#include "source_esolver/esolver_ks_pw_tddft.h"

#include "source_base/constants.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/timer.h"
#include "source_cell/cell_tools.h"
#include "source_estate/elecstate_pw.h"
#include "source_estate/elecstate_tools.h"
#include "source_estate/module_charge/chg_symm.h"
#include "source_hamilt/hamilt_hs_adapter.h"
#include "source_hamilt/module_ewald/h_ewald_pw.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_ctrl/ctrl_output_td_pw.h"
#include "source_io/module_efield/td_vecpot_io.h"
#include "source_io/module_output/output_log.h"
#include "source_io/module_parameter/parameter.h"
#include "source_pw/module_pwdft/hamilt_pw.h"
#include "source_pw/module_rt/td_endpoint_pw.h"
#include "source_pw/module_rt/td_pw.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <type_traits>

namespace ModuleESolver
{

template <typename T, typename Device>
ESolver_KS_PW_TDDFT<T, Device>::ESolver_KS_PW_TDDFT() : log_(GlobalV::ofs_running)
{
    this->classname = "ESolver_KS_PW_TDDFT";
    this->basisname = "PW";
}

template <typename T, typename Device>
double ESolver_KS_PW_TDDFT<T, Device>::cal_energy()
{
    return material_energy_;
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::before_all_runners(BaseCell &basecell, const Input_para &inp)
{
    basecell.require_kind(BaseCell::Kind::unitcell, __FUNCTION__);
    UnitCell &ucell = static_cast<UnitCell &>(basecell);

    // 1) Prepare the external field and optional vector-potential samples for electronic time stepping.
    this->field_ = elecstate::create_td_field_manager(inp);
    if (inp.td_max_halvings > 0)
    {
        const int parts = 1 << inp.td_max_halvings;
        for (const elecstate::TDField &field : this->field_->fields())
        {
            if (field.subdivisions() > std::numeric_limits<int>::max() / parts)
            {
                ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "td_max_halvings exceeds the field sampling integer range.");
            }
        }
    }
    if (inp.init_vecpot)
    {
        // An empty directory selects the calculation working directory.
        const std::vector<ModuleBase::Vector3<double>> samples = ModuleIO::read_td_vecpot("");
        this->field_->set_A_samples(samples);
    }

    // 2) Initialize the common PW data before checking the resolved XC functional and RT capabilities.
    ESolver_KS_PW<T, Device>::before_all_runners(ucell, inp);

    // The parent resolves the XC functional, including pseudopotential defaults.
    const int xc_type = XC_Functional::get_func_type();
    pw::check_td_input(inp, xc_type, this->general_exx_info_.cal_exx);

    // 3) Prepare ionic operators and endpoint observables using the existing PW grids and charge storage.
    const bool has_float = std::is_same<Real, float>::value;
    ion_ops_.reset(new pw::TDIonPW(&ucell, &this->sf, this->Pgrid, *this->pw_rhod, &this->chr, this->locpp.numeric, has_float));

    current_req_.charge_total = inp.out_current == 1;
    current_req_.charge_per_k = current_req_.charge_total && inp.out_current_k;
    current_req_.spin_current = inp.nspin == 4 && inp.td_out_spincur;
    vel_.initialize(ucell, *this->pw_wfc, this->kv, this->ppcell);

    if (inp.nspin == 4)
    {
        spin_.initialize(
            ucell,
            *this->pw_rhod,
            *this->pw_wfc,
            this->kv,
            this->sf,
            this->ppcell,
            inp.td_out_mag,
            inp.orbital_dir,
            inp.onsite_radius,
            inp.td_out_torque);
    }

    // 4) Attach the field to the potential and create the selected propagator and accepted-state storage.
    this->pelec->pot->set_td_field_manager(this->field_);
#ifdef __MPI
    const hsolver::diag_comm_info comm(this->pw_wfc->pool_world, this->pw_wfc->poolrank, this->pw_wfc->poolnproc);
#else
    const hsolver::diag_comm_info comm(0, 1);
#endif
    const int npol = ucell.get_npol();
    this->td_prop_ = pw::make_td_prop<T, Device>(*this->pw_wfc, inp, npol, this->kv.ik2iktot, comm, log_);
    this->history_.prepare(*this->pelec->pot);

    const int rank = GlobalV::MY_RANK;
    const std::string &out_dir = PARAM.globalv.global_out_dir;

    // 5) Prepare the enabled field output files before sampling begins.
    ModuleIO::prep_td_out(*this->field_, out_dir, inp.out_vecpot, inp.out_efield, inp.init_vecpot, rank);
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::runner(BaseCell &basecell, const int istep)
{
    basecell.require_kind(BaseCell::Kind::unitcell, __FUNCTION__);
    UnitCell &cell = static_cast<UnitCell &>(basecell);

    const int steps_per_md = this->inp_->estep_per_md;

    // 1) Establish the initial electronic state once through the common SCF driver and its after_scf hook.
    if (istep == 0)
    {
        if (unitcell::if_atoms_can_move(cell.atoms, cell.ntype))
        {
            ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "Fixed-ion MD scheduling requires movement flags 0 0 0.");
        }
        ESolver_KS::runner(cell, 0);
        if (this->inp_->mdp.md_nstep != 1)
        {
            return;
        }
    }

    // 2) Map this fixed-ion MD scheduling call to its inclusive range of electronic steps.
    int first_step = (istep - 1) * steps_per_md + 1;
    int last_step = istep * steps_per_md;
    if (this->inp_->mdp.md_nstep == 1)
    {
        // A single MD call includes both initialization and the requested electronic steps.
        first_step = 1;
        last_step = steps_per_md;
    }

    // 3) Supply the accepted geometry at all operator and retry times, then advance the electronic steps.
    const double dt = this->field_->dt_ha();
    const double left = (first_step - 1) * dt;
    const double right = last_step * dt;

    const unitcell::IonicState &left_ions = this->history_.ions();
    const pw::TDIonPath ions(left_ions, left, right);

    propagate(&cell, first_step, last_step, ions);
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::propagate(
    UnitCell *cell,
    int first_step,
    int last_step,
    const pw::TDIonPath &ions)
{
    // 1) Check that the requested path and step range can propagate the initialized electronic state.
    if (!ions.is_fixed())
    {
        pw::check_td_ions(*this->inp_);
    }
    if (first_step < 1 || last_step < first_step || !this->history_.ready())
    {
        ModuleBase::WARNING_QUIT(
            "ESolver_KS_PW_TDDFT", "Propagation requires an initialized electronic state and an increasing positive step range.");
    }

    // 2) Join the supplied ionic path to the accepted state at the first interval's left endpoint.
    const double dt = this->field_->dt_ha();
    const double time = (first_step - 1) * dt;
    const unitcell::IonicState &saved = this->history_.ions();
    ions.check(*cell, saved, time);

    // 3) Advance one electronic step at a time; run_prop handles each step's trials and endpoint output.
    for (int step = first_step; step <= last_step; ++step)
    {
        run_prop(cell, step, ions);
    }
}

template <typename T, typename Device>
bool ESolver_KS_PW_TDDFT<T, Device>::defer_iter_output(const int istep) const
{
    return istep > 0;
}

template <typename T, typename Device>
int ESolver_KS_PW_TDDFT<T, Device>::scf_iter_limit(const int istep) const
{
    return istep > 0 ? this->inp_->td_scf_nmax : this->inp_->scf_nmax;
}

template <typename T, typename Device>
bool ESolver_KS_PW_TDDFT<T, Device>::advance_interval(
    UnitCell *cell,
    int step,
    int index,
    int depth,
    const elecstate::TDFieldState &full,
    const pw::TDIonPath &ions,
    hsolver::TDOrthStats *stats)
{
    // 1) Start each retry subinterval from the latest accepted electronic state and bind its external field.
    const int parts = 1 << depth;
    if (depth > 0)
    {
        psi::Psi<T, Device> &current = *this->stp.template get_psi_t<T, Device>();
        this->history_.restore(&current, this->pelec, &this->chr);
        this->td_prop_->orbitals().reset_orth_stats();

        const elecstate::TDFieldState trial = this->field_->subinterval(full, index, parts);
        this->field_->bind_field(trial);
    }

    // 2) Locate the method's ionic sample and the right endpoint using times derived from the original step.
    // CN uses midpoint ions; ETRS uses left-endpoint ions for its cached first half-step.
    const double left_ha = (step - 1 + static_cast<double>(index) / parts) * full.dt;
    const double right_ha = (step - 1 + static_cast<double>(index + 1) / parts) * full.dt;
    const double sample_time = this->td_prop_->sample_time(left_ha, right_ha);
    prop_ions_ = ions.at(sample_time);
    right_ions_ = ions.at(right_ha);

    // 3) Prepare the right-endpoint potential, core density and ion-ion energy for the electronic iterations.
    ion_ops_->update_pot(right_ions_, this->pelec->pot);
    if (!ions.is_fixed() && !this->inp_->test_skip_ewald)
    {
        this->pelec->f_en.ewald_energy = H_Ewald_pw::compute_ewald(*cell, this->pw_rhod, this->sf.strucFac, this->inp_->test_energy, log_);
    }
    this->prepare_init_hamilt(*cell);

    const double left_fs = left_ha * ModuleBase::AU_to_FS;
    const double right_fs = right_ha * ModuleBase::AU_to_FS;
    const std::string interval_text = ModuleIO::print_td_trial(left_fs, right_fs, depth, &log_, &std::cout);

    // 4) Reset trial caches and iterate the endpoint density through the common SCF loop.
    // Each hamilt2rho_single call propagates from the accepted left state; iter_finish updates the endpoint potential.
    const int iter_limit = this->scf_iter_limit(step);
    this->td_prop_->begin_trial();
    const bool converged = this->scf_loop(*cell, step, iter_limit);

    // 5) Accept a converged trial or one at the subdivision limit; otherwise solve the two halves in order.
    const bool accept_trial = converged || depth == this->inp_->td_max_halvings;
    ModuleIO::print_td_result(this->niter, this->drho, converged, accept_trial, &log_);
    if (!converged && depth < this->inp_->td_max_halvings)
    {
        ModuleIO::print_td_retry(interval_text, &std::cout);

        const int next_depth = depth + 1;
        const int left_index = 2 * index;
        const int right_index = left_index + 1;

        // Accept the left half first so the right half starts from its updated electronic and ionic history.
        const bool left_converged = advance_interval(cell, step, left_index, next_depth, full, ions, stats);
        const bool right_converged = advance_interval(cell, step, right_index, next_depth, full, ions, stats);
        return left_converged && right_converged;
    }

    // 6) At the retry limit, rebuild consistent endpoint quantities from the final orbitals and continue propagation.
    if (!converged)
    {
        this->rebuild_end(cell);
    }

    // 7) Accumulate accepted orthogonality statistics and save electrons and ions at the same right endpoint.
    // This history supplies the starting state for the next subinterval or electronic step.
    const hsolver::TDOrthStats &trial_stats = this->td_prop_->orbitals().orth_stats();
    stats->before = std::max(stats->before, trial_stats.before);
    stats->after = std::max(stats->after, trial_stats.after);

    const psi::Psi<T, Device> *current = this->stp.template get_psi_t<T, Device>();
    const bool save_retry = this->inp_->td_max_halvings > 0;
    this->history_.save(*current, *this->pelec, this->chr, right_ions_, save_retry);

    return converged;
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::run_prop(UnitCell *cell, int step, const pw::TDIonPath &ions)
{
    ModuleBase::timer::start("ESolver_KS_PW_TDDFT", "run_prop");
    const ModuleBase::TimePoint started = ModuleBase::get_time();

    // 1) Prepare the external field and diagnostics for the full electronic step.
    this->before_scf(*cell, step);
    ModuleBase::GlobalFunc::DONE(log_, "INIT SCF");

    // 2) Advance the full interval, allowing advance_interval to accept smaller subintervals when needed.
    const elecstate::TDFieldState full = this->field_->field_state();
    hsolver::TDOrthStats stats;
    const bool converged = advance_interval(cell, step, 0, 0, full, ions, &stats);

    // 3) Restore full-step field metadata and retain the statistics and density of the accepted endpoint.
    this->field_->bind_field(full);
    this->td_prop_->orbitals().set_orth_stats(stats);
    this->chr.save_rho_before_sum_band();

    // 4) Write density, orbital and field results once all accepted subintervals reach the full-step endpoint.
    this->out_prop(*cell, step, converged);

    const double duration = ModuleBase::get_duration(started, ModuleBase::get_time());
    this->record_iteration(*cell, duration);

    const int rank = GlobalV::MY_RANK;
    const std::string &out_dir = PARAM.globalv.global_out_dir;
    ModuleIO::out_td_field(*this->field_, out_dir, step, this->inp_->out_vecpot, this->inp_->out_efield, rank);

    // 5) Finalize endpoint energies and observables, retaining the convergence status of the full step.
    this->after_scf(*cell, step, converged);
    this->conv_esolver = converged;

    ModuleBase::timer::end("ESolver_KS_PW_TDDFT", "run_prop");
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::before_scf(UnitCell &ucell, const int istep)
{
    // 1) Select the field for this electronic step and reset its orthogonality diagnostics.
    this->print_step(istep);
    this->prep_td_step(istep);
    this->td_prop_->orbitals().reset_orth_stats();

    // 2) Initialize the common PW state and bind the initial ions at step zero.
    // During propagation, advance_interval applies the ionic samples required by each trial.
    if (istep == 0)
    {
        ESolver_KS_PW<T, Device>::before_scf(ucell, istep);
        const unitcell::IonicState initial = unitcell::capture_ions(ucell, 0.0, 0.0);
        ion_ops_->bind(initial);
    }

    // 3) Ensure the accepted-potential buffer matches the initialized grid before saving any endpoint.
    this->history_.prepare(*this->pelec->pot);
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::prepare_init_hamilt(const UnitCell &ucell)
{
    if (this->field_->gauge() == 1)
    {
        // 1) Cache the unshifted momentum bound of the fixed PW basis during initialization.
        if (this->field_->current_step() == 0)
        {
            qmax_ = pw::td_momentum_bound(*this->pw_wfc, ucell.tpiba);
        }
        // 2) Extend radial projector tables to cover the endpoint vector potentials and their interpolated samples.
        const ModuleBase::Vector3<double> &left = this->field_->A_left_ha();
        const ModuleBase::Vector3<double> &right = this->field_->A_right_ha();
        pw::ensure_td_vnl(ucell, qmax_, right, left, &this->ppcell);

        // 3) Bind the right-endpoint potential and vector potential before applying the Hamiltonian.
        // This also includes a file-supplied nonzero A(0) when preparing the initial wavefunctions.
        elecstate::Potential &potential = *this->pelec->pot;
        const Real *veff = potential.template get_veff_smooth_data<Real>();
        hamilt::HamiltPW<T, Device> *hamilt_pw = static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
        hamilt_pw->bind_td_state(veff, this->field_->A_right_ha());
    }
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::prep_td_step(const int istep)
{
    ModuleBase::timer::start("ESolver_KS_PW_TDDFT", "prep_td_step");

    // 1) Reuse an already prepared step so repeated setup preserves its field sampling state.
    if (this->field_->current_step() == istep)
    {
        ModuleBase::timer::end("ESolver_KS_PW_TDDFT", "prep_td_step");
        return;
    }

    // 2) Sample the initial or length-gauge field at the endpoint; prepare a full interval for velocity-gauge propagation.
    if (this->field_->gauge() == 0 || istep == 0)
    {
        this->field_->prepare_sample(istep);
    }
    else
    {
        const int interval_left = istep - 1;
        this->field_->prepare_interval(interval_left, istep);
    }

    // 3) Report the prepared field; propagation field files are written by run_prop after interval acceptance.
    if (this->field_->gauge() == 1)
    {
        const ModuleBase::Vector3<double> &A_right_ha = this->field_->A_right_ha();
        ModuleBase::GlobalFunc::OUT(log_, "Cartesian vector potential Ax(t)", A_right_ha[0]);
        ModuleBase::GlobalFunc::OUT(log_, "Cartesian vector potential Ay(t)", A_right_ha[1]);
        ModuleBase::GlobalFunc::OUT(log_, "Cartesian vector potential Az(t)", A_right_ha[2]);
    }

    if (!this->defer_iter_output(istep))
    {
        const int rank = GlobalV::MY_RANK;
        const std::string &out_dir = PARAM.globalv.global_out_dir;
        ModuleIO::out_td_field(*this->field_, out_dir, istep, this->inp_->out_vecpot, this->inp_->out_efield, rank);
    }

    ModuleBase::timer::end("ESolver_KS_PW_TDDFT", "prep_td_step");
}

template <typename T, typename Device>
std::string ESolver_KS_PW_TDDFT<T, Device>::diag_policy(const int istep) const
{
    if (istep == 0)
    {
        return "ksdft";
    }
    return ESolver_KS::diag_policy(istep);
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::iter_init(UnitCell &ucell, const int istep, const int iter)
{
    int ionic_step = 0;
    if (this->inp_->mdp.md_nstep > 1 && istep > 0)
    {
        ionic_step = (istep - 1) / this->inp_->estep_per_md + 1;
    }
    ModuleIO::write_head_td(log_, ionic_step, istep, iter, this->basisname);
    ESolver_KS_PW<T, Device>::iter_init(ucell, istep, iter);
}

template <typename T, typename Device>
ESolver_KS::DensityStage ESolver_KS_PW_TDDFT<T, Device>::density_stage(const int istep, const int iter) const
{
    // The first RT iteration supplies an unmixed density prediction to iter_finish for rebuilding the endpoint potential.
    // Subsequent iterations use the common mixing and convergence checks to correct that prediction.
    if (istep > 0 && iter == 1)
    {
        return ESolver_KS::DensityStage::predictor;
    }
    return ESolver_KS::DensityStage::standard;
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::hamilt2rho_single(
    UnitCell &ucell,
    const int istep,
    const int iter,
    const double ethr)
{
    ModuleBase::timer::start("ESolver_KS_PW_TDDFT", "hamilt2rho_single");

    // 1) At step zero, diagonalize the KS Hamiltonian and apply the requested initial orbital orthogonality treatment.
    if (istep == 0)
    {
        ESolver_KS_PW<T, Device>::hamilt2rho_single(ucell, istep, iter, ethr);

        psi::Psi<T, Device> &current = *this->stp.template get_psi_t<T, Device>();
        elecstate::ElecStatePW<T, Device> &estate = *static_cast<elecstate::ElecStatePW<T, Device> *>(this->pelec);
        hamilt::Hamilt<T, Device> *hamilt = static_cast<hamilt::Hamilt<T, Device> *>(this->p_hamilt);
        hamilt::HamiltHSOperator<T, Device> op(hamilt, this->pw_wfc);
        const bool do_orth = this->inp_->td_orthonormal != "none";
        pw::orth_init_pw(
            this->td_prop_->orbitals(),
            op,
            current,
            estate,
            this->chr,
            ucell,
            *this->pw_rhod,
            this->inp_->nspin,
            this->inp_->nbands,
            iter,
            do_orth);

        ModuleBase::timer::end("ESolver_KS_PW_TDDFT", "hamilt2rho_single");
        return;
    }

    // 2) Propagate the accepted left state: CN correctors average endpoint potentials; ETRS correctors reuse the left half-step.
    // The propagator performs the requested orbital correction and restores right-endpoint operators.
    ModuleBase::TITLE("ESolver_KS_PW_TDDFT", "hamilt2rho_single (RT-TDDFT)");
    psi::Psi<T, Device> *current = this->stp.template get_psi_t<T, Device>();
    elecstate::Potential &potential = *this->pelec->pot;
    hamilt::HamiltPW<T, Device> *hamilt_pw = static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
    hamilt::HamiltHSOperator<T, Device> op(hamilt_pw, this->pw_wfc);

    const Real *veff = potential.template get_veff_smooth_data<Real>();
    const ModuleBase::matrix &shape = potential.get_veff_smooth();
    const int nveff = shape.nr * shape.nc;
    const bool detailed = this->inp_->out_level == "ie";
    this->td_prop_->evolve(
        hamilt_pw, veff, nveff, this->history_, *this->field_, ion_ops_.get(), prop_ions_, right_ions_, current, istep, iter, detailed);

    // 3) Construct the trial endpoint density from the propagated orbitals with the initial occupations held fixed.
    elecstate::ElecStatePW<T, Device> *estate = static_cast<elecstate::ElecStatePW<T, Device> *>(this->pelec);
    estate->psiToRho(*current);
    module_charge::symmetrize_rho(this->inp_->nspin, this->chr, this->pw_rhod, ucell.symm);

    // 4) Evaluate the band sum with the same endpoint potential used for the KS double-counting terms.
    // The common iter_finish then updates the potential from this density for the next corrector iteration.
    this->td_prop_->orbitals().cal_band_energy(op, *current, &this->pelec->ekb);
    elecstate::calEBand(this->pelec->ekb, this->pelec->wg, this->pelec->f_en);

    ModuleBase::timer::end("ESolver_KS_PW_TDDFT", "hamilt2rho_single");
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::after_scf(UnitCell &ucell, const int istep, const bool conv_esolver)
{
    const int rank = GlobalV::MY_RANK;
    const std::string &out_dir = PARAM.globalv.global_out_dir;
    // 1) Finalize the initial state's orthogonality check and rebuild its endpoint quantities if SCF is unconverged.
    if (istep == 0 && this->inp_->td_orthonormal == "none")
    {
        const psi::Psi<T, Device> *current = this->stp.template get_psi_t<T, Device>();
        this->td_prop_->orbitals().check_initial(*current, this->niter);
    }
    if (!conv_esolver && istep == 0)
    {
        this->rebuild_end(&ucell);
    }

    // 2) Bind the finalized endpoint Hamiltonian for optional Eij output and common PW post-SCF processing.
    hamilt::HamiltPW<T, Device> *hamilt = static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
    const Real *veff = this->pelec->pot->template get_veff_smooth_data<Real>();
    const ModuleBase::Vector3<double> &vecpot = this->field_->A_right_ha();
    hamilt->bind_td_state(veff, vecpot);
    this->out_eij(istep);

    ESolver_KS_PW<T, Device>::after_scf(ucell, istep, conv_esolver);

    // 3) Update the material energy returned to the driver and write endpoint energy diagnostics.
    this->out_energy(istep, rank, out_dir);

    // 4) Seed the accepted history after initial SCF; propagation endpoints are saved by advance_interval.
    if (istep == 0)
    {
        psi::Psi<T, Device> *current = this->stp.template get_psi_t<T, Device>();
        const unitcell::IonicState &initial_ions = ion_ops_->state();
        const bool save_retry = this->inp_->td_max_halvings > 0;
        this->history_.save(*current, *this->pelec, this->chr, initial_ions, save_retry);
    }

    // 5) Evaluate endpoint current, spin and conservation observables using the finalized electronic and ionic state.
    this->out_td(ucell, istep, rank, out_dir);
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::rebuild_end(UnitCell *cell)
{
    // Recompute endpoint density, potential and energy together when an iteration-limited state is accepted.
    psi::Psi<T, Device> &current = *this->stp.template get_psi_t<T, Device>();
    elecstate::ElecStatePW<T, Device> &estate = *static_cast<elecstate::ElecStatePW<T, Device> *>(this->pelec);
    hamilt::HamiltPW<T, Device> &hamilt = *static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
    const ModuleBase::Vector3<double> &vecpot = this->field_->A_right_ha();
    pw::rebuild_td_end(
        current,
        estate,
        this->chr,
        *cell,
        hamilt,
        this->td_prop_->orbitals(),
        *this->pw_wfc,
        *this->pw_rhod,
        vecpot,
        this->inp_->nspin,
        this->inp_->imp_sol,
        this->inp_->sc_mag_switch,
        this->inp_->dft_plus_u,
        this->inp_->assume_isolated);
}

template class ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>;
template class ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>;
template class ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>;
#endif

} // namespace ModuleESolver
