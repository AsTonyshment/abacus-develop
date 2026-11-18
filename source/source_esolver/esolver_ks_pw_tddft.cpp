#include "source_esolver/esolver_ks_pw_tddft.h"

#include "source_base/constants.h"
#include "source_base/global_variable.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_base/tool_title.h"
#include "source_cell/cell_tools.h"
#include "source_estate/elecstate_pw.h"
#include "source_estate/elecstate_tools.h"
#include "source_estate/module_charge/chg_symm.h"
#include "source_hamilt/hamilt_hs_adapter.h"
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

namespace ModuleESolver
{

template <typename T, typename Device>
ESolver_KS_PW_TDDFT<T, Device>::ESolver_KS_PW_TDDFT() : log_(GlobalV::ofs_running)
{
    this->classname = "ESolver_KS_PW_TDDFT";
    this->basisname = "PW";
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

    // 2) Initialize the common PW data (before checking the resolved XC functional and RT capabilities).
    ESolver_KS_PW<T, Device>::before_all_runners(ucell, inp);

    // The parent resolves the XC functional, including pseudopotential defaults.
    const int xc_type = XC_Functional::get_func_type();
    pw::check_td_input(inp, xc_type, this->general_exx_info_.cal_exx);

    // 3) Prepare ionic operators, the propagator and accepted-state storage on the initialized PW grids.
    this->init_td(&ucell, inp);

    // 4) Initialize the requested endpoint observables.
    this->init_obs(ucell, inp);

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
    const bool moving = unitcell::if_atoms_can_move(cell.atoms, cell.ntype);

    // 1) Establish the initial electronic state once through the common SCF driver and its after_scf hook.
    if (istep == 0)
    {
        if (moving)
        {
            pw::check_td_coupled(*this->inp_);
        }
        ESolver_KS::runner(cell, 0);

        if (moving || this->inp_->mdp.md_nstep != 1)
        {
            return;
        }
    }

    // 2) Ehrenfest currently requires estep_per_md=1, so each MD advance calls prop_step once.
    if (moving)
    {
        const bool need_force = true;
        this->prop_step(&cell, istep, nullptr, need_force);
        return;
    }

    // 3) Fixed ions allow estep_per_md=N, so this MD call schedules N electronic steps.
    const int md_interval = std::max(istep, 1);
    const int first_step = (md_interval - 1) * steps_per_md + 1;
    const int last_step = md_interval * steps_per_md;

    // 4) Use a constant path for all samples and let prop_path iterate over the scheduled electronic steps.
    const double dt = this->field_->dt_ha();
    const double left = (first_step - 1) * dt;
    const double right = last_step * dt;

    const unitcell::IonicState &left_ions = this->history_.ions();
    const pw::TDIonPath ions(left_ions, left, right);

    // MD requests a force after this runner call, at the last electronic endpoint.
    const bool need_force = true;
    prop_path(&cell, first_step, last_step, ions, need_force);
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::prop_path(
    UnitCell *cell,
    const int &first_step,
    const int &last_step,
    const pw::TDIonPath &ions,
    const bool &need_force)
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

    // 3) Delegate each step of the supplied fixed or moving path to prop_step, including electronic retries and output.
    for (int step = first_step; step <= last_step; ++step)
    {
        const bool end_force = need_force && step == last_step;
        prop_step(cell, step, &ions, end_force);
    }
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::prop_step(
    UnitCell *cell,
    const int &step,
    const pw::TDIonPath *prescribed,
    const bool &need_force)
{
    ModuleBase::timer::start("ESolver_KS_PW_TDDFT", "prop_step");
    const ModuleBase::TimePoint started = ModuleBase::get_time();

    // 1) Read the accepted left state, predict the ionic endpoint and preserve its origin when needed.
    // A null path selects Ehrenfest ionic self-consistency; a supplied path determines the ionic samples directly.
    const bool coupled = prescribed == nullptr;
    const unitcell::IonicState left = history_.ions();
    ModuleBase::matrix left_accel;
    const double dt = field_->dt_ha();
    const int limit = coupled ? this->inp_->td_ion_nmax : 1;
    const bool save_origin = coupled && limit > 1 && this->inp_->td_max_halvings > 0;
    const bool direct_etrs = this->inp_->propagator == 2 && !this->inp_->td_nl_motion;
    unitcell::IonicState trial_ions = left;
    if (coupled)
    {
        left_accel = this->ion_accel().accel;
        trial_ions = pw::predict_ions(left, left_accel, dt);
        // Use the electronic step index to avoid cumulative roundoff in accepted ionic times.
        trial_ions.t_pos = step * dt;
        trial_ions.t_vel = trial_ions.t_pos;

        // Preserve an immutable origin when later ionic trials can require repropagation.
        if (save_origin)
        {
            ion_origin_.copy_from(history_);
        }
    }

    // 2) Invalidate the old force after retaining the left acceleration, then prepare the full-step field.
    endpoint_force_.reset();
    this->before_scf(*cell, step);
    if (this->inp_->vnl_in_h)
    {
        hamilt::HamiltPW<T, Device> &hamilt = *static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
        hamilt.td_projectors().begin_step();
    }
    const elecstate::TDFieldState full = field_->field_state();

    // 3) Propagate electrons along each trial path and iterate the coupled endpoint velocity.
    bool conv = false;
    IntervalStats retained;
    ModuleBase::matrix ion_force;
    double end_ewald = 0.0;
    for (int iter = 1; iter <= limit; ++iter)
    {
        // 3.1) Start every ionic trial from the same original history and field with fresh statistics.
        if (iter > 1 && save_origin)
        {
            history_.copy_from(ion_origin_);
        }
        retained = IntervalStats();

        // 3.2) Select the prescribed path or construct a Hermite path from the trial endpoint velocity.
        const bool need_derivs = (coupled || need_force) && this->inp_->td_nl_motion && this->inp_->vnl_in_h;
        IntervalEnd interval;
        const double *saved_ewald = iter > 1 ? &end_ewald : nullptr;
        if (coupled)
        {
            ModuleIO::print_ion_trial(step, iter, limit, &log_, &std::cout);
            const pw::TDIonPath path = pw::TDIonPath::hermite(left, trial_ions);
            interval = advance_interval(cell, step, 0, 0, full, path, need_derivs, saved_ewald, &retained);
        }
        else
        {
            interval = advance_interval(cell, step, 0, 0, full, *prescribed, need_derivs, saved_ewald, &retained);
        }

        const bool elec_conv = interval.converged;
        end_ewald = interval.ewald;

        // 3.3) Restore the full-step field after internal electronic retries.
        field_->bind_field(full);
        conv = elec_conv;
        if (!coupled)
        {
            break;
        }

        // 3.4) Evaluate endpoint forces and velocity-Verlet residuals to determine convergence and acceptance.
        const ModuleBase::matrix *saved_ion = ion_force.nr == 0 ? nullptr : &ion_force;
        const pw::IonForce force = cal_ion_force(*cell, saved_ion);
        if (saved_ion == nullptr)
        {
            ion_force = force.forces.ion;
        }
        const double vel_res = pw::correct_ions(left, left_accel, force.dynamics.accel, dt, &trial_ions);

        // Ionic convergence ends the outer correction even when the electronic iteration budget was exhausted.
        // Unsplit ETRS without traveling projectors completes the Verlet kick without another electronic trial.
        const bool ion_conv = (direct_etrs && retained.max_depth == 0) || vel_res <= this->inp_->td_ion_thr;
        // A single propagation checks electronic convergence without imposing ionic self-consistency.
        conv = elec_conv && (limit == 1 || ion_conv);
        const bool accepted = ion_conv || iter == limit;
        ModuleIO::print_td_coupled(
            step,
            iter,
            limit,
            retained.iters,
            retained.residual,
            vel_res,
            this->inp_->td_ion_thr,
            force.dynamics.residual,
            force.dynamics.status,
            conv,
            accepted,
            &log_,
            &std::cout);

        // 3.5) Finalize the corrected endpoint or reuse its velocity in the next trial path.
        if (accepted)
        {
            this->correct_end(cell, trial_ions, force);
            break;
        }
    }

    if (this->inp_->vnl_in_h)
    {
        hamilt::HamiltPW<T, Device> &hamilt = *static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
        hamilt.td_projectors().end_step();
    }

    // 4) Save the complete corrected endpoint once, then synchronize its displacement with MD.
    const psi::Psi<T, Device> &current = *this->stp.template get_psi_t<T, Device>();
    history_.save(current, *this->pelec, this->chr, right_ions_);
    if (coupled)
    {
        if (!endpoint_force_)
        {
            ModuleBase::WARNING_QUIT("prop_step", "Missing force result for the accepted coupled endpoint.");
        }
        pw::store_ion_disp(left, right_ions_, cell);
    }

    // 5) Report retained statistics and original-endpoint observables, then store the convergence status.
    hsolver::TDOrthStats orth;
    orth.before = retained.before;
    orth.after = retained.after;
    this->td_prop_->orbitals().set_orth_stats(orth);
    this->chr.save_rho_before_sum_band();
    this->out_prop(*cell, step, conv);
    const ModuleBase::TimePoint finished = ModuleBase::get_time();
    const double duration = ModuleBase::get_duration(started, finished);
    this->record_iteration(*cell, duration);
    this->after_scf(*cell, step, conv);
    this->conv_esolver = conv;

    ModuleBase::timer::end("ESolver_KS_PW_TDDFT", "prop_step");
}

template <typename T, typename Device>
typename ESolver_KS_PW_TDDFT<T, Device>::IntervalEnd ESolver_KS_PW_TDDFT<T, Device>::advance_interval(
    UnitCell *cell,
    const int &step,
    const int &index,
    const int &depth,
    const elecstate::TDFieldState &full,
    const pw::TDIonPath &ions,
    const bool &need_derivs,
    const double *right_ewald,
    IntervalStats *stats)
{
    // 1) Start each retry subinterval from the latest accepted electronic state and bind its external field.
    hamilt::HamiltPW<T, Device> *sample_hamilt = nullptr;
    if (this->inp_->vnl_in_h)
    {
        sample_hamilt = static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
    }
    endpoint_.clear_sample(sample_hamilt);
    const int parts = 1 << depth;
    psi::Psi<T, Device> &wave = *this->stp.template get_psi_t<T, Device>();
    this->history_.restore(&wave, this->pelec, &this->chr);
    this->td_prop_->orbitals().reset_orth_stats();
    const elecstate::TDFieldState trial = this->field_->subinterval(full, index, parts);
    this->field_->bind_field(trial);

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
        // History restoration replaces energies, so always install the current endpoint's Ewald value.
        if (right_ewald != nullptr)
        {
            this->pelec->f_en.ewald_energy = *right_ewald;
        }
        else
        {
            this->pelec->f_en.ewald_energy = ion_ops_->cal_ewald(this->inp_->test_energy, &log_);
        }
    }
    IntervalEnd result;
    result.ewald = this->pelec->f_en.ewald_energy;
    this->prepare_init_hamilt(*cell);

    const double left_fs = left_ha * ModuleBase::AU_to_FS;
    const double right_fs = right_ha * ModuleBase::AU_to_FS;
    ModuleIO::print_td_trial(left_fs, right_fs, depth, &log_, &std::cout);

    // 4) Reset trial caches and iterate the endpoint density through the common SCF loop.
    // Each hamilt2rho_single call propagates from the accepted left state; iter_finish updates the endpoint potential.
    const int iter_limit = this->scf_iter_limit(step);
    this->td_prop_->begin_trial();
    const bool conv = this->scf_loop(*cell, step, iter_limit);

    // 5) Accept a converged trial or one at the subdivision limit; otherwise solve the two halves in order.
    const bool accept_trial = conv || depth == this->inp_->td_max_halvings;
    ModuleIO::print_td_result(this->niter, this->drho, conv, accept_trial, &log_);
    if (!conv && depth < this->inp_->td_max_halvings)
    {
        ModuleIO::print_td_retry(left_fs, right_fs, &std::cout);

        const int next_depth = depth + 1;
        const int left_index = 2 * index;
        const int right_index = left_index + 1;

        // Accept the left half first so the right half starts from its updated electronic and ionic history.
        const IntervalEnd left_end = advance_interval(cell, step, left_index, next_depth, full, ions, need_derivs, nullptr, stats);
        const IntervalEnd right_end = advance_interval(cell, step, right_index, next_depth, full, ions, need_derivs, &result.ewald, stats);
        result.converged = left_end.converged && right_end.converged;
        return result;
    }

    // 6) Finalize each retained endpoint with its orbital density before saving it.
    const bool final_sample = index + 1 == parts;
    elecstate::ElecStatePW<T, Device> *estate = static_cast<elecstate::ElecStatePW<T, Device> *>(this->pelec);
    pw::prep_td_rho(wave, estate, &this->chr, cell, *this->pw_rhod, conv, this->inp_->nspin, this->inp_->imp_sol);
    if (final_sample)
    {
        const bool need_products = this->inp_->vnl_in_h;
        this->finish_end(cell, need_products, need_derivs);
    }

    // 7) Save intermediate history for retries; prop_step commits the corrected full-step endpoint.
    const hsolver::TDOrthStats &trial_stats = this->td_prop_->orbitals().orth_stats();
    stats->before = std::max(stats->before, trial_stats.before);
    stats->after = std::max(stats->after, trial_stats.after);
    stats->residual = std::max(stats->residual, this->drho);
    stats->iters += this->niter;
    stats->max_depth = std::max(stats->max_depth, depth);

    if (!final_sample)
    {
        const psi::Psi<T, Device> *current = this->stp.template get_psi_t<T, Device>();
        this->history_.save(*current, *this->pelec, this->chr, right_ions_);
    }

    result.converged = conv;
    return result;
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
        // Clear endpoint forces and outer-trial snapshots before initializing a new trajectory.
        endpoint_force_.reset();
        hamilt::HamiltPW<T, Device> *old_hamilt = nullptr;
        if (this->p_hamilt != nullptr && this->inp_->vnl_in_h)
        {
            old_hamilt = static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
        }
        endpoint_.clear_sample(old_hamilt);
        ion_origin_ = pw::TDHistoryPW<T, Device>();
        ESolver_KS_PW<T, Device>::before_scf(ucell, istep);
        const unitcell::IonicState initial = unitcell::capture_ions(ucell, 0.0, 0.0);
        ion_ops_->bind(initial);
    }

    // 3) Ensure the accepted-potential buffer matches the initialized grid before saving any endpoint.
    this->history_.prepare(*this->pelec->pot);
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
    if (istep > 0 && (this->inp_->mdp.md_nstep > 1 || unitcell::if_atoms_can_move(ucell.atoms, ucell.ntype)))
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
    this->td_prop_->evolve(hamilt_pw, veff, nveff, this->history_, *this->field_, prop_ions_, right_ions_, current, istep, iter, detailed);

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

    // 1) Write the accepted propagation field; prep_td_step already writes the initial field.
    if (istep > 0)
    {
        ModuleIO::out_td_field(*this->field_, out_dir, istep, this->inp_->out_vecpot, this->inp_->out_efield, rank);
    }

    // 2) Finalize the initial state's orthogonality check and rebuild its endpoint quantities if SCF is unconverged.
    if (istep == 0 && this->inp_->td_orthonormal == "none")
    {
        const psi::Psi<T, Device> *current = this->stp.template get_psi_t<T, Device>();
        this->td_prop_->orbitals().check_initial(*current, this->niter);
    }
    if (!conv_esolver && istep == 0)
    {
        const psi::Psi<T, Device> &wave = *this->stp.template get_psi_t<T, Device>();
        elecstate::ElecStatePW<T, Device> *estate = static_cast<elecstate::ElecStatePW<T, Device> *>(this->pelec);
        const bool rho_ready = false;
        pw::prep_td_rho(wave, estate, &this->chr, &ucell, *this->pw_rhod, rho_ready, this->inp_->nspin, this->inp_->imp_sol);
        this->finish_end(&ucell, false, false);
    }

    // 3) Bind endpoint potential, field and nuclear velocities for optional Eij output and common PW post-SCF processing.
    hamilt::HamiltPW<T, Device> *hamilt = static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
    const Real *veff = this->pelec->pot->template get_veff_smooth_data<Real>();
    const ModuleBase::Vector3<double> &vecpot = this->field_->A_right_ha();
    const std::vector<ModuleBase::Vector3<double>> vel = pw::td_ion_vel(ucell, this->inp_->td_nl_motion);
    const std::vector<ModuleBase::Vector3<double>> tau = unitcell::cell_positions(ucell);
    hamilt->bind_td_state(veff, vecpot, vel, tau);
    this->out_eij(istep);

    ESolver_KS_PW<T, Device>::after_scf(ucell, istep, conv_esolver);

    // 4) Evaluate initial coupled dynamics once, including Pi for the endpoint energy.
    if (istep == 0 && unitcell::if_atoms_can_move(ucell.atoms, ucell.ntype))
    {
        const pw::IonForce force = cal_ion_force(ucell, nullptr);
        endpoint_force_.reset(new pw::IonForce(force));
    }

    // 5) Prepare shared endpoint overlaps and update the MD energy before reporting observables.
    ModuleBase::matrix ion_pi;
    const hamilt::NLOrbData<typename GetTypeReal<T>::type, Device> *nl_orbs = this->prep_end_orbs(ucell, &ion_pi);
    const pw::TDEnergy energy = this->cal_end_energy(ucell, ion_pi);
    md_energy_ = energy.potential;
    this->out_energy(energy, istep, rank, out_dir);

    // 6) Seed initial history; prop_step saves subsequent endpoints after their velocity correction.
    if (istep == 0)
    {
        psi::Psi<T, Device> *current = this->stp.template get_psi_t<T, Device>();
        const unitcell::IonicState &initial_ions = ion_ops_->state();
        this->history_.save(*current, *this->pelec, this->chr, initial_ions);
    }

    // 7) Pass the prepared overlaps explicitly to current and spin sampling, then report conservation.
    this->out_curr_spin(ucell, nl_orbs, istep, rank, out_dir);
    this->out_conserv(ucell, istep);
}

template class ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>;
template class ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>;
template class ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>;
#endif

} // namespace ModuleESolver
