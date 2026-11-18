#include "source_esolver/esolver_ks_pw_tddft.h"

#include "source_base/global_variable.h"
#include "source_base/module_parallel/para_bridge.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_cell/cell_tools.h"
#include "source_estate/elecstate_tools.h"
#include "source_estate/module_charge/chg_rhog_io.h"
#include "source_hamilt/hamilt_hs_adapter.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_ctrl/ctrl_output_pw.h"
#include "source_io/module_ctrl/ctrl_output_td_pw.h"
#include "source_io/module_energy/td_eij_pw.h"
#include "source_io/module_output/output_log.h"
#include "source_io/module_output/td_conserv_io.h"
#include "source_io/module_parameter/parameter.h"
#include "source_pw/module_pwdft/hamilt_pw.h"
#include "source_pw/module_rt/td_pw.h"

#include <iostream>

namespace ModuleESolver
{
namespace
{
template <typename Device>
const hamilt::NLOrbData<double, Device> *force_orbs(const hamilt::NLOrbData<double, Device> *orbs)
{
    return orbs;
}
template <typename Device>
const hamilt::NLOrbData<double, Device> *force_orbs(const hamilt::NLOrbData<float, Device> *)
{
    // PW forces use double-precision orbitals and therefore require independent projections.
    return nullptr;
}
} // namespace

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::init_obs(const UnitCell &ucell, const Input_para &inp)
{
    // Prepare current and spin evaluators once on the common PW grids.
    current_req_.charge_total = inp.out_current == 1;
    current_req_.charge_ions = current_req_.charge_total && unitcell::if_atoms_can_move(ucell.atoms, ucell.ntype);
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
}

template <typename T, typename Device>
const hamilt::NLOrbData<typename GetTypeReal<T>::type, Device> *ESolver_KS_PW_TDDFT<T, Device>::prep_end_orbs(
    const UnitCell &cell, ModuleBase::matrix *ion_pi)
{
    if (!this->inp_->vnl_in_h)
    {
        ion_pi->create(cell.nat, 3);
        return nullptr;
    }

    const pw::PiDerivs *derivs = endpoint_.derivatives(false);
    // Reuse derivatives evaluated with the accepted force or final endpoint band energies.
    if (endpoint_force_)
    {
        *ion_pi = endpoint_force_->pi;
    }
    else if (derivs != nullptr)
    {
        *ion_pi = derivs->pi;
    }
    else
    {
        // Initial fixed-ion endpoints prepare Pi and shared overlaps together.
        const psi::Psi<T, Device> &wave = *this->stp.template get_psi_t<T, Device>();
        const std::vector<ModuleBase::Vector3<double>> vel = pw::td_ion_vel(cell, this->inp_->td_nl_motion);
        ModuleBase::Vector3<double> nl_a{0.0, 0.0, 0.0};
        if (this->field_->gauge() == 1 && this->inp_->td_vg_nl != 0)
        {
            nl_a = this->field_->A_right_ha();
        }
        hamilt::HamiltPW<T, Device> &hamilt = *static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
        const bool need_grad = current_req_.charge_total || current_req_.charge_per_k || current_req_.spin_current;
        *ion_pi = endpoint_.cal_pi(
            cell,
            *this->pw_wfc,
            this->kv,
            &this->ppcell,
            wave,
            this->pelec->wg,
            nl_a,
            vel,
            need_grad,
            &hamilt,
            &this->td_prop_->orbitals(),
            nullptr);
    }

    // The borrowed overlaps remain valid until the next Pi evaluation.
    return this->ppcell.nkb > 0 ? &endpoint_.orb_data() : nullptr;
}

template <typename T, typename Device>
pw::TDEnergy ESolver_KS_PW_TDDFT<T, Device>::cal_end_energy(const UnitCell &cell, const ModuleBase::matrix &pi) const
{
    pw::TDEnergy energy;
    this->pelec->pot->td_field_energy(*this->pelec->charge, energy.elec_field, energy.ion_field);
    energy.smearing = this->pelec->f_en.demet;

    // Combine the accepted KS energy and field samples with the supplied nuclear velocity derivative.
    energy.potential = elecstate::cal_td_energy(
        this->pelec->f_en.etot,
        cell,
        pi,
        energy.elec_field,
        energy.ion_field,
        &energy.total,
        &energy.total_field,
        &energy.kinetic,
        &energy.vel_corr);
    return energy;
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::out_energy(
    const pw::TDEnergy &energy,
    const int &istep,
    const int &rank,
    const std::string &output_dir) const
{
    const double dt = this->field_->dt_ha();
    ModuleIO::out_td_energy(
        output_dir,
        istep,
        dt,
        energy.total,
        energy.total_field,
        energy.elec_field,
        energy.ion_field,
        energy.kinetic,
        energy.vel_corr,
        energy.potential,
        energy.smearing,
        this->inp_->td_out_stat,
        rank,
        &log_,
        &std::cout);
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::out_curr_spin(
    const UnitCell &ucell,
    const hamilt::NLOrbData<typename GetTypeReal<T>::type, Device> *nl_orbs,
    const int &istep,
    const int &rank,
    const std::string &output_dir)
{
    const psi::Psi<T, Device> &current = *this->stp.template get_psi_t<T, Device>();
    const ModuleBase::Vector3<double> &vecpot = this->field_->A_right_ha();
    ModuleBase::Vector3<double> nonlocal_a{0.0, 0.0, 0.0};
    if (this->inp_->td_vg_nl != 0)
    {
        nonlocal_a = vecpot;
    }
    // Evaluate observables with the accepted endpoint geometry, field and nuclear velocities.
    const std::vector<ModuleBase::Vector3<double>> ion_vel = pw::td_ion_vel(ucell, this->inp_->td_nl_motion);
    const pw::TDVelResultPW currents = vel_.calculate(current, *this->pelec, vecpot, nonlocal_a, ion_vel, current_req_, nl_orbs);

    pw::TDSpinResultPW moments;
    if (this->inp_->nspin == 4 && (this->inp_->td_out_mag != "none" || this->inp_->td_out_torque))
    {
        moments = spin_.calculate(current, *this->pelec, vecpot, nonlocal_a, ion_vel, nl_orbs);
    }

    const double dt_ha = this->field_->dt_ha();
    const int nkstot = this->kv.get_nkstot();
    ModuleIO::out_curr_spin_pw(
        currents,
        moments,
        current_req_,
        this->inp_->td_out_mag,
        this->inp_->orbital_dir,
        this->inp_->onsite_radius,
        this->inp_->td_out_torque,
        ucell.orbital_fn,
        istep,
        dt_ha,
        this->inp_->nspin,
        nkstot,
        rank,
        output_dir,
        log_);
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::out_conserv(const UnitCell &ucell, const int &istep)
{
    if (this->inp_->td_out_stat)
    {
        const double local_npsi = this->td_prop_->orbitals().wave_electrons(this->pelec->wg);
        const hsolver::TDOrthStats &stats = this->td_prop_->orbitals().orth_stats();
        const bool full_gram = this->inp_->td_orthonormal != "none";
        const pw::TDConserv values =
            pw::cal_td_conserv(local_npsi, this->chr, ucell.omega, this->pw_wfc->poolrank, stats.before, stats.after, full_gram);
        if (istep == 0)
        {
            // Use the initial wavefunction and density electron counts as references for conservation diagnostics.
            npsi0_ = values.npsi;
            nrho0_ = values.nrho;
        }
        ModuleIO::print_conserv(values, npsi0_, nrho0_, istep, full_gram, this->inp_->td_orthonormal, log_);
    }
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::out_eij(int istep)
{
    if (this->inp_->td_print_eij >= 0.0)
    {
        psi::Psi<T, Device> &current = *this->stp.template get_psi_t<T, Device>();
        hamilt::HamiltPW<T, Device> *hamilt = static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
        hamilt::HamiltHSOperator<T, Device> op(hamilt, this->pw_wfc);
        ModuleIO::print_eij_pw(
            op, this->td_prop_->orbitals(), current, this->kv, this->inp_->nspin, istep, this->niter, this->inp_->td_print_eij, log_);
    }
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::print_step(const int istep) const
{
    const int evolution_step = istep + 1;
    std::cout << " -------------------------------------------" << std::endl;
    std::cout << " STEP OF ELECTRON EVOLVE : " << evolution_step << std::endl;
    std::cout << " -------------------------------------------" << std::endl;
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::out_prop(const UnitCell &cell, const int &step, const bool &conv)
{
    const System_para &system = PARAM.globalv;
    const bool write_restart = this->inp_->out_chg[0] != -1;
    const std::string prefix = system.global_out_dir + this->inp_->suffix;
    const bool write_tau = XC_Functional::get_ked_flag();
    const bool writer_pool = GlobalV::MY_POOL == 0;
    const Parallel::ParaWorld pw_world = Parallel::make_pw_world();

    module_charge::write_density_restart(
        this->chr,
        *this->pw_rhod,
        cell.GT,
        prefix,
        this->inp_->nspin,
        write_restart,
        system.gamma_only_pw,
        write_tau,
        writer_pool,
        pw_world,
        &GlobalV::ofs_warning);

    this->write_iter_bands();

    ModuleIO::ctrl_iter_pw(step, this->niter, conv, this->stp, this->kv, this->pw_wfc, *this->inp_, true, log_);
}

template <typename T, typename Device>
double ESolver_KS_PW_TDDFT<T, Device>::cal_energy()
{
    return md_energy_;
}

template <typename T, typename Device>
const pw::IonAccel &ESolver_KS_PW_TDDFT<T, Device>::ion_accel() const
{
    if (!endpoint_force_)
    {
        ModuleBase::WARNING_QUIT("ion_accel", "The accepted endpoint force has not been evaluated.");
    }
    return endpoint_force_->dynamics;
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::cal_force(BaseCell &basecell, ModuleBase::matrix &force)
{
    basecell.require_kind(BaseCell::Kind::unitcell, __FUNCTION__);
    UnitCell &cell = static_cast<UnitCell &>(basecell);

    // 1) Reuse the accepted coupled force or evaluate the fixed or prescribed endpoint on demand.
    if (!endpoint_force_)
    {
        const pw::IonForce evaluated = cal_ion_force(cell, nullptr);
        endpoint_force_.reset(new pw::IonForce(evaluated));
    }

    // 2) Return the complete endpoint force and report its contributions.
    const pw::IonForce &result = *endpoint_force_;
    force = result.forces.total;
    ModuleIO::out_force_pw(&log_, cell, result.forces, this->inp_->test_force);
    if (this->inp_->td_out_stat)
    {
        ModuleIO::out_ion_stat(&log_, result.dynamics.status, result.dynamics.residual);
    }
}

template <typename T, typename Device>
const hamilt::NLOrbData<double, Device> *ESolver_KS_PW_TDDFT<T, Device>::force_proj(ModuleBase::Vector3<double> *shift) const
{
    shift->set(0.0, 0.0, 0.0);
    if (this->field_->gauge() == 1)
    {
        const ModuleBase::Vector3<double> &a = this->field_->A_right_ha();
        if (this->inp_->td_vg_nl == 2)
        {
            *shift = a;
        }
        else if (this->inp_->td_vg_nl == 1 && a != *shift)
        {
            // The Hamiltonian uses shifted projectors while the force uses unshifted ones.
            return nullptr;
        }
    }
    if (!this->inp_->vnl_in_h || this->ppcell.nkb == 0)
    {
        return nullptr;
    }
    return force_orbs(&endpoint_.orb_data());
}

template <typename T, typename Device>
pw::IonForce ESolver_KS_PW_TDDFT<T, Device>::cal_ion_force(UnitCell &cell, const ModuleBase::matrix *ion_force)
{
    ModuleBase::timer::start("ESolver_KS_PW_TDDFT", "cal_ion_force");

    // 1) Reuse finalized nuclear derivatives or evaluate the initial/prescribed endpoint.
    pw::IonForce result;
    ModuleBase::matrix hess(cell.nat, 6);
    ModuleBase::matrix rate(cell.nat, 3);
    const bool motion = this->inp_->td_nl_motion && this->inp_->vnl_in_h;
    const pw::PiDerivs *derivs = endpoint_.derivatives(motion);
    const bool have_derivs = derivs != nullptr;
    if (have_derivs)
    {
        result.pi = derivs->pi;
        if (motion)
        {
            hess = derivs->hess;
            rate = derivs->rate;
        }
    }
    else if (this->inp_->vnl_in_h)
    {
        hamilt::HamiltPW<T, Device> &hamilt = *static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
        const Real *veff = this->pelec->pot->template get_veff_smooth_data<Real>();
        const ModuleBase::Vector3<double> &a = this->field_->A_right_ha();
        ModuleBase::Vector3<double> nl_a{0.0, 0.0, 0.0};
        ModuleBase::Vector3<double> adot{0.0, 0.0, 0.0};
        if (this->field_->gauge() == 1 && this->inp_->td_vg_nl != 0)
        {
            nl_a = a;
            adot = this->field_->efield_ha() * -1.0;
        }

        const std::vector<ModuleBase::Vector3<double>> vel = pw::td_ion_vel(cell, motion);
        const std::vector<ModuleBase::Vector3<double>> tau = unitcell::cell_positions(cell);
        hamilt.bind_td_state(veff, a, vel, tau);
        const psi::Psi<T, Device> &wave = *this->stp.template get_psi_t<T, Device>();
        if (motion)
        {
            const pw::PiDerivs &derivs = endpoint_.cal_pi_derivs(
                cell,
                *this->pw_wfc,
                this->kv,
                &this->ppcell,
                wave,
                this->pelec->wg,
                nl_a,
                adot,
                vel,
                &hamilt,
                &this->td_prop_->orbitals(),
                nullptr);
            hess = derivs.hess;
            rate = derivs.rate;
            result.pi = derivs.pi;
        }
        else
        {
            const bool need_grad = current_req_.charge_total || current_req_.charge_per_k || current_req_.spin_current;
            result.pi = endpoint_.cal_pi(
                cell,
                *this->pw_wfc,
                this->kv,
                &this->ppcell,
                wave,
                this->pelec->wg,
                nl_a,
                vel,
                need_grad,
                &hamilt,
                &this->td_prop_->orbitals(),
                nullptr);
        }
    }

    // 2) Evaluate position and direct field forces using matching endpoint overlaps.
    ModuleBase::Vector3<double> shift{0.0, 0.0, 0.0};
    const hamilt::NLOrbData<double, Device> *orbs = force_proj(&shift);
    const std::vector<ModuleBase::Vector3<double>> vel = pw::td_ion_vel(cell, this->inp_->td_nl_motion);
    const ModuleBase::Vector3<double> &electric = this->field_->efield_ha();
    result.forces = this->cal_force_terms(cell, shift, vel, electric, orbs, ion_force);

    // 3) Complete the constrained acceleration and physical force from the same endpoint derivatives.
    pw::finish_ion_force(cell, hess, rate, &result);

    ModuleBase::timer::end("ESolver_KS_PW_TDDFT", "cal_ion_force");
    return result;
}

template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::init_obs(const UnitCell &, const Input_para &);
template double ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::cal_energy();
template const pw::IonAccel &ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::ion_accel() const;
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::cal_force(BaseCell &, ModuleBase::matrix &);
template pw::IonForce ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::cal_ion_force(
    UnitCell &, const ModuleBase::matrix *);
template const hamilt::NLOrbData<double, base_device::DEVICE_CPU> *
ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::force_proj(ModuleBase::Vector3<double> *) const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::init_obs(const UnitCell &, const Input_para &);
template double ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::cal_energy();
template const pw::IonAccel &ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::ion_accel() const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::cal_force(BaseCell &, ModuleBase::matrix &);
template pw::IonForce ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::cal_ion_force(
    UnitCell &, const ModuleBase::matrix *);
template const hamilt::NLOrbData<double, base_device::DEVICE_CPU> *
ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::force_proj(ModuleBase::Vector3<double> *) const;
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::out_prop(const UnitCell &, const int &, const bool &);
template const hamilt::NLOrbData<float, base_device::DEVICE_CPU> *
ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::prep_end_orbs(const UnitCell &, ModuleBase::matrix *);
template pw::TDEnergy ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::cal_end_energy(
    const UnitCell &, const ModuleBase::matrix &) const;
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::out_energy(
    const pw::TDEnergy &,
    const int &,
    const int &,
    const std::string &) const;
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::out_curr_spin(
    const UnitCell &,
    const hamilt::NLOrbData<float, base_device::DEVICE_CPU> *,
    const int &,
    const int &,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::out_conserv(const UnitCell &, const int &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::out_eij(int);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::print_step(int) const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::out_prop(const UnitCell &, const int &, const bool &);
template const hamilt::NLOrbData<double, base_device::DEVICE_CPU> *
ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::prep_end_orbs(const UnitCell &, ModuleBase::matrix *);
template pw::TDEnergy ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::cal_end_energy(
    const UnitCell &, const ModuleBase::matrix &) const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::out_energy(
    const pw::TDEnergy &,
    const int &,
    const int &,
    const std::string &) const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::out_curr_spin(
    const UnitCell &,
    const hamilt::NLOrbData<double, base_device::DEVICE_CPU> *,
    const int &,
    const int &,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::out_conserv(const UnitCell &, const int &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::out_eij(int);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::print_step(int) const;
#if defined(__CUDA) || defined(__ROCM)
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::init_obs(const UnitCell &, const Input_para &);
template double ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::cal_energy();
template const pw::IonAccel &ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::ion_accel() const;
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::cal_force(BaseCell &, ModuleBase::matrix &);
template pw::IonForce ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::cal_ion_force(
    UnitCell &, const ModuleBase::matrix *);
template const hamilt::NLOrbData<double, base_device::DEVICE_GPU> *
ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::force_proj(ModuleBase::Vector3<double> *) const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::init_obs(const UnitCell &, const Input_para &);
template double ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::cal_energy();
template const pw::IonAccel &ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::ion_accel() const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::cal_force(BaseCell &, ModuleBase::matrix &);
template pw::IonForce ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::cal_ion_force(
    UnitCell &, const ModuleBase::matrix *);
template const hamilt::NLOrbData<double, base_device::DEVICE_GPU> *
ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::force_proj(ModuleBase::Vector3<double> *) const;
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::out_prop(const UnitCell &, const int &, const bool &);
template const hamilt::NLOrbData<float, base_device::DEVICE_GPU> *
ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::prep_end_orbs(const UnitCell &, ModuleBase::matrix *);
template pw::TDEnergy ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::cal_end_energy(
    const UnitCell &, const ModuleBase::matrix &) const;
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::out_energy(
    const pw::TDEnergy &,
    const int &,
    const int &,
    const std::string &) const;
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::out_curr_spin(
    const UnitCell &,
    const hamilt::NLOrbData<float, base_device::DEVICE_GPU> *,
    const int &,
    const int &,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::out_conserv(const UnitCell &, const int &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::out_eij(int);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::print_step(int) const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::out_prop(const UnitCell &, const int &, const bool &);
template const hamilt::NLOrbData<double, base_device::DEVICE_GPU> *
ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::prep_end_orbs(const UnitCell &, ModuleBase::matrix *);
template pw::TDEnergy ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::cal_end_energy(
    const UnitCell &, const ModuleBase::matrix &) const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::out_energy(
    const pw::TDEnergy &,
    const int &,
    const int &,
    const std::string &) const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::out_curr_spin(
    const UnitCell &,
    const hamilt::NLOrbData<double, base_device::DEVICE_GPU> *,
    const int &,
    const int &,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::out_conserv(const UnitCell &, const int &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::out_eij(int);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::print_step(int) const;
#endif
} // namespace ModuleESolver
