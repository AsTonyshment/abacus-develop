#include "source_esolver/esolver_ks_pw_tddft.h"

#include "source_base/global_variable.h"
#include "source_base/timer.h"
#include "source_estate/elecstate_pw.h"
#include "source_io/module_ctrl/ctrl_output_td_pw.h"
#include "source_io/module_parameter/parameter.h"
#include "source_pw/module_pwdft/hamilt_pw.h"
#include "source_pw/module_pwdft/force_pw.h"
#include "source_pw/module_rt/td_endpoint_pw.h"
#include "source_pw/module_rt/td_pw.h"

#include <type_traits>

namespace ModuleESolver
{
template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::init_td(UnitCell *cell, const Input_para &inp)
{
    // Bind ionic operators to the initialized PW grids and charge storage.
    const bool has_float = std::is_same<Real, float>::value;
    ion_ops_.reset(new pw::TDIonPW(cell, &this->sf, this->Pgrid, *this->pw_rhod, &this->chr, this->locpp.numeric, has_float));

    // Attach the field to the potential and create the selected propagator and accepted-state storage.
    this->pelec->pot->set_td_field_manager(this->field_);
#ifdef __MPI
    const hsolver::diag_comm_info comm(this->pw_wfc->pool_world, this->pw_wfc->poolrank, this->pw_wfc->poolnproc);
#else
    const hsolver::diag_comm_info comm(0, 1);
#endif
    const int npol = cell->get_npol();
    this->td_prop_ = pw::make_td_prop<T, Device>(*this->pw_wfc, inp, npol, this->kv.ik2iktot, comm, log_);
    this->history_.prepare(*this->pelec->pot);
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::prepare_init_hamilt(const UnitCell &ucell)
{
    // 1) Cache the unshifted momentum bound of the fixed PW basis during initialization.
    if (this->field_->current_step() == 0)
    {
        qmax_ = pw::td_momentum_bound(*this->pw_wfc, ucell.tpiba);
    }

    // 2) Cover the actual endpoint and propagation samples, including nuclear velocities in either gauge.
    const ModuleBase::Vector3<double> &left = this->field_->A_left_ha();
    const ModuleBase::Vector3<double> &right = this->field_->A_right_ha();
    const ModuleBase::Vector3<double> &prop = this->field_->A_prop_ha();
    const std::vector<ModuleBase::Vector3<double>> vel = pw::td_ion_vel(ucell, this->inp_->td_nl_motion);
    const bool shift_a = this->inp_->td_vg_nl != 0;
    const bool sample_vel = this->inp_->td_nl_motion && this->history_.ready();
    const unitcell::IonicState &saved = this->history_.ions();
    const double bound = pw::td_sample_bound(qmax_, left, right, prop, shift_a, vel, prop_ions_, saved, sample_vel);
    this->ppcell.ensure_vnl_range(ucell, bound);

    // 3) Bind the complete endpoint operator, including file-supplied A(0), before initializing orbitals.
    const Real *veff = this->pelec->pot->template get_veff_smooth_data<Real>();
    hamilt::HamiltPW<T, Device> *hamilt_pw = static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
    const std::vector<ModuleBase::Vector3<double>> tau = unitcell::cell_positions(ucell);
    hamilt_pw->bind_td_state(veff, right, vel, tau);
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

    // 3) Report the prepared field; after_scf writes propagation field files after interval acceptance.
    if (this->field_->gauge() == 1)
    {
        const ModuleBase::Vector3<double> &vecpot = this->field_->A_right_ha();
        ModuleIO::print_td_vecpot(vecpot, &log_);
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
void ESolver_KS_PW_TDDFT<T, Device>::finish_end(
    UnitCell *cell,
    const bool &need_products,
    const bool &need_accel)
{
    // Evaluate endpoint energies and derivatives with the already prepared density and potential.
    psi::Psi<T, Device> &current = *this->stp.template get_psi_t<T, Device>();
    elecstate::ElecStatePW<T, Device> &estate = *static_cast<elecstate::ElecStatePW<T, Device> *>(this->pelec);
    hamilt::HamiltPW<T, Device> &hamilt = *static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
    const ModuleBase::Vector3<double> &vecpot = this->field_->A_right_ha();
    const std::vector<ModuleBase::Vector3<double>> vel = pw::td_ion_vel(*cell, this->inp_->td_nl_motion);
    ModuleBase::Vector3<double> nl_a{0.0, 0.0, 0.0};
    ModuleBase::Vector3<double> adot{0.0, 0.0, 0.0};
    if (this->field_->gauge() == 1 && this->inp_->td_vg_nl != 0)
    {
        nl_a = vecpot;
        adot = this->field_->efield_ha() * -1.0;
    }

    const bool need_grad = current_req_.charge_total || current_req_.charge_per_k || current_req_.spin_current;
    pw::eval_td_end(
        current,
        &estate,
        *cell,
        &hamilt,
        &this->td_prop_->orbitals(),
        *this->pw_wfc,
        vecpot,
        vel,
        &endpoint_,
        need_products,
        need_accel,
        need_grad,
        nl_a,
        adot,
        &this->ppcell,
        this->inp_->imp_sol,
        this->inp_->sc_mag_switch,
        this->inp_->dft_plus_u,
        this->inp_->assume_isolated);
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::correct_end(
    UnitCell *cell, const unitcell::IonicState &ions, const pw::IonForce &force)
{
    ModuleBase::timer::start("ESolver_KS_PW_TDDFT", "correct_end");
    right_ions_ = ions;
    ion_ops_->bind(right_ions_);

    // Traveling projectors require energies and forces evaluated at the corrected velocity.
    if (this->inp_->td_nl_motion && this->inp_->vnl_in_h)
    {
        this->prepare_init_hamilt(*cell);
        const bool need_products = true;
        const bool need_accel = true;
        this->finish_end(cell, need_products, need_accel);
        const pw::PiDerivs &derivs = *endpoint_.derivatives(need_accel);
        pw::IonForce corrected;
        corrected.pi = derivs.pi;
        corrected.forces = force.forces;

        ModuleBase::Vector3<double> shift{0.0, 0.0, 0.0};
        const hamilt::NLOrbData<double, Device> *orbs = force_proj(&shift);
        const std::vector<ModuleBase::Vector3<double>> vel = pw::td_ion_vel(*cell, this->inp_->td_nl_motion);
        // The preceding full force evaluation synchronized the same endpoint orbitals in double precision.
        const psi::Psi<std::complex<double>, Device> *wave = this->stp.template get_psi_d<T, Device>();
        Forces<double, Device> ff(cell->nat);
        ff.update_nl_force(orbs, shift, vel, *this->pelec, &this->kv, this->pw_wfc, &this->sf,
                           this->ppcell, *cell, wave, &corrected.forces);
        pw::finish_ion_force(*cell, derivs.hess, derivs.rate, &corrected);
        endpoint_force_.reset(new pw::IonForce(corrected));
    }
    else
    {
        endpoint_force_.reset(new pw::IonForce(force));
    }
    ModuleBase::timer::end("ESolver_KS_PW_TDDFT", "correct_end");
}

template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::init_td(UnitCell *, const Input_para &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::prepare_init_hamilt(const UnitCell &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::prep_td_step(int);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::finish_end(UnitCell *, const bool &, const bool &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::correct_end(UnitCell *, const unitcell::IonicState &, const pw::IonForce &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::init_td(UnitCell *, const Input_para &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::prepare_init_hamilt(const UnitCell &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::prep_td_step(int);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::finish_end(UnitCell *, const bool &, const bool &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::correct_end(UnitCell *, const unitcell::IonicState &, const pw::IonForce &);
#if defined(__CUDA) || defined(__ROCM)
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::init_td(UnitCell *, const Input_para &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::prepare_init_hamilt(const UnitCell &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::prep_td_step(int);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::finish_end(UnitCell *, const bool &, const bool &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::correct_end(UnitCell *, const unitcell::IonicState &, const pw::IonForce &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::init_td(UnitCell *, const Input_para &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::prepare_init_hamilt(const UnitCell &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::prep_td_step(int);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::finish_end(UnitCell *, const bool &, const bool &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::correct_end(UnitCell *, const unitcell::IonicState &, const pw::IonForce &);
#endif
} // namespace ModuleESolver
