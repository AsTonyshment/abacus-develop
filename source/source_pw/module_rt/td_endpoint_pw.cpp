#include "source_pw/module_rt/td_endpoint_pw.h"

#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_cell/ionic_state.h"
#include "source_estate/elecstate_tools.h"
#include "source_estate/module_charge/chg_symm.h"
#include "source_hamilt/hamilt_hs_adapter.h"

namespace pw
{
template <typename T, typename Device>
void orth_init_pw(
    hsolver::TDOrbOps<T, Device> &orbitals,
    hsolver::HSOperator<T, Device> &op,
    psi::Psi<T, Device> &current,
    elecstate::ElecStatePW<T, Device> &estate,
    Charge &charge,
    UnitCell &cell,
    const ModulePW::PW_Basis &rhod,
    int nspin,
    int nbands,
    int iter,
    bool do_orth)
{
    ModuleBase::timer::start("pw", "orth_init_pw");
    // Padding remains zero even when physical orbital corrections are disabled.
    orbitals.zero_padding(&current);
    if (do_orth)
    {
        int changed = orbitals.orth_initial(&current, iter);
        // Occupations and density construction contain collectives across pools.
        Parallel_Reduce::reduce_all(changed);
        if (changed > 0)
        {
            orbitals.cal_band_energy(op, current, &estate.ekb);
            elecstate::calculate_weights(
                estate.ekb, estate.wg, estate.klist, estate.eferm, estate.f_en, estate.nelec_spin, nbands, estate.skip_weights);
            elecstate::calEBand(estate.ekb, estate.wg, estate.f_en);

            estate.psiToRho(current);
            module_charge::symmetrize_rho(nspin, charge, &rhod, cell.symm);
        }
    }
    ModuleBase::timer::end("pw", "orth_init_pw");
}


template <typename T, typename Device>
void prep_td_rho(const psi::Psi<T, Device> &current,
                 elecstate::ElecStatePW<T, Device> *estate,
                 Charge *charge,
                 UnitCell *cell,
                 const ModulePW::PW_Basis &rhod,
                 const bool &rho_ready,
                 const int &nspin,
                 const int &imp_sol)
{
    ModuleBase::timer::start("pw", "prep_td_rho");
    // Unconverged accepted trials recover density from the retained orbitals.
    if (!rho_ready)
    {
        estate->psiToRho(current);
        module_charge::symmetrize_rho(nspin, *charge, &rhod, cell->symm);
    }
    charge->save_rho_before_sum_band();
    // Retain the solvent update at electronic acceptance, independently of velocity correction.
    if (!rho_ready || imp_sol != 0)
    {
        estate->pot->update_from_charge(charge, cell);
    }
    ModuleBase::timer::end("pw", "prep_td_rho");
}

template <typename T, typename Device>
void eval_td_end(
    const psi::Psi<T, Device> &current,
    elecstate::ElecStatePW<T, Device> *estate,
    const UnitCell &cell,
    hamilt::HamiltPW<T, Device> *hamilt,
    hsolver::TDOrbOps<T, Device> *orbitals,
    const ModulePW::PW_Basis_K &basis,
    const ModuleBase::Vector3<double> &vecpot,
    const std::vector<ModuleBase::Vector3<double>> &ion_vel,
    TDEndpointPW<typename GetTypeReal<T>::type, Device> *endpoint,
    const bool &need_products,
    const bool &need_accel,
    const bool &need_grad,
    const ModuleBase::Vector3<double> &nl_a,
    const ModuleBase::Vector3<double> &adot,
    pseudopot_cell_vnl *pp,
    const int &imp_sol,
    const bool &sc_mag,
    const int &dft_u,
    const std::string &assume_iso)
{
    ModuleBase::timer::start("pw", "eval_td_end");
    using Real = typename GetTypeReal<T>::type;
    const Real *veff = estate->pot->template get_veff_smooth_data<Real>();
    // Rebind the accepted endpoint velocities together with its potential and field before recomputing band energies.
    const std::vector<ModuleBase::Vector3<double>> tau = unitcell::cell_positions(cell);
    hamilt->bind_td_state(veff, vecpot, ion_vel, tau);
    hamilt::HamiltHSOperator<T, Device> op(hamilt, &basis);
    if (need_products && need_accel)
    {
        endpoint->cal_pi_derivs(cell, basis, *estate->klist, pp,
                                       current, estate->wg, nl_a, adot, ion_vel, hamilt, orbitals, &estate->ekb);
    }
    else if (need_products)
    {
        endpoint->cal_pi(cell, basis, *estate->klist, pp, current, estate->wg, nl_a, ion_vel, need_grad,
                                    hamilt, orbitals, &estate->ekb);
    }
    else
    {
        orbitals->cal_band_energy(op, current, &estate->ekb);
    }
    elecstate::calEBand(estate->ekb, estate->wg, estate->f_en);
    estate->f_en.deband = estate->cal_delta_eband(cell);
    estate->f_en.descf = 0.0;

    // Do not reuse a force correction from an earlier self-consistent endpoint.
    estate->vnew_exist = false;
    estate->cal_energies(2, imp_sol, sc_mag, dft_u, assume_iso);
    ModuleBase::timer::end("pw", "eval_td_end");
}

template void orth_init_pw(
    hsolver::TDOrbOps<std::complex<float>, base_device::DEVICE_CPU> &,
    hsolver::HSOperator<std::complex<float>, base_device::DEVICE_CPU> &,
    psi::Psi<std::complex<float>, base_device::DEVICE_CPU> &,
    elecstate::ElecStatePW<std::complex<float>, base_device::DEVICE_CPU> &,
    Charge &,
    UnitCell &,
    const ModulePW::PW_Basis &,
    int,
    int,
    int,
    bool);
template void prep_td_rho(const psi::Psi<std::complex<float>, base_device::DEVICE_CPU> &,
                          elecstate::ElecStatePW<std::complex<float>, base_device::DEVICE_CPU> *,
                          Charge *, UnitCell *, const ModulePW::PW_Basis &, const bool &, const int &, const int &);
template void eval_td_end(
    const psi::Psi<std::complex<float>, base_device::DEVICE_CPU> &,
    elecstate::ElecStatePW<std::complex<float>, base_device::DEVICE_CPU> *,
    const UnitCell &,
    hamilt::HamiltPW<std::complex<float>, base_device::DEVICE_CPU> *,
    hsolver::TDOrbOps<std::complex<float>, base_device::DEVICE_CPU> *,
    const ModulePW::PW_Basis_K &,
    const ModuleBase::Vector3<double> &,
    const std::vector<ModuleBase::Vector3<double>> &,
    TDEndpointPW<float, base_device::DEVICE_CPU> *,
    const bool &,
    const bool &,
    const bool &,
    const ModuleBase::Vector3<double> &,
    const ModuleBase::Vector3<double> &,
    pseudopot_cell_vnl *,
    const int &,
    const bool &,
    const int &,
    const std::string &);
template void orth_init_pw(
    hsolver::TDOrbOps<std::complex<double>, base_device::DEVICE_CPU> &,
    hsolver::HSOperator<std::complex<double>, base_device::DEVICE_CPU> &,
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU> &,
    elecstate::ElecStatePW<std::complex<double>, base_device::DEVICE_CPU> &,
    Charge &,
    UnitCell &,
    const ModulePW::PW_Basis &,
    int,
    int,
    int,
    bool);
template void prep_td_rho(const psi::Psi<std::complex<double>, base_device::DEVICE_CPU> &,
                          elecstate::ElecStatePW<std::complex<double>, base_device::DEVICE_CPU> *,
                          Charge *, UnitCell *, const ModulePW::PW_Basis &, const bool &, const int &, const int &);
template void eval_td_end(
    const psi::Psi<std::complex<double>, base_device::DEVICE_CPU> &,
    elecstate::ElecStatePW<std::complex<double>, base_device::DEVICE_CPU> *,
    const UnitCell &,
    hamilt::HamiltPW<std::complex<double>, base_device::DEVICE_CPU> *,
    hsolver::TDOrbOps<std::complex<double>, base_device::DEVICE_CPU> *,
    const ModulePW::PW_Basis_K &,
    const ModuleBase::Vector3<double> &,
    const std::vector<ModuleBase::Vector3<double>> &,
    TDEndpointPW<double, base_device::DEVICE_CPU> *,
    const bool &,
    const bool &,
    const bool &,
    const ModuleBase::Vector3<double> &,
    const ModuleBase::Vector3<double> &,
    pseudopot_cell_vnl *,
    const int &,
    const bool &,
    const int &,
    const std::string &);
#if defined(__CUDA) || defined(__ROCM)
template void orth_init_pw(
    hsolver::TDOrbOps<std::complex<float>, base_device::DEVICE_GPU> &,
    hsolver::HSOperator<std::complex<float>, base_device::DEVICE_GPU> &,
    psi::Psi<std::complex<float>, base_device::DEVICE_GPU> &,
    elecstate::ElecStatePW<std::complex<float>, base_device::DEVICE_GPU> &,
    Charge &,
    UnitCell &,
    const ModulePW::PW_Basis &,
    int,
    int,
    int,
    bool);
template void prep_td_rho(const psi::Psi<std::complex<float>, base_device::DEVICE_GPU> &,
                          elecstate::ElecStatePW<std::complex<float>, base_device::DEVICE_GPU> *,
                          Charge *, UnitCell *, const ModulePW::PW_Basis &, const bool &, const int &, const int &);
template void eval_td_end(
    const psi::Psi<std::complex<float>, base_device::DEVICE_GPU> &,
    elecstate::ElecStatePW<std::complex<float>, base_device::DEVICE_GPU> *,
    const UnitCell &,
    hamilt::HamiltPW<std::complex<float>, base_device::DEVICE_GPU> *,
    hsolver::TDOrbOps<std::complex<float>, base_device::DEVICE_GPU> *,
    const ModulePW::PW_Basis_K &,
    const ModuleBase::Vector3<double> &,
    const std::vector<ModuleBase::Vector3<double>> &,
    TDEndpointPW<float, base_device::DEVICE_GPU> *,
    const bool &,
    const bool &,
    const bool &,
    const ModuleBase::Vector3<double> &,
    const ModuleBase::Vector3<double> &,
    pseudopot_cell_vnl *,
    const int &,
    const bool &,
    const int &,
    const std::string &);
template void orth_init_pw(
    hsolver::TDOrbOps<std::complex<double>, base_device::DEVICE_GPU> &,
    hsolver::HSOperator<std::complex<double>, base_device::DEVICE_GPU> &,
    psi::Psi<std::complex<double>, base_device::DEVICE_GPU> &,
    elecstate::ElecStatePW<std::complex<double>, base_device::DEVICE_GPU> &,
    Charge &,
    UnitCell &,
    const ModulePW::PW_Basis &,
    int,
    int,
    int,
    bool);
template void prep_td_rho(const psi::Psi<std::complex<double>, base_device::DEVICE_GPU> &,
                          elecstate::ElecStatePW<std::complex<double>, base_device::DEVICE_GPU> *,
                          Charge *, UnitCell *, const ModulePW::PW_Basis &, const bool &, const int &, const int &);
template void eval_td_end(
    const psi::Psi<std::complex<double>, base_device::DEVICE_GPU> &,
    elecstate::ElecStatePW<std::complex<double>, base_device::DEVICE_GPU> *,
    const UnitCell &,
    hamilt::HamiltPW<std::complex<double>, base_device::DEVICE_GPU> *,
    hsolver::TDOrbOps<std::complex<double>, base_device::DEVICE_GPU> *,
    const ModulePW::PW_Basis_K &,
    const ModuleBase::Vector3<double> &,
    const std::vector<ModuleBase::Vector3<double>> &,
    TDEndpointPW<double, base_device::DEVICE_GPU> *,
    const bool &,
    const bool &,
    const bool &,
    const ModuleBase::Vector3<double> &,
    const ModuleBase::Vector3<double> &,
    pseudopot_cell_vnl *,
    const int &,
    const bool &,
    const int &,
    const std::string &);
#endif
} // namespace pw
