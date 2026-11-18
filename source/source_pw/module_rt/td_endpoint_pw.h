#ifndef TD_ENDPOINT_PW_H
#define TD_ENDPOINT_PW_H

#include "source_estate/elecstate_pw.h"
#include "source_pw/module_pwdft/hamilt_pw.h"
#include "source_pw/module_rt/td_orbital_ops.h"
#include "source_pw/module_rt/td_end_sample.h"

namespace pw
{
/** @brief Clean initial orbitals and rebuild occupations and density if orthogonalization changes them. */
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
    bool do_orth);

//! Prepare orbital density and its potential once for each retained electronic interval.
template <typename T, typename Device>
void prep_td_rho(const psi::Psi<T, Device> &current,
                 elecstate::ElecStatePW<T, Device> *estate,
                 Charge *charge,
                 UnitCell *cell,
                 const ModulePW::PW_Basis &rhod,
                 const bool &rho_ready,
                 const int &nspin,
                 const int &imp_sol);

/** @brief Evaluate endpoint energies and derivatives with the prepared density and potential.
 * @param[in] current Retained endpoint orbitals, with fixed coefficients and occupations.
 * @param[in,out] estate Electronic density-dependent energies and effective potential.
 * @param[in] cell Endpoint geometry and nuclear velocities.
 * @param[in,out] hamilt Endpoint operator bindings and projector workspace.
 * @param[in,out] orbitals Band-energy workspace.
 */
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
    const std::string &assume_iso);
} // namespace pw
#endif
