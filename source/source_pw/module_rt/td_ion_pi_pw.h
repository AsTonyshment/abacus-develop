#ifndef PW_TD_ION_PI_PW_H
#define PW_TD_ION_PI_PW_H

#include "source_cell/klist.h"
#include "source_hsolver/hs_operator.h"
#include "source_psi/psi.h"
#include "source_pw/module_pwdft/projector_gradient.h"
#include "source_pw/module_pwdft/velocity_workspace.h"

namespace pw
{
//! Nuclear velocity derivatives of the RT-TDDFT nonlocal energy.
template <typename Real, typename Device>
class TDIonPiPW
{
  private:
    using Complex = std::complex<Real>;
    using CtDevice = typename ct::PsiToContainer<Device>::type;

    ct::Tensor beta_; // Projectors, momentum gradients and Hessians for one k point.
    ct::Tensor offsets_{ct::DataType::DT_INT, ct::DeviceTypeToEnum<CtDevice>::value, {0}};    // Atom offsets and projector owners.
    ct::Tensor weights_{ct::DataType::DT_DOUBLE, ct::DeviceTypeToEnum<CtDevice>::value, {0}}; // Occupation and k-point weights.
    ct::Tensor sums_{ct::DataType::DT_DOUBLE, ct::DeviceTypeToEnum<CtDevice>::value, {0}};    // Atom sums and band-tile partial sums.
    hamilt::ProjectorGradient<Real, Device> grad_;                                            // Momentum derivative evaluator.
    hamilt::VelocityWorkspace<Real, Device> work_; // Wavefunction projections and D contractions.
    std::vector<int> host_off_;                    // Host atom offsets and projector owners.
    ct::Tensor hpsi_;                              // Hpsi in Hartree units.

    // Momenta and nuclear velocities for translation-phase derivatives, with host momentum staging.
    ct::Tensor phase_q_;
    ct::Tensor phase_vel_;
    std::vector<Real> phase_host_;

    // Separate projections of Hpsi, translation-phase derivatives and momentum Hessians.
    hamilt::VelocityWorkspace<Real, Device> h_work_;
    hamilt::VelocityWorkspace<Real, Device> r_work_;
    hamilt::VelocityWorkspace<Real, Device> q_work_;

    /**
     * @brief Prepare derivative buffers and atom indices.
     * @param[in] cell Atom types and projector counts.
     * @param[in] ion_vel Nuclear velocities in type-major order and Hartree atomic units.
     * @param[in] moving Whether translation-phase derivatives are needed.
     * @param[in] ld Plane-wave leading dimension.
     * @param[in] nkb Total projector count.
     * @param[in] cols Band count times spinor-component count.
     */
    void prep_pi_derivs(
        const UnitCell &cell,
        const std::vector<ModuleBase::Vector3<double>> &ion_vel,
        bool moving,
        int ld,
        int nkb,
        int cols);

  public:
    /**
     * @brief Compute the nuclear velocity derivative Pi = dE/dv.
     * @param[in] cell Atomic geometry and pseudopotentials.
     * @param[in] basis Wavefunction plane-wave basis.
     * @param[in] kv K-point spin indices.
     * @param[in,out] pp Nonlocal projectors and radial-table caches.
     * @param[in] psi Electronic wavefunctions.
     * @param[in] wg Occupations including k-point weights, indexed by (k point, band).
     * @param[in] a Nonlocal momentum shift in Hartree atomic units.
     * @param[in] ion_vel Nuclear velocities in type-major order and Hartree atomic units.
     * @return Pi(atom, Cartesian component) in Hartree atomic units.
     */
    ModuleBase::matrix cal_pi(
        const UnitCell &cell,
        const ModulePW::PW_Basis_K &basis,
        const K_Vectors &kv,
        pseudopot_cell_vnl *pp,
        const psi::Psi<Complex, Device> &psi,
        const ModuleBase::matrix &wg,
        const ModuleBase::Vector3<double> &a,
        const std::vector<ModuleBase::Vector3<double>> &ion_vel);

    /**
     * @brief Compute Q and g in dPi/dt = g + Q*accel, in Hartree atomic units.
     * @param[in] cell Atomic geometry and pseudopotentials.
     * @param[in] basis Wavefunction plane-wave basis.
     * @param[in] kv K-point spin indices.
     * @param[in,out] pp Nonlocal projectors and radial-table caches.
     * @param[in] psi Electronic wavefunctions.
     * @param[in] wg Occupations including k-point weights, indexed by (k point, band).
     * @param[in] a Nonlocal momentum shift in Hartree atomic units.
     * @param[in] adot Time derivative of a in Hartree atomic units.
     * @param[in] ion_vel Nuclear velocities in type-major order and Hartree atomic units.
     * @param[in,out] op Hamiltonian applied to psi, updated for each k point.
     * @param[out] hess Velocity Hessian Q(nat, 6), ordered xx, xy, xz, yy, yz, zz.
     * @param[out] rate Acceleration-independent rate g(nat, 3), ordered x, y, z.
     */
    void cal_pi_derivs(
        const UnitCell &cell,
        const ModulePW::PW_Basis_K &basis,
        const K_Vectors &kv,
        pseudopot_cell_vnl *pp,
        const psi::Psi<Complex, Device> &psi,
        const ModuleBase::matrix &wg,
        const ModuleBase::Vector3<double> &a,
        const ModuleBase::Vector3<double> &adot,
        const std::vector<ModuleBase::Vector3<double>> &ion_vel,
        hsolver::HSOperator<Complex, Device> *op,
        ModuleBase::matrix *hess,
        ModuleBase::matrix *rate);
};
} // namespace pw
#endif
