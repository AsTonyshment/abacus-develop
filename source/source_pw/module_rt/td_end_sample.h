#ifndef TD_END_SAMPLE_H
#define TD_END_SAMPLE_H
#include "source_pw/module_rt/td_ion_pi_pw.h"
#include "source_pw/module_rt/td_orbital_ops.h"
#include "source_pw/module_pwdft/hamilt_pw.h"
#include "source_pw/module_pwdft/nl_orb_data.h"
#include "source_pw/module_pwdft/projector_gradient.h"

namespace pw
{
//! Own endpoint products and coordinate their single-pass evaluation.
template <typename Real, typename Device>
class TDEndpointPW
{
  public:
    using Complex = std::complex<Real>;

    //! Clear borrowed products and release their projector protection together.
    void clear_sample(hamilt::HamiltPW<Complex, Device> *hamilt);
    //! Borrow completed derivatives; nullptr requests evaluation of the missing order.
    const PiDerivs *derivatives(const bool &need_accel) const
    {
        return ready_ && (!need_accel || derivs_.hess.nr > 0) ? &derivs_ : nullptr;
    }
    //! Borrow products until the next endpoint evaluation or invalidation.
    const hamilt::NLOrbData<Real, Device> &orb_data() const { return orbs_; }
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
     * @param[in] need_grad Request momentum-gradient overlaps for current observables.
     * @param[in,out] hamilt Endpoint Hamiltonian and sampled projectors.
     * @param[in,out] orbitals Shared band-product workspace.
     * @param[out] energies Optional endpoint band energies; nullptr retains the existing values.
     * @return Pi(atom, Cartesian component) in Hartree atomic units.
     */
    const ModuleBase::matrix &cal_pi(
        const UnitCell &cell,
        const ModulePW::PW_Basis_K &basis,
        const K_Vectors &kv,
        pseudopot_cell_vnl *pp,
        const psi::Psi<Complex, Device> &psi,
        const ModuleBase::matrix &wg,
        const ModuleBase::Vector3<double> &a,
        const std::vector<ModuleBase::Vector3<double>> &ion_vel,
        const bool &need_grad,
        hamilt::HamiltPW<Complex, Device> *hamilt,
        hsolver::TDOrbOps<Complex, Device> *orbitals,
        ModuleBase::matrix *energies);

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
     * @param[in,out] hamilt Endpoint Hamiltonian and sampled projectors.
     * @param[in,out] orbitals Shared band-product workspace.
     * @param[out] energies Optional endpoint band energies; nullptr retains the existing values.
     * @return Pi, velocity Hessian Q(xx,xy,xz,yy,yz,zz), and acceleration-independent rate g(x,y,z).
     */
    const PiDerivs &cal_pi_derivs(
        const UnitCell &cell,
        const ModulePW::PW_Basis_K &basis,
        const K_Vectors &kv,
        pseudopot_cell_vnl *pp,
        const psi::Psi<Complex, Device> &psi,
        const ModuleBase::matrix &wg,
        const ModuleBase::Vector3<double> &a,
        const ModuleBase::Vector3<double> &adot,
        const std::vector<ModuleBase::Vector3<double>> &ion_vel,
        hamilt::HamiltPW<Complex, Device> *hamilt,
        hsolver::TDOrbOps<Complex, Device> *orbitals,
        ModuleBase::matrix *energies);

  private:
    ct::Tensor hpsi_;
    ct::Tensor deriv_;
    hamilt::ProjectorGradient<Real, Device> grad_;
    hamilt::VelocityWorkspace<Real, Device> work_;
    hamilt::NLOrbData<Real, Device> orbs_;
    TDIonPiPW<Real, Device> pi_;
    PiDerivs derivs_;
    bool ready_ = false;

    void evaluate(const UnitCell &cell, const ModulePW::PW_Basis_K &basis, const K_Vectors &kv,
                     pseudopot_cell_vnl *pp, const psi::Psi<Complex, Device> &psi,
                     const ModuleBase::matrix &wg, const ModuleBase::Vector3<double> &a,
                     const ModuleBase::Vector3<double> &adot,
                     const std::vector<ModuleBase::Vector3<double>> &ion_vel,
                     const bool &accel, const bool &need_grad,
                     hamilt::HamiltPW<Complex, Device> *hamilt,
                     hsolver::TDOrbOps<Complex, Device> *orbitals, ModuleBase::matrix *energies);
};
}
#endif
