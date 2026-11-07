#ifndef PW_TD_ION_PI_PW_H
#define PW_TD_ION_PI_PW_H

#include "source_cell/klist.h"
#include "source_psi/psi.h"
#include "source_pw/module_pwdft/projector_gradient.h"
#include "source_pw/module_pwdft/velocity_workspace.h"

namespace pw
{
/** @brief RT-TDDFT nuclear velocity derivatives of the nonlocal energy, with reusable one-k-point storage. */
template <typename Real, typename Device>
class TDIonPiPW
{
  private:
    using Complex = std::complex<Real>;
    using CtDevice = typename ct::PsiToContainer<Device>::type;
    ct::Tensor beta_;
    ct::Tensor offsets_{ct::DataType::DT_INT, ct::DeviceTypeToEnum<CtDevice>::value, {0}};
    ct::Tensor weights_{ct::DataType::DT_DOUBLE, ct::DeviceTypeToEnum<CtDevice>::value, {0}};
    ct::Tensor sums_{ct::DataType::DT_DOUBLE, ct::DeviceTypeToEnum<CtDevice>::value, {0}};
    hamilt::ProjectorGradient<Real, Device> grad_;
    hamilt::VelocityWorkspace<Real, Device> work_;
    std::vector<int> host_off_;

  public:
    /** @brief Return [atom][Cartesian] dE/dv in Hartree atomic units; empty velocities select zero derivative. */
    ModuleBase::matrix cal_pi(
        const UnitCell &cell,
        const ModulePW::PW_Basis_K &basis,
        const K_Vectors &kv,
        pseudopot_cell_vnl *pp,
        const psi::Psi<Complex, Device> &psi,
        const ModuleBase::matrix &wg,
        const ModuleBase::Vector3<double> &a,
        const std::vector<ModuleBase::Vector3<double>> &ion_vel);
};
} // namespace pw
#endif
