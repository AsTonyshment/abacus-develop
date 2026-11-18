#ifndef PW_TD_ION_PI_PW_H
#define PW_TD_ION_PI_PW_H
#include "source_pw/module_pwdft/velocity_workspace.h"

namespace pw
{
//! Nuclear velocity derivative and its instantaneous acceleration equation.
struct PiDerivs
{
    ModuleBase::matrix pi;
    ModuleBase::matrix hess;
    ModuleBase::matrix rate;
};

//! Accumulate nuclear derivatives from explicitly supplied endpoint products.
template <typename Real, typename Device>
class TDIonPiPW
{
  public:
    using Complex = std::complex<Real>;

    //! Initialize atom metadata and zero the requested derivative sums.
    void begin(const UnitCell &cell, const std::vector<ModuleBase::Vector3<double>> &ion_vel,
               const ModuleBase::matrix &wg, const bool &moving, const bool &accel,
               const bool &pool_root, const int &ld, const int &nkb, const int &cols);
    //! Accumulate one k point; hp is in Hartree and extra contains six momentum Hessians.
    void accum_k(const UnitCell &cell, const ModulePW::PW_Basis_K &basis, const pseudopot_cell_vnl &pp,
                 const int &ik, const int &spin, const int &cols, const int &npol,
                 const bool &moving, const bool &accel, const Complex *wave, const Complex *hp,
                 const Complex *sample_beta, const Complex *grad, Complex *extra, const Complex *coeff);
    //! Reduce the accumulated derivatives and include the external-field rate.
    PiDerivs finish(const UnitCell &cell, const ModuleBase::Vector3<double> &adot,
                    const bool &accel, const bool &pool_root);

  private:
    ct::Tensor offsets_;
    ct::Tensor weights_;
    ct::Tensor sums_;
    ct::Tensor phase_q_;
    ct::Tensor phase_vel_;
    std::vector<int> host_off_;
    std::vector<Real> phase_host_;
    hamilt::VelocityWorkspace<Real, Device> h_work_;
    hamilt::VelocityWorkspace<Real, Device> r_work_;
    hamilt::VelocityWorkspace<Real, Device> q_work_;
};
}
#endif
