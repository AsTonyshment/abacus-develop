#ifndef PW_PROJECTOR_GRADIENT_H
#define PW_PROJECTOR_GRADIENT_H
#include "source_base/module_container/ATen/core/tensor.h"
#include "source_pw/module_pwdft/vnl_pw.h"

namespace hamilt
{
/** @brief One-k-point gradient workspace with versioned radial tables. */
template <typename Real, typename Device>
class ProjectorGradient
{
  private:
    using Complex = std::complex<Real>;
    ct::Tensor radial_;
    ct::Tensor derivative_;
    ct::Tensor second_;
    ct::Tensor moments_;
    ct::Tensor metadata_;
    ct::Tensor momentum_;
    ct::Tensor structure_;
    ct::Tensor vel_;
    size_t version_ = 0;
    size_t hess_ver_ = 0;
    std::vector<Real> host_q_;
    std::vector<int> host_metadata_;

    template <typename Value>
    void reserve(ct::Tensor *tensor, const int64_t count);

  public:
    /** @brief Evaluate derivatives with explicitly sampled device phases. */
    void cal_sample(pseudopot_cell_vnl *pp, const UnitCell &cell, const ModulePW::PW_Basis_K &basis,
                    const int &ik, const ModuleBase::Vector3<double> &a,
                    const std::vector<ModuleBase::Vector3<double>> &vel, const int &order,
                    const Complex *sk, Complex *output);

    /** @brief Evaluate momentum derivatives at G+k+A-v_I for the current operator.
     * @param ion_vel RT-TDDFT nuclear velocities in type-major order and Hartree atomic units; non-RT callers pass an empty vector.
     * @param order One returns x,y,z; two returns x,y,z,xx,xy,xz,yy,yz,zz for RT-TDDFT inertia.
     */
    void calculate(
        pseudopot_cell_vnl *pp,
        const UnitCell &cell,
        const ModulePW::PW_Basis_K &basis,
        const int &ik,
        const ModuleBase::Vector3<double> &a,
        const std::vector<ModuleBase::Vector3<double>> &ion_vel,
        const int &order,
        Complex *output);
};
} // namespace hamilt
#endif
