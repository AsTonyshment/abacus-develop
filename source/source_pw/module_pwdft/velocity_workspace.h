#ifndef HAMILT_VELOCITY_WORKSPACE_H
#define HAMILT_VELOCITY_WORKSPACE_H

#include "source_base/module_container/ATen/core/tensor.h"
#include "source_pw/module_pwdft/vnl_pw.h"

namespace hamilt
{
/** @brief Reusable projection storage and device contraction for a velocity operator. */
template <typename Real, typename Device>
class VelocityWorkspace
{
  private:
    using Complex = std::complex<Real>;
    ct::Tensor coefficients_;
    std::vector<Complex> host_;

  public:
    /** @brief Share beta and momentum-gradient overlaps between electronic velocity and RT-TDDFT nuclear energy derivatives. */
    Complex *project(
        const Complex *beta,
        const Complex *grad,
        const Complex *psi,
        int npw,
        int ld,
        int nkb,
        int cols,
        int ncomp);

    /** @brief Reduce four overlap blocks and contract the first ncomp blocks with D; bands counts spin-component columns. */
    // ncomp=4 serves the general velocity operator; ncomp=1 supplies D<beta|psi> for the RT-TDDFT nuclear derivative.
    void contract(
        const UnitCell &cell,
        const pseudopot_cell_vnl &pp,
        const int spin,
        const int bands,
        const int npol,
        const int ncomp,
        const Real scale,
        const ModulePW::PW_Basis_K &basis,
        Complex *buffer);
};
} // namespace hamilt
#endif
