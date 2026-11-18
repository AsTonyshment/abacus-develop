#ifndef NL_ORB_DATA_H
#define NL_ORB_DATA_H

#include "source_base/module_container/ATen/core/tensor.h"
#include "source_cell/ionic_state.h"
#include <memory>
#include <vector>

namespace hamilt
{
//! Typed endpoint products, published only after every local k point is complete.
template <typename Real, typename Device>
class NLOrbData
{
  public:
    using Complex = std::complex<Real>;

    //! Invalidate all borrowed data while retaining allocated coefficient storage.
    void clear();
    //! Prepare storage for a new physical endpoint.
    void begin(const UnitCell &cell, const int &nks, const int &ld, const int &cols,
               const int &nkb, const int &ngrad, const ModuleBase::Vector3<double> &a,
               const std::vector<ModuleBase::Vector3<double>> &vel);
    //! Save one complete k-point block in the common B/gradient/DB layout.
    void save(const int &ik, const int64_t &count, const Complex *coeff, const Complex *beta);
    //! Publish a fully prepared endpoint.
    void publish();
    //! Validate layout and identify a matching physical operator sample.
    bool matches(const UnitCell &cell, const ModuleBase::Vector3<double> &a,
                 const std::vector<ModuleBase::Vector3<double>> &vel,
                 const int &nks, const int &ld, const int &cols, const int &nkb, const int &ngrad) const;
    //! Borrow B = beta-adjoint * psi with projectors contiguous in each spin-component column.
    const Complex *proj(const int &ik) const;
    //! Borrow the three interleaved momentum-gradient projections.
    const Complex *grad(const int &ik) const;
    //! Borrow D*B with spin-component columns contiguous, in Hartree units.
    const Complex *dproj(const int &ik) const;
    //! Borrow the three consecutive D*gradient blocks, in Hartree units.
    const Complex *dgrad(const int &ik) const;
    //! Borrow the sampled projectors.
    const Complex *beta(const int &ik) const;
    int ld() const { return ld_; }
    int cols() const { return cols_; }
    int nkb() const { return nkb_; }

  private:
    std::vector<std::unique_ptr<ct::Tensor>> coeff_;
    std::vector<const Complex *> beta_;
    std::vector<int64_t> counts_;
    std::vector<ModuleBase::Vector3<double>> tau_;
    std::vector<ModuleBase::Vector3<double>> vel_;
    ModuleBase::Vector3<double> a_{0.0, 0.0, 0.0};
    int ld_ = 0;
    int cols_ = 0;
    int nkb_ = 0;
    int ngrad_ = 0;
    bool ready_ = false;

    void require_ready() const;
    //! Read complete projection blocks after checking their stored extent.
    const Complex *part(const int &ik, const int &offset, const int &blocks) const;
};
}
#endif
