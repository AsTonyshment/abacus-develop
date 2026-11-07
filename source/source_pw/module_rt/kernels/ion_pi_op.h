#ifndef PW_RT_ION_PI_OP_H
#define PW_RT_ION_PI_OP_H

#include "source_base/module_device/types.h"

#include <complex>

namespace pw
{
constexpr int pi_tile = 256;

/** @brief Accumulate RT-TDDFT nuclear energy derivatives from gradient overlaps and D<beta|psi>.
 * Array pointers use native device storage; offsets delimit each atom's projectors and partial holds band-tile sums.
 */
template <typename Real, typename Device>
class ion_pi_op
{
  public:
    void operator()(
        const int &nat,
        const int &nkb,
        const int &cols,
        const int &npol,
        const int *offsets,
        const double *weight,
        const std::complex<Real> *grad,
        const std::complex<Real> *db,
        double *partial,
        double *pi) const;
};

#if defined(__CUDA) || defined(__ROCM)
template <typename Real>
class ion_pi_op<Real, base_device::DEVICE_GPU>
{
  public:
    void operator()(
        const int &nat,
        const int &nkb,
        const int &cols,
        const int &npol,
        const int *offsets,
        const double *weight,
        const std::complex<Real> *grad,
        const std::complex<Real> *db,
        double *partial,
        double *pi) const;
};
#endif
} // namespace pw
#endif
