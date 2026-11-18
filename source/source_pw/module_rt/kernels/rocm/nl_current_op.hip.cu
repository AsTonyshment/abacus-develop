#include "source_pw/module_rt/kernels/nl_current_op.h"
#include <hipcub/hipcub.hpp>
#include <hip/hip_runtime.h>
#include <thrust/complex.h>

namespace pw
{
namespace
{
template <typename Real>
__global__ void nl_current_kernel(int nkb, int bands, int npol, const thrust::complex<Real> *proj,
                                  const thrust::complex<Real> *grad, const thrust::complex<Real> *dproj,
                                  const thrust::complex<Real> *dgrad, double *out)
{
    const int group = blockIdx.x;
    const double value = nl_current_sum(group / bands, group % bands, nkb, bands, npol, threadIdx.x, blockDim.x, proj, grad, dproj, dgrad);
    using Reduce = hipcub::BlockReduce<double, 256>;
    __shared__ typename Reduce::TempStorage storage;
    const double sum = Reduce(storage).Sum(value);
    if (threadIdx.x == 0)
    {
        out[group] = sum;
    }
}
}
template <typename Real>
void nl_current_op<Real, base_device::DEVICE_GPU>::operator()(const int &nkb, const int &bands, const int &npol, const int &ncomp,
                                                             const std::complex<Real> *proj, const std::complex<Real> *grad,
                    const std::complex<Real> *dproj, const std::complex<Real> *dgrad, double *out) const
{
    if (bands > 0)
    {
        const thrust::complex<Real> *b = reinterpret_cast<const thrust::complex<Real> *>(proj);
        const thrust::complex<Real> *g = reinterpret_cast<const thrust::complex<Real> *>(grad);
        const thrust::complex<Real> *db = reinterpret_cast<const thrust::complex<Real> *>(dproj);
        const thrust::complex<Real> *dg = reinterpret_cast<const thrust::complex<Real> *>(dgrad);
        nl_current_kernel<<<ncomp * bands, 256>>>(nkb, bands, npol, b, g, db, dg, out);
    }
}
template class nl_current_op<float, base_device::DEVICE_GPU>;
template class nl_current_op<double, base_device::DEVICE_GPU>;
}
