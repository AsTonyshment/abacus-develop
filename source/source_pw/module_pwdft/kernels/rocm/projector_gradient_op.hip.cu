#include "source_pw/module_pwdft/kernels/projector_gradient_op.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <thrust/complex.h>
namespace hamilt
{
template <typename Real, int Order>
__global__ void gradient_kernel(
    int npw,
    int ld,
    int nkb,
    int nbeta,
    int nq,
    Real dq,
    Real tpiba,
    const int *metadata,
    const Real *q,
    const Real *vel,
    const Real *tab,
    const Real *derivative,
    const Real *second,
    const Real *moments,
    const thrust::complex<Real> *sk,
    thrust::complex<Real> *out)
{
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw)
    {
        projector_elem<Real, thrust::complex<Real>, Order>(
            ig,
            static_cast<int>(blockIdx.y),
            npw,
            ld,
            nkb,
            nbeta,
            nq,
            dq,
            tpiba,
            metadata,
            q,
            vel,
            tab,
            derivative,
            second,
            moments,
            sk,
            out);
    }
}
template <typename Real, int Order>
void proj_eval_op<Real, base_device::DEVICE_GPU, Order>::operator()(
    int npw,
    int ld,
    int nkb,
    int nbeta,
    int nq,
    Real dq,
    Real tpiba,
    const int *metadata,
    const Real *q,
    const Real *vel,
    const Real *tab,
    const Real *derivative,
    const Real *second,
    const Real *moments,
    const std::complex<Real> *sk,
    std::complex<Real> *out) const
{
    if (npw == 0 || nkb == 0)
    {
        return;
    }
    // Compile ordinary projector calls independently of the RT-TDDFT Hessian workload.
    gradient_kernel<Real, Order><<<dim3((npw + 255) / 256, nkb), 256>>>(
        npw,
        ld,
        nkb,
        nbeta,
        nq,
        dq,
        tpiba,
        metadata,
        q,
        vel,
        tab,
        derivative,
        second,
        moments,
        reinterpret_cast<const thrust::complex<Real> *>(sk),
        reinterpret_cast<thrust::complex<Real> *>(out));
    const hipError_t error = hipGetLastError();
    if (error != hipSuccess)
    {
        throw std::runtime_error(hipGetErrorString(error));
    }
}
template class proj_eval_op<float, base_device::DEVICE_GPU, 0>;
template class proj_eval_op<float, base_device::DEVICE_GPU, 1>;
template class proj_eval_op<float, base_device::DEVICE_GPU, 2>;
template class proj_eval_op<double, base_device::DEVICE_GPU, 0>;
template class proj_eval_op<double, base_device::DEVICE_GPU, 1>;
template class proj_eval_op<double, base_device::DEVICE_GPU, 2>;
} // namespace hamilt
