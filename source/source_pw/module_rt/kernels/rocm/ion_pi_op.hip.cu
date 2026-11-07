#include "source_pw/module_rt/kernels/ion_pi_op.h"

#include <base/macros/macros.h>
#include <hip/hip_runtime.h>
#include <hipcub/block/block_reduce.hpp>
#include <thrust/complex.h>

#include <cstdint>

namespace pw
{
namespace
{
template <typename Real>
__global__ void ion_pi_kernel(
    int nkb,
    int cols,
    int npol,
    int tiles,
    const int *offsets,
    const double *weight,
    const thrust::complex<Real> *grad,
    const thrust::complex<Real> *db,
    double *partial,
    double *pi)
{
    const int component = blockIdx.x;
    const int atom = component / 3;
    const int d = component % 3;
    const int col = blockIdx.y * pi_tile + threadIdx.x;
    double sum = 0.0;
    if (col < cols)
    {
        const double scale = -2.0 * weight[col / npol];
        for (int p = offsets[atom]; p < offsets[atom + 1]; ++p)
        {
            const std::int64_t ig = (3 * static_cast<std::int64_t>(col) + d) * nkb + p;
            const std::int64_t ip = static_cast<std::int64_t>(p) * cols + col;
            const double product = static_cast<double>(grad[ig].real()) * static_cast<double>(db[ip].real()) +
                                   static_cast<double>(grad[ig].imag()) * static_cast<double>(db[ip].imag());
            sum += scale * product;
        }
    }
    using Reduce = hipcub::BlockReduce<double, pi_tile>;
    __shared__ typename Reduce::TempStorage storage;
    const double total = Reduce(storage).Sum(sum);
    if (threadIdx.x == 0)
    {
        if (tiles == 1)
        {
            pi[component] += total;
        }
        else
        {
            partial[static_cast<std::int64_t>(component) * tiles + blockIdx.y] = total;
        }
    }
}

__global__ void ion_pi_sum(int tiles, const double *partial, double *pi)
{
    double sum = 0.0;
    const std::int64_t start = static_cast<std::int64_t>(blockIdx.x) * tiles;
    for (int tile = threadIdx.x; tile < tiles; tile += blockDim.x)
    {
        sum += partial[start + tile];
    }
    using Reduce = hipcub::BlockReduce<double, pi_tile>;
    __shared__ typename Reduce::TempStorage storage;
    const double total = Reduce(storage).Sum(sum);
    if (threadIdx.x == 0)
    {
        pi[blockIdx.x] += total;
    }
}
} // namespace

template <typename Real>
void ion_pi_op<Real, base_device::DEVICE_GPU>::operator()(
    const int &nat,
    const int &nkb,
    const int &cols,
    const int &npol,
    const int *offsets,
    const double *weight,
    const std::complex<Real> *grad,
    const std::complex<Real> *db,
    double *partial,
    double *pi) const
{
    if (nat == 0 || cols == 0)
    {
        return;
    }
    const int tiles = (cols + pi_tile - 1) / pi_tile;
    const dim3 grid(3 * nat, tiles);
    const dim3 block(pi_tile);
    const thrust::complex<Real> *grad_dev = reinterpret_cast<const thrust::complex<Real> *>(grad);
    const thrust::complex<Real> *db_dev = reinterpret_cast<const thrust::complex<Real> *>(db);
    // Each RT-TDDFT atom component owns its output, avoiding floating-point atomics between tiles.
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(ion_pi_kernel<Real>), grid, block, 0, 0, nkb, cols, npol, tiles, offsets, weight, grad_dev, db_dev, partial, pi);
    if (tiles > 1)
    {
        const dim3 sum_grid(3 * nat);
        hipLaunchKernelGGL(ion_pi_sum, sum_grid, block, 0, 0, tiles, partial, pi);
    }
    hipErrcheck(hipGetLastError());
    hipCheckOnDebug();
}

template class ion_pi_op<float, base_device::DEVICE_GPU>;
template class ion_pi_op<double, base_device::DEVICE_GPU>;
} // namespace pw
