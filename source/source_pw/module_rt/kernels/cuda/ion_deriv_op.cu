#include "source_pw/module_rt/kernels/ion_deriv_op.h"
#include <base/macros/macros.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <thrust/complex.h>

namespace pw
{
namespace
{
template <typename Real>
__global__ void phase_kernel(
    int npw,
    int ld,
    int nkb,
    const int *atoms,
    const Real *q,
    const Real *vel,
    const thrust::complex<Real> *beta,
    const thrust::complex<Real> *grad,
    thrust::complex<Real> *out)
{
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = blockIdx.y;
    if (ig >= npw)
    {
        return;
    }
    Real qv = Real(0);
    for (int d = 0; d < 3; ++d)
    {
        qv += q[3 * ig + d] * vel[3 * atoms[p] + d];
    }
    // Differentiate the retained exp(-i(G+k).R) translation phase for RT-TDDFT nuclear motion.
    const thrust::complex<Real> factor(0, -qv);
    const std::int64_t block = static_cast<std::int64_t>(nkb) * ld;
    const std::int64_t index = static_cast<std::int64_t>(p) * ld + ig;
    out[index] = factor * beta[index];
    for (int d = 0; d < 3; ++d)
    {
        out[(d + 1) * block + index] = factor * grad[d * block + index];
    }
}

template <typename Real>
__global__ void deriv_kernel(
    int nkb,
    int cols,
    int npol,
    int tiles,
    const int *offsets,
    const double *weight,
    const thrust::complex<Real> *coeff,
    const thrust::complex<Real> *hpsi,
    const thrust::complex<Real> *phase,
    const thrust::complex<Real> *hess,
    double *partial,
    double *sum)
{
    const int component = blockIdx.x;
    const int col = blockIdx.y * pi_tile + threadIdx.x;
    double value = 0.0;
    if (col < cols)
    {
        value = ion_deriv_col(component / 9, component % 9, col, nkb, cols, npol, offsets, weight, coeff, hpsi, phase, hess);
    }
    using Reduce = cub::BlockReduce<double, pi_tile>;
    __shared__ typename Reduce::TempStorage storage;
    const double total = Reduce(storage).Sum(value);
    if (threadIdx.x == 0)
    {
        if (tiles == 1)
        {
            sum[component] += total;
        }
        else
        {
            partial[static_cast<std::int64_t>(component) * tiles + blockIdx.y] = total;
        }
    }
}

// Merge disjoint band-tile sums in a fixed reduction tree to avoid atomic updates to the same ionic component.
__global__ void deriv_sum(int tiles, const double *partial, double *sum)
{
    double value = 0.0;
    for (int tile = threadIdx.x; tile < tiles; tile += blockDim.x)
    {
        value += partial[static_cast<std::int64_t>(blockIdx.x) * tiles + tile];
    }
    using Reduce = cub::BlockReduce<double, pi_tile>;
    __shared__ typename Reduce::TempStorage storage;
    const double total = Reduce(storage).Sum(value);
    if (threadIdx.x == 0)
    {
        sum[blockIdx.x] += total;
    }
}
} // namespace

template <typename Real>
void ion_phase_op<Real, base_device::DEVICE_GPU>::operator()(
    int npw,
    int ld,
    int nkb,
    const int *atoms,
    const Real *q,
    const Real *vel,
    const std::complex<Real> *beta,
    const std::complex<Real> *grad,
    std::complex<Real> *out) const
{
    if (npw == 0 || nkb == 0)
    {
        return;
    }
    const dim3 grid((npw + 255) / 256, nkb);
    const thrust::complex<Real> *b = reinterpret_cast<const thrust::complex<Real> *>(beta);
    const thrust::complex<Real> *g = reinterpret_cast<const thrust::complex<Real> *>(grad);
    thrust::complex<Real> *o = reinterpret_cast<thrust::complex<Real> *>(out);
    phase_kernel<<<grid, 256>>>(npw, ld, nkb, atoms, q, vel, b, g, o);
    CHECK_LAST_CUDA_ERROR("ion_phase_op");
}

template <typename Real>
void ion_deriv_op<Real, base_device::DEVICE_GPU>::operator()(
    int nat,
    int nkb,
    int cols,
    int npol,
    const int *offsets,
    const double *weight,
    const std::complex<Real> *coeff,
    const std::complex<Real> *hpsi,
    const std::complex<Real> *phase,
    const std::complex<Real> *hess,
    double *partial,
    double *sum) const
{
    if (nat == 0 || cols == 0)
    {
        return;
    }
    const int tiles = (cols + pi_tile - 1) / pi_tile;
    const dim3 grid(9 * nat, tiles);
    const thrust::complex<Real> *c = reinterpret_cast<const thrust::complex<Real> *>(coeff);
    const thrust::complex<Real> *h = reinterpret_cast<const thrust::complex<Real> *>(hpsi);
    const thrust::complex<Real> *r = reinterpret_cast<const thrust::complex<Real> *>(phase);
    const thrust::complex<Real> *q = reinterpret_cast<const thrust::complex<Real> *>(hess);
    deriv_kernel<<<grid, pi_tile>>>(nkb, cols, npol, tiles, offsets, weight, c, h, r, q, partial, sum);
    if (tiles > 1)
    {
        deriv_sum<<<9 * nat, pi_tile>>>(tiles, partial, sum);
    }
    CHECK_LAST_CUDA_ERROR("ion_deriv_op");
    CHECK_CUDA_SYNC();
}
template class ion_phase_op<float, base_device::DEVICE_GPU>;
template class ion_phase_op<double, base_device::DEVICE_GPU>;
template class ion_deriv_op<float, base_device::DEVICE_GPU>;
template class ion_deriv_op<double, base_device::DEVICE_GPU>;
} // namespace pw
