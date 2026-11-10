#include "source_pw/module_rt/kernels/ion_deriv_op.h"
#include <algorithm>

namespace pw
{
template <typename Real, typename Device>
void ion_phase_op<Real, Device>::operator()(
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
    const std::int64_t block = static_cast<std::int64_t>(nkb) * ld;
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int p = 0; p < nkb; ++p)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            Real qv = Real(0);
            for (int d = 0; d < 3; ++d)
            {
                qv += q[3 * ig + d] * vel[3 * atoms[p] + d];
            }
            // Differentiate the retained exp(-i(G+k).R) translation phase for RT-TDDFT nuclear motion.
            const std::complex<Real> factor(0, -qv);
            const std::int64_t index = static_cast<std::int64_t>(p) * ld + ig;
            out[index] = factor * beta[index];
            for (int d = 0; d < 3; ++d)
            {
                out[(d + 1) * block + index] = factor * grad[d * block + index];
            }
        }
    }
}

template <typename Real, typename Device>
void ion_deriv_op<Real, Device>::operator()(
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
    const int tiles = (cols + pi_tile - 1) / pi_tile;
    const std::int64_t groups = static_cast<std::int64_t>(9) * nat * tiles;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (groups > 9)
#endif
    for (std::int64_t group = 0; group < groups; ++group)
    {
        const int tile = group % tiles;
        const int component = group / tiles;
        const int end = std::min(cols, (tile + 1) * pi_tile);
        double value = 0.0;
        for (int col = tile * pi_tile; col < end; ++col)
        {
            value += ion_deriv_col(component / 9, component % 9, col, nkb, cols, npol, offsets, weight, coeff, hpsi, phase, hess);
        }
        if (tiles == 1)
        {
            sum[component] += value;
        }
        else
        {
            partial[group] = value;
        }
    }
    // Merge disjoint band-tile sums in a fixed order to avoid concurrent writes to the same ionic component.
    if (tiles > 1)
    {
        for (int component = 0; component < 9 * nat; ++component)
        {
            for (int tile = 0; tile < tiles; ++tile)
            {
                sum[component] += partial[static_cast<std::int64_t>(component) * tiles + tile];
            }
        }
    }
}
template class ion_phase_op<float, base_device::DEVICE_CPU>;
template class ion_phase_op<double, base_device::DEVICE_CPU>;
template class ion_deriv_op<float, base_device::DEVICE_CPU>;
template class ion_deriv_op<double, base_device::DEVICE_CPU>;
} // namespace pw
