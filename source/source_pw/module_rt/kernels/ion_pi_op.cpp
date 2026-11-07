#include "source_pw/module_rt/kernels/ion_pi_op.h"

#include <algorithm>
#include <cstdint>

namespace pw
{
template <typename Real, typename Device>
void ion_pi_op<Real, Device>::operator()(
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
    const int tiles = (cols + pi_tile - 1) / pi_tile;
    const std::int64_t groups = static_cast<std::int64_t>(3) * nat * tiles;
    // Disjoint tiles allow parallel RT-TDDFT accumulation with a fixed summation order for each result.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (groups > 6)
#endif
    for (std::int64_t group = 0; group < groups; ++group)
    {
        const int tile = group % tiles;
        const int component = group / tiles;
        const int atom = component / 3;
        const int d = component % 3;
        const int end = std::min(cols, (tile + 1) * pi_tile);
        double sum = 0.0;
        for (int col = tile * pi_tile; col < end; ++col)
        {
            const double scale = -2.0 * weight[col / npol];
            for (int p = offsets[atom]; p < offsets[atom + 1]; ++p)
            {
                const std::int64_t ig = (3 * static_cast<std::int64_t>(col) + d) * nkb + p;
                const std::int64_t ip = static_cast<std::int64_t>(p) * cols + col;
                // Promote both factors before multiplication to retain precision with float projection buffers.
                const double product = static_cast<double>(grad[ig].real()) * static_cast<double>(db[ip].real()) +
                                       static_cast<double>(grad[ig].imag()) * static_cast<double>(db[ip].imag());
                sum += scale * product;
            }
        }
        if (tiles == 1)
        {
            pi[component] += sum;
        }
        else
        {
            partial[group] = sum;
        }
    }
    if (tiles > 1)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (nat > 2)
#endif
        for (int component = 0; component < 3 * nat; ++component)
        {
            double sum = 0.0;
            for (int tile = 0; tile < tiles; ++tile)
            {
                sum += partial[static_cast<std::int64_t>(component) * tiles + tile];
            }
            pi[component] += sum;
        }
    }
}

template class ion_pi_op<float, base_device::DEVICE_CPU>;
template class ion_pi_op<double, base_device::DEVICE_CPU>;
} // namespace pw
