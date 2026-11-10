#ifndef PW_PROJECTOR_GRADIENT_OP_H
#define PW_PROJECTOR_GRADIENT_OP_H
#include "source_base/module_device/types.h"
#include <cmath>
#include <complex>
#include <cstdint>
namespace hamilt
{
/** @brief Cartesian solid harmonics and their derivatives share the same polynomial recurrence. */
template <typename Real, int Order>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
    void solid_harmonic(
        int l,
        int lm,
        const Real *q,
        Real *value)
{
    constexpr int ai[6] = {0, 0, 0, 1, 1, 2};
    constexpr int aj[6] = {0, 1, 2, 1, 2, 2};
    const int index = lm - l * l;
    const int m = (index + 1) / 2;
    const bool imaginary = index > 0 && index % 2 == 0;
    constexpr int count = Order == 2 ? 10 : (Order == 1 ? 4 : 1);
    const Real r2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
    Real re[count] = {Real(1)};
    Real im[count] = {};
    for (int k = 1; k <= m; ++k)
    {
        Real nr[count] = {};
        Real ni[count] = {};
        const Real factor = Real(-(2 * k - 1));
        for (int d = 0; d < count; ++d)
        {
            nr[d] = q[0] * re[d] - q[1] * im[d];
            ni[d] = q[0] * im[d] + q[1] * re[d];
            if (d > 0 && d < 4)
            {
                nr[d] += (d == 1 ? re[0] : Real(0)) - (d == 2 ? im[0] : Real(0));
                ni[d] += (d == 1 ? im[0] : Real(0)) + (d == 2 ? re[0] : Real(0));
            }
            if (d >= 4)
            {
                const int i = ai[d - 4];
                const int j = aj[d - 4];
                nr[d] += (i == 0 ? re[j + 1] : Real(0)) + (j == 0 ? re[i + 1] : Real(0)) - (i == 1 ? im[j + 1] : Real(0)) -
                         (j == 1 ? im[i + 1] : Real(0));
                ni[d] += (i == 0 ? im[j + 1] : Real(0)) + (j == 0 ? im[i + 1] : Real(0)) + (i == 1 ? re[j + 1] : Real(0)) +
                         (j == 1 ? re[i + 1] : Real(0));
            }
        }
        for (int d = 0; d < count; ++d)
        {
            re[d] = factor * nr[d];
            im[d] = factor * ni[d];
        }
    }
    Real previous[count] = {};
    Real current[count] = {};
    for (int d = 0; d < count; ++d)
    {
        current[d] = imaginary ? im[d] : re[d];
    }
    for (int n = m + 1; n <= l; ++n)
    {
        Real next[count] = {};
        for (int d = 0; d < count; ++d)
        {
            Real zterm = q[2] * current[d];
            Real rterm = r2 * previous[d];
            if (d > 0 && d < 4)
            {
                zterm += d == 3 ? current[0] : Real(0);
                rterm += Real(2) * q[d - 1] * previous[0];
            }
            if (d >= 4)
            {
                const int i = ai[d - 4];
                const int j = aj[d - 4];
                zterm += (i == 2 ? current[j + 1] : Real(0)) + (j == 2 ? current[i + 1] : Real(0));
                rterm += Real(2) * (q[i] * previous[j + 1] + q[j] * previous[i + 1] + (i == j ? previous[0] : Real(0)));
            }
            next[d] = (Real(2 * n - 1) * zterm - Real(n + m - 1) * rterm) / Real(n - m);
        }
        for (int d = 0; d < count; ++d)
        {
            previous[d] = current[d];
            current[d] = next[d];
        }
    }
    Real norm = Real(2 * l + 1) / Real(12.56637061435917295385);
    for (int k = l - m + 1; k <= l + m; ++k)
    {
        norm /= Real(k);
    }
    norm = sqrt(norm * (m == 0 ? Real(1) : Real(2)));
    for (int d = 0; d < count; ++d)
    {
        value[d] = norm * current[d];
    }
}

/** @brief Convert solid harmonic derivatives to normalized angular derivatives away from the origin. */
template <typename Real, int Order>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
    void gradient_harmonic(
        int l,
        int lm,
        const Real *q,
        Real *value)
{
    const Real radius = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    constexpr int count = Order == 2 ? 10 : (Order == 1 ? 4 : 1);
    // The caller evaluates the origin neighborhood with the regular radial expansion.
    const Real u[3] = {q[0] / radius, q[1] / radius, q[2] / radius};
    Real solid[count] = {};
    solid_harmonic<Real, Order>(l, lm, u, solid);
    value[0] = solid[0];
    if (Order > 0)
    {
        for (int i = 0; i < 3; ++i)
        {
            value[i + 1] = (solid[i + 1] - Real(l) * solid[0] * u[i]) / radius;
        }
    }
    if (Order == 2)
    {
        constexpr int ai[6] = {0, 0, 0, 1, 1, 2};
        constexpr int aj[6] = {0, 1, 2, 1, 2, 2};
        for (int d = 0; d < 6; ++d)
        {
            const int i = ai[d];
            const int j = aj[d];
            value[4 + d] = (solid[4 + d] - Real(l) * (u[i] * solid[j + 1] + u[j] * solid[i + 1] + (i == j ? solid[0] : Real(0))) +
                               Real(l * (l + 2)) * solid[0] * u[i] * u[j]) /
                           (radius * radius);
        }
    }
}

/** @brief Device projector assembly; metadata is [type,beta,l,lm,atom].
 * Orders 0, 1 and 2 return values, three gradients and nine gradient/Hessian blocks, respectively.
 */
template <typename Real, typename Device, int Order>
class proj_eval_op
{
  public:
    void operator()(
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
        std::complex<Real> *out) const;
};

#if defined(__CUDA) || defined(__ROCM)
template <typename Real, int Order>
class proj_eval_op<Real, base_device::DEVICE_GPU, Order>
{
  public:
    void operator()(
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
        std::complex<Real> *out) const;
};
#endif

template <typename Real>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
    Real gradient_interpolate(const Real *table, const Real position)
{
    const int index = static_cast<int>(position);
    const Real x = position - Real(index);
    return table[index] * (Real(1) - x) * (Real(2) - x) * (Real(3) - x) / Real(6) +
           table[index + 1] * x * (Real(2) - x) * (Real(3) - x) / Real(2) - table[index + 2] * x * (Real(1) - x) * (Real(3) - x) / Real(2) +
           table[index + 3] * x * (Real(1) - x) * (Real(2) - x) / Real(6);
}

/** @brief Use one regular polynomial for projector values and momentum derivatives near the origin. */
template <typename Real, int Order>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
    void projector_origin(
        int l,
        int lm,
        const Real *q,
        const Real *mom,
        Real *out)
{
    const Real s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
    Real harmonic[Order == 2 ? 10 : (Order == 1 ? 4 : 1)] = {};
    solid_harmonic<Real, Order>(l, lm, q, harmonic);
    Real u = mom[4];
    Real du = Real(0);
    Real d2u = Real(0);
    // Differentiate the same polynomial in |q|^2 so ordinary projector values and RT-TDDFT derivatives stay consistent.
    for (int n = 3; n >= 0; --n)
    {
        if (Order == 2)
        {
            d2u = d2u * s + Real(2) * du;
        }
        if (Order > 0)
        {
            du = du * s + u;
        }
        u = u * s + mom[n];
    }
    out[0] = u * harmonic[0];
    if (Order > 0)
    {
        for (int i = 0; i < 3; ++i)
        {
            out[1 + i] = u * harmonic[1 + i] + Real(2) * du * q[i] * harmonic[0];
        }
    }
    if (Order == 2)
    {
        constexpr int ai[6] = {0, 0, 0, 1, 1, 2};
        constexpr int aj[6] = {0, 1, 2, 1, 2, 2};
        for (int d = 0; d < 6; ++d)
        {
            const int i = ai[d];
            const int j = aj[d];
            const Real delta = i == j ? harmonic[0] : Real(0);
            out[4 + d] = u * harmonic[4 + d] + Real(2) * du * (q[i] * harmonic[j + 1] + q[j] * harmonic[i + 1] + delta) +
                         Real(4) * d2u * q[i] * q[j] * harmonic[0];
        }
    }
}

/** @brief Share radial and angular intermediates through the requested momentum derivative order. */
template <typename Real, int Order>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
    void proj_radial(
        int l,
        int lm,
        const Real *q,
        Real dq,
        const Real *radial,
        const Real *first,
        const Real *second,
        const Real *mom,
        Real *value)
{
    const Real r2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
    const Real radius = sqrt(r2);
    // Use the dimensionless |q|rmax bound to select one regular origin representation for every derivative order.
    if (radius * mom[5] < Real(0.05))
    {
        projector_origin<Real, Order>(l, lm, q, mom, value);
        return;
    }
    constexpr int count = Order == 2 ? 10 : (Order == 1 ? 4 : 1);
    Real harmonic[count] = {};
    gradient_harmonic<Real, Order>(l, lm, q, harmonic);
    const Real position = radius / dq;
    const Real f = gradient_interpolate(radial, position);
    value[0] = f * harmonic[0];
    if (Order > 0)
    {
        const Real df = gradient_interpolate(first, position);
        for (int i = 0; i < 3; ++i)
        {
            value[1 + i] = harmonic[0] * df * q[i] / radius + f * harmonic[1 + i];
        }
        if (Order == 2)
        {
            constexpr int ai[6] = {0, 0, 0, 1, 1, 2};
            constexpr int aj[6] = {0, 1, 2, 1, 2, 2};
            const Real d2f = gradient_interpolate(second, position);
            for (int d = 0; d < 6; ++d)
            {
                const int i = ai[d];
                const int j = aj[d];
                const Real ui = q[i] / radius;
                const Real uj = q[j] / radius;
                const Real delta = i == j ? Real(1) : Real(0);
                value[4 + d] = harmonic[0] * (d2f * ui * uj + df * (delta - ui * uj) / radius) +
                               df * (ui * harmonic[j + 1] + uj * harmonic[i + 1]) + f * harmonic[4 + d];
            }
        }
    }
}

template <typename Real, typename Complex, int Order>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
    void projector_elem(
        const int ig,
        const int projector,
        const int npw,
        const int ld,
        const int nkb,
        const int nbeta,
        const int nq,
        const Real dq,
        const Real tpiba,
        const int *metadata,
        const Real *q,
        const Real *vel,
        const Real *tab,
        const Real *derivative,
        const Real *second,
        const Real *moments,
        const Complex *sk,
        Complex *out)
{
    const int *meta = metadata + 5 * static_cast<std::int64_t>(projector);
    const int l = meta[2];
    const int lm = meta[3];
    Real momentum[3];
    for (int d = 0; d < 3; ++d)
    {
        momentum[d] = q[3 * static_cast<std::int64_t>(ig) + d];
        if (vel != nullptr)
        {
            // Convert the owning atom's Hartree-unit velocity to the reciprocal-grid units of q before subtracting it.
            momentum[d] -= vel[3 * static_cast<std::int64_t>(meta[4]) + d] / tpiba;
        }
    }
    const std::int64_t radial_offset = (static_cast<std::int64_t>(meta[0]) * nbeta + meta[1]) * nq;
    const int phase_index = l % 4;
    const Complex phase =
        phase_index == 0 ? Complex(1, 0) : (phase_index == 1 ? Complex(0, -1) : (phase_index == 2 ? Complex(-1, 0) : Complex(0, 1)));
    // Keep the original translation phase: the omitted atom-common shift phase cancels in each projector outer product.
    const std::int64_t structure_index = static_cast<std::int64_t>(meta[4]) * npw + ig;
    const std::int64_t mom_offset = (static_cast<std::int64_t>(meta[0]) * nbeta + meta[1]) * 6;
    const Real physical_q[3] = {momentum[0] * tpiba, momentum[1] * tpiba, momentum[2] * tpiba};
    const Real *radial = tab + radial_offset;
    const Real *first = Order > 0 ? derivative + radial_offset : nullptr;
    const Real *radial_d2 = Order == 2 ? second + radial_offset : nullptr;
    const Real *mom = moments + mom_offset;
    constexpr int count = Order == 2 ? 10 : (Order == 1 ? 4 : 1);
    Real value[count] = {};
    proj_radial<Real, Order>(l, lm, physical_q, dq, radial, first, radial_d2, mom, value);
    // RT-TDDFT inertia consumes gradients followed by Hessians from the same angular recurrence.
    constexpr int first_out = Order == 0 ? 0 : 1;
    for (int d = first_out; d < count; ++d)
    {
        const std::int64_t offset = (static_cast<std::int64_t>(d - first_out) * nkb + projector) * ld + ig;
        out[offset] = value[d] * sk[structure_index] * phase;
    }
}
} // namespace hamilt
#endif
