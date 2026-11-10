#ifndef PW_RT_ION_DERIV_OP_H
#define PW_RT_ION_DERIV_OP_H

#include "source_pw/module_rt/kernels/ion_pi_op.h"
#include <cstdint>

namespace pw
{
/** @brief Form the RT-TDDFT projector time derivative caused by nuclear translation. */
template <typename Real, typename Device>
class ion_phase_op
{
  public:
    void operator()(
        int npw,
        int ld,
        int nkb,
        const int *atoms,
        const Real *q,
        const Real *vel,
        const std::complex<Real> *beta,
        const std::complex<Real> *grad,
        std::complex<Real> *out) const;
};

/** @brief Reduce the RT-TDDFT velocity Hessian and acceleration-independent nuclear derivative rate. */
template <typename Real, typename Device>
class ion_deriv_op
{
  public:
    void operator()(
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
        double *sum) const;
};

#if defined(__CUDA) || defined(__ROCM)
template <typename Real>
class ion_phase_op<Real, base_device::DEVICE_GPU>
{
  public:
    void operator()(
        int npw,
        int ld,
        int nkb,
        const int *atoms,
        const Real *q,
        const Real *vel,
        const std::complex<Real> *beta,
        const std::complex<Real> *grad,
        std::complex<Real> *out) const;
};
template <typename Real>
class ion_deriv_op<Real, base_device::DEVICE_GPU>
{
  public:
    void operator()(
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
        double *sum) const;
};
#endif

template <typename Complex>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
    double ion_pair(const Complex &a, const Complex &b)
{
    return static_cast<double>(a.real()) * static_cast<double>(b.real()) + static_cast<double>(a.imag()) * static_cast<double>(b.imag());
}

template <typename Complex>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
    double ion_deriv_col(
        int atom,
        int d,
        int col,
        int nkb,
        int cols,
        int npol,
        const int *offsets,
        const double *weight,
        const Complex *coeff,
        const Complex *hpsi,
        const Complex *phase,
        const Complex *hess)
{
    // The first six outputs pack the symmetric velocity Hessian as xx, xy, xz, yy, yz, zz, followed by three rate components.
    constexpr int ai[6] = {0, 0, 0, 1, 1, 2};
    constexpr int aj[6] = {0, 1, 2, 1, 2, 2};
    const std::int64_t block = static_cast<std::int64_t>(nkb) * cols;
    double sum = 0.0;
    for (int p = offsets[atom]; p < offsets[atom + 1]; ++p)
    {
        const std::int64_t ip = static_cast<std::int64_t>(p) * cols + col;
        if (d < 6)
        {
            // The energy Hessian combines a second projector derivative paired with beta and two first derivatives paired together.
            const std::int64_t ig = block + (3 * static_cast<std::int64_t>(col) + ai[d]) * nkb + p;
            const std::int64_t ih =
                d == 0 ? static_cast<std::int64_t>(col) * nkb + p : block + (5 * static_cast<std::int64_t>(col) + d - 1) * nkb + p;
            const std::int64_t idg = (5 + aj[d]) * block + ip;
            sum += ion_pair(hess[ih], coeff[4 * block + ip]) + ion_pair(coeff[ig], coeff[idg]);
        }
        else
        {
            const int dir = d - 6;
            const std::int64_t ib = static_cast<std::int64_t>(col) * nkb + p;
            const std::int64_t ig = block + (3 * static_cast<std::int64_t>(col) + dir) * nkb + p;
            // dot(psi)=-iHpsi and the moving translation phase both contribute to each projected amplitude.
            Complex db(hpsi[ib].imag(), -hpsi[ib].real());
            Complex dg(hpsi[ig].imag(), -hpsi[ig].real());
            if (phase != nullptr)
            {
                db += phase[ib];
                dg += phase[ig];
            }
            // The minus sign converts momentum gradients into nuclear velocity derivatives of the shifted projector.
            sum -= ion_pair(dg, coeff[4 * block + ip]) + ion_pair(coeff[(5 + dir) * block + ip], db);
        }
    }
    return 2.0 * weight[col / npol] * sum;
}
} // namespace pw
#endif
