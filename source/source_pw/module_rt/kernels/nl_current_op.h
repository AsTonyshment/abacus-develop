#ifndef NL_CURRENT_OP_H
#define NL_CURRENT_OP_H

#include "source_base/module_device/types.h"
#include <complex>
#include <cstdint>

namespace pw
{
/** @brief Charge and Pauli current contractions in the reduced projector space. */
template <typename Real, typename Device>
class nl_current_op
{
  public:
    void operator()(const int &nkb, const int &bands, const int &npol, const int &ncomp,
                    const std::complex<Real> *proj, const std::complex<Real> *grad,
                    const std::complex<Real> *dproj, const std::complex<Real> *dgrad, double *out) const;
};
#if defined(__CUDA) || defined(__ROCM)
template <typename Real>
class nl_current_op<Real, base_device::DEVICE_GPU>
{
  public:
    void operator()(const int &nkb, const int &bands, const int &npol, const int &ncomp,
                    const std::complex<Real> *proj, const std::complex<Real> *grad,
                    const std::complex<Real> *dproj, const std::complex<Real> *dgrad, double *out) const;
};
#endif

template <typename Complex>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
double nl_vel_pair(int d, int left, int right, int p, int nkb, int cols, bool imag, const Complex *proj, const Complex *grad, const Complex *dproj, const Complex *dgrad)
{
    const int64_t block = static_cast<int64_t>(nkb) * cols;
    const Complex b = proj[static_cast<int64_t>(left) * nkb + p];
    const Complex g = grad[(3 * static_cast<int64_t>(left) + d) * nkb + p];
    const Complex db = dproj[static_cast<int64_t>(p) * cols + right];
    const Complex dg = dgrad[d * block + static_cast<int64_t>(p) * cols + right];
    if (imag)
    {
        return static_cast<double>(g.real()) * db.imag() - static_cast<double>(g.imag()) * db.real()
               + static_cast<double>(b.real()) * dg.imag() - static_cast<double>(b.imag()) * dg.real();
    }
    return static_cast<double>(g.real()) * db.real() + static_cast<double>(g.imag()) * db.imag()
           + static_cast<double>(b.real()) * dg.real() + static_cast<double>(b.imag()) * dg.imag();
}

template <typename Complex>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
double nl_current_sum(int comp, int band, int nkb, int bands, int npol,
                      int first, int stride, const Complex *proj, const Complex *grad, const Complex *dproj, const Complex *dgrad)
{
    const int cols = bands * npol;
    const int up = band * npol;
    const int dn = up + 1;
    const int d = comp < 3 ? comp : (comp - 3) / 3;
    double sum = 0.0;
    for (int p = first; p < nkb; p += stride)
    {
        if (comp < 3)
        {
            sum += nl_vel_pair(d, up, up, p, nkb, cols, false, proj, grad, dproj, dgrad);
            if (npol == 2)
            {
                sum += nl_vel_pair(d, dn, dn, p, nkb, cols, false, proj, grad, dproj, dgrad);
            }
        }
        else if (comp % 3 == 0)
        {
            sum += 0.5 * (nl_vel_pair(d, up, dn, p, nkb, cols, false, proj, grad, dproj, dgrad)
                          + nl_vel_pair(d, dn, up, p, nkb, cols, false, proj, grad, dproj, dgrad));
        }
        else if (comp % 3 == 1)
        {
            sum += 0.5 * (nl_vel_pair(d, up, dn, p, nkb, cols, true, proj, grad, dproj, dgrad)
                          - nl_vel_pair(d, dn, up, p, nkb, cols, true, proj, grad, dproj, dgrad));
        }
        else
        {
            sum += 0.5 * (nl_vel_pair(d, up, up, p, nkb, cols, false, proj, grad, dproj, dgrad)
                          - nl_vel_pair(d, dn, dn, p, nkb, cols, false, proj, grad, dproj, dgrad));
        }
    }
    return sum;
}
}
#endif
