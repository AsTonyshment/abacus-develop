#include "source_pw/module_pwdft/kernels/projector_gradient_op.h"
namespace hamilt
{
template <typename Real, typename Device, int Order>
void proj_eval_op<Real, Device, Order>::operator()(
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
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int p = 0; p < nkb; ++p)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            projector_elem<Real, std::complex<Real>, Order>(
                ig, p, npw, ld, nkb, nbeta, nq, dq, tpiba, metadata, q, vel, tab, derivative, second, moments, sk, out);
        }
    }
}
template class proj_eval_op<float, base_device::DEVICE_CPU, 0>;
template class proj_eval_op<float, base_device::DEVICE_CPU, 1>;
template class proj_eval_op<float, base_device::DEVICE_CPU, 2>;
template class proj_eval_op<double, base_device::DEVICE_CPU, 0>;
template class proj_eval_op<double, base_device::DEVICE_CPU, 1>;
template class proj_eval_op<double, base_device::DEVICE_CPU, 2>;
} // namespace hamilt
