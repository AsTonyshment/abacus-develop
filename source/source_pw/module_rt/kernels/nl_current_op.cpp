#include "source_pw/module_rt/kernels/nl_current_op.h"

namespace pw
{
template <typename Real, typename Device>
void nl_current_op<Real, Device>::operator()(const int &nkb, const int &bands, const int &npol, const int &ncomp,
                                             const std::complex<Real> *proj, const std::complex<Real> *grad,
                    const std::complex<Real> *dproj, const std::complex<Real> *dgrad, double *out) const
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int group = 0; group < ncomp * bands; ++group)
    {
        out[group] = nl_current_sum(group / bands, group % bands, nkb, bands, npol, 0, 1, proj, grad, dproj, dgrad);
    }
}
template class nl_current_op<float, base_device::DEVICE_CPU>;
template class nl_current_op<double, base_device::DEVICE_CPU>;
}
