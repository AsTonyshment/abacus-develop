#include "source_pw/module_pwdft/op_pw_ekin_td.h"

#include "source_base/module_device/memory_op.h"
#include "source_base/timer.h"
#include "source_pw/module_pwdft/kernels/ekinetic_op.h"

#include <algorithm>
#include <vector>

namespace hamilt
{

template <typename T, typename Device>
TDEkineticPW<T, Device>::TDEkineticPW(const ModulePW::PW_Basis_K* wfc_basis_in, const Real tpiba_in)
    : wfc_basis(wfc_basis_in), tpiba(tpiba_in)
{
    this->classname = "TDEkineticPW";
    this->cal_type = calculation_type::pw_ekinetic_td;
}

template <typename T, typename Device>
TDEkineticPW<T, Device>::~TDEkineticPW() = default;

template <typename T, typename Device>
void TDEkineticPW<T, Device>::init(const int ik)
{
    ModuleBase::timer::start("TDEkineticPW", "init");
    this->ik = ik;
    const int npw = this->wfc_basis->npwk[ik];
    using CtDevice = typename ct::PsiToContainer<Device>::type;
    if (this->factor_.NumElements() < std::max(1, npw)
        || this->factor_.data_type() != ct::DataTypeToEnum<Real>::value
        || this->factor_.device_type() != ct::DeviceTypeToEnum<CtDevice>::value)
    {
        this->factor_ = ct::Tensor(ct::DataTypeToEnum<Real>::value,
                                  ct::DeviceTypeToEnum<CtDevice>::value, {std::max(1, npw)});
    }
    const ModuleBase::Vector3<double> a = elecstate::H_TDDFT_pw::At / 2.0;
    std::vector<Real> factor(npw);
    for (int ig = 0; ig < npw; ++ig)
    {
        const ModuleBase::Vector3<double> momentum = this->wfc_basis->getgpluskcar(ik, ig) * static_cast<double>(this->tpiba);
        factor[ig] = static_cast<Real>(2.0 * (momentum.x * a.x + momentum.y * a.y + momentum.z * a.z) + a.norm2());
    }
    if (npw > 0)
    {
        base_device::memory::synchronize_memory_op<Real, Device, base_device::DEVICE_CPU>()(
            this->factor_.template data<Real>(), factor.data(), npw);
    }
    if (this->next_op != nullptr)
    {
        this->next_op->init(ik);
    }
    ModuleBase::timer::end("TDEkineticPW", "init");
}

template <typename T, typename Device>
void TDEkineticPW<T, Device>::act(const int nbands, const int nbasis, const int npol,
                                 const T* tmpsi_in, T* tmhpsi, const int ngk_ik,
                                 const bool is_first_node) const
{
    ModuleBase::timer::start("TDEkineticPW", "act");
    // Reuse the CPU/CUDA/HIP diagonal kinetic kernel for 2*A*p + A^2.
    if (nbands > 0)
    {
        ekinetic_pw_op<Real, Device>()(this->ctx, nbands, ngk_ik, nbasis / npol, is_first_node,
                                      Real(1), this->factor_.template data<Real>(), tmhpsi, tmpsi_in);
    }
    ModuleBase::timer::end("TDEkineticPW", "act");
}

template class TDEkineticPW<std::complex<float>, base_device::DEVICE_CPU>;
template class TDEkineticPW<std::complex<double>, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class TDEkineticPW<std::complex<float>, base_device::DEVICE_GPU>;
template class TDEkineticPW<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace hamilt
