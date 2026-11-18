#include "source_pw/module_pwdft/nl_orb_data.h"
#include "source_base/tool_quit.h"
#include "source_base/module_container/ATen/core/tensor_reserve.h"
#include "source_base/module_device/memory_op.h"

namespace hamilt
{
template <typename Real, typename Device>
void NLOrbData<Real, Device>::require_ready() const
{
    if (!ready_)
    {
        ModuleBase::WARNING_QUIT("NLOrbData", "Endpoint products are not ready.");
    }
}

template <typename Real, typename Device>
void NLOrbData<Real, Device>::clear()
{
    ready_ = false;
    beta_.clear();
    counts_.clear();
}

template <typename Real, typename Device>
void NLOrbData<Real, Device>::begin(const UnitCell &cell, const int &nks, const int &ld, const int &cols,
                                  const int &nkb, const int &ngrad, const ModuleBase::Vector3<double> &a,
                                  const std::vector<ModuleBase::Vector3<double>> &vel)
{
    if (ngrad != 0 && ngrad != 3)
    {
        ModuleBase::WARNING_QUIT("NLOrbData", "Endpoint products require zero or three gradient components.");
    }
    clear();
    coeff_.resize(nks);
    beta_.assign(nks, nullptr);
    counts_.assign(nks, 0);
    tau_ = unitcell::cell_positions(cell);
    vel_ = vel;
    a_ = a;
    ld_ = ld;
    cols_ = cols;
    nkb_ = nkb;
    ngrad_ = ngrad;
}

template <typename Real, typename Device>
void NLOrbData<Real, Device>::save(const int &ik, const int64_t &count, const Complex *coeff, const Complex *beta)
{
    const int64_t block = static_cast<int64_t>(nkb_) * cols_;
    const int64_t projected = (1 + ngrad_) * block;
    if (ready_ || coeff == nullptr || beta == nullptr || beta_.size() != coeff_.size()
        || (count != projected && count != 2 * projected))
    {
        ModuleBase::WARNING_QUIT("NLOrbData", "Invalid endpoint product write.");
    }
    std::unique_ptr<ct::Tensor> &buffer = coeff_.at(ik);
    if (!buffer)
    {
        buffer.reset(new ct::Tensor);
    }
    ct::reserve_buffer<Complex, Device>(buffer.get(), count);
    Complex *target = buffer->template data<Complex>();
    base_device::memory::synchronize_memory_op<Complex, Device, Device>()(target, coeff, count);
    counts_.at(ik) = count;
    beta_.at(ik) = beta;
}

template <typename Real, typename Device>
void NLOrbData<Real, Device>::publish()
{
    if (beta_.size() != coeff_.size())
    {
        ModuleBase::WARNING_QUIT("NLOrbData", "Endpoint products have been cleared.");
    }
    for (size_t ik = 0; ik < beta_.size(); ++ik)
    {
        if (!coeff_[ik] || beta_[ik] == nullptr)
        {
            ModuleBase::WARNING_QUIT("NLOrbData", "Incomplete endpoint products.");
        }
    }
    ready_ = true;
}

template <typename Real, typename Device>
bool NLOrbData<Real, Device>::matches(const UnitCell &cell, const ModuleBase::Vector3<double> &a,
                                     const std::vector<ModuleBase::Vector3<double>> &vel,
                                     const int &nks, const int &ld, const int &cols, const int &nkb, const int &ngrad) const
{
    require_ready();
    if (nks != static_cast<int>(coeff_.size()) || ld != ld_ || cols != cols_ || nkb != nkb_ || ngrad > ngrad_)
    {
        ModuleBase::WARNING_QUIT("NLOrbData", "Incompatible endpoint product layout.");
    }
    return a == a_ && vel == vel_ && unitcell::same_positions(cell, tau_);
}

template <typename Real, typename Device>
const std::complex<Real> *NLOrbData<Real, Device>::part(const int &ik, const int &offset, const int &blocks) const
{
    require_ready();
    const int64_t block = static_cast<int64_t>(nkb_) * cols_;
    if (counts_.at(ik) < (offset + blocks) * block)
    {
        ModuleBase::WARNING_QUIT("NLOrbData", "Requested endpoint products were not evaluated on this rank.");
    }
    return coeff_.at(ik)->template data<Complex>() + offset * block;
}

template <typename Real, typename Device>
const std::complex<Real> *NLOrbData<Real, Device>::proj(const int &ik) const
{
    return part(ik, 0, 1);
}

template <typename Real, typename Device>
const std::complex<Real> *NLOrbData<Real, Device>::grad(const int &ik) const
{
    if (ngrad_ != 3)
    {
        ModuleBase::WARNING_QUIT("NLOrbData", "Endpoint gradient products were not requested.");
    }
    return part(ik, 1, 3);
}

template <typename Real, typename Device>
const std::complex<Real> *NLOrbData<Real, Device>::dproj(const int &ik) const
{
    const int offset = 1 + ngrad_;
    return part(ik, offset, 1);
}

template <typename Real, typename Device>
const std::complex<Real> *NLOrbData<Real, Device>::dgrad(const int &ik) const
{
    if (ngrad_ != 3)
    {
        ModuleBase::WARNING_QUIT("NLOrbData", "Endpoint gradient products were not requested.");
    }
    const int offset = 2 + ngrad_;
    return part(ik, offset, 3);
}

template <typename Real, typename Device>
const std::complex<Real> *NLOrbData<Real, Device>::beta(const int &ik) const
{
    require_ready();
    return beta_.at(ik);
}

template class NLOrbData<float, base_device::DEVICE_CPU>;
template class NLOrbData<double, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class NLOrbData<float, base_device::DEVICE_GPU>;
template class NLOrbData<double, base_device::DEVICE_GPU>;
#endif

}
