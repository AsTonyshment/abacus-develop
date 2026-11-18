#include "source_pw/module_pwdft/nl_proj_pw.h"

#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_base/module_container/ATen/core/tensor_reserve.h"
#include <limits>
#include <algorithm>

namespace hamilt
{
template <typename Real, typename Device>
NLProjPW<Real, Device>::NLProjPW(const UnitCell &cell, const ModulePW::PW_Basis_K &basis, const pseudopot_cell_vnl &pp)
    : cell_(cell), basis_(basis), pp_(pp)
{
}

template <typename Real, typename Device>
void NLProjPW<Real, Device>::select(const std::vector<ModuleBase::Vector3<double>> &tau,
                                   const std::vector<ModuleBase::Vector3<double>> &vel,
                                   const ModuleBase::Vector3<double> &a)
{
    if (tau.size() != static_cast<size_t>(cell_.nat) || (!vel.empty() && vel.size() != tau.size()))
    {
        ModuleBase::WARNING_QUIT("NLProjPW", "Sample positions and velocities must match the atom count.");
    }
    for (size_t next = samples_.size(); next > 0; --next)
    {
        const size_t i = next - 1;
        const std::unique_ptr<Sample> &sample = samples_[i];
        if (sample->a == a && sample->tau == tau && sample->vel == vel)
        {
            active_ = sample.get();
            std::rotate(samples_.begin() + i, samples_.begin() + i + 1, samples_.end());
            return;
        }
    }
    std::unique_ptr<Sample> sample;
    constexpr size_t capacity = 4;
    if (samples_.size() < capacity)
    {
        sample.reset(new Sample);
    }
    else
    {
        // At most three distinct samples are protected: origin, operator and endpoint.
        for (size_t i = 0; i < samples_.size(); ++i)
        {
            Sample *entry = samples_[i].get();
            if (entry != origin_ && entry != active_ && entry != endpoint_)
            {
                sample = std::move(samples_[i]);
                samples_.erase(samples_.begin() + i);
                break;
            }
        }
        if (!sample)
        {
            ModuleBase::WARNING_QUIT("NLProjPW", "No unborrowed projector sample is available.");
        }
    }
    sample->tau = tau;
    sample->vel = vel;
    sample->a = a;
    sample->phase.prep(cell_, pp_.psf->phase_grid(), tau);
    sample->beta.resize(basis_.nks);
    sample->version.assign(basis_.nks, std::numeric_limits<size_t>::max());
    active_ = sample.get();
    samples_.push_back(std::move(sample));
}

template <typename Real, typename Device>
void NLProjPW<Real, Device>::begin_step()
{
    origin_ = active_;
}

template <typename Real, typename Device>
void NLProjPW<Real, Device>::end_step()
{
    origin_ = nullptr;
}

template <typename Real, typename Device>
void NLProjPW<Real, Device>::hold_endpoint()
{
    endpoint_ = active_;
}

template <typename Real, typename Device>
void NLProjPW<Real, Device>::drop_endpoint()
{
    endpoint_ = nullptr;
}

template <typename Real, typename Device>
const std::complex<Real> *NLProjPW<Real, Device>::phases(const int &ik)
{
    if (active_ == nullptr)
    {
        ModuleBase::WARNING_QUIT("NLProjPW", "Select an atomic sample before evaluating projectors.");
    }
    const int64_t size = static_cast<int64_t>(cell_.nat) * basis_.npwk[ik];
    ct::reserve_buffer<Complex, Device>(&sk_, size);
    Complex *sk = sk_.template data<Complex>();
    active_->phase.cal_sk(basis_, ik, sk);
    return sk;
}

template <typename Real, typename Device>
const std::complex<Real> *NLProjPW<Real, Device>::beta(const int &ik)
{
    if (active_ == nullptr)
    {
        ModuleBase::WARNING_QUIT("NLProjPW", "Select an atomic sample before evaluating projectors.");
    }
    std::unique_ptr<ct::Tensor> &buffer = active_->beta.at(ik);
    if (!buffer)
    {
        buffer.reset(new ct::Tensor);
    }
    const int64_t size = static_cast<int64_t>(pp_.nkb) * basis_.npwk_max;
    ct::reserve_buffer<Complex, Device>(buffer.get(), size);
    if (active_->version[ik] != pp_.table_version())
    {
        const Complex *sk = phases(ik);
        pp_.build_vnl(static_cast<Device *>(nullptr), cell_, ik, active_->a, active_->vel, sk,
                      buffer->template data<Complex>(), basis_.npwk_max);
        active_->version[ik] = pp_.table_version();
    }
    return buffer->template data<Complex>();
}

template class NLProjPW<float, base_device::DEVICE_CPU>;
template class NLProjPW<double, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class NLProjPW<float, base_device::DEVICE_GPU>;
template class NLProjPW<double, base_device::DEVICE_GPU>;
#endif
}
