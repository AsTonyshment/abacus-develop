#include "source_pw/module_rt/td_history_pw.h"

#include "source_base/module_device/memory_op.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"

#include <algorithm>

namespace pw
{
namespace
{

template <typename T, typename Device>
void copy_wave(const psi::Psi<T, Device> &source, psi::Psi<T, Device> *target)
{
    // Copy all k points into existing storage and preserve the active cursor for later operator calls.
    const T *data = source.get_pointer() - source.get_psi_bias();
    target->set_all_psi(data, source.size());
    target->fix_k(source.get_current_k());
    target->fix_kb(source.get_current_k(), source.get_current_b());
}

} // namespace

template <typename T, typename Device>
void TDHistoryPW<T, Device>::restore(psi::Psi<T, Device> *current, elecstate::ElecState *estate, Charge *charge) const
{
    ModuleBase::timer::start("TDHistoryPW", "restore");

    copy_wave(*previous_, current);
    estate->f_en = energy_;
    estate->ekb = ekb_;
    estate->wg = wg_;
    estate->eferm = eferm_;

    // Discard the failed trial's potential correction before rebuilding a potential from the restored density.
    estate->vnew_exist = false;
    for (int spin = 0; spin < charge->nspin; ++spin)
    {
        const double *density = rho_.data() + spin * charge->nrxx;
        const double *end = density + charge->nrxx;
        std::copy(density, end, charge->rho[spin]);
    }
    charge->save_rho_before_sum_band();

    ModuleBase::timer::end("TDHistoryPW", "restore");
}

template <typename T, typename Device>
void TDHistoryPW<T, Device>::copy_from(const TDHistoryPW &source)
{
    ModuleBase::timer::start("TDHistoryPW", "copy_from");

    if (!previous_)
    {
        previous_.reset(new psi::Psi<T, Device>(source.previous()));
    }
    else
    {
        copy_wave(source.previous(), previous_.get());
    }

    // Keep independent storage while reusing the allocation across ionic trials and original steps.
    const int64_t size = source.veff_prev_.NumElements();
    const ct::DataType dtype = source.veff_prev_.data_type();
    const ct::DeviceType device = source.veff_prev_.device_type();
    if (veff_prev_.NumElements() < size || veff_prev_.data_type() != dtype || veff_prev_.device_type() != device)
    {
        const ct::TensorShape &shape = source.veff_prev_.shape();
        veff_prev_ = ct::Tensor(dtype, device, shape);
    }
    Real *target = veff_prev_.template data<Real>();
    const Real *saved = source.veff_prev_.template data<Real>();
    base_device::memory::synchronize_memory_op<Real, Device, Device>()(target, saved, size);

    // Copy the ionic state and electronic metadata belonging to the same endpoint.
    ions_ = source.ions_;
    energy_ = source.energy_;
    ekb_ = source.ekb_;
    wg_ = source.wg_;
    eferm_ = source.eferm_;
    rho_ = source.rho_;

    ModuleBase::timer::end("TDHistoryPW", "copy_from");
}

template <typename T, typename Device>
void TDHistoryPW<T, Device>::prepare(const elecstate::Potential &potential)
{
    ModuleBase::timer::start("TDHistoryPW", "prepare");

    using CtDevice = typename ct::PsiToContainer<Device>::type;
    const ct::DeviceType device = ct::DeviceTypeToEnum<CtDevice>::value;
    const ModuleBase::matrix &shape = potential.get_veff_smooth();
    const int64_t size = std::max<int64_t>(1, static_cast<int64_t>(shape.nr) * shape.nc);

    // Reuse the potential snapshot buffer while its capacity, precision and device remain suitable.
    if (veff_prev_.NumElements() < size || veff_prev_.data_type() != ct::DataTypeToEnum<Real>::value || veff_prev_.device_type() != device)
    {
        veff_prev_ = ct::Tensor(ct::DataTypeToEnum<Real>::value, device, {size});
    }

    ModuleBase::timer::end("TDHistoryPW", "prepare");
}

template <typename T, typename Device>
const typename GetTypeReal<T>::type *TDHistoryPW<T, Device>::left_potential() const
{
    return veff_prev_.template data<Real>();
}

template <typename T, typename Device>
void TDHistoryPW<T, Device>::save(
    const psi::Psi<T, Device> &current,
    const elecstate::ElecState &estate,
    const Charge &charge,
    const unitcell::IonicState &ions)
{
    ModuleBase::timer::start("TDHistoryPW", "save");

    if (!previous_)
    {
        previous_.reset(new psi::Psi<T, Device>(current));
    }
    else
    {
        copy_wave(current, previous_.get());
    }

    // Save endpoint metadata and density together so retries can restore the same electronic state.
    ions_ = ions;
    energy_ = estate.f_en;
    ekb_ = estate.ekb;
    wg_ = estate.wg;
    eferm_ = estate.eferm;

    rho_.resize(charge.nspin * charge.nrxx);
    for (int spin = 0; spin < charge.nspin; ++spin)
    {
        const double *density = charge.rho[spin];
        const double *end = density + charge.nrxx;
        double *saved = rho_.data() + spin * charge.nrxx;
        std::copy(density, end, saved);
    }

    // Copy the accepted potential on its device so the next corrector retains a fixed left endpoint.
    const ModuleBase::matrix &shape = estate.pot->get_veff_smooth();
    const Real *potential = estate.pot->template get_veff_smooth_data<Real>();
    const int size = shape.nr * shape.nc;
    if (size > 0)
    {
        base_device::memory::synchronize_memory_op<Real, Device, Device>()(veff_prev_.template data<Real>(), potential, size);
    }

    ModuleBase::timer::end("TDHistoryPW", "save");
}

template <typename T, typename Device>
const psi::Psi<T, Device> &TDHistoryPW<T, Device>::previous() const
{
    if (!previous_)
    {
        ModuleBase::WARNING_QUIT("TDHistoryPW", "No endpoint state has been saved for propagation.");
    }
    return *previous_;
}

template class TDHistoryPW<std::complex<float>, base_device::DEVICE_CPU>;
template class TDHistoryPW<std::complex<double>, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class TDHistoryPW<std::complex<float>, base_device::DEVICE_GPU>;
template class TDHistoryPW<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace pw
