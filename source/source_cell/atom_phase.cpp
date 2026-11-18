#include "source_cell/atom_phase.h"

#include "source_base/constants.h"
#include "source_base/libm/libm.h"
#include "source_base/module_container/ATen/core/tensor_reserve.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/timer.h"
#include "source_cell/kernels/wf_op.h"

namespace unitcell
{
template <typename Real, typename Device>
void AtomPhase<Real, Device>::prep(const UnitCell &cell, const ModulePW::PW_Basis &rho,
                                     const std::vector<ModuleBase::Vector3<double>> &tau)
{
    ModuleBase::timer::start("AtomPhase", "prep");
    nr_[0] = rho.nx;
    nr_[1] = rho.ny;
    nr_[2] = rho.nz;
    ntype_ = cell.ntype;
    host_na_.resize(ntype_);
    host_tau_.resize(3 * tau.size());
    for (int it = 0; it < ntype_; ++it)
    {
        host_na_[it] = cell.atoms[it].na;
    }
    for (size_t ia = 0; ia < tau.size(); ++ia)
    {
        for (int d = 0; d < 3; ++d)
        {
            host_tau_[3 * ia + d] = tau[ia][d];
        }
    }
    const int64_t ntau = host_tau_.size();
    ct::reserve_buffer<Real, Device>(&tau_, ntau);
    ct::reserve_buffer<int, Device>(&na_, ntype_);
    base_device::memory::cast_memory_op<Real, double, Device, base_device::DEVICE_CPU>()(
        tau_.template data<Real>(), host_tau_.data(), ntau);
    base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>()(
        na_.template data<int>(), host_na_.data(), ntype_);

    // Match the ordinary structure-factor arithmetic at every sampled geometry.
    const std::complex<double> phase = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI;
    for (int d = 0; d < 3; ++d)
    {
        const int width = 2 * nr_[d] + 1;
        const int64_t count = static_cast<int64_t>(cell.nat) * width;
        host_phase_.resize(count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int ia = 0; ia < cell.nat; ++ia)
        {
            const ModuleBase::Vector3<double> gtau = cell.G * tau[ia];
            for (int n = -nr_[d]; n <= nr_[d]; ++n)
            {
                const double arg = n * gtau[d];
                host_phase_[static_cast<int64_t>(ia) * width + n + nr_[d]] = ModuleBase::libm::exp(phase * arg);
            }
        }
        ct::reserve_buffer<std::complex<Real>, Device>(&phase_[d], count);
        base_device::memory::cast_memory_op<std::complex<Real>, std::complex<double>, Device, base_device::DEVICE_CPU>()(
            phase_[d].template data<std::complex<Real>>(), host_phase_.data(), count);
    }
    ModuleBase::timer::end("AtomPhase", "prep");
}

template <typename Real, typename Device>
void AtomPhase<Real, Device>::cal_sk(const ModulePW::PW_Basis_K &basis, const int &ik, std::complex<Real> *out) const
{
    ModuleBase::timer::start("AtomPhase", "cal_sk");
    if (basis.npwk[ik] == 0)
    {
        ModuleBase::timer::end("AtomPhase", "cal_sk");
        return;
    }
    const bool gpu = base_device::get_device_type(static_cast<Device *>(nullptr)) == base_device::GpuDevice;
    const int *map = gpu ? basis.d_igl2isz_k : basis.igl2isz_k;
    const int *grid = gpu ? basis.d_is2fftixy : basis.is2fftixy;
    const int nx = 2 * nr_[0] + 1;
    const int ny = 2 * nr_[1] + 1;
    const int nz = 2 * nr_[2] + 1;
    hamilt::cal_sk_op<Real, Device>()(
        nullptr, ik, ntype_, basis.nx, basis.ny, basis.nz, nr_[0], nr_[1], nr_[2], basis.npwk[ik],
        basis.npwk_max, basis.fftny, nx, ny, nz, na_.template data<int>(), map, grid, ModuleBase::TWO_PI,
        basis.template get_kvec_c_data<Real>(), tau_.template data<Real>(), phase_[0].template data<std::complex<Real>>(),
        phase_[1].template data<std::complex<Real>>(), phase_[2].template data<std::complex<Real>>(), out);
    ModuleBase::timer::end("AtomPhase", "cal_sk");
}

template class AtomPhase<float, base_device::DEVICE_CPU>;
template class AtomPhase<double, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class AtomPhase<float, base_device::DEVICE_GPU>;
template class AtomPhase<double, base_device::DEVICE_GPU>;
#endif
}
