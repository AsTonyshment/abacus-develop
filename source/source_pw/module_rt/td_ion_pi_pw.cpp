#include "source_pw/module_rt/td_ion_pi_pw.h"

#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_pw/module_rt/kernels/ion_pi_op.h"

#include <algorithm>
#include <cstdint>

namespace pw
{
template <typename Real, typename Device>
ModuleBase::matrix TDIonPiPW<Real, Device>::cal_pi(
    const UnitCell &cell,
    const ModulePW::PW_Basis_K &basis,
    const K_Vectors &kv,
    pseudopot_cell_vnl *pp,
    const psi::Psi<Complex, Device> &psi,
    const ModuleBase::matrix &wg,
    const ModuleBase::Vector3<double> &a,
    const std::vector<ModuleBase::Vector3<double>> &ion_vel)
{
    ModuleBase::timer::start("TDIonPiPW", "cal_pi");
    ModuleBase::matrix result(cell.nat, 3);
    // Empty velocities disable the velocity-dependent model; explicit zero velocities still require evaluating dE/dv.
    if (ion_vel.empty() || pp->nkb == 0)
    {
        ModuleBase::timer::end("TDIonPiPW", "cal_pi");
        return result;
    }
    const ct::DeviceType device = ct::DeviceTypeToEnum<CtDevice>::value;
    const int npol = psi.get_npol();
    const int cols = psi.get_nbands() * npol;
    const std::int64_t block = static_cast<std::int64_t>(cols) * pp->nkb;
    const std::int64_t nproj = static_cast<std::int64_t>(basis.npwk_max) * pp->nkb;
    const std::int64_t size = std::max<std::int64_t>(1, 4 * nproj);
    if (beta_.NumElements() < size || beta_.data_type() != ct::DataTypeToEnum<Complex>::value || beta_.device_type() != device)
    {
        beta_ = ct::Tensor(ct::DataTypeToEnum<Complex>::value, device, {size});
    }
    // Reuse one k-point buffer for beta followed by its three Cartesian momentum derivatives.
    Complex *beta = beta_.template data<Complex>();
    Complex *grad = beta + nproj;
    const bool gpu = device == ct::DeviceType::GpuDevice;
    const int *offsets = nullptr;
    const double *weights = wg.c;
    double *sum = result.c;
    double *partial = nullptr;
    if (basis.poolrank == 0)
    {
        // Prepare small RT-TDDFT reduction inputs once per sample and reuse them across all local k points.
        host_off_.resize(cell.nat + 1);
        host_off_[0] = 0;
        int atom = 0;
        for (int it = 0; it < cell.ntype; ++it)
        {
            for (int ia = 0; ia < cell.atoms[it].na; ++ia, ++atom)
            {
                host_off_[atom + 1] = host_off_[atom] + cell.atoms[it].ncpp.nh;
            }
        }
        const int tiles = (cols + pi_tile - 1) / pi_tile;
        const std::int64_t partial_size = tiles > 1 ? 3 * static_cast<std::int64_t>(cell.nat) * tiles : 0;
        const std::int64_t sum_size = partial_size + (gpu ? 3 * cell.nat : 0);
        if (sums_.NumElements() < sum_size)
        {
            sums_.resize({sum_size});
        }
        offsets = host_off_.data();
        partial = sums_.template data<double>();
        // CPU reduction uses the caller's arrays directly; only GPU sampling needs input and result transfers.
        if (gpu)
        {
            const std::int64_t weight_size = static_cast<std::int64_t>(wg.nr) * wg.nc;
            if (offsets_.NumElements() < cell.nat + 1)
            {
                offsets_.resize({cell.nat + 1});
            }
            if (weights_.NumElements() < weight_size)
            {
                weights_.resize({weight_size});
            }
            base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>()(
                offsets_.template data<int>(), host_off_.data(), cell.nat + 1);
            base_device::memory::synchronize_memory_op<double, Device, base_device::DEVICE_CPU>()(
                weights_.template data<double>(), wg.c, weight_size);
            offsets = offsets_.template data<int>();
            weights = weights_.template data<double>();
            sum = sums_.template data<double>();
            partial = sum + 3 * cell.nat;
            base_device::memory::set_memory_op<double, Device>()(sum, 0, 3 * cell.nat);
        }
    }
    const int old_k = psi.get_current_k();
    for (int ik = 0; ik < basis.nks; ++ik)
    {
        psi.fix_k(ik);
        const int npw = basis.npwk[ik];
        pp->getvnl(static_cast<Device *>(nullptr), cell, ik, a, ion_vel, beta, basis.npwk_max);
        grad_.calculate(pp, cell, basis, ik, a, ion_vel, grad);
        Complex *coeff = work_.project(beta, grad, psi.get_pointer(), npw, basis.npwk_max, pp->nkb, cols, 1);
        // D is stored in Ry; the factor converts dE/dv to Hartree units after pool reduction.
        work_.contract(cell, *pp, kv.isk[ik], cols, npol, 1, Real(0.5), basis, coeff);
        // Pool reduction replicates the overlaps, so only the pool root accumulates them for the final global sum.
        if (basis.poolrank != 0)
        {
            continue;
        }
        const double *weight = weights + static_cast<std::int64_t>(ik) * wg.nc;
        const Complex *grad_b = coeff + block;
        const Complex *db = coeff + 4 * block;
        ion_pi_op<Real, Device>()(cell.nat, pp->nkb, cols, npol, offsets, weight, grad_b, db, partial, sum);
    }
    psi.fix_k(old_k);
    // Transfer only the accumulated atom derivatives after every k point has contributed on its native device.
    if (basis.poolrank == 0 && gpu)
    {
        base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, Device>()(
            result.c, sums_.template data<double>(), 3 * cell.nat);
    }
    // Each pool contributes its replicated projection result once, including all k-point weights.
    Parallel_Reduce::reduce_all(result.c, 3 * cell.nat);
    ModuleBase::timer::end("TDIonPiPW", "cal_pi");
    return result;
}

template class TDIonPiPW<float, base_device::DEVICE_CPU>;
template class TDIonPiPW<double, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class TDIonPiPW<float, base_device::DEVICE_GPU>;
template class TDIonPiPW<double, base_device::DEVICE_GPU>;
#endif
} // namespace pw
