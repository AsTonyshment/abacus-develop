#include "source_pw/module_rt/td_ion_pi_pw.h"

#include "source_base/kernels/math_kernel_op.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_hsolver/module_linalg/linear_algebra.h"
#include "source_pw/module_rt/kernels/ion_deriv_op.h"
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
        grad_.calculate(pp, cell, basis, ik, a, ion_vel, 1, grad);

        Complex *coeff = work_.project(beta, grad, psi.get_pointer(), npw, basis.npwk_max, pp->nkb, cols, 3, 1);
        // D is stored in Ry; the factor converts dE/dv to Hartree units after pool reduction.
        work_.contract(cell, *pp, kv.isk[ik], cols, npol, 3, 1, Real(0.5), basis, coeff);

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

template <typename Real, typename Device>
void TDIonPiPW<Real, Device>::prep_pi_derivs(
    const UnitCell &cell,
    const std::vector<ModuleBase::Vector3<double>> &ion_vel,
    bool moving,
    int ld,
    int nkb,
    int cols)
{
    const std::int64_t proj_size = static_cast<std::int64_t>(ld) * nkb;
    const std::int64_t orb_size = static_cast<std::int64_t>(ld) * cols;
    hsolver::linear_buffer<Complex, Device>(&beta_, 10 * proj_size);
    hsolver::linear_buffer<Complex, Device>(&hpsi_, orb_size);

    const int meta_size = cell.nat + 1 + (moving ? nkb : 0);
    hsolver::linear_buffer<int, Device>(&offsets_, meta_size);

    // Each pool rank needs atom metadata for its local RT-TDDFT projector phase derivatives.
    host_off_.resize(meta_size);
    std::vector<int> &indices = host_off_;
    int atom = 0;
    int proj = 0;
    for (int it = 0; it < cell.ntype; ++it)
    {
        for (int ia = 0; ia < cell.atoms[it].na; ++ia, ++atom)
        {
            indices[atom] = proj;
            for (int p = 0; p < cell.atoms[it].ncpp.nh; ++p, ++proj)
            {
                if (moving)
                {
                    indices[cell.nat + 1 + proj] = atom;
                }
            }
        }
    }
    indices[cell.nat] = nkb;
    base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>()(
        offsets_.template data<int>(), indices.data(), indices.size());

    if (moving)
    {
        hsolver::linear_buffer<Real, Device>(&phase_q_, 3 * ld);
        hsolver::linear_buffer<Real, Device>(&phase_vel_, 3 * cell.nat);

        base_device::memory::cast_memory_op<Real, double, Device, base_device::DEVICE_CPU>()(
            phase_vel_.template data<Real>(), reinterpret_cast<const double *>(ion_vel.data()), 3 * cell.nat);
        phase_host_.resize(3 * ld);
    }
}

template <typename Real, typename Device>
void TDIonPiPW<Real, Device>::cal_pi_derivs(
    const UnitCell &cell,
    const ModulePW::PW_Basis_K &basis,
    const K_Vectors &kv,
    pseudopot_cell_vnl *pp,
    const psi::Psi<Complex, Device> &psi,
    const ModuleBase::matrix &wg,
    const ModuleBase::Vector3<double> &a,
    const ModuleBase::Vector3<double> &adot,
    const std::vector<ModuleBase::Vector3<double>> &ion_vel,
    hsolver::HSOperator<Complex, Device> *op,
    ModuleBase::matrix *hess,
    ModuleBase::matrix *rate)
{
    ModuleBase::timer::start("TDIonPiPW", "cal_pi_derivs");

    hess->create(cell.nat, 6);
    rate->create(cell.nat, 3);

    if (ion_vel.empty() || pp->nkb == 0)
    {
        ModuleBase::timer::end("TDIonPiPW", "cal_pi_derivs");
        return;
    }

    const int nkb = pp->nkb;
    const int ld = basis.npwk_max;
    const int npol = psi.get_npol();
    const int bands = psi.get_nbands();
    const int cols = bands * npol;
    const std::int64_t proj_size = static_cast<std::int64_t>(ld) * nkb;
    const std::int64_t orb_size = static_cast<std::int64_t>(ld) * cols;

    bool moving = false;
    for (const ModuleBase::Vector3<double> &vel : ion_vel)
    {
        moving = moving || vel.x != 0.0 || vel.y != 0.0 || vel.z != 0.0;
    }
    prep_pi_derivs(cell, ion_vel, moving, ld, nkb, cols);

    const bool pool_root = basis.poolrank == 0;
    const int ncomp = pool_root ? 4 : 0;
    const std::int64_t sum_size = 9 * static_cast<std::int64_t>(cell.nat);
    double *sums = nullptr;
    double *partial = nullptr;

    if (pool_root)
    {
        // Only the pool root accumulates RT-TDDFT ionic derivatives from the reduced overlaps.
        const std::int64_t weight_size = static_cast<std::int64_t>(wg.nr) * wg.nc;
        hsolver::linear_buffer<double, Device>(&weights_, weight_size);
        base_device::memory::synchronize_memory_op<double, Device, base_device::DEVICE_CPU>()(
            weights_.template data<double>(), wg.c, weight_size);

        const int tiles = (cols + pi_tile - 1) / pi_tile;
        const std::int64_t partial_size = tiles > 1 ? sum_size * tiles : 0;
        hsolver::linear_buffer<double, Device>(&sums_, sum_size + partial_size);
        sums = sums_.template data<double>();
        partial = sums + sum_size;
        base_device::memory::set_memory_op<double, Device>()(sums, 0, sum_size);
    }

    Complex *beta = beta_.template data<Complex>();
    Complex *grad = beta + proj_size;
    Complex *extra = beta + 4 * proj_size;
    Complex *hp = hpsi_.template data<Complex>();

    const int *offsets = offsets_.template data<int>();
    const int *atoms = offsets + cell.nat + 1;
    std::vector<Real> &q = phase_host_;

    const int old_k = psi.get_current_k();
    for (int ik = 0; ik < basis.nks; ++ik)
    {
        psi.fix_k(ik);
        op->update_k(ik);
        const int npw = basis.npwk[ik];
        const Complex *wave = psi.get_pointer();

        // Apply the accepted Hamiltonian once and convert its Ry result to the Hartree time derivative.
        base_device::memory::set_memory_op<Complex, Device>()(hp, 0, orb_size);
        op->hpsi(wave, hp, psi.get_nbasis(), bands);
        const Complex half(Real(0.5), Real(0));
        ModuleBase::scal_op<Real, Device>()(orb_size, &half, hp, 1);

        pp->getvnl(static_cast<Device *>(nullptr), cell, ik, a, ion_vel, beta, ld);
        // Prepare geometry once and fill the adjacent gradient and Hessian blocks for RT-TDDFT inertia.
        grad_.calculate(pp, cell, basis, ik, a, ion_vel, 2, grad);

        // All pool ranks reduce overlaps, while only the root forms the D contractions consumed below.
        Complex *coeff = work_.project(beta, grad, wave, npw, ld, nkb, cols, 3, ncomp);
        work_.contract(cell, *pp, kv.isk[ik], cols, npol, 3, ncomp, Real(0.5), basis, coeff);

        Complex *hc = h_work_.project(beta, grad, hp, npw, ld, nkb, cols, 3, 0);
        h_work_.contract(cell, *pp, kv.isk[ik], cols, npol, 3, 0, Real(1), basis, hc);

        // Pack xx separately from the other five Hessians to reuse projection storage without repeating beta overlaps.
        const Complex *other_hess = extra + proj_size;
        Complex *qc = q_work_.project(extra, other_hess, wave, npw, ld, nkb, cols, 5, 0);
        q_work_.contract(cell, *pp, kv.isk[ik], cols, npol, 5, 0, Real(1), basis, qc);

        // Stationary nuclei have no translation-phase rate, while Hpsi and the velocity Hessian remain active in RT-TDDFT.
        Complex *rc = nullptr;
        if (moving)
        {
            // The retained translation phase depends on G+k, while A-v shifts only the projector's radial and angular factors.
            for (int ig = 0; ig < npw; ++ig)
            {
                const ModuleBase::Vector3<double> gk = basis.getgpluskcar(ik, ig) * cell.tpiba;
                for (int d = 0; d < 3; ++d)
                {
                    q[3 * ig + d] = static_cast<Real>(gk[d]);
                }
            }

            base_device::memory::synchronize_memory_op<Real, Device, base_device::DEVICE_CPU>()(
                phase_q_.template data<Real>(), q.data(), 3 * npw);

            const Real *qptr = phase_q_.template data<Real>();
            const Real *vel = phase_vel_.template data<Real>();
            ion_phase_op<Real, Device>()(npw, ld, nkb, atoms, qptr, vel, beta, grad, extra);

            const Complex *phase_grad = extra + proj_size;
            rc = r_work_.project(extra, phase_grad, wave, npw, ld, nkb, cols, 3, 0);
            r_work_.contract(cell, *pp, kv.isk[ik], cols, npol, 3, 0, Real(1), basis, rc);
        }

        if (pool_root)
        {
            const double *weights = weights_.template data<double>() + static_cast<std::int64_t>(ik) * wg.nc;
            ion_deriv_op<Real, Device>()(cell.nat, nkb, cols, npol, offsets, weights, coeff, hc, rc, qc, partial, sums);
        }
    }
    psi.fix_k(old_k);

    ModuleBase::matrix packed(cell.nat, 9);
    if (pool_root)
    {
        base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, Device>()(packed.c, sums, sum_size);
    }
    Parallel_Reduce::reduce_all(packed.c, sum_size);

    constexpr int component[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    for (int ia = 0; ia < cell.nat; ++ia)
    {
        for (int d = 0; d < 6; ++d)
        {
            (*hess)(ia, d) = packed(ia, d);
        }

        for (int i = 0; i < 3; ++i)
        {
            (*rate)(ia, i) = packed(ia, i + 6);
            // In the coupled nonlocal projector, dPi/dA is minus the nuclear velocity Hessian.
            for (int j = 0; j < 3; ++j)
            {
                (*rate)(ia, i) -= (*hess)(ia, component[i][j]) * adot[j];
            }
        }
    }

    ModuleBase::timer::end("TDIonPiPW", "cal_pi_derivs");
}

template class TDIonPiPW<float, base_device::DEVICE_CPU>;
template class TDIonPiPW<double, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class TDIonPiPW<float, base_device::DEVICE_GPU>;
template class TDIonPiPW<double, base_device::DEVICE_GPU>;
#endif
} // namespace pw
