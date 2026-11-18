#include "source_pw/module_rt/td_ion_pi_pw.h"
#include "source_base/parallel_reduce.h"
#include "source_hsolver/module_linalg/linear_algebra.h"
#include "source_pw/module_rt/kernels/ion_deriv_op.h"
#include "source_pw/module_rt/kernels/ion_pi_op.h"
#include <algorithm>

namespace pw
{
template <typename Real, typename Device>
void TDIonPiPW<Real, Device>::begin(
    const UnitCell &cell,
    const std::vector<ModuleBase::Vector3<double>> &ion_vel,
    const ModuleBase::matrix &wg,
    const bool &moving,
    const bool &accel,
    const bool &pool_root,
    const int &ld,
    const int &nkb,
    const int &cols)
{
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
    const std::int64_t sum_size = (accel ? 12 : 3) * static_cast<std::int64_t>(cell.nat);
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
        double *sums = sums_.template data<double>();
        base_device::memory::set_memory_op<double, Device>()(sums, 0, sum_size);
    }

}

template <typename Real, typename Device>
void TDIonPiPW<Real, Device>::accum_k(
    const UnitCell &cell, const ModulePW::PW_Basis_K &basis, const pseudopot_cell_vnl &pp,
    const int &ik, const int &spin, const int &cols, const int &npol,
    const bool &moving, const bool &accel,
    const Complex *wave, const Complex *hp, const Complex *sample_beta,
    const Complex *grad, Complex *extra, const Complex *coeff)
{
    const int nkb = pp.nkb;
    const int ld = basis.npwk_max;
    const int npw = basis.npwk[ik];
    const int64_t proj_size = static_cast<int64_t>(ld) * nkb;
    const int64_t sum_size = (accel ? 12 : 3) * static_cast<int64_t>(cell.nat);
    const bool pool_root = basis.poolrank == 0;
    const int *offsets = offsets_.template data<int>();
    const int *atoms = offsets + cell.nat + 1;
    double *sums = pool_root ? sums_.template data<double>() : nullptr;
    double *partial = pool_root ? sums + sum_size : nullptr;
    std::vector<Real> &q = phase_host_;
    if (accel)
    {
        Complex *hc = h_work_.project(sample_beta, grad, hp, npw, ld, nkb, cols, 3, 0, nullptr);
        h_work_.contract(cell, pp, spin, cols, npol, 3, 0, Real(1), basis, hc, false);

        // Pack xx separately from the other five Hessians to reuse projection storage without repeating beta overlaps.
        const Complex *other_hess = extra + proj_size;
        Complex *qc = q_work_.project(extra, other_hess, wave, npw, ld, nkb, cols, 5, 0, nullptr);
        q_work_.contract(cell, pp, spin, cols, npol, 5, 0, Real(1), basis, qc, false);

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
            ion_phase_op<Real, Device>()(npw, ld, nkb, atoms, qptr, vel, sample_beta, grad, extra);

            const Complex *phase_grad = extra + proj_size;
            rc = r_work_.project(extra, phase_grad, wave, npw, ld, nkb, cols, 3, 0, nullptr);
            r_work_.contract(cell, pp, spin, cols, npol, 3, 0, Real(1), basis, rc, false);
        }

        if (pool_root)
        {
            const int bands = cols / npol;
            const double *weights = weights_.template data<double>() + static_cast<int64_t>(ik) * bands;
            ion_deriv_op<Real, Device>()(cell.nat, nkb, cols, npol, offsets, weights, coeff, hc, rc, qc, partial, sums);
        }
    }
    if (pool_root)
    {
        const int bands = cols / npol;
        const double *weights = weights_.template data<double>() + static_cast<int64_t>(ik) * bands;
        const int64_t block = static_cast<int64_t>(nkb) * cols;
        const Complex *grad_b = coeff + block;
        const Complex *db = coeff + 4 * block;
        double *pi_sum = sums + (accel ? 9 * cell.nat : 0);
        ion_pi_op<Real, Device>()(cell.nat, nkb, cols, npol, offsets, weights, grad_b, db, partial, pi_sum);
    }
}

template <typename Real, typename Device>
PiDerivs TDIonPiPW<Real, Device>::finish(const UnitCell &cell, const ModuleBase::Vector3<double> &adot,
                                       const bool &accel, const bool &pool_root)
{
    PiDerivs result;
    result.pi.create(cell.nat, 3);
    const int64_t sum_size = (accel ? 12 : 3) * static_cast<int64_t>(cell.nat);
    const double *sums = pool_root ? sums_.template data<double>() : nullptr;
    const int components = accel ? 12 : 3;
    ModuleBase::matrix packed(cell.nat, components);
    if (pool_root)
    {
        base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, Device>()(packed.c, sums, sum_size);
    }
    Parallel_Reduce::reduce_all(packed.c, sum_size);

    if (!accel)
    {
        std::copy(packed.c, packed.c + 3 * cell.nat, result.pi.c);
        return result;
    }
    result.hess.create(cell.nat, 6);
    result.rate.create(cell.nat, 3);
    ModuleBase::matrix *pi = &result.pi;
    ModuleBase::matrix *hess = &result.hess;
    ModuleBase::matrix *rate = &result.rate;
    constexpr int component[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    for (int ia = 0; ia < cell.nat; ++ia)
    {
        for (int d = 0; d < 6; ++d)
        {
            (*hess)(ia, d) = packed.c[9 * ia + d];
        }

        for (int i = 0; i < 3; ++i)
        {
            (*pi)(ia, i) = packed.c[9 * cell.nat + 3 * ia + i];
            (*rate)(ia, i) = packed.c[9 * ia + i + 6];
            // In the coupled nonlocal projector, dPi/dA is minus the nuclear velocity Hessian.
            for (int j = 0; j < 3; ++j)
            {
                (*rate)(ia, i) -= (*hess)(ia, component[i][j]) * adot[j];
            }
        }
    }

    return result;
}

template class TDIonPiPW<float, base_device::DEVICE_CPU>;
template class TDIonPiPW<double, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class TDIonPiPW<float, base_device::DEVICE_GPU>;
template class TDIonPiPW<double, base_device::DEVICE_GPU>;
#endif
} // namespace pw
