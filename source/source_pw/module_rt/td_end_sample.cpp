#include "source_pw/module_rt/td_end_sample.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/timer.h"
#include "source_hamilt/hamilt_hs_adapter.h"
#include "source_hsolver/module_linalg/linear_algebra.h"

namespace pw
{
template <typename Real, typename Device>
void TDEndpointPW<Real, Device>::clear_sample(hamilt::HamiltPW<Complex, Device> *hamilt)
{
    ready_ = false;
    orbs_.clear();
    if (hamilt != nullptr)
    {
        hamilt->td_projectors().drop_endpoint();
    }
}

template <typename Real, typename Device>
void TDEndpointPW<Real, Device>::evaluate(
    const UnitCell &cell, const ModulePW::PW_Basis_K &basis, const K_Vectors &kv,
    pseudopot_cell_vnl *pp, const psi::Psi<Complex, Device> &psi,
    const ModuleBase::matrix &wg, const ModuleBase::Vector3<double> &a,
    const ModuleBase::Vector3<double> &adot,
    const std::vector<ModuleBase::Vector3<double>> &ion_vel,
    const bool &accel, const bool &need_grad,
    hamilt::HamiltPW<Complex, Device> *hamilt,
    hsolver::TDOrbOps<Complex, Device> *orbitals, ModuleBase::matrix *energies)
{
    ModuleBase::timer::start("TDEndpointPW", "evaluate");
    clear_sample(hamilt);
    derivs_ = PiDerivs();
    PiDerivs &result = derivs_;
    result.pi.create(cell.nat, 3);
    if (accel)
    {
        result.hess.create(cell.nat, 6);
        result.rate.create(cell.nat, 3);
    }
    hamilt::HamiltHSOperator<Complex, Device> op(hamilt, &basis);
    if (pp->nkb == 0)
    {
        if (energies != nullptr)
        {
            orbitals->cal_band_energy(op, psi, energies);
        }
        ModuleBase::timer::end("TDEndpointPW", "evaluate");
        ready_ = true;
        return;
    }

    const bool motion = !ion_vel.empty();
    const bool full = accel && motion;
    const bool pool_root = basis.poolrank == 0;
    const int npol = psi.get_npol();
    const int bands = psi.get_nbands();
    const int cols = bands * npol;
    const int ld = basis.npwk_max;
    const int ngrad = need_grad || motion ? 3 : 0;
    const int ncomp = pool_root ? 1 + ngrad : 0;
    const int order = full ? 2 : 1;
    const int64_t block = static_cast<int64_t>(pp->nkb) * cols;
    const int64_t proj_size = static_cast<int64_t>(ld) * pp->nkb;
    const int64_t orb_size = static_cast<int64_t>(ld) * cols;
    const int64_t deriv_size = (full ? 9 : ngrad) * proj_size;
    hsolver::linear_buffer<Complex, Device>(&deriv_, deriv_size);
    Complex *grad = deriv_.template data<Complex>();
    Complex *extra = full ? grad + 3 * proj_size : nullptr;
    Complex *hp = nullptr;
    if (full || energies != nullptr)
    {
        hsolver::linear_buffer<Complex, Device>(&hpsi_, orb_size);
        hp = hpsi_.template data<Complex>();
    }
    bool moving = false;
    for (const ModuleBase::Vector3<double> &vel : ion_vel)
    {
        moving = moving || vel.x != 0.0 || vel.y != 0.0 || vel.z != 0.0;
    }
    const bool phase_rate = full && moving;
    if (motion)
    {
        pi_.begin(cell, ion_vel, wg, phase_rate, full, pool_root, ld, pp->nkb, cols);
    }
    orbs_.begin(cell, basis.nks, ld, cols, pp->nkb, ngrad, a, ion_vel);
    hamilt::NLProjPW<Real, Device> &projectors = hamilt->td_projectors();
    const std::vector<ModuleBase::Vector3<double>> tau = unitcell::cell_positions(cell);
    projectors.select(tau, ion_vel, a);
    projectors.hold_endpoint();

    const int old_k = psi.get_current_k();
    for (int ik = 0; ik < basis.nks; ++ik)
    {
        psi.fix_k(ik);
        const int npw = basis.npwk[ik];
        const Complex *wave = psi.get_pointer();
        const Complex *beta_proj = nullptr;
        if (hp != nullptr)
        {
            base_device::memory::set_memory_op<Complex, Device>()(hp, 0, orb_size);
            op.update_k(ik);
            op.hpsi(wave, hp, psi.get_nbasis(), bands);
            if (energies != nullptr)
            {
                const int dim = npol == 2 ? psi.get_nbasis() : npw;
                double *values = energies->c + ik * energies->nc;
                orbitals->cal_band_dot(wave, hp, psi.get_nbasis(), dim, bands, values);
            }
            beta_proj = hamilt->td_projections();
            if (full)
            {
                const Complex half(Real(0.5), Real(0));
                ModuleBase::scal_op<Real, Device>()(orb_size, &half, hp, 1);
            }
        }
        const Complex *beta = projectors.beta(ik);
        if (ngrad > 0)
        {
            const Complex *sk = projectors.phases(ik);
            grad_.cal_sample(pp, cell, basis, ik, a, ion_vel, order, sk, grad);
        }
        Complex *coeff = work_.project(beta, grad, wave, npw, ld, pp->nkb, cols, ngrad, ncomp, beta_proj);
        const bool reduced = beta_proj != nullptr;
        const Real scale = Real(0.5);
        work_.contract(cell, *pp, kv.isk[ik], cols, npol, ngrad, ncomp, scale, basis, coeff, reduced);
        const int64_t count = (1 + ngrad + ncomp) * block;
        orbs_.save(ik, count, coeff, beta);
        if (motion)
        {
            pi_.accum_k(cell, basis, *pp, ik, kv.isk[ik], cols, npol, phase_rate, full,
                        wave, hp, beta, grad, extra, coeff);
        }
    }
    psi.fix_k(old_k);
    orbs_.publish();
    if (motion)
    {
        result = pi_.finish(cell, adot, full, pool_root);
    }
    ready_ = true;
    ModuleBase::timer::end("TDEndpointPW", "evaluate");
}

template <typename Real, typename Device>
const ModuleBase::matrix &TDEndpointPW<Real, Device>::cal_pi(
    const UnitCell &cell, const ModulePW::PW_Basis_K &basis, const K_Vectors &kv,
    pseudopot_cell_vnl *pp, const psi::Psi<Complex, Device> &psi,
    const ModuleBase::matrix &wg, const ModuleBase::Vector3<double> &a,
    const std::vector<ModuleBase::Vector3<double>> &ion_vel, const bool &need_grad,
    hamilt::HamiltPW<Complex, Device> *hamilt,
    hsolver::TDOrbOps<Complex, Device> *orbitals, ModuleBase::matrix *energies)
{
    const ModuleBase::Vector3<double> adot{0.0, 0.0, 0.0};
    const bool accel = false;
    evaluate(cell, basis, kv, pp, psi, wg, a, adot, ion_vel, accel, need_grad, hamilt, orbitals, energies);
    return derivs_.pi;
}

template <typename Real, typename Device>
const PiDerivs &TDEndpointPW<Real, Device>::cal_pi_derivs(
    const UnitCell &cell, const ModulePW::PW_Basis_K &basis, const K_Vectors &kv,
    pseudopot_cell_vnl *pp, const psi::Psi<Complex, Device> &psi,
    const ModuleBase::matrix &wg, const ModuleBase::Vector3<double> &a,
    const ModuleBase::Vector3<double> &adot,
    const std::vector<ModuleBase::Vector3<double>> &ion_vel,
    hamilt::HamiltPW<Complex, Device> *hamilt,
    hsolver::TDOrbOps<Complex, Device> *orbitals, ModuleBase::matrix *energies)
{
    const bool accel = true;
    const bool need_grad = true;
    evaluate(cell, basis, kv, pp, psi, wg, a, adot, ion_vel, accel, need_grad, hamilt, orbitals, energies);
    return derivs_;
}

template class TDEndpointPW<float, base_device::DEVICE_CPU>;
template class TDEndpointPW<double, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class TDEndpointPW<float, base_device::DEVICE_GPU>;
template class TDEndpointPW<double, base_device::DEVICE_GPU>;
#endif
}
