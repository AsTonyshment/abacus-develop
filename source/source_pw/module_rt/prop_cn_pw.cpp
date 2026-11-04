#include "source_pw/module_rt/prop_cn_pw.h"

#include "source_base/kernels/math_kernel_op.h"
#include "source_pw/module_rt/td_ion_pw.h"

namespace pw
{
template <typename T, typename Device>
CNPropPW<T, Device>::CNPropPW(
    const ModulePW::PW_Basis_K &basis,
    const hsolver::CNControls &cn_ctrl,
    int npol,
    const std::vector<int> &global_k,
    const hsolver::diag_comm_info &comm,
    std::ostream &log)
    : solver_(basis, cn_ctrl, npol, global_k, comm, log)
{
}

template <typename T, typename Device>
void CNPropPW<T, Device>::evolve(
    TDHamiltPW<T, Device> *hamilt,
    const TDHistoryPW<T, Device> &history,
    const typename GetTypeReal<T>::type *veff,
    int nveff,
    const elecstate::TDFieldManager &field,
    const unitcell::IonicState &sample,
    const unitcell::IonicState &right,
    psi::Psi<T, Device> *current,
    int elec_step,
    int scf_iter,
    bool log_detail)
{
    using Real = typename T::value_type;
    if (scf_iter > 1)
    {
        // Approximate the midpoint local potential by averaging the accepted left and corrected right potentials.
        const int size = nveff;
        hsolver::linear_buffer<Real, Device>(&veff_mid_, size);
        Real *buffer = veff_mid_.template data<Real>();
        const Real *previous = history.left_potential();
        ModuleBase::vector_add_vector_op<Real, Device, Real>()(size, buffer, previous, Real(0.5), veff, Real(0.5));
        veff = buffer;
    }

    ModuleBase::Vector3<double> vecpot_ha(0, 0, 0);
    if (field.gauge() == 1)
    {
        vecpot_ha = field.A_prop_ha();
    }

    // Use explicit midpoint ions for the nonlocal operator with the CN propagation field and local potential.
    hamilt->bind(veff, vecpot_ha, sample);
    const psi::Psi<T, Device> &previous = history.previous();
    const double dt = field.interval_dt_ha();
    solver_.solve(*hamilt, previous, current, dt, vecpot_ha, elec_step, scf_iter, log_detail);
}
template class CNPropPW<std::complex<float>, base_device::DEVICE_CPU>;
template class CNPropPW<std::complex<double>, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class CNPropPW<std::complex<float>, base_device::DEVICE_GPU>;
template class CNPropPW<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace pw
