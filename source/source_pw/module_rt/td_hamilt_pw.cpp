#include "source_pw/module_rt/td_hamilt_pw.h"

namespace pw
{
template <typename T, typename Device>
TDHamiltPW<T, Device>::TDHamiltPW(
    hamilt::HamiltPW<T, Device> *hamilt,
    const ModulePW::PW_Basis_K &basis,
    const bool &motion)
    : hamilt_(*hamilt), op_(hamilt, &basis), motion_(motion)
{
}
template <typename T, typename Device>
void TDHamiltPW<T, Device>::bind(const Real *veff, const ModuleBase::Vector3<double> &vecpot_ha, const unitcell::IonicState &state)
{
    // Sampled projectors own their geometry while the shared cell stays at the right endpoint.
    std::vector<ModuleBase::Vector3<double>> tau;
    tau.reserve(state.atoms.size());
    for (const unitcell::IonicAtom &atom : state.atoms)
    {
        tau.push_back(atom.frac * state.lattice);
    }
    // Extract velocities from this propagation sample so the projector phase follows the sampled ionic path.
    std::vector<ModuleBase::Vector3<double>> vel;
    if (motion_)
    {
        vel.reserve(state.atoms.size());
        for (const unitcell::IonicAtom &atom : state.atoms)
        {
            vel.push_back(atom.velocity);
        }
    }
    hamilt_.bind_td_state(veff, vecpot_ha, vel, tau);
}
template <typename T, typename Device>
void TDHamiltPW<T, Device>::update_k(int ik)
{
    // Select cached projectors and the current local potential for this k point.
    op_.update_k(ik);
}
template <typename T, typename Device>
void TDHamiltPW<T, Device>::hpsi(
    const T *x,
    T *hx,
    int ld,
    int bands) const
{
    op_.hpsi(x, hx, ld, bands);
}
template <typename T, typename Device>
void TDHamiltPW<T, Device>::spsi(
    const T *x,
    T *sx,
    int ld,
    int bands) const
{
    op_.spsi(x, sx, ld, bands);
}
template class TDHamiltPW<std::complex<float>, base_device::DEVICE_CPU>;
template class TDHamiltPW<std::complex<double>, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class TDHamiltPW<std::complex<float>, base_device::DEVICE_GPU>;
template class TDHamiltPW<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace pw
