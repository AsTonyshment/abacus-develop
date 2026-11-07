#include "fs_nonlocal_tools.h"

#include "source_base/global_variable.h"
#include "source_base/parallel_device.h"
#include "source_base/timer.h"

#include "source_base/parallel_comm.h" // different MPI worlds (POOL_WORLD)

namespace hamilt
{

template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_vkb(
    const int &ik,
    const int &nbdall,
    const ModuleBase::Vector3<double> &a,
    const std::vector<ModuleBase::Vector3<double>> &ion_vel)
{
    ModuleBase::timer::start("FS_Nonlocal_tools", "cal_vkb");
    const int columns = nbdall * this->ucell_->get_npol();
    const int projection_elements = columns * this->nkb;
    if (this->becp == nullptr)
    {
        resmem_complex_op()(this->becp, projection_elements);
    }
    base_device::memory::set_memory_op<std::complex<FPTYPE>, Device>()(this->becp, 0, projection_elements);
    const int npw = this->wfc_basis_->npwk[ik];
    if (npw > 0)
    {
        // RT-TDDFT supplies field and nuclear velocity shifts; other force and stress callers use zero a and empty velocities.
        this->nlpp_->getvnl(this->ctx, *this->ucell_, ik, a, ion_vel, this->ppcell_vkb, npw);
    }
    ModuleBase::timer::end("FS_Nonlocal_tools", "cal_vkb");
}

// cal_becp
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_becp(
    const int &ik,
    const int &npm,
    const std::complex<FPTYPE> *ppsi,
    const int &nbd0)
{
    if (npm == 0)
    {
        return;
    }
    const int npol = this->ucell_->get_npol();
    const int npw = this->wfc_basis_->npwk[ik];
    const char transa = 'C';
    const char transb = 'N';
    const int npm_npol = npm * npol;
    const int index0 = nbd0 * npol * nkb;

    if (npw == 0)
    {
        std::complex<FPTYPE> *destination = this->becp + index0;
        const int count = npm_npol * this->nkb;
        base_device::memory::set_memory_op<std::complex<FPTYPE>, Device>()(destination, 0, count);
        return;
    }

    gemm_op()(
        transa,
        transb,
        this->nkb,
        npm_npol,
        npw,
        &ModuleBase::ONE,
        this->ppcell_vkb,
        npw,
        ppsi,
        this->max_npw,
        &ModuleBase::ZERO,
        this->becp + index0,
        this->nkb);
}

template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::reduce_pool_becp(const int &npm)
{
    const int npol = this->ucell_->get_npol();
    const int size_becp_act = npm * npol * this->nkb;
    // becp calculate is over , now we should broadcast this data.
#ifdef __MPI
    if (GlobalV::NPROC_IN_POOL > 1)
    {
        Parallel_Common::reduce_data(this->becp, size_becp_act, POOL_WORLD);
    }
#endif
}

// template instantiation
template void FS_Nonlocal_tools<double, base_device::DEVICE_CPU>::cal_vkb(
    const int &,
    const int &,
    const ModuleBase::Vector3<double> &,
    const std::vector<ModuleBase::Vector3<double>> &);
template void FS_Nonlocal_tools<double, base_device::DEVICE_CPU>::cal_becp(
    const int &,
    const int &,
    const std::complex<double> *,
    const int &);
template void FS_Nonlocal_tools<double, base_device::DEVICE_CPU>::reduce_pool_becp(const int &);
#if ((defined __CUDA) || (defined __ROCM))
template void FS_Nonlocal_tools<double, base_device::DEVICE_GPU>::cal_vkb(
    const int &,
    const int &,
    const ModuleBase::Vector3<double> &,
    const std::vector<ModuleBase::Vector3<double>> &);
template void FS_Nonlocal_tools<double, base_device::DEVICE_GPU>::cal_becp(
    const int &,
    const int &,
    const std::complex<double> *,
    const int &);
template void FS_Nonlocal_tools<double, base_device::DEVICE_GPU>::reduce_pool_becp(const int &);
#endif

} // namespace hamilt
