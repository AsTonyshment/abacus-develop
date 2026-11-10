#include "source_pw/module_pwdft/velocity_workspace.h"

#include "source_base/kernels/math_kernel_op.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/parallel_device.h"
#include "source_base/timer.h"
#include "source_pw/module_pwdft/kernels/nonlocal_op.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace hamilt
{
template <typename Real, typename Device>
std::complex<Real> *VelocityWorkspace<Real, Device>::project(
    const Complex *beta,
    const Complex *grad,
    const Complex *psi,
    int npw,
    int ld,
    int nkb,
    int cols,
    int nderiv,
    int ncomp)
{
    using CtDevice = typename ct::PsiToContainer<Device>::type;
    const std::int64_t block = static_cast<std::int64_t>(nkb) * cols;
    // Allocate only the requested derivative and D-contraction blocks for velocity and RT-TDDFT sampling.
    const std::int64_t capacity = std::max<std::int64_t>(1, (1 + nderiv + ncomp) * block);
    if (coefficients_.NumElements() < capacity || coefficients_.data_type() != ct::DataTypeToEnum<Complex>::value ||
        coefficients_.device_type() != ct::DeviceTypeToEnum<CtDevice>::value)
    {
        coefficients_ = ct::Tensor(ct::DataTypeToEnum<Complex>::value, ct::DeviceTypeToEnum<CtDevice>::value, {capacity});
    }
    // Store beta overlaps, interleaved derivative overlaps, then space for up to four D-contracted blocks.
    Complex *buffer = coefficients_.template data<Complex>();
    if (npw == 0)
    {
        // Empty PW ranks still contribute zeros to the following collective projection reduction.
        base_device::memory::set_memory_op<Complex, Device>()(buffer, 0, (1 + nderiv) * block);
    }
    else
    {
        const Complex one(1, 0);
        const Complex zero(0, 0);
        const int nkb3 = nderiv * nkb;
        if (cols == 1)
        {
            ModuleBase::gemv_op<Complex, Device>()('C', npw, nkb, &one, beta, ld, psi, 1, &zero, buffer, 1);
            ModuleBase::gemv_op<Complex, Device>()('C', npw, nkb3, &one, grad, ld, psi, 1, &zero, buffer + block, 1);
        }
        else
        {
            ModuleBase::gemm_op<Complex, Device>()('C', 'N', nkb, cols, npw, &one, beta, ld, psi, ld, &zero, buffer, nkb);
            ModuleBase::gemm_op<Complex, Device>()('C', 'N', nkb3, cols, npw, &one, grad, ld, psi, ld, &zero, buffer + block, nkb3);
        }
    }
    return buffer;
}

template <typename Real, typename Device>
void VelocityWorkspace<Real, Device>::contract(
    const UnitCell &cell,
    const pseudopot_cell_vnl &pp,
    const int spin,
    const int bands,
    const int npol,
    const int nderiv,
    const int ncomp,
    const Real scale,
    const ModulePW::PW_Basis_K &basis,
    Complex *buffer)
{
    ModuleBase::timer::start("VelocityWorkspace", "contract");
    const std::int64_t count = static_cast<std::int64_t>(bands) * pp.nkb;
    if (count == 0)
    {
        ModuleBase::timer::end("VelocityWorkspace", "contract");
        return;
    }
    const std::int64_t projection_elements = (1 + nderiv) * count;
    // Complete the distributed overlaps before multiplying by D so both observables use the same full projections.
#ifdef __MPI
    if (basis.poolnproc > 1)
    {
        Complex *host = nullptr;
#if !defined(__CUDA_MPI) && !defined(__NCCL_PARALLEL_DEVICE)
        // Stage GPU overlaps on the host only when the selected pool communication requires it.
        if (std::is_same<Device, base_device::DEVICE_GPU>::value)
        {
            if (host_.size() < static_cast<std::size_t>(projection_elements))
            {
                host_.resize(projection_elements);
            }
            host = host_.data();
        }
#endif
        // The complex reduction wrapper doubles the MPI element count.
        const std::int64_t max_chunk = std::numeric_limits<int>::max() / 2;
        for (std::int64_t offset = 0; offset < projection_elements;)
        {
            const std::int64_t remaining = projection_elements - offset;
            const int chunk_size = static_cast<int>(std::min(max_chunk, remaining));
            Complex *chunk = buffer + offset;
            Complex *host_chunk = host == nullptr ? nullptr : host + offset;
            Parallel_Common::reduce_dev<Complex, Device>(chunk, chunk_size, basis.pool_world, host_chunk);
            offset += chunk_size;
        }
    }
#endif
    // RT-TDDFT derivative-only projections and non-root sampling ranks require the collective reduction above without D contraction.
    if (ncomp == 0)
    {
        ModuleBase::timer::end("VelocityWorkspace", "contract");
        return;
    }
    Complex *output = buffer + projection_elements;
    const std::int64_t out_size = ncomp * count;
    base_device::memory::set_memory_op<Complex, Device>()(output, 0, out_size);
    for (int component = 0; component < ncomp; ++component)
    {
        // The ordinary overlap is packed separately from the three interleaved gradient overlaps.
        int sum = 0;
        int atom = 0;
        const int stride = component == 0 ? pp.nkb : nderiv * pp.nkb;
        const std::int64_t gradient_offset = static_cast<std::int64_t>(component - 1) * pp.nkb;
        const Complex *input = component == 0 ? buffer : buffer + count + gradient_offset;
        Complex *component_output = output + component * count;
        for (int type = 0; type < cell.ntype; ++type)
        {
            // Both kernels advance sum and atom by reference.
            if (npol == 2)
            {
                nonlocal_pw_op<Real, Device>()(
                    nullptr,
                    cell.atoms[type].na,
                    bands,
                    cell.atoms[type].ncpp.nh,
                    sum,
                    atom,
                    stride,
                    pp.deeq_nc.getBound2(),
                    pp.deeq_nc.getBound3(),
                    pp.deeq_nc.getBound4(),
                    pp.template get_deeq_nc_data<Real>(),
                    component_output,
                    input);
            }
            else
            {
                nonlocal_pw_op<Real, Device>()(
                    nullptr,
                    cell.atoms[type].na,
                    bands,
                    cell.atoms[type].ncpp.nh,
                    sum,
                    atom,
                    spin,
                    stride,
                    pp.deeq.getBound2(),
                    pp.deeq.getBound3(),
                    pp.deeq.getBound4(),
                    pp.template get_deeq_data<Real>(),
                    component_output,
                    input);
            }
        }
    }
    if (scale != Real(1))
    {
        // Velocity contractions convert the Ry-valued D matrix to Hartree without changing the Hamiltonian's storage.
        const Complex factor(scale, 0);
        const std::int64_t max_chunk = std::numeric_limits<int>::max();
        for (std::int64_t offset = 0; offset < out_size;)
        {
            const std::int64_t remaining = out_size - offset;
            const int chunk_size = static_cast<int>(std::min(max_chunk, remaining));
            Complex *chunk = output + offset;
            ModuleBase::scal_op<Real, Device>()(chunk_size, &factor, chunk, 1);
            offset += chunk_size;
        }
    }
    ModuleBase::timer::end("VelocityWorkspace", "contract");
}
template class VelocityWorkspace<float, base_device::DEVICE_CPU>;
template class VelocityWorkspace<double, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class VelocityWorkspace<float, base_device::DEVICE_GPU>;
template class VelocityWorkspace<double, base_device::DEVICE_GPU>;
#endif
} // namespace hamilt
