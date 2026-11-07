#include "source_base/global_function.h"
#include "source_base/math_ylmreal.h"
#include "source_base/module_device/device.h"
#include "source_base/timer.h"
#include "source_pw/module_pwdft/kernels/projector_gradient_op.h"
#include "source_pw/module_pwdft/kernels/vnl_op.h"
#include "vnl_pw.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
template <typename Value, typename Device>
Value *reserve_vnl_buffer(ct::Tensor &buffer, const std::int64_t count)
{
    using CtDevice = typename ct::PsiToContainer<Device>::type;
    const ct::DataType data_type = ct::DataTypeToEnum<Value>::value;
    const ct::DeviceType device_type = ct::DeviceTypeToEnum<CtDevice>::value;
    if (buffer.NumElements() < count || buffer.data_type() != data_type || buffer.device_type() != device_type)
    {
        buffer = ct::Tensor(data_type, device_type, {count});
    }
    return buffer.data<Value>();
}
} // namespace

/**
 * @brief Calculate nonlocal projectors at k+G+A(t)-v_I(t).
 *
 * The radial and angular projector factors use the kinetic momentum shifted
 * by the Hartree-unit vector potential and atomic velocity. The structure factor remains evaluated
 * on the static reciprocal grid because its atom-dependent common phase cancels in
 * each same-atom projector outer product.
 */
template <typename FPTYPE, typename Device>
void pseudopot_cell_vnl::getvnl(
    Device *ctx,
    const UnitCell &ucell,
    const int &ik,
    const ModuleBase::Vector3<double> &vector_potential,
    const std::vector<ModuleBase::Vector3<double>> &ion_vel,
    std::complex<FPTYPE> *vkb_in,
    const int lead_dim) const
{
    ModuleBase::timer::start("pp_cell_vnl", "getvnl");

    using cal_vnl_op = hamilt::cal_vnl_op<FPTYPE, Device>;
    using syncmem_int_op = base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>;
    using castmem_var_h2d_op = base_device::memory::cast_memory_op<FPTYPE, double, Device, base_device::DEVICE_CPU>;
    using castmem_var_h2h_op = base_device::memory::cast_memory_op<FPTYPE, double, base_device::DEVICE_CPU, base_device::DEVICE_CPU>;

    if (this->lmaxkb < 0 || this->wfcpw->npwk[ik] == 0)
    {
        ModuleBase::timer::end("pp_cell_vnl", "getvnl");
        return;
    }

    const int npw = this->wfcpw->npwk[ik];
    const int precision = std::is_same<FPTYPE, float>::value ? 0 : 1;

    FPTYPE *shifted_gk_data = nullptr;
    FPTYPE *tab_ptr = this->get_tab_data<FPTYPE>();
    const std::int64_t momentum_elements = static_cast<std::int64_t>(npw) * 3;

    const ModuleBase::Vector3<double> reduced_vector_potential = vector_potential / ucell.tpiba;
    this->vnl_host_momentum_.resize(npw);
    std::vector<ModuleBase::Vector3<double>> &shifted_gk = this->vnl_host_momentum_;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int ig = 0; ig < npw; ++ig)
    {
        shifted_gk[ig] = this->wfcpw->getgpluskcar(ik, ig) + reduced_vector_potential;
    }

    double vmax = 0.0;
    for (const ModuleBase::Vector3<double> &vel : ion_vel)
    {
        vmax = std::max(vmax, vel.norm());
    }
    // Bound every atom's shifted momentum without constructing atom-sized PW grids.
    for (int ig = 0; ig < npw; ++ig)
    {
        const FPTYPE x = static_cast<FPTYPE>(shifted_gk[ig].x);
        const FPTYPE y = static_cast<FPTYPE>(shifted_gk[ig].y);
        const FPTYPE z = static_cast<FPTYPE>(shifted_gk[ig].z);
        const FPTYPE q = std::sqrt(x * x + y * y + z * z) * static_cast<FPTYPE>(ucell.tpiba);
        const FPTYPE index = (q + static_cast<FPTYPE>(vmax)) / static_cast<FPTYPE>(this->table_dq_);
        // Also cover small host/device differences in the norm and division.
        const double guarded_index = static_cast<double>(index) * (1.0 + 8.0 * std::numeric_limits<FPTYPE>::epsilon());
        this->check_vnl_index(guarded_index, false);
    }

    if (this->use_gpu_)
    {
        shifted_gk_data = reserve_vnl_buffer<FPTYPE, Device>(this->vnl_momentum_[precision], momentum_elements);
        castmem_var_h2d_op()(shifted_gk_data, reinterpret_cast<double *>(shifted_gk.data()), momentum_elements);
    }
    else
    {
        if (std::is_same<FPTYPE, float>::value)
        {
            shifted_gk_data = reserve_vnl_buffer<FPTYPE, Device>(this->vnl_momentum_[precision], momentum_elements);
            castmem_var_h2h_op()(shifted_gk_data, reinterpret_cast<double *>(shifted_gk.data()), momentum_elements);
        }
        else
        {
            shifted_gk_data = reinterpret_cast<FPTYPE *>(shifted_gk.data());
        }
    }

    const std::int64_t structure_elements = static_cast<std::int64_t>(ucell.nat) * npw;
    std::complex<FPTYPE> *structure_factor =
        reserve_vnl_buffer<std::complex<FPTYPE>, Device>(this->vnl_structure_[precision], structure_elements);
    this->psf->get_sk(ctx, ik, this->wfcpw, structure_factor);

    if (vmax > 0.0)
    {
        // RT-TDDFT traveling pseudopotentials require atom-specific radial and angular factors at nonzero nuclear velocities.
        if (this->vnl_host_meta_.empty())
        {
            // Cache each projector's radial channel, angular indices and owning atom for the shared device evaluator.
            int atom = 0;
            for (int it = 0; it < ucell.ntype; ++it)
            {
                for (int ia = 0; ia < ucell.atoms[it].na; ++ia, ++atom)
                {
                    for (int p = 0; p < ucell.atoms[it].ncpp.nh; ++p)
                    {
                        this->vnl_host_meta_.push_back(it);
                        this->vnl_host_meta_.push_back(static_cast<int>(this->indv(it, p)));
                        this->vnl_host_meta_.push_back(static_cast<int>(this->nhtol(it, p)));
                        this->vnl_host_meta_.push_back(static_cast<int>(this->nhtolm(it, p)));
                        this->vnl_host_meta_.push_back(atom);
                    }
                }
            }
            int *meta = reserve_vnl_buffer<int, Device>(this->vnl_meta_, this->vnl_host_meta_.size());
            syncmem_int_op()(meta, this->vnl_host_meta_.data(), this->vnl_host_meta_.size());
        }
        const std::int64_t size = 3 * static_cast<std::int64_t>(ion_vel.size());
        FPTYPE *vel = reserve_vnl_buffer<FPTYPE, Device>(this->vnl_vel_[precision], size);
        base_device::memory::cast_memory_op<FPTYPE, double, Device, base_device::DEVICE_CPU>()(
            vel, reinterpret_cast<const double *>(ion_vel.data()), size);
        hamilt::proj_eval_op<FPTYPE, Device>()(
            npw,
            lead_dim,
            this->nkb,
            this->tab.getBound2(),
            this->tab.getBound3(),
            static_cast<FPTYPE>(this->table_dq_),
            static_cast<FPTYPE>(ucell.tpiba),
            this->vnl_meta_.data<int>(),
            shifted_gk_data,
            vel,
            tab_ptr,
            nullptr,
            structure_factor,
            vkb_in);
        // The atom-resolved evaluator has filled vkb_in, completing the moving-projector calculation.
        ModuleBase::timer::end("pp_cell_vnl", "getvnl");
        return;
    }

    // Non-RT and zero-velocity RT-TDDFT calls share spherical harmonics and reuse each type's radial factors across atoms.
    const int ylm_count = (this->lmaxkb + 1) * (this->lmaxkb + 1);
    const std::int64_t ylm_elements = static_cast<std::int64_t>(ylm_count) * npw;
    const std::int64_t radial_elements = static_cast<std::int64_t>(this->nhm) * npw;
    FPTYPE *ylm = reserve_vnl_buffer<FPTYPE, Device>(this->vnl_ylm_[precision], ylm_elements);
    FPTYPE *vkb_radial = reserve_vnl_buffer<FPTYPE, Device>(this->vnl_radial_[precision], radial_elements);
    FPTYPE *indv_ptr = this->get_indv_data<FPTYPE>();
    FPTYPE *nhtol_ptr = this->get_nhtol_data<FPTYPE>();
    FPTYPE *nhtolm_ptr = this->get_nhtolm_data<FPTYPE>();
    const std::int64_t metadata_elements = 3 * static_cast<std::int64_t>(ucell.ntype);
    this->vnl_host_atom_counts_.resize(metadata_elements);
    int *host_atom_nh = this->vnl_host_atom_counts_.data();
    int *host_atom_nb = host_atom_nh + ucell.ntype;
    int *host_atom_na = host_atom_nb + ucell.ntype;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        host_atom_nb[it] = ucell.atoms[it].ncpp.nbeta;
        host_atom_nh[it] = ucell.atoms[it].ncpp.nh;
        host_atom_na[it] = ucell.atoms[it].na;
    }
    int *atom_nh = host_atom_nh;
    int *atom_nb = host_atom_nb;
    int *atom_na = host_atom_na;
    if (this->use_gpu_)
    {
        // These type-level counts belong to the zero-velocity kernel; traveling projectors use their own metadata.
        atom_nh = reserve_vnl_buffer<int, Device>(this->vnl_atom_counts_, metadata_elements);
        atom_nb = atom_nh + ucell.ntype;
        atom_na = atom_nb + ucell.ntype;
        syncmem_int_op()(atom_nh, host_atom_nh, metadata_elements);
    }
    ModuleBase::YlmReal::Ylm_Real(ctx, ylm_count, npw, shifted_gk_data, ylm);

    cal_vnl_op()(
        ctx,
        ucell.ntype,
        npw,
        lead_dim,
        this->nhm,
        this->tab.getBound2(),
        this->tab.getBound3(),
        atom_na,
        atom_nb,
        atom_nh,
        static_cast<FPTYPE>(this->table_dq_),
        static_cast<FPTYPE>(ucell.tpiba),
        static_cast<std::complex<FPTYPE>>(ModuleBase::NEG_IMAG_UNIT),
        shifted_gk_data,
        ylm,
        indv_ptr,
        nhtol_ptr,
        nhtolm_ptr,
        tab_ptr,
        vkb_radial,
        structure_factor,
        vkb_in);

    ModuleBase::timer::end("pp_cell_vnl", "getvnl");
}

// Explicit instantiations for CPU/GPU and float/double precision.
template void pseudopot_cell_vnl::getvnl<float, base_device::DEVICE_CPU>(
    base_device::DEVICE_CPU *,
    const UnitCell &,
    const int &,
    const ModuleBase::Vector3<double> &,
    const std::vector<ModuleBase::Vector3<double>> &,
    std::complex<float> *,
    int) const;
template void pseudopot_cell_vnl::getvnl<double, base_device::DEVICE_CPU>(
    base_device::DEVICE_CPU *,
    const UnitCell &,
    const int &,
    const ModuleBase::Vector3<double> &,
    const std::vector<ModuleBase::Vector3<double>> &,
    std::complex<double> *,
    int) const;
#if defined(__CUDA) || defined(__ROCM)
template void pseudopot_cell_vnl::getvnl<float, base_device::DEVICE_GPU>(
    base_device::DEVICE_GPU *,
    const UnitCell &,
    const int &,
    const ModuleBase::Vector3<double> &,
    const std::vector<ModuleBase::Vector3<double>> &,
    std::complex<float> *,
    int) const;
template void pseudopot_cell_vnl::getvnl<double, base_device::DEVICE_GPU>(
    base_device::DEVICE_GPU *,
    const UnitCell &,
    const int &,
    const ModuleBase::Vector3<double> &,
    const std::vector<ModuleBase::Vector3<double>> &,
    std::complex<double> *,
    int) const;
#endif
