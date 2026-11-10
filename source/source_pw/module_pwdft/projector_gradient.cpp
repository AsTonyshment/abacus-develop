#include "source_pw/module_pwdft/projector_gradient.h"

#include "source_base/timer.h"
#include "source_pw/module_pwdft/kernels/projector_gradient_op.h"

#include <algorithm>
#include <limits>

namespace hamilt
{
template <typename Real, typename Device>
template <typename Value>
void ProjectorGradient<Real, Device>::reserve(ct::Tensor *tensor, const int64_t count)
{
    using CtDevice = typename ct::PsiToContainer<Device>::type;
    const int64_t capacity = std::max<int64_t>(1, count);
    if (tensor->NumElements() < capacity || tensor->data_type() != ct::DataTypeToEnum<Value>::value ||
        tensor->device_type() != ct::DeviceTypeToEnum<CtDevice>::value)
    {
        *tensor = ct::Tensor(ct::DataTypeToEnum<Value>::value, ct::DeviceTypeToEnum<CtDevice>::value, {capacity});
    }
}

template <typename Real, typename Device>
void ProjectorGradient<Real, Device>::calculate(
    pseudopot_cell_vnl *pp,
    const UnitCell &cell,
    const ModulePW::PW_Basis_K &basis,
    const int ik,
    const ModuleBase::Vector3<double> &A,
    const std::vector<ModuleBase::Vector3<double>> &ion_vel,
    int order,
    Complex *output)
{
    ModuleBase::timer::start("ProjectorGradient", "calculate");
    if (pp->nkb == 0)
    {
        ModuleBase::timer::end("ProjectorGradient", "calculate");
        return;
    }
    pp->ensure_grad_table(cell);
    const double dq = pp->table_dq();
    if (version_ != pp->table_version())
    {
        reserve<Real>(&radial_, pp->tab.getSize());
        reserve<Real>(&derivative_, pp->tab_dq.getSize());
        reserve<Real>(&moments_, pp->rad_mom.getSize());
        base_device::memory::cast_memory_op<Real, double, Device, base_device::DEVICE_CPU>()(
            moments_.template data<Real>(), pp->rad_mom.ptr, pp->rad_mom.getSize());
        base_device::memory::cast_memory_op<Real, double, Device, base_device::DEVICE_CPU>()(
            radial_.template data<Real>(), pp->tab.ptr, pp->tab.getSize());
        base_device::memory::cast_memory_op<Real, double, Device, base_device::DEVICE_CPU>()(
            derivative_.template data<Real>(), pp->tab_dq.ptr, pp->tab_dq.getSize());
        host_metadata_.clear();
        int atom = 0;
        for (int type = 0; type < cell.ntype; ++type)
        {
            for (int ia = 0; ia < cell.atoms[type].na; ++ia, ++atom)
            {
                for (int p = 0; p < cell.atoms[type].ncpp.nh; ++p)
                {
                    host_metadata_.push_back(type);
                    host_metadata_.push_back(static_cast<int>(pp->indv(type, p)));
                    host_metadata_.push_back(static_cast<int>(pp->nhtol(type, p)));
                    host_metadata_.push_back(static_cast<int>(pp->nhtolm(type, p)));
                    host_metadata_.push_back(atom);
                }
            }
        }
        reserve<int>(&metadata_, host_metadata_.size());
        base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>()(
            metadata_.template data<int>(), host_metadata_.data(), host_metadata_.size());
        version_ = pp->table_version();
    }
    // Second derivatives are needed only by RT-TDDFT nuclear inertia, so ordinary velocity calls keep first-order storage.
    const Real *second = nullptr;
    const Real *moments = moments_.template data<Real>();
    if (order == 2)
    {
        pp->ensure_hess(cell);
        if (hess_ver_ != pp->table_version())
        {
            reserve<Real>(&second_, pp->tab_d2.getSize());
            base_device::memory::cast_memory_op<Real, double, Device, base_device::DEVICE_CPU>()(
                second_.template data<Real>(), pp->tab_d2.ptr, pp->tab_d2.getSize());
            hess_ver_ = pp->table_version();
        }
        second = second_.template data<Real>();
    }
    const int npw = basis.npwk[ik];
    if (npw == 0)
    {
        ModuleBase::timer::end("ProjectorGradient", "calculate");
        return;
    }
    const std::int64_t momentum_elements = 3 * static_cast<std::int64_t>(npw);
    double vmax = 0.0;
    for (const ModuleBase::Vector3<double> &vel : ion_vel)
    {
        vmax = std::max(vmax, vel.norm());
    }
    // A null velocity pointer selects the ordinary momentum gradient used by ground-state velocity operators.
    Real *vel = nullptr;
    if (!ion_vel.empty())
    {
        const std::int64_t size = 3 * static_cast<std::int64_t>(ion_vel.size());
        reserve<Real>(&vel_, size);
        vel = vel_.template data<Real>();
        base_device::memory::cast_memory_op<Real, double, Device, base_device::DEVICE_CPU>()(
            vel, reinterpret_cast<const double *>(ion_vel.data()), size);
    }
    host_q_.resize(momentum_elements);
    for (int ig = 0; ig < npw; ++ig)
    {
        const ModuleBase::Vector3<double> q = basis.getgpluskcar(ik, ig) + A / cell.tpiba;
        for (int d = 0; d < 3; ++d)
        {
            host_q_[3 * static_cast<std::int64_t>(ig) + d] = static_cast<Real>(q[d]);
        }
        const Real *v = host_q_.data() + 3 * static_cast<std::int64_t>(ig);
        const Real radius = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) * static_cast<Real>(cell.tpiba);
        const Real position = (radius + static_cast<Real>(vmax)) / static_cast<Real>(dq);
        pp->check_vnl_index(static_cast<double>(position) * (1.0 + 8.0 * std::numeric_limits<Real>::epsilon()), true);
    }
    const std::int64_t structure_elements = static_cast<std::int64_t>(cell.nat) * npw;
    reserve<Real>(&momentum_, momentum_elements);
    reserve<Complex>(&structure_, structure_elements);
    base_device::memory::synchronize_memory_op<Real, Device, base_device::DEVICE_CPU>()(
        momentum_.template data<Real>(), host_q_.data(), momentum_elements);
    Device *ctx = nullptr;
    pp->psf->get_sk(ctx, ik, &basis, structure_.template data<Complex>());
    // Select a compiled derivative order before launching the shared CPU/GPU evaluator.
    const Real grid_dq = static_cast<Real>(dq);
    const Real tpiba = static_cast<Real>(cell.tpiba);
    const int *meta = metadata_.template data<int>();
    const Real *q = momentum_.template data<Real>();
    const Real *radial = radial_.template data<Real>();
    const Real *first = derivative_.template data<Real>();
    const Complex *sk = structure_.template data<Complex>();
    if (order == 1)
    {
        proj_eval_op<Real, Device, 1>()(
            npw,
            basis.npwk_max,
            pp->nkb,
            pp->tab.getBound2(),
            pp->tab.getBound3(),
            grid_dq,
            tpiba,
            meta,
            q,
            vel,
            radial,
            first,
            second,
            moments,
            sk,
            output);
    }
    else
    {
        proj_eval_op<Real, Device, 2>()(
            npw,
            basis.npwk_max,
            pp->nkb,
            pp->tab.getBound2(),
            pp->tab.getBound3(),
            grid_dq,
            tpiba,
            meta,
            q,
            vel,
            radial,
            first,
            second,
            moments,
            sk,
            output);
    }
    ModuleBase::timer::end("ProjectorGradient", "calculate");
}
template class ProjectorGradient<float, base_device::DEVICE_CPU>;
template class ProjectorGradient<double, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class ProjectorGradient<float, base_device::DEVICE_GPU>;
template class ProjectorGradient<double, base_device::DEVICE_GPU>;
#endif
} // namespace hamilt
