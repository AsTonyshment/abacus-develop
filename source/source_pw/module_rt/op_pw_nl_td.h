#ifndef OP_PW_NL_TD_H
#define OP_PW_NL_TD_H

#include "source_pw/module_pwdft/nonlocal_workspace.h"
#include "source_pw/module_pwdft/op_pw.h"
#include "source_pw/module_pwdft/vnl_pw.h"

namespace hamilt
{

template <typename T, typename Device = base_device::DEVICE_CPU>
class TDNonlocalPW : public OperatorPW<T, Device>
{
  public:
    /** @brief Bind the RT-TDDFT vector potential and type-major nuclear velocities in Hartree units before init(). */
    void bind(const ModuleBase::Vector3<double> &A, const std::vector<ModuleBase::Vector3<double>> &ion_vel)
    {
        // Electromagnetic coupling and nuclear motion are independent, so the field switch leaves the velocity phase active.
        if (nonlocal_field_)
        {
            A_ha_ = A;
        }
        // Own the sampled velocities so later projector updates remain valid after the caller's temporary array expires.
        ion_vel_ = ion_vel;
    }
    TDNonlocalPW(
        const int *isk_in,
        const pseudopot_cell_vnl *ppcell_in,
        const UnitCell *ucell_in,
        const ModulePW::PW_Basis_K *wfc_basis,
        bool nonlocal_field);

    virtual ~TDNonlocalPW();
    virtual void init(const int ik_in) override;

    virtual void act(
        const int nbands,
        const int nbasis,
        const int npol,
        const T *tmpsi_in,
        T *tmhpsi,
        const int ngk_ik = 0,
        const bool is_first_node = false) const override;

  private:
    const bool nonlocal_field_;
    mutable NonlocalWorkspace<T, Device> workspace_;

    const int *isk = nullptr;
    const pseudopot_cell_vnl *ppcell = nullptr;
    const UnitCell *ucell = nullptr;
    const ModulePW::PW_Basis_K *wfcpw = nullptr;

    mutable T *vkb_td = nullptr; // Time-dependent projector cache.

    ModuleBase::Vector3<double> A_ha_{0.0, 0.0, 0.0};
    std::vector<ModuleBase::Vector3<double>> ion_vel_;
    Device *ctx = {};
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
};

} // namespace hamilt
#endif
