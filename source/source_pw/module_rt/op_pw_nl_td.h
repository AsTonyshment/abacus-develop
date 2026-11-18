#ifndef OP_PW_NL_TD_H
#define OP_PW_NL_TD_H

#include "source_pw/module_pwdft/nonlocal_workspace.h"
#include "source_pw/module_pwdft/nl_proj_pw.h"
#include "source_pw/module_pwdft/op_pw.h"
#include "source_pw/module_pwdft/vnl_pw.h"

namespace hamilt
{

template <typename T, typename Device = base_device::DEVICE_CPU>
class TDNonlocalPW : public OperatorPW<T, Device>
{
  public:
    /** @brief Bind the RT-TDDFT vector potential and type-major nuclear velocities in Hartree units before init(). */
    void bind(const ModuleBase::Vector3<double> &a, const std::vector<ModuleBase::Vector3<double>> &ion_vel,
              const std::vector<ModuleBase::Vector3<double>> &tau)
    {
        // Electromagnetic coupling and nuclear motion are independent, so the field switch leaves the velocity phase active.
        ModuleBase::Vector3<double> nl_a{0.0, 0.0, 0.0};
        if (nonlocal_field_)
        {
            nl_a = a;
        }
        // The projector cache owns the selected geometry, velocities and field.
        projectors_->select(tau, ion_vel, nl_a);
    }
    /** @brief Share sampled projectors with endpoint numerical evaluators. */
    NLProjPW<typename GetTypeReal<T>::type, Device> &projectors() { return *projectors_; }
    /** @brief Borrow overlaps from the immediately preceding Hamiltonian action. */
    const T *projections() const { return workspace_.projections(); }
    TDNonlocalPW(
        const int *isk_in,
        const pseudopot_cell_vnl *ppcell_in,
        const UnitCell *ucell_in,
        const ModulePW::PW_Basis_K *wfc_basis,
        const bool &nonlocal_field);

    ~TDNonlocalPW() override = default;
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
    std::unique_ptr<NLProjPW<typename GetTypeReal<T>::type, Device>> projectors_;

    const int *isk_ = nullptr;
    const pseudopot_cell_vnl *ppcell_ = nullptr;
    const UnitCell *ucell_ = nullptr;
    const ModulePW::PW_Basis_K *wfcpw_ = nullptr;

    const T *vkb_td_ = nullptr; // Borrowed from the selected physical sample.
};

} // namespace hamilt
#endif
