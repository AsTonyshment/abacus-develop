#ifndef TD_CURRENT_PW_H
#define TD_CURRENT_PW_H

#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_estate/elecstate.h"
#include "source_psi/psi.h"
#include "source_pw/module_pwdft/vnl_pw.h"
#include "source_pw/module_pwdft/op_pw_vel.h"
#include "source_pw/module_pwdft/op_pw_vel_td.h"
#include <memory>

namespace ModuleIO
{
/** @brief Persistent one-k-point current workspace for a propagation run. */
template <typename FPTYPE, typename Device>
class CurrentPW
{
  private:
    std::unique_ptr<hamilt::Velocity<FPTYPE, Device>> velocity_;
    std::unique_ptr<hamilt::TDVelocity<FPTYPE, Device>> td_velocity_;
    ct::Tensor vpsi_;
    ct::Tensor dots_;
    std::vector<std::complex<FPTYPE>> band_current_;
  public:
    /** @brief Evaluate and write the current using native wavefunctions. */
    void write(const int istep, const UnitCell& ucell, const ModulePW::PW_Basis_K* wfcpw,
               psi::Psi<std::complex<FPTYPE>, Device>* psi, const elecstate::ElecState* pelec,
               const K_Vectors& kv, pseudopot_cell_vnl* ppcell);
};
/**
 * @brief Calculate and write the current in a plane-wave basis.
 * @param istep Current electronic propagation step.
 * @param ucell Unit cell.
 * @param wfcpw Plane-wave basis.
 * @param psi Wavefunctions for all k points.
 * @param pelec Electronic state containing occupations.
 * @param kv K points including spin and local-to-global indices.
 * @param ppcell Nonlocal pseudopotential projectors.
 */
template <typename FPTYPE, typename Device = base_device::DEVICE_CPU>
void write_current_pw(const int istep,
                      const UnitCell& ucell,
                      const ModulePW::PW_Basis_K* wfcpw,
                      psi::Psi<std::complex<FPTYPE>, Device>* psi,
                      const elecstate::ElecState* pelec,
                      const K_Vectors& kv,
                      pseudopot_cell_vnl* ppcell);
} // namespace ModuleIO
#endif
