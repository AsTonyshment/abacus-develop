#ifndef FORCE_PW_H
#define FORCE_PW_H

#include "source_base/global_function.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/matrix.h"
#include "source_base/module_device/memory_op.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/klist.h"
#include "source_cell/module_symmetry/symmetry.h"
#include "source_cell/stru_fac.h"
#include "source_estate/elecstate.h"
#include "source_psi/psi.h"
#include "source_pw/module_pwdft/dftu_base.h"
#include "source_pw/module_pwdft/force_result.h"
#include "source_pw/module_pwdft/nl_orb_data.h"
#include "source_pw/module_pwdft/vl_pw.h"

#include <string>
#include <vector>

class pseudopot_cell_vnl;

namespace vdw
{
struct VdwResult;
}

// forward declaration so that the dH module (out_mat_dh_vl) can reuse cal_force_loc
namespace hamilt
{
template <class T>
class Veff;
}

template <typename FPTYPE, typename Device = base_device::DEVICE_CPU>
class Forces
{
  public:
    template <typename T>
    friend class Force_Stress_LCAO;
    template <class T>
    friend class hamilt::Veff;
    template <typename T>
    friend class ForcePWTerms;
    /* This routine is a driver routine which compute the forces
     * acting on the atoms, the complete forces in plane waves
     * is computed from 4 main parts
     * (1) cal_force_loc: contribution due to local potential.
     * (2) cal_foce_ew: contribution due to ewald potential.
     * (3) cal_force_cc: contributino due to NLCC.
     * (4) cal_nl: contribution due to the non-local pseudopotential.
     * (5) cal_force_us: contribution due to US pseudopotential.
     * (6) cal_scc: contributino due to incomplete SCF calculation.
     */
    Forces(const int nat_in) : nat(nat_in) {};
    ~Forces() {};

    /**
     * @brief Evaluate PW force contributions in Ry/Bohr.
     * @param ion_force Optional Ewald force at the same positions and lattice.
     * @param esolver_type Selects the RT-TDDFT instantaneous-force treatment for "tddft".
     * @param nonlocal_a RT-TDDFT nonlocal momentum shift in Hartree atomic units; zero for ground-state calls.
     * @param ion_vel RT-TDDFT nuclear velocities in type-major order and Hartree atomic units; non-RT callers pass an empty vector.
     * @param electric RT-TDDFT electric field in Hartree atomic units; zero for ground-state calls.
     * @param[out] result Force contributions and total force; must point to writable storage.
     */
    void cal_force(
        const hamilt::NLOrbData<FPTYPE, Device> *nl_orbs,
        const ModuleBase::matrix *ion_force,
        const std::string &esolver_type,
        const ModuleBase::Vector3<double> &nonlocal_a,
        const std::vector<ModuleBase::Vector3<double>> &ion_vel,
        const ModuleBase::Vector3<double> &electric,
        const int &nspin,
        const bool &domag,
        const bool &domag_z,
        const int &gga_grad,
        const bool &use_onsite_projection,
        UnitCell &ucell,
        PWForceResult *result,
        const vdw::VdwResult *vdw_result,
        const elecstate::ElecState &elec,
        const ModulePW::PW_Basis *const rho_basis,
        ModuleSymmetry::Symmetry *p_symm,
        Structure_Factor *p_sf,
        surchem &solvent,
        const Plus_U_Base *p_dftu,
        const pseudopot_cell_vl *locpp,
        const pseudopot_cell_vnl *nlpp = nullptr,
        K_Vectors *pkv = nullptr,
        ModulePW::PW_Basis_K *psi_basis = nullptr,
        const psi::Psi<std::complex<FPTYPE>, Device> *psi_in = nullptr);

    /** @brief Reevaluate traveling-projector forces at fixed orbitals, density, positions and field.
     * All velocity-independent contributions must already belong to this endpoint.
     * This entry supports the norm-conserving RT model.
     */
    void update_nl_force(
        const hamilt::NLOrbData<FPTYPE, Device> *nl_orbs,
        const ModuleBase::Vector3<double> &nonlocal_a,
        const std::vector<ModuleBase::Vector3<double>> &ion_vel,
        const elecstate::ElecState &elec,
        K_Vectors *kv,
        ModulePW::PW_Basis_K *basis,
        Structure_Factor *sf,
        const pseudopot_cell_vnl &pp,
        UnitCell &cell,
        const psi::Psi<std::complex<FPTYPE>, Device> *wave,
        PWForceResult *result);

  protected:
    //! Assemble force contributions and apply the established net-force and symmetry treatment.
    void sum_force(const UnitCell &cell, ModuleSymmetry::Symmetry *symm, PWForceResult *result);

    int nat = 0;
    int npwx = 0;

    void cal_force_loc(
        const UnitCell &ucell,
        ModuleBase::matrix &forcelc,
        const ModulePW::PW_Basis *const rho_basis,
        const ModuleBase::matrix &vloc,
        const Charge *const chr);
    void cal_force_ew(
        const UnitCell &ucell,
        ModuleBase::matrix &forceion,
        const ModulePW::PW_Basis *const rho_basis,
        const Structure_Factor *p_sf);
    void cal_force_cc(
        ModuleBase::matrix &forcecc,
        const ModulePW::PW_Basis *const rho_basis,
        const Charge *const chr,
        const bool *numeric,
        UnitCell &ucell_in,
        const int nspin,
        const bool domag,
        const bool domag_z,
        const int gga_grad);
    /**
     * @brief This routine computes the atomic force of non-local pseudopotential
     *    F^{NL}_i = \sum_{n,k}f_{nk}\sum_I \sum_{lm,l'm'}D_{l,l'}^{I} [
     *               \sum_G \langle c_{nk}(\mathbf{G+K})|\beta_{lm}^I(\mathbf{G+K})\rangle *
     *               \sum_{G'}\langle \beta_{lm}^I(\mathbf{G+K})*(-j)^l(\mathbf{G+K})_i |c_{nk}(\mathbf{G+K})\rangle ]
     *    there would be three parts in the above equation:
     *    (1) sum over becp and dbecp with D_{l,l'}^{I} ----- first line in the above equation
     *    (2) calculate becp = <psi | beta> ----- second line in the above equation
     *    (3) calculate dbecp = <psi | \nabla beta> ----- third line in the above equation
     */
    // nonlocal_a selects the RT-TDDFT projector shift in Hartree atomic units; ground-state calls pass zero.
    void cal_force_nl(
        const hamilt::NLOrbData<FPTYPE, Device> *nl_orbs,
        ModuleBase::matrix &forcenl,
        const ModuleBase::Vector3<double> &nonlocal_a,
        const std::vector<ModuleBase::Vector3<double>> &ion_vel,
        const ModuleBase::matrix &wg,
        const ModuleBase::matrix &ekb,
        const K_Vectors *p_kv,
        const ModulePW::PW_Basis_K *psi_basis,
        const Structure_Factor *p_sf,
        const pseudopot_cell_vnl &nlpp_in,
        const UnitCell &ucell_in,
        const psi::Psi<std::complex<FPTYPE>, Device> *psi_in = nullptr);
    /// @brief atomic force for DFT+U and DeltaSpin
    /// @param force_onsite , the output atomic force
    /// @param wg , the weight of k points
    /// @param wfc_basis , the plane wave basis
    /// @param ucell_in , the unit cell
    /// @param psi_in , the wave function
    void cal_force_onsite(
        ModuleBase::matrix &force_onsite,
        const ModuleBase::matrix &wg,
        const ModulePW::PW_Basis_K *wfc_basis,
        const UnitCell &ucell_in,
        const Plus_U_Base &dftu,
        const psi::Psi<std::complex<FPTYPE>, Device> *psi_in = nullptr);

    void cal_force_scc(
        ModuleBase::matrix &forcescc,
        const ModulePW::PW_Basis *const rho_basis,
        const ModuleBase::matrix &v_current,
        const bool vnew_exist,
        const bool *numeric,
        const UnitCell &ucell_in);
    void cal_force_us(
        ModuleBase::matrix &forcenl,
        const ModulePW::PW_Basis *const rho_basis,
        const pseudopot_cell_vnl &ppcell_in,
        const elecstate::ElecState &elec,
        const UnitCell &ucell);
    void cal_ylm(
        int lmax,
        int npw,
        const FPTYPE *gk_in,
        FPTYPE *ylm);
    void deriv_drhoc(
        const bool &numeric,
        const int mesh,
        const FPTYPE *r,
        const FPTYPE *rab,
        const FPTYPE *rhoc,
        FPTYPE *drhocg,
        const ModulePW::PW_Basis *const rho_basis,
        int type,
        const UnitCell &ucell_in); // used in nonlinear core correction stress
    void deriv_drhoc_scc(
        const bool &numeric,
        const int mesh,
        const FPTYPE *r,
        const FPTYPE *rab,
        const FPTYPE *rhoc,
        FPTYPE *drhocg,
        const ModulePW::PW_Basis *const rho_basis,
        const UnitCell &ucell_in); // used in nonlinear core correction stress
  protected:
    Device *ctx = {};
    base_device::DEVICE_CPU *cpu_ctx = {};
    base_device::AbacusDevice_t device = {};

  private:
    using gemm_op = ModuleBase::gemm_op<std::complex<FPTYPE>, Device>;

    using resmem_complex_op = base_device::memory::resize_memory_op<std::complex<FPTYPE>, Device>;
    using resmem_complex_h_op = base_device::memory::resize_memory_op<std::complex<FPTYPE>, base_device::DEVICE_CPU>;
    using delmem_complex_op = base_device::memory::delete_memory_op<std::complex<FPTYPE>, Device>;
    using delmem_complex_h_op = base_device::memory::delete_memory_op<std::complex<FPTYPE>, base_device::DEVICE_CPU>;
    using syncmem_complex_h2d_op = base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, base_device::DEVICE_CPU>;
    using syncmem_complex_d2h_op = base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, base_device::DEVICE_CPU, Device>;

    using resmem_var_op = base_device::memory::resize_memory_op<FPTYPE, Device>;
    using delmem_var_op = base_device::memory::delete_memory_op<FPTYPE, Device>;
    using syncmem_var_h2d_op = base_device::memory::synchronize_memory_op<FPTYPE, Device, base_device::DEVICE_CPU>;
    using syncmem_var_d2h_op = base_device::memory::synchronize_memory_op<FPTYPE, base_device::DEVICE_CPU, Device>;

    using resmem_int_op = base_device::memory::resize_memory_op<int, Device>;
    using delmem_int_op = base_device::memory::delete_memory_op<int, Device>;
    using syncmem_int_h2d_op = base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>;
};

#endif
