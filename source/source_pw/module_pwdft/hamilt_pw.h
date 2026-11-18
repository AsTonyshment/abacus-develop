#ifndef HAMILTPW_H
#define HAMILTPW_H

#include "source_base/kernels/math_kernel_op.h"
#include "source_base/macros.h"
#include "source_cell/klist.h"
#include "source_estate/module_pot/potential_new.h"
#include "source_hamilt/hamilt.h"
#include "source_pw/module_pwdft/dftu_base.h"
#include "source_pw/module_pwdft/exx_helper.h"
#include "source_pw/module_pwdft/vnl_pw.h"
#include "source_pw/module_pwdft/nl_proj_pw.h"

/// General_Exx_Info forward declaration, full definition in general_exx_info.h
struct General_Exx_Info;

namespace hamilt
{

template <typename T, typename Device = base_device::DEVICE_CPU>
class HamiltPW : public Hamilt<T, Device>
{
  private:
    // Note GetTypeReal<T>::type will
    // return T if T is real type(float, double),
    // otherwise return the real type of T(complex<float>, std::complex<double>)
    using Real = typename GetTypeReal<T>::type;
    using syncmem_complex_h2d_op = base_device::memory::synchronize_memory_op<T, Device, base_device::DEVICE_CPU>;

  public:
    // nonlocal_field controls nonlocal electromagnetic coupling only in RT-TDDFT velocity gauge.
    HamiltPW(
        elecstate::Potential *pot_in,
        ModulePW::PW_Basis_K *wfc_basis,
        K_Vectors *p_kv,
        pseudopot_cell_vnl *nlpp,
        Plus_U_Base *p_dftu,
        const UnitCell *ucell,
        const General_Exx_Info *exx_info,
        bool nonlocal_field);

    ~HamiltPW();

    // for target K point, update consequence of hPsi() and matrix()
    void updateHk(const int ik) override;

    /** @brief Bind RT-TDDFT local-potential buffers, vector potential and nuclear velocities at the same operator time.
     *  @param ion_vel Type-major velocities in Hartree atomic units; empty selects RT-TDDFT without the nuclear velocity phase.
     *  @param tau Explicit Cartesian sampling positions in units of lat0.
     *  @note The caller keeps potential buffers alive and selects endpoint operators after propagation.
     */
    void bind_td_state(const Real *veff, const ModuleBase::Vector3<double> &A_ha,
                       const std::vector<ModuleBase::Vector3<double>> &ion_vel,
                       const std::vector<ModuleBase::Vector3<double>> &tau);

    /** @brief Access RT projectors owned by the nonlocal operator. */
    NLProjPW<Real, Device> &td_projectors();
    /** @brief Borrow the nonlocal overlaps produced by the last Hpsi call. */
    const T *td_projections() const;

    void sPsi(
        const T *psi_in, // psi
        T *spsi,         // spsi
        const int nrow,  // dimension of spsi: nbands * nrow
        const int npw,   // number of plane waves
        const int nbands // number of bands
    ) const override;

    void set_exx_helper(Exx_Helper<T, Device> &exx_helper_in);

  protected:
    // used in sPhi, which are calculated in hPsi or sPhi
    const pseudopot_cell_vnl *ppcell = nullptr;
    const UnitCell *const ucell = nullptr;
    mutable T *vkb = nullptr;
    Real *qq_nt = nullptr;
    T *qq_so = nullptr;

    Device *ctx = {};
    using gemv_op = ModuleBase::gemv_op<T, Device>;
    using gemm_op = ModuleBase::gemm_op<T, Device>;
    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    using syncmem_op = base_device::memory::synchronize_memory_op<T, Device, Device>;
};

} // namespace hamilt

#endif
