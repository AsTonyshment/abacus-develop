#ifndef OP_PW_VEL_TD_H
#define OP_PW_VEL_TD_H

#include "op_pw.h"
#include "source_pw/module_pwdft/velocity_workspace.h"
#include "source_pw/module_pwdft/projector_gradient.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/vnl_pw.h"

namespace hamilt
{

// Velocity operator in velocity gauge: mv = p + A(t) + i[V_NL(t), r]
template <typename FPTYPE, typename Device = base_device::DEVICE_CPU>
class TDVelocity
{
  public:
    TDVelocity(const ModulePW::PW_Basis_K* wfcpw_in,
               const int* isk_in,
               pseudopot_cell_vnl* ppcell_in,
               const UnitCell* ucell_in,
               const bool nonlocal_in = true);
    ~TDVelocity();

    void init(const int ik_in);
    /** @brief Refresh the borrowed spin map after basis redistribution. */
    void set_spin(const int* spins) { isk = spins; }

    /**
     * @brief calculate \hat{v}_{\mathrm{V}}|\psi>
     * * @param psi_in Psi class which contains wavefunction information
     * @param n_npwx nbands * NPOL
     * @param tmpsi_in |\psi_i>    size: n_npwx * npwx
     * @param tmvpsi \hat{v}|\psi> size: 3 * n_npwx * npwx
     * @param add true : tmvpsi = tmvpsi + v|\psi>  false: tmvpsi = v|\psi>
     */
    void act(const psi::Psi<std::complex<FPTYPE>, Device>* psi_in,
             const int n_npwx,
             const std::complex<FPTYPE>* tmpsi_in,
             std::complex<FPTYPE>* tmvpsi,
             const bool add = false) const;

    bool nonlocal = true;

  private:
    const ModulePW::PW_Basis_K* wfcpw = nullptr;
    const int* isk = nullptr;
    pseudopot_cell_vnl* ppcell = nullptr;
    const UnitCell* ucell = nullptr;
    int ik = 0;
    double tpiba = 0.0;
    int momentum_capacity_ = 0;
    int projector_capacity_ = 0;
    ProjectorGradient<FPTYPE, Device> gradient_;
    mutable VelocityWorkspace<FPTYPE, Device> contraction_;
    Device* ctx = {}; // Device context corresponding to target architecture

    FPTYPE* gx_ = nullptr; ///<[Device, npwx] x component of p(t) = G+K+A(t)
    FPTYPE* gy_ = nullptr; ///<[Device, npwx] y component of p(t) = G+K+A(t)
    FPTYPE* gz_ = nullptr; ///<[Device, npwx] z component of p(t) = G+K+A(t)

    std::complex<FPTYPE>* vkb_td_ = nullptr;     ///<[Device, nkb * npwk_max] TD nonlocal pseudopotential vkb
    std::complex<FPTYPE>* gradvkb_td_ = nullptr; ///<[Device, 3*nkb * npwk_max] gradient of TD nonlocal pseudopotential gradvkb

    using Complex = std::complex<FPTYPE>;
    using resmem_var_op = base_device::memory::resize_memory_op<FPTYPE, Device>;
    using delmem_var_op = base_device::memory::delete_memory_op<FPTYPE, Device>;
    using syncmem_var_h2d_op = base_device::memory::synchronize_memory_op<FPTYPE, Device, base_device::DEVICE_CPU>;
    using resmem_complex_op = base_device::memory::resize_memory_op<std::complex<FPTYPE>, Device>;
    using setmem_complex_op = base_device::memory::set_memory_op<std::complex<FPTYPE>, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<std::complex<FPTYPE>, Device>;
    using syncmem_complex_d2h_op = base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, base_device::DEVICE_CPU, Device>;
    using syncmem_complex_h2d_op = base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, base_device::DEVICE_CPU>;
};

} // namespace hamilt

#endif
