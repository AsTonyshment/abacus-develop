#ifndef OP_PW_NL_TD_H
#define OP_PW_NL_TD_H

#include "op_pw.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_pw/module_pwdft/kernels/nonlocal_op.h"
#include "source_pw/module_pwdft/vnl_pw.h"

namespace hamilt
{

template <typename T, typename Device = base_device::DEVICE_CPU>
class TDNonlocalPW : public OperatorPW<T, Device>
{
  private:
    using Real = typename GetTypeReal<T>::type;

  public:
    TDNonlocalPW(const int* isk_in, const pseudopot_cell_vnl* ppcell_in, const UnitCell* ucell_in, const ModulePW::PW_Basis_K* wfc_basis);

    virtual ~TDNonlocalPW();
    virtual void init(const int ik_in) override;

    virtual void act(const int nbands,
                     const int nbasis,
                     const int npol,
                     const T* tmpsi_in,
                     T* tmhpsi,
                     const int ngk_ik = 0,
                     const bool is_first_node = false) const override;

  private:
    void add_nonlocal_pp(T* hpsi_in, const T* becp, const int m) const;

    mutable int npw = 0;
    mutable int max_npw = 0;
    mutable int npol = 0;
    mutable size_t nkb_m = 0;
    mutable size_t ps_capacity_ = 0;

    const int* isk = nullptr;
    const pseudopot_cell_vnl* ppcell = nullptr;
    const UnitCell* ucell = nullptr;
    const ModulePW::PW_Basis_K* wfcpw = nullptr;

    mutable T* ps = nullptr;
    mutable T* becp = nullptr;
    mutable T* vkb_td = nullptr; // Time-dependent projector cache.

    Device* ctx = {};
    Real* deeq = nullptr;
    T* deeq_nc = nullptr;

    using gemv_op = ModuleBase::gemv_op<T, Device>;
    using gemm_op = ModuleBase::gemm_op<T, Device>;
    using nonlocal_op = nonlocal_pw_op<Real, Device>;
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;

    T one{1, 0};
    T zero{0, 0};
};

} // namespace hamilt
#endif
