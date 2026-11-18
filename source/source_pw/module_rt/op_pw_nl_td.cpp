#include "source_pw/module_rt/op_pw_nl_td.h"

#include "source_base/timer.h"
#include "source_base/tool_quit.h"

namespace hamilt
{

template <typename T, typename Device>
TDNonlocalPW<T, Device>::TDNonlocalPW(
    const int *isk_in,
    const pseudopot_cell_vnl *ppcell_in,
    const UnitCell *ucell_in,
    const ModulePW::PW_Basis_K *wfc_basis,
    const bool &nonlocal_field)
    : nonlocal_field_(nonlocal_field)
{
    if (isk_in == nullptr || ppcell_in == nullptr || ucell_in == nullptr || wfc_basis == nullptr)
    {
        ModuleBase::WARNING_QUIT("TDNonlocalPW", "Constructor failed, null pointers detected!");
    }

    this->classname = "TDNonlocalPW";
    this->cal_type = calculation_type::pw_nonlocal;
    this->isk_ = isk_in;
    this->ppcell_ = ppcell_in;
    this->ucell_ = ucell_in;
    this->wfcpw_ = wfc_basis;
    projectors_.reset(new NLProjPW<typename GetTypeReal<T>::type, Device>(*ucell_in, *wfc_basis, *ppcell_in));
}

template <typename T, typename Device>
void TDNonlocalPW<T, Device>::init(const int ik_in)
{
    ModuleBase::timer::start("TDNonlocalPW", "init");
    this->ik = ik_in;

    // Refresh the shifted projectors when nonlocal projectors are present.
    if (this->ppcell_->nkb > 0 && this->wfcpw_->npwk[ik_in] > 0)
    {
        // Density correctors share the projectors of this physical sampling point.
        this->vkb_td_ = projectors_->beta(ik_in);
    }

    if (this->next_op != nullptr)
    {
        this->next_op->init(ik_in);
    }
    ModuleBase::timer::end("TDNonlocalPW", "init");
}

template <typename T, typename Device>
void TDNonlocalPW<T, Device>::act(
    const int nbands,
    const int nbasis,
    const int npol,
    const T *tmpsi_in,
    T *tmhpsi,
    const int ngk_ik,
    const bool is_first_node) const
{
    this->workspace_.apply(
        *this->ucell_,
        *this->ppcell_,
        *this->wfcpw_,
        this->isk_[this->ik],
        nbands,
        nbasis,
        npol,
        ngk_ik,
        is_first_node,
        this->vkb_td_,
        tmpsi_in,
        tmhpsi);
}

// Explicit CPU template instantiations.
template class TDNonlocalPW<std::complex<float>, base_device::DEVICE_CPU>;
template class TDNonlocalPW<std::complex<double>, base_device::DEVICE_CPU>;

// ================= GPU explicit instantiations =================
#if ((defined __CUDA) || (defined __ROCM))
template class TDNonlocalPW<std::complex<float>, base_device::DEVICE_GPU>;
template class TDNonlocalPW<std::complex<double>, base_device::DEVICE_GPU>;
#endif
// ===============================================================

} // namespace hamilt
