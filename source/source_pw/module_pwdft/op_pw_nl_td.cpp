#include "op_pw_nl_td.h"

#include "source_base/parallel_reduce.h"
#include "source_base/parallel_comm.h"
#include "source_base/parallel_device.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_estate/module_pot/h_tddft_pw.h"
#include "source_io/module_parameter/parameter.h"

namespace hamilt
{

template <typename T, typename Device>
TDNonlocalPW<T, Device>::TDNonlocalPW(const int* isk_in,
                                      const pseudopot_cell_vnl* ppcell_in,
                                      const UnitCell* ucell_in,
                                      const ModulePW::PW_Basis_K* wfc_basis)
{
    if (isk_in == nullptr || ppcell_in == nullptr || ucell_in == nullptr)
    {
        ModuleBase::WARNING_QUIT("TDNonlocalPW", "Constructor failed, null pointers detected!");
    }

    this->classname = "TDNonlocalPW";
    this->cal_type = calculation_type::pw_nonlocal;
    this->isk = isk_in;
    this->ppcell = ppcell_in;
    this->ucell = ucell_in;
    this->wfcpw = wfc_basis;
    this->deeq = this->ppcell->template get_deeq_data<Real>();
    // Non-collinear spin is not supported by this operator.
    this->deeq_nc = this->ppcell->template get_deeq_nc_data<Real>();
}

template <typename T, typename Device>
TDNonlocalPW<T, Device>::~TDNonlocalPW()
{
    delmem_complex_op()(this->ps);
    delmem_complex_op()(this->becp);
    delmem_complex_op()(this->vkb_td);
}

template <typename T, typename Device>
void TDNonlocalPW<T, Device>::init(const int ik_in)
{
    ModuleBase::timer::start("TDNonlocalPW", "init_vkb_td");
    this->ik = ik_in;

    // Refresh the shifted projectors when nonlocal projectors are present.
    if (this->ppcell->nkb > 0 && this->wfcpw->npwk[ik_in] > 0)
    {
        // Allocate the time-dependent projector cache on first use.
        if (this->vkb_td == nullptr)
        {
            resmem_complex_op()(this->vkb_td, this->ppcell->nkb * this->wfcpw->npwk_max, "TDNL::vkb_td");
        }

        // Convert the legacy mirror to the Hartree-unit vector potential.
        ModuleBase::Vector3<double> A_au = elecstate::H_TDDFT_pw::At / 2.0;

        // Generate projectors shifted by the current vector potential.
        this->ppcell->getvnl_td(this->ctx, *this->ucell, this->ik, A_au, this->vkb_td);
    }

    if (this->next_op != nullptr)
    {
        this->next_op->init(ik_in);
    }
    ModuleBase::timer::end("TDNonlocalPW", "init_vkb_td");
}

template <typename T, typename Device>
void TDNonlocalPW<T, Device>::add_nonlocal_pp(T* hpsi_in, const T* becp_in, const int m) const
{
    ModuleBase::timer::start("TDNonlocalPW", "add_nonlocal_pp");

    int nkb = this->ppcell->nkb;
    if (this->ps_capacity_ < static_cast<size_t>(nkb) * m)
    {
        resmem_complex_op()(this->ps, nkb * m, "TDNL::ps");
        this->ps_capacity_ = static_cast<size_t>(nkb) * m;
    }
    setmem_complex_op()(this->ps, 0, nkb * m);

    int sum = 0;
    int iat = 0;

    // This path assumes one spinor component.
    const int current_spin = this->isk[this->ik];

    for (int it = 0; it < this->ucell->ntype; it++)
    {
        const int nproj = this->ucell->atoms[it].ncpp.nh;

        nonlocal_op()(this->ctx, // device context
                      this->ucell->atoms[it].na,
                      m,
                      nproj, // four loop size
                      sum,
                      iat,
                      current_spin,
                      nkb, // additional index params
                      this->ppcell->deeq.getBound2(),
                      this->ppcell->deeq.getBound3(),
                      this->ppcell->deeq.getBound4(),
                      this->deeq, // array of data
                      this->ps,
                      becp_in);
    }

    // Contract the coefficients with the shifted projectors into hpsi.
    char transa = 'N';
    char transb = 'T';

    if (m == 1)
    {
        int inc = 1;
        gemv_op()(transa, this->npw, nkb, &this->one, this->vkb_td, this->ppcell->vkbnc, this->ps, inc, &this->one, hpsi_in, inc);
    }
    else
    {
#ifdef __DSP
        ModuleBase::gemm_op_mt<T, Device>()
#else
        gemm_op()
#endif
            (transa,
             transb,
             this->npw,
             m,
             nkb,
             &this->one,
             this->vkb_td,
             this->ppcell->vkbnc,
             this->ps,
             m,
             &this->one,
             hpsi_in,
             this->max_npw);
    }
    ModuleBase::timer::end("TDNonlocalPW", "add_nonlocal_pp");
}

template <typename T, typename Device>
void TDNonlocalPW<T, Device>::act(const int nbands,
                                  const int nbasis,
                                  const int npol,
                                  const T* tmpsi_in,
                                  T* tmhpsi,
                                  const int ngk_ik,
                                  const bool is_first_node) const
{
    ModuleBase::timer::start("Operator", "TDNonlocalPW");

    // ---------------- Debug output; remove after development ---------------- //
    // std::cout << "TDNonlocalPW::act called with nbands=" << nbands << ", nbasis=" << nbasis << ", npol=" << npol
    //           << ", ngk_ik=" << ngk_ik << ", is_first_node=" << is_first_node << std::endl;
    // ------------------------------------------------------------------------ //

    if (is_first_node)
    {
        setmem_complex_op()(tmhpsi, 0, nbasis * nbands / npol);
    }

    this->npw = ngk_ik;
    this->max_npw = nbasis / npol;
    this->npol = npol;

    if (this->ppcell->nkb > 0)
    {
        int nkb = this->ppcell->nkb;
        if (this->nkb_m < nbands * nkb)
        {
            resmem_complex_op()(this->becp, nbands * nkb, "TDNL::becp");
            this->nkb_m = nbands * nkb;
        }

        char transa = 'C';
        char transb = 'N';

        if (this->npw == 0)
        {
            setmem_complex_op()(this->becp, 0, nbands * nkb);
        }
        else if (nbands == 1)
        {
            int inc = 1;
            gemv_op()(transa, this->npw, nkb, &this->one, this->vkb_td, this->ppcell->vkbnc, tmpsi_in, inc, &this->zero, this->becp, inc);
        }
        else
        {
#ifdef __DSP
            ModuleBase::gemm_op_mt<T, Device>()
#else
            gemm_op()
#endif
                (transa,
                 transb,
                 nkb,
                 nbands,
                 this->npw,
                 &this->one,
                 this->vkb_td,
                 this->ppcell->vkbnc,
                 tmpsi_in,
                 this->max_npw,
                 &this->zero,
                 this->becp,
                 nkb);
        }

#ifdef __MPI
        Parallel_Common::reduce_dev<T, Device>(this->becp, nkb * nbands, POOL_WORLD);
#endif
        if (this->npw > 0)
        {
            this->add_nonlocal_pp(tmhpsi, this->becp, nbands);
        }
    }
    ModuleBase::timer::end("Operator", "TDNonlocalPW");
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
