#include "op_pw_vel_td.h"

#include "source_base/kernels/math_kernel_op.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_io/module_parameter/parameter.h"
#include "source_estate/module_pot/h_tddft_pw.h"

namespace hamilt
{

template <typename FPTYPE, typename Device>
TDVelocity<FPTYPE, Device>::TDVelocity(const ModulePW::PW_Basis_K* wfcpw_in,
                                       const int* isk_in,
                                       pseudopot_cell_vnl* ppcell_in,
                                       const UnitCell* ucell_in,
                                       const bool nonlocal_in)
{
    if (wfcpw_in == nullptr || isk_in == nullptr || ppcell_in == nullptr || ucell_in == nullptr)
    {
        ModuleBase::WARNING_QUIT("TDVelocity", "Constructor of Operator::TDVelocity failed, please check your code!");
    }
    this->wfcpw = wfcpw_in;
    this->isk = isk_in;
    this->ppcell = ppcell_in;
    this->ucell = ucell_in;
    this->nonlocal = nonlocal_in;
    this->tpiba = ucell_in->tpiba;

    // Prepare the radial derivative table used by the nonlocal term.
    if (this->nonlocal)
    {
        this->ppcell->ensure_grad_table(*this->ucell);
    }
}

template <typename FPTYPE, typename Device>
TDVelocity<FPTYPE, Device>::~TDVelocity()
{
    delmem_var_op()(this->gx_);
    delmem_var_op()(this->gy_);
    delmem_var_op()(this->gz_);
    delmem_complex_op()(this->vkb_td_);
    delmem_complex_op()(this->gradvkb_td_);
}

template <typename FPTYPE, typename Device>
void TDVelocity<FPTYPE, Device>::init(const int ik_in)
{
    ModuleBase::timer::start("Operator", "TDVelocity_init");
    this->ik = ik_in;
    this->tpiba = this->ucell->tpiba;
    const int npw = this->wfcpw->npwk[ik_in];
    const int npwk_max = this->wfcpw->npwk_max;

    // init memory for shifted p(t)
    if (npw > momentum_capacity_)
    {
        resmem_var_op()(gx_, npw);
        resmem_var_op()(gy_, npw);
        resmem_var_op()(gz_, npw);
        momentum_capacity_ = npw;
    }

    // Obtain the vector potential for the current time step.
    // H_TDDFT_pw::At is effectively 2 * A_au, so we divide by 2.0
    ModuleBase::Vector3<double> A_au = elecstate::H_TDDFT_pw::At / 2.0;

    std::vector<FPTYPE> gtmp_x(npw), gtmp_y(npw), gtmp_z(npw);
    std::vector<FPTYPE*> gtmp_ptr = {this->gx_, this->gy_, this->gz_};

    for (int ig = 0; ig < npw; ++ig)
    {
        const ModuleBase::Vector3<double> tmpg = wfcpw->getgpluskcar(this->ik, ig);
        // p(t) = (k + G) * tpiba + A(t)
        gtmp_x[ig] = static_cast<FPTYPE>(tmpg.x * this->tpiba + A_au.x);
        gtmp_y[ig] = static_cast<FPTYPE>(tmpg.y * this->tpiba + A_au.y);
        gtmp_z[ig] = static_cast<FPTYPE>(tmpg.z * this->tpiba + A_au.z);
    }
    if (npw > 0)
    {
        syncmem_var_h2d_op()(gtmp_ptr[0], gtmp_x.data(), npw);
        syncmem_var_h2d_op()(gtmp_ptr[1], gtmp_y.data(), npw);
        syncmem_var_h2d_op()(gtmp_ptr[2], gtmp_z.data(), npw);
    }

    // Calculate time-dependent nonlocal pseudopotential vkb and its gradient gradvkb
    if (npw > 0 && this->ppcell->nkb > 0 && this->nonlocal)
    {
        const int nkb = this->ppcell->nkb;
        const int npwk_max = this->wfcpw->npwk_max;

        if (nkb*npwk_max > projector_capacity_)
        {
            resmem_complex_op()(vkb_td_, nkb*npwk_max, "TDVel::vkb_td");
            resmem_complex_op()(gradvkb_td_, 3*nkb*npwk_max, "TDVel::gradvkb_td");
            projector_capacity_ = nkb*npwk_max;
        }

        // Generate projectors shifted by the current vector potential.
        this->ppcell->getvnl_td(this->ctx, *this->ucell, ik_in, A_au, this->vkb_td_);

        this->gradient_.calculate(this->ppcell, *this->ucell, *this->wfcpw, ik_in, A_au,
                                  PARAM.globalv.dq, this->gradvkb_td_);
    }

    ModuleBase::timer::end("Operator", "TDVelocity_init");
}

template <typename FPTYPE, typename Device>
void TDVelocity<FPTYPE, Device>::act(const psi::Psi<std::complex<FPTYPE>, Device>* psi_in,
                                     const int n_npwx,
                                     const std::complex<FPTYPE>* psi0,
                                     std::complex<FPTYPE>* vpsi,
                                     const bool add) const
{
    ModuleBase::timer::start("Operator", "TDVelocity_act");
    const int npw = this->wfcpw->npwk[this->ik];
    const int max_npw = this->wfcpw->npwk_max;
    const int npol = psi_in->get_npol();

    std::vector<FPTYPE*> gtmp_ptr = {this->gx_, this->gy_, this->gz_};

    // ---------------------------------------------
    //       p(t) = \hat{p} + A(t)
    // ---------------------------------------------
    for (int id = 0; id < 3; ++id)
    {
        const Complex* tmpsi_in = psi0;
        Complex* tmpvpsi = vpsi + id * n_npwx * max_npw;
        for (int ib = 0; npw > 0 && ib < n_npwx; ++ib)
        {
            ModuleBase::vector_mul_vector_op<Complex, Device, FPTYPE>()(npw, tmpvpsi, tmpsi_in, gtmp_ptr[id], add);
            tmpvpsi += max_npw;
            tmpsi_in += max_npw;
        }
    }

    // ---------------------------------------------
    // i[V_NL(t), r] = (\nabla_p+\nabla_p')V_{NL}(p(t),p'(t))
    // |\beta><\beta|\psi>
    // ---------------------------------------------
    if (this->ppcell->nkb <= 0 || !this->nonlocal)
    {
        ModuleBase::timer::end("Operator", "TDVelocity_act");
        return;
    }

    // 1. <\beta|\psi> and <\nabla\beta|\psi>
    const int block = this->ppcell->nkb * n_npwx;
    Complex* becp1_ = this->contraction_.prepare(block);
    Complex* becp2_ = becp1_ + block;
    Complex* ps1_ = becp1_ + 4 * block;
    Complex* ps2_ = ps1_ + block;

    const int nkb = this->ppcell->nkb;
    const int nkb3 = 3 * nkb;
    Complex one = 1.0;
    Complex zero = 0.0;

    Complex* vkb_d = this->vkb_td_;
    Complex* gradvkb_d = this->gradvkb_td_;

    if (npw == 0)
    {
        // Empty local projections must contribute zero to the pool reduction.
        base_device::memory::set_memory_op<Complex, Device>()(becp1_, 0, 4 * block);
    }
    else if (n_npwx == 1)
    {
        int inc = 1;
        ModuleBase::gemv_op<Complex, Device>()('C', npw, nkb, &one, vkb_d, max_npw, psi0, inc, &zero, becp1_, inc);
        ModuleBase::gemv_op<Complex, Device>()('C', npw, nkb3, &one, gradvkb_d, max_npw, psi0, inc, &zero, becp2_, inc);
    }
    else
    {
        ModuleBase::gemm_op<Complex, Device>()('C', 'N', nkb, n_npwx, npw, &one, vkb_d, max_npw, psi0, max_npw, &zero, becp1_, nkb);
        ModuleBase::gemm_op<Complex, Device>()('C', 'N', nkb3, n_npwx, npw, &one, gradvkb_d, max_npw, psi0, max_npw, &zero, becp2_, nkb3);
    }

    if (npol != 1)
    {
        ModuleBase::WARNING_QUIT("Velocity", "Non-collinear velocity is not supported.");
    }
    this->contraction_.contract(*this->ucell, *this->ppcell, this->isk[this->ik],
                               n_npwx, FPTYPE(0.5), GlobalV::NPROC_IN_POOL, becp1_);

    if (npw == 0)
    {
        ModuleBase::timer::end("Operator", "TDVelocity_act");
        return;
    }

    if (n_npwx == 1)
    {
        int inc = 1;
        for (int id = 0; id < 3; ++id)
        {
            int vkbshift = id * max_npw * nkb;
            int ps2shift = id * nkb;
            int npwshift = id * max_npw;

            ModuleBase::gemv_op<Complex,
                                Device>()('N', npw, nkb, &one, gradvkb_d + vkbshift, max_npw, ps1_, inc, &one, vpsi + npwshift, inc);
            ModuleBase::gemv_op<Complex, Device>()('N', npw, nkb, &one, vkb_d, max_npw, ps2_ + ps2shift, inc, &one, vpsi + npwshift, inc);
        }
    }
    else
    {
        for (int id = 0; id < 3; ++id)
        {
            int vkbshift = id * max_npw * nkb;
            int ps2shift = id * n_npwx * nkb;
            int npwshift = id * max_npw * n_npwx;

            ModuleBase::gemm_op<Complex, Device>()('N',
                                                   'T',
                                                   npw,
                                                   n_npwx,
                                                   nkb,
                                                   &one,
                                                   gradvkb_d + vkbshift,
                                                   max_npw,
                                                   ps1_,
                                                   n_npwx,
                                                   &one,
                                                   vpsi + npwshift,
                                                   max_npw);
            ModuleBase::gemm_op<Complex, Device>()('N',
                                                   'T',
                                                   npw,
                                                   n_npwx,
                                                   nkb,
                                                   &one,
                                                   vkb_d,
                                                   max_npw,
                                                   ps2_ + ps2shift,
                                                   n_npwx,
                                                   &one,
                                                   vpsi + npwshift,
                                                   max_npw);
        }
    }



    ModuleBase::timer::end("Operator", "TDVelocity_act");
    return;
}

template class TDVelocity<double, base_device::DEVICE_CPU>;
template class TDVelocity<float, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class TDVelocity<double, base_device::DEVICE_GPU>;
template class TDVelocity<float, base_device::DEVICE_GPU>;
#endif

} // namespace hamilt
