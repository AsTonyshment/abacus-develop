#include "vnl_pw.h"
#include "source_base/math_sphbes.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/timer.h"
#include "source_base/math_ylmreal.h"
#include "source_base/math_integral.h"
#include "source_base/math_polyint.h"

#include <vector>

void pseudopot_cell_vnl::ensure_grad_table(const UnitCell& cell)
{
    if (this->nkb > 0 && this->gradient_version_ != this->table_version_)
    {
        this->initgradq_vnl(cell);
    }
}

void pseudopot_cell_vnl::initgradq_vnl(const UnitCell &cell)
{
    this->gradient_version_ = this->table_version_;
    const int ntype = cell.ntype;
    this->tab_dq.create(ntype, this->tab.getBound2(), this->tab.getBound3());
    gradvkb.create(3, nkb, this->wfcpw->npwk_max);

    const double pref = ModuleBase::FOUR_PI / sqrt(cell.omega);
    for (int it = 0;it < ntype;it++)  
    {
        const int nbeta = cell.atoms[it].ncpp.nbeta;
        int kkbeta = cell.atoms[it].ncpp.kkbeta;
        if ( (kkbeta%2 == 0) && kkbeta>0 )
        {
            kkbeta--;
        }

        std::vector<double> djl(kkbeta);
        std::vector<double> aux(kkbeta);

        for (int ib = 0;ib < nbeta;ib++)
        {
            const int l = cell.atoms[it].ncpp.lll[ib];
            for (int iq=0; iq<this->tab_dq.getBound3(); iq++)
            {
                const double q = iq * PARAM.globalv.dq;
                ModuleBase::Sphbes::dSpherical_Bessel_dx(kkbeta, cell.atoms[it].ncpp.r.data(), q, l, djl.data());

                for (int ir = 0;ir < kkbeta;ir++)
                {
                    aux[ir] = cell.atoms[it].ncpp.betar(ib, ir) *
                              djl[ir] * pow(cell.atoms[it].ncpp.r[ir],2);
                } 
                double vqint = 0.0;
                ModuleBase::Integral::Simpson_Integral(kkbeta, aux.data(), cell.atoms[it].ncpp.rab.data(), vqint);
                this->tab_dq(it, ib, iq) = vqint * pref;
            }
        }
    }

}

void pseudopot_cell_vnl::getgradq_vnl(const UnitCell& ucell,
                                      const int ik)
{
    if(PARAM.inp.test_pp) ModuleBase::TITLE("pseudopot_cell_vnl","getvnl");
    ModuleBase::timer::start("pp_cell_vnl","getvnl");

    if(lmaxkb < 0) 
    {
        return;
    }

    const int npw = this->wfcpw->npwk[ik];

    // When the internal memory is large enough, it is better to make tmpgradvkb and tmpvkb be the number of pseudopot_cell_vnl
    // We only need to initialize them once as long as the cell is unchanged.
    ModuleBase::realArray tmpgradvkb(3, nhm, npw);
    ModuleBase::matrix tmpvkb(nhm, npw);
    std::vector<double> vq(npw);
    std::vector<double> dvq(npw);

    const int x1= (lmaxkb + 1)*(lmaxkb + 1);

    ModuleBase::matrix ylm(x1, npw);
    std::vector<ModuleBase::matrix> dylm(3);
    dylm[0].create(x1, npw);
    dylm[1].create(x1, npw);
    dylm[2].create(x1, npw);

    std::vector<ModuleBase::Vector3<double>> gk(npw);
    for (int ig = 0;ig < npw;ig++) 
    {
        gk[ig] = this->wfcpw->getgpluskcar(ik,ig);
    }

    ModuleBase::YlmReal::grad_Ylm_Real(x1, npw, gk.data(), ylm, dylm[0], dylm[1], dylm[2]);

    // GPU path skips vkb allocation in init(); allocate now if needed
    if (this->vkb.nc == 0 && this->nkb > 0 && this->vkbnc > 0) {
        this->vkb.create(this->nkb, this->vkbnc);
    }

    int jkb = 0;
    for(int it = 0;it < ucell.ntype;it++)
    {
        // calculate beta in G-space using an interpolation table
        const int nbeta = ucell.atoms[it].ncpp.nbeta;
        const int nh = ucell.atoms[it].ncpp.nh;
        int nb0 = -1;
        for( int ih = 0; ih < nh; ++ih)
        {
            int nb = this->indv(it, ih);
            if(nb != nb0)
            {
                for (int ig = 0;ig < npw;++ig)
                {
                    const double gnorm = gk[ig].norm() * ucell.tpiba;
                    vq [ig] = ModuleBase::PolyInt::Polynomial_Interpolation(
                            this->tab, it, nb, this->tab.getBound3(), PARAM.globalv.dq, gnorm );
                    dvq[ig] =ModuleBase::PolyInt::Polynomial_Interpolation(
                            this->tab_dq, it, nb, this->tab_dq.getBound3(), PARAM.globalv.dq, gnorm );
                }
                nb0 = nb;
            }
            double lmmat[9] = {0, 0, 1, -1, 0, 0, 0, -1, 0};
            for(int id = 0; id < 3; ++id)
            {
                const int lm = static_cast<int>( nhtolm(it, ih) );
                for (int ig = 0;ig < npw;++ig)
                {
                    ModuleBase::Vector3<double> gg = gk[ig];
                    double ggnorm = gg.norm();
                    if(ggnorm < 1e-8)
                    {
                        if(lm == 0 || lm > 3)
                        {
                            tmpgradvkb(id, ih, ig) = 0.0;
                        }
                        else//lm = 1,2,3;  l = 1
                        {
                            //q \to 0 : \nabla(f(q)Y(\hat{q})) = f(q)/q*sqrt(3/4/pi)*vec(-delta(lm,2), -delta(lm,3), delta(lm,1))
                            // tmpgradvkb(id, ih, ig) = 0.0;
                            tmpgradvkb(id, ih, ig) = dvq[ig] * sqrt(3.0/4.0/M_PI) * lmmat[(lm-1)*3 + id];
                        }
                    }
                    else
                    {
                        tmpgradvkb(id, ih, ig) = ylm(lm, ig) * dvq[ig] * gg[id] / ggnorm
                                                 + dylm[id](lm, ig) / this->wfcpw->tpiba
                                                       * vq[ig]; // note: dylm/d(tpiba * gx) = 1/tpiba * dylm/dgx
                    }
                    tmpvkb(ih, ig) = ylm(lm,ig) * vq[ig];
                }
            }
        }

        // vkb1 contains all betas including angular part for type nt
        // now add the structure factor and factor (-i)^l
        for (int ia=0; ia<ucell.atoms[it].na; ia++) 
        {
            std::complex<double> *sk = this->psf->get_sk(ik, it, ia, this->wfcpw);

            for (int ih = 0;ih < nh;++ih)
            {
                std::complex<double> pref = pow( ModuleBase::NEG_IMAG_UNIT, nhtol(it, ih));
                std::complex<double>* pvkb = &this->vkb(jkb, 0);
                for (int id = 0; id < 3 ; ++id)
                {
                    std::complex<double>* pgvkb = &this->gradvkb(id, jkb, 0);
                    for (int ig = 0;ig < npw;++ig)
                    {
                        std::complex<double> skig = sk[ig];
                        pvkb[ig] = tmpvkb(ih, ig) * skig * pref;
                        // std::complex<double> dskig = ModuleBase::NEG_IMAG_UNIT * (ucell.atoms[it].tau[ia][id] * this->wfcpw->lat0) * skig;
                        // pgvkb[ig] = tmpgradvkb(id, ih, ig) * skig * pref +  tmpvkb(ih, ig) * dskig * pref;
                        // The second term will be eliminate when doing <psi|beta>Dij<beta|psi> or we can say (\nabla_q+\nabla_q')S(q'-q) = 0
                        pgvkb[ig] = tmpgradvkb(id, ih, ig) * skig * pref;
                    }
                } //end id
                ++jkb;
            } // end ih
            
            delete [] sk;
        } // end ia
    } // enddo

    ModuleBase::timer::end("pp_cell_vnl","getvnl");

    return;
}

// ====================================================================
// getgradq_vnl_td:
// Calculate the gradient of nonlocal pseudopotential projectors in
// velocity gauge: gradient_p beta(p), where p = k + G + A(t).
// ====================================================================
void pseudopot_cell_vnl::getgradq_vnl_td(const UnitCell& ucell, const int ik, const ModuleBase::Vector3<double>& vector_potential) const
{
    ModuleBase::timer::start("pp_cell_vnl", "getgradq_vnl_td");

    if (this->lmaxkb < 0)
    {
        ModuleBase::timer::end("pp_cell_vnl", "getgradq_vnl_td");
        return;
    }

    const int npw = this->wfcpw->npwk[ik];
    ModuleBase::realArray projector_gradient(3, this->nhm, npw);
    std::vector<double> radial_value(npw);
    std::vector<double> radial_derivative(npw);

    const int ylm_count = (this->lmaxkb + 1) * (this->lmaxkb + 1);
    ModuleBase::matrix ylm(ylm_count, npw);
    std::vector<ModuleBase::matrix> ylm_gradient(3);
    for (int direction = 0; direction < 3; ++direction)
    {
        ylm_gradient[direction].create(ylm_count, npw);
    }

    const ModuleBase::Vector3<double> reduced_vector_potential = vector_potential / ucell.tpiba;
    std::vector<ModuleBase::Vector3<double>> shifted_gk(npw);
    for (int ig = 0; ig < npw; ++ig)
    {
        shifted_gk[ig] = this->wfcpw->getgpluskcar(ik, ig) + reduced_vector_potential;
    }

    ModuleBase::YlmReal::grad_Ylm_Real(ylm_count, npw, shifted_gk.data(), ylm, ylm_gradient[0], ylm_gradient[1], ylm_gradient[2]);

    if (this->gradvkb.ptr == nullptr && this->nkb > 0 && this->wfcpw->npwk_max > 0)
    {
        this->gradvkb.create(3, this->nkb, this->wfcpw->npwk_max);
    }

    static const double l1_direction[9] = {0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0};
    int projector_index = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        const int projector_count = ucell.atoms[it].ncpp.nh;
        int previous_beta = -1;
        for (int projector = 0; projector < projector_count; ++projector)
        {
            const int beta = static_cast<int>(this->indv(it, projector));
            if (beta != previous_beta)
            {
                for (int ig = 0; ig < npw; ++ig)
                {
                    const double momentum_norm = shifted_gk[ig].norm() * ucell.tpiba;
                    this->check_vnl_range(momentum_norm, PARAM.globalv.dq, true);
                    radial_value[ig] = ModuleBase::PolyInt::Polynomial_Interpolation(this->tab,
                                                                                     it,
                                                                                     beta,
                                                                                     this->tab.getBound3(),
                                                                                     PARAM.globalv.dq,
                                                                                     momentum_norm);
                    radial_derivative[ig] = ModuleBase::PolyInt::Polynomial_Interpolation(this->tab_dq,
                                                                                          it,
                                                                                          beta,
                                                                                          this->tab_dq.getBound3(),
                                                                                          PARAM.globalv.dq,
                                                                                          momentum_norm);
                }
                previous_beta = beta;
            }

            const int lm = static_cast<int>(this->nhtolm(it, projector));
            for (int direction = 0; direction < 3; ++direction)
            {
                for (int ig = 0; ig < npw; ++ig)
                {
                    const ModuleBase::Vector3<double>& momentum = shifted_gk[ig];
                    const double reduced_norm = momentum.norm();
                    if (reduced_norm < 1.0e-8)
                    {
                        if (lm == 0 || lm > 3)
                        {
                            projector_gradient(direction, projector, ig) = 0.0;
                        }
                        else
                        {
                            projector_gradient(direction, projector, ig)
                                = radial_derivative[ig] * std::sqrt(3.0 / (4.0 * M_PI)) * l1_direction[(lm - 1) * 3 + direction];
                        }
                    }
                    else
                    {
                        projector_gradient(direction, projector, ig)
                            = ylm(lm, ig) * radial_derivative[ig] * momentum[direction] / reduced_norm
                              + ylm_gradient[direction](lm, ig) * radial_value[ig] / this->wfcpw->tpiba;
                    }
                }
            }
        }

        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            // The atom-dependent A phase cancels between the two projectors in
            // each same-atom nonlocal outer product, so use the static factor.
            std::complex<double>* structure_factor = this->psf->get_sk(ik, it, ia, this->wfcpw);

            for (int projector = 0; projector < projector_count; ++projector)
            {
                const std::complex<double> angular_phase = std::pow(ModuleBase::NEG_IMAG_UNIT, this->nhtol(it, projector));
                for (int direction = 0; direction < 3; ++direction)
                {
                    std::complex<double>* output = &this->gradvkb(direction, projector_index, 0);
                    for (int ig = 0; ig < npw; ++ig)
                    {
                        output[ig] = projector_gradient(direction, projector, ig) * structure_factor[ig] * angular_phase;
                    }
                }
                ++projector_index;
            }
            delete[] structure_factor;
        }
    }

    ModuleBase::timer::end("pp_cell_vnl", "getgradq_vnl_td");
}
