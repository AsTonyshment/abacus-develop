#include "source_base/math_integral.h"
#include "source_base/math_sphbes.h"
#include "source_base/timer.h"
#include "source_pw/module_pwdft/vnl_pw.h"

#include <vector>

void pseudopot_cell_vnl::ensure_grad_table(const UnitCell &cell)
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

    const double pref = ModuleBase::FOUR_PI / sqrt(cell.omega);
    for (int it = 0; it < ntype; it++)
    {
        const int nbeta = cell.atoms[it].ncpp.nbeta;
        int kkbeta = cell.atoms[it].ncpp.kkbeta;
        if ((kkbeta % 2 == 0) && kkbeta > 0)
        {
            kkbeta--;
        }

        std::vector<double> djl(kkbeta);
        std::vector<double> aux(kkbeta);

        for (int ib = 0; ib < nbeta; ib++)
        {
            const int l = cell.atoms[it].ncpp.lll[ib];
            for (int iq = 0; iq < this->tab_dq.getBound3(); iq++)
            {
                const double q = iq * this->table_dq_;
                ModuleBase::Sphbes::dSpherical_Bessel_dx(kkbeta, cell.atoms[it].ncpp.r.data(), q, l, djl.data());

                for (int ir = 0; ir < kkbeta; ir++)
                {
                    aux[ir] = cell.atoms[it].ncpp.betar(ib, ir) * djl[ir] * pow(cell.atoms[it].ncpp.r[ir], 2);
                }
                double vqint = 0.0;
                ModuleBase::Integral::Simpson_Integral(kkbeta, aux.data(), cell.atoms[it].ncpp.rab.data(), vqint);
                this->tab_dq(it, ib, iq) = vqint * pref;
            }
        }
    }
}

void pseudopot_cell_vnl::ensure_hess(const UnitCell &cell)
{
    ModuleBase::timer::start("pseudopot_cell_vnl", "ensure_hess");
    if (this->nkb == 0 || this->hess_version_ == this->table_version_)
    {
        ModuleBase::timer::end("pseudopot_cell_vnl", "ensure_hess");
        return;
    }
    // Build the radial Hessian table on demand for RT-TDDFT using the same channels and momentum grid as tab.
    this->tab_d2.create(cell.ntype, this->tab.getBound2(), this->tab.getBound3());
    const double pref = ModuleBase::FOUR_PI / std::sqrt(cell.omega);
    for (int it = 0; it < cell.ntype; ++it)
    {
        const Atom &atom = cell.atoms[it];
        int nr = atom.ncpp.kkbeta;
        if (nr > 0 && nr % 2 == 0)
        {
            --nr;
        }
        std::vector<double> aux(nr, 0.0);
        for (int ib = 0; ib < atom.ncpp.nbeta; ++ib)
        {
            const int l = atom.ncpp.lll[ib];
            for (int iq = 0; iq < this->tab.getBound3(); ++iq)
            {
                const double q = iq * this->table_dq_;
                for (int ir = 0; ir < nr; ++ir)
                {
                    const double r = atom.ncpp.r[ir];
                    const double x = q * r;
                    // Bessel recurrences retain the analytic second derivative at q=0 without division by q.
                    double d2 = -ModuleBase::Sphbes::dsphbesj(1, x);
                    if (l > 0)
                    {
                        d2 = (l * ModuleBase::Sphbes::dsphbesj(l - 1, x) - (l + 1) * ModuleBase::Sphbes::dsphbesj(l + 1, x)) / (2 * l + 1);
                    }
                    aux[ir] = atom.ncpp.betar(ib, ir) * r * r * r * d2;
                }
                double integral = 0.0;
                ModuleBase::Integral::Simpson_Integral(nr, aux.data(), atom.ncpp.rab.data(), integral);
                this->tab_d2(it, ib, iq) = pref * integral;
            }
        }
    }
    this->hess_version_ = this->table_version_;
    ModuleBase::timer::end("pseudopot_cell_vnl", "ensure_hess");
}
