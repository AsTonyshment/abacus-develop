#include "source_pw/module_pwdft/td_pw.h"

#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"

#include <algorithm>

namespace pw
{

void check_td_input(const Input_para& input, const bool needs_ked)
{
    if ((input.nspin != 1 && input.nspin != 2) || input.bndpar != 1 || (input.td_stype != 0 && input.td_stype != 1))
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT requires nspin=1 or 2, bndpar=1 and td_stype=0 or 1.");
    }
    if (input.estep_per_md != 1)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT currently requires estep_per_md to be 1.");
    }
    if (input.mdp.md_restart || input.restart_load)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT restart is not supported yet.");
    }
    if (input.td_stype == 1 && needs_ked)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT",
                                 "PW RT-TDDFT with a kinetic-energy-density functional requires td_stype=0; "
                                 "the velocity-gauge coupling of the functional is not implemented.");
    }
}

double td_momentum_bound(const ModulePW::PW_Basis_K& basis, const double tpiba)
{
    ModuleBase::timer::start("pw", "td_momentum_bound");
    double bound = 0.0;
    for (int ik = 0; ik < basis.nks; ++ik)
    {
        for (int ig = 0; ig < basis.npwk[ik]; ++ig)
        {
            bound = std::max(bound, basis.getgpluskcar(ik, ig).norm() * tpiba);
        }
    }
    Parallel_Reduce::reduce_max(bound);
    ModuleBase::timer::end("pw", "td_momentum_bound");
    return bound;
}

void ensure_td_vnl(const UnitCell& cell, const double unshifted_bound,
                   const ModuleBase::Vector3<double>& endpoint,
                   const ModuleBase::Vector3<double>& midpoint,
                   const double dq, pseudopot_cell_vnl* projectors)
{
    ModuleBase::timer::start("pw", "ensure_td_vnl");
    const double field_bound = std::max(endpoint.norm(), midpoint.norm()) / 2.0;
    projectors->ensure_vnl_range(cell, unshifted_bound + field_bound, dq);
    ModuleBase::timer::end("pw", "ensure_td_vnl");
}

} // namespace pw
