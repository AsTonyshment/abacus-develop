#include "source_pw/module_rt/td_pw.h"

#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"

#include <algorithm>

namespace pw
{
void check_td_input(const Input_para &input, int xc_type, bool cal_exx)
{
    if (input.out_current != 0 && input.out_current != 1)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT requires out_current=0 or 1.");
    }
    if (input.scf_nmax < 1)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT requires scf_nmax >= 1.");
    }
    if (input.mdp.md_type == "npt" || input.mdp.md_type == "msst")
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT requires a fixed cell; NPT and MSST are not supported.");
    }
    if (input.dft_plus_u != 0)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT does not support DFT+U; set dft_plus_u=0.");
    }
    if (input.sc_mag_switch)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT does not support spin constraints; set sc_mag_switch=false.");
    }
    if (xc_type == 3 || xc_type == 5)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT does not support meta-GGA.");
    }
    if (cal_exx)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT does not support exact exchange (EXX); use a non-hybrid functional.");
    }
    if ((input.nspin != 1 && input.nspin != 2 && input.nspin != 4) || input.bndpar != 1 || (input.td_stype != 0 && input.td_stype != 1))
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT requires nspin=1, 2 or 4, bndpar=1 and td_stype=0 or 1.");
    }
    if (input.nspin == 4 && !input.noncolin)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "nspin=4 PW RT-TDDFT requires noncolin=true.");
    }
    if (input.out_elf[0] > 0)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT does not support ELF.");
    }
    if (input.mdp.md_restart || input.restart_load)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "PW RT-TDDFT restart is not supported yet.");
    }
}

void check_td_ions(const Input_para &input)
{
    // Restrict moving paths to energy terms whose geometry-dependent data are refreshed by the PW RT workflow.
    if ((input.vdw_method != "none" || input.efield_flag || input.gate_flag || input.imp_sol == 1 || input.dfthalf_type != 0 ||
            input.assume_isolated == "makov-payne"))
    {
        ModuleBase::WARNING_QUIT(
            "ESolver_KS_PW_TDDFT", "Prescribed PW ionic samples support the RT field and LDA/GGA pseudopotential energy terms.");
    }
}

double td_momentum_bound(const ModulePW::PW_Basis_K &basis, const double tpiba)
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

std::vector<ModuleBase::Vector3<double>> td_ion_vel(const UnitCell &cell, bool motion)
{
    // In RT-TDDFT, explicit zero velocities can yield a nonzero dE/dv, while an empty array disables the velocity phase.
    std::vector<ModuleBase::Vector3<double>> vel;
    if (motion)
    {
        vel.reserve(cell.nat);
        for (int it = 0; it < cell.ntype; ++it)
        {
            for (int ia = 0; ia < cell.atoms[it].na; ++ia)
            {
                vel.push_back(cell.atoms[it].vel[ia]);
            }
        }
    }
    return vel;
}

} // namespace pw
