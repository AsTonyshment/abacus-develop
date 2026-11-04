#include "source_base/matrix.h"
#include "source_estate/elecstate.h"
#include "source_estate/module_charge/chg_tools.h"
#include "source_estate/write_init.h"

namespace elecstate
{

void init_scf(
    const UnitCell &ucell,
    const Parallel_Grid &pgrid,
    const ModuleBase::ComplexMatrix &strucfac,
    const bool *numeric,
    const int istep,
    const std::string &out_dir,
    const Input_para &inp,
    ElecState *pelec)
{
    // Prepare radial core data for this SCF geometry before assembling the core density.
    ModuleBase::matrix core_rad;
    module_charge::cal_core_rad(ucell, numeric, *pelec->charge->rhopw, &core_rad);
    module_charge::set_rho_core(ucell, strucfac, core_rad, pelec->charge->rho_core, pelec->charge->rhog_core, *pelec->charge->rhopw);

    //! renormalize the charge density
    if (PARAM.inp.init_chg != "dm_no_renormalize")
    {
        pelec->charge->renormalize_rho(inp.nelec, ucell.omega);
    }

    //! initialize the potential
    pelec->pot->init_pot(pelec->charge);

    //! output the initial potential
    ModuleIO::write_pot_init(ucell, pgrid, pelec, istep, out_dir, inp, PARAM.globalv.two_fermi, GlobalV::ofs_running);
}

} // namespace elecstate
