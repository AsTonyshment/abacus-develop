#ifndef TD_ION_PW_H
#define TD_ION_PW_H

#include "source_base/matrix.h"
#include "source_base/parallel_grid.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/ionic_state.h"
#include "source_cell/stru_fac.h"
#include "source_estate/module_charge/charge.h"
#include "source_estate/module_pot/potential_new.h"

namespace pw
{
/** @brief Explicit samples at the nodes required by the propagation method. */
class TDIonPath
{
  public:
    explicit TDIonPath(const std::vector<unitcell::IonicState> &samples);
    /** @brief A constant geometry over a time interval, including arbitrary retry nodes. */
    TDIonPath(const unitcell::IonicState &stationary, double left, double right);
    unitcell::IonicState at(double time) const;
    /// @brief Validate samples and their connection to the accepted propagation origin.
    void check(const UnitCell &cell, const unitcell::IonicState &accepted, double time) const;
    bool is_fixed() const { return stationary_; }

  private:
    std::vector<unitcell::IonicState> samples_;
    const bool stationary_;
};

/** @brief Refresh ionic operators on the existing PW grids. */
class TDIonPW
{
  public:
    TDIonPW(
        UnitCell *cell,
        Structure_Factor *sf,
        const Parallel_Grid &grid,
        const ModulePW::PW_Basis &rho_basis,
        Charge *charge,
        const bool *numeric,
        bool has_float);
    void bind(const unitcell::IonicState &state);
    void update_pot(const unitcell::IonicState &state, elecstate::Potential *pot);
    const unitcell::IonicState &state() const;

  private:
    UnitCell &cell_;
    Structure_Factor &sf_;
    const Parallel_Grid &grid_;
    const ModulePW::PW_Basis &rho_basis_;
    Charge &charge_;
    const bool has_float_;
    unitcell::IonicState state_;
    ModuleBase::matrix core_rad_;
    std::vector<ModuleBase::Vector3<double>> end_frac_;
};
} // namespace pw

#endif
