#include "elecstate_tools.h"

#include "source_base/timer.h"

namespace elecstate
{
double cal_td_energy(
    const ElecState &state,
    double &electronic_field,
    double &ionic_field,
    double &hamilton)
{
    ModuleBase::timer::start("ElecState", "cal_td_energy");
    state.pot->td_field_energy(*state.charge, electronic_field, ionic_field);
    const double internal = state.f_en.etot - state.f_en.demet;
    hamilton = internal + ionic_field;
    const double material = internal - electronic_field;
    ModuleBase::timer::end("ElecState", "cal_td_energy");
    return material;
}
} // namespace elecstate
