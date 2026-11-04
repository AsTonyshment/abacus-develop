#ifndef CTRL_OUTPUT_TD_PW_H
#define CTRL_OUTPUT_TD_PW_H

#include "source_estate/module_pot/td_field_manager.h"
#include "source_pw/module_rt/td_obs_pw.h"

#include <ostream>
#include <string>

namespace ModuleIO
{
/** @brief Prepare enabled field files on the designated writer rank. */
void prep_td_out(
    const elecstate::TDFieldManager &field,
    const std::string &out_dir,
    bool out_a,
    bool out_e,
    bool read_a,
    int rank);

/** @brief Write the current field samples without advancing the field. */
void out_td_field(
    const elecstate::TDFieldManager &field,
    const std::string &out_dir,
    int step,
    bool out_a,
    bool out_e,
    int rank);

/** @brief Write optional energy statistics and print the endpoint total energy. */
void out_td_energy(
    const std::string &out_dir,
    int step,
    double dt,
    double material,
    double hamilt,
    double elec_field,
    double ion_field,
    double etot,
    bool out_stat,
    int rank,
    std::ostream *log,
    std::ostream *out);

std::string print_td_trial(
    double left,
    double right,
    int depth,
    std::ostream *log,
    std::ostream *out);
void print_td_result(
    int iter,
    double residual,
    bool converged,
    bool accepted,
    std::ostream *log);
void print_td_retry(const std::string &interval, std::ostream *out);

/** @brief Write precomputed endpoint observables on the designated output rank. */
void ctrl_output_td_pw(
    const pw::TDVelResultPW &currents,
    const pw::TDSpinResultPW &moments,
    const pw::CurrentRequest &request,
    const std::string &mag_partition,
    const std::string &orbital_dir,
    double onsite_radius,
    bool calc_torque,
    const std::vector<std::string> &orbital_fn,
    int step,
    double dt_ha,
    int nspin,
    int nkstot,
    int rank,
    const std::string &directory,
    std::ostream &log);
} // namespace ModuleIO
#endif
