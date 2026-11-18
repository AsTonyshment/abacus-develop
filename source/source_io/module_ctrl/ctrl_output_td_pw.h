#ifndef CTRL_OUTPUT_TD_PW_H
#define CTRL_OUTPUT_TD_PW_H

#include "source_base/vector3.h"
#include "source_estate/module_pot/td_field_manager.h"
#include "source_pw/module_rt/td_obs_pw.h"

#include <fstream>
#include <ostream>
#include <string>

namespace ModuleIO
{
//! Print the accepted endpoint's effective-inertia solve status and relative residual.
void out_ion_stat(std::ofstream *log, const int &status, const double &residual);

//! Prepare enabled field files on the designated writer rank.
void prep_td_out(
    const elecstate::TDFieldManager &field,
    const std::string &out_dir,
    bool out_a,
    bool out_e,
    bool read_a,
    int rank);

//! Write the current field samples without advancing the field.
void out_td_field(
    const elecstate::TDFieldManager &field,
    const std::string &out_dir,
    int step,
    bool out_a,
    bool out_e,
    int rank);

//! Print accepted RT endpoint energies and the MD potential, and write optional statistics.
void out_td_energy(
    const std::string &out_dir,
    int step,
    double dt,
    double total,
    double total_field,
    double elec_field,
    double ion_field,
    double kinetic,
    double vel_corr,
    double potential,
    double smearing,
    bool out_stat,
    int rank,
    std::ostream *log,
    std::ostream *out);

//! Log the trial interval in fs and its subdivision depth, and print substeps.
void print_td_trial(
    const double &left,
    const double &right,
    const int &depth,
    std::ostream *log,
    std::ostream *out);

//! Log the trial iteration count, residual, convergence and acceptance status.
void print_td_result(
    const int &iter,
    const double &res,
    const bool &conv,
    const bool &accepted,
    std::ostream *log);

//! Report that an unconverged trial interval will be split into two half steps.
void print_td_retry(const double &left, const double &right, std::ostream *out);

//! Log the original step and print the iteration when outer ionic self-consistency is enabled.
void print_ion_trial(
    const int &step,
    const int &iter,
    const int &limit,
    std::ostream *log,
    std::ostream *out);

//! Log coupled diagnostics and print convergence details when outer ionic self-consistency is enabled.
void print_td_coupled(
    const int &step,
    const int &iter,
    const int &limit,
    const int &elec_iters,
    const double &elec_res,
    const double &vel_res,
    const double &vel_thr,
    const double &inertia_res,
    const int &inertia_status,
    const bool &conv,
    const bool &accepted,
    std::ostream *log,
    std::ostream *out);

//! Print the prepared Cartesian vector potential in Hartree atomic units.
void print_td_vecpot(const ModuleBase::Vector3<double> &vecpot, std::ofstream *log);

//! Write precomputed PW endpoint currents and spin observables on the designated output rank.
void out_curr_spin_pw(
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
