#include "source_io/module_ctrl/ctrl_output_td_pw.h"

#include "source_base/constants.h"
#include "source_io/module_current/td_current_pw.h"
#include "source_io/module_current/td_spin_pw_io.h"
#include "source_io/module_efield/td_efield_io.h"
#include "source_io/module_efield/td_vecpot_io.h"
#include "source_io/module_output/td_conserv_io.h"

#include <iomanip>
#include <sstream>

namespace ModuleIO
{
void prep_td_out(
    const elecstate::TDFieldManager &field,
    const std::string &out_dir,
    bool out_a,
    bool out_e,
    bool read_a,
    int rank)
{
    if (rank != 0)
    {
        return;
    }
    if (out_e)
    {
        prep_td_field_out(out_dir, field, false);
    }
    if (out_a)
    {
        if (read_a)
        {
            // Check for input/output path overlap before preparing the vector-potential output file.
            check_td_vecpot_paths("", out_dir);
        }
        prep_td_vecpot_out(out_dir, false);
    }
}

void out_td_field(
    const elecstate::TDFieldManager &field,
    const std::string &out_dir,
    int step,
    bool out_a,
    bool out_e,
    int rank)
{
    if (rank != 0)
    {
        return;
    }
    if (out_a)
    {
        const ModuleBase::Vector3<double> &vecpot = field.A_right_ha();
        write_td_vecpot(out_dir, step, vecpot);
    }
    if (out_e)
    {
        write_td_field(field, out_dir);
    }
}

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
    std::ostream *out)
{
    if (out_stat && rank == 0)
    {
        write_td_energy(out_dir, step, dt, material, hamilt, elec_field, ion_field, *log);
    }
    const std::streamsize precision = out->precision();
    *out << " Potential (Ry): " << std::setprecision(15) << etot << std::endl;
    out->precision(precision);
}

std::string print_td_trial(
    double left,
    double right,
    int depth,
    std::ostream *log,
    std::ostream *out)
{
    const std::streamsize precision = log->precision();
    *log << " TD trial: [" << std::setprecision(12) << left << ", " << right << "] fs, depth=" << depth << '\n';
    log->precision(precision);

    std::ostringstream interval;
    interval << std::setprecision(12) << "[" << left << ", " << right << "] fs";
    const std::string text = interval.str();
    if (depth > 0)
    {
        *out << " TD substep: " << text << std::endl;
    }
    return text;
}

void print_td_result(
    int iter,
    double residual,
    bool converged,
    bool accepted,
    std::ostream *log)
{
    // Report convergence and acceptance separately because an iteration-limited endpoint can still advance propagation.
    const std::ios::fmtflags flags = log->flags();
    const std::streamsize precision = log->precision();
    *log << " TD trial result: iterations=" << iter << ", residual=" << std::scientific << std::setprecision(12) << residual
         << ", converged=" << converged << ", accepted=" << accepted << '\n';
    // Restore stream formatting so TD diagnostics leave subsequent shared-log output unchanged.
    log->flags(flags);
    log->precision(precision);
}

void print_td_retry(const std::string &interval, std::ostream *out)
{
    *out << " TD retry: SCF not converged; splitting " << interval << " into two half steps." << std::endl;
}

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
    std::ostream &log)
{
    if (rank != 0)
    {
        return;
    }
    if (request.charge_total)
    {
        const int spin_mult = nspin == 2 ? 2 : 1;
        const int nk = nkstot / spin_mult;
        write_current_pw(currents, step, nk, request.charge_per_k, directory);
    }
    if (nspin == 4)
    {
        const double time = step * dt_ha * ModuleBase::AU_to_FS;
        write_td_spin_pw(
            directory,
            step,
            time,
            orbital_fn,
            mag_partition,
            orbital_dir,
            onsite_radius,
            calc_torque,
            request.spin_current,
            moments,
            currents,
            log);
    }
}
} // namespace ModuleIO
