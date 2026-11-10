#include "source_io/module_ctrl/ctrl_output_td_pw.h"

#include "source_base/constants.h"
#include "source_base/formatter.h"
#include "source_io/module_current/td_current_pw.h"
#include "source_io/module_current/td_spin_pw_io.h"
#include "source_io/module_efield/td_efield_io.h"
#include "source_io/module_efield/td_vecpot_io.h"
#include "source_io/module_output/output_log.h"
#include "source_io/module_output/td_conserv_io.h"

#include <iomanip>
#include <sstream>
#include <vector>

namespace ModuleIO
{
void out_ion_force(
    const UnitCell &cell,
    const pw::IonAccel &ions,
    const ModuleBase::matrix &force,
    bool out_stat,
    int rank,
    std::ofstream &log)
{
    if (rank != 0)
    {
        return;
    }

    // Report physical force components whenever RT-TDDFT forces are requested.
    ModuleBase::matrix derivative(ions.dpi);
    ModuleBase::matrix reaction(ions.reaction);
    derivative *= 2.0;
    reaction *= 2.0;
    ModuleIO::print_force(log, cell, "NUCLEAR VELOCITY FORCE (eV/Angstrom)", derivative, false);
    ModuleIO::print_force(log, cell, "CONSTRAINT REACTION (eV/Angstrom)", reaction, false);

    if (out_stat)
    {
        const std::ios::fmtflags flags = log.flags();
        const std::streamsize precision = log.precision();
        log << " RT ionic inertia: LAPACK info=" << ions.status << ", relative residual=" << std::scientific << std::setprecision(12)
            << ions.residual << '\n';
        log.flags(flags);
        log.precision(precision);
    }

    ModuleIO::print_force(log, cell, "TOTAL-FORCE (eV/Angstrom)", force, false);
}

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
    std::ostream *out)
{
    if (rank != 0)
    {
        return;
    }

    if (out_stat)
    {
        write_td_energy(out_dir, step, dt, potential, total, total_field, kinetic, vel_corr, smearing, elec_field, ion_field, *log);
    }

    // Use the same accepted RT energy as cal_energy() so the standalone and MD potential outputs agree.
    const std::ios::fmtflags flags = out->flags();
    const std::streamsize precision = out->precision();
    *out << " Potential (Ry): " << std::defaultfloat << std::setprecision(15) << potential << std::endl;
    out->flags(flags);
    out->precision(precision);

    const std::vector<std::string> labels{
        "E_potential", "E_ionic_kinetic", "E_velocity_corr", "E_total", "E_elec_field", "E_ion_field", "E_total_field"};
    const std::vector<double> energy_ry{potential, kinetic, vel_corr, total, elec_field, ion_field, total_field};
    std::vector<double> energy_ev(energy_ry.size());
    for (std::size_t i = 0; i < energy_ry.size(); ++i)
    {
        energy_ev[i] = energy_ry[i] * ModuleBase::Ry_to_eV;
    }
    FmtTable table(
        {"RT energy", "Rydberg", "eV"},
        labels.size(),
        {"%-16s", "%20.10f", "%20.10f"},
        1,
        {FmtTable::Align::LEFT, FmtTable::Align::CENTER});
    table << labels << energy_ry << energy_ev;
    const std::ios::fmtflags log_flags = log->flags();
    const std::streamsize log_prec = log->precision();
    *log << " RT endpoint energy: step " << step + 1 << ", time " << std::fixed << std::setprecision(6) << step * dt * ModuleBase::AU_to_FS
         << " fs\n"
         << table.str() << '\n';
    log->flags(log_flags);
    log->precision(log_prec);
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
