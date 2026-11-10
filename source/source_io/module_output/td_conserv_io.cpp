#include "source_io/module_output/td_conserv_io.h"

#include "source_base/constants.h"
#include "source_base/global_function.h"
#include "source_base/tool_quit.h"

#include <iomanip>
#include <limits.h>
#include <unistd.h>

namespace ModuleIO
{
void write_td_energy(
    const std::string &directory,
    int step,
    double dt_ha,
    double potential,
    double total,
    double total_field,
    double kinetic,
    double vel_corr,
    double smearing,
    double elec_field,
    double ion_field,
    std::ostream &log)
{
    const std::string filename = directory + "td_energy.txt";
    const std::ios::openmode mode = step == 0 ? std::ios::out | std::ios::trunc : std::ios::app;
    std::ofstream output(filename, mode);

    if (!output)
    {
        ModuleBase::WARNING_QUIT("write_td_energy", "Cannot open " + filename);
    }

    if (step == 0)
    {
        std::string full_path = filename;
        if (filename[0] != '/')
        {
            char cwd[PATH_MAX];
            if (getcwd(cwd, sizeof(cwd)) == nullptr)
            {
                ModuleBase::WARNING_QUIT("write_td_energy", "Cannot resolve the output directory.");
            }
            full_path = std::string(cwd) + "/" + filename;
        }
        log << " Write file: " << full_path << '\n';
        output << "# energy_unit=Rydberg\n";
        output << std::left << std::setw(6) << "#step" << "  " << std::right << std::setw(10) << "time_fs";
        for (const char *label :
            {"potential", "total", "total_field", "ionic_kinetic", "velocity_corr", "smearing", "electronic_field", "ionic_field"})
        {
            output << "  " << std::setw(19) << label;
        }
        output << '\n';
    }

    output << std::right << std::setw(6) << step + 1 << "  " << std::fixed << std::setprecision(6) << std::setw(10)
           << step * dt_ha * ModuleBase::AU_to_FS;
    output << std::scientific << std::setprecision(12);
    for (double energy : {potential, total, total_field, kinetic, vel_corr, smearing, elec_field, ion_field})
    {
        output << "  " << std::setw(19) << energy;
    }
    output << '\n';
    output.close();

    if (!output)
    {
        ModuleBase::WARNING_QUIT("write_td_energy", "Failed to write " + filename);
    }
}

void print_conserv(
    const pw::TDConserv &values,
    double npsi0,
    double nrho0,
    int step,
    bool full_gram,
    const std::string &orth,
    std::ofstream &log)
{
    if (!log.good())
    {
        return;
    }

    const char *stage = step == 0 ? "initial state" : "propagation";
    const int evolve_step = step + 1;
    const double delta_wave = values.npsi - npsi0;
    const double delta_rho = values.nrho - nrho0;
    const std::ios::fmtflags saved_flags = log.flags();
    const std::streamsize saved_prec = log.precision();
    log << " PW RT-TDDFT conservation: evolution step " << evolve_step << ", " << stage << ", " << orth << '\n';
    log << std::right << std::fixed << std::setprecision(10);
    ModuleBase::GlobalFunc::OUT(log, "Npsi", values.npsi);
    ModuleBase::GlobalFunc::OUT(log, "Nrho", values.nrho);

    log << std::scientific << std::setprecision(6);
    ModuleBase::GlobalFunc::OUT(log, "dNpsi", delta_wave);
    ModuleBase::GlobalFunc::OUT(log, "dNrho", delta_rho);
    if (full_gram)
    {
        if (step == 0)
        {
            ModuleBase::GlobalFunc::OUT(log, "initial_orth_error", values.orth_after);
        }
        else
        {
            ModuleBase::GlobalFunc::OUT(log, "orth_before_step_max", values.orth_before);
            ModuleBase::GlobalFunc::OUT(log, "orth_after_step_max", values.orth_after);
        }
    }
    log.flags(saved_flags);
    log.precision(saved_prec);
}
} // namespace ModuleIO
