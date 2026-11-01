#ifndef TD_CONSERV_IO_H
#define TD_CONSERV_IO_H

#include "source_pw/module_rt/td_conserv_pw.h"

#include <fstream>
#include <string>

namespace ModuleIO
{
/** @brief Write accepted endpoint energies in Rydberg. */
void write_td_energy(
    const std::string &directory,
    int step,
    double dt_ha,
    double material,
    double hamilton,
    double electronic_field,
    double ionic_field,
    std::ostream &log);

/** @brief Print endpoint conservation relative to the caller's initial counts. */
void print_conserv(
    const pw::TDConserv &values,
    double npsi0,
    double nrho0,
    int step,
    bool full_gram,
    const std::string &orth,
    std::ofstream &log);
} // namespace ModuleIO
#endif
