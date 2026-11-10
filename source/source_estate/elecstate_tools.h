#ifndef ELECSTATE_TOOLS_H
#define ELECSTATE_TOOLS_H
#include "elecstate.h"
#include "source_base/matrix.h"

namespace elecstate
{
//! Sum eigenvalues weighted by occupations and k-point weights into the band energy across all partitions.
void calEBand(const ModuleBase::matrix &ekb, const ModuleBase::matrix &wg, fenergy &f_en);

//! Combine RT-TDDFT endpoint KS and scalar-field energies with ionic terms in Ry (retaining the smearing contribution).
double cal_td_energy(
    const double &etot,
    const UnitCell &cell,
    const ModuleBase::matrix &pi,
    const double &elec_field,
    const double &ion_field,
    double *total,
    double *total_field,
    double *kinetic,
    double *vel_corr);

//! Update occupation weights, Fermi levels and the smearing energy contribution unless fixed weights are supplied.
void calculate_weights(
    const ModuleBase::matrix &ekb,
    ModuleBase::matrix &wg,
    const K_Vectors *klist,
    Efermi &eferm,
    fenergy &f_en,
    std::vector<double> &nelec_spin,
    const int global_nbands,
    const bool skip_weights);

//! Validate user-supplied occupations, assign them to local k-points and bands, and disable weight recalculation.
void fixed_weights(
    const std::vector<double> &ocp_kb,
    const int &nbands,
    const double &nelec,
    const K_Vectors *klist,
    ModuleBase::matrix &wg,
    bool &skip_weights);
} // namespace elecstate

#endif
