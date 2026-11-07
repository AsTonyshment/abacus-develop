#ifndef TD_PW_H
#define TD_PW_H

#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/unitcell.h"
#include "source_io/module_parameter/input_parameter.h"

namespace pw
{

/** @brief Check PW propagation restrictions after resolving the XC functional and EXX configuration. */
void check_td_input(const Input_para &input, int xc_type, bool cal_exx);

/** @brief Validate terms requiring geometry refresh for a moving ionic path. */
void check_td_ions(const Input_para &input);

/** @brief Return the global unshifted momentum bound after updating the distributed basis. */
double td_momentum_bound(const ModulePW::PW_Basis_K &basis, const double tpiba);

/** @brief Extract RT-TDDFT nuclear velocities in type-major order and Hartree units; empty selects position-dependent pseudopotentials. */
std::vector<ModuleBase::Vector3<double>> td_ion_vel(const UnitCell &cell, bool motion);

} // namespace pw
#endif
