#ifndef PW_TD_ION_ACCEL_H
#define PW_TD_ION_ACCEL_H

#include "source_base/matrix.h"
#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/force_result.h"

namespace pw
{
//! Instantaneous RT-TDDFT ionic quantities in Hartree atomic units after a force evaluation.
struct IonAccel
{
    //! Ionic accelerations with constrained Cartesian components set to zero.
    ModuleBase::matrix accel;

    //! Full time derivative of the nuclear velocity derivative Pi, including the acceleration contribution.
    ModuleBase::matrix dpi;

    //! Reaction forces that maintain zero acceleration in constrained directions.
    ModuleBase::matrix reaction;

    //! LAPACK status of the most recent effective-inertia solve, with zero indicating success.
    int status = 0;

    //! Maximum relative force-balance residual over all movable Cartesian components.
    double residual = 0.0;
};

//! Evaluated position and nuclear-velocity forces, including the constrained acceleration.
struct IonForce
{
    //! Nuclear energy derivative evaluated with the same endpoint orbitals and projectors.
    ModuleBase::matrix pi;

    //! Complete force and its contributions in Ry/Bohr.
    PWForceResult forces;

    //! Acceleration, full Pi rate and constraint reactions in Hartree atomic units.
    IonAccel dynamics;
};

//! Solve the RT-TDDFT effective-inertia equation in movable directions and recover the full Pi rate and constraint reactions.
IonAccel cal_ion_acc(
    const UnitCell &cell,
    const ModuleBase::matrix &force,
    const ModuleBase::matrix &hess,
    const ModuleBase::matrix &rate);

//! Complete ionic dynamics in Hartree units and total force in Ry/Bohr from the evaluated position force.
void finish_ion_force(
    const UnitCell &cell,
    const ModuleBase::matrix &hess,
    const ModuleBase::matrix &rate,
    IonForce *force);

} // namespace pw

#endif
