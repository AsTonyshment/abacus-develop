#ifndef IONIC_STATE_H
#define IONIC_STATE_H

#include "source_base/matrix3.h"
#include "source_base/vector3.h"

#include "source_cell/unitcell.h"

#include <vector>

namespace unitcell
{
/** @brief Stable type-major atom identity and physical Cartesian data. */
struct IonicAtom
{
    int id = 0;
    int type = 0;
    int index = 0;
    ModuleBase::Vector3<int> movable;
    ModuleBase::Vector3<double> frac;
    ModuleBase::Vector3<double> position; ///< Continuous Cartesian position in Bohr.
    ModuleBase::Vector3<double> velocity; ///< Bohr per Hartree atomic time.
    ModuleBase::Vector3<double> disp;     ///< Unwrapped MD displacement in Bohr.
};

/** @brief An ionic sample; MD drift positions and kick velocities have separate times. */
struct IonicState
{
    double t_pos = 0.0;
    double t_vel = 0.0;
    double lat0 = 0.0;
    ModuleBase::Matrix3 lattice;
    std::vector<IonicAtom> atoms;
};

// Record position and velocity times in Hartree atomic units to preserve the MD integrator's sampling times.
IonicState capture_ions(const UnitCell &cell, double t_pos, double t_vel);
} // namespace unitcell

#endif
