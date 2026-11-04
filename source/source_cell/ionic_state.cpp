#include "source_cell/ionic_state.h"

namespace unitcell
{
IonicState capture_ions(const UnitCell &cell, double t_pos, double t_vel)
{
    IonicState state;
    state.t_pos = t_pos;
    state.t_vel = t_vel;
    state.lat0 = cell.lat0;
    state.lattice = cell.latvec;
    // Preserve type-major atom identity so snapshots can be matched across Cell, MD and PW operators.
    for (int type = 0; type < cell.ntype; ++type)
    {
        const Atom &atoms = cell.atoms[type];
        for (int index = 0; index < atoms.na; ++index)
        {
            IonicAtom atom;
            atom.id = static_cast<int>(state.atoms.size());
            atom.type = type;
            atom.index = index;
            atom.movable = atoms.mbl[index];
            atom.frac = atoms.taud[index];
            // Convert cell-scaled coordinates to Bohr; continuous trajectories are maintained by the MD sampling layer.
            atom.position = atoms.tau[index] * cell.lat0;
            atom.velocity = atoms.vel[index];
            if (!atoms.dis.empty())
            {
                atom.disp = atoms.dis[index] * cell.latvec * cell.lat0;
            }
            state.atoms.push_back(atom);
        }
    }
    return state;
}

} // namespace unitcell
