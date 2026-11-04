#include "source_md/md_ionic_state.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace MD_func
{
unitcell::IonicState sample_ions(const MDCell &cell, double t_pos, double t_vel)
{
    unitcell::IonicState state;
    state.t_pos = t_pos;
    state.t_vel = t_vel;
    state.lat0 = cell.lat0();
    state.lattice = cell.latvec();
    std::vector<double> cart;
    std::vector<double> frac;
    std::vector<double> vel;
    std::vector<double> disp;
    std::vector<int> shift;
    std::vector<int> mbl;
    // Gather owned MD atoms directly so sampling leaves the backing UnitCell and its update flags unchanged.
    cell.collect_md(&cart, &frac, &vel, &disp, &shift, &mbl);
    const std::vector<std::int64_t> &counts = cell.type_atom_counts();
    const std::size_t nat = cart.size() / 3;
    state.atoms.resize(nat);
    int id = 0;
    for (std::size_t type = 0; type < counts.size(); ++type)
    {
        for (int index = 0; index < counts[type]; ++index, ++id)
        {
            unitcell::IonicAtom &atom = state.atoms[id];
            atom.id = id;
            atom.type = type;
            atom.index = index;
            for (int axis = 0; axis < 3; ++axis)
            {
                const int pos = 3 * id + axis;
                atom.frac[axis] = frac[pos];
                atom.position[axis] = cart[pos] * state.lat0;
                atom.velocity[axis] = vel[pos];
                atom.disp[axis] = disp[pos];
                atom.movable[axis] = mbl[pos];
            }
            // Convert the unwrapped fractional MD step to Bohr for continuous-position accumulation.
            atom.disp = atom.disp * state.lattice * state.lat0;
        }
    }
    return state;
}
void unwrap_ions(unitcell::IonicState *state, const unitcell::IonicState &previous)
{
    // Treat roundoff-equivalent position times as repeated samples to avoid adding the same MD displacement twice.
    const double abs_time = std::abs(state->t_pos);
    const double abs_prev = std::abs(previous.t_pos);
    const double magnitude = std::max(abs_time, abs_prev);
    const double scale = std::max(1.0, magnitude);
    const double tolerance = 32.0 * std::numeric_limits<double>::epsilon() * scale;
    const bool same_time = std::abs(state->t_pos - previous.t_pos) <= tolerance;
    for (std::size_t i = 0; i < state->atoms.size(); ++i)
    {
        unitcell::IonicAtom &atom = state->atoms[i];
        const unitcell::IonicAtom &old = previous.atoms[i];
        if (same_time)
        {
            // Both displacements share the same MD step origin, so apply only the change since the previous sample.
            atom.position = old.position + (atom.disp - old.disp);
        }
        else
        {
            // Accumulate the actual MD displacement to retain continuous motion across periodic boundaries.
            atom.position = old.position + atom.disp;
        }
    }
}
} // namespace MD_func
