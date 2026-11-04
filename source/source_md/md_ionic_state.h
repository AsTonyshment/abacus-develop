#ifndef MD_IONIC_STATE_H
#define MD_IONIC_STATE_H

#include "source_cell/ionic_state.h"
#include "source_cell/mdcell.h"

namespace MD_func
{
/** @brief Collect actual MD data in type-major order without changing either cell representation. */
unitcell::IonicState sample_ions(const MDCell &cell, double t_pos, double t_vel);
/** @brief Continue samples from the same MD system using their actual step displacements. */
void unwrap_ions(unitcell::IonicState *state, const unitcell::IonicState &previous);
} // namespace MD_func

#endif
