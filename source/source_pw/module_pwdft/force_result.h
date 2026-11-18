#ifndef PW_FORCE_RESULT_H
#define PW_FORCE_RESULT_H

#include "source_base/matrix.h"

/** @brief PW force contributions in Ry/Bohr, independent of their presentation. */
struct PWForceResult
{
    //! Complete force after all contributions of the active physical model are evaluated.
    ModuleBase::matrix total;
    //! RT position and direct field force; empty for ground-state calculations.
    ModuleBase::matrix pos_field;
    //! RT nuclear-velocity contribution dPi/dt; empty for ground-state calculations.
    ModuleBase::matrix vel_force;
    //! RT constraint reaction, stored separately from total; empty for ground-state calculations.
    ModuleBase::matrix reaction;
    ModuleBase::matrix local;
    ModuleBase::matrix ion;
    ModuleBase::matrix nonlocal;
    ModuleBase::matrix core;
    ModuleBase::matrix scc;
    ModuleBase::matrix onsite;
    ModuleBase::matrix vdw;
    ModuleBase::matrix field;
    ModuleBase::matrix static_field;
    ModuleBase::matrix gate;
    ModuleBase::matrix solvent;
    ModuleBase::matrix pcc;
    bool shifted = false;
};

#endif
