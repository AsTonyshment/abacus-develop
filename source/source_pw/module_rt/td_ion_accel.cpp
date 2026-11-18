#include "source_pw/module_rt/td_ion_accel.h"

#include "source_base/constants.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace pw
{
IonAccel cal_ion_acc(
    const UnitCell &cell,
    const ModuleBase::matrix &force,
    const ModuleBase::matrix &hess,
    const ModuleBase::matrix &rate)
{
    ModuleBase::timer::start("pw", "cal_ion_acc");

    IonAccel result;
    result.accel.create(cell.nat, 3);
    result.dpi.create(cell.nat, 3);
    result.reaction.create(cell.nat, 3);
    constexpr int component[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

    int atom = 0;
    for (int it = 0; it < cell.ntype; ++it)
    {
        const double mass = cell.atoms[it].mass / ModuleBase::AU_to_MASS;
        for (int ia = 0; ia < cell.atoms[it].na; ++ia, ++atom)
        {
            // Restrict the RT-TDDFT inertia equation to movable directions, keeping constrained accelerations zero.
            int free[3] = {};
            int n = 0;
            for (int i = 0; i < 3; ++i)
            {
                if (cell.atoms[it].mbl[ia][i] != 0)
                {
                    free[n++] = i;
                }
            }

            // Assemble (M-Q)a=F+g, where Q is the velocity Hessian and g is the acceleration-independent Pi rate.
            double matrix[9] = {};
            double rhs[3] = {};
            for (int i = 0; i < n; ++i)
            {
                rhs[i] = force(atom, free[i]) + rate(atom, free[i]);
                for (int j = 0; j < n; ++j)
                {
                    matrix[j * n + i] = (i == j ? mass : 0.0) - hess(atom, component[free[i]][free[j]]);
                }
            }

            if (n > 0)
            {
                // The symmetric indefinite solver handles physical inertia corrections without assuming positive definiteness.
                constexpr char uplo = 'U';
                constexpr int nrhs = 1;
                constexpr int lwork = 3;
                int pivot[3] = {};
                double work[3] = {};
                dsysv_(&uplo, &n, &nrhs, matrix, &n, pivot, rhs, &n, work, &lwork, &result.status);
                if (result.status != 0)
                {
                    const std::string message = "Effective inertia solve failed for atom " + std::to_string(atom + 1) +
                                                ", LAPACK info=" + std::to_string(result.status);
                    ModuleBase::WARNING_QUIT("cal_ion_acc", message);
                }
                for (int i = 0; i < n; ++i)
                {
                    result.accel(atom, free[i]) = rhs[i];
                }
            }

            // Recover the full Pi rate g+Qa after solving the coupled Cartesian accelerations.
            for (int i = 0; i < 3; ++i)
            {
                double dpi = rate(atom, i);
                double scale = std::abs(force(atom, i) + rate(atom, i));
                for (int j = 0; j < 3; ++j)
                {
                    const double q = hess(atom, component[i][j]);
                    dpi += q * result.accel(atom, j);
                    scale += std::abs(((i == j ? mass : 0.0) - q) * result.accel(atom, j));
                }
                result.dpi(atom, i) = dpi;
                const double residual = mass * result.accel(atom, i) - force(atom, i) - dpi;
                if (!std::isfinite(residual) || !std::isfinite(scale))
                {
                    ModuleBase::WARNING_QUIT("cal_ion_acc", "Nonfinite effective-inertia solution or force balance.");
                }

                if (cell.atoms[it].mbl[ia][i] == 0)
                {
                    // The force balance in a fixed direction gives the reaction required to maintain its zero acceleration.
                    result.reaction(atom, i) = residual;
                }
                else
                {
                    const double denominator = std::max(scale, std::numeric_limits<double>::min());
                    result.residual = std::max(result.residual, std::abs(residual) / denominator);
                }
            }
        }
    }

    ModuleBase::timer::end("pw", "cal_ion_acc");
    return result;
}

void finish_ion_force(
    const UnitCell &cell,
    const ModuleBase::matrix &hess,
    const ModuleBase::matrix &rate,
    IonForce *force)
{
    // Solve the effective-inertia equation using the position force in Hartree/Bohr.
    PWForceResult &forces = force->forces;
    forces.pos_field = forces.total;
    ModuleBase::matrix force_ha(forces.pos_field);
    force_ha *= 0.5;
    force->dynamics = cal_ion_acc(cell, force_ha, hess, rate);

    // Store RT contributions in Ry/Bohr and keep constraint reactions separate from total.
    forces.vel_force.create(cell.nat, 3);
    forces.reaction.create(cell.nat, 3);
    for (int ia = 0; ia < cell.nat; ++ia)
    {
        for (int d = 0; d < 3; ++d)
        {
            forces.vel_force(ia, d) = 2.0 * force->dynamics.dpi(ia, d);
            forces.reaction(ia, d) = 2.0 * force->dynamics.reaction(ia, d);
            forces.total(ia, d) += forces.vel_force(ia, d);
        }
    }
}

} // namespace pw
