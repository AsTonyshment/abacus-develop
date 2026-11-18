#include "source_pw/module_rt/td_ion_pw.h"

#include "source_base/tool_quit.h"
#include "source_estate/module_charge/chg_tools.h"
#include "source_estate/module_pot/td_field_manager.h"
#include "source_hamilt/module_ewald/h_ewald_pw.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pw
{
namespace
{
void wrap_ion(unitcell::IonicAtom *atom, const unitcell::IonicState &state)
{
    // Wrap fractional coordinates while retaining the continuous Cartesian trajectory.
    atom->frac = atom->position / state.lat0 * state.lattice.Inverse();
    for (int d = 0; d < 3; ++d)
    {
        if (!std::isfinite(atom->frac[d]) || !std::isfinite(atom->velocity[d]))
        {
            ModuleBase::WARNING_QUIT("wrap_ion", "Nonfinite coupled ionic sample.");
        }
        atom->frac[d] -= std::floor(atom->frac[d]);
    }
}
} // namespace

unitcell::IonicState predict_ions(const unitcell::IonicState &left, const ModuleBase::matrix &accel, const double &dt)
{
    unitcell::IonicState right = left;
    right.t_pos += dt;
    right.t_vel = right.t_pos;

    // Drift with the half-step velocity and predict the endpoint velocity from the left acceleration.
    for (unitcell::IonicAtom &atom : right.atoms)
    {
        const unitcell::IonicAtom &origin = left.atoms[atom.id];
        for (int d = 0; d < 3; ++d)
        {
            const double half = origin.velocity[d] + 0.5 * dt * accel(atom.id, d);
            atom.disp[d] = atom.movable[d] ? dt * half : 0.0;
            atom.position[d] = origin.position[d] + atom.disp[d];
            atom.velocity[d] = atom.movable[d] ? origin.velocity[d] + dt * accel(atom.id, d) : 0.0;
        }
        wrap_ion(&atom, right);
    }

    return right;
}

double correct_ions(const unitcell::IonicState &left,
                    const ModuleBase::matrix &left_accel,
                    const ModuleBase::matrix &right_accel,
                    const double &dt,
                    unitcell::IonicState *right)
{
    double residual = 0.0;

    for (unitcell::IonicAtom &atom : right->atoms)
    {
        for (int d = 0; d < 3; ++d)
        {
            // Correct the endpoint velocity using the accelerations at both ends of the interval.
            double velocity = 0.0;
            if (atom.movable[d])
            {
                velocity = left.atoms[atom.id].velocity[d] + 0.5 * dt * (left_accel(atom.id, d) + right_accel(atom.id, d));
            }

            const double velocity_error = std::abs(velocity - atom.velocity[d]);
            if (!std::isfinite(velocity_error))
            {
                ModuleBase::WARNING_QUIT("correct_ions", "Nonfinite ionic velocity correction.");
            }

            residual = std::max(residual, velocity_error);
            atom.velocity[d] = velocity;
        }
    }

    return residual;
}

void store_ion_disp(const unitcell::IonicState &left, const unitcell::IonicState &right, UnitCell *cell)
{
    for (const unitcell::IonicAtom &atom : right.atoms)
    {
        // Validate the fixed drift once when committing the original ionic step.
        for (int d = 0; d < 3; ++d)
        {
            const double origin = left.atoms[atom.id].position[d];
            const double position = atom.position[d];
            const double disp = atom.disp[d];
            const double scale = std::max({1.0, std::abs(origin), std::abs(position), std::abs(disp)});
            const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
            const double error = position - origin - disp;
            if (!std::isfinite(error) || std::abs(error) > tolerance)
            {
                ModuleBase::WARNING_QUIT("store_ion_disp", "Accepted position is inconsistent with the predicted displacement.");
            }
        }
        Atom &target = cell->atoms[atom.type];
        const int ia = atom.index;

        // Convert the continuous Cartesian displacement to the fractional increment stored by MD.
        target.dis[ia] = atom.disp / cell->lat0 * cell->GT;

        // Count periodic crossings from the previous wrapped position and the full displacement.
        const ModuleBase::Vector3<double> unwrapped = left.atoms[atom.id].frac + target.dis[ia];
        for (int d = 0; d < 3; ++d)
        {
            const double crossing = -std::floor(unwrapped[d]);
            if (!std::isfinite(crossing) || crossing < std::numeric_limits<int>::min()
                || crossing > std::numeric_limits<int>::max())
            {
                ModuleBase::WARNING_QUIT("store_ion_disp", "Periodic crossing count exceeds the MD integer range.");
            }
            target.boundary_shift[ia][d] = static_cast<int>(crossing);
        }
    }
}

namespace
{
bool same_time(double a, double b)
{
    // Allow roundoff from equivalent time expressions when locating explicit propagation samples.
    const double abs_a = std::abs(a);
    const double abs_b = std::abs(b);
    const double magnitude = std::max(abs_a, abs_b);
    const double scale = std::max(1.0, magnitude);
    const double difference = a - b;
    return std::abs(difference) <= 32.0 * std::numeric_limits<double>::epsilon() * scale;
}

bool same_value(double a, double b)
{
    const double abs_a = std::abs(a);
    const double abs_b = std::abs(b);
    const double magnitude = std::max(abs_a, abs_b);
    const double scale = std::max(1.0, magnitude);
    return std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
}

bool same_lattice(const ModuleBase::Matrix3 &a, const ModuleBase::Matrix3 &b)
{
    return same_value(a.e11, b.e11) && same_value(a.e12, b.e12) && same_value(a.e13, b.e13) && same_value(a.e21, b.e21) &&
           same_value(a.e22, b.e22) && same_value(a.e23, b.e23) && same_value(a.e31, b.e31) && same_value(a.e32, b.e32) &&
           same_value(a.e33, b.e33);
}

bool same_vector(const ModuleBase::Vector3<double> &a, const ModuleBase::Vector3<double> &b)
{
    for (int axis = 0; axis < 3; ++axis)
    {
        if (!same_value(a[axis], b[axis]))
        {
            return false;
        }
    }
    return true;
}

void check_ions(const unitcell::IonicState &state, const UnitCell &cell)
{
    if (!same_value(state.lat0, cell.lat0) || !same_lattice(state.lattice, cell.latvec) ||
        state.atoms.size() != static_cast<std::size_t>(cell.nat))
    {
        throw std::invalid_argument("Ionic sample has an invalid atom count or fixed cell.");
    }
    int id = 0;
    for (int type = 0; type < cell.ntype; ++type)
    {
        for (int index = 0; index < cell.atoms[type].na; ++index, ++id)
        {
            const unitcell::IonicAtom &atom = state.atoms[id];
            if (atom.id != id || atom.type != type || atom.index != index)
            {
                throw std::invalid_argument("Ionic sample changed atom identity.");
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                if (!std::isfinite(atom.position[axis]) || !std::isfinite(atom.frac[axis]) || !std::isfinite(atom.velocity[axis]) ||
                    !std::isfinite(atom.disp[axis]))
                {
                    throw std::invalid_argument("Invalid ionic coordinate, velocity or displacement.");
                }
            }
        }
    }
}

bool same_ions(const unitcell::IonicState &left, const unitcell::IonicState &right)
{
    if (!same_time(left.t_pos, right.t_pos) || !same_time(left.t_vel, right.t_vel))
    {
        return false;
    }
    for (std::size_t i = 0; i < left.atoms.size(); ++i)
    {
        const unitcell::IonicAtom &a = left.atoms[i];
        const unitcell::IonicAtom &b = right.atoms[i];
        if (!same_vector(a.position, b.position) || !same_vector(a.velocity, b.velocity))
        {
            return false;
        }
    }
    return true;
}

bool precedes(const unitcell::IonicState &state, double time) { return state.t_pos < time; }
} // namespace

TDIonPath::TDIonPath(const std::vector<unitcell::IonicState> &samples) : samples_(samples), stationary_(false)
{
    if (samples_.empty())
    {
        throw std::invalid_argument("An ionic path needs explicit time samples.");
    }
    for (std::size_t i = 0; i < samples_.size(); ++i)
    {
        const unitcell::IonicState &state = samples_[i];
        if (!std::isfinite(state.t_pos) || !std::isfinite(state.t_vel) || !same_time(state.t_pos, state.t_vel) ||
            (i > 0 && state.t_pos <= samples_[i - 1].t_pos))
        {
            throw std::invalid_argument("Ionic path samples must be simultaneous and strictly ordered in time.");
        }
    }
}

unitcell::IonicState TDIonPath::at(double time) const
{
    if (!std::isfinite(time))
    {
        throw std::invalid_argument("Ionic sampling time must be finite.");
    }
    if (hermite_)
    {
        const unitcell::IonicState &left = samples_.front();
        const unitcell::IonicState &right = samples_.back();
        if (same_time(time, left.t_pos))
        {
            return left;
        }
        if (same_time(time, right.t_pos))
        {
            return right;
        }
        if (time < left.t_pos || time > right.t_pos)
        {
            throw std::invalid_argument("Hermite sample lies outside the ionic interval.");
        }
        const double h = right.t_pos - left.t_pos;
        const double s = (time - left.t_pos) / h;
        const double s2 = s * s;
        const double s3 = s2 * s;

        unitcell::IonicState result = left;
        result.t_pos = time;
        result.t_vel = time;

        // Differentiate the same Hermite position polynomial to obtain simultaneous velocities.
        for (unitcell::IonicAtom &atom : result.atoms)
        {
            const unitcell::IonicAtom &a = left.atoms[atom.id];
            const unitcell::IonicAtom &b = right.atoms[atom.id];
            const ModuleBase::Vector3<double> delta = b.position - a.position;
            atom.position = a.position + delta * (3.0 * s2 - 2.0 * s3)
                            + a.velocity * (h * (s3 - 2.0 * s2 + s)) + b.velocity * (h * (s3 - s2));
            atom.velocity = delta * ((6.0 * s - 6.0 * s2) / h)
                            + a.velocity * (3.0 * s2 - 4.0 * s + 1.0) + b.velocity * (3.0 * s2 - 2.0 * s);
            atom.disp = atom.position - a.position;

            atom.frac = atom.position / result.lat0 * result.lattice.Inverse();
            for (int d = 0; d < 3; ++d)
            {
                atom.frac[d] -= std::floor(atom.frac[d]);
            }
        }
        return result;
    }
    if (stationary_ && time >= samples_.front().t_pos && time <= samples_.back().t_pos)
    {
        unitcell::IonicState sample = samples_.front();
        sample.t_pos = time;
        sample.t_vel = time;
        return sample;
    }
    const std::vector<unitcell::IonicState>::const_iterator begin = samples_.begin();
    // Check both neighbors because roundoff can place the requested time on either side of a stored node.
    const std::vector<unitcell::IonicState>::const_iterator end = samples_.end();
    const std::vector<unitcell::IonicState>::const_iterator it = std::lower_bound(begin, end, time, precedes);
    if (it != samples_.end() && same_time(it->t_pos, time))
    {
        unitcell::IonicState sample = *it;
        sample.t_pos = time;
        sample.t_vel = time;
        return sample;
    }
    if (it != begin)
    {
        const unitcell::IonicState &previous = *(it - 1);
        if (same_time(previous.t_pos, time))
        {
            unitcell::IonicState sample = previous;
            sample.t_pos = time;
            sample.t_vel = time;
            return sample;
        }
    }
    throw std::invalid_argument("Missing explicit ionic sample at the requested operator time.");
}

TDIonPath TDIonPath::hermite(const unitcell::IonicState &left, const unitcell::IonicState &right)
{
    const std::vector<unitcell::IonicState> samples{left, right};
    TDIonPath path(samples);
    path.hermite_ = true;
    return path;
}

void TDIonPath::check(const UnitCell &cell, const unitcell::IonicState &accepted, double time) const
{
    // Fixed-cell and atom-identity checks belong to this PW path, while Cell stores general ionic samples.
    for (const unitcell::IonicState &sample : samples_)
    {
        check_ions(sample, cell);
    }
    const unitcell::IonicState origin = at(time);
    if (!same_ions(origin, accepted))
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW_TDDFT", "Path origin differs from the accepted ionic state.");
    }
}

TDIonPath::TDIonPath(const unitcell::IonicState &state, double left, double right) : stationary_(true)
{
    if (!std::isfinite(left) || !std::isfinite(right) || !(right > left))
    {
        throw std::invalid_argument("A stationary ionic interval must have finite increasing times.");
    }
    samples_.assign(2, state);
    for (unitcell::IonicState &sample : samples_)
    {
        for (unitcell::IonicAtom &atom : sample.atoms)
        {
            atom.velocity.set(0.0, 0.0, 0.0);
        }
    }
    samples_[0].t_pos = left;
    samples_[0].t_vel = left;
    samples_[1].t_pos = right;
    samples_[1].t_vel = right;
}

TDIonPW::TDIonPW(
    UnitCell *cell,
    Structure_Factor *sf,
    const Parallel_Grid &grid,
    const ModulePW::PW_Basis &rho_basis,
    Charge *charge,
    const bool *numeric,
    bool has_float)
    : cell_(*cell), sf_(*sf), grid_(grid), rho_basis_(rho_basis), charge_(*charge), has_float_(has_float)
{
    // Cache element core transforms once because ion motion changes only their structure factors on a fixed grid.
    module_charge::cal_core_rad(cell_, numeric, rho_basis_, &core_rad_);
}

void TDIonPW::bind(const unitcell::IonicState &sample)
{
    // Apply endpoint positions and velocities before refreshing its density-dependent quantities.
    bool changed = state_.atoms.empty();
    for (const unitcell::IonicAtom &atom : sample.atoms)
    {
        Atom &target = cell_.atoms[atom.type];
        changed = changed || !(target.taud[atom.index] == atom.frac);
        if (!state_.atoms.empty())
        {
            changed = changed || !(state_.atoms[atom.id].frac == atom.frac);
        }
        target.taud[atom.index] = atom.frac;
        target.tau[atom.index] = atom.frac * cell_.latvec;
        target.vel[atom.index] = atom.velocity;
    }
    if (changed)
    {
        // Refresh endpoint phases used by the ordinary PW force and potential interfaces.
        sf_.cal_phase(cell_, rho_basis_, has_float_);
    }
    state_ = sample;
}

void TDIonPW::update_pot(const unitcell::IonicState &sample, elecstate::Potential *pot)
{
    // The shared geometry always belongs to a density endpoint.
    bool changed = state_.atoms.empty();
    for (const unitcell::IonicAtom &atom : sample.atoms)
    {
        changed = changed || !(state_.atoms[atom.id].frac == atom.frac);
    }
    bind(sample);
    if (changed)
    {
        sf_.cal_sf(cell_, grid_, rho_basis_);
        module_charge::set_rho_core(cell_, sf_.strucFac, core_rad_, charge_.rho_core, charge_.rhog_core, rho_basis_);
    }
    const elecstate::TDFieldManager *field = pot->td_field_manager();
    // The length-gauge scalar field is part of the fixed potential components.
    if (changed || field->gauge() == 0)
    {
        pot->init_pot(&charge_);
    }
    else
    {
        // Reuse the fixed ionic potential and update only charge-dependent terms for an unchanged velocity-gauge endpoint.
        pot->update_from_charge(&charge_, &cell_);
    }
}

const unitcell::IonicState &TDIonPW::state() const { return state_; }

double TDIonPW::cal_ewald(const int &test_energy, std::ofstream *log)
{
    return H_Ewald_pw::compute_ewald(cell_, &rho_basis_, sf_.strucFac, test_energy, *log);
}

} // namespace pw
