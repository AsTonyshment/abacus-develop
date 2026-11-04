#include "source_pw/module_rt/td_ion_pw.h"

#include "source_base/tool_quit.h"
#include "source_estate/module_charge/chg_tools.h"
#include "source_estate/module_pot/td_field_manager.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pw
{
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
    // Apply operator-time positions and velocities while preserving the endpoint density and local potential.
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
        // Refresh per-atom phases so subsequent projector construction uses the bound ionic geometry.
        sf_.cal_phase(cell_, rho_basis_, has_float_);
    }
    state_ = sample;
}

void TDIonPW::update_pot(const unitcell::IonicState &sample, elecstate::Potential *pot)
{
    bind(sample);
    // Track the potential's endpoint separately because intermediate operator bindings also change the active geometry.
    bool changed = end_frac_.empty();
    for (const unitcell::IonicAtom &atom : sample.atoms)
    {
        changed = changed || !(end_frac_[atom.id] == atom.frac);
    }
    if (changed)
    {
        sf_.cal_sf(cell_, grid_, rho_basis_);
        module_charge::set_rho_core(cell_, sf_.strucFac, core_rad_, charge_.rho_core, charge_.rhog_core, rho_basis_);
        end_frac_.resize(sample.atoms.size());
        for (const unitcell::IonicAtom &atom : sample.atoms)
        {
            end_frac_[atom.id] = atom.frac;
        }
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

} // namespace pw
