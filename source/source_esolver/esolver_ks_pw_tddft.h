#ifndef ESOLVER_KS_PW_TDDFT_H
#define ESOLVER_KS_PW_TDDFT_H

#include "source_esolver/esolver_ks_pw.h"
#include "source_estate/module_pot/td_field_manager.h"
#include "source_pw/module_rt/td_history_pw.h"
#include "source_pw/module_rt/td_ion_accel.h"
#include "source_pw/module_rt/td_ion_pi_pw.h"
#include "source_pw/module_rt/td_ion_pw.h"
#include "source_pw/module_rt/td_prop_pw.h"
#include "source_pw/module_rt/td_spin_pw.h"
#include "source_pw/module_rt/td_vel_pw.h"

#include <iosfwd>
#include <memory>

namespace ModuleESolver
{

template <typename T, typename Device = base_device::DEVICE_CPU>
class ESolver_KS_PW_TDDFT : public ESolver_KS_PW<T, Device>
{
  public:
    ESolver_KS_PW_TDDFT();
    ~ESolver_KS_PW_TDDFT() override = default;

    //! Initialize the field, propagation solver and accepted-state storage.
    void before_all_runners(BaseCell &basecell, const Input_para &inp) override;

    //! Adapt the fixed-ion MD schedule to initialization and electronic propagation.
    void runner(BaseCell &basecell, const int istep) override;

    //! Propagate the inclusive electronic-step range with explicit ionic samples.
    void propagate(
        UnitCell *cell,
        int first_step,
        int last_step,
        const pw::TDIonPath &ions);

    //! Return the accepted endpoint potential in Ry for the MD energy calculation.
    double cal_energy() override;

    //! Return position, field and nuclear velocity derivative forces in Ry/Bohr.
    void cal_force(BaseCell &basecell, ModuleBase::matrix &force) override;

    //! Expose ionic accelerations, Pi rates and constraint reactions from the most recent force evaluation.
    const pw::IonAccel &ion_accel() const { return ion_accel_; }

    //! Expose accepted endpoint dE/dv by atom and Cartesian direction in Hartree atomic units.
    const ModuleBase::matrix &ion_pi() const { return ion_pi_; }

  protected:
    //! Prepare the step's field and diagnostics and initialize the common PW state at step zero.
    void before_scf(UnitCell &ucell, const int istep) override;

    //! Extend radial tables for shifted projectors and bind endpoint field and velocities before applying the Hamiltonian.
    void prepare_init_hamilt(const UnitCell &ucell) override;

    //! Select independent iteration limits for initial-state SCF and propagation.
    int scf_iter_limit(const int istep) const override;

    //! Print the TDDFT iteration header and initialize the PW iteration.
    void iter_init(UnitCell &ucell, const int istep, const int iter) override;

    //! Use the ground-state diagonalization policy for the initial electronic step.
    std::string diag_policy(const int istep) const override;

    //! Obtain the electronic density through initial-state diagonalization or real-time propagation.
    void hamilt2rho_single(
        UnitCell &ucell,
        const int istep,
        const int iter,
        const double ethr) override;

    //! Select an unmixed first-iteration predictor during propagation and standard density updates otherwise.
    ESolver_KS::DensityStage density_stage(const int istep, const int iter) const override;

    //! Defer propagation iteration output until the full electronic-step endpoint is accepted.
    bool defer_iter_output(const int istep) const override;

    //! Finalize endpoint energy and observables and seed the accepted history after initial SCF.
    void after_scf(UnitCell &ucell, const int istep, const bool conv_esolver) override;

  private:
    using Real = typename GetTypeReal<T>::type; // Real scalar type matching the orbital precision.

    // Propagation and accepted history.
    std::shared_ptr<elecstate::TDFieldManager> field_; // External-field samples shared with the potential.
    std::unique_ptr<pw::TDPropPW<T, Device>> td_prop_; // Selected electronic propagator and orbital workspace.
    pw::TDHistoryPW<T, Device> history_;               // Accepted electronic and ionic state used to start propagation trials.

    // Ionic samples and geometry-dependent operators.
    std::unique_ptr<pw::TDIonPW> ion_ops_; // Updates geometry-dependent PW operators from ionic samples.
    unitcell::IonicState prop_ions_;       // Ionic sample used by the propagation scheme's Hamiltonian.
    unitcell::IonicState right_ions_;      // Right-endpoint ionic sample used for density, energy and observables.
    double qmax_ = 0.0;                    // Maximum unshifted |G+k| in inverse Bohr for radial-table bounds.

    // Observable evaluators and endpoint results.
    pw::CurrentRequest current_req_;      // Enabled total charge current, per-k-point charge current and spin-current outputs.
    pw::TDVelPW<Real, Device> vel_;       // Charge and spin-current evaluator for accepted endpoints.
    pw::TDSpinPW<Real, Device> spin_;     // Endpoint magnetization and spin-torque evaluator.
    pw::TDIonPiPW<Real, Device> pi_calc_; // Nuclear velocity derivative evaluator with reusable projection buffers.
    ModuleBase::matrix ion_pi_;           // Accepted endpoint dE/dv by atom and direction in Hartree atomic units.
    pw::IonAccel ion_accel_;              // Ionic accelerations, Pi rates and constraint reactions from the latest force evaluation.

    // Energy, conservation references and output.
    double md_energy_ = 0.0; // Accepted endpoint potential in Ry returned to MD.
    double npsi0_ = 0.0;     // Initial orbital electron count used as the conservation reference.
    double nrho0_ = 0.0;     // Initial density electron count used as the conservation reference.
    std::ofstream &log_;     // Running log shared with the driver.

    //! Advance and finalize one electronic step along the supplied ionic path.
    void run_prop(UnitCell *cell, int step, const pw::TDIonPath &ions);

    //! Prepare the step's external-field samples and report the field according to the output schedule.
    void prep_td_step(const int istep);

    //! Advance a trial interval with recursive halving and report convergence of its accepted subintervals.
    bool advance_interval(
        UnitCell *cell,
        int step,
        int index,
        int depth,
        const elecstate::TDFieldState &full,
        const pw::TDIonPath &ions,
        hsolver::TDOrthStats *stats);

    //! Rebuild consistent endpoint density, potential and energy when accepting an iteration-limited state.
    void rebuild_end(UnitCell *cell);

    //! Write density restart data and orbital results at the accepted full-step endpoint.
    void out_prop(const UnitCell &cell, int step, bool converged);

    //! Sample Pi, update the MD potential and report the accepted endpoint energies.
    void out_energy(
        const UnitCell &cell,
        int step,
        int rank,
        const std::string &out_dir);

    //! Evaluate and output requested endpoint currents, spin observables and conservation diagnostics.
    void out_td(
        const UnitCell &cell,
        int step,
        int rank,
        const std::string &out_dir);

    //! Write requested Hamiltonian matrix elements in the propagated orbital basis.
    void out_eij(int step);

    //! Print the one-based electronic step header.
    void print_step(const int istep) const;
};

} // namespace ModuleESolver
#endif
