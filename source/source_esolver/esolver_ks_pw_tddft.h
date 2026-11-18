#ifndef ESOLVER_KS_PW_TDDFT_H
#define ESOLVER_KS_PW_TDDFT_H

#include "source_esolver/esolver_ks_pw.h"
#include "source_estate/module_pot/td_field_manager.h"
#include "source_pw/module_pwdft/nl_orb_data.h"
#include "source_pw/module_rt/td_conserv_pw.h"
#include "source_pw/module_rt/td_end_sample.h"
#include "source_pw/module_rt/td_history_pw.h"
#include "source_pw/module_rt/td_ion_accel.h"
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

    //! Initialize once and advance fixed-ion or coupled NVE propagation on the MD schedule.
    void runner(BaseCell &basecell, const int istep) override;

    //! Advance an inclusive step range on a supplied path, preparing full nuclear derivatives at its last step when requested.
    void prop_path(
        UnitCell *cell,
        const int &first_step,
        const int &last_step,
        const pw::TDIonPath &ions,
        const bool &need_force);

    //! Return the accepted endpoint potential in Ry for the MD energy calculation.
    double cal_energy() override;

    //! Return position, field and nuclear velocity derivative forces in Ry/Bohr.
    void cal_force(BaseCell &basecell, ModuleBase::matrix &force) override;

    //! Borrow dynamics from the evaluated accepted endpoint; valid until the next propagation or initialization.
    const pw::IonAccel &ion_accel() const;

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

    //! Accepted interval result with reusable right-endpoint Ewald energy.
    struct IntervalEnd
    {
        bool converged = false;
        double ewald = 0.0;
    };

    //! Statistics accumulated over retained electronic intervals.
    struct IntervalStats
    {
        double before = 0.0;
        double after = 0.0;
        double residual = 0.0;
        int iters = 0;
        int max_depth = 0;
    };

    // Propagation and accepted history.
    std::shared_ptr<elecstate::TDFieldManager> field_; // External-field samples shared with the potential.
    std::unique_ptr<pw::TDPropPW<T, Device>> td_prop_; // Selected electronic propagator and orbital workspace.
    pw::TDHistoryPW<T, Device> history_;               // Accepted electronic and ionic state used to start propagation trials.
    pw::TDHistoryPW<T, Device> ion_origin_;            // Reusable immutable origin for repeated ionic trials.

    // Ionic samples and geometry-dependent operators.
    std::unique_ptr<pw::TDIonPW> ion_ops_; // Updates geometry-dependent PW operators from ionic samples.
    unitcell::IonicState prop_ions_;       // Ionic sample used by the propagation scheme's Hamiltonian.
    unitcell::IonicState right_ions_;      // Right-endpoint ionic sample used for density, energy and observables.
    double qmax_ = 0.0;                    // Maximum unshifted |G+k| in inverse Bohr for radial-table bounds.

    // Observable evaluators and endpoint results.
    pw::CurrentRequest current_req_;          // Enabled total charge current, per-k-point charge current and spin-current outputs.
    pw::TDVelPW<Real, Device> vel_;           // Charge and spin-current evaluator for accepted endpoints.
    pw::TDSpinPW<Real, Device> spin_;         // Endpoint magnetization and spin-torque evaluator.
    pw::TDEndpointPW<Real, Device> endpoint_; // Endpoint products and derivative workspace.
    std::unique_ptr<const pw::IonForce> endpoint_force_; // Complete force result for the current accepted endpoint.

    // Energy, conservation references and output.
    double md_energy_ = 0.0; // Accepted endpoint potential in Ry returned to MD.
    double npsi0_ = 0.0;     // Initial orbital electron count used as the conservation reference.
    double nrho0_ = 0.0;     // Initial density electron count used as the conservation reference.
    std::ofstream &log_;     // Running log shared with the driver.

    //! Initialize ionic operators, the propagator and accepted-state storage on the common PW grids.
    void init_td(UnitCell *cell, const Input_para &inp);

    //! Initialize current and spin evaluators for the requested endpoint observables.
    void init_obs(const UnitCell &cell, const Input_para &inp);

    //! Evaluate forces and instantaneous dynamics at the current trial endpoint; the caller decides acceptance.
    pw::IonForce cal_ion_force(UnitCell &cell, const ModuleBase::matrix *ion_force);

    //! Select the nonlocal shift and compatible endpoint overlaps for the ionic position force.
    const hamilt::NLOrbData<double, Device> *force_proj(ModuleBase::Vector3<double> *shift) const;

    //! Advance one original electronic step with internal retries, following a supplied path or solving coupled ionic motion.
    void prop_step(
        UnitCell *cell,
        const int &step,
        const pw::TDIonPath *prescribed,
        const bool &need_force);

    //! Prepare the step's external-field samples and report the field according to the output schedule.
    void prep_td_step(const int istep);

    //! Advance a trial interval with recursive halving and report convergence of its accepted subintervals.
    IntervalEnd advance_interval(
        UnitCell *cell,
        const int &step,
        const int &index,
        const int &depth,
        const elecstate::TDFieldState &full,
        const pw::TDIonPath &ions,
        const bool &need_derivs,
        const double *right_ewald,
        IntervalStats *stats);

    //! Evaluate endpoint energy and derivatives using the prepared density and potential.
    void finish_end(UnitCell *cell, const bool &need_products, const bool &need_accel);

    //! Commit corrected velocities and evaluate the complete endpoint for the next step.
    void correct_end(UnitCell *cell, const unitcell::IonicState &ions, const pw::IonForce &force);

    //! Write density restart data and orbital results at the accepted full-step endpoint.
    void out_prop(const UnitCell &cell, const int &step, const bool &conv);

    //! Prepare Pi and borrow endpoint overlaps until the next Pi evaluation; nullptr denotes absent nonlocal projectors.
    const hamilt::NLOrbData<typename GetTypeReal<T>::type, Device> *prep_end_orbs(const UnitCell &cell, ModuleBase::matrix *ion_pi);

    //! Evaluate accepted endpoint energy contributions from the supplied nuclear velocity derivative.
    pw::TDEnergy cal_end_energy(const UnitCell &cell, const ModuleBase::matrix &pi) const;

    //! Print precomputed endpoint energies without changing the accepted physical state.
    void out_energy(
        const pw::TDEnergy &energy,
        const int &step,
        const int &rank,
        const std::string &out_dir) const;

    //! Evaluate and output currents and spin; nullptr selects independent nonlocal projection when no shared overlaps exist.
    void out_curr_spin(
        const UnitCell &cell,
        const hamilt::NLOrbData<typename GetTypeReal<T>::type, Device> *nl_orbs,
        const int &step,
        const int &rank,
        const std::string &out_dir);

    //! Establish initial electron-count references and print accepted endpoint conservation diagnostics.
    void out_conserv(const UnitCell &cell, const int &step);

    //! Write requested Hamiltonian matrix elements in the propagated orbital basis.
    void out_eij(int step);

    //! Print the one-based electronic step header.
    void print_step(const int istep) const;
};

} // namespace ModuleESolver
#endif
