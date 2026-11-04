#ifndef ESOLVER_KS_PW_TDDFT_H
#define ESOLVER_KS_PW_TDDFT_H

#include "source_esolver/esolver_ks_pw.h"
#include "source_estate/module_pot/td_field_manager.h"
#include "source_pw/module_rt/td_history_pw.h"
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

    /** @brief Initialize the field, propagation solver and time-step history. */
    void before_all_runners(BaseCell &basecell, const Input_para &inp) override;
    /** @brief Adapt the fixed-ion MD schedule to initialization and electronic propagation. */
    void runner(BaseCell &basecell, const int istep) override;
    /** @brief Propagate the inclusive electronic-step range with explicit ionic samples. */
    void propagate(
        UnitCell *cell,
        int first_step,
        int last_step,
        const pw::TDIonPath &ions);
    double cal_energy() override;

  protected:
    /** @brief Use independent iteration limits for initial-state SCF and propagation. */
    int scf_iter_limit(const int istep) const override;
    /** @brief Use ground-state diagonalization for the initial electronic step. */
    std::string diag_policy(const int istep) const override;

    /** @brief Require an unmixed endpoint prediction before accepting a corrected step. */
    ESolver_KS::DensityStage density_stage(const int istep, const int iter) const override;
    bool defer_iter_output(const int istep) const override;

    void before_scf(UnitCell &ucell, const int istep) override;

    /** @brief Cover shifted projectors and bind endpoint A before the first Hamiltonian application. */
    void prepare_init_hamilt(const UnitCell &ucell) override;

    /** @brief Print the TDDFT iteration header and initialize the PW iteration. */
    void iter_init(UnitCell &ucell, const int istep, const int iter) override;

    /** @brief Switch between ground-state diagonalization and real-time propagation. */
    void hamilt2rho_single(
        UnitCell &ucell,
        const int istep,
        const int iter,
        const double ethr) override;

    /** @brief Save the converged or iteration-limited endpoint for the next propagation step. */
    void after_scf(UnitCell &ucell, const int istep, const bool conv_esolver) override;

  private:
    using Real = typename GetTypeReal<T>::type;
    std::ofstream &log_;
    std::unique_ptr<pw::TDPropPW<T, Device>> td_prop_;

    std::shared_ptr<elecstate::TDFieldManager> field_;
    pw::TDVelPW<Real, Device> vel_;
    pw::TDSpinPW<Real, Device> spin_;
    pw::CurrentRequest current_req_;

    pw::TDHistoryPW<T, Device> history_;
    std::unique_ptr<pw::TDIonPW> ion_ops_;
    // Keep the method's operator sample separate from the endpoint used for density and observables.
    unitcell::IonicState prop_ions_;
    unitcell::IonicState right_ions_;
    double qmax_ = 0.0;
    double npsi0_ = 0.0;
    double nrho0_ = 0.0;
    double material_energy_ = 0.0;

    void run_prop(UnitCell *cell, int step, const pw::TDIonPath &ions);
    bool advance_interval(
        UnitCell *cell,
        int step,
        int index,
        int depth,
        const elecstate::TDFieldState &full,
        const pw::TDIonPath &ions,
        hsolver::TDOrthStats *stats);
    void rebuild_end(UnitCell *cell);
    void out_prop(const UnitCell &cell, int step, bool converged);
    void out_energy(int step, int rank, const std::string &out_dir);
    void out_td(
        const UnitCell &cell,
        int step,
        int rank,
        const std::string &out_dir);
    void out_eij(int step);
    /** @brief Print the one-based electronic step header. */
    void print_step(const int istep) const;
    void prep_td_step(const int istep);
};

} // namespace ModuleESolver
#endif
