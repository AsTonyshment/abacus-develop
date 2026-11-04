#ifndef PW_PROP_CN_PW_H
#define PW_PROP_CN_PW_H
#include "source_pw/module_rt/cn_solver_pw.h"
#include "source_pw/module_rt/td_prop_pw.h"

namespace pw
{
template <typename T, typename Device>
class CNPropPW final : public PWProp<T, Device>
{
  public:
    CNPropPW(
        const ModulePW::PW_Basis_K &basis,
        const hsolver::CNControls &cn_ctrl,
        int npol,
        const std::vector<int> &global_k,
        const hsolver::diag_comm_info &comm,
        std::ostream &log);
    void begin_trial() override
    {
        // A new interval changes the CN system, so discard its reusable linear-solver basis.
        solver_.invalidate_basis();
    }
    double sample_time(double left, double right) const override { return left + 0.5 * (right - left); }
    void evolve(
        TDHamiltPW<T, Device> *hamilt,
        const TDHistoryPW<T, Device> &history,
        const typename GetTypeReal<T>::type *veff,
        int nveff,
        const elecstate::TDFieldManager &field,
        const unitcell::IonicState &sample,
        const unitcell::IonicState &right,
        psi::Psi<T, Device> *current,
        int elec_step,
        int scf_iter,
        bool log_detail) override;

  private:
    hsolver::CNSolverPW<T, Device> solver_;
    ct::Tensor veff_mid_;
};
} // namespace pw
#endif
