#ifndef PW_PROP_ETRS_PW_H
#define PW_PROP_ETRS_PW_H
#include "source_hsolver/diag_comm_info.h"
#include "source_hsolver/module_linalg/exp_action.h"
#include "source_pw/module_rt/td_prop_pw.h"

namespace pw
{
template <typename T, typename Device>
class ETRSPropPW final : public PWProp<T, Device>
{
  public:
    ETRSPropPW(
        const hsolver::ExpControls &exp_ctrl,
        int npol,
        const std::vector<int> &global_k,
        const hsolver::diag_comm_info &comm,
        std::ostream &log);
    void begin_trial() override;
    double sample_time(double left, double right) const override { return left; }
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
    const hsolver::diag_comm_info pool_comm_;
    std::unique_ptr<hsolver::ExpAction<T, Device>> exponential_;
    ct::Tensor half_psi_;
    const int npol_;
    const std::vector<int> global_k_;
    std::ostream &log_;
    bool half_ready_ = false;
    void report(
        const hsolver::ExpResult &result,
        int ik,
        int elec_step,
        int scf_iter,
        bool log_detail) const;
};
} // namespace pw
#endif
