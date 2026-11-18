#ifndef TD_HISTORY_PW_H
#define TD_HISTORY_PW_H

#include "source_base/module_container/ATen/core/tensor.h"
#include "source_cell/ionic_state.h"
#include "source_estate/elecstate.h"
#include "source_estate/fp_energy.h"
#include "source_estate/module_charge/charge.h"
#include "source_estate/module_pot/potential_new.h"
#include "source_psi/psi.h"

#include <memory>
#include <vector>

namespace pw
{
/** @brief Accepted ionic state and electronic snapshots of orbitals, density, potential, energies and occupations. */
template <typename T, typename Device>
class TDHistoryPW
{
  public:
    /** @brief Prepare storage after the potential grids have been updated. */
    void prepare(const elecstate::Potential &potential);

    /** @brief Borrow the immutable accepted left-endpoint potential. */
    const typename GetTypeReal<T>::type *left_potential() const;

    /** @brief Check whether an endpoint has been saved. */
    bool ready() const { return previous_ != nullptr; }

    /** @brief Deep-copy an immutable origin before nested propagation changes the working history. */
    void copy_from(const TDHistoryPW &source);

    /** @brief Save the accepted endpoint; the caller owns the convergence decision. */
    void save(
        const psi::Psi<T, Device> &current,
        const elecstate::ElecState &estate,
        const Charge &charge,
        const unitcell::IonicState &ions);

    /** @brief Restore accepted endpoint data before rebuilding the trial potential. */
    void restore(psi::Psi<T, Device> *current, elecstate::ElecState *estate, Charge *charge) const;

    /** @brief Borrow the ionic state at the saved electronic endpoint. */
    const unitcell::IonicState &ions() const { return ions_; }

    /** @brief Access the fixed origin of every corrector solve in the next step. */
    const psi::Psi<T, Device> &previous() const;

  private:
    using Real = typename GetTypeReal<T>::type;
    std::unique_ptr<psi::Psi<T, Device>> previous_;
    ct::Tensor veff_prev_;
    unitcell::IonicState ions_;
    elecstate::fenergy energy_;
    ModuleBase::matrix ekb_;
    ModuleBase::matrix wg_;
    elecstate::Efermi eferm_;
    std::vector<double> rho_;
};

} // namespace pw
#endif
