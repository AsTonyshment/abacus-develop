#ifndef PW_TD_HAMILT_PW_H
#define PW_TD_HAMILT_PW_H
#include "source_hamilt/hamilt_hs_adapter.h"
#include "source_pw/module_pwdft/hamilt_pw.h"

#include "source_cell/ionic_state.h"

namespace pw
{
/** @brief A call-scoped PW sample selecting immutable nonlocal projectors. */
template <typename T, typename Device>
class TDHamiltPW final : public hsolver::HSOperator<T, Device>
{
  public:
    using Real = typename T::value_type;
    TDHamiltPW(
        hamilt::HamiltPW<T, Device> *hamilt,
        const ModulePW::PW_Basis_K &basis,
        const bool &motion);
    /** @brief Apply an ionic sample and borrow the potential for subsequent k-point applications. */
    void bind(const Real *veff, const ModuleBase::Vector3<double> &vecpot_ha, const unitcell::IonicState &state);
    void update_k(int ik) override;
    void hpsi(
        const T *x,
        T *hx,
        int ld,
        int bands) const override;
    void spsi(
        const T *x,
        T *sx,
        int ld,
        int bands) const override;

  private:
    hamilt::HamiltPW<T, Device> &hamilt_;
    hamilt::HamiltHSOperator<T, Device> op_;
    const bool motion_;
};
} // namespace pw
#endif
