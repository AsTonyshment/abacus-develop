#ifndef ATOM_PHASE_H
#define ATOM_PHASE_H

#include "source_base/module_container/ATen/core/tensor.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/unitcell.h"

namespace unitcell
{
/** @brief Atomic translation phases owned by one immutable operator geometry. */
template <typename Real, typename Device>
class AtomPhase
{
  public:
    /** @brief Prepare phases from explicit Cartesian positions in units of lat0. */
    void prep(const UnitCell &cell, const ModulePW::PW_Basis &rho,
              const std::vector<ModuleBase::Vector3<double>> &tau);
    /** @brief Form exp[-i(G+k).R] on the native device; output has nat*npw entries. */
    void cal_sk(const ModulePW::PW_Basis_K &basis, const int &ik, std::complex<Real> *out) const;

  private:
    ct::Tensor phase_[3];
    ct::Tensor tau_;
    ct::Tensor na_;
    int nr_[3] = {0, 0, 0};
    int ntype_ = 0;
    std::vector<std::complex<double>> host_phase_;
    std::vector<double> host_tau_;
    std::vector<int> host_na_;
};
}
#endif
