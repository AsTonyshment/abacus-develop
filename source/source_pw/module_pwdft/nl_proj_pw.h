#ifndef NL_PROJ_PW_H
#define NL_PROJ_PW_H

#include "source_cell/atom_phase.h"
#include "source_pw/module_pwdft/vnl_pw.h"
#include <memory>

namespace hamilt
{
/** @brief Device-resident projectors indexed by their physical sampling geometry. */
template <typename Real, typename Device>
class NLProjPW
{
  public:
    NLProjPW(const UnitCell &cell, const ModulePW::PW_Basis_K &basis, const pseudopot_cell_vnl &pp);
    /** @brief Select a physical operator sample, independently of the shared cell geometry. */
    void select(const std::vector<ModuleBase::Vector3<double>> &tau,
                const std::vector<ModuleBase::Vector3<double>> &vel,
                const ModuleBase::Vector3<double> &a);
    /** @brief Protect the accepted origin throughout the next ionic step. */
    void begin_step();
    //! Release the original endpoint after completing the ionic step.
    void end_step();
    //! Protect the selected sample while endpoint observables borrow its projectors.
    void hold_endpoint();
    //! Release endpoint protection after clearing its borrowed projector pointers.
    void drop_endpoint();
    /** @brief Borrow projectors for the selected sample and local k point. */
    const std::complex<Real> *beta(const int &ik);
    /** @brief Borrow atomic phases for derivative construction at the selected sample. */
    const std::complex<Real> *phases(const int &ik);

  private:
    using Complex = std::complex<Real>;
    struct Sample
    {
        std::vector<ModuleBase::Vector3<double>> tau;
        std::vector<ModuleBase::Vector3<double>> vel;
        ModuleBase::Vector3<double> a{0.0, 0.0, 0.0};
        unitcell::AtomPhase<Real, Device> phase;
        std::vector<std::unique_ptr<ct::Tensor>> beta;
        std::vector<size_t> version;
    };
    const UnitCell &cell_;
    const ModulePW::PW_Basis_K &basis_;
    const pseudopot_cell_vnl &pp_;
    std::vector<std::unique_ptr<Sample>> samples_;
    Sample *active_ = nullptr;
    Sample *origin_ = nullptr;
    Sample *endpoint_ = nullptr;
    ct::Tensor sk_;
};
}
#endif
