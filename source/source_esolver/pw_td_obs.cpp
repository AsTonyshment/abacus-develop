#include "source_esolver/esolver_ks_pw_tddft.h"

#include "source_base/global_variable.h"
#include "source_base/module_parallel/para_bridge.h"
#include "source_estate/elecstate_tools.h"
#include "source_estate/module_charge/chg_rhog_io.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_ctrl/ctrl_output_pw.h"
#include "source_io/module_ctrl/ctrl_output_td_pw.h"
#include "source_io/module_energy/td_eij_pw.h"
#include "source_io/module_output/td_conserv_io.h"
#include "source_io/module_parameter/parameter.h"
#include "source_pw/module_rt/td_pw.h"

#include <iostream>

namespace ModuleESolver
{
template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::out_energy(
    const UnitCell &cell,
    int istep,
    int rank,
    const std::string &output_dir)
{
    // Sample Pi before energy evaluation so the velocity correction uses the same accepted endpoint.
    const psi::Psi<T, Device> &wave = *this->stp.template get_psi_t<T, Device>();
    const std::vector<ModuleBase::Vector3<double>> vel = pw::td_ion_vel(cell, this->inp_->td_nl_motion);
    ModuleBase::Vector3<double> nl_a{0.0, 0.0, 0.0};
    if (this->field_->gauge() == 1 && this->inp_->td_vg_nl != 0)
    {
        nl_a = this->field_->A_right_ha();
    }
    if (this->inp_->vnl_in_h)
    {
        ion_pi_ = pi_calc_.cal_pi(cell, *this->pw_wfc, this->kv, &this->ppcell, wave, this->pelec->wg, nl_a, vel);
    }
    else
    {
        ion_pi_.create(cell.nat, 3);
    }

    double elec_field = 0.0;
    double ionic_field = 0.0;
    double total_field = 0.0;
    double total = 0.0;
    double kinetic = 0.0;
    double vel_corr = 0.0;

    // Sample scalar-field energies at the accepted RT-TDDFT endpoint before combining its energy contributions.
    this->pelec->pot->td_field_energy(*this->pelec->charge, elec_field, ionic_field);

    // MD adds ionic kinetic energy to the same potential used by the RT-TDDFT endpoint totals.
    md_energy_ = elecstate::cal_td_energy(
        this->pelec->f_en.etot, cell, ion_pi_, elec_field, ionic_field, &total, &total_field, &kinetic, &vel_corr);

    const double dt = this->field_->dt_ha();
    ModuleIO::out_td_energy(
        output_dir,
        istep,
        dt,
        total,
        total_field,
        elec_field,
        ionic_field,
        kinetic,
        vel_corr,
        md_energy_,
        this->pelec->f_en.demet,
        this->inp_->td_out_stat,
        rank,
        &log_,
        &std::cout);
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::out_td(
    const UnitCell &ucell,
    int istep,
    int rank,
    const std::string &output_dir)
{
    if (istep >= 0)
    {
        const psi::Psi<T, Device> &current = *this->stp.template get_psi_t<T, Device>();
        const ModuleBase::Vector3<double> &vecpot = this->field_->A_right_ha();
        ModuleBase::Vector3<double> nonlocal_a{0.0, 0.0, 0.0};
        if (this->inp_->td_vg_nl != 0)
        {
            nonlocal_a = vecpot;
        }
        // Evaluate observables with the accepted endpoint geometry, field and nuclear velocities.
        const std::vector<ModuleBase::Vector3<double>> ion_vel = pw::td_ion_vel(ucell, this->inp_->td_nl_motion);
        const pw::TDVelResultPW currents = vel_.calculate(current, *this->pelec, vecpot, nonlocal_a, ion_vel, current_req_);

        pw::TDSpinResultPW moments;
        if (this->inp_->nspin == 4 && (this->inp_->td_out_mag != "none" || this->inp_->td_out_torque))
        {
            moments = spin_.calculate(current, *this->pelec, vecpot, nonlocal_a, ion_vel);
        }

        const double dt_ha = this->field_->dt_ha();
        const int nkstot = this->kv.get_nkstot();
        ModuleIO::ctrl_output_td_pw(
            currents,
            moments,
            current_req_,
            this->inp_->td_out_mag,
            this->inp_->orbital_dir,
            this->inp_->onsite_radius,
            this->inp_->td_out_torque,
            ucell.orbital_fn,
            istep,
            dt_ha,
            this->inp_->nspin,
            nkstot,
            rank,
            output_dir,
            log_);
    }

    if (this->inp_->td_out_stat)
    {
        const double local_npsi = this->td_prop_->orbitals().wave_electrons(this->pelec->wg);
        const hsolver::TDOrthStats &stats = this->td_prop_->orbitals().orth_stats();
        const bool full_gram = this->inp_->td_orthonormal != "none";
        const pw::TDConserv values =
            pw::cal_td_conserv(local_npsi, this->chr, ucell.omega, this->pw_wfc->poolrank, stats.before, stats.after, full_gram);
        if (istep == 0)
        {
            // Use the initial wavefunction and density electron counts as references for conservation diagnostics.
            npsi0_ = values.npsi;
            nrho0_ = values.nrho;
        }
        ModuleIO::print_conserv(values, npsi0_, nrho0_, istep, full_gram, this->inp_->td_orthonormal, log_);
    }
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::out_eij(int istep)
{
    if (this->inp_->td_print_eij >= 0.0)
    {
        psi::Psi<T, Device> &current = *this->stp.template get_psi_t<T, Device>();
        hamilt::HamiltPW<T, Device> *hamilt = static_cast<hamilt::HamiltPW<T, Device> *>(this->p_hamilt);
        hamilt::HamiltHSOperator<T, Device> op(hamilt, this->pw_wfc);
        ModuleIO::print_eij_pw(
            op, this->td_prop_->orbitals(), current, this->kv, this->inp_->nspin, istep, this->niter, this->inp_->td_print_eij, log_);
    }
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::print_step(const int istep) const
{
    const int evolution_step = istep + 1;
    std::cout << " -------------------------------------------" << std::endl;
    std::cout << " STEP OF ELECTRON EVOLVE : " << evolution_step << std::endl;
    std::cout << " -------------------------------------------" << std::endl;
}

template <typename T, typename Device>
void ESolver_KS_PW_TDDFT<T, Device>::out_prop(const UnitCell &cell, int step, bool converged)
{
    const System_para &system = PARAM.globalv;
    const bool write_restart = this->inp_->out_chg[0] != -1;
    const std::string prefix = system.global_out_dir + this->inp_->suffix;
    const bool write_tau = XC_Functional::get_ked_flag();
    const bool writer_pool = GlobalV::MY_POOL == 0;
    const Parallel::ParaWorld pw_world = Parallel::make_pw_world();
    module_charge::write_density_restart(
        this->chr,
        *this->pw_rhod,
        cell.GT,
        prefix,
        this->inp_->nspin,
        write_restart,
        system.gamma_only_pw,
        write_tau,
        writer_pool,
        pw_world,
        &GlobalV::ofs_warning);

    this->write_iter_bands();
    ModuleIO::ctrl_iter_pw(step, this->niter, converged, this->stp, this->kv, this->pw_wfc, *this->inp_, true, log_);
}

template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::out_prop(const UnitCell &, int, bool);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::out_energy(
    const UnitCell &,
    int,
    int,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::out_td(
    const UnitCell &,
    int,
    int,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::out_eij(int);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_CPU>::print_step(int) const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::out_prop(const UnitCell &, int, bool);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::out_energy(
    const UnitCell &,
    int,
    int,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::out_td(
    const UnitCell &,
    int,
    int,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::out_eij(int);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_CPU>::print_step(int) const;
#if defined(__CUDA) || defined(__ROCM)
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::out_prop(const UnitCell &, int, bool);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::out_energy(
    const UnitCell &,
    int,
    int,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::out_td(
    const UnitCell &,
    int,
    int,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::out_eij(int);
template void ESolver_KS_PW_TDDFT<std::complex<float>, base_device::DEVICE_GPU>::print_step(int) const;
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::out_prop(const UnitCell &, int, bool);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::out_energy(
    const UnitCell &,
    int,
    int,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::out_td(
    const UnitCell &,
    int,
    int,
    const std::string &);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::out_eij(int);
template void ESolver_KS_PW_TDDFT<std::complex<double>, base_device::DEVICE_GPU>::print_step(int) const;
#endif
} // namespace ModuleESolver
