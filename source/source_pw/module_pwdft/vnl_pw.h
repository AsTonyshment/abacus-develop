#ifndef VNL_IN_PW_H
#define VNL_IN_PW_H

#include "source_base/complexarray.h"
#include "source_base/complexmatrix.h"
#include "source_base/intarray.h"
#include "source_base/module_container/ATen/core/tensor.h"
#include "source_base/realarray.h"
#include "source_cell/soc.h"
#include "source_cell/stru_fac.h"
#include "source_cell/unitcell.h"
#include "source_psi/psi.h"

#include <array>
#include <vector>

//! Nonlocal pseudopotential projectors and augmentation terms in a plane-wave basis.
class pseudopot_cell_vnl
{
  public:
    pseudopot_cell_vnl();
    ~pseudopot_cell_vnl();

    /**
     * @brief Set projector dimensions and allocate tables and buffers.
     * @param[in] cell Atomic types and pseudopotentials.
     * @param[in] psf_in Structure-factor provider.
     * @param[in] wfc_basis Wavefunction plane-wave basis.
     * @param[in] allocate_vkb Whether to allocate the CPU vkb matrix.
     */
    void init(
        const UnitCell &cell,
        Structure_Factor *psf_in,
        const ModulePW::PW_Basis_K *wfc_basis = nullptr,
        const bool allocate_vkb = true);

    int nkb = 0;    // Total projectors over all atoms, including angular components.
    int lmaxkb = 0; // Maximum projector angular momentum; -1 for no projectors.

    /**
     * @brief Initialize projector tables, bare D coefficients and USPP overlaps.
     * @param[in,out] cell Pseudopotentials, including USPP angular-channel limits to adjust.
     * @param[in] rho_basis Density basis for G=0 augmentation integrals.
     */
    void init_vnl(UnitCell &cell, const ModulePW::PW_Basis *rho_basis);

    //! Rescale radial data to volume omega_in in cubic Bohr.
    void rescale_vnl(const double &omega_in);

    //! Extend the radial table for four-point interpolation up to qmax in inverse Bohr.
    void ensure_vnl_range(const UnitCell &cell, const double &qmax);

    //! Check the interpolation index against tab and, if derivative is true, tab_dq.
    void check_vnl_index(const double &index, const bool &derivative) const;

    //! Return the radial momentum-grid spacing in inverse Bohr.
    double table_dq() const { return table_dq_; }

    /**
     * @brief Build nonlocal projectors at G+k+A-v_I.
     * @param[in] ctx Device context.
     * @param[in] ucell Cell geometry and atom ordering.
     * @param[in] ik Local k-point index.
     * @param[in] vecpot RT-TDDFT momentum shift in Hartree atomic units.
     * @param[in] ion_vel RT-TDDFT nuclear velocities in type-major order and Hartree atomic units.
     * @param[out] vkb_in Projectors stored as vkb_in[projector * lead_dim + ig].
     * @param[in] lead_dim Projector stride, at least the local plane-wave count.
     */
    template <typename FPTYPE, typename Device>
    void getvnl(
        Device *ctx,
        const UnitCell &ucell,
        const int &ik,
        const ModuleBase::Vector3<double> &vecpot,
        const std::vector<ModuleBase::Vector3<double>> &ion_vel,
        std::complex<FPTYPE> *vkb_in,
        int lead_dim) const;

    /** @brief Build shifted projectors using explicit native-device atomic phases. */
    template <typename FPTYPE, typename Device>
    void build_vnl(Device *ctx, const UnitCell &cell, const int &ik,
                   const ModuleBase::Vector3<double> &a,
                   const std::vector<ModuleBase::Vector3<double>> &vel,
                   const std::complex<FPTYPE> *sk, std::complex<FPTYPE> *beta, const int &ld) const;

    //! Rebuild the radial first-derivative table by Bessel quadrature.
    void initgradq_vnl(const UnitCell &cell);
    //! Return the radial-table revision.
    size_t table_version() const { return table_version_; }
    //! Refresh the first-derivative table when outdated.
    void ensure_grad_table(const UnitCell &cell);
    //! Prepare radial second derivatives for RT-TDDFT ionic inertia.
    void ensure_hess(const UnitCell &cell);

  private:
    double table_dq_ = 0.0;       // Radial momentum-grid spacing in inverse Bohr.
    size_t table_version_ = 1;    // Current radial-table revision.
    size_t gradient_version_ = 0; // Revision of tab_dq.
    size_t hess_version_ = 0;     // Revision of tab_d2.

  public:
    int nhm = 0;    // Maximum projectors per atom, including angular components.
    int nbetam = 0; // Maximum radial projector channels per type.
    int lmaxq = 0;  // Augmentation angular-channel count: 2*lmaxkb+1.

    ModuleBase::matrix indv;   // Radial channel for each (type, projector).
    ModuleBase::matrix nhtol;  // Angular momentum l for each (type, projector).
    ModuleBase::matrix nhtolm; // Real-harmonic index l*l+m, m in [0, 2*l], for each (type, projector).
    ModuleBase::matrix nhtoj;  // Total angular momentum j for each (type, projector).

    ModuleBase::realArray dvan;       // Bare D(type, projector, projector).
    ModuleBase::ComplexArray dvan_so; // Bare SOC D(spin block, type, projector, projector); added by zhengdy-soc

    ModuleBase::realArray tab;     // Radial projectors (type, channel, momentum), normalized by 4*pi/sqrt(volume).
    ModuleBase::realArray tab_at;  // Atomic radial transforms (type, orbital, momentum).
    ModuleBase::realArray tab_dq;  // First momentum derivatives of tab.
    ModuleBase::realArray tab_d2;  // Second momentum derivatives for RT-TDDFT ionic inertia.
    ModuleBase::realArray rad_mom; // Five origin coefficients and rmax per radial channel.

    ModuleBase::realArray deeq;                // Effective D(spin, atom, projector, projector).
    bool multi_proj = false;                   // Nonzero off-diagonal scalar D coefficients.
    float *s_deeq = nullptr;                   // Single-precision deeq copy.
    double *d_deeq = nullptr;                  // Double-precision deeq device copy or CPU alias.
    ModuleBase::ComplexArray deeq_nc;          // Spinor D(spin block, atom, projector, projector).
    std::complex<float> *c_deeq_nc = nullptr;  // Single-precision deeq_nc copy.
    std::complex<double> *z_deeq_nc = nullptr; // Double-precision deeq_nc device copy or CPU alias.

    // liuyu add 2023-10-03
    // USPP augmentation and overlap data.
    std::vector<int> indv_ijkb0;    // First global projector index per atom.
    ModuleBase::realArray qq_nt;    // Augmentation integrals (type, projector, projector).
    ModuleBase::ComplexArray qq_so; // Spinor augmentation integrals (type, spin block, projector, projector).
    ModuleBase::realArray ap;       // Real-harmonic product coefficients (LM, lm1, lm2).
    ModuleBase::IntArray lpx;       // Allowed LM count per harmonic pair (lm1, lm2).
    ModuleBase::IntArray lpl;       // Allowed LM indices per harmonic pair (lm1, lm2).
    ModuleBase::realArray qrad;     // Augmentation radial transforms (type, L, packed radial pair, momentum).

    float *s_qq_nt = nullptr;                // Single-precision qq_nt copy.
    double *d_qq_nt = nullptr;               // Double-precision qq_nt device copy or CPU alias.
    std::complex<float> *c_qq_so = nullptr;  // Single-precision qq_so copy.
    std::complex<double> *z_qq_so = nullptr; // Double-precision qq_so device copy or CPU alias.

    mutable ModuleBase::ComplexMatrix vkb; // CPU projectors (global projector, local plane wave).
    Structure_Factor *psf = nullptr;       // Non-owning structure-factor provider.

    int vkbnc = 0; // Projector leading dimension npwx.

    //! Print the radial projector table.
    void print_vnl(std::ofstream &ofs);

    /**
     * @brief Assemble augmentation Q(G) from radial tables and real harmonics.
     * @param[in] ng Number of reciprocal vectors.
     * @param[in] ih First angular-resolved projector index within the type.
     * @param[in] jh Second angular-resolved projector index within the type.
     * @param[in] itype Atomic type index.
     * @param[in] qnorm Magnitudes in inverse Bohr, length ng.
     * @param[in] ylm Real harmonics (LM, ig).
     * @param[out] qg Q(G) without atomic-position phases, length ng.
     */
    void radial_fft_q(
        const int ng,
        const int ih,
        const int jh,
        const int itype,
        const double *qnorm,
        const ModuleBase::matrix ylm,
        std::complex<double> *qg) const;

    /**
     * @brief Assemble augmentation Q(G) with raw CPU arrays in float or double precision.
     * @param[in] ctx CPU device context.
     * @param[in] ng Number of reciprocal vectors.
     * @param[in] ih First angular-resolved projector index within the type.
     * @param[in] jh Second angular-resolved projector index within the type.
     * @param[in] itype Atomic type index.
     * @param[in] qnorm Magnitudes in inverse Bohr, length ng.
     * @param[in] ylm Real harmonics stored as ylm[LM * ng + ig].
     * @param[out] qg Q(G) without atomic-position phases, length ng.
     */
    template <typename FPTYPE, typename Device>
    void radial_fft_q(
        Device *ctx,
        const int ng,
        const int ih,
        const int jh,
        const int itype,
        const FPTYPE *qnorm,
        const FPTYPE *ylm,
        std::complex<FPTYPE> *qg) const;

    /**
     * @brief Assemble effective D coefficients, including USPP potential corrections.
     * @param[in] veff Effective potential in Ry (spin component, local grid point).
     * @param[in] rho_basis Density basis and FFT grids.
     * @param[in] cell Atomic geometry and pseudopotentials.
     */
    void cal_effective_D(const ModuleBase::matrix &veff, const ModulePW::PW_Basis *rho_basis, UnitCell &cell);

    //! Return projector angular momenta.
    template <typename FPTYPE>
    FPTYPE *get_nhtol_data() const;

    //! Return projector real-harmonic indices.
    template <typename FPTYPE>
    FPTYPE *get_nhtolm_data() const;

    //! Return the projector-to-radial-channel map.
    template <typename FPTYPE>
    FPTYPE *get_indv_data() const;

    //! Return the radial projector table.
    template <typename FPTYPE>
    FPTYPE *get_tab_data() const;

    //! Return scalar effective D coefficients.
    template <typename FPTYPE>
    FPTYPE *get_deeq_data() const;

    //! Return type-resolved augmentation integrals.
    template <typename FPTYPE>
    FPTYPE *get_qq_nt_data() const;

    //! Return spinor augmentation integrals.
    template <typename FPTYPE>
    std::complex<FPTYPE> *get_qq_so_data() const;

    //! Return the internal projector buffer.
    template <typename FPTYPE>
    std::complex<FPTYPE> *get_vkb_data() const;

    //! Return spinor effective D coefficients.
    template <typename FPTYPE>
    std::complex<FPTYPE> *get_deeq_nc_data() const;

    //! Release owned buffers and projector caches.
    void release_memory();

  private:
    // Reusable projector buffers, indexed by precision: 0 for float, 1 for double.
    mutable std::array<ct::Tensor, 2> vnl_ylm_;
    mutable std::array<ct::Tensor, 2> vnl_radial_;
    mutable std::array<ct::Tensor, 2> vnl_momentum_;
    mutable std::array<ct::Tensor, 2> vnl_structure_;
    mutable std::array<ct::Tensor, 2> vnl_vel_;     // RT-TDDFT nuclear velocities.
    mutable std::array<ct::Tensor, 2> vnl_mom_;     // Small-momentum expansion data.
    mutable std::array<size_t, 2> mom_ver_{{0, 0}}; // Origin-data revision per precision.

    // Projector channel and atom indices for RT-TDDFT traveling pseudopotentials.
    mutable ct::Tensor vnl_meta_;
    mutable std::vector<int> vnl_host_meta_;

    // Atom counts by type for zero-velocity projectors.
    mutable ct::Tensor vnl_atom_counts_;
    mutable std::vector<int> vnl_host_atom_counts_;
    mutable std::vector<ModuleBase::Vector3<double>> vnl_host_momentum_; // Cartesian G+k+A in units of tpiba.

    bool memory_released = false; // Whether raw-pointer buffers have been released.

    // Single-precision index, radial-table and projector buffers.
    float *s_nhtol = nullptr;
    float *s_nhtolm = nullptr;
    float *s_indv = nullptr;
    float *s_tab = nullptr;
    std::complex<float> *c_vkb = nullptr;

    // Double-precision device copies or CPU aliases; DSP uses separate projector storage.
    double *d_nhtol = nullptr;
    double *d_nhtolm = nullptr;
    double *d_indv = nullptr;
    double *d_tab = nullptr;
    std::complex<double> *z_vkb = nullptr;

    const ModulePW::PW_Basis_K *wfcpw = nullptr; // Non-owning wavefunction basis.

    Soc soc; // Harmonic rotations and spin-orbit coupling coefficients.

    double omega_old = 0;  // Volume used to normalize radial tables.
    bool use_gpu_ = false; // Use GPU buffers and transfers.

    //! Compute radial projectors and origin coefficients by quadrature.
    void fill_vnl_table(const UnitCell &cell);

    //! Synchronize radial-table copies and CPU aliases.
    void sync_vnl_table();

    //! Compute USPP radial transforms in qrad with normalization 4*pi/volume.
    void compute_qrad(UnitCell &cell);

    /**
     * @brief Store integrals of the effective potential against USPP augmentation functions in deeq.
     * @param[in] veff Effective potential in Ry (spin component, local grid point).
     * @param[in] rho_basis Density basis and FFT grids.
     * @param[in] cell Atomic geometry and pseudopotentials.
     */
    void newq(const ModuleBase::matrix &veff, const ModulePW::PW_Basis *rho_basis, UnitCell &cell);

    //! Assemble SOC D coefficients for atom iat from bare terms and spin-rotated USPP integrals.
    void newd_so(const int &iat, UnitCell &cell);

    //! Assemble spinor D coefficients for atom iat from bare terms and scalar and magnetic USPP integrals.
    void newd_nc(const int &iat, UnitCell &cell);
};

#endif // VNL_IN_PW_H
