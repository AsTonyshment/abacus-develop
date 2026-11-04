#include "source_cell/stru_fac.h"
#include "source_base/constants.h"
#include "source_base/global_function.h"
#include "source_base/libm/libm.h"
#include "source_base/math_bspline.h"
#include "source_base/memory_recorder.h"
#include "source_base/timer.h"
#include "source_io/module_parameter/parameter.h"

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

Structure_Factor::Structure_Factor()
{
    // LCAO basis doesn't support GPU acceleration on this function currently.
    if (PARAM.inp.basis_type == "pw")
    {
        this->device = PARAM.inp.device;
    }
}

Structure_Factor::~Structure_Factor()
{
    if (device == "gpu")
    {
        delmem_cd_op()(this->c_eigts1);
        delmem_cd_op()(this->c_eigts2);
        delmem_cd_op()(this->c_eigts3);
        delmem_zd_op()(this->z_eigts1);
        delmem_zd_op()(this->z_eigts2);
        delmem_zd_op()(this->z_eigts3);
    }
    else
    {
        delmem_ch_op()(this->c_eigts1);
        delmem_ch_op()(this->c_eigts2);
        delmem_ch_op()(this->c_eigts3);
        // There's no need to delete double precision pointers while in a CPU environment.
    }
}

// called in input.cpp
void Structure_Factor::set(const ModulePW::PW_Basis *rho_basis_in, const int &nbspline_in)
{
    ModuleBase::TITLE("Structure_Factor", "set");
    this->rho_basis = rho_basis_in;
    this->nbspline = nbspline_in;
    return;
}

// Peize Lin optimize and add OpenMP 2021.04.01
//  Calculate structure factor
void Structure_Factor::setup(
    const UnitCell *Ucell,
    const Parallel_Grid &pgrid,
    const ModulePW::PW_Basis *rho_basis,
    const bool has_float_data)
{
    ModuleBase::TITLE("Structure_Factor", "setup");
    ModuleBase::timer::start("Structure_Factor", "setup");

    // Keep full geometry initialization together while allowing RT callers to refresh its two parts independently.
    cal_sf(*Ucell, pgrid, *rho_basis);
    cal_phase(*Ucell, *rho_basis, has_float_data);
    ModuleBase::timer::end("Structure_Factor", "setup");
}

void Structure_Factor::cal_sf(const UnitCell &cell, const Parallel_Grid &grid, const ModulePW::PW_Basis &basis)
{
    // Sum atomic phases by element for the local ionic potential and nonlinear core density.
    ModuleBase::timer::start("Structure_Factor", "cal_sf");
    this->ucell = &cell;
    const std::complex<double> phase = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI;
    strucFac.create(cell.ntype, basis.npw);
    ModuleBase::Memory::record("SF::strucFac", sizeof(std::complex<double>) * cell.ntype * basis.npw);
    if (nbspline > 0)
    {
        nbspline = ((nbspline + 1) / 2) * 2;
        bspline_sf(nbspline, &cell, grid, &basis);
    }
    else
    {
        for (int it = 0; it < cell.ntype; ++it)
        {
            const int na = cell.atoms[it].na;
            const ModuleBase::Vector3<double> *tau = cell.atoms[it].tau.data();
            const int npw = basis.npw;
            const ModuleBase::Vector3<double> *gcar = basis.gcar;
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int ig = 0; ig < npw; ++ig)
            {
                const ModuleBase::Vector3<double> g = gcar[ig];
                std::complex<double> sum = ModuleBase::ZERO;
                for (int ia = 0; ia < na; ++ia)
                {
                    sum += ModuleBase::libm::exp(phase * (g * tau[ia]));
                }
                strucFac(it, ig) = sum;
            }
        }
    }
    ModuleBase::timer::end("Structure_Factor", "cal_sf");
}

void Structure_Factor::cal_phase(const UnitCell &cell, const ModulePW::PW_Basis &basis, bool has_float)
{
    // Refresh per-atom projector phases independently so operator-time bindings retain endpoint structure factors.
    ModuleBase::timer::start("Structure_Factor", "cal_phase");
    this->ucell = &cell;
    const std::complex<double> phase = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI;
    eigts1.create(cell.nat, 2 * basis.nx + 1);
    eigts2.create(cell.nat, 2 * basis.ny + 1);
    eigts3.create(cell.nat, 2 * basis.nz + 1);
    const int elements = eigts1.size + eigts2.size + eigts3.size;
    ModuleBase::Memory::record("SF::eigts123", sizeof(std::complex<double>) * elements);

    int iat = 0;
    for (int it = 0; it < cell.ntype; ++it)
    {
        for (int ia = 0; ia < cell.atoms[it].na; ++ia, ++iat)
        {
            const ModuleBase::Vector3<double> gtau = cell.G * cell.atoms[it].tau[ia];
#ifdef _OPENMP
#pragma omp parallel
            {
#pragma omp for schedule(static, 16)
#endif
                for (int n = -basis.nx; n <= basis.nx; ++n)
                {
                    const double arg = n * gtau.x;
                    eigts1(iat, n + basis.nx) = ModuleBase::libm::exp(phase * arg);
                }
#ifdef _OPENMP
#pragma omp for schedule(static, 16)
#endif
                for (int n = -basis.ny; n <= basis.ny; ++n)
                {
                    const double arg = n * gtau.y;
                    eigts2(iat, n + basis.ny) = ModuleBase::libm::exp(phase * arg);
                }
#ifdef _OPENMP
#pragma omp for schedule(static, 16)
#endif
                for (int n = -basis.nz; n <= basis.nz; ++n)
                {
                    const double arg = n * gtau.z;
                    eigts3(iat, n + basis.nz) = ModuleBase::libm::exp(phase * arg);
                }
#ifdef _OPENMP
            }
#endif
        }
    }
    sync_phase(has_float);
    ModuleBase::timer::end("Structure_Factor", "cal_phase");
}

void Structure_Factor::sync_phase(bool has_float_data)
{
    ModuleBase::timer::start("Structure_Factor", "sync_phase");
    const ModuleBase::ComplexMatrix *phases[3] = {&eigts1, &eigts2, &eigts3};
    std::complex<float> **floats[3] = {&c_eigts1, &c_eigts2, &c_eigts3};
    std::complex<double> **doubles[3] = {&z_eigts1, &z_eigts2, &z_eigts3};
    for (int axis = 0; axis < 3; ++axis)
    {
        const int size = phases[axis]->size;
        // Reuse conversion and device buffers across ionic updates while the phase-table dimensions stay fixed.
        const bool resized = phase_size_[axis] != size;
        if (resized)
        {
            if (device == "gpu")
            {
                delmem_cd_op()(*floats[axis]);
                delmem_zd_op()(*doubles[axis]);
                *doubles[axis] = nullptr;
            }
            else
            {
                delmem_ch_op()(*floats[axis]);
            }
            *floats[axis] = nullptr;
            phase_size_[axis] = size;
        }
        if (device == "gpu")
        {
            if (*doubles[axis] == nullptr)
            {
                resmem_zd_op()(*doubles[axis], size);
            }
            syncmem_z2z_h2d_op()(*doubles[axis], phases[axis]->c, size);
            if (has_float_data)
            {
                if (*floats[axis] == nullptr)
                {
                    resmem_cd_op()(*floats[axis], size);
                }
                castmem_z2c_h2d_op()(*floats[axis], phases[axis]->c, size);
            }
        }
        else
        {
            // CPU double precision aliases the matrix storage; float and device data require refreshed copies.
            *doubles[axis] = phases[axis]->c;
            if (has_float_data)
            {
                if (*floats[axis] == nullptr)
                {
                    resmem_ch_op()(*floats[axis], size);
                }
                castmem_z2c_h2h_op()(*floats[axis], phases[axis]->c, size);
            }
        }
    }
    ModuleBase::timer::end("Structure_Factor", "sync_phase");
}

//
// DESCRIPTION:
//    Calculate structure factor with Cardinal B-spline interpolation
//    Ref: J. Chem. Phys. 103, 8577 (1995)
//    qianrui create 2021-9-17
// INPUT LIST:
//    norder: the order of Cardinal B-spline base functions
// FURTHER OPTIMIZATION:
//    1. Use "r2c" fft
//
void Structure_Factor::bspline_sf(
    const int norder,
    const UnitCell *Ucell,
    const Parallel_Grid &pgrid,
    const ModulePW::PW_Basis *rho_basis)
{
    (void)pgrid;
    std::vector<double> tmpr(rho_basis->nrxx);
    std::vector<std::complex<double>> b1(rho_basis->nx);
    std::vector<std::complex<double>> b2(rho_basis->ny);
    std::vector<std::complex<double>> b3(rho_basis->nz);
    const int nplane = rho_basis->nplane;
    const int startz = rho_basis->startz_current;

    // Each rank owns the same atoms; populate only its local FFT z slab.
    for (int it = 0; it < Ucell->ntype; it++)
    {
        const int na = Ucell->atoms[it].na;
        const ModuleBase::Vector3<double> *const taud = Ucell->atoms[it].taud.data();
        ModuleBase::GlobalFunc::ZEROS(tmpr.data(), rho_basis->nrxx);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int ia = 0; ia < na; ++ia)
        {
            const double gridx = taud[ia].x * rho_basis->nx;
            const double gridy = taud[ia].y * rho_basis->ny;
            const double gridz = taud[ia].z * rho_basis->nz;
            const double dx = gridx - floor(gridx);
            const double dy = gridy - floor(gridy);
            const double dz = gridz - floor(gridz);
            // I'm not sure if there is a mod function for double data

            ModuleBase::Bspline bsx, bsy, bsz;
            bsx.init(norder, 1, 0);
            bsy.init(norder, 1, 0);
            bsz.init(norder, 1, 0);
            bsx.getbspline(dx);
            bsy.getbspline(dy);
            bsz.getbspline(dz);

            for (int iz = 0; iz <= norder; ++iz)
            {
                const int icz = int(rho_basis->nz * 10 - iz + floor(gridz)) % rho_basis->nz;
                if (icz < startz || icz >= startz + nplane)
                {
                    continue;
                }
                const int local_z = icz - startz;
                for (int iy = 0; iy <= norder; ++iy)
                {
                    const int icy = int(rho_basis->ny * 10 - iy + floor(gridy)) % rho_basis->ny;
                    for (int ix = 0; ix <= norder; ++ix)
                    {
                        const int icx = int(rho_basis->nx * 10 - ix + floor(gridx)) % rho_basis->nx;
#ifdef _OPENMP
#pragma omp atomic
#endif
                        tmpr[(icx * rho_basis->ny + icy) * nplane + local_z] +=
                            bsz.bezier_ele(iz) * bsy.bezier_ele(iy) * bsx.bezier_ele(ix);
                    }
                }
            }
        }

        // It should be optimized with r2c
        rho_basis->real2recip(tmpr.data(), &strucFac(it, 0));
        this->bsplinecoef(b1.data(), b2.data(), b3.data(), rho_basis->nx, rho_basis->ny, rho_basis->nz, norder);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 128)
#endif
        for (int ig = 0; ig < rho_basis->npw; ++ig)
        {
            const int idx = int(rho_basis->gdirect[ig].x + 0.1 + rho_basis->nx) % rho_basis->nx;
            const int idy = int(rho_basis->gdirect[ig].y + 0.1 + rho_basis->ny) % rho_basis->ny;
            const int idz = int(rho_basis->gdirect[ig].z + 0.1 + rho_basis->nz) % rho_basis->nz;
            strucFac(it, ig) *= (b1[idx] * b2[idy] * b3[idz] * double(rho_basis->nxyz));
        }
    }

    return;
}

void Structure_Factor::bsplinecoef(
    std::complex<double> *b1,
    std::complex<double> *b2,
    std::complex<double> *b3,
    const int nx,
    const int ny,
    const int nz,
    const int norder)
{
    const std::complex<double> ci_tpi = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI;
    ModuleBase::Bspline bsp;
    bsp.init(norder, 1, 0);
    bsp.getbspline(1.0);
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp for schedule(static, 16)
#endif
        for (int ix = 0; ix < nx; ++ix)
        {
            std::complex<double> fracx = 0;
            for (int io = 0; io < norder; ++io)
            {
                fracx += bsp.bezier_ele(io) * ModuleBase::libm::exp(ci_tpi * double(ix) / double(nx) * double(io));
            }
            b1[ix] = ModuleBase::libm::exp(ci_tpi * double(norder * ix) / double(nx)) / fracx;
        }
#ifdef _OPENMP
#pragma omp for schedule(static, 16)
#endif
        for (int iy = 0; iy < ny; ++iy)
        {
            std::complex<double> fracy = 0;
            for (int io = 0; io < norder; ++io)
            {
                fracy += bsp.bezier_ele(io) * ModuleBase::libm::exp(ci_tpi * double(iy) / double(ny) * double(io));
            }
            b2[iy] = ModuleBase::libm::exp(ci_tpi * double(norder * iy) / double(ny)) / fracy;
        }
#ifdef _OPENMP
#pragma omp for schedule(static, 16)
#endif
        for (int iz = 0; iz < nz; ++iz)
        {
            std::complex<double> fracz = 0;
            for (int io = 0; io < norder; ++io)
            {
                fracz += bsp.bezier_ele(io) * ModuleBase::libm::exp(ci_tpi * double(iz) / double(nz) * double(io));
            }
            b3[iz] = ModuleBase::libm::exp(ci_tpi * double(norder * iz) / double(nz)) / fracz;
        }
#ifdef _OPENMP
    }
#endif
}

template <>
std::complex<float> *Structure_Factor::get_eigts1_data() const
{
    return this->c_eigts1;
}
template <>
std::complex<double> *Structure_Factor::get_eigts1_data() const
{
    return this->z_eigts1;
}

template <>
std::complex<float> *Structure_Factor::get_eigts2_data() const
{
    return this->c_eigts2;
}
template <>
std::complex<double> *Structure_Factor::get_eigts2_data() const
{
    return this->z_eigts2;
}

template <>
std::complex<float> *Structure_Factor::get_eigts3_data() const
{
    return this->c_eigts3;
}
template <>
std::complex<double> *Structure_Factor::get_eigts3_data() const
{
    return this->z_eigts3;
}
