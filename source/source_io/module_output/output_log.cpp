#include "output_log.h"

#include "source_estate/fp_energy.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/constants.h"
#include "source_base/formatter.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_reduce.h"
#include "source_base/parallel_comm.h"
#include <cmath>
#include <iomanip>

#ifdef __MPI
#include <mpi.h>
#endif

namespace ModuleIO
{
void output_convergence_after_scf(const bool &convergence, double& energy, std::ofstream& ofs_running)
{
    if (convergence)
    {
        ofs_running << " #SCF IS CONVERGED#" << std::endl;
//      ofs_running << " final etot is " << std::setprecision(11) << energy * ModuleBase::Ry_to_eV << " eV" << std::endl;
        ofs_running << " #TOTAL ENERGY# " << std::setprecision(11) << energy * ModuleBase::Ry_to_eV << " eV" << std::endl;
    }
    else
    {
        ofs_running << " !!SCF IS NOT CONVERGED!!" << std::endl;
        std::cout << " !!SCF IS NOT CONVERGED!!" << std::endl;
    }
}

void output_after_relax(bool conv_ion, bool conv_esolver, std::ofstream& ofs_running)
{
    if (conv_ion && !conv_esolver)
    {
        std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        std::cout << " Relaxation is converged, but the SCF is unconverged! The results are unreliable. " << std::endl;
        std::cout << " It is suggested to increase the maximum SCF step and/or perform the relaxation again."
                  << std::endl;
        std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        ofs_running << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        ofs_running << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        ofs_running << "\n Relaxation is converged, but the SCF is unconverged! The results are unreliable.. "
                    << std::endl;
        ofs_running << "\n It is suggested to increase the maximum SCF step and/or perform the relaxation again. "
                    << std::endl;
        ofs_running << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        ofs_running << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    }
}

void output_efermi(const bool &convergence, const elecstate::Efermi& eferm, std::ofstream& ofs_running)
{
    if (convergence && PARAM.inp.out_level != "m")
    {
        if (eferm.two_efermi)
        {
            ofs_running << " #FERMI ENERGY# "
                        << std::setprecision(11) << eferm.ef_up * ModuleBase::Ry_to_eV
                        << " " << eferm.ef_dw * ModuleBase::Ry_to_eV
                        << " eV    # two fermi energies (up, down)" << std::endl;
        }
        else
        {
            ofs_running << " #FERMI ENERGY# "
                        << std::setprecision(11) << eferm.ef * ModuleBase::Ry_to_eV
                        << " eV" << std::endl;
        }
    }
}

void output_vacuum_level(const UnitCell* ucell,
                         const double* const* rho,
                         const double* v_elecstat,
                         const int& nx,
                         const int& ny,
                         const int& nz,
                         const int& nxyz,
                         const int& nrxx,
                         const int& nplane,
                         const int& startz_current,
                         std::ofstream& ofs_running)
{
    // determine the vacuum direction
    double vacuum[3] = {0.0};
    for (int dir = 0; dir < 3; dir++)
    {
        std::vector<double> pos;
        for (int it = 0; it < ucell->ntype; ++it)
        {
            for (int ia = 0; ia < ucell->atoms[it].na; ++ia)
            {
                pos.push_back(ucell->atoms[it].taud[ia][dir]);
            }
        }

        std::sort(pos.begin(), pos.end());
        for (int i = 1; i < pos.size(); i++)
        {
            vacuum[dir] = std::max(vacuum[dir], pos[i] - pos[i - 1]);
        }

        // consider the periodic boundary condition
        vacuum[dir] = std::max(vacuum[dir], pos[0] + 1 - pos[pos.size() - 1]);
    }

    // we assume that the cell is a cuboid
    // get the direction with the largest vacuum
    int direction = 2;
    vacuum[0] *= ucell->latvec.e11;
    vacuum[1] *= ucell->latvec.e22;
    vacuum[2] *= ucell->latvec.e33;
    if (vacuum[0] > vacuum[2])
    {
        direction = 0;
    }
    if (vacuum[1] > vacuum[direction])
    {
        direction = 1;
    }

    int length = nz;
    if (direction == 0)
    {
        length = nx;
    }
    else if (direction == 1)
    {
        length = ny;
    }

    // get the average along the direction in real space
    auto average = [](const int& ny,
                      const int& nxyz,
                      const int& nrxx,
                      const int& nplane,
                      const int& startz_current,
                      const int& direction,
                      const int& length,
                      const double* v,
                      double* ave,
                      bool abs) {
        for (int ir = 0; ir < nrxx; ++ir)
        {
            int index = 0;
            if (direction == 0)
            {
                index = ir / (ny * nplane);
            }
            else if (direction == 1)
            {
                int i = ir / (ny * nplane);
                index = ir / nplane - i * ny;
            }
            else if (direction == 2)
            {
                index = ir % nplane + startz_current;
            }

            double value = abs ? std::fabs(v[ir]) : v[ir];

            ave[index] += value;
        }

#ifdef __MPI
        Parallel_Reduce::reduce_pool(ave, length);
#endif

        int surface = nxyz / length;
        for (int i = 0; i < length; ++i)
        {
            ave[i] /= surface;
        }
    };

    // average charge density along direction
    std::vector<double> totchg(nrxx, 0.0);
    for (int ir = 0; ir < nrxx; ++ir)
    {
        totchg[ir] = rho[0][ir];
    }
    if (PARAM.inp.nspin == 2)
    {
        for (int ir = 0; ir < nrxx; ++ir)
        {
            totchg[ir] += rho[1][ir];
        }
    }

    std::vector<double> ave(length, 0.0);
    average(ny, nxyz, nrxx, nplane, startz_current, direction, length, totchg.data(), ave.data(), true);

    // set vacuum to be the point in space where the electronic charge density is the minimum
    // get the index corresponding to the minimum charge density
    int min_index = 0;
    double min_value = 1e9;
    double windows[7] = {0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1};
    for (int i = 0; i < length; i++)
    {
        double sum = 0;
        int temp = i - 3 + length;
        // use a sliding average to smoothen in charge density
        for (int win = 0; win < 7; win++)
        {
            int index = (temp + win) % length;
            sum += ave[index] * windows[win];
        }

        if (sum < min_value)
        {
            min_value = sum;
            min_index = i;
        }
    }

    // average electrostatic potential along direction
    ave.assign(ave.size(), 0.0);
    average(ny, nxyz, nrxx, nplane, startz_current, direction, length, v_elecstat, ave.data(), false);

    // get the vacuum level
    double vacuum_level = ave[min_index] * ModuleBase::Ry_to_eV;
    ofs_running << "The vacuum level is " << vacuum_level << " eV" << std::endl;
}

void out_force_pw(
    std::ofstream *log,
    const UnitCell &cell,
    const PWForceResult &result,
    const bool &details)
{
    // Report the net-force treatment already applied by the physical force evaluator.
    if (!result.shifted)
    {
        *log << " Notice: Atomic forces are not adjusted to enforce zero net force for gate fields, electric fields, or RT-TDDFT!\n";
    }

    // Completed RT force results carry the position and direct field subtotal.
    const bool rt_force = result.pos_field.nr > 0;

    // Print each available contribution once in eV/Angstrom.
    if (details)
    {
        print_force(*log, cell, "LOCAL    FORCE (eV/Angstrom)", result.local, false);
        print_force(*log, cell, "NONLOCAL FORCE (eV/Angstrom)", result.nonlocal, false);
        print_force(*log, cell, "NLCC     FORCE (eV/Angstrom)", result.core, false);
        print_force(*log, cell, "ION      FORCE (eV/Angstrom)", result.ion, false);
        print_force(*log, cell, "SCC      FORCE (eV/Angstrom)", result.scc, false);
        if (rt_force)
        {
            print_force(*log, cell, "RT FIELD FORCE (eV/Angstrom)", result.field, false);
        }

        const ModuleBase::matrix *extra[] = {&result.vdw, &result.static_field, &result.gate, &result.solvent, &result.onsite, &result.pcc};
        const char *labels[] = {
            "VDW      FORCE (eV/Angstrom)",
            "EFIELD   FORCE (eV/Angstrom)",
            "GATEFIELD FORCE (eV/Angstrom)",
            "IMP_SOL  FORCE (eV/Angstrom)",
            "ONSITE_PROJ FORCE (eV/Angstrom)",
            "PCC      FORCE (eV/Angstrom)"
        };
        for (int i = 0; i < 6; ++i)
        {
            if (extra[i]->nr > 0)
            {
                print_force(*log, cell, labels[i], *extra[i], false);
            }
        }
    }

    // Report RT force balances before the complete force shared with MD.
    if (rt_force)
    {
        print_force(*log, cell, "POSITION+FIELD FORCE (eV/Angstrom)", result.pos_field, false);
        print_force(*log, cell, "NUCLEAR VELOCITY FORCE (eV/Angstrom)", result.vel_force, false);
        print_force(*log, cell, "CONSTRAINT REACTION (eV/Angstrom)", result.reaction, false);
    }
    print_force(*log, cell, "TOTAL-FORCE (eV/Angstrom)", result.total, false);
}

void print_force(
    std::ofstream &ofs_running,
    const UnitCell &cell,
    const std::string &name,
    const ModuleBase::matrix &force,
    bool ry)
{
    // Apply the display cutoff in Ry/Bohr before converting to the requested output unit.
    constexpr double output_acc = 1.0e-8;
    double fac = 1.0;
    if (!ry)
    {
        fac = ModuleBase::Ry_to_eV / ModuleBase::BOHR_TO_A;
    }

    std::vector<std::string> atom_label;
    std::vector<double> force_x;
    std::vector<double> force_y;
    std::vector<double> force_z;
    atom_label.reserve(cell.nat);
    force_x.reserve(cell.nat);
    force_y.reserve(cell.nat);
    force_z.reserve(cell.nat);

    std::vector<std::string> titles({"Atoms", "Force_x", "Force_y", "Force_z"});
    int iat = 0;
    for (int it = 0; it < cell.ntype; ++it)
    {
        for (int ia = 0; ia < cell.atoms[it].na; ++ia)
        {
            const std::string atom_labels = cell.atoms[it].label + std::to_string(ia + 1);
            const double fx = std::abs(force(iat, 0)) > output_acc ? force(iat, 0) * fac : 0.0;
            const double fy = std::abs(force(iat, 1)) > output_acc ? force(iat, 1) * fac : 0.0;
            const double fz = std::abs(force(iat, 2)) > output_acc ? force(iat, 2) * fac : 0.0;
            atom_label.push_back(atom_labels);
            force_x.push_back(fx);
            force_y.push_back(fy);
            force_z.push_back(fz);
            ++iat;
        }
    }
    FmtTable fmt(
        /*titles=*/titles,
        /*nrows=*/atom_label.size(),
        /*formats=*/{"%8s", "%20.10f", "%20.10f", "%20.10f"},
        /*indent*/ 1,
        {FmtTable::Align::RIGHT, FmtTable::Align::RIGHT});
    fmt << atom_label << force_x << force_y << force_z;
    const std::string table = fmt.str();
    const std::string heading = "\n #" + name + "#\n";

    // Running logs and stdout follow the process-wide output setup.
    ofs_running << heading << table;
    if (PARAM.inp.test_force)
    {
        std::cout << heading << table;
    }
}

void print_stress(const std::string& name, const ModuleBase::matrix& scs, 
  const bool screen, const bool ry, std::ofstream &ofs)
{
    double unit_transform = 1;
    std::string title = name;
    std::string unit = "";
    if (ry)
    {
        title += " (a.u.)";
        unit = " a.u.";
    }
    else
    {
        title += " (kbar)";
        unit = " kbar";
        unit_transform = ModuleBase::RYDBERG_SI / pow(ModuleBase::BOHR_RADIUS_SI, 3) * 1.0e-8;
    }
    std::vector<double> stress_x;
    std::vector<double> stress_y;
    std::vector<double> stress_z;
    std::string table;

    ofs << "\n #" << title << "#" << std::endl;

    std::vector<std::string> titles({"Stress_x", "Stress_y", "Stress_z"});
    for (int i = 0; i < 3; i++)
    {
        double sx = scs(i, 0) * unit_transform;
        double sy = scs(i, 1) * unit_transform;
        double sz = scs(i, 2) * unit_transform;
        stress_x.push_back(sx);
        stress_y.push_back(sy);
        stress_z.push_back(sz);
    }

    double pressure = (scs(0, 0) + scs(1, 1) + scs(2, 2)) / 3.0 * unit_transform;

    FmtTable fmt(/*titles=*/titles, 
                 /*nrows=*/3, 
                 /*formats=*/{"%20.10f", "%20.10f", "%20.10f"}, 
                 /*indent*/1,
                 {FmtTable::Align::RIGHT,FmtTable::Align::RIGHT});


    fmt << stress_x << stress_y << stress_z;
    table = fmt.str();
    ofs << table;
    if (name == "TOTAL-STRESS" && PARAM.inp.calculation != "md")
    {
        ofs << " #TOTAL-PRESSURE# (EXCLUDE KINETIC PART OF IONS): " << std::fixed 
                             << std::setprecision(6) << pressure << unit
                             << std::endl;
    }
    if (screen)
    {
        std::cout << table;
        if (name == "TOTAL-STRESS")
        {
            std::cout << " TOTAL-PRESSURE (EXCLUDE KINETIC PART OF IONS): " << std::fixed 
                      << std::setprecision(6) << pressure << unit
                      << std::endl;
        }
    }
    return;
}

void write_head(std::ofstream& ofs, const int& istep, const int& iter, const std::string& basisname)
{
    ofs << std::right;
    ofs << "\n";
    ofs << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<< std::endl;
    ofs << " --> #ION MOVE#" << std::setw(10) << istep+1
        << "  #ELEC ITER#" << std::setw(10) << iter << "\n"; 
    ofs << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<< std::endl;

}
void write_head_td(std::ofstream& ofs, const int& istep, const int& estep, const int& iter, const std::string& basisname)
{
    ofs << std::right;
    ofs << "\n";
    ofs << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<< std::endl;
    ofs << " --> #ION MOVE#" << std::setw(10) << istep+1
        << " #ELEC PROP#" << std::setw(10) << estep+1
        << "  #ELEC ITER#" << std::setw(10) << iter << "\n"; 
    ofs << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<< std::endl;

}
}// namespace ModuleIO
