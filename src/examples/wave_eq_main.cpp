/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

*****************************************************************************/

// Linear wave equation test case.
// Contributed by Tuomas Karna, Intel Corp.

#include "yask_kernel_api.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <chrono>

using namespace std;
using namespace yask;

// gravitational acceleration
constexpr double g = 9.81;

// water depth
constexpr double h = 1.0;

/*
 * Exact solution for elevation field.
 * 
 * Returns time-dependent elevation of a 2D standing wave in a
 * rectangular domain.
 */
double exact_elev(double x, double y, double t, double lx, double ly) {
    double amp = 0.5;
    double c = sqrt(g * h);
    size_t n = 1;
    double sol_x = cos(2 * n * M_PI * x / lx);
    size_t m = 1;
    double sol_y = cos(2 * m * M_PI * y / ly);
    double omega = c * M_PI * hypot(n/lx, m/ly);
    double sol_t = cos(2 * omega * t);
    return amp * sol_x * sol_y * sol_t;
}

double initial_elev(double x, double y, double lx, double ly) {
    return exact_elev(x, y, 0.0, lx, ly);
}

int main(int argc, char** argv) {

    int rank_num = -1;
    try {

        // The factory from which all other kernel objects are made.
        yk_factory kfac;

        // Initalize MPI, etc.
        auto env = kfac.new_env();
        rank_num = env->get_rank_index();

        // parse command line options
        bool benchmark_mode = false;
        idx_t grid_size = 128;
        int i = 0;
        while (i < argc) {
            if (string(argv[i]) == "-t") {
                benchmark_mode = true;
            }
            if (string(argv[i]) == "-n") {
                i++;
                grid_size = atoi(argv[i]);
            }
            i++;
        }

        cout << "Number of MPI ranks: " << env->get_num_ranks() << endl;

        yk_env::disable_debug_output();
        //yk_env::set_trace_enabled(true);

        // Create solution.
        auto soln = kfac.new_solution(env);

        // Set default domain and block sizes.
        soln->set_overall_domain_size("x", grid_size+1);
        soln->set_overall_domain_size("y", grid_size+1);
        soln->set_block_size("x", 72);
        soln->set_block_size("y", 72);
        
        // Apply any YASK command-line options.
        // These may override the defaults set above.
        soln->apply_command_line_options(argc, argv);

        cout << "YASK solution: " << soln->get_name() << endl;
        assert(soln->get_name() == "wave2d");
        cout << "precision: " << soln->get_element_bytes() << " bytes" << endl;
        assert(soln->get_element_bytes() == 8);
        cout << "num threads: " << (soln->get_num_outer_threads() * soln->get_num_inner_threads()) << endl;
        cout << "block size: " << soln->get_block_size("x") << " x " << soln->get_block_size("y") << endl << flush;

        // Get access to YASK vars from kernel.
        auto e = soln->get_var("e");
        auto u = soln->get_var("u");
        auto v = soln->get_var("v");

        // Make additional YASK vars.
        // x and y coordinates in the cell center (T points).
        // NOTE the last element is not used; last valid element is grid_size-1.
        auto x = soln->new_var("x", {"x"});
        auto y = soln->new_var("y", {"y"});
        
        // Allocate memory for any vars that do not have storage set.
        // Set other data structures needed for stencil application.
        soln->prepare_solution();

        // YASK domains.
        auto firstx = soln->get_first_rank_domain_index("x");
        auto lastx = soln->get_last_rank_domain_index("x");
        auto xsz = lastx - firstx + 1; // number of cells in YASK domain.
        auto firsty = soln->get_first_rank_domain_index("y");
        auto lasty = soln->get_last_rank_domain_index("y");
        auto ysz = lasty - firsty + 1; // number of cells in YASK domain.
        cout << "YASK local domain size: " << xsz << " x " << ysz << endl;

        // Define domain coordinates at cell centers
        idx_t nx = grid_size;  // number of cells
        double x_min = -1.0;
        double x_max = 1.0;
        double lx = x_max - x_min;
        double dx = lx/nx;
        idx_t ny = grid_size;  // number of cells
        double y_min = -1.0;
        double y_max = 1.0;
        double ly = y_max - y_min;
        double dy = ly/ny;

        for (idx_t i = firstx; i <= lastx; i++) {
            double xi = x_min + dx/2 + i * dx;  // cell center
            x->set_element(xi, {i});
        }
        for (idx_t j = firsty; j <= lasty; j++) {
            double yj = y_min + dy/2 + j * dy;  // cell center
            y->set_element(yj, {j});
        }

        cout << "Grid size: " << nx << " x " << ny << endl;
        cout << "Elevation DOFs: " << nx*ny << endl;
        cout << "Velocity  DOFs: " << (nx+1)*ny + nx*(ny+1) << endl;
        cout << "Total     DOFs: " << nx*ny + (nx+1)*ny + nx*(ny+1);
        cout << endl;

        // compute time step
        double t_end = 1.0;
        double t_export = 0.02;

        double c = sqrt(g*h);
        double alpha = 0.5;
        double dt = alpha * dx / c;
        dt = t_export / static_cast<int>(ceil(t_export / dt));
        idx_t nt = static_cast<int>(ceil(t_end / dt));
        if (benchmark_mode) {
            nt = 100;
            dt = 1e-5;
            t_export = 25*dt;
            t_end = nt*dt;
        }
        cout << "Time step: " << (dt * 1e3) << " ms" << endl;
        cout << "Total simulation time: " << fixed << setprecision(1);
        cout << (t_end * 1e3) << " ms, ";
        cout << nt << " time steps" << endl;

        auto dx_var = soln->get_var("dx");
        auto dy_var = soln->get_var("dy");
        auto dt_var = soln->get_var("dt");
        dx_var->set_element(dx, {});
        dy_var->set_element(dy, {});
        dt_var->set_element(dt, {});

        soln->get_var("g")->set_element(g, {});
        soln->get_var("depth")->set_element(h, {});

        // Some 1D buffers used to copy slices of data to/from YASK vars.
        double bufs[2][ysz];
        
        // Init vars.
        e->set_all_elements_same(0.0);
        u->set_all_elements_same(0.0);
        v->set_all_elements_same(0.0);

        // Set elevation by slices.
        for (idx_t i = firstx; i <= lastx-1; i++) {
            double xi = x->get_element({i});
            for (idx_t j = 0; j < ysz-1; j++) {
                double yj = y->get_element({j + firsty});
                double val = initial_elev(xi, yj, lx, ly);
                bufs[0][j] = val;
            }
            e->set_elements_in_slice(bufs[0], ysz, { 0, i, firsty }, { 0, i, lasty-1 });
        }

        // Apply the stencil solution to the data.
        soln->reset_auto_tuner(false);
        env->global_barrier();

        double initial_v = 0;
        idx_t i_export = 0;
        double t = 0.0;

        // Main simulation loop.
        auto tic = std::chrono::steady_clock::now();
        for (idx_t ti = 0; ti < nt+1; ) {
            t = ti * dt;

            // Get stats from YASK vars by slices.
            double e_sum = 0.0, e_max = 0.0, u_max = 0.0;
            for (idx_t i = firstx; i <= lastx-1; i++) {
                for (idx_t j = 0; j < ysz-1; j++)
                    bufs[1][j] = 99.;
                auto n = e->get_elements_in_slice(bufs[0], ysz, { ti, i, firsty }, { ti, i, lasty });
                assert(n == ysz);
                n = u->get_elements_in_slice(bufs[1], ysz, { ti, i, firsty }, { ti, i, lasty });
                assert(n == ysz);

                #pragma omp simd
                for (idx_t j = 0; j < ysz-1; j++) {
                    auto eval = bufs[0][j];
                    auto uval = bufs[1][j];
                    //cout << " ** u[" << i << "," << j << "] = " << uval <<
                    //", " << u->get_element({ ti, i, j}) << endl;
                    e_sum += eval;
                    e_max = std::max(e_max, eval);
                    u_max = std::max(u_max, uval);
                }
            }

            // Compute and export stats.
            double total_v = (e_sum + h) * dx * dy;
            if (ti==0)
                initial_v = total_v;
            double diff_v = total_v - initial_v;

            printf("%2lu %4u %.3f ", i_export, i, t);
            printf("elev=%7.5f ", e_max);
            printf("u=%7.5f ", u_max);
            printf("dV=% 6.3e ", diff_v);
            printf("\n");

            if (e_max > 1e3) {
                std::cout << "Invalid elevation value: " << e_max;
                std::cout << std::endl;
                return 1;
            }
            i_export += 1;
            idx_t ti_end = ti + idx_t(ceil(t_export / dt)) - 1;
            ti_end = min(ti_end, nt);
            ti_end = max(ti_end, ti);

            // simulation steps.
            soln->run_solution(ti, ti_end);
            ti = ti_end + 1;
        }
        auto toc = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = toc - tic;
        std::cout << "Duration: " << std::setprecision(2) << duration.count() <<
            " s" << std::endl <<
            "Rate: " << (1e3 * duration.count() / (nt+1)) << " ms/step" << std::endl;

        // Compute error against exact solution
        double err_L2 = 0.0;
        y->get_elements_in_slice(bufs[0], ysz, { firsty }, { lasty });
        for (idx_t i = firstx; i <= lastx-1; i++) {
            double xi = x->get_element({i});
            e->get_elements_in_slice(bufs[1], ysz, { nt+1, i, firsty }, { nt+1, i, lasty });
            for (idx_t j = 0; j <= ysz-1; j++) {
                double yi = bufs[0][j];
                double elev_exact = exact_elev(xi, yi, t, lx, ly);
                double elev = bufs[1][j]; // e->get_element({nt+1, i, j});
                double err = abs(elev - elev_exact);
                err_L2 += (err * err) * dx * dy / lx / ly;
            }
        }
        
        err_L2 = std::sqrt(err_L2);
        std::cout << "L2 error: " << std::setw(7) << std::scientific;
        std::cout << std::setprecision(5) << err_L2 << std::endl;

        if (!benchmark_mode) {
            if (nx < 128 || ny < 128) {
                std::cout << "Skipping correctness test due to small problem size." << std::endl;
            } else {
                double tolerance = 1e-2;
                if (err_L2 > tolerance) {
                    std::cout << "ERROR: L2 error exceeds tolerance: " << err_L2 << " > " << tolerance << std::endl;
                    return 1;
                } else {
                    std::cout << "SUCCESS" << std::endl;
                }
            }
        }

        soln->end_solution();
        soln->get_stats();
        env->finalize();
        return 0;
    }
    catch (yask_exception e) {
        cerr << "Wave equation: " << e.get_message() <<
            " on rank " << rank_num << ".\n";
        return 1;
    }
}
