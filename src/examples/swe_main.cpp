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

// linear wave equation test case

#include <assert.h>
#include "yask_kernel_api.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <cmath>
#include <chrono>

using namespace std;
using namespace yask;

// gravitational acceleration
constexpr double g = 9.81;

// coriolis parameter
constexpr double coriolis = 10.0;

// exact solution for a stationary geostrophic gyre
double exact_elev(double x, double y, double t, double lx, double ly) {
    double amp = 0.02;
    double sigma = 0.4;
    return amp*exp(-(x*x + y*y)/sigma/sigma);
}

double exact_u(double x, double y, double t, double lx, double ly) {
    double elev = exact_elev(x, y, t, lx, ly);
    double sigma = 0.4;
    return g/coriolis*2*y/sigma/sigma*elev;
}

double exact_v(double x, double y, double t, double lx, double ly) {
    double elev = exact_elev(x, y, t, lx, ly);
    double sigma = 0.4;
    return -g/coriolis*2*x/sigma/sigma*elev;
}

inline double bathymetry(double x, double y, double lx, double ly) {
    return 1.0;
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

            // NB: This is similar to YASK's -g option,
            // but one (1) element is added to this value
            // before passing it to YASK.
            if (string(argv[i]) == "-n") {
                i++;
                grid_size = atoi(argv[i]);
            }
            i++;
        }

        cout << "Number of MPI ranks: " << env->get_num_ranks() << endl;
        if (env->get_num_ranks() > 1) {

            // Need to update the code below to process boundaries
            // only around the global domain, not each local domain.
            cerr << "This app is not yet enabled for >1 rank.\n";
            exit(1);
        }

        #ifdef TRACE
        if (rank_num == 0)
            yk_env::set_trace_enabled(true);
        else
            yk_env::disable_debug_output();
        #else
        yk_env::disable_debug_output();
        #endif

        // Create YASK solution.
        auto soln = kfac.new_solution(env);

        // Set default domain and block sizes.
        soln->set_overall_domain_size_vec({grid_size+1, grid_size+1});
        soln->set_block_size_vec({128, 48});

        // Apply YASK command-line options.
        // These may override the defaults set above.
        soln->apply_command_line_options(argc, argv);

        cout << "YASK threads: " << soln->get_num_outer_threads() << endl;
        cout << "Block size: " << soln->get_block_size("x") << " x " << soln->get_block_size("y") << endl;

        auto u = soln->get_var("u");
        auto v = soln->get_var("v");
        auto e = soln->get_var("e");
        auto h = soln->get_var("h");

        auto q = soln->get_var("q");
        auto ke = soln->get_var("ke");

        auto xt = soln->new_var("xt", {"x"});
        auto yt = soln->new_var("yt", {"y"});
        auto xu = soln->new_var("xu", {"x"});
        auto yv = soln->new_var("yv", {"y"});

        // Allocate memory for any vars that do not have storage set.
        // Set other data structures needed for stencil application.
        soln->prepare_solution();

        // Define domain coordinates at cell centers
        idx_t nx = grid_size;  // number of cells
        double x_min = -1.0;
        double x_max = 1.0;
        double lx = x_max - x_min;
        double dx = lx/nx;
        double inv_dx = double(nx)/lx;
        idx_t ny = grid_size;  // number of cells
        double y_min = -1.0;
        double y_max = 1.0;
        double ly = y_max - y_min;
        double dy = ly/ny;
        double inv_dy = double(ny)/ly;

        // YASK domains.
        auto firstx = soln->get_first_rank_domain_index("x");
        auto lastx = soln->get_last_rank_domain_index("x");
        auto xsz = lastx - firstx + 1; // number of cells in YASK domain.
        auto firsty = soln->get_first_rank_domain_index("y");
        auto lasty = soln->get_last_rank_domain_index("y");
        auto ysz = lasty - firsty + 1; // number of cells in YASK domain.
        //auto xysz = xsz * ysz;

        // x and y coordinates in the cell center (T points)
        // NOTE the last element is not used; last valid element is grid_size-1
        for (idx_t i = firstx; i <= lastx; i++) {
            double xi = x_min + i * dx;
            xt->set_element(xi + dx/2, {i});  // cell center
            xu->set_element(xi, {i});
        }
        for (idx_t j = firsty; j <= lasty; j++) {
            double yj = y_min + j * dy;
            yt->set_element(yj + dy/2, {j});  // cell center
            yv->set_element(yj, {j});
        }

        // Some 1D buffers used to copy slices of data to/from YASK vars.
        double bufs[7][ysz];

        cout << "Grid size: " << nx << " x " << ny << endl;
        cout << "Elevation DOFs: " << nx*ny << endl;
        cout << "Velocity  DOFs: " << (nx+1)*ny + nx*(ny+1) << endl;
        cout << "Total     DOFs: " << nx*ny + (nx+1)*ny + nx*(ny+1) << endl;

        // compute time step
        double t_end = 1.0;
        double t_export = 0.02;

        // Init vars.
        h->set_all_elements_same(1.0);
        e->set_all_elements_same(0.0);
        u->set_all_elements_same(0.0);
        v->set_all_elements_same(0.0);
        q->set_all_elements_same(0.0);
        ke->set_all_elements_same(0.0);

        double h2_sum = 0.0;
        for (idx_t i = firstx; i <= lastx-1; i++) {
            double xti = xt->get_element({i});
            for (idx_t j = 0; j < ysz-1; j++) {
                double ytj = yt->get_element({j + firsty});
                double elev = initial_elev(xti, ytj, lx, ly);
                double bath = bathymetry(xti, ytj, lx, ly);
                bufs[0][j] = elev; // for e.
                bufs[1][j] = bath; // for h.
                h2_sum += bath*bath;
            }
            e->set_elements_in_slice(bufs[0], { 0, i, firsty }, { 0, i, lasty-1 });
            h->set_elements_in_slice(bufs[1], { i, firsty }, { i, lasty-1 });
        }

        for (idx_t i = firstx+1; i <= lastx-1; i++) {
            double xui = xu->get_element({i});
            for (idx_t j = 0; j < ysz-1; j++) {
                double yti = yt->get_element({j + firsty});
                double u_val = exact_u(xui, yti, 0, lx, ly);
                bufs[0][j] = u_val;
            }
            u->set_elements_in_slice(bufs[0], { 0, i, firsty }, { 0, i, lasty-1 });
        }
        for (idx_t i = firstx; i <= lastx-1; i++) {
            double xti = xt->get_element({i});
            bufs[0][0] = 0.0;
            for (idx_t j = 1; j <= ysz-1; j++) {
                double yvi = yv->get_element({j+firsty});
                double v_val = exact_v(xti, yvi, 0, lx, ly);
                bufs[0][j] = v_val;
            }
            v->set_elements_in_slice(bufs[0], { 0, i, firsty }, { 0, i, lasty-1 });
        }

        // compute initial energy & h reductions.
        double sum_h = 0.0, max_h = -1e30;
        double pe_offset = 0.5 * g * h2_sum / nx / ny;
        double* bu0 = bufs[0];
        u->get_elements_in_slice(bu0, { 0, firstx, firsty }, { 0, firstx, lasty });
        double* bu1 = bufs[1];
        for (idx_t i = firstx; i <= lastx-1; i++) {
            u->get_elements_in_slice(bu1, { 0, i+1, firsty }, { 0, i+1, lasty });
            v->get_elements_in_slice(bufs[2], { 0, i, firsty }, { 0, i, lasty });
            e->get_elements_in_slice(bufs[3], { 0, i, firsty }, { 0, i, lasty });
            h->get_elements_in_slice(bufs[4], { i, firsty }, { i, lasty });
            for (idx_t j = 0; j < ysz-1; j++) {
                double u0 = bu0[j]; // u->get_element({0, i, j});
                double u1 = bu1[j]; // u->get_element({0, i+1, j});
                double v0 = bufs[2][j]; // v->get_element({0, i, j});
                double v1 = bufs[2][j+1]; // v->get_element({0, i, j+1});
                double ke_init = 0.25 * (u1*u1 + u0*u0 + v1*v1 + v0*v0);
                bufs[6][j] = ke_init;
                sum_h += bufs[4][j];
                max_h = std::max(max_h, bufs[4][j]);
            }
            ke->set_elements_in_slice(bufs[6], { 0, i, firsty }, { 0, i, lasty-1 });

            // Swap i & i+1 u-ptrs.
            std::swap(bu0, bu1);
        }

        double c = sqrt(g * max_h);
        double alpha = 0.5;
        double dt = alpha * dx / c;
        dt = t_export / static_cast<int>(ceil(t_export / dt));
        idx_t nt = static_cast<idx_t>(ceil(t_end / dt));
        if (benchmark_mode) {
            nt = 100;
            dt = 1e-5;
            t_export = 25*dt;
            t_end = dt*nt;
        }
        cout << "Time step: " << dt << " s" << endl;
        cout << "Total run time: " << fixed << setprecision(1);
        cout << t_end << " s, ";
        cout << nt << " time steps" << endl;

        // Set YASK scalars.
        soln->get_var("dx")->set_element(dx, {});
        soln->get_var("dy")->set_element(dy, {});
        soln->get_var("inv_dx")->set_element(inv_dx, {});
        soln->get_var("inv_dy")->set_element(inv_dy, {});
        soln->get_var("dt")->set_element(dt, {});
        soln->get_var("g")->set_element(g, {});
        soln->get_var("coriolis")->set_element(coriolis, {});
        auto ti_exp = soln->get_var("ti_exp");
        ti_exp->set_element(0, {});

        // Apply the stencil solution to the data.
        soln->reset_auto_tuner(false);
        env->global_barrier();

        double initial_v = 0;
        double initial_e = 0;
        double diff_v = 0;
        double diff_e = 0;
        idx_t i_export = 0;
        double t = 0.0;

        // Main simulation loop.
        auto tic = std::chrono::steady_clock::now();
        for (idx_t ti = 0; ti < nt+1; ) {
            t = ti * dt;

            // Compute and export stats.
            double sum_keH = 0.0, sum_e = 0.0, elev_max = 0.0,
                pe_sum = 0.0, u_max = 0.0, q_max = 0.0;
            for (idx_t i = firstx; i <= lastx-1; i++) {
                e->get_elements_in_slice(bufs[0], { ti, i, firsty }, { ti, i, lasty });
                h->get_elements_in_slice(bufs[1], { i, firsty }, { i, lasty });
                ke->get_elements_in_slice(bufs[2], { ti, i, firsty }, { ti, i, lasty });
                u->get_elements_in_slice(bufs[4], { ti, i, firsty }, { ti, i, lasty });
                q->get_elements_in_slice(bufs[5], { ti, i, firsty }, { ti, i, lasty });

                #pragma omp simd
                for (idx_t j = 0; j < ysz-1; j++) {
                    auto eval = bufs[0][j];
                    auto hval = bufs[1][j];
                    auto keval = bufs[2][j];
                    auto H = eval + hval;
                    sum_keH += keval * H;
                    sum_e += eval;
                    elev_max = std::max(elev_max, eval);
                    pe_sum += 0.5 * g * H * (eval - hval) + pe_offset;
                    u_max = std::max(u_max, bufs[4][j]);
                    q_max = std::max(q_max, bufs[5][j]);
                }
            }
            double total_v = (sum_e + sum_h) * dx * dy;
            double total_pe = pe_sum * dx * dy;
            double total_ke = sum_keH * dx * dy;
            double total_e = total_ke + total_pe;
            if (ti == 0) {
                initial_e = total_e;
                initial_v = total_v;
            }
            diff_e = total_e - initial_e;
            diff_v = total_v - initial_v;

            printf("%2lu %4lu %.3f ", i_export, ti, t);
            printf("elev=%7.5f ", elev_max);
            printf("u=%7.5f ", u_max);
            printf("q=%8.5f ", q_max);
            printf("dV=% 6.3e ", diff_v);
            printf("PE=%7.2e ", total_pe);
            printf("KE=%7.2e ", total_ke);
            printf("dE=% 6.3e ", diff_e);
            printf("\n");

            if (elev_max > 1e3) {
                std::cout << "Invalid elevation value: " << elev_max;
                std::cout << std::endl;
                return 1;
            }
            i_export += 1;
            idx_t ti_end = ti + idx_t(ceil(t_export / dt)) - 1;
            ti_end = min(ti_end, nt);
            ti_end = max(ti_end, ti);

            // Pass time index for exporting data to YASK kernel.
            ti_exp->set_element(ti_end, {});

            // simulation steps
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
        yt->get_elements_in_slice(bufs[0], { firsty }, { lasty });
        for (idx_t i = firstx; i <= lastx-1; i++) {
            double xi = xt->get_element({i});
            e->get_elements_in_slice(bufs[1], { nt+1, i, firsty }, { nt+1, i, lasty });
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
                double tolerance_l2 = 1e-4;
                double tolerance_ene = 1e-7;
                bool fail = 0;
                if (err_L2 > tolerance_l2) {
                    std::cout << "ERROR: L2 error exceeds tolerance: " << err_L2 << " > " << tolerance_l2 << std::endl;
                    fail = 1;
                }
                if (diff_e > tolerance_ene) {
                    std::cout << "ERROR: Energy error exceeds tolerance: " << diff_e << " > " << tolerance_ene << std::endl;
                    fail = 1;
                }
                if (fail) {
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
        cerr << "YASK SWE: " << e.get_message() <<
            " on rank " << rank_num << ".\n";
        return 1;
    }
}
