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
#ifdef USE_MPI
#include "mpi.h"
#endif

using namespace std;
using namespace yask;

// gravitational acceleration
constexpr double g = 9.80665; // https://en.wikipedia.org/wiki/Standard_gravity

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
    yk_env_ptr env;
    try {

        // YASK factories that make other YASK objects.
        yk_factory kfac;
        yask_output_factory ofac;

        // Initalize MPI, etc.
        env = kfac.new_env();
        rank_num = env->get_rank_index();
        int num_ranks = env->get_num_ranks();

        // Command-line options
        bool help = false;          // help requested.
        bool do_debug = false;
        bool do_trace = false;
        int msg_rank = 0;              // rank to print debug msgs.
        bool benchmark_mode = false;

        // Parser for app options.
        command_line_parser parser;
        parser.add_option(make_shared<command_line_parser::bool_option>
                          ("help",
                           "Print help message.",
                           help));
        parser.add_option(make_shared<command_line_parser::int_option>
                          ("msg_rank",
                           "Index of MPI rank that will print messages.",
                           msg_rank));
        parser.add_option(make_shared<command_line_parser::bool_option>
                          ("yask_debug",
                           "Print informational messages from YASK library to 'msg_rank' rank.",
                           do_debug));
        parser.add_option(make_shared<command_line_parser::bool_option>
                          ("yask_trace",
                           "Print internal tracing messages to 'msg_rank' rank if "
                           "YASK libary was compiled with tracing enabled.",
                           do_trace));
        parser.add_option(make_shared<command_line_parser::bool_option>
                          ("bench",
                           "Run in peformance-measurement mode instead of validation mode.",
                           benchmark_mode));

        // Parse 'args' and 'argv' cmd-line options, which sets option values.
        // Any remaining strings will be returned.
        auto rem_args = parser.parse_args(argc, argv);
        string pgm_name = argv[0];

        // Enable output only on requested rank.
        if (do_trace) {
            do_debug = true;
            yk_env::set_trace_enabled(true);
        }
        if (msg_rank != rank_num || !do_debug)
            yk_env::disable_debug_output(); // no YASK lib output.
        auto nullos = ofac.new_null_output();
        ostream& os = (msg_rank == rank_num) ? cout :
            nullos->get_ostream();
        
        // Create YASK solution.
        auto soln = kfac.new_solution(env);

        // Set default sizes.
        if (benchmark_mode) {
            soln->set_overall_domain_size_vec({ 8192, 8192 }); // override with '-g <N>'.

            // Configure for temporal blocking.
            soln->set_block_size("t", 12);          // override with '-bt <N>'.
            soln->set_block_size_vec({ 256, 92 }); // override with '-b <N>'.
        }
        else
            soln->set_overall_domain_size_vec({ 129, 129 });

        // Print help and exit if requested.  This is done after the
        // defaults are set so that the defaults will be shown in the help
        // message.
        if (help && msg_rank == rank_num) {
            os << "Usage: " << pgm_name << " [options]\n"
                "Options from the '" << pgm_name << "' binary:\n";
            parser.print_help(os);
            os << "Options from the YASK library:\n" <<
                soln->get_command_line_help();
            env->exit(1);
        }

        // Parse standard options for YASK library.
        // These may override the defaults set above.
        rem_args = soln->apply_command_line_options(rem_args);
        if (rem_args.length())
            THROW_YASK_EXCEPTION("extraneous parameter(s): '" +
                                 rem_args +
                                 "'; run with '-help' option for usage");
        
        os << "Number of MPI ranks: " << num_ranks << endl;
        os << "YASK solution: " << soln->get_name() << endl;
        assert(soln->get_name() == "wave2d");
        os << "Precision: " << soln->get_element_bytes() << " bytes" << endl;
        assert(soln->get_element_bytes() == sizeof(double));
        os << "Num threads: " << (soln->get_num_outer_threads() * soln->get_num_inner_threads()) << endl;
        os << "Block size: " << soln->get_block_size("x") << " * " << soln->get_block_size("y") << endl;
        os << "Mode: " << (benchmark_mode ? "benchmark" : "validation") << endl << flush;

        // Get access to YASK vars from kernel.
        auto e = soln->get_var("e");
        auto u = soln->get_var("u");
        auto v = soln->get_var("v");

        // Allocate var memory, MPI buffers, etc.
        soln->prepare_solution();

        // Computational global domain (one less than overall YASK domain)
        // and domain coordinates.
        idx_t nx = soln->get_overall_domain_size("x") - 1;
        double x_min = -1.0;
        double x_max = 1.0;
        double lx = x_max - x_min;
        double dx = lx/nx;
        double inv_dx = double(nx) / lx;
        idx_t ny = soln->get_overall_domain_size("y") - 1;
        double y_min = -1.0;
        double y_max = 1.0;
        double ly = y_max - y_min;
        double dy = ly/ny;
        double inv_dy = double(ny) / ly;

        // Computational local domain, i.e., intersection of local YASK
        // domain with computational global domain.
        auto firstx = soln->get_first_rank_domain_index("x");
        auto lastx = min(soln->get_last_rank_domain_index("x"), nx-1);
        auto xsz = lastx - firstx + 1;
        auto firsty = soln->get_first_rank_domain_index("y");
        auto lasty = min(soln->get_last_rank_domain_index("y"), ny-1);
        auto ysz = lasty - firsty + 1;

        os << "Global grid size for computation: " << nx << " * " << ny << endl;
        os << "Elevation DOFs: " << nx*ny << endl;
        os << "Velocity  DOFs: " << (nx+1)*ny + nx*(ny+1) << endl;
        os << "Total     DOFs: " << nx*ny + (nx+1)*ny + nx*(ny+1) << endl;
        os << "Local domain size for computation: " << xsz << " x " << ysz << endl;

        // Functions to return coords at each local grid index.
        auto xpos =
            [=](idx_t xi) {
                return x_min + dx/2 + xi * dx;
            };
        auto ypos =
            [=](idx_t yi) {
                return y_min + dy/2 + yi * dy;
            };

        // compute time step
        double t_end = 1.0;
        double t_export = 0.02;

        double c = sqrt(g * h);
        double alpha = 0.5;
        double dt = alpha * dx / c;
        dt = t_export / ceil(t_export / dt);
        idx_t nt = ceil(t_end / dt);
        if (benchmark_mode) {
            nt = 100;
            dt = 1e-5;
            t_export = 25.0 * dt;
            t_end = nt * dt;
        }

        // Set YASK scalars.
        soln->get_var("inv_dx")->set_element(inv_dx, {});
        soln->get_var("inv_dy")->set_element(inv_dy, {});
        soln->get_var("dt")->set_element(dt, {});
        soln->get_var("g")->set_element(g, {});
        soln->get_var("depth")->set_element(h, {});

        // Buffer used to copy slices of data to/from YASK var.
        auto* ebuf = new double[ysz];
        
        // Init vars.
        e->set_all_elements_same(0.0);
        u->set_all_elements_same(0.0);
        v->set_all_elements_same(0.0);

        // Set initial elevation by slices.
        for (idx_t xi = firstx; xi <= lastx; xi++) {

            #pragma omp simd
            for (idx_t j = 0; j < ysz; j++) {
                idx_t yj = j + firsty;
                double val = initial_elev(xpos(xi), ypos(yj), lx, ly);
                ebuf[j] = val;
            }
            auto n = e->set_elements_in_slice(ebuf, ysz,
                                              { 0, xi, firsty },
                                              { 0, xi, lasty });
            assert(n == ysz);
        }

        // Apply the stencil solution to the data.
        soln->reset_auto_tuner(false);
        env->global_barrier();

        os << fixed << setprecision(3) <<
            "Time step (ms): " << (dt * 1e3) << endl <<
            "Total simulation time (ms): " << (t_end * 1e3) << endl <<
            "Time steps: " << nt << endl << flush;
        double initial_v = 0;
        idx_t i_export = 0;
        double t = 0.0;

        os.unsetf(ios_base::floatfield); // default FP formatting.
        os << setprecision(4) << left;

        // Main simulation loop.
        auto tic = chrono::steady_clock::now();
        for (idx_t ti = 0; ti < nt+1; ) {
            t = ti * dt;

            // Get YASK var stats for this rank.
            auto e_red = e->reduce_elements_in_slice(yk_var::yk_sum_reduction |
                                                     yk_var::yk_max_reduction,
                                                     { ti, firstx, firsty },
                                                     { ti, lastx, lasty });
            auto u_red = u->reduce_elements_in_slice(yk_var::yk_max_reduction,
                                                     { ti, firstx, firsty },
                                                     { ti, lastx, lasty });

            // Local reductions.
            auto e_lsum = e_red->get_sum();
            auto e_lmax = e_red->get_max();
            auto u_lmax = u_red->get_max();

            // Global reductions.
            double e_gsum = e_lsum;
            double e_gmax = e_lmax;
            double u_gmax = u_lmax;
            #ifdef USE_MPI
            if (num_ranks > 1) {
                MPI_Reduce(&e_lsum, &e_gsum, 1, MPI_DOUBLE,
                           MPI_SUM, msg_rank, MPI_COMM_WORLD);
                double ltmp[2] = { e_lmax, u_lmax };
                double gtmp[2];
                MPI_Reduce(ltmp, gtmp, 2, MPI_DOUBLE,
                           MPI_MAX, msg_rank, MPI_COMM_WORLD);
                e_gmax = gtmp[0];
                u_gmax = gtmp[1];
            }
            #endif
            
            // Compute and export stats.
            double total_v = (e_gsum + h) * dx * dy;
            if (ti==0)
                initial_v = total_v;
            double diff_v = total_v - initial_v;

            os << i_export << '\t' << ti << '\t' << t <<
                "\telev=" << setw(8) << e_gmax << setw(0) <<
                "\tu=" << setw(8) << u_gmax << setw(0) <<
                "\tdV=" << setw(10) << diff_v << setw(0) << endl;

            if (e_gmax > 1e3) {
                os << "Invalid elevation value: " << e_gmax;
                os << endl;
                env->exit(1);
            }

            // Next eval point.
            i_export += 1;
            idx_t ti_end = ti + idx_t(ceil(t_export / dt)) - 1;
            ti_end = min(ti_end, nt);
            ti_end = max(ti_end, ti);

            // Run simulation steps until next export point.
            soln->run_solution(ti, ti_end);
            ti = ti_end + 1;
        }
        auto toc = chrono::steady_clock::now();

        // Report perf stas.
        chrono::duration<double> duration = toc - tic;
        auto dur = duration.count();
        auto ystats = soln->get_stats();
        auto ydur = ystats->get_elapsed_secs();
        os << fixed << setprecision(2) <<
            "Duration (s): " << dur << endl <<
            "Rate (ms/step): " << (1e3 * dur / nt) << endl <<
            "Time in YASK kernel (s): " << ydur <<
            " (" << (100. * ydur / dur) << "%)" << endl <<
            flush;

        // Compute local error against exact solution.
        double lerr_L2 = 0.0;
        for (idx_t xi = firstx; xi <= lastx; xi++) {

            auto n = e->get_elements_in_slice(ebuf, ysz,
                                              { nt+1, xi, firsty },
                                              { nt+1, xi, lasty });
            assert(n == ysz);
            
            #pragma omp simd
            for (idx_t j = 0; j < ysz; j++) {
                idx_t yj = j + firsty;

                double elev_exact = exact_elev(xpos(xi), ypos(yj), t, lx, ly);
                double elev = ebuf[j];
                double err = abs(elev - elev_exact);
                lerr_L2 += (err * err) * dx * dy / lx / ly;
            }
        }

        // Sum err across ranks.
        double oerr_L2 = lerr_L2;
        #ifdef USE_MPI
        if (num_ranks > 1) {
            MPI_Reduce(&lerr_L2, &oerr_L2, 1, MPI_DOUBLE,
                       MPI_SUM, msg_rank, MPI_COMM_WORLD);
        }
        #endif

        // Free buffer.
        delete[] ebuf;
        
        // Report global stats.
        if (rank_num == msg_rank) {
            oerr_L2 = sqrt(oerr_L2);
            double tolerance = 1e-2;
            bool do_check = (nx >= 128 && ny >= 128);
            os << scientific;
            os << "Overall L2 error: " << oerr_L2 << endl;
            if (do_check) {
                bool ok = true;
                if (oerr_L2 > tolerance) {
                    os << "  ERROR: L2 error exceeds tolerance: " << oerr_L2 << " > " << tolerance << endl;
                    ok = false;
                }
                if (ok)
                    os << "  SUCCESS" << endl;
                else
                    env->exit(1);
            } else {
                os << "  Skipping correctness test due to small problem size." << endl;
            }
            os << flush;
        }
        env->global_barrier();  // Ensure failing rank exits.

        // Clean exit.
        soln->end_solution();
        env->finalize();
        env->exit(0);
    }
    catch (yask_exception e) {
        cerr << "Wave equation: " << e.get_message() <<
            " on rank " << rank_num << ".\n";
        if (env)
            env->exit(1);
        exit(1);
    }
}
