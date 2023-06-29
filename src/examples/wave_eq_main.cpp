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

        // YASK factories that make other YASK objects.
        yk_factory kfac;
        yask_output_factory ofac;

        // Initalize MPI, etc.
        auto env = kfac.new_env();
        rank_num = env->get_rank_index();

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
                           "YASK libary compiled with tracing enabled.",
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
        
        // Create solution.
        auto soln = kfac.new_solution(env);

        // Set default sizes.
        if (benchmark_mode)
            soln->set_overall_domain_size_vec({ 2048, 2048 });
        else
            soln->set_overall_domain_size_vec({ 160, 160 });
        soln->set_block_size_vec({ 72, 72 });

        // Print help and exit if requested.  This is done after the
        // defaults are set so that the defaults will be shown in the help
        // message.
        if (help) {
            os << "Usage: " << pgm_name << " [options]\n"
                "Options from the '" << pgm_name << "' binary:\n";
            parser.print_help(os);
            os << "Options from the YASK library:\n" <<
                soln->get_command_line_help();
            env->exit(1);
        }

        // Apply any YASK command-line options.
        // These may override the defaults set above.
        // Parse standard args from YASK library.
        rem_args = soln->apply_command_line_options(rem_args);
        if (rem_args.length())
            THROW_YASK_EXCEPTION("extraneous parameter(s): '" +
                                 rem_args +
                                 "'; run with '-help' option for usage");
        
        os << "Number of MPI ranks: " << env->get_num_ranks() << endl;
        os << "YASK solution: " << soln->get_name() << endl;
        assert(soln->get_name() == "wave2d");
        os << "Precision: " << soln->get_element_bytes() << " bytes" << endl;
        assert(soln->get_element_bytes() == sizeof(double));
        os << "Num threads: " << (soln->get_num_outer_threads() * soln->get_num_inner_threads()) << endl;
        os << "Block size: " << soln->get_block_size("x") << " * " << soln->get_block_size("y") << endl << flush;

        // Get access to YASK vars from kernel.
        auto e = soln->get_var("e");
        auto u = soln->get_var("u");
        auto v = soln->get_var("v");

        // Allocate var memory, MPI buffers, etc.
        soln->prepare_solution();

        // Computational global domain (one less than YASK domain) and domain
        // coordinates.
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

        os << "Global grid size for wave eq: " << nx << " * " << ny << endl;
        os << "Elevation DOFs: " << nx*ny << endl;
        os << "Velocity  DOFs: " << (nx+1)*ny + nx*(ny+1) << endl;
        os << "Total     DOFs: " << nx*ny + (nx+1)*ny + nx*(ny+1) << endl;

        // Functions to return domain positions at each local grid index.
        auto xpos =
            [=](idx_t xi) {
                return x_min + dx/2 + xi * dx;
            };
        auto ypos =
            [=](idx_t yi) {
                return y_min + dy/2 + yi * dy;
            };

        // YASK local domain.
        auto firstx = soln->get_first_rank_domain_index("x");
        auto lastx = soln->get_last_rank_domain_index("x");
        auto xsz = lastx - firstx + 1; // number of cells in YASK domain.
        auto firsty = soln->get_first_rank_domain_index("y");
        auto lasty = soln->get_last_rank_domain_index("y");
        auto ysz = lasty - firsty + 1; // number of cells in YASK domain.

        os << "YASK local domain size: " << xsz << " x " << ysz << endl;

        // compute time step
        double t_end = 1.0;
        double t_export = 0.02;

        double c = sqrt(g*h);
        double alpha = 0.5;
        double dt = alpha * dx / c;
        dt = t_export / ceil(t_export / dt);
        idx_t nt = ceil(t_end / dt);
        if (benchmark_mode) {
            nt = 100;
            dt = 1e-5;
            t_export = 25*dt;
            t_end = nt*dt;
        }

        // Set YASK scalars.
        soln->get_var("inv_dx")->set_element(inv_dx, {});
        soln->get_var("inv_dy")->set_element(inv_dy, {});
        soln->get_var("dt")->set_element(dt, {});
        soln->get_var("g")->set_element(g, {});
        soln->get_var("depth")->set_element(h, {});

        // Some buffers used to copy slices of data to/from YASK vars.
        constexpr idx_t xbsz = 256; // buf size in x dim.
        const idx_t ybsz = ysz - 1; // buf size in y dim.
        double ebuf[xbsz][ybsz], ubuf[xbsz][ybsz];
        const idx_t xybsz = xbsz * ybsz;
        
        // Init vars.
        e->set_all_elements_same(0.0);
        u->set_all_elements_same(0.0);
        v->set_all_elements_same(0.0);

        // Set initial elevation by slices.
        for (idx_t firstbx = firstx; firstbx <= lastx - 1; firstbx += xbsz) {
            idx_t lastbx = min(lastx - 1, firstbx + xbsz - 1);
            assert(lastbx >= firstbx);
            idx_t nbx = lastbx - firstbx + 1;
            assert(nbx <= xbsz);
            idx_t nbxy = nbx * ybsz;
            assert(nbxy <= xybsz);

            for (idx_t i = 0; i < nbx; i++) {
                idx_t xi = i + firstbx;
                
                #pragma omp simd
                for (idx_t j = 0; j < ybsz; j++) {
                    idx_t yj = j + firsty;
                    double val = initial_elev(xpos(xi), ypos(yj), lx, ly);
                    ebuf[i][j] = val;
                }
            }
            auto n = e->set_elements_in_slice(ebuf[0], xybsz,
                                              { 0, firstbx, firsty },
                                              { 0, lastbx, lasty-1 });
            assert(n == nbxy);
        }

        // Apply the stencil solution to the data.
        soln->reset_auto_tuner(false);
        env->global_barrier();

        os << "Time step: " << (dt * 1e3) << " ms" << endl;
        os << "Total simulation time: " << fixed << setprecision(1);
        os << (t_end * 1e3) << " ms, ";
        os << nt << " time steps" << endl;
        double initial_v = 0;
        idx_t i_export = 0;
        double t = 0.0;

        os << flush;
        sleep(1);

        // Main simulation loop.
        auto tic = chrono::steady_clock::now();
        for (idx_t ti = 0; ti < nt+1; ) {
            t = ti * dt;

            // Get stats from YASK vars by slices.
            double e_sum = 0.0, e_max = 0.0, u_max = 0.0;
            for (idx_t firstbx = firstx; firstbx <= lastx - 1; firstbx += xbsz) {
                idx_t lastbx = min(lastx - 1, firstbx + xbsz - 1);
                idx_t nbx = lastbx - firstbx + 1;
                idx_t nbxy = nbx * ybsz;

                auto n = e->get_elements_in_slice(ebuf[0], xybsz,
                                                  { ti, firstbx, firsty },
                                                  { ti, lastbx, lasty-1 });
                assert(n == nbxy);
                n = u->get_elements_in_slice(ubuf[0], xybsz,
                                             { ti, firstbx, firsty },
                                             { ti, lastbx, lasty-1 });
                assert(n == nbxy);
            
                for (idx_t i = 0; i < nbx; i++) {
                
                    #pragma omp simd
                    for (idx_t j = 0; j < ybsz; j++) {

                        double eval = ebuf[i][j];
                        double uval = ubuf[i][j];
                        e_sum += eval;
                        e_max = max(e_max, eval);
                        u_max = max(u_max, uval);
                    }
                }
            }

            // Compute and export stats.
            double total_v = (e_sum + h) * dx * dy;
            if (ti==0)
                initial_v = total_v;
            double diff_v = total_v - initial_v;

            os << setprecision(4) <<
                i_export << '\t' << ti << '\t' << t <<
                "\telev=" << e_max <<
                "\tu=" << u_max <<
                "\tdV=" << diff_v << endl;

            if (e_max > 1e3) {
                os << "Invalid elevation value: " << e_max;
                os << endl;
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
        auto toc = chrono::steady_clock::now();

        chrono::duration<double> duration = toc - tic;
        auto ystats = soln->get_stats();
        os << "Duration (s): " << setprecision(2) << duration.count() << endl <<
            "Rate (ms/step): " << (1e3 * duration.count() / (nt+1)) << endl <<
            "Time in YASK kernel (s): " << ystats->get_elapsed_secs() << endl;
        os << flush;
        sleep(1);

        // Compute error against exact solution.
        double err_L2 = 0.0;
        for (idx_t firstbx = firstx; firstbx <= lastx - 1; firstbx += xbsz) {
            idx_t lastbx = min(lastx - 1, firstbx + xbsz - 1);
            assert(lastbx >= firstbx);
            idx_t nbx = lastbx - firstbx + 1;
            assert(nbx <= xbsz);
            idx_t nbxy = nbx * ybsz;
            assert(nbxy <= xybsz);

            auto n = e->get_elements_in_slice(ebuf[0], xybsz,
                                              { nt+1, firstbx, firsty },
                                              { nt+1, lastbx, lasty-1 });
            assert(n == nbxy);
            
            for (idx_t i = 0; i < nbx; i++) {
                idx_t xi = i + firstbx;
                
                #pragma omp simd
                for (idx_t j = 0; j < ybsz; j++) {
                    idx_t yj = j + firsty;

                    double elev_exact = exact_elev(xpos(xi), ypos(yj), t, lx, ly);
                    double elev = ebuf[i][j];
                    double err = abs(elev - elev_exact);
                    err_L2 += (err * err) * dx * dy / lx / ly;
                }
            }
        }
        
        err_L2 = sqrt(err_L2);
        os << "L2 error: " << setw(7) << scientific;
        os << setprecision(5) << err_L2 << endl;

        if (!benchmark_mode) {
            if (nx < 128 || ny < 128) {
                os << "Skipping correctness test due to small problem size." << endl;
            } else {
                double tolerance = 1e-2;
                if (err_L2 > tolerance) {
                    os << "ERROR: L2 error exceeds tolerance: " << err_L2 << " > " << tolerance << endl;
                    return 1;
                } else {
                    os << "SUCCESS" << endl;
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
