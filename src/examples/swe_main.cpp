/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2024, Intel Corporation

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

// Linear shallow-water-equation test case.
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
        ostream& os = (msg_rank == rank_num) ? cout :
            ofac.new_null_output()->get_ostream();
        
        // Create YASK solution.
        auto soln = kfac.new_solution(env);

        // Set default sizes.
        if (benchmark_mode) {
            soln->set_overall_domain_size_vec({ 2048, 2048 });
            soln->set_block_size_vec({128, 48});
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

        // Apply any YASK command-line options.
        // These may override the defaults set above.
        // Parse standard args from YASK library.
        rem_args = soln->apply_command_line_options(rem_args);
        if (rem_args.length())
            THROW_YASK_EXCEPTION("extraneous parameter(s): '" +
                                 rem_args +
                                 "'; run with '-help' option for usage");
        
        os << "Number of MPI ranks: " << num_ranks << endl;
        os << "YASK solution: " << soln->get_name() << endl;
        assert(soln->get_name() == "swe2d");
        os << "Precision: " << soln->get_element_bytes() << " bytes" << endl;
        assert(soln->get_element_bytes() == sizeof(double));
        os << "Num threads: " << (soln->get_num_outer_threads() * soln->get_num_inner_threads()) << endl;
        os << "Block size: " << soln->get_block_size("x") << " * " << soln->get_block_size("y") << endl << flush;

        // Get access to YASK vars from kernel.
        auto u = soln->get_var("u");
        auto v = soln->get_var("v");
        auto e = soln->get_var("e");
        auto h = soln->get_var("h");
        auto q = soln->get_var("q");
        auto pe = soln->get_var("pe");
        auto keH = soln->get_var("keH");

        // Allocate memory for any vars, etc.
        // Set other data structures needed for stencil application.
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
        double inv_dy = double(ny)/ly;

        // See sub-domain diagram in stencil DSL code.

        // Global domain for e.
        idx_t first_gx_e = 0;
        idx_t last_gx_e = nx - 1; // right x border.
        idx_t first_gy_e = 0;
        idx_t last_gy_e = ny - 1; // top y border.

        // Global domain for u.
        idx_t first_gx_u = 1;     // left x border.
        idx_t last_gx_u = nx - 1; // right x border.
        idx_t first_gy_u = 0;
        idx_t last_gy_u = ny - 1; // top y border.

        // Global domain for v;
        idx_t first_gx_v = 0;
        idx_t last_gx_v = nx - 1; // right x border.
        idx_t first_gy_v = 1;     // bottom y border.
        idx_t last_gy_v = ny - 1; // top y border.

        
        // YASK local computational domain.
        auto firstx = soln->get_first_rank_domain_index("x");
        auto lastx = soln->get_last_rank_domain_index("x");
        auto firsty = soln->get_first_rank_domain_index("y");
        auto lasty = soln->get_last_rank_domain_index("y");
        auto ysz = lasty - firsty + 1;

        // Computational local domain for e: intersection of YASK local
        // domain with global domain for e.
        auto firstx_e = max(firstx, first_gx_e);
        auto lastx_e = min(lastx, last_gx_e);
        auto xsz_e = lastx_e - firstx_e + 1;
        auto firsty_e = max(firsty, first_gy_e);
        auto lasty_e = min(lasty, last_gy_e);
        auto ysz_e = lasty_e - firsty_e + 1;

        // Computational local domain for u: intersection of YASK local
        // domain with global domain for u.
        auto firstx_u = max(firstx, first_gx_u);
        auto lastx_u = min(lastx, last_gx_u);
        auto firsty_u = max(firsty, first_gy_u);
        auto lasty_u = min(lasty, last_gy_u);

        os << "Global grid size for elev computation: " << nx << " * " << ny << endl;
        os << "Elevation DOFs: " << nx*ny << endl;
        os << "Velocity  DOFs: " << (nx+1)*ny + nx*(ny+1) << endl;
        os << "Total     DOFs: " << nx*ny + (nx+1)*ny + nx*(ny+1) << endl;
        os << "Local domain size for elev computation: " << xsz_e << " x " << ysz_e << endl;

        // Functions to return domain positions at each global grid index.
        auto xu =
            [=](idx_t xi) {
                return x_min + xi * dx;
            };
        auto yv =
            [=](idx_t yi) {
                return y_min + yi * dy;
            };
        auto xt =
            [=](idx_t xi) {
                return x_min + dx/2 + xi * dx;
            };
        auto yt =
            [=](idx_t yi) {
                return y_min + dy/2 + yi * dy;
            };

        // Functions to determine whether global grid indices
        // are in bounds.
        auto in_e =
            [=](idx_t xi, idx_t yi) {
                return xi >= first_gx_e && xi <= last_gx_e &&
                    yi >= first_gy_e && yi <= last_gy_e;
            };
        auto in_u =
            [=](idx_t xi, idx_t yi) {
                return xi >= first_gx_u && xi <= last_gx_u &&
                    yi >= first_gy_u && yi <= last_gy_u;
            };
        auto in_v =
            [=](idx_t xi, idx_t yi) {
                return xi >= first_gx_v && xi <= last_gx_v &&
                    yi >= first_gy_v && yi <= last_gy_v;
            };
        
        // Functions to return exact values at a global grid coord.
        auto exact_elev_idx =
            [=](idx_t xi, idx_t yi, double t) {
                return in_e(xi, yi) ?
                    exact_elev(xt(xi), yt(yi), t, lx, ly) : 0.0;
            };
        auto exact_u_idx =
            [=](idx_t xi, idx_t yi, double t) {
                return in_u(xi, yi) ?
                    exact_u(xu(xi), yt(yi), t, lx, ly) : 0.0;
            };
        auto exact_v_idx =
            [=](idx_t xi, idx_t yi, double t) {
                return in_v(xi, yi) ?
                    exact_v(xt(xi), yv(yi), t, lx, ly): 0.0;
            };
        auto bathymetry_idx =
            [=](idx_t xi, idx_t yi) {
                return in_e(xi, yi) ?
                    bathymetry(xt(xi), yt(yi), lx, ly): 1.0;
            };
                
        
        // compute time step
        double t_end = 1.0;
        double t_export = 0.02;

        // Some 1D buffers used to copy slices of data to/from YASK vars.
        auto* hbuf = new double[ysz];
        auto* ebuf = new double[ysz];
        auto* ubuf = new double[ysz];
        auto* vbuf = new double[ysz];
        auto* keHbuf = new double[ysz];
        auto* pebuf = new double[ysz];

        // Init whole vars to default values.
        h->set_all_elements_same(1.0);
        e->set_all_elements_same(0.0);
        u->set_all_elements_same(0.0);
        v->set_all_elements_same(0.0);
        q->set_all_elements_same(0.0);
        keH->set_all_elements_same(0.0);
        pe->set_all_elements_same(0.0);

        // Init vars to exact values by slices within YASK domain.
        // Find local and global initial reductions.
        double h2_lsum = 0.0;
        for (idx_t xi = firstx; xi <= lastx; xi++) {
            for (idx_t j = 0; j < ysz; j++) {
                auto yj = firsty + j;

                double h0 = bathymetry_idx(xi, yj);
                hbuf[j] = h0;
                double e0 = exact_elev_idx(xi, yj, 0.0);
                ebuf[j] = e0;
                double u0 = exact_u_idx(xi, yj, 0.0);
                ubuf[j] = u0;
                double v0 = exact_v_idx(xi, yj, 0.0);
                vbuf[j] = v0;
            }
            h->set_elements_in_slice(hbuf, ysz, { xi, firsty }, { xi, lasty });
            e->set_elements_in_slice(ebuf, ysz, { 0, xi, firsty }, { 0, xi, lasty });
            u->set_elements_in_slice(ubuf, ysz, { 0, xi, firsty }, { 0, xi, lasty });
            v->set_elements_in_slice(vbuf, ysz, { 0, xi, firsty }, { 0, xi, lasty });
        }
        double h2_sum = h2_lsum;
        #ifdef USE_MPI
        if (num_ranks > 1)
            MPI_Allreduce(&h2_lsum, &h2_sum, 1, MPI_DOUBLE,
                       MPI_SUM, MPI_COMM_WORLD);
        #endif
        double pe_offset = 0.5 * g * h2_sum / (nx * ny);

        double h_lsum = 0.0, h_lmax = -1e30;
        for (idx_t xi = firstx; xi <= lastx; xi++) {
            for (idx_t j = 0; j < ysz; j++) {
                auto yj = firsty + j;

                double h0 = bathymetry_idx(xi, yj);
                double e0 = exact_elev_idx(xi, yj, 0.0);
                double u0 = exact_u_idx(xi, yj, 0.0);
                double u1 = exact_v_idx(xi+1, yj, 0.0);
                double v0 = exact_v_idx(xi, yj, 0.0);
                double v1 = exact_v_idx(xi, yj+1, 0.0);

                double ke_init = 0.25 * (u1*u1 + u0*u0 + v1*v1 + v0*v0);
                double H = e0 + h0;
                double keH_init = ke_init * H;
                keHbuf[j] = in_e(xi, yj) ? keH_init : 0.0;

                double pe_init = 0.5 * g * H * (e0 - h0) + pe_offset;
                pebuf[j] = in_e(xi, yj) ? pe_init : 0.0;
                
                if (in_e(xi, yj)) {
                    h2_lsum += h0 * h0;
                    h_lsum += h0;
                    h_lmax = std::max(h_lmax, h0);
                }
            }
            keH->set_elements_in_slice(keHbuf, ysz, { 0, xi, firsty }, { 0, xi, lasty });
            pe->set_elements_in_slice(pebuf, ysz, { 0, xi, firsty }, { 0, xi, lasty });
        }
        double h_sum = h_lsum;
        #ifdef USE_MPI
        if (num_ranks > 1)
            MPI_Allreduce(&h_lsum, &h_sum, 1, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD);
        #endif
        double h_max = h_lmax;
        #ifdef USE_MPI
        if (num_ranks > 1)
            MPI_Allreduce(&h_lmax, &h_max, 1, MPI_DOUBLE,
                          MPI_MAX, MPI_COMM_WORLD);
        #endif

        double c = sqrt(g * h_max);
        double alpha = 0.5;
        double dt = alpha * dx / c;
        dt = t_export / ceil(t_export / dt);
        idx_t nt = ceil(t_end / dt);
        if (benchmark_mode) {
            nt = 100;
            dt = 1e-5;
            t_export = 25.0 * dt;
            t_end = dt * nt;
        }
        os << fixed << setprecision(3) <<
            "Time step (ms): " << (dt * 1e3) << endl <<
            "Total simulation time (ms): " << (t_end * 1e3) << endl <<
            "Time steps: " << nt << endl << flush;

        // Set YASK scalars.
        soln->get_var("dx")->set_element(dx, {});
        soln->get_var("dy")->set_element(dy, {});
        soln->get_var("inv_dx")->set_element(inv_dx, {});
        soln->get_var("inv_dy")->set_element(inv_dy, {});
        soln->get_var("dt")->set_element(dt, {});
        soln->get_var("g")->set_element(g, {});
        soln->get_var("coriolis")->set_element(coriolis, {});
        soln->get_var("pe_offset")->set_element(pe_offset, {});

        // Init var instructing YASK when to calculate
        // result needed for export.
        auto ti_exp = soln->get_var("ti_exp");
        ti_exp->set_element(0, {});

        // Final prep before running sim.
        soln->reset_auto_tuner(false);
        soln->exchange_halos();
        env->global_barrier();

        double initial_v = 0;
        double initial_e = 0;
        double diff_v = 0;
        double diff_e = 0;
        idx_t i_export = 0;
        double t = 0.0;

        // Apply the stencil solution to the data.
        // Main simulation loop.
        auto tic = std::chrono::steady_clock::now();
        for (idx_t ti = 0; ti < nt+1; ) {
            t = ti * dt;

            // Get YASK var stats for this rank.
            auto e_red = e->reduce_elements_in_slice(yk_var::yk_sum_reduction |
                                                     yk_var::yk_max_reduction,
                                                     { ti, firstx_e, firsty_e },
                                                     { ti, lastx_e, lasty_e });
            auto u_red = u->reduce_elements_in_slice(yk_var::yk_max_reduction,
                                                     { ti, firstx_u, firsty_u },
                                                     { ti, lastx_u, lasty_u });
            auto q_red = q->reduce_elements_in_slice(yk_var::yk_max_reduction,
                                                     { ti, firstx_e, firsty_e },
                                                     { ti, lastx_e, lasty_e });
            auto keH_red = keH->reduce_elements_in_slice(yk_var::yk_sum_reduction,
                                                         { ti, firstx_e, firsty_e },
                                                         { ti, lastx_e, lasty_e });
            auto pe_red = pe->reduce_elements_in_slice(yk_var::yk_sum_reduction,
                                                       { ti, firstx_e, firsty_e },
                                                       { ti, lastx_e, lasty_e });

            // Local reductions.
            auto e_lsum = e_red->get_sum();
            auto keH_lsum = keH_red->get_sum();
            auto pe_lsum = pe_red->get_sum();
            auto e_lmax = e_red->get_max();
            auto u_lmax = u_red->get_max();
            auto q_lmax = q_red->get_max();

            // Global reductions.
            auto e_sum = e_lsum;
            auto keH_sum = keH_lsum;
            auto pe_sum = pe_lsum;
            auto e_max = e_lmax;
            auto u_max = u_lmax;
            auto q_max = q_lmax;
            #ifdef USE_MPI
            if (num_ranks > 1) {

                // Can use MPI_Reduce() instead of MPI_Allreduce() below
                // because we only use the stats on 'msg_rank'.

                // Sums.
                double ltmp[3] = { e_lsum, keH_lsum, pe_lsum };
                double gtmp[3];
                MPI_Reduce(ltmp, gtmp, 3, MPI_DOUBLE,
                           MPI_SUM, msg_rank, MPI_COMM_WORLD);
                e_sum = gtmp[0];
                keH_sum = gtmp[1];
                pe_sum = gtmp[2];

                // Maxes.
                ltmp[0] = e_lmax;
                ltmp[1] = u_lmax;
                ltmp[2] = q_lmax;
                MPI_Reduce(ltmp, gtmp, 3, MPI_DOUBLE,
                           MPI_MAX, msg_rank, MPI_COMM_WORLD);
                e_max = gtmp[0];
                u_max = gtmp[1];
                q_max = gtmp[2];
            }
            #endif

            // Compute and export stats.
            if (rank_num == msg_rank) {
                double total_v = (e_sum + h_sum) * dx * dy;
                double total_pe = pe_sum * dx * dy;
                double total_ke = keH_sum * dx * dy;
                double total_e = total_ke + total_pe;
                if (ti == 0) {
                    initial_e = total_e;
                    initial_v = total_v;
                }
                diff_e = total_e - initial_e;
                diff_v = total_v - initial_v;

                os << setprecision(4) << fixed << right <<
                    setw(3) << i_export << ' ' <<
                    setw(3) << ti << ' ' <<
                    setw(7) << t << 
                    " elev=" << setw(7) << e_max <<
                    " u=" << setw(7) << u_max <<
                    " q=" << setw(7) << q_max <<
                    scientific <<
                    " dV=" << setw(11) << diff_v <<
                    " PE=" << setw(11) << total_pe <<
                    " KE=" << setw(11) << total_ke <<
                    " dE=" << setw(11) << diff_e <<
                    endl;
            
                if (e_max > 1e3) {
                    cerr << "Invalid elevation value: " << e_max << endl;
                    env->exit(1);
                }
            }

            // Next eval point.
            i_export += 1;
            idx_t ti_end = ti + idx_t(ceil(t_export / dt)) - 1;
            ti_end = min(ti_end, nt);
            ti_end = max(ti_end, ti);

            // Pass time index for exporting data to YASK kernel.
            ti_exp->set_element(ti_end, {});

            // Run simulation steps until next export point.
            soln->run_solution(ti, ti_end);
            ti = ti_end + 1;

        } // Sim loop.
        auto toc = std::chrono::steady_clock::now();

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

        // Compute error against exact solution.
        double lerr_L2 = 0.0;
        for (idx_t xi = firstx; xi <= lastx; xi++) {
            e->get_elements_in_slice(ebuf, ysz, { nt+1, xi, firsty }, { nt+1, xi, lasty });
            for (idx_t j = 0; j < ysz; j++) {
                auto yj = firsty + j;
                if (in_e(xi, yj)) {
                    auto elev_exact = exact_elev_idx(xi, yj, t);
                    auto elev = ebuf[j];
                    double err = abs(elev - elev_exact);
                    lerr_L2 += (err * err) * (dx * dy) / (lx * ly);
                }
            }
        }
        
        // Sum across ranks.
        double gerr_L2 = lerr_L2;
        #ifdef USE_MPI
        if (num_ranks > 1) {
            MPI_Reduce(&lerr_L2, &gerr_L2, 1, MPI_DOUBLE,
                       MPI_SUM, msg_rank, MPI_COMM_WORLD);
        }
        #endif

        // Free buffers.
        delete[] hbuf;
        delete[] ebuf;
        delete[] ubuf;
        delete[] vbuf;
        delete[] keHbuf;
        delete[] pebuf;
        
        // Report global error.
        if (rank_num == msg_rank) {
            gerr_L2 = sqrt(gerr_L2);
            double tolerance_l2 = 1e-4;
            double tolerance_ene = 1e-7;
            bool do_check = (nx >= 128 && ny >= 128);
            os << scientific;
            os << "Overall L2 error: " << gerr_L2 << endl;
            if (do_check) {
                bool ok = true;
                if (gerr_L2 > tolerance_l2) {
                    os << "  ERROR: L2 error exceeds tolerance: " << gerr_L2 <<
                        " > " << tolerance_l2 << endl;
                    ok = false;
                }
                if (abs(diff_e) > tolerance_ene) {
                    os << "ERROR: Energy error (dE) exceeds tolerance: " << abs(diff_e) <<
                        " > " << tolerance_ene << std::endl;
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

        soln->end_solution();
        env->finalize();
        env->exit(0);
    }
    catch (yask_exception e) {
        cerr << "YASK SWE: " << e.get_message() <<
            " on rank " << rank_num << ".\n";
        if (env)
            env->exit(1);
        exit(1);
    }
}
