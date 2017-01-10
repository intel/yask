/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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

// This code is the YASK stencil performance-measurement tool.  It can be
// used as a starting-point for integrating a YASK stencil kernel into other
// applications.

// Base classes for stencil code.
#include "stencil_calc.hpp"

// Additional header(s) needed by this app.
#include <sstream>

using namespace std;
using namespace yask;

// Some command-line options for this application, i.e., settings not stored
// in the YASK stencil context.
struct AppOptions {
    bool help;                  // help requested.
    bool doWarmup;              // whether to do warmup run.
    idx_t num_trials;           // number of trials.
    bool validate;              // whether to do validation run.
    int pre_trial_sleep_time;   // sec to sleep before each trial.
    string error_msg;
    bool parse_ok;
    
    AppOptions() :
        help(false),
        doWarmup(true),
        num_trials(3),
        validate(false),
        pre_trial_sleep_time(1),
        parse_ok(true)
    { }

    // Parse options from the command-line and set corresponding vars.
    // On error, return false and set error_msg;
    bool parse(int argc, char** argv,
               StencilContext& context) {

        // First, parse cmd-line options provided by the Settings class.
        // Any remaining strings will be left in args.
        vector<string> args;
        context.parseArgs(argc, argv, args);

        // Process remaining args added for this app.
        for (int argi = 0; argi < args.size(); argi++) {
            string& opt = args[argi];
            if (opt.length() >= 2 && opt[0] == '-') {

                if (opt == "-h" || opt == "-help" || opt == "--help")
                    help = true;
                
                else if (opt == "-no_warmup")
                    doWarmup = false;

                else if (opt == "-validate") {
                    validate = true;
                }

                else if (opt == "-v") {
                    validate = true;
                    doWarmup = false;
                    context.dx = context.dy = context.dz = 64;
                    context.dt = 1;
                    num_trials = 1;
                }

                else if (opt == "-t")
                    num_trials = context.int_val(args, argi);
                
                else if (opt == "-sleep")
                    pre_trial_sleep_time = context.int_val(args, argi);
            
                else {
                    error_msg = "option '" + opt + "' not recognized";
                    parse_ok = false;
                    break;
                }
            }
            else {
                error_msg = "extraneous parameter '" + args[argi] + "'";
                parse_ok = false;
                break;
            }
        }
        return parse_ok;
    }
    
    // Print usage.
    void usage(ostream& os, const string& pgmName, StencilContext& context)
    {
        ostringstream oss;
        oss <<
            " -t <n>           number of trials, default=" <<
            num_trials << endl <<
            " -validate        validate by comparing to a scalar run (see notes below)\n" <<
            " -v               shorthand for '-validate -no_warmup -d 64 -dt 1 -t 1\n" <<
            " -sleep <n>       seconds to sleep before each trial, default=" <<
            pre_trial_sleep_time << endl <<
            " -no_warmup       skip warmup iterations\n"
            " -h               print usage and exit\n";
        string appOptions = oss.str();
        string appNotes =
            " Validation is very slow and uses 2x memory,\n"
            "  so run with very small sizes and number of time-steps.\n"
            "  If validation fails, it may be due to rounding error;\n"
            "  try building with 8-byte reals.\n";

        context.print_usage(os, pgmName, appOptions, appNotes);
    }

    // Print splash banner and invocation string.
    // Exit with help message if requested.
    void splash(int argc, char** argv,
                StencilContext& context)
    {
        ostream& os = context.get_ostr();
        os <<
            "┌──────────────────────────────────────────┐\n"
            "│  Y.A.S.K. ── Yet Another Stencil Kernel  │\n"
            "│            https://01.org/yask           │\n"
            "│    Intel Corporation, copyright 2017     │\n"
            "└──────────────────────────────────────────┘\n"
            "\nStencil name: " YASK_STENCIL_NAME << endl;

        // Echo invocation parameters for record-keeping.
        os << "Invocation:";
        for (int argi = 0; argi < argc; argi++)
            os << " " << argv[argi];
        os << endl;
        
        // Exit due to help request or error msg.
        if (help || !parse_ok) {

            // It's possible help was requested or error appeard on some rank
            // that is not the msg-rank. So, we print on all ranks to make sure
            // some message is printed, but we delay output on non-msg ranks to
            // reduce clutter in the normal case.
            if (context.my_rank != context.msg_rank)
                sleep(1);

            if (!parse_ok)
                cerr << "\nError on command-line: " << error_msg << endl;
            if (help) {
                usage(cerr, argv[0], context);
                cerr << "\nExiting due to help request." << endl;
            }
            exit(1);
        }
    }
};

// Parse command-line args, run kernel, run validation if requested.
int main(int argc, char** argv)
{
    // Object containing data and parameters for stencil eval.
    YASK_STENCIL_CONTEXT context;

    // Parse cmd-line options.
    AppOptions opts;
    opts.parse(argc, argv, context);

    // Init MPI, OMP, etc.
    context.initEnv(&argc, &argv);
    ostream& os = context.get_ostr();

    // Splash banner, etc.
    opts.splash(argc, argv, context);

    // Alloc memory, create lists of grids, etc.
    context.allocAll();

    // Exit if nothing to do.
    if (opts.num_trials < 1) {
        cerr << "Exiting because no trials are specified." << endl;
        exit_yask(1);
    }
    if (context.tot_numpts_dt < 1) {
        cerr << "Exiting because there are zero points to evaluate." << endl;
        exit_yask(1);
    }

    // init data in grids and params.
    context.initData();

    // just a line.
    string divLine;
    for (int i = 0; i < 60; i++)
        divLine += "─";
    divLine += "\n";
    
    // warmup caches, threading, etc.
    if (opts.doWarmup) {

        // Temporarily set dt to a temp value for warmup.
        idx_t dt = context.dt;
        idx_t tmp_dt = min<idx_t>(context.dt, TIME_DIM_SIZE);
        context.dt = tmp_dt;

        os << endl << divLine <<
            "Running " << context.dt << " time step(s) for warm-up...\n" << flush;
        context.calc_rank_opt();

        // Replace temp setting with correct value.
        context.dt = dt;
        context.global_barrier();
    }

    // variables for measuring performance.
    double wstart, wstop;
    float best_elapsed_time=0.0f, best_apps=0.0f, best_dpps=0.0f, best_flops=0.0f;

    // Performance runs.
    os << endl << divLine <<
        "Running " << opts.num_trials << " performance trial(s) of " <<
        context.dt << " time step(s) each...\n";
    for (idx_t tr = 0; tr < opts.num_trials; tr++) {
        os << divLine;

        // init data befor each trial for comparison if validating.
        if (opts.validate)
            context.initDiff();

        // Stabilize.
        os << flush;
        if (opts.pre_trial_sleep_time > 0)
            sleep(opts.pre_trial_sleep_time);
        context.global_barrier();

        // Start timing.
        VTUNE_RESUME;
        context.mpi_time = 0.0;
        wstart = getTimeInSecs();

        // Actual work (must wait until all ranks are done).
        context.calc_rank_opt();
        context.global_barrier();

        // Stop timing.
        wstop =  getTimeInSecs();
        VTUNE_PAUSE;
            
        // calc and report perf.
        float elapsed_time = (float)(wstop - wstart);
        float apps = float(context.tot_numpts_dt) / elapsed_time;
        float dpps = float(context.tot_domain_dt) / elapsed_time;
        float flops = float(context.tot_numFpOps_dt) / elapsed_time;
        os << 
            "time (sec):                             " << printWithPow10Multiplier(elapsed_time) << endl <<
            "throughput (prob-size-points/sec):      " << printWithPow10Multiplier(dpps) << endl <<
            "throughput (points-updated/sec):        " << printWithPow10Multiplier(apps) << endl <<
            "throughput (est-FLOPS):                 " << printWithPow10Multiplier(flops) << endl;
#ifdef USE_MPI
        os <<
            "time in halo exch (sec):                " << printWithPow10Multiplier(context.mpi_time) << endl;
#endif

        if (apps > best_apps) {
            best_apps = apps;
            best_dpps = dpps;
            best_elapsed_time = elapsed_time;
            best_flops = flops;
        }
    }

    os << divLine <<
        "best-time (sec):                        " << printWithPow10Multiplier(best_elapsed_time) << endl <<
        "best-throughput (prob-size-points/sec): " << printWithPow10Multiplier(best_dpps) << endl <<
        "best-throughput (points-updated/sec):   " << printWithPow10Multiplier(best_apps) << endl <<
        "best-throughput (est-FLOPS):            " << printWithPow10Multiplier(best_flops) << endl <<
        divLine <<
        "Notes:\n" <<
        " prob-size-points/sec is based on problem-size as described above.\n" <<
        " points-updated/sec is based on grid-points-updated as described above.\n" <<
        " est-FLOPS is based on est-FP-ops as described above.\n" <<
        endl;
    
    if (opts.validate) {
        context.global_barrier();
        os << endl << divLine <<
            "Setup for validation...\n";

        // Make a reference context for comparisons w/new grids:
        // Copy the settings from context, then re-alloc grids.
        YASK_STENCIL_CONTEXT ref_context = context;
        ref_context.name += "-reference";
        ref_context.allocAll();

        // init to same value used in context.
        ref_context.initDiff();

        // Debug code to determine if data immediately after init matches.
#if CHECK_INIT
        {
            context.initDiff();
            idx_t errs = context.compareData(ref_context);
            if( errs == 0 ) {
                os << "INIT CHECK PASSED." << endl;
                exit_yask(0);
            } else {
                cerr << "INIT CHECK FAILED: " << errs << " mismatch(es)." << endl;
                exit_yask(1);
            }
        }
#endif

        // Ref trial.
        os << endl << divLine <<
            "Running " << context.dt << " time step(s) for validation...\n" << flush;
        ref_context.calc_rank_ref();

        // check for equality.
        context.global_barrier();
        os << "Checking results on rank " << context.my_rank << "..." << endl;
        idx_t errs = context.compareData(ref_context);
        if( errs == 0 ) {
            os << "TEST PASSED." << endl;
        } else {
            cerr << "TEST FAILED: " << errs << " mismatch(es)." << endl;
            if (REAL_BYTES < 8)
                cerr << "This is not uncommon for low-precision FP; try with 8-byte reals." << endl;
            exit_yask(1);
        }
    }
    else
        os << "\nRESULTS NOT VERIFIED.\n";

    context.global_barrier();
    MPI_Finalize();
    os << "YASK DONE." << endl << divLine;
    
    return 0;
}
