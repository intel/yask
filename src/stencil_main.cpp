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

using namespace std;
using namespace yask;

// Add some command-line options for this application to the
// default ones provided by YASK.
struct AppSettings : public StencilSettings {
    bool help;                  // help requested.
    bool doWarmup;              // whether to do warmup run.
    int num_trials;             // number of trials.
    bool validate;              // whether to do validation run.
    bool dump;                  // whether to dump grids to disk.
    int pre_trial_sleep_time;   // sec to sleep before each trial.

    AppSettings() :
        help(false),
        doWarmup(true),
        num_trials(3),
        validate(false),
        dump(false),
        pre_trial_sleep_time(1)
    { }

    // A custom option-handler for validation.
    class ValOption : public CommandLineParser::OptionBase {
        AppSettings& _as;

    public:
        ValOption(AppSettings& as) :
                OptionBase("v",
                           "Shortcut for -validate -no-warmup -t 1 -dt 1 -d 64 -b 24."),
                _as(as) { }

        // Set multiple vars.
        virtual bool check_arg(std::vector<std::string>& args,
                               int& argi) {
            if (_check_arg(args, argi, _name)) {
                _as.validate = true;
                _as.dump     = false;
                _as.doWarmup = false;
                _as.num_trials = 1;
                _as.dt = 1;
                _as.dx = _as.dy = _as.dz = 64;
                _as.bx = _as.by = _as.bz = 24;
                return true;
            }
            return false;
        }
    };

    // Parse options from the command-line and set corresponding vars.
    // Exit with message on error or request for help.
    void parse(int argc, char** argv) {

        // Create a parser and add base options to it.
        CommandLineParser parser;
        add_options(parser);

        // Add more options for this app.
        parser.add_option(new CommandLineParser::BoolOption("h",
                                         "Print help message.",
                                         help));
        parser.add_option(new CommandLineParser::BoolOption("help",
                                         "Print help message.",
                                         help));
        parser.add_option(new CommandLineParser::BoolOption("warmup",
                                         "Run warmup iteration(s) before performance trial(s).",
                                         doWarmup));
        parser.add_option(new CommandLineParser::IntOption("t",
                                        "Number of performance trials.",
                                        num_trials));
        parser.add_option(new CommandLineParser::IntOption("sleep",
                                        "Number of seconds to sleep before each performance trial.",
                                        pre_trial_sleep_time));
        parser.add_option(new CommandLineParser::BoolOption("validate",
                                         "Run validation iteration(s) after performance trial(s).",
                                         validate));
        parser.add_option(new CommandLineParser::BoolOption("dump",
                                         "Dump grids to disk.",
                                         dump));
        parser.add_option(new ValOption(*this));
        
        // Parse cmd-line options.
        // Any remaining strings will be left in args.
        vector<string> args;
        parser.parse_args(argc, argv, args);

        if (help) {
            string appNotes =
                " Validation is very slow and uses 2x memory,\n"
                "  so run with very small sizes and number of time-steps.\n"
                "  If validation fails, it may be due to rounding error;\n"
                "   try building with 8-byte reals.\n";
            vector<string> appExamples;
            appExamples.push_back("-t 2");
            appExamples.push_back("-v");
            print_usage(cout, parser, argv[0], appNotes, appExamples);
            exit_yask(1);
        }
        
        if (args.size()) {
            cerr << "Error: extraneous parameter(s):";
            for (auto arg : args)
                cerr << " '" << arg << "'";
            cerr << ".\nRun with '-help' option for usage.\n" << flush;
            exit_yask(1);
        }
    }
    
    // Print splash banner and invocation string.
    // Exit with help message if requested.
    void splash(ostream& os, int argc, char** argv)
    {
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
    }
};

// Parse command-line args, run kernel, run validation if requested.
int main(int argc, char** argv)
{
    // Parse cmd-line options.
    AppSettings opts;
    opts.parse(argc, argv);

    // Object containing data and parameters for stencil eval.
    YASK_STENCIL_CONTEXT context(opts);

    // Init MPI, OMP, etc.
    context.initEnv(&argc, &argv);

    // Print splash banner and related info.
    ostream& os = context.get_ostr();
    opts.splash(os, argc, argv);

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

    if (opts.dump)
      context.dump_grids( std::string( "dump_ini" ) );

    // just a line.
    string divLine;
    for (int i = 0; i < 60; i++)
        divLine += "─";
    divLine += "\n";
    
    // warmup caches, threading, etc.
    if (opts.doWarmup) {

        // Temporarily set dt to a temp value for warmup.
        idx_t dt = opts.dt;
        idx_t tmp_dt = min<idx_t>(opts.dt, 1);
        opts.dt = tmp_dt;

        os << endl << divLine <<
            "Running " << opts.dt << " time step(s) for warm-up...\n" << flush;
        context.calc_rank_opt();

        // Replace temp setting with correct value.
        opts.dt = dt;
        context.global_barrier();
    }

    // variables for measuring performance.
    double wstart, wstop;
    float best_elapsed_time=0.0f, best_apps=0.0f, best_dpps=0.0f, best_flops=0.0f;

    // Performance runs.
    os << endl << divLine <<
        "Running " << opts.num_trials << " performance trial(s) of " <<
        opts.dt << " time step(s) each...\n";
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

        // Dump grids to disk
        if (opts.dump)
            context.dump_grids( std::string( "dump_fin" ) );

        // Calc and report perf.
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

        // Make a reference context for comparisons w/new grids.
        YASK_STENCIL_CONTEXT ref_context(opts);
        ref_context.name += "-reference";
        ref_context.copyEnv(context);
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
            "Running " << opts.dt << " time step(s) for validation...\n" << flush;
        ref_context.calc_rank_ref();

        // Dump grids to disk
        if (opts.dump)
            ref_context.dump_grids( std::string( "dump_fin" ) );

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
