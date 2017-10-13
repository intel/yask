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

// This code implements the YASK stand-alone performance-measurement tool.

#include "yask.hpp"
using namespace std;
using namespace yask;

// Add some command-line options for this application in addition to the
// default ones provided by YASK.
struct AppSettings : public KernelSettings {
    bool help = false;          // help requested.
    bool doWarmup = true;       // whether to do warmup run.
    int num_trials = 3;         // number of trials.
    bool validate = false;      // whether to do validation run.
    int pre_trial_sleep_time = 1; // sec to sleep before each trial.
    int debug_sleep = 0;          // sec to sleep for debug attach.

    AppSettings(DimsPtr dims) :
        KernelSettings(dims) { }

    // A custom option-handler for validation.
    class ValOption : public CommandLineParser::OptionBase {
        AppSettings& _as;

    public:
        ValOption(AppSettings& as) :
                OptionBase("v",
                           "Shortcut for '-validate -no-warmup -t 1 -dt 1 -d 64 -b 32'."),
                _as(as) { }

        // Set multiple vars.
        virtual bool check_arg(std::vector<std::string>& args,
                               int& argi) {
            if (_check_arg(args, argi, _name)) {
                _as.validate = true;
                _as.doWarmup = false;
                _as.num_trials = 1;
                _as._rank_sizes[_as._dims->_step_dim] = 1;
                for (auto dim : _as._dims->_domain_dims.getDims()) {
                    auto& dname = dim.getName();
                    _as._rank_sizes[dname] = 64;
                    _as._block_sizes[dname] = 24;
                }
                return true;
            }
            return false;
        }
    };

#ifndef DEF_ARGS
#define DEF_ARGS ""
#endif
    
    // Parse options from the command-line and set corresponding vars.
    // Exit with message on error or request for help.
    void parse(int argc, char** argv) {

        // Set default for time-domain size.
        auto& step_dim = _dims->_step_dim;
        _rank_sizes[step_dim] = 50;

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
        parser.add_option(new CommandLineParser::IntOption("debug_sleep",
                                        "Number of seconds to sleep for debug attach.",
                                        debug_sleep));
        parser.add_option(new CommandLineParser::BoolOption("validate",
                                         "Run validation iteration(s) after performance trial(s).",
                                         validate));
        parser.add_option(new ValOption(*this));
        
        // Parse cmd-line options, which sets values.
        // Any remaining strings will be left in args.
        vector<string> args;
        parser.set_args(DEF_ARGS, args);
        parser.parse_args(argc, argv, args);

        if (help) {
            string appNotes =
                "Validation is very slow and uses 2x memory,\n"
                " so run with very small sizes and number of time-steps.\n"
                " If validation fails, it may be due to rounding error;\n"
                "  try building with 8-byte reals.\n";
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
        os << "Default arguments: " DEF_ARGS << endl;
        os << "Invocation:";
        for (int argi = 0; argi < argc; argi++)
            os << " " << argv[argi];
        os << endl;
        
        os << "PID: " << getpid() << endl;
        if (debug_sleep) {
            os << "Sleeping " << debug_sleep <<
                " second(s) to allow debug attach...\n";
            sleep(debug_sleep);
            os << "Resuming...\n";
        }
    }
};

// Parse command-line args, run kernel, run validation if requested.
int main(int argc, char** argv)
{
    // Stop collecting VTune data.
    // Even better to use -start-paused option.
    VTUNE_PAUSE;

    // Bootstrap factory from kernel API.
    yk_factory kfac;

    // Set up the environment (mostly MPI).
    auto kenv = kfac.new_env();

    // Problem dimensions.
    auto dims = YASK_STENCIL_CONTEXT::new_dims();

    // Parse cmd-line options.
    // TODO: do this through APIs.
    auto opts = make_shared<AppSettings>(dims);
    opts->parse(argc, argv);

    // Object containing data and parameters for stencil eval.
    // TODO: do everything through API without cast to StencilContext.
    auto ksoln = kfac.new_solution(kenv);
    auto context = dynamic_pointer_cast<StencilContext>(ksoln);
    assert(context.get());
    context->set_settings(opts);
    ostream& os = context->set_ostr();
    
    // Make sure any MPI/OMP debug data is dumped from all ranks before continuing.
    kenv->global_barrier();
    
    // Print splash banner and related info.
    opts->splash(os, argc, argv);

    // Alloc memory, etc.
    ksoln->prepare_solution();

    // Exit if nothing to do.
    if (opts->num_trials < 1) {
        cerr << "Exiting because no trials are specified." << endl;
        exit_yask(1);
    }
    if (context->tot_numpts_dt < 1) {
        cerr << "Exiting because there are zero points to evaluate." << endl;
        exit_yask(1);
    }

    // init data in grids and params.
    if (opts->doWarmup || !opts->validate)
        context->initData();

    // just a line.
    string divLine;
    for (int i = 0; i < 60; i++)
        divLine += "─";
    divLine += "\n";
    
    // warmup caches, threading, etc.
    if (opts->doWarmup) {

        idx_t dt = 1;
        os << endl << divLine <<
            "Running " << dt << " step(s) for warm-up...\n" << flush;
        context->run_solution(0, dt);

        kenv->global_barrier();
    }

    // variables for measuring performance.
    double wstart, wstop;
    float best_elapsed_time=0.0f, best_apps=0.0f, best_dpps=0.0f, best_flops=0.0f;

    /////// Performance run(s).
    auto& step_dim = opts->_dims->_step_dim;
    idx_t dt = opts->_rank_sizes[step_dim];
    os << endl << divLine <<
        "Running " << opts->num_trials << " performance trial(s) of " <<
        dt << " step(s) each...\n" << flush;
    for (idx_t tr = 0; tr < opts->num_trials; tr++) {
        os << divLine;

        // init data before each trial for comparison if validating.
        if (opts->validate)
            context->initDiff();

        // Stabilize.
        os << flush;
        if (opts->pre_trial_sleep_time > 0)
            sleep(opts->pre_trial_sleep_time);
        kenv->global_barrier();

        // Start timing.
        VTUNE_RESUME;
        context->mpi_time = 0.0;
        wstart = getTimeInSecs();

        // Actual work.
        context->calc_rank_opt();

        // Stop timing.
        wstop =  getTimeInSecs();
        VTUNE_PAUSE;
            
        // Calc and report perf.
        float elapsed_time = (float)(wstop - wstart);
        float apps = float(context->tot_numpts_dt) / elapsed_time;
        float dpps = float(context->tot_domain_dt) / elapsed_time;
        float flops = float(context->tot_numFpOps_dt) / elapsed_time;
        os << 
            "time (sec):                             " << makeNumStr(elapsed_time) << endl <<
            "throughput (prob-size-points/sec):      " << makeNumStr(dpps) << endl <<
            "throughput (point-updates/sec):         " << makeNumStr(apps) << endl <<
            "throughput (est-FLOPS):                 " << makeNumStr(flops) << endl;
#ifdef USE_MPI
        os <<
            "time in halo exch (sec):                " << makeNumStr(context->mpi_time);
        if (elapsed_time > 0.f) {
            float pct = 100.f * context->mpi_time / elapsed_time;
            os << " (" << pct << "%)";
        }
        os << endl;
#endif

        if (apps > best_apps) {
            best_apps = apps;
            best_dpps = dpps;
            best_elapsed_time = elapsed_time;
            best_flops = flops;
        }
    }

    os << divLine <<
        "best-time (sec):                        " << makeNumStr(best_elapsed_time) << endl <<
        "best-throughput (prob-size-points/sec): " << makeNumStr(best_dpps) << endl <<
        "best-throughput (point-updates/sec):    " << makeNumStr(best_apps) << endl <<
        "best-throughput (est-FLOPS):            " << makeNumStr(best_flops) << endl <<
        divLine <<
        "Notes:\n" <<
        " prob-size-points/sec is based on problem-size as described above.\n" <<
        " point-updates/sec is based on grid-point-updates as described above.\n" <<
        " est-FLOPS is based on est-FP-ops as described above.\n" <<
        endl;
    
    /////// Validation run.
    if (opts->validate) {
        kenv->global_barrier();
        os << endl << divLine <<
            "Setup for validation...\n";

        // Make a reference context for comparisons w/new grids.
        auto ref_soln = kfac.new_solution(kenv, ksoln);
        auto ref_context = dynamic_pointer_cast<StencilContext>(ref_soln);
        assert(ref_context.get());
        ref_context->name += "-reference";
        ref_soln->prepare_solution();

        // init to same value used in context.
        ref_context->initDiff();
        
#ifdef CHECK_INIT

        // Debug code to determine if data compares immediately after init matches.
        os << endl << divLine <<
            "Reinitializing data for minimal validation...\n" << flush;
        context->initDiff();
#else

        // Ref trial.
        os << endl << divLine <<
            "Running " << dt << " time step(s) for validation...\n" << flush;
        ref_context->calc_rank_ref();
#endif

        // check for equality.
        os << "Checking results..." << endl;
        idx_t errs = context->compareData(*ref_context);
        if( errs == 0 ) {
            os << "TEST PASSED." << endl;
        } else {
            cerr << "TEST FAILED: >= " << errs << " mismatch(es)." << endl;
            if (REAL_BYTES < 8)
                cerr << "This is not uncommon for low-precision FP; try with 8-byte reals." << endl;
            exit_yask(1);
        }
    }
    else
        os << "\nRESULTS NOT VERIFIED.\n";

    kenv->global_barrier();
    MPI_Finalize();
    os << "YASK DONE." << endl << divLine;
    
    return 0;
}
