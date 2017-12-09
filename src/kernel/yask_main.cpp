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
    bool doPreAutoTune = true;  // whether to do pre-auto-tuning.
    bool doAutoTune = false;    // whether to do auto-tuning.
    int step_alloc = 0;         // if >0, override number of steps to alloc.
    int num_trials = 3;         // number of trials.
    bool validate = false;      // whether to do validation run.
    int pre_trial_sleep_time = 1; // sec to sleep before each trial.
    int debug_sleep = 0;          // sec to sleep for debug attach.

    AppSettings(DimsPtr dims) :
        KernelSettings(dims) { }

    // A custom option-handler for '-v'.
    class ValOption : public CommandLineParser::OptionBase {
        AppSettings& _as;

    public:

        ValOption(AppSettings& as) :
                OptionBase("v",
                           "Shortcut for '-validate -no-pre-auto_tune -no-auto_tune -no-warmup -t 1 -dt 1 -d 64 -b 32'."),
                _as(as) { }

        // Set multiple vars.
        virtual bool check_arg(std::vector<std::string>& args,
                               int& argi) {
            if (_check_arg(args, argi, _name)) {
                _as.validate = true;
                _as.doPreAutoTune = false;
                _as.doAutoTune = false;
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
        parser.add_option(new CommandLineParser::BoolOption
                          ("h",
                           "Print help message.",
                           help));
        parser.add_option(new CommandLineParser::BoolOption
                          ("help",
                           "Print help message.",
                           help));
        parser.add_option(new CommandLineParser::BoolOption
                          ("pre_auto_tune",
                           "Run iteration(s) *before* performance trial(s) to find good-performing "
                           "values for block sizes. "
                           "Uses default values or command-line-provided values as a starting point.",
                           doPreAutoTune));
        parser.add_option(new CommandLineParser::BoolOption
                          ("auto_tune",
                           "Run iteration(s) *during* performance trial(s) to find good-performing "
                           "values for block sizes. "
                           "Uses default values or command-line-provided values as a starting point.",
                           doAutoTune));
        parser.add_option(new CommandLineParser::BoolOption
                          ("warmup",
                           "Run warmup iteration(s) before performance "
                           "trial(s) and after auto-tuning iterations, if enabled.",
                           doWarmup));
        parser.add_option(new CommandLineParser::IntOption
                          ("step_alloc",
                           "Number of steps to allocate in relevant grids, "
                           "overriding default value from YASK compiler.",
                           step_alloc));
        parser.add_option(new CommandLineParser::IntOption
                          ("t",
                           "Number of performance trials.",
                           num_trials));
        parser.add_option(new CommandLineParser::IntOption
                          ("sleep",
                           "Number of seconds to sleep before each performance trial.",
                           pre_trial_sleep_time));
        parser.add_option(new CommandLineParser::IntOption
                          ("debug_sleep",
                           "Number of seconds to sleep for debug attach.",
                           debug_sleep));
        parser.add_option(new CommandLineParser::BoolOption
                          ("validate",
                           "Run validation iteration(s) after performance trial(s).",
                           validate));
        parser.add_option(new ValOption(*this));
        
        // Tokenize default args.
        vector<string> args;
        parser.set_args(DEF_ARGS, args);

        // Parse cmd-line options, which sets values.
        // Any remaining strings will be left in args.
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
            "\n"
            "Version: " << yask_get_version_string() << endl <<
            "Stencil name: " YASK_STENCIL_NAME << endl;

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

// Override step allocation.
void alloc_steps(yk_solution_ptr soln, const AppSettings& opts) {
    if (opts.step_alloc <= 0)
        return;

    // Find grids with steps.
    auto step_dim = soln->get_step_dim_name();
    auto grids = soln->get_grids();
    for (auto grid : grids) {
        if (grid->is_dim_used(step_dim))

            // override num steps.
            grid->set_alloc_size(step_dim, opts.step_alloc);
    }
}

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

    // Override alloc if requested.
    alloc_steps(ksoln, *opts);
    
    // Alloc memory, etc.
    ksoln->prepare_solution();

    // Exit if nothing to do.
    if (opts->num_trials < 1) {
        cerr << "Exiting because no trials are specified." << endl;
        exit_yask(1);
    }
    if (context->bb_num_points < 1) {
        cerr << "Exiting because there are no points in the domain." << endl;
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

    // Invoke auto-tuner.
    if (opts->doPreAutoTune)
        ksoln->run_auto_tuner_now();

    // Enable/disable further auto-tuning.
    ksoln->reset_auto_tuner(opts->doAutoTune);

    // warmup caches, threading, etc.
    if (opts->doWarmup) {

        idx_t dt = 1;
        os << endl << divLine <<
            "Running " << dt << " step(s) for warm-up...\n" << flush;
        ksoln->run_solution(0, dt-1);

    }
    kenv->global_barrier();

    // variables for measuring performance.
    double best_elapsed_time=0., best_apps=0., best_dpps=0., best_flops=0.;

    /////// Performance run(s).
    auto& step_dim = opts->_dims->_step_dim;
    idx_t dt = opts->_rank_sizes[step_dim];
    os << endl << divLine <<
        "Running " << opts->num_trials << " performance trial(s) of " <<
        dt << " step(s) each...\n" << flush;
    for (idx_t tr = 0; tr < opts->num_trials; tr++) {
        os << divLine << flush;

        // init data before each trial for comparison if validating.
        if (opts->validate)
            context->initDiff();

        // Warn if tuning.
        if (ksoln->is_auto_tuner_enabled())
            os << "auto-tuner is active during this trial, so results may not be representative.\n";

        // Stabilize.
        if (opts->pre_trial_sleep_time > 0) {
            os << flush;
            sleep(opts->pre_trial_sleep_time);
        }
        kenv->global_barrier();

        // Start vtune collection.
        VTUNE_RESUME;

        // Actual work.
        context->clear_timers();
        context->calc_rank_opt();

        // Stop vtune collection.
        VTUNE_PAUSE;
            
        // Calc and report perf.
        auto stats = context->get_stats();

        // Remember best.
        if (context->domain_pts_ps > best_dpps) {
            best_dpps = context->domain_pts_ps;
            best_apps = context->writes_ps;
            best_flops = context->flops;
            best_elapsed_time = stats->get_elapsed_run_secs();
        }
    }

    os << divLine <<
        "best-elapsed-time (sec):           " << makeNumStr(best_elapsed_time) << endl <<
        "best-throughput (num-points/sec):  " << makeNumStr(best_dpps) << endl <<
        "best-throughput (est-FLOPS):       " << makeNumStr(best_flops) << endl <<
        "best-throughput (num-writes/sec):  " << makeNumStr(best_apps) << endl <<
        divLine <<
        "Notes:\n" <<
        " Num-points is based on overall-problem-size as described above.\n" <<
        " Num-writes is based on num-writes-required as described above.\n" <<
        " Est-FLOPS is based on est-FP-ops as described above.\n" <<
        endl;
    
    /////// Validation run.
    bool ok = true;
    if (opts->validate) {
        kenv->global_barrier();
        os << endl << divLine <<
            "Setup for validation...\n";

        // Make a reference context for comparisons w/new grids.
        auto ref_soln = kfac.new_solution(kenv, ksoln);
        auto ref_context = dynamic_pointer_cast<StencilContext>(ref_soln);
        assert(ref_context.get());
        ref_context->name += "-reference";
        ref_context->allow_vec_exchange = false;
        alloc_steps(ref_soln, *opts);
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
        auto ri = kenv->get_rank_index();
        if( errs == 0 ) {
            os << "TEST PASSED on rank " << ri << ".\n" << flush;
        } else {
            cerr << "TEST FAILED on rank " << ri << ": >= " << errs << " mismatch(es).\n" << flush;
            if (REAL_BYTES < 8)
                cerr << "This is not uncommon for low-precision FP; try with 8-byte reals." << endl;
            ok = false;
        }
    }
    else
        os << "\nRESULTS NOT VERIFIED.\n";

    kenv->global_barrier();
    if (!ok)
        exit_yask(1);

    MPI_Finalize();
    os << "YASK DONE." << endl << divLine << flush;
    
    return 0;
}
