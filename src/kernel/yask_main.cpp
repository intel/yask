/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

#include "yask_stencil.hpp"
using namespace std;
using namespace yask;

// Auto-generated stencil code that extends base types.
#define DEFINE_CONTEXT
#include YSTR2(YK_CODE_FILE)

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
    int trial_steps = 0;        // number of steps in each trial.
    int trial_time = 10;        // sec to run each trial if trial_steps == 0.
    int pre_trial_sleep_time = 1; // sec to sleep before each trial.
    int debug_sleep = 0;          // sec to sleep for debug attach.

    AppSettings(DimsPtr dims, KernelEnvPtr env) :
        KernelSettings(dims, env) { }

    // A custom option-handler for '-v'.
    class ValOption : public CommandLineParser::OptionBase {
        AppSettings& _as;

    public:

        ValOption(AppSettings& as) :
                OptionBase("v",
                           "Minimal validation: shortcut for '-validate -no-pre-auto_tune -no-auto_tune"
                           " -no-warmup -t 1 -trial_steps 1 -d 63 -b 24'."),
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
                _as.trial_steps = 1;
                for (auto dim : _as._dims->_domain_dims.getDims()) {
                    auto& dname = dim.getName();
                    _as._rank_sizes[dname] = 63;
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
                          ("trial_steps",
                           "Number of steps to run each performance trial. "
                           "If zero, the 'trial_time' value is used.",
                           trial_steps));
        parser.add_option(new CommandLineParser::IntOption
                          ("dt",
                           "Same as 'trial_steps'; for backward-compatibility.",
                           trial_steps));
        parser.add_option(new CommandLineParser::IntOption
                          ("trial_time",
                           "Approximate number of seconds to run each performance trial. "
                           "When the 'trial_steps' value is zero, the number of steps is "
                           "based on the rate measured in the warmup phase. "
                           "(Thus, warmup cannot be disabled when the number of steps is zero.)",
                           trial_time));
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
            yask_exception e;
            stringstream err;
            err << "Error: extraneous parameter(s):";
            for (auto arg : args)
                err << " '" << arg << "'";
            err << "; run with '-help' option for usage";
            THROW_YASK_EXCEPTION(err.str());
        }
    }

    // Print splash banner and invocation string.
    // Exit with help message if requested.
    void splash(ostream& os, int argc, char** argv)
    {
        os <<
            "┌────────────────────────────────────────────┐\n"
            "│   Y.A.S.K. ── Yet Another Stencil Kernel   │\n"
            "│       https://github.com/intel/yask        │\n"
            "│ Copyright (c) 2014-2018, Intel Corporation │\n"
            "└────────────────────────────────────────────┘\n"
            "\n"
            "Version: " << yask_get_version_string() << endl <<
            "Stencil name: " YASK_STENCIL_NAME << endl;

        // Echo invocation parameters for record-keeping.
        os << "Default arguments: " DEF_ARGS << endl;
        os << "Binary invocation:";
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
    // just a line.
    string divLine;
    for (int i = 0; i < 70; i++)
        divLine += "─";
    divLine += "\n";

    try {
        // Stop collecting VTune data.
        // Even better to use -start-paused option.
        VTUNE_PAUSE;

        // Bootstrap factories from kernel API.
        yk_factory kfac;
        yask_output_factory yof;

        // Set up the environment (mostly MPI).
        auto kenv = kfac.new_env();
        auto ep = dynamic_pointer_cast<KernelEnv>(kenv);
        auto num_ranks = kenv->get_num_ranks();

        // Problem dimensions.
        auto dims = YASK_STENCIL_CONTEXT::new_dims();

        // Parse cmd-line options.
        // TODO: do this through APIs.
        auto opts = make_shared<AppSettings>(dims, ep);
        opts->parse(argc, argv);

        // Make sure warmup is on if needed.
        if (opts->trial_steps <= 0 && opts->trial_time > 0)
            opts->doWarmup = true;

        // Object containing data and parameters for stencil eval.
        // TODO: do everything through API without cast to StencilContext.
        auto ksoln = kfac.new_solution(kenv);
        auto context = dynamic_pointer_cast<StencilContext>(ksoln);
        assert(context.get());

        // Replace the default settings with 'opts'.
        context->set_settings(opts);

        // Make sure any MPI/OMP debug data is dumped from all ranks before continuing.
        kenv->global_barrier();

        // Print splash banner and related info.
        ostream& os = context->set_ostr();
        opts->splash(os, argc, argv);

        // Override alloc if requested.
        alloc_steps(ksoln, *opts);

        // Alloc memory, etc.
        ksoln->prepare_solution();

        // Exit if nothing to do.
        if (context->rank_bb.bb_num_points < 1)
            THROW_YASK_EXCEPTION("Exiting because there are no points in the domain");

        // init data in grids and params.
        if (opts->doWarmup || !opts->validate)
            context->initData();

        // Invoke auto-tuner.
        if (opts->doPreAutoTune)
            ksoln->run_auto_tuner_now();

        // Enable/disable further auto-tuning.
        ksoln->reset_auto_tuner(opts->doAutoTune);

        // Warmup caches, threading, etc.
        // Measure time to change number of steps.
        if (opts->doWarmup) {

            // Turn off debug.
            auto dbg_out = context->get_debug_output();
            context->set_debug_output(yof.new_null_output());
            os << endl << divLine;

            // Warmup phases.
            double rate = 1.0;
            idx_t warmup_steps = 1;
            idx_t max_wsteps = 10;
            for (int n = 0; n < 3; n++) {

                // Run steps.
                // Always run warmup forward, even for reverse stencils.
                // (The result will be meaningless, but that doesn't matter.)
                os << "Running " << warmup_steps << " step(s) for " <<
                    (n ? "calibration" : "warm-up") << "...\n" << flush;
                kenv->global_barrier();
                ksoln->run_solution(0, warmup_steps-1);
                kenv->global_barrier();
                auto stats = context->get_stats();
                auto wtime = stats->get_elapsed_secs();
                os << "  Done in " << makeNumStr(wtime) << " secs.\n";
                rate = (wtime > 0.) ? double(warmup_steps) / wtime : 0;

                // Done if time est. isn't needed.
                if (opts->trial_steps > 0)
                    break;

                // Use time to set number of steps for next trial.
                double warmup_time = 0.5 * (n + 1);
                warmup_steps = ceil(rate * warmup_time);
                warmup_steps = min(warmup_steps, max_wsteps);
                max_wsteps *= max_wsteps;

                // Average across all ranks because it is critical that
                // all ranks use the same number of steps to avoid deadlock.
                warmup_steps = CEIL_DIV(sumOverRanks(warmup_steps, ep->comm), num_ranks);
            }

            // Set final number of steps.
            if (opts->trial_steps <= 0) {
                idx_t tsteps = ceil(rate * opts->trial_time);
                tsteps = CEIL_DIV(sumOverRanks(tsteps, ep->comm), num_ranks);
                opts->trial_steps = tsteps;
            }
            
            // Restore debug.
            context->set_debug_output(dbg_out);
        }
        kenv->global_barrier();

        // Exit if nothing to do.
        if (opts->num_trials < 1)
            THROW_YASK_EXCEPTION("Exiting because zero trials are specified");
        if (opts->trial_steps <= 0)
            THROW_YASK_EXCEPTION("Exiting because zero steps per trial are specified");

        // Track best trial.
        shared_ptr<Stats> best_trial;

        // First & last steps.
        idx_t first_t = 0;
        idx_t last_t = opts->trial_steps - 1;

        // Stencils seem to be backward?
        // (This is just a heuristic, but the direction
        // is not usually critical to perf measurement.)
        if (opts->_dims->_step_dir < 0) {
            first_t = last_t;
            last_t = 0;
        }
        
        /////// Performance run(s).
        os << endl << divLine <<
            "Running " << opts->num_trials << " performance trial(s) of " <<
            opts->trial_steps << " step(s) each...\n" << flush;
        for (idx_t tr = 0; tr < opts->num_trials; tr++) {
            os << divLine <<
                "Trial number:                      " << (tr + 1) << endl << flush;

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
            ksoln->run_solution(first_t, last_t);
            kenv->global_barrier();

            // Stop vtune collection.
            VTUNE_PAUSE;

            // Calc and report perf.
            auto trial_stats = context->get_stats();
            auto stats = dynamic_pointer_cast<Stats>(trial_stats);

            // Remember best.
            if (best_trial == nullptr || stats->run_time < best_trial->run_time)
                best_trial = stats;
        }

        if (best_trial != nullptr) {
            os << divLine <<
                "Performance stats of best trial:\n"
                " best-num-steps-done:              " << makeNumStr(best_trial->nsteps) << endl <<
                " best-elapsed-time (sec):          " << makeNumStr(best_trial->run_time) << endl <<
                " best-throughput (num-reads/sec):  " << makeNumStr(best_trial->reads_ps) << endl <<
                " best-throughput (num-writes/sec): " << makeNumStr(best_trial->writes_ps) << endl <<
                " best-throughput (est-FLOPS):      " << makeNumStr(best_trial->flops) << endl <<
                " best-throughput (num-points/sec): " << makeNumStr(best_trial->pts_ps) << endl <<
                divLine <<
                "Notes:\n"
                " Num-reads and writes/sec and FLOPS are metrics based on\n"
                "  stencil specifications and can vary due to differences in\n"
                "  implementations and optimizations.\n"
                " Num-points/sec is based on overall problem size and is\n"
                "  a more reliable performance metric, esp. when comparing\n"
                "  across implementations.\n";
        }

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
            auto& ref_opts = ref_context->get_settings();

            // Change some settings.
            ref_context->name += "-reference";
            ref_context->allow_vec_exchange = false;   // exchange scalars in halos.
            ref_opts->overlap_comms = false;
            ref_opts->use_shm = false;
            ref_opts->_numa_pref = yask_numa_none;

            // TODO: re-enable the region and block settings below;
            // requires allowing consistent init of different-sized grids
            // in kernel code.
#if 0
            auto sdim = ref_soln->get_step_dim_name();
            ref_soln->set_region_size(sdim, 0);
            ref_soln->set_block_size(sdim, 0);
            for (auto ddim : ref_soln->get_domain_dim_names()) {
                ref_soln->set_region_size(ddim, 0);
                ref_soln->set_block_size(ddim, 0);
            }
#endif

            // Override allocations and prep solution as with ref soln.
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
            // Do same number as last perf run.
            os << endl << divLine <<
                "Running " << opts->trial_steps << " step(s) for validation...\n" << flush;
            ref_context->run_ref(first_t, last_t);

            // Discard perf report.
            auto dbg_out = ref_context->get_debug_output();
            ref_context->set_debug_output(yof.new_null_output());
            auto rstats = ref_context->get_stats();
            ref_context->set_debug_output(dbg_out);
            os << "  Done in " << makeNumStr(rstats->get_elapsed_secs()) << " secs.\n";
#endif
            // check for equality.
            os << "\nChecking results..." << endl;
            idx_t errs = context->compareData(*ref_context);
            auto ri = kenv->get_rank_index();
            if( errs == 0 ) {
                os << "TEST PASSED on rank " << ri << ".\n" << flush;
            } else {
                cerr << "TEST FAILED on rank " << ri << ": >= " << errs << " mismatch(es).\n" << flush;
                if (REAL_BYTES < 8)
                    cerr << "Small differences are not uncommon for low-precision FP; try with 8-byte reals." << endl;
                ok = false;
            }
            ref_soln->end_solution();
        }
        else
            os << "\nRESULTS NOT VERIFIED.\n";
        ksoln->end_solution();

        os << "Stencil '" << ksoln->get_name() << "'.\n";
        if (!ok)
            exit_yask(1);

        MPI_Finalize();
        os << "YASK DONE." << endl << divLine << flush;
    }
    catch (yask_exception e) {
        cerr << "YASK Kernel: " << e.get_message() << ".\n";
        exit_yask(1);
    }

    return 0;
}
