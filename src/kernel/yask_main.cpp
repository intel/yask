/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2019, Intel Corporation

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
struct MySettings {
    bool help = false;          // help requested.
    bool doWarmup = true;       // whether to do warmup run.
    bool doPreAutoTune = true;  // whether to do pre-auto-tuning.
    int step_alloc = 0;         // if >0, override number of steps to alloc.
    int num_trials = 3;         // number of trials.
    bool validate = false;      // whether to do validation run.
    int trial_steps = 0;        // number of steps in each trial.
    int trial_time = 10;        // sec to run each trial if trial_steps == 0.
    int pre_trial_sleep_time = 1; // sec to sleep before each trial.
    int debug_sleep = 0;          // sec to sleep for debug attach.
    bool doTrace = false;         // tracing.
    int msgRank = 0;              // rank to print debug msgs.

    // Ptr to the soln.
    yk_solution_ptr _ksoln;

    MySettings(yk_solution_ptr ksoln) :
        _ksoln(ksoln) { }

    // A custom option-handler for '-v'.
    class ValOption : public CommandLineParser::OptionBase {
        MySettings& _as;
        static constexpr idx_t _lsz=63, _bsz=24;

    public:

        ValOption(MySettings& as) :
                OptionBase("v",
                           "Minimal validation: shortcut for '-validate -no-pre_auto_tune -no-auto_tune"
                           " -no-warmup -t 1 -trial_steps 1 -l " + to_string(_lsz) +
                           " -b " + to_string(_bsz) + "'."),
                _as(as) { }

        // Set multiple vars.
        virtual bool check_arg(const std::vector<std::string>& args,
                               int& argi) {
            if (_check_arg(args, argi, _name)) {
                _as.validate = true;
                _as.doPreAutoTune = false;
                _as.doWarmup = false;
                _as.num_trials = 1;
                _as.trial_steps = 1;

                // Create soln options and parse them if there is a soln.
                if (_as._ksoln) {
                    for (auto& dname : _as._ksoln->get_domain_dim_names()) {

                        // Local domain size, e.g., "-lx 63".
                        string arg = "-l" + dname + " " + to_string(_lsz);

                        // Block size, e.g., "-bx 24".
                        arg += " -b" + dname + " " + to_string(_bsz);

                        // Parse 'arg'.
                        auto rem = _as._ksoln->apply_command_line_options(arg);
                        assert(rem.length() == 0);
                    }
                }
                return true;
            }
            return false;
        }
    };

    // Parse options from the command-line and set corresponding vars.
    // Exit with message on error or request for help.
    void parse(int argc, char** argv) {

        // Create a parser and add options to it.
        CommandLineParser parser;
        parser.add_option(new CommandLineParser::BoolOption
                          ("help",
                           "Print help message.",
                           help));
        parser.add_option(new CommandLineParser::IntOption
                          ("msg_rank",
                           "Index of MPI rank that will print informational messages.",
                           msgRank));
        parser.add_option(new CommandLineParser::BoolOption
                          ("trace",
                           "Print internal debug messages if compiled with"
                           " general and/or memory-access tracing enabled.",
                           doTrace));
        parser.add_option(new CommandLineParser::BoolOption
                          ("pre_auto_tune",
                           "Run iteration(s) *before* performance trial(s) to find good-performing "
                           "values for block sizes. "
                           "Uses default values or command-line-provided values as a starting point.",
                           doPreAutoTune));
        parser.add_option(new CommandLineParser::BoolOption
                          ("warmup",
                           "Run warmup iteration(s) before performance "
                           "trial(s) and after auto-tuning iterations, if enabled.",
                           doWarmup));
        parser.add_option(new CommandLineParser::IntOption
                          ("step_alloc",
                           "Number of steps to allocate in relevant vars, "
                           "overriding default value from YASK compiler.",
                           step_alloc));
        parser.add_option(new CommandLineParser::IntOption
                          ("num_trials",
                           "Number of performance trials.",
                           num_trials));
        parser.add_option(new CommandLineParser::IntOption
                          ("t",
                           "Alias for '-num_trials'; for backward-compatibility.",
                           num_trials));
        parser.add_option(new CommandLineParser::IntOption
                          ("trial_steps",
                           "Number of steps to run each performance trial. "
                           "If zero, the 'trial_time' value is used.",
                           trial_steps));
        parser.add_option(new CommandLineParser::IntOption
                          ("dt",
                           "Alias for '-trial_steps'; for backward-compatibility.",
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

        // Parse 'args' and 'argv' cmd-line options, which sets values.
        // Any remaining strings will be returned.
        auto rem_args = parser.parse_args(argc, argv);

        // Handle additional knobs and help if there is a soln.
        if (_ksoln) {
        
            // Parse standard args not handled by this parser.
            rem_args = _ksoln->apply_command_line_options(rem_args);

            if (help) {
                string appNotes =
                    "\nValidation is very slow and uses 2x memory,\n"
                    " so run with very small sizes and number of time-steps.\n"
                    " If validation fails, it may be due to rounding error;\n"
                    " try building with 8-byte reals.\n";
                vector<string> appExamples;
                appExamples.push_back("-g 768 -num_trials 2");
                appExamples.push_back("-v");
                
                // TODO: make an API for this.
                auto context = dynamic_pointer_cast<StencilContext>(_ksoln);
                assert(context.get());
                auto& opts = context->get_settings();
                opts->print_usage(cout, parser, argv[0], appNotes, appExamples);
                exit_yask(1);
            }

            if (rem_args.length())
                THROW_YASK_EXCEPTION("Error: extraneous parameter(s): '" +
                                     rem_args +
                                     "'; run with '-help' option for usage");
        }
    }

    // Print splash banner and invocation string.
    // Exit with help message if requested.
    void splash(ostream& os, int argc, char** argv)
    {
        // See https://en.wikipedia.org/wiki/Box-drawing_character.
        os <<
            " \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n"
            " \u2502     Y.A.S.K. \u2500\u2500 Yet Another Stencil Kit    \u2502\n"
            " \u2502       https://github.com/intel/yask        \u2502\n"
            " \u2502 Copyright (c) 2014-2019, Intel Corporation \u2502\n"
            " \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\n"
            "\n"
            "Version: " << yask_get_version_string() << endl <<
            "Stencil name: " YASK_STENCIL_NAME << endl;

        // Echo invocation parameters for record-keeping.
#ifdef DEF_ARGS
        os << "Default arguments: " DEF_ARGS << endl;
#endif
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
};                              // AppSettings.

// Override step allocation.
void alloc_steps(yk_solution_ptr soln, const MySettings& opts) {
    if (opts.step_alloc <= 0)
        return;

    // Find vars with steps.
    auto step_dim = soln->get_step_dim_name();
    auto vars = soln->get_vars();
    for (auto var : vars) {
        if (var->is_dim_used(step_dim))

            // override num steps.
            var->set_alloc_size(step_dim, opts.step_alloc);
    }
}

// Parse command-line args, run kernel, run validation if requested.
int main(int argc, char** argv)
{
    // just a line.
    string divLine;
    for (int i = 0; i < 70; i++)
        divLine += "\u2500";
    divLine += "\n";

    try {
        // Bootstrap factories from kernel API.
        yk_factory kfac;
        yask_output_factory yof;

        // Parse custom options once just to get vars needed for env.
        MySettings opts1(nullptr);
        opts1.parse(argc, argv);
        
        // Set up the environment (mostly MPI).
        auto kenv = kfac.new_env();
        kenv->set_trace_enabled(opts1.doTrace);
        if (opts1.msgRank == kenv->get_rank_index())
            kenv->set_debug_output(yof.new_stdout_output());
        else
            kenv->set_debug_output(yof.new_null_output());
        auto ep = dynamic_pointer_cast<KernelEnv>(kenv);
        auto num_ranks = kenv->get_num_ranks();
        auto& os = kenv->get_debug_output()->get_ostream();

        // Make solution object containing data and parameters for stencil eval.
        // TODO: do everything through API without cast to StencilContext.
        auto ksoln = kfac.new_solution(kenv);
        auto context = dynamic_pointer_cast<StencilContext>(ksoln);
        assert(context.get());
        auto& copts = context->get_settings();
        assert(copts);

        // Parse custom and library-provided cmd-line options and
        // exit on -help or error.
        // TODO: do this through APIs.
        MySettings opts(ksoln);
        opts.parse(argc, argv);

        // Make sure warmup is on if needed.
        if (opts.trial_steps <= 0 && opts.trial_time > 0)
            opts.doWarmup = true;

        // Make sure any MPI/OMP debug data is dumped from all ranks before continuing.
        kenv->global_barrier();

        // Print splash banner and related info.
        opts.splash(os, argc, argv);

        // Override alloc if requested.
        alloc_steps(ksoln, opts);

        // Alloc memory, etc.
        ksoln->prepare_solution();

        // Exit if nothing to do.
        if (context->rank_bb.bb_num_points < 1)
            THROW_YASK_EXCEPTION("Exiting because there are no points in the domain");

        // init data in vars and params.
        if (opts.doWarmup || !opts.validate)
            context->initData();

        // Invoke auto-tuner.
        if (opts.doPreAutoTune)
            ksoln->run_auto_tuner_now();

        // Enable/disable further auto-tuning.
        ksoln->reset_auto_tuner(copts->_do_auto_tune);

        // Warmup caches, threading, etc.
        // Measure time to change number of steps.
        if (opts.doWarmup) {

            // Turn off debug.
            auto dbg_out = context->get_debug_output();
            context->set_debug_output(yof.new_null_output());
            os << endl << divLine;

            // Warmup and calibration phases.
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
                if (opts.trial_steps > 0)
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
            if (opts.trial_steps <= 0) {
                idx_t tsteps = ceil(rate * opts.trial_time);
                tsteps = CEIL_DIV(sumOverRanks(tsteps, ep->comm), num_ranks);

                // Round up to multiple of temporal tiling if not too big.
                auto step_dim = ksoln->get_step_dim_name();
                auto rt = copts->_region_sizes[step_dim];
                auto bt = copts->_block_sizes[step_dim];
                auto tt = max(rt, bt);
                const idx_t max_mult = 5;
                if (tt > 1 && tt < max_mult * tsteps)
                    tsteps = ROUND_UP(tsteps, tt);
                
                opts.trial_steps = tsteps;
            }
            
            // Restore debug.
            context->set_debug_output(dbg_out);
        }
        kenv->global_barrier();

        // Exit if nothing to do.
        if (opts.num_trials < 1)
            THROW_YASK_EXCEPTION("Exiting because zero trials are specified");
        if (opts.trial_steps <= 0)
            THROW_YASK_EXCEPTION("Exiting because zero steps per trial are specified");

        // Track results.
        vector<shared_ptr<Stats>> trial_stats;

        // First & last steps.
        idx_t first_t = 0;
        idx_t last_t = opts.trial_steps - 1;

        // Stencils seem to be backward?
        // (This is just a heuristic, but the direction
        // is not usually critical to perf measurement.)
        if (copts->_dims->_step_dir < 0) {
            first_t = last_t;
            last_t = 0;
        }
        
        /////// Performance run(s).
        os << endl << divLine <<
            "Running " << opts.num_trials << " performance trial(s) of " <<
            opts.trial_steps << " step(s) each...\n" << flush;
        for (idx_t tr = 0; tr < opts.num_trials; tr++) {
            os << divLine <<
                "Trial number:                      " << (tr + 1) << endl << flush;

            // init data before each trial for comparison if validating.
            if (opts.validate)
                context->initData();

            // Warn if tuning.
            if (ksoln->is_auto_tuner_enabled())
                os << "auto-tuner is active during this trial, so results may not be representative.\n";

            // Stabilize.
            if (opts.pre_trial_sleep_time > 0) {
                os << flush;
                sleep(opts.pre_trial_sleep_time);
            }
            kenv->global_barrier();

            // Actual work.
            context->clear_timers();
            ksoln->run_solution(first_t, last_t);
            kenv->global_barrier();

            // Calc and report perf.
            auto tstats = context->get_stats();
            auto stats = dynamic_pointer_cast<Stats>(tstats);

            // Remember stats.
            trial_stats.push_back(stats);
        }

        // Report stats.
        if (trial_stats.size()) {

            // Sort based on time.
            sort(trial_stats.begin(), trial_stats.end(),
                 [](const shared_ptr<Stats>& lhs, const shared_ptr<Stats>& rhs) {
                     return lhs->run_time < rhs->run_time; });

            // Pick best and 50%-percentile.
            // See https://en.wikipedia.org/wiki/Percentile.
            auto& best_trial = trial_stats.front();
            auto r50 = trial_stats.size() / 2;
            auto& mid_trial = trial_stats.at(r50);
            
            os << divLine <<
                "Performance stats of best trial:\n"
                " best-num-steps-done:              " << best_trial->nsteps << endl <<
                " best-elapsed-time (sec):          " << makeNumStr(best_trial->run_time) << endl <<
                " best-throughput (num-reads/sec):  " << makeNumStr(best_trial->reads_ps) << endl <<
                " best-throughput (num-writes/sec): " << makeNumStr(best_trial->writes_ps) << endl <<
                " best-throughput (est-FLOPS):      " << makeNumStr(best_trial->flops) << endl <<
                " best-throughput (num-points/sec): " << makeNumStr(best_trial->pts_ps) << endl <<
                divLine <<
                "Performance stats of 50th-percentile trial:\n"
                " mid-num-steps-done:               " << mid_trial->nsteps << endl <<
                " mid-elapsed-time (sec):           " << makeNumStr(mid_trial->run_time) << endl <<
                " mid-throughput (num-reads/sec):   " << makeNumStr(mid_trial->reads_ps) << endl <<
                " mid-throughput (num-writes/sec):  " << makeNumStr(mid_trial->writes_ps) << endl <<
                " mid-throughput (est-FLOPS):       " << makeNumStr(mid_trial->flops) << endl <<
                " mid-throughput (num-points/sec):  " << makeNumStr(mid_trial->pts_ps) << endl <<
                divLine <<
                "Notes:\n"
                " The 50th-percentile trial is the same as the median trial\n"
                "  when there is an odd number of trials. When there is an even\n"
                "  number of trials, the nearest-rank method is used. An odd\n"
                "  number of trials is recommended.\n"
                " Num-reads/sec, num-writes/sec, and FLOPS are metrics based on\n"
                "  stencil specifications and can vary due to differences in\n"
                "  implementations and optimizations.\n"
                " Num-points/sec is based on overall problem size and is\n"
                "  a more reliable performance metric, esp. when comparing\n"
                "  across implementations.\n";
            context->print_warnings();
        }

        /////// Validation run.
        bool ok = true;
        if (opts.validate) {
            kenv->global_barrier();
            os << endl << divLine <<
                "Setup for validation...\n";

            // Make a reference context for comparisons w/new vars.
            // Reuse 'kenv' and copy library settings from 'ksoln'.
            auto ref_soln = kfac.new_solution(kenv, ksoln);
            auto ref_context = dynamic_pointer_cast<StencilContext>(ref_soln);
            assert(ref_context.get());
            auto& ref_opts = ref_context->get_settings();

            // Reapply cmd-line options to override default settings.
            MySettings my_ref_opts(ref_soln);
            my_ref_opts.parse(argc, argv);

            // Change some settings.
            ref_context->name += "-reference";
            ref_context->allow_vec_exchange = false;   // exchange scalars in halos.
            ref_opts->overlap_comms = false;
            ref_opts->use_shm = false;
            ref_opts->_numa_pref = yask_numa_none;

            // TODO: re-enable the region and block settings below;
            // requires allowing consistent init of different-sized vars
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
            alloc_steps(ref_soln, my_ref_opts);
            ref_soln->prepare_solution();

            // init to same value used in context.
            ref_context->initData();

#ifdef CHECK_INIT

            // Debug code to determine if data compares immediately after init matches.
            os << endl << divLine <<
                "Reinitializing data for minimal validation...\n" << flush;
            context->initData();
#else

            // Ref trial.
            // Do same number as last perf run.
            os << endl << divLine <<
                "Running " << opts.trial_steps << " step(s) for validation...\n" << flush;
            ref_context->run_ref(first_t, last_t);

            // Discard perf report.
            auto dbg_out = ref_context->get_debug_output();
            ref_context->set_debug_output(yof.new_null_output());
            auto rstats = ref_context->get_stats();
            ref_context->set_debug_output(dbg_out);
            os << "  Done in " << makeNumStr(rstats->get_elapsed_secs()) << " secs.\n" << flush;
#endif
            // check for equality.
            os << "\nChecking results...\n";
            idx_t errs = context->compareData(*ref_context);
            auto ri = kenv->get_rank_index();

            // Trick to emulate MPI critical section.
            // This cannot be in a conditional block--otherwise
            // it will deadlock if some ranks pass and some fail.
            for (int r = 0; r < kenv->get_num_ranks(); r++) {
                kenv->global_barrier();
                if (r == ri) {
                    if( errs == 0 )
                        os << "TEST PASSED on rank " << ri << ".\n" << flush;
                    else {

                        // Use 'cerr' to print on all ranks in case rank printing to 'os'
                        // passed and other(s) failed.
                        cerr << "TEST FAILED on rank " << ri << ": " << errs << " mismatch(es).\n";
                        if (REAL_BYTES < 8)
                            cerr << " Small differences are not uncommon for low-precision FP; "
                                "try with 8-byte reals.\n";
                        cerr << flush;
                        ok = false;
                    }
                }
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
    catch (yask_exception& e) {
        cerr << "YASK Kernel: " << e.get_message() << ".\n";
        exit_yask(1);
    }

    return 0;
}
