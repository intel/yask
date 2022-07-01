/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2022, Intel Corporation

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

// Add some command-line options for this application in addition to the
// default ones provided by YASK.
struct MySettings {
    static constexpr double def_init_val = -99.;
    
    bool help = false;          // help requested.
    bool do_warmup = true;       // whether to do warmup run.
    bool do_pre_auto_tune = true;  // whether to do pre-auto-tuning.
    int step_alloc = 0;         // if >0, override number of steps to alloc.
    int num_trials = 3;         // number of trials.
    bool validate = false;      // whether to do validation run.
    int trial_steps = 0;        // number of steps in each trial.
    int trial_time = 10;        // sec to run each trial if trial_steps == 0.
    int pre_trial_sleep_time = 1; // sec to sleep before each trial.
    int debug_sleep = 0;          // sec to sleep for debug attach.
    bool do_trace = false;         // tracing.
    int msg_rank = 0;              // rank to print debug msgs.
    double init_val = def_init_val;        // value to init all points.

    // Ptr to the soln.
    yk_solution_ptr _ksoln;

    MySettings(yk_solution_ptr ksoln) :
        _ksoln(ksoln) { }

    // Parse options from the command-line and set corresponding vars.
    // Exit with message on error or request for help.
    // Return settings.
    string parse(int argc, char** argv) {
        string values;

        // Create a parser and add options to it.
        CommandLineParser parser;
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("help",
                           "Print help message.",
                           help));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("msg_rank",
                           "Index of MPI rank that will print informational messages.",
                           msg_rank));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("trace",
                           "Print internal debug messages if compiled with"
                           " general and/or memory-access tracing enabled.",
                           do_trace));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("pre_auto_tune",
                           "Run iteration(s) *before* performance trial(s) to find good-performing "
                           "values for block sizes. "
                           "Uses default values or command-line-provided values as a starting point.",
                           do_pre_auto_tune));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("warmup",
                           "Run warmup iteration(s) before performance "
                           "trial(s) and after auto-tuning iterations, if enabled.",
                           do_warmup));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("step_alloc",
                           "Number of steps to allocate in relevant vars, "
                           "overriding default value from YASK compiler.",
                           step_alloc));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("num_trials",
                           "Number of performance trials.",
                           num_trials));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("t",
                           "[Deprecated] Use '-num_trials'.",
                           num_trials));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("trial_steps",
                           "Number of steps to run each performance trial. "
                           "If zero, the 'trial_time' value is used to determine the number of steps to run.",
                           trial_steps));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("dt",
                           "[Deprecated] Use '-trial_steps'.",
                           trial_steps));
        parser.add_option(make_shared<CommandLineParser::DoubleOption>
                          ("init_val",
                           string("Initialize all points in all stencil vars to given value. ") +
                           "If value is " + to_string(MySettings::def_init_val) +
                           ", points are set to varying values.",
                           init_val));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("trial_time",
                           "Approximate number of seconds to run each performance trial. "
                           "When the 'trial_steps' value is zero, the number of steps is "
                           "based on the rate measured in the warmup phase. "
                           "(Thus, warmup cannot be disabled when the number of steps is zero.)",
                           trial_time));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("sleep",
                           "Number of seconds to sleep before each performance trial.",
                           pre_trial_sleep_time));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("debug_delay",
                           "[Debug] Number of seconds to sleep for debug attach.",
                           debug_sleep));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("validate",
                           "Run validation iteration(s) after performance trial(s).",
                           validate));

        // Parse 'args' and 'argv' cmd-line options, which sets values.
        // Any remaining strings will be returned.
        auto rem_args = parser.parse_args(argc, argv);

        // Handle additional knobs and help if there is a soln.
        if (_ksoln) {

            // TODO: make an API for this.
            auto context = dynamic_pointer_cast<StencilContext>(_ksoln);
            assert(context.get());
            auto& req_opts = context->get_req_opts();
        
            // Parse standard args not handled by this parser.
            rem_args = _ksoln->apply_command_line_options(rem_args);

            if (help) {
                string pgm_name(argv[0]);
                cout << "Usage: " << pgm_name << " [options]\n"
                    "Options from the binary:\n";
                parser.print_help(cout);

                cout << "Options from the YASK library:\n";
                req_opts->print_usage(cout);
                cout << 
                    "\nValidation is very slow and uses 2x memory,\n"
                    " so run with very small sizes and number of time-steps.\n"
                    " If validation fails, it may be due to rounding error;\n"
                    " try building with 8-byte reals.\n";

                // Make example knobs across dims.
                string exg, exnr, exb;
                int i = 1;
                for (auto& dname : _ksoln->get_domain_dim_names()) {
                    exg += " -g" + dname + " " + to_string(i * 128);
                    exb += " -b" + dname + " " + to_string(i * 16);
                    exnr += " -nr" + dname + " " + to_string(i + 1);
                    i++;
                }
                cout <<
                    "\nExamples:\n"
                    " " << pgm_name << " -g 768  # global-domain size in all dims same.\n"
                    " " << pgm_name << exg << "  # global-domain size in each dim separately.\n"
                    " " << pgm_name << " -l 128  # local-domain (per-rank) size.\n"
                    " " << pgm_name << " -g 512" << exnr << "  # number of ranks in each dim.\n" <<
                    " " << pgm_name << " -g 512" << exb << " -no-pre_auto_tune  # manual block size.\n" <<
                    flush;
                exit_yask(1);
            }

            // Add settings.
            ostringstream oss;
            oss << "Options from the binary:\n";
            parser.print_values(oss);
            oss << "Options from the YASK library:\n";
            req_opts->print_values(oss);
            values = oss.str();
            
            if (rem_args.length())
                THROW_YASK_EXCEPTION("Error: extraneous parameter(s): '" +
                                     rem_args +
                                     "'; run with '-help' option for usage");
        }
        return values;
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
            " \u2502 Copyright (c) 2014-2022, Intel Corporation \u2502\n"
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

// Init values in vars.
void init_vars(MySettings& opts, std::shared_ptr<yask::StencilContext> context) {
    if (opts.init_val != MySettings::def_init_val)
        context->init_same(opts.init_val);
    else {
        double seed = opts.validate ? 1.0 : 0.1;
        context->init_diff(seed);
    }
}

// Parse command-line args, run kernel, run validation if requested.
int main(int argc, char** argv)
{
    // Stop collecting VTune data.
    // Even better to use -start-paused option.
    VTUNE_PAUSE;

    // just a line.
    string div_line;
    for (int i = 0; i < 70; i++)
        div_line += "\u2500";
    div_line += "\n";

    try {
        // Bootstrap factories from kernel API.
        yk_factory kfac;
        yask_output_factory yof;

        // Parse only custom options just to get vars needed to set up env.
        MySettings opts1(nullptr);
        opts1.parse(argc, argv);
        yk_env::set_trace_enabled(opts1.do_trace);

        // Set up the environment.
        auto kenv = kfac.new_env();
        auto ep = dynamic_pointer_cast<KernelEnv>(kenv);
        auto num_ranks = kenv->get_num_ranks();

        // Enable debug only on requested rank.
        if (opts1.msg_rank == kenv->get_rank_index())
            yk_env::set_debug_output(yof.new_stdout_output());
        else
            yk_env::set_debug_output(yof.new_null_output());
        auto& os = kenv->get_debug_output()->get_ostream();
        
        // Make solution object containing data and parameters for stencil eval.
        // TODO: do everything through API without cast to StencilContext.
        auto ksoln = kfac.new_solution(kenv);
        auto context = dynamic_pointer_cast<StencilContext>(ksoln);
        assert(context.get());
        auto& copts = context->get_actl_opts();
        assert(copts);

        // Parse custom and library-provided cmd-line options and
        // exit on -help or error.
        // TODO: do this through APIs.
        MySettings opts(ksoln);
        auto opts_str = opts.parse(argc, argv);

        // Make sure any MPI/OMP debug data is dumped from all ranks before continuing.
        kenv->global_barrier();

        // Print splash banner and related info.
        opts.splash(os, argc, argv);
        os << "\n" << opts_str;

        // Print PID and sleep for debug if needed.
        os << "\nPID: " << getpid() << endl;
        if (opts.debug_sleep) {
            os << "Sleeping " << opts.debug_sleep <<
                " second(s) to allow debug attach...\n";
            sleep(opts.debug_sleep);
            os << "Resuming...\n";
        }
        
        // Override alloc if requested.
        alloc_steps(ksoln, opts);

        // Alloc memory, etc.
        ksoln->prepare_solution();

        // Exit if nothing to do.
        if (context->rank_bb.bb_num_points < 1)
            THROW_YASK_EXCEPTION("Exiting because there are no points in the domain");

        // Init data in vars and params.
        init_vars(opts, context);

        // Copy vars now instead of waiting for run_solution() to do it
        // automatically. This will remove overhead from first call.
        context->copy_vars_to_device();

        // Invoke auto-tuner.
        if (opts.do_pre_auto_tune)
            ksoln->run_auto_tuner_now();

        // Enable/disable further auto-tuning.
        ksoln->reset_auto_tuner(copts->_do_auto_tune);

        // Make sure warmup is on if needed.
        if (opts.trial_steps <= 0 && opts.trial_time > 0)
            opts.do_warmup = true;

        // Warmup caches, threading, etc.
        // Measure time to change number of steps.
        if (opts.do_warmup) {

            // Turn off debug.
            auto dbg_out = context->get_debug_output();
            context->set_debug_output(yof.new_null_output());
            os << endl << div_line;

            // Warmup and calibration phases.
            double rate = 1.0;
            {
                idx_t warmup_steps = 1;
                idx_t max_wsteps = 10;
                int nruns = 3;
                for (int n = 0; n < nruns; n++) {

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
                    os << "  Done in " << make_num_str(wtime) << " secs.\n";
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
                    warmup_steps = CEIL_DIV(sum_over_ranks(warmup_steps, ep->comm), num_ranks);

                    // Done if only 1 step to do.
                    if (warmup_steps <= 1)
                        break;
                }
            }

            // Set final number of steps.
            if (opts.trial_steps <= 0) {
                idx_t tsteps = ceil(rate * opts.trial_time);
                tsteps = CEIL_DIV(sum_over_ranks(tsteps, ep->comm), num_ranks);

                // Round up to multiple of temporal tiling if not too big.
                auto step_dim = ksoln->get_step_dim_name();
                auto rt = copts->_mega_block_sizes[step_dim];
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
        os << endl << div_line <<
            "Running " << opts.num_trials << " performance trial(s) of " <<
            opts.trial_steps << " step(s) each...\n" << flush;
        for (idx_t tr = 0; tr < opts.num_trials; tr++) {
            os << div_line <<
                "Trial number:  " << (tr + 1) << endl << flush;

            // re-init data before each trial for comparison if validating.
            if (opts.validate) {
                init_vars(opts, context);
                context->copy_vars_to_device();
            }

            // Warn if tuning.
            if (ksoln->is_auto_tuner_enabled())
                os << "auto-tuner is active during this trial, so results may not be representative.\n";

            // Pause before trial.
            if (opts.pre_trial_sleep_time > 0) {
                os << flush;
                sleep(opts.pre_trial_sleep_time);
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
            auto tstats = context->get_stats();
            auto stats = dynamic_pointer_cast<Stats>(tstats);

            // Remember stats.
            trial_stats.push_back(stats);
        }

        // Done with vtune.
        VTUNE_DETACH;

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

            // Additional stats.
            double sum_pps = 0., sum2_pps = 0., max_pps = 0., min_pps = 0.;
            for (auto ts : trial_stats) {
                auto pps = ts->pts_ps;
                sum_pps += pps;
                sum2_pps += pps * pps;
                if (max_pps == 0. || pps > max_pps)
                    max_pps = pps;
                if (min_pps == 0. || pps < min_pps)
                    min_pps = pps;
            }
            auto n = trial_stats.size();
            double ave_pps = sum_pps / n;
            double var_pps = (sum2_pps - (sum_pps * sum_pps) / n) / (n - 1.);
            double sd_pps = sqrt(var_pps);
            
            os << div_line <<
                "Performance stats of best trial:\n"
                " best-num-steps-done:              " << best_trial->nsteps << endl <<
                " best-elapsed-time (sec):          " << make_num_str(best_trial->run_time) << endl <<
                " best-throughput (num-reads/sec):  " << make_num_str(best_trial->reads_ps) << endl <<
                " best-throughput (num-writes/sec): " << make_num_str(best_trial->writes_ps) << endl <<
                " best-throughput (est-FLOPS):      " << make_num_str(best_trial->flops) << endl <<
                " best-throughput (num-points/sec): " << make_num_str(best_trial->pts_ps) << endl <<
                div_line <<
                "Performance stats of 50th-percentile trial:\n"
                " mid-num-steps-done:               " << mid_trial->nsteps << endl <<
                " mid-elapsed-time (sec):           " << make_num_str(mid_trial->run_time) << endl <<
                " mid-throughput (num-reads/sec):   " << make_num_str(mid_trial->reads_ps) << endl <<
                " mid-throughput (num-writes/sec):  " << make_num_str(mid_trial->writes_ps) << endl <<
                " mid-throughput (est-FLOPS):       " << make_num_str(mid_trial->flops) << endl <<
                " mid-throughput (num-points/sec):  " << make_num_str(mid_trial->pts_ps) << endl <<
                div_line <<
                "Performance stats across trials:\n"
                " num-trials:                          " << n << endl <<
                " min-throughput (num-points/sec):     " << make_num_str(min_pps) << endl <<
                " max-throughput (num-points/sec):     " << make_num_str(max_pps) << endl <<
                " ave-throughput (num-points/sec):     " << make_num_str(ave_pps) << endl;
            if (n > 2)
                os << 
                    " std-dev-throughput (num-points/sec): " << make_num_str(sd_pps) << endl;
            os << div_line <<
                "Notes:\n";
            if (n == 1)
                os << " Since there was only one trial, the best trial and the\n"
                    "  50th-percentile trial are the one and only trial.\n";
            else {
                if (n % 2 == 1)
                    os << " Since there was an odd number of trials, the\n"
                        "  50th-percentile trial is the trial with the median performance:\n";
                else
                    os << " Since there was an even number of trials, the\n"
                        "  50th-percentile trial is chosen with the nearest-rank method:\n";
                os << "  the trial with performance ranked " << (r50+1) << " out of " <<
                    trial_stats.size() << ".\n";
            }
            os <<
                " Num-reads/sec, num-writes/sec, and FLOPS are metrics based on\n"
                "  stencil specifications and can vary due to differences in\n"
                "  implementations and optimizations.\n"
                " Num-points/sec is based on the number of results computed and\n"
                "  is a more reliable performance metric, esp. when comparing\n"
                "  across architectures and/or implementations.\n";
            context->print_warnings();
        }

        /////// Validation run.
        bool ok = true;
        if (opts.validate) {
            kenv->global_barrier();
            os << endl << div_line <<
                "Setup for validation...\n";

            // Make a reference context for comparisons w/new vars.
            // Reuse 'kenv' and copy library settings from 'ksoln'.
            auto ref_soln = kfac.new_solution(kenv, ksoln);
            auto ref_context = dynamic_pointer_cast<StencilContext>(ref_soln);
            assert(ref_context.get());

            // Reapply cmd-line options to override default settings.
            MySettings my_ref_opts(ref_soln);
            my_ref_opts.parse(argc, argv);

            // Change some settings.
            ref_context->name += "-reference";
            auto& ref_opts = ref_context->get_actl_opts();
            ref_opts->force_scalar_exchange = true;
            ref_opts->do_halo_exchange = true;
            ref_opts->overlap_comms = false;
            ref_opts->use_shm = false;
            ref_opts->_numa_pref = yask_numa_none;

            // Override allocations and prep solution as with ref soln.
            alloc_steps(ref_soln, my_ref_opts);
            ref_soln->prepare_solution();

            // init to same value used in context.
            init_vars(opts, ref_context);

            #ifdef CHECK_INIT

            // Debug code to determine if data compares immediately after init matches.
            os << endl << div_line <<
                "Reinitializing data for minimal validation...\n" << flush;
            init_vars(opts, context);
            #else

            // Ref trial.
            // Do same number as last perf run.
            os << endl << div_line <<
                "Running " << opts.trial_steps << " step(s) for validation...\n" << flush;
            ref_context->run_ref(first_t, last_t);

            // Discard perf report.
            auto dbg_out = ref_context->get_debug_output();
            ref_context->set_debug_output(yof.new_null_output());
            auto rstats = ref_context->get_stats();
            ref_context->set_debug_output(dbg_out);
            os << "  Done in " << make_num_str(rstats->get_elapsed_secs()) << " secs.\n" << flush;
            #endif
            // check for equality.
            os << "\nChecking results...\n";
            idx_t errs = context->compare_data(*ref_context);
            auto ri = kenv->get_rank_index();

            // Trick to emulate MPI critical section.
            // This cannot be in a conditional block--otherwise
            // it will deadlock if some ranks pass and some fail.
            for (int r = 0; r < kenv->get_num_ranks(); r++) {
                kenv->global_barrier();
                if (r == ri) {

                    // Use 'cerr' to print on all ranks in case rank printing to 'os'
                    // passed and other(s) failed.
                    if( errs == 0 )
                        cerr << "TEST PASSED on rank " << ri << ".\n" << flush;
                    else {
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
            os << "\nResults NOT VERIFIED.\n";
        ksoln->end_solution();

        os << "Stencil '" << context->get_description() << "'.\n";
        if (!ok)
            exit_yask(1);

        MPI_Finalize();
        os << "YASK DONE." << endl << div_line << flush;
    }
    catch (yask_exception& e) {
        cerr << "YASK Kernel: " << e.get_message() << ".\n";
        exit_yask(1);
    }

    return 0;
}
