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

// This file contains implementations of AutoTuner methods and
// use of the tuner from StencilContext.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Ctor.
    AutoTuner::AutoTuner(StencilContext* context,
                         KernelSettings* settings) :
        ContextLinker(context),
        _settings(settings),
        _name("auto-tuner")
    {
        STATE_VARS(this);
        assert(settings);
        _prefix = string(" ") + _name + ": ";

        clear(settings->_do_auto_tune);
    }

    // Switch target ptr to next one.
    // Sets other target-specific settings.
    // Return 'true' if more to do; 'false' if done.
    // TODO: replace this with a less error-prone data structure
    // and algorithm that will check dependencies more cleanly.
    bool AutoTuner::next_target() {
        STATE_VARS(this);

        trial_secs = _settings->_tuner_trial_secs;
        max_radius = _settings->_tuner_radius;

        outerp = 0;
        min_blks = 1;
        min_pts = 1;

        // Move to next target.
        if (targetp == 0)
            targeti = 0;
        else
            targeti++;
        if (targeti >= _settings->_tuner_targets.size()) {
            targetp = 0;
            return false;
        }
        auto& target_str = _settings->_tuner_targets.at(targeti);
        AT_TRACE_MSG("next target is '" << target_str << "'");

        // Mega-blocks?
        if (target_str == _settings->_mega_block_str) {
            targetp = &_settings->_mega_block_sizes;
            outerp = &_settings->_rank_sizes;
            AT_DEBUG_MSG("searching mega-block sizes...");
        }
        
        // Blocks?
        else if (target_str == _settings->_block_str) {
            targetp = &_settings->_block_sizes;
            outerp = &_settings->_mega_block_sizes;

            // Set min blocks and pts.
            #ifndef USE_OFFLOAD
            int rt=0, bt=0;
            get_num_comp_threads(rt, bt);
            min_blks = max<int>(rt / 2, 1); // At least 1 for every 2 threads.

            // Set min pts (512=8^3).
            min_pts = max<int>(min<int>(get_num_domain_points(*outerp) /
                                        min_blks, 512), 1);
            #endif

            AT_DEBUG_MSG("searching block sizes...");
        }

        // Micro-blocks?
        else if (target_str == _settings->_micro_block_str) {
            targetp = &_settings->_micro_block_sizes;
            outerp = &_settings->_block_sizes;
            AT_DEBUG_MSG("searching micro-block sizes...");
        }

        // Nano-blocks?
        else if (target_str == _settings->_nano_block_str) {
            targetp = &_settings->_nano_block_sizes;
            outerp = &_settings->_micro_block_sizes;
            AT_DEBUG_MSG("searching nano-block sizes...");
        }

        // Pico-blocks?
        else if (target_str == _settings->_pico_block_str) {
            targetp = &_settings->_pico_block_sizes;
            outerp = &_settings->_nano_block_sizes;
            AT_DEBUG_MSG("searching pico-block sizes...");
        }

        else {
            THROW_YASK_EXCEPTION("unrecognized auto-tuner target '" +
                                 target_str + "'");
        }
        assert(targetp);
        assert(outerp);

        // Reset search state.
        at_state.init(this, false);
            
        // Get initial search center from current target.
        at_state.center_sizes = *targetp;

        // Prepare for next target.
        AT_TRACE_MSG("starting size: "  << at_state.center_sizes.make_dim_val_str(" * "));
        AT_TRACE_MSG("starting search radius: " << at_state.radius);
        adjust_settings(false);
        return true;
    }
    
    // Print the best settings.
    void AutoTuner::print_settings() const {
        STATE_VARS(this);
        _context->print_sizes(_prefix);
        _context->print_temporal_tiling_info(_prefix);
        _settings->adjust_settings(_context);
    }

    // Reset the auto-tuner.
    void AutoTuner::clear(bool mark_done, bool verbose) {
        STATE_VARS(this);

#ifdef TRACE
        this->verbose = true;
#else
        this->verbose = verbose;
#endif

        // Apply the best known settings from existing data, if any.
        apply_best();

        // Mark done?
        done = mark_done;
        
        // Reset all vars to be ready to start a new tuning.
        timer.clear();
        steps_done = 0;
        targetp = 0;
        outerp = 0;
        targeti = 0;
        at_state.init(this, true);

    } // clear.

    // Check whether sizes within search limits.
    bool AutoTuner::check_sizes(const IdxTuple& bsize) {
        bool ok = true;
        AT_TRACE_MSG("checking size "  <<
                     bsize.make_dim_val_str(" * "));

        // Too small?
        if (get_num_domain_points(bsize) < min_pts) {
            at_state.n2small++;
            AT_TRACE_MSG("too small");
            ok = false;
        }

        // Too few?
        else {
            idx_t nblks = get_num_domain_points(*outerp) /
                get_num_domain_points(bsize);
            if (nblks < min_blks) {
                AT_TRACE_MSG("too big");
                ok = false;
                at_state.n2big++;
            }
        }
        return ok;
    }

    // This is a "call-back" routine from run_solution().  If a trial is
    // over, it will evaluate that trial and set the state for the next
    // auto-tuner step before returning. If a trial is not over, it will
    // return to get more data.
    void AutoTuner::eval() {
        STATE_VARS(this);

        // Get elapsed time and steps; reset them.
        double etime = timer.get_elapsed_secs();
        timer.clear();
        idx_t steps = steps_done;
        steps_done = 0;

        // Leave if done.
        if (done)
            return;

        // Cumulative stats and rate.
        at_state.csteps += steps;
        at_state.ctime += etime;
        double crate = (at_state.ctime > 0.) ? (double(at_state.csteps) / at_state.ctime) : 0.;
        AT_TRACE_MSG("eval() callback: " << steps << " step(s) in " <<
                     etime << " secs; " << at_state.csteps << " step(s) in " <<
                     at_state.ctime << " secs (" << crate <<
                     " steps/sec) cumulative; best-rate = " << at_state.best_rate <<
                     "; trial-secs = " << trial_secs);

        // Still in warmup?
        if (at_state.in_warmup) {

            // Warmup not done?
            if (at_state.ctime < max(warmup_secs, trial_secs) &&
                at_state.csteps < warmup_steps)
                return; // Keep running.

            // Warmup is done.
            AT_DEBUG_MSG("finished warmup for " <<
                         at_state.csteps << " steps(s) in " <<
                         make_num_str(at_state.ctime) << " secs");

            // Set first target.
            targetp = 0;
            if (!next_target()) {

                // No targets.
                clear(true);
                AT_DEBUG_MSG("no enabled auto-tuner targets");
                return;
            }
            
            return; // Start first trial.
        }

        // Determine whether we've done enough.
        bool rate_ok = false;

        // If the current rate is much less than the best,
        // we don't need a better measurement.
        if (crate > 0. && at_state.best_rate > 0. &&
            crate < at_state.best_rate * cutoff)
            rate_ok = true;

        // Enough time or steps to get a good measurement?
        else if (at_state.ctime >= trial_secs || at_state.csteps >= trial_steps)
            rate_ok = true;

        // Return from eval if we need to do more work.
        if (!rate_ok)
            return; // Get more data for this trial.

        // Save current result.
        at_state.results[*targetp] = crate;
        bool is_better = crate > at_state.best_rate;
        if (is_better) {
            at_state.best_sizes = *targetp;
            at_state.best_rate = crate;
            at_state.better_neigh_found = true;
        }

        // Print progress and reset vars for next time.
        AT_DEBUG_MSG("search-dist=" << at_state.radius << ": " <<
                     make_num_str(crate) << " steps/sec (" <<
                     at_state.csteps << " steps(s) in " << make_num_str(at_state.ctime) <<
                     " secs) with size " <<
                     targetp->remove_dim(step_posn).make_dim_val_str(" * ") <<
                     (is_better ? " -- best so far" : ""));
        at_state.csteps = 0;
        at_state.ctime = 0.;

        // At this point, we have gathered perf info on the current settings.
        // Now, we need to determine next unevaluated point in search space.
        // When found, we 'return' from this call-back function.
        while (true) {

            // Gradient-descent(GD) search:
            // Use the neighborhood info from MPI to track neighbors.
            // TODO: move to a more general place.
            // Valid neighbor index?
            if (at_state.neigh_idx < mpi_info->neighborhood_size) {

                // Convert index to offsets in each domain dim.
                auto ofs = mpi_info->neighborhood_sizes.unlayout(at_state.neigh_idx);

                // Next neighbor of center point.
                at_state.neigh_idx++;

                // Determine new size.
                IdxTuple bsize(at_state.center_sizes);
                bool ok = true;
                int mdist = 0; // manhattan dist from center.
                for (auto odim : ofs) {
                    auto& dname = odim._get_name(); // a domain-dim name.
                    auto& dofs = odim.get_val(); // always [0..2].

                    // Min and max sizes in this dim.
                    auto dmin = dims->_fold_pts[dname];
                    auto dmax = (*outerp)[dname];

                    // Determine distance of GD neighbors.
                    auto dist = dmin; // stride by vector size.
                    dist = max(dist, min_dist);
                    dist *= at_state.radius;

                    auto sz = at_state.center_sizes[dname]; // current size in 'odim'.
                    switch (dofs) {
                    case 0:     // reduce size in 'odim'.
                        sz -= dist;
                        mdist++;
                        break;
                    case 1:     // keep size in 'odim'.
                        break;
                    case 2:     // increase size in 'odim'.
                        if (sz < dist)
                            sz = dist;
                        else
                            sz += dist;
                        mdist++;
                        break;
                    default:
                        assert(false && "internal error in tune_settings()");
                    }

                    // Don't look in all dim combos.
                    if (mdist > 3) {
                        at_state.n2far++;
                        ok = false;
                        break;  // out of dim-loop.
                    }

                    // Adjustments.
                    sz = max(sz, dmin);
                    sz = ROUND_UP(sz, dmin);
                    sz = min(sz, dmax);

                    // Save.
                    bsize[dname] = sz;

                } // domain dims.

                // Check sizes.
                if (ok && !check_sizes(bsize))
                    ok = false;

                // Valid size and not already evaluated?
                if (ok) {
                    if (at_state.results.count(bsize) > 0)
                        AT_TRACE_MSG("already evaluated");
                    else {
                        AT_TRACE_MSG("exiting eval() with new size");

                        // Run next step with this size.
                        *targetp = bsize;
                        adjust_settings(false);
                        return;
                    }
                }

            } // valid neighbor index.

            // Beyond last neighbor of current center?
            // Determine next search setting.
            else {

                // Should GD continue at this radius from the new best
                // point?
                bool stop_gd = !at_state.better_neigh_found;

                // Make new center at best size so far.
                at_state.center_sizes = at_state.best_sizes;

                // Reset search vars.
                at_state.neigh_idx = 0;
                at_state.better_neigh_found = false;

                // Check another point at this radius?
                if (!stop_gd)
                    AT_TRACE_MSG("continuing search from " <<
                                 at_state.center_sizes.make_dim_val_str(" * "));

                // No new best point, so this is the end of the
                // GD search at this radius.
                else {

                    // Move to next radius.
                    at_state.radius /= 2;
                    if (at_state.radius >= 1)
                        AT_TRACE_MSG("new search radius=" << at_state.radius);

                    // No more radii for this target.
                    else {

                        // Apply current best result for this target.
                        apply_best();

                        // Move to next target.
                        if (next_target()) {
                            AT_TRACE_MSG("exiting eval() with new target");
                            return;
                        }

                        // No more targets.
                        else {
                            
                            // Reset AT and disable.
                            clear(true);
                            AT_DEBUG_MSG("done");
                            return;
                        }
                    }
                }
            } // beyond next neighbor of center.
        } // while(true) search for new setting to try.
    } // eval.

    // Apply best settings if avail, and adjust other settings.
    // Returns true if set.
    bool AutoTuner::apply_best() {
        STATE_VARS(this);
        if (at_state.best_rate > 0. && targetp) {
            AT_DEBUG_MSG("applying size "  <<
                         at_state.best_sizes.make_dim_val_str(" * "));
            *targetp = at_state.best_sizes;

            // Save these results as requested options.
            if (targetp == &_settings->_mega_block_sizes)
                req_opts->_mega_block_sizes = *targetp;
            if (targetp == &_settings->_block_sizes)
                req_opts->_block_sizes = *targetp;
            if (targetp == &_settings->_micro_block_sizes)
                req_opts->_micro_block_sizes = *targetp;
            if (targetp == &_settings->_nano_block_sizes)
                req_opts->_nano_block_sizes = *targetp;
            if (targetp == &_settings->_pico_block_sizes)
                req_opts->_pico_block_sizes = *targetp;

            // Adjust other settings based on target.
            adjust_settings(false);
            return true;
        }
        return false;
    }
    
    // Adjust related kernel settings to prepare for next run.
    void AutoTuner::adjust_settings(bool do_print) {
        STATE_VARS(this);
        assert(targetp);

        // Reset non-target settings to requested ones.  This is done so
        // that ajustment will be applied based on requested ones and this
        // target instead of adjusted ones.
        if (targetp != &_settings->_mega_block_sizes)
            _settings->_mega_block_sizes = req_opts->_mega_block_sizes;
        if (targetp != &_settings->_block_sizes)
            _settings->_block_sizes = req_opts->_block_sizes;
        if (targetp != &_settings->_micro_block_sizes)
            _settings->_micro_block_sizes = req_opts->_micro_block_sizes;
        if (targetp != &_settings->_nano_block_sizes)
            _settings->_nano_block_sizes = req_opts->_nano_block_sizes;
        if (targetp != &_settings->_pico_block_sizes)
            _settings->_pico_block_sizes = req_opts->_pico_block_sizes;
        
        // Save debug output and set to null.
        auto saved_op = get_debug_output();
        if (!do_print) {
            yask_output_factory yof;
            auto nullop = yof.new_null_output();
            set_debug_output(nullop);
        }

        // The following sequence is the required subset of what
        // is done in prepare_solution().
        
        // Make sure everything is adjusted based on new target size.
        _settings->adjust_settings(do_print ? this : 0);
        
        // Update temporal blocking info.
        // (Normally called from update_var_info().)
        _context->update_tb_info();

        // Reallocate scratch vars based on new micro-block size.
        // TODO: only do this when needed.
        _context->alloc_scratch_data();

        // Restore debug output.
        set_debug_output(saved_op);
    }

    ///// StencilContext methods to control the auto-tuner(s).

    // Eval auto-tuner for given number of steps.
    void StencilContext::eval_auto_tuner() {
        _at.eval();
    }

    // Reset auto-tuners.
    void StencilContext::reset_auto_tuner(bool enable, bool verbose) {
        _at.clear(!enable, verbose);
    }

    // Determine if any auto tuners are running.
    bool StencilContext::is_auto_tuner_enabled() const {
        return !_at.is_done();
   }

    // Apply auto-tuning immediately, i.e., not as part of normal processing.
    // Will alter data in vars.
    void StencilContext::run_auto_tuner_now(bool verbose) {
        STATE_VARS(this);
        if (!is_prepared())
            THROW_YASK_EXCEPTION("run_auto_tuner_now() called without calling prepare_solution() first");

        DEBUG_MSG("\nAuto-tuning...");
        YaskTimer at_timer;
        at_timer.start();

        // Copy vars to device now so that automatic copy in
        // run_solution() will not impact timing.
        copy_vars_to_device();

        // Temporarily disable halo exchange to tune intra-rank.
        // Will not produce valid results and will corrupt data.
        auto save_halo_exchange = actl_opts->do_halo_exchange;
        actl_opts->do_halo_exchange = false;

        // Init tuners.
        reset_auto_tuner(true, verbose);

        // Reset stats.
        clear_timers();

        // Determine number of steps to run.
        // If wave-fronts are enabled, run a max number of these steps.
        idx_t step_dir = dims->_step_dir; // +/- 1.
        idx_t stride_t = min(max(wf_steps, idx_t(1)), +AutoTuner::max_stride_t) * step_dir;

        // Run time-steps until AT converges.
        for (idx_t t = 0; ; t += stride_t) {

            // Run stride_t time-step(s).
            run_solution(t, t + stride_t - step_dir);

            // AT done on this rank?
            if (!is_auto_tuner_enabled())
                break;
        }

        // Wait for all ranks to finish.
        #if USE_MPI
        DEBUG_MSG("Waiting for auto-tuner to converge on all ranks...");
        env->global_barrier();
        #endif

        // reenable normal operation.
        actl_opts->do_halo_exchange = save_halo_exchange;

        // Report results.
        at_timer.stop();
        DEBUG_MSG("Auto-tuner done after " <<
                  make_num_str(at_timer.get_elapsed_secs()) << " secs");
        DEBUG_MSG("Final settings:");
        _at.print_settings();

        // Reset stats.
        clear_timers();
    }

} // namespace yask.
