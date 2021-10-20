/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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
                         KernelSettings* settings,
                         const std::string& name) :
        ContextLinker(context),
        _settings(settings),
        _name("auto-tuner") {
        assert(settings);
        if (name.length())
            _name += "(" + name + ")";
        clear(settings->_do_auto_tune);
    }

    // Eval auto-tuner for given number of steps.
    void StencilContext::eval_auto_tuner(idx_t num_steps) {
        STATE_VARS(this);
        _at.steps_done += num_steps;
        _at.timer.stop();

        if (state->_use_stage_tuners) {
            for (auto& sp : st_stages)
                sp->get_at().eval();
        }
        else
            _at.eval();
    }

    // Reset auto-tuners.
    void StencilContext::reset_auto_tuner(bool enable, bool verbose) {
        for (auto& sp : st_stages)
            sp->get_at().clear(!enable, verbose);
        _at.clear(!enable, verbose);
    }

    // Determine if any auto tuners are running.
    bool StencilContext::is_auto_tuner_enabled() const {
        STATE_VARS(this);
        bool done = true;
        if (state->_use_stage_tuners) {
            for (auto& sp : st_stages)
                if (!sp->get_at().is_done())
                    done = false;
        } else
            done = _at.is_done();
        return !done;
    }

    // Apply auto-tuning immediately, i.e., not as part of normal processing.
    // Will alter data in vars.
    void StencilContext::run_auto_tuner_now(bool verbose) {
        STATE_VARS(this);
        if (!is_prepared())
            THROW_YASK_EXCEPTION("Error: run_auto_tuner_now() called without calling prepare_solution() first");

        DEBUG_MSG("Auto-tuning...");
        YaskTimer at_timer;
        at_timer.start();

        // Temporarily disable halo exchange to tune intra-rank.
        enable_halo_exchange = false;

        // Temporarily ignore step conditions to force eval of conditional
        // bundles.  NB: may affect perf, e.g., if stages A and B run in
        // AAABAAAB sequence, perf may be [very] different if run as
        // ABABAB..., esp. w/temporal tiling.  TODO: work around this.
        check_step_conds = false;

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
        DEBUG_MSG("Waiting for auto-tuner to converge on all ranks...");
        env->global_barrier();

        // reenable normal operation.
#ifndef NO_HALO_EXCHANGE
        enable_halo_exchange = true;
#endif
        check_step_conds = true;

        // Report results.
        at_timer.stop();
        DEBUG_MSG("Auto-tuner done after " << steps_done << " step(s) in " <<
                  make_num_str(at_timer.get_elapsed_secs()) << " secs.");
        if (state->_use_stage_tuners) {
            for (auto& sp : st_stages)
                sp->get_at().print_settings();
        } else
            _at.print_settings();
        print_temporal_tiling_info();

        // Reset stats.
        clear_timers();
    }

    // Print the best settings.
    void AutoTuner::print_settings() const {
        STATE_VARS(this);
        if (tune_mini_blks())
            DEBUG_MSG(_name << ": best-mini-block-size: " <<
                      target_sizes().remove_dim(step_posn).make_dim_val_str(" * "));
        else
            DEBUG_MSG(_name << ": best-block-size: " <<
                      target_sizes().remove_dim(step_posn).make_dim_val_str(" * ") << endl <<
                      _name << ": mini-block-size: " <<
                      _settings->_mini_block_sizes.remove_dim(step_posn).make_dim_val_str(" * "));
        DEBUG_MSG(_name << ": sub-block-size: " <<
                  _settings->_sub_block_sizes.remove_dim(step_posn).make_dim_val_str(" * "));
    }

    // Access settings.
    bool AutoTuner::tune_mini_blks() const {
        return _context->get_settings()->_tune_mini_blks;
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
        if (best_rate > 0.) {
            target_sizes() = best_sizes;
            apply();
            DEBUG_MSG(_name << ": applying size "  <<
                      best_sizes.make_dim_val_str(" * "));
        }

        // Reset all vars.
        results.clear();
        n2big = n2small = n2far = 0;
        best_rate = 0.;
        radius = max_radius;
        done = mark_done;
        neigh_idx = 0;
        better_neigh_found = false;
        ctime = 0.;
        csteps = 0;
        in_warmup = true;
        timer.clear();
        steps_done = 0;
        target_steps = target_sizes()[step_dim];
        center_sizes = target_sizes();
        best_sizes = target_sizes();

        // Set min blocks to number of region threads.
        int rt=0, bt=0;
        get_num_comp_threads(rt, bt);
        min_blks = rt;

    } // clear.

    // Check whether sizes within search limits.
    bool AutoTuner::check_sizes(const IdxTuple& bsize) {
        bool ok = true;

        // Too small?
        if (ok && get_num_domain_points(bsize) < min_pts) {
            n2small++;
            ok = false;
        }

        // Too few?
        else if (ok) {
            idx_t nblks = get_num_domain_points(outer_sizes()) /
                get_num_domain_points(bsize);
            if (nblks < min_blks) {
                ok = false;
                n2big++;
            }
        }
        return ok;
    }

    // Evaluate the previous run and take next auto-tuner step.
    void AutoTuner::eval() {
        STATE_VARS(this);

        // Get elapsed time and reset.
        double etime = timer.get_elapsed_secs();
        timer.clear();
        idx_t steps = steps_done;
        steps_done = 0;

        // Leave if done.
        if (done)
            return;

        // Setup not done?
        if (!nullop)
            return;

        // Cumulative stats and rate.
        csteps += steps;
        ctime += etime;
        double rate = (ctime > 0.) ? (double(csteps) / ctime) : 0.;
        double min_secs = _settings->_tuner_min_secs;
        TRACE_MSG(_name << " eval() callback: " << steps << " step(s) in " <<
                  etime << " secs; " << csteps << " step(s) in " <<
                  ctime << " secs (" << rate <<
                  " steps/sec) cumulative; best-rate = " << best_rate <<
                  "; min-secs = " << min_secs);

        // Still in warmup?
        if (in_warmup) {

            // Warmup not done?
            if (ctime < max(warmup_secs, min_secs) && csteps < warmup_steps)
                return;

            // Done.
            DEBUG_MSG(_name << ": finished warmup for " <<
                      csteps << " steps(s) in " <<
                      make_num_str(ctime) << " secs\n" <<
                      _name << ": tuning " << (tune_mini_blks() ? "mini-" : "") <<
                      "block sizes...");
            in_warmup = false;

            // Restart for first real measurement.
            csteps = 0;
            ctime = 0;

            // Set center point for search.
            center_sizes = target_sizes();

            // Pick better starting point if needed.
            if (!check_sizes(center_sizes)) {
                for (auto dim : center_sizes) {
                    auto& dname = dim._get_name();
                    auto& dval = dim.get_val();
                    if (dname != step_dim) {
                        auto dmax = max(idx_t(1), outer_sizes()[dname] / 2);
                        center_sizes[dname] = dmax;
                    }
                }
            }

            // Set vars to starting point.
            best_sizes = center_sizes;
            target_sizes() = center_sizes;
            apply();
            TRACE_MSG(_name << ": starting size: "  << center_sizes.make_dim_val_str(" * "));
            TRACE_MSG(_name << ": starting search radius: " << radius);
            return;
        }

        // Determine whether we've done enough.
        bool rate_ok = false;

        // If the current rate is much less than the best,
        // we don't need a better measurement.
        if (rate > 0. && best_rate > 0. && rate < best_rate * cutoff)
            rate_ok = true;

        // Enough time or steps to get a good measurement?
        else if (ctime >= min_secs || csteps >= min_steps)
            rate_ok = true;

        // Return from eval if we need to do more work.
        if (!rate_ok)
            return;

        // Save result.
        results[target_sizes()] = rate;
        bool is_better = rate > best_rate;
        if (is_better) {
            best_sizes = target_sizes();
            best_rate = rate;
            better_neigh_found = true;
        }

        // Print progress and reset vars for next time.
        DEBUG_MSG(_name << ": search-dist=" << radius << ": " <<
                  make_num_str(rate) << " steps/sec (" <<
                  csteps << " steps(s) in " << make_num_str(ctime) <<
                  " secs) with size " <<
                  target_sizes().remove_dim(step_posn).make_dim_val_str(" * ") <<
                  (is_better ? " -- best so far" : ""));
        csteps = 0;
        ctime = 0.;

        // At this point, we have gathered perf info on the current settings.
        // Now, we need to determine next unevaluated point in search space.
        while (true) {

            // Gradient-descent(GD) search:
            // Use the neighborhood info from MPI to track neighbors.
            // TODO: move to a more general place.
            // Valid neighbor index?
            if (neigh_idx < mpi_info->neighborhood_size) {

                // Convert index to offsets in each domain dim.
                auto ofs = mpi_info->neighborhood_sizes.unlayout(neigh_idx);

                // Next neighbor of center point.
                neigh_idx++;

                // Determine new size.
                IdxTuple bsize(center_sizes);
                bool ok = true;
                int mdist = 0; // manhattan dist from center.
                for (auto odim : ofs) {
                    auto& dname = odim._get_name(); // a domain-dim name.
                    auto& dofs = odim.get_val(); // always [0..2].

                    // Min and max sizes of this dim.
                    auto dmin = dims->_cluster_pts[dname];
                    auto dmax = outer_sizes()[dname];

                    // Determine distance of GD neighbors.
                    auto dist = dmin; // stride by cluster size.
                    dist = max(dist, min_dist);
                    dist *= radius;

                    auto sz = center_sizes[dname];
                    switch (dofs) {
                    case 0:     // reduce size in 'odim'.
                        sz -= dist;
                        mdist++;
                        break;
                    case 1:     // keep size in 'odim'.
                        break;
                    case 2:     // increase size in 'odim'.
                        sz += dist;
                        mdist++;
                        break;
                    default:
                        assert(false && "internal error in tune_settings()");
                    }

                    // Don't look in far corners.
                    if (mdist > 2) {
                        n2far++;
                        ok = false;
                        break;  // out of dim-loop.
                    }

                    // Too small?
                    if (sz < dmin) {
                        n2small++;
                        ok = false;
                        break;  // out of dim-loop.
                    }

                    // Adjustments.
                    sz = min(sz, dmax);
                    sz = ROUND_UP(sz, dmin);

                    // Save.
                    bsize[dname] = sz;

                } // domain dims.
                TRACE_MSG(_name << ": checking size "  <<
                          bsize.make_dim_val_str(" * "));

                // Check sizes.
                if (ok && !check_sizes(bsize))
                    ok = false;


                // Valid size and not already checked?
                if (ok && results.count(bsize) == 0) {

                    // Run next step with this size.
                    target_sizes() = bsize;
                    break;      // out of block-search loop.
                }

            } // valid neighbor index.

            // Beyond last neighbor of current center?
            else {

                // Should GD continue?
                bool stop_gd = !better_neigh_found;

                // Make new center at best size so far.
                center_sizes = best_sizes;

                // Reset search vars.
                neigh_idx = 0;
                better_neigh_found = false;

                // No new best point, so this is the end of this
                // GD search.
                if (stop_gd) {

                    // Move to next radius.
                    radius /= 2;

                    // Done?
                    if (radius < 1) {

                        // Reset AT and disable.
                        clear(true);
                        DEBUG_MSG(_name << ": done");
                        return;
                    }
                    TRACE_MSG(_name << ": new search radius=" << radius);
                }
                else {
                    TRACE_MSG(_name << ": continuing search from " <<
                               center_sizes.make_dim_val_str(" * "));
                }
            } // beyond next neighbor of center.
        } // search for new setting to try.

        // Fix settings for next step.
        apply();
        TRACE_MSG(_name << ": next size "  <<
                  target_sizes().make_dim_val_str(" * "));
    } // eval.

    // Adjust related kernel settings to prepare for a run.
    // Does *not* set the settings being tuned.
    void AutoTuner::apply() {
        STATE_VARS(this);

        // Restore step-dim value for block.
        target_sizes()[step_posn] = target_steps;

        // Change derived sizes to 0 so adjust_settings()
        // will set them to the default.
        if (!tune_mini_blks()) {
            _settings->_block_group_sizes.set_vals_same(0);
            _settings->_mini_block_sizes.set_vals_same(0);
        }
        _settings->_mini_block_group_sizes.set_vals_same(0);
        _settings->_sub_block_sizes.set_vals_same(0);
        _settings->_sub_block_group_sizes.set_vals_same(0);

        // Save debug output and set to null.
        auto saved_op = get_debug_output();
        set_debug_output(nullop);

        // Make sure everything is resized based on block size.
        _settings->adjust_settings();

        // Update temporal blocking info.
        _context->update_tb_info();

        // Reallocate scratch data based on new mini-block size.
        // TODO: only do this when blocks have increased or
        // decreased by a certain percentage.
        _context->alloc_scratch_data();

        // Restore debug output.
        set_debug_output(saved_op);
    }

} // namespace yask.
