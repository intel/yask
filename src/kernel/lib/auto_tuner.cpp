/*****************************************************************************

YASK: Yet Another Stencil Kernel
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

        if (state->_use_pack_tuners) {
            for (auto& sp : stPacks)
                sp->getAT().eval();
        }
        else
            _at.eval();
    }
    
    // Reset auto-tuners.
    void StencilContext::reset_auto_tuner(bool enable, bool verbose) {
        for (auto& sp : stPacks)
            sp->getAT().clear(!enable, verbose);
        _at.clear(!enable, verbose);
    }

    // Determine if any auto tuners are running.
    bool StencilContext::is_auto_tuner_enabled() const {
        STATE_VARS(this);
        bool done = true;
        if (state->_use_pack_tuners) {
            for (auto& sp : stPacks)
                if (!sp->getAT().is_done())
                    done = false;
        } else
            done = _at.is_done();
        return !done;
    }
    
    // Apply auto-tuning immediately, i.e., not as part of normal processing.
    // Will alter data in grids.
    void StencilContext::run_auto_tuner_now(bool verbose) {
        STATE_VARS(this);
        if (!rank_bb.bb_valid)
            THROW_YASK_EXCEPTION("Error: run_auto_tuner_now() called without calling prepare_solution() first");

        os << "Auto-tuning...\n" << flush;
        YaskTimer at_timer;
        at_timer.start();

        // Temporarily disable halo exchange to tune intra-rank.
        enable_halo_exchange = false;

        // Temporarily ignore step conditions to force eval of conditional
        // bundles.  NB: may affect perf, e.g., if packs A and B run in
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
        idx_t step_t = min(max(wf_steps, idx_t(1)), +AutoTuner::max_step_t) * step_dir;

        // Run time-steps until AT converges.
        for (idx_t t = 0; ; t += step_t) {

            // Run step_t time-step(s).
            run_solution(t, t + step_t - step_dir);

            // AT done on this rank?
            if (!is_auto_tuner_enabled())
                break;
        }

        // Wait for all ranks to finish.
        os << "Waiting for auto-tuner to converge on all ranks...\n";
        env->global_barrier();

        // reenable normal operation.
#ifndef NO_HALO_EXCHANGE
        enable_halo_exchange = true;
#endif
        check_step_conds = true;

        // Report results.
        at_timer.stop();
        os << "Auto-tuner done after " << steps_done << " step(s) in " <<
            makeNumStr(at_timer.get_elapsed_secs()) << " secs.\n";
        if (state->_use_pack_tuners) {
            for (auto& sp : stPacks)
                sp->getAT().print_settings(os);
        } else
            _at.print_settings(os);
        print_temporal_tiling_info();

        // Reset stats.
        clear_timers();
    }

    // Print the best settings.
    void AutoTuner::print_settings(ostream& os) const {
        if (tune_mini_blks())
            os << _name << ": best-mini-block-size: " <<
                target_sizes().makeDimValStr(" * ") << endl;
        else
            os << _name << ": best-block-size: " <<
                target_sizes().makeDimValStr(" * ") << endl <<
                _name << ": mini-block-size: " <<
                _settings->_mini_block_sizes.makeDimValStr(" * ") << endl;
        os << _name << ": sub-block-size: " <<
            _settings->_sub_block_sizes.makeDimValStr(" * ") << endl <<
            flush;
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
            os << _name << ": applying size "  <<
                best_sizes.makeDimValStr(" * ") << endl;
        }

        // Reset all vars.
        results.clear();
        n2big = n2small = n2far = 0;
        best_sizes = target_sizes();
        best_rate = 0.;
        center_sizes = best_sizes;
        radius = max_radius;
        done = mark_done;
        neigh_idx = 0;
        better_neigh_found = false;
        ctime = 0.;
        csteps = 0;
        in_warmup = true;
        timer.clear();
        steps_done = 0;

        // Set min blocks to number of region threads.
        int rt=0, bt=0;
        get_num_comp_threads(rt, bt);
        min_blks = rt;

        // Adjust starting block if needed.
        for (auto dim : center_sizes.getDims()) {
            auto& dname = dim.getName();
            auto& dval = dim.getVal();

            if (dname == step_dim) {
                target_steps = target_sizes()[dname]; // save value.
                center_sizes[dname] = target_steps;
            } else {
                auto dmax = max(idx_t(1), outer_sizes()[dname] / 2);
                if (dval > dmax || dval < 1)
                    center_sizes[dname] = dmax;
            }
        }
        if (!done) {
            TRACE_MSG(_name << ": starting size: "  <<
                      center_sizes.makeDimValStr(" * "));
            TRACE_MSG(_name << ": starting search radius: " << radius);
        }
    } // clear.

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

        // Cumulative stats.
        csteps += steps;
        ctime += etime;

        // Still in warmup?
        if (in_warmup) {

            // Warmup not done?
            if (ctime < warmup_secs && csteps < warmup_steps)
                return;

            // Done.
            os << _name << ": finished warmup for " <<
                csteps << " steps(s) in " <<
                makeNumStr(ctime) << " secs\n" <<
                _name << ": tuning " << (tune_mini_blks() ? "mini-" : "") <<
                "block sizes...\n";
            in_warmup = false;

            // Restart for first measurement.
            csteps = 0;
            ctime = 0;

            // Fix settings for next step.
            apply();
            TRACE_MSG(_name << ": first size "  <<
                      target_sizes().makeDimValStr(" * "));
            return;
        }

        // Need more steps to get a good measurement?
        if (ctime < min_secs && csteps < min_steps)
            return;

        // Calc perf and reset vars for next time.
        double rate = (ctime > 0.) ? (double(csteps) / ctime) : 0.;
        os << _name << ": search-radius=" << radius << ": " <<
            csteps << " steps(s) in " <<
            makeNumStr(ctime) << " secs (" <<
            makeNumStr(rate) << " steps/sec) with size " <<
            target_sizes().makeDimValStr(" * ") << endl;
        csteps = 0;
        ctime = 0.;

        // Save result.
        results[target_sizes()] = rate;
        bool is_better = rate > best_rate;
        if (is_better) {
            best_sizes = target_sizes();
            best_rate = rate;
            better_neigh_found = true;
        }

        // At this point, we have gathered perf info on the current settings.
        // Now, we need to determine next unevaluated point in search space.
        while (true) {

            // Gradient-descent(GD) search:
            // Use the neighborhood info from MPI to track neighbors.
            // TODO: move to a more general place.
            // Valid neighbor index?
            if (neigh_idx < mpiInfo->neighborhood_size) {

                // Convert index to offsets in each domain dim.
                auto ofs = mpiInfo->neighborhood_sizes.unlayout(neigh_idx);

                // Next neighbor of center point.
                neigh_idx++;

                // Determine new size.
                IdxTuple bsize(center_sizes);
                bool ok = true;
                int mdist = 0; // manhattan dist from center.
                for (auto odim : ofs.getDims()) {
                    auto& dname = odim.getName(); // a domain-dim name.
                    auto& dofs = odim.getVal(); // always [0..2].

                    // Min and max sizes of this dim.
                    auto dmin = dims->_cluster_pts[dname];
                    auto dmax = outer_sizes()[dname];

                    // Determine distance of GD neighbors.
                    auto dist = dmin; // step by cluster size.
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
                          bsize.makeDimValStr(" * "));

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

                // Valid size and not already checked?
                if (ok && !results.count(bsize)) {

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
                        os << _name << ": done" << endl;
                        return;
                    }
                    TRACE_MSG(_name << ": new search radius=" << radius);
                }
                else {
                    TRACE_MSG(_name << ": continuing search from " <<
                               center_sizes.makeDimValStr(" * "));
                }
            } // beyond next neighbor of center.
        } // search for new setting to try.

        // Fix settings for next step.
        apply();
        TRACE_MSG(_name << ": next size "  <<
                  target_sizes().makeDimValStr(" * "));
    } // eval.

    // Apply auto-tuner settings to prepare for a run.
    // Does *not* set the settings being tuned.
    void AutoTuner::apply() {
        STATE_VARS(this);

        // Restore step-dim value for block.
        target_sizes()[step_posn] = target_steps;
        
        // Change derived sizes to 0 so adjustSettings()
        // will set them to the default.
        if (!tune_mini_blks()) {
            _settings->_block_group_sizes.setValsSame(0);
            _settings->_mini_block_sizes.setValsSame(0);
        }
        _settings->_mini_block_group_sizes.setValsSame(0);
        _settings->_sub_block_sizes.setValsSame(0);
        _settings->_sub_block_group_sizes.setValsSame(0);

        // Save debug output and set to null.
        auto saved_op = get_debug_output();
        set_debug_output(nullop);

        // Make sure everything is resized based on block size.
        _settings->adjustSettings();

        // Update temporal blocking info.
        _context->update_tb_info();

        // Reallocate scratch data based on new mini-block size.
        // TODO: only do this when blocks have increased or
        // decreased by a certain percentage.
        _context->allocScratchData();

        // Restore debug output.
        set_debug_output(saved_op);
    }

} // namespace yask.
