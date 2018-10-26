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

// This file contains implementations of AutoTuner methods.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Print the best settings.
    void AutoTuner::print_settings(ostream& os) const {
        os << _name << ": best-block-size: " <<
            _settings->_block_sizes.makeDimValStr(" * ") << endl <<
            _name << ": mini-block-size: " <<
            _settings->_mini_block_sizes.makeDimValStr(" * ") << endl <<
            _name << ": sub-block-size: " <<
            _settings->_sub_block_sizes.makeDimValStr(" * ") << endl <<
            flush;
    }
    
    // Reset the auto-tuner.
    void AutoTuner::clear(bool mark_done, bool verbose) {

        // Output.
        ostream& os = _context->get_ostr();
#ifdef TRACE
        this->verbose = true;
#else
        this->verbose = verbose;
#endif

        // Apply the best known settings from existing data, if any.
        if (best_rate > 0.) {
            _settings->_block_sizes = best_block;
            apply();
            os << _name << ": applying block-size "  <<
                best_block.makeDimValStr(" * ") << endl;
        }

        // Reset all vars.
        results.clear();
        n2big = n2small = n2far = 0;
        best_block = _settings->_block_sizes;
        best_rate = 0.;
        center_block = best_block;
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
        min_blks = _context->set_region_threads();

        // Adjust starting block if needed.
        auto& opts = _context->get_settings();
        for (auto dim : center_block.getDims()) {
            auto& dname = dim.getName();
            auto& dval = dim.getVal();

            if (dname == opts->_dims->_step_dim) {
                block_steps = opts->_block_sizes[dname];
                center_block[dname] = block_steps;
            } else {
                auto dmax = max(idx_t(1), opts->_region_sizes[dname] / 2);
                if (dval > dmax || dval < 1)
                    center_block[dname] = dmax;
            }
        }
        if (!done) {
            TRACE_MSG2(_name << ": starting block-size: "  <<
                       center_block.makeDimValStr(" * "));
            TRACE_MSG2(_name << ": starting search radius: " << radius);
        }
    } // clear.

    // Evaluate the previous run and take next auto-tuner step.
    void AutoTuner::eval() {
        CONTEXT_VARS(_context);

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
            os << _name << ": in warmup for " << ctime << " secs" << endl;
            in_warmup = false;

            // Measure this step only.
            csteps = steps;
            ctime = etime;
        }

        // Need more steps to get a good measurement?
        if (ctime < min_secs && csteps < min_steps)
            return;

        // Calc perf and reset vars for next time.
        double rate = (ctime > 0.) ? double(csteps) / ctime : 0.;
        os << _name << ": radius=" << radius << ": " <<
            csteps << " steps(s) in " << ctime <<
            " secs (" << rate <<
            " steps/sec) with block-size " <<
            _settings->_block_sizes.makeDimValStr(" * ");
        if (_context->tb_steps > 0)
            os << ", " << _context->tb_steps << " TB step(s)";
        os << endl;
        csteps = 0;
        ctime = 0.;

        // Save result.
        results[_settings->_block_sizes] = rate;
        bool is_better = rate > best_rate;
        if (is_better) {
            best_block = _settings->_block_sizes;
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

                // Determine new block size.
                IdxTuple bsize(center_block);
                bool ok = true;
                int mdist = 0; // manhattan dist from center.
                for (auto odim : ofs.getDims()) {
                    auto& dname = odim.getName(); // a domain-dim name.
                    auto& dofs = odim.getVal(); // always [0..2].

                    // Min and max sizes of this dim.
                    auto dmin = dims->_cluster_pts[dname];
                    auto dmax = opts->_region_sizes[dname];

                    // Determine distance of GD neighbors.
                    auto step = dmin; // step by cluster size.
                    step = max(step, min_step);
                    step *= radius;

                    auto sz = center_block[dname];
                    switch (dofs) {
                    case 0:     // reduce size in 'odim'.
                        sz -= step;
                        mdist++;
                        break;
                    case 1:     // keep size in 'odim'.
                        break;
                    case 2:     // increase size in 'odim'.
                        sz += step;
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
                TRACE_MSG2(_name << ": checking block-size "  <<
                          bsize.makeDimValStr(" * "));

                // Too small?
                if (ok && get_num_domain_points(bsize) < min_pts) {
                    n2small++;
                    ok = false;
                }

                // Too few?
                else if (ok) {
                    auto& opts = _context->get_settings();
                    idx_t nblks = get_num_domain_points(opts->_region_sizes) /
                        get_num_domain_points(bsize);
                    if (nblks < min_blks) {
                        ok = false;
                        n2big++;
                    }
                }

                // Valid size and not already checked?
                if (ok && !results.count(bsize)) {

                    // Run next step with this size.
                    _settings->_block_sizes = bsize;
                    break;      // out of block-search loop.
                }

            } // valid neighbor index.

            // Beyond last neighbor of current center?
            else {

                // Should GD continue?
                bool stop_gd = !better_neigh_found;

                // Make new center at best block so far.
                center_block = best_block;

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
                    TRACE_MSG2(_name << ": new search radius=" << radius);
                }
                else {
                    TRACE_MSG2(_name << ": continuing search from block " <<
                               center_block.makeDimValStr(" * "));
                }
            } // beyond next neighbor of center.
        } // search for new setting to try.

        // Fix settings for next step.
        // Assumption is that block size in one pack doesn't affect
        // perf in another pack.
        apply();
        TRACE_MSG2(_name << ": next block-size "  <<
                  _settings->_block_sizes.makeDimValStr(" * "));
    } // eval.

    // Apply auto-tuner settings to prepare for a run.
    void AutoTuner::apply() {
        CONTEXT_VARS(_context);

        // Restore step-dim value for block.
        _settings->_block_sizes[step_posn] = block_steps;
        
        // Change block-based sizes to 0 so adjustSettings()
        // will set them to the default.
        // TODO: tune mini- and sub-block sizes also.
        _settings->_sub_block_sizes.setValsSame(0);
        _settings->_sub_block_group_sizes.setValsSame(0);
        _settings->_mini_block_sizes.setValsSame(0);
        _settings->_mini_block_group_sizes.setValsSame(0);
        _settings->_block_group_sizes.setValsSame(0);

        // Make sure everything is resized based on block size.
        auto saved_op = cp->get_debug_output();
        cp->set_debug_output(nullop);
        _settings->adjustSettings(cp->get_env());

        // Update temporal blocking info.
        cp->update_tb_info();

        // Reallocate scratch data based on new block size.
        // TODO: only do this when blocks have increased or
        // decreased by a certain percentage.
        _context->allocScratchData();
        cp->set_debug_output(saved_op);
    }

} // namespace yask.
