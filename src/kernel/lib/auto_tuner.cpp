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
        n2big = n2small = 0;
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

        // Set min blocks to number of region threads.
        min_blks = _context->set_region_threads();

        // Adjust starting block if needed.
        for (auto dim : center_block.getDims()) {
            auto& dname = dim.getName();
            auto& dval = dim.getVal();

            auto& opts = _context->get_settings();
            auto dmax = max(idx_t(1), opts->_region_sizes[dname] / 2);
            if (dval > dmax || dval < 1)
                center_block[dname] = dmax;
        }
        if (!done) {
            os << _name << ": starting block-size: "  <<
                center_block.makeDimValStr(" * ") << endl;
            os << _name << ": starting search radius: " << radius << endl;
        }
    } // clear.

    // Evaluate the previous run and take next auto-tuner step.
    void AutoTuner::eval(idx_t steps) {
        ostream& os = _context->get_ostr();
        auto& mpiInfo = _context->get_mpi_info();
        auto& dims = _context->get_dims();
        auto& opts = _context->get_settings();

        // Get elapsed time and reset.
        double etime = timer.get_elapsed_secs();
        timer.clear();
        
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
        os << _name << ": " << csteps << " steps(s) at " << rate <<
            " steps/sec with block-size " <<
            _settings->_block_sizes.makeDimValStr(" * ") << endl;
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

                // Convert index to offsets in each dim.
                auto ofs = mpiInfo->neighborhood_sizes.unlayout(neigh_idx);

                // Next neighbor of center point.
                neigh_idx++;

                // Determine new block size.
                IdxTuple bsize(center_block);
                bool ok = true;
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
                    case 0:
                        sz -= step;
                        break;
                    case 1:
                        break;
                    case 2:
                        sz += step;
                        break;
                    default:
                        assert(false && "internal error in tune_settings()");
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
                if (ok && bsize.product() < min_pts) {
                    n2small++;
                    ok = false;
                }

                // Too few?
                else if (ok) {
                    auto& opts = _context->get_settings();
                    idx_t nblks = opts->_region_sizes.product() / bsize.product();
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
                    os << _name << ": new search radius: " << radius << endl;
                }
                else {
                    TRACE_MSG2(_name << ": continuing search from block " <<
                               center_block.makeDimValStr(" * "));
                }
            } // beyond next neighbor of center.
        } // search for new setting to try.

        // Fix settings for next step.
        apply();
        TRACE_MSG2(_name << ": next block-size "  <<
                  _settings->_block_sizes.makeDimValStr(" * "));
    } // eval.

    // Apply auto-tuner settings to prepare for a run.
    void AutoTuner::apply() {
        ostream& os = _context->get_ostr();
        auto& mpiInfo = _context->get_mpi_info();
        auto& dims = _context->get_dims();
        auto& opts = _context->get_settings();
        auto& env = _context->get_env();
        auto step_posn = +Indices::step_posn;

        // Change block-related sizes to 0 so adjustSettings()
        // will set them to the default.
        // Save and restore step-dim value.
        // TODO: tune sub-block sizes also.
        auto step_size = _settings->_sub_block_sizes[step_posn];
        _settings->_sub_block_sizes.setValsSame(0);
        _settings->_sub_block_sizes[step_posn] = step_size;
        step_size = _settings->_sub_block_group_sizes[step_posn];
        _settings->_sub_block_group_sizes.setValsSame(0);
        _settings->_sub_block_group_sizes[step_posn] = step_size;
        step_size = _settings->_block_group_sizes[step_posn];
        _settings->_block_group_sizes.setValsSame(0);
        _settings->_block_group_sizes[step_posn] = step_size;

        // Make sure everything is resized based on block size.
        _settings->adjustSettings(nullop->get_ostream(), env);

        // Update temporal blocking info.
        _context->update_block_info();

        // Reallocate scratch data based on new block size.
        _context->allocScratchData(nullop->get_ostream());
    }

} // namespace yask.
