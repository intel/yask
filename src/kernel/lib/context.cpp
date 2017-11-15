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

#include "yask.hpp"
using namespace std;

namespace yask {

    // APIs.
    // See yask_kernel_api.hpp.

#define GET_SOLN_API(api_name, expr, step_ok, domain_ok, misc_ok)   \
    idx_t StencilContext::api_name(const string& dim) const {           \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        return expr;                                                    \
    }
    GET_SOLN_API(get_first_rank_domain_index, bb_begin[dim], false, true, false)
    GET_SOLN_API(get_last_rank_domain_index, bb_end[dim] - 1, false, true, false)
    GET_SOLN_API(get_overall_domain_size, overall_domain_sizes[dim], false, true, false)
    GET_SOLN_API(get_rank_domain_size, _opts->_rank_sizes[dim], false, true, false)
    GET_SOLN_API(get_min_pad_size, _opts->_min_pad_sizes[dim], false, true, false)
    GET_SOLN_API(get_block_size, _opts->_block_sizes[dim], false, true, false)
    GET_SOLN_API(get_num_ranks, _opts->_num_ranks[dim], false, true, false)
    GET_SOLN_API(get_rank_index, _opts->_rank_indices[dim], false, true, false)
#undef GET_SOLN_API

#define SET_SOLN_API(api_name, expr, step_ok, domain_ok, misc_ok)       \
    void StencilContext::api_name(const string& dim, idx_t n) {         \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        expr;                                                           \
        update_grids();                                                 \
    }
    SET_SOLN_API(set_rank_domain_size, _opts->_rank_sizes[dim] = n, false, true, false)
    SET_SOLN_API(set_min_pad_size, _opts->_min_pad_sizes[dim] = n, false, true, false)
    SET_SOLN_API(set_block_size, _opts->_block_sizes[dim] = n, false, true, false)
    SET_SOLN_API(set_num_ranks, _opts->_num_ranks[dim] = n, false, true, false)
#undef SET_SOLN_API
    
    void StencilContext::share_grid_storage(yk_solution_ptr source) {
        auto sp = dynamic_pointer_cast<StencilContext>(source);
        assert(sp);
        
        for (auto gp : gridPtrs) {
            auto gname = gp->get_name();
            auto si = sp->gridMap.find(gname);
            if (si != sp->gridMap.end()) {
                auto sgp = si->second;
                gp->share_storage(sgp);
            }
        }
    }

    string StencilContext::apply_command_line_options(const string& args) {

        // Create a parser and add base options to it.
        CommandLineParser parser;
        _opts->add_options(parser);

        // Tokenize default args.
        vector<string> argsv;
        parser.set_args(args, argsv);

        // Parse cmd-line options, which sets values in settings.
        parser.parse_args("YASK", argsv);

        // Return any left-over strings.
        string rem;
        for (auto r : argsv) {
            if (rem.length())
                rem += " ";
            rem += r;
        }
        return rem;
    }
    
    ///// StencilContext functions:

    // Set debug output to cout if my_rank == msg_rank
    // or a null stream otherwise.
    ostream& StencilContext::set_ostr() {
        yask_output_factory yof;
        if (_env->my_rank == _opts->msg_rank)
            set_debug_output(yof.new_stdout_output());
        else
            set_debug_output(yof.new_null_output());
        assert(_ostr);
        return *_ostr;
    }
    
    ///// Top-level methods for evaluating reference and optimized stencils.

    // Eval stencil equation group(s) over grid(s) using scalar code.
    void StencilContext::calc_rank_ref()
    {
        run_time.start();

        auto& step_dim = _dims->_step_dim;
        auto step_posn = Indices::step_posn;
        int ndims = _dims->_stencil_dims.getNumDims();
        idx_t begin_t = 0;
        idx_t end_t = _opts->_rank_sizes[step_dim];
        idx_t step_t = _dims->_step_dir;
        assert(abs(step_t) == 1);
        steps_done += abs(end_t - begin_t);

        // backward?
        if (step_t < 0) {
            begin_t = end_t + step_t;
            end_t = step_t;
        }

        // Begin, end, last tuples.
        IdxTuple begin(_dims->_stencil_dims);
        begin.setVals(bb_begin, false);
        begin[step_dim] = begin_t;
        IdxTuple end(_dims->_stencil_dims);
        end.setVals(bb_end, false);
        end[step_dim] = end_t;
        
        TRACE_MSG("calc_rank_ref: " << begin.makeDimValStr() << " ... (end before) " <<
                  end.makeDimValStr());

        // Use the number of threads for a region to help avoid expensive
        // thread-number changing.
        set_region_threads();
        
        // Indices needed for the generated misc loops.
        ScanIndices misc_idxs(ndims);
        misc_idxs.begin = begin;
        misc_idxs.end = end;

        // Initial halo exchange.
        // (Needed in case there are 0 time-steps).
        exchange_halos_all();

        // Number of iterations to get from begin_t, stopping before end_t,
        // stepping by step_t.
        const idx_t num_t = (abs(end_t - begin_t) + (abs(step_t) - 1)) / abs(step_t);
        for (idx_t index_t = 0; index_t < num_t; index_t++)
        {
            // This value of index_t steps from start_t to stop_t-1.
            const idx_t start_t = begin_t + (index_t * step_t);
            const idx_t stop_t = (step_t > 0) ?
                min(start_t + step_t, end_t) :
                max(start_t + step_t, end_t);

            // Set indices that will pass through generated code
            // because the step loop is coded here.
            misc_idxs.index[step_posn] = index_t;
            misc_idxs.start[step_posn] = start_t;
            misc_idxs.stop[step_posn] = stop_t;
            misc_idxs.step[step_posn] = step_t;
        
            // Loop thru eq-groups.
            for (auto* eg : eqGroups) {

                // Halo exchange(s) needed for this eq-group.
                // TODO: do overall problem here w/o halos to catch
                // halo bugs during validation.
                exchange_halos_all();

                // Define misc-loop function.  Since step is always 1, we
                // ignore misc_stop.  If point is in sub-domain for this eq
                // group, then evaluate the reference scalar code.
#define misc_fn(misc_idxs)                                  \
                if (eg->is_in_valid_domain(misc_idxs.start)) \
                    eg->calc_scalar(misc_idxs.start)
                
                // Scan through n-D space.
                TRACE_MSG("calc_rank_ref: step " << start_t);
#include "yask_misc_loops.hpp"
#undef misc_fn
                
                // Remember grids that have been written to by this eq-group,
                // updated at step 'start_t + step_t'.
                mark_grids_dirty(*eg, start_t + step_t);
                
            } // eq-groups.
        } // iterations.

        // Final halo exchange.
        exchange_halos_all();

        run_time.stop();
    }

    // Eval equation group(s) over grid(s) using optimized code.
    void StencilContext::run_solution(idx_t first_step_index,
                                      idx_t last_step_index)
    {
        run_time.start();
        
        auto& step_dim = _dims->_step_dim;
        auto step_posn = Indices::step_posn;
        idx_t begin_t = first_step_index;
        idx_t step_t = _opts->_region_sizes[step_dim] * _dims->_step_dir;
        idx_t end_t = last_step_index + _dims->_step_dir; // end is beyond last.
        int ndims = _dims->_stencil_dims.size();
        steps_done += abs(end_t - begin_t);

        // Begin, end, step, last tuples.
        IdxTuple begin(_dims->_stencil_dims);
        begin.setVals(bb_begin, false);
        begin[step_dim] = begin_t;
        IdxTuple end(_dims->_stencil_dims);
        end.setVals(bb_end, false);
        end[step_dim] = end_t;
        IdxTuple step(_dims->_stencil_dims);
        step.setVals(_opts->_region_sizes, false); // step by region sizes.
        step[step_dim] = step_t;

        TRACE_MSG("run_solution: " << begin.makeDimValStr() << " ... (end before) " <<
                  end.makeDimValStr() << " by " << step.makeDimValStr());
        if (!bb_valid) {
            cerr << "Error: run_solution() called without calling prepare_solution() first.\n";
            exit_yask(1);
        }
        if (bb_size < 1) {
            TRACE_MSG("nothing to do in solution");
            return;
        }
        
#ifdef MODEL_CACHE
        ostream& os = get_ostr();
        if (context.my_rank != context.msg_rank)
            cache_model.disable();
        if (cache_model.isEnabled())
            os << "Modeling cache...\n";
#endif
        
        // Extend end points for overlapping regions due to wavefront angle.
        // For each subsequent time step in a region, the spatial location
        // of each block evaluation is shifted by the angle for each
        // eq-group. So, the total shift in a region is the angle * num
        // stencils * num timesteps. Thus, the number of overlapping regions
        // is ceil(total shift / region size). This assumes all eq-groups
        // are inter-dependent to find minimum extension. Actual required
        // extension may be less, but this will just result in some calls to
        // calc_region() that do nothing.
        //
        // Conceptually (showing 4 regions in t and x dims):
        // -----------------------------  t = rt
        //  \    |\     \     \ |   \     .
        //   \   | \     \     \|    \    .
        //    \  |  \     \     |     \   .
        //     \ |r0 \  r1 \ r2 |\ r3  \  .
        //      \|    \     \   | \     \ .
        // ------------------------------ t = 0
        //       |              |     |
        // x = begin_dx      end_dx end_dx
        //                   (orig) (after extension)
        //
        idx_t nshifts = (idx_t(eqGroups.size()) * abs(step_t)) - 1;
        for (auto& dim : _dims->_domain_dims.getDims()) {
            auto& dname = dim.getName();
            end[dname] += angles[dname] * nshifts;
        }
        TRACE_MSG("after wave-front adjustment: " <<
                  begin.makeDimValStr() << " ... (end before) " <<
                  end.makeDimValStr());

        // Indices needed for the 'rank' loops.
        ScanIndices rank_idxs(ndims);
        rank_idxs.begin = begin;
        rank_idxs.end = end;
        rank_idxs.step = step;

        // Make sure threads are set properly for a region.
        set_region_threads();

        // Initial halo exchange.
        exchange_halos_all();

        // Number of iterations to get from begin_t to end_t-1,
        // stepping by step_t.
        const idx_t num_t = (abs(end_t - begin_t) + (abs(step_t) - 1)) / abs(step_t);
        for (idx_t index_t = 0; index_t < num_t; index_t++)
        {
            // This value of index_t steps from start_t to stop_t-1.
            const idx_t start_t = begin_t + (index_t * step_t);
            const idx_t stop_t = (step_t > 0) ?
                min(start_t + step_t, end_t) :
                max(start_t + step_t, end_t);

            // Set indices that will pass through generated code.
            rank_idxs.index[step_posn] = index_t;
            rank_idxs.start[step_posn] = start_t;
            rank_idxs.stop[step_posn] = stop_t;
            rank_idxs.step[step_posn] = step_t;
            
            // If doing only one time step in a region (default), loop
            // through equations here, and do only one equation group at a
            // time in calc_region(). This is similar to loop in
            // calc_rank_ref().
            if (abs(step_t) == 1) {

                for (auto* eg : eqGroups) {

                    // Halo exchange(s) needed for this eq-group.
                    exchange_halos(start_t, stop_t, *eg);

                    // Eval this eq-group in calc_region().
                    EqGroupSet eqGroup_set;
                    eqGroup_set.insert(eg);
                    EqGroupSet* eqGroup_ptr = &eqGroup_set;

                    // Include automatically-generated loop code that calls
                    // calc_region() for each region.
                    TRACE_MSG("run_solution: step " << start_t);
#include "yask_rank_loops.hpp"

                    // Remember grids that have been written to by this eq-group,
                    // updated at step 'start_t + step_t'.
                    mark_grids_dirty(*eg, start_t + step_t);
                }
            }

            // If doing more than one time step in a region (temporal wave-front),
            // must loop through all eq-groups in calc_region().
            else {

                // TODO: enable reverse time w/wave-fronts.
                if (step_t < 0) {
                    cerr << "Error: reverse time with wave-fronts not yet supported.\n";
                    exit_yask(1);
                }

                // TODO: enable halo exchange for wave-fronts.
                if (_env->num_ranks > 1) {
                    cerr << "Error: halo exchange with wave-fronts not yet supported.\n";
                    exit_yask(1);
                }
                
                // Eval all equation groups.
                EqGroupSet* eqGroup_ptr = NULL;
                
                // Include automatically-generated loop code that calls calc_region() for each region.
                TRACE_MSG("run_solution: steps " << start_t << " ... (end before) " << stop_t);
#include "yask_rank_loops.hpp"
            }

        }

        // Final halo exchange.
        exchange_halos_all();

#ifdef MODEL_CACHE
        // Print cache stats, then disable.
        // Thus, cache is only modeled for first call.
        if (cache_model.isEnabled()) {
            os << "Done modeling cache...\n";
            cache_model.dumpStats();
            cache_model.disable();
        }
#endif
        run_time.stop();
    }

    // Apply solution for time-steps specified in _rank_sizes.
    void StencilContext::calc_rank_opt()
    {
        auto& step_dim = _dims->_step_dim;
        idx_t first_t = 0;
        idx_t last_t = _opts->_rank_sizes[step_dim] - 1;

        // backward?
        if (_dims->_step_dir < 0) {
            first_t = last_t;
            last_t = 0;
        }

        run_solution(first_t, last_t);
    }

    // Calculate results within a region.
    // Each region is typically computed in a separate OpenMP 'for' region.
    // In it, we loop over the time steps and the stencil
    // equations and evaluate the blocks in the region.
    void StencilContext::calc_region(EqGroupSet* eqGroup_set,
                                     const ScanIndices& rank_idxs) {

        int ndims = _dims->_stencil_dims.size();
        auto& step_dim = _dims->_step_dim;
        auto step_posn = Indices::step_posn;
        TRACE_MSG("calc_region: " << rank_idxs.start.makeValStr(ndims) <<
                  " ... (end before) " << rank_idxs.stop.makeValStr(ndims));

        // Init region begin & end from rank start & stop indices.
        ScanIndices region_idxs(ndims);
        region_idxs.initFromOuter(rank_idxs);

        // Make a copy of the original start and stop indices because
        // we will be shifting these for temporal wavefronts.
        Indices rank_start(rank_idxs.start);
        Indices rank_stop(rank_idxs.stop);

        // Not yet supporting temporal blocking.
        if (_opts->_block_sizes[step_dim] != 1) {
            cerr << "Error: temporal blocking not yet supported." << endl;
            exit_yask(1);
        }
        
        // Steps within a region are based on block sizes.
        region_idxs.step = _opts->_block_sizes;

        // Groups in region loops are based on block-group sizes.
        region_idxs.group_size = _opts->_block_group_sizes;

        // Time loop.
        idx_t begin_t = region_idxs.begin[step_posn];
        idx_t end_t = region_idxs.end[step_posn];
        idx_t step_t = region_idxs.step[step_posn];
        const idx_t num_t = (abs(end_t - begin_t) + (abs(step_t) - 1)) / abs(step_t);
        for (idx_t index_t = 0; index_t < num_t; index_t++)
        {
            // This value of index_t steps from start_t to stop_t-1.
            const idx_t start_t = begin_t + (index_t * step_t);
            const idx_t stop_t = (step_t > 0) ?
                min(start_t + step_t, end_t) :
                max(start_t + step_t, end_t);

            // Set indices that will pass through generated code.
            region_idxs.index[step_posn] = index_t;
            region_idxs.start[step_posn] = start_t;
            region_idxs.stop[step_posn] = stop_t;
            
            // equations to evaluate at this time step.
            for (auto* eg : eqGroups) {
                if (!eqGroup_set || eqGroup_set->count(eg)) {
                    TRACE_MSG("calc_region: eq-group '" << eg->get_name() << "' w/BB " <<
                              eg->bb_begin.makeDimValStr() << " ... (end before) " <<
                              eg->bb_end.makeDimValStr());

                    // For wavefront adjustments, see conceptual diagram in
                    // calc_rank_opt().  In this function, 1 of the 4
                    // parallelogram-shaped regions is being evaluated.  At
                    // each time-step, the parallelogram may be trimmed
                    // based on the BB.
                    
                    // Actual region boundaries must stay within BB for this eq group.
                    // Note that i-loop is over domain vars only (skipping over step var).
                    bool ok = true;
                    for (int i = step_posn + 1; i < ndims; i++) {
                        auto& dname = _dims->_stencil_dims.getDimName(i);
                        assert(eg->bb_begin.lookup(dname));
                        region_idxs.begin[i] = max<idx_t>(rank_start[i], eg->bb_begin[dname]);
                        assert(eg->bb_end.lookup(dname));
                        region_idxs.end[i] = min<idx_t>(rank_stop[i], eg->bb_end[dname]);
                        if (region_idxs.end[i] <= region_idxs.begin[i])
                            ok = false;
                    }
                    TRACE_MSG("calc_region, after trimming: " <<
                              region_idxs.begin.makeValStr(ndims) <<
                              " ... (end before) " << region_idxs.end.makeValStr(ndims));
                    
                    // Only need to loop through the spatial extent of the
                    // region if any of its blocks are at least partly
                    // inside the domain. For overlapping regions, they may
                    // start outside the domain but enter the domain as time
                    // progresses and their boundaries shift. So, we don't
                    // want to return if this condition isn't met.
                    if (ok) {

                        // Include automatically-generated loop code that
                        // calls calc_block() for each block in this region.
                        // Loops through x from begin_rx to end_rx-1;
                        // similar for y and z.  This code typically
                        // contains the outer OpenMP loop(s).
#include "yask_region_loops.hpp"

                    }
            
                    // Shift spatial region boundaries for next iteration to
                    // implement temporal wavefront.  We only shift
                    // backward, so region loops must increment. They may do
                    // so in any order.  TODO: shift only what is needed by
                    // this eq-group, not the global max.
                    // Note that i-loop is over domain vars only (skipping over step var).
                    for (int i = step_posn + 1; i < ndims; i++) {
                        auto& dname = _dims->_stencil_dims.getDimName(i);
                        auto angle = angles[dname];
                        rank_start[i] -= angle;
                        rank_stop[i] -= angle;
                    }
                }            
            } // stencil equations.
        } // time.
    }

    // Apply auto-tuning to some of the settings.
    void StencilContext::tune_settings() {
        if (!bb_valid) {
            cerr << "Error: tune_settings() called without calling prepare_solution() first.\n";
            exit_yask(1);
        }
        ostream& os = get_ostr();
        os << "Auto-tuning...\n" << flush;
        YaskTimer at_timer;
        at_timer.start();

        // Temporarily disable halo exchange because tuning is intra-rank.
        enable_halo_exchange = false;
        
        // Results.
        map<IdxTuple, double> run_times;
        int n2big = 0, n2small = 0;

        // AT parameters.
        double min_secs = 0.1;
        double first_min_secs = 1.;
        idx_t min_step = 4;
        idx_t max_radius = 64;
        idx_t min_pts = 512; // 8^3.
        idx_t min_blks = 4;

        // Null stream to throw away debug info.
        yask_output_factory yof;
        auto nullop = yof.new_null_output();
        auto nullosp = &nullop->get_ostream();

        // Best so far; start with current setting.
        IdxTuple best_block(_opts->_block_sizes);
        double best_time = 0.;

        // If the starting block is already very big, reduce it.
        for (auto dim : best_block.getDims()) {
            auto& dname = dim.getName();
            auto& dval = dim.getVal();

            auto dmax = max(idx_t(1), _opts->_region_sizes[dname] / 2);
            if (dval > dmax)
                best_block[dname] = dmax;
        }
        TRACE_MSG("tune_settings: starting block-size "  <<
                  best_block.makeDimValStr(" * "));

        // Loop through decreasing radius for gradient-descent neighbors.
        int nat = 0;
        for (idx_t r = max_radius; r >= 1; r /= 2) {
            os << "  auto-tuner search-radius " << r << "...\n";
        
            // Gradient-descent algorithm loop.
            bool found;
            do {
                found = false;
        
                // Explore immediately-neighboring settings.
                // Use the neighborhood tuple from MPI to do this.
                // TODO: move that tuple to a more general place.
                _mpiInfo->neighborhood_sizes.visitAllPoints
                    ([&](const IdxTuple& ofs, size_t idx) {

                        // Convert offset (0..2) to new block size.
                        bool ok = true;
                        IdxTuple bsize(best_block);
                        for (auto odim : ofs.getDims()) {
                            auto& dname = odim.getName();
                            auto& dofs = odim.getVal();

                            // Min and max sizes of this dim.
                            auto dmin = _dims->_cluster_pts[dname];
                            auto dmax = _opts->_region_sizes[dname];
                            
                            // Determine distance of GD neighbors.
                            auto step = dmin; // step by cluster size.
                            step = max(step, min_step);
                            step *= r;

                            auto sz = best_block[dname];
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
                                ok = false;
                                n2small++;
                                break;
                            }
                            
                            // Adjustments.
                            sz = min(sz, dmax);
                            sz = ROUND_UP(sz, dmin);

                            // Save.
                            bsize[dname] = sz;
                        }
                        TRACE_MSG("tune_settings: checking block-size "  <<
                                  bsize.makeDimValStr(" * "));

                        // Too small?
                        if (ok && bsize.product() < min_pts) {
                            ok = false;
                            n2small++;
                        }

                        // Too few?
                        else if (ok) {
                            idx_t nblks = _opts->_region_sizes.product() / bsize.product();
                            if (nblks < min_blks) {
                                ok = false;
                                n2big++;
                            }
                        }

                        // Bad size or already got these results?
                        if (!ok || run_times.count(bsize))
                            return true; // keep visiting.
                
                        // Set the size in the settings.
                        _opts->_block_sizes = bsize;

                        // Change sub-block size to 0 so adjustSettings()
                        // will set it to the default.
                        _opts->_sub_block_sizes.setValsSame(0);
                        _opts->_sub_block_group_sizes.setValsSame(0);
                
                        // Make sure everything is resized.
                        _opts->adjustSettings(*nullosp);

                        // Run steps until the min time is reached.
                        // Run a few dummy steps the first time for warmup.
                        nat++;
                        int nruns = (nat == 1) ? 2 : 1;
                        double rrate = 0.;
                        for (int i = 0; i < nruns; i++) {
                            clear_timers();
                            double rtime = 0.;
                            bool is_last = i == nruns - 1;
                            double mtime = is_last ? min_secs : first_min_secs;
                            int ndone = 0;
                            do {
                                TRACE_MSG("tune_settings: running step " << ndone <<
                                          " with block-size "  << bsize.makeDimValStr(" * "));
                                run_solution(ndone);
                                rtime = run_time.get_elapsed_secs();
                                ndone++;
                            } while (rtime < mtime);

                            // save perf results.
                            rrate = rtime / ndone; // time per step.
                            run_times[bsize] = rrate;
                        }

                        // Better than previous best?
                        bool is_best = best_time == 0. || rrate < best_time;
                        if (is_best) {
                            best_time = rrate;
                            best_block = bsize;
                            found = true;
                        }
                        
                        // Print results.
                        os << "  auto-tuner experiment # " << nat <<
                            " with block-size " << bsize.makeDimValStr(" * ") << ": " <<
                            run_times[bsize] << " sec/step";
                        if (is_best)
                            os << ", best";
                        os << endl << flush;
                
                        return true;    // keep visiting.

                    });         // neighbors of best point.
            } while (found); // stop GD when no new best is found.
        } // GD radius.

        at_timer.stop();
        os << "Auto-tuner done in " << at_timer.get_elapsed_secs() << " secs.\n";
        if (!nat)
            os << "  No block-size alternatives found (" << n2small << " too small, " <<
            n2big << " too large).\n";
        else {
            os << "  Best block-size: " << best_block.makeDimValStr(" * ") << endl;

            // Set the size in the settings.
            _opts->_block_sizes = best_block;
            
            // Change sub-block size to 0 so adjustSettings()
            // will set it to the default.
            _opts->_sub_block_sizes.setValsSame(0);
            _opts->_sub_block_group_sizes.setValsSame(0);
            
            // Make sure everything is resized.
            _opts->adjustSettings(os);

            // Reset timers a final time for 'real' work.
            clear_timers();
        }

        // reenable halo exchange.
        enable_halo_exchange = true;
    }
    
    // Add a new grid to the containers.
    void StencilContext::addGrid(YkGridPtr gp, bool is_output) {
        auto& gname = gp->get_name();
        if (gridMap.count(gname)) {
            cerr << "Error: grid '" << gname << "' already exists.\n";
            exit_yask(1);
        }

        // Add to list and map.
        gridPtrs.push_back(gp);
        gridMap[gname] = gp;

        // Add to output list and map if 'is_output'.
        if (is_output) {
            outputGridPtrs.push_back(gp);
            outputGridMap[gname] = gp;
        }
    }
    
    // Init MPI-related vars and other vars related to my rank's place in
    // the global problem: rank index, offset, etc.  Need to call this even
    // if not using MPI to properly init these vars.  Called from
    // prepare_solution(), so it doesn't normally need to be called from user code.
    void StencilContext::setupRank()
    {
        ostream& os = get_ostr();
        auto& step_dim = _dims->_step_dim;
        auto me = _env->my_rank;

        // Check ranks.
        idx_t req_ranks = _opts->_num_ranks.product();
        if (req_ranks != _env->num_ranks) {
            cerr << "error: " << req_ranks << " rank(s) requested (" <<
                _opts->_num_ranks.makeDimValStr(" * ") << "), but " <<
                _env->num_ranks << " rank(s) are active." << endl;
            exit_yask(1);
        }
        assertEqualityOverRanks(_opts->_rank_sizes[step_dim], _env->comm, "num steps");

        // Determine my coordinates if not provided already.
        // TODO: do this more intelligently based on proximity.
        if (_opts->find_loc)
            _opts->_rank_indices = _opts->_num_ranks.unlayout(me);

        // A table of rank-coordinates for everyone.
        auto num_ddims = _opts->_rank_indices.size(); // domain-dims only!
        idx_t coords[_env->num_ranks][num_ddims];

        // Init coords for this rank.
        for (int i = 0; i < num_ddims; i++)
            coords[me][i] = _opts->_rank_indices[i];

        // A table of rank-domain sizes for everyone.
        idx_t rsizes[_env->num_ranks][num_ddims];

        // Init sizes for this rank.
        for (int di = 0; di < num_ddims; di++) {
            auto& dname = _opts->_rank_indices.getDimName(di);
            rsizes[me][di] = _opts->_rank_sizes[dname];
        }

#ifdef USE_MPI
        // Exchange coord and size info between all ranks.
        for (int rn = 0; rn < _env->num_ranks; rn++) {
            MPI_Bcast(&coords[rn][0], num_ddims, MPI_INTEGER8,
                      rn, _env->comm);
            MPI_Bcast(&rsizes[rn][0], num_ddims, MPI_INTEGER8,
                      rn, _env->comm);
        }
        // Now, the tables are filled in for all ranks.
#endif

        // Init offsets and total sizes.
        rank_domain_offsets.setValsSame(0);
        overall_domain_sizes.setValsSame(0);

        // Loop over all ranks, including myself.
        int num_neighbors = 0;
        for (int rn = 0; rn < _env->num_ranks; rn++) {

            // Coord offset of rn from me: prev => negative, self => 0, next => positive.
            IdxTuple rcoords(_dims->_domain_dims);
            IdxTuple rdeltas(_dims->_domain_dims);
            for (int di = 0; di < num_ddims; di++) {
                rcoords[di] = coords[rn][di];
                rdeltas[di] = coords[rn][di] - _opts->_rank_indices[di];
            }
        
            // Manhattan distance from rn (sum of abs deltas in all dims).
            // Max distance in any dim.
            int mandist = 0;
            int maxdist = 0;
            for (int di = 0; di < num_ddims; di++) {
                mandist += abs(rdeltas[di]);
                maxdist = max(maxdist, abs(int(rdeltas[di])));
            }
            
            // Myself.
            if (rn == me) {
                if (mandist != 0) {
                    cerr << "Internal error: distance to own rank == " << mandist << endl;
                    exit_yask(1);
                }
            }

            // Someone else.
            else {
                if (mandist == 0) {
                    cerr << "Error: ranks " << me <<
                        " and " << rn << " at same coordinates." << endl;
                    exit_yask(1);
                }
            }

            // Loop through domain dims.
            for (int di = 0; di < num_ddims; di++) {
                auto& dname = _opts->_rank_indices.getDimName(di);

                // Is rank 'rn' in-line with my rank in 'dname' dim?
                // True when deltas in other dims are zero.
                bool is_inline = true;
                for (int dj = 0; dj < num_ddims; dj++) {
                    if (di != dj && rdeltas[dj] != 0) {
                        is_inline = false;
                        break;
                    }
                }

                // Process ranks that are in-line in 'dname', including self.
                if (is_inline) {
                    
                    // Accumulate total problem size in each dim for ranks that
                    // intersect with this rank, including myself.
                    overall_domain_sizes[dname] += rsizes[rn][di];

                    // Adjust my offset in the global problem by adding all domain
                    // sizes from prev ranks only.
                    if (rdeltas[di] < 0)
                        rank_domain_offsets[dname] += rsizes[rn][di];

                    // Make sure all the other dims are the same size.
                    // This ensures that all the ranks' domains line up
                    // properly along their edges and at their corners.
                    for (int dj = 0; dj < num_ddims; dj++) {
                        if (di != dj) {
                            auto mysz = rsizes[me][dj];
                            auto rnsz = rsizes[rn][dj];
                            if (mysz != rnsz) {
                                auto& dnamej = _opts->_rank_indices.getDimName(dj);
                                cerr << "Error: rank " << rn << " and " << me <<
                                    " are both at rank-index " << coords[me][di] <<
                                    " in the '" << dname <<
                                    "' dimension , but their rank-domain sizes are " <<
                                    rnsz << " and " << mysz <<
                                    " (resp.) in the '" << dj <<
                                    "' dimension, making them unaligned.\n";
                                exit_yask(1);
                            }
                        }
                    }
                }
            }

            // Rank rn is myself or my immediate neighbor if its distance <= 1 in
            // every dim.  Assume we do not need to exchange halos except
            // with immediate neighbor. We validate this assumption below by
            // making sure that the rank domain size is at least as big as the
            // largest halo.
            if (maxdist <= 1) {

                // At this point, rdeltas contains only -1..+1 for each domain dim.
                // Add one to -1..+1 to get 0..2 range for my_neighbors offsets.
                IdxTuple roffsets = rdeltas.addElements(1);
                assert(rdeltas.min() >= -1);
                assert(rdeltas.max() <= 1);
                assert(roffsets.min() >= 0);
                assert(roffsets.max() <= 2);

                // Convert the offsets into a 1D index.
                auto rn_ofs = _mpiInfo->getNeighborIndex(roffsets);
                TRACE_MSG("neighborhood size = " << _mpiInfo->neighborhood_sizes.makeDimValStr() <<
                          " & roffsets of rank " << rn << " = " << roffsets.makeDimValStr() <<
                          " => " << rn_ofs);
                assert(idx_t(rn_ofs) < _mpiInfo->neighborhood_size);

                // Save rank of this neighbor into the MPI info object.
                _mpiInfo->my_neighbors.at(rn_ofs) = rn;
                if (rn != me) {
                    num_neighbors++;
                    os << "Neighbor #" << num_neighbors << " is rank " << rn <<
                        " at absolute rank indices " << rcoords.makeDimValStr() <<
                        " (" << rdeltas.makeDimValOffsetStr() << " relative to rank " <<
                        me << ")\n";
                }

                // Save manhattan dist.
                _mpiInfo->man_dists.at(rn_ofs) = mandist;

                // Loop through domain dims.
                bool vlen_mults = true;
                for (int di = 0; di < num_ddims; di++) {
                    auto& dname = _opts->_rank_indices.getDimName(di);

                    // Does rn have all VLEN-multiple sizes?
                    auto rnsz = rsizes[rn][di];
                    auto vlen = _dims->_fold_pts[di];
                    if (rnsz % vlen != 0) {
                        TRACE_MSG("cannot use vector halo exchange with rank " << rn <<
                                  " because its size in '" << dname << "' is " << rnsz);
                        vlen_mults = false;
                    }

                    // Is domain size at least as large as halo in direction
                    // with multiple ranks?
                    if (_opts->_num_ranks[dname] > 1 && rnsz < max_halos[di]) {
                        cerr << "Error: rank-domain size of " << rnsz << " in '" <<
                            dname << "' in rank " << rn <<
                            " is less than largest halo size of " << max_halos[di] << endl;
                        exit_yask(1);
                    }
                }

                // Save vec-mult flag.
                _mpiInfo->has_all_vlen_mults.at(rn_ofs) = vlen_mults;
                
            } // self or immediate neighbor in any direction.
            
        } // ranks.

        // Set offsets in grids.
        update_grids();
    }

    // Allocate memory for grids that do not already have storage.
    // Create MPI buffers.
    // TODO: allow different types of memory for different grids, MPI bufs, etc.
    void StencilContext::allocData() {
        ostream& os = get_ostr();

        // Remove any old MPI data. We do this early to give preference
        // to grids for any HBM memory that might be available.
        mpiData.clear();

        // Base ptrs for all default-alloc'd data.
        // These pointers will be shared by the ones in the grid
        // objects, which will take over ownership when these go
        // out of scope.
        shared_ptr<char> _grid_data_buf;
        shared_ptr<char> _mpi_data_buf;

        // Alloc grid memory.
        // Pass 0: count required size, allocate chunk of memory at end.
        // Pass 1: distribute parts of already-allocated memory chunk.
        for (int pass = 0; pass < 2; pass++) {
            TRACE_MSG("allocData pass " << pass << " for " <<
                      gridPtrs.size() << " grid(s)");
        
            // Determine how many bytes are needed and actually alloc'd.
            size_t gbytes = 0, agbytes = 0;
            int ngrids = 0;
        
            // Grids.
            for (auto gp : gridPtrs) {
                if (!gp)
                    continue;
                auto& gname = gp->get_name();

                // Grid data.
                // Don't alloc if already done.
                if (!gp->is_storage_allocated()) {

                    // Set storage if buffer has been allocated.
                    if (pass == 1) {
                        gp->set_storage(_grid_data_buf, agbytes);
                        gp->print_info(os);
                        os << endl;
                    }

                    // Determine size used (also offset to next location).
                    gbytes += gp->get_num_storage_bytes();
                    agbytes += ROUND_UP(gp->get_num_storage_bytes() + _data_buf_pad,
                                        CACHELINE_BYTES);
                    ngrids++;
                    TRACE_MSG(" grid '" << gname << "' needs " <<
                              makeByteStr(gp->get_num_storage_bytes()));
                }
            }

            // Don't need pad after last one.
            if (agbytes >= _data_buf_pad)
                agbytes -= _data_buf_pad;

            // Allocate data.
            if (pass == 0) {
                os << "Allocating " << makeByteStr(agbytes) <<
                    " for " << ngrids << " grid(s)...\n" << flush;
                _grid_data_buf = shared_ptr<char>(alignedAlloc(agbytes), AlignedDeleter());
            }
        }
        
#ifdef USE_MPI
        int num_exchanges = 0;
        auto me = _env->my_rank;
        
        // Need to determine the size and shape of all MPI buffers.
        // Visit all neighbors.
        _mpiInfo->visitNeighbors
            ([&](const IdxTuple& noffsets, int nrank, int nidx) {
                if (nrank == MPI_PROC_NULL)
                    return;

                // Manhattan dist.
                int mandist = _mpiInfo->man_dists.at(nidx);
                    
                // Check against max dist needed.  TODO: determine max dist
                // automatically from stencil equations; may not be same for all
                // grids.
#ifndef MAX_EXCH_DIST
#define MAX_EXCH_DIST 3
#endif

                // Check distance.
                // TODO: calculate and use exch dist for each grid.
                if (mandist > MAX_EXCH_DIST) {
                    TRACE_MSG("no halo exchange needed with rank " << nrank <<
                              " because L1-norm = " << mandist);
                    return;     // from lambda fn.
                }
        
                // Determine size of MPI buffers between nrank and my rank.
                // Create send and receive buffers for each grid that has a halo
                // between rn and me.
                for (auto gp : gridPtrs) {
                    if (!gp)
                        continue;
                    auto& gname = gp->get_name();

                    // Lookup first & last domain indices and halo sizes
                    // for this grid.
                    bool found_delta = false;
                    IdxTuple halo_sizes, first_idx, last_idx;
                    for (auto& dim : _dims->_domain_dims.getDims()) {
                        auto& dname = dim.getName();
                        if (gp->is_dim_used(dname)) {

                            // Get domain stats for this grid.
                            first_idx.addDimBack(dname, gp->get_first_rank_domain_index(dname));
                            last_idx.addDimBack(dname, gp->get_last_rank_domain_index(dname));
                            auto halo_size = gp->get_halo_size(dname);
                            halo_sizes.addDimBack(dname, halo_size);

                            // Vectorized exchange allowed based on domain sizes?
                            // Both my rank and neighbor rank must have all domain sizes
                            // of vector multiples.
                            bool vec_ok = allow_vec_exchange &&
                                _mpiInfo->has_all_vlen_mults[_mpiInfo->my_neighbor_index] &&
                                _mpiInfo->has_all_vlen_mults[nidx];
                            
                            // Round up halo sizes if vectorized exchanges allowed.
                            // TODO: add a heuristic to avoid increasing by a large factor.
                            if (vec_ok) {
                                auto vec_size = _dims->_fold_pts[dname];
                                halo_sizes.setVal(dname, ROUND_UP(halo_size, vec_size));
                            }
                            
                            // Is there a neighbor in this domain direction?
                            if (noffsets[dname] != MPIInfo::rank_self)
                                found_delta = true;
                        }
                    }

                    // Is buffer needed?
                    if (!found_delta) {
                        TRACE_MSG("no halo exchange needed for grid '" << gname <<
                                  "' with rank " << nrank <<
                                  " because the neighbor is not in a direction"
                                  " corresponding to a grid dim");
                        continue;
                    }

                    // Make a buffer in both directions (send & receive).
                    for (int bd = 0; bd < MPIBufs::nBufDirs; bd++) {

                        // Begin/end vars to indicate what part
                        // of main grid to read from or write to based on
                        // the current neighbor being processed.
                        IdxTuple copy_begin = gp->get_allocs();
                        IdxTuple copy_end = gp->get_allocs();

                        // Adjust along domain dims in this grid.
                        for (auto& dim : halo_sizes.getDims()) {
                            auto& dname = dim.getName();

                            // Init range to whole rank domain (inside halos).
                            // These may be changed below depending on the
                            // neighbor's direction.
                            copy_begin[dname] = first_idx[dname];
                            copy_end[dname] = last_idx[dname] + 1; // end = last + 1.

                            // Neighbor direction in this dim.
                            auto neigh_ofs = noffsets[dname];
                            
                            // Region to read from, i.e., data from inside
                            // this rank's halo to be put into receiver's
                            // halo.
                            if (bd == MPIBufs::bufSend) {

                                // Is this neighbor 'before' me in this dim?
                                if (neigh_ofs == idx_t(MPIInfo::rank_prev)) {

                                    // Only read slice as wide as halo from beginning.
                                    copy_end[dname] = first_idx[dname] + halo_sizes[dname];
                                }
                            
                                // Is this neighbor 'after' me in this dim?
                                else if (neigh_ofs == idx_t(MPIInfo::rank_next)) {
                                    
                                    // Only read slice as wide as halo before end.
                                    copy_begin[dname] = last_idx[dname] + 1 - halo_sizes[dname];
                                }
                            
                                // Else, this neighbor is in same posn as I am in this dim,
                                // so we leave the default begin/end settings.
                            }
                        
                            // Region to write to, i.e., into this rank's halo.
                            else if (bd == MPIBufs::bufRecv) {

                                // Is this neighbor 'before' me in this dim?
                                if (neigh_ofs == idx_t(MPIInfo::rank_prev)) {

                                    // Only read slice as wide as halo before beginning.
                                    copy_begin[dname] = first_idx[dname] - halo_sizes[dname];
                                    copy_end[dname] = first_idx[dname];
                                }
                            
                                // Is this neighbor 'after' me in this dim?
                                else if (neigh_ofs == idx_t(MPIInfo::rank_next)) {
                                    
                                    // Only read slice as wide as halo after end.
                                    copy_begin[dname] = last_idx[dname] + 1;
                                    copy_end[dname] = last_idx[dname] + 1 + halo_sizes[dname];
                                }
                                
                                // Else, this neighbor is in same posn as I am in this dim,
                                // so we leave the default begin/end settings.
                            }
                        } // domain dims in this grid.

                        // Sizes of buffer in all dims of this grid.
                        // Also, set begin/end value for non-domain dims.
                        IdxTuple buf_sizes = gp->get_allocs();
                        bool vlen_mults = true;
                        for (auto& dname : gp->get_dim_names()) {
                            idx_t dsize = 1;

                            // domain dim?
                            if (halo_sizes.lookup(dname)) {
                                dsize = copy_end[dname] - copy_begin[dname];

                                // Check whether size is multiple of vlen.
                                auto vlen = _dims->_fold_pts[dname];
                                if (dsize % vlen != 0)
                                    vlen_mults = false;
                            }

                            // step dim?
                            // Assume only one time-step to exchange.
                            // TODO: fix this when MPI + wave-front is enabled.
                            else if (dname == _dims->_step_dim) {

                                // Use 0..1 as a place-holder range.
                                // The actual values will be supplied during
                                // halo exchange.
                                copy_begin[dname] = 0;
                                copy_end[dname] = 1;
                            }

                            // misc?
                            // Copy over entire range.
                            // TODO: make dirty flags for misc dims in grids.
                            else {
                                dsize = gp->get_alloc_size(dname);
                                copy_begin[dname] = gp->get_first_misc_index(dname);
                                copy_end[dname] = gp->get_last_misc_index(dname) + 1;
                            }

                            // Save computed size.
                            buf_sizes[dname] = dsize;
                                
                        } // all dims in this grid.

                        // Does buffer have non-zero size?
                        if (buf_sizes.size() == 0 || buf_sizes.product() == 0) {
                            TRACE_MSG("no halo exchange needed for grid '" << gname <<
                                      "' with rank " << nrank << " because there is no data to exchange");
                            continue;
                        }

                        // At this point, buf_sizes, copy_begin, and copy_end
                        // should be set for each dim in this grid.
                        // Convert end to last.
                        IdxTuple copy_last = copy_end.subElements(1);

                        // Unique name for buffer based on grid name, direction, and ranks.
                        ostringstream oss;
                        oss << gname;
                        if (bd == MPIBufs::bufSend)
                            oss << "_send_halo_from_" << me << "_to_" << nrank;
                        else if (bd == MPIBufs::bufRecv)
                            oss << "_recv_halo_at_" << me << "_from_" << nrank;
                        string bufname = oss.str();

                        // Make MPI data entry for this grid.
                        auto gbp = mpiData.emplace(gname, _mpiInfo);
                        auto& gbi = gbp.first; // iterator from pair returned by emplace().
                        auto& gbv = gbi->second; // value from iterator.
                        auto& buf = gbv.getBuf(MPIBufs::BufDir(bd), noffsets);

                        // Config buffer for this grid.
                        // (But don't allocate storage yet.)
                        buf.begin_pt = copy_begin;
                        buf.last_pt = copy_last;
                        buf.num_pts = buf_sizes;
                        buf.name = bufname;
                        buf.has_all_vlen_mults = vlen_mults;
                        
                        TRACE_MSG("configured MPI buffer object '" << buf.name <<
                                  "' for rank at relative offsets " <<
                                  noffsets.subElements(1).makeDimValStr() << " with " <<
                                  buf.num_pts.makeDimValStr(" * ") << " = " << buf.get_size() <<
                                  " element(s) to be copied from " << buf.begin_pt.makeDimValStr() <<
                                  " to " << buf.last_pt.makeDimValStr());
                        num_exchanges++;

                    } // send, recv.
                } // grids.
            });   // neighbors.
        TRACE_MSG("number of halo-exchanges needed on this rank: " << num_exchanges);

        // Allocate MPI buffers.
        // Pass 0: count required size, allocate chunk of memory at end.
        // Pass 1: distribute parts of already-allocated memory chunk.
        for (int pass = 0; pass < 2; pass++) {
            TRACE_MSG("allocData pass " << pass << " for " <<
                      mpiData.size() << " MPI data set(s)");
        
            // Determine how many bytes are needed and actually alloc'd.
            size_t bbytes = 0, abbytes = 0; // for MPI buffers.
            int nbufs = 0;
        
            // Grids.
            for (auto gp : gridPtrs) {
                if (!gp)
                    continue;
                auto& gname = gp->get_name();

                // MPI bufs for this grid.
                if (mpiData.count(gname)) {
                    auto& grid_mpi_data = mpiData.at(gname);

                    // Visit buffers for each neighbor for this grid.
                    grid_mpi_data.visitNeighbors
                        ([&](const IdxTuple& roffsets,
                             int rank,
                             int idx,
                             MPIBufs& bufs) {

                            // Send and recv.
                            for (int bd = 0; bd < MPIBufs::nBufDirs; bd++) {
                                auto& buf = grid_mpi_data.getBuf(MPIBufs::BufDir(bd), roffsets);
                                if (buf.get_size() == 0)
                                    continue;

                                if (pass == 1)
                                    buf.set_storage(_mpi_data_buf, abbytes);

                                auto sbytes = buf.get_bytes();
                                bbytes += sbytes;
                                abbytes += ROUND_UP(sbytes + _data_buf_pad,
                                                    CACHELINE_BYTES);
                                nbufs++;
                                TRACE_MSG("  MPI buf '" << buf.name << "' needs " <<
                                          makeByteStr(sbytes));
                            }
                        } );
                }
            }

            // Don't need pad after last one.
            if (abbytes >= _data_buf_pad)
                abbytes -= _data_buf_pad;

            // Allocate data.
            if (pass == 0) {
                os << "Allocating " << makeByteStr(abbytes) <<
                    " for " << nbufs << " MPI buffer(s)...\n" << flush;
                _mpi_data_buf = shared_ptr<char>(alignedAlloc(abbytes), AlignedDeleter());
            }
        }
#endif
    }

    // Set grid sizes and offsets based on settings.
    // Set max halos across grids.
    // This should be called anytime a setting or rank offset is changed.
    void StencilContext::update_grids()
    {
        assert(_opts);

        // Reset halos.
        max_halos = _dims->_domain_dims;
        
        // Loop through each grid.
        for (auto gp : gridPtrs) {

            // Loop through each domain dim.
            for (auto& dim : _dims->_domain_dims.getDims()) {
                auto& dname = dim.getName();

                if (gp->is_dim_used(dname)) {

                    // Rank domains.
                    gp->_set_domain_size(dname, _opts->_rank_sizes[dname]);
                    
                    // Pads.
                    // Set via both 'extra' and 'min'; larger result will be used.
                    gp->set_extra_pad_size(dname, _opts->_extra_pad_sizes[dname]);
                    gp->set_min_pad_size(dname, _opts->_min_pad_sizes[dname]);
                    
                    // Offsets.
                    gp->_set_offset(dname, rank_domain_offsets[dname]);

                    // Update max halo across grids, used for wavefront angles.
                    auto hsz = gp->get_halo_size(dname);
                    max_halos[dname] = max(max_halos[dname], hsz);
                }
            }
        }
    }
    
    // Allocate grids and MPI bufs.
    // Initialize some data structures.
    void StencilContext::prepare_solution() {

        // reset time keepers.
        clear_timers();

        // Don't continue until all ranks are this far.
        _env->global_barrier();

        ostream& os = get_ostr();
#ifdef DEBUG
        os << "*** WARNING: YASK compiled with DEBUG; ignore performance results.\n";
#endif
#if defined(NO_INTRINSICS) && (VLEN > 1)
        os << "*** WARNING: YASK compiled with NO_INTRINSICS; ignore performance results.\n";
#endif
#ifdef MODEL_CACHE
        os << "*** WARNING: YASK compiled with MODEL_CACHE; ignore performance results.\n";
#endif
#ifdef TRACE_MEM
        os << "*** WARNING: YASK compiled with TRACE_MEM; ignore performance results.\n";
#endif
#ifdef TRACE_INTRINSICS
        os << "*** WARNING: YASK compiled with TRACE_INTRINSICS; ignore performance results.\n";
#endif
        
        // Adjust all settings before setting MPI buffers or sizing grids.
        // Prints out final settings.
        _opts->adjustSettings(os);

        // Size grids based on finalized settings.
        update_grids();
        
        // Report ranks.
        os << endl;
        os << "Num ranks: " << _env->get_num_ranks() << endl;
        os << "This rank index: " << _env->get_rank_index() << endl;

        // report threads.
        os << "Num OpenMP procs: " << omp_get_num_procs() << endl;
        set_all_threads();
        os << "Num OpenMP threads: " << omp_get_max_threads() << endl;
        set_region_threads(); // Temporary; just for reporting.
        os << "  Num threads per region: " << omp_get_max_threads() << endl;
        set_block_threads(); // Temporary; just for reporting.
        os << "  Num threads per block: " << omp_get_max_threads() << endl;

        // Set the number of threads for a region to help avoid expensive
        // thread-number changing.
        set_region_threads();

        // Run a dummy nested OMP loop to make sure nested threading is
        // initialized.
#ifdef _OPENMP
#pragma omp parallel for
        for (int i = 0; i < omp_get_max_threads() * 100; i++) {

            idx_t dummy = 0;
            set_block_threads();
#pragma omp parallel for reduction(+:dummy)
            for (int j = 0; j < i * 100; j++) {
                dummy += j;
            }
        }
#endif
        
        // TODO: enable multi-rank wave-front tiling.
        auto& step_dim = _dims->_step_dim;
        if (_opts->_region_sizes[step_dim] > 1 && _env->num_ranks > 1) {
            cerr << "MPI communication is not currently enabled with wave-front tiling." << endl;
            exit_yask(1);
        }

        os << endl;
        os << "Num grids: " << gridPtrs.size() << endl;
        os << "Num grids to be updated: " << outputGridPtrs.size() << endl;
        os << "Num stencil equation-groups: " << eqGroups.size() << endl;
        
        // Set up data based on MPI rank, including grid positions.
        setupRank();

        // Determine bounding-boxes for all eq-groups.
        find_bounding_boxes();

        // Alloc grids and MPI bufs.
        allocData();

        // Report some stats.
        idx_t dt = _opts->_rank_sizes[step_dim];
        os << "\nProblem sizes in points (from smallest to largest):\n"
            " vector-size:          " << _dims->_fold_pts.makeDimValStr(" * ") << endl <<
            " cluster-size:         " << _dims->_cluster_pts.makeDimValStr(" * ") << endl <<
            " sub-block-size:       " << _opts->_sub_block_sizes.makeDimValStr(" * ") << endl <<
            " sub-block-group-size: " << _opts->_sub_block_group_sizes.makeDimValStr(" * ") << endl <<
            " block-size:           " << _opts->_block_sizes.makeDimValStr(" * ") << endl <<
            " block-group-size:     " << _opts->_block_group_sizes.makeDimValStr(" * ") << endl <<
            " region-size:          " << _opts->_region_sizes.makeDimValStr(" * ") << endl <<
            " rank-domain-size:     " << _opts->_rank_sizes.makeDimValStr(" * ") << endl <<
            " overall-problem-size: " << overall_domain_sizes.makeDimValStr(" * ") << endl <<
            endl <<
            "Other settings:\n"
#ifdef USE_MPI
            " num-ranks:            " << _opts->_num_ranks.makeDimValStr(" * ") << endl <<
            " rank-indices:         " << _opts->_rank_indices.makeDimValStr() << endl <<
            " rank-domain-offsets:  " << rank_domain_offsets.makeDimValOffsetStr() << endl <<
#endif
            " stencil-name:         " << get_name() << endl << 
            " vector-len:           " << VLEN << endl <<
            " extra-padding:        " << _opts->_extra_pad_sizes.makeDimValStr() << endl <<
            " minimum-padding:      " << _opts->_min_pad_sizes.makeDimValStr() << endl <<
            " wave-front-angles:    " << angles.makeDimValStr() << endl <<
            " max-halos:            " << max_halos.makeDimValStr() << endl <<
            " L1-prefetch-distance: " << PFD_L1 << endl <<
            " L2-prefetch-distance: " << PFD_L2 << endl <<
            endl;
        
        // sums across eqs for this rank.
        rank_numWrites_1t = 0;
        rank_reads_1t = 0;
        rank_numFpOps_1t = 0;
        for (auto* eg : eqGroups) {
            idx_t updates1 = eg->get_scalar_points_written();
            idx_t updates_domain = updates1 * eg->bb_num_points;
            rank_numWrites_1t += updates_domain;
            idx_t reads1 = eg->get_scalar_points_read();
            idx_t reads_domain = reads1 * eg->bb_num_points;
            rank_reads_1t += reads_domain;
            idx_t fpops1 = eg->get_scalar_fp_ops();
            idx_t fpops_domain = fpops1 * eg->bb_num_points;
            rank_numFpOps_1t += fpops_domain;
            os << "Stats for equation-group '" << eg->get_name() << "':\n" <<
                " sub-domain:                 " << eg->bb_begin.makeDimValStr() <<
                " ... " << eg->bb_end.subElements(1).makeDimValStr() << endl <<
                " sub-domain size:            " << eg->bb_len.makeDimValStr(" * ") << endl <<
                " valid points in sub domain: " << makeNumStr(eg->bb_num_points) << endl <<
                " grid-updates per point:     " << updates1 << endl <<
                " grid-updates in sub-domain: " << makeNumStr(updates_domain) << endl <<
                " grid-reads per point:       " << reads1 << endl <<
                " grid-reads in sub-domain:   " << makeNumStr(reads_domain) << endl <<
                " est FP-ops per point:       " << fpops1 << endl <<
                " est FP-ops in sub-domain:   " << makeNumStr(fpops_domain) << endl;
        }

        // Report total allocation.
        rank_nbytes = get_num_bytes();
        os << "Total allocation in this rank: " <<
            makeByteStr(rank_nbytes) << "\n";
        tot_nbytes = sumOverRanks(rank_nbytes, _env->comm);
        os << "Total overall allocation in " << _env->num_ranks << " rank(s): " <<
            makeByteStr(tot_nbytes) << "\n";
    
        // Various metrics for amount of work.
        rank_numWrites_dt = rank_numWrites_1t * dt;
        tot_numWrites_1t = sumOverRanks(rank_numWrites_1t, _env->comm);
        tot_numWrites_dt = tot_numWrites_1t * dt;

        rank_reads_dt = rank_reads_1t * dt;
        tot_reads_1t = sumOverRanks(rank_reads_1t, _env->comm);
        tot_reads_dt = tot_reads_1t * dt;

        rank_numFpOps_dt = rank_numFpOps_1t * dt;
        tot_numFpOps_1t = sumOverRanks(rank_numFpOps_1t, _env->comm);
        tot_numFpOps_dt = tot_numFpOps_1t * dt;

        rank_domain_1t = bb_num_points;
        rank_domain_dt = rank_domain_1t * dt; // same as _opts->_rank_sizes.product();
        tot_domain_1t = sumOverRanks(rank_domain_1t, _env->comm);
        tot_domain_dt = tot_domain_1t * dt;
    
        // Print some more stats.
        os << endl <<
            "Amount-of-work stats:\n" <<
            " domain-size in this rank for one time-step: " <<
            makeNumStr(rank_domain_1t) << endl <<
            " overall-problem-size in all ranks for one time-step: " <<
            makeNumStr(tot_domain_1t) << endl <<
            endl <<
            " num-writes-required in this rank for one time-step: " <<
            makeNumStr(rank_numWrites_1t) << endl <<
            " num-writes-required in all ranks for one time-step: " <<
            makeNumStr(tot_numWrites_1t) << endl <<
            endl <<
            " num-reads-required in this rank for one time-step: " <<
            makeNumStr(rank_reads_1t) << endl <<
            " num-reads-required in all ranks for one time-step: " <<
            makeNumStr(tot_reads_1t) << endl <<
            endl <<
            " est-FP-ops in this rank for one time-step: " <<
            makeNumStr(rank_numFpOps_1t) << endl <<
            " est-FP-ops in all ranks for one time-step: " <<
            makeNumStr(tot_numFpOps_1t) << endl <<
            endl;

        if (dt > 1) {
            os <<
                " domain-size in this rank for all time-steps: " <<
                makeNumStr(rank_domain_dt) << endl <<
                " overall-problem-size in all ranks for all time-steps: " <<
                makeNumStr(tot_domain_dt) << endl <<
                endl <<
                " num-writes-required in this rank for all time-steps: " <<
                makeNumStr(rank_numWrites_dt) << endl <<
                " num-writes-required in all ranks for all time-steps: " <<
                makeNumStr(tot_numWrites_dt) << endl <<
                endl <<
                " num-reads-required in this rank for all time-steps: " <<
                makeNumStr(rank_reads_dt) << endl <<
                " num-reads-required in all ranks for all time-steps: " <<
                makeNumStr(tot_reads_dt) << endl <<
                endl <<
                " est-FP-ops in this rank for all time-steps: " <<
                makeNumStr(rank_numFpOps_dt) << endl <<
                " est-FP-ops in all ranks for all time-steps: " <<
                makeNumStr(tot_numFpOps_dt) << endl <<
                endl;
        }
        os <<
            "Notes:\n"
            " Domain-sizes and overall-problem-sizes are based on rank-domain sizes\n"
            "  and number of ranks regardless of number of grids or sub-domains.\n"
            " Num-writes-required is based on sum of grid-updates in sub-domain across equation-group(s).\n"
            " Num-reads-required is based on sum of grid-reads in sub-domain across equation-group(s).\n"
            " Est-FP-ops are based on sum of est-FP-ops in sub-domain across equation-group(s).\n"
            "\n";
    }

    /// Get statistics associated with preceding calls to run_solution().
    yk_stats_ptr StencilContext::get_stats() {
        ostream& os = get_ostr();

        // Calc and report perf.
        double rtime = run_time.get_elapsed_secs();
        double mtime = mpi_time.get_elapsed_secs();
        if (rtime > 0.) {
            domain_pts_ps = double(tot_domain_1t * steps_done) / rtime;
            writes_ps= double(tot_numWrites_1t * steps_done) / rtime;
            flops = double(tot_numFpOps_1t * steps_done) / rtime;
        }
        else
            domain_pts_ps = writes_ps = flops = 0.;
        if (steps_done > 0) {
            os <<
                "num-points-per-step:                    " << makeNumStr(tot_domain_1t) << endl <<
                "num-writes-per-step:                    " << makeNumStr(tot_numWrites_1t) << endl <<
                "num-est-FP-ops-per-step:                " << makeNumStr(tot_numFpOps_1t) << endl <<
                "num-steps-done:                         " << makeNumStr(steps_done) << endl <<
                "elapsed-time (sec):                     " << makeNumStr(rtime) << endl <<
                "throughput (num-points/sec):            " << makeNumStr(domain_pts_ps) << endl <<
                "throughput (est-FLOPS):                 " << makeNumStr(flops) << endl <<
                "throughput (num-writes/sec):            " << makeNumStr(writes_ps) << endl;
#ifdef USE_MPI
            os <<
                "time in halo exch (sec):                " << makeNumStr(mtime);
            float pct = 100. * mtime / rtime;
            os << " (" << pct << "%)" << endl;
#endif
        }

        // Fill in return object.
        auto p = make_shared<Stats>();
        p->npts = tot_domain_1t;
        p->nwrites = tot_numWrites_1t;
        p->nfpops = tot_numFpOps_1t;
        p->nsteps = steps_done;
        p->run_time = rtime;
        p->mpi_time = mtime;

        // Clear counters.
        clear_timers();

        return p;
    }
    
    // Dealloc grids, etc.
    void StencilContext::end_solution() {

        // Release any MPI data.
        mpiData.clear();

        // Release grid data.
        for (auto gp : gridPtrs) {
            if (!gp)
                continue;
            gp->release_storage();
        }            
    }

    // Init all grids & params by calling initFn.
    void StencilContext::initValues(function<void (YkGridPtr gp, 
                                                   real_t seed)> realInitFn) {
        ostream& os = get_ostr();
        real_t v = 0.1;
        os << "Initializing grids..." << endl;
        for (auto gp : gridPtrs) {
            realInitFn(gp, v);
            v += 0.01;
        }
    }

    // Compare grids in contexts.
    // Return number of mis-compares.
    idx_t StencilContext::compareData(const StencilContext& ref) const {
        ostream& os = get_ostr();

        os << "Comparing grid(s) in '" << name << "' to '" << ref.name << "'..." << endl;
        if (gridPtrs.size() != ref.gridPtrs.size()) {
            cerr << "** number of grids not equal." << endl;
            return 1;
        }
        idx_t errs = 0;
        for (size_t gi = 0; gi < gridPtrs.size(); gi++) {
            TRACE_MSG("Grid '" << ref.gridPtrs[gi]->get_name() << "'...");
            errs += gridPtrs[gi]->compare(ref.gridPtrs[gi].get());
        }

        return errs;
    }

    // Compute convenience values for a bounding-box.
    void BoundingBox::update_bb(ostream& os,
                                const string& name,
                                StencilContext& context,
                                bool force_full) {

        auto dims = context.get_dims();
        auto& domain_dims = dims->_domain_dims;
        bb_len = bb_end.subElements(bb_begin);
        bb_size = bb_len.product();
        if (force_full)
            bb_num_points = bb_size;
        bb_simple = true;       // assume ok.

        // Solid rectangle?
        if (bb_num_points != bb_size) {
            os << "Warning: '" << name << "' domain has only " <<
                makeNumStr(bb_num_points) <<
                " valid point(s) inside its bounding-box of " <<
                makeNumStr(bb_size) <<
                " point(s); slower scalar calculations will be used.\n";
            bb_simple = false;
        }

        else {

            // Lengths are cluster-length multiples?
            bool is_cluster_mult = true;
            for (auto& dim : domain_dims.getDims()) {
                auto& dname = dim.getName();
                if (bb_len[dname] % dims->_cluster_pts[dname]) {
                    is_cluster_mult = false;
                    break;
                }
            }
            if (!is_cluster_mult) {
                os << "Warning: '" << name << "' domain"
                    " has one or more sizes that are not vector-cluster multiples;"
                    " slower scalar calculations will be used.\n";
                bb_simple = false;
            }

            else {

                // Does everything start on a vector-length boundary?
                bool is_aligned = true;
                for (auto& dim : domain_dims.getDims()) {
                    auto& dname = dim.getName();
                    if ((bb_begin[dname] - context.rank_domain_offsets[dname]) %
                        dims->_fold_pts[dname]) {
                        is_aligned = false;
                        break;
                    }
                }
                if (!is_aligned) {
                    os << "Warning: '" << name << "' domain"
                        " has one or more starting edges not on vector boundaries;"
                        " slower scalar calculations will be used.\n";
                    bb_simple = false;
                }
            }
        }

        // All done.
        bb_valid = true;
    }
    
    // Set the bounding-box for each eq-group and whole domain.
    // Also sets wave-front angles.
    void StencilContext::find_bounding_boxes()
    {
        ostream& os = get_ostr();

        // Find BB for each eq group.
        for (auto eg : eqGroups)
            eg->find_bounding_box();

        // Overall BB based only on rank offsets and rank domain sizes.
        bb_begin = rank_domain_offsets;
        bb_end = rank_domain_offsets.addElements(_opts->_rank_sizes, false);
        update_bb(os, "rank", *this, true);

        // Determine the max spatial skewing angles for temporal wavefronts
        // based on the max halos.  This assumes the smallest granularity of
        // calculation is CPTS_* in each dim.  We only need non-zero angles
        // if the region size is less than the rank size, i.e., if the
        // region covers the whole rank in a given dimension, no wave-front
        // is needed in thar dim.
        // TODO: make rounding-up an option.
        for (auto& dim : _dims->_domain_dims.getDims()) {
            auto& dname = dim.getName();
            angles[dname] = (_opts->_region_sizes[dname] < bb_len[dname]) ?
                ROUND_UP(max_halos[dname], _dims->_cluster_pts[dname]) : 0;
        }
    }

    // Exchange dirty halo data for all grids, regardless
    // of their eq-group.
    void StencilContext::exchange_halos_all() {

#ifdef USE_MPI
        TRACE_MSG("exchange_halos_all()...");

        // Find max steps stored over all grids.
        auto& sd = _dims->_step_dim;
        idx_t start = 0, stop = 1;
        for (auto gp : gridPtrs) {
            if (gp->is_dim_used(sd)) {
                start = min(start, gp->_get_first_alloc_index(sd));
                stop = max(stop, gp->_get_last_alloc_index(sd) + 1);
            }
        }
        
        // Initial halo exchange for each eq-group.
        for (auto* eg : eqGroups) {

            // Do exchange over max steps.
            exchange_halos(start, stop, *eg);
        }
#endif
    }
    
    // Exchange halo data needed by eq-group 'eg' at the given time.
    // Data is needed for input grids that have not already been updated.
    // [BIG] TODO: overlap halo exchange with computation.
    void StencilContext::exchange_halos(idx_t start, idx_t stop, EqGroupBase& eg)
    {
#ifdef USE_MPI
        if (!enable_halo_exchange || _env->num_ranks < 2)
            return;
        mpi_time.start();
        TRACE_MSG("exchange_halos: " << start << " ... (end before) " << stop <<
                  " for eq-group '" << eg.get_name() << "'");
        auto opts = get_settings();
        auto& sd = _dims->_step_dim;

        // 1D array to store send request handles.
        // We use a 1D array so we can call MPI_Waitall().
        MPI_Request send_reqs[eg.inputGridPtrs.size() * _mpiInfo->neighborhood_size];

        // 2D array for receive request handles.
        // We use a 2D array to simplify individual indexing.
        MPI_Request recv_reqs[eg.inputGridPtrs.size()][_mpiInfo->neighborhood_size];

        // Loop through steps.  This loop has to be outside halo-step loop
        // because we only have one buffer per step. Normally, we only
        // exchange one step; in that case, it doesn't matter.
        // TODO: this will need to be addressed if/when comm/compute overlap is added.
        assert(start != stop);
        idx_t step = (start < stop) ? 1 : -1;
        for (idx_t t = start; t != stop; t += step) {
            int num_send_reqs = 0;

            // Sequence of things to do for each grid's neighbors
            // (isend includes packing).
            enum halo_steps { halo_irecv, halo_pack_isend, halo_unpack, halo_nsteps };
            for (int hi = 0; hi < halo_nsteps; hi++) {

                if (hi == halo_irecv)
                    TRACE_MSG("exchange_halos: requesting data for step " << t << "...");
                else if (hi == halo_pack_isend)
                    TRACE_MSG("exchange_halos: packing and sending data for step " << t << "...");
                else if (hi == halo_unpack)
                    TRACE_MSG("exchange_halos: unpacking data for step " << t << "...");
            
                // Loop thru all input grids in this group.
                for (size_t gi = 0; gi < eg.inputGridPtrs.size(); gi++) {
                    auto gp = eg.inputGridPtrs[gi];
                    MPI_Request* grid_recv_reqs = recv_reqs[gi];

                    // Only need to swap grids whose halos are not up-to-date
                    // for this step.
                    if (!gp->is_dirty(t))
                        continue;

                    // Only need to swap grids that have MPI buffers.
                    auto& gname = gp->get_name();
                    if (mpiData.count(gname) == 0)
                        continue;
                    TRACE_MSG(" for grid '" << gname << "'...");

                    // Visit all this rank's neighbors.
                    auto& grid_mpi_data = mpiData.at(gname);
                    grid_mpi_data.visitNeighbors
                        ([&](const IdxTuple& offsets, // NeighborOffset.
                             int neighbor_rank,
                             int ni, // 1D index.
                             MPIBufs& bufs) {
                            auto& sendBuf = bufs.bufs[MPIBufs::bufSend];
                            auto& recvBuf = bufs.bufs[MPIBufs::bufRecv];
                            
                            // Nothing to do if there are no buffers.
                            if (sendBuf.get_size() == 0)
                                return;
                            assert(recvBuf.get_size() != 0);
                            assert(sendBuf._elems != 0);
                            assert(recvBuf._elems != 0);
                            TRACE_MSG("  with rank " << neighbor_rank << " at relative position " <<
                                      offsets.subElements(1).makeDimValOffsetStr() << "...");

                            // Vectorized exchange allowed based on domain sizes?
                            // Both my rank and neighbor rank must have all domain sizes
                            // of vector multiples.
                            // We will also need to check the sizes of the buffers.
                            // This is required to guarantee that the vector alignment
                            // would be identical between buffers.
                            bool vec_ok = allow_vec_exchange &&
                                _mpiInfo->has_all_vlen_mults[_mpiInfo->my_neighbor_index] &&
                                _mpiInfo->has_all_vlen_mults[ni];
                         
                            // Submit async request to receive data from neighbor.
                            if (hi == halo_irecv) {
                                auto nbytes = recvBuf.get_bytes();
                                void* buf = (void*)recvBuf._elems;
                                TRACE_MSG("   requesting " << makeByteStr(nbytes) << "...");
                                MPI_Irecv(buf, nbytes, MPI_BYTE,
                                          neighbor_rank, int(gi), _env->comm, &grid_recv_reqs[ni]);
                            }

                            // Pack data into send buffer, then send to neighbor.
                            else if (hi == halo_pack_isend) {

                                // Vec ok?
                                // Domain sizes must be ok, and buffer size must be ok
                                // as calculated when buffers were created.
                                bool send_vec_ok = vec_ok && sendBuf.has_all_vlen_mults;

                                // Get first and last ranges.
                                IdxTuple first = sendBuf.begin_pt;
                                IdxTuple last = sendBuf.last_pt;

                                // The code in allocData() pre-calculated the size and
                                // starting points of each buffer, except for the starting
                                // point of the time-step, which is an argument to this
                                // function.  So, we need to set that value now.
                                if (gp->is_dim_used(sd)) {
                                    first.setVal(sd, t);
                                    last.setVal(sd, t);
                                }
                                TRACE_MSG("   packing " << sendBuf.num_pts.makeDimValStr(" * ") <<
                                          " points from " << first.makeDimValStr() <<
                                          " to " << last.makeDimValStr() <<
                                          (send_vec_ok ? " with" : " without") <<
                                          " vector copy...");

                                // Copy data from grid to buffer.
                                void* buf = (void*)sendBuf._elems;
                                if (send_vec_ok)
                                    gp->get_vecs_in_slice(buf, first, last);
                                else
                                    gp->get_elements_in_slice(buf, first, last);

                                // Send packed buffer to neighbor.
                                auto nbytes = sendBuf.get_bytes();
                                TRACE_MSG("   sending " << makeByteStr(nbytes) << "...");
                                MPI_Isend(buf, nbytes, MPI_BYTE,
                                          neighbor_rank, int(gi), _env->comm,
                                          &send_reqs[num_send_reqs++]);
                            }

                            // Wait for data from neighbor, then unpack it.
                            else if (hi == halo_unpack) {

                                // Wait for data from neighbor before unpacking it.
                                TRACE_MSG("   waiting for MPI data...");
                                MPI_Wait(&grid_recv_reqs[ni], MPI_STATUS_IGNORE);

                                // Vec ok?
                                bool recv_vec_ok = vec_ok && recvBuf.has_all_vlen_mults;

                                // Get first and last ranges.
                                IdxTuple first = recvBuf.begin_pt;
                                IdxTuple last = recvBuf.last_pt;

                                // Set step val as above.
                                if (gp->is_dim_used(sd)) {
                                    first.setVal(sd, t);
                                    last.setVal(sd, t);
                                }
                                TRACE_MSG("   got data; unpacking " << recvBuf.num_pts.makeDimValStr(" * ") <<
                                          " points into " << first.makeDimValStr() <<
                                          " to " << last.makeDimValStr() <<
                                          (recv_vec_ok ? " with" : " without") <<
                                          " vector copy...");

                                // Copy data from buffer to grid.
                                void* buf = (void*)recvBuf._elems;
                                idx_t n = 0;
                                if (recv_vec_ok)
                                    n = gp->set_vecs_in_slice(buf, first, last);
                                else
                                    n = gp->set_elements_in_slice(buf, first, last);
                                assert(n == recvBuf.get_size());
                            }
                        }); // visit neighbors.

                } // grids.
            } // exchange sequence.
            
            // Mark grids as up-to-date.
            for (size_t gi = 0; gi < eg.inputGridPtrs.size(); gi++) {
                auto gp = eg.inputGridPtrs[gi];
                if (gp->is_dirty(t)) {
                    gp->set_dirty(false, t);
                    TRACE_MSG("grid '" << gp->get_name() <<
                              "' marked as clean at step " << t);
                }
            }

            // Wait for all send requests to complete.
            if (num_send_reqs) {
                TRACE_MSG("exchange_halos: waiting for " << num_send_reqs <<
                          " MPI send request(s) to complete...");
                MPI_Waitall(num_send_reqs, send_reqs, MPI_STATUS_IGNORE);
                TRACE_MSG(" done waiting for MPI send request(s)");
            }
            else
                TRACE_MSG("exchange_halos: no MPI send requests to wait for");
        } // steps.
        
        mpi_time.stop();
#endif
    }

    // Mark grids that have been written to by eq-group 'eg'.
    // TODO: only mark grids that are written to in their halo-read area.
    // TODO: add index for misc dim(s).
    void StencilContext::mark_grids_dirty(EqGroupBase& eg, idx_t step_idx)
    {
        for (auto gp : eg.outputGridPtrs) {
            gp->set_dirty(true, step_idx);
            TRACE_MSG("grid '" << gp->get_name() <<
                      "' marked as dirty at step " << step_idx);
        }
    }

} // namespace yask.
