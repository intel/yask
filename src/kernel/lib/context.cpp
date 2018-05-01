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

// This file contains implementations of StencilContext methods.
// Also see context_setup.cpp.

#include "yask.hpp"
using namespace std;

namespace yask {

    // APIs.
    // See yask_kernel_api.hpp.

#define GET_SOLN_API(api_name, expr, step_ok, domain_ok, misc_ok, prep_req) \
    idx_t StencilContext::api_name(const string& dim) const {           \
        if (prep_req && !rank_bb.bb_valid)                              \
            THROW_YASK_EXCEPTION("Error: '" #api_name \
                                 "()' called before calling 'prepare_solution()'"); \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        return expr;                                                    \
    }
    GET_SOLN_API(get_rank_domain_size, _opts->_rank_sizes[dim], false, true, false, false)
    GET_SOLN_API(get_region_size, _opts->_region_sizes[dim], true, true, false, false)
    GET_SOLN_API(get_min_pad_size, _opts->_min_pad_sizes[dim], false, true, false, false)
    GET_SOLN_API(get_block_size, _opts->_block_sizes[dim], false, true, false, false)
    GET_SOLN_API(get_num_ranks, _opts->_num_ranks[dim], false, true, false, false)
    GET_SOLN_API(get_first_rank_domain_index, rank_bb.bb_begin[dim], false, true, false, true)
    GET_SOLN_API(get_last_rank_domain_index, rank_bb.bb_end[dim] - 1, false, true, false, true)
    GET_SOLN_API(get_overall_domain_size, overall_domain_sizes[dim], false, true, false, true)
    GET_SOLN_API(get_rank_index, _opts->_rank_indices[dim], false, true, false, true)
#undef GET_SOLN_API

    // The grid sizes updated any time these settings are changed.
#define SET_SOLN_API(api_name, expr, step_ok, domain_ok, misc_ok, reset_prep) \
    void StencilContext::api_name(const string& dim, idx_t n) {         \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        expr;                                                           \
        update_grids();                                                 \
        if (reset_prep) rank_bb.bb_valid = ext_bb.bb_valid = false;     \
    }
    SET_SOLN_API(set_min_pad_size, _opts->_min_pad_sizes[dim] = n, false, true, false, false)
    SET_SOLN_API(set_block_size, _opts->_block_sizes[dim] = n, false, true, false, false)
    SET_SOLN_API(set_region_size, _opts->_region_sizes[dim] = n, true, true, false, true)
    SET_SOLN_API(set_rank_domain_size, _opts->_rank_sizes[dim] = n, false, true, false, true)
    SET_SOLN_API(set_num_ranks, _opts->_num_ranks[dim] = n, false, true, false, true)
#undef SET_SOLN_API

    // Constructor.
    StencilContext::StencilContext(KernelEnvPtr env,
                                   KernelSettingsPtr settings) :
    _ostr(&std::cout),
        _env(env),
        _opts(settings),
        _dims(settings->_dims),
        _at(this)
    {
        // Set debug output object.
        yask_output_factory yof;
        set_debug_output(yof.new_stdout_output());

        // Create MPI Info object.
        _mpiInfo = std::make_shared<MPIInfo>(settings->_dims);

        // Init various tuples to make sure they have the correct dims.
        rank_domain_offsets = _dims->_domain_dims;
        rank_domain_offsets.setValsSame(-1); // indicates prepare_solution() not called.
        overall_domain_sizes = _dims->_domain_dims;
        max_halos = _dims->_domain_dims;
        wf_angles = _dims->_domain_dims;
        wf_shifts = _dims->_domain_dims;
        left_wf_exts = _dims->_domain_dims;
        right_wf_exts = _dims->_domain_dims;
            
        // Set output to msg-rank per settings.
        set_ostr();
    }

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

    // Eval stencil bundle(s) over grid(s) using reference scalar code.
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

        // Begin & end tuples.
        // Based on rank bounding box, not extended
        // BB because we don't use wave-fronts in the ref code.
        IdxTuple begin(_dims->_stencil_dims);
        begin.setVals(rank_bb.bb_begin, false);
        begin[step_dim] = begin_t;
        IdxTuple end(_dims->_stencil_dims);
        end.setVals(rank_bb.bb_end, false);
        end[step_dim] = end_t;
        
        TRACE_MSG("calc_rank_ref: " << begin.makeDimValStr() << " ... (end before) " <<
                  end.makeDimValStr());

        // Force region & block sizes to whole rank size so that scratch
        // grids will be large enough.
        _opts->_region_sizes.setValsSame(0);
        _opts->_block_sizes.setValsSame(0);
        _at.apply();

        // Use only one set of scratch grids.
        int scratch_grid_idx = 0;
        
        // Indices to loop through.
        // Init from begin & end tuples.
        ScanIndices rank_idxs(*_dims, false, &rank_domain_offsets);
        rank_idxs.begin = begin;
        rank_idxs.end = end;

        // Set offsets in scratch grids.
        // Requires scratch grids to be allocated for whole
        // rank instead of smaller grid size.
        update_scratch_grids(scratch_grid_idx, rank_idxs.begin);
            
        // Initial halo exchange.
        // (Needed in case there are 0 time-steps).
        // TODO: get rid of all halo exchanges in this function,
        // and calculate overall problem in one rank.
        exchange_halos_all();

        // Number of iterations to get from begin_t, stopping before end_t,
        // stepping by step_t.
        const idx_t num_t = abs(end_t - begin_t);
        for (idx_t index_t = 0; index_t < num_t; index_t++)
        {
            // This value of index_t steps from start_t to stop_t-1.
            const idx_t start_t = begin_t + (index_t * step_t);
            const idx_t stop_t = (step_t > 0) ?
                min(start_t + step_t, end_t) :
                max(start_t + step_t, end_t);

            // Set indices that will pass through generated code
            // because the step loop is coded here.
            rank_idxs.index[step_posn] = index_t;
            rank_idxs.start[step_posn] = start_t;
            rank_idxs.stop[step_posn] = stop_t;
            rank_idxs.step[step_posn] = step_t;

            // Loop thru bundles.
            for (auto* asg : stBundles) {

                // Don't do scratch updates here.
                if (asg->is_scratch())
                    continue;

                // Scan through n-D space.
                TRACE_MSG("calc_rank_ref: step " << start_t <<
                          " in non-scratch group '" << asg->get_name());
                
                // Exchange all dirty halos.
                exchange_halos_all();

                // Find the groups that need to be processed.
                // This will be the prerequisite scratch-grid
                // groups plus this non-scratch group.
                auto sg_list = asg->get_scratch_deps();
                sg_list.push_back(asg);

                // Loop through all the needed bundles.
                for (auto* sg : sg_list) {

                    // Indices needed for the generated misc loops.  Will normally be a
                    // copy of rank_idxs except when updating scratch-grids.
                    ScanIndices misc_idxs = sg->adjust_scan(scratch_grid_idx, rank_idxs);
                    misc_idxs.step.setFromConst(1); // ensure unit step.
                
                    // Define misc-loop function.  Since step is always 1, we
                    // ignore misc_stop.  If point is in sub-domain for this
                    // bundle, then evaluate the reference scalar code.
                    // TODO: fix domain of scratch grids.
#define misc_fn(misc_idxs)   do {                                       \
                        if (sg->is_in_valid_domain(misc_idxs.start))    \
                            sg->calc_scalar(scratch_grid_idx, misc_idxs.start); \
                    } while(0)
                
                    // Scan through n-D space.
                    TRACE_MSG("calc_rank_ref: step " << start_t <<
                              " in bundle '" << sg->get_name() << "': " <<
                              misc_idxs.begin.makeValStr(ndims) <<
                              " ... (end before) " << misc_idxs.end.makeValStr(ndims));
#include "yask_misc_loops.hpp"
#undef misc_fn
                
                    // Mark grids that [may] have been written to by this bundle,
                    // updated at next step (+/- 1).
                    // Mark grids as dirty even if not actually written by this
                    // rank. This is needed because neighbors will not know what
                    // grids are actually dirty, and all ranks must have the same
                    // information about which grids are possibly dirty.
                    mark_grids_dirty(start_t + step_t, stop_t + step_t, *sg);
                
                } // needed bundles.
            } // all bundles.

        } // iterations.

        // Final halo exchange.
        exchange_halos_all();

        run_time.stop();
    }

    // Eval stencil bundle(s) over grid(s) using optimized code.
    void StencilContext::run_solution(idx_t first_step_index,
                                      idx_t last_step_index)
    {
        run_time.start();
        
        auto& step_dim = _dims->_step_dim;
        auto step_posn = Indices::step_posn;
        auto step_dir = _dims->_step_dir;
        int ndims = _dims->_stencil_dims.size();

        // Find begin, step and end in step-dim.
        idx_t begin_t = first_step_index;

        // Step-size in step-dim is number of region steps.
        // Then, it is multipled by +/-1 to get proper direction.
        idx_t step_t = _opts->_region_sizes[step_dim];
        step_t *= step_dir;
        assert(step_t);
        idx_t end_t = last_step_index + step_dir; // end is beyond last.

        // Begin, end, step tuples.
        // Based on overall bounding box, which includes
        // any needed extensions for wave-fronts.
        IdxTuple begin(_dims->_stencil_dims);
        begin.setVals(ext_bb.bb_begin, false);
        begin[step_dim] = begin_t;
        IdxTuple end(_dims->_stencil_dims);
        end.setVals(ext_bb.bb_end, false);
        end[step_dim] = end_t;
        IdxTuple step(_dims->_stencil_dims);
        step.setVals(_opts->_region_sizes, false); // step by region sizes.
        step[step_dim] = step_t;

        TRACE_MSG("run_solution: " << begin.makeDimValStr() << " ... (end before) " <<
                  end.makeDimValStr() << " by " << step.makeDimValStr());
        if (!rank_bb.bb_valid)
            THROW_YASK_EXCEPTION("Error: run_solution() called without calling prepare_solution() first");
        if (ext_bb.bb_size < 1) {
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
        // stencil-bundle. So, the total shift in a region is the angle * num
        // stencils * num timesteps. This assumes all bundles
        // are inter-dependent to find maximum extension. Actual required
        // extension may be less, but this will just result in some calls to
        // calc_region() that do nothing.
        //
        // Conceptually (showing 2 ranks in t and x dims):
        // -----------------------------  t = rt ------------------------------
        //   \   | \     \     \|  \      .          / |  \     \     \|  \   |
        //    \  |  \     \     |   \     .         / \|   \     \     |   \  |
        //     \ |r0 \  r1 \ r2 |\ r3\    .        /r0 | r1 \  r2 \ r3 |\ r4\ |
        //      \|    \     \   | \   \   .       /    |\    \     \   | \   \|
        // ------------------------------ t = 0 -------------------------------
        //       |   rank 0     |      |         |     |   rank 1      |      |
        // x = begin_dx      end_dx end_dx    begin_dx begin_dx     end_dx end_dx
        //     (rank)        (rank) (ext)     (ext)    (rank)       (rank) (adj)
        //
        //                      |XXXXXX|         |XXXXX|  <- redundant calculations.          
        // XXXXXX|  <- areas outside of outer ranks not calculated ->  |XXXXXXX
        //
        if (abs(step_t) > 1) {
            for (auto& dim : _dims->_domain_dims.getDims()) {
                auto& dname = dim.getName();

                // The end should be adjusted if there is not
                // already an extension.
                if (right_wf_exts[dname] == 0)
                    end[dname] += wf_shifts[dname];
            }
            TRACE_MSG("after adjustment for " << num_wf_shifts <<
                      " wave-front shift(s): " <<
                      begin.makeDimValStr() << " ... (end before) " <<
                      end.makeDimValStr());
        }

        // Indices needed for the 'rank' loops.
        ScanIndices rank_idxs(*_dims, true, &rank_domain_offsets);
        rank_idxs.begin = begin;
        rank_idxs.end = end;
        rank_idxs.step = step;

        // Make sure threads are set properly for a region.
        set_region_threads();

        // Initial halo exchange.
        exchange_halos_all();

        // Number of iterations to get from begin_t to end_t-1,
        // stepping by step_t.
        const idx_t num_t = CEIL_DIV(abs(end_t - begin_t), abs(step_t));
        for (idx_t index_t = 0; index_t < num_t; index_t++)
        {
            YaskTimer rtime;   // just for these step_t steps.
            rtime.start();
            
            // This value of index_t steps from start_t to stop_t-1.
            const idx_t start_t = begin_t + (index_t * step_t);
            const idx_t stop_t = (step_t > 0) ?
                min(start_t + step_t, end_t) :
                max(start_t + step_t, end_t);
            idx_t this_num_t = abs(stop_t - start_t);

            // Set indices that will pass through generated code.
            rank_idxs.index[step_posn] = index_t;
            rank_idxs.start[step_posn] = start_t;
            rank_idxs.stop[step_posn] = stop_t;
            rank_idxs.step[step_posn] = step_t;
            
            // If no wave-fronts (default), loop through bundles here, and do
            // only one bundle at a time in calc_region(). This is similar to
            // loop in calc_rank_ref().
            if (step_t == 1) {

                for (auto* sg : stBundles) {

                    // Don't do scratch updates here.
                    if (sg->is_scratch())
                        continue;

                    // Exchange halo(s) needed for this bundle.
                    exchange_halos(start_t, stop_t, sg);

                    // Eval this bundle in calc_region().
                    StencilBundleSet stBundle_set;
                    stBundle_set.insert(sg);
                    StencilBundleSet* stBundle_ptr = &stBundle_set;

                    // Include automatically-generated loop code that calls
                    // calc_region() for each region.
                    TRACE_MSG("run_solution: step " << start_t <<
                              " in bundle '" << sg->get_name() << "'");
#include "yask_rank_loops.hpp"
                }
            }

            // If doing wave-fronts, must loop through all bundles in
            // calc_region().  TODO: consider making this the only case,
            // allowing all bundles to be done between MPI exchanges, even
            // w/o wave-fronts.
            else {

                // Exchange all dirty halo(s).
                exchange_halos_all();
                
                // Eval all stencil bundles.
                StencilBundleSet* stBundle_ptr = NULL;
                
                // Include automatically-generated loop code that calls calc_region() for each region.
                TRACE_MSG("run_solution: steps " << start_t << " ... (end before) " << stop_t);
#include "yask_rank_loops.hpp"
            }

            steps_done += this_num_t;
            rtime.stop();   // for these steps.

            // Call the auto-tuner to evaluate these steps.
            // TODO: remove MPI time.
            auto elapsed_time = rtime.get_elapsed_secs();
            _at.eval(this_num_t, elapsed_time);
            
        } // step loop.

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

        // Final halo exchange.
        exchange_halos_all();
    }

    // Calculate results within a region.
    // Each region is typically computed in a separate OpenMP 'for' region.
    // In it, we loop over the time steps and the stencils
    // and evaluate the blocks in the region.
    void StencilContext::calc_region(StencilBundleSet* stBundle_set,
                                     const ScanIndices& rank_idxs) {

        int ndims = _dims->_stencil_dims.size();
        auto& step_dim = _dims->_step_dim;
        auto step_posn = Indices::step_posn;
        TRACE_MSG("calc_region: " << rank_idxs.start.makeValStr(ndims) <<
                  " ... (end before) " << rank_idxs.stop.makeValStr(ndims));

        // Init region begin & end from rank start & stop indices.
        ScanIndices region_idxs(*_dims, true, &rank_domain_offsets);
        region_idxs.initFromOuter(rank_idxs);

        // Make a copy of the original start and stop indices because
        // we will be shifting these for temporal wavefronts.
        Indices start(rank_idxs.start);
        Indices stop(rank_idxs.stop);

        // Not yet supporting temporal blocking.
        if (_opts->_block_sizes[step_dim] != 1)
            THROW_YASK_EXCEPTION("Error: temporal blocking not yet supported");
        
        // Steps within a region are based on block sizes.
        region_idxs.step = _opts->_block_sizes;

        // Groups in region loops are based on block-group sizes.
        region_idxs.group_size = _opts->_block_group_sizes;

        // Time loop.
        idx_t begin_t = region_idxs.begin[step_posn];
        idx_t end_t = region_idxs.end[step_posn];
        idx_t step_t = region_idxs.step[step_posn];
        const idx_t num_t = CEIL_DIV(abs(end_t - begin_t), abs(step_t));
        for (idx_t index_t = 0; index_t < num_t; index_t++) {
            
            // This value of index_t steps from start_t to stop_t-1.
            const idx_t start_t = begin_t + (index_t * step_t);
            const idx_t stop_t = (step_t > 0) ?
                min(start_t + step_t, end_t) :
                max(start_t + step_t, end_t);

            // Set indices that will pass through generated code.
            region_idxs.index[step_posn] = index_t;
            region_idxs.start[step_posn] = start_t;
            region_idxs.stop[step_posn] = stop_t;
            
            // Stencil bundles to evaluate at this time step.
            for (auto* sg : stBundles) {

                // Don't do scratch updates here.
                if (sg->is_scratch())
                    continue;

                // Bundle selected?
                if (stBundle_set && !stBundle_set->count(sg))
                    continue;
                
                TRACE_MSG("calc_region: stencil-bundle '" << sg->get_name() << "' w/BB " <<
                          sg->bb_begin.makeDimValStr() << " ... (end before) " <<
                          sg->bb_end.makeDimValStr());

                // For wavefront adjustments, see conceptual diagram in
                // run_solution().  In this function, one of the
                // parallelogram-shaped regions is being evaluated.  At
                // each time-step, the parallelogram may be trimmed
                // based on the BB and WF extensions outside of the rank-BB.
                    
                // Actual region boundaries must stay within [extended] BB for this bundle.
                bool ok = true;
                for (int i = 0; i < ndims; i++) {
                    if (i == step_posn) continue;
                    auto& dname = _dims->_stencil_dims.getDimName(i);
                    auto angle = wf_angles[dname];
                    idx_t dbegin = rank_bb.bb_begin[dname];
                    idx_t dend = rank_bb.bb_end[dname];

                    assert(sg->bb_begin.lookup(dname));
                    idx_t rbegin = max<idx_t>(start[i], sg->bb_begin[dname]);
                    if (rbegin < dbegin) // in left WF ext?
                        rbegin = max(rbegin, dbegin - left_wf_exts[dname] + index_t * angle);
                    region_idxs.begin[i] = rbegin;

                    assert(sg->bb_end.lookup(dname));
                    idx_t rend = min<idx_t>(stop[i], sg->bb_end[dname]);
                    if (rend > dend) // in right WF ext?
                        rend = min(rend, dend + right_wf_exts[dname] - index_t * angle);
                    region_idxs.end[i] = rend;

                    if (rend <= rbegin)
                        ok = false;
                }
                TRACE_MSG("calc_region, after trimming for step " << start_t << ": " <<
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
            
                // Mark grids that [may] have been written to by this bundle,
                // updated at next step (+/- 1).
                // Mark grids as dirty even if not actually written by this
                // rank. This is needed because neighbors will not know what
                // grids are actually dirty, and all ranks must have the same
                // information about which grids are possibly dirty.
                // TODO: make this smarter.
                mark_grids_dirty(start_t + step_t, stop_t + step_t, *sg);

                    // Shift spatial region boundaries for next iteration to
                // implement temporal wavefront.  Between regions, we only shift
                // backward, so region loops must strictly increment. They may do
                // so in any order.  TODO: shift only what is needed by
                // this bundle, not the global max.
                for (int i = 0; i < ndims; i++) {
                    if (i == step_posn) continue;
                    auto& dname = _dims->_stencil_dims.getDimName(i);
                    auto angle = wf_angles[dname];

                    start[i] -= angle;
                    stop[i] -= angle;
                }

            } // stencil bundles.
        } // time.
    } // calc_region.

    // Reset the auto-tuner.
    void StencilContext::AT::clear(bool mark_done, bool verbose) {

        // Output.
        ostream& os = _context->get_ostr();
#ifdef TRACE
        this->verbose = true;
#else
        this->verbose = verbose;
#endif

        // Null stream to throw away debug info.
        yask_output_factory yof;
        nullop = yof.new_null_output();

        // Apply the best known settings from existing data, if any.
        auto _opts = _context->_opts;
        if (best_rate > 0.) {
            _opts->_block_sizes = best_block;
            apply();
            os << "auto-tuner: applying block-size "  <<
                best_block.makeDimValStr(" * ") << endl;
        }
        
        // Reset all vars.
        results.clear();
        n2big = n2small = 0;
        best_block = _opts->_block_sizes;
        best_rate = 0.;
        center_block = best_block;
        radius = max_radius;
        done = mark_done;
        neigh_idx = 0;
        better_neigh_found = false;
        ctime = 0.;
        csteps = 0;
        in_warmup = true;

        // Set min blocks to number of region threads.
        min_blks = _context->set_region_threads();
        
        // Adjust starting block if needed.
        for (auto dim : center_block.getDims()) {
            auto& dname = dim.getName();
            auto& dval = dim.getVal();
                    
            auto dmax = max(idx_t(1), _opts->_region_sizes[dname] / 2);
            if (dval > dmax || dval < 1)
                center_block[dname] = dmax;
        }
        if (!done) {
            os << "auto-tuner: starting block-size: "  <<
                center_block.makeDimValStr(" * ") << endl;
            os << "auto-tuner: starting search radius: " << radius << endl;
        }
    } // clear.

    // Evaluate the previous run and take next auto-tuner step.
    void StencilContext::AT::eval(idx_t steps, double etime) {
        ostream& os = _context->get_ostr();
        
        // Leave if done.
        if (done)
            return;

        // Setup not done?
        if (!nullop)
            return;

        // Handy ptrs.
        auto _opts = _context->_opts;
        auto _mpiInfo = _context->_mpiInfo;
        auto _dims = _context->_dims;

        // Cumulative stats.
        csteps += steps;
        ctime += etime;

        // Still in warmup?
        if (in_warmup) {

            // Warmup not done?
            if (ctime < warmup_secs && csteps < warmup_steps)
                return;

            // Done.
            os << "auto-tuner: in warmup for " << ctime << " secs" << endl;
            in_warmup = false;

            // Measure this step only.
            csteps = steps;
            ctime = etime;
        }
            
        // Need more steps to get a good measurement?
        if (ctime < min_secs && csteps < min_steps)
            return;

        // Calc perf and reset vars for next time.
        double rate = double(csteps) / ctime;
        os << "auto-tuner: " << csteps << " steps(s) at " << rate <<
            " steps/sec with block-size " <<
            _opts->_block_sizes.makeDimValStr(" * ") << endl;
        csteps = 0;
        ctime = 0.;

        // Save result.
        results[_opts->_block_sizes] = rate;
        bool is_better = rate > best_rate;
        if (is_better) {
            best_block = _opts->_block_sizes;
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
            if (neigh_idx < _mpiInfo->neighborhood_size) {

                // Convert index to offsets in each dim.
                auto ofs = _mpiInfo->neighborhood_sizes.unlayout(neigh_idx);

                // Next neighbor of center point.
                neigh_idx++;
                
                // Determine new block size.
                IdxTuple bsize(center_block);
                bool ok = true;
                for (auto odim : ofs.getDims()) {
                    auto& dname = odim.getName(); // a domain-dim name.
                    auto& dofs = odim.getVal(); // always [0..2].

                    // Min and max sizes of this dim.
                    auto dmin = _dims->_cluster_pts[dname];
                    auto dmax = _opts->_region_sizes[dname];
                            
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
                TRACE_MSG2("auto-tuner: checking block-size "  <<
                          bsize.makeDimValStr(" * "));

                // Too small?
                if (ok && bsize.product() < min_pts) {
                    n2small++;
                    ok = false;
                }

                // Too few?
                else if (ok) {
                    idx_t nblks = _opts->_region_sizes.product() / bsize.product();
                    if (nblks < min_blks) {
                        ok = false;
                        n2big++;
                    }
                }
            
                // Valid size and not already checked?
                if (ok && !results.count(bsize)) {

                    // Run next step with this size.
                    _opts->_block_sizes = bsize;
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
                        os << "auto-tuner: done" << endl;
                        return;
                    }
                    os << "auto-tuner: new search radius: " << radius << endl;
                }
                else {
                    TRACE_MSG2("auto-tuner: continuing search from block " <<
                               center_block.makeDimValStr(" * "));
                }
            } // beyond next neighbor of center.
        } // search for new setting to try.

        // Fix settings for next step.
        apply();
        TRACE_MSG2("auto-tuner: next block-size "  <<
                  _opts->_block_sizes.makeDimValStr(" * "));
    } // eval.

    // Apply auto-tuner settings.
    void StencilContext::AT::apply() {
        auto _opts = _context->_opts;
        auto _env = _context->_env;

        // Change block-related sizes to 0 so adjustSettings()
        // will set them to the default.
        // TODO: tune sub-block sizes also.
        _opts->_sub_block_sizes.setValsSame(0);
        _opts->_sub_block_group_sizes.setValsSame(0);
        _opts->_block_group_sizes.setValsSame(0);
                
        // Make sure everything is resized based on block size.
        _opts->adjustSettings(nullop->get_ostream(), _env);

        // Reallocate scratch data based on new block size.
        _context->allocScratchData(nullop->get_ostream());
    }
    
    // Apply auto-tuning to some of the settings.
    void StencilContext::run_auto_tuner_now(bool verbose) {
        if (!rank_bb.bb_valid)
            THROW_YASK_EXCEPTION("Error: tune_settings() called without calling prepare_solution() first");

        ostream& os = get_ostr();
        os << "Auto-tuning...\n" << flush;
        YaskTimer at_timer;
        at_timer.start();

        // Temporarily disable halo exchange to tune intra-rank.
        enable_halo_exchange = false;
        
        // Init tuner.
        _at.clear(false, verbose);

        // Reset stats.
        clear_timers();

        // Determine number of sets to run.
        // If wave-fronts are enabled, run a max number of these steps.
        // TODO: only run one region during AT.
        idx_t region_steps = _opts->_region_sizes[_dims->_step_dim];
        idx_t step_t = min(region_steps, _at.max_step_t);
        
        // Run time-steps until AT converges.
        bool done = false;
        for (idx_t t = 0; !done; t += step_t) {

            // Run step_t time-step(s).
            run_solution(t, t + step_t - 1);

            // AT done on this rank?
            done = _at.is_done();
        }

        // Wait for all ranks to finish.
        _env->global_barrier();

        // reenable halo exchange.
        enable_halo_exchange = true;
        
        // Report results.
        at_timer.stop();
        os << "Auto-tuner done after " << steps_done << " step(s) in " <<
            at_timer.get_elapsed_secs() << " secs.\n";
        os << "best-block-size: " << _opts->_block_sizes.makeDimValStr(" * ") << endl << flush;
        os << "best-sub-block-size: " << _opts->_sub_block_sizes.makeDimValStr(" * ") << endl << flush;

        // Reset stats.
        clear_timers();
    }
    
    // Add a new grid to the containers.
    void StencilContext::addGrid(YkGridPtr gp, bool is_output) {
        assert(gp);
        auto& gname = gp->get_name();
        if (gridMap.count(gname))
            THROW_YASK_EXCEPTION("Error: grid '" << gname << "' already exists");

        // Add to list and map.
        gridPtrs.push_back(gp);
        gridMap[gname] = gp;

        // Add to output list and map if 'is_output'.
        if (is_output) {
            outputGridPtrs.push_back(gp);
            outputGridMap[gname] = gp;
        }
    }

    // Adjust offsets of scratch grids based on thread number 'thread_idx'
    // and beginning point of block 'idxs'.  Each scratch-grid is assigned
    // to a thread, so it must "move around" as the thread is assigned to
    // each block.  This move is accomplished by changing the grids' global
    // and local offsets.
    void StencilContext::update_scratch_grids(int thread_idx,
                                              const Indices& idxs) {
        auto dims = get_dims();
        int nsdims = dims->_stencil_dims.size();
        auto step_posn = Indices::step_posn;

        // Loop thru vecs of scratch grids.
        for (auto* sv : scratchVecs) {
            assert(sv);

            // Get ptr to the scratch grid for this thread.
            auto gp = sv->at(thread_idx);
            assert(gp);
            assert(gp->is_scratch());

            // i: index for stencil dims, j: index for domain dims.
            for (int i = 0, j = 0; i < nsdims; i++) {
                if (i != step_posn) {
                    auto& dim = dims->_stencil_dims.getDim(i);
                    auto& dname = dim.getName();

                    // Is this dim used in this grid?
                    int posn = gp->get_dim_posn(dname);
                    if (posn >= 0) {

                        // |        +------+       |
                        // |  loc   |      |       | 
                        // |  ofs   |      |       | 
                        // |<------>|      |       | 
                        // |        +------+       |
                        // ^        ^
                        // |        |
                        // |        start of grid-domain/0-idx of block
                        // first rank-domain index
                        
                        // Set offset of grid based on starting point of block.
                        // This is a global index, so it will include the rank offset.
                        gp->_set_offset(posn, idxs[i]);

                        // Local offset is the offset of this grid
                        // relative to the current rank.
                        // Set local offset to diff between global offset
                        // and rank offset.
                        auto rofs = rank_domain_offsets[j];
                        auto lofs = idxs[i] - rofs;
                        gp->_set_local_offset(posn, lofs);

                        // For a vectorized grid, the local offset must
                        // be a vector multiple. This is necessary for
                        // vector and cluster operations to work properly.
                        assert(imod_flr(lofs, gp->_get_vec_lens(posn)) == 0);
                    }
                    j++;
                }
            }
        }
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
                "elapsed-time (sec):                     " << makeNumStr(rtime) << endl;
#ifdef USE_MPI
            os <<
                "time in halo exch (sec):                " << makeNumStr(mtime);
            float pct = 100. * mtime / rtime;
            os << " (" << pct << "%)" << endl;
#endif
            os <<
                "throughput (num-writes/sec):            " << makeNumStr(writes_ps) << endl <<
                "throughput (est-FLOPS):                 " << makeNumStr(flops) << endl <<
                "throughput (num-points/sec):            " << makeNumStr(domain_pts_ps) << endl;
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

    // Exchange dirty halo data for all grids and all steps, regardless
    // of their stencil-bundle.
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
        
        exchange_halos(start, stop, 0);
#endif
    }
    
    // Exchange halo data needed by stencil-bundle 'sg' at the given time.
    // If sg==null, check all bundles.
    // Data is needed for input grids that have not already been updated.
    // [BIG] TODO: overlap halo exchange with computation.
    void StencilContext::exchange_halos(idx_t start, idx_t stop, StencilBundleBase* sgp)
    {
#ifdef USE_MPI
        if (!enable_halo_exchange || _env->num_ranks < 2)
            return;

        mpi_time.start();
        TRACE_MSG("exchange_halos: " << start << " ... (end before) " << stop);
        auto opts = get_settings();
        auto& sd = _dims->_step_dim;

        // Loop through steps.  This loop has to be outside halo-step loop
        // because we only have one buffer per step. Normally, we only
        // exchange one step; in that case, it doesn't matter. It would be more
        // efficient to allow packing and unpacking multiple steps, esp. with WFs.
        // TODO: this will need to be addressed if/when comm/compute overlap is added.
        assert(start != stop);
        idx_t step = (start < stop) ? 1 : -1;
        for (idx_t t = start; t != stop; t += step) {

            // Get list of grids that need to be swapped.
            // Use an ordered map to make sure grids are in
            // same order on all ranks.
            GridPtrMap gridsToSwap;

            // Loop thru all stencil bundles.
            for (auto* sg : stBundles) {

                // Don't exchange for scratch groups.
                if (sg->is_scratch())
                    return;

                // Bundle selected?
                if (sgp && sgp != sg)
                    continue;
                
                // Find the bundles that need to be processed.
                // This will be the prerequisite scratch-grid
                // bundles plus this non-scratch bundle.
                // We need to loop thru the scratch-grid
                // bundles so we can consider the inputs
                // to them for exchanges.
                auto sg_list = sg->get_scratch_deps();
                sg_list.push_back(sg);

                // Loop through all the needed bundles.
                for (auto* csg : sg_list) {

                    TRACE_MSG("exchange_halos: checking " << csg->inputGridPtrs.size() <<
                              " input grid(s) to bundle '" << csg->get_name() <<
                              "' that is needed for bundle '" << sg->get_name() << "'");

                    // Loop thru all *input* grids in this bundle.
                    for (auto gp : csg->inputGridPtrs) {

                        // Don't swap scratch grids.
                        if (gp->is_scratch())
                            continue;
                    
                        // Only need to swap grids whose halos are not up-to-date
                        // for this step.
                        if (!gp->is_dirty(t))
                            continue;

                        // Only need to swap grids that have any MPI buffers.
                        auto& gname = gp->get_name();
                        if (mpiData.count(gname) == 0)
                            continue;

                        // Swap this grid.
                        gridsToSwap[gname] = gp;
                    }
                } // needed bundles.
            } // all bundles.
            TRACE_MSG("exchange_halos: need to exchange halos for " <<
                      gridsToSwap.size() << " grid(s)");

            // 1D array to store send request handles.
            // We use a 1D array so we can call MPI_Waitall().
            MPI_Request send_reqs[gridsToSwap.size() * _mpiInfo->neighborhood_size];
            
            // 2D array for receive request handles.
            // We use a 2D array to simplify individual indexing.
            MPI_Request recv_reqs[gridsToSwap.size()][_mpiInfo->neighborhood_size];

            // Sequence of things to do for each grid's neighbors
            // (isend includes packing).
            int num_send_reqs = 0;
            int num_recv_reqs = 0;
            enum halo_steps { halo_irecv, halo_pack_isend, halo_unpack, halo_nsteps };
            for (int halo_step = 0; halo_step < halo_nsteps; halo_step++) {

                if (halo_step == halo_irecv)
                    TRACE_MSG("exchange_halos: requesting data for step " << t << "...");
                else if (halo_step == halo_pack_isend)
                    TRACE_MSG("exchange_halos: packing and sending data for step " << t << "...");
                else if (halo_step == halo_unpack)
                    TRACE_MSG("exchange_halos: unpacking data for step " << t << "...");

                // Loop thru all grids to swap.
                // Use 'gi' as a unique MPI index.
                int gi = -1;
                for (auto gtsi : gridsToSwap) {
                    auto& gname = gtsi.first;
                    auto gp = gtsi.second;
                    gi++;
                    MPI_Request* grid_recv_reqs = recv_reqs[gi];
                    TRACE_MSG(" for grid #" << gi << ", '" << gname << "'...");

                    // Visit all this rank's neighbors.
                    auto& grid_mpi_data = mpiData.at(gname);
                    grid_mpi_data.visitNeighbors
                        ([&](const IdxTuple& offsets, // NeighborOffset.
                             int neighbor_rank,
                             int ni, // unique neighbor index.
                             MPIBufs& bufs) {
                            auto& sendBuf = bufs.bufs[MPIBufs::bufSend];
                            auto& recvBuf = bufs.bufs[MPIBufs::bufRecv];
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
                            if (halo_step == halo_irecv) {
                                auto nbytes = recvBuf.get_bytes();
                                if (nbytes) {
                                    void* buf = (void*)recvBuf._elems;
                                    TRACE_MSG("   requesting " << makeByteStr(nbytes) << "...");
                                    MPI_Irecv(buf, nbytes, MPI_BYTE,
                                              neighbor_rank, int(gi), _env->comm, &grid_recv_reqs[ni]);
                                    num_recv_reqs++;
                                }
                                else
                                    TRACE_MSG("   0B to request");
                            }

                            // Pack data into send buffer, then send to neighbor.
                            else if (halo_step == halo_pack_isend) {
                                auto nbytes = sendBuf.get_bytes();
                                if (nbytes) {

                                    // Vec ok?
                                    // Domain sizes must be ok, and buffer size must be ok
                                    // as calculated when buffers were created.
                                    bool send_vec_ok = vec_ok && sendBuf.vec_copy_ok;

                                    // Get first and last ranges.
                                    IdxTuple first = sendBuf.begin_pt;
                                    IdxTuple last = sendBuf.last_pt;

                                    // The code in allocMpiData() pre-calculated the first and
                                    // last points of each buffer, except in the step dim.
                                    // So, we need to set that value now.
                                    // TODO: update this if we expand the buffers to hold
                                    // more than one step.
                                    if (gp->is_dim_used(sd)) {
                                        first.setVal(sd, t);
                                        last.setVal(sd, t);
                                    }
                                    TRACE_MSG("   packing " << sendBuf.num_pts.makeDimValStr(" * ") <<
                                              " points from " << first.makeDimValStr() <<
                                              " ... " << last.makeDimValStr() <<
                                              (send_vec_ok ? " with" : " without") <<
                                              " vector copy...");

                                    // Copy (pack) data from grid to buffer.
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
                                else
                                    TRACE_MSG("   0B to send");
                            }

                            // Wait for data from neighbor, then unpack it.
                            else if (halo_step == halo_unpack) {
                                auto nbytes = recvBuf.get_bytes();
                                if (nbytes) {

                                    // Wait for data from neighbor before unpacking it.
                                    TRACE_MSG("   waiting for " << makeByteStr(nbytes) << "...");
                                    MPI_Wait(&grid_recv_reqs[ni], MPI_STATUS_IGNORE);

                                    // Vec ok?
                                    bool recv_vec_ok = vec_ok && recvBuf.vec_copy_ok;

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
                                              " ... " << last.makeDimValStr() <<
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
                                else
                                    TRACE_MSG("   0B to wait for");
                            }
                        }); // visit neighbors.

                } // grids.
            } // exchange sequence.
            
            // Mark grids as up-to-date.
            for (auto gtsi : gridsToSwap) {
                auto& gname = gtsi.first;
                auto gp = gtsi.second;
                if (gp->is_dirty(t)) {
                    gp->set_dirty(false, t);
                    TRACE_MSG("grid '" << gname <<
                              "' marked as clean at step " << t);
                }
            }

            // Wait for all send requests to complete.
            TRACE_MSG("exchange_halos: " << num_recv_reqs <<
                      " MPI receive request(s) completed");
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

    // Mark grids that have been written to by stencil-bundle 'sg'.
    // TODO: only mark grids that are written to in their halo-read area.
    // TODO: add index for misc dim(s).
    void StencilContext::mark_grids_dirty(idx_t start, idx_t stop, StencilBundleBase& sg) {
        idx_t step = (start < stop) ? 1 : -1;
        for (auto gp : sg.outputGridPtrs) {
            for (idx_t t = start; t != stop; t += step) {
                gp->set_dirty(true, t);
                TRACE_MSG("grid '" << gp->get_name() << "' marked as dirty at step " << t);
            }
        }
    }

} // namespace yask.
