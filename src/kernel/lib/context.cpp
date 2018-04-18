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
        ScanIndices rank_idxs(*_dims, false);
        rank_idxs.begin = begin;
        rank_idxs.end = end;

        // Set offsets in scratch grids.
        // Requires scratch grids to be allocated for whole
        // rank instead of smaller grid size.
        update_scratch_grids(scratch_grid_idx, rank_idxs);
            
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
            // For this reference-code implementation, we
            // will do all stencil bundles at this level,
            // even scratch-grid ones.
            for (auto* sg : stBundles) {

                // Exchange all dirty halos.
                exchange_halos_all();

                // Indices needed for the generated misc loops.  Will normally be a
                // copy of rank_idxs except when updating scratch-grids.
                ScanIndices misc_idxs = sg->adjust_scan(scratch_grid_idx, rank_idxs);
                misc_idxs.step.setFromConst(1); // ensure unit step.
                
                // Define misc-loop function.  Since step is always 1, we
                // ignore misc_stop.  If point is in sub-domain for this
                // bundle, then evaluate the reference scalar code.
#define misc_fn(misc_idxs)   do {                                       \
                    if (sg->is_in_valid_domain(misc_idxs.start))        \
                        sg->calc_scalar(scratch_grid_idx, misc_idxs.start);   \
                } while(0)
                
                // Scan through n-D space.
                TRACE_MSG("calc_rank_ref: step " << start_t <<
                          " in bundle '" << sg->get_name() << "': " <<
                          misc_idxs.begin.makeValStr(ndims) <<
                          " ... (end before) " << misc_idxs.end.makeValStr(ndims));
#include "yask_misc_loops.hpp"
#undef misc_fn
                
                // Remember grids that have been written to by this bundle,
                // updated at next step (+/- 1).
                mark_grids_dirty(start_t + step_t, stop_t + step_t, *sg);
                
            } // bundles.
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
        ScanIndices rank_idxs(*_dims, true);
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
                    exchange_halos(start_t, stop_t, *sg);

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
            // calc_region().
            // TODO: make this the only case, allowing all bundles to be done
            // between MPI exchanges, even w/o wave-fronts.
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
        ScanIndices region_idxs(*_dims, true);
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

                // Bundle not selected.
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

                    // Remember grids that have been written to by this bundle,
                    // updated at next step (+/- 1).
                    mark_grids_dirty(start_t + step_t, stop_t + step_t, *sg);
                }
            
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
    
    // Init MPI-related vars and other vars related to my rank's place in
    // the global problem: rank index, offset, etc.  Need to call this even
    // if not using MPI to properly init these vars.  Called from
    // prepare_solution(), so it doesn't normally need to be called from user code.
    void StencilContext::setupRank() {
        ostream& os = get_ostr();
        auto& step_dim = _dims->_step_dim;
        auto me = _env->my_rank;

        // Check ranks.
        idx_t req_ranks = _opts->_num_ranks.product();
        if (req_ranks != _env->num_ranks) {
            THROW_YASK_EXCEPTION("error: " << req_ranks << " rank(s) requested (" <<
                _opts->_num_ranks.makeDimValStr(" * ") << "), but " <<
                _env->num_ranks << " rank(s) are active");
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
                if (mandist != 0)
                    THROW_YASK_EXCEPTION("Internal error: distance to own rank == " << mandist);
            }

            // Someone else.
            else {
                if (mandist == 0)
                    THROW_YASK_EXCEPTION("Error: ranks " << me <<
                                         " and " << rn << " at same coordinates");
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
                                THROW_YASK_EXCEPTION("Error: rank " << rn << " and " << me <<
                                    " are both at rank-index " << coords[me][di] <<
                                    " in the '" << dname <<
                                    "' dimension , but their rank-domain sizes are " <<
                                    rnsz << " and " << mysz <<
                                    " (resp.) in the '" << dj <<
                                    "' dimension, making them unaligned");
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
                }

                // Save vec-mult flag.
                _mpiInfo->has_all_vlen_mults.at(rn_ofs) = vlen_mults;
                
            } // self or immediate neighbor in any direction.
            
        } // ranks.

        // Set offsets in grids and find WF extensions
        // based on the grids' halos.
        update_grids();

        // Determine bounding-boxes for all bundles.
        // This must be done after finding WF extensions.
        find_bounding_boxes();

    } // setupRank.

    // Alloc 'nbytes' on each requested NUMA node.
    // Map keys are preferred NUMA nodes or -1 for local.
    // Pointers are returned in '_data_buf'.
    // 'ngrids' and 'type' are only used for debug msg.
    void StencilContext::_alloc_data(const map <int, size_t>& nbytes,
                                     const map <int, size_t>& ngrids,
                                     map <int, shared_ptr<char>>& data_buf,
                                     const std::string& type) {
        ostream& os = get_ostr();

        for (const auto& i : nbytes) {
            int numa_pref = i.first;
            size_t nb = i.second;
            size_t ng = ngrids.at(numa_pref);

            // Don't need pad after last one.
            if (nb >= _data_buf_pad)
                nb -= _data_buf_pad;

            // Allocate data.
            os << "Allocating " << makeByteStr(nb) <<
                " for " << ng << " " << type << "(s)";
#ifdef USE_NUMA
            if (numa_pref >= 0)
                os << " preferring NUMA node " << numa_pref;
            else
                os << " using NUMA policy " << numa_pref;
#endif
            os << "...\n" << flush;
            auto p = shared_numa_alloc<char>(nb, numa_pref);
            TRACE_MSG("Got memory at " << static_cast<void*>(p.get()));

            // Save using original key.
            data_buf[numa_pref] = p;
        }
    }
    
    // Allocate memory for grids that do not already have storage.
    void StencilContext::allocGridData(ostream& os) {

        // Base ptrs for all default-alloc'd data.
        // These pointers will be shared by the ones in the grid
        // objects, which will take over ownership when these go
        // out of scope.
        // Key is preferred numa node or -1 for local.
        map <int, shared_ptr<char>> _grid_data_buf;

        // Pass 0: count required size for each NUMA node, allocate chunk of memory at end.
        // Pass 1: distribute parts of already-allocated memory chunk.
        for (int pass = 0; pass < 2; pass++) {
            TRACE_MSG("allocGridData pass " << pass << " for " <<
                      gridPtrs.size() << " grid(s)");
        
            // Count bytes needed and number of grids for each NUMA node.
            map <int, size_t> npbytes, ngrids;
        
            // Grids.
            for (auto gp : gridPtrs) {
                if (!gp)
                    continue;
                auto& gname = gp->get_name();

                // Grid data.
                // Don't alloc if already done.
                if (!gp->is_storage_allocated()) {
                    int numa_pref = gp->get_numa_preferred();

                    // Set storage if buffer has been allocated in pass 0.
                    if (pass == 1) {
                        auto p = _grid_data_buf[numa_pref];
                        assert(p);
                        gp->set_storage(p, npbytes[numa_pref]);
                        os << gp->make_info_string() << endl;
                    }

                    // Determine padded size (also offset to next location).
                    size_t nbytes = gp->get_num_storage_bytes();
                    npbytes[numa_pref] += ROUND_UP(nbytes + _data_buf_pad,
                                                  CACHELINE_BYTES);
                    ngrids[numa_pref]++;
                    if (pass == 0)
                        TRACE_MSG(" grid '" << gname << "' needs " << makeByteStr(nbytes) <<
                                  " on NUMA node " << numa_pref);
                }
            }

            // Alloc for each node.
            if (pass == 0)
                _alloc_data(npbytes, ngrids, _grid_data_buf, "grid");

        } // grid passes.
    };
    
    // Create MPI and allocate buffers.
    void StencilContext::allocMpiData(ostream& os) {

        // Remove any old MPI data.
        freeMpiData(os);

#ifdef USE_MPI

        int num_exchanges = 0;
        auto me = _env->my_rank;
        
        // Need to determine the size and shape of all MPI buffers.
        // Visit all neighbors of this rank.
        _mpiInfo->visitNeighbors
            ([&](const IdxTuple& neigh_offsets, int neigh_rank, int neigh_idx) {
                if (neigh_rank == MPI_PROC_NULL)
                    return; // from lambda fn.

                // Determine max dist needed.  TODO: determine max dist
                // automatically from stencils; may not be same for all
                // grids.
#ifndef MAX_EXCH_DIST
#define MAX_EXCH_DIST (NUM_STENCIL_DIMS - 1)
#endif
                // Always use max dist with WF.
                // TODO: determine if this is overkill.
                int maxdist = MAX_EXCH_DIST;
                if (num_wf_shifts > 0)
                    maxdist = NUM_STENCIL_DIMS - 1;

                // Manhattan dist.
                int mandist = _mpiInfo->man_dists.at(neigh_idx);
                    
                // Check distance.
                // TODO: calculate and use exch dist for each grid.
                if (mandist > maxdist) {
                    TRACE_MSG("no halo exchange needed with rank " << neigh_rank <<
                              " because L1-norm = " << mandist);
                    return;     // from lambda fn.
                }
        
                // Determine size of MPI buffers between neigh_rank and my rank
                // for each grid and create those that are needed.
                for (auto gp : gridPtrs) {
                    if (!gp)
                        continue;
                    auto& gname = gp->get_name();

                    // Lookup first & last domain indices and calc exchange sizes
                    // for this grid.
                    bool found_delta = false;
                    IdxTuple my_halo_sizes, neigh_halo_sizes;
                    IdxTuple first_inner_idx, last_inner_idx;
                    IdxTuple first_outer_idx, last_outer_idx;
                    for (auto& dim : _dims->_domain_dims.getDims()) {
                        auto& dname = dim.getName();
                        if (gp->is_dim_used(dname)) {

                            // Get domain indices for this grid.
                            // If there are no more ranks in the given direction, extend
                            // the index into the outer halo to make sure all data are sync'd.
                            // This is critical for WFs.
                            idx_t fidx = gp->get_first_rank_domain_index(dname);
                            idx_t lidx = gp->get_last_rank_domain_index(dname);
                            first_inner_idx.addDimBack(dname, fidx);
                            last_inner_idx.addDimBack(dname, lidx);
                            if (_opts->is_first_rank(dname))
                                fidx -= gp->get_left_halo_size(dname);
                            if (_opts->is_last_rank(dname))
                                lidx += gp->get_right_halo_size(dname);
                            first_outer_idx.addDimBack(dname, fidx);
                            last_outer_idx.addDimBack(dname, lidx);

                            // Determine size of exchange. This will be the actual halo size
                            // plus any wave-front extensions. In the current implementation,
                            // we need the wave-front extensions regardless of whether there
                            // is a halo on a given grid. This is because each stencil-bundle
                            // gets shifted by the WF angles at each step in the WF.

                            // Neighbor is to the left.
                            if (neigh_offsets[dname] == MPIInfo::rank_prev) {
                                auto ext = left_wf_exts[dname];

                                // my halo.
                                auto halo_size = gp->get_left_halo_size(dname);
                                halo_size += ext;
                                my_halo_sizes.addDimBack(dname, halo_size);

                                // neighbor halo.
                                halo_size = gp->get_right_halo_size(dname); // their right is on my left.
                                halo_size += ext;
                                neigh_halo_sizes.addDimBack(dname, halo_size);
                            }

                            // Neighbor is to the right.
                            else if (neigh_offsets[dname] == MPIInfo::rank_next) {
                                auto ext = right_wf_exts[dname];

                                // my halo.
                                auto halo_size = gp->get_right_halo_size(dname);
                                halo_size += ext;
                                my_halo_sizes.addDimBack(dname, halo_size);

                                // neighbor halo.
                                halo_size = gp->get_left_halo_size(dname); // their left is on my right.
                                halo_size += ext;
                                neigh_halo_sizes.addDimBack(dname, halo_size);
                            }

                            // Neighbor in-line.
                            else {
                                my_halo_sizes.addDimBack(dname, 0);
                                neigh_halo_sizes.addDimBack(dname, 0);
                            }

                            // Vectorized exchange allowed based on domain sizes?
                            // Both my rank and neighbor rank must have all domain sizes
                            // of vector multiples.
                            bool vec_ok = allow_vec_exchange &&
                                _mpiInfo->has_all_vlen_mults[_mpiInfo->my_neighbor_index] &&
                                _mpiInfo->has_all_vlen_mults[neigh_idx];
                            
                            // Round up halo sizes if vectorized exchanges allowed.
                            // TODO: add a heuristic to avoid increasing by a large factor.
                            if (vec_ok) {
                                auto vec_size = _dims->_fold_pts[dname];
                                my_halo_sizes.setVal(dname, ROUND_UP(my_halo_sizes[dname], vec_size));
                                neigh_halo_sizes.setVal(dname, ROUND_UP(neigh_halo_sizes[dname], vec_size));
                            }
                            
                            // Is this neighbor before or after me in this domain direction?
                            if (neigh_offsets[dname] != MPIInfo::rank_self)
                                found_delta = true;
                        }
                    }

                    // Is buffer needed?
                    // Example: if this grid is 2D in y-z, but only neighbors are in
                    // x-dim, we don't need any exchange.
                    if (!found_delta) {
                        TRACE_MSG("no halo exchange needed for grid '" << gname <<
                                  "' with rank " << neigh_rank <<
                                  " because the neighbor is not in a direction"
                                  " corresponding to a grid dim");
                        continue; // to next grid.
                    }

                    // Make a buffer in both directions (send & receive).
                    for (int bd = 0; bd < MPIBufs::nBufDirs; bd++) {

                        // Begin/end vars to indicate what part
                        // of main grid to read from or write to based on
                        // the current neighbor being processed.
                        IdxTuple copy_begin = gp->get_allocs();
                        IdxTuple copy_end = gp->get_allocs();

                        // Adjust along domain dims in this grid.
                        for (auto& dim : _dims->_domain_dims.getDims()) {
                            auto& dname = dim.getName();

                            // Init range to whole rank domain (including
                            // outer halos).  These may be changed below
                            // depending on the neighbor's direction.
                            copy_begin[dname] = first_outer_idx[dname];
                            copy_end[dname] = last_outer_idx[dname] + 1; // end = last + 1.

                            // Neighbor direction in this dim.
                            auto neigh_ofs = neigh_offsets[dname];
                            
                            // Region to read from, i.e., data from inside
                            // this rank's domain to be put into neighbor's
                            // halo.
                            if (bd == MPIBufs::bufSend) {

                                // Neighbor is to the left.
                                if (neigh_ofs == idx_t(MPIInfo::rank_prev)) {

                                    // Only read slice as wide as halo from beginning.
                                    copy_end[dname] = first_inner_idx[dname] + neigh_halo_sizes[dname];
                                }
                            
                                // Neighbor is to the right.
                                else if (neigh_ofs == idx_t(MPIInfo::rank_next)) {
                                    
                                    // Only read slice as wide as halo before end.
                                    copy_begin[dname] = last_inner_idx[dname] + 1 - neigh_halo_sizes[dname];
                                }
                            
                                // Else, this neighbor is in same posn as I am in this dim,
                                // so we leave the default begin/end settings.
                            }
                        
                            // Region to write to, i.e., into this rank's halo.
                            else if (bd == MPIBufs::bufRecv) {

                                // Neighbor is to the left.
                                if (neigh_ofs == idx_t(MPIInfo::rank_prev)) {

                                    // Only read slice as wide as halo before beginning.
                                    copy_begin[dname] = first_inner_idx[dname] - my_halo_sizes[dname];
                                    copy_end[dname] = first_inner_idx[dname];
                                }
                            
                                // Neighbor is to the right.
                                else if (neigh_ofs == idx_t(MPIInfo::rank_next)) {
                                    
                                    // Only read slice as wide as halo after end.
                                    copy_begin[dname] = last_inner_idx[dname] + 1;
                                    copy_end[dname] = last_inner_idx[dname] + 1 + my_halo_sizes[dname];
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
                            if (_dims->_domain_dims.lookup(dname)) {
                                dsize = copy_end[dname] - copy_begin[dname];

                                // Check whether size is multiple of vlen.
                                auto vlen = _dims->_fold_pts[dname];
                                if (dsize % vlen != 0)
                                    vlen_mults = false;
                            }

                            // step dim?
                            // Allowing only one step to be exchanged.
                            // TODO: consider exchanging mutiple steps at once for WFs.
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
                                      "' with rank " << neigh_rank <<
                                      " because there is no data to exchange");
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
                            oss << "_send_halo_from_" << me << "_to_" << neigh_rank;
                        else if (bd == MPIBufs::bufRecv)
                            oss << "_recv_halo_from_" << neigh_rank << "_to_" << me;
                        string bufname = oss.str();

                        // Make MPI data entry for this grid.
                        auto gbp = mpiData.emplace(gname, _mpiInfo);
                        auto& gbi = gbp.first; // iterator from pair returned by emplace().
                        auto& gbv = gbi->second; // value from iterator.
                        auto& buf = gbv.getBuf(MPIBufs::BufDir(bd), neigh_offsets);

                        // Config buffer for this grid.
                        // (But don't allocate storage yet.)
                        buf.begin_pt = copy_begin;
                        buf.last_pt = copy_last;
                        buf.num_pts = buf_sizes;
                        buf.name = bufname;
                        buf.has_all_vlen_mults = vlen_mults;
                        
                        TRACE_MSG("configured MPI buffer object '" << buf.name <<
                                  "' for rank at relative offsets " <<
                                  neigh_offsets.subElements(1).makeDimValStr() << " with " <<
                                  buf.num_pts.makeDimValStr(" * ") << " = " << buf.get_size() <<
                                  " element(s) at " << buf.begin_pt.makeDimValStr() <<
                                  " ... " << buf.last_pt.makeDimValStr());
                        num_exchanges++;

                    } // send, recv.
                } // grids.
            });   // neighbors.
        TRACE_MSG("number of halo-exchanges needed on this rank: " << num_exchanges);

        // Base ptrs for all alloc'd data.
        // These pointers will be shared by the ones in the grid
        // objects, which will take over ownership when these go
        // out of scope.
        map <int, shared_ptr<char>> _mpi_data_buf;

        // Allocate MPI buffers.
        // Pass 0: count required size, allocate chunk of memory at end.
        // Pass 1: distribute parts of already-allocated memory chunk.
        for (int pass = 0; pass < 2; pass++) {
            TRACE_MSG("allocMpiData pass " << pass << " for " <<
                      mpiData.size() << " MPI buffer set(s)");
        
            // Count bytes needed and number of buffers for each NUMA node.
            map <int, size_t> npbytes, nbufs;
        
            // Grids.
            for (auto gp : gridPtrs) {
                if (!gp)
                    continue;
                auto& gname = gp->get_name();
                int numa_pref = gp->get_numa_preferred();

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
                                
                                // Set storage if buffer has been allocated in pass 0.
                                if (pass == 1) {
                                    auto p = _mpi_data_buf[numa_pref];
                                    assert(p);
                                    buf.set_storage(p, npbytes[numa_pref]);
                                }

                                // Determine padded size (also offset to next location).
                                auto sbytes = buf.get_bytes();
                                npbytes[numa_pref] += ROUND_UP(sbytes + _data_buf_pad,
                                                               CACHELINE_BYTES);
                                nbufs[numa_pref]++;
                                if (pass == 0)
                                    TRACE_MSG("  MPI buf '" << buf.name << "' needs " <<
                                              makeByteStr(sbytes) <<
                                              " on NUMA node " << numa_pref);
                            }
                        } );
                }
            }

            // Alloc for each node.
            if (pass == 0)
                _alloc_data(npbytes, nbufs, _mpi_data_buf, "MPI buffer");

        } // MPI passes.
#endif
    }

    // Allocate memory for scratch grids based on number of threads and
    // block sizes.
    void StencilContext::allocScratchData(ostream& os) {

        // Remove any old scratch data.
        freeScratchData(os);

        // Base ptrs for all alloc'd data.
        // This pointer will be shared by the ones in the grid
        // objects, which will take over ownership when it goes
        // out of scope.
        map <int, shared_ptr<char>> _scratch_data_buf;

        // Make sure the right number of threads are set so we
        // have the right number of scratch grids.
        int rthreads = set_region_threads();

        // Delete any existing scratch grids.
        // Create new scratch grids.
        makeScratchGrids(rthreads);
        
        // Pass 0: count required size, allocate chunk of memory at end.
        // Pass 1: distribute parts of already-allocated memory chunk.
        for (int pass = 0; pass < 2; pass++) {
            TRACE_MSG("allocScratchData pass " << pass << " for " <<
                      scratchVecs.size() << " set(s) of scratch grids");
        
            // Count bytes needed and number of grids for each NUMA node.
            map <int, size_t> npbytes, ngrids;

            // Loop through each scratch grid vector.
            for (auto* sgv : scratchVecs) {
                assert(sgv);

                // Loop through each scratch grid in this vector.
                // There will be one for each region thread.
                assert(int(sgv->size()) == rthreads);
                int thr_num = 0;
                for (auto gp : *sgv) {
                    assert(gp);
                    auto& gname = gp->get_name();
                    int numa_pref = gp->get_numa_preferred();
            
                    // Loop through each domain dim.
                    for (auto& dim : _dims->_domain_dims.getDims()) {
                        auto& dname = dim.getName();

                        if (gp->is_dim_used(dname)) {

                            // Set domain size of grid to block size.
                            gp->_set_domain_size(dname, _opts->_block_sizes[dname]);
                    
                            // Pads.
                            // Set via both 'extra' and 'min'; larger result will be used.
                            gp->set_extra_pad_size(dname, _opts->_extra_pad_sizes[dname]);
                            gp->set_min_pad_size(dname, _opts->_min_pad_sizes[dname]);
                        }
                    } // dims.
                
                    // Set storage if buffer has been allocated.
                    if (pass == 1) {
                        auto p = _scratch_data_buf[numa_pref];
                        assert(p);
                        gp->set_storage(p, npbytes[numa_pref]);
                        TRACE_MSG(gp->make_info_string());
                    }

                    // Determine size used (also offset to next location).
                    size_t nbytes = gp->get_num_storage_bytes();
                    npbytes[numa_pref] += ROUND_UP(nbytes + _data_buf_pad,
                                                   CACHELINE_BYTES);
                    ngrids[numa_pref]++;
                    if (pass == 0)
                        TRACE_MSG(" scratch grid '" << gname << "' for thread " <<
                                  thr_num << " needs " << makeByteStr(nbytes) <<
                                  " on NUMA node " << numa_pref);
                    thr_num++;
                } // scratch grids.
            } // scratch-grid vecs.

            // Alloc for each node.
            if (pass == 0)
                _alloc_data(npbytes, ngrids, _scratch_data_buf, "scratch grid");

        } // scratch-grid passes.
    }

    // Adjust offsets of scratch grids based
    // on thread and scan indices.
    // Each scratch-grid is assigned to a thread, so it must
    // "move around" as the thread is assigned to each block.
    void StencilContext::update_scratch_grids(int thread_idx,
                                              const ScanIndices& idxs) {
        auto dims = get_dims();
        int nsdims = dims->_stencil_dims.size();
        auto step_posn = Indices::step_posn;

        // Loop thru vecs of scratch grids.
        for (auto* sv : scratchVecs) {
            assert(sv);

            // Get the one for this thread.
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

                        // Set offset of grid based on starting point of block.
                        // This is global, so it will include the rank offset.
                        gp->_set_offset(posn, idxs.begin[i]);

                        // Set local offset to diff between global offset
                        // and rank offset. Must be vec-multiple.
                        auto rofs = rank_domain_offsets[j];
                        auto lofs = idxs.begin[i] - rofs;
                        gp->_set_local_offset(posn, lofs);
                        assert(imod_flr(lofs, dims->_fold_pts[j]) == 0);
                    }
                    j++;
                }
            }
        }
    }

    
    // Set non-scratch grid sizes and offsets based on settings.
    // Set wave-front settings.
    // This should be called anytime a setting or rank offset is changed.
    void StencilContext::update_grids()
    {
        assert(_opts);

        // Reset halos to zero.
        max_halos = _dims->_domain_dims;

        // Loop through each non-scratch grid.
        for (auto gp : gridPtrs) {
            assert(gp);

            // Ignore manually-sized grid.
            if (gp->is_fixed_size())
                continue;

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
                    max_halos[dname] = max(max_halos[dname], gp->get_left_halo_size(dname));
                    max_halos[dname] = max(max_halos[dname], gp->get_right_halo_size(dname));
                }
            }
        } // grids.

        // Calculate wave-front settings based on max halos.
        // See the wavefront diagram in run_solution() for description
        // of angles and extensions.
        auto& step_dim = _dims->_step_dim;
        auto wf_steps = _opts->_region_sizes[step_dim];
        num_wf_shifts = 0;
        if (wf_steps > 1)

            // TODO: don't shift for scratch grids.
            num_wf_shifts = max((idx_t(stBundles.size()) * wf_steps) - 1, idx_t(0));
        for (auto& dim : _dims->_domain_dims.getDims()) {
            auto& dname = dim.getName();
            auto rksize = _opts->_rank_sizes[dname];
            auto nranks = _opts->_num_ranks[dname];

            // Determine the max spatial skewing angles for temporal
            // wave-fronts based on the max halos.  We only need non-zero
            // angles if the region size is less than the rank size and
            // there are no other ranks in this dim, i.e., if the region
            // covers the global domain in a given dim, no wave-front is
            // needed in that dim.  TODO: make rounding-up an option.
            idx_t angle = 0;
            if (_opts->_region_sizes[dname] < rksize || nranks > 0)
                angle = ROUND_UP(max_halos[dname], _dims->_cluster_pts[dname]);
            wf_angles[dname] = angle;

            // Determine the total WF shift to be added in each dim.
            idx_t shifts = angle * num_wf_shifts;
            wf_shifts[dname] = shifts;

            // Is domain size at least as large as halo + wf_ext in direction
            // when there are multiple ranks?
            auto min_size = max_halos[dname] + shifts;
            if (_opts->_num_ranks[dname] > 1 && rksize < min_size) {
                THROW_YASK_EXCEPTION("Error: rank-domain size of " << rksize << " in '" <<
                                     dname << "' dim is less than minimum size of " << min_size <<
                                     ", which is based on stencil halos and temporal wave-front sizes");
            }

            // If there is another rank to the left, set wave-front
            // extension on the left.
            left_wf_exts[dname] = _opts->is_first_rank(dname) ? 0 : shifts;

            // If there is another rank to the right, set wave-front
            // extension on the right.
            right_wf_exts[dname] = _opts->is_last_rank(dname) ? 0 : shifts;
        }            
            
        // Now that wave-front settings are known, we can push this info
        // back to the grids. It's useful to store this redundant info
        // in the grids, because there it's indexed by grid dims instead
        // of domain dims. This makes it faster to do grid indexing.
        for (auto gp : gridPtrs) {
            assert(gp);

            // Ignore manually-sized grid.
            if (gp->is_fixed_size())
                continue;

            // Loop through each domain dim.
            for (auto& dim : _dims->_domain_dims.getDims()) {
                auto& dname = dim.getName();
                if (gp->is_dim_used(dname)) {

                    // Set extensions to be the same as the global ones.
                    gp->_set_left_wf_ext(dname, left_wf_exts[dname]);
                    gp->_set_right_wf_ext(dname, right_wf_exts[dname]);
                }
            }
        }
    }
    
    // Allocate grids and MPI bufs.
    // Initialize some data structures.
    void StencilContext::prepare_solution() {
        auto& step_dim = _dims->_step_dim;

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
        
        // reset time keepers.
        clear_timers();

        // Init auto-tuner to run silently during normal operation.
        _at.clear(false, false);

        // Adjust all settings before setting MPI buffers or sizing grids.
        // Prints final settings.
        // TODO: print settings again after auto-tuning.
        _opts->adjustSettings(os, _env);

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

        // Set the number of threads for a region. It should stay this
        // way for top-level OpenMP parallel sections.
        int rthreads = set_region_threads();

        // Run a dummy nested OMP loop to make sure nested threading is
        // initialized.
#ifdef _OPENMP
#pragma omp parallel for
        for (int i = 0; i < rthreads * 100; i++) {

            idx_t dummy = 0;
            set_block_threads();
#pragma omp parallel for reduction(+:dummy)
            for (int j = 0; j < i * 100; j++) {
                dummy += j;
            }
        }
#endif

        // Some grid stats.
        os << endl;
        os << "Num grids: " << gridPtrs.size() << endl;
        os << "Num grids to be updated: " << outputGridPtrs.size() << endl;
        
        // Set up data based on MPI rank, including grid positions.
        // Update all the grid sizes.
        setupRank();

        // Alloc grids, scratch grids, MPI bufs.
        // This is the order in which preferred NUMA nodes (e.g., HBW mem)
        // will be used.
        // We free the scratch and MPI data first to give grids preference.
        freeScratchData(os);
        freeMpiData(os);
        allocGridData(os);
        allocScratchData(os);
        allocMpiData(os);

        // Report total allocation.
        rank_nbytes = get_num_bytes();
        os << "Total allocation in this rank: " <<
            makeByteStr(rank_nbytes) << "\n";
        tot_nbytes = sumOverRanks(rank_nbytes, _env->comm);
        os << "Total overall allocation in " << _env->num_ranks << " rank(s): " <<
            makeByteStr(tot_nbytes) << "\n";
    
        // Report some stats.
        idx_t dt = _opts->_rank_sizes[step_dim];
        os << "\nProblem sizes in points (from smallest to largest):\n"
            " vector-size:           " << _dims->_fold_pts.makeDimValStr(" * ") << endl <<
            " cluster-size:          " << _dims->_cluster_pts.makeDimValStr(" * ") << endl <<
            " sub-block-size:        " << _opts->_sub_block_sizes.makeDimValStr(" * ") << endl <<
            " sub-block-group-size:  " << _opts->_sub_block_group_sizes.makeDimValStr(" * ") << endl <<
            " block-size:            " << _opts->_block_sizes.makeDimValStr(" * ") << endl <<
            " block-group-size:      " << _opts->_block_group_sizes.makeDimValStr(" * ") << endl <<
            " region-size:           " << _opts->_region_sizes.makeDimValStr(" * ") << endl <<
            " rank-domain-size:      " << _opts->_rank_sizes.makeDimValStr(" * ") << endl <<
            " overall-problem-size:  " << overall_domain_sizes.makeDimValStr(" * ") << endl <<
            endl <<
            "Other settings:\n"
            " yask-version:          " << yask_get_version_string() << endl <<
            " stencil-name:          " << get_name() << endl <<
            " element-size:          " << makeByteStr(get_element_bytes()) << endl <<
#ifdef USE_MPI
            " num-ranks:             " << _opts->_num_ranks.makeDimValStr(" * ") << endl <<
            " rank-indices:          " << _opts->_rank_indices.makeDimValStr() << endl <<
            " rank-domain-offsets:   " << rank_domain_offsets.makeDimValOffsetStr() << endl <<
#endif
            " rank-domain:           " << rank_bb.bb_begin.makeDimValStr() <<
                " ... " << rank_bb.bb_end.subElements(1).makeDimValStr() << endl <<
            " vector-len:            " << VLEN << endl <<
            " extra-padding:         " << _opts->_extra_pad_sizes.makeDimValStr() << endl <<
            " minimum-padding:       " << _opts->_min_pad_sizes.makeDimValStr() << endl <<
            " L1-prefetch-distance:  " << PFD_L1 << endl <<
            " L2-prefetch-distance:  " << PFD_L2 << endl <<
            " max-halos:             " << max_halos.makeDimValStr() << endl;
        if (num_wf_shifts > 0) {
            os <<
                " wave-front-angles:     " << wf_angles.makeDimValStr() << endl <<
                " num-wave-front-shifts: " << num_wf_shifts << endl <<
                " wave-front-shift-lens: " << wf_shifts.makeDimValStr() << endl <<
                " left-wave-front-exts:  " << left_wf_exts.makeDimValStr() << endl <<
                " right-wave-front-exts: " << right_wf_exts.makeDimValStr() << endl <<
                " ext-rank-domain:       " << ext_bb.bb_begin.makeDimValStr() <<
                " ... " << ext_bb.bb_end.subElements(1).makeDimValStr() << endl;
        }
        os << endl;
        
        // sums across bundles for this rank.
        rank_numWrites_1t = 0;
        rank_reads_1t = 0;
        rank_numFpOps_1t = 0;
        os << "Num stencil bundles: " << stBundles.size() << endl;
        for (auto* sg : stBundles) {
            idx_t updates1 = sg->get_scalar_points_written();
            idx_t updates_domain = updates1 * sg->bb_num_points;
            rank_numWrites_1t += updates_domain;
            idx_t reads1 = sg->get_scalar_points_read();
            idx_t reads_domain = reads1 * sg->bb_num_points;
            rank_reads_1t += reads_domain;
            idx_t fpops1 = sg->get_scalar_fp_ops();
            idx_t fpops_domain = fpops1 * sg->bb_num_points;
            rank_numFpOps_1t += fpops_domain;
            os << "Stats for bundle '" << sg->get_name() << "':\n" <<
                " sub-domain:                 " << sg->bb_begin.makeDimValStr() <<
                " ... " << sg->bb_end.subElements(1).makeDimValStr() << endl <<
                " sub-domain size:            " << sg->bb_len.makeDimValStr(" * ") << endl <<
                " valid points in sub domain: " << makeNumStr(sg->bb_num_points) << endl <<
                " grid-updates per point:     " << updates1 << endl <<
                " grid-updates in sub-domain: " << makeNumStr(updates_domain) << endl <<
                " grid-reads per point:       " << reads1 << endl <<
                " grid-reads in sub-domain:   " << makeNumStr(reads_domain) << endl <<
                " est FP-ops per point:       " << fpops1 << endl <<
                " est FP-ops in sub-domain:   " << makeNumStr(fpops_domain) << endl;
        }

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

        rank_domain_1t = rank_bb.bb_num_points;
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
            " Num-writes-required is based on sum of grid-updates in sub-domain across stencil-bundle(s).\n"
            " Num-reads-required is based on sum of grid-reads in sub-domain across stencil-bundle(s).\n"
            " Est-FP-ops are based on sum of est-FP-ops in sub-domain across stencil-bundle(s).\n"
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
    
    // Dealloc grids, etc.
    void StencilContext::end_solution() {

        // Final halo exchange.
        exchange_halos_all();

        // Release any MPI data.
        mpiData.clear();

        // Release grid data.
        for (auto gp : gridPtrs) {
            if (!gp)
                continue;
            gp->release_storage();
        }

	// Reset threads to original value.
	set_max_threads();
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

        // Solid rectangle?
        bb_is_full = true;
        if (bb_num_points != bb_size) {
            os << "Warning: '" << name << "' domain has only " <<
                makeNumStr(bb_num_points) <<
                " valid point(s) inside its bounding-box of " <<
                makeNumStr(bb_size) <<
                " point(s); slower scalar calculations will be used.\n";
            bb_is_full = false;
        }

        // Does everything start on a vector-length boundary?
        bb_is_aligned = true;
        for (auto& dim : domain_dims.getDims()) {
            auto& dname = dim.getName();
            if ((bb_begin[dname] - context.rank_domain_offsets[dname]) %
                dims->_fold_pts[dname] != 0) {
                os << "Note: '" << name << "' domain"
                    " has one or more starting edges not on vector boundaries;"
                    " masked calculations will be used in peel and remainder sub-blocks.\n";
                bb_is_aligned = false;
                break;
            }
        }

        // Lengths are cluster-length multiples?
        bb_is_cluster_mult = true;
        for (auto& dim : domain_dims.getDims()) {
            auto& dname = dim.getName();
            if (bb_len[dname] % dims->_cluster_pts[dname] != 0) {
                if (bb_is_full && bb_is_aligned)
                    os << "Note: '" << name << "' domain"
                        " has one or more sizes that are not vector-cluster multiples;"
                        " masked calculations will be used in peel and remainder sub-blocks.\n";
                bb_is_cluster_mult = false;
                break;
            }
        }

        // All done.
        bb_valid = true;
    }
    
    // Set the bounding-box for each stencil-bundle and whole domain.
    void StencilContext::find_bounding_boxes()
    {
        ostream& os = get_ostr();

        // Rank BB is based only on rank offsets and rank domain sizes.
        rank_bb.bb_begin = rank_domain_offsets;
        rank_bb.bb_end = rank_domain_offsets.addElements(_opts->_rank_sizes, false);
        rank_bb.update_bb(os, "rank", *this, true);

        // Overall BB may be extended for wave-fronts.
        ext_bb.bb_begin = rank_bb.bb_begin.subElements(left_wf_exts);
        ext_bb.bb_end = rank_bb.bb_end.addElements(right_wf_exts);
        ext_bb.update_bb(os, "extended-rank", *this, true);

        // Find BB for each bundle.
        for (auto sg : stBundles)
            sg->find_bounding_box();
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
        
        // Initial halo exchange for each bundle.
        for (auto* sg : stBundles) {

            // Do exchange over max steps.
            exchange_halos(start, stop, *sg);
        }
#endif
    }
    
    // Exchange halo data needed by stencil-bundle 'sg' at the given time.
    // Data is needed for input grids that have not already been updated.
    // [BIG] TODO: overlap halo exchange with computation.
    void StencilContext::exchange_halos(idx_t start, idx_t stop, StencilBundleBase& sg)
    {
#ifdef USE_MPI
        if (!enable_halo_exchange || _env->num_ranks < 2)
            return;
        mpi_time.start();
        TRACE_MSG("exchange_halos: " << start << " ... (end before) " << stop <<
                  " for stencil-bundle '" << sg.get_name() << "'");
        auto opts = get_settings();
        auto& sd = _dims->_step_dim;

        // 1D array to store send request handles.
        // We use a 1D array so we can call MPI_Waitall().
        MPI_Request send_reqs[sg.inputGridPtrs.size() * _mpiInfo->neighborhood_size];

        // 2D array for receive request handles.
        // We use a 2D array to simplify individual indexing.
        MPI_Request recv_reqs[sg.inputGridPtrs.size()][_mpiInfo->neighborhood_size];

        // Loop through steps.  This loop has to be outside halo-step loop
        // because we only have one buffer per step. Normally, we only
        // exchange one step; in that case, it doesn't matter. It would be more
        // efficient to allow packing and unpacking multiple steps, esp. with WFs.
        // TODO: this will need to be addressed if/when comm/compute overlap is added.
        assert(start != stop);
        idx_t step = (start < stop) ? 1 : -1;
        for (idx_t t = start; t != stop; t += step) {
            int num_send_reqs = 0;

            // Sequence of things to do for each grid's neighbors
            // (isend includes packing).
            enum halo_steps { halo_irecv, halo_pack_isend, halo_unpack, halo_nsteps };
            for (int halo_step = 0; halo_step < halo_nsteps; halo_step++) {

                if (halo_step == halo_irecv)
                    TRACE_MSG("exchange_halos: requesting data for step " << t << "...");
                else if (halo_step == halo_pack_isend)
                    TRACE_MSG("exchange_halos: packing and sending data for step " << t << "...");
                else if (halo_step == halo_unpack)
                    TRACE_MSG("exchange_halos: unpacking data for step " << t << "...");
            
                // Loop thru all input grids in this bundle.
                for (size_t gi = 0; gi < sg.inputGridPtrs.size(); gi++) {
                    auto gp = sg.inputGridPtrs[gi];
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
                                }
                            }

                            // Pack data into send buffer, then send to neighbor.
                            else if (halo_step == halo_pack_isend) {
                                auto nbytes = sendBuf.get_bytes();
                                if (nbytes) {

                                    // Vec ok?
                                    // Domain sizes must be ok, and buffer size must be ok
                                    // as calculated when buffers were created.
                                    bool send_vec_ok = vec_ok && sendBuf.has_all_vlen_mults;

                                    // Get first and last ranges.
                                    IdxTuple first = sendBuf.begin_pt;
                                    IdxTuple last = sendBuf.last_pt;

                                    // The code in allocData() pre-calculated the first and
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
                            }

                            // Wait for data from neighbor, then unpack it.
                            else if (halo_step == halo_unpack) {
                                auto nbytes = recvBuf.get_bytes();
                                if (nbytes) {

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
                            }
                        }); // visit neighbors.

                } // grids.
            } // exchange sequence.
            
            // Mark grids as up-to-date.
            for (size_t gi = 0; gi < sg.inputGridPtrs.size(); gi++) {
                auto gp = sg.inputGridPtrs[gi];
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
