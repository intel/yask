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
// Also see setup.cpp.

#include "yask_stencil.hpp"
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
        update_grid_info();                                                 \
        if (reset_prep) rank_bb.bb_valid = ext_bb.bb_valid = false;     \
    }
    SET_SOLN_API(set_min_pad_size, _opts->_min_pad_sizes[dim] = n, false, true, false, false)
    SET_SOLN_API(set_block_size, _opts->_block_sizes[dim] = n, false, true, false, true)
    SET_SOLN_API(set_region_size, _opts->_region_sizes[dim] = n, true, true, false, true)
    SET_SOLN_API(set_rank_domain_size, _opts->_rank_sizes[dim] = n, false, true, false, true)
    SET_SOLN_API(set_num_ranks, _opts->_num_ranks[dim] = n, false, true, false, true)
    SET_SOLN_API(set_rank_index, _opts->_rank_indices[dim] = n, false, true, false, true)
#undef SET_SOLN_API

    // Constructor.
    StencilContext::StencilContext(KernelEnvPtr env,
                                   KernelSettingsPtr settings) :
    _ostr(&std::cout),
        _env(env),
        _opts(settings),
        _dims(settings->_dims)
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

        ostream& os = get_ostr();
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

        TRACE_MSG("calc_rank_ref: [" << begin.makeDimValStr() << " ... " <<
                  end.makeDimValStr() << ")");

        // Force region & block sizes to whole rank size so that scratch
        // grids will be large enough. Turn off any temporal blocking.
        _opts->_region_sizes.setValsSame(0);
        _opts->_region_sizes[_dims->_step_dim] = 1;
        _opts->_block_sizes.setValsSame(0);
        _opts->_block_sizes[_dims->_step_dim] = 1;
        _opts->adjustSettings(get_env());

        // Copy these settings to packs and realloc scratch grids.
        for (auto& sp : stPacks)
            sp->getSettings() = *_opts;
        allocScratchData(os);

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
        update_scratch_grid_info(scratch_grid_idx, rank_idxs.begin);

        // Initial halo exchange.
        // TODO: get rid of all halo exchanges in this function,
        // and calculate overall problem in one rank.
        exchange_halos();

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

            // Loop thru bundles. We ignore bundle packs here
            // because packing bundles is an optional optimizations.
            for (auto* asg : stBundles) {

                // Scan through n-D space.
                TRACE_MSG("calc_rank_ref: step " << start_t <<
                          " in non-scratch group '" << asg->get_name());

                // Exchange all dirty halos.
                exchange_halos();

                // Find the groups that need to be processed.
                // This will be the prerequisite scratch-grid
                // groups plus this non-scratch group.
                auto sg_list = asg->get_reqd_bundles();

                // Loop through all the needed bundles.
                ext_time.start();
                for (auto* sg : sg_list) {

                    // Indices needed for the generated misc loops.  Will normally be a
                    // copy of rank_idxs except when updating scratch-grids.
                    ScanIndices misc_idxs = sg->adjust_span(scratch_grid_idx, rank_idxs);
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
                              " in bundle '" << sg->get_name() << "': [" <<
                              misc_idxs.begin.makeValStr(ndims) <<
                              " ... " << misc_idxs.end.makeValStr(ndims) << ")");
#include "yask_misc_loops.hpp"
#undef misc_fn
                } // needed bundles.

                // Mark grids that [may] have been written to.
                // Mark grids as dirty even if not actually written by this
                // rank. This is needed because neighbors will not know what
                // grids are actually dirty, and all ranks must have the same
                // information about which grids are possibly dirty.
                mark_grids_dirty(nullptr, start_t, stop_t);

                ext_time.stop();
            } // all bundles.

        } // iterations.

        // Final halo exchange.
        exchange_halos();

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

    // Eval stencil bundle pack(s) over grid(s) using optimized code.
    void StencilContext::run_solution(idx_t first_step_index,
                                      idx_t last_step_index)
    {
        run_time.start();

        auto& step_dim = _dims->_step_dim;
        auto step_posn = Indices::step_posn;
        int ndims = _dims->_stencil_dims.size();

        // Determine step dir from order of first/last.
        idx_t step_dir = (first_step_index > last_step_index) ? -1 : 1;
        
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

        TRACE_MSG("run_solution: [" << begin.makeDimValStr() << " ... " <<
                  end.makeDimValStr() << ") by " << step.makeDimValStr());
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
        // bundle pack. So, the total shift in a region is the angle * num
        // packs * num timesteps. This assumes all bundle packs
        // are inter-dependent to find maximum extension. Actual required
        // extension may be less, but this will just result in some calls to
        // calc_region() that do nothing.
        //
        // Conceptually (showing 2 ranks in t and x dims):
        // -----------------------------  t = rt ------------------------------
        //   \   | \     \     \|  \   |  .      |   / |  \     \     \|  \   |
        //    \  |  \     \     |   \  |  .      |  / \|   \     \     |   \  |
        //     \ |r0 \  r1 \ r2 |\ r3\ |  .      | /r0 | r1 \  r2 \ r3 |\ r4\ |
        //      \|    \     \   | \   \|  .      |/    |\    \     \   | \   \|
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
                      " wave-front shift(s): [" <<
                      begin.makeDimValStr() << " ... " <<
                      end.makeDimValStr() << ")");
        }

        // Indices needed for the 'rank' loops.
        ScanIndices rank_idxs(*_dims, true, &rank_domain_offsets);
        rank_idxs.begin = begin;
        rank_idxs.end = end;
        rank_idxs.step = step;

        // Make sure threads are set properly for a region.
        set_region_threads();

        // Initial halo exchange.
        exchange_halos();

        // Number of iterations to get from begin_t to end_t-1,
        // stepping by step_t.
        const idx_t num_t = CEIL_DIV(abs(end_t - begin_t), abs(step_t));
        for (idx_t index_t = 0; index_t < num_t; index_t++)
        {
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

            // If no wave-fronts (default), loop through packs here, and do
            // only one pack at a time in calc_region(). This is similar to
            // loop in calc_rank_ref(), but with packs instead of bundles.
            if (step_t == 1) {

                // Loop thru packs.
                for (auto& bp : stPacks) {

                    // Make 2 passes. 1: compute data needed for MPI
                    // send and send that data. 2: compute remaining
                    // data and unpack received MPI data.
                    for (int pass = 0; pass < 2; pass++) {

                        // If there is an MPI interior defined, set
                        // the proper flags.
                        if (mpi_interior.bb_valid) {
                            if (pass == 0) {
                                do_mpi_exterior = true;
                                do_mpi_interior = false;
                            } else {
                                do_mpi_exterior = false;
                                do_mpi_interior = true;
                            }
                        } else {
                            do_mpi_exterior = true;
                            do_mpi_interior = true;

                            // Only 1 pass needed when needed when not
                            // overlapping comms and compute.
                            if (pass > 0)
                                break;
                        }
                        
                        // Include automatically-generated loop code that calls
                        // calc_region(bp) for each region.
                        TRACE_MSG("run_solution: step " << start_t <<
                                  " for bundle-pack '" << bp->get_name() << "'");
                        if (do_mpi_exterior)
                            TRACE_MSG(" within MPI exterior");
                        if (do_mpi_interior)
                            TRACE_MSG(" within MPI interior");
#include "yask_rank_loops.hpp"

                        // Do the appropriate steps for halo exchange.
                        exchange_halos();

                    } // passes.

                    // Set the flags back to default.
                    do_mpi_exterior = true;
                    do_mpi_interior = true;
                }
            }

            // If doing wave-fronts, must loop through all packs in
            // calc_region().  TODO: optionally enable this when there are
            // multiple packs but step_t == 1.
            else {

                // Null ptr => Eval all stencil bundles each time
                // calc_region() is called.
                BundlePackPtr bp;

                // Include automatically-generated loop code that calls
                // calc_region() for each region.
                TRACE_MSG("run_solution: steps [" << start_t <<
                          " ... " << stop_t << ")");
#include "yask_rank_loops.hpp"

                // Exchange dirty halo(s).
                exchange_halos();
            }

            steps_done += this_num_t;

            // Call the auto-tuner to evaluate these steps.
            for (auto& sp : stPacks)
                sp->getAT().eval(this_num_t);

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

    // Calculate results within a region.  Each region is typically computed
    // in a separate OpenMP 'for' region.  In this function, we loop over
    // the time steps and bundle packs and evaluate a pack in each of
    // the blocks in the region.  If 'sel_bp' is null, eval all packs; else
    // eval only the one pointed to.
    void StencilContext::calc_region(BundlePackPtr& sel_bp,
                                     const ScanIndices& rank_idxs) {

        int ndims = _dims->_stencil_dims.size();
        auto& step_dim = _dims->_step_dim;
        auto step_posn = Indices::step_posn;
        TRACE_MSG("calc_region: [" << rank_idxs.start.makeValStr(ndims) <<
                  " ... " << rank_idxs.stop.makeValStr(ndims) << ")");

        // Track time (use "else" to avoid double-counting).
        if (do_mpi_exterior)
            ext_time.start();
        else if (do_mpi_interior)
            int_time.start();

        // Init region begin & end from rank start & stop indices.
        ScanIndices region_idxs(*_dims, true, &rank_domain_offsets);
        region_idxs.initFromOuter(rank_idxs);

        // Make a copy of the original index span because
        // we will be shifting it for temporal wavefronts.
        Indices start(region_idxs.begin);
        Indices stop(region_idxs.end);

        // Step (usually time) loop.
        // When doing WF tiling, this loop will step through
        // several time-steps in each region.
        idx_t begin_t = region_idxs.begin[step_posn];
        idx_t end_t = region_idxs.end[step_posn];
        idx_t step_t = _opts->_block_sizes[step_posn];
        if (step_t != 1)
            FORMAT_AND_THROW_YASK_EXCEPTION
                ("Error: temporal block size " << step_t << " not allowed");
        const idx_t num_t = CEIL_DIV(abs(end_t - begin_t), abs(step_t));
        idx_t shift_num = 0;
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

            // Stencil bundle packs to evaluate at this time step.
            for (auto& bp : stPacks) {

                // Not a selected bundle pack?
                if (sel_bp && sel_bp != bp)
                    continue;

                TRACE_MSG("calc_region: bundle-pack '" << bp->get_name() << "' in step(s) [" <<
                          start_t << " ... " << stop_t << ")");

                // Start timers for this bundle.
                bp->getAT().timer.start();
                bp->timer.start();

                // Steps within a region are based on block sizes.
                // These may have been tweaked by the auto-tuner.
                auto& blksize = bp->getSettings()._block_sizes;
                region_idxs.step = blksize;
                region_idxs.step[step_posn] = step_t; // only wanted domain sizes.

                // Groups in region loops are based on block-group sizes.
                auto& bgsize = bp->getSettings()._block_group_sizes;
                region_idxs.group_size = bgsize;
                region_idxs.group_size[step_posn] = step_t; // only wanted domain sizes.
                
                // For wavefront adjustments, see conceptual diagram in
                // run_solution().  In this function, one of the
                // parallelogram-shaped regions is being evaluated.  These
                // shapes may extend beyond actual boundaries. So, at each
                // time-step, the parallelogram may be trimmed based on the
                // BB and WF extensions outside of the rank-BB.

                // Actual region boundaries must stay within [extended] pack BB.
                // We have to calculate the posn in the extended rank at each
                // value of 'shift_num' because it is being shifted spatially.
                bool ok = true;
                auto& pbb = bp->getBB();
                for (int i = 0, j = 0; i < ndims; i++) {
                    if (i == step_posn) continue;
                    auto angle = wf_angles[j];

                    // Begin point.
                    idx_t dbegin = rank_bb.bb_begin[j]; // non-extended domain.
                    idx_t rbegin = max<idx_t>(start[i], pbb.bb_begin[j]);
                    if (rbegin < dbegin) // in left WF ext?
                        rbegin = max(rbegin, dbegin - left_wf_exts[j] + shift_num * angle);
                    region_idxs.begin[i] = rbegin;

                    // End point.
                    idx_t dend = rank_bb.bb_end[j]; // non-extended domain.
                    idx_t rend = min<idx_t>(stop[i], pbb.bb_end[j]);
                    if (rend > dend) // in right WF ext?
                        rend = min(rend, dend + right_wf_exts[j] - shift_num * angle);
                    region_idxs.end[i] = rend;

                    // Anything to do?
                    if (rend <= rbegin)
                        ok = false;

                    j++; // next domain index.
                }
                TRACE_MSG("calc_region: region span after trimming: [" <<
                          region_idxs.begin.makeValStr(ndims) <<
                          " ... " << region_idxs.end.makeValStr(ndims) << ")");

                // Only need to loop through the span of the region if it is
                // at least partly inside the extended BB. For overlapping
                // regions, they may start outside the domain but enter the
                // domain as time progresses and their boundaries shift. So,
                // we don't want to return if this condition isn't met.
                if (ok) {

                    // Include automatically-generated loop code that
                    // calls calc_block() for each block in this region.
                    // Loops through x from begin_rx to end_rx-1;
                    // similar for y and z.  This code typically
                    // contains the outer OpenMP loop(s).
#include "yask_region_loops.hpp"
                }

                // Mark grids that [may] have been written to by this pack,
                // updated at next step (+/- 1).  Only mark for exterior
                // computation, because we don't care about blocks not
                // needed for MPI sends.  Mark grids as dirty even if not
                // actually written by this rank, perhaps due to
                // sub-domains. This is needed because neighbors will not
                // know what grids are actually dirty, and all ranks must
                // have the same information about which grids are possibly
                // dirty.  TODO: make this smarter to save unneeded MPI
                // exchanges.
                if (do_mpi_exterior)
                    mark_grids_dirty(bp, start_t, stop_t);

                // Shift spatial region boundaries for next iteration to
                // implement temporal wavefront.  Between regions, we only shift
                // left, so region loops must strictly increment. They may do
                // so in any order.  TODO: shift only what is needed by
                // this pack, not the global max.
                for (int i = 0, j = 0; i < ndims; i++) {
                    if (i == step_posn) continue;
                    auto angle = wf_angles[j];

                    start[i] -= angle;
                    stop[i] -= angle;
                    j++;
                }
                shift_num++;

                // Stop timers for this bundle.
                bp->getAT().timer.stop();
                bp->timer.stop();

            } // stencil bundle packs.
        } // time.

        if (do_mpi_exterior) {
            double ext_delta = ext_time.stop();
            TRACE_MSG("secs spent in this region for rank-exterior blocks: " << makeNumStr(ext_delta));
        }
        else if (do_mpi_interior) {
            double int_delta = int_time.stop();
            TRACE_MSG("secs spent in this region for rank-interior blocks: " << makeNumStr(int_delta));
        }

    } // calc_region.

    // Calculate results within a block. This function calls
    // 'calc_block' for each bundle in the specified pack.
    // Typically called by a top-level OMP thread from calc_region().
    void StencilContext::calc_block(BundlePackPtr& sel_bp,
                                    const ScanIndices& region_idxs) {

        int nsdims = _dims->_stencil_dims.size();
        auto& step_dim = _dims->_step_dim;
        auto step_posn = Indices::step_posn;
        auto* bp = sel_bp.get();
        assert(bp);
        int thread_idx = omp_get_thread_num();
        TRACE_MSG("calc_block for pack '" << bp->get_name() << "': [" <<
                  region_idxs.start.makeValStr(nsdims) <<
                  " ... " << region_idxs.stop.makeValStr(nsdims) <<
                  ") in thread " << thread_idx);

        // If we are not calculating some of the blocks, determine
        // whether this block is *completely* inside the interior.
        // A block even partially in the exterior is not considered
        // "inside".
        if (!do_mpi_interior || !do_mpi_exterior) {
            assert(do_mpi_interior || do_mpi_exterior);
            assert(mpi_interior.bb_valid);

            // Starting point and ending point must be in BB.
            bool inside = true;
            for (int i = 0, j = 0; i < nsdims; i++) {
                if (i == step_posn) continue;

                // Starting before beginning of interior?
                if (region_idxs.start[i] < mpi_interior.bb_begin[j]) {
                    inside = false;
                    break;
                }

                // Stopping after ending of interior?
                if (region_idxs.stop[i] > mpi_interior.bb_end[j]) {
                    inside = false;
                    break;
                }

                j++;
            }
            if (do_mpi_interior) {
                if (inside)
                    TRACE_MSG(" calculating because block is interior");
                else {
                    TRACE_MSG(" *not* calculating because block is exterior");
                    return;
                }
            }
            if (do_mpi_exterior) {
                if (!inside)
                    TRACE_MSG(" calculating because block is exterior");
                else {
                    TRACE_MSG(" *not* calculating because block is interior");
                    return;
                }
            }
        }

        // Hack to promote forward progress in MPI when calc'ing
        // interior only.
        // We do this only on thread 0 to avoid stacking up useless
        // MPI requests by many threads.
        if (do_mpi_interior && !do_mpi_exterior && thread_idx == 0)
            exchange_halos(true);

        // Init block begin & end from region start & stop indices.
        ScanIndices block_idxs(*_dims, true, 0);
        block_idxs.initFromOuter(region_idxs);

        // Steps within a block are based on sub-block sizes.
        // These may have been tweaked by the auto-tuner.
        auto& sbsize = bp->getSettings()._sub_block_sizes;
        block_idxs.step = sbsize;

        // Groups in block loops are based on sub-block-group sizes.
        auto& sbgsize = bp->getSettings()._sub_block_group_sizes;
        block_idxs.group_size = sbgsize;

        // Loop through bundles in this pack.
        for (auto* sb : *bp) {
            sb->calc_block(block_idxs);
        }
    }

    // Reset auto-tuners.
    void StencilContext::reset_auto_tuner(bool enable, bool verbose) {
        for (auto& sp : stPacks)
            sp->getAT().clear(!enable, verbose);
    }

    // Determine if any auto tuners are running.
    bool StencilContext::is_auto_tuner_enabled() const {
        bool done = true;
        for (auto& sp : stPacks)
            if (!sp->getAT().is_done())
                done = false;
        return !done;
    }
    
    // Apply auto-tuning immediately, i.e., not as part of normal processing.
    // Will alter data in grids.
    void StencilContext::run_auto_tuner_now(bool verbose) {
        if (!rank_bb.bb_valid)
            THROW_YASK_EXCEPTION("Error: tune_settings() called without calling prepare_solution() first");

        ostream& os = get_ostr();
        os << "Auto-tuning...\n" << flush;
        YaskTimer at_timer;
        at_timer.start();

        // Temporarily disable halo exchange to tune intra-rank.
        enable_halo_exchange = false;

        // Init tuners.
        for (auto& sp : stPacks)
            sp->getAT().clear(false, verbose);

        // Reset stats.
        clear_timers();

        // Determine number of sets to run.
        // If wave-fronts are enabled, run a max number of these steps.
        // TODO: only run one region during AT.
        idx_t region_steps = _opts->_region_sizes[_dims->_step_dim];
        idx_t step_dir = _dims->_step_dir;
        idx_t step_t = min(region_steps, AutoTuner::max_step_t) * step_dir;

        // Run time-steps until AT converges.
        for (idx_t t = 0; ; t += step_t) {

            // Run step_t time-step(s).
            run_solution(t, t + step_t - step_dir);

            // AT done on this rank?
            if (!is_auto_tuner_enabled())
                break;
        }

        // Wait for all ranks to finish.
        _env->global_barrier();

        // reenable halo exchange.
        enable_halo_exchange = true;

        // Report results.
        at_timer.stop();
        os << "Auto-tuner done after " << steps_done << " step(s) in " <<
            at_timer.get_elapsed_secs() << " secs.\n";
        for (auto& sp : stPacks) {
            auto& settings = sp->getSettings();
            os << " for pack '" << sp->get_name() << "':\n" <<
                "  best-block-size: " << settings._block_sizes.makeDimValStr(" * ") << endl <<
                "  best-sub-block-size: " << settings._sub_block_sizes.makeDimValStr(" * ") << endl << flush;
        }

        // Reset stats.
        clear_timers();
    }

    // Add a new grid to the containers.
    void StencilContext::addGrid(YkGridPtr gp, bool is_output) {
        assert(gp);
        auto& gname = gp->get_name();
        if (gridMap.count(gname))
            THROW_YASK_EXCEPTION("Error: grid '" + gname + "' already exists");

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
    void StencilContext::update_scratch_grid_info(int thread_idx,
                                                  const Indices& idxs) {
        auto& dims = get_dims();
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

                        // | ... |        +------+       |
                        // |  global ofs  |      |       |
                        // |<------------>|grid/ |       |
                        // |     |  loc   | blk  |       |
                        // |rank |  ofs   |domain|       |
                        // | ofs |<------>|      |       |
                        // |<--->|        +------+       |
                        // ^     ^        ^              ^
                        // |     |        |              last rank-domain index
                        // |     |        start of grid-domain/0-idx of block
                        // |     first rank-domain index
                        // first overall-domain index

                        // Local offset is the offset of this grid
                        // relative to the current rank.
                        // Set local offset to diff between global offset
                        // and rank offset.
                        // Round down to make sure it's vec-aligned.
                        auto rofs = rank_domain_offsets[j];
                        auto vlen = gp->_get_vec_len(posn);
                        auto lofs = round_down_flr(idxs[i] - rofs, vlen);
                        gp->_set_local_offset(posn, lofs);

                        // Set global offset of grid based on starting point of block.
                        // This is a global index, so it will include the rank offset.
                        // Thus, it it not necessarily a vec mult.
                        // Need to use calculated local offset to adjust for any
                        // rounding that was done above.
                        gp->_set_offset(posn, rofs + lofs);
                    }
                    j++;
                }
            }
        }
    }

    static void print_pct(ostream& os, double ntime, double dtime) {
        if (dtime > 0.) {
            float pct = 100. * ntime / dtime;
            os << " (" << pct << "%)";
        }
        os << endl;
    }

    /// Get statistics associated with preceding calls to run_solution().
    yk_stats_ptr StencilContext::get_stats() {
        ostream& os = get_ostr();

        // Calc and report perf.
        double rtime = run_time.get_elapsed_secs();
        double htime = min(halo_time.get_elapsed_secs(), rtime);
        double wtime = min(wait_time.get_elapsed_secs(), htime);
        double etime = min(ext_time.get_elapsed_secs(), rtime - htime);
        double itime = min(int_time.get_elapsed_secs(), rtime - htime - etime);
        double ctime = etime + itime;
        double otime = max(rtime - ctime - htime, 0.);
        if (rtime > 0.) {
            domain_pts_ps = double(tot_domain_1t * steps_done) / rtime;
            writes_ps= double(tot_numWrites_1t * steps_done) / rtime;
            flops = double(tot_numFpOps_1t * steps_done) / rtime;
        }
        else
            domain_pts_ps = writes_ps = flops = 0.;
        if (steps_done > 0) {
            os <<
                "Amount-of-work stats:\n"
                " num-points-per-step:              " << makeNumStr(tot_domain_1t) << endl <<
                " num-writes-per-step:              " << makeNumStr(tot_numWrites_1t) << endl <<
                " num-est-FP-ops-per-step:          " << makeNumStr(tot_numFpOps_1t) << endl <<
                " num-steps-done:                   " << makeNumStr(steps_done) << endl <<
                "Performance stats:\n"
                " elapsed-time (sec):               " << makeNumStr(rtime) << endl <<
                " Time breakdown by activity type:\n"
                "  compute (sec):                    " << makeNumStr(ctime);
            print_pct(os, ctime, rtime);
#ifdef USE_MPI
            os <<
                "  halo exchange (sec):              " << makeNumStr(htime);
            print_pct(os, htime, rtime);
#endif
            os <<
                "  other (sec):                      " << makeNumStr(otime);
            print_pct(os, otime, rtime);
            os <<
                " Compute-time breakdown by stencil pack(s):\n";
            double tptime = 0.;
            for (auto& sp : stPacks) {
                double ptime = min(sp->timer.get_elapsed_secs(), ctime - tptime);
                if (ptime > 0.) {
                    os <<
                        "  pack '" << sp->get_name() << "' (sec):      " << makeNumStr(ptime);
                    print_pct(os, ptime, ctime);
                    tptime += ptime;
                }
            }
            double optime = max(ctime - tptime, 0.);
            os <<
                "  other (sec):                      " << makeNumStr(optime);
            print_pct(os, optime, ctime);
#ifdef USE_MPI
            os <<
                " Compute-time breakdown by halo area:\n"
                "  rank-exterior compute (sec):      " << makeNumStr(etime);
            print_pct(os, etime, ctime);
            os <<
                "  rank-interior compute (sec):      " << makeNumStr(itime);
            print_pct(os, itime, ctime);
            os <<
                " Halo-time breakdown:\n"
                "  MPI waits (sec):                  " << makeNumStr(wtime);
            print_pct(os, wtime, htime);
            double ohtime = max(htime - wtime, 0.);
            os <<
                "  packing, unpacking, etc. (sec):   " << makeNumStr(ohtime);
            print_pct(os, ohtime, htime);
#endif
            os <<
                " throughput (num-writes/sec):      " << makeNumStr(writes_ps) << endl <<
                " throughput (est-FLOPS):           " << makeNumStr(flops) << endl <<
                " throughput (num-points/sec):      " << makeNumStr(domain_pts_ps) << endl;
        }

        // Fill in return object.
        auto p = make_shared<Stats>();
        p->npts = tot_domain_1t;
        p->nwrites = tot_numWrites_1t;
        p->nfpops = tot_numFpOps_1t;
        p->nsteps = steps_done;
        p->run_time = rtime;
        p->halo_time = htime;

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

    // Exchange dirty halo data for all grids and all steps.
    void StencilContext::exchange_halos(bool test_only) {

#ifdef USE_MPI
        if (!enable_halo_exchange || _env->num_ranks < 2)
            return;

        halo_time.start();
        double wait_delta = 0.;
        TRACE_MSG("exchange_halos");
        if (test_only)
            TRACE_MSG(" testing only");
        else {
            if (do_mpi_exterior)
                TRACE_MSG(" following calc of MPI exterior");
            if (do_mpi_interior)
                TRACE_MSG(" following calc of MPI interior");
        }
        auto opts = get_settings();
        auto& sd = _dims->_step_dim;

        // Vars for list of grids that need to be swapped and their step indices.
        GridPtrMap gridsToSwap;
        map<string, vector_set<idx_t>> stepsToSwap;
        int num_swaps = 0;
        size_t max_steps = 0;

        // TODO: move this into a separate function.
        if (test_only) {
            int num_tests = 0;

            // Call MPI_Test() on all unfinished requests to promote MPI progress.
            // TODO: replace with more direct and less intrusive techniques.
            
            // Loop thru MPI data.
            for (auto& mdi : mpiData) {
                auto& gname = mdi.first;
                auto& grid_mpi_data = mdi.second;
                MPI_Request* grid_recv_reqs = grid_mpi_data.recv_reqs.data();
                MPI_Request* grid_send_reqs = grid_mpi_data.send_reqs.data();

                int flag;
                for (size_t i = 0; i < grid_mpi_data.recv_reqs.size(); i++) {
                    auto& r = grid_recv_reqs[i];
                    if (r != MPI_REQUEST_NULL) {
                        //TRACE_MSG(gname << " recv test &MPI_Request = " << &r);
                        MPI_Test(&r, &flag, MPI_STATUS_IGNORE);
                        num_tests++;
                    }
                }
                for (size_t i = 0; i < grid_mpi_data.send_reqs.size(); i++) {
                    auto& r = grid_send_reqs[i];
                    if (r != MPI_REQUEST_NULL) {
                        //TRACE_MSG(gname << " send test &MPI_Request = " << &r);
                        MPI_Test(&r, &flag, MPI_STATUS_IGNORE);
                        num_tests++;
                    }
                }
            }
            TRACE_MSG("exchange_halos: " << num_tests << " MPI test(s) issued");
        }

        else {

            // Loop thru all bundle packs.
            // TODO: do this only once per step.
            // TODO: expand this to hold misc indices also.  Use an ordered map
            // by name to make sure grids are in same order on all ranks.
            for (auto& bp : stPacks) {

                // Loop thru stencil bundles in this pack.
                for (auto* sg : *bp) {

                    // Find the bundles that need to be processed.
                    // This will be any prerequisite scratch-grid
                    // bundles plus this non-scratch bundle.
                    // We need to loop thru the scratch-grid
                    // bundles so we can consider the inputs
                    // to them for exchanges.
                    auto sg_list = sg->get_reqd_bundles();

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

                            // Only need to swap grids that have any MPI buffers.
                            auto& gname = gp->get_name();
                            if (mpiData.count(gname) == 0)
                                continue;

                            // Check all allocated step indices.
                            idx_t start = 0, stop = 1;
                            if (gp->is_dim_used(sd)) {
                                start = min(start, gp->_get_first_alloc_index(sd));
                                stop = max(stop, gp->_get_last_alloc_index(sd) + 1);
                            }
                            for (idx_t t = start; t < stop; t++) {
                            
                                // Only need to swap grids whose halos are not up-to-date
                                // for this step.
                                if (!gp->is_dirty(t))
                                    continue;

                                // Swap this grid.
                                gridsToSwap[gname] = gp;
                                stepsToSwap[gname].insert(t);
                                num_swaps++;
                                max_steps = max(max_steps, stepsToSwap[gname].size());

                                // Cannot swap >1 step if overlapping comms/calc
                                // because we only have one step buffer per grid.
                                // TODO: fix this.
                                if (!do_mpi_exterior || !do_mpi_interior)
                                    assert(stepsToSwap[gname].size() == 1);

                            } // steps.
                        } // grids.
                    } // needed bundles.
                } // bundles in pack.
            } // packs.
            TRACE_MSG("exchange_halos: need to exchange halos for " <<
                      num_swaps << " steps(s) in " <<
                      gridsToSwap.size() << " grid(s)");
            assert(gridsToSwap.size() == stepsToSwap.size());
        }

        // Loop thru step-vector indices.
        // This loop is outside because we only have one buffer
        // per grid. Thus, we have to complete comms before
        // transerring another step. TODO: fix this.
        for (size_t svi = 0; svi < max_steps; svi++) {

            // Sequence of things to do for each grid's neighbors.
            enum halo_steps { halo_irecv, halo_pack_isend, halo_unpack, halo_final };
            vector<halo_steps> steps_to_do;

            // Flags indicate what part of grids were most recently calc'd.
            if (do_mpi_exterior) {
                steps_to_do.push_back(halo_irecv);
                steps_to_do.push_back(halo_pack_isend);
            }
            if (do_mpi_interior) {
                steps_to_do.push_back(halo_unpack);
                steps_to_do.push_back(halo_final);
            }
            int num_send_reqs = 0;
            int num_recv_reqs = 0;
            for (auto halo_step : steps_to_do) {

                if (halo_step == halo_irecv)
                    TRACE_MSG("exchange_halos: requesting data phase");
                else if (halo_step == halo_pack_isend)
                    TRACE_MSG("exchange_halos: packing and sending data phase");
                else if (halo_step == halo_unpack)
                    TRACE_MSG("exchange_halos: waiting for and unpacking data phase");
                else if (halo_step == halo_final)
                    TRACE_MSG("exchange_halos: waiting for send to finish phase");
                else
                    THROW_YASK_EXCEPTION("internal error: unknown halo-exchange step");

                // Loop thru all grids to swap.
                // Use 'gi' as an MPI tag.
                int gi = 0;
                for (auto gtsi : gridsToSwap) {
                    gi++;
                    auto& gname = gtsi.first;
                    auto gp = gtsi.second;
                    auto& grid_mpi_data = mpiData.at(gname);
                    MPI_Request* grid_recv_reqs = grid_mpi_data.recv_reqs.data();
                    MPI_Request* grid_send_reqs = grid_mpi_data.send_reqs.data();

                    // Get needed step in this grid.
                    auto& steps = stepsToSwap[gname];
                    if (steps.size() <= svi)
                        continue; // no step at this index.
                    idx_t si = steps.at(svi);
                    TRACE_MSG(" for grid '" << gname << "' w/step-index " << si);

                    // Loop thru all this rank's neighbors.
                    grid_mpi_data.visitNeighbors
                        ([&](const IdxTuple& offsets, // NeighborOffset.
                             int neighbor_rank,
                             int ni, // unique neighbor index.
                             MPIBufs& bufs) {
                            auto& sendBuf = bufs.bufs[MPIBufs::bufSend];
                            auto& recvBuf = bufs.bufs[MPIBufs::bufRecv];
                            TRACE_MSG("  with rank " << neighbor_rank << " at relative position " <<
                                      offsets.subElements(1).makeDimValOffsetStr());

                            // Submit async request to receive data from neighbor.
                            if (halo_step == halo_irecv) {
                                auto nbytes = recvBuf.get_bytes();
                                if (nbytes) {
                                    void* buf = (void*)recvBuf._elems;
                                    TRACE_MSG("   requesting " << makeByteStr(nbytes));
                                    auto& r = grid_recv_reqs[ni];
                                    //TRACE_MSG(gname << " Irecv &MPI_Request = " << &r);
                                    MPI_Irecv(buf, nbytes, MPI_BYTE,
                                              neighbor_rank, int(gi),
                                              _env->comm, &r);
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
                                    bool send_vec_ok = allow_vec_exchange && sendBuf.vec_copy_ok;

                                    // Get first and last ranges.
                                    IdxTuple first = sendBuf.begin_pt;
                                    IdxTuple last = sendBuf.last_pt;

                                    // The code in allocMpiData() pre-calculated the first and
                                    // last points of each buffer, except in the step dim.
                                    // So, we need to set that value now.
                                    // TODO: update this if we expand the buffers to hold
                                    // more than one step.
                                    if (gp->is_dim_used(sd)) {
                                        first.setVal(sd, si);
                                        last.setVal(sd, si);
                                    }
                                    TRACE_MSG("   packing " << sendBuf.num_pts.makeDimValStr(" * ") <<
                                              " points from [" << first.makeDimValStr() <<
                                              " ... " << last.makeDimValStr() << ") " <<
                                              (send_vec_ok ? "with" : "without") <<
                                              " vector copy");

                                    // Copy (pack) data from grid to buffer.
                                    void* buf = (void*)sendBuf._elems;
                                    if (send_vec_ok)
                                        gp->get_vecs_in_slice(buf, first, last);
                                    else
                                        gp->get_elements_in_slice(buf, first, last);

                                    // Send packed buffer to neighbor.
                                    auto nbytes = sendBuf.get_bytes();
                                    TRACE_MSG("   sending " << makeByteStr(nbytes));
                                    auto& r = grid_send_reqs[ni];
                                    //TRACE_MSG(gname << " Isend &MPI_Request = " << &r);
                                    MPI_Isend(buf, nbytes, MPI_BYTE,
                                              neighbor_rank, int(gi), _env->comm, &r);
                                    num_send_reqs++;
                                }
                                else
                                    TRACE_MSG("   0B to send");
                            }

                            // Wait for data from neighbor, then unpack it.
                            else if (halo_step == halo_unpack) {
                                auto nbytes = recvBuf.get_bytes();
                                if (nbytes) {

                                    // Wait for data from neighbor before unpacking it.
                                    auto& r = grid_recv_reqs[ni];
                                    //TRACE_MSG(gname << " recv wait &MPI_Request = " << &r);
                                    if (r != MPI_REQUEST_NULL) {
                                        TRACE_MSG("   waiting for receipt of " << makeByteStr(nbytes));
                                        wait_time.start();
                                        MPI_Wait(&r, MPI_STATUS_IGNORE);
                                        wait_delta += wait_time.stop();
                                    }

                                    // Vec ok?
                                    bool recv_vec_ok = allow_vec_exchange && recvBuf.vec_copy_ok;

                                    // Get first and last ranges.
                                    IdxTuple first = recvBuf.begin_pt;
                                    IdxTuple last = recvBuf.last_pt;

                                    // Set step val as above.
                                    if (gp->is_dim_used(sd)) {
                                        first.setVal(sd, si);
                                        last.setVal(sd, si);
                                    }
                                    TRACE_MSG("   got data; unpacking " << recvBuf.num_pts.makeDimValStr(" * ") <<
                                              " points into [" << first.makeDimValStr() <<
                                              " ... " << last.makeDimValStr() << ") " <<
                                              (recv_vec_ok ? "with" : "without") <<
                                              " vector copy");

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

                            // Final steps.
                            else if (halo_step == halo_final) {
                                auto nbytes = sendBuf.get_bytes();
                                if (nbytes) {

                                    // Wait for send to finish.
                                    // TODO: consider using MPI_WaitAll.
                                    // TODO: strictly, we don't have to wait on the
                                    // send to finish until we want to reuse this buffer,
                                    // so we could wait on the *previous* send right before
                                    // doing another one.
                                    auto& r = grid_send_reqs[ni];
                                    //TRACE_MSG(gname << " send wait &MPI_Request = " << &r);
                                    if (r != MPI_REQUEST_NULL) {
                                        TRACE_MSG("   waiting to finish send of " << makeByteStr(nbytes));
                                        wait_time.start();
                                        MPI_Wait(&grid_send_reqs[ni], MPI_STATUS_IGNORE);
                                        wait_delta += wait_time.stop();
                                    }
                                }

                                // Mark grids as up-to-date when done.
                                if (gp->is_dirty(si)) {
                                    gp->set_dirty(false, si);
                                    TRACE_MSG("grid '" << gname <<
                                              "' marked as clean at step-index " << si);
                                }
                            }
                            
                        }); // visit neighbors.

                } // grids.
            } // exchange sequence.

            TRACE_MSG("exchange_halos: " << num_recv_reqs << " MPI receive request(s) issued");
            TRACE_MSG("exchange_halos: " << num_send_reqs << " MPI send request(s) issued");

        } // step indices.

        auto mpi_call_time = halo_time.stop();
        TRACE_MSG("exchange_halos: secs spent in MPI waits: " << makeNumStr(wait_delta));
        TRACE_MSG("exchange_halos: secs spent in this call: " << makeNumStr(mpi_call_time));
#endif
    }

    // Mark grids that have been written to by bundle pack 'sel_bp'.
    // TODO: only mark grids that are written to in their halo-read area.
    // TODO: add index for misc dim(s).
    // TODO: track sub-domain of grid that is dirty.
    void StencilContext::mark_grids_dirty(const BundlePackPtr& sel_bp,
                                          idx_t start, idx_t stop) {
        idx_t step = (start > stop) ? -1 : 1;
        map<YkGridPtr, set<idx_t>> grids_done;

        // Stencil bundle packs.
        for (auto& bp : stPacks) {

            // Not a selected bundle pack?
            if (sel_bp && sel_bp != bp)
                continue;

            // Each input step.
            for (idx_t t = start; t != stop; t += step) {

                // Each bundle in this pack.
                for (auto* sb : *bp) {

                    // Get output step for this bundle, if any.
                    // For many stencils, this will be t+1 or
                    // t-1 if stepping backward.
                    idx_t t_out = 0;
                    if (!sb->get_output_step_index(t, t_out))
                        continue;

                    // Output grids for this bundle.  NB: don't need to mark
                    // scratch grids as dirty because they are never exchanged.
                    for (auto gp : sb->outputGridPtrs) {

                        // Mark output step as dirty if not already done.
                        if (grids_done[gp].count(t_out) == 0) {
                            gp->set_dirty(true, t_out);
                            TRACE_MSG("grid '" << gp->get_name() <<
                                      "' marked as dirty at step " << t_out);
                            grids_done[gp].insert(t_out);
                        }
                    }
                }
            }
        }
    }

    // Reset elapsed times to zero.
    void StencilContext::clear_timers() {
        run_time.clear();
        ext_time.clear();
        int_time.clear();
        halo_time.clear();
        wait_time.clear();
        steps_done = 0;
        for (auto& sp : stPacks)
            sp->timer.clear();
    }

    
} // namespace yask.
