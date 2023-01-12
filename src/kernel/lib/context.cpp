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

// This file contains implementations of some StencilContext methods.
// Also see setup.cpp, halo.cpp, and soln_apis.cpp.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    ///// Top-level methods for evaluating reference and optimized stencils.

    // Set the core vars that are needed for running kernels.
    void CommonCoreData::set_core(const StencilContext *cxt) {
        STATE_VARS_CONST(cxt);
        _global_sizes.set_from_tuple(actl_opts->_global_sizes);
        _rank_sizes.set_from_tuple(actl_opts->_rank_sizes);
        _rank_domain_offsets = cxt->rank_domain_offsets;
    }
    
    // Eval stencil bundle(s) over var(s) using reference scalar code.
    // Does NOT offload computations.
    void StencilContext::run_ref(idx_t first_step_index,
                                 idx_t last_step_index) {
        STATE_VARS(this);
        run_time.start();

        // Since any APIs may have been called in other ranks, mark all
        // neighbor vars as possibly dirty.
        set_all_neighbor_vars_dirty();

        // Disable offload.
        bool save_offload = KernelEnv::_use_offload;
        KernelEnv::_use_offload = false;

        // Determine step dir from order of first/last.
        idx_t step_dir = (last_step_index >= first_step_index) ? 1 : -1;

        // Find begin, stride and end in step-dim.
        idx_t begin_t = first_step_index;
        idx_t stride_t = step_dir; // always +/- 1 for ref run.
        assert(stride_t);
        idx_t end_t = last_step_index + step_dir; // end is beyond last.

        // Begin & end tuples.
        // Based on rank bounding box, not extended
        // BB because we don't use wave-fronts in the ref code.
        IdxTuple begin(stencil_dims);
        begin.set_vals(rank_bb.bb_begin_tuple(domain_dims), false); // 'false' because dims aren't same.
        begin[step_dim] = begin_t;
        IdxTuple end(stencil_dims);
        end.set_vals(rank_bb.bb_end_tuple(domain_dims), false);
        end[step_dim] = end_t;

        TRACE_MSG("run_ref: [" << begin.make_dim_val_str() << " ... " <<
                  end.make_dim_val_str() << ")");

        // Force sub-sizes to whole rank size so that scratch
        // vars will be large enough. Turn off any temporal blocking.
        actl_opts->_mega_block_sizes = actl_opts->_rank_sizes;
        actl_opts->_block_sizes = actl_opts->_rank_sizes;
        actl_opts->_micro_block_sizes = actl_opts->_rank_sizes;
        actl_opts->_nano_block_sizes = actl_opts->_rank_sizes;
        actl_opts->_pico_block_sizes = actl_opts->_rank_sizes;
        actl_opts->adjust_settings(); // Don't print settings.
        update_var_info(true);

        // Realloc scratch vars.
        alloc_scratch_data();

        // Indices to loop through.
        // Init from begin & end tuples.
        ScanIndices rank_idxs(false, &rank_domain_offsets);
        rank_idxs.begin = begin;
        rank_idxs.end = end;

        // Use only one set of scratch vars, i.e.,
        // we don't have one for each thread.
        int scratch_var_idx = 0;

        // Set offsets in scratch vars.  For this reference run, scratch
        // vars are allocated for the whole rank instead of smaller var
        // size.
        update_scratch_var_info(scratch_var_idx, rank_idxs.begin);

        // Doing all parts.
        MpiSection mpisec(this);
        mpisec.init();
        
        // Initial halo exchange.
        // TODO: get rid of all halo exchanges in this function,
        // and calculate overall problem in one rank.
        exchange_halos(mpisec);

        // Number of iterations to get from begin_t, stopping before end_t,
        // jumping by stride_t.
        const idx_t num_t = abs(end_t - begin_t);
        for (idx_t index_t = 0; index_t < num_t; index_t++)
        {
            // This value of index_t steps from start_t to stop_t-1.
            const idx_t start_t = begin_t + (index_t * stride_t);
            const idx_t stop_t = (stride_t > 0) ?
                min(start_t + stride_t, end_t) :
                max(start_t + stride_t, end_t);

            // Set indices that will pass through generated code
            // because the step loop is coded here.
            rank_idxs.index[step_posn] = index_t;
            rank_idxs.start[step_posn] = start_t;
            rank_idxs.stop[step_posn] = stop_t;
            rank_idxs.stride[step_posn] = stride_t;

            // Loop thru bundles. We ignore stages here
            // because staging is an optional optimization.
            for (auto* asg : st_bundles) {

                // Scan through n-D space.
                TRACE_MSG("run_ref: step " << start_t <<
                          " in non-scratch bundle '" << asg->get_name());

                // Check step.
                if (check_step_conds && !asg->is_in_valid_step(start_t)) {
                    TRACE_MSG("run_ref: not valid for step " << start_t);
                    continue;
                }

                // Exchange all dirty halos.
                exchange_halos(mpisec);

                // Find the bundles that need to be processed.
                // This will be the prerequisite scratch-var
                // bundles plus this non-scratch tile.
                auto sg_list = asg->get_reqd_bundles();

                // Loop through all the needed bundles.
                for (auto* sg : sg_list) {

                    // Indices needed for the generated misc loops.  Will normally be a
                    // copy of rank_idxs except when updating scratch-vars.
                    ScanIndices misc_idxs = sg->adjust_span(scratch_var_idx, rank_idxs);
                    misc_idxs.stride.set_from_const(1); // ensure unit stride.

                    // Scan through n-D space.
                    TRACE_MSG("run_ref: step " << start_t <<
                              " in bundle '" << sg->get_name() << "': [" <<
                              misc_idxs.begin.make_val_str() <<
                              " ... " << misc_idxs.end.make_val_str() << ")");
                    sg->calc_in_domain(scratch_var_idx, misc_idxs);

                } // needed bundles.

                // Mark vars that were updated in this rank.
                asg->update_var_info(YkVarBase::self, start_t, true, false, false);

                // Mark vars that *may* have been written to by any rank.
                update_var_info(nullptr, start_t, stop_t, true, false);

           } // all bundles.

         } // iterations.
        steps_done += abs(end_t - begin_t);

        // Final halo exchange.
        exchange_halos(mpisec);

        run_time.stop();

        // Restore offload setting.
        KernelEnv::_use_offload = save_offload;

    } // run_ref.

    // Eval stage(s) over var(s) using optimized code.
    void StencilContext::run_solution(idx_t first_step_index,
                                      idx_t last_step_index)
    {
        TRACE_MSG("running steps " << first_step_index << " ... " << last_step_index);
        STATE_VARS(this);

        // User-provided code.
        call_2idx_hooks(_before_run_solution_hooks,
                        first_step_index, last_step_index);
        // Start main timer.
        run_time.start();

        // Since any APIs may have been called in other ranks, mark all
        // neighbor vars as possibly dirty.
        set_all_neighbor_vars_dirty();

        // Determine step dir from order of first/last.
        idx_t step_dir = (last_step_index >= first_step_index) ? 1 : -1;

        // Find begin, stride and end in step-dim.
        idx_t begin_t = first_step_index;

        // Stride-size in step-dim is number of mega-block steps.
        // Then, it is multipled by +/-1 to get proper direction.
        idx_t stride_t = max(wf_steps, idx_t(1)) * step_dir;
        assert(stride_t);
        idx_t end_t = last_step_index + step_dir; // end is beyond last.

        // Begin, end, stride tuples.
        // Based on overall bounding box, which includes
        // any needed extensions for wave-fronts.
        IdxTuple begin(stencil_dims);
        begin.set_vals(ext_bb.bb_begin_tuple(domain_dims), false);
        begin[step_posn] = begin_t;
        IdxTuple end(stencil_dims);
        end.set_vals(ext_bb.bb_end_tuple(domain_dims), false);
        end[step_posn] = end_t;
        IdxTuple stride(stencil_dims);
        stride.set_vals(actl_opts->_mega_block_sizes, false); // stride by mega-block sizes.
        stride[step_posn] = stride_t;

        TRACE_MSG("running area [" <<
                  begin.make_dim_val_str() << " ... " <<
                  end.make_dim_val_str() << ") by " <<
                  stride.make_dim_val_str());
        if (!is_prepared())
            THROW_YASK_EXCEPTION("run_solution() called without calling prepare_solution() first");
        if (ext_bb.bb_size < 1) {
            TRACE_MSG("nothing to do in solution");
        }
        else {

            // Copy vars to device as needed. The vars will be left updated
            // on the device but not on the host after this call. Thus, if
            // this function is called multiple times without accessing any
            // data on the host, this should only trigger copying on the
            // first call.
            copy_vars_to_device();

            #ifdef MODEL_CACHE
            if (env.my_rank != env.msg_rank)
                cache_model.disable();
            if (cache_model.is_enabled())
                os << "Modeling cache...\n";
            #endif

            // Adjust end points for overlapping mega-blocks due to wavefront angle.
            // For each subsequent time step in a mega-block, the spatial location
            // of each block evaluation is shifted by the angle for each
            // stage. So, the total shift in a mega-block is the angle * num
            // stages * num timesteps. This assumes all stages
            // are inter-dependent to find maximum extension. Actual required
            // size may be less, but this will just result in some calls to
            // calc_mega_block() that do nothing.
            //
            // Conceptually (showing 2 ranks in t and x dims):
            // -----------------------------  t = rt ------------------------------
            //   \   | \     \     \|  \   |    .    |   / |  \     \     \|  \   |
            //    \  |  \     \     |   \  |    .    |  / \|   \     \     |   \  |
            //     \ |r0 \  r1 \ r2 |\ r3\ |    .    | /r0 | r1 \  r2 \ r3 |\ r4\ |
            //      \|    \     \   | \   \|         |/    |\    \     \   | \   \|
            // ------------------------------ t = 0 -------------------------------
            //       |   rank 0     |      |         |     |   rank 1      |      |
            // x = begin[x]       end[x] end[x]  begin[x] begin[x]       end[x] end[x]
            //     (rank)        (rank) (ext)     (ext)    (rank)       (rank) (adj)
            //
            //                      |XXXXXX|         |XXXXX|  <- redundant calculations.
            // XXXXXX|  <- areas outside of outer ranks not calculated ->  |XXXXXXX
            //
            if (wf_steps > 0) {
                DOMAIN_VAR_LOOP_FAST(i, j) {

                    // The end should be adjusted only if an extension doesn't
                    // exist.  Extentions exist between ranks, so additional
                    // adjustments are only needed at the end of the right-most
                    // rank in each dim.  See "(adj)" in diagram above.
                    if (right_wf_exts[j] == 0)
                        end[i] += wf_shift_pts[j];
                }
            }

            // At this point, 'begin' and 'end' should describe the *max* range
            // needed in the domain for this rank for the first time step.  At
            // any subsequent time step, this max may be shifted for temporal
            // wavefronts or blocking. Also, for each time step, the *actual*
            // range will be adjusted as needed before any actual stencil
            // calculations are made.

            // Indices needed for the 'rank' loops.
            ScanIndices rank_idxs(true, &rank_domain_offsets);
            rank_idxs.begin = begin;
            rank_idxs.end = end;
            rank_idxs.stride = stride;
            rank_idxs.tile_size = actl_opts->_rank_tile_sizes;
            rank_idxs.adjust_from_settings(actl_opts->_rank_sizes,
                                           actl_opts->_rank_tile_sizes,
                                           actl_opts->_mega_block_sizes);
            TRACE_MSG("after adjustment for " << num_wf_shifts <<
                      " wave-front shift(s): [" <<
                      rank_idxs.begin.make_val_str() << " ... " <<
                      rank_idxs.end.make_val_str() << ") by " <<
                      rank_idxs.stride.make_val_str());
            
            // Make sure threads are set properly for a mega-block.
            set_num_outer_threads();

            // Initial halo exchange.
            MpiSection mpisec(this);
            exchange_halos(mpisec);

            // Number of iterations to get from begin_t to end_t-1,
            // jumping by stride_t.
            const idx_t num_t = CEIL_DIV(abs(end_t - begin_t), abs(stride_t));
            for (idx_t index_t = 0; index_t < num_t; index_t++)
            {
                // This value of index_t steps from start_t to stop_t-1.
                const idx_t start_t = begin_t + (index_t * stride_t);
                const idx_t stop_t = (stride_t > 0) ?
                    min(start_t + stride_t, end_t) :
                    max(start_t + stride_t, end_t);
                idx_t this_num_t = abs(stop_t - start_t);

                // Set indices that will pass through generated code.
                rank_idxs.index[step_posn] = index_t;
                rank_idxs.start[step_posn] = start_t;
                rank_idxs.stop[step_posn] = stop_t;
                rank_idxs.stride[step_posn] = stride_t;

                // Start timer for auto-tuner.
                _at.timer.start();

                // If no wave-fronts (default), loop through stages here, and do
                // only one stage at a time in calc_mega_block(). This is similar to
                // loop in calc_rank_ref(), but with stages instead of bundles.
                if (wf_steps == 0) {

                    // Loop thru stages.
                    for (auto& bp : st_stages) {

                        // Check step.
                        if (check_step_conds && !bp->is_in_valid_step(start_t)) {
                            TRACE_MSG("step " << start_t <<
                                      " not valid for stage '" <<
                                      bp->get_name() << "'");
                            continue;
                        }

                        // Do MPI-external parts separately?
                        if (mpi_interior.bb_valid) {
                            mpisec.do_mpi_interior = false;

                            // Overlap comms and computation by restricting
                            // mega-block boundaries.  Make an external pass for
                            // each side of each domain dim, e.g., 'left x',
                            // 'right x', 'left y', ...
                            DOMAIN_VAR_LOOP(i, j) {
                                for (bool is_left : { true, false }) {

                                    // Skip if no halo to calculate in this
                                    // section.
                                    if (!does_exterior_exist(j, is_left))
                                        continue;

                                    // Set the proper flags to indicate what
                                    // section we're working on.
                                    mpisec.do_mpi_left = is_left;
                                    mpisec.do_mpi_right = !is_left;
                                    mpisec.mpi_exterior_dim = j;
                                    assert(mpisec.is_overlap_active());

                                    // Include automatically-generated loop
                                    // code to call calc_mega_block() for
                                    // each mega-block. The mega-block will be trimmed
                                    // to the active MPI exterior section.
                                    TRACE_MSG("step " << start_t <<
                                              " for stage '" << bp->get_name() <<
                                              "' in MPI exterior " <<
                                              (is_left ? "left-" : "right-") <<
                                              domain_dims.get_dim_name(j));

                                    // Loop prefix.
                                    #define RANK_LOOP_INDICES rank_idxs
                                    #define RANK_BODY_INDICES mega_block_range
                                    #define RANK_USE_LOOP_PART_0
                                    #include "yask_rank_loops.hpp"

                                    // Loop body.
                                    calc_mega_block(bp, mega_block_range, mpisec);

                                    // Loop suffix.
                                    #define RANK_USE_LOOP_PART_1
                                    #include "yask_rank_loops.hpp"
                                    
                                } // left/right.
                            } // domain dims.

                            // Mark vars that *may* have been written to by
                            // this stage by any rank. Mark vars as dirty
                            // even if not actually written by this rank,
                            // perhaps due to sub-domains or asymmetrical
                            // stencils. This is needed because neighbors
                            // will not know what vars are actually dirty,
                            // and all ranks must have the same information
                            // about which vars are possibly dirty.
                            update_var_info(bp, start_t, stop_t, true);

                            // Do the appropriate steps for halo exchange of exterior.
                            mpisec.do_mpi_left = mpisec.do_mpi_right = true;
                            exchange_halos(mpisec);

                            // Do interior only in next pass.
                            mpisec.do_mpi_left = mpisec.do_mpi_right = false;
                            mpisec.do_mpi_interior = true;

                        } // Exterior only for overlapping comms.

                        // Include automatically-generated loop code to call
                        // calc_mega_block() for each mega-block. If overlapping
                        // comms, this will be just the interior.  If not, it
                        // will cover the whole rank.
                        TRACE_MSG("step " << start_t <<
                                  " for stage '" << bp->get_name() << "'");

                        // Loop prefix.
                        #define RANK_LOOP_INDICES rank_idxs
                        #define RANK_BODY_INDICES mega_block_range
                        #define RANK_USE_LOOP_PART_0
                        #include "yask_rank_loops.hpp"

                        // Loop body.
                        calc_mega_block(bp, mega_block_range, mpisec);

                        // Loop suffix.
                        #define RANK_USE_LOOP_PART_1
                        #include "yask_rank_loops.hpp"
 
                        // Mark as dirty only if we just did exterior.
                        bool mark_dirty = mpisec.do_mpi_left || mpisec.do_mpi_right;
                        update_var_info(bp, start_t, stop_t, mark_dirty);

                        // Do the appropriate steps for halo exchange depending
                        // on 'do_mpi_*' flags.
                        exchange_halos(mpisec);

                        // Set the overlap flags back to default.
                        mpisec.init();

                    } // stages.
                } // No WF tiling.

                // If doing wave-fronts, must loop through all stages in
                // calc_mega_block().
                else {

                    // Null ptr => Eval all stages each time
                    // calc_mega_block() is called.
                    StagePtr bp;

                    // Do MPI-external parts separately?
                    if (mpi_interior.bb_valid) {
                        mpisec.do_mpi_interior = false;

                        // Overlap comms and computation by restricting
                        // mega-block boundaries.  Make an external pass for
                        // each side of each domain dim, e.g., 'left x',
                        // 'right x', 'left y', ...
                        DOMAIN_VAR_LOOP(i, j) {
                            for (bool is_left : { true, false }) {

                                // Skip if no halo to calculate in this
                                // section.
                                if (!does_exterior_exist(j, is_left))
                                    continue;

                                // Set the proper flags to indicate what
                                // section we're working on.
                                mpisec.do_mpi_left = is_left;
                                mpisec.do_mpi_right = !is_left;
                                mpisec.mpi_exterior_dim = j;
                                assert(mpisec.is_overlap_active());

                                // Include automatically-generated loop
                                // code to call calc_mega_block(bp) for
                                // each mega-block. The mega-block will be trimmed
                                // to the active MPI exterior section.
                                TRACE_MSG("WF steps [" << start_t <<
                                          " ... " << stop_t <<
                                          ") in MPI exterior " <<
                                          (is_left ? "left-" : "right-") <<
                                          domain_dims.get_dim_name(j));

                                // Loop prefix.
                                #define RANK_LOOP_INDICES rank_idxs
                                #define RANK_BODY_INDICES mega_block_range
                                #define RANK_USE_LOOP_PART_0
                                #include "yask_rank_loops.hpp"

                                // Loop body.
                                calc_mega_block(bp, mega_block_range, mpisec);

                                // Loop suffix.
                                #define RANK_USE_LOOP_PART_1
                                #include "yask_rank_loops.hpp"
                                
                            } // left/right.
                        } // domain dims.

                        // Mark vars dirty for all stages.
                        update_var_info(bp, start_t, stop_t, true);

                        // Do the appropriate steps for halo exchange of exterior.
                        mpisec.do_mpi_left = mpisec.do_mpi_right = true;
                        exchange_halos(mpisec);

                        // Do interior only in next pass.
                        mpisec.do_mpi_left = mpisec.do_mpi_right = false;
                        mpisec.do_mpi_interior = true;

                    } // Exterior only for overlapping comms.

                    // Include automatically-generated loop code to call
                    // calc_mega_block() for each mega-block. If overlapping
                    // comms, this will be just the interior.  If not, it
                    // will cover the whole rank.
                    TRACE_MSG("steps [" << start_t <<
                              " ... " << stop_t << ")");

                    // Loop prefix.
                    #define RANK_LOOP_INDICES rank_idxs
                    #define RANK_BODY_INDICES mega_block_range
                    #define RANK_USE_LOOP_PART_0
                    #include "yask_rank_loops.hpp"

                    // Loop body.
                    calc_mega_block(bp, mega_block_range, mpisec);
                    
                    // Loop suffix.
                    #define RANK_USE_LOOP_PART_1
                    #include "yask_rank_loops.hpp"

                    // Mark as dirty only if we just did exterior.
                    bool mark_dirty = mpisec.do_mpi_left || mpisec.do_mpi_right;
                    update_var_info(bp, start_t, stop_t, mark_dirty);

                    // Do the appropriate steps for halo exchange depending
                    // on 'do_mpi_*' flags.
                    exchange_halos(mpisec);

                    // Set the overlap flags back to default.
                    mpisec.init();

                } // With WF tiling.

                // Overall steps.
                steps_done += this_num_t;

                // Count steps for each stage to properly account for
                // step conditions when using temporal tiling.
                for (auto& bp : st_stages) {
                    idx_t num_stage_steps = 0;

                    if (!check_step_conds)
                        num_stage_steps = this_num_t;
                    else {

                        // Loop through each step.
                        assert(abs(step_dir) == 1);
                        for (idx_t t = start_t; t != stop_t; t += step_dir) {

                            // Check step cond for this t.
                            if (bp->is_in_valid_step(t))
                                num_stage_steps++;
                        }
                    }

                    // Add to steps done for this stage.
                    bp->add_steps(num_stage_steps);
                }

                // Call the auto-tuner to evaluate these steps and change
                // settings when enough time has passed.
                // FIXME: in-situ AT will not work properly with temporal conditions
                // because not all sequences of N steps will do the same amount of work.
                auto this_time = _at.timer.stop();
                _at.steps_done += this_num_t;
                eval_auto_tuner();
                TRACE_MSG("did " << this_num_t << " step(s) in " << this_time << " secs.");

            } // step loop.

            #ifdef MODEL_CACHE
            // Print cache stats, then disable.
            // Thus, cache is only modeled for first call.
            if (cache_model.is_enabled()) {
                os << "Done modeling cache...\n";
                cache_model.dump_stats();
                cache_model.disable();
            }
            #endif

        } // Something to do.
        
        // Stop timer.
        run_time.stop();

        // User-provided code.
        call_2idx_hooks(_after_run_solution_hooks,
                        first_step_index, last_step_index);

    } // run_solution().

    // Calculate results within a mega-block.  Each mega-block is typically computed
    // via a separate OpenMP 'for' region.  In this function, we loop over
    // the time steps and stages and evaluate a stage in each of
    // the blocks in the mega-block.  If 'sel_bp' is null, eval all stages; else
    // eval only the one pointed to.
    void StencilContext::calc_mega_block(StagePtr& sel_bp,
                                         const ScanIndices& rank_idxs,
                                         MpiSection& mpisec) {
        STATE_VARS(this);
        TRACE_MSG("calc_mega_block: mega-block [" <<
                  rank_idxs.start.make_val_str() << " ... " <<
                  rank_idxs.stop.make_val_str() << ") within possibly-adjusted rank [" <<
                  rank_idxs.begin.make_val_str() << " ... " <<
                  rank_idxs.end.make_val_str() << ") for " <<
                  mpisec.make_descr());

        // Track time separately for MPI exterior and interior.
        if (mpisec.is_exterior_active())
            ext_time.start();
        else
            int_time.start();

        // Init mega-block begin & end from rank start & stop indices.
        ScanIndices mega_block_idxs = rank_idxs.create_inner();

        // Time range.
        // When doing WF rank tiling, this loop will stride through
        // several time-steps in each mega-block.
        // When also doing TB, it will stride by the block strides.
        idx_t begin_t = mega_block_idxs.begin[step_posn];
        idx_t end_t = mega_block_idxs.end[step_posn];
        idx_t step_dir = (end_t >= begin_t) ? 1 : -1;
        idx_t stride_t = max(tb_steps, idx_t(1)) * step_dir;
        assert(stride_t);
        const idx_t num_t = CEIL_DIV(abs(end_t - begin_t), abs(stride_t));

        // Time loop.
        idx_t mega_block_shift_num = 0;
        for (idx_t index_t = 0; index_t < num_t; index_t++) {

            // This value of index_t steps from start_t to stop_t-1.
            // Be sure to handle reverse steps.
            const idx_t start_t = begin_t + (index_t * stride_t);
            const idx_t stop_t = (stride_t > 0) ?
                min(start_t + stride_t, end_t) :
                max(start_t + stride_t, end_t);

            // Set step indices that will pass through generated code.
            mega_block_idxs.index[step_posn] = index_t;
            mega_block_idxs.start[step_posn] = start_t;
            mega_block_idxs.stop[step_posn] = stop_t;

            // If no temporal blocking (default), loop through stages here,
            // and do only one stage at a time in calc_block(). If there is
            // no WF blocking either, the stage loop body will only execute
            // with one active stage, and 'mega_block_shift_num' will never be > 0.
            if (tb_steps == 0) {

                // Stages to evaluate at this time step.
                for (auto& bp : st_stages) {

                    // Not a selected stage?
                    if (sel_bp && sel_bp != bp)
                        continue;

                    TRACE_MSG("no TB; stage '" <<
                              bp->get_name() << "' in step(s) [" <<
                              start_t << " ... " << stop_t << ")");

                    // Check step.
                    if (check_step_conds && !bp->is_in_valid_step(start_t)) {
                        TRACE_MSG("step " << start_t <<
                                  " not valid for stage '" << bp->get_name() << "'");
                        continue;
                    }

                    // Strides within a mega-block are based on block sizes.
                    mega_block_idxs.stride = actl_opts->_block_sizes;
                    mega_block_idxs.stride[step_posn] = stride_t;

                    // Tiles in mega-block loops.
                    mega_block_idxs.tile_size = actl_opts->_mega_block_tile_sizes;
                    
                    // Set mega_block_idxs begin & end based on shifted rank
                    // start & stop (original mega-block begin & end), rank
                    // boundaries, and stage BB. This will be the base of the
                    // mega-block loops.
                    bool ok = shift_mega_block(rank_idxs.start, rank_idxs.stop,
                                               mega_block_shift_num, bp,
                                               mega_block_idxs, mpisec);
                    mega_block_idxs.adjust_from_settings(actl_opts->_mega_block_sizes,
                                                         actl_opts->_mega_block_tile_sizes,
                                                         actl_opts->_block_sizes);

                    // Only need to loop through the span of the mega-block if it is
                    // at least partly inside the extended BB. For overlapping
                    // mega-blocks, they may start outside the domain but enter the
                    // domain as time progresses and their boundaries shift. So,
                    // we don't want to return if this condition isn't met.
                    if (ok) {
                        idx_t nphases = 1; // Only 1 phase w/o TB.
                        idx_t phase = 0;

                        // Include automatically-generated loop code to
                        // call calc_block() for each block in this mega-block.
                        // Loops through x from begin_rx to end_rx-1;
                        // similar for y and z.  This code typically
                        // contains the outer OpenMP loop(s).

                        // Loop prefix.
                        #define MEGA_BLOCK_LOOP_INDICES mega_block_idxs
                        #define MEGA_BLOCK_BODY_INDICES blk_range
                        #define MEGA_BLOCK_USE_LOOP_PART_0
                        #include "yask_mega_block_loops.hpp"

                        // Loop body.
                        calc_block(bp, mega_block_shift_num, nphases, phase,
                                   rank_idxs, blk_range, mpisec);

                        // Loop suffix.
                        #define MEGA_BLOCK_USE_LOOP_PART_1
                        #include "yask_mega_block_loops.hpp"
                    }

                    // Need to shift for next stage and/or time.
                    mega_block_shift_num++;

                } // stages.
            } // no temporal blocking.

            // If using TB, iterate thru steps in a WF and stages in calc_block().
            else {

                TRACE_MSG("calc_mega_block: w/TB in step(s) [" <<
                          start_t << " ... " << stop_t << ")");

                // Null ptr => Eval all stages each time
                // calc_block() is called.
                StagePtr bp;

                // Strides within a mega-block are based on rank block sizes.
                // Cannot use different strides per stage with TB.
                auto& settings = *actl_opts;
                mega_block_idxs.stride = settings._block_sizes;
                mega_block_idxs.stride[step_posn] = stride_t;

                // Tiles in mega-block loops.
                mega_block_idxs.tile_size = settings._mega_block_tile_sizes;
                
                // Set mega_block_idxs begin & end based on shifted start & stop
                // and rank boundaries.  This will be the base of the mega-block
                // loops. The bounds in mega_block_idxs may be outside the
                // actual rank because we're starting with the expanded rank.
                bool ok = shift_mega_block(rank_idxs.start, rank_idxs.stop,
                                           mega_block_shift_num, bp,
                                           mega_block_idxs, mpisec);
                mega_block_idxs.adjust_from_settings(settings._mega_block_sizes,
                                                     settings._mega_block_tile_sizes,
                                                     settings._block_sizes);

                // Should always be valid because we just shifted (no trim).
                // Trimming will be done at the micro-block level.
                assert(ok);

                // To tesselate n-D domain space, we use n+1 distinct
                // "phases".  For example, 1-D TB uses "upward" trapezoids
                // and "downward" trapezoids. Outer OMP threads sync after
                // every phase. Thus, the phase loop is here around the
                // generated OMP loops.  TODO: schedule phases and their
                // shapes via task dependencies.
                idx_t nphases = nddims + 1;
                for (idx_t phase = 0; phase < nphases; phase++) {

                    // Call calc_block() on every block concurrently.  Only
                    // the shapes corresponding to the current 'phase' will
                    // be calculated.

                    // Loop prefix.
                    #define MEGA_BLOCK_LOOP_INDICES mega_block_idxs
                    #define MEGA_BLOCK_BODY_INDICES blk_range
                    #define MEGA_BLOCK_USE_LOOP_PART_0
                    #include "yask_mega_block_loops.hpp"

                    // Loop body.
                    calc_block(bp, mega_block_shift_num, nphases, phase,
                               rank_idxs, blk_range, mpisec);

                    // Loop suffix.
                    #define MEGA_BLOCK_USE_LOOP_PART_1
                    #include "yask_mega_block_loops.hpp"
                }

                // Loop thru stages that were evaluated in
                // these 'tb_steps' to increment shift for next mega-block
                // "layer", if any. This is needed when there are more WF
                // steps than TB steps.  TODO: consider moving this inside
                // calc_block().
                for (idx_t t = start_t; t != stop_t; t += step_dir) {
                    for (auto& bp : st_stages) {

                        // Check step.
                        if (check_step_conds && !bp->is_in_valid_step(t))
                            continue;

                        // One shift for each stage in each TB step.
                        mega_block_shift_num++;
                    }
                }
            } // with temporal blocking.
        } // time.

        if (mpisec.is_exterior_active()) {
            double ext_delta = ext_time.stop();
            TRACE_MSG("secs spent in this mega-block for rank-exterior blocks: " << make_num_str(ext_delta));
        }
        else {
            double int_delta = int_time.stop();
            TRACE_MSG("secs spent in this mega-block for rank-interior blocks: " << make_num_str(int_delta));
        }

    } // calc_mega_block.

    // Calculate results within a block. This function calls
    // 'calc_micro_block()' for the specified stage or all stages if 'sel_bp'
    // is null.  When using TB, only the shape(s) needed for the tesselation
    // 'phase' are computed.  Typically called by a top-level OMP thread
    // from calc_mega_block().
    void StencilContext::calc_block(StagePtr& sel_bp,
                                    idx_t mega_block_shift_num,
                                    idx_t nphases, idx_t phase,
                                    const ScanIndices& rank_idxs,
                                    const ScanIndices& mega_block_idxs,
                                    MpiSection& mpisec) {

        STATE_VARS(this);
        auto* bp = sel_bp.get();
        int outer_thread_idx = omp_get_thread_num();
        TRACE_MSG("calc_block: phase " << phase << ", block [" <<
                  mega_block_idxs.start.make_val_str() << " ... " <<
                  mega_block_idxs.stop.make_val_str() <<
                  ") within mega-block [" <<
                  mega_block_idxs.begin.make_val_str() << " ... " <<
                  mega_block_idxs.end.make_val_str() <<
                  ") by mega-block thread " << outer_thread_idx);

        // Init block begin & end from mega-block start & stop indices.
        ScanIndices block_idxs = mega_block_idxs.create_inner();

        // Time range.
        // When not doing TB, there is only one step.
        // When doing TB, we will only do one iteration here
        // that covers all steps,
        // and calc_micro_block() will loop over all steps.
        idx_t begin_t = block_idxs.begin[step_posn];
        idx_t end_t = block_idxs.end[step_posn];
        idx_t step_dir = (end_t >= begin_t) ? 1 : -1;
        idx_t stride_t = max(tb_steps, idx_t(1)) * step_dir;
        assert(stride_t);
        const idx_t num_t = CEIL_DIV(abs(end_t - begin_t), abs(stride_t));

        // If TB is not being used, just process the given stage.
        // No need for a time loop.
        // No need to check bounds, because they were checked in
        // calc_mega_block() when not using TB.
        if (tb_steps == 0) {
            assert(bp);
            assert(abs(stride_t) == 1);
            assert(abs(end_t - begin_t) == 1);
            assert(num_t == 1);

            // Set step indices that will pass through generated code.
            block_idxs.index[step_posn] = 0;
            block_idxs.start[step_posn] = begin_t;
            block_idxs.stop[step_posn] = end_t;

            // Strides within a block are based on micro-block sizes.
            block_idxs.stride = actl_opts->_micro_block_sizes;
            block_idxs.stride[step_posn] = stride_t;

            // Tiles in block loops.
            block_idxs.tile_size = actl_opts->_block_tile_sizes;

            // Default settings for no TB.
            StagePtr bp = sel_bp;
            assert(phase == 0);
            idx_t nshapes = 1;
            idx_t shape = 0;
            idx_t shift_num = 0;
            bit_mask_t bridge_mask = 0;
            ScanIndices adj_block_idxs = block_idxs;
            adj_block_idxs.adjust_from_settings(actl_opts->_block_sizes,
                                                actl_opts->_block_tile_sizes,
                                                actl_opts->_micro_block_sizes);

            // Include automatically-generated loop code to
            // call calc_micro_block() for each micro-block in this block.

            // Loop prefix.
            #define BLOCK_LOOP_INDICES adj_block_idxs
            #define BLOCK_BODY_INDICES micro_blk_range
            #define BLOCK_USE_LOOP_PART_0
            #include "yask_block_loops.hpp"

            // Loop body.
            calc_micro_block(outer_thread_idx, bp, mega_block_shift_num,
                             nphases, phase, nshapes, shape, bridge_mask,
                             rank_idxs, mega_block_idxs, block_idxs, micro_blk_range,
                             mpisec);
            
            // Loop suffix.
            #define BLOCK_USE_LOOP_PART_1
            #include "yask_block_loops.hpp"
        } // no TB.

        // If TB is active, loop thru each required shape.
        else {
            assert(phase >= 0);
            assert(phase < nphases); // E.g., phase = 0..2 for 2D.

            // Determine number of shapes for this 'phase'. First and last
            // phase need one shape. Other (bridge) phases need one shape
            // for each combination of domain dims. E.g., need 'x' and
            // 'y' bridges for 2D problem in phase 1.
            idx_t nshapes = n_choose_k(nddims, phase);
            bit_mask_t bridge_mask = 0;

            // Set temporal indices to full range.
            block_idxs.index[step_posn] = 0; // only one index.
            block_idxs.start[step_posn] = begin_t;
            block_idxs.stop[step_posn] = end_t;

            // Strides within a block are based on rank micro-block sizes.
            auto& settings = *actl_opts;
            block_idxs.stride = settings._micro_block_sizes;
            block_idxs.stride[step_posn] = step_dir;

            // Tiles in block loops.
            block_idxs.tile_size = settings._block_tile_sizes;

            // Increase range of block to cover all phases and
            // shapes.
            ScanIndices adj_block_idxs = block_idxs;
            DOMAIN_VAR_LOOP_FAST(i, j) {

                // TB shapes can extend to the right only.  They can
                // cover a range as big as this block's base plus the
                // next block in all dims, so we add the width of the
                // current block to the end.  This makes the adjusted
                // blocks overlap, but the size of each micro-block is
                // trimmed at each step to the proper active size.
                // TODO: find a way to make this more efficient to avoid
                // calling calc_micro_block() many times with nothing to
                // do.
                auto width = mega_block_idxs.stop[i] - mega_block_idxs.start[i];
                adj_block_idxs.end[i] += width;
            }
            adj_block_idxs.adjust_from_settings(settings._block_sizes,
                                                settings._block_tile_sizes,
                                                settings._micro_block_sizes);
            TRACE_MSG("calc_block: phase " << phase <<
                      ", adjusted block [" <<
                      adj_block_idxs.begin.make_val_str() << " ... " <<
                      adj_block_idxs.end.make_val_str() <<
                      ") with micro-block stride " <<
                      adj_block_idxs.stride.make_val_str());

            // Loop thru shapes.
            for (idx_t shape = 0; shape < nshapes; shape++) {

                // Get 'shape'th combo of 'phase' things from 'nddims'.
                // These will be used to create bridge shapes.
                bridge_mask = n_choose_k_set(nddims, phase, shape);

                // Can only be one time iteration here when doing TB
                // because micro-block temporal size is always same
                // as block temporal size.
                assert(num_t == 1);

                // Include automatically-generated loop code to call
                // calc_micro_block() for each micro-block in this block.
                StagePtr bp; // null.

                // Loop prefix.
                #define BLOCK_LOOP_INDICES adj_block_idxs
                #define BLOCK_BODY_INDICES micro_blk_range
                #define BLOCK_USE_LOOP_PART_0
                #include "yask_block_loops.hpp"

                // Loop body.
                calc_micro_block(outer_thread_idx, bp, mega_block_shift_num,
                                 nphases, phase, nshapes, shape, bridge_mask,
                                 rank_idxs, mega_block_idxs, block_idxs, micro_blk_range,
                                 mpisec);
            
                // Loop suffix.
                #define BLOCK_USE_LOOP_PART_1
                #include "yask_block_loops.hpp"
                
            } // shape loop.
        } // TB.
    } // calc_block().

    // Calculate results within a micro-block.
    // This function calls 'StencilBundleBase::calc_micro_block()'
    // for each bundle in the specified stage or all stages if 'sel_bp' is
    // null. When using TB, only the 'shape' needed for the tesselation
    // 'phase' are computed. The starting 'shift_num' is relative
    // to the bottom of the current mega-block and block.
    void StencilContext::calc_micro_block(int outer_thread_idx,
                                          StagePtr& sel_bp,
                                          idx_t mega_block_shift_num,
                                          idx_t nphases, idx_t phase,
                                          idx_t nshapes, idx_t shape,
                                          const bit_mask_t& bridge_mask,
                                          const ScanIndices& rank_idxs,
                                          const ScanIndices& base_mega_block_idxs,
                                          const ScanIndices& base_block_idxs,
                                          const ScanIndices& adj_block_idxs,
                                          MpiSection& mpisec) {

        STATE_VARS(this);
        TRACE_MSG("calc_micro_block: phase " << phase <<
                  ", shape " << shape <<
                  ", micro-block [" <<
                  adj_block_idxs.start.make_val_str() << " ... " <<
                  adj_block_idxs.stop.make_val_str() << ") within base-block [" <<
                  base_block_idxs.begin.make_val_str() << " ... " <<
                  base_block_idxs.end.make_val_str() << ") within base-mega-block [" <<
                  base_mega_block_idxs.begin.make_val_str() << " ... " <<
                  base_mega_block_idxs.end.make_val_str() <<
                  ") by mega-block thread " << outer_thread_idx);

        // Promote forward progress in MPI when calc'ing interior
        // only. Call from one thread only.
        // Let all other threads continue.
        if (mpisec.is_overlap_active() && mpisec.do_mpi_interior) {
            if (outer_thread_idx == 0)
                adv_halo_exchange();
        }

        // Init micro-block begin & end from blk start & stop indices.
        ScanIndices micro_block_idxs = adj_block_idxs.create_inner();

        // Time range.
        // No more temporal blocks below micro-blocks, so we always stride
        // by +/- 1.
        idx_t begin_t = micro_block_idxs.begin[step_posn];
        idx_t end_t = micro_block_idxs.end[step_posn];
        idx_t step_dir = (end_t >= begin_t) ? 1 : -1;
        idx_t stride_t = 1 * step_dir;        // +/- 1.
        assert(stride_t);
        const idx_t num_t = CEIL_DIV(abs(end_t - begin_t), abs(stride_t));

        // Time loop.
        idx_t shift_num = 0;
        for (idx_t index_t = 0; index_t < num_t; index_t++) {

            // This value of index_t steps from start_t to stop_t-1.
            const idx_t start_t = begin_t + (index_t * stride_t);
            const idx_t stop_t = (stride_t > 0) ?
                min(start_t + stride_t, end_t) :
                max(start_t + stride_t, end_t);
            TRACE_MSG("calc_micro_block: phase " << phase <<
                      ", shape " << shape <<
                      ", in step " << start_t);
            assert(abs(stop_t - start_t) == 1); // no more TB.

            // Set step indices that will pass through generated code.
            micro_block_idxs.index[step_posn] = index_t;
            micro_block_idxs.begin[step_posn] = start_t;
            micro_block_idxs.end[step_posn] = stop_t;
            micro_block_idxs.start[step_posn] = start_t;
            micro_block_idxs.stop[step_posn] = stop_t;

            // Stages to evaluate at this time step.
            for (auto& bp : st_stages) {

                // Not a selected stage?
                if (sel_bp && sel_bp != bp)
                    continue;

                // Check step.
                if (check_step_conds && !bp->is_in_valid_step(start_t)) {
                    TRACE_MSG("calc_micro_block: step " << start_t <<
                              " not valid for stage '" <<
                              bp->get_name() << "'");
                    continue;
                }
                TRACE_MSG("calc_micro_block: phase " << phase <<
                          ", shape " << shape <<
                          ", step " << start_t <<
                          ", stage '" << bp->get_name() <<
                          "', shift-num " << shift_num);

                // Start timers for this stage.  Tracking only on thread
                // 0. TODO: track all threads and report cross-thread stats.
                if (outer_thread_idx == 0)
                    bp->start_timers();

                // Strides within a micro-blk are based on nano-blk sizes.
                // This will get overridden later if thread binding is enabled.
                micro_block_idxs.stride = actl_opts->_nano_block_sizes;
                micro_block_idxs.stride[step_posn] = stride_t;

                // Tiles in micro-blk loops.
                micro_block_idxs.tile_size = actl_opts->_micro_block_tile_sizes;

                // Set micro_block_idxs begin & end based on shifted rank
                // start & stop (original mega-block begin & end), rank
                // boundaries, and stage BB. There may be several TB layers
                // within a mega-block WF, so we need to add the mega-block and
                // local micro-block shift counts.
                bool ok = shift_mega_block(rank_idxs.start, rank_idxs.stop,
                                           mega_block_shift_num + shift_num, bp,
                                           micro_block_idxs, mpisec);

                // Set micro_block_idxs begin & end based on shifted begin &
                // end of block for given phase & shape.  This will be the
                // base for the micro-block loops, which have no temporal
                // tiling.
                if (ok)
                    ok = shift_micro_block(adj_block_idxs.start, adj_block_idxs.stop,
                                           adj_block_idxs.begin, adj_block_idxs.end,
                                           base_block_idxs.begin, base_block_idxs.end,
                                           base_mega_block_idxs.begin, base_mega_block_idxs.end,
                                           shift_num,
                                           nphases, phase,
                                           nshapes, shape,
                                           bridge_mask,
                                           micro_block_idxs);

                if (ok) {
                    micro_block_idxs.adjust_from_settings(actl_opts->_micro_block_sizes,
                                                          actl_opts->_micro_block_tile_sizes,
                                                          actl_opts->_nano_block_sizes);

                    // Update offsets of scratch vars based on the current
                    // micro-block location.
                    if (scratch_vecs.size())
                        update_scratch_var_info(outer_thread_idx, micro_block_idxs.begin);

                    // Call calc_micro_block() for each non-scratch bundle.
                    for (auto* sb : *bp)
                        if (sb->get_bb().bb_num_points)
                            sb->calc_micro_block(outer_thread_idx, *actl_opts, micro_block_idxs, mpisec);

                    // Make sure streaming stores are visible for later loads.
                    make_stores_visible();
                }

                // Need to shift for next stage and/or time-step.
                shift_num++;

                // Stop timers for this stage.
                if (outer_thread_idx == 0)
                    bp->stop_timers();

            } // stages.
        } // time-steps.

    } // calc_micro_block().

    // Find boundaries within mega-block with 'base_start' to 'base_stop'
    // shifted 'shift_num' times, which should start at 0 and increment for
    // each stage in each time-step.  Trim to ext-BB and MPI section if 'bp' if
    // not null.  Write results into 'begin' and 'end' in 'idxs'.  Return
    // 'true' if resulting area is non-empty, 'false' if empty.
    bool StencilContext::shift_mega_block(const Indices& base_start,
                                          const Indices& base_stop,
                                          idx_t shift_num,
                                          StagePtr& bp,
                                          ScanIndices& idxs,
                                          const MpiSection& mpisec) {
        STATE_VARS(this);

        // For wavefront adjustments, see conceptual diagram in
        // run_solution().  At each stage and time-step, the parallelogram
        // may be trimmed based on the BB and WF extensions outside of the
        // rank-BB.

        // Actual mega-block boundaries must stay within [extended] stage BB.
        // We have to calculate the posn in the extended rank at each
        // value of 'shift_num' because it is being shifted spatially.
        bool ok = true;
        DOMAIN_VAR_LOOP(i, j) {
            auto angle = wf_angles[j];
            idx_t shift_amt = angle * shift_num;

            // Shift initial spatial mega-block boundaries for this iteration of
            // temporal wavefront.  Mega-Blocks only shift left, so mega-block loops
            // must strictly increment. They may do so in any order.  Shift
            // by pts in one WF step.  Always shift left in WFs.
            idx_t rstart = base_start[i] - shift_amt;
            idx_t rstop = base_stop[i] - shift_amt;

            // Trim only if stage is specified.
            if (bp.get()) {

                // Trim to extended BB of stage. This will also trim
                // to the extended BB of the rank.
                auto& pbb = bp.get()->get_bb();
                rstart = max(rstart, pbb.bb_begin[j]);
                rstop = min(rstop, pbb.bb_end[j]);

                // Find non-extended domain. We'll use this to determine if
                // we're in an extension, where special rules apply.
                idx_t dbegin = rank_bb.bb_begin[j];
                idx_t dend = rank_bb.bb_end[j];

                // In left ext, add 'angle' points for every shift to get
                // mega-block boundary in ext.
                if (rstart < dbegin && left_wf_exts[j])
                    rstart = max(rstart, dbegin - left_wf_exts[j] + shift_amt);

                // In right ext, subtract 'angle' points for every shift.
                if (rstop > dend && right_wf_exts[j])
                    rstop = min(rstop, dend + right_wf_exts[j] - shift_amt);

                // Trim mega-block based on current MPI section if overlapping.
                if (mpisec.is_overlap_active()) {

                    // Interior boundaries.
                    idx_t int_begin = mpi_interior.bb_begin[j];
                    idx_t int_end = mpi_interior.bb_end[j];

                    if (wf_steps > 0) {

                        // If doing WF tiling, each exterior shape is a
                        // trapezoid with its height in the time dim.  Each
                        // shift reduces the width of the trapezoid until it is
                        // the minimum width at the top.  Thus, the interior is
                        // an inverted trapezoid between the exterior ones.

                        //       +----+---------------+----+
                        // t    / ext  \  interior   / ext  \    .
                        // ^   /  left  \           /  right \   .
                        // |  +----------+---------+----------+
                        // +--->x        ^          ^
                        //               |          |
                        //             int_begin  int_end

                        // Modify interior if there is an external MPI
                        // section on either side.  Reduce interior by
                        // 'wf_shift_pts' to get size at base of mega-block,
                        // then expand by current shift amount to get size
                        // at current shift number.
                        if (does_exterior_exist(j, true)) { // left.
                            int_begin += wf_shift_pts[j];
                            int_begin -= shift_amt;
                        }
                        if (does_exterior_exist(j, false)) { // right.
                            int_end -= wf_shift_pts[j];
                            int_end += shift_amt;
                        }
                    }

                    // In interior.
                    if (mpisec.do_mpi_interior) {
                        rstart = max(rstart, int_begin);
                        rstop = min(rstop, int_end);
                    }

                    // In one of the exterior sections.
                    else {

                        // Should be doing either left or right, not both.
                        assert(mpisec.do_mpi_left != mpisec.do_mpi_right);

                        // Nothing to do if specified exterior section
                        // doesn't exist.
                        if (!does_exterior_exist(mpisec.mpi_exterior_dim, mpisec.do_mpi_left)) {
                            ok = false;
                            break;
                        }

                        // Example in 2D:
                        // +------+------------+------+
                        // |      | ext left y |      |
                        // |      |            |      |
                        // | ext  +------------+ ext  | <-- mpi_interior.bb_begin[y]
                        // | left |  interior  | right|
                        // | x    |            | x    |
                        // |      +------------+      |
                        // |      | ext right y|      | <-- mpi_interior.bb_end[y]
                        // |      |            |      |
                        // +------+------------+------+
                        //        ^             ^
                        //        |             |
                        //        |           mpi_interior.bb_end[x]
                        //      mpi_interior.bb_begin[x]

                        // Trim left or right for current dim.
                        if (j == mpisec.mpi_exterior_dim) {
                            if (mpisec.do_mpi_left)
                                rstop = min(rstop, int_begin);

                            else {
                                assert(mpisec.do_mpi_right);
                                rstart = max(rstart, int_end);

                                // For right, also need to trim to avoid
                                // overlap with left. This could happen
                                // when the width of the rank is less
                                // than twice the amount of temporal
                                // shifting. This implies left always
                                // needs to be done before right.
                                rstart = max(rstart, int_begin);
                            }
                        }

                        // Trim across all dims up to current one, e.g.,
                        // trim overlap between 'x' and 'y' from 'y'.
                        // See above diagram. This implies dims need
                        // to be done in ascending numerical order.
                        if (j < mpisec.mpi_exterior_dim) {
                            rstart = max(rstart, int_begin);
                            rstop = min(rstop, int_end);
                        }
                    } // exterior.
                } // overlapping.

                // Anything to do in the adjusted mega-block?
                if (rstop <= rstart) {
                    ok = false;
                    break;
                }
            } // Trimming.

            // Copy result into idxs.
            idxs.begin[i] = rstart;
            idxs.end[i] = rstop;
        }
        TRACE_MSG("shift_mega_block: updated span: [" <<
                  idxs.begin.make_val_str() << " ... " <<
                  idxs.end.make_val_str() << ") for " <<
                  mpisec.make_descr() << 
                  " within mega-block base [" <<
                  base_start.make_val_str() << " ... " <<
                  base_stop.make_val_str() << ") shifted " <<
                  shift_num << " time(s) is " <<
                  (ok ? "not " : "") << "empty");
        return ok;
    }

    // For given 'phase' and 'shape', find boundaries within micro-block at
    // 'mb_base_start' to 'mb_base_stop' shifted by 'mb_shift_num', which
    // should start at 0 and increment for each stage in each time-step.
    // 'mb_base' is subset of 'adj_block_base'.  Also trim to block at
    // 'block_base_start' to 'block_base_stop' shifted by 'mb_shift_num'.
    // Input 'begin' and 'end' of 'idxs' should be trimmed to mega-block.  Writes
    // results back into 'begin' and 'end' of 'idxs'.  Returns 'true' if
    // resulting area is non-empty, 'false' if empty.
    bool StencilContext::shift_micro_block(const Indices& mb_base_start,
                                           const Indices& mb_base_stop,
                                           const Indices& adj_block_base_start,
                                           const Indices& adj_block_base_stop,
                                           const Indices& block_base_start,
                                           const Indices& block_base_stop,
                                           const Indices& mega_block_base_start,
                                           const Indices& mega_block_base_stop,
                                           idx_t mb_shift_num,
                                           idx_t nphases, idx_t phase,
                                           idx_t nshapes, idx_t shape,
                                           const bit_mask_t& bridge_mask,
                                           ScanIndices& idxs) {
        STATE_VARS(this);
        auto nstages = st_stages.size();
        bool ok = true;

        // Loop thru dims, breaking out if any dim has no work.
        DOMAIN_VAR_LOOP(i, j) {

            // Determine range of this block for current phase, shape, and
            // shift. For each dim, we'll first compute the L & R sides of
            // the base block and the L side of the next block.
            auto tb_angle = tb_angles[j];

            // Is this block first and/or last in mega-block?
            bool is_first_blk = block_base_start[i] <= mega_block_base_start[i];
            bool is_last_blk = block_base_stop[i] >= mega_block_base_stop[i];

            // Is there only one blk in the mega-block in this dim?
            bool is_one_blk = is_first_blk && is_last_blk;

            // Initial start and stop point of phase-0 block.
            idx_t blk_start = block_base_start[i];
            idx_t blk_stop = block_base_stop[i];

            // If more than one blk, adjust for base of phase-0 trapezoid.
            if (nphases > 1 && !is_one_blk)
                blk_stop = min(blk_start + tb_widths[j], block_base_stop[i]);

            // Starting point of the *next* block.  This is used to create
            // bridge shapes between blocks.  Initially, the beginning of
            // the next block is the end of this block.
            idx_t next_blk_start = block_base_stop[i];

            // Adjust these based on current shift.  Adjust by pts in one TB
            // step, reducing size on R & L sides.  But if block is first
            // and/or last, clamp to mega-block.  TODO: have different R & L
            // angles. TODO: have different shifts for each stage.

            // Shift start to right unless first.  First block will be a
            // parallelogram or trapezoid clamped to beginning of mega-block.
            blk_start += tb_angle * mb_shift_num;
            if (is_first_blk)
                blk_start = idxs.begin[i];

            // Shift stop to left. If there will be no bridges, clamp
            // last block to end of mega-block.
            blk_stop -= tb_angle * mb_shift_num;
            if ((nphases == 1 || is_one_blk) && is_last_blk)
                blk_stop = idxs.end[i];

            // Shift start of next block. Last bridge will be
            // clamped to end of mega-block.
            next_blk_start += tb_angle * mb_shift_num;
            if (is_last_blk)
                next_blk_start = idxs.end[i];

            // Use these 3 values to determine the beginning and end
            // of the current shape for the current phase.
            // For phase 0, limits are simply the base start and stop.
            idx_t shape_start = blk_start;
            idx_t shape_stop = blk_stop;

            // Depending on the phase and shape, create a bridge between
            // from RHS of base block to the LHS of the next block
            // until all dims are bridged at last phase.
            // Use list of dims to bridge for this shape
            // computed earlier.
            if (phase > 0 && is_bit_set(bridge_mask, j)) {
                TRACE_MSG("shift_micro_block: phase " << phase <<
                          ", shape " << shape <<
                          ": bridging dim " << j);

                // Start at end of base block, but not
                // before start of block.
                shape_start = max(blk_stop, blk_start);

                // Stop at beginning of next block.
                shape_stop = next_blk_start;
            }

            // We now have bounds of this shape in shape_{start,stop}
            // for given phase and shift.
            if (shape_stop <= shape_start)
                ok = false;
            else {

                // Is this micro-block first and/or last in block?
                bool is_first_mb = mb_base_start[i] <= adj_block_base_start[i];
                bool is_last_mb = mb_base_stop[i] >= adj_block_base_stop[i];

                // Is there only one MB?
                bool is_one_mb = is_first_mb && is_last_mb;

                // Beginning and end of min-block.
                idx_t mb_start = mb_base_start[i];
                idx_t mb_stop = mb_base_stop[i];

                // Shift micro-block by MB angles unless there is only one.
                // MB is a wave-front, so only shift left.
                if (!is_one_mb) {
                    auto mb_angle = mb_angles[j];
                    mb_start -= mb_angle * mb_shift_num;
                    mb_stop -= mb_angle * mb_shift_num;
                }

                // Clamp first & last MB to shape boundaries.
                if (is_first_mb)
                    mb_start = shape_start;
                if (is_last_mb)
                    mb_stop = shape_stop;

                // Trim micro-block to fit in mega-block.
                mb_start = max(mb_start, idxs.begin[i]);
                mb_stop = min(mb_stop, idxs.end[i]);

                // Trim micro-block range to fit in shape.
                mb_start = max(mb_start, shape_start);
                mb_stop = min(mb_stop, shape_stop);

                // Update 'idxs'.
                idxs.begin[i] = mb_start;
                idxs.end[i] = mb_stop;

                // No work to do?
                if (mb_stop <= mb_start)
                    ok = false;
            }
            if (!ok)
                break;

        } // dims.

        TRACE_MSG("shift_micro_block: phase " << phase << "/" << nphases <<
                  ", shape " << shape << "/" << nshapes <<
                  ", updated span: [" <<
                  idxs.begin.make_val_str() << " ... " <<
                  idxs.end.make_val_str() << ") from original micro-block [" <<
                  mb_base_start.make_val_str() << " ... " <<
                  mb_base_stop.make_val_str() << ") shifted " <<
                  mb_shift_num << " time(s) within adj-block base [" <<
                  adj_block_base_start.make_val_str() << " ... " <<
                  adj_block_base_stop.make_val_str() << ") and actual block base [" <<
                  block_base_start.make_val_str() << " ... " <<
                  block_base_stop.make_val_str() << ") and mega-block base [" <<
                  mega_block_base_start.make_val_str() << " ... " <<
                  mega_block_base_stop.make_val_str() << ") is " <<
                  (ok ? "not " : "") << "empty");
        return ok;
    }

    // Compare output vars in contexts.
    // Return number of mis-compares.
    idx_t StencilContext::compare_data(const StencilContext& ref) const {
        STATE_VARS_CONST(this);
        copy_vars_from_device();

        DEBUG_MSG("Comparing output var(s) in '" << name << "' to '" << ref.name << "'...");
        if (output_var_ptrs.size() != ref.output_var_ptrs.size()) {
            TRACE_MSG("** number of output vars not equal");
            return 1;
        }
        idx_t errs = 0;
        for (size_t gi = 0; gi < output_var_ptrs.size(); gi++) {
            auto& gb = output_var_ptrs[gi]->gb();
            auto* rgbp = ref.output_var_ptrs[gi]->gbp();
            TRACE_MSG("Var '" << gb.get_name() << "'...");
            errs += gb.compare(rgbp);
        }

        return errs;
    }

    // Update data in vars that *may* have been written to by stage 'sel_bp'
    // in any rank: set the last "valid step" and mark vars as "dirty",
    // i.e., indicate that we may need to do a halo exchange.
    void StencilContext::update_var_info(const StagePtr& sel_bp,
                                         idx_t start, idx_t stop,
                                         bool mark_dirty,
                                         bool mod_dev_data) {
        STATE_VARS(this);
        idx_t stride = (start > stop) ? -1 : 1;

        // Stages.
        for (auto& bp : st_stages) {

            // Not a selected stage?
            if (sel_bp && sel_bp != bp)
                continue;

            // Each input step.
            for (idx_t t = start; t != stop; t += stride) {

                // Each bundle in this stage.
                for (auto* sb : *bp) {

                    // Output vars for this bundle.
                    sb->update_var_info(YkVarBase::others, t, mark_dirty, mod_dev_data, true);

                } // bundles.
            } // steps.
        } // stages.
    } // update_var_info().

    // Reset any locks, etc.
    void StencilContext::reset_locks() {

        // MPI buffer locks.
        for (auto& mdi : mpi_data) {
            auto& md = mdi.second;
            md.reset_locks();
        }
    }

    // Copy vars from host to device as needed.
    void StencilContext::copy_vars_to_device() const {
         for (auto gp : orig_var_ptrs) {
            assert(gp);
            gp->gb().const_copy_data_to_device();
        }
     }
    
    // Copy vars from device to host as needed.
    void StencilContext::copy_vars_from_device() const {
         for (auto gp : orig_var_ptrs) {
            assert(gp);
            gp->gb().const_copy_data_from_device();
        }
    }
    
} // namespace yask.
