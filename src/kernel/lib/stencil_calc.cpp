/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2022, Intel Corporation

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

// This file contains implementations of bundle and stage methods.
// Also see context_setup.cpp.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Calculate results within a micro-block defined by 'micro_block_idxs'.
    // This is called by StencilContext::calc_micro_block() for each bundle.
    // It is here that any required scratch-var stencils are evaluated
    // first and then the non-scratch stencils in the stencil bundle.
    // It is also here that the boundaries of the bounding-box(es) of the bundle
    // are respected. There must not be any temporal blocking at this point.
    void StencilBundleBase::calc_micro_block(int outer_thread_idx,
                                             KernelSettings& settings,
                                             const ScanIndices& micro_block_idxs) {
        STATE_VARS(this);
        TRACE_MSG("calc_micro_block('" << get_name() << "'): [" <<
                   micro_block_idxs.begin.make_val_str() << " ... " <<
                   micro_block_idxs.end.make_val_str() << ") by " <<
                   micro_block_idxs.stride.make_val_str() <<
                   " by outer thread " << outer_thread_idx);
        assert(!is_scratch());

        // No temporal blocking allowed here.
        assert(abs(micro_block_idxs.get_overall_range(step_posn)) == 1);
        auto t = micro_block_idxs.begin[step_posn];
        assert(abs(micro_block_idxs.end[step_posn] - t) == 1);

        // Nothing to do if outer BB is empty.
        if (_bundle_bb.bb_num_points == 0) {
            TRACE_MSG("calc_micro_block: empty BB");
            return;
        }

        // TODO: if >1 BB, check limits of outer one first to save time.

        // Set number of threads in this block.
        // This will be the number of nano-blocks done in parallel.
        int nbt = _context->set_num_inner_threads();

        // Thread-binding info.
        // We only bind threads if there is more than one block thread
        // and binding is enabled.
        bool bind_threads = nbt > 1 && settings.bind_inner_threads;
        int bind_posn = settings._bind_posn;
        idx_t bind_slab_pts = settings._nano_block_sizes[bind_posn]; // Other sizes not used.

        // Loop through each solid BB for this bundle.
        // For each BB, calc intersection between it and 'micro_block_idxs'.
        // If this is non-empty, apply the bundle to all its required nano-blocks.
        TRACE_MSG("calc_micro_block('" << get_name() << "'): checking " <<
                   _bb_list.size() << " BB(s)");
        int bbn = 0;
  	for (auto& bb : _bb_list) {
            bbn++;
            bool bb_ok = true;
            if (bb.bb_num_points == 0)
                bb_ok = false;

            // Trim the micro-block indices based on the bounding box(es)
            // for this bundle.
            ScanIndices mb_idxs(micro_block_idxs);
            DOMAIN_VAR_LOOP_FAST(i, j) {

                // Begin point.
                auto bbegin = max(micro_block_idxs.begin[i], bb.bb_begin[j]);
                mb_idxs.begin[i] = bbegin;

                // End point.
                auto bend = min(micro_block_idxs.end[i], bb.bb_end[j]);
                mb_idxs.end[i] = bend;

                // Anything to do?
                if (bend <= bbegin) {
                    bb_ok = false;
                    break;
                }
            }

            // nothing to do?
            if (!bb_ok) {
                TRACE_MSG("calc_micro_block for bundle '" << get_name() <<
                           "': no overlap between bundle " << bbn << " and current block");
                continue; // to next BB.
            }

            TRACE_MSG("calc_micro_block('" << get_name() <<
                       "'): after trimming for BB " << bbn << ": [" <<
                       mb_idxs.begin.make_val_str() <<
                       " ... " << mb_idxs.end.make_val_str() << ")");

            // Get the bundles that need to be processed in
            // this block. This will be any prerequisite scratch-var
            // bundles plus the current non-scratch bundle.
            auto sg_list = get_reqd_bundles();

            // Loop through all the needed bundles.
            for (auto* sg : sg_list) {

                // Indices needed for the generated loops.  Will normally be a
                // copy of 'mb_idxs' except when updating scratch-vars.
                ScanIndices adj_mb_idxs = sg->adjust_span(outer_thread_idx, mb_idxs);

                // Tweak settings for adjusted indices.
                adj_mb_idxs.adjust_from_settings(settings._micro_block_sizes,
                                                 settings._micro_block_tile_sizes,
                                                 settings._nano_block_sizes);

                // If binding threads to data.
                if (bind_threads) {

                    // Tweak settings for adjusted indices.  This sets
                    // up the nano-blocks as multiple slabs perpendicular
                    // to the binding dim within the micro-block.
                    DOMAIN_VAR_LOOP_FAST(i, j) {

                        // If this is the binding dim, set stride size
                        // and alignment granularity to the slab
                        // width. Setting the alignment keeps slabs
                        // aligned between stages and/or steps.
                        if (i == bind_posn) {
                            adj_mb_idxs.stride[i] = bind_slab_pts;
                            adj_mb_idxs.align[i] = bind_slab_pts;
                        }

                        // If this is not the binding dim, set stride
                        // size to full width.  For now, this is the
                        // only option for micro-block shapes when
                        // binding.  TODO: consider other options.
                        else
                            adj_mb_idxs.stride[i] = adj_mb_idxs.get_overall_range(i);
                    }

                    TRACE_MSG("calc_micro_block('" << get_name() << "'): " <<
                              " for reqd bundle '" << sg->get_name() << "': [" <<
                              adj_mb_idxs.begin.make_val_str() << " ... " <<
                              adj_mb_idxs.end.make_val_str() << ") by " <<
                              adj_mb_idxs.stride.make_val_str() <<
                              " by outer thread " << outer_thread_idx <<
                              " with " << nbt << " block thread(s) bound to data");

                    // Start threads within a block.  Each of these threads
                    // will eventually work on a separate nano-block.  This
                    // is nested within an OMP outer thread.
                    _Pragma("omp parallel proc_bind(spread)") {
                        assert(omp_get_level() == 2);
                        assert(omp_get_num_threads() == nbt);
                        int inner_thread_idx = omp_get_thread_num();

                        // Run the micro-block loops on all block threads and
                        // call calc_nano_block() only by the designated
                        // thread for the given slab index in the binding
                        // dim. This is an explicit replacement for "normal"
                        // OpenMP scheduling.
                        
                        // Disable the OpenMP construct in the micro-block loop
                        // because we're already in the parallel section.
                        #define MICRO_BLOCK_OMP_PRAGMA

                        // Loop prefix.
                        #define MICRO_BLOCK_LOOP_INDICES adj_mb_idxs
                        #define MICRO_BLOCK_BODY_INDICES nano_blk_range
                        #define MICRO_BLOCK_USE_LOOP_PART_0
                        #include "yask_micro_block_loops.hpp"

                        // Loop body.
                        const idx_t idx_ofs = 0x1000; // to help keep pattern when idx is neg.
                        auto bind_elem_idx = nano_blk_range.start[bind_posn];
                        auto bind_slab_idx = idiv_flr(bind_elem_idx + idx_ofs, bind_slab_pts);
                        auto bind_thr = imod_flr<idx_t>(bind_slab_idx, nbt);
                        if (inner_thread_idx == bind_thr)
                            sg->calc_nano_block(outer_thread_idx, inner_thread_idx,
                                               settings, nano_blk_range);

                        // Loop sufffix.
                        #define MICRO_BLOCK_USE_LOOP_PART_1
                        #include "yask_micro_block_loops.hpp"

                    } // Parallel region.
                } // Binding threads to data.

                // If not binding or there is only one block per thread.
                // (This is the more common case.)
                else {

                    TRACE_MSG("calc_micro_block('" << get_name() << "'): " <<
                              " for reqd bundle '" << sg->get_name() << "': [" <<
                              adj_mb_idxs.begin.make_val_str() << " ... " <<
                              adj_mb_idxs.end.make_val_str() << ") by " <<
                              adj_mb_idxs.stride.make_val_str() <<
                              " by outer thread " << outer_thread_idx <<
                              " with " << nbt << " block thread(s) NOT bound to data");

                    // Call calc_nano_block() with a different thread for
                    // each nano-block using standard OpenMP scheduling.
                    
                    // Loop prefix.
                    #define MICRO_BLOCK_LOOP_INDICES adj_mb_idxs
                    #define MICRO_BLOCK_BODY_INDICES nano_blk_range
                    #define MICRO_BLOCK_USE_LOOP_PART_0
                    #include "yask_micro_block_loops.hpp"

                    // Loop body.
                    int inner_thread_idx = omp_get_thread_num();
                    sg->calc_nano_block(outer_thread_idx, inner_thread_idx,
                                       settings, nano_blk_range);

                    // Loop suffix.
                    #define MICRO_BLOCK_USE_LOOP_PART_1
                    #include "yask_micro_block_loops.hpp"

                } // OMP parallel when binding threads to data.
            } // bundles.

            // Mark exterior dirty for halo exchange if exterior was done.
            bool mark_dirty = _context->do_mpi_left || _context->do_mpi_right;
            update_var_info(YkVarBase::self, t, mark_dirty, true, false);
            
        } // BB list.
    } // calc_micro_block().

    // Mark vars dirty that are updated by this bundle and/or
    // update last valid step.
    void StencilBundleBase::update_var_info(YkVarBase::dirty_idx whose,
                                            idx_t t,
                                            bool mark_extern_dirty,
                                            bool mod_dev_data,
                                            bool update_valid_step) {
        STATE_VARS(this);

        // Get output step for this bundle, if any.  For most stencils, this
        // will be t+1 or t-1 if striding backward.
        idx_t t_out = 0;
        if (!get_output_step_index(t, t_out)) {
            TRACE_MSG("not updating because output step is not available");
            return;
        }

        // Output vars for this bundle.  NB: don't need to mark
        // scratch vars as dirty because they are never exchanged.
        for (auto gp : output_var_ptrs) {
            auto& gb = gp->gb();

            // Mark given dirty flag.
            // This flag will be false if we're only updating the interior,
            // i.e., we don't need to trigger a halo exchange.
            if (mark_extern_dirty) {
                gb.set_dirty(whose, true, t_out);
                TRACE_MSG(gb.get_name() << " marked dirty");
            }

            // Mark the entire var as dirty on the device, regardless
            // of whether this is the interior or exterior.
            if (mod_dev_data)
                gb.get_coh().mod_dev();

            // Update last valid step.
            if (update_valid_step)
                gb.update_valid_step(t_out);
        }
    }

    // If this bundle is updating scratch var(s),
    // expand begin & end of 'idxs' by sizes of halos.
    // Stride indices may also change.
    // NB: it is not necessary that the domain of each var
    // is the same as the span of 'idxs'. However, it should be
    // at least that large to ensure that var is able to hold
    // calculated results. This is checked when 'CHECK' is defined.
    // In other words, var can be larger than span of 'idxs', but
    // its halo sizes are still used to specify how much to
    // add to 'idxs'.
    // Returns adjusted indices.
    ScanIndices StencilBundleBase::adjust_span(int outer_thread_idx,
                                               const ScanIndices& idxs) const {
        STATE_VARS(this);
        ScanIndices adj_idxs(idxs);

        // Loop thru vecs of scratch vars for this bundle.
        for (auto* sv : output_scratch_vecs) {
            assert(sv);

            // Get the one for this thread.
            auto& gp = sv->at(outer_thread_idx);
            assert(gp);
            auto& gb = gp->gb();
            assert(gb.is_scratch());

            // i: index for stencil dims, j: index for domain dims.
            DOMAIN_VAR_LOOP_FAST(i, j) {
                auto& dim = dims->_stencil_dims.get_dim(i);
                auto& dname = dim._get_name();

                // Is this dim used in this var?
                int posn = gb.get_dim_posn(dname);
                if (posn >= 0) {

                    // Get halos, which need to be written to for
                    // scratch vars.
                    idx_t lh = gp->get_left_halo_size(posn);
                    idx_t rh = gp->get_right_halo_size(posn);

                    // Round up halos to vector sizes.
                    // TODO: consider cluster sizes, but need to make changes
                    // elsewhere in code.
                    lh = ROUND_UP(lh, fold_pts[j]);
                    rh = ROUND_UP(rh, fold_pts[j]);

                    // Adjust begin & end scan indices based on halos.
                    adj_idxs.begin[i] = idxs.begin[i] - lh;
                    adj_idxs.end[i] = idxs.end[i] + rh;

                    // Make sure var covers index bounds.
                    TRACE_MSG("adjust_span: micro-blk [" <<
                              idxs.begin[i] << "..." <<
                              idxs.end[i] << ") adjusted to [" <<
                              adj_idxs.begin[i] << "..." <<
                              adj_idxs.end[i] << ") within scratch-var '" <<
                              gp->get_name() << "' with halos " <<
                              gp->get_left_halo_size(posn) << " and " <<
                              gp->get_right_halo_size(posn) << " allocated [" <<
                              gp->get_first_local_index(posn) << "..." <<
                              gp->get_last_local_index(posn) << "] in dim '" << dname << "'");
                    assert(adj_idxs.begin[i] >= gp->get_first_local_index(posn));
                    assert(adj_idxs.end[i] <= gp->get_last_local_index(posn) + 1);

                    // If existing stride is >= whole tile, adjust it also.
                    idx_t width = idxs.end[i] - idxs.begin[i];
                    if (idxs.stride[i] >= width) {
                        idx_t adj_width = adj_idxs.end[i] - adj_idxs.begin[i];
                        adj_idxs.stride[i] = adj_width;
                    }
                }
            }

            // Only need to get info from one var.
            // TODO: check that vars are consistent.
            break;
        }
        return adj_idxs;
    } // adjust_span().

    // Timer methods.
    // Start and stop stage timers for final stats and track steps done.
    void Stage::start_timers() {
        timer.start();
    }
    void Stage::stop_timers() {
        timer.stop();
    }
    void Stage::add_steps(idx_t num_steps) {
        steps_done += num_steps;
    }

    static void print_var_list(ostream& os, const VarPtrs& gps, const string& type) {
        os << "  num " << type << " vars:";
        for (size_t i = 0; i < max(21ULL - type.length(), 1ULL); i++)
            os << ' ';
        os << gps.size() << endl;
        if (gps.size()) {
            os << "  " << type << " vars:";
            for (size_t i = 0; i < max(25ULL - type.length(), 1ULL); i++)
                os << ' ';
            int i = 0;
            for (auto gp : gps) {
                if (i++) os << ", ";
                os << gp->get_name();
            }
            os << endl;
        }
    }

    // Calc the work stats.
    // Contains MPI barriers!
    void Stage::init_work_stats() {
        STATE_VARS(this);

        num_reads_per_step = 0;
        num_writes_per_step = 0;
        num_fpops_per_step = 0;

        DEBUG_MSG("Stage '" << get_name() << "':\n" <<
                  " num bundles:                 " << size() << endl <<
                  " stage scope:                 " << _stage_bb.make_range_string(domain_dims));

        // Bundles.
        for (auto* sg : *this) {

            // Stats for this bundle for 1 pt.
            idx_t writes1 = 0, reads1 = 0, fpops1 = 0;

            // Loop through all the needed bundles to count stats for
            // scratch bundles.  Does not count extra ops needed in scratch
            // halos since this varies depending on block size.
            auto sg_list = sg->get_reqd_bundles();
            for (auto* rsg : sg_list) {
                reads1 += rsg->get_scalar_points_read();
                writes1 += rsg->get_scalar_points_written();
                fpops1 += rsg->get_scalar_fp_ops();
            }

            // Multiply by valid pts in BB for this bundle.
            auto& bb = sg->get_bb();
            idx_t writes_bb = writes1 * bb.bb_num_points;
            num_writes_per_step += writes_bb;
            idx_t reads_bb = reads1 * bb.bb_num_points;
            num_reads_per_step += reads_bb;
            idx_t fpops_bb = fpops1 * bb.bb_num_points;
            num_fpops_per_step += fpops_bb;

            DEBUG_MSG(" Bundle '" << sg->get_name() << "':\n" <<
                      "  num reqd scratch bundles:   " << (sg_list.size() - 1));
            // TODO: add info on scratch bundles here.

            if (sg->is_sub_domain_expr())
                DEBUG_MSG("  sub-domain expr:            '" << sg->get_domain_description() << "'");
            if (sg->is_step_cond_expr())
                DEBUG_MSG("  step-condition expr:        '" << sg->get_step_cond_description() << "'");

            DEBUG_MSG("  bundle size (points):       " << make_num_str(bb.bb_size));
            if (bb.bb_size) {
                DEBUG_MSG("  valid points in bundle:     " << make_num_str(bb.bb_num_points));
                if (bb.bb_num_points) {
                    DEBUG_MSG("  bundle scope:               " << bb.make_range_string(domain_dims) <<
                              "\n  bundle bounding-box size:   " << bb.make_len_string(domain_dims));
                }
            }
            DEBUG_MSG("  num full rectangles in box: " << sg->get_bbs().size());
            if (sg->get_bbs().size() > 1) {
                for (size_t ri = 0; ri < sg->get_bbs().size(); ri++) {
                    auto& rbb = sg->get_bbs()[ri];
                    DEBUG_MSG("   Rectangle " << ri << ":\n"
                              "    num points in rect:       " << make_num_str(rbb.bb_num_points));
                    if (rbb.bb_num_points) {
                        DEBUG_MSG("    rect scope:               " << rbb.make_range_string(domain_dims) <<
                                  "\n    rect size:                " << rbb.make_len_string(domain_dims));
                    }
                }
            }
            DEBUG_MSG("  var-reads per point:        " << reads1 << endl <<
                      "  var-reads in rank:          " << make_num_str(reads_bb) << endl <<
                      "  var-writes per point:       " << writes1 << endl <<
                      "  var-writes in rank:         " << make_num_str(writes_bb) << endl <<
                      "  est FP-ops per point:       " << fpops1 << endl <<
                      "  est FP-ops in rank:         " << make_num_str(fpops_bb));

            // Classify vars.
            VarPtrs idvars, imvars, odvars, omvars, iodvars, iomvars; // i[nput], o[utput], d[omain], m[isc].
            for (auto gp : sg->input_var_ptrs) {
                auto& gb = gp->gb();
                bool isdom = gb.is_domain_var();
                auto& ogps = sg->output_var_ptrs;
                bool isout = find(ogps.begin(), ogps.end(), gp) != ogps.end();
                if (isout) {
                    if (isdom)
                        iodvars.push_back(gp);
                    else
                        iomvars.push_back(gp);
                } else {
                    if (isdom)
                        idvars.push_back(gp);
                    else
                        imvars.push_back(gp);
                }
            }
            for (auto gp : sg->output_var_ptrs) {
                auto& gb = gp->gb();
                bool isdom = gb.is_domain_var();
                auto& igps = sg->input_var_ptrs;
                bool isin = find(igps.begin(), igps.end(), gp) != igps.end();
                if (!isin) {
                    if (isdom)
                        odvars.push_back(gp);
                    else
                        omvars.push_back(gp);
                }
            }
            yask_output_ptr op = ksbp->get_debug_output();
            ostream& os = op->get_ostream();
            print_var_list(os, idvars, "input-only domain");
            print_var_list(os, odvars, "output-only domain");
            print_var_list(os, iodvars, "input-output domain");
            print_var_list(os, imvars, "input-only other");
            print_var_list(os, omvars, "output-only other");
            print_var_list(os, iomvars, "input-output other");

        } // bundles.

        // Sum across ranks.
        tot_reads_per_step = sum_over_ranks(num_reads_per_step, env->comm);
        tot_writes_per_step = sum_over_ranks(num_writes_per_step, env->comm);
        tot_fpops_per_step = sum_over_ranks(num_fpops_per_step, env->comm);

    } // init_work_stats().

} // namespace yask.
