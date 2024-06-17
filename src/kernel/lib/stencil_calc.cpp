/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2024, Intel Corporation

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

// This file contains implementations of part and stage methods.
// Also see context_setup.cpp.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Calculate results within a micro-block defined by 'micro_block_idxs'.
    // This is called by StencilContext::calc_micro_block() for each part.
    // It is here that any required scratch-var stencils are evaluated
    // first and then the non-scratch stencils in the stencil part.
    // It is also here that the boundaries of the bounding-box(es) of the part
    // are respected. There must not be any temporal blocking at this point.
    void StencilPartBase::calc_micro_block(int outer_thread_idx,
                                           KernelSettings& settings,
                                           const ScanIndices& micro_block_idxs,
                                           MpiSection& mpisec,
                                           StencilPartUSet& parts_done,
                                           VarPtrUSet& vars_written) {
        STATE_VARS(this);
        TRACE_MSG("in '" << get_name() << "': " <<
                  micro_block_idxs.make_range_str(true) <<
                  " via outer thread " << outer_thread_idx);
        assert(!is_scratch());

        // No temporal blocking allowed here.
        assert(abs(micro_block_idxs.get_overall_range(step_posn)) == 1);
        auto t = micro_block_idxs.begin[step_posn];
        assert(abs(micro_block_idxs.end[step_posn] - t) == 1);

        // Nothing to do if outer BB is empty.
        if (_part_bb.bb_num_points == 0) {
            TRACE_MSG("empty BB");
            return;
        }

        // Set number of threads in this block.
        // This will be the number of nano-blocks done in parallel.
        int nbt = _context->set_num_inner_threads();

        // Thread-binding info.
        // We only bind threads if there is more than one inner thread
        // and binding is enabled.
        bool bind_threads = nbt > 1 && settings.bind_inner_threads;
        int bind_posn = settings._bind_posn;
        idx_t bind_slab_pts = settings._nano_block_sizes[bind_posn]; // Other sizes not used.

        // Get the parts that need to be processed in
        // this block. This will be any prerequisite scratch-var
        // parts plus the current non-scratch part.
        auto rp_list = get_reqd_parts();

        // Loop through all the needed parts.
        for (auto* rp : rp_list) {
            TRACE_MSG("processing reqd part '" << rp->get_name() << "'");
            bool is_scratch = rp->is_scratch();

            // Initialize any scratch output vars that haven't been written
            // to yet. For each scratch var, do this only when the first
            // scratch part that writes to it is scheduled. This helps with
            // cache locality: we don't want to init a var long before it's
            // used, which might evict another var's data from a cache.
            // Also, we do this before we check the step-condition. This is
            // so that a var will be properly initialized in the case where
            // it is read from after a part that would have written to it,
            // but the step condition wasn't true.
            if (is_scratch && actl_opts->_init_scratch_vars) {
                for (auto* sv : rp->output_scratch_vecs) {
                    assert(sv);
                    auto vp = sv->at(outer_thread_idx); // Var for current thread.
                    assert(vp);
                    if (vars_written.count(vp) == 0) {

                        // Only need to init if there are conditional
                        // expressions; otherwise, all points will be
                        // written do, so no need to init var.
                        if (rp->is_sub_domain_expr() ||
                            rp->is_step_cond_expr()) {
                            TRACE_MSG("initializing " << vp->get_name());
                            vp->set_all_elements_same(0.0);
                        }
                        vars_written.insert(vp); // Mark as written.
                    }
                }
            }
            
            // Check step.
            if (!rp->is_in_valid_step(t)) {
                TRACE_MSG(rp->get_name() << " not needed for step " << t);
                continue;
            }

            // Already done?  This is tracked across calls to this func
            // because >1 non-scratch part in a stage can depend on some
            // common scratch part(s). This is also the reason we don't
            // trim scratch part(s) to the BB(s) of the non-scratch
            // part(s) that they depend on: the non-scratch parts may be
            // used by >1 non-scratch parts with different BBs.
            if (parts_done.count(rp)) {
                TRACE_MSG(rp->get_name() << " already done for this micro-blk");
                continue;
            }
            
            // For scratch-vars, expand indices based on write halo.
            ScanIndices mb_idxs2(micro_block_idxs);
            if (is_scratch) {
                mb_idxs2 = rp->adjust_scratch_span(outer_thread_idx, mb_idxs2,
                                                   settings);
                TRACE_MSG("micro-block adjusted to " << mb_idxs2.make_range_str(true) <<
                          " per scratch write halo");
            }

            // Loop through all the full BBs in this reqd part.
            TRACE_MSG("checking " << rp->get_bbs().size() <<
                      " full BB(s) for reqd part '" << rp->get_name() << "'");
            auto fbbs = rp->get_bbs();
            int fbbn = 0;
            for (auto& fbb : fbbs) {
                fbbn++;
                TRACE_MSG("reqd BB " << fbbn << ": " << fbb.make_range_str_dbg(domain_dims));

                // Find intersection between full BB and 'mb_idxs2'.
                ScanIndices mb_idxs3(mb_idxs2);
                bool fbb_ok = fbb.bb_num_points > 0;
                DOMAIN_VAR_LOOP_FAST(i, j) {

                    // Begin point.
                    auto bbegin = max(mb_idxs2.begin[i], fbb.bb_begin[j]);
                    mb_idxs3.begin[i] = bbegin;

                    // End point.
                    auto bend = min(mb_idxs2.end[i], fbb.bb_end[j]);
                    mb_idxs3.end[i] = bend;

                    // Anything to do?
                    if (bend <= bbegin)
                        fbb_ok = false;
                }
                if (!fbb_ok) {
                    TRACE_MSG("full reqd BB " << fbbn << " is empty");
                    continue;
                }
                TRACE_MSG("micro-block trimmed to " <<
                          mb_idxs3.make_range_str(true) << " within BB " << fbbn);

                ///// Bounds set for this BB; ready to evaluate it.

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
                            mb_idxs3.stride[i] = bind_slab_pts;
                            mb_idxs3.align[i] = bind_slab_pts;
                        }

                        // If this is not the binding dim, set stride
                        // size to > full width.  For now, this is the
                        // only option for micro-block shapes when
                        // binding.  TODO: consider other options.
                        else
                            mb_idxs3.stride[i] = ROUND_UP(mb_idxs3.get_overall_range(i),
                                                          fold_pts[j]) * 2;
                    }

                    TRACE_MSG("reqd part '" << rp->get_name() << "': " <<
                              mb_idxs3.make_range_str(true) <<
                              " via outer thread " << outer_thread_idx <<
                              " with " << nbt << " inner thread(s) bound to data...");

                    // Start threads within a block.  Each of these threads
                    // will eventually work on a separate nano-block.  This
                    // is nested within an OMP outer thread.
                    _Pragma("omp parallel") {
                        assert(omp_get_level() == 2);
                        assert(omp_get_num_threads() == nbt);
                        int inner_thread_idx = omp_get_thread_num();

                        // Run the micro-block loops on all inner threads and
                        // call calc_nano_block() only by the designated
                        // thread for the given slab index in the binding
                        // dim. This is an explicit replacement for "normal"
                        // OpenMP scheduling.
                        
                        // Disable the OpenMP construct in the micro-block loop
                        // because we're already in the parallel section.
                        #define MICRO_BLOCK_OMP_PRAGMA

                        // Loop prefix.
                        #define MICRO_BLOCK_LOOP_INDICES mb_idxs3
                        #define MICRO_BLOCK_BODY_INDICES nano_blk_range
                        #define MICRO_BLOCK_USE_LOOP_PART_0
                        #include "yask_micro_block_loops.hpp"

                        // Loop body.
                        const idx_t idx_ofs = 0x1000; // to help keep pattern when idx is neg.
                        auto bind_elem_idx = nano_blk_range.start[bind_posn];
                        auto bind_slab_idx = idiv_flr(bind_elem_idx + idx_ofs, bind_slab_pts);
                        auto bind_thr = imod_flr<idx_t>(bind_slab_idx, nbt);
                        if (inner_thread_idx == bind_thr)
                            rp->calc_nano_block(outer_thread_idx, inner_thread_idx,
                                                settings, nano_blk_range);

                        // Loop sufffix.
                        #define MICRO_BLOCK_USE_LOOP_PART_1
                        #include "yask_micro_block_loops.hpp"

                    } // Parallel region.
                } // Binding threads to data.

                // If not binding or there is only one block per thread.
                // (This is the more common case.)
                else {

                    TRACE_MSG("reqd part '" << rp->get_name() << "': " <<
                              mb_idxs3.make_range_str(true) << 
                              " via outer thread " << outer_thread_idx <<
                              " with " << nbt << " inner thread(s) NOT bound to data...");

                    // Call calc_nano_block() with a different thread for
                    // each nano-block using standard OpenMP scheduling.
                    
                    // Loop prefix.
                    #define MICRO_BLOCK_LOOP_INDICES mb_idxs3
                    #define MICRO_BLOCK_BODY_INDICES nano_blk_range
                    #define MICRO_BLOCK_USE_LOOP_PART_0
                    #include "yask_micro_block_loops.hpp"

                    // Loop body.
                    int inner_thread_idx = omp_get_thread_num();
                    rp->calc_nano_block(outer_thread_idx, inner_thread_idx,
                                        settings, nano_blk_range);

                    // Loop suffix.
                    #define MICRO_BLOCK_USE_LOOP_PART_1
                    #include "yask_micro_block_loops.hpp"

                } // not binding threads to data.
            } // full BBs in this required part.

            // Make sure streaming stores are visible for later loads.
            make_stores_visible();
            
            // Mark this part done. This avoid re-evaluating
            // scratch parts that are used more than once in
            // a stage.
            parts_done.insert(rp);

        } // required parts.

        // Mark exterior dirty for halo exchange if exterior was done.
        bool mark_dirty = mpisec.do_mpi_left || mpisec.do_mpi_right;
        update_var_info(YkVarBase::self, t, mark_dirty, true, false);
            
    } // calc_micro_block().

    // Mark vars dirty that are updated by this part and/or
    // update last valid step.
    void StencilPartBase::update_var_info(YkVarBase::dirty_idx whose,
                                            idx_t t,
                                            bool mark_extern_dirty,
                                            bool mod_dev_data,
                                            bool update_valid_step) {
        STATE_VARS(this);

        // Get output step for this part, if any.  For most stencils, this
        // will be t+1 or t-1 if striding backward.
        idx_t t_out = 0;
        if (!get_output_step_index(t, t_out)) {
            TRACE_MSG("not updating because output step is not available");
            return;
        }

        // Output vars for this part.  NB: don't need to mark
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
    
    // Expand begin & end of 'idxs' by sizes of write halos.
    // Stride indices may also change.
    // NB: it is not necessary that the domain of each var
    // is the same as the span of 'idxs'. However, it should be
    // at least that large to ensure that var is able to hold
    // calculated results. This is checked when 'CHECK' is defined.
    // In other words, var can be larger than span of 'idxs', but
    // its halo sizes are still used to specify how much to
    // add to 'idxs'.
    // Returns adjusted indices.
    ScanIndices StencilPartBase::adjust_scratch_span(int outer_thread_idx,
                                                       const ScanIndices& idxs,
                                                       KernelSettings& settings) const {
        assert(is_scratch());
        STATE_VARS(this);
        assert(max_write_halo_left.get_num_dims() == NUM_DOMAIN_DIMS);
        assert(max_write_halo_right.get_num_dims() == NUM_DOMAIN_DIMS);

        // Init return indices.
        ScanIndices adj_idxs(idxs);

        // Adjust for each dim.
        // i: index for stencil dims, j: index for domain dims.
        DOMAIN_VAR_LOOP(i, j) {
        
            // Adjust begin & end scan indices based on write halos.
            idx_t ab = idxs.begin[i] - max_write_halo_left[j];
            idx_t ae = idxs.end[i] + max_write_halo_right[j];

            #if 0
            // Currently disabled because it causes the bad-FP-value
            // checker to throw exceptions when outside normal bounds.
            //
            // Round up halos to vector sizes.  This is to [try to] avoid
            // costly masking around edges of scratch write area. For
            // scratch vars, it won't hurt to calculate extra values outside
            // of the min write area because those values should never be
            // used. Also, reading is always done at vector-lengths; only
            // writing is masked. Rounding must be rank-local, so
            // rounding is after by subtracting rank offsets, and then they
            // are re-added.  Be careful to allocate enough memory in
            // StencilContext::alloc_scratch_data().  NB: When a scratch var
            // has domain conditions, this won't always succeed in reducing
            // masking.
            idx_t ro = _context->rank_domain_offsets[j];
            ab = round_down_flr(ab - ro, fold_pts[j]) + ro;
            ae = round_up_flr(ae - ro, fold_pts[j]) + ro;
            #endif

            adj_idxs.begin[i] = ab;
            adj_idxs.end[i] = ae;

            // Adjust strides and/or tiles as needed.
            adj_idxs.adjust_from_settings(settings._micro_block_sizes,
                                          settings._micro_block_tile_sizes,
                                          settings._nano_block_sizes);

            auto& dim = dims->_domain_dims.get_dim(j);
            auto& dname = dim._get_name();

            // Make sure size of scratch vars cover new index bounds.
            // TODO: check size of input vars, incl. read halos.
            #ifdef CHECK
            for (auto* sv : output_scratch_vecs) {
                assert(sv);

                // Get the one for this thread.
                auto& gp = sv->at(outer_thread_idx);
                assert(gp);
                auto& gb = gp->gb();
                
                // Is this dim used in this var?
                int posn = gb.get_dim_posn(dname);
                if (posn >= 0) {

                    TRACE_MSG("micro-blk adjusted from [" <<
                              idxs.begin[i] << "..." <<
                              idxs.end[i] << ") to [" <<
                              adj_idxs.begin[i] << "..." <<
                              adj_idxs.end[i] << "); checking" <<
                              " against scratch-var '" <<
                              gp->get_name() << "' with halos " <<
                              gp->get_left_halo_size(posn) << " and " <<
                              gp->get_right_halo_size(posn) << " allocated [" <<
                              gp->get_first_local_index(posn) << "..." <<
                              gp->get_last_local_index(posn) << "] in dim '" << dname << "'");
                
                    assert(ab >= gp->get_first_local_index(posn));
                    assert(ae <= gp->get_last_local_index(posn) + 1);
                }
            }
            #endif // check.
        } // dims.
        
        return adj_idxs;
    } // adjust_scratch_span().
    
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
    // NB: Contains MPI barriers to sum work across ranks!
    void Stage::init_work_stats() {
        STATE_VARS(this);

        num_reads_per_step = 0;
        num_writes_per_step = 0;
        num_fpops_per_step = 0;

        DEBUG_MSG("Stage '" << get_name() << "':\n" <<
                  " num non-scratch parts:     " << size() << endl <<
                  " stage scope:                 " << _stage_bb.make_range_string(domain_dims));

        // Non-scratch parts.
        for (auto* sg : *this) {

            // This part and its scratch parts.
            auto sc_list = sg->get_scratch_children();
            auto sg_list = sg->get_reqd_parts();
            DEBUG_MSG(" Non-scratch part '" << sg->get_name() << "':\n" <<
                      "  num reqd scratch parts:   " << sc_list.size());
            
            // Stats for each part.
            typedef map<string, idx_t> bstats;
            bstats npts, writes, reads, fpops;

            // Loop through all the full BBs in this part.
            auto fbbs = sg->get_bbs();
            for (auto& fbb : fbbs) {

                // Loop through all the needed parts.
                for (auto* rsg : sg_list) {
                    auto& bname = rsg->get_name();

                    // Loop through all full BBs in needed part.
                    auto fnbbs = rsg->get_bbs();
                    for (auto& fnbb : fnbbs) {
                    
                        // Find intersection between BBs.
                        // NB: If fbb == fnbb, then bbi = fbb;
                        // TODO: add scratch halos in pad area.
                        auto bbi = fbb.intersection_with(fnbb, _context);
                        auto nptsi = bbi.bb_num_points;

                        // Add stats.
                        npts[bname] += nptsi;
                        reads[bname] += rsg->get_scalar_points_read() * nptsi;
                        writes[bname] += rsg->get_scalar_points_written() * nptsi;
                        fpops[bname] += rsg->get_scalar_fp_ops() * nptsi;
                    }
                }
            }

            // Loop through all needed parts.
            for (auto* rsg : sg_list) {
                auto& bname = rsg->get_name();
                num_reads_per_step += reads[bname];
                num_writes_per_step += writes[bname];
                num_fpops_per_step += fpops[bname];

                if (actl_opts->_verbose) {
                    DEBUG_MSG("  part '" << rsg->get_name() << "':");

                    if (rsg->is_sub_domain_expr())
                        DEBUG_MSG("   sub-domain expr:            '" << rsg->get_domain_description() << "'");
                    if (rsg->is_step_cond_expr())
                        DEBUG_MSG("   step-condition expr:        '" << rsg->get_step_cond_description() << "'");

                    DEBUG_MSG("   points to eval in part:   " << make_num_str(npts[bname]) << endl <<
                              "   var-reads per point:        " << rsg->get_scalar_points_read() << endl <<
                              "   var-writes per point:       " << rsg->get_scalar_points_written() << endl <<
                              "   est FP-ops per point:       " << rsg->get_scalar_fp_ops() << endl <<
                              "   var-reads in rank:          " << make_num_str(reads[bname]) << endl <<
                              "   var-writes in rank:         " << make_num_str(writes[bname]) << endl <<
                              "   est FP-ops in rank:         " << make_num_str(fpops[bname]));

                    auto& bb = rsg->get_bb();
                    DEBUG_MSG("   part scope:                 " << bb.make_range_string(domain_dims));
                    auto& bbs = rsg->get_bbs();
                    DEBUG_MSG("   num full rectangles in box: " << bbs.size());
                    for (size_t ri = 0; ri < bbs.size(); ri++) {
                        auto& rbb = bbs[ri];
                        DEBUG_MSG("    Rectangle " << ri << ":\n"
                                  "     num points in rect:       " << make_num_str(rbb.bb_size));
                        if (rbb.bb_size) {
                            DEBUG_MSG("     rect scope:               " << rbb.make_range_string(domain_dims) <<
                                      "\n     rect size:                " << rbb.make_len_string(domain_dims));
                        }
                    }
                }
            }

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

        } // parts.

        // Sum across ranks.
        tot_reads_per_step = env->sum_over_ranks(num_reads_per_step);
        tot_writes_per_step = env->sum_over_ranks(num_writes_per_step);
        tot_fpops_per_step = env->sum_over_ranks(num_fpops_per_step);

    } // init_work_stats().

} // namespace yask.
