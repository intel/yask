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

    // Calculate results within a block.
    // Typically called by an OMP thread team.
    void StencilGroupBase::calc_block(const ScanIndices& region_idxs) {

        auto opts = _generic_context->get_settings();
        auto dims = _generic_context->get_dims();
        int ndims = dims->_stencil_dims.size();
        auto& step_dim = dims->_step_dim;
        int thread_idx = omp_get_thread_num();
        TRACE_MSG3("calc_block:" <<
                   " in non-scratch group '" << get_name() << "': " <<
                   region_idxs.start.makeValStr(ndims) <<
                   " ... (end before) " << region_idxs.stop.makeValStr(ndims) <<
                   " by thread " << thread_idx);
        assert(!is_scratch());

        // Init block begin & end from region start & stop indices.
        ScanIndices def_block_idxs(ndims);
        def_block_idxs.initFromOuter(region_idxs);

        // Steps within a block are based on sub-block sizes.
        def_block_idxs.step = opts->_sub_block_sizes;

        // Groups in block loops are based on sub-block-group sizes.
        def_block_idxs.group_size = opts->_sub_block_group_sizes;

        // Indices needed for the generated loops.  Will normally be a copy
        // of def_block_idxs except when updating scratch-grids.
        ScanIndices block_idxs(ndims);
        
        // Define the groups that need to be processed in
        // this block. This will be the prerequisite scratch-grid
        // groups plus this non-scratch group.
        auto sg_list = get_scratch_deps();
        sg_list.push_back(this);

        // Set number of threads for a block.
        // This should be nested within a top-level OpenMP task.
        _generic_context->set_block_threads();

        // Loop through all the needed groups.
        for (auto* sg : sg_list) {

            // Copy of default indices for generated loop.
            block_idxs = def_block_idxs;
        
            // If this group is updating scratch grid(s),
            // expand indices to calculate values in halo.
            sg->adjust_scan(block_idxs);

            TRACE_MSG3("calc_block: " <<
                       " in group '" << sg->get_name() << "': " <<
                       block_idxs.begin.makeValStr(ndims) <<
                       " ... (end before) " << block_idxs.end.makeValStr(ndims) <<
                       " by thread " << thread_idx);
            
            // Include automatically-generated loop code that calls
            // calc_sub_block() for each sub-block in this block.  Loops through
            // x from begin_bx to end_bx-1; similar for y and z.  This
            // code typically contains the nested OpenMP loop(s).
#include "yask_block_loops.hpp"
        }
    }

    // Normalize the indices, i.e., divide by vector len in each dim.
    // Ranks offsets must already be subtracted.
    // Each dim in 'orig' must be a multiple of corresponding vec len.
    void StencilGroupBase::normalize_indices(const Indices& orig, Indices& norm) const {
        auto* cp = _generic_context;
        auto dims = cp->get_dims();
        int nsdims = dims->_stencil_dims.size();
        auto step_posn = Indices::step_posn;
        assert(orig.getNumDims() == nsdims);
        assert(norm.getNumDims() == nsdims);
        
        for (int i = 0, j = 0; i < nsdims; i++) {
            if (i != step_posn) {

                // Divide indices by fold lengths as needed by
                // read/writeVecNorm().  Use idiv_flr() instead of '/'
                // because begin/end vars may be negative (if in halo).
                norm[i] = idiv_flr<idx_t>(orig[i], dims->_fold_pts[j]);

                // Check for no remainder.
                assert(imod_flr<idx_t>(orig[i], dims->_fold_pts[j]) == 0);

                // Next domain index.
                j++;
            }
        }
    }
    
    // Calculate results for one sub-block.
    // Typically called by a single OMP thread.
    // The index ranges in 'block_idxs' are sub-divided
    // into full vector-clusters, full vectors, and sub-vectors
    // and finally evaluated by the YASK-compiler-generated loops.
    void StencilGroupBase::calc_sub_block(int thread_idx,
                                          const ScanIndices& block_idxs) {
        auto* cp = _generic_context;
        auto opts = cp->get_settings();
        auto dims = cp->get_dims();
        int nddims = dims->_domain_dims.size();
        int nsdims = dims->_stencil_dims.size();
        auto& step_dim = dims->_step_dim;
        auto step_posn = Indices::step_posn;
        TRACE_MSG3("calc_sub_block:" <<
                   " in group '" << get_name() << "': " <<
                   block_idxs.start.makeValStr(nsdims) <<
                   " ... (end before) " << block_idxs.stop.makeValStr(nsdims));

        // Init sub-block begin & end from block start & stop indices.
        // These indices are in element units and rank-relative.
        ScanIndices sub_block_idxs(nsdims);
        sub_block_idxs.initFromOuter(block_idxs);

        // Subset of sub-block that is full clusters.
        // These indices are in element units and rank-relative.
        ScanIndices sub_block_fcidxs(sub_block_idxs);

        // Subset of sub-block that is full vectors.
        // These indices are in element units and rank-relative.
        ScanIndices sub_block_fvidxs(sub_block_idxs);
        
        // Superset of sub-block that is full or partial vectors.
        // These indices are in element units and rank-relative.
        ScanIndices sub_block_vidxs(sub_block_idxs);

        // Masks for computing partial vectors in each dim.
        // Init to all-ones (no masking).
        Indices peel_masks(nsdims), rem_masks(nsdims);
        peel_masks.setFromConst(-1);
        rem_masks.setFromConst(-1);
        
        // Determine what part of this sub-block can be done with full clusters/vectors.
        // Init to full clusters only.
        bool do_clusters = true; // any clusters to do?
        bool do_vectors = false; // any vectors to do? (assume not)
        bool do_scalars = false; // any scalars to do? (assume not)
        bool scalar_for_peel_rem = false; // using the scalar code for peel and/or remainder.

        // If BB is not full, do whole block with
        // scalar code.  We should only get here when using sub-domains.
        // TODO: do as much vectorization as possible-- this current code is
        // functionally correct but very poor perf.
        if (!bb_is_full) {

            do_clusters = false;
            do_vectors = false;
            do_scalars = true;
            scalar_for_peel_rem = false;

            // None of these will be used.
            sub_block_fcidxs.begin.setFromConst(0);
            sub_block_fcidxs.end.setFromConst(0);
            sub_block_fvidxs.begin.setFromConst(0);
            sub_block_fvidxs.end.setFromConst(0);
            sub_block_vidxs.begin.setFromConst(0);
            sub_block_vidxs.end.setFromConst(0);
        }

        // If it isn't guaranteed that this block is only aligned cluster
        // mults in all dims, determine the subset of this sub-block that is
        // clusters/vectors.  TODO: pre-calc this info for each block.
        else if (is_scratch() || !bb_is_cluster_mult || !bb_is_aligned) {

            // (i: index for stencil dims, j: index for domain dims).
            for (int i = 0, j = 0; i < nsdims; i++) {
                if (i != step_posn) {

                    // Rank offset.
                    auto rofs = cp->rank_domain_offsets[j];
                
                    // Begin/end of rank-relative scalar elements in this dim.
                    auto ebgn = sub_block_idxs.begin[i] - rofs;
                    auto eend = sub_block_idxs.end[i] - rofs;

                    // Find range of whole clusters.
                    // Note that fcend <= eend because we round
                    // down to get whole clusters only.
                    auto cpts = dims->_cluster_pts[j];
                    auto fcbgn = round_up_flr(ebgn, cpts);
                    auto fcend = round_down_flr(eend, cpts);
                    sub_block_fcidxs.begin[i] = fcbgn;
                    sub_block_fcidxs.end[i] = fcend;

                    // Find range of whole or partial vectors.
                    // Note that fvend <= eend because we round
                    // down to get whole vectors only.
                    // Note that vend >= eend because we round
                    // up to include partial vectors.
                    // Similar but opposite for begin vars.
                    // We make a vector mask to pick the
                    // right elements.
                    auto vpts = dims->_fold_pts[j];
                    auto fvbgn = round_up_flr(ebgn, vpts);
                    auto fvend = round_down_flr(eend, vpts);
                    auto vbgn = round_down_flr(ebgn, vpts);
                    auto vend = round_up_flr(eend, vpts);
                    if (i == _inner_posn) {

                        // Don't do any vectors in plane of inner dim.
                        // We'll do these with scalars.
                        // This is unusual because vector folding is
                        // normally done in a plane perpendicular to
                        // the inner dim for >= 2D domains.
                        fvbgn = vbgn = fcbgn;
                        fvend = vend = fcend;
                    }
                    sub_block_fvidxs.begin[i] = fvbgn;
                    sub_block_fvidxs.end[i] = fvend;
                    sub_block_vidxs.begin[i] = vbgn;
                    sub_block_vidxs.end[i] = vend;

                    // Any vectors to do (full or partial)?
                    if (vbgn < fcbgn || vend > fcend)
                        do_vectors = true;

                    // Calculate peel mask in this dim for partial vector.
                    // All such masks will be ANDed together to form the
                    // final mask over all domain dims.
                    if (vbgn < fvbgn) {
                        idx_t mask = 0;

                        // Need to set upper bit.
                        idx_t mbit = 0x1 << (dims->_fold_pts.product() - 1);
                        
                        // Visit points in a vec-fold.
                        dims->_fold_pts.visitAllPoints
                            ([&](const IdxTuple& pt, size_t idx) {

                                // Shift mask to next posn.
                                mask >>= 1;
                                
                                // If this point is within the sub-block,
                                // set the next bit in the mask.
                                idx_t pi = vbgn + pt[j];
                                if (pi >= ebgn)
                                    mask |= mbit;

                                // Keep visiting.
                                return true;
                            });

                        // Save mask.
                        peel_masks[i] = mask;
                    }

                    // Calculate remainder mask in this dim for partial vector.
                    // All such masks will be ANDed together to form the
                    // final mask over all domain dims.
                    if (vend > fvend) {
                        idx_t mask = 0;

                        // Need to set upper bit.
                        idx_t mbit = 0x1 << (dims->_fold_pts.product() - 1);
                        
                        // Visit points in a vec-fold.
                        dims->_fold_pts.visitAllPoints
                            ([&](const IdxTuple& pt, size_t idx) {

                                // Shift mask to next posn.
                                mask >>= 1;
                                
                                // If this point is within the sub-block,
                                // put a 1 in the mask.
                                idx_t pi = fvend + pt[j];
                                if (pi < eend)
                                    mask |= mbit;

                                // Keep visiting.
                                return true;
                            });

                        // Save mask.
                        rem_masks[i] = mask;
                    }

                    // Any remainder?
                    // This will only be needed in inner dim because we
                    // will do partial vectors in other dims.
                    // Set 'scalar_for_peel_rem' to indicate we only want to
                    // do peel and/or rem in scalar loop.
                    if (i == _inner_posn && (ebgn < vbgn || eend > vend)) {
                        do_scalars = true;
                        scalar_for_peel_rem = true;
                    }

                    // Next domain index.
                    j++;
                }
            }
        }

        // Normalized indices needed for sub-block loop.
        ScanIndices norm_sub_block_idxs(sub_block_idxs);
        
        // Normalize the cluster indices.
        // These will be the bounds of the sub-block loops.
        // Set both begin/end and start/stop to ensure start/stop
        // vars get passed through to calc_loop_of_clusters()
        // for the inner loop.
        normalize_indices(sub_block_fcidxs.begin, norm_sub_block_idxs.begin);
        norm_sub_block_idxs.start = norm_sub_block_idxs.begin;
        normalize_indices(sub_block_fcidxs.end, norm_sub_block_idxs.end);
        norm_sub_block_idxs.stop = norm_sub_block_idxs.end;
        
        // Full rectangular polytope of aligned clusters: use optimized code.
        if (do_clusters) {
            TRACE_MSG3("calc_sub_block:  using cluster code for " <<
                       sub_block_fcidxs.begin.makeValStr(nsdims) <<
                       " ... (end before) " << sub_block_fcidxs.end.makeValStr(nsdims));

            // Step sizes are based on cluster lengths (in vector units).
            // The step in the inner loop is hard-coded in the generated code.
            for (int i = 0, j = 0; i < nsdims; i++) {
                if (i != step_posn) {
                    norm_sub_block_idxs.step[i] = dims->_cluster_mults[j];
                    j++;
                }
            }

            // Define the function called from the generated loops
            // to simply call the loop-of-clusters functions.
#define calc_inner_loop(thread_idx, loop_idxs) calc_loop_of_clusters(thread_idx, loop_idxs)

            // Include automatically-generated loop code that calls
            // calc_inner_loop(). This is different from the higher-level
            // loops because it does not scan the inner dim.
#include "yask_sub_block_loops.hpp"
#undef calc_inner_loop
        }
        
        // Full and partial peel/remainder vectors.
        if (do_vectors) {
            TRACE_MSG3("calc_sub_block:  using vector code for " <<
                       sub_block_vidxs.begin.makeValStr(nsdims) <<
                       " ... (end before) " << sub_block_vidxs.end.makeValStr(nsdims) <<
                       " before and/or after full vector-clusters in " <<
                       sub_block_fcidxs.begin.makeValStr(nsdims) <<
                       " ... (end before) " << sub_block_fcidxs.end.makeValStr(nsdims));

            // Keep a copy of the normalized cluster indices
            // that were calculated above.
            // The full clusters were already done above, so
            // we only need to do vectors before or after the
            // clusters in each dim.
            // We'll exclude them below.
            ScanIndices norm_sub_block_fcidxs(norm_sub_block_idxs);

            // Normalize the vector indices.
            // These will be the bounds of the sub-block loops.
            // Set both begin/end and start/stop to ensure start/stop
            // vars get passed through to calc_loop_of_clusters()
            // for the inner loop.
            normalize_indices(sub_block_vidxs.begin, norm_sub_block_idxs.begin);
            norm_sub_block_idxs.start = norm_sub_block_idxs.begin;
            normalize_indices(sub_block_vidxs.end, norm_sub_block_idxs.end);
            norm_sub_block_idxs.stop = norm_sub_block_idxs.end;

            // Step sizes are one vector.
            // The step in the inner loop is hard-coded in the generated code.
            for (int i = 0, j = 0; i < nsdims; i++) {
                if (i != step_posn) {
                    norm_sub_block_idxs.step[i] = 1;
                    j++;
                }
            }

            // Also normalize the *full* vector indices to determine if
            // we need a mask at each vector index.
            // We don't need start, stop, or step for this.
            ScanIndices norm_sub_block_fvidxs(sub_block_idxs);
            normalize_indices(sub_block_fvidxs.begin, norm_sub_block_fvidxs.begin);
            normalize_indices(sub_block_fvidxs.end, norm_sub_block_fvidxs.end);

            // Define the function called from the generated loops to
            // determine whether a loop of vectors is within the peel
            // range (before the cluster) or remainder
            // range (after the clusters). If so, call the
            // loop-of-vectors function w/appropriate mask.
            // Since step is always 1, we ignore loop_idxs.stop.
#define calc_inner_loop(thread_idx, loop_idxs) \
            bool ok = false;                                            \
            idx_t mask = idx_t(-1);                                     \
            for (int i = 0; i < nsdims; i++) {                          \
                if (i != step_posn &&                                   \
                    i != _inner_posn &&                                 \
                    (loop_idxs.start[i] < norm_sub_block_fcidxs.begin[i] || \
                     loop_idxs.start[i] >= norm_sub_block_fcidxs.end[i])) { \
                        ok = true;                                      \
                        if (loop_idxs.start[i] < norm_sub_block_fvidxs.begin[i]) \
                            mask &= peel_masks[i];                      \
                        if (loop_idxs.start[i] >= norm_sub_block_fvidxs.end[i]) \
                            mask &= rem_masks[i];                       \
                }                                                       \
            }                                                           \
            if (ok) calc_loop_of_vectors(thread_idx, loop_idxs, mask);

            // Include automatically-generated loop code that calls
            // calc_inner_loop(). This is different from the higher-level
            // loops because it does not scan the inner dim.
#include "yask_sub_block_loops.hpp"
#undef calc_inner_loop
        }
        
        // Use scalar code for anything not done above.
        if (do_scalars) {

#ifdef TRACE
            string msg = "calc_sub_block:  using scalar code for ";
            msg += scalar_for_peel_rem ? "peel/remainder of" : "entire";
            msg += " sub-block ";
            msg += bb_is_full ? "without" : "with";
            msg += " sub-domain checking";
            TRACE_MSG3(msg);
#endif

            // Use the 'misc' loop. The OMP will be ignored because we're already in
            // a nested OMP region. TODO: check this if there is only one block thread.
            ScanIndices misc_idxs(sub_block_idxs);

            // Define misc-loop function.
            // If point is in sub-domain for this
            // group, then evaluate the reference scalar code.
            // If no holes, don't need to check each point in domain.
            // Since step is always 1, we ignore misc_idxs.stop.
#define misc_fn(misc_idxs)  do {                                        \
            bool ok = true;                                             \
            if (scalar_for_peel_rem) {                                  \
                ok = false;                                             \
                for (int i = 0, j = 0; i < nsdims; i++) {               \
                    if (i != step_posn) {                               \
                        auto rofs = cp->rank_domain_offsets[j];         \
                        if (misc_idxs.start[i] < rofs + sub_block_vidxs.begin[i] || \
                            misc_idxs.start[i] >= rofs + sub_block_vidxs.end[i]) { \
                            ok = true; break; }                         \
                        j++;                                            \
                    }                                                   \
                }                                                       \
            }                                                           \
            if (ok && (bb_is_full || is_in_valid_domain(misc_idxs.start))) { \
                calc_scalar(thread_idx, misc_idxs.start);               \
            }                                                           \
        } while(0)

            // Scan through n-D space.
#include "yask_misc_loops.hpp"
#undef misc_fn
        }

        // Make sure streaming stores are visible for later loads.
        make_stores_visible();

    } // calc_sub_block.

    // Calculate a series of cluster results within an inner loop.
    // The 'loop_idxs' must specify a range only in the inner dim.
    // Indices must be rank-relative.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    void StencilGroupBase::calc_loop_of_clusters(int thread_idx,
                                                 const ScanIndices& loop_idxs) {
        auto* cp = _generic_context;
        auto dims = cp->get_dims();
        int nsdims = dims->_stencil_dims.size();
        auto step_posn = Indices::step_posn;
        TRACE_MSG3("calc_loop_of_clusters: local vector-indices " <<
                   loop_idxs.start.makeValStr(nsdims) <<
                   " ... (end before) " << loop_idxs.stop.makeValStr(nsdims));

#ifdef DEBUG
        // Check that only the inner dim has a range greater than one cluster.
        for (int i = 0, j = 0; i < nsdims; i++) {
            if (i != step_posn) {
                if (i != _inner_posn)
                    assert(loop_idxs.start[i] + dims->_cluster_mults[j] >=
                           loop_idxs.stop[i]);
                j++;
            }
        }
#endif

        // Need all starting indices.
        const Indices& start_idxs = loop_idxs.start;

        // Need stop for inner loop only.
        idx_t stop_inner = loop_idxs.stop[_inner_posn];

        // Call code from stencil compiler.
        calc_loop_of_clusters(thread_idx, start_idxs, stop_inner);
    }

    // Calculate a series of vector results within an inner loop.
    // The 'loop_idxs' must specify a range only in the inner dim.
    // Indices must be rank-relative.
    // Indices must be normalized, i.e., already divided by VLEN_*.
void StencilGroupBase::calc_loop_of_vectors(int thread_idx,
                                            const ScanIndices& loop_idxs,
                                            idx_t write_mask) {
        auto* cp = _generic_context;
        auto dims = cp->get_dims();
        int nsdims = dims->_stencil_dims.size();
        auto step_posn = Indices::step_posn;
        TRACE_MSG3("calc_loop_of_vectors: local vector-indices " <<
                   loop_idxs.start.makeValStr(nsdims) <<
                   " ... (end before) " << loop_idxs.stop.makeValStr(nsdims) <<
                   " w/write-mask = 0x" << hex << write_mask << dec);

#ifdef DEBUG
        // Check that only the inner dim has a range greater than one vector.
        for (int i = 0; i < nsdims; i++) {
            if (i != step_posn && i != _inner_posn)
                assert(loop_idxs.start[i] + 1 >= loop_idxs.stop[i]);
        }
#endif

        // Need all starting indices.
        const Indices& start_idxs = loop_idxs.start;

        // Need stop for inner loop only.
        idx_t stop_inner = loop_idxs.stop[_inner_posn];

        // Call code from stencil compiler.
        calc_loop_of_vectors(thread_idx, start_idxs, stop_inner, write_mask);
    }

    // If this group is updating scratch grid(s),
    // expand indices to calculate values in halo.
    bool StencilGroupBase::adjust_scan(ScanIndices& idxs) const {

        // Adjust scan based on first grid's halo.
        for (auto* sv : outputScratchVecs) {
            for (auto gp : *sv) {

                assert(gp->is_scratch());

                // Loop thru dims.
                // TODO: cache these values in the object so we don't need to
                // do this expensive scan.
                auto dims = get_dims();
                int i = 0;
                for (auto& dim : dims->_stencil_dims.getDims()) {
                    auto& dname = dim.getName();

                    // Get halo from grid in this dim.
                    int posn = gp->get_dim_posn(dname);
                    if (posn >= 0) {
                        idx_t lh = gp->get_left_halo_size(posn);
                        idx_t rh = gp->get_right_halo_size(posn);

                        // Adjust scan indices.
                        idxs.begin[i] -= lh;
                        idxs.end[i] += rh;
                    }
                    i++;
                }

                // Only need data for first grid since all halos should be same.
                return true;
            }
        }
        return false;
    }
    
    // Set the bounding-box vars for this group in this rank.
    void StencilGroupBase::find_bounding_box() {
        StencilContext& context = *_generic_context;
        ostream& os = context.get_ostr();
        auto settings = context.get_settings();
        auto dims = context.get_dims();
        auto& domain_dims = dims->_domain_dims;
        auto& step_dim = dims->_step_dim;
        auto& stencil_dims = dims->_stencil_dims;
        auto ndims = stencil_dims.size();

        // Init min vars w/max val and vice-versa.
        Indices min_pts(idx_max, ndims);
        Indices max_pts(idx_min, ndims);
        idx_t npts = 0;

        // Begin, end tuples.
        // Scan across domain in this rank.
        IdxTuple begin(stencil_dims);
        begin.setVals(context.rank_domain_offsets, false);
        begin[step_dim] = 0;
        IdxTuple end = begin.addElements(settings->_rank_sizes);
        end[step_dim] = 1;      // one time-step only.

        // Indices needed for the generated 'misc' loops.
        ScanIndices misc_idxs(ndims);
        misc_idxs.begin = begin;
        misc_idxs.end = end;

        // Define misc-loop function.  Since step is always 1, we ignore
        // misc_stop.  Update only if point is in domain for this group.
#define misc_fn(misc_idxs) do {                                  \
        if (is_in_valid_domain(misc_idxs.start)) {               \
            min_pts = min_pts.minElements(misc_idxs.start);      \
            max_pts = max_pts.maxElements(misc_idxs.start);      \
            npts++; \
        } } while(0)

        // Define OMP reductions to be used in generated code.
#ifdef OMP_PRAGMA_SUFFIX
#undef OMP_PRAGMA_SUFFIX
#endif
#define OMP_PRAGMA_SUFFIX reduction(+:npts)     \
            reduction(min_idxs:min_pts)         \
            reduction(max_idxs:max_pts)

        // Scan through n-D space.  This scan sets min_pts & max_pts for all
        // stencil dims (including step dim) and npts to the number of valid
        // points.
#include "yask_misc_loops.hpp"
#undef misc_fn
#undef OMP_PRAGMA_SUFFIX

        // Init bb vars to ensure they contain correct dims.
        bb_begin = domain_dims;
        bb_end = domain_dims;
        
        // If any points, set begin vars to min indices and end vars to one
        // beyond max indices.
        if (npts) {
            IdxTuple tmp(stencil_dims); // create tuple w/stencil dims.
            min_pts.setTupleVals(tmp);  // convert min_pts to tuple.
            bb_begin.setVals(tmp, false); // set bb_begin to domain dims of min_pts.

            max_pts.setTupleVals(tmp); // convert min_pts to tuple.
            bb_end.setVals(tmp, false); // set bb_end to domain dims of max_pts.
            bb_end = bb_end.addElements(1); // end = last + 1.
        }

        // No points, just set to zero.
        else {
            bb_begin.setValsSame(0);
            bb_end.setValsSame(0);
        }
        bb_num_points = npts;
        
        // Finalize BB.
        update_bb(os, get_name(), context);
    }
    
} // namespace yask.
