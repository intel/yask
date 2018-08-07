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

// This file contains implementations of StencilBundleBase methods.
// Also see context_setup.cpp.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Calculate results within a block defined by 'def_block_idxs'.
    // It is here that any required scratch-grid stencils are evaluated
    // first and then the non-scratch stencils in the stencil bundle.
    void StencilBundleBase::calc_block(const ScanIndices& def_block_idxs) {

        auto& opts = _generic_context->get_settings();
        auto& dims = _generic_context->get_dims();
        int nsdims = dims->_stencil_dims.size();
        auto& step_dim = dims->_step_dim;
        auto step_posn = Indices::step_posn;
        int thread_idx = omp_get_thread_num(); // used to index the scratch grids.
        TRACE_MSG3("calc_block for bundle '" << get_name() << "': [" <<
                   def_block_idxs.begin.makeValStr(nsdims) <<
                   " ... " << def_block_idxs.end.makeValStr(nsdims) <<
                   ") by " << def_block_idxs.step.makeValStr(nsdims) <<
                   " in thread " << thread_idx);
        assert(!is_scratch());

        // TODO: if >1 BB, check outer one first to save time.
        
        // Loop through each solid BB.
        // For each BB, calc intersection between it and 'def_block_idxs'.
        // If this is non-empty, apply the bundle to all its required sub-blocks.
        TRACE_MSG3("calc_block for bundle '" << get_name() << "': checking " <<
                   _bb_list.size() << " BB(s)");
        int bbn = 0;
  	for (auto& bb : _bb_list) {
            bbn++;
            bool bb_ok = true;

            // Trim the default block indices based on the bounding box(es)
            // for this bundle.
            ScanIndices bb_idxs(def_block_idxs);
            for (int i = 0, j = 0; i < nsdims; i++) {
                if (i == step_posn) continue;

                // Begin point.
                auto bbegin = max(def_block_idxs.begin[i], bb.bb_begin[j]);
                bb_idxs.begin[i] = bbegin;

                // End point.
                auto bend = min(def_block_idxs.end[i], bb.bb_end[j]);
                bb_idxs.end[i] = bend;
		
                // Anything to do?
                if (bend <= bbegin) {
                    bb_ok = false;
                    break;
                }
                j++;            // next domain index.
            }

            // nothing to do?
            if (!bb_ok) {
                TRACE_MSG3("calc_block for bundle '" << get_name() <<
                           "': no overlap between bundle " << bbn << " and current block");
                continue; // to next BB.
            }
            
            TRACE_MSG3("calc_block for bundle '" << get_name() <<
                       "': after trimming for BB " << bbn << ": [" <<
                       bb_idxs.begin.makeValStr(nsdims) <<
                       " ... " << bb_idxs.end.makeValStr(nsdims) << ")");

            // Update offsets of scratch grids based on this bundle's location.
            _generic_context->update_scratch_grid_info(thread_idx, bb_idxs.begin);

            // Get the bundles that need to be processed in
            // this block. This will be any prerequisite scratch-grid
            // bundles plus this non-scratch bundle.
            auto sg_list = get_reqd_bundles();

            // Set number of threads for a block.
            // Each of these threads will work on a sub-block.
            // This should be nested within a top-level OpenMP task.
            _generic_context->set_block_threads();

            // Loop through all the needed bundles.
            for (auto* sg : sg_list) {

                // Indices needed for the generated loops.  Will normally be a
                // copy of 'bb_idxs' except when updating scratch-grids.
                ScanIndices block_idxs = sg->adjust_span(thread_idx, bb_idxs);

                TRACE_MSG3("calc_block for bundle '" << get_name() << "': " <<
                           " in reqd bundle '" << sg->get_name() << "': [" <<
                           block_idxs.begin.makeValStr(nsdims) <<
                           " ... " << block_idxs.end.makeValStr(nsdims) <<
                           ") in thread " << thread_idx);

                // Include automatically-generated loop code that calls
                // calc_sub_block() for each sub-block in this block. This
                // code typically contains the nested OpenMP loop(s).
#include "yask_block_loops.hpp"
            }
        } // BB list.
    }

    // Normalize the indices, i.e., divide by vector len in each dim.
    // Ranks offsets must already be subtracted because rank offsets
    // are not necessarily vec-multiples.
    // Each dim in 'orig' must be a multiple of corresponding vec len.
    void StencilBundleBase::normalize_indices(const Indices& orig, Indices& norm) const {
        auto* cp = _generic_context;
        auto& dims = cp->get_dims();
        int nsdims = dims->_stencil_dims.size();
        auto step_posn = Indices::step_posn;
        assert(orig.getNumDims() == nsdims);
        assert(norm.getNumDims() == nsdims);

        // i: index for stencil dims, j: index for domain dims.
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
    void StencilBundleBase::calc_sub_block(int thread_idx,
                                           const ScanIndices& block_idxs) {
        auto* cp = _generic_context;
        auto& opts = cp->get_settings();
        auto& dims = cp->get_dims();
        int nddims = dims->_domain_dims.size();
        int nsdims = dims->_stencil_dims.size();
        auto& step_dim = dims->_step_dim;
        auto step_posn = Indices::step_posn;
        TRACE_MSG3("calc_sub_block for reqd bundle '" << get_name() << "': [" <<
                   block_idxs.start.makeValStr(nsdims) <<
                   " ... " << block_idxs.stop.makeValStr(nsdims) << ")");

        /*
          Indices in each domain dim:

          sub_block_eidxs.begin                      rem_masks used here
          |peel_masks used here                      | sub_block_eidxs.end
          ||                                         | |
          vv                                         v v
          |---|-------|---------------------------|---|---|  <- "|" on vec boundaries.
          ^   ^       ^                            ^   ^   ^
          |   |       |                            |   |   |
          |   |       sub_block_fcidxs.begin       |   |   sub_block_vidxs.end
          |   sub_block_fvidxs.begin               |   sub_block_fvidxs.end
          sub_block_vidxs.begin                    sub_block_fcidxs.end
        */

        // Init sub-block begin & end from block start & stop indices.
        // These indices are in element units and global (NOT rank-relative).
        ScanIndices sub_block_idxs(*dims, true, 0);
        sub_block_idxs.initFromOuter(block_idxs);

        // Sub block indices in element units and rank-relative.
        ScanIndices sub_block_eidxs(sub_block_idxs);

        // Subset of sub-block that is full clusters.
        // These indices are in element units and rank-relative.
        ScanIndices sub_block_fcidxs(sub_block_idxs);

        // Subset of sub-block that is full vectors.
        // These indices are in element units and rank-relative.
        ScanIndices sub_block_fvidxs(sub_block_idxs);

        // Superset of sub-block that is full or partial (masked) vectors.
        // These indices are in element units and rank-relative.
        ScanIndices sub_block_vidxs(sub_block_idxs);

        // These will be set to rank-relative, so set ofs to zero.
        sub_block_eidxs.align_ofs.setFromConst(0);
        sub_block_fcidxs.align_ofs.setFromConst(0);
        sub_block_fvidxs.align_ofs.setFromConst(0);
        sub_block_vidxs.align_ofs.setFromConst(0);

        // Masks for computing partial vectors in each dim.
        // Init to all-ones (no masking).
        Indices peel_masks(nsdims), rem_masks(nsdims);
        peel_masks.setFromConst(-1);
        rem_masks.setFromConst(-1);

        // Flags that indicate what type of processing needs to be done.
        bool do_clusters = false; // any clusters to do?
        bool do_vectors = false; // any vectors to do? (assume not)
        bool do_scalars = false; // any scalars to do? (assume not)
        bool scalar_for_peel_rem = false; // using the scalar code for peel and/or remainder.

        // Do only scalar code--no clusters or vectors--for debug.
#ifdef FORCE_SCALAR
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
    
        // Adjust indices to be rank-relative.
        // Determine the subset of this sub-block that is
        // clusters, vectors, and partial vectors.
#else
        do_clusters = true;
        do_vectors = false;
        do_scalars = false;

        // i: index for stencil dims, j: index for domain dims.
        for (int i = 0, j = 0; i < nsdims; i++) {
            if (i != step_posn) {

                // Rank offset.
                auto rofs = cp->rank_domain_offsets[j];

                // Begin/end of rank-relative scalar elements in this dim.
                auto ebgn = sub_block_idxs.begin[i] - rofs;
                auto eend = sub_block_idxs.end[i] - rofs;
                sub_block_eidxs.begin[i] = ebgn;
                sub_block_eidxs.end[i] = eend;

                // Find range of full clusters.
                // Note that fcend <= eend because we round
                // down to get whole clusters only.
                // Similarly, fcbgn >= ebgn.
                auto cpts = dims->_cluster_pts[j];
                auto fcbgn = round_up_flr(ebgn, cpts);
                auto fcend = round_down_flr(eend, cpts);
                sub_block_fcidxs.begin[i] = fcbgn;
                sub_block_fcidxs.end[i] = fcend;

                // Any clusters to do?
                if (fcend <= fcbgn)
                    do_clusters = false;

                // If anything before or after clusters, continue with
                // setting vector indices and peel/rem masks.
                if (fcbgn > ebgn || fcend < eend) {

                    // Find range of full and/or partial vectors.
                    // Note that fvend <= eend because we round
                    // down to get whole vectors only.
                    // Note that vend >= eend because we round
                    // up to include partial vectors.
                    // Similar but opposite for begin vars.
                    // We make a vector mask to pick the
                    // right elements.
                    // TODO: use compile-time consts instead
                    // of _fold_pts for more efficiency.
                    auto vpts = dims->_fold_pts[j];
                    auto fvbgn = round_up_flr(ebgn, vpts);
                    auto fvend = round_down_flr(eend, vpts);
                    auto vbgn = round_down_flr(ebgn, vpts);
                    auto vend = round_up_flr(eend, vpts);
                    if (i == _inner_posn) {

                        // Don't do any full and/or partial vectors in
                        // plane of inner dim.  We'll do these with
                        // scalars.  This is unusual because vector
                        // folding is normally done in a plane
                        // perpendicular to the inner dim for >= 2D
                        // domains.
                        fvbgn = vbgn = fcbgn;
                        fvend = vend = fcend;
                    }
                    sub_block_fvidxs.begin[i] = fvbgn;
                    sub_block_fvidxs.end[i] = fvend;
                    sub_block_vidxs.begin[i] = vbgn;
                    sub_block_vidxs.end[i] = vend;

                    // Any vectors to do (full and/or partial)?
                    if (vbgn < fcbgn || vend > fcend)
                        do_vectors = true;

                    // Calculate masks in this dim for partial vectors.
                    // All such masks will be ANDed together to form the
                    // final masks over all domain dims.
                    // Example: assume folding is x=4*y=4.
                    // Possible 'x' peel mask to exclude 1st 2 cols:
                    //   0 0 1 1
                    //   0 0 1 1
                    //   0 0 1 1
                    //   0 0 1 1
                    // Possible 'y' peel mask to exclude 1st row:
                    //   0 0 0 0
                    //   1 1 1 1
                    //   1 1 1 1
                    //   1 1 1 1
                    // Along 'x' face, the 'x' peel mask is used.
                    // Along 'y' face, the 'y' peel mask is used.
                    // Along an 'x-y' edge, they are ANDed to make this mask:
                    //   0 0 0 0
                    //   0 0 1 1
                    //   0 0 1 1
                    //   0 0 1 1
                    // so that the 6 corner elements are updated.

                    if (vbgn < fvbgn || vend > fvend) {
                        idx_t pmask = 0, rmask = 0;

                        // Need to set upper bit.
                        idx_t mbit = 0x1 << (dims->_fold_pts.product() - 1);

                        // Visit points in a vec-fold.
                        dims->_fold_pts.visitAllPoints
                            ([&](const IdxTuple& pt, size_t idx) {

                                // Shift masks to next posn.
                                pmask >>= 1;
                                rmask >>= 1;

                                // If the peel point is within the sub-block,
                                // set the next bit in the mask.
                                idx_t pi = vbgn + pt[j];
                                if (pi >= ebgn)
                                    pmask |= mbit;

                                // If the rem point is within the sub-block,
                                // put a 1 in the mask.
                                pi = fvend + pt[j];
                                if (pi < eend)
                                    rmask |= mbit;

                                // Keep visiting.
                                return true;
                            });

                        // Save masks in this dim.
                        peel_masks[i] = pmask;
                        rem_masks[i] = rmask;
                    }

                    // Anything not covered?
                    // This will only be needed in inner dim because we
                    // will do partial vectors in other dims.
                    // Set 'scalar_for_peel_rem' to indicate we only want to
                    // do peel and/or rem in scalar loop.
                    if (i == _inner_posn && (ebgn < vbgn || eend > vend)) {
                        do_scalars = true;
                        scalar_for_peel_rem = true;
                    }
                }

                // If no peel or rem, just set vec indices to same as
                // full cluster.
                else {
                    sub_block_fvidxs.begin[i] = fcbgn;
                    sub_block_fvidxs.end[i] = fcend;
                    sub_block_vidxs.begin[i] = fcbgn;
                    sub_block_vidxs.end[i] = fcend;
                }

                // Next domain index.
                j++;
            }
        }
#endif
            
        // Normalized indices needed for sub-block loop.
        ScanIndices norm_sub_block_idxs(sub_block_eidxs);

        // Normalize the cluster indices.
        // These will be the bounds of the sub-block loops.
        // Set both begin/end and start/stop to ensure start/stop
        // vars get passed through to calc_loop_of_clusters()
        // for the inner loop.
        normalize_indices(sub_block_fcidxs.begin, norm_sub_block_idxs.begin);
        norm_sub_block_idxs.start = norm_sub_block_idxs.begin;
        normalize_indices(sub_block_fcidxs.end, norm_sub_block_idxs.end);
        norm_sub_block_idxs.stop = norm_sub_block_idxs.end;
        norm_sub_block_idxs.align.setFromConst(1); // one vector.

        // Full rectilinear polytope of aligned clusters: use optimized code.
        if (do_clusters) {
            TRACE_MSG3("calc_sub_block:  using cluster code for [" <<
                       sub_block_fcidxs.begin.makeValStr(nsdims) <<
                       " ... " << sub_block_fcidxs.end.makeValStr(nsdims) << ")");

            // Step sizes are based on cluster lengths (in vector units).
            // The step in the inner loop is hard-coded in the generated code.
            for (int i = 0, j = 0; i < nsdims; i++) {
                if (i == step_posn) continue;
                norm_sub_block_idxs.step[i] = dims->_cluster_mults[j]; // N vecs.
                j++;
            }

            // Define the function called from the generated loops
            // to simply call the loop-of-clusters functions.
#define calc_inner_loop(thread_idx, loop_idxs)                  \
            calc_loop_of_clusters(thread_idx, loop_idxs)

            // Include automatically-generated loop code that calls
            // calc_inner_loop(). This is different from the higher-level
            // loops because it does not scan the inner dim.
#include "yask_sub_block_loops.hpp"
#undef calc_inner_loop
        }

        // Full and partial peel/remainder vectors.
        if (do_vectors) {
            TRACE_MSG3("calc_sub_block:  using vector code for [" <<
                       sub_block_vidxs.begin.makeValStr(nsdims) <<
                       " ... " << sub_block_vidxs.end.makeValStr(nsdims) <<
                       ") *not* within full vector-clusters at [" <<
                       sub_block_fcidxs.begin.makeValStr(nsdims) <<
                       " ... " << sub_block_fcidxs.end.makeValStr(nsdims) << ")");

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
            norm_sub_block_idxs.step.setFromConst(1);

            // Also normalize the *full* vector indices to determine if
            // we need a mask at each vector index.
            // We just need begin and end indices for this.
            ScanIndices norm_sub_block_fvidxs(sub_block_eidxs);
            normalize_indices(sub_block_fvidxs.begin, norm_sub_block_fvidxs.begin);
            normalize_indices(sub_block_fvidxs.end, norm_sub_block_fvidxs.end);
            norm_sub_block_fvidxs.align.setFromConst(1); // one vector.

            // Define the function called from the generated loops to
            // determine whether a loop of vectors is within the peel
            // range (before the cluster) and/or remainder
            // range (after the clusters). If so, call the
            // loop-of-vectors function w/appropriate mask.
            // See the mask diagrams above that show how the
            // masks are ANDed together.
            // Since step is always 1, we ignore loop_idxs.stop.
#define calc_inner_loop(thread_idx, loop_idxs)                          \
            bool ok = false;                                            \
            idx_t mask = idx_t(-1);                                     \
            for (int i = 0; i < nsdims; i++) {                          \
                if (i != step_posn &&                                   \
                    i != _inner_posn &&                                 \
                    (loop_idxs.start[i] < norm_sub_block_fcidxs.begin[i] || \
                     loop_idxs.start[i] >= norm_sub_block_fcidxs.end[i])) { \
                    ok = true;                                          \
                    if (loop_idxs.start[i] < norm_sub_block_fvidxs.begin[i]) \
                        mask &= peel_masks[i];                          \
                    if (loop_idxs.start[i] >= norm_sub_block_fvidxs.end[i]) \
                        mask &= rem_masks[i];                           \
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

            // Use the 'misc' loops. Indices for these loops will be scalar and
            // global rather than normalized as in the cluster and vector loops.
            ScanIndices misc_idxs(sub_block_idxs);
            
            // Step sizes and alignment are one element.
            misc_idxs.step.setFromConst(1);
            misc_idxs.align.setFromConst(1);

            TRACE_MSG3((scalar_for_peel_rem ? "peel/remainder of" : "entire") <<
                       " sub-block [" << misc_idxs.begin.makeValStr(nsdims) <<
                       " ... " << misc_idxs.end.makeValStr(nsdims) << ")");

            // Define misc-loop function.
            // If point is in sub-domain for this
            // bundle, then evaluate the reference scalar code.
            // If no holes, don't need to check each point in domain.
            // Since step is always 1, we ignore misc_idxs.stop.
#define misc_fn(pt_idxs)  do {                                          \
                TRACE_MSG3("calc_sub_block:   at pt " << pt_idxs.start.makeValStr(nsdims)); \
                bool ok = false;                                        \
                if (scalar_for_peel_rem) {                              \
                    for (int i = 0, j = 0; i < nsdims; i++) {           \
                        if (i == step_posn) continue;                   \
                        auto rofs = cp->rank_domain_offsets[j];         \
                        if (pt_idxs.start[i] < rofs + sub_block_vidxs.begin[i] || \
                            pt_idxs.start[i] >= rofs + sub_block_vidxs.end[i]) { \
                            ok = true; break; }                         \
                        j++;                                            \
                    }                                                   \
                }                                                       \
                else ok = is_in_valid_domain(pt_idxs.start);            \
                if (ok) calc_scalar(thread_idx, pt_idxs.start);         \
            } while(0)

            // Scan through n-D space.
            // The OMP in the misc loops will be ignored if we're already in
            // the max allowed nested OMP region.
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
    void StencilBundleBase::calc_loop_of_clusters(int thread_idx,
                                                  const ScanIndices& loop_idxs) {
        auto* cp = _generic_context;
        auto& dims = cp->get_dims();
        int nsdims = dims->_stencil_dims.size();
        auto step_posn = Indices::step_posn;
        TRACE_MSG3("calc_loop_of_clusters: local vector-indices [" <<
                   loop_idxs.start.makeValStr(nsdims) <<
                   " ... " << loop_idxs.stop.makeValStr(nsdims) << ")");

#ifdef CHECK
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
    void StencilBundleBase::calc_loop_of_vectors(int thread_idx,
                                                 const ScanIndices& loop_idxs,
                                                 idx_t write_mask) {
        auto* cp = _generic_context;
        auto& dims = cp->get_dims();
        int nsdims = dims->_stencil_dims.size();
        auto step_posn = Indices::step_posn;
        TRACE_MSG3("calc_loop_of_vectors: local vector-indices [" <<
                   loop_idxs.start.makeValStr(nsdims) <<
                   " ... " << loop_idxs.stop.makeValStr(nsdims) <<
                   ") w/write-mask = 0x" << hex << write_mask << dec);

#ifdef CHECK
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

    // If this bundle is updating scratch grid(s),
    // expand begin & end of 'idxs' by sizes of halos.
    // This will often change vec-len aligned indices to non-aligned.
    // Step indices may also change.
    // NB: it is not necessary that the domain of each grid
    // is the same as the span of 'idxs'. However, it should be
    // at least that large to ensure that grid is able to hold
    // calculated results.
    // In other words, grid can be larger than span of 'idxs', but
    // its halo sizes are still used to specify how much to
    // add to 'idxs'.
    // Return adjusted indices.
    ScanIndices StencilBundleBase::adjust_span(int thread_idx,
                                               const ScanIndices& idxs) const {
        ScanIndices adj_idxs(idxs);
        auto* cp = _generic_context;
        auto& dims = cp->get_dims();
        int nsdims = dims->_stencil_dims.size();
        auto step_posn = Indices::step_posn;

        // Loop thru vecs of scratch grids for this bundle.
        for (auto* sv : outputScratchVecs) {
            assert(sv);

            // Get the one for this thread.
            auto gp = sv->at(thread_idx);
            assert(gp);
            assert(gp->is_scratch());

            // i: index for stencil dims, j: index for domain dims.
            for (int i = 0, j = 0; i < nsdims; i++) {
                if (i == step_posn) continue;
                auto& dim = dims->_stencil_dims.getDim(i);
                auto& dname = dim.getName();

                // Is this dim used in this grid?
                int posn = gp->get_dim_posn(dname);
                if (posn >= 0) {

                    // Adjust begin & end scan indices based on halos.
                    idx_t lh = gp->get_left_halo_size(posn);
                    idx_t rh = gp->get_right_halo_size(posn);
                    adj_idxs.begin[i] = idxs.begin[i] - lh;
                    adj_idxs.end[i] = idxs.end[i] + rh;

                    // Make sure grid covers block.
                    assert(adj_idxs.begin[i] >= gp->get_first_rank_alloc_index(posn));
                    assert(adj_idxs.end[i] <= gp->get_last_rank_alloc_index(posn) + 1);

                    // If existing step is >= whole tile, adjust it also.
                    idx_t width = idxs.end[i] - idxs.begin[i];
                    if (idxs.step[i] >= width) {
                        idx_t adj_width = adj_idxs.end[i] - adj_idxs.begin[i];
                        adj_idxs.step[i] = adj_width;
                    }
                }
                j++;
            }

            // Only need to get info from one grid.
            // TODO: check that grids are consistent.
            break;
        }
        return adj_idxs;
    }

} // namespace yask.
