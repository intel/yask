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

    // Calculate results within a block.
    // Typically called by an OMP thread team.
    void StencilGroupBase::calc_block(const ScanIndices& region_idxs) {

        auto opts = _generic_context->get_settings();
        auto dims = _generic_context->get_dims();
        int ndims = dims->_stencil_dims.size();
        auto& step_dim = dims->_step_dim;
        TRACE_MSG3("calc_block: " << region_idxs.start.makeValStr(ndims) <<
                  " ... (end before) " << region_idxs.stop.makeValStr(ndims));

        // Init block begin & end from region start & stop indices.
        ScanIndices block_idxs(ndims);
        block_idxs.initFromOuter(region_idxs);

        // Steps within a block are based on sub-block sizes.
        block_idxs.step = opts->_sub_block_sizes;

        // Groups in block loops are based on sub-block-group sizes.
        block_idxs.group_size = opts->_sub_block_group_sizes;

        // Set number of threads for a block.
        // This should be nested within a top-level OpenMP task.
        _generic_context->set_block_threads();

        // Include automatically-generated loop code that calls
        // calc_sub_block() for each sub-block in this block.  Loops through
        // x from begin_bx to end_bx-1; similar for y and z.  This
        // code typically contains the nested OpenMP loop(s).
#include "yask_block_loops.hpp"
    }
    
    // Calculate results for one sub-block.
    // Typically called by a single OMP thread.
    void StencilGroupBase::calc_sub_block(const ScanIndices& block_idxs) {

        auto* cp = _generic_context;
        auto opts = cp->get_settings();
        auto dims = cp->get_dims();
        int nddims = dims->_domain_dims.size();
        int nsdims = dims->_stencil_dims.size();
        auto& step_dim = dims->_step_dim;
        auto step_posn = Indices::step_posn;
        TRACE_MSG3("calc_sub_block: " << block_idxs.start.makeValStr(nsdims) <<
                  " ... (end before) " << block_idxs.stop.makeValStr(nsdims));

        // Init sub-block begin & end from block start & stop indices.
        // These indices are in element units.
        ScanIndices sub_block_idxs(nsdims);
        sub_block_idxs.initFromOuter(block_idxs);

        // Portion of sub-block that is full clusters/vectors.
        // These indices are in element units.
        ScanIndices sub_block_cidxs(sub_block_idxs);
        ScanIndices sub_block_vidxs(sub_block_idxs);
        
        // Determine what part of this sub-block can be done with full clusters/vectors.
        // Init to full clusters only.
        bool do_clusters = true; // any clusters to do?
        bool do_vectors = false; // any vectors to do? (assume not)
        bool do_scalars = false; // any scalars to do? (assume not)
        bool do_rem = false;     // check scalar code for partial remainders?

        // If BB is not full or not aligned, do whole block with scalar code.
        // We should only get here when using sub-domains.
        // TODO: do as much vectorization as possible--
        // this current code is functionally correct but very
        // poor perf.
        if (!bb_is_full || !bb_is_aligned) {

            do_clusters = false;
            do_vectors = false;
            do_scalars = true;
            do_rem = false;
            sub_block_cidxs.begin.setFromConst(0);
            sub_block_cidxs.end.setFromConst(0);
            sub_block_vidxs.begin.setFromConst(0);
            sub_block_vidxs.end.setFromConst(0);
        }

        // If whole BB is not a cluster mult, determine the subset of this
        // sub-block that is whole clusters/vectors.
        else if (!bb_is_cluster_mult) {

            // Assume we need vectors and scalars.
            do_vectors = true;
            do_scalars = true;
            do_rem = true;

            // (i: index for stencil dims, j: index for domain dims).
            for (int i = 0, j = 0; i < nsdims; i++) {
                if (i != step_posn) {

                    // End of scalars in this dim.
                    auto send = sub_block_idxs.end[i];

                    // Find end of whole clusters.
                    auto cpts = dims->_cluster_pts[j];
                    auto cend = ROUND_DOWN(send, cpts);
                    sub_block_cidxs.end[i] = cend;

                    // Find end of whole vectors.
                    auto vpts = dims->_fold_pts[j];
                    auto vend = ROUND_DOWN(send, vpts);
                    if (i == _inner_posn)
                        vend = cend; // don't do vectors in inner dim.
                    sub_block_vidxs.end[i] = vend;

                    // Next domain index.
                    j++;
                }
            }
        }

        // Normalized indices needed for sub-block loop.
        ScanIndices norm_sub_block_idxs(sub_block_idxs);
        
        // Full rectangular polytope of aligned clusters: use optimized code.
        if (do_clusters) {
            TRACE_MSG3("calc_sub_block: using cluster code for " <<
                       sub_block_cidxs.begin.makeValStr(nsdims) <<
                       " ... (end before) " << sub_block_cidxs.end.makeValStr(nsdims));

            // Normalize the cluster indices.
            int j = 0;          // domain dim index.
            for (int i = 0; i < nsdims; i++) {
                if (i != step_posn) {
                
                    // Subtract rank offset and divide indices by fold lengths
                    // as needed by read/writeVecNorm().  Use idiv_flr() instead
                    // of '/' because begin/end vars may be negative (if in
                    // halo).
                    // Set both begin/end and start/stop to ensure start/stop
                    // vars get passed through to calc_loop_of_clusters()
                    // for the inner loop.
                    idx_t nbegin = idiv_flr<idx_t>(sub_block_cidxs.begin[i] -
                                                   cp->rank_domain_offsets[j],
                                                   dims->_fold_pts[j]);
                    norm_sub_block_idxs.begin[i] = nbegin;
                    norm_sub_block_idxs.start[i] = nbegin;
                    idx_t nend = idiv_flr<idx_t>(sub_block_cidxs.end[i] -
                                                 cp->rank_domain_offsets[j],
                                                 dims->_fold_pts[j]);
                    norm_sub_block_idxs.end[i] = nend;
                    norm_sub_block_idxs.stop[i] = nend;

                    // Step sizes are based on cluster lengths (in vector units).
                    // The step in the inner loop is hard-coded in the generated code.
                    norm_sub_block_idxs.step[i] = dims->_cluster_mults[j];
                    j++;
                }
            }

            // Define the function called from the generated loops
            // to simply call the loop-of-clusters functions.
#define calc_inner_loop(loop_idxs) calc_loop_of_clusters(loop_idxs)

            // Include automatically-generated loop code that calls
            // calc_inner_loop(). This is different from the higher-level
            // loops because it does not scan the inner dim.
#include "yask_sub_block_loops.hpp"
#undef calc_inner_loop
        }
        
        // Full remainder vectors.
        if (do_vectors) {
            TRACE_MSG3("calc_sub_block: using vector code for " <<
                       sub_block_vidxs.begin.makeValStr(nsdims) <<
                       " ... (end before) " << sub_block_vidxs.end.makeValStr(nsdims) <<
                       " remaining after clusters in " <<
                       sub_block_cidxs.begin.makeValStr(nsdims) <<
                       " ... (end before) " << sub_block_cidxs.end.makeValStr(nsdims));

            // Keep a copy of the normalized cluster indices.
            ScanIndices norm_sub_block_cidxs(norm_sub_block_idxs);

            // Normalize the vector indices.
            int j = 0;          // domain dim index.
            for (int i = 0; i < nsdims; i++) {
                if (i != step_posn) {
                
                    // Subtract rank offset and divide indices by fold lengths
                    // as needed by read/writeVecNorm().  Use idiv_flr() instead
                    // of '/' because begin/end vars may be negative (if in
                    // halo).
                    // Set both begin/end and start/stop to ensure start/stop
                    // vars get passed through to calc_loop_of_vectors()
                    // for the inner loop.
                    idx_t nbegin = idiv_flr<idx_t>(sub_block_vidxs.begin[i] -
                                                   cp->rank_domain_offsets[j],
                                                   dims->_fold_pts[j]);
                    norm_sub_block_idxs.begin[i] = nbegin;
                    norm_sub_block_idxs.start[i] = nbegin;
                    idx_t nend = idiv_flr<idx_t>(sub_block_vidxs.end[i] -
                                                 cp->rank_domain_offsets[j],
                                                 dims->_fold_pts[j]);
                    norm_sub_block_idxs.end[i] = nend;
                    norm_sub_block_idxs.stop[i] = nend;

                    // Step sizes are one vector.
                    // The step in the inner loop is hard-coded in the generated code.
                    norm_sub_block_idxs.step[i] = 1;
                    j++;
                }
            }

            // Define the function called from the generated loops to
            // determine whether a loop of vectors is within the remainder
            // range, i.e., after the clusters. If so, set the extent of the
            // loop and call the loop-of-vectors function w/an all-1s mask.
#define calc_inner_loop(loop_idxs) \
            bool ok = false;                                            \
            for (int i = 0; i < nsdims; i++) {                          \
                if (i != step_posn &&                                   \
                    i != _inner_posn &&                                 \
                    loop_idxs.start[i] >= norm_sub_block_cidxs.end[i]) { \
                    ok = true; break; }                                 \
            }                                                           \
            if (ok) calc_loop_of_vectors(loop_idxs, idx_t(-1));
            // TODO: handle inner-loop vector remainder.

            // Include automatically-generated loop code that calls
            // calc_inner_loop(). This is different from the higher-level
            // loops because it does not scan the inner dim.
#include "yask_sub_block_loops.hpp"
#undef calc_inner_loop
        }
        
        // Use scalar code for anything not done above.
        if (do_scalars) {

#ifdef TRACE
            string msg = "calc_sub_block: using scalar code for ";
            msg += do_rem ? "remainder of" : "entire";
            msg += " sub-block ";
            msg += bb_is_full ? "without" : "with";
            msg += " sub-domain checking";
            TRACE_MSG3(msg);
#endif

            // Use the 'misc' loop. The OMP will be ignored because we're already in
            // a nested OMP region. TODO: check this if there is only one block thread.
            ScanIndices misc_idxs(sub_block_idxs);

            // Define misc-loop function.  Since step is always 1, we
            // ignore misc_stop.  If point is in sub-domain for this
            // group, then evaluate the reference scalar code.
            // If no holes, don't need to check each point in domain.
#define misc_fn(misc_idxs)                                              \
            bool ok = true;                                             \
            if (do_rem) { ok = false;                                   \
                for (int i = 0; i < nsdims; i++)                        \
                    if (i != step_posn &&                               \
                        misc_idxs.start[i] >= sub_block_vidxs.end[i]) { \
                        ok = true; break; }                             \
            }                                                           \
            if (ok && (bb_is_full || is_in_valid_domain(misc_idxs.start))) \
                calc_scalar(misc_idxs.start)
                
            // Scan through n-D space.
#include "yask_misc_loops.hpp"
#undef misc_fn
        }

        // Make sure streaming stores are visible for later loads.
        make_stores_visible();
    }

    // Calculate a series of cluster results within an inner loop.
    // The 'loop_idxs' must specify a range only in the inner dim.
    // Indices must be rank-relative.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    void StencilGroupBase::calc_loop_of_clusters(const ScanIndices& loop_idxs) {
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
        calc_loop_of_clusters(start_idxs, stop_inner);
    }

    // Calculate a series of vector results within an inner loop.
    // The 'loop_idxs' must specify a range only in the inner dim.
    // Indices must be rank-relative.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    void StencilGroupBase::calc_loop_of_vectors(const ScanIndices& loop_idxs, idx_t write_mask) {
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
        calc_loop_of_vectors(start_idxs, stop_inner, write_mask);
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
#define misc_fn(misc_idxs)                                        \
        if (is_in_valid_domain(misc_idxs.start)) {               \
            min_pts = min_pts.minElements(misc_idxs.start);      \
            max_pts = max_pts.maxElements(misc_idxs.start);      \
            npts++; \
        }

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
