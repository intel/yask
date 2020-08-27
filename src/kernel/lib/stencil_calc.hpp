/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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

#pragma once

namespace yask {

    // Classes that support evaluation of one stencil bundle
    // and a stage of bundles.
    // A stencil solution contains one or more stages.

    // A pure-virtual class base for a stencil bundle.
    class StencilBundleBase :
        public ContextLinker {

    protected:

        // Other bundles that this one depends on.
        StencilBundleSet _depends_on;

        // List of scratch-var bundles that need to be evaluated
        // before this bundle. Listed in eval order first-to-last.
        StencilBundleList _scratch_children;

        // Overall bounding box for the bundle.
        // This may or may not be solid, i.e., it
        // may contain some invalid points.
        // This must fit inside the extended BB for this rank.
        BoundingBox _bundle_bb;

	// Bounding box(es) that indicate where this bundle is valid.
	// These must be non-overlapping. These do NOT contain
        // any invalid points. These will all be inside '_bundle_bb'.
	BBList _bb_list;

        // Normalize the indices, i.e., divide by vector len in each dim.
        // Ranks offsets must already be subtracted.
        // Each dim in 'orig' must be a multiple of corresponding vec len.
        void normalize_indices(const Indices& orig, Indices& norm) const {
            STATE_VARS(this);
            assert(orig._get_num_dims() == nsdims);
            assert(norm._get_num_dims() == nsdims);

            // i: index for stencil dims, j: index for domain dims.
            DOMAIN_VAR_LOOP(i, j) {

                // Divide indices by fold lengths as needed by
                // read/write_vec_norm().  Use idiv_flr() instead of '/'
                // because begin/end vars may be negative (if in halo).
                norm[i] = idiv_flr<idx_t>(orig[i], fold_pts[j]);

                // Check for no remainder.
                assert(imod_flr<idx_t>(orig[i], fold_pts[j]) == 0);
            }
        }

    public:

        // Vars that are written to by these stencils.
        VarPtrs output_var_ptrs;

        // Vars that are read by these stencils (not necessarify
        // read-only, i.e., a var can be input and output).
        VarPtrs input_var_ptrs;

        // Vectors of scratch vars that are written to/read from.
        ScratchVecs output_scratch_vecs;
        ScratchVecs input_scratch_vecs;

        // ctor, dtor.
        StencilBundleBase(StencilContext* context) :
            ContextLinker(context) { }
        virtual ~StencilBundleBase() { }

        // Access to BBs.
        BoundingBox& get_bb() { return _bundle_bb; }
        BBList& get_bbs() { return _bb_list; }

        // Add dependency.
        void add_dep(StencilBundleBase* eg) {
            _depends_on.insert(eg);
        }

        // Get dependencies.
        const StencilBundleSet& get_deps() const {
            return _depends_on;
        }

        // Add needed scratch-bundle.
        void add_scratch_child(StencilBundleBase* eg) {
            _scratch_children.push_back(eg);
        }

        // Get needed scratch-bundle(s).
        const StencilBundleList& get_scratch_children() const {
            return _scratch_children;
        }

        // Get scratch children plus self.
        StencilBundleList get_reqd_bundles() {
            auto sg_list = get_scratch_children(); // Do children first.
            sg_list.push_back(this); // Do self last.
            return sg_list;
        }

        // If this bundle is updating scratch var(s),
        // expand indices to calculate values in halo.
        // Adjust offsets in vars based on original idxs.
        // Return adjusted indices.
        ScanIndices adjust_span(int thread_idx, const ScanIndices& idxs) const;

        // Set the bounding-box vars for this bundle in this rank.
        void find_bounding_box();

        // Copy BB vars from another.
        void copy_bounding_box(const StencilBundleBase* src);

        // Calculate results for an arbitrary tile for points in the valid domain.
        // Scratch vars, if any, are indexed via 'scratch_var_idx'.
        virtual void
        calc_in_domain(int scratch_var_idx, const ScanIndices& misc_idxs) =0;

        // Calculate results within a mini-block.
        void
        calc_mini_block(int region_thread_idx,
                        KernelSettings& settings,
                        const ScanIndices& mini_block_idxs);

        // Calculate results within a sub-block.
        virtual void
        calc_sub_block(int region_thread_idx,
                       int block_thread_idx,
                       KernelSettings& settings,
                       const ScanIndices& mini_block_idxs) =0;

        // Functions below are stubs for the code generated
        // by the stencil compiler.
        
        // Get name of this bundle.
        virtual const std::string
        get_name() const =0;

        // Get estimated number of FP ops done for one scalar eval.
        virtual int
        get_scalar_fp_ops() const =0;

        // Get number of points read and written for one scalar eval.
        virtual int
        get_scalar_points_read() const =0;
        virtual int
        get_scalar_points_written() const =0;

        // Whether this bundle updates scratch var(s)?
        virtual bool
        is_scratch() const =0;

        // Determine whether indices are in [sub-]domain.
        virtual bool
        is_in_valid_domain(const Indices& idxs) const =0;

        // Return true if there are any non-default conditions.
        virtual bool
        is_sub_domain_expr() const =0;
        virtual bool
        is_step_cond_expr() const =0;

        // Return human-readable description of conditions.
        virtual std::string
        get_domain_description() const =0;
        virtual std::string
        get_step_cond_description() const =0;

        // Determine whether step index is enabled.
        virtual bool
        is_in_valid_step(idx_t input_step_index) const =0;

        // If bundle updates var(s) with the step index,
        // set 'output_step_index' to the step that an update
        // occurs when calling one of the calc_*() methods with
        // 'input_step_index' and return 'true'.
        // Else, return 'false';
        virtual bool
        get_output_step_index(idx_t input_step_index,
                              idx_t& output_step_index) const =0;

    };                          // StencilBundleBase.

    // A template that is instantiated with the stencil-compiler
    // output class.
    template <typename StencilBundleImplT,
              typename StencilCoreDataT>
    class StencilBundleTempl:
        public StencilBundleBase {

    protected:
        StencilBundleImplT _bundle;

        // Access core data.
        // TODO: use dynamic_cast in CHECK mode.
        inline StencilCoreDataT* _corep() {
            return static_cast<StencilCoreDataT*>(_context->corep());
        }
        inline const StencilCoreDataT* _corep() const {
            return static_cast<const StencilCoreDataT*>(_context->corep());
        }
        
    public:

        // Ctor.
        StencilBundleTempl(StencilContext* context):
            StencilBundleBase(context) { }

        // Dtor.
        virtual ~StencilBundleTempl() { }

        // Get name of this bundle.
        const std::string get_name() const override {
            return _bundle._name;
        }

        // Get estimated number of FP ops done for one scalar eval.
        int get_scalar_fp_ops() const override {
            return _bundle._scalar_fp_ops;
        }

        // Get number of points read and written for one scalar eval.
        int get_scalar_points_read() const override {
            return _bundle._scalar_points_read;
        }
        int get_scalar_points_written() const override {
            return _bundle._scalar_points_written;
        }

        // Whether this bundle updates scratch var(s)?
        bool is_scratch() const override {
            return _bundle._is_scratch;
        }

        // Determine whether indices are in [sub-]domain.
        bool is_in_valid_domain(const Indices& idxs) const override {
            return _bundle.is_in_valid_domain(_corep(), idxs);
        }

        // Return true if there are any non-default conditions.
        bool is_sub_domain_expr() const override {
            return _bundle.is_sub_domain_expr();
        }
        bool is_step_cond_expr() const override {
            return _bundle.is_step_cond_expr();
        }

        // Return human-readable description of conditions.
        std::string get_domain_description() const override {
            return _bundle.get_domain_description();
        }
        std::string get_step_cond_description() const override {
            return _bundle.get_step_cond_description();
        }

        // Determine whether step index is enabled.
        bool is_in_valid_step(idx_t input_step_index) const override {
            return _bundle.is_in_valid_step(_corep(), input_step_index);
        }

        // If bundle updates var(s) with the step index,
        // set 'output_step_index' to the step that an update
        // occurs when calling one of the calc_*() methods with
        // 'input_step_index' and return 'true'.
        // Else, return 'false';
        bool get_output_step_index(idx_t input_step_index,
                                   idx_t& output_step_index) const override {
            return _bundle.get_output_step_index(input_step_index,
                                                 output_step_index);
        }

        // Calculate results for an arbitrary tile for points in the valid domain.
        // Scratch vars, if any are used, are indexed via 'scratch_var_idx'.
        // This is very slow and used for reference calculations.
        void
        calc_in_domain(int scratch_var_idx, const ScanIndices& misc_idxs) override {
            auto* cp = _corep();

            // Define misc-loop function.  Since stride is always 1, we
            // ignore misc_indx.stop.  If point is in sub-domain for this
            // bundle, then execute the reference scalar code.
            // TODO: fix domain of scratch vars.
            #define MISC_FN(misc_idxs)                                  \
                do {                                                    \
                    if (_bundle.is_in_valid_domain(cp, misc_idxs.start)) \
                        _bundle.calc_scalar(cp, scratch_var_idx, misc_idxs.start); \
                } while(0)
            #include "yask_misc_loops.hpp"
            #undef MISC_FN
        }
        
        // Calculate results within a sub-block.
        // Essentially a chooser between the scalar or vector version.
        void
        calc_sub_block(int region_thread_idx,
                       int block_thread_idx,
                       KernelSettings& settings,
                       const ScanIndices& mini_block_idxs) override {

            // Choose between scalar and vector impls.
            if (settings.force_scalar)
                calc_sub_block_scalar(region_thread_idx, block_thread_idx,
                                      settings, mini_block_idxs);
            else
                calc_sub_block_vec(region_thread_idx, block_thread_idx,
                                   settings, mini_block_idxs);
        }

        // Calculate results for one sub-block using pure scalar code.
        // This is very slow and used for debug.
        void
        calc_sub_block_scalar(int region_thread_idx,
                              int block_thread_idx,
                              KernelSettings& settings,
                              const ScanIndices& mini_block_idxs) {
            STATE_VARS(this);
            TRACE_MSG("calc_sub_block_scalar for bundle '" << get_name() << "': [" <<
                      mini_block_idxs.start.make_val_str() <<
                      " ... " << mini_block_idxs.stop.make_val_str() <<
                      ") by region thread " << region_thread_idx <<
                      " and block thread " << block_thread_idx);

            auto* cp = _corep();

            // Init sub-block begin & end from block start & stop indices.
            // Use the 'misc' loops. Indices for these loops will be scalar and
            // global rather than normalized as in the cluster and vector loops.
            ScanIndices misc_idxs(*dims, true);
            misc_idxs.init_from_outer(mini_block_idxs);

            // Stride sizes and alignment are one element.
            misc_idxs.stride.set_from_const(1);
            misc_idxs.align.set_from_const(1);

            // Define misc-loop function.
            // Since stride is always 1, we ignore pt_idxs.stop.
            #define MISC_FN(pt_idxs)                                    \
                FORCE_INLINE                                            \
                    _bundle.calc_scalar(cp, region_thread_idx, pt_idxs.start)

            // Set OMP loop to offload or disable OMP.
            #ifdef USE_OFFLOAD
            #define OMP_PRAGMA _Pragma("omp target parallel for device(KernelEnv::_omp_devn)")
            #else
            #define OMP_PRAGMA
            #endif
            
            // Scan through n-D space.
            #include "yask_misc_loops.hpp"
            #undef MISC_FN
        }

        // Calculate results for one sub-block.
        // The index ranges in 'mini_block_idxs' are sub-divided
        // into full vector-clusters, full vectors, and sub-vectors
        // and finally evaluated by the YASK-compiler-generated loops.
        void
        calc_sub_block_vec(int region_thread_idx,
                           int block_thread_idx,
                           KernelSettings& settings,
                           const ScanIndices& mini_block_idxs) {
            STATE_VARS(this);
            TRACE_MSG("calc_sub_block_vec for bundle '" << get_name() << "': [" <<
                      mini_block_idxs.start.make_val_str() <<
                      " ... " << mini_block_idxs.stop.make_val_str() <<
                      ") by region thread " << region_thread_idx <<
                      " and block thread " << block_thread_idx);
            auto* cp = _corep();

            /*
              Indices in each non-inner domain dim:

               sb_eidxs.begin                              rem_masks used here
               | peel_masks used here                      | sb_eidxs.end
               | |    full vecs here       full vecs here  | |
               | |    |                                 |  | |
               v v    v         full clusters here      v  v v
              +---+-------+---------------------------+---+---+   "+" => vec boundaries.
              ^   ^       ^                            ^   ^   ^
              |   |       |                            |   |   |
              |   |       sb_fcidxs.begin              |   |   sb_vidxs.end (rounded up)
              |   sb_fvidxs.begin                      |   sb_fvidxs.end (rounded down)
              sb_vidxs.begin                           sb_fcidxs.end (rounded up)

              For the inner domain dim, sb_vidxs = sb_fvidxs = sb_fcidxs.
            */

            // Init sub-block begin & end from block start & stop indices.
            // These indices are in element units and global (NOT rank-relative).
            ScanIndices sb_idxs(*dims, true);
            sb_idxs.init_from_outer(mini_block_idxs);

            // Sub block indices in element units and rank-relative.
            ScanIndices sb_eidxs(sb_idxs);

            // Subset of sub-block that is full clusters.
            // These indices are in element units and rank-relative.
            ScanIndices sb_fcidxs(sb_idxs);

            // Subset of sub-block that is full vectors.
            // These indices are in element units and rank-relative.
            ScanIndices sb_fvidxs(sb_idxs);

            // Superset of sub-block that is full or partial (masked) vectors.
            // These indices are in element units and rank-relative.
            ScanIndices sb_vidxs(sb_idxs);

            // These will be set to rank-relative, so set ofs to zero.
            sb_eidxs.align_ofs.set_from_const(0);
            sb_fcidxs.align_ofs.set_from_const(0);
            sb_fvidxs.align_ofs.set_from_const(0);
            sb_vidxs.align_ofs.set_from_const(0);

            // Masks for computing partial vectors in each dim.
            // Init to all-ones (no masking).
            Indices peel_masks(nsdims), rem_masks(nsdims);
            peel_masks.set_from_const(-1);
            rem_masks.set_from_const(-1);

            // Flags that indicate what type of processing needs to be done.
            bool do_clusters = true; // any clusters to do?
            bool do_vectors = false; // any vectors to do?
            bool do_scalars_left = false, do_scalars_right = false; // any scalars to do?
            idx_t scalar_left_end = 0, scalar_right_begin = 0;

            /*
              3D-view, where vertical (z) dim is inner dim:
                   _________________
                  /                /|
                 /                //|
                /                // |
               /_______________ // /|
               |______________<---------- do_scalars_left
               | detail below  | //
               |_______________|//
               |______________<---------- do_scalars_right

               Detail of section between left and right scalars:
                   _________________
                  / ____________   /|
                 / /        <-------------do_clusters
                / /___________/ <-------- do_vectors
               /_______________ / /
               |               | /
               |_______________|/

            */
            
            // Adjust indices to be rank-relative.
            // Determine the subset of this sub-block that is
            // clusters, vectors, and partial vectors.
            _DOMAIN_VAR_LOOP(i, j) {

                // Rank offset.
                auto rofs = _context->rank_domain_offsets[j];

                // Begin/end of rank-relative scalar elements in this dim.
                auto ebgn = sb_idxs.begin[i] - rofs;
                auto eend = sb_idxs.end[i] - rofs;
                sb_eidxs.begin[i] = ebgn;
                sb_eidxs.end[i] = eend;

                // Find range of full clusters.
                // Note that fcend <= eend because we round
                // down to get whole clusters only.
                // Similarly, fcbgn >= ebgn.
                auto cpts = dims->_cluster_pts[j];
                auto fcbgn = round_up_flr(ebgn, cpts);
                auto fcend = round_down_flr(eend, cpts);
                sb_fcidxs.begin[i] = fcbgn;
                sb_fcidxs.end[i] = fcend;

                // Any clusters to do?
                if (fcend <= fcbgn)
                    do_clusters = false;

                // If anything before or after clusters, continue with
                // setting vector indices and peel/rem masks.
                if (fcbgn > ebgn || fcend < eend) {

                    // Find range of full and/or partial vectors.  Note that
                    // fvend <= eend because we round down to get whole
                    // vectors only, and vend >= eend because we
                    // round up to include partial vectors.  Similar but
                    // opposite for begin vars.  We make a vector mask to
                    // pick the right elements.
                    auto vpts = fold_pts[j];
                    auto fvbgn = round_up_flr(ebgn, vpts);
                    auto fvend = round_down_flr(eend, vpts);
                    auto vbgn = round_down_flr(ebgn, vpts);
                    auto vend = round_up_flr(eend, vpts);
                    if (i == inner_posn) {

                        // Don't do any full and/or partial vectors in plane of
                        // inner domain dim.  We'll do these with scalars.  This
                        // should be unusual because vector folding is normally
                        // done in a plane perpendicular to the inner dim for >=
                        // 2D domains.
                        fvbgn = vbgn = fcbgn;
                        fvend = vend = fcend;

                        // No vectors to do at all: do one scalar slab.
                        if (vend <= vbgn) {
                            do_scalars_left = true;
                            scalar_left_end = sb_idxs.end[i];
                        }

                        // Need slab(s) before or after vectors.
                        else {
                            if (ebgn < vbgn) {
                                do_scalars_left = true;
                                scalar_left_end = vbgn + rofs;
                            }
                            if (eend > vend) {
                                do_scalars_right = true;
                                scalar_right_begin = vend + rofs;
                            }
                        }
                    }
                    sb_fvidxs.begin[i] = fvbgn;
                    sb_fvidxs.end[i] = fvend;
                    sb_vidxs.begin[i] = vbgn;
                    sb_vidxs.end[i] = vend;

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
                    // so that the 6 corner elements are updated in this example.

                    if (vbgn < fvbgn || vend > fvend) {
                        idx_t pmask = 0, rmask = 0;

                        // Need to set upper bit.
                        idx_t mbit = 0x1 << (dims->_fold_pts.product() - 1);

                        // Visit points in a vec-fold to set bits for this dim's
                        // masks per the diagram above.  TODO: make this more
                        // efficient.
                        dims->_fold_pts.visit_all_points
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
                }

                // If no peel or rem, just set vec indices to same as
                // full cluster.
                else {
                    sb_fvidxs.begin[i] = fcbgn;
                    sb_fvidxs.end[i] = fcend;
                    sb_vidxs.begin[i] = fcbgn;
                    sb_vidxs.end[i] = fcend;
                }
            }

            // Normalized indices needed for sub-block loop.
            ScanIndices norm_sb_idxs(sb_eidxs);

            // Normalize the cluster indices.
            // These will be the bounds of the sub-block loops.
            // Set both begin/end and start/stop to ensure start/stop
            // vars get passed through to calc_loop_of_clusters()
            // for the inner loop.
            normalize_indices(sb_fcidxs.begin, norm_sb_idxs.begin);
            norm_sb_idxs.start = norm_sb_idxs.begin;
            normalize_indices(sb_fcidxs.end, norm_sb_idxs.end);
            norm_sb_idxs.stop = norm_sb_idxs.end;
            norm_sb_idxs.align.set_from_const(1); // one vector.

            // Full rectilinear polytope of aligned clusters: use optimized
            // code for full clusters w/o masking.
            if (do_clusters) {
                TRACE_MSG("calc_sub_block_vec:  using cluster code for rank-indices [" <<
                          sb_fcidxs.begin.make_val_str() <<
                          " ... " << sb_fcidxs.end.make_val_str() <<
                          ") by region thread " << region_thread_idx <<
                          " and block thread " << block_thread_idx);

                // Stride sizes are based on cluster lengths (in vector units).
                // The stride in the inner loop is hard-coded in the generated code.
                DOMAIN_VAR_LOOP(i, j) {
                    norm_sb_idxs.stride[i] = dims->_cluster_mults[j]; // N vecs.
                }

                // Perform the calculations in this block.
                calc_clusters(cp, region_thread_idx, block_thread_idx,
                              norm_sb_idxs, inner_posn);
                
            } // whole clusters.

            // Full and partial peel/remainder vectors in all dims except
            // the inner one.
            // An alternative way to do this would be to do the left and
            // right slabs in each domain dim separately.
            if (do_vectors) {
                TRACE_MSG("calc_sub_block_vec:  using vector code for local indices [" <<
                          sb_vidxs.begin.make_val_str() <<
                          " ... " << sb_vidxs.end.make_val_str() <<
                          ") *not* within full vector-clusters at [" <<
                          sb_fcidxs.begin.make_val_str() <<
                          " ... " << sb_fcidxs.end.make_val_str() <<
                          ") by region thread " << region_thread_idx <<
                          " and block thread " << block_thread_idx);
                #ifdef USE_OFFLOAD
                THROW_YASK_EXCEPTION("Internal error: vector border-code not expected when offloading");
                #else

                // Keep a copy of the normalized cluster indices
                // that were calculated above.
                // The full clusters were already done above, so
                // we only need to do vectors before or after the
                // clusters in each dim.
                // We'll exclude them below.
                ScanIndices norm_sb_fcidxs(norm_sb_idxs);

                // Normalize the vector indices.
                // These will be the bounds of the sub-block loops.
                // Set both begin/end and start/stop to ensure start/stop
                // vars get passed through to calc_loop_of_clusters()
                // for the inner loop.
                normalize_indices(sb_vidxs.begin, norm_sb_idxs.begin);
                norm_sb_idxs.start = norm_sb_idxs.begin;
                normalize_indices(sb_vidxs.end, norm_sb_idxs.end);
                norm_sb_idxs.stop = norm_sb_idxs.end;

                // Stride sizes are one vector.
                // The stride in the inner loop is hard-coded in the generated code.
                norm_sb_idxs.stride.set_from_const(1);

                // Also normalize the *full* vector indices to determine if
                // we need a mask at each vector index.
                // We just need begin and end indices for this.
                ScanIndices norm_sb_fvidxs(sb_eidxs);
                normalize_indices(sb_fvidxs.begin, norm_sb_fvidxs.begin);
                normalize_indices(sb_fvidxs.end, norm_sb_fvidxs.end);
                norm_sb_fvidxs.align.set_from_const(1); // one vector.

                // Perform the calculations around the outside of this block.
                calc_outer_vectors(cp, region_thread_idx, block_thread_idx,
                                   norm_sb_idxs, norm_sb_fcidxs, norm_sb_fvidxs,
                                   peel_masks, rem_masks, inner_posn);
                #endif
            } // do vectors.

            // Use scalar code for anything not done above.  This should only be
            // called if vectorizing on the inner loop and the sub-block size in
            // that dim is not a multiple of the inner-dim vector len, so that
            // situation should be avoided.
            // Unfortunately, this is common when using "legacy" layouts, where
            // the inner dim is vectorized.
            // TODO: enable masking in the inner dim.
            if (do_scalars_left || do_scalars_right) {
                #ifdef USE_OFFLOAD
                THROW_YASK_EXCEPTION("Internal error: scalar remainder-code not expected when offloading");
                #else

                // Use the 'misc' loops. Indices for these loops will be scalar and
                // global rather than normalized as in the cluster and vector loops.
                ScanIndices misc_idxs(sb_idxs);

                // Stride sizes and alignment are one element.
                misc_idxs.stride.set_from_const(1);
                misc_idxs.align.set_from_const(1);

                // Define misc-loop function.  This is called at each point
                // in the sub-block.  Since stride is always 1, we ignore
                // misc_idxs.stop.  TODO: handle more efficiently: calculate
                // masks, and call vector code.
                #define MISC_FN(pt_idxs)                                \
                    _bundle.calc_scalar(cp, region_thread_idx, pt_idxs.start)
                
                // Left slab: compute scalars from beginning of sub-block to
                // beginning of first full vector (or end of sub-block if less).
                if (do_scalars_left) {
                    int i = inner_posn;
                    int j = i - 1;
                    misc_idxs.begin[i] = sb_idxs.begin[i];
                    misc_idxs.end[i] = scalar_left_end;
                    TRACE_MSG("calc_sub_block_vec:  using scalar code "
                              "for left slab global indices [" <<
                              misc_idxs.begin.make_val_str() << " ... " <<
                              misc_idxs.end.make_val_str() <<
                              ") by region thread " << region_thread_idx <<
                              " and block thread " << block_thread_idx);

                    // Scan through n-D space.
                    #define OMP_PRAGMA
                    #include "yask_misc_loops.hpp"
                }

                // Right slab: compute scalars from end of last full vector
                // (or beginning of sub-block if greater) to end of sub-block.
                if (do_scalars_right) {
                    int i = inner_posn;
                    int j = i - 1;
                    misc_idxs.begin[i] = scalar_right_begin;
                    misc_idxs.end[i] = sb_idxs.end[i];
                    TRACE_MSG("calc_sub_block_vec:  using scalar code "
                              "for right slab global indices [" <<
                              misc_idxs.begin.make_val_str() << " ... " <<
                              misc_idxs.end.make_val_str() <<
                              ") by region thread " << region_thread_idx <<
                              " and block thread " << block_thread_idx);

                    // Scan through n-D space.
                    #define OMP_PRAGMA
                    #include "yask_misc_loops.hpp"
                }

                #undef MISC_FN
                #endif
            } // do scalars.
        } // calc_sub_block_vec.

        // Calculate a block of clusters.
        // This should be the hottest function for most stencils.
        void
        calc_clusters(StencilCoreDataT* corep,
                      int region_thread_idx,
                      int block_thread_idx,
                      ScanIndices& norm_sb_idxs,
                      int inner_posn) {

            // Define the function called from the generated loops to simply
            // call the loop-of-clusters function.
            #define CALC_INNER_LOOP(loop_idxs)                          \
                FORCE_INLINE                                            \
                    calc_loop_of_clusters(corep, region_thread_idx, block_thread_idx, \
                                          loop_idxs, inner_posn)

            // Include automatically-generated loop code that calls
            // CALC_INNER_LOOP().
            FORCE_INLINE_RECURSIVE {
                #include "yask_sub_block_loops.hpp"
            }
            #undef CALC_INNER_LOOP
        }

        // Calculate a series of cluster results within an inner loop.
        // This is a simple wrapper around the YASK compiler-generated
        // code that reformats the indices.
        // The 'loop_idxs' must specify a range only in the inner dim.
        // Indices must be rank-relative.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE void
        calc_loop_of_clusters(StencilCoreDataT* corep,
                              int region_thread_idx,
                              int block_thread_idx,
                              const ScanIndices& loop_idxs,
                              int inner_posn) {
            #ifndef USE_OFFLOAD
            #ifdef TRACE
            {
                STATE_VARS(this);
                TRACE_MSG("calc_loop_of_clusters: local vector-indices [" <<
                          loop_idxs.start.make_val_str() <<
                          " ... " << loop_idxs.stop.make_val_str() <<
                          ") by region thread " << region_thread_idx <<
                          " and block thread " << block_thread_idx);
            }
            #endif
            #ifdef CHECK
            {
                STATE_VARS(this);

                // Check that only the inner dim has a range greater than one cluster.
                DOMAIN_VAR_LOOP(i, j) {
                    if (i != inner_posn)
                        assert(loop_idxs.start[i] + dims->_cluster_mults[j] >=
                               loop_idxs.stop[i]);
                }
            }
            #endif
            #endif

            // Need all starting indices.
            const Indices& start_idxs = loop_idxs.start;

            // Need stop for inner loop only.
            idx_t stop_inner = loop_idxs.stop[inner_posn];

            // Call code from stencil compiler.
            FORCE_INLINE
                _bundle.calc_loop_of_clusters(corep, region_thread_idx, block_thread_idx,
                                              start_idxs, stop_inner);
        }

        // Calculate a block of vectors outside of the cluster area.
        void
        calc_outer_vectors(StencilCoreDataT* corep,
                           int region_thread_idx,
                           int block_thread_idx,
                           ScanIndices& norm_sb_idxs,
                           ScanIndices& norm_sb_fcidxs,
                           ScanIndices& norm_sb_fvidxs,
                           Indices& peel_masks,
                           Indices& rem_masks,
                           int inner_posn) {

            // Define the function called from the generated loops.
            #define CALC_INNER_LOOP(loop_idxs)                          \
                calc_loop_of_outer_vectors(corep, region_thread_idx, block_thread_idx, \
                    loop_idxs, norm_sb_idxs, norm_sb_fcidxs, norm_sb_fvidxs, \
                    peel_masks, rem_masks, inner_posn);

            // Include automatically-generated loop code that calls
            // CALC_INNER_LOOP().
            FORCE_INLINE_RECURSIVE {
                #include "yask_sub_block_loops.hpp"
            }
            #undef CALC_INNER_LOOP
        }
            
        // Calculate a loop of vectors outside of the cluster area.
        ALWAYS_INLINE void
        calc_loop_of_outer_vectors(StencilCoreDataT* corep,
                                   int region_thread_idx,
                                   int block_thread_idx,
                                   ScanIndices& loop_idxs,
                                   ScanIndices& norm_sb_idxs,
                                   ScanIndices& norm_sb_fcidxs,
                                   ScanIndices& norm_sb_fvidxs,
                                   Indices& peel_masks,
                                   Indices& rem_masks,
                                   int inner_posn) {

            // Determine whether a loop of vectors is within the peel
            // range (before the cluster) and/or remainder range (after
            // the clusters)--setting the 'ok' flag. In other words, the
            // vectors should be used only outside of the inner block of
            // clusters. Then, call the loop-of-vectors function
            // w/appropriate mask.  See the mask diagrams above that
            // show how the masks are ANDed together.  Since stride is
            // always 1, we ignore loop_idxs.stop.
            bool ok = false;
            idx_t mask = idx_t(-1);
            DOMAIN_VAR_LOOP(i, j) {
                auto iidx = loop_idxs.start[i];

                // Is inner loop outside of full clusters?
                if (i != inner_posn &&
                    (iidx < norm_sb_fcidxs.begin[i] || iidx >= norm_sb_fcidxs.end[i])) {
                    ok = true;

                    // Is inner loop outside of full vectors?
                    // If so, apply mask to left or right.
                    if (iidx < norm_sb_fvidxs.begin[i])
                        mask &= peel_masks[i];
                    if (iidx >= norm_sb_fvidxs.end[i])
                        mask &= rem_masks[i];
                }
            }

            // Continue only if outside of at least one dim.
            if (ok) {
                #ifdef TRACE
                {
                    STATE_VARS(this);
                    TRACE_MSG("calc_loop_of_outer_vectors: local vector-indices [" <<
                              loop_idxs.start.make_val_str() <<
                              " ... " << loop_idxs.stop.make_val_str() <<
                              ") w/write-mask = 0x" << std::hex << mask << std::dec <<
                              " by region thread " << region_thread_idx <<
                              " and block thread " << block_thread_idx);
                }
                #endif
                #ifdef CHECK
                {
                    STATE_VARS(this);
                    // Check that only the inner dim has a range greater than one vector.
                    for (int i = 0; i < nsdims; i++) {
                        if (i != step_posn && i != inner_posn)
                            assert(loop_idxs.start[i] + 1 >= loop_idxs.stop[i]);
                    }
                }
                #endif

                // Need all starting indices.
                const Indices& start_idxs = loop_idxs.start;

                // Need stop for inner loop only.
                idx_t stop_inner = loop_idxs.stop[inner_posn];

                // Call code from stencil compiler.
                FORCE_INLINE
                    _bundle.calc_loop_of_vectors(corep, region_thread_idx, block_thread_idx,
                                                 start_idxs, stop_inner, mask);
            }
        }
    }; // StencilBundleBase.
    
    // A collection of independent stencil bundles.
    // "Independent" implies that they may be evaluated
    // in any order.
    class Stage :
        public ContextLinker,
        public std::vector<StencilBundleBase*> {

    protected:
        std::string _name;

        // Union of bounding boxes for all bundles in this stage.
        BoundingBox _stage_bb;

        // Local stage settings.
        // Only some of these will be used.
        KernelSettings _stage_opts;

        // Auto-tuner for stage settings.
        AutoTuner _at;

    public:

        // Perf stats for this stage.
        YaskTimer timer;
        idx_t steps_done = 0;
        Stats stats;

        // Work needed across points in this rank.
        idx_t num_reads_per_step = 0;
        idx_t num_writes_per_step = 0;
        idx_t num_fpops_per_step = 0;

        // Work done across all ranks.
        idx_t tot_reads_per_step = 0;
        idx_t tot_writes_per_step = 0;
        idx_t tot_fpops_per_step = 0;

        Stage(StencilContext* context,
                   const std::string& name) :
            ContextLinker(context),
            _name(name),
            _stage_opts(*context->get_state()->_opts), // init w/a copy of the base settings.
            _at(context, &_stage_opts, name) { }
        virtual ~Stage() { }

        const std::string& get_name() {
            return _name;
        }

        // Update the amount of work stats.
        // Print to current debug stream.
        void init_work_stats();

        // Determine whether step index is enabled.
        bool
        is_in_valid_step(idx_t input_step_index) const {
            if (!size())
                return false;

            // All step conditions must be the same, so
            // we call first one.
            return front()->is_in_valid_step(input_step_index);
        }

        // Accessors.
        BoundingBox& get_bb() { return _stage_bb; }
        AutoTuner& get_at() { return _at; }
        KernelSettings& get_local_settings() { return _stage_opts; }

        // If using separate stage tuners, return local settings.
        // Otherwise, return one in context.
        KernelSettings& get_active_settings() {
            STATE_VARS(this);
            return use_stage_tuners() ? _stage_opts : *opts;
        }

        // Perf-tracking methods.
        void start_timers();
        void stop_timers();
        void add_steps(idx_t num_steps);

    }; // Stage.

} // yask namespace.
