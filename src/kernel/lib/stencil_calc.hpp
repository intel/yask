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

#pragma once

namespace yask {

    // Classes that support evaluation of one stencil bundle
    // and a 'pack' of bundles.
    // A stencil context contains one or more packs.

    // A pure-virtual class base for a stencil bundle.
    class StencilBundleBase {
    protected:
        StencilContext* _generic_context = 0;
        std::string _name;
        int _scalar_fp_ops = 0;
        int _scalar_points_read = 0;
        int _scalar_points_written = 0;

        // Position of inner dim in stencil-dims tuple.
        int _inner_posn = 0;

        // Other bundles that this one depends on.
        StencilBundleSet _depends_on;

        // List of scratch-grid bundles that need to be evaluated
        // before this bundle. Listed in eval order first-to-last.
        StencilBundleList _scratch_children;

        // Whether this updates scratch grid(s);
        bool _is_scratch = false;

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
        void normalize_indices(const Indices& orig, Indices& norm) const;

    public:

        // Grids that are written to by these stencils.
        GridPtrs outputGridPtrs;

        // Grids that are read by these stencils (not necessarify
        // read-only, i.e., a grid can be input and output).
        GridPtrs inputGridPtrs;

        // Vectors of scratch grids that are written to/read from.
        ScratchVecs outputScratchVecs;
        ScratchVecs inputScratchVecs;

        // ctor, dtor.
        StencilBundleBase(StencilContext* context) :
            _generic_context(context) {

            // Find index posn of inner loop.
            auto& dims = context->get_dims();
            int ndims = dims->_stencil_dims.getNumDims();
            for (int i = 0; i < ndims; i++) {
                auto& dname = dims->_stencil_dims.getDimName(i);
                if (dname == dims->_inner_dim) {
                    _inner_posn = i;
                    break;
                }
            }
        }

        virtual ~StencilBundleBase() { }

        // Access to dims and MPI info.
        virtual DimsPtr& get_dims() const {
            return _generic_context->get_dims();
        }
        virtual MPIInfoPtr& get_mpi_info() {
            return _generic_context->get_mpi_info();
        }

        // Get name of this bundle.
        virtual const std::string& get_name() const { return _name; }

        // Get estimated number of FP ops done for one scalar eval.
        virtual int get_scalar_fp_ops() const { return _scalar_fp_ops; }

        // Get number of points read and written for one scalar eval.
        virtual int get_scalar_points_read() const { return _scalar_points_read; }
        virtual int get_scalar_points_written() const { return _scalar_points_written; }

        // Scratch accessors.
        bool is_scratch() const { return _is_scratch; }
        void set_scratch(bool is_scratch) { _is_scratch = is_scratch; }

        // Access to BBs.
        BoundingBox& getBB() { return _bundle_bb; }
        BBList& getBBs() { return _bb_list; }

        // Add dependency.
        virtual void add_dep(StencilBundleBase* eg) {
            _depends_on.insert(eg);
        }

        // Get dependencies.
        virtual const StencilBundleSet& get_deps() const {
            return _depends_on;
        }

        // Add needed scratch-bundle.
        virtual void add_scratch_child(StencilBundleBase* eg) {
            _scratch_children.push_back(eg);
        }

        // Get needed scratch-bundle(s).
        virtual const StencilBundleList& get_scratch_children() const {
            return _scratch_children;
        }

        // Get scratch children plus self.
        virtual StencilBundleList get_reqd_bundles() {
            auto sg_list = get_scratch_children();
            sg_list.push_back(this);
            return sg_list;
        }

        // If this bundle is updating scratch grid(s),
        // expand indices to calculate values in halo.
        // Adjust offsets in grids based on original idxs.
        // Return adjusted indices.
        virtual ScanIndices adjust_span(int thread_idx, const ScanIndices& idxs) const;

        // Set the bounding-box vars for this bundle in this rank.
        virtual void find_bounding_box();

        // Determine whether indices are in [sub-]domain.
        virtual bool
        is_in_valid_domain(const Indices& idxs) const =0;

        // Determine whether step index is enabled.
        virtual bool
        is_in_valid_step(idx_t input_step_index) const =0;

        // If bundle updates grid(s) with the step index,
        // set 'output_step_index' to the step that an update
        // occurs when calling one of the calc_*() methods with
        // 'input_step_index' and return 'true'.
        // Else, return 'false';
        virtual bool
        get_output_step_index(idx_t input_step_index,
                              idx_t& output_step_index) const =0;

        // Calculate one scalar result.
        virtual void
        calc_scalar(int thread_idx, const Indices& idxs) =0;

        // Calculate results within a block.
        virtual void
        calc_block(const ScanIndices& def_block_idxs);

        // Calculate results within a sub-block.
        virtual void
        calc_sub_block(int thread_idx, const ScanIndices& block_idxs);

        // Calculate a series of cluster results within an inner loop.
        // All indices start at 'start_idxs'. Inner loop iterates to
        // 'stop_inner' by 'step_inner'.
        // Indices must be rank-relative.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void
        calc_loop_of_clusters(int thread_idx,
                              const Indices& start_idxs,
                              idx_t stop_inner) =0;

        // Calculate a series of cluster results within an inner loop.
        // The 'loop_idxs' must specify a range only in the inner dim.
        // Indices must be rank-relative.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void
        calc_loop_of_clusters(int thread_idx,
                              const ScanIndices& loop_idxs);

        // Calculate a series of vector results within an inner loop.
        // All indices start at 'start_idxs'. Inner loop iterates to
        // 'stop_inner' by 'step_inner'.
        // Indices must be rank-relative.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        // Each vector write is masked by 'write_mask'.
        virtual void
        calc_loop_of_vectors(int thread_idx,
                             const Indices& start_idxs,
                             idx_t stop_inner,
                             idx_t write_mask) =0;

        // Calculate a series of vector results within an inner loop.
        // The 'loop_idxs' must specify a range only in the inner dim.
        // Indices must be rank-relative.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        // Each vector write is masked by 'write_mask'.
        virtual void
        calc_loop_of_vectors(int thread_idx,
                             const ScanIndices& loop_idxs,
                             idx_t write_mask);

    };                          // StencilBundleBase.

    // A collection of independent stencil bundles.
    // "Independent" implies that they may be evaluated
    // in any order.
    class BundlePack :
        public std::vector<StencilBundleBase*> {

    protected:
        std::string _name;

        // Parent solution.
        StencilContext* _context = 0;
        
        // Union of bounding boxes for all bundles in this pack.
        BoundingBox _pack_bb;

        // Local pack settings.
        // Only some of these will be used.
        KernelSettings _opts;

        // Auto-tuner for pack settings.
        AutoTuner _at;
        
    public:

        // Perf stats for this pack.
        YaskTimer timer;
        idx_t steps_done = 0;

        // Work needed across points in this rank.
        idx_t num_reads_per_step = 0;
        idx_t num_writes_per_step = 0;
        idx_t num_fpops_per_step = 0;

        // Work done across all ranks.
        idx_t tot_reads_per_step = 0;
        idx_t tot_writes_per_step = 0;
        idx_t tot_fpops_per_step = 0;
        
        BundlePack(const std::string& name,
                   StencilContext* ctx) :
            _name(name),
            _context(ctx),
            _opts(*ctx->get_settings()),
            _at(ctx, &_opts, name) { }
        virtual ~BundlePack() { }

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
        BoundingBox& getBB() { return _pack_bb; }
        AutoTuner& getAT() { return _at; }
        KernelSettings& getSettings() { return _opts; }

        // Perf-tracking methods.
        void start_timers();
        void stop_timers();
        void add_steps(idx_t num_steps);

    }; // BundlePack.

} // yask namespace.
