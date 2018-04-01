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
    
    /// Classes that support evaluation of one stencil group.
    /// A stencil context contains one or more groups.

    // A pure-virtual class base for a stencil group.
    class StencilGroupBase : public BoundingBox {
    protected:
        StencilContext* _generic_context = 0;
        std::string _name;
        int _scalar_fp_ops = 0;
        int _scalar_points_read = 0;
        int _scalar_points_written = 0;

        // Position of inner dim in stencil-dims tuple.
        int _inner_posn = 0;

        // Other groups that this one depends on.
        std::map<DepType, StencilGroupSet> _depends_on;

        // List of scratch-grid groups that need to be evaluated
        // before this group. Listed in eval order first-to-last.
        StencilGroupList _scratch_deps;

        // Whether this updates scratch grid(s);
        bool _is_scratch = false;

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
        StencilGroupBase(StencilContext* context) :
            _generic_context(context) {

            // Make sure map entries exist.
            for (DepType dt = DepType(0); dt < num_deps; dt = DepType(dt+1)) {
                _depends_on[dt];
            }

            // Find index posn of inner loop.
            auto dims = context->get_dims();
            int ndims = dims->_stencil_dims.getNumDims();
            for (int i = 0; i < ndims; i++) {
                auto& dname = dims->_stencil_dims.getDimName(i);
                if (dname == dims->_inner_dim) {
                    _inner_posn = i;
                    break;
                }
            }
        }

        virtual ~StencilGroupBase() { }

        // Access to dims and MPI info.
        virtual DimsPtr get_dims() const {
            return _generic_context->get_dims();
        }
        virtual MPIInfoPtr get_mpi_info() {
            return _generic_context->get_mpi_info();
        }

        // Get name of this group.
        virtual const std::string& get_name() const { return _name; }

        // Get estimated number of FP ops done for one scalar eval.
        virtual int get_scalar_fp_ops() const { return _scalar_fp_ops; }

        // Get number of points read and written for one scalar eval.
        virtual int get_scalar_points_read() const { return _scalar_points_read; }
        virtual int get_scalar_points_written() const { return _scalar_points_written; }

        // Scratch accessors.
        virtual bool is_scratch() const { return _is_scratch; }
        virtual void set_scratch(bool is_scratch) { _is_scratch = is_scratch; }
        
        // Add dependency.
        virtual void add_dep(DepType dt, StencilGroupBase* eg) {
            _depends_on.at(dt).insert(eg);
        }

        // Get dependencies.
        virtual const StencilGroupSet& get_deps(DepType dt) const {
            return _depends_on.at(dt);
        }

        // Add needed scratch-group.
        virtual void add_scratch_dep(StencilGroupBase* eg) {
            _scratch_deps.push_back(eg);
        }

        // Get needed scratch-group(s).
        virtual const StencilGroupList& get_scratch_deps() const {
            return _scratch_deps;
        }

        // If this group is updating scratch grid(s),
        // expand indices to calculate values in halo.
        // Adjust offsets in grids based on original idxs.
        // Return adjusted indices.
        virtual ScanIndices adjust_scan(int thread_idx, const ScanIndices& idxs) const;
        
        // Set the bounding-box vars for this group in this rank.
        virtual void find_bounding_box();

        // Determine whether indices are in [sub-]domain.
        virtual bool
        is_in_valid_domain(const Indices& idxs) =0;

        // Calculate one scalar result at time t.
        virtual void
        calc_scalar(int thread_idx, const Indices& idxs) =0;

        // Calculate results within a block.
        // Each block is typically computed in a separate OpenMP thread team.
        virtual void
        calc_block(const ScanIndices& region_idxs);

        // Calculate results within a sub-block.
        // Each sub-block is typically computed in a separate nested OpenMP thread.
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
    };

} // yask namespace.
