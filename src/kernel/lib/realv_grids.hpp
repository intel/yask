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

    // Underlying storage.
    typedef GenericGridTemplate<real_t> RealElemGrid;
    typedef GenericGridTemplate<real_vec_t> RealVecGrid;

    // Base class implementing all yk_grids. Can be used for grids
    // that contain either individual elements or vectors.
    class YkGridBase :
        public virtual yk_grid {

        // Rank and local offsets in domain dim:

        // | ... |        +------+       |
        // |  global ofs  |      |       |
        // |<------------>|grid  |       |
        // |     |  loc   |domain|       |
        // |rank |  ofs   |      |       |
        // | ofs |<------>|      |       |
        // |<--->|        +------+       |
        // ^     ^        ^              ^
        // |     |        |              last rank-domain index
        // |     |        0 index in underlying storage.
        // |     first rank-domain index
        // first overall-domain index

        // Rank offset is not necessarily a vector multiple.
        // Local offset must be a vector multiple.

    protected:
        // Underlying storage.  A GenericGrid is similar to a YkGrid, but it
        // doesn't have stencil features like padding, halos, offsets, etc.
        // Holds name of grid, names of dims, sizes of dims, memory layout,
        // actual data, message stream.
        GenericGridBase* _ggb = 0;

        // Problem dimensions. (NOT grid dims.)
        DimsPtr _dims;

        // The following masks have one bit for each dim in the grid.
        idx_t _step_dim_mask = 0;
        idx_t _domain_dim_mask = 0;
        idx_t _misc_dim_mask = 0;

        // The following indices have one value for each dim in the grid.
        // All values are in units of reals, not underlying elements, if different.
        // See diagram above for '_rank_offsets' and '_local_offsets'.
        // Comments show settings for domain dims | non-domain dims.
        Indices _domains;   // size of "interior" of grid | alloc size.
        Indices _req_left_pads, _req_right_pads; // requested extra space around domains | zero.
        Indices _actl_left_pads, _actl_right_pads; // actual extra space around domains | zero.
        Indices _left_halos, _right_halos; // space within pads for halo exchange | zero.
        Indices _left_wf_exts, _right_wf_exts; // additional halos for wave-fronts | zero.
        Indices _rank_offsets;   // offsets of this grid domain in overall problem | zero.
        Indices _local_offsets; // offsets of this grid domain in this rank | first index for misc.
        Indices _allocs;    // actual grid alloc in reals | same.

        // Sizes in vectors for sizes that are always vec lens (to avoid division).
        // Each entry _vec_lens may be same as _dims->_fold_pts or one, depending
        // on whether grid is fully vectorized.
        Indices _vec_lens;  // num reals in each elem | one.
        Indices _vec_left_pads; // same as _actl_left_pads.
        Indices _vec_allocs; // same as _allocs.
        Indices _vec_local_offsets; // same as _local_offsets.

        // Whether step dim is used.
        // If true, will always be in Indices::step_posn.
        bool _has_step_dim = false;

        // Whether certain dims can be changed.
        bool _is_dynamic_step_alloc = false;
        bool _is_dynamic_misc_alloc = false;

        // Data that needs to be copied to neighbor's halos if using MPI.
        // If this grid has the step dim, there is one bit per alloc'd step.
        // Otherwise, only bit 0 is used.
        std::vector<bool> _dirty_steps;

        // Data layout for slice APIs.
        bool _is_col_major = false;

        // Whether to resize this grid based on solution parameters.
        bool _fixed_size = false;

        // Whether this is a scratch grid.
        bool _is_scratch = false;

        // Whether this was created via an API.
        bool _is_new_grid = false;

        // Convenience function to format indices like
        // "x=5, y=3".
        virtual std::string makeIndexString(const Indices& idxs,
                                            std::string separator=", ",
                                            std::string infix="=",
                                            std::string prefix="",
                                            std::string suffix="") const;

        // Determine required padding from halos.
        // Does not include user-specified min padding or
        // final rounding for left pad.
        virtual Indices getReqdPad(const Indices& halos, const Indices& wf_exts) const;

        // Check whether dim exists and is of allowed type.
        virtual void checkDimType(const std::string& dim,
                                  const std::string& fn_name,
                                  bool step_ok,
                                  bool domain_ok,
                                  bool misc_ok) const;

        // Share data from source grid of type GT.
        template<typename GT>
        bool _share_data(YkGridBase* src,
                         bool die_on_failure) {
            auto* tp = dynamic_cast<GT*>(_ggb);
            if (!tp) {
                if (die_on_failure)
                    THROW_YASK_EXCEPTION("Error in share_data(): "
                                         "target grid not of expected type (internal inconsistency)");
                return false;
            }
            auto* sp = dynamic_cast<GT*>(src->_ggb);
            if (!sp) {
                if (die_on_failure)
                    THROW_YASK_EXCEPTION("Error in share_data(): source grid " +
                                         src->make_info_string() +
                                         " not of same type as target grid " +
                                         make_info_string());
                return false;
            }

            // shallow copy.
            *tp = *sp;
            return true;
        }

        // Share data from source grid.
        // Must be implemented by a concrete class
        // using the templated function above.
        virtual bool share_data(YkGridBase* src, bool die_on_failure) =0;

        // Resize or fail if already allocated.
        virtual void resize();

        // Set dirty flags in range.
        void set_dirty_in_slice(const Indices& first_indices,
                                const Indices& last_indices);

        // Make tuple needed for slicing.
        IdxTuple get_slice_range(const Indices& first_indices,
                                 const Indices& last_indices) const;

    public:
        YkGridBase(GenericGridBase* ggb,
                   const GridDimNames& dimNames,
                   DimsPtr dims);
        virtual ~YkGridBase() { }

        // Halo-exchange flag accessors.
        virtual bool is_dirty(idx_t step_idx) const;
        virtual void set_dirty(bool dirty, idx_t step_idx);
        virtual void set_dirty_all(bool dirty);
        inline void set_dirty_using_alloc_index(bool dirty, idx_t alloc_idx) {
            _dirty_steps[alloc_idx] = dirty;
        }

        // Resize flag accessors.
        virtual bool is_fixed_size() const { return _fixed_size; }
        virtual void set_fixed_size(bool is_fixed) {
            _fixed_size = is_fixed;
            if (is_fixed) {
                _rank_offsets.setFromConst(0);
                _is_dynamic_step_alloc = true;
                _is_dynamic_misc_alloc = true;
            }
        }
        virtual bool is_dynamic_step_alloc() const {
            return _is_dynamic_step_alloc;
        }
        virtual void _set_dynamic_step_alloc(bool is_dynamic) {
            _is_dynamic_step_alloc = is_dynamic;
        }
        virtual bool is_dynamic_misc_alloc() const {
            return _is_dynamic_misc_alloc;
        }
        virtual void _set_dynamic_misc_alloc(bool is_dynamic) {
            _is_dynamic_misc_alloc = is_dynamic;
        }
        
        // Scratch accessors.
        virtual bool is_scratch() const { return _is_scratch; }
        virtual void set_scratch(bool is_scratch) {
            _is_scratch = is_scratch;
            if (is_scratch)
                _rank_offsets.setFromConst(0);
        }

        // New-grid accessors.
        virtual bool is_new_grid() const { return _is_new_grid; }
        virtual void set_new_grid(bool is_new_grid) {
            _is_new_grid = is_new_grid;
            if (_is_new_grid)
                _is_dynamic_step_alloc = true;
                _is_dynamic_misc_alloc = true;
        }        

        // NUMA accessors.
        virtual int get_numa_preferred() const { return _ggb->get_numa_pref(); }
        virtual bool set_numa_preferred(int numa_node) {
            return _ggb->set_numa_pref(numa_node);
        }

        // Lookup position by dim name.
        // Return -1 or die if not found, depending on flag.
        virtual int get_dim_posn(const std::string& dim,
                                 bool die_on_failure = false,
                                 const std::string& die_msg = "") const;

        // Get dim name by posn.
        virtual const std::string& get_dim_name(int n) const {
            assert(n >= 0);
            assert(n < get_num_dims());
            return _ggb->get_dim_name(n);
        }

        // Adjust logical time index to 0-based index
        // using temporal allocation size.
        inline idx_t _wrap_step(idx_t t) const {

            // Index wraps in tdim.
            // Examples based on tdim == 2:
            //  t => return value.
            // ---  -------------
            // -2 => 0.
            // -1 => 1.
            //  0 => 0.
            //  1 => 1.
            //  2 => 0.

            // Avoid discontinuity caused by dividing negative numbers by
            // using floored-mod.
            idx_t res = imod_flr(t, _domains[+Indices::step_posn]);
            return res;
        }

        // Convert logical step index to index in allocated range.
        // If this grid doesn't use the step dim, returns 0.
        inline idx_t get_alloc_step_index(const Indices& indices) const {
            return _has_step_dim ? _wrap_step(indices[+Indices::step_posn]) : 0;
        }

        // Get grid dims with allocations in number of reals.
        virtual IdxTuple get_allocs() const {
            IdxTuple allocs = _ggb->get_dims(); // make a copy.
            _allocs.setTupleVals(allocs);
            return allocs;
        }

        // Get the messsage output stream.
        virtual std::ostream& get_ostr() const {
            return _ggb->get_ostr();
        }

        // Make a human-readable description.
        virtual std::string make_info_string() const =0;

        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const YkGridBase* ref,
                              real_t epsilon = EPSILON,
                              int maxPrint = 20,
                              std::ostream& os = std::cerr) const;

        // Make sure indices are in range.
        // Optionally fix them to be in range and return in 'fixed_indices'.
        // If 'normalize', make rank-relative, divide by vlen and return in 'fixed_indices'.
        virtual bool checkIndices(const Indices& indices,
                                  const std::string& fn,
                                  bool strict_indices,
                                  bool normalize,
                                  Indices* fixed_indices = NULL) const;

        // Set elements to a sequence of values using seed.
        // Cf. set_all_elements_same().
        virtual void set_all_elements_in_seq(double seed) =0;

        // Get a pointer to one element.
        // Indices are relative to overall problem domain.
        // 'alloc_step_idx' is the pre-computed step index "wrapped"
        // to be within the allocated space. This avoids lots of 'idiv' instrs.
        // Methods are implemented in concrete classes for efficiency.
        virtual const real_t* getElemPtr(const Indices& idxs,
                                         idx_t alloc_step_idx,
                                         bool checkBounds=true) const =0;
        virtual real_t* getElemPtr(const Indices& idxs,
                                   idx_t alloc_step_idx,
                                   bool checkBounds=true) =0;

        // Read one element.
        // Indices are relative to overall problem domain.
        virtual real_t readElem(const Indices& idxs,
                                idx_t alloc_step_idx,
                                int line) const =0;

        // Write one element.
        // Indices are relative to overall problem domain.
        inline void writeElem(real_t val,
                              const Indices& idxs,
                              idx_t alloc_step_idx,
                              int line) {
            real_t* ep = getElemPtr(idxs, alloc_step_idx);
            *ep = val;
#ifdef TRACE_MEM
            printElem("writeElem", idxs, val, line);
#endif
        }

        // Update one element.
        // Indices are relative to overall problem domain.
        inline void addToElem(real_t val,
                              const Indices& idxs,
                              idx_t alloc_step_idx,
                              int line) {
            real_t* ep = getElemPtr(idxs, alloc_step_idx);

#pragma omp atomic update
            *ep += val;
#ifdef TRACE_MEM
            printElem("addToElem", idxs, *ep, line);
#endif
        }

        // Print one element.
        virtual void printElem(const std::string& msg,
                               const Indices& idxs,
                               real_t e,
                               int line,
                               bool newline = true) const;

        // Print one vector.
        // Indices must be normalized and rank-relative.
        virtual void printVecNorm(const std::string& msg,
                                  const Indices& idxs,
                                  const real_vec_t& val,
                                  int line,
                                  bool newline = true) const;

        // APIs not defined above.
        // See yask_kernel_api.hpp.
        virtual const std::string& get_name() const {
            return _ggb->get_name();
        }
        virtual bool is_dim_used(const std::string& dim) const {
            return _ggb->is_dim_used(dim);
        }
        virtual int get_num_dims() const {
            return _ggb->get_num_dims();
        }
        virtual GridDimNames get_dim_names() const {
            std::vector<std::string> dims;
            for (int i = 0; i < get_num_dims(); i++)
                dims.push_back(get_dim_name(i));
            return dims;
        }

        // Possibly vectorized version of set/get_elements_in_slice().
        virtual idx_t set_vecs_in_slice(const void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices) {
            return set_elements_in_slice(buffer_ptr, first_indices, last_indices);
        }
        virtual idx_t get_vecs_in_slice(void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices) const {
            return get_elements_in_slice(buffer_ptr, first_indices, last_indices);
        }

#define GET_GRID_API(api_name)                                      \
        virtual idx_t api_name(const std::string& dim) const;       \
        virtual idx_t api_name(int posn) const;
#define SET_GRID_API(api_name)                                      \
        virtual void api_name(const std::string& dim, idx_t n);     \
        virtual void api_name(int posn, idx_t n);

        // Settings that should never be exposed as APIs because
        // they can break the usage model.
        // They are not protected because they are used from outside
        // this class hierarchy.
        GET_GRID_API(_get_first_alloc_index)
        GET_GRID_API(_get_last_alloc_index)
        GET_GRID_API(_get_left_wf_ext)
        GET_GRID_API(_get_local_offset)
        GET_GRID_API(_get_rank_offset)
        GET_GRID_API(_get_right_wf_ext)
        GET_GRID_API(_get_vec_len)

        SET_GRID_API(_set_alloc_size)
        SET_GRID_API(_set_domain_size)
        SET_GRID_API(_set_left_pad_size)
        SET_GRID_API(_set_left_wf_ext)
        SET_GRID_API(_set_local_offset)
        SET_GRID_API(_set_rank_offset)
        SET_GRID_API(_set_right_pad_size)
        SET_GRID_API(_set_right_wf_ext)

        // Exposed APIs.
        GET_GRID_API(get_rank_domain_size)
        GET_GRID_API(get_first_rank_domain_index)
        GET_GRID_API(get_last_rank_domain_index)
        GET_GRID_API(get_left_halo_size)
        GET_GRID_API(get_right_halo_size)
        GET_GRID_API(get_halo_size)
        GET_GRID_API(get_first_rank_halo_index)
        GET_GRID_API(get_last_rank_halo_index)
        GET_GRID_API(get_left_extra_pad_size)
        GET_GRID_API(get_right_extra_pad_size)
        GET_GRID_API(get_extra_pad_size)
        GET_GRID_API(get_left_pad_size)
        GET_GRID_API(get_right_pad_size)
        GET_GRID_API(get_pad_size)
        GET_GRID_API(get_alloc_size)
        GET_GRID_API(get_first_rank_alloc_index)
        GET_GRID_API(get_last_rank_alloc_index)
        GET_GRID_API(get_first_misc_index)
        GET_GRID_API(get_last_misc_index)

        SET_GRID_API(set_left_halo_size)
        SET_GRID_API(set_right_halo_size)
        SET_GRID_API(set_halo_size)
        SET_GRID_API(set_left_min_pad_size)
        SET_GRID_API(set_right_min_pad_size)
        SET_GRID_API(set_min_pad_size)
        SET_GRID_API(set_left_extra_pad_size)
        SET_GRID_API(set_right_extra_pad_size)
        SET_GRID_API(set_extra_pad_size)
        SET_GRID_API(set_alloc_size)
        SET_GRID_API(set_first_misc_index)

#undef GET_GRID_API
#undef SET_GRID_API

        virtual std::string format_indices(const Indices& indices) const {
            std::string str = get_name() + "(" + makeIndexString(indices) + ")";
            return str;
        }
        virtual std::string format_indices(const GridIndices& indices) const {
            const Indices indices2(indices);
            return format_indices(indices2);
        }
        virtual std::string format_indices(const std::initializer_list<idx_t>& indices) const {
            const Indices indices2(indices);
            return format_indices(indices2);
        }

        virtual bool is_element_allocated(const Indices& indices) const;
        virtual bool is_element_allocated(const GridIndices& indices) const {
            const Indices indices2(indices);
            return is_element_allocated(indices2);
        }
        virtual bool is_element_allocated(const std::initializer_list<idx_t>& indices) const {
            const Indices indices2(indices);
            return is_element_allocated(indices2);
        }

        virtual double get_element(const Indices& indices) const;
        virtual double get_element(const GridIndices& indices) const {
            const Indices indices2(indices);
            return get_element(indices2);
        }
        virtual double get_element(const std::initializer_list<idx_t>& indices) const {
            const Indices indices2(indices);
            return get_element(indices2);
        }
        virtual idx_t get_elements_in_slice(void* buffer_ptr,
                                            const Indices& first_indices,
                                            const Indices& last_indices) const;
        virtual idx_t get_elements_in_slice(void* buffer_ptr,
                                            const GridIndices& first_indices,
                                            const GridIndices& last_indices) const {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return get_elements_in_slice(buffer_ptr, first, last);
        }
        virtual idx_t set_element(double val,
                                  const Indices& indices,
                                  bool strict_indices = false);
        virtual idx_t set_element(double val,
                                  const GridIndices& indices,
                                  bool strict_indices = false) {
            const Indices indices2(indices);
            return set_element(val, indices2, strict_indices);
        }
        virtual idx_t set_element(double val,
                                  const std::initializer_list<idx_t>& indices,
                                  bool strict_indices = false) {
            const Indices indices2(indices);
            return set_element(val, indices2, strict_indices);
        }
        virtual idx_t add_to_element(double val,
                                  const Indices& indices,
                                  bool strict_indices = false);
        virtual idx_t add_to_element(double val,
                                  const GridIndices& indices,
                                  bool strict_indices = false) {
            const Indices indices2(indices);
            return add_to_element(val, indices2, strict_indices);
        }
        virtual idx_t add_to_element(double val,
                                  const std::initializer_list<idx_t>& indices,
                                  bool strict_indices = false) {
            const Indices indices2(indices);
            return add_to_element(val, indices2, strict_indices);
        }

        virtual void set_all_elements_same(double val) =0;
        virtual idx_t set_elements_in_slice_same(double val,
                                                 const Indices& first_indices,
                                                 const Indices& last_indices,
                                                 bool strict_indices);
        virtual idx_t set_elements_in_slice_same(double val,
                                                 const GridIndices& first_indices,
                                                 const GridIndices& last_indices,
                                                 bool strict_indices) {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return set_elements_in_slice_same(val, first, last, strict_indices);
        }

        virtual idx_t set_elements_in_slice(const void* buffer_ptr,
                                            const Indices& first_indices,
                                            const Indices& last_indices);
        virtual idx_t set_elements_in_slice(const void* buffer_ptr,
                                            const GridIndices& first_indices,
                                            const GridIndices& last_indices) {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return set_elements_in_slice(buffer_ptr, first, last);
        }

        virtual void alloc_storage() {
            _ggb->default_alloc();
            get_ostr() << make_info_string() << std::endl;
        }
        virtual void release_storage() {
            _ggb->release_storage();
        }
        virtual void share_storage(yk_grid_ptr source);
        virtual bool is_storage_allocated() const {
            return _ggb->get_storage() != 0;
        }
        virtual idx_t get_num_storage_bytes() const {
            return idx_t(_ggb->get_num_bytes());
        }
        virtual idx_t get_num_storage_elements() const {
            return _allocs.product();
        }
        virtual bool is_storage_layout_identical(const yk_grid_ptr other) const;
        virtual void* get_raw_storage_buffer() {
            return _ggb->get_storage();
        }
        virtual void set_storage(std::shared_ptr<char> base, size_t offset) {
            _ggb->set_storage(base, offset);
        }
    };

    // YASK grid of real elements.
    // Used for grids that do not contain folded vectors.
    // If '_use_step_idx', then index to step dim will wrap around.
    template <typename LayoutFn, bool _use_step_idx>
    class YkElemGrid : public YkGridBase {

    protected:
        typedef GenericGrid<real_t, LayoutFn> _grid_type;
        _grid_type _data;

        // Share data from source grid.
        virtual bool share_data(YkGridBase* src, bool die_on_failure) {
            return _share_data<_grid_type>(src, die_on_failure);
        }

    public:
        YkElemGrid(DimsPtr dims,
                   std::string name,
                   const GridDimNames& dimNames,
                   KernelSettingsPtr* settings,
                   std::ostream** ostr) :
            YkGridBase(&_data, dimNames, dims),
            _data(name, dimNames, settings, ostr) {
            _has_step_dim = _use_step_idx;
            resize();
        }

        // Get num dims from compile-time const.
        virtual int get_num_dims() const final {
            return _data.get_num_dims();
        }

        // Make a human-readable description.
        virtual std::string make_info_string() const {
            return _data.make_info_string("FP");
        }

        // Init data.
        virtual void set_all_elements_same(double seed) {
            _data.set_elems_same(seed);
            set_dirty_all(true);
        }
        virtual void set_all_elements_in_seq(double seed) {
            _data.set_elems_in_seq(seed);
            set_dirty_all(true);
        }

        // Get a pointer to given element.
        virtual const real_t* getElemPtr(const Indices& idxs,
                                         idx_t alloc_step_idx,
                                         bool checkBounds=true) const final {

#ifdef TRACE_MEM
            _data.get_ostr() << get_name() << "." << "YkElemGrid::getElemPtr(" <<
                idxs.makeValStr(get_num_dims()) << ")";
#endif
            const auto n = _data.get_num_dims();
            Indices adj_idxs(n);

            // Special handling for step index.
            auto sp = +Indices::step_posn;
            if (_use_step_idx) {
                assert(alloc_step_idx == _wrap_step(idxs[sp]));
                adj_idxs[sp] = alloc_step_idx;
            }

#pragma unroll
            // All other indices.
            for (int i = 0; i < n; i++) {
                if (!(_use_step_idx && i == sp)) {

                    // Adjust for offsets and padding.
                    // This gives a positive 0-based local element index.
                    idx_t ai = idxs[i] + _actl_left_pads[i] -
                        (_rank_offsets[i] + _local_offsets[i]);
                    assert(ai >= 0);
                    adj_idxs[i] = uidx_t(ai);
                }
            }

#ifdef TRACE_MEM
            if (checkBounds)
                _data.get_ostr() << " => " << _data.get_index(adj_idxs);
            _data.get_ostr() << std::endl << std::flush;
#endif

            // Get pointer via layout in _data.
            return _data.getPtr(adj_idxs, checkBounds);
        }

        // Non-const version.
        virtual real_t* getElemPtr(const Indices& idxs,
                                   idx_t alloc_step_idx,
                                   bool checkBounds=true) final {

            const real_t* p =
                const_cast<const YkElemGrid*>(this)->getElemPtr(idxs, alloc_step_idx, checkBounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        // Indices are relative to overall problem domain.
        virtual real_t readElem(const Indices& idxs,
                                idx_t alloc_step_idx,
                                int line) const final {
            const real_t* ep = YkElemGrid::getElemPtr(idxs, alloc_step_idx);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem("readElem", idxs, e, line);
#endif
            return e;
        }

    };                          // YkElemGrid.

    // YASK grid of real vectors.
    // Used for grids that contain all the folded dims.
    // If '_use_step_idx', then index to step dim will wrap around.
    // The '_templ_vec_lens' arguments must contain a list of vector lengths
    // corresponding to each dim in the grid.
    template <typename LayoutFn, bool _use_step_idx, idx_t... _templ_vec_lens>
    class YkVecGrid : public YkGridBase {

    protected:
        typedef GenericGrid<real_vec_t, LayoutFn> _grid_type;
        _grid_type _data;

        // Positions of grid dims in vector fold dims.
        Indices _vec_fold_posns;

        // Share data from source grid.
        virtual bool share_data(YkGridBase* src, bool die_on_failure) {
            return _share_data<_grid_type>(src, die_on_failure);
        }

    public:
        YkVecGrid(DimsPtr dims,
                  const std::string& name,
                  const GridDimNames& dimNames,
                  KernelSettingsPtr* settings,
                  std::ostream** ostr) :
            YkGridBase(&_data, dimNames, dims),
            _data(name, dimNames, settings, ostr),
            _vec_fold_posns(idx_t(0), int(dimNames.size())) {
            _has_step_dim = _use_step_idx;

            // Template vec lengths.
            const int nvls = sizeof...(_templ_vec_lens);
            const idx_t vls[nvls] { _templ_vec_lens... };
            assert((size_t)nvls == dimNames.size());

            // Init vec sizes.
            // For each dim in the grid, use the number of vector
            // fold points or 1 if not set.
            for (size_t i = 0; i < dimNames.size(); i++) {
                auto& dname = dimNames.at(i);
                auto* p = dims->_vec_fold_pts.lookup(dname);
                idx_t dval = p ? *p : 1;
                _vec_lens[i] = dval;
                _vec_allocs[i] = dval;

                // Compare to template parameter pack.
                assert(dval == vls[i]);
            }

            // Init grid-dim positions of fold dims.
            assert(dims->_vec_fold_pts.getNumDims() == NUM_VEC_FOLD_DIMS);
            for (int i = 0; i < NUM_VEC_FOLD_DIMS; i++) {
                auto& fdim = dims->_vec_fold_pts.getDimName(i);
                int j = get_dim_posn(fdim, true,
                                     "internal error: folded grid missing folded dim");
                assert(j >= 0);
                _vec_fold_posns[i] = j;
            }

            resize();
        }

        // Get num dims from compile-time const.
        virtual int get_num_dims() const final {
            return _data.get_num_dims();
        }

        // Make a human-readable description.
        virtual std::string make_info_string() const {
            return _data.make_info_string("SIMD FP");
        }

        // Init data.
        virtual void set_all_elements_same(double seed) {
            real_vec_t seedv = seed; // bcast.
            _data.set_elems_same(seedv);
            set_dirty_all(true);
        }
        virtual void set_all_elements_in_seq(double seed) {
            real_vec_t seedv;
            auto n = seedv.get_num_elems();

            // Init elements to values between seed and 2*seed.
            // For example if n==4, init to
            // seed * 1.0, seed * 1.25, seed * 1.5, seed * 1.75.
            for (int i = 0; i < n; i++)
                seedv[i] = seed * (1.0 + double(i) / n);
            _data.set_elems_in_seq(seedv);
            set_dirty_all(true);
        }

        // Get a pointer to given element.
        virtual const real_t* getElemPtr(const Indices& idxs,
                                         idx_t alloc_step_idx,
                                         bool checkBounds=true) const final {

#ifdef TRACE_MEM
            _data.get_ostr() << get_name() << "." << "YkVecGrid::getElemPtr(" <<
                idxs.makeValStr(get_num_dims()) << ")";
#endif
            // Use template vec lengths instead of run-time values for
            // efficiency.
            static constexpr int nvls = sizeof...(_templ_vec_lens);
            static constexpr uidx_t vls[nvls] { _templ_vec_lens... };
            Indices vec_idxs(nvls), elem_ofs(nvls);
#ifdef DEBUG_LAYOUT
            const auto nd = _data.get_num_dims();
            assert(nd == nvls);
#endif

            // Special handling for step index.
            auto sp = +Indices::step_posn;
            if (_use_step_idx) {
                assert(alloc_step_idx == _wrap_step(idxs[sp]));
                vec_idxs[sp] = alloc_step_idx;
                elem_ofs[sp] = 0;
            }

            // Try to force compiler to use shifts instead of DIV and MOD
            // when the vec-lengths are 2^n.
#pragma unroll
#pragma novector
            // All other indices.
            for (int i = 0; i < nvls; i++) {
                if (!(_use_step_idx && i == sp)) {

                    // Adjust for offset and padding.
                    // This gives a positive 0-based local element index.
                    idx_t ai = idxs[i] + _actl_left_pads[i] -
                        (_rank_offsets[i] + _local_offsets[i]);
                    assert(ai >= 0);
                    uidx_t adj_idx = uidx_t(ai);

                    // Get vector index and offset.
                    // Use unsigned DIV and MOD to avoid compiler having to
                    // emit code for preserving sign when using shifts.
                    vec_idxs[i] = idx_t(adj_idx / vls[i]);
                    elem_ofs[i] = idx_t(adj_idx % vls[i]);
                    assert(vec_idxs[i] == idx_t(adj_idx / _vec_lens[i]));
                    assert(elem_ofs[i] == idx_t(adj_idx % _vec_lens[i]));
                }
            }

            // Get only the vectorized fold offsets, i.e., those
            // with vec-lengths > 1.
            // And, they need to be in the original folding order,
            // which might be different than the grid-dim order.
            Indices fold_ofs(NUM_VEC_FOLD_DIMS);
#pragma unroll
            for (int i = 0; i < NUM_VEC_FOLD_DIMS; i++) {
                int j = _vec_fold_posns[i];
                fold_ofs[i] = elem_ofs[j];
            }

            // Get 1D element index into vector.
            auto i = _dims.get()->getElemIndexInVec(fold_ofs);

#ifdef DEBUG_LAYOUT
            // Compare to more explicit offset extraction.
            IdxTuple eofs = get_allocs(); // get dims for this grid.
            elem_ofs.setTupleVals(eofs);  // set vals from elem_ofs.
            auto i2 = _dims->getElemIndexInVec(eofs);
            assert(i == i2);
#endif

#ifdef TRACE_MEM
            if (checkBounds)
                _data.get_ostr() << " => " << _data.get_index(vec_idxs) <<
                    "[" << i << "]";
            _data.get_ostr() << std::endl << std::flush;
#endif

            // Get pointer to vector.
            const real_vec_t* vp = _data.getPtr(vec_idxs, checkBounds);

            // Get pointer to element.
            const real_t* ep = &(*vp)[i];
            return ep;
        }

        // Non-const version.
        virtual real_t* getElemPtr(const Indices& idxs,
                                   idx_t alloc_step_idx,
                                   bool checkBounds=true) final {

            const real_t* p =
                const_cast<const YkVecGrid*>(this)->getElemPtr(idxs, alloc_step_idx,
                                                               checkBounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        // Indices are relative to overall problem domain.
        virtual real_t readElem(const Indices& idxs,
                                idx_t alloc_step_idx,
                                int line) const final {
            const real_t* ep = YkVecGrid::getElemPtr(idxs, alloc_step_idx);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem("readElem", idxs, e, line);
#endif
            return e;
        }

        // Get a pointer to given vector.
        // Indices must be normalized and rank-relative.
        // It's important that this function be efficient, since
        // it's indiectly used from the stencil kernel.
        inline const real_vec_t* getVecPtrNorm(const Indices& vec_idxs,
                                               idx_t alloc_step_idx,
                                               bool checkBounds=true) const {

#ifdef TRACE_MEM
            _data.get_ostr() << get_name() << "." << "YkVecGrid::getVecPtrNorm(" <<
                vec_idxs.makeValStr(get_num_dims()) << ")";
#endif

            static constexpr int nvls = sizeof...(_templ_vec_lens);
#ifdef DEBUG_LAYOUT
            const auto nd = _data.get_num_dims();
            assert(nd == nvls);
#endif
            Indices adj_idxs(nvls);

            // Special handling for step index.
            auto sp = +Indices::step_posn;
            if (_use_step_idx) {
                assert(alloc_step_idx == _wrap_step(vec_idxs[sp]));
                adj_idxs[sp] = alloc_step_idx;
            }

#pragma unroll
            // Domain indices.
            for (int i = 0; i < nvls; i++) {
                if (!(_use_step_idx && i == sp)) {

                    // Adjust for padding.
                    // Since the indices are rank-relative, subtract only
                    // the local offsets. (Compare to getElemPtr().)
                    // This gives a 0-based local *vector* index.
                    adj_idxs[i] = vec_idxs[i] + _vec_left_pads[i] - _vec_local_offsets[i];
                }
            }

#ifdef TRACE_MEM
            _data.get_ostr() << " => " << _data.get_index(adj_idxs, checkBounds) <<
                std::endl << std::flush;
#endif

            // Get ptr via layout in _data.
            return _data.getPtr(adj_idxs, checkBounds);
        }

        // Non-const version.
        inline real_vec_t* getVecPtrNorm(const Indices& vec_idxs,
                                         idx_t alloc_step_idx,
                                         bool checkBounds=true) {

            const real_vec_t* p =
                const_cast<const YkVecGrid*>(this)->getVecPtrNorm(vec_idxs,
                                                                  alloc_step_idx, checkBounds);
            return const_cast<real_vec_t*>(p);
        }

        // Read one vector.
        // Indices must be normalized and rank-relative.
        inline real_vec_t readVecNorm(const Indices& vec_idxs,
                                      idx_t alloc_step_idx,
                                      int line) const {
            const real_vec_t* vp = getVecPtrNorm(vec_idxs, alloc_step_idx);
            real_vec_t v = *vp;
#ifdef TRACE_MEM
            printVecNorm("readVecNorm", vec_idxs, v, line);
#endif
            return v;
        }

        // Write one vector.
        // Indices must be normalized and rank-relative.
        inline void writeVecNorm(real_vec_t val,
                                 const Indices& vec_idxs,
                                 idx_t alloc_step_idx,
                                 int line) {
            real_vec_t* vp = getVecPtrNorm(vec_idxs, alloc_step_idx);
            *vp = val;
#ifdef TRACE_MEM
            printVecNorm("writeVecNorm", vec_idxs, val, line);
#endif
        }

        // Prefetch one vector.
        // Indices must be normalized and rank-relative.
        template <int level>
        ALWAYS_INLINE
        void prefetchVecNorm(const Indices& vec_idxs,
                             idx_t alloc_step_idx,
                             int line) const {
#ifdef TRACE_MEM
            std::cout << "prefetchVecNorm<" << level << ">(" <<
                makeIndexString(vec_idxs.multElements(_vec_lens)) <<
                ")" << std::endl;
#endif
            auto p = getVecPtrNorm(vec_idxs, alloc_step_idx, false);
            prefetch<level>(p);
#ifdef MODEL_CACHE
            cache_model.prefetch(p, level, line);
#endif
        }

        // Vectorized version of set/get_elements_in_slice().
        virtual idx_t set_vecs_in_slice(const void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices) {
            if (!is_storage_allocated())
                return 0;
            Indices firstv, lastv;
            checkIndices(first_indices, "set_vecs_in_slice", true, true, &firstv);
            checkIndices(last_indices, "set_vecs_in_slice", true, true, &lastv);

            // Find range.
            IdxTuple numVecsTuple = get_slice_range(firstv, lastv);
            TRACE_MSG0(get_ostr(), "set_vecs_in_slice: setting " <<
                       numVecsTuple.makeDimValStr(" * ") << " vecs at " <<
                       makeIndexString(firstv) << " ... " <<
                       makeIndexString(lastv));

            // Visit points in slice.
            // TODO: visit points in all but last dim and write a simple loop for last dim.
            numVecsTuple.visitAllPointsInParallel
                ([&](const IdxTuple& ofs,
                     size_t idx) {
                    Indices pt = firstv.addElements(ofs);
                    real_vec_t val = ((real_vec_t*)buffer_ptr)[idx];

                    // TODO: move this outside of parallel loop when
                    // step index is const.
                    idx_t asi = get_alloc_step_index(pt);

                    writeVecNorm(val, pt, asi, __LINE__);
                    return true;    // keep going.
                });

            // Set appropriate dirty flag(s).
            set_dirty_in_slice(first_indices, last_indices);

            return numVecsTuple.product() * VLEN;
        }
        virtual idx_t get_vecs_in_slice(void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices) const {
            if (!is_storage_allocated()) {
                yask_exception e;
                std::stringstream err;
                err << "Error: call to 'get_vecs_in_slice' with no data allocated for grid '" <<
                    get_name() << "'.\n";
                e.add_message(err.str());
                throw e;
                //exit_yask(1);
            }
            Indices firstv, lastv;
            checkIndices(first_indices, "get_vecs_in_slice", true, true, &firstv);
            checkIndices(last_indices, "get_vecs_in_slice", true, true, &lastv);

            // Find range.
            IdxTuple numVecsTuple = get_slice_range(firstv, lastv);
            TRACE_MSG0(get_ostr(), "get_vecs_in_slice: getting " <<
                       numVecsTuple.makeDimValStr(" * ") << " vecs at " <<
                       makeIndexString(firstv) << " ... " <<
                       makeIndexString(lastv));

            // Visit points in slice.
            numVecsTuple.visitAllPointsInParallel
                ([&](const IdxTuple& ofs,
                     size_t idx) {
                    Indices pt = firstv.addElements(ofs);

                    // TODO: move this outside of parallel loop when
                    // step index is const.
                    idx_t asi = get_alloc_step_index(pt);

                    real_vec_t val = readVecNorm(pt, asi, __LINE__);
                    ((real_vec_t*)buffer_ptr)[idx] = val;
                    return true;    // keep going.
                });
            return numVecsTuple.product() * VLEN;
        }

    };                          // YkVecGrid.

}                               // namespace.
