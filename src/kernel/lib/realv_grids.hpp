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

#pragma once

namespace yask {

    // Underlying storage.
    typedef GenericGridTemplate<real_t> RealElemGrid;
    typedef GenericGridTemplate<real_vec_t> RealVecGrid;

    // Base class implementing all yk_grids. Can be used for grids
    // that contain either individual elements or vectors.
    class YkGridBase :
        public virtual yk_grid {

    protected:
        // Underlying storage.  A GenericGrid is similar to a YkGrid, but it
        // doesn't have stencil features like padding, halos, offsets, etc.
        // Holds name of grid, names of dims, sizes of dims, memory layout,
        // actual data, message stream.
        GenericGridBase* _ggb = 0;

        // Problem dimensions. (NOT grid dims.)
        DimsPtr _dims;

        // The following indices have values for all dims in the grid.
        // All values are in units of reals, not underlying elements, if different.
        // Settings for domain dims | non-domain dims.
        Indices _domains;   // rank domain sizes copied from the solution | alloc size.
        Indices _pads;      // extra space around domains | zero.
        Indices _halos;     // space within pads for halo exchange | zero.
        Indices _offsets;   // offsets of this rank in overall domain | first index.
        Indices _vec_lens;  // num reals in each elem | one.
        Indices _allocs;    // actual grid allocation in reals | as domain dims.

        // Indices in vectors for sizes that are always vec lens (to avoid division).
        Indices _vec_pads;
        Indices _vec_allocs;

        // Whether step dim is used.
        // If true, will always be in Indices::step_posn.
        bool _has_step_dim = false;
        
        // Data that needs to be copied to neighbor's halos if using MPI.
        // If this grid has the step dim, there is one bit per alloc'd step.
        // Otherwise, only bit 0 is used.
        std::vector<bool> _dirty_steps;

        // Data layout for slice APIs.
        bool _is_col_major = false;

        // Convenience function to format indices like
        // "x=5, y=3".
        virtual std::string makeIndexString(const Indices& idxs,
                                            std::string separator=", ",
                                            std::string infix="=",
                                            std::string prefix="",
                                            std::string suffix="") const;

        // Adjust logical time index to 0-based index
        // using temporal allocation size.
        inline idx_t wrap_step(idx_t t) const {

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
            // adding a large offset to the t index.  So, t can be negative,
            // but not so much that it would still be negative after adding
            // the offset.  This should not be a practical restriction.
            t += idx_t(0x10000);
            assert(t >= 0);

            // Do MOD with unsigned ints--simpler generated code.
            uidx_t res = uidx_t(t) % uidx_t(_domains[Indices::step_posn]);
            return idx_t(res);
        }
        
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
                if (die_on_failure) {
                    std::cerr << "Error in share_data(): "
                        "target grid not of expected type (internal inconsistency).\n";
                    exit_yask(1);
                }
                return false;
            }
            auto* sp = dynamic_cast<GT*>(src->_ggb);
            if (!sp) {
                if (die_on_failure) {
                    std::cerr << "Error in share_data(): source grid ";
                    src->print_info(std::cerr);
                    std::cerr << " not of same type as target grid ";
                    print_info(std::cerr);
                    std::cerr << ".\n";
                    exit_yask(1);
                }
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

    public:
        YkGridBase(GenericGridBase* ggb, size_t ndims, DimsPtr dims) :
            _ggb(ggb), _dims(dims) {

            assert(ggb);
            assert(dims.get());

            // Init indices.
            int n = int(ndims);
            _domains.setFromConst(0, n);
            _pads.setFromConst(0, n);
            _halos.setFromConst(0, n);
            _offsets.setFromConst(0, n);
            _vec_lens.setFromConst(1, n);
            _allocs.setFromConst(1, n);
            _vec_pads.setFromConst(1, n);
            _vec_allocs.setFromConst(1, n);
            
        }
        virtual ~YkGridBase() { }

        // Halo-exchange flag accessors.
        virtual bool is_dirty(idx_t step_idx) const;
        virtual void set_dirty(bool dirty, idx_t step_idx);
        virtual void set_dirty_all(bool dirty);
        
        // Lookup position by dim name.
        // Return -1 or die if not found, depending on flag.
        virtual int get_dim_posn(const std::string& dim,
                                 bool die_on_failure = false,
                                 const std::string& die_msg = "") const;

        // Get grid dims with allocations in number of reals.
        virtual IdxTuple get_allocs() const {
            IdxTuple allocs = _ggb->get_dims();
            _allocs.setTupleVals(allocs);
            return allocs;
        }
        
        // Get the messsage output stream.
        virtual std::ostream& get_ostr() const {
            return _ggb->get_ostr();
        }

        // Print some info to 'os'.
        virtual void print_info(std::ostream& os) const =0;
  
        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const YkGridBase* ref,
                              real_t epsilon = EPSILON,
                              int maxPrint = 20,
                              std::ostream& os = std::cerr) const;

        // Make sure indices are in range.
        // Optionally fix them to be in range and return in 'fixed_indices'.
        virtual bool checkIndices(const Indices& indices,
                                  const std::string& fn,
                                  bool strict_indices,
                                  Indices* fixed_indices = NULL) const;

        // Set elements to a sequence of values using seed.
        // Cf. set_all_elements_same().
        virtual void set_all_elements_in_seq(double seed) =0;
        
        // Get a pointer to one element.
        // Indices are relative to overall problem domain.
        // Implemented in concrete classes for efficiency.
        virtual const real_t* getElemPtr(const Indices& idxs,
                                         bool checkBounds=true) const =0;
        virtual real_t* getElemPtr(const Indices& idxs,
                                   bool checkBounds=true) =0;

        // Read one element.
        // Indices are relative to overall problem domain.
        virtual real_t readElem(const Indices& idxs,
                                int line) const =0;

        // Write one element.
        // Indices are relative to overall problem domain.
        inline void writeElem(real_t val,
                              const Indices& idxs,
                              int line) {
            real_t* ep = getElemPtr(idxs);
            *ep = val;
#ifdef TRACE_MEM
            printElem("writeElem", idxs, val, line);
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
        virtual const std::string& get_dim_name(int n) const {
            assert(n >= 0);
            assert(n < get_num_dims());
            return _ggb->get_dim_name(n);
        }
        virtual GridDimNames get_dim_names() const {
            std::vector<std::string> dims;
            for (int i = 0; i < get_num_dims(); i++)
                dims.push_back(get_dim_name(i));
            return dims;
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
        GET_GRID_API(_get_offset)
        GET_GRID_API(_get_first_alloc_index)
        GET_GRID_API(_get_last_alloc_index)
        SET_GRID_API(_set_domain_size)
        SET_GRID_API(_set_pad_size)
        SET_GRID_API(_set_offset)

        // Exposed APIs.
        GET_GRID_API(get_rank_domain_size)
        GET_GRID_API(get_first_rank_domain_index)
        GET_GRID_API(get_last_rank_domain_index)
        GET_GRID_API(get_halo_size)
        GET_GRID_API(get_extra_pad_size)
        GET_GRID_API(get_pad_size)
        GET_GRID_API(get_alloc_size)
        GET_GRID_API(get_first_rank_alloc_index)
        GET_GRID_API(get_last_rank_alloc_index)
        GET_GRID_API(get_first_misc_index)
        GET_GRID_API(get_last_misc_index)

        SET_GRID_API(set_halo_size)
        SET_GRID_API(set_min_pad_size)
        SET_GRID_API(set_extra_pad_size)
        SET_GRID_API(set_alloc_size)
        SET_GRID_API(set_first_misc_index)

#undef GET_GRID_API
#undef SET_GRID_API
        
        virtual void set_all_elements_same(double val) =0;
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
    // If '_wrap_1st_idx', then index to 1st dim will wrap around.
    template <typename LayoutFn, bool _wrap_1st_idx>
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
                   std::ostream** ostr) :
            YkGridBase(&_data, dimNames.size(), dims),
            _data(name, dimNames, ostr) {
            _has_step_dim = _wrap_1st_idx;
            resize();
        }

        // Get num dims from compile-time const.
        virtual int get_num_dims() const final {
            return _data.get_num_dims();
        }

        // Print some info to 'os'.
        virtual void print_info(std::ostream& os) const {
            _data.print_info(os, "FP");
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
                                         bool checkBounds=true) const final {

#ifdef TRACE_MEM
            _data.get_ostr() << get_name() << "." << "YkElemGrid::getElemPtr(" <<
                idxs.makeValStr(get_num_dims()) << ")";
#endif

            const auto n = _data.get_num_dims();
            Indices adj_idxs(n);
#pragma unroll
            for (int i = 0; i < n; i++) {

                // Special handling for 1st index.
                if (_wrap_1st_idx && i == 0)
                    adj_idxs[0] = wrap_step(idxs[0]);

                // Adjust for offset and padding.
                // This gives a 0-based local element index.
                else
                    adj_idxs[i] = idxs[i] - _offsets[i] + _pads[i];
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
                                   bool checkBounds=true) final {

            const real_t* p =
                const_cast<const YkElemGrid*>(this)->getElemPtr(idxs, checkBounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        // Indices are relative to overall problem domain.
        virtual real_t readElem(const Indices& idxs,
                                int line) const final {
            const real_t* ep = YkElemGrid::getElemPtr(idxs);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem("readElem", idxs, e, line);
#endif
            return e;
        }

    };                          // YkElemGrid.
    
    // YASK grid of real vectors.
    // Used for grids that contain all the folded dims.
    // If '_wrap_1st_idx', then index to 1st dim will wrap around.
    template <typename LayoutFn, bool _wrap_1st_idx>
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
                  std::ostream** ostr) :
            YkGridBase(&_data, dimNames.size(), dims),
            _data(name, dimNames, ostr),
            _vec_fold_posns(idx_t(0), int(dimNames.size())) {
            _has_step_dim = _wrap_1st_idx;

            // Init vec sizes.
            // For each dim in the grid, use the number of vector
            // fold points or 1 if not set.
            for (size_t i = 0; i < dimNames.size(); i++) {
                auto& dname = dimNames.at(i);
                auto* p = dims->_vec_fold_pts.lookup(dname);
                idx_t dval = p ? *p : 1;
                _vec_lens[i] = dval;
                _vec_allocs[i] = dval;
            }

            // Init grid positions of fold dims.
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

        // Print some info to 'os'.
        virtual void print_info(std::ostream& os) const {
            _data.print_info(os, "SIMD FP");
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
                                         bool checkBounds=true) const final {

#ifdef TRACE_MEM
            _data.get_ostr() << get_name() << "." << "YkVecGrid::getElemPtr(" <<
                idxs.makeValStr(get_num_dims()) << ")";
#endif

            const auto n = _data.get_num_dims();
            Indices vec_idxs(n), elem_ofs(n);
#pragma unroll
            for (int i = 0; i < n; i++) {

                // Special handling for step index.
                if (_wrap_1st_idx && i == Indices::step_posn) {
                    vec_idxs[i] = wrap_step(idxs[i]);
                    elem_ofs[i] = 0;
                }

                else {

                    // Adjust for offset and padding.
                    // This gives a 0-based local element index.
                    idx_t adj_idx = idxs[i] - _offsets[i] + _pads[i];
                    
                    // Get vector index and offset.
                    vec_idxs[i] = adj_idx / _vec_lens[i];
                    elem_ofs[i] = adj_idx % _vec_lens[i];
                }
            }

            // Get only the vectorized fold offsets.
            Indices fold_ofs(n);
#pragma unroll
            for (int i = 0; i < NUM_VEC_FOLD_DIMS; i++) {
                int j = _vec_fold_posns[i];
                fold_ofs[i] = elem_ofs[j];
            }
            
            // Get 1D element index into vector.
            auto i = _dims->getElemIndexInVec(fold_ofs);

#ifdef DEBUG_LAYOUT
            // Compare to more explicit offset extraction.
            IdxTuple eofs = get_allocs(); // get dims for this grid.
            elem_ofs.setTupleVals(eofs);  // set vals from elem_ofs.
            auto i2 = getElemIndexInVec(eofs);
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
                                   bool checkBounds=true) final {

            const real_t* p =
                const_cast<const YkVecGrid*>(this)->getElemPtr(idxs, checkBounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        // Indices are relative to overall problem domain.
        virtual real_t readElem(const Indices& idxs,
                                int line) const final {
            const real_t* ep = YkVecGrid::getElemPtr(idxs);
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
        inline const real_vec_t* getVecPtrNorm(const Indices& idxs,
                                               bool checkBounds=true) const {

#ifdef TRACE_MEM
            _data.get_ostr() << get_name() << "." << "YkVecGrid::getVecPtrNorm(" <<
                idxs.makeValStr(get_num_dims()) << ")";
#endif

            const auto n = _data.get_num_dims();
            Indices adj_idxs(n);
#pragma unroll
            for (int i = 0; i < n; i++) {

                // Special handling for 1st index.
                if (_wrap_1st_idx && i == Indices::step_posn)
                    adj_idxs[i] = wrap_step(idxs[i]);

                // Adjust for padding.
                // This gives a 0-based local *vector* index.
                else
                    adj_idxs[i] = idxs[i] + _vec_pads[i];
            }

#ifdef TRACE_MEM
            if (checkBounds)
                _data.get_ostr() << " => " << _data.get_index(adj_idxs);
            _data.get_ostr() << std::endl << std::flush;
#endif

            // Get ptr via layout in _data.
            return _data.getPtr(adj_idxs, checkBounds);
        }

        // Non-const version.
        inline real_vec_t* getVecPtrNorm(const Indices& idxs,
                                         bool checkBounds=true) {

            const real_vec_t* p =
                const_cast<const YkVecGrid*>(this)->getVecPtrNorm(idxs, checkBounds);
            return const_cast<real_vec_t*>(p);
        }

        // Read one vector.
        // Indices must be normalized and rank-relative.
        inline real_vec_t readVecNorm(const Indices& idxs,
                                  int line) const {
            const real_vec_t* vp = getVecPtrNorm(idxs);
            real_vec_t v = *vp;
#ifdef TRACE_MEM
            printVecNorm("readVecNorm", idxs, v, line);
#endif
            return v;
        }

        // Write one vector.
        // Indices must be normalized and rank-relative.
        inline void writeVecNorm(real_vec_t val,
                                 const Indices& idxs,
                                 int line) {
            real_vec_t* vp = getVecPtrNorm(idxs);
            *vp = val;
#ifdef TRACE_MEM
            printVecNorm("writeVecNorm", idxs, val, line);
#endif
        }

        // Prefetch one vector.
        // Indices must be normalized and rank-relative.
        template <int level>
        ALWAYS_INLINE
        void prefetchVecNorm(const Indices& idxs,
                             int line) const {
#ifdef TRACE_MEM
            std::cout << "prefetchVecNorm<" << level << ">(" <<
                makeIndexString(idxs.multElements(_vec_lens)) <<
                ")" << std::endl;
#endif
            auto p = getVecPtrNorm(idxs, false);
            prefetch<level>(p);
#ifdef MODEL_CACHE
            cache_model.prefetch(p, level, line);
#endif
        }

    };                          // YkVecGrid.

}                               // namespace.
