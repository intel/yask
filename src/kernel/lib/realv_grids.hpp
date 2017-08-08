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

#ifndef REAL_VEC_GRIDS
#define REAL_VEC_GRIDS

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
        // actual data.
        GenericGridBase* _ggb;

        // Positions of step and domain dims.
        // Indexed by name.
        // Either may be empty.
        // Remaining dims are "misc" dims.
        IdxTuple _step_dim_posn;
        IdxTuple _domain_dim_posns;
        
        // Settings for domain dims | non-domain dims.
        Indices _domains;       // values copied from the solution | alloc size.
        Indices _pads;          // extra space around domains | zero.
        Indices _halos;         // space within pads for halo exchange | zero.
        Indices _offsets;       // offsets of this rank's domain within overall domain | zero.

        // Data has been received from neighbors' halos, if applicable.
        bool _is_updated = false;

        // Convenience function to format indices like
        // "x=5, y=3".
        virtual std::string makeIndexString(const Indices& idxs,
                                            std::string separator=", ",
                                            std::string infix="=",
                                            std::string prefix="",
                                            std::string suffix="") const;

        // Check whether dim is of allowed type.
        virtual void checkDimType(const std::string& dim,
                                  const std::string& fn_name,
                                  bool domain_ok, bool non_domain_ok) const;
        
        // Share data from source grid.
        template<typename GT>
        bool _share_data(YkGridBase* src) {
            auto* tp = dynamic_cast<GT*>(_ggb);
            if (!tp) return false;
            auto* sp = dynamic_cast<GT*>(src->_ggb);
            if (!sp) return false;
            *tp = *sp; // shallow copy.
            return true;
        }

        // Share data from source grid.
        virtual bool share_data(YkGridBase* src) =0;

    public:
        YkGridBase(GenericGridBase* ggb) :
            _ggb(ggb) {
            assert(ggb);
        }
        virtual ~YkGridBase() { }

        // Halo-exchange flag accessors.
        bool is_updated() { return _is_updated; }
        void set_updated(bool is_updated) { _is_updated = is_updated; }
        
        // Lookup position by dim name.
        // Return -1 or die if not found, depending on flag.
        virtual int get_dim_posn(const std::string& dim,
                                 bool die_on_failure = false,
                                 const std::string& die_msg = "") const;
        
        // Resize only if not allocated.
        virtual void resize();

        // Print some info to 'os'.
        virtual void print_info(std::ostream& os) const =0;
  
        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const YkGridBase* ref,
                              real_t epsilon = EPSILON,
                              int maxPrint = 20,
                              std::ostream& os = std::cerr) const;

        // Make sure indices are in range.
        virtual bool checkIndices(const GridIndices& indices,
                                  const std::string& fn,
                                  bool strict_indices,
                                  GridIndices* fixed_indices = NULL) const;

        // Set elements to a sequence of values using seed.
        // Cf. set_all_elements_same().
        virtual void set_all_elements_in_seq(double seed);
        
        // Get a pointer to one element.
        // Indices are relative to overall problem domain.
        virtual const real_t* getElemPtr(const Indices& idxs,
                                         bool checkBounds=true) const =0;
        virtual real_t* getElemPtr(const Indices& idxs,
                                   bool checkBounds=true) =0;

        // Read one element.
        // Indices are relative to overall problem domain.
        inline real_t readElem(const Indices& idxs,
                                int line) const {
            const real_t* ep = getElemPtr(idxs);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem(std::cout, "readElem", idxs, e, line);
#endif
            return e;
        }

        // Write one element.
        // Indices are relative to overall problem domain.
        inline void writeElem(real_t val,
                              const Indices& idxs,
                              int line) {
            real_t* ep = getElemPtr(idxs);
            *ep = val;
#ifdef TRACE_MEM
            printElem(std::cout, "writeElem", idxs, val, line);
#endif
        }

        // Print one element.
        virtual void printElem(std::ostream& os,
                               const std::string& msg,
                               const Indices& idxs,
                               real_t e,
                               int line,
                               bool newline = true) const;

        // APIs not defined above.
        // See yask_kernel_api.hpp.
        virtual const std::string& get_name() const {
            return _ggb->get_name();
        }
        virtual int get_num_dims() const {
            return _ggb->get_num_dims();
        }
        virtual bool is_dim_used(const std::string& dim) const {
            return _ggb->is_dim_used(dim);
        }
        virtual bool is_step_dim(const std::string& dim) const {
            return _step_dim_posn.lookup(dim) != NULL;
        }
        virtual bool is_domain_dim(const std::string& dim) const {
            return _domain_dim_posns.lookup(dim) != NULL;
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

        virtual idx_t get_rank_domain_size(const std::string& dim) const; // not exposed.
        virtual idx_t get_first_rank_domain_index(const std::string& dim) const; // not exposed.
        virtual idx_t get_last_rank_domain_index(const std::string& dim) const; // not exposed.
        virtual idx_t get_offset(const std::string& dim) const; // not exposed.
        virtual idx_t get_halo_size(const std::string& dim) const;
        virtual idx_t get_extra_pad_size(const std::string& dim) const;
        virtual idx_t get_pad_size(const std::string& dim) const;
        virtual idx_t get_alloc_size(const std::string& dim) const;
        virtual idx_t get_first_rank_alloc_index(const std::string& dim) const;
        virtual idx_t get_last_rank_alloc_index(const std::string& dim) const;
        virtual idx_t get_first_index(const std::string& dim) const;
        virtual idx_t get_last_index(const std::string& dim) const;

        virtual void set_domain_size(const std::string& dim, idx_t size); // not exposed.
        virtual void set_pad_size(const std::string& dim, idx_t size); // not exposed.
        virtual void set_offset(const std::string& dim, idx_t size); // not exposed.
        virtual void set_halo_size(const std::string& dim, idx_t size);
        virtual void set_min_pad_size(const std::string& dim, idx_t size);
        virtual void set_extra_pad_size(const std::string& dim, idx_t size);
        virtual void set_alloc_size(const std::string& dim, idx_t size);
        virtual void set_first_index(const std::string& dim, idx_t size);

        virtual void set_all_elements_same(double val);
        virtual double get_element(idx_t dim1_index, idx_t dim2_index,
                                   idx_t dim3_index, idx_t dim4_index,
                                   idx_t dim5_index, idx_t dim6_index) const;
        virtual double get_element(const GridIndices& indices) const;
        virtual idx_t get_elements_in_slice(void* buffer_ptr,
                                            const GridIndices& first_indices,
                                            const GridIndices& last_indices) const;
        virtual idx_t set_element(double val,
                                  idx_t dim1_index, idx_t dim2_index,
                                  idx_t dim3_index, idx_t dim4_index,
                                  idx_t dim5_index, idx_t dim6_index);
        virtual idx_t set_element(double val,
                                  const GridIndices& indices,
                                  bool strict_indices);
        virtual idx_t set_elements_in_slice_same(double val,
                                                 const GridIndices& first_indices,
                                                 const GridIndices& last_indices,
                                                 bool strict_indices);
        virtual idx_t set_elements_in_slice(const void* buffer_ptr,
                                            const GridIndices& first_indices,
                                            const GridIndices& last_indices);
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
            return _ggb->get_num_elems();
        }
        virtual bool is_storage_layout_identical(const yk_grid_ptr other) const;
        virtual void* get_raw_storage_buffer() {
            return _ggb->get_storage();
        }
        virtual void set_storage(std::shared_ptr<char> base, size_t offset) {
            _ggb->set_storage(base, offset);
        }
    };

    // Some derivations.
    typedef std::shared_ptr<YkGridBase> YkGridPtr;
    typedef std::vector<YkGridPtr> GridPtrs;
    typedef std::set<YkGridPtr> GridPtrSet;
    
    // YASK grid of real elements.
    // Used for grids that do not contain all the folded dims.
    template <typename LayoutFn>
    class YkElemGrid : public YkGridBase {

    protected:
        typedef GenericGrid<real_t, LayoutFn> _grid_type;
        _grid_type _data;
        
        // Share data from source grid.
        virtual bool share_data(YkGridBase* src) {
            return _share_data<_grid_type>(src);
        }

    public:
        YkElemGrid(std::string name,
                   const GridDimNames& dimNames) :
            YkGridBase(&_data), _data(name, dimNames) { }

        // Print some info to 'os'.
        virtual void print_info(std::ostream& os) const;
  
        // Get a pointer to given element.
        virtual const real_t* getElemPtr(const Indices& idxs,
                                         bool checkBounds=true) const final {

#ifdef TRACE_MEM
            std::cout << get_name() << "." << "YkElemGrid::getElemPtr(" <<
                idxs.makeValStr(get_num_dims()) << ")";
#endif
        
            // Adjust for offset and padding.
            // FIXME: handle step idx.
            Indices idxs2;
            #pragma unroll
            for (int i = 0; i < _data.get_num_dims(); i++)
                idxs2[i] = idxs[i] - _offsets[i] + _pads[i];

#ifdef TRACE_MEM
            if (checkBounds)
                std::cout << " => " << _data.get_index(idxs2);
            std::cout << std::endl << flush;
#endif

            // Get pointer via layout in _data.
            return &_data(idxs2, checkBounds);
        }

        // Non-const version.
        virtual real_t* getElemPtr(const Indices& idxs,
                                   bool checkBounds=true) final {

            const real_t* p =
                const_cast<const YkElemGrid*>(this)->getElemPtr(idxs, checkBounds);
            return const_cast<real_t*>(p);
        }
    
    };
    
    // YASK grid of real vectors.
    // Used for grids that contain all the folded dims.
    template <typename LayoutFn>
    class YkVecGrid : public YkGridBase {

    protected:
        typedef GenericGrid<real_vec_t, LayoutFn> _grid_type;
        _grid_type _data;
        
        // Share data from source grid.
        virtual bool share_data(YkGridBase* src) {
            return _share_data<_grid_type>(src);
        }

        // sizes in units of real_vec_t elements.
        Indices _vec_allocs;
        Indices _vec_pads;

    public:
        
        // Print some info to 'os'.
        virtual void print_info(std::ostream& os) const;

#warning FIXME
#if 0
        
        
        // Normalize element indices to rank-relative vector indices and
        // element offsets.
        ALWAYS_INLINE
        void normalize(idx_t elem_index,
                       idx_t& vec_index,
                       idx_t& elem_ofs,
                       idx_t vec_len,
                       idx_t vec_pad,
                       idx_t rank_ofs) const {

            // Remove offset to make rank-relative.
            // Add padding before division to ensure negative
            // indices work under truncated division.
            idx_t padded_index = elem_index - rank_ofs + (vec_pad * vec_len);

            // Normalize and remove added padding.
            vec_index = (padded_index / vec_len) - vec_pad;

            // Divide values with padding in numerator to avoid negative indices to get offsets.
            elem_ofs = padded_index % vec_len;
        }

        // Adjust logical time index to 0-based index
        // using temporal allocation size.
        ALWAYS_INLINE
        idx_t get_index_t(idx_t t) const {

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
            t += 0x1000;
            assert(t >= 0);
            return t % _tdim;
        }

    public:
        RealVecGrid(RealVecGrid* gp) :
            _gp(gp) { }
        virtual ~RealVecGrid() { }

        // Initialize memory to a given value.
        virtual void set_same(real_t val) {
            real_vec_t rn;
            rn = val;               // broadcast thru vector.
            _gp->set_same(rn);
        }

        // Initialize memory to incrementing values based on val.
        virtual void set_diff(real_t val);

        // Get number of real_vecs, including halos & padding.
        virtual idx_t get_num_real_vecs() const {
            return _gp->get_num_elems();
        }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _gp->get_num_bytes() / sizeof(real_t);
        }

        // Get size in bytes.
        virtual size_t get_num_bytes() const {
            return _gp->get_num_bytes();
        }

        // Normalize element indices to vector indices and element offsets.
        ALWAYS_INLINE
        void normalize_x(idx_t x, idx_t& vec_index, idx_t& elem_ofs) const {
            normalize(x, vec_index, elem_ofs, VLEN_X, _pxv, _ox);
        }
        ALWAYS_INLINE
        void normalize_y(idx_t y, idx_t& vec_index, idx_t& elem_ofs) const {
            normalize(y, vec_index, elem_ofs, VLEN_Y, _pyv, _oy);
        }
        ALWAYS_INLINE
        void normalize_z(idx_t z, idx_t& vec_index, idx_t& elem_ofs) const {
            normalize(z, vec_index, elem_ofs, VLEN_Z, _pzv, _oz);
        }

        // Read one element.
        virtual real_t readElem_TXYZ(idx_t t, idx_t x, idx_t y, idx_t z,
                                     int line) const =0;

        // Write one element.
        virtual void writeElem_TXYZ(real_t val,
                                     idx_t t, idx_t x, idx_t y, idx_t z,                               
                                     int line) =0;

        // Read one vector at *vector* offset.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TXYZ(idx_t t, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const =0;
        
        // Write one vector at *vector* offset.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TXYZ(const real_vec_t& v,
                                        idx_t t, idx_t xv, idx_t yv, idx_t zv,
                                        int line) =0;

        // Print one element.
        virtual void printElem_TXYZ(std::ostream& os, const std::string& m,
                                     idx_t t, idx_t x, idx_t y, idx_t z,
                                     real_t e, int line, bool newline = true) const;

        // Print one vector at *vector* offset.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void printVecNorm_TXYZ(std::ostream& os, const std::string& m,
                                        idx_t t, idx_t xv, idx_t yv, idx_t zv,
                                        const real_vec_t& v,
                                        int line) const;

        // Direct access to data.
        real_vec_t* get_storage() {
            return _gp->get_storage();
        }
        const real_vec_t* get_storage() const {
            return _gp->get_storage();
        }
        void set_storage(std::shared_ptr<char> base, size_t offset) {
            _gp->set_storage(base, offset);
        }
        RealVecGrid* getGenericGrid() {
            return _gp;
        }
        const RealVecGrid* getGenericGrid() const {
            return _gp;
        }
#endif
    };

#if 0
    // A 3D (x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn> class RealVecGrid_XYZ :
        public RealVecGridBase {

    protected:

        typedef GenericGrid3d<real_vec_t, LayoutFn> _grid_type;
        _grid_type _data;

        virtual void resize_g() {
            _data.set_d1(_axv);
            _data.set_d2(_ayv);
            _data.set_d3(_azv);
        }
        
    public:

        // Ctor.
        RealVecGrid_XYZ(const std::string& name) :
            RealVecGridBase(&_data),
            _data(name, "x", "y", "z") { }

        // Determine what dims are defined.
        virtual bool got_x() const { return true; }
        virtual bool got_y() const { return true; }
        virtual bool got_z() const { return true; }

        // Get pointer to the real_vec_t at vector offset xv, yv, zv.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        const real_vec_t* getVecPtrNorm(idx_t xv, idx_t yv, idx_t zv,
                                        bool checkBounds=true) const {

#ifdef TRACE_MEM
            std::cout << get_name() << "." << "RealVecGrid_XYZ::getVecPtrNorm(" <<
                xv << "," << yv << "," << zv << ")";
#endif
        
            // adjust for padding.
            xv += _pxv;
            yv += _pyv;
            zv += _pzv;

#ifdef TRACE_MEM
            if (checkBounds)
                std::cout << " => " << _data.get_index(xv, yv, zv);
            std::cout << std::endl << flush;
#endif

            // Get pointer via layout in _data.
            return &_data(xv, yv, zv, checkBounds);
        }

        // Non-const version.
        ALWAYS_INLINE
        real_vec_t* getVecPtrNorm(idx_t xv, idx_t yv, idx_t zv,
                                  bool checkBounds=true) {

            const real_vec_t* vp =
                const_cast<const RealVecGrid_XYZ*>(this)->getVecPtrNorm(xv, yv, zv,
                                                                        checkBounds);
            return const_cast<real_vec_t*>(vp);
        }
    
        // Get a pointer to one real_t.
        ALWAYS_INLINE
        const real_t* getElemPtr(idx_t x, idx_t y, idx_t z,
                                 bool checkBounds=true) const {
            idx_t xv, xe, yv, ye, zv, ze;
            normalize_x(x, xv, xe);
            normalize_y(y, yv, ye);
            normalize_z(z, zv, ze);

            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(xv, yv, zv, checkBounds);

            // Extract point from vector.
            return &(*vp)(xe, ye, ze);
        }

        // non-const version.
        ALWAYS_INLINE
        real_t* getElemPtr(idx_t x, idx_t y, idx_t z,
                           bool checkBounds=true) {
            const real_t* p = const_cast<const RealVecGrid_XYZ*>(this)->getElemPtr(x, y, z,
                                                                               checkBounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        ALWAYS_INLINE
        real_t readElem(idx_t x, idx_t y, idx_t z,
                        int line) const {
            const real_t* ep = getElemPtr(x, y, z);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem(std::cout, "readElem", x, y, z, e, line);
#endif
            return e;
        }

        // Write one element.
        ALWAYS_INLINE
        void writeElem(real_t val, idx_t x, idx_t y, idx_t z,
                       int line) {
            real_t* ep = getElemPtr(x, y, z);
            *ep = val;
#ifdef TRACE_MEM
            printElem(std::cout, "writeElem", x, y, z, val, line);
#endif
        }

        // Read one vector at vector offset xv, yv, zv.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        real_vec_t readVecNorm(idx_t xv, idx_t yv, idx_t zv,
                               int line) const {
#ifdef TRACE_MEM
            std::cout << "readVecNorm(" << xv << "," << yv << "," << zv << ")..." << std::endl;
#endif        
            const real_vec_t* p = getVecPtrNorm(xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            real_vec_t v;
            v.loadFrom(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "readVec", xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.read(p, line);
#endif
            return v;
        }

        // Write one vector at vector offset xv, yv, zv.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        void writeVecNorm(const real_vec_t& v, idx_t xv, idx_t yv, idx_t zv,
                          int line) {
            real_vec_t* p = getVecPtrNorm(xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            v.storeTo(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "writeVec", xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.write(p, line);
#endif
        }

        // Prefetch one vector at vector offset xv, yv, zv.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        template <int level>
        ALWAYS_INLINE
        void prefetchVecNorm(idx_t xv, idx_t yv, idx_t zv,
                             int line) const {
#ifdef TRACE_MEM
            std::cout << "prefetchVecNorm<" << level << ">(" <<
                xv << "," << yv << "," << zv << ")..." << std::endl;
#endif        
            const char* p = (const char*)getVecPtrNorm(xv, yv, zv, false);
            __assume_aligned(p, CACHELINE_BYTES);
            _mm_prefetch (p, level);
#ifdef MODEL_CACHE
            cache_model.prefetch(p, level, line);
#endif
        }

        // Read one element.
        virtual real_t readElem_TXYZ(idx_t t, idx_t x, idx_t y, idx_t z,
                                      int line) const {
            assert(t == 0);
            return readElem(x, y, z, line);
        }

        // Write one element.
        virtual void writeElem_TXYZ(real_t val,
                                     idx_t t, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            assert(t == 0);
            writeElem(val, x, y, z, line);
        }
        
        // Read one vector at *vector* offset.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TXYZ(idx_t t, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const {
            assert(t == 0);
            return readVecNorm(xv, yv, zv, line);
        }
        
        // Write one vector at *vector* offset.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TXYZ(const real_vec_t& val,
                                        idx_t t, idx_t xv, idx_t yv, idx_t zv,
                                        int line) {
            assert(t == 0);
            writeVecNorm(val, xv, yv, zv, line);
        }

        // Print one vector.
        void printVecNorm(std::ostream& os, const std::string& m,
                      idx_t xv, idx_t yv, idx_t zv, const real_vec_t& v,
                      int line) const {
            printVecNorm_TXYZ(os, m, 0, xv, yv, zv, v, line);
        }

        // Print one element.
        void printElem(std::ostream& os, const std::string& m,
                       idx_t x, idx_t y, idx_t z, real_t e,
                       int line) const {
            printElem_TXYZ(os, m, 0, x, y, z, e, line);
        }
    };


    // A 4D (t, x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each spatial dimension.
    template <typename LayoutFn> class RealVecGrid_TXYZ :
        public RealVecGridBase {
    
    protected:

        typedef GenericGrid4d<real_vec_t, LayoutFn> _grid_type;
        _grid_type _data;

        // Share data from source grid.
        virtual bool share_data(RealVecGridBase* src) {
            return _share_data<_grid_type>(src);
        }
        
        virtual void resize_g() {
            _data.set_d1(_tdim);
            _data.set_d2(_axv);
            _data.set_d3(_ayv);
            _data.set_d4(_azv);
        }
        
    public:

        // Ctor.
        RealVecGrid_TXYZ(const std::string& name) :
            RealVecGridBase(&_data),
            _data(name, "t", "x", "y", "z") { }

        // Determine what dims are defined.
        virtual bool got_t() const { return true; }
        virtual bool got_x() const { return true; }
        virtual bool got_y() const { return true; }
        virtual bool got_z() const { return true; }
        
        // Get pointer to the real_vec_t at vector offset t, xv, yv, zv.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        const real_vec_t* getVecPtrNorm(idx_t t, idx_t xv, idx_t yv, idx_t zv,
                                        bool checkBounds=true) const {

#ifdef TRACE_MEM
            std::cout << get_name() << "." << "RealVecGrid_TXYZ::getVecPtrNorm(" <<
                t << "," << xv << "," << yv << "," << zv << ")";
#endif

            // Wrap time index.
            t = get_index_t(t);

            // adjust for padding.
            xv += _pxv;
            yv += _pyv;
            zv += _pzv;

#ifdef TRACE_MEM
            if (checkBounds)
                std::cout << " => " << _data.get_index(t, xv, yv, zv);
            std::cout << std::endl << flush;
#endif
            return &_data(t, xv, yv, zv, checkBounds);
        }

        // Non-const version.
        ALWAYS_INLINE
        real_vec_t* getVecPtrNorm(idx_t t, idx_t xv, idx_t yv, idx_t zv,
                                  bool checkBounds=true) {

            const real_vec_t* vp =
                const_cast<const RealVecGrid_TXYZ*>(this)->getVecPtrNorm(t, xv, yv, zv,
                                                                          checkBounds);
            return const_cast<real_vec_t*>(vp);
        }
    
        // Get a pointer to one real_t.
        ALWAYS_INLINE
        const real_t* getElemPtr(idx_t t, idx_t x, idx_t y, idx_t z,
                                 bool checkBounds=true) const {
            idx_t xv, xe, yv, ye, zv, ze;
            normalize_x(x, xv, xe);
            normalize_y(y, yv, ye);
            normalize_z(z, zv, ze);

            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(t, xv, yv, zv, checkBounds);

            // Extract point from vector.
            return &(*vp)(xe, ye, ze);
        }

        // non-const version.
        ALWAYS_INLINE
        real_t* getElemPtr(idx_t t, idx_t x, idx_t y, idx_t z,
                           bool checkBounds=true) {
            const real_t* p =
                const_cast<const RealVecGrid_TXYZ*>(this)->getElemPtr(t, x, y, z,
                                                                       checkBounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        ALWAYS_INLINE
        real_t readElem(idx_t t, idx_t x, idx_t y, idx_t z,
                        int line) const {
            const real_t* ep = getElemPtr(t, x, y, z);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem(std::cout, "readElem", t, x, y, z, e, line);
#endif
            return e;
        }

        // Write one element.
        ALWAYS_INLINE
        void writeElem(real_t val,
                       idx_t t, idx_t x, idx_t y, idx_t z,
                       int line) {
            real_t* ep = getElemPtr(t, x, y, z);
            *ep = val;
#ifdef TRACE_MEM
            printElem(std::cout, "writeElem", t, x, y, z, val, line);
#endif
        }

        // Read one vector at vector offset t, xv, yv, zv.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        real_vec_t readVecNorm(idx_t t, idx_t xv, idx_t yv, idx_t zv,
                               int line) const {
#ifdef TRACE_MEM
            std::cout << "readVecNorm(" << t << "," << xv <<
                "," << yv << "," << zv << ")..." << std::endl;
#endif        
            const real_vec_t* p = getVecPtrNorm(t, xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            real_vec_t v;
            v.loadFrom(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "readVec", t, xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.read(p, line);
#endif
            return v;
        }

        // Write one vector at vector offset t, xv, yv, zv.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE void
        writeVecNorm(const real_vec_t& v,
                     idx_t t, idx_t xv, idx_t yv, idx_t zv,
                     int line) {
            real_vec_t* p = getVecPtrNorm(t, xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            v.storeTo(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "writeVec", t, xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.write(p, line);
#endif
        }

        // Prefetch one vector at vector offset t, xv, yv, zv.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        template <int level>
        ALWAYS_INLINE
        void prefetchVecNorm(idx_t t, idx_t xv, idx_t yv, idx_t zv,
                             int line) const {
#ifdef TRACE_MEM
            std::cout << "prefetchVecNorm<" << level << ">(" << t << "," <<
                xv << "," << yv << "," << zv << ")..." << std::endl;
#endif
            const char* p = (const char*)getVecPtrNorm(t, xv, yv, zv, false);
            __assume_aligned(p, CACHELINE_BYTES);
            _mm_prefetch (p, level);
#ifdef MODEL_CACHE
            cache_model.prefetch(p, level, line);
#endif
        }

        // Read one element.
        virtual real_t readElem_TXYZ(idx_t t, idx_t x, idx_t y, idx_t z,
                                      int line) const {
            return readElem(t, x, y, z, line);
        }
        
        // Write one element.
        virtual void writeElem_TXYZ(real_t val,
                                     idx_t t, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            writeElem(val, t, x, y, z, line);
        }

        // Read one vector at *vector* offset.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TXYZ(idx_t t, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const {
            return readVecNorm(t, xv, yv, zv, line);
        }
        
        // Write one vector at *vector* offset.
        // Indices must be relative to rank, i.e., offset is already subtracted.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TXYZ(const real_vec_t& val,
                                        idx_t t, idx_t xv, idx_t yv, idx_t zv,
                                        int line) {
            writeVecNorm(val, t, xv, yv, zv, line);
        }

        // Print one vector.
        void printVecNorm(std::ostream& os, const std::string& m,
                          idx_t t, idx_t xv, idx_t yv, idx_t zv,
                          const real_vec_t& v,
                          int line) const {
            printVecNorm_TXYZ(os, m, t, xv, yv, zv, v, line);
        }

        // Print one element.
        void printElem(std::ostream& os, const std::string& m,
                       idx_t t, idx_t x, idx_t y, idx_t z,
                       real_t e,
                       int line) const {
            printElem_TXYZ(os, m, t, x, y, z, e, line);
        }
    };
#endif

#if GRID_TEST
    // Dummy vars to test template compiles.
    GridDimNames dummyDims;
    GenericGrid<idx_t, Layout_123> dummy1("d1", dummyDims);
    GenericGrid<real_t, Layout_123> dummy2("d2", dummyDims);
    GenericGrid<real_vec_t, Layout_123> dummy3("d3", dummyDims);
    YkElemGrid<Layout_123> dummy4("d4", dummyDims);
#endif
    
}
#endif
