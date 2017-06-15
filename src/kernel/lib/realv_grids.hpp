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

    typedef GenericGridBase<real_t> RealGrid;
    typedef GenericGridBase<real_vec_t> RealVecGrid;

    // Pure-virtual base class for real_vec_t grids.  Supports 3
    // spatial dims and one temporal dim.  Indices used to access
    // points are global and logical.  Example: if there are two ranks in
    // the x-dim, each of which has domain-size of 100 and a pad-size of 10,
    // the 1st rank's x indices will be -10..110, and the 2nd rank's x
    // indices will be 90..210.  TODO: allow different pos and neg-side
    // halos and/or padding.
    class RealVecGridBase :
        public virtual yk_grid {

    protected:
        RealVecGrid* _gp;

        // Allocation in time domain.
        idx_t _tdim = 1;

        // real_t sizes for 3 spatial dims.
        idx_t _dx=1, _dy=1, _dz=1; // domain sizes.
        idx_t _hx=0, _hy=0, _hz=0; // halo sizes.
        idx_t _px=0, _py=0, _pz=0; // pad sizes, which include halos.
        idx_t _ox=0, _oy=0, _oz=0; // offsets into global problem domain.

        // real_vec_t sizes for 3 spatial dims.
        idx_t _axv=1, _ayv=1, _azv=1; // alloc size.
        idx_t _pxv=0, _pyv=0, _pzv=0; // padding before domain.

        // Dynamic data.
        bool _is_updated = false; // data has been received from neighbors' halos.

        // Share data from source grid.
        virtual bool share_data(RealVecGridBase* src) =0;

        // Share data from source grid.
        template<typename GT>
        bool _share_data(RealVecGridBase* src) {
            auto* tp = dynamic_cast<GT*>(_gp);
            if (!tp) return false;
            auto* sp = dynamic_cast<GT*>(src->_gp);
            if (!sp) return false;
            *tp = *sp; // shallow copy.
            return true;
        }
        
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
            // t_idx => return value.
            // -2 => 0.
            // -1 => 1.
            //  0 => 0.
            //  1 => 1.
            //  2 => 0.

            // Avoid discontinuity caused by dividing negative numbers by
            // adding a large offset to the t index.  So, t can be negative,
            // but not so much that it would still be negative after adding
            // the offset.  This should not be a practical restriction.
            assert(t % CPTS_T == 0);
            t += 0x1000 * idx_t(CPTS_T);
            assert(t >= 0);
            idx_t t_idx = t / idx_t(CPTS_T);
            return t_idx % _tdim;
        }

        // Resize only if not allocated.
        virtual void resize();

        // Resize the underlying grid based on the current settings.
        virtual void resize_g() =0;

        // Make sure indices are in range.
        virtual void checkIndices(const GridIndices& indices,
                                  const std::string& fn) const;
        
    public:
        RealVecGridBase(RealVecGrid* gp) :
            _gp(gp) { }
        virtual ~RealVecGridBase() { }

        // Get name.
        const std::string& get_name() const { return _gp->get_name(); }

        // Determine what dims are actually used for derived type.
        virtual bool got_t() const { return false; }
        virtual bool got_x() const { return false; }
        virtual bool got_y() const { return false; }
        virtual bool got_z() const { return false; }
        virtual int get_num_dims() const {
            return (got_t() ? 1 : 0) +
                (got_x() ? 1 : 0) +
                (got_y() ? 1 : 0) +
                (got_z() ? 1 : 0);
        }

        // Get storage allocation in number of points.
        inline idx_t get_alloc_t() const { return _tdim; }
        inline idx_t get_alloc_x() const { return _axv * VLEN_X; }
        inline idx_t get_alloc_y() const { return _ayv * VLEN_Y; }
        inline idx_t get_alloc_z() const { return _azv * VLEN_Z; }
        
        // Get domain-size for this rank.
        inline idx_t get_dx() const { return _dx; }
        inline idx_t get_dy() const { return _dy; }
        inline idx_t get_dz() const { return _dz; }

        // Get halo-size.
        inline idx_t get_halo_x() const { return _hx; }
        inline idx_t get_halo_y() const { return _hy; }
        inline idx_t get_halo_z() const { return _hz; }

        // Get padding-size after round-up.
        // This includes the halo.
        inline idx_t get_pad_x() const { return _px; }
        inline idx_t get_pad_y() const { return _py; }
        inline idx_t get_pad_z() const { return _pz; }

        // Get extra-padding-size after round-up.
        // Since the extra pad is in addition to the halo, these
        // values may not be multiples of the vector lengths.
        inline idx_t get_extra_pad_x() const { return _px - _hx; }
        inline idx_t get_extra_pad_y() const { return _py - _hy; }
        inline idx_t get_extra_pad_z() const { return _pz - _hz; }

        // Get first logical index in domain on this rank.
        inline idx_t get_first_x() const { return _ox; }
        inline idx_t get_first_y() const { return _oy; }
        inline idx_t get_first_z() const { return _oz; }

        // Get last logical index in domain on this rank.
        inline idx_t get_last_x() const { return _ox + _dx - 1; }
        inline idx_t get_last_y() const { return _oy + _dy - 1; }
        inline idx_t get_last_z() const { return _oz + _dz - 1; }

        // Set temporal storage allocation.
        void set_alloc_t(idx_t tdim) { _tdim = tdim; resize(); }

        // Set domain-size for this rank.
        void set_dx(idx_t dx) { _dx = dx; resize(); }
        void set_dy(idx_t dy) { _dy = dy; resize(); }
        void set_dz(idx_t dz) { _dz = dz; resize(); }

        // Set halo sizes.
        // Automatically increase padding if less than halo.
        // Halo sizes are not rounded up.
        void set_halo_x(idx_t hx) { _hx = hx; set_pad_x(_px); }
        void set_halo_y(idx_t hy) { _hy = hy; set_pad_y(_py); }
        void set_halo_z(idx_t hz) { _hz = hz; set_pad_z(_pz); }

        // Set padding and round-up.
        // Padding will be increased to size of halo if needed.
        void set_pad_x(idx_t px) { _px = std::max(px, _hx); resize(); }
        void set_pad_y(idx_t py) { _py = std::max(py, _hy); resize(); }
        void set_pad_z(idx_t pz) { _pz = std::max(pz, _hz); resize(); }

        // Increase padding if below minimum.
        void set_min_pad_x(idx_t mpx) { if (_px < mpx) set_pad_x(mpx); }
        void set_min_pad_y(idx_t mpy) { if (_py < mpy) set_pad_y(mpy); }
        void set_min_pad_z(idx_t mpz) { if (_pz < mpz) set_pad_z(mpz); }
        
        // Set offset for current rank.
        void set_ofs_x(idx_t ox) { _ox = ox; }
        void set_ofs_y(idx_t oy) { _oy = oy; }
        void set_ofs_z(idx_t oz) { _oz = oz; }

        // Dynamic data accessors.
        bool is_updated() { return _is_updated; }
        void set_updated(bool is_updated) { _is_updated = is_updated; }
        
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

        // Print some info
        virtual void print_info(std::ostream& os);
  
        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const RealVecGridBase& ref,
                              real_t epsilon = EPSILON,
                              int maxPrint = 20,
                              std::ostream& os = std::cerr) const;

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


        // APIs not defined above.
        // See yask_kernel_api.hpp.
        virtual bool is_dim_used(const std::string& dim) const {
            return _gp->is_dim_used(dim);
        }
        virtual const std::string& get_dim_name(int n) const {
            assert(n >= 0);
            assert(n < get_num_dims());
            return _gp->get_dim_name(n);
        }
        virtual std::vector<std::string> get_dim_names() const {
            std::vector<std::string> dims;
            for (int i = 0; i < get_num_dims(); i++)
                dims.push_back(get_dim_name(i));
            return dims;
        }
        virtual idx_t get_rank_domain_size(const std::string& dim) const; // not exposed.
        virtual idx_t get_first_rank_domain_index(const std::string& dim) const; // not exposed.
        virtual idx_t get_last_rank_domain_index(const std::string& dim) const; // not exposed.
        virtual idx_t get_halo_size(const std::string& dim) const;
        virtual idx_t get_extra_pad_size(const std::string& dim) const;
        virtual idx_t get_pad_size(const std::string& dim) const;
        virtual idx_t get_alloc_size(const std::string& dim) const;
        virtual void set_halo_size(const std::string& dim, idx_t size);
        virtual void set_pad_size(const std::string& dim, idx_t size); // not exposed.
        virtual void set_min_pad_size(const std::string& dim, idx_t size);
        virtual void set_alloc_size(const std::string& dim, idx_t size);
        virtual void set_all_elements_same(double val) {
            set_same(real_t(val));
        }
        virtual void alloc_storage() {
            _gp->default_alloc();
        }
        virtual void release_storage() {
            _gp->release_storage();
        }            
        virtual void share_storage(yk_grid_ptr source);
        virtual bool is_storage_allocated() const {
            return get_storage() != 0;
        }
    };
    
    // A 3D (x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn> class RealVecGrid_XYZ :
        public RealVecGridBase {

    protected:

        typedef GenericGrid3d<real_vec_t, LayoutFn> _grid_type;
        _grid_type _data;

        // Share data from source grid.
        virtual bool share_data(RealVecGridBase* src) {
            return _share_data<_grid_type>(src);
        }
        
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
        virtual double get_element(idx_t dim1_index, idx_t dim2_index,
                                   idx_t dim3_index, idx_t dim4_index,
                                   idx_t dim5_index, idx_t dim6_index) const {
            GridIndices idx = {dim1_index, dim2_index, dim3_index};
            return get_element(idx);
        }
        virtual double get_element(const GridIndices& indices) const {
            checkIndices(indices, "get_element");
            return double(readElem(indices.at(0), indices.at(1),
                                   indices.at(2), __LINE__));
        }
        virtual idx_t get_elements_in_slice(void* buffer_ptr,
                                            const GridIndices& first_indices,
                                            const GridIndices& last_indices) const {
            checkIndices(first_indices, "get_elements_in_slice");
            checkIndices(last_indices, "get_elements_in_slice");
            idx_t n = 0;
            for (idx_t x = first_indices[0]; x <= last_indices[0]; x++)
                for (idx_t y = first_indices[1]; y <= last_indices[1]; y++)
                    for (idx_t z = first_indices[2]; z <= last_indices[2]; z++, n++) {
                        real_t val = readElem(x, y, z, __LINE__);
                        ((real_t*)buffer_ptr)[n] = val;
                    }
            return n;
        }

        // Write one element.
        virtual void writeElem_TXYZ(real_t val,
                                     idx_t t, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            assert(t == 0);
            writeElem(val, x, y, z, line);
        }
        virtual void set_element(double val,
                                 idx_t dim1_index, idx_t dim2_index,
                                 idx_t dim3_index, idx_t dim4_index,
                                 idx_t dim5_index, idx_t dim6_index) {
            GridIndices idx = {dim1_index, dim2_index, dim3_index};
            set_element(val, idx);
        }            
        virtual void set_element(double val, const GridIndices& indices) {
            checkIndices(indices, "set_element");
            writeElem(real_t(val),
                      indices.at(0), indices.at(1),
                      indices.at(2), __LINE__);
            set_updated(false);
        }
        virtual idx_t set_elements_in_slice_same(double val,
                                                 const GridIndices& first_indices,
                                                 const GridIndices& last_indices) {
            checkIndices(first_indices, "set_elements_in_slice_same");
            checkIndices(last_indices, "set_elements_in_slice_same");
            idx_t n = 0;
            for (idx_t x = first_indices[0]; x <= last_indices[0]; x++)
                for (idx_t y = first_indices[1]; y <= last_indices[1]; y++)
                    for (idx_t z = first_indices[2]; z <= last_indices[2]; z++, n++)
                        writeElem(real_t(val), x, y, z, __LINE__);
            set_updated(false);
            return n;
        }
        virtual idx_t set_elements_in_slice(const void* buffer_ptr,
                                            const std::vector<idx_t>& first_indices,
                                            const std::vector<idx_t>& last_indices) {
            checkIndices(first_indices, "set_elements_in_slice");
            checkIndices(last_indices, "set_elements_in_slice");
            idx_t n = 0;
            for (idx_t x = first_indices[0]; x <= last_indices[0]; x++)
                for (idx_t y = first_indices[1]; y <= last_indices[1]; y++)
                    for (idx_t z = first_indices[2]; z <= last_indices[2]; z++, n++) {
                        real_t val = ((real_t*)buffer_ptr)[n];
                        writeElem(val, x, y, z, __LINE__);
                    }
            set_updated(false);
            return n;
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
        virtual double get_element(idx_t dim1_index, idx_t dim2_index,
                                   idx_t dim3_index, idx_t dim4_index,
                                   idx_t dim5_index, idx_t dim6_index) const {
            GridIndices idx = {dim1_index, dim2_index, dim3_index, dim4_index};
            return get_element(idx);
        }
        virtual double get_element(const GridIndices& indices) const {
            checkIndices(indices, "get_element");
            return double(readElem(indices.at(0), indices.at(1),
                                   indices.at(2), indices.at(3), __LINE__));
        }
        virtual idx_t get_elements_in_slice(void* buffer_ptr,
                                            const GridIndices& first_indices,
                                            const GridIndices& last_indices) const {
            checkIndices(first_indices, "get_elements_in_slice");
            checkIndices(last_indices, "get_elements_in_slice");
            idx_t n = 0;
            for (idx_t t = first_indices[0]; t <= last_indices[0]; t++)
                for (idx_t x = first_indices[1]; x <= last_indices[1]; x++)
                    for (idx_t y = first_indices[2]; y <= last_indices[2]; y++)
                        for (idx_t z = first_indices[3]; z <= last_indices[3]; z++, n++) {
                            real_t val = readElem(t, x, y, z, __LINE__);
                            ((real_t*)buffer_ptr)[n] = val;
                        }
            return n;
        }
        
        // Write one element.
        virtual void writeElem_TXYZ(real_t val,
                                     idx_t t, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            writeElem(val, t, x, y, z, line);
        }
        virtual void set_element(double val,
                                 idx_t dim1_index, idx_t dim2_index,
                                 idx_t dim3_index, idx_t dim4_index,
                                 idx_t dim5_index, idx_t dim6_index) {
            GridIndices idx = {dim1_index, dim2_index, dim3_index, dim4_index};
            set_element(val, idx);
        }            
        virtual void set_element(double val, const GridIndices& indices) {
            checkIndices(indices, "set_element");
            writeElem(real_t(val),
                      indices.at(0), indices.at(1),
                      indices.at(2), indices.at(3), __LINE__);
            set_updated(false);
        }
        virtual idx_t set_elements_in_slice_same(double val,
                                                 const GridIndices& first_indices,
                                                 const GridIndices& last_indices) {
            checkIndices(first_indices, "set_elements_in_slice_same");
            checkIndices(last_indices, "set_elements_in_slice_same");
            idx_t n = 0;
            for (idx_t t = first_indices[0]; t <= last_indices[0]; t++)
                for (idx_t x = first_indices[1]; x <= last_indices[1]; x++)
                    for (idx_t y = first_indices[2]; y <= last_indices[2]; y++)
                        for (idx_t z = first_indices[3]; z <= last_indices[3]; z++, n++)
                            writeElem(real_t(val), t, x, y, z, __LINE__);
            set_updated(false);
            return n;
        }
        virtual idx_t set_elements_in_slice(const void* buffer_ptr,
                                            const std::vector<idx_t>& first_indices,
                                            const std::vector<idx_t>& last_indices) {
            checkIndices(first_indices, "set_elements_in_slice");
            checkIndices(last_indices, "set_elements_in_slice");
            idx_t n = 0;
            for (idx_t t = first_indices[0]; t <= last_indices[0]; t++)
                for (idx_t x = first_indices[1]; x <= last_indices[1]; x++)
                    for (idx_t y = first_indices[2]; y <= last_indices[2]; y++)
                        for (idx_t z = first_indices[3]; z <= last_indices[3]; z++, n++) {
                            real_t val = ((real_t*)buffer_ptr)[n];
                            writeElem(val, t, x, y, z, __LINE__);
                    }
            set_updated(false);
            return n;
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

}
#endif
