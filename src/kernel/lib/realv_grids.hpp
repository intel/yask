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

#define USE_GET_INDEX 0

namespace yask {

    typedef GenericGridBase<real_t> RealGrid;
    typedef GenericGridBase<real_vec_t> RealVecGrid;

    // Pure-virtual base class for real_vec_t grids.  Supports up to 4
    // spatial dims and optional temporal dim.  Indices used to access
    // points are global and logical.  Example: if there are two ranks in
    // the x-dim, each of which has domain-size of 100 and a pad-size of 10,
    // the 1st rank's x indices will be -10..110, and the 2nd rank's x
    // indices will be 90..210.  TODO: allow different pos and neg-side
    // halos and/or padding.
    class RealVecGridBase :
        public virtual yk_grid {

    protected:
        RealVecGrid* _gp;

        // real_t sizes for up to 4 spatial dims.
        idx_t _dw=VLEN_W, _dx=VLEN_X, _dy=VLEN_Y, _dz=VLEN_Z; // domain sizes.
        idx_t _hw=0, _hx=0, _hy=0, _hz=0; // halo sizes.
        idx_t _pw=0, _px=0, _py=0, _pz=0; // halo + extra-pad sizes.
        idx_t _ow=0, _ox=0, _oy=0, _oz=0; // offsets into global problem domain.

        // real_vec_t sizes for up to 4 spatial dims.
        // halo vector-sizes are not given here, because they are not rounded up.
        idx_t _dwv=1, _dxv=1, _dyv=1, _dzv=1;
        idx_t _pwv=0, _pxv=0, _pyv=0, _pzv=0;
        idx_t _owv=0, _oxv=0, _oyv=0, _ozv=0;

        // Dynamic data.
        bool _is_updated = false; // data has been received from neighbors' halos.
        
        // Normalize element indices to vector indices and element offsets.
        ALWAYS_INLINE
        void normalize(idx_t elem_index,
                       idx_t& vec_index,
                       idx_t& elem_ofs,
                       idx_t vec_len,
                       idx_t vec_pad,
                       idx_t elem_pad) const {

            // Add padding before division to ensure negative indices work under
            // truncated division.
            idx_t padded_index = elem_index + elem_pad;

            // Normalize and remove added padding.
            vec_index = padded_index / vec_len - vec_pad;

            // Divide values with padding in numerator to avoid negative indices to get offsets.
            elem_ofs = padded_index % vec_len;
        }

        // Adjust logical spatial vector index to 0-based internal index by
        // adding padding and removing offset.  TODO: currently, the
        // compiler isn't able to eliminate some common sub-expressions in
        // addr calculation when these functions are used. Until this is
        // resolved, alternative code is used in derived classes if the
        // macro USE_GET_INDEX is not set.
        ALWAYS_INLINE idx_t get_index(idx_t vec_index,
                                      idx_t vec_pad,
                                      idx_t vec_ofs) const {
            return vec_index + vec_pad - vec_ofs;
        }
        ALWAYS_INLINE idx_t get_index_w(idx_t vec_index) const {
            return get_index(vec_index, _pwv, _owv);
        }
        ALWAYS_INLINE idx_t get_index_x(idx_t vec_index) const {
            return get_index(vec_index, _pxv, _oxv);
        }
        ALWAYS_INLINE idx_t get_index_y(idx_t vec_index) const {
            return get_index(vec_index, _pyv, _oyv);
        }
        ALWAYS_INLINE idx_t get_index_z(idx_t vec_index) const {
            return get_index(vec_index, _pzv, _ozv);
        }

        // Resize the underlying grid based on the current settings.
        virtual void resize_g() =0;
        
    public:
        RealVecGridBase(RealVecGrid* gp) :
            _gp(gp) { }
        virtual ~RealVecGridBase() { }

        // Get name.
        const std::string& get_name() const { return _gp->get_name(); }

        // Determine what dims are actually used for derived type.
        virtual bool got_t() const { return false; }
        virtual bool got_w() const { return false; }
        virtual bool got_x() const { return false; }
        virtual bool got_y() const { return false; }
        virtual bool got_z() const { return false; }
        virtual int get_num_dims() const {
            return (got_t() ? 1 : 0) +
                (got_w() ? 1 : 0) +
                (got_x() ? 1 : 0) +
                (got_y() ? 1 : 0) +
                (got_z() ? 1 : 0);
        }

        // Get temporal allocation.
        virtual inline idx_t get_tdim() const { return 1; }
        
        // Get domain-size for this rank after round-up.
        inline idx_t get_dw() const { return _dw; }
        inline idx_t get_dx() const { return _dx; }
        inline idx_t get_dy() const { return _dy; }
        inline idx_t get_dz() const { return _dz; }

        // Get halo-size (NOT rounded up).
        inline idx_t get_halo_w() const { return _hw; }
        inline idx_t get_halo_x() const { return _hx; }
        inline idx_t get_halo_y() const { return _hy; }
        inline idx_t get_halo_z() const { return _hz; }

        // Get padding-size after round-up.
        // This includes the halo.
        inline idx_t get_pad_w() const { return _pw; }
        inline idx_t get_pad_x() const { return _px; }
        inline idx_t get_pad_y() const { return _py; }
        inline idx_t get_pad_z() const { return _pz; }

        // Get extra-padding-size after round-up.
        // Since the extra pad is in addition to the halo, these
        // values may not be multiples of the vector lengths.
        inline idx_t get_extra_pad_w() const { return _pw - _hw; }
        inline idx_t get_extra_pad_x() const { return _px - _hx; }
        inline idx_t get_extra_pad_y() const { return _py - _hy; }
        inline idx_t get_extra_pad_z() const { return _pz - _hz; }

        // Get first logical index in domain on this rank.
        inline idx_t get_first_w() const { return _ow; }
        inline idx_t get_first_x() const { return _ox; }
        inline idx_t get_first_y() const { return _oy; }
        inline idx_t get_first_z() const { return _oz; }

        // Get last logical index in domain on this rank.
        inline idx_t get_last_w() const { return _ow + _dw - 1; }
        inline idx_t get_last_x() const { return _ox + _dx - 1; }
        inline idx_t get_last_y() const { return _oy + _dy - 1; }
        inline idx_t get_last_z() const { return _oz + _dz - 1; }

        // Set domain-size for this rank and round-up.
        inline void set_dw(idx_t dw) {
            _dw = ROUND_UP(dw, VLEN_W); _dwv = _dw / VLEN_W; resize_g(); }
        inline void set_dx(idx_t dx) {
            _dx = ROUND_UP(dx, VLEN_X); _dxv = _dx / VLEN_X; resize_g(); }
        inline void set_dy(idx_t dy) {
            _dy = ROUND_UP(dy, VLEN_Y); _dyv = _dy / VLEN_Y; resize_g(); }
        inline void set_dz(idx_t dz) {
            _dz = ROUND_UP(dz, VLEN_Z); _dzv = _dz / VLEN_Z; resize_g(); }

        // Set halo sizes.
        // Automatically increase padding if less than halo.
        // Halo sizes are not rounded up.
        inline void set_halo_w(idx_t hw) { _hw = hw; set_pad_w(_pw); }
        inline void set_halo_x(idx_t hx) { _hx = hx; set_pad_x(_px); }
        inline void set_halo_y(idx_t hy) { _hy = hy; set_pad_y(_py); }
        inline void set_halo_z(idx_t hz) { _hz = hz; set_pad_z(_pz); }

        // Set padding and round-up.
        // Padding will be increased to size of halo if needed.
        inline void set_pad_w(idx_t pw) {
            _pw = ROUND_UP(std::max(pw, _hw), VLEN_W); _pwv = _pw / VLEN_W; resize_g(); }
        inline void set_pad_x(idx_t px) {
            _px = ROUND_UP(std::max(px, _hx), VLEN_X); _pxv = _px / VLEN_X; resize_g(); }
        inline void set_pad_y(idx_t py) {
            _py = ROUND_UP(std::max(py, _hy), VLEN_Y); _pyv = _py / VLEN_Y; resize_g(); }
        inline void set_pad_z(idx_t pz) {
            _pz = ROUND_UP(std::max(pz, _hz), VLEN_Z); _pzv = _pz / VLEN_Z; resize_g(); }

        // Set padding in addition to halo size and round-up.
        inline void set_extra_pad_w(idx_t epw) { set_pad_w(_hw + epw); }
        inline void set_extra_pad_x(idx_t epx) { set_pad_x(_hx + epx); }
        inline void set_extra_pad_y(idx_t epy) { set_pad_y(_hy + epy); }
        inline void set_extra_pad_z(idx_t epz) { set_pad_z(_hz + epz); }
        
        // Set offset and round-up.
        inline void set_ofs_w(idx_t ow) {
            _ow = ROUND_UP(ow, VLEN_W); _owv = _ow / VLEN_W; }
        inline void set_ofs_x(idx_t ox) {
            _ox = ROUND_UP(ox, VLEN_X); _oxv = _ox / VLEN_X; }
        inline void set_ofs_y(idx_t oy) {
            _oy = ROUND_UP(oy, VLEN_Y); _oyv = _oy / VLEN_Y; }
        inline void set_ofs_z(idx_t oz) {
            _oz = ROUND_UP(oz, VLEN_Z); _ozv = _oz / VLEN_Z; }

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
        inline idx_t get_num_real_vecs() const {
            return _gp->get_num_elems();
        }

        // Get number of elements.
        inline idx_t get_num_elems() {
            return _gp->get_num_bytes() / sizeof(real_t);
        }

        // Get size in bytes.
        inline size_t get_num_bytes() const {
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
        void normalize_w(idx_t w, idx_t& vec_index, idx_t& elem_ofs) const {
            normalize(w, vec_index, elem_ofs, VLEN_W, _pwv, _pw);
        }
        ALWAYS_INLINE
        void normalize_x(idx_t x, idx_t& vec_index, idx_t& elem_ofs) const {
            normalize(x, vec_index, elem_ofs, VLEN_X, _pxv, _px);
        }
        ALWAYS_INLINE
        void normalize_y(idx_t y, idx_t& vec_index, idx_t& elem_ofs) const {
            normalize(y, vec_index, elem_ofs, VLEN_Y, _pyv, _py);
        }
        ALWAYS_INLINE
        void normalize_z(idx_t z, idx_t& vec_index, idx_t& elem_ofs) const {
            normalize(z, vec_index, elem_ofs, VLEN_Z, _pzv, _pz);
        }

        // Read one element.
        virtual real_t readElem_TWXYZ(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                                      int line) const =0;

        // Write one element.
        virtual void writeElem_TWXYZ(real_t val,
                                     idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,                               
                                     int line) =0;

        // Read one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TWXYZ(idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const =0;
        
        // Write one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TWXYZ(const real_vec_t& v,
                                        idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                        int line) =0;

        // Print one element.
        virtual void printElem_TWXYZ(std::ostream& os, const std::string& m,
                                     idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                                     real_t e, int line, bool newline = true) const;

        // Print one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void printVecNorm_TWXYZ(std::ostream& os, const std::string& m,
                                        idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                        const real_vec_t& v,
                                        int line) const;

        // Direct access to data.
        real_vec_t* get_storage() {
            return _gp->get_storage();
        }
        const real_vec_t* get_storage() const {
            return _gp->get_storage();
        }
        void set_storage(void* buf, size_t offset) {
            _gp->set_storage(buf, offset);
        }
        RealVecGrid* getGenericGrid() {
            return _gp;
        }
        const RealVecGrid* getGenericGrid() const {
            return _gp;
        }


        // APIs not defined above.
        // See yask_kernel_api.hpp.
        virtual const std::string& get_dim_name(int n) const {
            assert(n >= 0);
            assert(n < get_num_dims());
            return _gp->get_dim_name(n);
        }
    };
    
    // A 3D (x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn> class RealVecGrid_XYZ :
        public RealVecGridBase {

    protected:

        GenericGrid3d<real_vec_t, LayoutFn> _data;

        virtual void resize_g() {
            _data.set_d1(_dxv + 2 * _pxv);
            _data.set_d2(_dyv + 2 * _pyv);
            _data.set_d3(_dzv + 2 * _pzv);
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
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        const real_vec_t* getVecPtrNorm(idx_t xv, idx_t yv, idx_t zv,
                                        bool checkBounds=true) const {

#ifdef TRACE_MEM
            std::cout << _name << "." << "RealVecGrid_XYZ::getVecPtrNorm(" <<
                xv << "," << yv << "," << zv << ")";
#endif
        
            // adjust for padding and offset.
#if USE_GET_INDEX
            xv = get_index_x(xv);
            yv = get_index_y(yv);
            zv = get_index_z(zv);
#else
            xv += _pxv - _oxv;
            yv += _pyv - _oyv;
            zv += _pzv - _ozv;
#endif

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
            return &(*vp)(0, xe, ye, ze);
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
        virtual real_t readElem_TWXYZ(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                                      int line) const {
            assert(t == 0);
            assert(w == 0);
            return readElem(x, y, z, line);
        }

        // Write one element.
        virtual void writeElem_TWXYZ(real_t val,
                                     idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            assert(t == 0);
            assert(w == 0);
            writeElem(val, x, y, z, line);
        }

        // Read one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TWXYZ(idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const {
            assert(t == 0);
            assert(wv == 0);
            return readVecNorm(xv, yv, zv, line);
        }
        
        // Write one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TWXYZ(const real_vec_t& val,
                                        idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                        int line) {
            assert(t == 0);
            assert(wv == 0);
            writeVecNorm(val, xv, yv, zv, line);
        }

        // Print one vector.
        void printVecNorm(std::ostream& os, const std::string& m,
                      idx_t xv, idx_t yv, idx_t zv, const real_vec_t& v,
                      int line) const {
            printVecNorm_TWXYZ(0, 0, xv, yv, zv, v, line);
        }

        // Print one element.
        void printElem(std::ostream& os, const std::string& m,
                       idx_t x, idx_t y, idx_t z, real_t e,
                       int line) const {
            printElem_TWXYZ(0, 0, x, y, z, e, line);
        }
    };

    // A 4D (w, x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn> class RealVecGrid_WXYZ :
        public RealVecGridBase {
    
    protected:

        GenericGrid4d<real_vec_t, LayoutFn> _data;

        virtual void resize_g() {
            _data.set_d1(_dwv + 2 * _pwv);
            _data.set_d2(_dxv + 2 * _pxv);
            _data.set_d3(_dyv + 2 * _pyv);
            _data.set_d4(_dzv + 2 * _pzv);
        }
        
    public:

        // Ctor.
        RealVecGrid_WXYZ(const std::string& name) :
            RealVecGridBase(&_data),
            _data(name, "w", "x", "y", "z") { }

        // Determine what dims are defined.
        virtual bool got_w() const { return true; }
        virtual bool got_x() const { return true; }
        virtual bool got_y() const { return true; }
        virtual bool got_z() const { return true; }
        
        // Get pointer to the real_vec_t at vector offset wv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        const real_vec_t* getVecPtrNorm(idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                        bool checkBounds=true) const {

#ifdef TRACE_MEM
            std::cout << _name << "." << "RealVecGrid_WXYZ::getVecPtrNorm(" <<
                wv << "," << xv << "," << yv << "," << zv << ")";
#endif
        
            // adjust for padding and offset.
#if USE_GET_INDEX
            wv = get_index_w(wv);
            xv = get_index_x(xv);
            yv = get_index_y(yv);
            zv = get_index_z(zv);
#else
            wv += _pwv - _owv;
            xv += _pxv - _oxv;
            yv += _pyv - _oyv;
            zv += _pzv - _ozv;
#endif

#ifdef TRACE_MEM
            if (checkBounds)
                std::cout << " => " << _data.get_index(wv, xv, yv, zv);
            std::cout << std::endl << flush;
#endif
            return &_data(wv, xv, yv, zv, checkBounds);
        }

        // Non-const version.
        ALWAYS_INLINE
        real_vec_t* getVecPtrNorm(idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                  bool checkBounds=true) {

            const real_vec_t* vp =
                const_cast<const RealVecGrid_WXYZ*>(this)->getVecPtrNorm(wv, xv, yv, zv,
                                                                       checkBounds);
            return const_cast<real_vec_t*>(vp);
        }
    
        // Get a pointer to one real_t.
        ALWAYS_INLINE
        const real_t* getElemPtr(idx_t w, idx_t x, idx_t y, idx_t z,
                                 bool checkBounds=true) const {
            idx_t wv, we, xv, xe, yv, ye, zv, ze;
            normalize_w(w, wv, we);
            normalize_x(x, xv, xe);
            normalize_y(y, yv, ye);
            normalize_z(z, zv, ze);

            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(wv, xv, yv, zv, checkBounds);

            // Extract point from vector.
            return &(*vp)(we, xe, ye, ze);
        }

        // non-const version.
        ALWAYS_INLINE
        real_t* getElemPtr(idx_t w, idx_t x, idx_t y, idx_t z,
                           bool checkBounds=true) {
            const real_t* p = const_cast<const RealVecGrid_WXYZ*>(this)->getElemPtr(w, x, y, z,
                                                                                    checkBounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        ALWAYS_INLINE
        real_t readElem(idx_t w, idx_t x, idx_t y, idx_t z,
                        int line) const {
            const real_t* ep = getElemPtr(w, x, y, z);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem(std::cout, "readElem", w, x, y, z, e, line);
#endif
            return e;
        }

        // Write one element.
        ALWAYS_INLINE
        void writeElem(real_t val, idx_t w, idx_t x, idx_t y, idx_t z,
                       int line) {
            real_t* ep = getElemPtr(w, x, y, z);
            *ep = val;
#ifdef TRACE_MEM
            printElem(std::cout, "writeElem", w, x, y, z, val, line);
#endif
        }

        // Read one vector at vector offset wv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        real_vec_t readVecNorm(idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                               int line) const {
#ifdef TRACE_MEM
            std::cout << "readVecNorm(" << wv << "," << xv << "," << yv << "," << zv << ")..." << std::endl;
#endif        
            const real_vec_t* p = getVecPtrNorm(wv, xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            real_vec_t v;
            v.loadFrom(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "readVec", wv, xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.read(p, line);
#endif
            return v;
        }

        // Write one vector at vector offset wv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        void writeVecNorm(const real_vec_t& v, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                          int line) {
            real_vec_t* p = getVecPtrNorm(wv, xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            v.storeTo(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "writeVec", wv, xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.write(p, line);
#endif
        }

        // Prefetch one vector at vector offset wv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        template <int level>
        ALWAYS_INLINE
        void prefetchVecNorm(idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                             int line) const {
#ifdef TRACE_MEM
            std::cout << "prefetchVecNorm<" << level << ">(" <<
                wv << "," << xv << "," << yv << "," << zv << ")..." << std::endl;
#endif        
            const char* p = (const char*)getVecPtrNorm(wv, xv, yv, zv, false);
            __assume_aligned(p, CACHELINE_BYTES);
            _mm_prefetch (p, level);
#ifdef MODEL_CACHE
            cache_model.prefetch(p, level, line);
#endif
        }

        // Read one element.
        virtual real_t readElem_TWXYZ(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                                      int line) const {
            assert(t == 0);
            return readElem(w, x, y, z, line);
        }

        // Write one element.
        virtual void writeElem_TWXYZ(real_t val,
                                     idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            assert(t == 0);
            writeElem(val, w, x, y, z, line);
        }

        // Read one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TWXYZ(idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const {
            assert(t == 0);
            return readVecNorm(wv, xv, yv, zv, line);
        }
        
        // Write one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TWXYZ(const real_vec_t& val,
                                        idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                        int line) {
            assert(t == 0);
            writeVecNorm(val, wv, xv, yv, zv, line);
        }

        // Print one vector.
        void printVecNorm(std::ostream& os, const std::string& m,
                      idx_t wv, idx_t xv, idx_t yv, idx_t zv, const real_vec_t& v,
                      int line) const {
            printVecNorm_TWXYZ(0, wv, xv, yv, zv, v, line);
        }

        // Print one element.
        void printElem(std::ostream& os, const std::string& m,
                       idx_t w, idx_t x, idx_t y, idx_t z, real_t e,
                       int line) const {
            printElem_TWXYZ(0, w, x, y, z, e, line);
        }
    };

    // Base class that adds a templated temporal size for
    // index-calculation efficiency.
    template <idx_t tdim>
    class RealVecGrid_T : public RealVecGridBase {

    protected:
        idx_t _tdim = 1;        // FIXME: remove this test.

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

            // Avoid discontinuity caused by negative time by adding a large
            // offset to the t index.  So, t can be negative, but not so
            // much that it would still be negative after adding the offset.
            // This should not be a practical restriction.
            t += 256 * _tdim;
            assert(t >= 0);
            assert(t % CPTS_T == 0);
            idx_t t_idx = t / idx_t(CPTS_T);
            return t_idx % _tdim;
        }

    public:

        RealVecGrid_T(RealVecGrid* gp) :
            RealVecGridBase(gp), _tdim(tdim) { }

        // Get temporal allocation.
        virtual inline idx_t get_tdim() const final {
            return _tdim;
        }
    };

    // A 4D (t, x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn, idx_t _tdim> class RealVecGrid_TXYZ :
        public RealVecGrid_T<_tdim> {
    
    protected:

        GenericGrid4d<real_vec_t, LayoutFn> _data;

        virtual void resize_g() {
            _data.set_d1(_tdim);
            _data.set_d2(this->_dxv + 2 * this->_pxv);
            _data.set_d3(this->_dyv + 2 * this->_pyv);
            _data.set_d4(this->_dzv + 2 * this->_pzv);
        }
        
    public:

        // Ctor.
        RealVecGrid_TXYZ(const std::string& name) :
            RealVecGrid_T<_tdim>(&_data),
            _data(name, "t", "x", "y", "z") { }

        // Determine what dims are defined.
        virtual bool got_t() const { return true; }
        virtual bool got_x() const { return true; }
        virtual bool got_y() const { return true; }
        virtual bool got_z() const { return true; }
        
        // Get pointer to the real_vec_t at vector offset t, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        const real_vec_t* getVecPtrNorm(idx_t t, idx_t xv, idx_t yv, idx_t zv,
                                        bool checkBounds=true) const {

#ifdef TRACE_MEM
            std::cout << _name << "." << "RealVecGrid_TWXYZ::getVecPtrNorm(" <<
                t << "," << << xv << "," << yv << "," << zv << ")";
#endif
        
            // adjust for padding and offset.
            t = this->get_index_t(t);
#if USE_GET_INDEX
            xv = this->get_index_x(xv);
            yv = this->get_index_y(yv);
            zv = this->get_index_z(zv);
#else
            xv += this->_pxv - this->_oxv;
            yv += this->_pyv - this->_oyv;
            zv += this->_pzv - this->_ozv;
#endif

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
            this->normalize_x(x, xv, xe);
            this->normalize_y(y, yv, ye);
            this->normalize_z(z, zv, ze);

            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(t, xv, yv, zv, checkBounds);

            // Extract point from vector.
            return &(*vp)(0, xe, ye, ze);
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
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        real_vec_t readVecNorm(idx_t t, idx_t xv, idx_t yv, idx_t zv,
                               int line) const {
#ifdef TRACE_MEM
            std::cout << "readVecNorm(" << t "," << xv <<
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
        virtual real_t readElem_TWXYZ(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                                      int line) const {
            assert(w == 0);
            return readElem(t, x, y, z, line);
        }

        // Write one element.
        virtual void writeElem_TWXYZ(real_t val,
                                     idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            assert(w == 0);
            writeElem(val, t, x, y, z, line);
        }

        // Read one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TWXYZ(idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const {
            assert(wv == 0);
            return readVecNorm(t, xv, yv, zv, line);
        }
        
        // Write one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TWXYZ(const real_vec_t& val,
                                        idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                        int line) {
            assert(wv == 0);
            writeVecNorm(val, t, xv, yv, zv, line);
        }

        // Print one vector.
        void printVecNorm(std::ostream& os, const std::string& m,
                          idx_t t, idx_t xv, idx_t yv, idx_t zv,
                          const real_vec_t& v,
                          int line) const {
            printVecNorm_TWXYZ(t, 0, xv, yv, zv, v, line);
        }

        // Print one element.
        void printElem(std::ostream& os, const std::string& m,
                       idx_t t, idx_t x, idx_t y, idx_t z,
                       real_t e,
                       int line) const {
            printElem_TWXYZ(t, 0, x, y, z, e, line);
        }
    };

    // A 5D (t, w, x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn, idx_t _tdim> class RealVecGrid_TWXYZ :
        public RealVecGrid_T<_tdim> {
    
    protected:

        GenericGrid5d<real_vec_t, LayoutFn> _data;

        virtual void resize_g() {
            _data.set_d1(_tdim);
            _data.set_d2(this->_dwv + 2 * this->_pwv);
            _data.set_d3(this->_dxv + 2 * this->_pxv);
            _data.set_d4(this->_dyv + 2 * this->_pyv);
            _data.set_d5(this->_dzv + 2 * this->_pzv);
        }
        
    public:

        // Ctor.
        RealVecGrid_TWXYZ(const std::string& name) :
            RealVecGrid_T<_tdim>(&_data),
            _data(name, "t", "w", "x", "y", "z") { }

        // Determine what dims are defined.
        virtual bool got_t() const { return true; }
        virtual bool got_w() const { return true; }
        virtual bool got_x() const { return true; }
        virtual bool got_y() const { return true; }
        virtual bool got_z() const { return true; }
        
        // Get pointer to the real_vec_t at vector offset t, wv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        const real_vec_t* getVecPtrNorm(idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                        bool checkBounds=true) const {

#ifdef TRACE_MEM
            std::cout << _name << "." << "RealVecGrid_TWXYZ::getVecPtrNorm(" <<
                t << "," << wv << "," << xv << "," << yv << "," << zv << ")";
#endif
        
            // adjust for padding and offset.
            t = this->get_index_t(t);
#if USE_GET_INDEX
            wv = this->get_index_w(wv);
            xv = this->get_index_x(xv);
            yv = this->get_index_y(yv);
            zv = this->get_index_z(zv);
#else
            wv += this->_pwv - this->_owv;
            xv += this->_pxv - this->_oxv;
            yv += this->_pyv - this->_oyv;
            zv += this->_pzv - this->_ozv;
#endif

#ifdef TRACE_MEM
            if (checkBounds)
                std::cout << " => " << _data.get_index(t, wv, xv, yv, zv);
            std::cout << std::endl << flush;
#endif
            return &_data(t, wv, xv, yv, zv, checkBounds);
        }

        // Non-const version.
        ALWAYS_INLINE
        real_vec_t* getVecPtrNorm(idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                  bool checkBounds=true) {

            const real_vec_t* vp =
                const_cast<const RealVecGrid_TWXYZ*>(this)->getVecPtrNorm(t, wv, xv, yv, zv,
                                                                          checkBounds);
            return const_cast<real_vec_t*>(vp);
        }
    
        // Get a pointer to one real_t.
        ALWAYS_INLINE
        const real_t* getElemPtr(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                                 bool checkBounds=true) const {
            idx_t wv, we, xv, xe, yv, ye, zv, ze;
            this->normalize_w(w, wv, we);
            this->normalize_x(x, xv, xe);
            this->normalize_y(y, yv, ye);
            this->normalize_z(z, zv, ze);

            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(t, wv, xv, yv, zv, checkBounds);

            // Extract point from vector.
            return &(*vp)(we, xe, ye, ze);
        }

        // non-const version.
        ALWAYS_INLINE
        real_t* getElemPtr(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                           bool checkBounds=true) {
            const real_t* p =
                const_cast<const RealVecGrid_TWXYZ*>(this)->getElemPtr(t, w, x, y, z,
                                                                       checkBounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        ALWAYS_INLINE
        real_t readElem(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                        int line) const {
            const real_t* ep = getElemPtr(t, w, x, y, z);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem(std::cout, "readElem", t, w, x, y, z, e, line);
#endif
            return e;
        }

        // Write one element.
        ALWAYS_INLINE
        void writeElem(real_t val,
                       idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                       int line) {
            real_t* ep = getElemPtr(t, w, x, y, z);
            *ep = val;
#ifdef TRACE_MEM
            printElem(std::cout, "writeElem", t, w, x, y, z, val, line);
#endif
        }

        // Read one vector at vector offset t, wv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        real_vec_t readVecNorm(idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                               int line) const {
#ifdef TRACE_MEM
            std::cout << "readVecNorm(" << t "," << wv << "," << xv <<
                "," << yv << "," << zv << ")..." << std::endl;
#endif        
            const real_vec_t* p = getVecPtrNorm(t, wv, xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            real_vec_t v;
            v.loadFrom(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "readVec", t, wv, xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.read(p, line);
#endif
            return v;
        }

        // Write one vector at vector offset t, wv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE void
        writeVecNorm(const real_vec_t& v,
                     idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                     int line) {
            real_vec_t* p = getVecPtrNorm(t, wv, xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            v.storeTo(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "writeVec", t, wv, xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.write(p, line);
#endif
        }

        // Prefetch one vector at vector offset t, wv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        template <int level>
        ALWAYS_INLINE
        void prefetchVecNorm(idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                             int line) const {
#ifdef TRACE_MEM
            std::cout << "prefetchVecNorm<" << level << ">(" << t << "," <<
                wv << "," << xv << "," << yv << "," << zv << ")..." << std::endl;
#endif        
            const char* p = (const char*)getVecPtrNorm(t, wv, xv, yv, zv, false);
            __assume_aligned(p, CACHELINE_BYTES);
            _mm_prefetch (p, level);
#ifdef MODEL_CACHE
            cache_model.prefetch(p, level, line);
#endif
        }

        // Read one element.
        virtual real_t readElem_TWXYZ(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                                      int line) const {
            return readElem(t, w, x, y, z, line);
        }

        // Write one element.
        virtual void writeElem_TWXYZ(real_t val,
                                     idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            writeElem(val, t, w, x, y, z, line);
        }

        // Read one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TWXYZ(idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const {
            return readVecNorm(t, wv, xv, yv, zv, line);
        }
        
        // Write one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TWXYZ(const real_vec_t& val,
                                        idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                        int line) {
            writeVecNorm(val, t, wv, xv, yv, zv, line);
        }

        // Print one vector.
        void printVecNorm(std::ostream& os, const std::string& m,
                          idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                          const real_vec_t& v,
                          int line) const {
            printVecNorm_TWXYZ(t, wv, xv, yv, zv, v, line);
        }

        // Print one element.
        void printElem(std::ostream& os, const std::string& m,
                       idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                       real_t e,
                       int line) const {
            printElem_TWXYZ(t, w, x, y, z, e, line);
        }
    };

}
#endif
