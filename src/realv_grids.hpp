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

#include "generic_grids.hpp"

#ifdef MODEL_CACHE
extern yask::Cache cache_model;
#endif

namespace yask {

    typedef GenericGridBase<real_t> RealGrid;
    typedef GenericGridBase<real_vec_t> RealVecGrid;

    // Base class for real_vec_t grids.  Supports up to 4 spatial dims and
    // optional temporal dim.  Indices used to access points are global and
    // logical.  Example: if there are two ranks in the x-dim, each of which
    // has domain-size of 100 and a pad-size of 10, the 1st rank's x indices
    // will be -10..110, and the 2nd rank's x indices will be 90..210.
    // TODO: allow different pos and neg-side halos and/or padding.
    class RealVecGridBase {

    protected:
        std::string _name;
        RealVecGrid* _gp;

        // real_t sizes for up to 4 spatial dims.
        idx_t _dn=1, _dx=1, _dy=1, _dz=1; // domain sizes.
        idx_t _hn=0, _hx=0, _hy=0, _hz=0; // halo sizes.
        idx_t _pn=0, _px=0, _py=0, _pz=0; // halo + extra-pad sizes.
        idx_t _on=0, _ox=0, _oy=0, _oz=0; // offsets into global problem domain.

        // real_vec_t sizes for up to 4 spatial dims.
        // halo vector-sizes are not given here, because they are not rounded up.
        idx_t _dnv, _dxv, _dyv, _dzv;
        idx_t _pnv, _pxv, _pyv, _pzv;
        idx_t _onv, _oxv, _oyv, _ozv;

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
        
    public:
        RealVecGridBase(idx_t dn, idx_t dx, idx_t dy, idx_t dz, // rank domain sizes.
                        idx_t hn, idx_t hx, idx_t hy, idx_t hz, // halo sizes.
                        idx_t pn, idx_t px, idx_t py, idx_t pz, // extra-pad sizes.
                        idx_t on, idx_t ox, idx_t oy, idx_t oz, // offsets within global problem.
                        const std::string& name,
                        RealVecGrid* gp);
        virtual ~RealVecGridBase() { }

        // Get name.
        const std::string& get_name() { return _name; }

        // Get temporal allocation.
        virtual inline idx_t get_tdim() const =0;
        
        // Get domain-size parameters after round-up.
        inline idx_t get_dn() const { return _dn; }
        inline idx_t get_dx() const { return _dx; }
        inline idx_t get_dy() const { return _dy; }
        inline idx_t get_dz() const { return _dz; }

        // Get halo-size parameters (NOT rounded up).
        inline idx_t get_halo_n() const { return _hn; }
        inline idx_t get_halo_x() const { return _hx; }
        inline idx_t get_halo_y() const { return _hy; }
        inline idx_t get_halo_z() const { return _hz; }

        // Get extra-padding-size parameters after round-up.
        // Since the extra pad is in addition to the halo, these
        // values may not be multiples of the vector lengths.
        inline idx_t get_pad_n() const { return _pn - _hn; }
        inline idx_t get_pad_x() const { return _px - _hx; }
        inline idx_t get_pad_y() const { return _py - _hy; }
        inline idx_t get_pad_z() const { return _pz - _hz; }

        // Get first logical index in domain on this rank.
        inline idx_t get_first_n() const { return _on; }
        inline idx_t get_first_x() const { return _ox; }
        inline idx_t get_first_y() const { return _oy; }
        inline idx_t get_first_z() const { return _oz; }

        // Get last logical index in domain on this rank.
        inline idx_t get_last_n() const { return _on + _dn - 1; }
        inline idx_t get_last_x() const { return _ox + _dx - 1; }
        inline idx_t get_last_y() const { return _oy + _dy - 1; }
        inline idx_t get_last_z() const { return _oz + _dz - 1; }
        
        // Determine what dims are defined.
        virtual bool got_t() const { return false; }
        virtual bool got_n() const { return false; }
        virtual bool got_x() const { return false; }
        virtual bool got_y() const { return false; }
        virtual bool got_z() const { return false; }

        // Initialize memory to a given value.
        virtual void set_same(real_t val) {
            real_vec_t rn;
            rn = val;               // broadcast.
            _gp->set_same(rn);
        }

        // Initialize memory to incrementing values based on val.
        virtual void set_diff(real_t val);

        // Get number of real_vecs, including halos & padding.
        inline idx_t get_num_real_vecs() const {
            return _gp->get_num_elems();
        }

        // Get size in bytes.
        inline idx_t get_num_bytes() const {
            return _gp->get_num_bytes();
        }

  
        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const RealVecGridBase& ref,
                              real_t epsilon = EPSILON,
                              int maxPrint = 20,
                              std::ostream& os = std::cerr) const;

        // Normalize element indices to vector indices and element offsets.
        ALWAYS_INLINE
        void normalize_n(idx_t n, idx_t& vec_index, idx_t& elem_ofs) const {
            normalize(n, vec_index, elem_ofs, VLEN_N, _pnv, _pn);
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
        virtual real_t readElem_TNXYZ(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                                      int line) const =0;

        // Write one element.
        virtual void writeElem_TNXYZ(real_t val,
                                     idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,                               
                                     int line) =0;

        // Read one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TNXYZ(idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const =0;
        
        // Write one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TNXYZ(const real_vec_t& v,
                                        idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                        int line) =0;

        // Print one element.
        virtual void printElem_TNXYZ(std::ostream& os, const std::string& m,
                                     idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                                     real_t e, int line, bool newline = true) const;

        // Print one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void printVecNorm_TNXYZ(std::ostream& os, const std::string& m,
                                        idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                        const real_vec_t& v,
                                        int line) const;

        // Direct access to data (dangerous!).
        real_vec_t* getRawData() {
            return _gp->getRawData();
        }
        const real_vec_t* getRawData() const {
            return _gp->getRawData();
        }
        RealVecGrid* getGenericGrid() {
            return _gp;
        }
        const RealVecGrid* getGenericGrid() const {
            return _gp;
        }

    };

    // Base class that adds a templated temporal size for
    // index-calculation efficiency.
    template <idx_t _tdim>
    class RealVecGridTemplate : public RealVecGridBase {

    protected:

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
            // much that it would still be negaive after adding the offset.
            // This should not be a practical restriction.
            t += 256 * _tdim;
            assert(t >= 0);
            assert(t % CPTS_T == 0);
            idx_t t_idx = t / idx_t(CPTS_T);
            return t_idx % _tdim;
        }

        // Adjust logical spatial vector index to 0-based internal index by
        // adding padding and removing offset.  TODO: currently, the
        // compiler isn't able to eliminate some common sub-expressions in
        // addr calculation when these functions are used. Until this is
        // resolved, alternative code is used below if the macro
        // USE_GET_INDEX is not set.
        ALWAYS_INLINE idx_t get_index(idx_t vec_index,
                                      idx_t vec_pad,
                                      idx_t vec_ofs) const {
            return vec_index + vec_pad - vec_ofs;
        }
        ALWAYS_INLINE idx_t get_index_n(idx_t vec_index) const {
            return get_index(vec_index, _pnv, _onv);
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

    public:

        RealVecGridTemplate(idx_t dn, idx_t dx, idx_t dy, idx_t dz,
                            idx_t hn, idx_t hx, idx_t hy, idx_t hz,
                            idx_t pn, idx_t px, idx_t py, idx_t pz,
                            idx_t on, idx_t ox, idx_t oy, idx_t oz,
                            const std::string& name,
                            RealVecGrid* gp) :
            RealVecGridBase(dn, dx, dy, dz,
                            hn, hx, hy, hz,
                            pn, px, py, pz,
                            on, ox, oy, oz,
                            name, gp) { }

        // Get temporal allocation.
        virtual inline idx_t get_tdim() const final {
            return _tdim;
        }
    };

    
    // A 3D (x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn> class RealVecGrid_XYZ :
        public RealVecGridTemplate<1> {

    protected:

        GenericGrid3d<real_vec_t, LayoutFn> _data;
    
    public:

        // Ctor.
        // Dimensions are real_t elements, not real_vec_t.
        RealVecGrid_XYZ(idx_t dx, idx_t dy, idx_t dz,
                        idx_t hx, idx_t hy, idx_t hz,
                        idx_t px, idx_t py, idx_t pz,
                        idx_t ox, idx_t oy, idx_t oz,
                        const std::string& name,
                        bool use_hbw,
                        std::ostream& msg_stream) :
            RealVecGridTemplate(1, dx, dy, dz,
                                0, hx, hy, hz,
                                0, px, py, pz,
                                0, ox, oy, oz,
                                name, &_data),

            // Alloc space for required number of real_vec_t's.
            _data(_dxv + 2*_pxv,
                  _dyv + 2*_pyv,
                  _dzv + 2*_pzv,
                  GRID_ALIGNMENT,
                  use_hbw)
        {
            _data.print_info(name, msg_stream);
        }

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
            idx_t xv, ie, yv, je, zv, ke;
            normalize_x(x, xv, ie);
            normalize_y(y, yv, je);
            normalize_z(z, zv, ke);

            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(xv, yv, zv, checkBounds);

            // Extract point from vector.
            return &(*vp)(0, ie, je, ke);
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
        virtual real_t readElem_TNXYZ(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                                      int line) const {
            assert(t == 0);
            assert(n == 0);
            return readElem(x, y, z, line);
        }

        // Write one element.
        virtual void writeElem_TNXYZ(real_t val,
                                     idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            assert(t == 0);
            assert(n == 0);
            writeElem(val, x, y, z, line);
        }

        // Read one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TNXYZ(idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const {
            assert(t == 0);
            assert(nv == 0);
            return readVecNorm(xv, yv, zv, line);
        }
        
        // Write one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TNXYZ(const real_vec_t& val,
                                        idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                        int line) {
            assert(t == 0);
            assert(nv == 0);
            writeVecNorm(val, xv, yv, zv, line);
        }

        // Print one vector.
        void printVecNorm(std::ostream& os, const std::string& m,
                      idx_t xv, idx_t yv, idx_t zv, const real_vec_t& v,
                      int line) const {
            printVecNorm_TNXYZ(0, 0, xv, yv, zv, v, line);
        }

        // Print one element.
        void printElem(std::ostream& os, const std::string& m,
                       idx_t x, idx_t y, idx_t z, real_t e,
                       int line) const {
            printElem_TNXYZ(0, 0, xv, yv, zv, e, line);
        }

    };

    // A 4D (n, x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn> class RealVecGrid_NXYZ :
        public RealVecGridTemplate<1> {
    
    protected:

        GenericGrid4d<real_vec_t, LayoutFn> _data;

    public:

        // Ctor.
        // Dimensions are real_t elements, not real_vec_t.
        RealVecGrid_NXYZ(idx_t dn, idx_t dx, idx_t dy, idx_t dz,
                         idx_t hn, idx_t hx, idx_t hy, idx_t hz,
                         idx_t pn, idx_t px, idx_t py, idx_t pz,
                         idx_t on, idx_t ox, idx_t oy, idx_t oz,
                         const std::string& name,
                         bool use_hbw,
                         std::ostream& msg_stream) :
            RealVecGridTemplate(dn, dx, dy, dz,
                                hn, hx, hy, hz,
                                pn, px, py, pz,
                                on, ox, oy, oz,
                                name, &_data),

            // Alloc space for required number of real_vec_t's.
            _data(_dnv + 2*_pnv,
                  _dxv + 2*_pxv,
                  _dyv + 2*_pyv,
                  _dzv + 2*_pzv,
                  GRID_ALIGNMENT,
                  use_hbw)
        {
            _data.print_info(name, msg_stream);
        }

        // Determine what dims are defined.
        virtual bool got_n() const { return true; }
        virtual bool got_x() const { return true; }
        virtual bool got_y() const { return true; }
        virtual bool got_z() const { return true; }
        
        // Get pointer to the real_vec_t at vector offset nv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        const real_vec_t* getVecPtrNorm(idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                        bool checkBounds=true) const {

#ifdef TRACE_MEM
            std::cout << _name << "." << "RealVecGrid_NXYZ::getVecPtrNorm(" <<
                nv << "," << xv << "," << yv << "," << zv << ")";
#endif
        
            // adjust for padding and offset.
#if USE_GET_INDEX
            nv = get_index_n(nv);
            xv = get_index_x(xv);
            yv = get_index_y(yv);
            zv = get_index_z(zv);
#else
            nv += _pnv - _onv;
            xv += _pxv - _oxv;
            yv += _pyv - _oyv;
            zv += _pzv - _ozv;
#endif

#ifdef TRACE_MEM
            if (checkBounds)
                std::cout << " => " << _data.get_index(nv, xv, yv, zv);
            std::cout << std::endl << flush;
#endif
            return &_data(nv, xv, yv, zv, checkBounds);
        }

        // Non-const version.
        ALWAYS_INLINE
        real_vec_t* getVecPtrNorm(idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                  bool checkBounds=true) {

            const real_vec_t* vp =
                const_cast<const RealVecGrid_NXYZ*>(this)->getVecPtrNorm(nv, xv, yv, zv,
                                                                       checkBounds);
            return const_cast<real_vec_t*>(vp);
        }
    
        // Get a pointer to one real_t.
        ALWAYS_INLINE
        const real_t* getElemPtr(idx_t n, idx_t x, idx_t y, idx_t z,
                                 bool checkBounds=true) const {
            idx_t nv, ne, xv, ie, yv, je, zv, ke;
            normalize_n(n, nv, ne);
            normalize_x(x, xv, ie);
            normalize_y(y, yv, je);
            normalize_z(z, zv, ke);

            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(nv, xv, yv, zv, checkBounds);

            // Extract point from vector.
            return &(*vp)(ne, ie, je, ke);
        }

        // non-const version.
        ALWAYS_INLINE
        real_t* getElemPtr(idx_t n, idx_t x, idx_t y, idx_t z,
                           bool checkBounds=true) {
            const real_t* p = const_cast<const RealVecGrid_NXYZ*>(this)->getElemPtr(n, x, y, z,
                                                                                    checkBounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        ALWAYS_INLINE
        real_t readElem(idx_t n, idx_t x, idx_t y, idx_t z,
                        int line) const {
            const real_t* ep = getElemPtr(n, x, y, z);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem(std::cout, "readElem", n, x, y, z, e, line);
#endif
            return e;
        }

        // Write one element.
        ALWAYS_INLINE
        void writeElem(real_t val, idx_t n, idx_t x, idx_t y, idx_t z,
                       int line) {
            real_t* ep = getElemPtr(n, x, y, z);
            *ep = val;
#ifdef TRACE_MEM
            printElem(std::cout, "writeElem", n, x, y, z, val, line);
#endif
        }

        // Read one vector at vector offset nv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        real_vec_t readVecNorm(idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                               int line) const {
#ifdef TRACE_MEM
            std::cout << "readVecNorm(" << nv << "," << xv << "," << yv << "," << zv << ")..." << std::endl;
#endif        
            const real_vec_t* p = getVecPtrNorm(nv, xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            real_vec_t v;
            v.loadFrom(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "readVec", nv, xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.read(p, line);
#endif
            return v;
        }

        // Write one vector at vector offset nv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        void writeVecNorm(const real_vec_t& v, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                          int line) {
            real_vec_t* p = getVecPtrNorm(nv, xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            v.storeTo(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "writeVec", nv, xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.write(p, line);
#endif
        }

        // Prefetch one vector at vector offset nv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        template <int level>
        ALWAYS_INLINE
        void prefetchVecNorm(idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                             int line) const {
#ifdef TRACE_MEM
            std::cout << "prefetchVecNorm<" << level << ">(" <<
                nv << "," << xv << "," << yv << "," << zv << ")..." << std::endl;
#endif        
            const char* p = (const char*)getVecPtrNorm(nv, xv, yv, zv, false);
            __assume_aligned(p, CACHELINE_BYTES);
            _mm_prefetch (p, level);
#ifdef MODEL_CACHE
            cache_model.prefetch(p, level, line);
#endif
        }

        // Read one element.
        virtual real_t readElem_TNXYZ(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                                      int line) const {
            assert(t == 0);
            return readElem(n, x, y, z, line);
        }

        // Write one element.
        virtual void writeElem_TNXYZ(real_t val,
                                     idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            assert(t == 0);
            writeElem(val, n, x, y, z, line);
        }

        // Read one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TNXYZ(idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const {
            assert(t == 0);
            return readVecNorm(nv, xv, yv, zv, line);
        }
        
        // Write one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TNXYZ(const real_vec_t& val,
                                        idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                        int line) {
            assert(t == 0);
            writeVecNorm(val, nv, xv, yv, zv, line);
        }

        // Print one vector.
        void printVecNorm(std::ostream& os, const std::string& m,
                      idx_t nv, idx_t xv, idx_t yv, idx_t zv, const real_vec_t& v,
                      int line) const {
            printVecNorm_TNXYZ(0, nv, xv, yv, zv, v, line);
        }

        // Print one element.
        void printElem(std::ostream& os, const std::string& m,
                       idx_t n, idx_t x, idx_t y, idx_t z, real_t e,
                       int line) const {
            printElem_TNXYZ(0, n, xv, yv, zv, e, line);
        }
    };

    // A 4D (t, x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn, idx_t _tdim> class RealVecGrid_TXYZ :
        public RealVecGridTemplate<_tdim> {
    
    protected:

        GenericGrid4d<real_vec_t, LayoutFn> _data;

    public:

        // Ctor.
        // Dimensions are real_t elements, not real_vec_t.
        RealVecGrid_TXYZ(idx_t dx, idx_t dy, idx_t dz,
                         idx_t hx, idx_t hy, idx_t hz,
                         idx_t px, idx_t py, idx_t pz,                         
                         idx_t ox, idx_t oy, idx_t oz,
                         const std::string& name,
                         bool use_hbw,
                         std::ostream& msg_stream) :
            RealVecGridTemplate<_tdim>(1, dx, dy, dz,
                                       0, hx, hy, hz,
                                       0, px, py, pz,
                                       0, ox, oy, oz,
                                       name, &_data),

            // Alloc space for required number of real_vec_t's.
            _data(_tdim,
                  RealVecGridTemplate<_tdim>::_dxv + 2*RealVecGridTemplate<_tdim>::_pxv,
                  RealVecGridTemplate<_tdim>::_dyv + 2*RealVecGridTemplate<_tdim>::_pyv,
                  RealVecGridTemplate<_tdim>::_dzv + 2*RealVecGridTemplate<_tdim>::_pzv,
                  GRID_ALIGNMENT,
                  use_hbw)
        {
            _data.print_info(name, msg_stream);
        }

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
            std::cout << _name << "." << "RealVecGrid_TNXYZ::getVecPtrNorm(" <<
                t << "," << << xv << "," << yv << "," << zv << ")";
#endif
        
            // adjust for padding and offset.
            t = RealVecGridTemplate<_tdim>::get_index_t(t);
#if USE_GET_INDEX
            xv = RealVecGridTemplate<_tdim>::get_index_x(xv);
            yv = RealVecGridTemplate<_tdim>::get_index_y(yv);
            zv = RealVecGridTemplate<_tdim>::get_index_z(zv);
#else
            xv += RealVecGridTemplate<_tdim>::_pxv - RealVecGridTemplate<_tdim>::_oxv;
            yv += RealVecGridTemplate<_tdim>::_pyv - RealVecGridTemplate<_tdim>::_oyv;
            zv += RealVecGridTemplate<_tdim>::_pzv - RealVecGridTemplate<_tdim>::_ozv;
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
            idx_t xv, ie, yv, je, zv, ke;
            RealVecGridTemplate<_tdim>::normalize_x(x, xv, ie);
            RealVecGridTemplate<_tdim>::normalize_y(y, yv, je);
            RealVecGridTemplate<_tdim>::normalize_z(z, zv, ke);

            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(t, xv, yv, zv, checkBounds);

            // Extract point from vector.
            return &(*vp)(0, ie, je, ke);
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
        virtual real_t readElem_TNXYZ(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                                      int line) const {
            assert(n == 0);
            return readElem(t, x, y, z, line);
        }

        // Write one element.
        virtual void writeElem_TNXYZ(real_t val,
                                     idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            assert(n == 0);
            writeElem(val, t, x, y, z, line);
        }

        // Read one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TNXYZ(idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const {
            assert(nv == 0);
            return readVecNorm(t, xv, yv, zv, line);
        }
        
        // Write one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TNXYZ(const real_vec_t& val,
                                        idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                        int line) {
            assert(nv == 0);
            writeVecNorm(val, t, xv, yv, zv, line);
        }

        // Print one vector.
        void printVecNorm(std::ostream& os, const std::string& m,
                          idx_t t, idx_t xv, idx_t yv, idx_t zv,
                          const real_vec_t& v,
                          int line) const {
            printVecNorm_TNXYZ(t, 0, xv, yv, zv, v, line);
        }

        // Print one element.
        void printElem(std::ostream& os, const std::string& m,
                       idx_t t, idx_t x, idx_t y, idx_t z,
                       real_t e,
                       int line) const {
            printElem_TNXYZ(t, 0, xv, yv, zv, e, line);
        }
    };

    // A 5D (t, n, x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn, idx_t _tdim> class RealVecGrid_TNXYZ :
        public RealVecGridTemplate<_tdim> {
    
    protected:

        GenericGrid5d<real_vec_t, LayoutFn> _data;

    public:

        // Ctor.
        // Dimensions are real_t elements, not real_vec_t.
        RealVecGrid_TNXYZ(idx_t dn, idx_t dx, idx_t dy, idx_t dz,
                          idx_t hn, idx_t hx, idx_t hy, idx_t hz,
                          idx_t pn, idx_t px, idx_t py, idx_t pz,
                          idx_t on, idx_t ox, idx_t oy, idx_t oz,
                          const std::string& name,
                          bool use_hbw,
                          std::ostream& msg_stream) :
            RealVecGridTemplate<_tdim>(dn, dx, dy, dz,
                                       hn, hx, hy, hz,
                                       pn, px, py, pz,
                                       on, ox, oy, oz,
                                       name, &_data),

            // Alloc space for required number of real_vec_t's.
            _data(_tdim,
                  RealVecGridTemplate<_tdim>::_dnv + 2*RealVecGridTemplate<_tdim>::_pnv,
                  RealVecGridTemplate<_tdim>::_dxv + 2*RealVecGridTemplate<_tdim>::_pxv,
                  RealVecGridTemplate<_tdim>::_dyv + 2*RealVecGridTemplate<_tdim>::_pyv,
                  RealVecGridTemplate<_tdim>::_dzv + 2*RealVecGridTemplate<_tdim>::_pzv,
                  GRID_ALIGNMENT,
                  use_hbw)
        {
            _data.print_info(name, msg_stream);
        }

        // Determine what dims are defined.
        virtual bool got_t() const { return true; }
        virtual bool got_n() const { return true; }
        virtual bool got_x() const { return true; }
        virtual bool got_y() const { return true; }
        virtual bool got_z() const { return true; }
        
        // Get pointer to the real_vec_t at vector offset t, nv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        const real_vec_t* getVecPtrNorm(idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                        bool checkBounds=true) const {

#ifdef TRACE_MEM
            std::cout << _name << "." << "RealVecGrid_TNXYZ::getVecPtrNorm(" <<
                t << "," << nv << "," << xv << "," << yv << "," << zv << ")";
#endif
        
            // adjust for padding and offset.
            t = RealVecGridTemplate<_tdim>::get_index_t(t);
#if USE_GET_INDEX
            nv = RealVecGridTemplate<_tdim>::get_index_n(nv);
            xv = RealVecGridTemplate<_tdim>::get_index_x(xv);
            yv = RealVecGridTemplate<_tdim>::get_index_y(yv);
            zv = RealVecGridTemplate<_tdim>::get_index_z(zv);
#else
            nv += RealVecGridTemplate<_tdim>::_pnv - RealVecGridTemplate<_tdim>::_onv;
            xv += RealVecGridTemplate<_tdim>::_pxv - RealVecGridTemplate<_tdim>::_oxv;
            yv += RealVecGridTemplate<_tdim>::_pyv - RealVecGridTemplate<_tdim>::_oyv;
            zv += RealVecGridTemplate<_tdim>::_pzv - RealVecGridTemplate<_tdim>::_ozv;
#endif

#ifdef TRACE_MEM
            if (checkBounds)
                std::cout << " => " << _data.get_index(t, nv, xv, yv, zv);
            std::cout << std::endl << flush;
#endif
            return &_data(t, nv, xv, yv, zv, checkBounds);
        }

        // Non-const version.
        ALWAYS_INLINE
        real_vec_t* getVecPtrNorm(idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                  bool checkBounds=true) {

            const real_vec_t* vp =
                const_cast<const RealVecGrid_TNXYZ*>(this)->getVecPtrNorm(t, nv, xv, yv, zv,
                                                                          checkBounds);
            return const_cast<real_vec_t*>(vp);
        }
    
        // Get a pointer to one real_t.
        ALWAYS_INLINE
        const real_t* getElemPtr(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                                 bool checkBounds=true) const {
            idx_t nv, ne, xv, ie, yv, je, zv, ke;
            RealVecGridTemplate<_tdim>::normalize_n(n, nv, ne);
            RealVecGridTemplate<_tdim>::normalize_x(x, xv, ie);
            RealVecGridTemplate<_tdim>::normalize_y(y, yv, je);
            RealVecGridTemplate<_tdim>::normalize_z(z, zv, ke);

            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(t, nv, xv, yv, zv, checkBounds);

            // Extract point from vector.
            return &(*vp)(ne, ie, je, ke);
        }

        // non-const version.
        ALWAYS_INLINE
        real_t* getElemPtr(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                           bool checkBounds=true) {
            const real_t* p =
                const_cast<const RealVecGrid_TNXYZ*>(this)->getElemPtr(t, n, x, y, z,
                                                                       checkBounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        ALWAYS_INLINE
        real_t readElem(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                        int line) const {
            const real_t* ep = getElemPtr(t, n, x, y, z);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem(std::cout, "readElem", t, n, x, y, z, e, line);
#endif
            return e;
        }

        // Write one element.
        ALWAYS_INLINE
        void writeElem(real_t val,
                       idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                       int line) {
            real_t* ep = getElemPtr(t, n, x, y, z);
            *ep = val;
#ifdef TRACE_MEM
            printElem(std::cout, "writeElem", t, n, x, y, z, val, line);
#endif
        }

        // Read one vector at vector offset t, nv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE
        real_vec_t readVecNorm(idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                               int line) const {
#ifdef TRACE_MEM
            std::cout << "readVecNorm(" << t "," << nv << "," << xv <<
                "," << yv << "," << zv << ")..." << std::endl;
#endif        
            const real_vec_t* p = getVecPtrNorm(t, nv, xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            real_vec_t v;
            v.loadFrom(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "readVec", t, nv, xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.read(p, line);
#endif
            return v;
        }

        // Write one vector at vector offset t, nv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE void
        writeVecNorm(const real_vec_t& v,
                     idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                     int line) {
            real_vec_t* p = getVecPtrNorm(t, nv, xv, yv, zv);
            __assume_aligned(p, CACHELINE_BYTES);
            v.storeTo(p);
#ifdef TRACE_MEM
            printVecNorm(std::cout, "writeVec", t, nv, xv, yv, zv, v, line);
#endif
#ifdef MODEL_CACHE
            cache_model.write(p, line);
#endif
        }

        // Prefetch one vector at vector offset t, nv, xv, yv, zv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        template <int level>
        ALWAYS_INLINE
        void prefetchVecNorm(idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                             int line) const {
#ifdef TRACE_MEM
            std::cout << "prefetchVecNorm<" << level << ">(" << t << "," <<
                nv << "," << xv << "," << yv << "," << zv << ")..." << std::endl;
#endif        
            const char* p = (const char*)getVecPtrNorm(t, nv, xv, yv, zv, false);
            __assume_aligned(p, CACHELINE_BYTES);
            _mm_prefetch (p, level);
#ifdef MODEL_CACHE
            cache_model.prefetch(p, level, line);
#endif
        }

        // Read one element.
        virtual real_t readElem_TNXYZ(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                                      int line) const {
            return readElem(t, n, x, y, z, line);
        }

        // Write one element.
        virtual void writeElem_TNXYZ(real_t val,
                                     idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,                               
                                     int line) {
            writeElem(val, t, n, x, y, z, line);
        }

        // Read one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual real_vec_t readVecNorm_TNXYZ(idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                             int line) const {
            return readVecNorm(t, nv, xv, yv, zv, line);
        }
        
        // Write one vector at *vector* offset.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void writeVecNorm_TNXYZ(const real_vec_t& val,
                                        idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                        int line) {
            writeVecNorm(val, t, nv, xv, yv, zv, line);
        }

        // Print one vector.
        void printVecNorm(std::ostream& os, const std::string& m,
                          idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                          const real_vec_t& v,
                          int line) const {
            printVecNorm_TNXYZ(t, nv, xv, yv, zv, v, line);
        }

        // Print one element.
        void printElem(std::ostream& os, const std::string& m,
                       idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                       real_t e,
                       int line) const {
            printElem_TNXYZ(t, n, xv, yv, zv, e, line);
        }
    };

}
#endif
