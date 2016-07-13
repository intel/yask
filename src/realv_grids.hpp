/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

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

// rounding macros for integer types.
#define CEIL_DIV(numer, denom) (((numer) + (denom) - 1) / (denom))
#define ROUND_UP(n, mult) (CEIL_DIV(n, mult) * (mult))

namespace yask {

    // Base class for real_vec_t grids.
    // Provides generic-grid support.
    class RealVecGridBase {
    protected:
        std::string _name;
        GenericGridBase<real_vec_t>* _gp;

    public:
        RealVecGridBase(std::string name, GenericGridBase<real_vec_t>* gp) :
            _name(name), _gp(gp) { }

        const std::string& get_name() { return _name; }
    
        // Initialize memory to a given value.
        virtual void set_same(real_t val) {
            real_vec_t rn;
            rn = val;               // broadcast.
            _gp->set_same(rn);
        }

        // Initialize memory to incrementing values based on val.
        virtual void set_diff(real_t val) {

            // make a real_vec_t pattern.
            real_vec_t rn;
            for (int i = 0; i < VLEN; i++)
                rn[i] = real_t(i * VLEN + 1) * val / VLEN;
        
            _gp->set_diff(rn);
        }

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
        idx_t compare(const RealVecGridBase& ref,
                      real_t epsilon = EPSILON,
                      int maxPrint = 20,
                      std::ostream& os = std::cerr) const {
            real_vec_t ev;
            ev = epsilon;           // broadcast to real_vec_t elements.
            return _gp->compare(ref._gp, ev, maxPrint, os);
        }

        // Direct access to data (dangerous!).
        real_vec_t* getRawData() {
            return _gp->getRawData();
        }
        const real_vec_t* getRawData() const {
            return _gp->getRawData();
        }
    };


    // A 3D (x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn> class RealVecGrid_XYZ :
        public RealVecGridBase {
    protected:

        // real sizes.
        idx_t _dx, _dy, _dz;
        idx_t _px, _py, _pz;

        // real_vec_t sizes.
        idx_t _dxv, _dyv, _dzv;
        idx_t _pxv, _pyv, _pzv;
    
        GenericGrid3d<real_vec_t, LayoutFn> _data;
    
    public:

        // Ctor.
        // Dimensions are real_t elements, not real_vecs.
        RealVecGrid_XYZ(idx_t dx, idx_t dy, idx_t dz,
                      idx_t px, idx_t py, idx_t pz,
                        const std::string& name,
                        std::ostream& msg_stream = std::cout) :
            RealVecGridBase(name, &_data),

            // Round up each dim to multiple of dim in real_vec_t.
            _dx(ROUND_UP(dx, VLEN_X)),
            _dy(ROUND_UP(dy, VLEN_Y)),
            _dz(ROUND_UP(dz, VLEN_Z)),
            _px(ROUND_UP(px, VLEN_X)),
            _py(ROUND_UP(py, VLEN_Y)),
            _pz(ROUND_UP(pz, VLEN_Z)),
                                
            // Determine number of real_vec_t's.
            _dxv(_dx / VLEN_X),
            _dyv(_dy / VLEN_Y),
            _dzv(_dz / VLEN_Z),
            _pxv(_px / VLEN_X),
            _pyv(_py / VLEN_Y),
            _pzv(_pz / VLEN_Z),

            // Alloc space for required number of real_vec_t's.
            _data(_dxv + 2*_pxv,
                  _dyv + 2*_pyv,
                  _dzv + 2*_pzv,
                  ALLOC_ALIGNMENT)
        {
            _data.print_info(name, msg_stream);

            // Should not be using grid w/o N dimension with folding in N.
            assert(VLEN_N == 1);
        }

        // Get parameters after round-up.
        inline idx_t get_dx() { return _dx; }
        inline idx_t get_dy() { return _dy; }
        inline idx_t get_dz() { return _dz; }
        inline idx_t get_px() { return _px; }
        inline idx_t get_py() { return _py; }
        inline idx_t get_pz() { return _pz; }

        // Get pointer to the real_vec_t at vector offset iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE const real_vec_t* getVecPtrNorm(idx_t iv, idx_t jv, idx_t kv,
                                                 bool checkBounds=true) const {

#ifdef TRACE_MEM
            std::cout << _name << "." << "RealVecGrid_XYZ::getVecPtrNorm(" <<
                iv << "," << jv << "," << kv << ")";
#endif
        
            // adjust for padding.
            iv += _pxv;
            jv += _pyv;
            kv += _pzv;

#ifdef TRACE_MEM
            if (checkBounds)
                std::cout << " => " << _data.get_index(iv, jv, kv);
            std::cout << std::endl << flush;
#endif
            return &_data(iv, jv, kv, checkBounds);
        }

        // Non-const version.
        ALWAYS_INLINE real_vec_t* getVecPtrNorm(idx_t iv, idx_t jv, idx_t kv,
                                           bool checkBounds=true) {

            const real_vec_t* vp =
                const_cast<const RealVecGrid_XYZ*>(this)->getVecPtrNorm(iv, jv, kv,
                                                                      checkBounds);
            return const_cast<real_vec_t*>(vp);
        }
    
        // Get a pointer to one real_t.
        ALWAYS_INLINE const real_t* getElemPtr(idx_t i, idx_t j, idx_t k,
                                             bool checkBounds=true) const {
#if 1
            // add padding before division to ensure negative indices work.
            idx_t ip = i + _px;
            idx_t jp = j + _py;
            idx_t kp = k + _pz;

            // normalize and remove padding.
            idx_t iv = ip / VLEN_X - _pxv;
            idx_t jv = jp / VLEN_Y - _pyv;
            idx_t kv = kp / VLEN_Z - _pzv;

            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(iv, jv, kv, checkBounds);

            // intra-vector element indices.
            idx_t ie = ip % VLEN_X;
            idx_t je = jp % VLEN_Y;
            idx_t ke = kp % VLEN_Z;
#else
            // normalize.
            idx_t iv = idiv<idx_t>(i, VLEN_X);
            idx_t jv = idiv<idx_t>(j, VLEN_Y);
            idx_t kv = idiv<idx_t>(k, VLEN_Z);
        
            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(iv, jv, kv, checkBounds);

            // intra-vector element indices.
            idx_t ie = imod<idx_t>(i, VLEN_X);
            idx_t je = imod<idx_t>(j, VLEN_Y);
            idx_t ke = imod<idx_t>(k, VLEN_Z);
#endif

            // Extract point from vector.
            return &(*vp)(0, ie, je, ke);
        }

        // non-const version.
        ALWAYS_INLINE real_t* getElemPtr(idx_t i, idx_t j, idx_t k,
                                       bool checkBounds=true) {
            const real_t* p = const_cast<const RealVecGrid_XYZ*>(this)->getElemPtr(i, j, k,
                                                                               checkBounds);
            return const_cast<real_t*>(p);
        }

        // Print one vector.
        void printVec(const std::string& m, idx_t iv, idx_t jv, idx_t kv, const real_vec_t& v,
                      int line) const {
            idx_t i = iv * VLEN_X;
            idx_t j = jv * VLEN_Y;
            idx_t k = kv * VLEN_Z;
            for (int k2 = 0; k2 < VLEN_Z; k2++) {
                for (int j2 = 0; j2 < VLEN_Y; j2++) {
                    for (int i2 = 0; i2 < VLEN_X; i2++) {
                        real_t e = v(0, i2, j2, k2);
                        real_t e2 = readElem(i+i2, j+j2, k+k2, line);

                        std::cout << m << ": " << _name << "[" <<
                            (i+i2) << ", " << (j+j2) << ", " << (k+k2) << "] = " << e;
                        if (line)
                            std::cout << " at line " << line;

                        // compare to per-element read.
                        if (e == e2)
                            std::cout << " (same as readElem())";
                        else
                            std::cout << " != " << e2 << " from readElem() <<<< ERROR";
                        std::cout << std::endl << std::flush;
                    }
                }
            }
        }

        // Print one element.
        void printElem(const std::string& m, idx_t i, idx_t j, idx_t k, real_t e,
                       int line) const {
            std::cout << m << ": " << _name << "[" <<
                i << ", " << j << ", " << k << "] = " << e;
            if (line)
                std::cout << " at line " << line;
            std::cout << std::endl << std::flush;
        }

        // Read one element.
        ALWAYS_INLINE real_t readElem(idx_t i, idx_t j, idx_t k,
                                    int line) const {
            const real_t* ep = getElemPtr(i, j, k);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem("readElem", i, j, k, e, line);
#endif
            return e;
        }

        // Write one element.
        ALWAYS_INLINE void writeElem(real_t val, idx_t i, idx_t j, idx_t k,
                                     int line) {
            real_t* ep = getElemPtr(i, j, k);
            *ep = val;
#ifdef TRACE_MEM
            printElem("writeElem", i, j, k, val, line);
#endif
        }

        // Read one vector at vector offset iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE const real_vec_t readVecNorm(idx_t iv, idx_t jv, idx_t kv,
                                              int line) const {
#ifdef TRACE_MEM
            std::cout << "readVecNorm(" << iv << "," << jv << "," << kv << ")..." << std::endl;
#endif        
            const real_vec_t* p = getVecPtrNorm(iv, jv, kv);
            __assume_aligned(p, CACHELINE_BYTES);
            real_vec_t v;
            v.loadFrom(p);
#ifdef TRACE_MEM
            printVec("readVec", iv, jv, kv, v, line);
#endif
#ifdef MODEL_CACHE
            cache.read(p, line);
#endif
            return v;
        }

        // Write one vector at vector offset nv, iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE void writeVecNorm(const real_vec_t& v, idx_t iv, idx_t jv, idx_t kv,
                                        int line) {
            real_vec_t* p = getVecPtrNorm(iv, jv, kv);
            __assume_aligned(p, CACHELINE_BYTES);
            v.storeTo(p);
#ifdef TRACE_MEM
            printVec("writeVec", iv, jv, kv, v, line);
#endif
#ifdef MODEL_CACHE
            cache.write(p, line);
#endif
        }

    };

    // A 4D (n, x, y, z) collection of real_vec_t elements.
    // Supports symmetric padding in each dimension.
    template <typename LayoutFn> class RealVecGrid_NXYZ :
        public RealVecGridBase {
    
    protected:

        // real sizes.
        idx_t _dn, _dx, _dy, _dz;
        idx_t _pn, _px, _py, _pz;

        // real_vec_t sizes.
        idx_t _dnv, _dxv, _dyv, _dzv;
        idx_t _pnv, _pxv, _pyv, _pzv;
    
        GenericGrid4d<real_vec_t, LayoutFn> _data;

    public:

        // Ctor.
        // Dimensions are real_t elements, not real_vecs.
        RealVecGrid_NXYZ(idx_t dn, idx_t dx, idx_t dy, idx_t dz,
                         idx_t pn, idx_t px, idx_t py, idx_t pz,
                         const std::string& name,
                         std::ostream& msg_stream = std::cout) :
            RealVecGridBase(name, &_data),

            // Round up each dim to multiple of dim in real_vec_t.
            _dn(ROUND_UP(dn, VLEN_N)),
            _dx(ROUND_UP(dx, VLEN_X)),
            _dy(ROUND_UP(dy, VLEN_Y)),
            _dz(ROUND_UP(dz, VLEN_Z)),
            _pn(ROUND_UP(pn, VLEN_N)),
            _px(ROUND_UP(px, VLEN_X)),
            _py(ROUND_UP(py, VLEN_Y)),
            _pz(ROUND_UP(pz, VLEN_Z)),

            // Determine number of real_vec_t's.
            _dnv(_dn / VLEN_N),
            _dxv(_dx / VLEN_X),
            _dyv(_dy / VLEN_Y),
            _dzv(_dz / VLEN_Z),
            _pnv(_pn / VLEN_N),
            _pxv(_px / VLEN_X),
            _pyv(_py / VLEN_Y),
            _pzv(_pz / VLEN_Z),

            // Alloc space for required number of real_vec_t's.
            _data(_dnv + 2*_pnv,
                  _dxv + 2*_pxv,
                  _dyv + 2*_pyv,
                  _dzv + 2*_pzv,
                  ALLOC_ALIGNMENT)
        {
            _data.print_info(name, msg_stream);
        }

        // Get pointer to the real_vec_t at vector offset nv, iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE const real_vec_t* getVecPtrNorm(idx_t nv, idx_t iv, idx_t jv, idx_t kv,
                                                 bool checkBounds=true) const {

#ifdef TRACE_MEM
            std::cout << _name << "." << "RealVecGrid_NXYZ::getVecPtrNorm(" <<
                nv << "," << iv << "," << jv << "," << kv << ")";
#endif
        
            // adjust for padding.
            nv += _pnv;
            iv += _pxv;
            jv += _pyv;
            kv += _pzv;

#ifdef TRACE_MEM
            if (checkBounds)
                std::cout << " => " << _data.get_index(nv, iv, jv, kv);
            std::cout << std::endl << flush;
#endif
            return &_data(nv, iv, jv, kv, checkBounds);
        }

        // Non-const version.
        ALWAYS_INLINE real_vec_t* getVecPtrNorm(idx_t nv, idx_t iv, idx_t jv, idx_t kv,
                                           bool checkBounds=true) {

            const real_vec_t* vp =
                const_cast<const RealVecGrid_NXYZ*>(this)->getVecPtrNorm(nv, iv, jv, kv,
                                                                       checkBounds);
            return const_cast<real_vec_t*>(vp);
        }
    
        // Get a pointer to one real_t.
        ALWAYS_INLINE const real_t* getElemPtr(idx_t n, idx_t i, idx_t j, idx_t k,
                                             bool checkBounds=true) const {
#if 1
            // add padding before division to ensure negative indices work.
            idx_t np = n + _pn;
            idx_t ip = i + _px;
            idx_t jp = j + _py;
            idx_t kp = k + _pz;

            // normalize and remove padding.
            idx_t nv = np / VLEN_N - _pnv;
            idx_t iv = ip / VLEN_X - _pxv;
            idx_t jv = jp / VLEN_Y - _pyv;
            idx_t kv = kp / VLEN_Z - _pzv;

            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(nv, iv, jv, kv, checkBounds);

            // intra-vector element indices.
            // use values with padding in numerator to avoid negative indices.
            idx_t ne = np % VLEN_N;
            idx_t ie = ip % VLEN_X;
            idx_t je = jp % VLEN_Y;
            idx_t ke = kp % VLEN_Z;
#else
            // normalize.
            idx_t nv = idiv<idx_t>(n, VLEN_N);
            idx_t iv = idiv<idx_t>(i, VLEN_X);
            idx_t jv = idiv<idx_t>(j, VLEN_Y);
            idx_t kv = idiv<idx_t>(k, VLEN_Z);
        
            // Get vector.
            const real_vec_t* vp = getVecPtrNorm(nv, iv, jv, kv, checkBounds);

            // intra-vector element indices.
            idx_t ne = imod<idx_t>(n, VLEN_N);
            idx_t ie = imod<idx_t>(i, VLEN_X);
            idx_t je = imod<idx_t>(j, VLEN_Y);
            idx_t ke = imod<idx_t>(k, VLEN_Z);
#endif
        
            // Extract point from vector.
            return &(*vp)(ne, ie, je, ke);
        }

        // non-const version.
        ALWAYS_INLINE real_t* getElemPtr(idx_t n, idx_t i, idx_t j, idx_t k,
                                       bool checkBounds=true) {
            const real_t* p = const_cast<const RealVecGrid_NXYZ*>(this)->getElemPtr(n, i, j, k,
                                                                                checkBounds);
            return const_cast<real_t*>(p);
        }

        // Print one vector.
        void printVec(const std::string& m, idx_t nv, idx_t iv, idx_t jv, idx_t kv, const real_vec_t& v,
                      int line) const {
            idx_t n = nv * VLEN_N;
            idx_t i = iv * VLEN_X;
            idx_t j = jv * VLEN_Y;
            idx_t k = kv * VLEN_Z;
            for (int k2 = 0; k2 < VLEN_Z; k2++) {
                for (int j2 = 0; j2 < VLEN_Y; j2++) {
                    for (int i2 = 0; i2 < VLEN_X; i2++) {
                        for (int n2 = 0; n2 < VLEN_N; n2++) {
                            real_t e = v(n2, i2, j2, k2);
#ifdef CHECK_VEC_ELEMS
                            real_t e2 = readElem(n+n2, i+i2, j+j2, k+k2, line);
#endif

                            std::cout << m << ": " << _name << "[" << (n+n2) << ", " <<
                                (i+i2) << ", " << (j+j2) << ", " << (k+k2) << "] = " << e;
                            if (line)
                                std::cout << " at line " << line;

#ifdef CHECK_VEC_ELEMS
                            // compare to per-element read.
                            if (e == e2)
                                std::cout << " (same as readElem())";
                            else
                                std::cout << " != " << e2 << " from readElem() <<<< ERROR";
#endif
                            std::cout << std::endl << std::flush;
                        }
                    }
                }
            }
        }

        // Print one element.
        void printElem(const std::string& m, idx_t n, idx_t i, idx_t j, idx_t k, real_t e,
                       int line) const {
            std::cout << m << ": " << _name << "[" <<
                n << ", " << i << ", " << j << ", " << k << "] = " << e;
            if (line)
                std::cout << " at line " << line;
            std::cout << std::endl << std::flush;
        }

        // Read one element.
        ALWAYS_INLINE real_t readElem(idx_t n, idx_t i, idx_t j, idx_t k,
                                    int line) const {
            const real_t* ep = getElemPtr(n, i, j, k);
            real_t e = *ep;
#ifdef TRACE_MEM
            printElem("readElem", n, i, j, k, e, line);
#endif
            return e;
        }

        // Write one element.
        ALWAYS_INLINE void writeElem(real_t val, idx_t n, idx_t i, idx_t j, idx_t k,
                                     int line) {
            real_t* ep = getElemPtr(n, i, j, k);
            *ep = val;
#ifdef TRACE_MEM
            printElem("writeElem", n, i, j, k, val, line);
#endif
        }

        // Read one vector at vector offset nv, iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE const real_vec_t readVecNorm(idx_t nv, idx_t iv, idx_t jv, idx_t kv,
                                              int line) const {
#ifdef TRACE_MEM
            std::cout << "readVecNorm(" << nv << "," << iv << "," << jv << "," << kv << ")..." << std::endl;
#endif        
            const real_vec_t* p = getVecPtrNorm(nv, iv, jv, kv);
            __assume_aligned(p, CACHELINE_BYTES);
            real_vec_t v;
            v.loadFrom(p);
#ifdef TRACE_MEM
            printVec("readVec", nv, iv, jv, kv, v, line);
#endif
#ifdef MODEL_CACHE
            cache.read(p, line);
#endif
            return v;
        }

        // Write one vector at vector offset nv, iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE void writeVecNorm(const real_vec_t& v, idx_t nv, idx_t iv, idx_t jv, idx_t kv,
                                        int line) {
            real_vec_t* p = getVecPtrNorm(nv, iv, jv, kv);
            __assume_aligned(p, CACHELINE_BYTES);
            v.storeTo(p);
#ifdef TRACE_MEM
            printVec("writeVec", nv, iv, jv, kv, v, line);
#endif
#ifdef MODEL_CACHE
            cache.write(p, line);
#endif
        }

    };

    // A 4D (t, x, y, z) collection of real_vec_t elements, but any value of 't'
    // is divided by CPTS_T and wrapped to TIME_DIM_SIZE indices.
    // Supports symmetric padding in each spatial dimension.
    template <typename LayoutFn> class RealVecGrid_TXYZ :
        public RealVecGrid_NXYZ<LayoutFn>  {
    
    public:

        // Ctor.
        RealVecGrid_TXYZ(idx_t dx, idx_t dy, idx_t dz,
                         idx_t px, idx_t py, idx_t pz,
                         const std::string& name,
                         std::ostream& msg_stream = std::cout) :
            RealVecGrid_NXYZ<LayoutFn>(TIME_DIM_SIZE, dx, dy, dz,
                                       0, px, py, pz,
                                       name, msg_stream)
        {
            if (VLEN_N > 1) {
                std::cerr << "Sorry, vectorizing in N dimension not yet supported." << std::endl;
                exit(1);
            }
        }

        // Get correct index based on time t.
        ALWAYS_INLINE idx_t getMatIndex(idx_t t) const {

#if ALLOW_NEG_TIME
            // Time t must be multiple of CPTS_T.
            // Use imod & idiv to allow t to be negative.
            assert(imod<idx_t>(t, CPTS_T) == 0);
            idx_t t_idx = idiv<idx_t>(t, CPTS_T);

            // Index wraps in TIME_DIM_SIZE.
            // Examples if TIME_DIM_SIZE == 2:
            // t_idx => return value.
            // -2 => 0.
            // -1 => 1.
            //  0 => 0.
            //  1 => 1.

            // Use imod to allow t to be negative.
            return imod<idx_t>(t_idx, TIME_DIM_SIZE);
#else
            // version that doesn't allow negative time.
            assert(t >= 0);
            assert(t % CPTS_T == 0);
            idx_t t_idx = t / idx_t(CPTS_T);
            return t_idx % idx_t(TIME_DIM_SIZE);
#endif
        }

        // Read one element.
        ALWAYS_INLINE real_t readElem(idx_t t, idx_t i, idx_t j, idx_t k,
                                    int line) const {
            idx_t n = getMatIndex(t);
            return RealVecGrid_NXYZ<LayoutFn>::readElem(n, i, j, k, line);
        }

        // Write one element of the grid.
        ALWAYS_INLINE void writeElem(real_t val, idx_t t, idx_t i, idx_t j, idx_t k,
                                     int line) {
            idx_t n = getMatIndex(t);
            RealVecGrid_NXYZ<LayoutFn>::writeElem(val, n, i, j, k, line);
        }

        // Read one vector at t and vector offset iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE const real_vec_t readVecNorm(idx_t t, idx_t iv, idx_t jv, idx_t kv,
                                              int line) const {
            idx_t n = getMatIndex(t);
            return RealVecGrid_NXYZ<LayoutFn>::readVecNorm(n, iv, jv, kv, line);
        }

        // Write one vector at t and vector offset iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE void writeVecNorm(const real_vec_t& v, idx_t t, idx_t iv, idx_t jv, idx_t kv,
                                        int line) {
            idx_t n = getMatIndex(t);
            RealVecGrid_NXYZ<LayoutFn>::writeVecNorm(v, n, iv, jv, kv, line);
        }

        // Get pointer to the real at t and offset i, j, k.
        ALWAYS_INLINE const real_t* getElemPtr(idx_t t, idx_t i, idx_t j, idx_t k,
                                             int line) const {
            idx_t n = getMatIndex(t);
            return RealVecGrid_NXYZ<LayoutFn>::getElemPtr(n, i, j, k, false);
        }
        ALWAYS_INLINE real_t* getElemPtr(idx_t t, idx_t i, idx_t j, idx_t k,
                                       int line) {
            idx_t n = getMatIndex(t);
            return RealVecGrid_NXYZ<LayoutFn>::getElemPtr(n, i, j, k, false);
        }

        // Get pointer to the real_vec_t at t and vector offset iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE const real_vec_t* getVecPtrNorm(idx_t t, idx_t iv, idx_t jv, idx_t kv,
                                                 int line) const {
            idx_t n = getMatIndex(t);
            return RealVecGrid_NXYZ<LayoutFn>::getVecPtrNorm(n, iv, jv, kv, false);
        }
        ALWAYS_INLINE real_vec_t* getVecPtrNorm(idx_t t, idx_t iv, idx_t jv, idx_t kv,
                                           int line) {
            idx_t n = getMatIndex(t);
            return RealVecGrid_NXYZ<LayoutFn>::getVecPtrNorm(n, iv, jv, kv, false);
        }
    };

    // A 5D (t, n, x, y, z) collection of real_vec_t elements, but any value of 't'
    // is divided by CPTS_T and wrapped to TIME_DIM_SIZE indices.
    // Supports symmetric padding in each spatial dimension.
    template <typename LayoutFn> class RealVecGrid_TNXYZ :
        public RealVecGrid_NXYZ<LayoutFn> {
    
    protected:
        idx_t _dn;

    public:

        // Ctor.
        RealVecGrid_TNXYZ(idx_t dn, idx_t dx, idx_t dy, idx_t dz,
                          idx_t pn, idx_t px, idx_t py, idx_t pz,
                          const std::string& name,
                          std::ostream& msg_stream = std::cout) :
            RealVecGrid_NXYZ<LayoutFn>(TIME_DIM_SIZE * dn, dx, dy, dz,
                                       pn, px, py, pz,
                                       name, msg_stream),
            _dn(dn)
        {
            if (VLEN_N > 1) {
                std::cerr << "Sorry, vectorizing in N dimension not yet supported." << std::endl;
                exit(1);
            }
        }

        // Get correct index based on t & n.
        ALWAYS_INLINE idx_t getMatIndex(idx_t t, idx_t n) const {

#if ALLOW_NEG_TIME
            // Time t must be multiple of CPTS_T.
            // Use imod & idiv to allow t to be negative.
            assert(imod<idx_t>(t, CPTS_T) == 0);
            idx_t t_idx = idiv<idx_t>(t, CPTS_T);

            // Index wraps in TIME_DIM_SIZE.
            // Examples if TIME_DIM_SIZE == 2:
            // t_idx => t_idx2.
            // -2 => 0.
            // -1 => 1.
            //  0 => 0.
            //  1 => 1.

            // Use imod to allow t to be negative.
            idx_t t_idx2 = imod<idx_t>(t_idx, TIME_DIM_SIZE);
#else
            // version that doesn't allow negative time.
            assert(t >= 0);
            assert(t % CPTS_T == 0);
            idx_t t_idx = t / idx_t(CPTS_T);
            idx_t t_idx2 = t_idx % idx_t(TIME_DIM_SIZE);
#endif        

            // Layout t_idx2 and n onto one dimension.
            return LAYOUT_21(n, t_idx2, _dn, TIME_DIM_SIZE);
        }

        // Read one element.
        ALWAYS_INLINE real_t readElem(idx_t t, idx_t n,
                                    idx_t i, idx_t j, idx_t k,
                                    int line) const {
            idx_t n2 = getMatIndex(t, n);
            return RealVecGrid_NXYZ<LayoutFn>::readElem(n2, i, j, k, line);
        }

        // Write one element of the grid.
        ALWAYS_INLINE void writeElem(real_t val, idx_t t, idx_t n,
                                     idx_t i, idx_t j, idx_t k,
                                     int line) {
            idx_t n2 = getMatIndex(t, n);
            RealVecGrid_NXYZ<LayoutFn>::writeElem(val, n2, i, j, k, line);
        }

        // Read one vector at t and vector offset nv, iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE const real_vec_t readVecNorm(idx_t t, idx_t nv,
                                              idx_t iv, idx_t jv, idx_t kv,
                                              int line) const {
            idx_t n2 = getMatIndex(t, nv);
            return RealVecGrid_NXYZ<LayoutFn>::readVecNorm(n2, iv, jv, kv, line);
        }

        // Write one vector at t and vector offset nv, iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE void writeVecNorm(const real_vec_t& v, idx_t t, idx_t nv,
                                        idx_t iv, idx_t jv, idx_t kv,
                                        int line) {
            idx_t n2 = getMatIndex(t, nv);
            RealVecGrid_NXYZ<LayoutFn>::writeVecNorm(v, n2, iv, jv, kv, line);
        }

        // Get pointer to the real at t and offset n, i, j, k.
        ALWAYS_INLINE const real_t* getElemPtr(idx_t t, idx_t n, idx_t i, idx_t j, idx_t k,
                                             int line) const {
            idx_t n2 = getMatIndex(t, n);
            return RealVecGrid_NXYZ<LayoutFn>::getElemPtr(n2, i, j, k, false);
        }
        ALWAYS_INLINE real_t* getElemPtr(idx_t t, idx_t n, idx_t i, idx_t j, idx_t k,
                                       int line) {
            idx_t n2 = getMatIndex(t, n);
            return RealVecGrid_NXYZ<LayoutFn>::getElemPtr(n2, i, j, k, false);
        }

        // Get pointer to the real_vec_t at t and vector offset nv, iv, jv, kv.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        ALWAYS_INLINE const real_vec_t* getVecPtrNorm(idx_t t, idx_t nv,
                                                 idx_t iv, idx_t jv, idx_t kv,
                                                 int line) const {
            idx_t n2 = getMatIndex(t, nv);
            return RealVecGrid_NXYZ<LayoutFn>::getVecPtrNorm(n2, iv, jv, kv, false);
        }
        ALWAYS_INLINE real_vec_t* getVecPtrNorm(idx_t t, idx_t nv,
                                           idx_t iv, idx_t jv, idx_t kv,
                                           int line) {
            idx_t n2 = getMatIndex(t, nv);
            return RealVecGrid_NXYZ<LayoutFn>::getVecPtrNorm(n2, iv, jv, kv, false);
        }
    };

}
#endif
