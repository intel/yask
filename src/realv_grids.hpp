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

#ifndef REALV_GRIDS
#define REALV_GRIDS

#include "generic_grids.hpp"

// rounding macros.
#define CEIL_DIV(numer, denom) (((numer) + (denom) - 1) / (denom))
#define ROUND_UP(n, mult) (CEIL_DIV(n, mult) * (mult))

// Base class for realv grids.
// Provides generic-grid support.
class RealvGridBase {
protected:
    string _name;
    GenericGridBase<realv>* _gp;

public:
    RealvGridBase(string name, GenericGridBase<realv>* gp) :
        _name(name), _gp(gp) { }

    const string& get_name() { return _name; }
    
    // Initialize memory to a given value.
    virtual void set_same(REAL val) {
        realv rn;
        rn = val;               // broadcast.
        _gp->set_same(rn);
    }

    // Initialize memory to incrementing values based on val.
    virtual void set_diff(REAL val) {

        // make a realv pattern.
        realv rn;
        for (int i = 0; i < VLEN; i++)
            rn[i] = REAL(i * VLEN + 1) * val / VLEN;
        
        _gp->set_diff(rn);
    }

    // Get number of elements with padding.
    inline idx_t get_num_elems() const {
        return _gp->get_num_elems();
    }

    // Get size in bytes.
    inline idx_t get_num_bytes() const {
        return _gp->get_num_bytes();
    }

    
    // Check for equality.
    // Return number of mismatches greater than epsilon.
    idx_t compare(const RealvGridBase& ref,
                   REAL epsilon = EPSILON,
                   int maxPrint = 20,
                   std::ostream& os = std::cerr) const {
        realv ev;
        ev = epsilon;           // broadcast to realv elements.
        return _gp->compare(ref._gp, ev, maxPrint, os);
    }
};


// A 3D (x, y, z) collection of realv elements.
// Supports symmetric padding in each dimension.
template <typename Mapfn> class RealvGrid_XYZ : public RealvGridBase {
protected:

    // real sizes.
    idx_t _dx, _dy, _dz;
    idx_t _px, _py, _pz;

    // realv sizes.
    idx_t _dxv, _dyv, _dzv;
    idx_t _pxv, _pyv, _pzv;
    
    GenericGrid3d<realv, Mapfn> _data;
    
public:

    // Ctor.
    // Dimensions are REAL elements, not realvs.
    RealvGrid_XYZ(idx_t dx, idx_t dy, idx_t dz,
                  idx_t px, idx_t py, idx_t pz,
                  const string& name) :

        // Round up each dim to size of realv.
        _dx(ROUND_UP(dx, VLEN_X)),
        _dy(ROUND_UP(dy, VLEN_Y)),
        _dz(ROUND_UP(dz, VLEN_Z)),
        _px(ROUND_UP(px, VLEN_X)),
        _py(ROUND_UP(py, VLEN_Y)),
        _pz(ROUND_UP(pz, VLEN_Z)),
                                
        // Determine number of realv's.
        _dxv(_dx / VLEN_X),
        _dyv(_dy / VLEN_Y),
        _dzv(_dz / VLEN_Z),
        _pxv(_px / VLEN_X),
        _pyv(_py / VLEN_Y),
        _pzv(_pz / VLEN_Z),

        // Alloc space for required number of realv's.
        _data(_dxv + 2*_pxv,
              _dyv + 2*_pyv,
              _dzv + 2*_pzv,
              ALLOC_ALIGNMENT),
        RealvGridBase(name, &_data)
    {
        _data.print_info(name);

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

    // Get pointer to the realv at vector offset iv, jv, kv.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    ALWAYS_INLINE const realv* getVecPtrNorm(idx_t iv, idx_t jv, idx_t kv,
                                             bool checkBounds=true) const {

#ifdef TRACE_MEM
        cout << _name << "." << "RealvGrid_XYZ::getVecPtrNorm(" <<
             iv << "," << jv << "," << kv << ")";
#endif
        
        // adjust for padding.
        iv += _pxv;
        jv += _pyv;
        kv += _pzv;

#ifdef TRACE_MEM
        if (checkBounds)
            cout << " => " << _data.get_index(iv, jv, kv);
        cout << endl << flush;
#endif
        return &_data(iv, jv, kv, checkBounds);
    }

    // Non-const version.
    ALWAYS_INLINE realv* getVecPtrNorm(idx_t iv, idx_t jv, idx_t kv,
                                       bool checkBounds=true) {

        const realv* vp =
            const_cast<const RealvGrid_XYZ*>(this)->getVecPtrNorm(iv, jv, kv,
                                                                   checkBounds);
        return const_cast<realv*>(vp);
    }
    
    // Get a pointer to one REAL.
    ALWAYS_INLINE const REAL* getElemPtr(idx_t i, idx_t j, idx_t k,
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
        const realv* vp = getVecPtrNorm(iv, jv, kv, checkBounds);

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
        const realv* vp = getVecPtrNorm(iv, jv, kv, checkBounds);

        // intra-vector element indices.
        idx_t ie = imod<idx_t>(i, VLEN_X);
        idx_t je = imod<idx_t>(j, VLEN_Y);
        idx_t ke = imod<idx_t>(k, VLEN_Z);
#endif

        // Extract point from vector.
        return &(*vp)(0, ie, je, ke);
    }

    // non-const version.
    ALWAYS_INLINE REAL* getElemPtr(idx_t i, idx_t j, idx_t k,
                                  bool checkBounds=true) {
        const REAL* p = const_cast<const RealvGrid_XYZ*>(this)->getElemPtr(i, j, k,
                                                                         checkBounds);
        return const_cast<REAL*>(p);
    }

    // Print one vector.
    void printVec(const string& m, idx_t iv, idx_t jv, idx_t kv, const realv& v,
                  int line) const {
        idx_t i = iv * VLEN_X;
        idx_t j = jv * VLEN_Y;
        idx_t k = kv * VLEN_Z;
        for (int k2 = 0; k2 < VLEN_Z; k2++) {
            for (int j2 = 0; j2 < VLEN_Y; j2++) {
                for (int i2 = 0; i2 < VLEN_X; i2++) {
                    REAL e = v(0, i2, j2, k2);
                    REAL e2 = readElem(i+i2, j+j2, k+k2, line);

                    cout << m << ": " << _name << "[" <<
                        (i+i2) << ", " << (j+j2) << ", " << (k+k2) << "] = " << e;
                    if (line)
                        cout << " at line " << line;

                    // compare to per-element read.
                    if (e == e2)
                        cout << " (same as readElem())";
                    else
                        cout << " != " << e2 << " from readElem() <<<< ERROR";
                    cout << endl << flush;
                }
            }
        }
    }

    // Print one element.
    void printElem(const string& m, idx_t i, idx_t j, idx_t k, REAL e,
                   int line) const {
        cout << m << ": " << _name << "[" <<
            i << ", " << j << ", " << k << "] = " << e;
        if (line)
            cout << " at line " << line;
        cout << endl << flush;
    }

    // Read one element.
    ALWAYS_INLINE REAL readElem(idx_t i, idx_t j, idx_t k,
                               int line) const {
        const REAL* ep = getElemPtr(i, j, k);
        REAL e = *ep;
#ifdef TRACE_MEM
        printElem("readElem", i, j, k, e, line);
#endif
        return e;
    }

    // Write one element.
    ALWAYS_INLINE void writeElem(REAL val, idx_t i, idx_t j, idx_t k,
                                int line) {
        REAL* ep = getElemPtr(i, j, k);
        *ep = val;
#ifdef TRACE_MEM
        printElem("writeElem", i, j, k, val, line);
#endif
    }

    // Read one vector at vector offset iv, jv, kv.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    ALWAYS_INLINE const realv readVecNorm(idx_t iv, idx_t jv, idx_t kv,
                                          int line) const {
#ifdef TRACE_MEM
        cout << "readVecNorm(" << iv << "," << jv << "," << kv << ")..." << endl;
#endif        
        const realv* p = getVecPtrNorm(iv, jv, kv);
        __assume_aligned(p, CACHELINE_BYTES);
        realv v;
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
    ALWAYS_INLINE void writeVecNorm(const realv& v, idx_t iv, idx_t jv, idx_t kv,
                                    int line) {
        realv* p = getVecPtrNorm(iv, jv, kv);
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

// A 4D (n, x, y, z) collection of realv elements.
// Supports symmetric padding in each dimension.
template <typename Mapfn> class RealvGrid_NXYZ : public RealvGridBase {
    
protected:

    // real sizes.
    idx_t _dn, _dx, _dy, _dz;
    idx_t _pn, _px, _py, _pz;

    // realv sizes.
    idx_t _dnv, _dxv, _dyv, _dzv;
    idx_t _pnv, _pxv, _pyv, _pzv;
    
    GenericGrid4d<realv, Mapfn> _data;

public:

    // Ctor.
    // Dimensions are REAL elements, not realvs.
    RealvGrid_NXYZ(idx_t dn, idx_t dx, idx_t dy, idx_t dz,
                   idx_t pn, idx_t px, idx_t py, idx_t pz,
                   const string& name) :

        // Round up each dim to multiple of realv.
        _dn(ROUND_UP(dn, VLEN_N)),
        _dx(ROUND_UP(dx, VLEN_X)),
        _dy(ROUND_UP(dy, VLEN_Y)),
        _dz(ROUND_UP(dz, VLEN_Z)),
        _pn(ROUND_UP(pn, VLEN_N)),
        _px(ROUND_UP(px, VLEN_X)),
        _py(ROUND_UP(py, VLEN_Y)),
        _pz(ROUND_UP(pz, VLEN_Z)),

        // Determine number of realv's.
        _dnv(_dn / VLEN_N),
        _dxv(_dx / VLEN_X),
        _dyv(_dy / VLEN_Y),
        _dzv(_dz / VLEN_Z),
        _pnv(_pn / VLEN_N),
        _pxv(_px / VLEN_X),
        _pyv(_py / VLEN_Y),
        _pzv(_pz / VLEN_Z),

        // Alloc space for required number of realv's.
        _data(_dnv + 2*_pnv,
              _dxv + 2*_pxv,
              _dyv + 2*_pyv,
              _dzv + 2*_pzv,
              ALLOC_ALIGNMENT),
        RealvGridBase(name, &_data)
    {
        _data.print_info(name);
    }

    // Get pointer to the realv at vector offset nv, iv, jv, kv.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    ALWAYS_INLINE const realv* getVecPtrNorm(idx_t nv, idx_t iv, idx_t jv, idx_t kv,
                                             bool checkBounds=true) const {

#ifdef TRACE_MEM
        cout << _name << "." << "RealvGrid_NXYZ::getVecPtrNorm(" <<
            nv << "," << iv << "," << jv << "," << kv << ")";
#endif
        
        // adjust for padding.
        nv += _pnv;
        iv += _pxv;
        jv += _pyv;
        kv += _pzv;

#ifdef TRACE_MEM
        if (checkBounds)
            cout << " => " << _data.get_index(nv, iv, jv, kv);
        cout << endl << flush;
#endif
        return &_data(nv, iv, jv, kv, checkBounds);
    }

    // Non-const version.
    ALWAYS_INLINE realv* getVecPtrNorm(idx_t nv, idx_t iv, idx_t jv, idx_t kv,
                                       bool checkBounds=true) {

        const realv* vp =
            const_cast<const RealvGrid_NXYZ*>(this)->getVecPtrNorm(nv, iv, jv, kv,
                                                                   checkBounds);
        return const_cast<realv*>(vp);
    }
    
    // Get a pointer to one REAL.
    ALWAYS_INLINE const REAL* getElemPtr(idx_t n, idx_t i, idx_t j, idx_t k,
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
        const realv* vp = getVecPtrNorm(nv, iv, jv, kv, checkBounds);

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
        const realv* vp = getVecPtrNorm(nv, iv, jv, kv, checkBounds);

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
    ALWAYS_INLINE REAL* getElemPtr(idx_t n, idx_t i, idx_t j, idx_t k,
                                  bool checkBounds=true) {
        const REAL* p = const_cast<const RealvGrid_NXYZ*>(this)->getElemPtr(n, i, j, k,
                                                                         checkBounds);
        return const_cast<REAL*>(p);
    }

    // Print one vector.
    void printVec(const string& m, idx_t nv, idx_t iv, idx_t jv, idx_t kv, const realv& v,
                  int line) const {
        idx_t n = nv * VLEN_N;
        idx_t i = iv * VLEN_X;
        idx_t j = jv * VLEN_Y;
        idx_t k = kv * VLEN_Z;
        for (int k2 = 0; k2 < VLEN_Z; k2++) {
            for (int j2 = 0; j2 < VLEN_Y; j2++) {
                for (int i2 = 0; i2 < VLEN_X; i2++) {
                    for (int n2 = 0; n2 < VLEN_N; n2++) {
                        REAL e = v(n2, i2, j2, k2);
#ifdef CHECK_VEC_ELEMS
                        REAL e2 = readElem(n+n2, i+i2, j+j2, k+k2, line);
#endif

                        cout << m << ": " << _name << "[" << (n+n2) << ", " <<
                            (i+i2) << ", " << (j+j2) << ", " << (k+k2) << "] = " << e;
                        if (line)
                            cout << " at line " << line;

#ifdef CHECK_VEC_ELEMS
                        // compare to per-element read.
                        if (e == e2)
                            cout << " (same as readElem())";
                        else
                            cout << " != " << e2 << " from readElem() <<<< ERROR";
#endif
                        cout << endl << flush;
                    }
                }
            }
        }
    }

    // Print one element.
    void printElem(const string& m, idx_t n, idx_t i, idx_t j, idx_t k, REAL e,
                   int line) const {
        cout << m << ": " << _name << "[" <<
            n << ", " << i << ", " << j << ", " << k << "] = " << e;
        if (line)
            cout << " at line " << line;
        cout << endl << flush;
    }

    // Read one element.
    ALWAYS_INLINE REAL readElem(idx_t n, idx_t i, idx_t j, idx_t k,
                               int line) const {
        const REAL* ep = getElemPtr(n, i, j, k);
        REAL e = *ep;
#ifdef TRACE_MEM
        printElem("readElem", n, i, j, k, e, line);
#endif
        return e;
    }

    // Write one element.
    ALWAYS_INLINE void writeElem(REAL val, idx_t n, idx_t i, idx_t j, idx_t k,
                               int line) {
        REAL* ep = getElemPtr(n, i, j, k);
        *ep = val;
#ifdef TRACE_MEM
        printElem("writeElem", n, i, j, k, val, line);
#endif
    }

    // Read one vector at vector offset nv, iv, jv, kv.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    ALWAYS_INLINE const realv readVecNorm(idx_t nv, idx_t iv, idx_t jv, idx_t kv,
                                          int line) const {
#ifdef TRACE_MEM
        cout << "readVecNorm(" << nv << "," << iv << "," << jv << "," << kv << ")..." << endl;
#endif        
        const realv* p = getVecPtrNorm(nv, iv, jv, kv);
        __assume_aligned(p, CACHELINE_BYTES);
        realv v;
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
    ALWAYS_INLINE void writeVecNorm(const realv& v, idx_t nv, idx_t iv, idx_t jv, idx_t kv,
                                    int line) {
        realv* p = getVecPtrNorm(nv, iv, jv, kv);
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

// A 4D (t, x, y, z) collection of realv elements, but any value of 't'
// is divided by CPTS_T and mapped to TIME_DIM_SIZE indices.
// Supports symmetric padding in each spatial dimension.
template <typename Mapfn> class RealvGrid_TXYZ : public RealvGrid_NXYZ<Mapfn>  {
    
public:

    // Ctor.
    RealvGrid_TXYZ(idx_t dx, idx_t dy, idx_t dz,
                   idx_t px, idx_t py, idx_t pz,
                   const string& name) :
        RealvGrid_NXYZ<Mapfn>(TIME_DIM_SIZE, dx, dy, dz,
                              0, px, py, pz,
                              name)
    {
        if (VLEN_N > 1) {
            cerr << "Sorry, vectorizing in N dimension not yet supported." << endl;
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
    ALWAYS_INLINE REAL readElem(idx_t t, idx_t i, idx_t j, idx_t k,
                               int line) const {
        idx_t n = getMatIndex(t);
        return RealvGrid_NXYZ<Mapfn>::readElem(n, i, j, k, line);
    }

    // Write one element of the grid.
    ALWAYS_INLINE void writeElem(REAL val, idx_t t, idx_t i, idx_t j, idx_t k,
                                int line) {
        idx_t n = getMatIndex(t);
        RealvGrid_NXYZ<Mapfn>::writeElem(val, n, i, j, k, line);
    }

    // Read one vector at t and vector offset iv, jv, kv.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    ALWAYS_INLINE const realv readVecNorm(idx_t t, idx_t iv, idx_t jv, idx_t kv,
                                          int line) const {
        idx_t n = getMatIndex(t);
        return RealvGrid_NXYZ<Mapfn>::readVecNorm(n, iv, jv, kv, line);
    }

    // Write one vector at t and vector offset iv, jv, kv.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    ALWAYS_INLINE void writeVecNorm(const realv& v, idx_t t, idx_t iv, idx_t jv, idx_t kv,
                                    int line) {
        idx_t n = getMatIndex(t);
        RealvGrid_NXYZ<Mapfn>::writeVecNorm(v, n, iv, jv, kv, line);
    }

    // Get pointer to the real at t and offset i, j, k.
    ALWAYS_INLINE const REAL* getElemPtr(idx_t t, idx_t i, idx_t j, idx_t k,
                                          int line) const {
        idx_t n = getMatIndex(t);
        return RealvGrid_NXYZ<Mapfn>::getElemPtr(n, i, j, k, false);
    }
    ALWAYS_INLINE REAL* getElemPtr(idx_t t, idx_t i, idx_t j, idx_t k,
                                       int line) {
        idx_t n = getMatIndex(t);
        return RealvGrid_NXYZ<Mapfn>::getElemPtr(n, i, j, k, false);
    }

    // Get pointer to the realv at t and vector offset iv, jv, kv.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    ALWAYS_INLINE const realv* getVecPtrNorm(idx_t t, idx_t iv, idx_t jv, idx_t kv,
                                             int line) const {
        idx_t n = getMatIndex(t);
        return RealvGrid_NXYZ<Mapfn>::getVecPtrNorm(n, iv, jv, kv, false);
    }
    ALWAYS_INLINE realv* getVecPtrNorm(idx_t t, idx_t iv, idx_t jv, idx_t kv,
                                       int line) {
        idx_t n = getMatIndex(t);
        return RealvGrid_NXYZ<Mapfn>::getVecPtrNorm(n, iv, jv, kv, false);
    }
};

// A 5D (t, n, x, y, z) collection of realv elements, but any value of 't'
// is divided by CPTS_T and mapped to TIME_DIM_SIZE indices.
// Supports symmetric padding in each spatial dimension.
template <typename Mapfn> class RealvGrid_TNXYZ : public RealvGrid_NXYZ<Mapfn> {
    
protected:
    idx_t _dn;

public:

    // Ctor.
    RealvGrid_TNXYZ(idx_t dn, idx_t dx, idx_t dy, idx_t dz,
                    idx_t pn, idx_t px, idx_t py, idx_t pz,
                    const string& name) :
        RealvGrid_NXYZ<Mapfn(TIME_DIM_SIZE * dn, dx, dy, dz,
                             pn, px, py, pz,
                             name),
        _dn(dn)
    {
        if (VLEN_N > 1) {
            cerr << "Sorry, vectorizing in N dimension not yet supported." << endl;
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

        // Map t_idx2 and n onto one dimension.
        return MAP21(n, t_idx2, _dn, TIME_DIM_SIZE);
    }

    // Read one element.
    ALWAYS_INLINE REAL readElem(idx_t t, idx_t n,
                               idx_t i, idx_t j, idx_t k,
                               int line) const {
        idx_t n2 = getMatIndex(t, n);
        return RealvGrid_NXYZ<Mapfn>::readElem(n2, i, j, k, line);
    }

    // Write one element of the grid.
    ALWAYS_INLINE void writeElem(REAL val, idx_t t, idx_t n,
                                idx_t i, idx_t j, idx_t k,
                                int line) {
        idx_t n2 = getMatIndex(t, n);
        RealvGrid_NXYZ<Mapfn>::writeElem(val, n2, i, j, k, line);
    }

    // Read one vector at t and vector offset nv, iv, jv, kv.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    ALWAYS_INLINE const realv& readVecNorm(idx_t t, idx_t nv,
                                           idx_t iv, idx_t jv, idx_t kv,
                                           int line) const {
        idx_t n2 = getMatIndex(t, nv);
        return RealvGrid_NXYZ<Mapfn>::readVecNorm(n2, iv, jv, kv, line);
    }

    // Write one vector at t and vector offset nv, iv, jv, kv.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    ALWAYS_INLINE void writeVecNorm(const realv& v, idx_t t, idx_t nv,
                                    idx_t iv, idx_t jv, idx_t kv,
                                      int line) {
        idx_t n2 = getMatIndex(t, nv);
        RealvGrid_NXYZ<Mapfn>::writeVecNorm(v, n2, iv, jv, kv, line);
    }

    // Get pointer to the real at t and offset n, i, j, k.
    ALWAYS_INLINE const REAL* getElemPtr(idx_t t, idx_t n, idx_t i, idx_t j, idx_t k,
                                          int line) const {
        idx_t n2 = getMatIndex(t, n);
        return RealvGrid_NXYZ<Mapfn>::getElemPtr(n2, i, j, k, false);
    }
    ALWAYS_INLINE REAL* getElemPtr(idx_t t, idx_t n, idx_t i, idx_t j, idx_t k,
                                       int line) {
        idx_t n2 = getMatIndex(t, n);
        return RealvGrid_NXYZ<Mapfn>::getElemPtr(n2, i, j, k, false);
    }

    // Get pointer to the realv at t and vector offset nv, iv, jv, kv.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    ALWAYS_INLINE const realv* getVecPtrNorm(idx_t t, idx_t nv,
                                             idx_t iv, idx_t jv, idx_t kv,
                                             int line) const {
        idx_t n2 = getMatIndex(t, nv);
        return RealvGrid_NXYZ<Mapfn>::getVecPtrNorm(n2, iv, jv, kv, false);
    }
    ALWAYS_INLINE realv* getVecPtrNorm(idx_t t, idx_t nv,
                                       idx_t iv, idx_t jv, idx_t kv,
                                       int line) {
        idx_t n2 = getMatIndex(t, nv);
        return RealvGrid_NXYZ<Mapfn>::getVecPtrNorm(n2, iv, jv, kv, false);
    }
};

#endif
