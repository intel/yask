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

// Base class for realv grids.
// Provides generic-grid support.
class RealvGridBase {
protected:
    string _name;
    GenericGridBase<realv>* _gp;

public:
    RealvGridBase(string name, GenericGridBase<realv>* gp) :
        _name(name), _gp(gp) { }
    
    // Initialize memory to a given value.
    virtual void set_same(REAL val) {
        realv rn;
        rn = val;               // broadcast.
        _gp->set_same(rn);
    }

    // Initialize memory to incrementing values based on val.
    virtual void set_diff(REAL val) {

        // make a realv pattern
        realv rn;
        for (int i = 0; i < VLEN; i++)
            rn.r[i] = REAL(i * VLEN + 1) * val / VLEN;
        
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
            << iv << "," << jv << "," << kv << ")";
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

        // normalize.
        idx_t iv = idiv<idx_t>(i, VLEN_X);
        idx_t jv = idiv<idx_t>(j, VLEN_Y);
        idx_t kv = idiv<idx_t>(k, VLEN_Z);
        
        // Get vector.
        const realv* vp = getVecPtrNorm(iv, jv, kv, checkBounds);

        // intra-vector element indices.
        idx_t ip = imod<idx_t>(i, VLEN_X);
        idx_t jp = imod<idx_t>(j, VLEN_Y);
        idx_t kp = imod<idx_t>(k, VLEN_Z);

        // Extract point from vector.
        return &(*vp)(0, ip, jp, kp);
    }

    // non-const version.
    ALWAYS_INLINE REAL* getElemPtr(idx_t i, idx_t j, idx_t k,
                                  bool checkBounds=true) {
        const REAL* p = const_cast<const RealvGrid_XYZ*>(this)->getElemPtr(i, j, k,
                                                                         checkBounds);
        return const_cast<REAL*>(p);
    }

    // Print one vector.
    void printVec(const string& m, idx_t i, idx_t j, idx_t k, const realv& v,
                  int line) const {
        for (int k2 = 0; k2 < VLEN_Z; k2++) {
            for (int j2 = 0; j2 < VLEN_Y; j2++) {
                for (int i2 = 0; i2 < VLEN_X; i2++) {
                    REAL e = v(0, i2, j2, k2);
                    cout << m << ": " << _name << "[" <<
                        (i+i2) << ", " << (j+j2) << ", " << (k+k2) << "] = " << e;
                    if (line)
                        cout << " at line " << line;
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

    // Read one vector at vector offset nv, iv, jv, kv.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    ALWAYS_INLINE const realv readVecNorm(idx_t iv, idx_t jv, idx_t kv,
                                          int line) const {
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

    // Check for equality.
    // Return number of mismatches greater than epsilon.
    idx_t compare(const RealvGrid_XYZ<Mapfn>& ref,
                   REAL epsilon = EPSILON,
                   int maxPrint = 20,
                   std::ostream& os = std::cerr) const {
        realv ev;
        ev = epsilon;           // broadcast.
        return _data.compare(ref._data, ev, maxPrint, os);
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

        // normalize.
        idx_t nv = idiv<idx_t>(n, VLEN_N);
        idx_t iv = idiv<idx_t>(i, VLEN_X);
        idx_t jv = idiv<idx_t>(j, VLEN_Y);
        idx_t kv = idiv<idx_t>(k, VLEN_Z);
        
        // Get vector.
        const realv* vp = getVecPtrNorm(nv, iv, jv, kv, checkBounds);

        // intra-vector element indices.
        idx_t np = imod<idx_t>(n, VLEN_N);
        idx_t ip = imod<idx_t>(i, VLEN_X);
        idx_t jp = imod<idx_t>(j, VLEN_Y);
        idx_t kp = imod<idx_t>(k, VLEN_Z);

        // Extract point from vector.
        return &(*vp)(np, ip, jp, kp);
    }

    // non-const version.
    ALWAYS_INLINE REAL* getElemPtr(idx_t n, idx_t i, idx_t j, idx_t k,
                                  bool checkBounds=true) {
        const REAL* p = const_cast<const RealvGrid_NXYZ*>(this)->getElemPtr(n, i, j, k,
                                                                         checkBounds);
        return const_cast<REAL*>(p);
    }

    // Print one vector.
    void printVec(const string& m, idx_t n, idx_t i, idx_t j, idx_t k, const realv& v,
                  int line) const {
        for (int k2 = 0; k2 < VLEN_Z; k2++) {
            for (int j2 = 0; j2 < VLEN_Y; j2++) {
                for (int i2 = 0; i2 < VLEN_X; i2++) {
                    for (int n2 = 0; n2 < VLEN_N; n2++) {
                    
                        REAL e = v(n2, i2, j2, k2);
                        cout << m << ": " << _name << "[" << (n+n2) << ", " <<
                            (i+i2) << ", " << (j+j2) << ", " << (k+k2) << "] = " << e;
                        if (line)
                            cout << " at line " << line;
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

    // Check for equality.
    // Return number of mismatches greater than epsilon.
    idx_t compare(const RealvGrid_NXYZ<Mapfn>& ref,
                   REAL epsilon = EPSILON,
                   int maxPrint = 20,
                   std::ostream& os = std::cerr) const {
        realv ev;
        ev = epsilon;           // broadcast.
        return _data.compare(ref._data, ev, maxPrint, os);
    }    
};

// A 4D (t, x, y, z) collection of realv elements, but any value of 't'
// is divided by TIME_STEPS and mapped to TIME_DIM indices.
// Supports symmetric padding in each spatial dimension.
template <typename Mapfn> class RealvGrid_TXYZ : public RealvGrid_NXYZ<Mapfn>  {
    
public:

    // Ctor.
    RealvGrid_TXYZ(idx_t dx, idx_t dy, idx_t dz,
                   idx_t px, idx_t py, idx_t pz,
                   const string& name) :
        RealvGrid_NXYZ<Mapfn>(TIME_DIM, dx, dy, dz,
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

#if 1
        // Time t must be multiple of TIME_STEPS.
        // Use imod & idiv to allow t to be negative.
        assert(imod<idx_t>(t, TIME_STEPS) == 0);
        idx_t t_idx = idiv<idx_t>(t, TIME_STEPS);

        // Index wraps in TIME_DIM.
        // Examples if TIME_DIM == 2:
        // t_idx => return value.
        // -2 => 0.
        // -1 => 1.
        //  0 => 0.
        //  1 => 1.

        // Use imod to allow t to be negative.
        return imod<idx_t>(t_idx, TIME_DIM);
#else
        // version that doesn't allow negative time.
        assert(t >= 0);
        assert(t % TIME_STEPS == 0);
        idx_t t_idx = t / idx_t(TIME_STEPS);
        return t_idx % idx_t(TIME_DIM);
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
// is divided by TIME_STEPS and mapped to TIME_DIM indices.
// Supports symmetric padding in each spatial dimension.
template <typename Mapfn> class RealvGrid_TNXYZ : public RealvGrid_NXYZ<Mapfn> {
    
protected:
    idx_t _dn;

public:

    // Ctor.
    RealvGrid_TNXYZ(idx_t dn, idx_t dx, idx_t dy, idx_t dz,
                    idx_t pn, idx_t px, idx_t py, idx_t pz,
                    const string& name) :
        RealvGrid_NXYZ<Mapfn(TIME_DIM * dn, dx, dy, dz,
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

        // Time t must be multiple of TIME_STEPS.
        // Use imod & idiv to allow t to be negative.
        assert(imod<idx_t>(t, TIME_STEPS) == 0);
        idx_t t_idx = idiv<idx_t>(t, TIME_STEPS);

        // Index wraps in TIME_DIM.
        // Examples if TIME_DIM == 2:
        // t_idx => t_idx2.
        // -2 => 0.
        // -1 => 1.
        //  0 => 0.
        //  1 => 1.
        // Use imod to allow t to be negative.
        idx_t t_idx2 = imod<idx_t>(t_idx, TIME_DIM);

        // Map t_idx2 and n onto one dimension.
        return MAP21(n, t_idx2, _dn, TIME_DIM);
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
