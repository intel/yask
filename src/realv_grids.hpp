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

// A 3D (x, y, z) collection of realv elements.
template <typename Mapfn> class RealvGrid3d {
    
protected:
    GenericGrid3d<realv, Mapfn> _data;
    string _name;

public:

    // Ctor.
    // Dimensions are REAL elements, not realvs.
    RealvGrid3d(int d1, int d2, int d3,
                int p1, int p2, int p3,
                const string& name) :
        _data(d1 / VLEN_X, d2 / VLEN_Y, d3 / VLEN_Z,
              p1 / VLEN_X, p2 / VLEN_Y, p3 / VLEN_Z,
              ALLOC_ALIGNMENT),
        _name(name)
    {
        _data.print_info(name);
    }

    // Get pointer to a realv.
    ALWAYS_INLINE const realv* getVecPtr(int veci, int vecj, int veck,
                                         bool safe=true) const {

#ifdef TRACE_MEM
        cout << _name << "." << "getVecPtr-3D(" <<
            veci << "," << vecj << "," << veck << ", " <<
            (safe ? "safe": "unsafe") << ")";
        if (safe)
            cout << " => " << _data.get_index(veci, vecj, veck);
        cout << endl << flush;
#endif
        return &_data(veci, vecj, veck, safe);
    }

    // Non-const version.
    ALWAYS_INLINE realv* getVecPtr(int veci, int vecj, int veck,
                                   bool safe=true) {
        const realv* vp =
            const_cast<const RealvGrid3d*>(this)->getVecPtr(veci, vecj, veck, safe);
        return const_cast<realv*>(vp);
    }
    
    // Get a pointer to one REAL.
    ALWAYS_INLINE const REAL* getElemPtr(int i, int j, int k,
                                        bool safe=true) const {
        const realv* vp = getVecPtr(i / VLEN_X, j / VLEN_Y, k / VLEN_Z, safe);
        return &(*vp)(abs(i % VLEN_X), abs(j % VLEN_Y), abs(k % VLEN_Z));
    }

    // non-const version.
    ALWAYS_INLINE REAL* getElemPtr(int i, int j, int k,
                                  bool safe=true) {
        const REAL* p = const_cast<const RealvGrid3d*>(this)->getElemPtr(i, j, k, safe);
        return const_cast<REAL*>(p);
    }

    // Print one vector.
    void printVec(const string& m, int veci, int vecj, int veck, const realv& v,
                  int line=0) const {
        int x1 = veci * VLEN_X;
        int y1 = vecj * VLEN_Y;
        int z1 = veck * VLEN_Z;
        cout << m << ": " << _name << "[" <<
            x1 << ".." << x1+VLEN_X-1 << " x " <<
            y1 << ".." << y1+VLEN_Y-1 << " x " <<
            z1 << ".." << z1+VLEN_Z-1 << "] = " << v;
        if (line)
            cout << " at line " << line;
        cout << endl << flush;
    }

    // Print one element.
    void printElem(const string& m, int i, int j, int k, REAL e,
                   int line=0) const {
        cout << m << ": " << _name << "[" <<
            i << ", " << j << ", " << k << "] = " << e;
        if (line)
            cout << " at line " << line;
        cout << endl << flush;
    }

    // Read one element.
    ALWAYS_INLINE REAL readElem(int i, int j, int k,
                               int line=0) const {
        const REAL* ep = getElemPtr(i, j, k);
        REAL e = *ep;
#ifdef TRACE_MEM
        printElem("readElem", i, j, k, e, line);
#endif
        return e;
    }

    // Write one element.
    ALWAYS_INLINE void writeElem(REAL val, int i, int j, int k,
                                int line=0) {
        REAL* ep = getElemPtr(i, j, k);
        *ep = val;
#ifdef TRACE_MEM
        printElem("writeElem", i, j, k, val, line);
#endif
    }

    // Read one vector at veci, vecj, veck.
    ALWAYS_INLINE const realv readVec(int veci, int vecj, int veck,
                                      int line=0) const {
        const realv* p = getVecPtr(veci, vecj, veck);
        __assume_aligned(p, CACHELINE_BYTES);
        realv v;
        v.loadFrom(p);
#ifdef TRACE_MEM
        printVec("readVec", veci, vecj, veck, v, line);
#endif
#ifdef MODEL_CACHE
        cache.read(p, line);
#endif
        return v;
    }

    // Write one vector at veci, vecj, veck.
    ALWAYS_INLINE void writeVec(const realv& v, int veci, int vecj, int veck,
                                int line=0) {
        realv* p = getVecPtr(veci, vecj, veck);
        __assume_aligned(p, CACHELINE_BYTES);
        v.storeTo(p);
#ifdef TRACE_MEM
        printVec("writeVec", veci, vecj, veck, v, line);
#endif
#ifdef MODEL_CACHE
        cache.write(p, line);
#endif
    }

    // Check for equality.
    // Return number of mismatches greater than epsilon.
    size_t compare(const RealvGrid3d<Mapfn>& ref,
                           REAL epsilon = 1e-3,
                           int maxPrint = 20,
                           std::ostream& os = std::cerr) const {
        realv ev;
        ev = epsilon;
        return _data.compare(ref._data, ev, maxPrint, os);
    }

    // Initialize memory to a given value.
    void set_same(REAL val) {
        realv rv;
        rv = val;
        _data.set_same(rv);
    }    
};

// A 4D (n, x, y, z) collection of realv elements.
template <typename Mapfn> class RealvGrid4d {
    
protected:
    GenericGrid4d<realv, Mapfn> _data;
    string _name;

public:

    // Ctor.
    // Note that 'num' is stored as the 4th dimension in _data,
    // even though the accessors below list it first.
    RealvGrid4d(int num,
                int d1, int d2, int d3,
                int p1, int p2, int p3,
                const string& name) :
        _data(d1 / VLEN_X, d2 / VLEN_Y, d3 / VLEN_Z, num,
              p1 / VLEN_X, p2 / VLEN_Y, p3 / VLEN_Z, 0,
              ALLOC_ALIGNMENT),
        _name(name)
    {
        _data.print_info(name);
    }

    // Get pointer to a realv.
    ALWAYS_INLINE const realv* getVecPtr(int num, int veci, int vecj, int veck,
                                         bool safe=true) const {

#ifdef TRACE_MEM
        cout << _name << "." << "getVecPtr.4D(" <<
            num << "," <<
            veci << "," << vecj << "," << veck << ", " <<
            (safe ? "safe": "unsafe") << ")";
        if (safe)
            cout << " => " << _data.get_index(veci, vecj, veck, num);
        cout << endl << flush;
#endif
        return &_data(veci, vecj, veck, num, safe);
    }

    // Non-const version.
    ALWAYS_INLINE realv* getVecPtr(int num, int veci, int vecj, int veck,
                                   bool safe=true) {
        const realv* vp =
            const_cast<const RealvGrid4d*>(this)->getVecPtr(num, veci, vecj, veck, safe);
        return const_cast<realv*>(vp);
    }
    
    // Get a pointer to one REAL.
    ALWAYS_INLINE const REAL* getElemPtr(int num, int i, int j, int k,
                                        bool safe=true) const {
        const realv* vp = getVecPtr(num, i / VLEN_X, j / VLEN_Y, k / VLEN_Z, safe);
        return &(*vp)(abs(i % VLEN_X), abs(j % VLEN_Y), abs(k % VLEN_Z));
    }

    // non-const version.
    ALWAYS_INLINE REAL* getElemPtr(int num, int i, int j, int k,
                                  bool safe=true) {
        const REAL* p = const_cast<const RealvGrid4d*>(this)->getElemPtr(num, i, j, k, safe);
        return const_cast<REAL*>(p);
    }

    // Print one vector.
    void printVec(const string& m, int num, int veci, int vecj, int veck, const realv& v,
                  int line=0) const {
        int x1 = veci * VLEN_X;
        int y1 = vecj * VLEN_Y;
        int z1 = veck * VLEN_Z;
        cout << m << ": " << _name << "[" <<
            num << ", " <<
            x1 << ".." << x1+VLEN_X-1 << " x " <<
            y1 << ".." << y1+VLEN_Y-1 << " x " <<
            z1 << ".." << z1+VLEN_Z-1 << "] = " << v;
        if (line)
            cout << " at line " << line;
        cout << endl << flush;
    }

    // Print one element.
    void printElem(const string& m, int num, int i, int j, int k, REAL e,
                   int line=0) const {
        cout << m << ": " << _name << "[" <<
            num << ", " <<
            i << ", " << j << ", " << k << "] = " << e;
        if (line)
            cout << " at line " << line;
        cout << endl << flush;
    }

    // Read one element.
    ALWAYS_INLINE REAL readElem(int num, int i, int j, int k,
                               int line=0) const {
        const REAL* ep = getElemPtr(num, i, j, k);
        REAL e = *ep;
#ifdef TRACE_MEM
        printElem("readElem", num, i, j, k, e, line);
#endif
        return e;
    }

    // Write one element.
    ALWAYS_INLINE void writeElem(REAL val, int num, int i, int j, int k,
                               int line=0) {
        REAL* ep = getElemPtr(num, i, j, k);
        *ep = val;
#ifdef TRACE_MEM
        printElem("writeElem", num, i, j, k, val, line);
#endif
    }

    // Read one vector at veci, vecj, veck.
    ALWAYS_INLINE const realv readVec(int num, int veci, int vecj, int veck,
                                      int line=0) const {
        const realv* p = getVecPtr(num, veci, vecj, veck);
        __assume_aligned(p, CACHELINE_BYTES);
        realv v;
        v.loadFrom(p);
#ifdef TRACE_MEM
        printVec("readVec", num, veci, vecj, veck, v, line);
#endif
#ifdef MODEL_CACHE
        cache.read(p, line);
#endif
        return v;
    }

    // Write one vector at veci, vecj, veck.
    ALWAYS_INLINE void writeVec(const realv& v, int num, int veci, int vecj, int veck,
                                int line=0) {
        realv* p = getVecPtr(num, veci, vecj, veck);
        __assume_aligned(p, CACHELINE_BYTES);
        v.storeTo(p);
#ifdef TRACE_MEM
        printVec("writeVec", num, veci, vecj, veck, v, line);
#endif
#ifdef MODEL_CACHE
        cache.write(p, line);
#endif
    }


    // Check for equality.
    // Return number of mismatches greater than epsilon.
    size_t compare(const RealvGrid4d<Mapfn>& ref,
                           REAL epsilon = 1e-3,
                           int maxPrint = 20,
                           std::ostream& os = std::cerr) const {
        realv ev;
        ev = epsilon;
        return _data.compare(ref._data, ev, maxPrint, os);
    }    

    // Initialize memory to a given value.
    void set_same(REAL val) {
        realv rv;
        rv = val;
        _data.set_same(rv);
    }    
};

// A pseudo-5D (time, var, x, y, z) collection of realv elements.  It is
// really a 4D grid; the time and var 'dimensions' are mapped onto the 4th
// dimension in a restricted way which only allows read from t and write to
// t+TIME_STEPS.
template <typename Mapfn> class RealvGridPseudo5d {
    
protected:
    RealvGrid4d<Mapfn> _data;

public:

    // Ctor.
    RealvGridPseudo5d(int d1, int d2, int d3,
                      int p1, int p2, int p3,
                      const string& name) :
        _data(NUM_GRIDS,
              d1, d2, d3,
              p1, p2, p3,
              name) { }

    // Get correct index based on time & variable number.
    // There will be overlap between indices if NUM_WORKS < NUM_VARS.
    ALWAYS_INLINE int getMatIndex(int t, int var) const {
        assert(t % TIME_STEPS == 0);

        /* Example with NUM_VARS=5 & NUM_WORKS=2:
           t=0...
           var 0: mat 0 => mat 5
           var 1: mat 1 => mat 6
           var 2: mat 2 => mat 0
           var 3: mat 3 => mat 1
           var 4: mat 4 => mat 2
           t=1...
           var 0: mat 5 => mat 3
           var 1: mat 6 => mat 4
           var 2: mat 0 => mat 5
           var 3: mat 1 => mat 6
           var 4: mat 2 => mat 0
        */

        return ((t / TIME_STEPS) * NUM_VARS + var) % NUM_GRIDS;
    }

    // Read one element.
    ALWAYS_INLINE REAL readElem(int t, int var,
                               int i, int j, int k,
                               int line=0) const {
        int n = getMatIndex(t, var);
        return _data.readElem(n, i, j, k, line);
    }

    // Write one element of the grid.
    ALWAYS_INLINE void writeElem(REAL val, int t, int var,
                                int i, int j, int k,
                                int line=0) {
        int n = getMatIndex(t, var);
        _data.writeElem(val, n, i, j, k, line);
    }

    // Read one vector at veci, vecj, veck.
    ALWAYS_INLINE const realv& readVec(int t, int var,
                                      int veci, int vecj, int veck,
                                      int line=0) const {
        int n = getMatIndex(t, var);
        return _data.readVec(n, veci, vecj, veck, line);
    }

    // Write one vector at veci, vecj, veck.
    ALWAYS_INLINE void writeVec(const realv& v, int t, int var,
                                      int veci, int vecj, int veck,
                                      int line=0) {
        int n = getMatIndex(t, var);
        _data.writeVec(v, n, veci, vecj, veck, line);
    }

    // Get a pointer to vector at veci, vecj, veck.
    ALWAYS_INLINE const realv* getVecPtr(int t, int var,
                                         int veci, int vecj, int veck,
                                         int line=0) const {
        int n = getMatIndex(t, var);
        return _data.getVecPtr(n, veci, vecj, veck, false);
    }
    ALWAYS_INLINE realv* getVecPtr(int t, int var,
                                         int veci, int vecj, int veck,
                                         int line=0) {
        int n = getMatIndex(t, var);
        return _data.getVecPtr(n, veci, vecj, veck, false);
    }


    // Check for equality.
    // Return number of mismatches greater than epsilon.
    size_t compare(const RealvGridPseudo5d<Mapfn>& ref,
                           REAL epsilon = 1e-3,
                           int maxPrint = 20,
                           std::ostream& os = std::cerr) const {
        return _data.compare(ref._data, epsilon, maxPrint, os);
    }

    // Initialize memory to a given value.
    void set_same(REAL val) {
        _data.set_same(val);
    }    
};

#endif
