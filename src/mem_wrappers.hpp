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

// This file defines wrappers to use for accessing memory.
// By using these wrappers, you can turn on the following:
// - use the cache model when CACHE_MODEL is set to 1 or 2.
// - trace accesses when TRACE_MEM is set.
// - check alignment when DEBUG is set.
// If none of these are activated, there is no cost.
// Also defines macros for PREFETCH and EVICT.

#ifndef _MEM_WRAPPERS_H
#define _MEM_WRAPPERS_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
using namespace std;

#define CACHELINE_BYTES   64

// Set MODEL_CACHE to 1 or 2 to model L1 or L2.
#ifdef MODEL_CACHE
#include "cache_model.hpp"
extern Cache cache;
#endif

    //#define ALLOC_ALIGNMENT CACHELINE_BYTES
#define ALLOC_ALIGNMENT 4096 // 4k-page

// Make an index and offset canonical, i.e., offset in [0..vecLen-1].
// Makes proper adjustments for negative inputs.
#define FIX_INDEX_OFFSET(indexIn, offsetIn, indexOut, offsetOut, vecLen) \
    do {                                                                \
        const int ofs = ((indexIn) * (vecLen)) + (offsetIn);            \
        indexOut = ofs / (vecLen);                                      \
        offsetOut = ofs % (vecLen);                                     \
        while(offsetOut < 0) {                                          \
            offsetOut += (vecLen);                                      \
            indexOut--;                                                 \
        }                                                               \
    } while(0)

// _numMats 3D matrices of vectors.
template <long _numMats> class RealMatrixBase {

 protected:
    string _name;               // for debug messages.
    
    realv* restrict _base;      // base address of data.

    // vector dimensions.
    long _vlen_x, _vlen_y, _vlen_z, _vlen;

    // following vars are per-matrix and are counts by vector-blocking dims.
    long _d1v, _d2v, _d3v;       // sizes of each dimension.
    long _p1v, _p2v, _p3v;       // sizes of each pad, per side.
    long _s1v, _s2v, _s3v;       // allocated sizes in each dimension.

    // following vars are for all matrices.
    size_t _sizev;               // number of vectors of reals.

 public:

    // Create a matrix.
    // All size arguments are in units of reals.
    // d1..d3 sizes should not include halos.
    // The padding must be large enough to include halos.
    // Read and write functions are relative to d1..d3.
    // p1..p3 are the padding sizes, per side.
    // Call read and write functions with negative numbers or
    // numbers >= dimensions to access halos.
    // d* and p* are rounded up to multiples of vlen_*.
    RealMatrixBase(long vlen_x, long vlen_y, long vlen_z,
                    long d1, long d2, long d3,
                    long p1, long p2, long p3,
                    const string& name) :
        _name(name),
        _vlen_x(vlen_x), _vlen_y(vlen_y), _vlen_z(vlen_z)
    {
        // problem dimensions.
        d1 = ROUND_UP(d1, vlen_x);
        d2 = ROUND_UP(d2, vlen_y);
        d3 = ROUND_UP(d3, vlen_z);
        _d1v = d1/vlen_x;
        _d2v = d2/vlen_y;
        _d3v = d3/vlen_z;

        // padding, including halos.
        p1 = ROUND_UP(p1, vlen_x);
        p2 = ROUND_UP(p2, vlen_y);
        p3 = ROUND_UP(p3, vlen_z);
        _p1v = p1/vlen_x;
        _p2v = p2/vlen_y;
        _p3v = p3/vlen_z;

        // totals.
        _s1v = _d1v + 2*_p1v;
        _s2v = _d2v + 2*_p2v;
        _s3v = _d3v + 2*_p3v;
        _sizev = _s1v * _s2v * _s3v * _numMats;
        _vlen = vlen_x * vlen_y * vlen_z;
        assert(VLEN == _vlen);

        const double oneK = 1024.;
        const double oneM = oneK * oneK;
        const double oneG = oneK * oneM;
        cout << "Allocating ";
        if (double(getNumBytes()) > oneM)
            cout << (double(getNumBytes()) / oneG) << " GBytes";
        else if (double(getNumBytes()) > oneK)
            cout << (double(getNumBytes()) / oneM) << " MBytes";
        else
            cout << (double(getNumBytes()) / oneK) << " kBytes";
        cout << " for " <<
            getNumReals() << " reals (" <<
            _numMats << " * " << _vlen << " * (" <<
            _s1v << " * " << _s2v << " * " << _s3v << ")) in " <<
            _numMats << " '" << _name << "' " << (_numMats == 1 ? "matrix" : "matrices");
        _base = (realv*)_mm_malloc( getNumBytes(), ALLOC_ALIGNMENT);
        if (!_base) {
            cerr << "error: cannot allocate " << getNumReals() << " reals." << endl;
            exit(1);
        }
    }

    virtual ~RealMatrixBase() {
        cout << "freeing " << getNumReals() << " reals for '" << _name << "'." << endl;
        _mm_free(_base);
        _base = 0;
        _sizev = 0;
    }

    inline const string& getName() const {
        return _name;
    }
    inline const char* getNameCStr() const {
        return _name.c_str();
    }

    // Vector sizes.
    inline long getVlen() const {
        return _vlen;
    }
    inline long getVlenX() const {
        return _vlen_x;
    }
    inline long getVlenY() const {
        return _vlen_y;
    }
    inline long getVlenZ() const {
        return _vlen_z;
    }

    // Matrix sizes (without padding).
    inline long getVecDimX() const {
        return _d1v;
    }
    inline long getVecDimY() const {
        return _d2v;
    }
    inline long getVecDimZ() const {
        return _d3v;
    }
    inline long getDimX() const {
        return _d1v * _vlen_x;
    }
    inline long getDimY() const {
        return _d2v * _vlen_y;
    }
    inline long getDimZ() const {
        return _d3v * _vlen_z;
    }

    // These functions return total sizes, not per-matrix.
    inline size_t getNumVecs() const {
        return _sizev;
    }
    inline size_t getNumReals() const {
        return getNumVecs() * _vlen;
    }
    inline size_t getNumBytes() const {
        return getNumReals() * sizeof(REAL);
    }

    // direct access to a REAL.
    inline REAL& raw(size_t ai) {
        assert(ai < getNumReals());
        size_t i, j;
        UNMAP21(ai, i, j, getNumVecs(), _vlen);
        assert(i < getNumVecs());
        return _base[i][j];
    }
    inline const REAL& raw(size_t ai) const {
        assert(ai < getNumReals());
        size_t i, j;
        UNMAP21(ai, i, j, getNumVecs(), _vlen);
        assert(i < getNumVecs());
        return _base[i][j];
    }

    // 'line' args are for debug.

    // Init data to various values.
    void init_data(unsigned int seed = 1)
    {
        if (!seed)
            seed = (unsigned int)(((size_t)(_base) >> 12) & 0xff); // multiplier based on some bits in base address.
        const REAL m = 1.f / (seed + 1);

        for(size_t i=0; i<getNumVecs(); i++) {
            REAL n = m * (i % seed + 1);
            SIMD_LOOP(j) 
                _base[i][j] = n + REAL(j)/_vlen;
        }
    }

    // Set data to identical values.
    void set_data(REAL val = 0.1f)
    {
        size_t n = getNumVecs();
#pragma omp parallel for
        for(size_t i=0; i<n; i++) {
            SIMD_LOOP(j) 
                _base[i][j] = val;
        }
    }

    // Compare and report errors if relative absolute error > maxErr and absolute error > maxErr.
    bool within_epsilon(const RealMatrixBase& ref,
                        REAL maxErr = 1e-3f, int max_errs = 20 ) const
    {
        size_t nok = 0;
        size_t okPrints = 0;
        size_t nerrs = 0;
        maxErr = fabsf(maxErr);
        REAL maxAE = 0.f;

        assert(getNumReals() == ref.getNumReals());
        cout << "comparing " << getNumReals() << " values..." << endl;
        for(size_t ai=0; ai<getNumReals(); ai++) {
            REAL ae = fabsf( ref.raw(ai) - raw(ai) );
            if (ae > maxAE)
                maxAE = ae;
            bool ok = ae <= maxErr;
            REAL rae = 0.f;
            if (fabsf(ref.raw(ai)) > 1.f) {
                rae = fabsf( ae / ref.raw(ai) );
                ok |= rae <= maxErr;
            }
            if( !ok ) {
                if (++nerrs < max_errs)
                    printf(" ERROR: elem %lu = %.4g in %s != %.4g in %s; AE=%.4g; RAE=%.4g\n",
                           ai, raw(ai), getNameCStr(),
                           ref.raw(ai), ref.getNameCStr(), ae, rae);
            }
                
            // just print a few matches.
            else if (++nok < okPrints) {
                printf(" INFO: elem %lu = %.4g in %s ~= %.4g in %s; AE=%.4g; RAE=%.4g\n",
                       ai, raw(ai), getNameCStr(),
                       ref.raw(ai), ref.getNameCStr(), ae, rae);
            }
        }
        cout << " max abs error = " << maxAE << 
            "; num errors = " << nerrs << "/" << getNumReals() << endl;
        return nerrs == 0;
    }
};

#include "real_matrices.hpp"

// L1 and L2 hints
#define L1 _MM_HINT_T0
#define L2 _MM_HINT_T1

#ifdef MODEL_CACHE
#if MODEL_CACHE==L1
#warning Modeling L1 cache
#elif MODEL_CACHE==L2
#warning Modeling L2 cache
#else
#warning Modeling UNKNOWN cache
#endif
#endif

// prefetch cannot be a function because the hint cannot be a var.
// define some optional prefix macros for cache modeling and tracing.
#ifdef NOPREFETCH
#define PREFETCH(hint, base, matNum, xv, yv, zv, line) true
#else

#ifdef MODEL_CACHE
#define MCP(p, hint, line) cache.prefetch(p, hint, line)
#else
#define MCP(p, hint, line)
#endif

#ifdef TRACE_MEM
#define TP(p, hint, base, matNum, xv, yv, zv, line) \
    printf("prefetch %s[%i][%i,%i,%i](%p) to L%i at line %i.\n", \
           base.getNameCStr(), matNum, (int)xv (int)yv, (int)zv, p, hint, line); \
        fflush(stdout)
#else 
#define TP(p, hint, base, matNum, xv, y, z, line)
#endif

#define PREFETCH(hint, base, matNum, xv, yv, zv, line)    \
    do {                                        \
        const REAL *p = base.getPtr(matNum, xv, yv, zv, false);        \
        TP(p, hint, base, matNum, xv, yv, zv, line); MCP(p, hint, line);  \
        _mm_prefetch((const char*)(p), hint);   \
    } while(0)
#endif

// evict cannot be a function because the hint cannot be a var.
// define some optional prefix macros for cache modeling and tracing.
#ifdef NOEVICT
#define EVICT(hint, base, matNum, xv, y, z, line) true
#else

#ifdef MODEL_CACHE
#define MCE(p, hint, line) cache.evict(p, hint, line),
#else
#define MCE(p, hint, line)
#endif

#ifdef TRACE_MEM
#define TE(p, hint, line) printf("evict %p from L%i at line %i.\n", p, hint, line); \
        fflush(stdout)
#else
#define TE(p, hint, line)
#endif

#define EVICT(hint, base, matNum, xv, yv, zv, line)    \
    do {                                        \
        const REAL *p = base.getPtr(matNum, xv, yv, zv, false);        \
        TE(p, hint, line); MCE(p, hint, line);  \
        _mm_clevict((const char*)(p), hint);    \
    } while(0)
#endif

////// Default prefetch distances.
// all prefetch distances are in units of vec-blocks.
// TODO: determine the default programmatically.

// how far to prefetch ahead for L1.
#ifndef PFDL1
#define PFDL1 1
#endif

// how far to prefetch ahead for L2.
#ifndef PFDL2
#define PFDL2 8
#endif

// make sure PFDL2 > PFDL1.
#if PFDL2 <= PFDL1
#undef PFDL2
#define PFDL2 (PFDL1+1)
#endif

#endif
