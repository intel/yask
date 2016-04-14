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

#ifndef _STENCIL_HPP
#define _STENCIL_HPP

// control assert() with DEBUG instead of NDEBUG.
#ifndef DEBUG
#define NDEBUG
#endif

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <map>
#include <vector>

using namespace std;

// type to use for indexing grids.
typedef long int idx_t;

#include "idiv.hpp"
#include "map_macros.hpp"

#ifndef WIN32
#include <unistd.h>
#include <stdint.h>
#include <immintrin.h>
#else
#define _mm_clevict(p,h) true
#define _Pragma(x)
#endif

#if defined(__GNUC__) && !defined(__ICC)
#define __forceinline inline
#define __assume(x) true
#define __declspec(x)
typedef short unsigned int __mmask16;
#endif

#if (defined(__GNUC__) && !defined(__ICC)) || defined(WIN32)
#define restrict
#define __assume_aligned(p,n)
#endif

// VTune and stub macros.
#ifdef USE_VTUNE
#include "sampling_MIC.h"
#define SEP_PAUSE  VTPauseSampling()
#define SEP_RESUME VTResumeSampling()
#else
#define SEP_PAUSE
#define SEP_RESUME
#endif

// OpenMP and stub functions.
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_threads() (1)
#define omp_get_max_threads() (1)
#define omp_get_thread_num()  (0)
#endif

// Enable hardware thread work crew if requested.
#if (__INTEL_CREW) && (defined(__MIC__) || (defined(__linux) && defined(__x86_64)))
extern "C" {
    extern void kmp_crew_create();
    extern void kmp_crew_destroy();
    extern int kmp_crew_get_max_size();
}
#define CREW_FOR_LOOP _Pragma("intel_crew parallel for")
#else
#define kmp_crew_create()  ((void)0)
#define kmp_crew_destroy() ((void)0)
#define kmp_crew_get_max_size() (1)
#define CREW_FOR_LOOP
#endif

// conditional inlining
#ifdef DEBUG
#define ALWAYS_INLINE inline
#else
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#endif

// macro for debug message.
#ifdef TRACE
#define TRACE_MSG(fmt,...) (printf(fmt "\n",__VA_ARGS__), fflush(0))
#else
#define TRACE_MSG(fmt,...) true
#endif

// auto-generated macros from foldBuilder.
#include "stencil_macros.hpp"

// max abs difference in validation.
#ifndef EPSILON
#define EPSILON (1e-3)
#endif

// check whether two reals are close enough.
template<typename T>
inline bool within_tolerance(T val, T ref, T epsilon) {
    bool ok;
    double adiff = fabs(val - ref);
    if (fabs(ref) > 1.0)
        epsilon = fabs(ref * epsilon);
    ok = adiff < epsilon;
#ifdef DEBUG_TOLERANCE
    if (!ok)
        cerr << "outside tolerance of " << epsilon << ": " << val << " != " << ref <<
            " because " << adiff << " >= " << epsilon << endl;
#endif
    return ok;
}

// rounding macros.
#define DIV_AND_ROUND_UP(n, denom) (((n) + (denom) - 1) / (denom))
#define ROUND_UP(n, mult) (DIV_AND_ROUND_UP(n, mult) * (mult))

// number of time steps calculated in each stencil-function call.
#ifndef TIME_STEPS
#define TIME_STEPS (1)
#endif

// Size of time dimension required in allocated memory.
// TODO: calculate this per-grid based on dependency tree and
// traversal order.
#ifndef TIME_DIM
#define TIME_DIM (2)
#endif

// vector sizes.
#ifndef VLEN_N
#define VLEN_N (1)
#endif
#ifndef VLEN_X
#define VLEN_X (1)
#endif
#ifndef VLEN_Y
#define VLEN_Y (1)
#endif
#ifndef VLEN_Z
#define VLEN_Z (1)
#endif

// cluster sizes in vectors.
#ifndef CLEN_N
#define CLEN_N (1)
#endif
#ifndef CLEN_X
#define CLEN_X (1)
#endif
#ifndef CLEN_Y
#define CLEN_Y (1)
#endif
#ifndef CLEN_Z
#define CLEN_Z (1)
#endif

// cluster sizes in points.
#define CPTS_N (CLEN_N * VLEN_N)
#define CPTS_X (CLEN_X * VLEN_X)
#define CPTS_Y (CLEN_Y * VLEN_Y)
#define CPTS_Z (CLEN_Z * VLEN_Z)
#define CPTS (CPTS_N * CPTS_X * CPTS_Y * CPTS_Z)

// default problem size.
#ifndef DEF_PROB_SIZE
#define DEF_PROB_SIZE (768)
#endif

// define a vector of reals.
#include "realv.hpp"

// Memory-accessing code.
#include "mem_macros.hpp"
#include "realv_grids.hpp"

// Default grid layouts.
// 3-d dims are 1=x, 2=y, 3=z.
// 4-d dims are 1=n/t, 2=x, 3=y, 4=z.
// Last number in 'Map' layout has unit stride, e.g.,
// Map321 & Map1432 have unit-stride in x.
// Map123 & Map4123 have unit-stride in z.
#ifndef MAP_4D
#define MAP_4D Map1432
#endif
#ifndef MAP_3D
#define MAP_3D Map321
#endif
typedef RealvGrid_XYZ<MAP_3D> Grid_XYZ;
typedef RealvGrid_NXYZ<MAP_4D> Grid_NXYZ;
typedef RealvGrid_TXYZ<MAP_4D> Grid_TXYZ;
typedef RealvGrid_TNXYZ<MAP_4D> Grid_TNXYZ; // T and N mapped to 1st dim.

// Context for calculating values.
class StencilContext {
public:

    GRID_PTR_DEFNS;
    vector<RealvGridBase*> gridPtrs;

    idx_t dn, dx, dy, dz;                   // problem size.
    idx_t rn, rx, ry, rz;                   // region size.
    idx_t bn, bx, by, bz;                   // block size.
    idx_t padn, padx, pady, padz; // padding, including halos.

    StencilContext() { }

    // Allocate grid memory and set gridPtrs.
    void allocGrids() {
        cout << "allocating matrices..." << endl;
        GRID_ALLOCS;
        idx_t nbytes = 0;
        for (auto gp : gridPtrs)
            nbytes += gp->get_num_bytes();
        cout << "total allocation: " << (float(nbytes)/1e9) << "G byte(s)." << endl;
    }

    // Init all grids w/same value within each grid,
    // but different values between grids.
    void initSame() {
        REAL v = 0.1;
        cout << "initializing matrices..." << endl;
        for (auto gp : gridPtrs) {
            gp->set_same(v);
            v += 0.01;
        }
    }

    // Init all grids w/different values.
    // Better for validation, but slower.
    void initDiff() {
        REAL v = 0.01;
        for (auto gp : gridPtrs) {
            gp->set_inc(v);
            v += 0.001;
        }
    }

    // Compare contexts.
    idx_t compare(const StencilContext& ref) const {
        return GRID_COMPARES;
    }
};

// Base classes for auto-generated stencil code.
#include "stencil_calc.hpp"

// Include auto-generated stencil code.
#include "stencil_code.hpp"

// Some utility functions.
extern double getTimeInSecs();
extern idx_t roundUp(idx_t dim, idx_t mult, string name);

// Reference stencil calculations.
void calc_steps_ref(StencilContext& context, Stencils& stencils, const int nreps);

// Vectorized and blocked stencil calculations.
void calc_steps_opt(StencilContext& context, Stencils& stencils, const int nreps);

#endif
