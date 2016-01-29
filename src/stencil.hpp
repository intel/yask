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

#ifndef _STENCIL_INCLUDE
#define _STENCIL_INCLUDE

// control assert() with DEBUG instead of NDEBUG.
#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdexcept>

#include "mapping.hpp"

#ifndef WIN32
#include <unistd.h>
#include <stdint.h>
#include <immintrin.h>
#else
#include <map>
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

#ifdef USE_SEP
#include "sampling_MIC.h"
#define SEP_PAUSE  VTPauseSampling()
#define SEP_RESUME VTResumeSampling()
#else
#define SEP_PAUSE
#define SEP_RESUME
#endif

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

// define a vector of reals.
#include "stencil_macros.hpp"
#include "realv.hpp"

// number of var grids to be calculated.
#ifndef NUM_VARS
#define NUM_VARS (1)
#endif

// number of work grids.
#ifndef NUM_WORKS
#define NUM_WORKS (1)
#endif

// total grids.
#define NUM_GRIDS (NUM_VARS + NUM_WORKS)

// cluster size in points.
#define CPTS_X (CLEN_X * VLEN_X)
#define CPTS_Y (CLEN_Y * VLEN_Y)
#define CPTS_Z (CLEN_Z * VLEN_Z)
#define CPTS (CPTS_X * CPTS_Y * CPTS_Z)

// rounding macro.
#define ROUND_UP(n, mult) (((n) + (mult) - 1) / (mult) * (mult))

// Memory-accessing code.
#include "mem_wrappers.hpp"

// Matrix implementations.
// Vars are 1=x, 2=y, 3=z, 4=(v&t).
// Last number is consecutive in memory, e.g.,
// RealMatrix4321 is C-like with consecutive x indices,
// RealMatrix4123 is Fortran-like with consecutive z indices.
#ifndef MATRIX_BASE
#define MATRIX_BASE RealMatrix4321
#endif
#define MATRIX_TYPE MATRIX_BASE<NUM_GRIDS>
#define VEL_MAT_TYPE MATRIX_BASE<1>

// A grid using the MATRIX_TYPE
#include "Grid.hpp"

// Context for calculating values.
class StencilContext {
public:
    Grid5d* grid;               // main grid for current and previous values.
    const Grid3d* vel;          // velocity constants.

    // all sizes in vector-lengths.
    int dx, dy, dz;                   // problem size.
    int rx, ry, rz;                   // region size.
    int bx, by, bz;                   // block size.

    StencilContext() {
        grid = NULL;
        vel = NULL;
    }
};

// Some utility functions.
extern double getTimeInSecs();
extern int roundUp(int dim, int mult, string name);

// Scalar version of stencil calculation.
#include STENCIL_HEADER

// Reference stencil calculations.
void calc_steps_ref(StencilContext& context, const int nreps);

// Vectorized and blocked stencil calculations.
void calc_steps_opt(StencilContext& context, const int nreps);

#endif
