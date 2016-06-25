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

// This file defines macros to use for accessing memory.
// By using these, you can turn on the following:
// - use the cache model when CACHE_MODEL is set to 1 or 2.
// - trace accesses when TRACE_MEM is set.
// - check alignment when DEBUG is set.
// If none of these are activated, there is no cost.
// Also defines macros for PREFETCH and EVICT.

#ifndef MEM_MACROS
#define MEM_MACROS

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>

#define CACHELINE_BYTES   64

    // Set MODEL_CACHE to 1 or 2 to model L1 or L2.
#ifdef MODEL_CACHE
#include "cache_model.hpp"
#endif

    //#define ALLOC_ALIGNMENT CACHELINE_BYTES
#define ALLOC_ALIGNMENT 4096 // 4k-page

    // Make an index and offset canonical, i.e., offset in [0..vecLen-1].
    // Makes proper adjustments for negative inputs.
#define FIX_INDEX_OFFSET(indexIn, offsetIn, indexOut, offsetOut, vecLen) \
    do {                                                                \
        const idx_t ofs = ((indexIn) * (vecLen)) + (offsetIn);          \
        indexOut = ofs / (vecLen);                                      \
        offsetOut = ofs % (vecLen);                                     \
        while(offsetOut < 0) {                                          \
            offsetOut += (vecLen);                                      \
            indexOut--;                                                 \
        }                                                               \
    } while(0)

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
#define TP(p, hint, base, matNum, xv, yv, zv, line)                     \
    printf("prefetch %s[%i][%i,%i,%i](%p) to L%i at line %i.\n",        \
           base.getNameCStr(), matNum, (int)xv (int)yv, (int)zv, p, hint, line); \
    fflush(stdout)
#else 
#define TP(p, hint, base, matNum, xv, y, z, line)
#endif

#define PREFETCH(hint, base, matNum, xv, yv, zv, line)                  \
    do {                                                                \
        const real_t *p = base.getPtr(matNum, xv, yv, zv, false);         \
        TP(p, hint, base, matNum, xv, yv, zv, line); MCP(p, hint, line); \
        _mm_prefetch((const char*)(p), hint);                           \
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

#define EVICT(hint, base, matNum, xv, yv, zv, line)                     \
        do {                                                            \
            const real_t *p = base.getPtr(matNum, xv, yv, zv, false);     \
            TE(p, hint, line); MCE(p, hint, line);                      \
            _mm_clevict((const char*)(p), hint);                        \
        } while(0)
#endif

        ////// Default prefetch distances.
        // These are only used if and when prefetch code is generated
        // by gen-loops.pl.

        // how far to prefetch ahead for L1.
#ifndef PFDL1
#define PFDL1 1
#endif

        // how far to prefetch ahead for L2.
#ifndef PFDL2
#define PFDL2 2
#endif

#endif
