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

// This file defines macros to use for accessing memory.
// By using these, you can turn on the following:
// - use the cache model when CACHE_MODEL is set to 1 or 2.
// - trace accesses when TRACE_MEM is set.
// - check alignment when DEBUG is set.
// If none of these are activated, there is no cost.
// Also defines macros for PREFETCH and EVICT.

#ifndef MEM_MACROS
#define MEM_MACROS

#define CACHELINE_BYTES  (64)
#define YASK_PAD (17) // cache-lines between data buffers.
#define YASK_ALIGNMENT (2 * 1024 * 1024) // 2MiB-page

    // Set MODEL_CACHE to 1 or 2 to model L1 or L2.
#ifdef MODEL_CACHE
#include "cache_model.hpp"
#endif

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
