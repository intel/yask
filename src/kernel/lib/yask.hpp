/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2019, Intel Corporation

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

// This file defines functions, types, and macros needed for common
// (non-stencil-specific) code. This file does not input generated files.

#pragma once
#include "yask_assert.hpp"

// Choose features
#define _POSIX_C_SOURCE 200809L

// MPI or stubs.
// This must come before including the API header to make sure
// MPI_VERSION is defined.
#ifdef USE_MPI
#include "mpi.h"
#else
#define MPI_Barrier(comm) ((void)0)
#define MPI_Finalize()    ((void)0)
typedef int MPI_Comm;
typedef int MPI_Win;
typedef int MPI_Group;
typedef int MPI_Request;
#define MPI_PROC_NULL     (-1)
#define MPI_COMM_NULL     ((MPI_Comm)0x04000000)
#define MPI_REQUEST_NULL  ((MPI_Request)0x2c000000)
#define MPI_GROUP_NULL    ((MPI_Group)0x08000000)
#ifdef MPI_VERSION
#undef MPI_VERSION
#endif
#endif

// Include the API as early as possible. This helps to ensure that it will stand alone.
#include "yask_kernel_api.hpp"

// Standard C and C++ headers.
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits.h>
#include <malloc.h>
#include <map>
#include <unordered_map>
#include <set>
#include <sstream>
#include <stddef.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>
#include <unistd.h>
#include <stdint.h>
#include <immintrin.h>
#include <sys/mman.h>
#ifdef USE_PMEM
#include <memkind.h>
#include <sys/syscall.h>
#endif

// Conditional inlining
#if defined(USE_ALWAYS_INLINE) && !defined(CHECK)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define ALWAYS_INLINE inline
#endif

// Additional type for unsigned indices.
typedef std::uint64_t uidx_t;

// Common utilities.
#include "common_utils.hpp"

// Floored integer divide and mod.
#include "idiv.hpp"

// Combinations.
#include "combo.hpp"

// Simple macros and stubs.

#ifdef WIN32
#define _Pragma(x)
#endif

#ifndef NO_VEC
#define _NO_VECTOR _Pragma("novector")
#define _VEC_ALIGNED _Pragma("vector aligned")
#define _VEC_UNALIGNED _Pragma("vector unaligned")
#define _VEC_ALWAYS _Pragma("vector always")
#define _VEC_STREAMING _Pragma("vector nontemporal")
#else
#define _NO_VECTOR
#define _VEC_ALIGNED
#define _VEC_UNALIGNED
#define _VEC_ALWAYS
#define _VEC_STREAMING
#endif

#ifndef NO_SIMD
#define _SIMD _Pragma("omp simd")
#else
#define _SIMD
#endif

#ifndef NO_UNROLL
#define _UNROLL _Pragma("unroll")
#else
#define _UNROLL
#endif

#ifdef NO_ASSUME
#define __assume(x) ((void)0)
#define __assume_aligned(p,n) ((void)0)
#endif

// VTune or stubs.
#ifdef USE_VTUNE
#include "ittnotify.h"
#define VTUNE_PAUSE  __itt_pause()
#define VTUNE_RESUME __itt_resume()
#else
#define VTUNE_PAUSE ((void)0)
#define VTUNE_RESUME ((void)0)
#endif

// Stringizing hacks for the C preprocessor.
#define YSTR1(s) #s
#define YSTR2(s) YSTR1(s)

// Default alloc settings.
#define CACHELINE_BYTES  (64)
#define YASK_PAD (3) // cache-lines between data buffers.
#define YASK_PAD_BYTES (CACHELINE_BYTES * YASK_PAD)
#define YASK_HUGE_ALIGNMENT (2 * 1024 * 1024) // 2MiB-page for large allocs.
#define CACHE_ALIGNED __attribute__ ((aligned (CACHELINE_BYTES)))
#ifndef USE_NUMA
#undef NUMA_PREF
#define NUMA_PREF yask_numa_none
#elif !defined NUMA_PREF
#define NUMA_PREF yask_numa_local
#endif

// Macro for debug message.

// 'os is an ostream.
#define DEBUG_MSG0(os, msg) do {                            \
        KernelEnv::set_debug_lock();                        \
        (os) << std::boolalpha << std::dec <<               \
            msg << std::endl << std::flush;                 \
        KernelEnv::unset_debug_lock();                      \
    } while(0)

// 'state' is a pointer to a KernelState.
#define DEBUG_MSG1(state, msg) do {                            \
    if (state->_env->_debug.get()) {                           \
        auto& os = state->_env->_debug.get()->get_ostream();   \
        DEBUG_MSG0(os, msg);                                   \
    } } while(0)

// Macro for debug message when 'state' is defined.
#define DEBUG_MSG(msg) DEBUG_MSG1(state, msg)

// Macro for trace message.
// Enabled only if compiled with TRACE macro and run with -trace option.
#ifdef TRACE

// 'os is an ostream.
#define TRACE_MSG0(os, msg) do {                            \
        if (state->_env->_trace) {                          \
            DEBUG_MSG0(os, "YASK: " << msg);                \
        } } while(0)

// 'state' is a pointer to a KernelState.
#define TRACE_MSG1(state, msg) do {                         \
        if (state->_env->_trace) {                          \
            DEBUG_MSG1(state, "YASK: " << msg);             \
        } } while(0)
#else
#define TRACE_MSG0(os, msg) ((void)0)
#define TRACE_MSG1(state, msg) ((void)0)
#endif

// Macro for trace message when 'state' is defined.
#define TRACE_MSG(msg) TRACE_MSG1(state, msg)

// Macro for mem-trace when 'state' is defined.
// Enabled only if compiled with TRACE_MEM macro and run with -trace option.
#ifdef TRACE_MEM
#define TRACE_MEM_MSG(msg) TRACE_MSG1(state, msg)
#else
#define TRACE_MEM_MSG(msg) ((void)0)
#endif

// breakpoint.
#define INT3 asm volatile("int $3")

// L1 and L2 hints
#define L1_HINT _MM_HINT_T0
#define L2_HINT _MM_HINT_T1

// Set MODEL_CACHE to 1 or 2 to model L1 or L2.
#ifdef MODEL_CACHE
#include "cache_model.hpp"
extern yask::Cache cache_model;
 #if MODEL_CACHE==L1
  #warning Modeling L1 cache
 #elif MODEL_CACHE==L2
  #warning Modeling L2 cache
 #else
  #warning Modeling UNKNOWN cache
 #endif
#endif

// Other utilities.
#include "utils.hpp"
#include "tuple.hpp"


