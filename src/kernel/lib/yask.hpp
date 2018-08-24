/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

// Choose features
#define _POSIX_C_SOURCE 200809L

// Include the API first. This helps to ensure that it will stand alone.
#include "yask_kernel_api.hpp"

// Control assert() by turning on with CHECK instead of turning off with
// NDEBUG. This makes it off by default.
#ifndef CHECK
#define NDEBUG
#endif

// Standard C and C++ headers.
#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits.h>
#include <malloc.h>
#include <map>
#include <math.h>
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
#ifdef USE_PMEM
#include <memkind.h>
#include <memkind_pmem.h>
#endif

// Additional type for unsigned indices.
typedef std::uint64_t uidx_t;

// Common utilities.
#include "common_utils.hpp"

// Floored integer divide and mod.
#include "idiv.hpp"

// Simple macros and stubs.

#ifdef WIN32
#define _Pragma(x)
#endif

#if defined(__GNUC__) && !defined(__ICC)
#define __assume(x) ((void)0)
//#define __declspec(x)
#endif

#if (defined(__GNUC__) && !defined(__ICC)) || defined(WIN32)
#define restrict
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

// MPI or stubs.
#ifdef USE_MPI
#include "mpi.h"
#else
#define MPI_PROC_NULL (-1)
#define MPI_Barrier(comm) ((void)0)
#define MPI_Comm int
#define MPI_Finalize() ((void)0)
#define MPI_Request int
#define MPI_REQUEST_NULL   ((MPI_Request)0x2c000000)
#endif

// OpenMP or stubs.
#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_num_procs() { return 1; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline void omp_set_num_threads(int n) { }
inline void omp_set_nested(int n) { }
#endif

// Stringizing hacks for the C preprocessor.
#define YSTR1(s) #s
#define YSTR2(s) YSTR1(s)

// Rounding macros for integer types.
#define CEIL_DIV(numer, denom) (((numer) + (denom) - 1) / (denom))
#define ROUND_UP(n, mult) (CEIL_DIV(n, mult) * (mult))
#define ROUND_DOWN(n, mult) (((n) / (mult)) * (mult))

// Default alloc settings.
#define CACHELINE_BYTES  (64)
#define YASK_PAD (7) // cache-lines between data buffers.
#define YASK_HUGE_ALIGNMENT (2 * 1024 * 1024) // 2MiB-page for large allocs.
#define CACHE_ALIGNED __attribute__ ((aligned (CACHELINE_BYTES)))
#ifndef USE_NUMA
#undef NUMA_PREF
#define NUMA_PREF yask_numa_none
#elif !defined NUMA_PREF
#define NUMA_PREF yask_numa_local
#endif

// macro for debug message.
#ifdef TRACE
#define TRACE_MSG0(os, msg) ((os) << "YASK: " << msg << std::endl << std::flush)
#else
#define TRACE_MSG0(os, msg) ((void)0)
#endif

// macro for debug message from a StencilContext method.
#define TRACE_MSG1(msg) TRACE_MSG0(get_ostr(), msg)
#define TRACE_MSG(msg) TRACE_MSG1(msg)

// macro for debug message when _context ptr is defined.
#define TRACE_MSG2(msg) TRACE_MSG0(_context->get_ostr(), msg)

// macro for debug message when _generic_context ptr is defined.
#define TRACE_MSG3(msg) TRACE_MSG0(_generic_context->get_ostr(), msg)

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


