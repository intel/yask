/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2022, Intel Corporation

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
#pragma once
#include <stdio.h>

// Hack macros for the C preprocessor.
#define YSTR1(s) #s
#define YSTR2(s) YSTR1(s)
#define YPRAGMA(x) _Pragma(#x)
#define YCAT(a, ...) YPRIM_CAT(a, __VA_ARGS__)
#define YPRIM_CAT(a, ...) a ## __VA_ARGS__
#define YIIF(c) YPRIM_CAT(IIF_, c)
#define YIIF_0(t, ...) __VA_ARGS__
#define YIIF_1(t, ...) t

// Control assert() by turning on with CHECK instead of turning off with
// NDEBUG. This makes it off by default.
#ifdef CHECK

// Temporarily replace assert() with printf() when offloading, but
// this doesn't cause program to halt.
// Also define host_assert() to be a stub.
#if defined(USE_OFFLOAD) && !defined(USE_OFFLOAD_X86)
#define assert(expr)                                                    \
    ((expr) ?                                                           \
     ((void)0) :                                                        \
     ((void)printf("YASK: ***** assertion '%s' failed at %s:%i\n",      \
                   YSTR1(expr), __FILE__, __LINE__)))
#define host_assert(expr) ((void)0)
#else
#include <cassert>
#define host_assert(expr) assert(expr)
#endif

#else
#define assert(expr) ((void)0)
#define host_assert(expr) ((void)0)
#define NDEBUG
#endif

