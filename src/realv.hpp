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

// This file defines macros and functions to use for vectors of floats or doubles.

#ifndef _REALV_H
#define _REALV_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
using namespace std;

#if REAL_BYTES == 4
#define REAL float
#define CTRL_INT unsigned __int32
#define CTRL_SEL_BIT 0x8
#define CTRL_IDX_MASK 0x7
#define V512_ELEMS 16
#define MMASK __mmask16
#elif REAL_BYTES == 8
#define REAL double
#define CTRL_INT unsigned __int64
#define CTRL_SEL_BIT 0x10
#define CTRL_IDX_MASK 0xf
#define V512_ELEMS 8
#define MMASK __mmask8
#else
#error "REAL_BYTES not set to 4 or 8"
#endif

// Emulate instrinsics for unsupported VLEN.
// Only 512-bit vectors supported.
#if VLEN != V512_ELEMS
#warning "Emulating intrinsics because VLEN elements != 512 bits"
#define EMU_INTRINSICS
#elif !defined(INTRIN512)
#warning "Emulating 512-bit intrinsics because INTRIN512 is not defined"
#define EMU_INTRINSICS
#endif

// Macro for looping through an aligned realv.
#ifdef DEBUG
#define SIMD_LOOP(i)                            \
    for (int i=0; i<VLEN; i++)
#else
#define SIMD_LOOP(i)                            \
    _Pragma("vector aligned") _Pragma("simd")   \
    for (int i=0; i<VLEN; i++)
#endif

// Type for a vector block.
// This must be an aggregate type to allow aggregate initialization,
// so no user-provided ctors, copy operator, virtual member functions, etc.
union realv {

    REAL r[VLEN];
    CTRL_INT ci[VLEN];

#ifndef EMU_INTRINSICS
    __m512i m512i;
#if REAL_BYTES == 4
    __m512  m512r;
#else
    __m512d m512r;
#endif
#endif

    // access a REAL linearly.
    inline REAL& operator[](int l) {
        return r[l];
    }
    inline const REAL& operator[](int l) const {
        return r[l];
    }

    // access a REAL by x,y,z vector-block indices.
    inline REAL& operator()(unsigned i, unsigned j, unsigned k) {
        unsigned l = MAP321(i, j, k, VLEN_X, VLEN_Y, VLEN_Z);
        return r[l];
    }
    inline const REAL& operator()(unsigned i, unsigned j, unsigned k) const {
        unsigned l = MAP321(i, j, k, VLEN_X, VLEN_Y, VLEN_Z);
        return r[l];
    }

    // copy whole vector.
    inline void copy(const realv& rhs) {
#ifdef EMU_INTRINSICS
        SIMD_LOOP(i) r[i] = rhs[i];
#else
        m512r = rhs.m512r;
#endif
    }

    // assignment: single value broadcast.
    inline void operator=(REAL val) {
#ifdef EMU_INTRINSICS
        SIMD_LOOP(i) r[i] = val;
#elif REAL_BYTES == 4
        m512r = _mm512_set1_ps(val);
#else
        m512r = _mm512_set1_pd(val);
#endif
    }

    // negate.
    inline realv operator-() const {
        realv res;
#ifdef EMU_INTRINSICS
        SIMD_LOOP(i) res[i] = -r[i];
#elif REAL_BYTES == 4
        res.m512r = _mm512_sub_ps(_mm512_setzero_ps(), this->m512r);
#else
        res.m512r = _mm512_sub_pd(_mm512_setzero_pd(), this->m512r);
#endif
        return res;
    }

    // add.
    inline realv operator+(realv rhs) const {
        realv res;
#ifdef EMU_INTRINSICS
        SIMD_LOOP(i) res[i] = r[i] + rhs[i];
#elif REAL_BYTES == 4
        res.m512r = _mm512_add_ps(this->m512r, rhs.m512r);
#else
        res.m512r = _mm512_add_pd(this->m512r, rhs.m512r);
#endif
        return res;
    }

    // sub.
    inline realv operator-(realv rhs) const {
        realv res;
#ifdef EMU_INTRINSICS
        SIMD_LOOP(i) res[i] = r[i] - rhs[i];
#elif REAL_BYTES == 4
        res.m512r = _mm512_sub_ps(this->m512r, rhs.m512r);
#else
        res.m512r = _mm512_sub_pd(this->m512r, rhs.m512r);
#endif
        return res;
    }

    // mul.
    inline realv operator*(realv rhs) const {
        realv res;
#ifdef EMU_INTRINSICS
        SIMD_LOOP(i) res[i] = r[i] * rhs[i];
#elif REAL_BYTES == 4
        res.m512r = _mm512_mul_ps(this->m512r, rhs.m512r);
#else
        res.m512r = _mm512_mul_pd(this->m512r, rhs.m512r);
#endif
        return res;
    }

    // div.
    inline realv operator/(realv rhs) const {
        realv res;
#ifdef EMU_INTRINSICS
        SIMD_LOOP(i) res[i] = r[i] / rhs[i];
#elif REAL_BYTES == 4
        res.m512r = _mm512_div_ps(this->m512r, rhs.m512r);
#else
        res.m512r = _mm512_div_pd(this->m512r, rhs.m512r);
#endif
        return res;
    }

    // load/store.
    // TODO: allow optional use of intrinsics like _mm512_load_ps.
    inline void loadFrom(const realv* from) {
        *this = *from;
    }
    inline void storeTo(realv* to) const {
#ifdef __INTEL_COMPILER
        _Pragma("vector nontemporal")
            SIMD_LOOP(i) (*to)[i] = r[i];
#else
            *to = *this;
#endif
    }

    // Output.
    inline void print_ctrls(ostream& os) const {
        for (int j = 0; j < VLEN; j++) {
            if (j) os << ",";
            os << "[" << j << "]=" << ci[j];
        }
        os << endl;
    }

    inline void print_reals(ostream& os) const {
        for (int j = 0; j < VLEN; j++) {
            if (j) os << ",";
            os << " [" << j << "]=" << r[j];
        }
        os << endl;
    }
    
}; // realv.

#ifdef __INTEL_COMPILER
#define ALIGNED_REALV __declspec(align(sizeof(realv))) realv
#else
#define ALIGNED_REALV realv __attribute__((aligned(sizeof(realv)))) 
#endif

// zero a VLEN-sized array.
#define ZERO_VEC(v) do {                        \
        SIMD_LOOP(i)                            \
            v[i] = (REAL)0.0;                   \
    } while(0)

// declare and zero a VLEN-sized array.
#define MAKE_VEC(v)                             \
    ALIGNED_REALV v(0.0)

// wrappers around some intrinsics w/non-intrinsic equivalents.
// TODO: move these into the realv union.

// get consecutive elements from two vectors.
ALWAYS_INLINE void realv_align(realv& res, const realv& v2, const realv& v3,
                                  const int count) {
#ifdef TRACE_INTRINSICS
    cout << "realv_align w/count=" << count << ":" << endl;
    cout << " v2: ";
    v2.print_reals(cout);
    cout << " v3: ";
    v3.print_reals(cout);
#endif

#ifdef EMU_INTRINSICS
    // (v2[VLEN-1], ..., v2[0], v3[VLEN-1], ..., v3[0]) >> count*32b.
    for (int i = 0; i < VLEN-count; i++)
        res.r[i] = v3.r[i + count];
    for (int i = VLEN-count; i < VLEN; i++)
        res.r[i] = v2.r[i + count - VLEN];
#elif REAL_BYTES == 4
    res.m512i = _mm512_alignr_epi32(v2.m512i, v3.m512i, count);
#elif defined(ARCH_KNC)
    // For KNC, for 64-bit align, have to use the 32-bit w/2x count.
    res.m512i = _mm512_alignr_epi32(v2.m512i, v3.m512i, count*2);
#else
    res.m512i = _mm512_alignr_epi64(v2.m512i, v3.m512i, count);
#endif

#ifdef TRACE_INTRINSICS
    cout << " res: ";
    res.print_reals(cout);
#endif
}

// get consecutive elements from two vectors w/masking.
ALWAYS_INLINE void realv_align(realv& res, const realv& v2, const realv& v3,
                               const int count, MMASK k1) {
#ifdef TRACE_INTRINSICS
    cout << "realv_align w/count=" << count << " w/mask:" << endl;
    cout << " v2: ";
    v2.print_reals(cout);
    cout << " v3: ";
    v3.print_reals(cout);
    cout << " res(before): ";
    res.print_reals(cout);
    cout << " mask: 0x" << hex << k1 << endl;
#endif

#ifdef EMU_INTRINSICS
    // (v2[VLEN-1], ..., v2[0], v3[VLEN-1], ..., v3[0]) >> count*32b.
    for (int i = 0; i < VLEN-count; i++)
        if ((k1 >> i) & 1)
            res.r[i] = v3.r[i + count];
    for (int i = VLEN-count; i < VLEN; i++)
        if ((k1 >> i) & 1)
            res.r[i] = v2.r[i + count - VLEN];
#elif REAL_BYTES == 4
    res.m512i = _mm512_mask_alignr_epi32(res.m512i, k1, v2.m512i, v3.m512i, count);
#elif defined(ARCH_KNC)
    cerr << "error: 64-bit align w/mask not supported on KNC" << endl;
    exit(1);
#else
    res.m512i = _mm512_mask_alignr_epi64(res.m512i, k1, v2.m512i, v3.m512i, count);
#endif

#ifdef TRACE_INTRINSICS
    cout << " res(after): ";
    res.print_reals(cout);
#endif
}

// rearrange elements in a vector.
ALWAYS_INLINE void realv_permute(realv& res, const realv& ctrl, const realv& v3) {

#ifdef TRACE_INTRINSICS
    cout << "realv_permute:" << endl;
    cout << " ctrl: ";
    ctrl.print_ctrls(cout);
    cout << " v3: ";
    v3.print_reals(cout);
#endif

#ifdef EMU_INTRINSICS
    // must make a temp copy in case &res == &v3.
    realv tmp = v3;
    for (int i = 0; i < VLEN; i++)
        res.r[i] = tmp.r[ctrl.ci[i]];
#elif REAL_BYTES == 4
    res.m512i = _mm512_permutevar_epi32(ctrl.m512i, v3.m512i);
#elif defined(ARCH_KNC)
    cerr << "error: 64-bit permute not supported on KNC" << endl;
    exit(1);
#else
    res.m512i = _mm512_permutexvar_epi64(ctrl.m512i, v3.m512i);
#endif

#ifdef TRACE_INTRINSICS
    cout << " res: ";
    res.print_reals(cout);
#endif
}

// rearrange elements in a vector w/masking.
ALWAYS_INLINE void realv_permute(realv& res, const realv& ctrl, const realv& v3,
                                 __mmask16 k1) {
#ifdef TRACE_INTRINSICS
    cout << "realv_permute w/mask:" << endl;
    cout << " ctrl: ";
    ctrl.print_ctrls(cout);
    cout << " v3: ";
    v3.print_reals(cout);
    cout << " mask: 0x" << hex << k1 << endl;
    cout << " res(before): ";
    res.print_reals(cout);
#endif

#ifdef EMU_INTRINSICS
    // must make a temp copy in case &res == &v3.
    realv tmp = v3;
    for (int i = 0; i < VLEN; i++) {
        if ((k1 >> i) & 1)
            res.r[i] = tmp.r[ctrl.ci[i]];
    }
#elif REAL_BYTES == 4
    res.m512i = _mm512_mask_permutevar_epi32(res.m512i, k1, ctrl.m512i, v3.m512i);
#elif defined(ARCH_KNC)
    cerr << "error: 64-bit permute w/mask not supported on KNC" << endl;
    exit(1);
#else
    res.m512i = _mm512_mask_permutexvar_epi64(res.m512i, k1, ctrl.m512i, v3.m512i);
#endif

#ifdef TRACE_INTRINSICS
    cout << " res(after): ";
    res.print_reals(cout);
#endif
}

// rearrange elements in 2 vectors.
// (the masking versions of these instrs do not preserve the source,
// so we don't have a masking version of this function.)
ALWAYS_INLINE void realv_permute2(realv& res, const realv& ctrl,
                                  const realv& a, const realv& b) {
#ifdef TRACE_INTRINSICS
    cout << "realv_permute2:" << endl;
    cout << " ctrl: ";
    ctrl.print_ctrls(cout);
    cout << " a: ";
    a.print_reals(cout);
    cout << " b: ";
    b.print_reals(cout);
#endif

#ifdef EMU_INTRINSICS
    // must make temp copies in case &res == &a or &b.
    realv tmpa = a, tmpb = b;
    for (int i = 0; i < VLEN; i++) {
        int sel = ctrl.ci[i] & CTRL_SEL_BIT; // 0 => a, 1 => b.
        int idx = ctrl.ci[i] & CTRL_IDX_MASK; // index.
        res.r[i] = sel ? tmpb.r[idx] : tmpa.r[idx];
    }

#elif defined(ARCH_KNC)
    cerr << "error: 2-input permute not supported on KNC" << endl;
    exit(1);
#elif REAL_BYTES == 4
    res.m512i = _mm512_permutex2var_epi32(a.m512i, ctrl.m512i, b.m512i);
#else
    res.m512i = _mm512_permutex2var_epi64(a.m512i, ctrl.m512i, b.m512i);
#endif

#ifdef TRACE_INTRINSICS
    cout << " res: ";
    res.print_reals(cout);
#endif
}

#endif
