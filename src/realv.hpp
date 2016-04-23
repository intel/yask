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

// This file defines a union to use for folded vectors of floats or doubles.

#ifndef _REALV_H
#define _REALV_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
using namespace std;

// values for 32-bit, single-precision reals.
#if REAL_BYTES == 4
#define REAL float
#define CTRL_INT unsigned __int32
#define CTRL_IDX_MASK 0xf
#define CTRL_SEL_BIT 0x10
#define MMASK __mmask16
#ifdef USE_INTRIN256
#define VEC_ELEMS 8
#define INAME(op) _mm256_ ## op ## _ps
#define INAMEI(op) _mm256_ ## op ## _epi32
#define IMEM_TYPE float
#elif defined(USE_INTRIN512)
#define VEC_ELEMS 16
#define INAME(op) _mm512_ ## op ## _ps
#define INAMEI(op) _mm512_ ## op ## _epi32
#define IMEM_TYPE void
#endif

// values for 64-bit, double-precision reals.
#elif REAL_BYTES == 8
#define REAL double
#define CTRL_INT unsigned __int64
#define CTRL_IDX_MASK 0x7
#define CTRL_SEL_BIT 0x8
#define MMASK __mmask8
#ifdef USE_INTRIN256
#define VEC_ELEMS 4
#define INAME(op) _mm256_ ## op ## _pd
#define INAMEI(op) _mm256_ ## op ## _epi64
#define IMEM_TYPE double
#elif defined(USE_INTRIN512)
#define VEC_ELEMS 8
#define INAME(op) _mm512_ ## op ## _pd
#define INAMEI(op) _mm512_ ## op ## _epi64
#define IMEM_TYPE void
#endif

#else
#error "REAL_BYTES not set to 4 or 8"
#endif

// Emulate instrinsics for unsupported VLEN.
// Only 256 and 512-bit vectors supported.
// VLEN == 1 also supported as scalar.
#if VLEN == 1
#define NO_INTRINSICS
// note: no warning here because intrinsics aren't wanted in this case.

#elif !defined(VEC_ELEMS)
#warning "Emulating intrinsics because HW vector length not defined; set USE_INTRIN256 or USE_INTRIN512"
#define NO_INTRINSICS

#elif VLEN != VEC_ELEMS
#warning "Emulating intrinsics because VLEN != HW vector length"
#define NO_INTRINSICS
#endif

// Macro for looping through an aligned realv.
#if defined(DEBUG) || (VLEN==1)
#define SIMD_LOOP(i)                            \
    for (int i=0; i<VLEN; i++)
#define SIMD_LOOP_UNALIGNED(i)                  \
    for (int i=0; i<VLEN; i++)
#else
#define SIMD_LOOP(i)                            \
    _Pragma("vector aligned") _Pragma("simd")   \
    for (int i=0; i<VLEN; i++)
#define SIMD_LOOP_UNALIGNED(i)                  \
    _Pragma("simd")                             \
    for (int i=0; i<VLEN; i++)
#endif

// Type for a vector block.
// This must be an aggregate type to allow aggregate initialization,
// so no user-provided ctors, copy operator, virtual member functions, etc.
union realv {

    REAL r[VLEN];
    CTRL_INT ci[VLEN];

#ifndef NO_INTRINSICS
    
    // 32-bit integer vector overlay.
#if defined(USE_INTRIN256)
    __m256i mi;
#elif defined(USE_INTRIN512)
    __m512i mi;
#endif

    // real vector.
#if REAL_BYTES == 4 && defined(USE_INTRIN256)
    __m256  mr;
#elif REAL_BYTES == 4 && defined(USE_INTRIN512)
    __m512  mr;
#elif REAL_BYTES == 8 && defined(USE_INTRIN256)
    __m256d mr;
#elif REAL_BYTES == 8 && defined(USE_INTRIN512)
    __m512d mr;
#endif

#endif
    
    // access a REAL linearly.
    inline REAL& operator[](idx_t l) {
        assert(l >= 0);
        assert(l < VLEN);
        return r[l];
    }
    inline const REAL& operator[](idx_t l) const {
        assert(l >= 0);
        assert(l < VLEN);
        return r[l];
    }

    // access a REAL by n,x,y,z vector-block indices.
    inline const REAL& operator()(idx_t n, idx_t i, idx_t j, idx_t k) const {
        assert(n >= 0);
        assert(n < VLEN_N);
        assert(i >= 0);
        assert(i < VLEN_X);
        assert(j >= 0);
        assert(j < VLEN_Y);
        assert(k >= 0);
        assert(k < VLEN_Z);

#if VLEN_FIRST_DIM_IS_UNIT_STRIDE

        // n dim is unit stride, followed by x, y, z.
        idx_t l = MAP4321(n, i, j, k, VLEN_N, VLEN_X, VLEN_Y, VLEN_Z);
#else

        // z dim is unit stride, followed by y, x, n.
        idx_t l = MAP1234(n, i, j, k, VLEN_N, VLEN_X, VLEN_Y, VLEN_Z);
#endif
        
        return r[l];
    }
    inline REAL& operator()(idx_t n, idx_t i, idx_t j, idx_t k) {
        const realv* ct = const_cast<const realv*>(this);
        const REAL& cr = (*ct)(n, i, j, k);
        return const_cast<REAL&>(cr);
    }

    // copy whole vector.
    inline void copy(const realv& rhs) {
#ifdef NO_INTRINSICS
        SIMD_LOOP(i) r[i] = rhs[i];
#else
        mr = rhs.mr;
#endif
    }

    // assignment: single value broadcast.
    inline void operator=(REAL val) {
#ifdef NO_INTRINSICS
        SIMD_LOOP(i) r[i] = val;
#else
        mr = INAME(set1)(val);
#endif
    }

    // broadcast with conversions.
    inline void operator=(int val) {
        operator=(REAL(val));
    }
    inline void operator=(long val) {
        operator=(REAL(val));
    }
#if REAL_BYTES == 4
    inline void operator=(double val) {
        operator=(REAL(val));
    }
#else
    inline void operator=(float val) {
        operator=(REAL(val));
    }
#endif
    
    // unary negate.
    inline realv operator-() const {
        realv res;
#ifdef NO_INTRINSICS
        SIMD_LOOP(i) res[i] = -r[i];
#else
        res.mr = INAME(sub)(INAME(setzero)(), mr);
#endif
        return res;
    }

    // add.
    inline realv operator+(realv rhs) const {
        realv res;
#ifdef NO_INTRINSICS
        SIMD_LOOP(i) res[i] = r[i] + rhs[i];
#else
        res.mr = INAME(add)(mr, rhs.mr);
#endif
        return res;
    }
    inline realv operator+(REAL rhs) const {
        realv rn;
        rn = rhs;               // broadcast.
        return (*this) + rn;
    }

    // sub.
    inline realv operator-(realv rhs) const {
        realv res;
#ifdef NO_INTRINSICS
        SIMD_LOOP(i) res[i] = r[i] - rhs[i];
#else
        res.mr = INAME(sub)(mr, rhs.mr);
#endif
        return res;
    }
    inline realv operator-(REAL rhs) const {
        realv rn;
        rn = rhs;               // broadcast.
        return (*this) - rn;
    }

    // mul.
    inline realv operator*(realv rhs) const {
        realv res;
#ifdef NO_INTRINSICS
        SIMD_LOOP(i) res[i] = r[i] * rhs[i];
#else
        res.mr = INAME(mul)(mr, rhs.mr);
#endif
        return res;
    }
    inline realv operator*(REAL rhs) const {
        realv rn;
        rn = rhs;               // broadcast.
        return (*this) * rn;
    }
    
    // div.
    inline realv operator/(realv rhs) const {
        realv res;
#ifdef NO_INTRINSICS
        SIMD_LOOP(i) res[i] = r[i] / rhs[i];
#else
        res.mr = INAME(div)(mr, rhs.mr);
#endif
        return res;
    }
    inline realv operator/(REAL rhs) const {
        realv rn;
        rn = rhs;               // broadcast.
        return (*this) / rn;
    }

    // less-than comparator.
    bool operator<(const realv& rhs) const {
        for (int j = 0; j < VLEN; j++) {
            if (r[j] < rhs.r[j])
                return true;
            else if (r[j] > rhs.r[j])
                return false;
        }
        return false;
    }

    // greater-than comparator.
    bool operator>(const realv& rhs) const {
        for (int j = 0; j < VLEN; j++) {
            if (r[j] > rhs.r[j])
                return true;
            else if (r[j] < rhs.r[j])
                return false;
        }
        return false;
    }
    
    // equal-to comparator for validation.
    bool operator==(const realv& rhs) const {
        for (int j = 0; j < VLEN; j++) {
            if (r[j] != rhs.r[j])
                return false;
        }
        return true;
    }
    
    // aligned load.
    inline void loadFrom(const realv* restrict from) {
#if defined(NO_INTRINSICS) || defined(NO_LOAD_INTRINSICS)
        SIMD_LOOP(i) r[i] = (*from)[i];
#else
        mr = INAME(load)((IMEM_TYPE const*)from);
#endif
    }

    // unaligned load.
    inline void loadUnalignedFrom(const realv* restrict from) {
#if defined(NO_INTRINSICS) || defined(NO_LOAD_INTRINSICS)
        SIMD_LOOP_UNALIGNED(i) r[i] = (*from)[i];
#else
        mr = INAME(loadu)((IMEM_TYPE const*)from);
#endif
    }

    // aligned store.
    inline void storeTo(realv* restrict to) const {
            SIMD_LOOP(i) (*to)[i] = r[i];
#if defined(NO_INTRINSICS) || defined(NO_STORE_INTRINSICS)
#if defined(__INTEL_COMPILER) && (VLEN > 1) && defined(USE_STREAMING_STORE)
        _Pragma("vector nontemporal")
#endif
        SIMD_LOOP(i) (*to)[i] = r[i];
#elif !defined(USE_STREAMING_STORE)
        INAME(store)((IMEM_TYPE*)to, mr);
#elif defined(ARCH_KNC)
        INAME(storenrngo)((IMEM_TYPE*)to, mr);
#else
        INAME(stream)((IMEM_TYPE*)to, mr);
#endif
    }

    // Output.
    inline void print_ctrls(ostream& os, bool doEnd=true) const {
        for (int j = 0; j < VLEN; j++) {
            if (j) os << ", ";
            os << "[" << j << "]=" << ci[j];
        }
        if (doEnd)
            os << endl;
    }

    inline void print_reals(ostream& os, bool doEnd=true) const {
        for (int j = 0; j < VLEN; j++) {
            if (j) os << ", ";
            os << "[" << j << "]=" << r[j];
        }
        if (doEnd)
            os << endl;
    }

}; // realv.

// Output using '<<'.
inline ostream& operator<<(ostream& os, const realv& rn) {
    rn.print_reals(os, false);
    return os;
}

// Compare two realv's.
inline bool within_tolerance(const realv& val, const realv& ref,
                             const realv& epsilon) {
        for (int j = 0; j < VLEN; j++) {
            if (!within_tolerance(val.r[j], ref.r[j], epsilon.r[j]))
                return false;
        }
        return true;
}

// wrappers around some intrinsics w/non-intrinsic equivalents.
// TODO: make these methods in the realv union.

// Get consecutive elements from two vectors.
// Concat a and b, shift right by count elements, keep rightmost elements.
// Thus, shift of 0 returns b; shift of VLEN returns a.
ALWAYS_INLINE void realv_align(realv& res, const realv& a, const realv& b,
                                  const int count) {
#ifdef TRACE_INTRINSICS
    cout << "realv_align w/count=" << count << ":" << endl;
    cout << " a: ";
    a.print_reals(cout);
    cout << " b: ";
    b.print_reals(cout);
#endif

#if defined(NO_INTRINSICS)
    // must make temp copies in case &res == &a or &b.
    realv tmpa = a, tmpb = b;
    for (int i = 0; i < VLEN-count; i++)
        res.r[i] = tmpb.r[i + count];
    for (int i = VLEN-count; i < VLEN; i++)
        res.r[i] = tmpa.r[i + count - VLEN];
    
#elif defined(USE_INTRIN256)
    // Not really an intrinsic, but not element-wise, either.
    // Put the 2 parts in a local array, then extract the desired part
    // using an unaligned load.
    REAL r2[VLEN * 2];
    *((realv*)(&r2[0])) = b;
    *((realv*)(&r2[VLEN])) = a;
    res = *((realv*)(&r2[count]));
    
#elif REAL_BYTES == 8 && defined(ARCH_KNC) && defined(USE_INTRIN512)
    // For KNC, for 64-bit align, use the 32-bit op w/2x count.
    res.mi = _mm512_alignr_epi32(a.mi, b.mi, count*2);

#else
    res.mi = INAMEI(alignr)(a.mi, b.mi, count);
#endif

#ifdef TRACE_INTRINSICS
    cout << " res: ";
    res.print_reals(cout);
#endif
}

// Get consecutive elements from two vectors w/masking.
// Concat a and b, shift right by count elements, keep rightmost elements.
// Elements in res corresponding to 0 bits in k1 are unchanged.
ALWAYS_INLINE void realv_align(realv& res, const realv& a, const realv& b,
                               const int count, unsigned int k1) {
#ifdef TRACE_INTRINSICS
    cout << "realv_align w/count=" << count << " w/mask:" << endl;
    cout << " a: ";
    a.print_reals(cout);
    cout << " b: ";
    b.print_reals(cout);
    cout << " res(before): ";
    res.print_reals(cout);
    cout << " mask: 0x" << hex << k1 << endl;
#endif

#ifdef NO_INTRINSICS
    // must make temp copies in case &res == &a or &b.
    realv tmpa = a, tmpb = b;
    for (int i = 0; i < VLEN-count; i++)
        if ((k1 >> i) & 1)
            res.r[i] = tmpb.r[i + count];
    for (int i = VLEN-count; i < VLEN; i++)
        if ((k1 >> i) & 1)
            res.r[i] = tmpa.r[i + count - VLEN];
#else
    res.mi = INAMEI(mask_alignr)(res.mi, MMASK(k1), a.mi, b.mi, count);
#endif

#ifdef TRACE_INTRINSICS
    cout << " res(after): ";
    res.print_reals(cout);
#endif
}

// Rearrange elements in a vector.
ALWAYS_INLINE void realv_permute(realv& res, const realv& ctrl, const realv& a) {

#ifdef TRACE_INTRINSICS
    cout << "realv_permute:" << endl;
    cout << " ctrl: ";
    ctrl.print_ctrls(cout);
    cout << " a: ";
    a.print_reals(cout);
#endif

#ifdef NO_INTRINSICS
    // must make a temp copy in case &res == &a.
    realv tmp = a;
    for (int i = 0; i < VLEN; i++)
        res.r[i] = tmp.r[ctrl.ci[i]];
#else
    res.mi = INAMEI(permutexvar)(ctrl.mi, a.mi);
#endif

#ifdef TRACE_INTRINSICS
    cout << " res: ";
    res.print_reals(cout);
#endif
}

// Rearrange elements in a vector w/masking.
// Elements in res corresponding to 0 bits in k1 are unchanged.
ALWAYS_INLINE void realv_permute(realv& res, const realv& ctrl, const realv& a,
                                 unsigned int k1) {
#ifdef TRACE_INTRINSICS
    cout << "realv_permute w/mask:" << endl;
    cout << " ctrl: ";
    ctrl.print_ctrls(cout);
    cout << " a: ";
    a.print_reals(cout);
    cout << " mask: 0x" << hex << k1 << endl;
    cout << " res(before): ";
    res.print_reals(cout);
#endif

#ifdef NO_INTRINSICS
    // must make a temp copy in case &res == &a.
    realv tmp = a;
    for (int i = 0; i < VLEN; i++) {
        if ((k1 >> i) & 1)
            res.r[i] = tmp.r[ctrl.ci[i]];
    }
#else
    res.mi = INAMEI(mask_permutexvar)(res.mi, MMASK(k1), ctrl.mi, a.mi);
#endif

#ifdef TRACE_INTRINSICS
    cout << " res(after): ";
    res.print_reals(cout);
#endif
}

// Rearrange elements in 2 vectors.
// (The masking versions of these instrs do not preserve the source,
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

#ifdef NO_INTRINSICS
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
#else
    res.mi = INAMEI(permutex2var)(a.mi, ctrl.mi, b.mi);
#endif

#ifdef TRACE_INTRINSICS
    cout << " res: ";
    res.print_reals(cout);
#endif
}

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


#endif
