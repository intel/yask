/*****************************************************************************

YASK: Yet Another Stencil Kernel
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

// This file defines a union to use for folded vectors of floats or doubles.

#ifndef _REAL_VEC_H
#define _REAL_VEC_H

namespace yask {

    // Max and min indices.
    const idx_t idx_max = INT64_MAX;
    const idx_t idx_min = INT64_MIN;

    // values for 32-bit, single-precision reals.
#if REAL_BYTES == 4
    typedef float real_t;
    typedef ::uint32_t ctrl_t;
    const ctrl_t ctrl_idx_mask = 0xf;
    const ctrl_t ctrl_sel_bit = 0x10;
#ifdef USE_INTRIN256
    typedef __m256 simd_t;
    typedef __m256i isimd_t;
    typedef float imem_t;
#define VEC_ELEMS 8
#define INAME(op) _mm256_ ## op ## _ps
#define INAMEI(op) _mm256_ ## op ## _epi32
#elif defined(USE_INTRIN512)
    typedef __m512 simd_t;
    typedef __m512i isimd_t;
    typedef void imem_t;
    typedef __mmask16 real_mask_t;
#define VEC_ELEMS 16
#define INAME(op) _mm512_ ## op ## _ps
#define INAMEI(op) _mm512_ ## op ## _epi32
#endif

    // values for 64-bit, double-precision reals.
#elif REAL_BYTES == 8
    typedef double real_t;
    typedef ::uint64_t ctrl_t;
    const ctrl_t ctrl_idx_mask = 0x7;
    const ctrl_t ctrl_sel_bit = 0x8;
#ifdef USE_INTRIN256
    typedef __m256d simd_t;
    typedef __m256i isimd_t;
    typedef double imem_t;
#define VEC_ELEMS 4
#define INAME(op) _mm256_ ## op ## _pd
#define INAMEI(op) _mm256_ ## op ## _epi64
#elif defined(USE_INTRIN512)
    typedef __m512d simd_t;
    typedef __m512i isimd_t;
    typedef void imem_t;
    typedef __mmask8 real_mask_t;
#define VEC_ELEMS 8
#define INAME(op) _mm512_ ## op ## _pd
#define INAMEI(op) _mm512_ ## op ## _epi64
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

#elif !defined(INAME)
#warning "Emulating intrinsics because HW vector length not defined; check setting of USE_INTRIN256 or USE_INTRIN512 in kernel Makefile"
#define NO_INTRINSICS

#elif VLEN != VEC_ELEMS
#warning "Emulating intrinsics because VLEN != HW vector length"
#define NO_INTRINSICS
#endif

#undef VEC_ELEMS

    // Macro for looping through an aligned real_vec_t.
#if defined(CHECK) || (VLEN==1) || !defined(__INTEL_COMPILER)
#define REAL_VEC_LOOP(i)                        \
    for (int i=0; i<VLEN; i++)
#define REAL_VEC_LOOP_UNALIGNED(i)              \
    for (int i=0; i<VLEN; i++)
#else
#define REAL_VEC_LOOP(i)                                                \
    _Pragma("vector aligned") _Pragma("vector always") _Pragma("omp simd") \
    for (int i=0; i<VLEN; i++)
#define REAL_VEC_LOOP_UNALIGNED(i)                      \
    _Pragma("vector always") _Pragma("omp simd")        \
    for (int i=0; i<VLEN; i++)
#endif

    // fence needed before loads after streaming stores.
    ALWAYS_INLINE void make_stores_visible() {
#if defined(USE_STREAMING_STORE)
        _mm_mfence();
#endif
    }

    // The following union is used to overlay C arrays with vector types.
    // It must be an aggregate type to allow aggregate initialization,
    // so no user-provided ctors, copy operator, virtual member functions, etc.
    // These features are in real_vec_t, which contains a real_vec_t_data.
    union real_vec_t_data {

        // Real array overlay.
        real_t r[VLEN];

        // Integer array overlay with ctrl_t same size as real_t.
        ctrl_t ci[VLEN];

#ifndef NO_INTRINSICS

        // Real SIMD-type overlay.
        simd_t mr;

        // Integer SIMD-type overlay.
        isimd_t mi;
#endif
    };


    // Type for a vector block.
    struct real_vec_t {

        // union of data types.
        real_vec_t_data u;

        // default ctor does not init data!
        ALWAYS_INLINE real_vec_t() {}

        // copy vector.
        ALWAYS_INLINE real_vec_t(const real_vec_t& val) {
            operator=(val);
        }
        ALWAYS_INLINE real_vec_t(const real_vec_t_data& val) {
            operator=(val);
        }
#ifndef NO_INTRINSICS
        ALWAYS_INLINE real_vec_t(const simd_t& val) {
            operator=(val);
        }
#endif

        // broadcast scalar.
        ALWAYS_INLINE real_vec_t(float val) {
            operator=(val);
        }
        ALWAYS_INLINE real_vec_t(double val) {
            operator=(val);
        }
        ALWAYS_INLINE real_vec_t(int val) {
            operator=(val);
        }
        ALWAYS_INLINE real_vec_t(long val) {
            operator=(val);
        }

        // get length.
        ALWAYS_INLINE int get_num_elems() const {
            return VLEN;
        }

        // copy whole vector.
        ALWAYS_INLINE real_vec_t& operator=(const real_vec_t& rhs) {
#ifdef NO_INTRINSICS
            REAL_VEC_LOOP(i) u.r[i] = rhs[i];
#else
            u.mr = rhs.u.mr;
#endif
            return *this;
        }
        ALWAYS_INLINE real_vec_t& operator=(const real_vec_t_data& rhs) {
            u = rhs;
            return *this;
        }
#ifndef NO_INTRINSICS
        ALWAYS_INLINE real_vec_t& operator=(const simd_t& rhs) {
            u.mr = rhs;
            return *this;
        }
#endif

        // assignment: single value broadcast.
        ALWAYS_INLINE void operator=(double val) {
#ifdef NO_INTRINSICS
            REAL_VEC_LOOP(i) u.r[i] = real_t(val);
#else
            u.mr = INAME(set1)(real_t(val));
#endif
        }
        ALWAYS_INLINE void operator=(float val) {
#ifdef NO_INTRINSICS
            REAL_VEC_LOOP(i) u.r[i] = real_t(val);
#else
            u.mr = INAME(set1)(real_t(val));
#endif
        }

        // broadcast with conversions.
        ALWAYS_INLINE void operator=(int val) {
            operator=(real_t(val));
        }
        ALWAYS_INLINE void operator=(long val) {
            operator=(real_t(val));
        }


        // access a real_t linearly.
        ALWAYS_INLINE real_t& operator[](idx_t l) {
            assert(l >= 0);
            assert(l < VLEN);
            return u.r[l];
        }
        ALWAYS_INLINE const real_t& operator[](idx_t l) const {
            assert(l >= 0);
            assert(l < VLEN);
            return u.r[l];
        }

        // unary negate.
        ALWAYS_INLINE real_vec_t operator-() const {
            real_vec_t res;
#ifdef NO_INTRINSICS
            REAL_VEC_LOOP(i) res[i] = -u.r[i];
#else
            // subtract from zero.
            res.u.mr = INAME(sub)(INAME(setzero)(), u.mr);
#endif
            return res;
        }

        // add.
        ALWAYS_INLINE real_vec_t operator+(real_vec_t rhs) const {
            real_vec_t res;
#ifdef NO_INTRINSICS
            REAL_VEC_LOOP(i) res[i] = u.r[i] + rhs[i];
#else
            res.u.mr = INAME(add)(u.mr, rhs.u.mr);
#endif
            return res;
        }
        ALWAYS_INLINE real_vec_t operator+(real_t rhs) const {
            real_vec_t rn;
            rn = rhs;               // broadcast.
            return (*this) + rn;
        }

        // sub.
        ALWAYS_INLINE real_vec_t operator-(real_vec_t rhs) const {
            real_vec_t res;
#ifdef NO_INTRINSICS
            REAL_VEC_LOOP(i) res[i] = u.r[i] - rhs[i];
#else
            res.u.mr = INAME(sub)(u.mr, rhs.u.mr);
#endif
            return res;
        }
        ALWAYS_INLINE real_vec_t operator-(real_t rhs) const {
            real_vec_t rn;
            rn = rhs;               // broadcast.
            return (*this) - rn;
        }

        // mul.
        ALWAYS_INLINE real_vec_t operator*(real_vec_t rhs) const {
            real_vec_t res;
#ifdef NO_INTRINSICS
            REAL_VEC_LOOP(i) res[i] = u.r[i] * rhs[i];
#else
            res.u.mr = INAME(mul)(u.mr, rhs.u.mr);
#endif
            return res;
        }
        ALWAYS_INLINE real_vec_t operator*(real_t rhs) const {
            real_vec_t rn;
            rn = rhs;               // broadcast.
            return (*this) * rn;
        }

        // div.
        ALWAYS_INLINE real_vec_t operator/(real_vec_t rhs) const {
            real_vec_t res, rcp;
#ifdef NO_INTRINSICS
            REAL_VEC_LOOP(i) res[i] = u.r[i] / rhs[i];
#elif defined(USE_RCP14)
            rcp.u.mr = INAME(rcp14)(rhs.u.mr);
            res.u.mr = INAME(mul)(u.mr, rcp.u.mr);
#elif defined(USE_RCP28)
            rcp.u.mr = INAME(rcp28)(rhs.u.mr);
            res.u.mr = INAME(mul)(u.mr, rcp.u.mr);
#else
            res.u.mr = INAME(div)(u.mr, rhs.u.mr);
#endif
            return res;
        }
        ALWAYS_INLINE real_vec_t operator/(real_t rhs) const {
            real_vec_t rn;
            rn = rhs;               // broadcast.
            return (*this) / rn;
        }

        // less-than comparator.
        bool operator<(const real_vec_t& rhs) const {
            for (int j = 0; j < VLEN; j++) {
                if (u.r[j] < rhs.u.r[j])
                    return true;
                else if (u.r[j] > rhs.u.r[j])
                    return false;
            }
            return false;
        }

        // greater-than comparator.
        bool operator>(const real_vec_t& rhs) const {
            for (int j = 0; j < VLEN; j++) {
                if (u.r[j] > rhs.u.r[j])
                    return true;
                else if (u.r[j] < rhs.u.r[j])
                    return false;
            }
            return false;
        }

        // equal-to comparator.
        bool operator==(const real_vec_t& rhs) const {
            for (int j = 0; j < VLEN; j++) {
                if (u.r[j] != rhs.u.r[j])
                    return false;
            }
            return true;
        }

        // not-equal-to comparator.
        bool operator!=(const real_vec_t& rhs) const {
            return !operator==(rhs);
        }

        // aligned load.
        ALWAYS_INLINE void loadFrom(const real_vec_t* __restrict__ from) {
#if defined(NO_INTRINSICS) || defined(NO_LOAD_INTRINSICS)
            REAL_VEC_LOOP(i) u.r[i] = (*from)[i];
#else
            u.mr = INAME(load)((imem_t const*)from);
#endif
        }

        // unaligned load.
        ALWAYS_INLINE void loadUnalignedFrom(const real_vec_t* __restrict__ from) {
#if defined(NO_INTRINSICS) || defined(NO_LOAD_INTRINSICS)
            REAL_VEC_LOOP_UNALIGNED(i) u.r[i] = (*from)[i];
#else
            u.mr = INAME(loadu)((imem_t const*)from);
#endif
        }

        // aligned store.
        ALWAYS_INLINE void storeTo(real_vec_t* __restrict__ to) const {

            // Using an explicit loop here instead of a store intrinsic may
            // allow the compiler to do more optimizations.  This is true
            // for icc 2016 r2--it may change for later versions. Try
            // defining and not defining NO_STORE_INTRINSICS and comparing
            // the sizes of the stencil computation loop and the overall
            // performance.
#if defined(NO_INTRINSICS) || defined(NO_STORE_INTRINSICS)
#if defined(__INTEL_COMPILER) && (VLEN > 1) && defined(USE_STREAMING_STORE)
            _Pragma("vector nontemporal")
#endif
                REAL_VEC_LOOP(i) (*to)[i] = u.r[i];
#elif !defined(USE_STREAMING_STORE)
            INAME(store)((imem_t*)to, u.mr);
#elif defined(ARCH_KNC)
            INAME(storenrngo)((imem_t*)to, u.mr);
#else
            INAME(stream)((imem_t*)to, u.mr);
#endif
        }

        // masked store.
        ALWAYS_INLINE void storeTo_masked(real_vec_t* __restrict__ to, uidx_t k1) const {

            // Using an explicit loop here instead of a store intrinsic may
            // allow the compiler to do more optimizations.  This is true
            // for icc 2016 r2--it may change for later versions. Try
            // defining and not defining NO_STORE_INTRINSICS and comparing
            // the sizes of the stencil computation loop and the overall
            // performance.
#if defined(NO_INTRINSICS) || defined(NO_STORE_INTRINSICS)
#if defined(__INTEL_COMPILER) && (VLEN > 1) && defined(USE_STREAMING_STORE)
            _Pragma("vector nontemporal")
#endif
                REAL_VEC_LOOP(i) if ((k1 >> i) & 1) (*to)[i] = u.r[i];

            // No masked streaming store intrinsic.
#elif defined USE_INTRIN256

            // Need to set [at least] upper bit in a SIMD reg to mask bit.
            real_vec_t_data mask;
            REAL_VEC_LOOP(i) mask.ci[i] = ((k1 >> i) & 1) ? ctrl_t(-1) : ctrl_t(0);
            INAME(maskstore)((imem_t*)to, mask.mi, u.mr);
#else
            INAME(mask_store)((imem_t*)to, real_mask_t(k1), u.mr);
#endif
        }

        // Output.
        void print_ctrls(std::ostream& os, bool doEnd=true) const {
            for (int j = 0; j < VLEN; j++) {
                if (j) os << ", ";
                os << "[" << j << "]=" << u.ci[j];
            }
            if (doEnd)
                os << std::endl;
        }

        void print_reals(std::ostream& os, bool doEnd=true) const {
            for (int j = 0; j < VLEN; j++) {
                if (j) os << ", ";
                os << "[" << j << "]=" << u.r[j];
            }
            if (doEnd)
                os << std::endl;
        }

    }; // real_vec_t.

    // Output using '<<'.
    inline std::ostream& operator<<(std::ostream& os, const real_vec_t& rn) {
        rn.print_reals(os, false);
        return os;
    }

    // More operator overloading.
    ALWAYS_INLINE real_vec_t operator+(real_t lhs, const real_vec_t& rhs) {
        return real_vec_t(lhs) + rhs;
    }
    ALWAYS_INLINE real_vec_t operator-(real_t lhs, const real_vec_t& rhs) {
        return real_vec_t(lhs) - rhs;
    }
    ALWAYS_INLINE real_vec_t operator*(real_t lhs, const real_vec_t& rhs) {
        return real_vec_t(lhs) * rhs;
    }
    ALWAYS_INLINE real_vec_t operator/(real_t lhs, const real_vec_t& rhs) {
        return real_vec_t(lhs) / rhs;
    }

    // Safe sqrt.
    ALWAYS_INLINE float sqrt_absf(float a) {
        return sqrtf(fabsf(a));
    }
    ALWAYS_INLINE double sqrt_abs(double a) {
        return sqrt(fabs(a));
    }

    // A macro to emulate missing SVML vector functions.
#if !defined(NO_INTRINSICS)
#define MAKE_INTRIN(op, libm_fn)                        \
    ALWAYS_INLINE simd_t INAME(op)(simd_t a) {          \
        real_vec_t rva(a);                              \
        real_vec_t res;                                 \
        REAL_VEC_LOOP(i) res[i] = libm_fn(rva.u.r[i]);  \
        return res.u.mr;                                \
    }

    // Need simd abs for 256 bits only.
#ifdef USE_INTRIN256
#if REAL_BYTES == 4
    MAKE_INTRIN(abs, fabsf)
#else
    MAKE_INTRIN(abs, fabs)
#endif
#endif

    // Safe sqrt(x), i.e., sqrt(abs(x)).
    // Used mainly for validation tests.
#if REAL_BYTES == 4
    MAKE_INTRIN(sqrt_abs, sqrt_absf)
#else
    MAKE_INTRIN(sqrt_abs, sqrt_abs)
#endif

#endif

    // math func wrappers.
#if REAL_BYTES == 4
#define SVML_1ARG_SCALAR(yask_fn, libm_dpfn, libm_spfn)        \
    ALWAYS_INLINE real_t yask_fn(const real_t& a) {             \
        return libm_spfn(a);                                    \
    }
#define SVML_2ARG_SCALAR(yask_fn, libm_dpfn, libm_spfn)        \
    ALWAYS_INLINE real_t yask_fn(const real_t& a, const real_t& b) {    \
        return libm_spfn(a, b);                                         \
    }
#else
#define SVML_1ARG_SCALAR(yask_fn, libm_dpfn, libm_spfn)        \
    ALWAYS_INLINE real_t yask_fn(const real_t& a) {             \
        return libm_dpfn(a);                                    \
    }
#define SVML_2ARG_SCALAR(yask_fn, libm_dpfn, libm_spfn)        \
    ALWAYS_INLINE real_t yask_fn(const real_t& a, const real_t& b) {    \
        return libm_dpfn(a, b);                                         \
    }
#endif

    // SVML emulation.
#if defined(NO_INTRINSICS) || !defined(USE_SVML)
#define SVML_1ARG(yask_fn, svml_fn, libm_dpfn, libm_spfn)      \
    SVML_1ARG_SCALAR(yask_fn, libm_dpfn, libm_spfn)            \
    ALWAYS_INLINE real_vec_t yask_fn(const real_vec_t& a) {     \
        real_vec_t res;                                         \
        REAL_VEC_LOOP(i) res[i] = yask_fn(a.u.r[i]);            \
        return res;                                             \
    }
#define SVML_2ARG(yask_fn, svml_fn, libm_dpfn, libm_spfn)      \
    SVML_2ARG_SCALAR(yask_fn, libm_dpfn, libm_spfn)            \
    ALWAYS_INLINE real_vec_t yask_fn(const real_vec_t& a, const real_vec_t& b) {    \
        real_vec_t res;                                         \
        REAL_VEC_LOOP(i) res[i] = yask_fn(a.u.r[i], b.u.r[i]);  \
        return res;                                             \
    }

    // SVML library wrappers.
#else
#define SVML_1ARG(yask_fn, svml_fn, libm_dpfn, libm_spfn)      \
    SVML_1ARG_SCALAR(yask_fn, libm_dpfn, libm_spfn)            \
    ALWAYS_INLINE real_vec_t yask_fn(const real_vec_t& a) {     \
        real_vec_t res;                                         \
        res.u.mr = INAME(svml_fn)(a.u.mr);                      \
        return res;                                             \
    }
#define SVML_2ARG(yask_fn, svml_fn, libm_dpfn, libm_spfn)      \
    SVML_2ARG_SCALAR(yask_fn, libm_dpfn, libm_spfn)            \
    ALWAYS_INLINE real_vec_t yask_fn(const real_vec_t& a, const real_vec_t& b) {    \
        real_vec_t res;                                         \
        res.u.mr = INAME(svml_fn)(a.u.mr, b.u.mr);              \
        return res;                                             \
    }
#endif

    // Use safe sqrt when running checked code to avoid NaNs.
    // In production usage, it is the responsibility of the user
    // to guarantee that the arguments to sqrt() are non-negative.
#ifdef CHECK
    SVML_1ARG(yask_sqrt, sqrt_abs, sqrt_abs, sqrt_absf) // square root.
#else
    SVML_1ARG(yask_sqrt, sqrt, sqrt, sqrtf) // square root.
#endif
    SVML_1ARG(yask_cbrt, cbrt, cbrt, cbrtf) // cube root.
    SVML_1ARG(yask_fabs, abs, fabs, fabsf) // abs value.
    SVML_1ARG(yask_erf, erf, erf, erff) // error fn.
    SVML_1ARG(yask_exp, exp, exp, expf) // natural exp.
    SVML_1ARG(yask_log, log, log, logf) // natural log.
    SVML_1ARG(yask_sin, sin, sin, sinf) // sine.
    SVML_1ARG(yask_cos, cos, cos, cosf) // cosine.
    SVML_1ARG(yask_atan, atan, atan, atanf) // inv (arc) tangent.
    SVML_2ARG(yask_pow, pow, pow, powf) // power.
#undef SVML_1ARG_SCALAR
#undef SVML_1ARG
#undef SVML_2ARG_SCALAR
#undef SVML_2ARG

    // Sin+cos.
    ALWAYS_INLINE void yask_sin_and_cos(real_t& sin_res, real_t& cos_res, real_t a) {
#if REAL_BYTES == 4
        sincosf(a, &sin_res, &cos_res);
#else
        sincos(a, &sin_res, &cos_res);
#endif
    }
    ALWAYS_INLINE void yask_cos_and_sin(real_t& cos_res, real_t& sin_res, real_t a) {
        yask_sin_and_cos(sin_res, cos_res, a);
    }
    ALWAYS_INLINE void yask_sin_and_cos(real_vec_t& sin_res, 
                                        real_vec_t& cos_res, 
                                        const real_vec_t& a) {
#if defined(NO_INTRINSICS) || !defined(USE_SVML)
        REAL_VEC_LOOP(i) yask_sin_and_cos(sin_res[i], cos_res[i], a[i]);
#else
        sin_res.u.mr = INAME(sincos)(&cos_res.u.mr, a.u.mr);
#endif
    }
    ALWAYS_INLINE void yask_cos_and_sin(real_vec_t& cos_res, 
                                        real_vec_t& sin_res, 
                                        const real_vec_t& a) {
        yask_sin_and_cos(sin_res, cos_res, a);
    }

    // Get consecutive elements from two vectors.
    // Concat a and b, shift right by count elements, keep _right_most elements.
    // Thus, shift of 0 returns b; shift of VLEN returns a.
    // Must be a template because count has to be known at compile-time.
    template<int count>
    ALWAYS_INLINE void real_vec_align(real_vec_t& res, const real_vec_t& a, const real_vec_t& b) {
#ifdef TRACE_INTRINSICS
        std::cout << "real_vec_align w/count=" << count << ":" << std::endl;
        std::cout << " a: ";
        a.print_reals(std::cout);
        std::cout << " b: ";
        b.print_reals(std::cout);
#endif
        assert(count >= 0);
        assert(count <= VLEN);
        if (count == 0)
            res.u = b.u;
        else if (count == VLEN)
            res.u = a.u;
        else {
        
#if defined(NO_INTRINSICS)
            // must make temp copies in case &res == &a or &b.
            real_vec_t tmpa = a, tmpb = b;
            for (int i = 0; i < VLEN-count; i++)
                res.u.r[i] = tmpb.u.r[i + count];
            for (int i = VLEN-count; i < VLEN; i++)
                res.u.r[i] = tmpa.u.r[i + count - VLEN];

            // For AVX2, use 8-bit op per 128-bit lane w/count*REAL_BYTES.
#elif defined(USE_AVX2)
            // See https://software.intel.com/en-us/blogs/2015/01/13/programming-using-avx2-permutations.
            // Each nybble of ctrl is
            // 0: lo part of A.
            // 1: hi part of A.
            // 2: lo part of B.
            // 3: hi part of B.
            auto tmp = _mm256_permute2x128_si256(b.u.mi, a.u.mi, 0x21);
#ifdef TRACE_INTRINSICS
            std::cout << " tmp: ";
            real_vec_t tmpa;
            tmpa.u.mi = tmp;
            tmpa.print_reals(std::cout);
#endif
            // count must be 1..VLEN-1.
            if (count == VLEN/2)
                res.u.mi = tmp;
            else if (count < VLEN/2)
                res.u.mi = _mm256_alignr_epi8(tmp, b.u.mi, count*REAL_BYTES);
            else
                res.u.mi = _mm256_alignr_epi8(a.u.mi, tmp, (count-(VLEN/2))*REAL_BYTES);

            // For AVX but not AVX2.
#elif defined(USE_INTRIN256)
            // Not really an intrinsic, but not element-wise, either.
            // Put the 2 parts in a local array, then extract the desired part
            // using an unaligned load.
            // The Intel compiler converts this into an efficient sequence
            // using vmovup*, vshufp*, vinsertf128, and/or vunpcklp* instrs.
            typedef real_t R2[VLEN * 2] CACHE_ALIGNED;
            R2 r2;
            *((real_vec_t*)(&r2[0])) = b;
            *((real_vec_t*)(&r2[VLEN])) = a;
            real_vec_t* p = (real_vec_t*)(&r2[count]); // not usually aligned.
            res.u.mr = INAME(loadu)((imem_t const*)p);

            // For DP on KNC, use 32-bit op w/2x count.
#elif REAL_BYTES == 8 && defined(ARCH_KNC) && defined(USE_INTRIN512)
            res.u.mi = _mm512_alignr_epi32(a.u.mi, b.u.mi, count*2);

            // Everything else.
#else
            res.u.mi = INAMEI(alignr)(a.u.mi, b.u.mi, count);
#endif
        }
        
#ifdef TRACE_INTRINSICS
        std::cout << " res: ";
        res.print_reals(std::cout);
#endif
    }

    // Get consecutive elements from two vectors w/masking.
    // Concat a and b, shift right by count elements, keep _right_most elements.
    // Elements in res corresponding to 0 bits in k1 are unchanged.
    template<int count>
    ALWAYS_INLINE void real_vec_align_masked(real_vec_t& res, const real_vec_t& a, const real_vec_t& b,
                                             uidx_t k1) {
#ifdef TRACE_INTRINSICS
        std::cout << "real_vec_align w/count=" << count << " w/mask:" << std::endl;
        std::cout << " a: ";
        a.print_reals(std::cout);
        std::cout << " b: ";
        b.print_reals(std::cout);
        std::cout << " res(before): ";
        res.print_reals(std::cout);
        std::cout << " mask: 0x" << std::hex << k1 << std::endl;
#endif

#if defined(NO_INTRINSICS) || !defined(USE_INTRIN512)
        // must make temp copies in case &res == &a or &b.
        real_vec_t tmpa = a, tmpb = b;
        for (int i = 0; i < VLEN-count; i++)
            if ((k1 >> i) & 1)
                res.u.r[i] = tmpb.u.r[i + count];
        for (int i = VLEN-count; i < VLEN; i++)
            if ((k1 >> i) & 1)
                res.u.r[i] = tmpa.u.r[i + count - VLEN];
#else
        res.u.mi = INAMEI(mask_alignr)(res.u.mi, real_mask_t(k1), a.u.mi, b.u.mi, count);
#endif

#ifdef TRACE_INTRINSICS
        std::cout << " res(after): ";
        res.print_reals(std::cout);
#endif
    }

    // Rearrange elements in a vector.
    ALWAYS_INLINE void real_vec_permute(real_vec_t& res, const real_vec_t& ctrl, const real_vec_t& a) {

#ifdef TRACE_INTRINSICS
        std::cout << "real_vec_permute:" << std::endl;
        std::cout << " ctrl: ";
        ctrl.print_ctrls(std::cout);
        std::cout << " a: ";
        a.print_reals(std::cout);
#endif

#if defined(NO_INTRINSICS) || !defined(USE_INTRIN512)
        // must make a temp copy in case &res == &a.
        real_vec_t tmp = a;
        for (int i = 0; i < VLEN; i++)
            res.u.r[i] = tmp.u.r[ctrl.u.ci[i]];
#else
        res.u.mi = INAMEI(permutexvar)(ctrl.u.mi, a.u.mi);
#endif

#ifdef TRACE_INTRINSICS
        std::cout << " res: ";
        res.print_reals(std::cout);
#endif
    }

    // Rearrange elements in a vector w/masking.
    // Elements in res corresponding to 0 bits in k1 are unchanged.
    ALWAYS_INLINE void real_vec_permute_masked(real_vec_t& res, const real_vec_t& ctrl, const real_vec_t& a,
                                               uidx_t k1) {
#ifdef TRACE_INTRINSICS
        std::cout << "real_vec_permute w/mask:" << std::endl;
        std::cout << " ctrl: ";
        ctrl.print_ctrls(std::cout);
        std::cout << " a: ";
        a.print_reals(std::cout);
        std::cout << " mask: 0x" << std::hex << k1 << std::endl;
        std::cout << " res(before): ";
        res.print_reals(std::cout);
#endif

#if defined(NO_INTRINSICS) || !defined(USE_INTRIN512)
        // must make a temp copy in case &res == &a.
        real_vec_t tmp = a;
        for (int i = 0; i < VLEN; i++) {
            if ((k1 >> i) & 1)
                res.u.r[i] = tmp.u.r[ctrl.u.ci[i]];
        }
#else
        res.u.mi = INAMEI(mask_permutexvar)(res.u.mi, real_mask_t(k1), ctrl.u.mi, a.u.mi);
#endif

#ifdef TRACE_INTRINSICS
        std::cout << " res(after): ";
        res.print_reals(std::cout);
#endif
    }

    // Rearrange elements in 2 vectors.
    // (The masking versions of these instrs do not preserve the source,
    // so we don't have a masking version of this function.)
    ALWAYS_INLINE void real_vec_permute2(real_vec_t& res, const real_vec_t& ctrl,
                                         const real_vec_t& a, const real_vec_t& b) {
#ifdef TRACE_INTRINSICS
        std::cout << "real_vec_permute2:" << std::endl;
        std::cout << " ctrl: ";
        ctrl.print_ctrls(std::cout);
        std::cout << " a: ";
        a.print_reals(std::cout);
        std::cout << " b: ";
        b.print_reals(std::cout);
#endif

#if defined(NO_INTRINSICS) || !defined(USE_INTRIN512)
        // must make temp copies in case &res == &a or &b.
        real_vec_t tmpa = a, tmpb = b;
        for (int i = 0; i < VLEN; i++) {
            int sel = ctrl.u.ci[i] & ctrl_sel_bit; // 0 => a, 1 => b.
            int idx = ctrl.u.ci[i] & ctrl_idx_mask; // index.
            res.u.r[i] = sel ? tmpb.u.r[idx] : tmpa.u.r[idx];
        }

#elif defined(ARCH_KNC)
        yask_exception e;
        std::stringstream err;
        err << "error: 2-input permute not supported on KNC" << std::endl;
        e.add_message(err.str());
        throw e;
        //exit_yask(1);
#else
        res.u.mi = INAMEI(permutex2var)(a.u.mi, ctrl.u.mi, b.u.mi);
#endif

#ifdef TRACE_INTRINSICS
        std::cout << " res: ";
        res.print_reals(std::cout);
#endif
    }

    // Prefetch wrapper.
    template <int level>
    inline void prefetch(const void* p) {
#if defined(__INTEL_COMPILER)
        _mm_prefetch((const char*)p, level);
#else
        _mm_prefetch(p, (enum _mm_hint)level);
#endif
    }

    // default max abs difference in validation.
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
            std::cerr << "outside tolerance of " << epsilon << ": " << val << " != " << ref <<
                " because " << adiff << " >= " << epsilon << std::endl;
#endif
        return ok;
    }

    // Compare two real_vec_t's.
    inline bool within_tolerance(const real_vec_t& val, const real_vec_t& ref,
                                 const real_vec_t& epsilon) {
        for (int j = 0; j < VLEN; j++) {
            if (!within_tolerance(val.u.r[j], ref.u.r[j], epsilon.u.r[j]))
                return false;
        }
        return true;
    }

}
#endif
