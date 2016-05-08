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

// Data and sizes for overall problem.
// This is a pure-virtual class that must be implemented
// for a specific problem.
struct StencilContext {

    // Name.
    string name;

    // A list of all grids.
    vector<RealvGridBase*> gridPtrs;

    // A list of all non-grid parameters.
    vector<GenericGridBase<REAL>*> paramPtrs;

    // Sizes.
    // - time size in steps to be done (not grid allocation).
    // - stencil size in equations to eval (not number of grids).
    // - spatial sizes in elements (not vectors).
    idx_t dt, ds, dn, dx, dy, dz;     // problem size.
    idx_t rt, rs, rn, rx, ry, rz;     // region size.
    idx_t bt, bs, bn, bx, by, bz;     // block size.
    idx_t padn, padx, pady, padz;     // spatial padding, including halos.
    idx_t angle_n, angle_x, angle_y, angle_z; // temporal skewing angles.

    StencilContext() { }
    virtual ~StencilContext() { }

    // Allocate grid memory and set gridPtrs.
    virtual void allocGrids() =0;

    // Allocate param memory and set paramPtrs.
    virtual void allocParams() =0;

    // Get total size.
    virtual idx_t get_num_bytes();

    // Init all grids & params w/same value within each,
    // but different values between them.
    virtual void initSame();

    // Init all grids & params w/different values.
    // Better for validation, but slower.
    virtual void initDiff();

    // Compare grids in contexts.
    // Params should not be written to, so they are not compared.
    // Return number of mis-compares.
    virtual idx_t compare(const StencilContext& ref) const;
};

/// Classes that support evaluation of one stencil equation.

// A pure-virtual class base for a stencil equation.
class StencilBase {

public:
    StencilBase() { }
    virtual ~StencilBase() { }

    // Get info from implementation.
    virtual const string& get_name() =0;
    virtual int get_scalar_fp_ops() =0;
    
    // Calculate one scalar result at time t.
    virtual void calc_scalar(StencilContext& generic_context,
                             idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) =0;

    // Calculate one block of results from begin to end-1 on each dimension.
    // Note: this interface cannot support temporal blocking with >1 stencil because
    // it only operates on one stencil.
    virtual void calc_block(StencilContext& generic_context, idx_t bt,
                            idx_t begin_bn, idx_t begin_bx, idx_t begin_by, idx_t begin_bz,
                            idx_t end_bn, idx_t end_bx, idx_t end_by, idx_t end_bz) =0;
};

// Macro for automatically adding N dimension.
// TODO: make all args programmatic.
#if USING_DIM_N
#define ARG_N(n) n,
#else
#define ARG_N(n)
#endif

// Prefetch a cluser starting at vector indices i, j, k.
// Generic macro to handle different cache levels.
#define PREFETCH_CLUSTER(fn, t, n, i, j, k)                             \
    do {                                                                \
        TRACE_MSG("%s.%s(%ld, %ld, %ld, %ld, %ld)",                     \
                  get_name().c_str(), #fn, t, n, i, j, k);              \
        _stencil.fn(context, t, ARG_N(n) i, j, k);                      \
    } while(0)

// A template that provides wrappers around a stencil-equation class created
// by the foldBuilder. A template is used instead of inheritance for performance.
// By using templates, the compiler can inline stencil code into loops and
// avoid indirect calls.
template <typename StencilEquationClass, typename ContextClass>
class StencilTemplate : public StencilBase {

protected:

    // _stencil must implement calc_scalar, calc_vector,
    // prefetch_L1_vector, prefetch_L2_vector, name.
    StencilEquationClass _stencil;

public:
    StencilTemplate() {}
    StencilTemplate(const string& name) :
        StencilBase(name) { }
    virtual ~StencilTemplate() {}

    // Get values from _stencil.
    virtual const string& get_name() {
        return _stencil.name;
    }
    virtual int get_scalar_fp_ops() {
        return _stencil.scalar_fp_ops;
    }
    
    // Calculate one scalar result.
    // This function implements the interface in the base class.
    virtual void calc_scalar(StencilContext& generic_context, idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) {

        // Convert to a problem-specific context.
        auto context = dynamic_cast<ContextClass&>(generic_context);

        // Call the generated code.
        _stencil.calc_scalar(context, t, ARG_N(n) x, y, z);
    }

    // Calculate results within a vector cluster.
    // Called from calc_block().
    // The begin/end_c* vars are the start/stop_b* vars from the block loops.
    // This function doesn't contain any loops; it's just a wrapper around 
    // calc_vector.
    ALWAYS_INLINE void
    calc_cluster (ContextClass& context, idx_t ct,
                  idx_t begin_cnv, idx_t begin_cxv, idx_t begin_cyv, idx_t begin_czv,
                  idx_t end_cnv, idx_t end_cxv, idx_t end_cyv, idx_t end_czv)
    {
        TRACE_MSG("%s.calc_cluster(%ld, %ld, %ld, %ld, %ld)",
                  get_name().c_str(), ct, begin_cnv, begin_cxv, begin_cyv, begin_czv);

        // The step vars are hard-coded in calc_block below, and there should
        // never be a partial step at this level. So, we can assume one var and
        // exactly CLEN_d steps in each given direction d are calculated in this
        // function.  Thus, we can ignore the end_* vars in the calc function.
        assert(end_cnv == begin_cnv + CLEN_N);
        assert(end_cxv == begin_cxv + CLEN_X);
        assert(end_cyv == begin_cyv + CLEN_Y);
        assert(end_czv == begin_czv + CLEN_Z);
        
        // Calculate results.
        _stencil.calc_vector(context, ct, ARG_N(begin_cnv) begin_cxv, begin_cyv, begin_czv);
    }

    // Prefetch a cluster into L1.
    // Called from calc_block().
    ALWAYS_INLINE void
    prefetch_L1_cluster (ContextClass& context, idx_t ct,
                         idx_t begin_cnv, idx_t begin_cxv, idx_t begin_cyv, idx_t begin_czv,
                         idx_t end_cnv, idx_t end_cxv, idx_t end_cyv, idx_t end_czv)
    {
        PREFETCH_CLUSTER(prefetch_L1_vector, ct, begin_cnv, begin_cxv, begin_cyv, begin_czv);
    }
    
    // Prefetch a cluster into L2.
    // Called from calc_block().
    ALWAYS_INLINE void
    prefetch_L2_cluster (ContextClass& context, idx_t ct,
                         idx_t begin_cnv, idx_t begin_cxv, idx_t begin_cyv, idx_t begin_czv,
                         idx_t end_cnv, idx_t end_cxv, idx_t end_cyv, idx_t end_czv)
    {
        PREFETCH_CLUSTER(prefetch_L2_vector, ct, begin_cnv, begin_cxv, begin_cyv, begin_czv);
    }
    
    // Calculate results within a cache block.
    // This function implements the interface in the base class.
    // Each block is typically computed in a separate OpenMP task.
    // The begin/end_b* vars are the start/stop_r* vars from the region loops.
    virtual void
    calc_block(StencilContext& generic_context, idx_t bt,
               idx_t begin_bn, idx_t begin_bx, idx_t begin_by, idx_t begin_bz,
               idx_t end_bn, idx_t end_bx, idx_t end_by, idx_t end_bz)
    {
        TRACE_MSG("%s.calc_block(%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld)", 
                  get_name().c_str(), bt,
                  begin_bn, end_bn-1,
                  begin_bx, end_bx-1,
                  begin_by, end_by-1,
                  begin_bz, end_bz-1);

        // Convert to a problem-specific context.
        auto context = dynamic_cast<ContextClass&>(generic_context);

        // Divide indices by vector lengths.
        const idx_t begin_bnv = idiv<idx_t>(begin_bn, VLEN_N);
        const idx_t begin_bxv = idiv<idx_t>(begin_bx, VLEN_X);
        const idx_t begin_byv = idiv<idx_t>(begin_by, VLEN_Y);
        const idx_t begin_bzv = idiv<idx_t>(begin_bz, VLEN_Z);
        const idx_t end_bnv = idiv<idx_t>(end_bn, VLEN_N);
        const idx_t end_bxv = idiv<idx_t>(end_bx, VLEN_X);
        const idx_t end_byv = idiv<idx_t>(end_by, VLEN_Y);
        const idx_t end_bzv = idiv<idx_t>(end_bz, VLEN_Z);

        // Vector-size steps based on cluster lengths.
        const idx_t step_bnv = CLEN_N;
        const idx_t step_bxv = CLEN_X;
        const idx_t step_byv = CLEN_Y;
        const idx_t step_bzv = CLEN_Z;
        
#if !defined(DEBUG) && defined(__INTEL_COMPILER)
        // Force loop body to be inline.
#pragma forceinline recursive
#endif
        {
            // Include automatically-generated loop code that calls calc_cluster()
            // and optionally, the prefetch functions().
#include "stencil_block_loops.hpp"
        }
    }

};

// Collection of all stencil equations to be evaluated.
struct StencilEquations {

    string name;

    // List of stencils.
    vector<StencilBase*> stencils;

    StencilEquations() {}
    virtual ~StencilEquations() {}

    // Reference stencil calculations.
    virtual void calc_problem_ref(StencilContext& context);

    // Vectorized and blocked stencil calculations.
    virtual void calc_problem_opt(StencilContext& context);

protected:
    
    // Calculate results within a region.
    void calc_region(StencilContext& context, 
                     idx_t begin_rt, idx_t begin_rn, idx_t begin_rx, idx_t begin_ry, idx_t begin_rz,
                     idx_t end_rt, idx_t end_rn, idx_t end_rx, idx_t end_ry, idx_t end_rz);
};
