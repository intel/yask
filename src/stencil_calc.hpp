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

// Macro for automatically adding N dimension.
// TODO: make all args programmatic.
#if USING_DIM_N
#define ARG_N(n) n,
#else
#define ARG_N(n)
#endif

// Prefetch a cluser starting at vector indices i, j, k.
// Generic macro to handle different cache levels and directions.
#define PREFETCH_CLUSTER(fn, t, n, i, j, k)                             \
    do {                                                                \
        TRACE_MSG("%s.%s(%ld, %ld, %ld, %ld, %ld)",                     \
                  _name.c_str(), #fn, t, n, i, j, k);                   \
        _stencil.fn(context, t, ARG_N(n) i, j, k);                      \
    } while(0)

//////////////// Main stencil class /////////////////

class StencilBase {

protected:
    string _name;

public:
    StencilBase() { }
    StencilBase(const string& name) :
        _name(name) { }

    virtual const string& get_name() { return _name; }
    
    // Calculate one scalar result.
    virtual void calc_scalar(StencilContext& context, idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) =0;

    // Calculate results within a region.
    virtual void calc_region(StencilContext& context, 
                             idx_t t,
                             idx_t begin_rn, idx_t begin_rx, idx_t begin_ry, idx_t begin_rz,
                             idx_t end_rn, idx_t end_rx, idx_t end_ry, idx_t end_rz) =0;
};

// A template that provides wrappers around a stencil class created
// by the foldBuilder.
template <typename StencilClass>
class StencilTemplate : public StencilBase {

protected:

    // _stencil must implement calc_scalar, calc_vector.
    StencilClass _stencil;

public:
    StencilTemplate(const string& name) :
        StencilBase(name) { }
    
    // Calculate one scalar result.
    virtual void calc_scalar(StencilContext& context, idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) {
        _stencil.calc_scalar(context, t, ARG_N(n) x, y, z);
    }

    // Calculate results within a vector cluster.
    // Called from calc_block().
    // The begin/end_c* vars are the start/stop_b* vars from the block loops.
    // This function doesn't contain any loops; it's just a wrapper around 
    // calc_vector.
    ALWAYS_INLINE void calc_cluster (StencilContext& context, idx_t t,
                                     idx_t begin_cnv, idx_t begin_cxv, idx_t begin_cyv, idx_t begin_czv,
                                     idx_t end_cnv, idx_t end_cxv, idx_t end_cyv, idx_t end_czv)
    {
        TRACE_MSG("%s.calc_cluster(%ld, %ld, %ld, %ld, %ld)",
                  _name.c_str(), t, begin_cnv, begin_cxv, begin_cyv, begin_czv);

        // The step vars are hard-coded in calc_block below, and there should
        // never be a partial step at this level. So, we can assume one var and
        // exactly CLEN_d steps in each given direction d are calculated in this
        // function.  Thus, we can ignore the end_* vars in the calc function.
        assert(end_cnv == begin_cnv + CLEN_N);
        assert(end_cxv == begin_cxv + CLEN_X);
        assert(end_cyv == begin_cyv + CLEN_Y);
        assert(end_czv == begin_czv + CLEN_Z);
        
        // Calculate results.
        _stencil.calc_vector(context, t, ARG_N(begin_cnv) begin_cxv, begin_cyv, begin_czv);
    }

    // Prefetch a cluster into L1.
    // Called from calc_block().
    ALWAYS_INLINE void prefetch_L1_cluster (StencilContext& context, idx_t t,
                                            idx_t begin_cnv, idx_t begin_cxv, idx_t begin_cyv, idx_t begin_czv,
                                            idx_t end_cnv, idx_t end_cxv, idx_t end_cyv, idx_t end_czv)
    {
        PREFETCH_CLUSTER(prefetch_L1_vector, t, begin_cnv, begin_cxv, begin_cyv, begin_czv);
    }
    
    // Prefetch a cluster into L2.
    // Called from calc_block().
    ALWAYS_INLINE void prefetch_L2_cluster (StencilContext& context, idx_t t,
                                            idx_t begin_cnv, idx_t begin_cxv, idx_t begin_cyv, idx_t begin_czv,
                                            idx_t end_cnv, idx_t end_cxv, idx_t end_cyv, idx_t end_czv)
    {
        PREFETCH_CLUSTER(prefetch_L2_vector, t, begin_cnv, begin_cxv, begin_cyv, begin_czv);
    }
    
    // Calculate results within a [cache] block.
    // Each block is typically computed in a separate OpenMP task.
    // The begin/end_b* vars are the start/stop_r* vars from the region loops.
    void calc_block(StencilContext& context, idx_t t,
                    idx_t begin_bn, idx_t begin_bx, idx_t begin_by, idx_t begin_bz,
                    idx_t end_bn, idx_t end_bx, idx_t end_by, idx_t end_bz)
    {
        TRACE_MSG("%s.calc_block(%ld, %ld, %ld, %ld, %ld)", 
                  _name.c_str(), t, begin_bn, begin_bx, begin_by, begin_bz);

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

    // Calculate results within a region.
    // Each region is typically computed in a separate OpenMP 'for' region.
    // The begin/end_r* vars are the start/stop_d* vars from the outer loops.
    virtual void calc_region(StencilContext& context, 
                             idx_t t,
                             idx_t begin_rn, idx_t begin_rx, idx_t begin_ry, idx_t begin_rz,
                             idx_t end_rn, idx_t end_rx, idx_t end_ry, idx_t end_rz)
    {
        TRACE_MSG("%s.calc_region(%ld, %ld, %ld, %ld, %ld)", 
                  _name.c_str(), t, begin_rn, begin_rx, begin_ry, begin_rz);

        // Steps based on block sizes.
        const idx_t step_rn = context.bn;
        const idx_t step_rx = context.bx;
        const idx_t step_ry = context.by;
        const idx_t step_rz = context.bz;
        
        // Start an OpenMP parallel region.
#pragma omp parallel
        {
            // Include automatically-generated loop code that calls calc_block().
            // This code typically contains OMP for loops.
#include "stencil_region_loops.hpp"
        }
    }
    
};

// List of stencils.
typedef vector<StencilBase*> Stencils;
