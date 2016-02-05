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

#include "stencil.hpp"

// Include auto-generated vectorized stencil code.
#include "stencil_vector_code.hpp"

//////////////// Macros for prefetching /////////////////

// Prefetch a cluser starting at vector indices veci, vecj, veck.
// Generic macro to handle different cache levels and directions.
// Called from generated loop code.
#define PREFETCH_CLUSTER(fn, context, t0, v0, veci, vecj, veck)       \
    do {                                                                \
        TRACE_MSG("prefetch_cluster(%s, context, %d, %d, %d, %d, %d)", \
                  #fn, t0, v0, veci, vecj, veck);                       \
        fn(context, t0, v0, veci, vecj, veck);              \
    } while(0)

// Use macros to call the correct prefetch function in each direction.
// TODO: generate the correct code automatically and remove these macros.
#define prefetch_L1_cluster(context, t0, v0, veci, vecj, veck, \
                             ev, eveci, evecj, eveck)       \
    PREFETCH_CLUSTER(prefetch_L1_stencil_vector, context, t0, v0, veci, vecj, veck)
#define prefetch_L1_cluster_bx(context, t0, v0, veci, vecj, veck, \
                             ev, eveci, evecj, eveck)               \
    PREFETCH_CLUSTER(prefetch_L1_stencil_vector_x, context, t0, v0, veci, vecj, veck)
#define prefetch_L1_cluster_by(context, t0, v0, veci, vecj, veck, \
                             ev, eveci, evecj, eveck)               \
    PREFETCH_CLUSTER(prefetch_L1_stencil_vector_y, context, t0, v0, veci, vecj, veck)
#define prefetch_L1_cluster_bz(context, t0, v0, veci, vecj, veck, \
                             ev, eveci, evecj, eveck)               \
    PREFETCH_CLUSTER(prefetch_L1_stencil_vector_z, context, t0, v0, veci, vecj, veck)

#define prefetch_L2_cluster(context, t0, v0, veci, vecj, veck, \
                             ev, eveci, evecj, eveck)               \
    PREFETCH_CLUSTER(prefetch_L2_stencil_vector, context, t0, v0, veci, vecj, veck)
#define prefetch_L2_cluster_bx(context, t0, v0, veci, vecj, veck, \
                             ev, eveci, evecj, eveck)               \
    PREFETCH_CLUSTER(prefetch_L2_stencil_vector_x, context, t0, v0, veci, vecj, veck)
#define prefetch_L2_cluster_by(context, t0, v0, veci, vecj, veck, \
                             ev, eveci, evecj, eveck)               \
    PREFETCH_CLUSTER(prefetch_L2_stencil_vector_y, context, t0, v0, veci, vecj, veck)
#define prefetch_L2_cluster_bz(context, t0, v0, veci, vecj, veck, \
                             ev, eveci, evecj, eveck)               \
    PREFETCH_CLUSTER(prefetch_L2_stencil_vector_z, context, t0, v0, veci, vecj, veck)

//////////////// Main stencil loops /////////////////

// Calculate results within a vector cluster.
// The begin/end_c* vars are the start/stop_b* vars from the block loops.
ALWAYS_INLINE static
void calc_cluster (StencilContext& context, int t0, 
                   int begin_cv, int begin_cx, int begin_cy, int begin_cz,
                   int end_cv, int end_cx, int end_cy, int end_cz)
{
    TRACE_MSG("calc_cluster(%d, %d, %d, %d, %d)", 
              t0, begin_cv, begin_cx, begin_cy, begin_cz);

    // The step vars are hard-coded in calc_block below, and there should
    // never be a partial step at this level. So, we can assume one var and
    // exactly CLEN_d steps in each given direction d are calculated in this
    // function.  Thus, we can ignore the end_* vars.
    assert(end_cv == begin_cv + 1);
    assert(end_cx == begin_cx + CLEN_X);
    assert(end_cy == begin_cy + CLEN_Y);
    assert(end_cz == begin_cz + CLEN_Z);

    // Calculate results.
    calc_stencil_vector(context, t0, begin_cv,
                        begin_cx, begin_cy, begin_cz);
}

// Calculate results within a cache block.
// Each block is typically computed in a separate OpenMP task.
// The begin/end_b* vars are the start/stop_r* vars from the region loops.
ALWAYS_INLINE static
void calc_block(StencilContext& context, int t0,
                   int begin_bv, int begin_bx, int begin_by, int begin_bz,
                   int end_bv, int end_bx, int end_by, int end_bz)
{
    TRACE_MSG("calc_block(%d, %d, %d, %d, %d)", 
              t0, begin_bv, begin_bx, begin_by, begin_bz);
    const MATRIX_TYPE& data = context.grid->getData();

    // Steps in vector lengths based on cluster lengths.
    const int step_bv = 1;
    const int step_bx = CLEN_X;
    const int step_by = CLEN_Y;
    const int step_bz = CLEN_Z;

    // Include automatically-generated loop code that calls calc_cluster().
#include "stencil_block_loops.hpp"
}

// Calculate results within a region.
// Each region is typically computed in a separate OpenMP 'for' region.
// The begin/end_r* vars are the start/stop_d* vars from the outer loops.
void calc_region(StencilContext& context, int t0,
                   int begin_rv, int begin_rx, int begin_ry, int begin_rz,
                   int end_rv, int end_rx, int end_ry, int end_rz)
{
    TRACE_MSG("calc_region(%d, %d, %d, %d, %d)", 
              t0, begin_rv, begin_rx, begin_ry, begin_rz);
    const MATRIX_TYPE& data = context.grid->getData();

    // Steps in vector lengths based on block sizes.
    const int step_rv = 1;
    const int step_rx = context.bx;
    const int step_ry = context.by;
    const int step_rz = context.bz;

    // Start an OpenMP parallel region.
#if !defined(DEBUG) && defined(__INTEL_COMPILER)
#pragma forceinline recursive
#endif
#pragma omp parallel
    {
        // Include automatically-generated loop code that calls calc_block().
        // This code typically contains OMP for loops.
#include "stencil_region_loops.hpp"
    }
}

// Calculate nreps time steps of stencil over grid using optimized code.
void calc_steps_opt(StencilContext& context, const int nreps)
{
    TRACE_MSG("calc_steps_opt(%d)", nreps);
    assert(context.grid);
    const MATRIX_TYPE& data = context.grid->getData();

    // problem begin points.
    const int begin_dv = 0, begin_dx = 0, begin_dy = 0, begin_dz = 0;
    
    // problem end-points in vector lengths.
    const int end_dv = NUM_VARS;
    const int end_dx = context.dx;
    const int end_dy = context.dy;
    const int end_dz = context.dz;

    // steps in vector lengths based on region sizes.
    const int step_dv = NUM_WORKS;
    const int step_dx = context.rx;
    const int step_dy = context.ry;
    const int step_dz = context.rz;
    
    printf("running %i optimized time steps(s)...\n", nreps);

    // For each iteration, calculate output for t0+TIME_STEPS based on t0.
    for(int t0 = 1; t0 <= nreps; t0 += TIME_STEPS) {

        // Include automatically-generated loop code that calls calc_region().
#include "stencil_outer_loops.hpp"

    } // iterations
}
