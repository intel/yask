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

// Calculate nreps time steps of stencil over grid using optimized code.
void calc_steps_opt(StencilContext& context, Stencils& stencils, const int nreps)
{
    TRACE_MSG("calc_steps_opt(%d)", nreps);

    // problem begin points.
    const idx_t begin_dn = 0, begin_dx = 0, begin_dy = 0, begin_dz = 0;
    
    // problem end-points.
    const idx_t end_dn = context.dn;
    const idx_t end_dx = context.dx;
    const idx_t end_dy = context.dy;
    const idx_t end_dz = context.dz;

    // steps based on region sizes.
    const idx_t step_dn = context.rn;
    const idx_t step_dx = context.rx;
    const idx_t step_dy = context.ry;
    const idx_t step_dz = context.rz;
    printf("running %i optimized time step(s)...\n", nreps);
    
    // time steps.
    // Start at a positive point to avoid any calculation referring
    // to negative time.
    idx_t t0 = TIME_DIM_SIZE * 2;
    for(idx_t t = t0; t < t0 + nreps; t += TIME_STEPS_PER_ITER) {

        // calculations to make.
        for (auto stencil : stencils) {
        
            // Include automatically-generated loop code that calls calc_region().
#include "stencil_outer_loops.hpp"

        } // calcs.
    } // iterations
}
