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

// Calculate nreps time steps of stencil over grid using scalar code.
void calc_steps_ref(StencilContext& context, Stencils& stencils, const int nreps)
{
    printf("running %i reference time step(s)...\n", nreps);

    // time steps.
    // Start at a positive point to avoid any calculation referring
    // to negative time.
    idx_t t0 = TIME_DIM_SIZE * 2;
    for(idx_t t = t0; t < t0 + nreps; t += TIME_STEPS_PER_ITER) {

        // calculations to make.
        for (auto stencil : stencils) {
        
            // grid index (not used in most stencils).
            for (idx_t n = 0; n < context.dn; n++) {

#pragma omp parallel for
                for(idx_t ix = 0; ix < context.dx; ix++) {

                    CREW_FOR_LOOP
                        for(idx_t iy = 0; iy < context.dy; iy++) {

                            for(idx_t iz = 0; iz < context.dz; iz++) {

                                TRACE_MSG("%s.calc_scalar(%ld, %ld, %ld, %ld, %ld)", 
                                          stencil->get_name().c_str(), t, n, ix, iy, iz);
                            
                                // Evaluate the reference scalar code.
                                stencil->calc_scalar(context, t, n, ix, iy, iz);
                            }
                        }
                }
            }
        }
    } // iterations.
}
