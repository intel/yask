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

// Include auto-generated scalar stencil code.
#include "stencil_scalar_code.hpp"

// Calculate nreps time steps of stencil over grid using scalar code.
void calc_steps_ref(StencilContext& context, const int nreps)
{
    // get problem sizes (in points, not vector lengths).
    MainGrid* grid = context.grid;
    assert(grid);

    // time steps.
    printf("running %i reference time step(s)...\n", nreps);
    for(int t0 = 1; t0 <= nreps; t0 += TIME_STEPS) {

        // variables.
        for (int v0 = 0; v0 < NUM_VARS; v0++) {

#pragma omp parallel for
            for(int iz = 0; iz < context.dz * VLEN_Z; iz++) {

                CREW_FOR_LOOP
                    for(int iy = 0; iy < context.dy * VLEN_Y; iy++) {

                        for(int ix = 0; ix < context.dx * VLEN_X; ix++) {

                            TRACE_MSG("calc_scalar(%d, %d, %d, %d, %d)", 
                                      t0, v0, ix, iy, iz);
                            
                            // Evaluate the reference scalar code.
                            calc_stencil_scalar(context, t0, v0, ix, iy, iz);
                        }
                    }
            }
        }
    } // iterations.
}
