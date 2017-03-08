/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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

// Simple stencil that calculates an average of the points in a cube.
// Similar to MG_STENCIL_3D27PT heat-transfer stencil from miniGhost benchmark.
// Uses the 'w' dimension to index independent grids.

#include "StencilBase.hpp"

class AveStencil : public StencilRadiusBase {

protected:
    Grid heat;            // W time-varying 3D grids.
    
public:
    AveStencil(StencilList& stencils, int radius=2) :
        StencilRadiusBase("ave", stencils, radius)
    {
        INIT_GRID_5D(heat, t, w, x, y, z);
    }

    // Define equation for grid n at t as average of
    // (2*radius+1)^3 cube of values from grid w at t-1.
    virtual void define(const IntTuple& offsets) {
        GET_OFFSET(t);
        GET_OFFSET(w);
        GET_OFFSET(x);
        GET_OFFSET(y);
        GET_OFFSET(z);
        
        // add values in cube of desired size.
        int rBegin = -_radius;
        int rEnd = _radius;
        int nPts = 0;
        GridValue v;
        for (int rx = rBegin; rx <= rEnd; rx++)
            for (int ry = rBegin; ry <= rEnd; ry++)
                for (int rz = rBegin; rz <= rEnd; rz++) {
                    v += heat(t, w, x+rx, y+ry, z+rz);
                    nPts++;
                }

        // divide by number of points to find average.
        v *= 1.0 / double(nPts);

        // define the grid value at t+1 to be equivalent to v.
        heat(t+1, w, x, y, z) EQUALS v;
    }
};

REGISTER_STENCIL(AveStencil);
