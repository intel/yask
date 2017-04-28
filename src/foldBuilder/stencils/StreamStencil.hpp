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

// Implement a "stream-like" stencil that just reads and writes
// with no spatial offsets.
// The radius controls how many reads are done.

#include "StencilBase.hpp"

class StreamStencil : public StencilRadiusBase {

protected:
    Grid grid;
    
public:

    StreamStencil(StencilList& stencils, int radius=8) :
        StencilRadiusBase("stream", stencils, radius)
    {
        INIT_GRID_4D(grid, t, x, y, z);
    }
    virtual ~StreamStencil() { }

    // Define equation to read '_radius' values and write one.
    virtual void define(const IntTuple& offsets) {
        GET_OFFSET(t);
        GET_OFFSET(x);
        GET_OFFSET(y);
        GET_OFFSET(z);

        GridValue v = constNum(1.0);

        // Add '_radius' values from past time-steps to ensure no spatial locality.
        for (int r = 0; r < _radius; r++) {
            v += grid(t-r, x, y, z);
        }

        // define the value at t+1 to be equivalent to v.
        grid(t+1, x, y, z) EQUALS v;
    }
};

REGISTER_STENCIL(StreamStencil);
