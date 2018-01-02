/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

// Some very simple tests.

#include "Soln.hpp"

// Simple tests to increment values in N spatial dims.

class Test1dStencil : public StencilBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x); // time-varying grid.
    
public:

    Test1dStencil(StencilList& stencils) :
        StencilBase("test_1d", stencils) { }
    virtual ~Test1dStencil() { }

    // Define equation to apply to all points in 'data' grid.
    virtual void define() {

        // define the value at t+1.
        data(t+1, x) EQUALS data(t, x) + 1.0;
    }
};

REGISTER_STENCIL(Test1dStencil);

class Test2dStencil : public StencilBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x, y); // time-varying grid.
    
public:

    Test2dStencil(StencilList& stencils) :
        StencilBase("test_2d", stencils) { }
    virtual ~Test2dStencil() { }

    // Define equation to apply to all points in 'data' grid.
    virtual void define() {

        // define the value at t+1.
        data(t+1, x, y) EQUALS data(t, x, y) + 1.0;
    }
};

REGISTER_STENCIL(Test2dStencil);

class Test3dStencil : public StencilBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_DOMAIN_INDEX(z);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x, y, z); // time-varying grid.
    
public:

    Test3dStencil(StencilList& stencils) :
        StencilBase("test_3d", stencils) { }
    virtual ~Test3dStencil() { }

    // Define equation to apply to all points in 'data' grid.
    virtual void define() {

        // define the value at t+1.
        data(t+1, x, y, z) EQUALS data(t, x, y, z) + 1.0;
    }
};

REGISTER_STENCIL(Test3dStencil);

class Test4dStencil : public StencilBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(w);         // spatial dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_DOMAIN_INDEX(z);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, w, x, y, z); // time-varying grid.
    
public:

    Test4dStencil(StencilList& stencils) :
        StencilBase("test_4d", stencils) { }
    virtual ~Test4dStencil() { }

    // Define equation to apply to all points in 'data' grid.
    virtual void define() {

        // define the value at t+1.
        data(t+1, w, x, y, z) EQUALS data(t, w, x, y, z) + 1.0;
    }
};

REGISTER_STENCIL(Test4dStencil);


// A "stream-like" stencil that just reads and writes
// with no spatial offsets.
// The radius controls how many reads are done in the time domain.
// Running with radius=2 should give performance comparable to
// (but not identical to) the stream 'triad' benchmark.

class StreamStencil : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_DOMAIN_INDEX(z);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x, y, z); // time-varying 3D grid.
    
public:

    StreamStencil(StencilList& stencils, int radius=2) :
        StencilRadiusBase("test_stream", stencils, radius) { }
    virtual ~StreamStencil() { }

    // Define equation to read '_radius' values and write one.
    virtual void define() {

        GridValue v = constNum(1.0);

        // Add '_radius' values from past time-steps to ensure no spatial locality.
        for (int r = 0; r < _radius; r++) {
            v += data(t-r, x, y, z);
        }

        // define the value at t+1 to be equivalent to v.
        data(t+1, x, y, z) EQUALS v;
    }
};

REGISTER_STENCIL(StreamStencil);

// Reverse-time stencil.
// In this test, data(t-1) depends on data(t).

class TestReverseStencil : public StencilBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x, y);
    
public:

    TestReverseStencil(StencilList& stencils) :
        StencilBase("test_reverse", stencils) { }
    virtual ~TestReverseStencil() { }

    // Define equation to do simple test.
    virtual void define() {

        data(t-1, x, y) EQUALS data(t, x, y) + 5.0;
    }
};

REGISTER_STENCIL(TestReverseStencil);
