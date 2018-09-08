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

class Test1dStencil : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x); // time-varying grid.

public:

    Test1dStencil(StencilList& stencils, int radius=2) :
        StencilRadiusBase("test_1d", stencils, radius) { }
    virtual ~Test1dStencil() { }

    // Define equation to apply to all points in 'data' grid.
    virtual void define() {

        // define the value at t+1 using asymmetric stencil.
        GridValue v = data(t, x) + 1.0;
        for (int r = 1; r <= _radius; r++)
            v += data(t, x + r);
        for (int r = 1; r <= _radius + 2; r++)
            v += data(t, x - r);
        data(t+1, x) EQUALS v;
    }
};

REGISTER_STENCIL(Test1dStencil);

class Test2dStencil : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x, y); // time-varying grid.

public:

    Test2dStencil(StencilList& stencils, int radius=2) :
        StencilRadiusBase("test_2d", stencils, radius) { }
    virtual ~Test2dStencil() { }

    // Define equation to apply to all points in 'data' grid.
    virtual void define() {

        // define the value at t+1 using asymmetric stencil.
        GridValue v = data(t, x, y) + 1.0;
        for (int r = 1; r <= _radius; r++)
            v += data(t, x + r, y);
        for (int r = 1; r <= _radius + 1; r++)
            v += data(t, x - r, y);
        for (int r = 1; r <= _radius + 2; r++)
            v += data(t, x, y + r);
        for (int r = 1; r <= _radius + 3; r++)
            v += data(t, x, y - r);
        data(t+1, x, y) EQUALS v;
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
        for (int r = 0; r < _radius; r++)
            v += data(t-r, x, y, z);

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

// Test the use of scratch-pad grids.

class TestScratchStencil1 : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x); // time-varying grid.

    // Temporary storage.
    MAKE_SCRATCH_GRID(t1, x);

public:

    TestScratchStencil1(StencilList& stencils, int radius=2) :
        StencilRadiusBase("test_scratch1", stencils, radius) { }

    // Define equation to apply to all points in 'data' grid.
    virtual void define() {

        // Set scratch var w/an asymmetrical stencil.
        GridValue u = data(t, x);
        for (int r = 1; r <= _radius; r++)
            u += data(t, x-r);
        for (int r = 1; r <= _radius + 1; r++)
            u += data(t, x+r);
        t1(x) EQUALS u / (_radius * 2 + 2);

        // Set data from scratch vars w/an asymmetrical stencil.
        GridValue v = t1(x);
        for (int r = 1; r <= _radius + 2; r++)
            v += t1(x-r);
        for (int r = 1; r <= _radius + 3; r++)
            v += t1(x+r);
        data(t+1, x) EQUALS v / (_radius * 2 + 6);
    }
};

REGISTER_STENCIL(TestScratchStencil1);

class TestScratchStencil2 : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_DOMAIN_INDEX(z);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x, y, z); // time-varying grid.

    // Temporary storage.
    MAKE_SCRATCH_GRID(t1, x, y, z);
    MAKE_SCRATCH_GRID(t2, x, y, z);
    MAKE_SCRATCH_GRID(t3, x, y, z);

public:

    TestScratchStencil2(StencilList& stencils, int radius=2) :
        StencilRadiusBase("test_scratch2", stencils, radius) { }

    // Define equation to apply to all points in 'data' grid.
    virtual void define() {

        // Set scratch vars.
        GridValue ix = constNum(1.0);
        GridValue iy = constNum(2.0);
        GridValue iz = constNum(3.0);
        for (int r = 1; r <= _radius; r++) {
            ix += data(t, x-r, y, z);
            iy += data(t, x, y+r, z);
            iz += data(t, x, y, z-r) + data(t, x, y, z+r);
        }
        t1(x, y, z) EQUALS ix;
        t2(x, y, z) EQUALS iy - iz;

        // Set a scratch var from other scratch vars.
        t3(x, y, z) EQUALS t1(x-1, y+1, z) + t2(x, y, z-1);

        // Update data from scratch vars.
        GridValue v = constNum(4.0);
        for (int r = 1; r <= _radius; r++) {
            v += t1(x-r, y, z) - t1(x+r, y, z);
            v += t2(x, y, z-r) - t2(x, y, z);
            v += t3(x-r, y-r, z) - t3(x+r, y+r, z);
        }
        data(t+1, x, y, z) EQUALS data(t, x, y, z) + v;
    }
};

REGISTER_STENCIL(TestScratchStencil2);

// Test the use of sub-domains.

class TestSubdomainStencil1 : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x); // time-varying grid.

public:

    TestSubdomainStencil1(StencilList& stencils, int radius=2) :
        StencilRadiusBase("test_subdomain_1d", stencils, radius) { }

    // Define equation to apply to all points in 'data' grid.
    virtual void define() {

        // Sub-domain.
        Condition sd0 = (x >= first_index(x) + 5) && (x <= last_index(x) - 3);
        
        // Set data w/different stencils.

        GridValue u = data(t, x);
        for (int r = 1; r <= _radius; r++)
            u += data(t, x-r);
        for (int r = 1; r <= _radius + 1; r++)
            u += data(t, x+r);
        data(t+1, x) EQUALS u / (_radius * 2 + 2) IF sd0;

        GridValue v = data(t, x);
        for (int r = 1; r <= _radius + 3; r++)
            v += data(t, x-r);
        for (int r = 1; r <= _radius + 2; r++)
            v += data(t, x+r);
        data(t+1, x) EQUALS u / (_radius * 2 + 6) IF !sd0;
    }
};

REGISTER_STENCIL(TestSubdomainStencil1);

class TestSubdomainStencil3 : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_DOMAIN_INDEX(z);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x, y, z); // time-varying grid.

public:

    TestSubdomainStencil3(StencilList& stencils, int radius=2) :
        StencilRadiusBase("test_subdomain_3d", stencils, radius) { }

    // Define equation to apply to all points in 'data' grid.
    virtual void define() {

        // Sub-domain is rectangle interior.
        Condition sd0 =
            (x >= first_index(x) + 5) && (x <= last_index(x) - 3) &&
            (y >= first_index(y) + 4) && (y <= last_index(y) - 6) &&
            (z >= first_index(z) + 6) && (z <= last_index(z) - 4);
        
        // Set data w/different stencils.

        GridValue u = data(t, x, y, z);
        for (int r = 1; r <= _radius; r++)
            u += data(t, x-r, y, z) + data(t, x+r, y, z) +
                data(t, x, y-r, z) + data(t, x, y+r, z) +
                data(t, x, y, z-r) + data(t, x, y, z+r);
        data(t+1, x, y, z) EQUALS u / (_radius * 6 + 1) IF sd0;

        GridValue v = data(t, x, y, z);
        for (int r = 1; r <= _radius; r++)
            v += data(t, x-r, y-r, z-r) + data(t, x+r, y+r, z+r);
        data(t+1, x, y, z) EQUALS u / (_radius * 2 + 1) IF !sd0;
    }
};

REGISTER_STENCIL(TestSubdomainStencil3);

class TestStepCondStencil1 : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x); // time-varying grid.

public:

    TestStepCondStencil1(StencilList& stencils, int radius=2) :
        StencilRadiusBase("test_step_cond_1d", stencils, radius) { }

    // Define equation to apply to all points in 'data' grid.
    virtual void define() {

        // Time condition.
        Condition tc0 = (t % 2 == 0);
        
        // Set data w/different stencils.

        GridValue u = data(t, x);
        for (int r = 1; r <= _radius; r++)
            u += data(t, x-r);
        for (int r = 1; r <= _radius + 1; r++)
            u += data(t, x+r);
        data(t+1, x) EQUALS u / (_radius * 2 + 2) IF_STEP tc0;

        GridValue v = data(t, x);
        for (int r = 1; r <= _radius + 3; r++)
            v += data(t, x-r);
        for (int r = 1; r <= _radius + 2; r++)
            v += data(t, x+r);
        data(t+1, x) EQUALS u / (_radius * 2 + 6) IF_STEP !tc0;
    }
};

REGISTER_STENCIL(TestStepCondStencil1);

// A stencil that has grids, but no stencil equation.
class TestEmptyStencil1 : public StencilBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x); // time-varying grid.

public:

    TestEmptyStencil1(StencilList& stencils) :
        StencilBase("test_empty1", stencils) { }

    virtual void define() { }
};

REGISTER_STENCIL(TestEmptyStencil1);
