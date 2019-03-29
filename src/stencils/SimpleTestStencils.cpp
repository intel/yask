/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2019, Intel Corporation

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

// Simple tests for various YASK DSL features.

// A base class for stencil tests.
class TestBase : public StencilRadiusBase {
    
protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(w);         // spatial dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_DOMAIN_INDEX(z);         // spatial dim.

    // Define simple stencil from var 'V' at 't0' centered around 'x0'.
    // Extend given radius left and/or right w/'*_ext'.
    virtual GridValue def_1d(Grid& V, const GridIndex& t0, const GridIndex& x0,
                             int left_ext, int right_ext) {
        GridValue v;
        int n = 1;
        for (int i = -_radius - left_ext; i <= _radius + right_ext; i++, n++)
            v += V(t0, x0+i);
        return v / n;
    }

    // Define simple stencil from scratch var 'V' centered around 'x0'.
    // Similar to 'def_1d()', but doesn't use time var.
    virtual GridValue def_scratch_1d(Grid& V, const GridIndex& x0,
                                     int left_ext, int right_ext) {
        GridValue v;
        int n = 1;
        for (int i = -_radius - left_ext; i <= _radius + right_ext; i++, n++)
            v += V(x0+i);
        return v / n;
    }

    // Define simple stencil from var 'V' at 't0' centered around 'x0', 'y0'.
    // Extend given radius left and/or right w/'*_ext'.
    // Use some points from the entire rectangle, not just on the axes.
    virtual GridValue def_2d(Grid& V, const GridIndex& t0,
                             const GridIndex& x0,
                             int x_left_ext, int x_right_ext,
                             const GridIndex& y0,
                             int y_left_ext, int y_right_ext) {
        GridValue v;
        int n = 0;
        for (int i = -_radius - x_left_ext; i <= _radius + x_right_ext; i++)
            for (int j = -_radius - y_left_ext; j <= _radius + y_right_ext; j++, n++)
                if (i * j == 0 || i == j)
                    v += V(t0, x0+i, y0+j);
        return v / n;
    }    

    // Define simple stencil from var 'V' at 't0' centered around 'x0', 'y0', 'z0'.
    // Extend given radius left and/or right w/'*_ext'.
    // Use some points from the entire rectangular polytope, not just on the axes.
    virtual GridValue def_3d(Grid& V, const GridIndex& t0,
                             const GridIndex& x0,
                             int x_left_ext, int x_right_ext,
                             const GridIndex& y0,
                             int y_left_ext, int y_right_ext,
                             const GridIndex& z0,
                             int z_left_ext, int z_right_ext) {
        GridValue v;
        int n = 0;
        for (int i = -_radius - x_left_ext; i <= _radius + x_right_ext; i++)
            for (int j = -_radius - y_left_ext; j <= _radius + y_right_ext; j++)
                for (int k = -_radius - z_left_ext; k <= _radius + z_right_ext; k++, n++)
                    if (i * j * k == 0 || (i == j && j == k))
                        v += V(t0, x0+i, y0+j, z0+k);
        return v / n;
    }

    // Define simple stencil from scratch var 'V' centered around 'x0', 'y0', 'z0'.
    // Extend given radius left and/or right w/'*_ext'.
    // Use some points from the entire rectangular polytope, not just on the axes.
    virtual GridValue def_scratch_3d(Grid& V,
                                     const GridIndex& x0,
                                     int x_left_ext, int x_right_ext,
                                     const GridIndex& y0,
                                     int y_left_ext, int y_right_ext,
                                     const GridIndex& z0,
                                     int z_left_ext, int z_right_ext) {
        GridValue v;
        int n = 0;
        for (int i = -_radius - x_left_ext; i <= _radius + x_right_ext; i++)
            for (int j = -_radius - y_left_ext; j <= _radius + y_right_ext; j++)
                for (int k = -_radius - z_left_ext; k <= _radius + z_right_ext; k++, n++)
                    if (i * j * k == 0 || (i == j && j == k))
                        v += V(x0+i, y0+j, z0+k);
        return v / n;
    }    

    // Define simple stencil from var 'V' at 't0' centered around 'w0', 'x0', 'y0', 'z0'.
    // Extend given radius left and/or right w/'*_ext'.
    // Use some points from the entire rectangular polytope, not just on the axes.
    virtual GridValue def_4d(Grid& V, const GridIndex& t0,
                             const GridIndex& w0,
                             int w_left_ext, int w_right_ext,
                             const GridIndex& x0,
                             int x_left_ext, int x_right_ext,
                             const GridIndex& y0,
                             int y_left_ext, int y_right_ext,
                             const GridIndex& z0,
                             int z_left_ext, int z_right_ext) {
        GridValue v;
        int n = 0;
        for (int h : { -_radius - w_left_ext, 0, _radius + w_right_ext })
            for (int i : { -_radius - x_left_ext, 0, _radius + x_right_ext })
                for (int j : { -_radius - y_left_ext, 0, _radius + y_right_ext })
                    for (int k : { -_radius - z_left_ext, 0, _radius + z_right_ext }) {
                        n++;
                        if (h * i * j * k == 0 || (h == i && i == j && j == k))
                            v += V(t0, w0+h, x0+i, y0+j, z0+k);
                    }
        return v / n;
    }    

public:

    TestBase(const string& name, StencilList& stencils, int radius) :
        StencilRadiusBase(name, stencils, radius) { }
};

class Test1dStencil : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x); // time-varying grid.

public:

    Test1dStencil(StencilList& stencils, int radius=2) :
        TestBase("test_1d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // define the value at t+1 using asymmetric stencil.
        A(t+1, x) EQUALS def_1d(A, t, x, 0, 2);
    }
};

REGISTER_STENCIL(Test1dStencil);

class Test2dStencil : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x, y); // time-varying grid.

public:

    Test2dStencil(StencilList& stencils, int radius=2) :
        TestBase("test_2d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // define the value at t+1 using asymmetric stencil.
        A(t+1, x, y) EQUALS def_2d(A, t, x, 0, 2, y, 4, 3);
    }
};

REGISTER_STENCIL(Test2dStencil);

class Test3dStencil : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x, y, z); // time-varying grid.

public:

    Test3dStencil(StencilList& stencils, int radius=2) :
        TestBase("test_3d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // define the value at t+1 using asymmetric stencil.
        A(t+1, x, y, z) EQUALS def_3d(A, t, x, 0, 2, y, 4, 3, z, 2, 1);
    }
};

REGISTER_STENCIL(Test3dStencil);

class Test4dStencil : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, w, x, y, z); // time-varying grid.

public:

    Test4dStencil(StencilList& stencils, int radius=1) :
        TestBase("test_4d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // define the value at t+1 using asymmetric stencil.
        A(t+1, w, x, y, z) EQUALS def_4d(A, t, w, 1, 2, x, 0, 2, y, 2, 1, z, 1, 0);
    }
};

REGISTER_STENCIL(Test4dStencil);

// Test misc indices.
class TestMisc2dStencil : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_MISC_INDEX(a);
    MAKE_MISC_INDEX(b);
    MAKE_MISC_INDEX(c);
    
    // Time-varying grid. Intermix last domain dim with misc dims to make
    // sure compiler creates correct layout.
    MAKE_GRID(A, t, x, a, y, b, c); 

public:

    TestMisc2dStencil(StencilList& stencils, int radius=2) :
        StencilRadiusBase("test_misc_2d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // Define the value at t+1 using asymmetric stencil
        // with various pos & neg indices in misc dims.
        GridValue v = A(t, x, 0, y, -1, 2) + 1.0;
        for (int r = 1; r <= _radius; r++)
            v += A(t, x + r, 3, y, 0, 1);
        for (int r = 1; r <= _radius + 1; r++)
            v += A(t, x - r, 4, y, 2, 1);
        for (int r = 1; r <= _radius + 2; r++)
            v += A(t, x, -2, y + r, 2, 0);
        for (int r = 1; r <= _radius + 3; r++)
            v += A(t, x, 0, y - r, 0, -1);
        A(t+1, x, 1, y, 2, 3) EQUALS v;
    }
};

REGISTER_STENCIL(TestMisc2dStencil);


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
    MAKE_GRID(A, t, x, y, z); // time-varying 3D grid.

public:

    StreamStencil(StencilList& stencils, int radius=2) :
        StencilRadiusBase("test_stream", stencils, radius) { }
    virtual ~StreamStencil() { }

    // Define equation to read '_radius' values and write one.
    virtual void define() {

        GridValue v = constNum(1.0);

        // Add '_radius' values from past time-steps to ensure no spatial locality.
        for (int r = 0; r < _radius; r++)
            v += A(t-r, x, y, z);

        // define the value at t+1 to be equivalent to v.
        A(t+1, x, y, z) EQUALS v;
    }
};

REGISTER_STENCIL(StreamStencil);

// Reverse-time stencil.
// In this test, A(t-1) depends on A(t).

class TestReverseStencil : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x, y);

public:

    TestReverseStencil(StencilList& stencils, int radius=2) :
        TestBase("test_reverse", stencils, radius) { }

    // Define equation to do simple test.
    virtual void define() {

        // Like the previous 2D test, but defines value at 't-1'.
        A(t-1, x, y) EQUALS def_2d(A, t, x, 0, 2, y, 4, 3);
    }
};

REGISTER_STENCIL(TestReverseStencil);

// Test dependent equations.
// These will create 2 stencil packs that will be applied in sequence
// for each time-step.
class TestDepStencil1 : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x); // time-varying grid.
    MAKE_GRID(B, t, x); // time-varying grid.

public:

    TestDepStencil1(StencilList& stencils, int radius=2) :
        TestBase("test_dep_1d", stencils, radius) { }

    // Define equation to apply to all points in 'A' and 'B' grids.
    virtual void define() {

        // Define A(t+1) from A(t) & stencil at B(t).
        A(t+1, x) EQUALS A(t, x) - def_1d(B, t, x, 0, 1);

        // Define B(t+1) from B(t) & stencil at A(t+1).
        B(t+1, x) EQUALS B(t, x) - def_1d(A, t+1, x, 3, 2);
    }
};

REGISTER_STENCIL(TestDepStencil1);

class TestDepStencil2 : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x, y); // time-varying grid.
    MAKE_GRID(B, t, x, y); // time-varying grid.

public:

    TestDepStencil2(StencilList& stencils, int radius=2) :
        TestBase("test_dep_2d", stencils, radius) { }

    // Define equation to apply to all points in 'A' and 'B' grids.
    virtual void define() {

        // Define A(t+1) from A(t) & stencil at B(t).
        A(t+1, x, y) EQUALS A(t, x, y) - def_2d(B, t, x, 0, 1, y, 2, 1);

        // Define B(t+1) from B(t) & stencil at A(t+1).
        B(t+1, x, y) EQUALS B(t, x, y) - def_2d(A, t+1, x, 3, 2, y, 0, 1);
    }
};

REGISTER_STENCIL(TestDepStencil2);

// Test the use of scratch-pad grids.

class TestScratchStencil1 : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x); // time-varying grid.

    // Temporary storage.
    MAKE_SCRATCH_GRID(B, x);

public:

    TestScratchStencil1(StencilList& stencils, int radius=2) :
        TestBase("test_scratch_1d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // Define values in scratch var 'B'.
        B(x) EQUALS def_1d(A, t, x, 1, 0);

        // Set 'A' from scratch var values.
        A(t+1, x) EQUALS def_scratch_1d(B, x-4, 2, 3) + def_scratch_1d(B, x+6, 0, 1);
    }
};

REGISTER_STENCIL(TestScratchStencil1);

class TestScratchStencil3 : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x, y, z); // time-varying grid.

    // Temporary storage.
    MAKE_SCRATCH_GRID(t1, x, y, z);
    MAKE_SCRATCH_GRID(t2, x, y, z);
    MAKE_SCRATCH_GRID(t3, x, y, z);

public:

    TestScratchStencil3(StencilList& stencils, int radius=2) :
        TestBase("test_scratch_3d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // Set scratch vars.
        t1(x, y, z) EQUALS def_3d(A, t, x, 0, 1, y, 2, 1, z, 1, 0);
        t2(x, y, z) EQUALS def_3d(A, t, x, 1, 0, y, 0, 2, z, 0, 1);

        // Set a scratch var from other scratch vars.
        t3(x, y, z) EQUALS t1(x-1, y+1, z) + t2(x, y, z-1);

        // Update A from scratch vars.
        A(t+1, x, y, z) EQUALS A(t, x, y, z) +
            def_scratch_3d(t1, x, 2, 0, y, 0, 1, z, 1, 0) +
            def_scratch_3d(t3, x, 1, 0, y, 0, 1, z, 0, 2);
    }
};

REGISTER_STENCIL(TestScratchStencil3);

// Test the use of sub-domains.
class TestSubdomainStencil1 : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x); // time-varying grid.

public:

    TestSubdomainStencil1(StencilList& stencils, int radius=2) :
        TestBase("test_subdomain_1d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // Define interior sub-domain.
        Condition sd0 = (x >= first_index(x) + 5) && (x <= last_index(x) - 3);
        
        // Define interior points.
        GridValue u = def_1d(A, t, x, 0, 1);
        A(t+1, x) EQUALS u IF sd0;

        // Define exterior points.
        A(t+1, x) EQUALS -u IF !sd0;
    }
};

REGISTER_STENCIL(TestSubdomainStencil1);

class TestSubdomainStencil2 : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x, y); // time-varying grid.

public:

    TestSubdomainStencil2(StencilList& stencils, int radius=2) :
        TestBase("test_subdomain_2d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // Sub-domain is rectangle interior.
        Condition sd0 =
            (x >= first_index(x) + 5) && (x <= last_index(x) - 3) &&
            (y >= first_index(y) + 4) && (y <= last_index(y) - 6);
        
        // Set A w/different stencils depending on condition.
        A(t+1, x, y) EQUALS def_2d(A, t, x, 0, 2, y, 1, 0) IF sd0;
        A(t+1, x, y) EQUALS def_2d(A, t, x, 1, 0, y, 0, 2) IF !sd0;
    }
};

REGISTER_STENCIL(TestSubdomainStencil2);

class TestSubdomainStencil3 : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x, y, z); // time-varying grid.

public:

    TestSubdomainStencil3(StencilList& stencils, int radius=2) :
        TestBase("test_subdomain_3d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // Sub-domain is rectangle interior.
        Condition sd0 =
            (x >= first_index(x) + 5) && (x <= last_index(x) - 3) &&
            (y >= first_index(y) + 4) && (y <= last_index(y) - 6) &&
            (z >= first_index(z) + 6) && (z <= last_index(z) - 4);
        
        // Set A w/different stencils depending on condition.
        A(t+1, x, y, z) EQUALS def_3d(A, t, x, 0, 2, y, 1, 0, z, 0, 1) IF sd0;
        A(t+1, x, y, z) EQUALS def_3d(A, t, x, 1, 0, y, 0, 2, z, 1, 0) IF !sd0;
    }
};

REGISTER_STENCIL(TestSubdomainStencil3);

// Test step condition.
class TestStepCondStencil1 : public TestBase {

protected:

    // Indices.
    MAKE_MISC_INDEX(b);

    // Vars.
    MAKE_GRID(A, t, x); // time-varying grid.
    MAKE_ARRAY(B, b);

public:

    TestStepCondStencil1(StencilList& stencils, int radius=2) :
        TestBase("test_step_cond_1d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // Time condition.
        Condition tc0 = (t % 2 == 0);

        // Var condition.
        Condition vc0 = (B(0) > B(1));
        
        // Set A w/different stencils depending on the conditions.  It is
        // the programmer's responsibility to ensure that the conditions are
        // exclusive when necessary. It is not checked at compile or
        // run-time.

        // Use this equation when t is even.
        A(t+1, x) EQUALS def_1d(A, t, x, 0, 0) IF_STEP tc0;

        // Use this equation when t is odd and B(0) > B(1).
        A(t+1, x) EQUALS def_1d(A, t, x, 1, 2) IF_STEP !tc0 && vc0;

        // Use this equation when t is even and B(0) <= B(1).
        A(t+1, x) EQUALS def_1d(A, t, x, 2, 0) IF_STEP !tc0 && !vc0;
    }
};

REGISTER_STENCIL(TestStepCondStencil1);

// Test the use of conditional updates with scratch-pad grids.
class TestScratchSubdomainStencil1 : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x); // time-varying grid.

    // Temporary storage.
    MAKE_SCRATCH_GRID(B, x);

public:

    TestScratchSubdomainStencil1(StencilList& stencils, int radius=2) :
        TestBase("test_scratch_subdomain_1d", stencils, radius) { }

    // Define equation to apply to all points in 'A' grid.
    virtual void define() {

        // Define values in scratch var 'B' using current values from 'A'.
        B(x) EQUALS def_1d(A, t, x, 1, 0);

        // Define sub-domain.
        Condition sd0 = (x >= first_index(x) + 5) && (x <= last_index(x) - 3);
        
        // Define next values for 'A' from scratch var values.
        auto v = def_scratch_1d(B, x-6, 2, 3) - def_scratch_1d(B, x+7, 0, 2);
        A(t+1, x) EQUALS v IF sd0;
        A(t+1, x) EQUALS -v IF !sd0;
    }
};

REGISTER_STENCIL(TestScratchSubdomainStencil1);

// A stencil that uses svml math functions.
class TestFuncStencil1 : public TestBase {

protected:

    // Vars.
    MAKE_GRID(A, t, x);
    MAKE_GRID(B, t, x);
    MAKE_GRID(C, t, x);

public:

    TestFuncStencil1(StencilList& stencils, int radius=1) :
        TestBase("test_func_1d", stencils, radius) { }

    virtual void define() {
        A(t+1, x) EQUALS cos(A(t, x)) - 2 * sin(A(t, x));
        B(t+1, x) EQUALS pow(def_1d(B, t, x, 0, 1), 1.0/2.5);
        C(t+1, x) EQUALS atan(def_1d(A, t+1, x, 1, 0) + cbrt(C(t, x+1)));
    }
};

REGISTER_STENCIL(TestFuncStencil1);

// A stencil that has grids, but no stencil equation.
class TestEmptyStencil1 : public StencilBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.

    // Vars.
    MAKE_GRID(A, t, x); // time-varying grid.

public:

    TestEmptyStencil1(StencilList& stencils) :
        StencilBase("test_empty1", stencils) { }

    virtual void define() { }
};

REGISTER_STENCIL(TestEmptyStencil1);
