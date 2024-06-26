/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2024, Intel Corporation

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

// Tests for various YASK DSL features.
// Most tests use whole-number calculations for more robust error-checking.

// YASK stencil solution(s) in this file will be integrated into the YASK compiler utility.
#include "yask_compiler_api.hpp"
using namespace std;
using namespace yask;

// Create an anonymous namespace to ensure that types are local.
namespace {

    // A base class for stencil tests.
    class TestBase : public yc_solution_with_radius_base {
    
    protected:

        // Indices & dimensions.
        // Not all these will be used in all tests.
        MAKE_STEP_INDEX(t);                          // step in time dim.
        MAKE_DOMAIN_INDEX(w);                        // spatial dim.
        MAKE_DOMAIN_INDEX(x);                        // spatial dim.
        MAKE_DOMAIN_INDEX(y);                        // spatial dim.
        MAKE_DOMAIN_INDEX(z);                        // spatial dim.

        // Define some stencils in different dimensions.
        // The size is based on the 'radius' option.
        // These will be asymmetrical if any of the '*_ext' params are not the same;
    
        // Define simple stencil from var 'V' at 't0' centered around 'x0'.
        // Extend given radius left and/or right w/'*_ext'.
        virtual yc_number_node_ptr def_t1d(yc_var_proxy& V, const yc_number_node_ptr& t0,
                                           const yc_number_node_ptr& x0,
                                           int left_ext, int right_ext) {
            auto r = get_radius();
            yc_number_node_ptr v = new_number_node(2.0);
            for (int i = -r - left_ext; i <= r + right_ext; i++)
                v += V(t0, x0+i);
            return v;
        }

        // Define simple stencil from scratch or read-only var 'V' centered
        // around 'x0'.  Similar to 'def_t1d()', but doesn't use step var.
        virtual yc_number_node_ptr def_1d(yc_var_proxy& V,
                                          const yc_number_node_ptr& x0,
                                          int left_ext, int right_ext) {
            auto r = get_radius();
            yc_number_node_ptr v = new_number_node(3.0);
            for (int i = -r - left_ext; i <= r + right_ext; i++)
                v += V(x0+i);
            return v;
        }

        // Define simple stencil from var 'V' at 't0' centered around 'x0', 'y0'.
        // Extend given radius left and/or right w/'*_ext'.
        // Use some points from the entire rectangle, not just on the axes.
        virtual yc_number_node_ptr def_t2d(yc_var_proxy& V, const yc_number_node_ptr& t0,
                                           const yc_number_node_ptr& x0,
                                           int x_left_ext, int x_right_ext,
                                           const yc_number_node_ptr& y0,
                                           int y_left_ext, int y_right_ext) {
            auto r = get_radius();
            yc_number_node_ptr v = new_number_node(4.0);
            for (int i : { -r - x_left_ext, 0, r + x_right_ext })
                for (int j : { -r - y_left_ext, 0, r + y_right_ext })
                    v += V(t0, x0+i, y0+j);
            return v;
        }    

        // Define simple stencil from scratch or read-only var 'V'
        // centered around 'x0', 'y0'.  Extend given radius left and/or right
        // w/'*_ext'.
        virtual yc_number_node_ptr def_2d(yc_var_proxy& V,
                                          const yc_number_node_ptr& x0,
                                          int x_left_ext, int x_right_ext,
                                          const yc_number_node_ptr& y0,
                                          int y_left_ext, int y_right_ext) {
            auto r = get_radius();
            yc_number_node_ptr v = new_number_node(5.0);
            for (int i : { -r - x_left_ext, 0, r + x_right_ext })
                for (int j : { -r - y_left_ext, 0, r + y_right_ext })
                    v += V(x0+i, y0+j);
            return v;
        }    

        // Define simple stencil from var 'V' at 't0' centered around 'x0', 'y0', 'z0'.
        // Extend given radius left and/or right w/'*_ext'.
        // Use some points from the entire rectangular polytope, not just on the axes.
        virtual yc_number_node_ptr def_t3d(yc_var_proxy& V, const yc_number_node_ptr& t0,
                                           const yc_number_node_ptr& x0,
                                           int x_left_ext, int x_right_ext,
                                           const yc_number_node_ptr& y0,
                                           int y_left_ext, int y_right_ext,
                                           const yc_number_node_ptr& z0,
                                           int z_left_ext, int z_right_ext) {
            auto r = get_radius();
            yc_number_node_ptr v = V(t0, x0, y0, z0);
            for (int i : { -r - x_left_ext, r + x_right_ext })
                for (int j : { -r - y_left_ext, r + y_right_ext })
                    for (int k : { -r - z_left_ext, r + z_right_ext })
                        v += V(t0, x0+i, y0+j, z0+k);
            return v;
        }

        // Define simple stencil from scratch or read-only var 'V' centered
        // around 'x0', 'y0', 'z0'.  Extend given radius left and/or right
        // w/'*_ext'.
        virtual yc_number_node_ptr def_3d(yc_var_proxy& V,
                                          const yc_number_node_ptr& x0,
                                          int x_left_ext, int x_right_ext,
                                          const yc_number_node_ptr& y0,
                                          int y_left_ext, int y_right_ext,
                                          const yc_number_node_ptr& z0,
                                          int z_left_ext, int z_right_ext) {
            auto r = get_radius();
            yc_number_node_ptr v = V(x0, y0, z0);
            for (int i : { -r - x_left_ext, r + x_right_ext })
                for (int j : { -r - y_left_ext, r + y_right_ext })
                    for (int k : { -r - z_left_ext, r + z_right_ext })
                        v += V(x0+i, y0+j, z0+k);
            return v;
        }    

        // Define simple stencil from var 'V' at 't0' centered around 'w0', 'x0', 'y0', 'z0'.
        // Extend given radius left and/or right w/'*_ext'.
        // Use some points from the entire rectangular polytope, not just on the axes.
        virtual yc_number_node_ptr def_t4d(yc_var_proxy& V, const yc_number_node_ptr& t0,
                                           const yc_number_node_ptr& w0,
                                           int w_left_ext, int w_right_ext,
                                           const yc_number_node_ptr& x0,
                                           int x_left_ext, int x_right_ext,
                                           const yc_number_node_ptr& y0,
                                           int y_left_ext, int y_right_ext,
                                           const yc_number_node_ptr& z0,
                                           int z_left_ext, int z_right_ext) {
            auto r = get_radius();
            yc_number_node_ptr v = V(t0, w0, x0, y0, z0);
            for (int h : { -r - w_left_ext, r + w_right_ext })
                for (int i : { -r - x_left_ext, r + x_right_ext })
                    for (int j : { -r - y_left_ext, r + y_right_ext })
                        for (int k : { -r - z_left_ext, r + z_right_ext })
                            v += V(t0, w0+h, x0+i, y0+j, z0+k);
            return v;
        }    

    public:

        TestBase(const string& name, int radius) :
            yc_solution_with_radius_base(name, radius) { }
    };

    // Simple 1D test.
    class Test1dStencil : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x); // time-varying var.

    public:

        Test1dStencil(int radius=2) :
            TestBase("test_1d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // define the value at t+1 using asymmetric stencil.
            A(t+1, x) EQUALS def_t1d(A, t, x, 0, 2);
        }
    };

    // Create an object of type 'Test1dStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(Test1dStencil);

    // Simple 2D test.
    class Test2dStencil : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x, y); // time-varying var.

    public:

        Test2dStencil(int radius=2) :
            TestBase("test_2d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // define the value at t+1 using asymmetric stencil.
            A(t+1, x, y) EQUALS def_t2d(A, t, x, 0, 2, y, 4, 3);
        }
    };

    // Create an object of type 'Test2dStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(Test2dStencil);

    // Simple 3D test.
    class Test3dStencil : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x, y, z); // time-varying var.

    public:

        Test3dStencil(int radius=2) :
            TestBase("test_3d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // define the value at t+1 using asymmetric stencil.
            A(t+1, x, y, z) EQUALS def_t3d(A, t, x, 0, 2, y, 4, 3, z, 2, 1);
        }
    };

    // Create an object of type 'Test3dStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(Test3dStencil);

    // Simple 4D test.
    class Test4dStencil : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, w, x, y, z); // time-varying var.

    public:

        Test4dStencil(int radius=1) :
            TestBase("test_4d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // define the value at t+1 using asymmetric stencil.
            A(t+1, w, x, y, z) EQUALS def_t4d(A, t, w, 1, 2, x, 0, 2, y, 2, 1, z, 1, 0);
        }
    };

    // Create an object of type 'Test4dStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(Test4dStencil);

    // Test vars that don't cover all domain dims.
    class TestPartialStencil3 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x, y, z); // time-varying var.
        MAKE_VAR(B, x); // 1D.
        MAKE_VAR(C, y);
        MAKE_VAR(D, z);
        MAKE_VAR(E, x, y); // 2D.
        MAKE_VAR(F, y, z);
        MAKE_VAR(G, z, y);
        MAKE_VAR(H, y, z, x); // 3D in different order.
        MAKE_SCALAR_VAR(I); // scalar.
        MAKE_VAR(J, t); // time-only.
        MAKE_VAR(K, t, y); // time + 1D.
        MAKE_VAR(L, t, y, z); // time + 2D.

    public:

        TestPartialStencil3(int radius=2) :
            TestBase("test_partial_3d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // define the value at t+1 using asymmetric stencil.
            A(t+1, x, y, z) EQUALS
                def_t3d(A, t, x, 0, 2, y, 4, 3, z, 2, 1) +
                def_1d(B, x, 0, 1) +
                def_1d(C, y, 1, 0) +
                def_1d(D, z, 0, 0) +
                def_2d(E, x, 0, 0, y, 1, 0) +
                def_2d(F, y, 0, 1, z, 0, 0) +
                def_2d(G, z, 1, 0, y, 0, 1) +
                def_3d(H, y, 1, 0, z, 0, 1, x, 1, 0) +
                I +
                J(t) +
                def_t1d(K, t, y, 0, 1) +
                def_t2d(L, t, y, 1, 0, z, 0, 1);
        }
    };

    // Create an object of type 'TestPartialStencil3',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestPartialStencil3);

    // Test misc indices.
    class TestMisc2dStencil : public yc_solution_with_radius_base {

    protected:

        // Indices & dimensions.
        MAKE_STEP_INDEX(t);                          // step in time dim.
        MAKE_DOMAIN_INDEX(x);                        // spatial dim.
        MAKE_DOMAIN_INDEX(y);                        // spatial dim.
        MAKE_MISC_INDEX(a);
        MAKE_MISC_INDEX(b);
        MAKE_MISC_INDEX(c);
    
        // Time-varying var. Intermix last domain dim with misc dims to make
        // sure compiler creates correct layout.
        MAKE_VAR(A, t, x, a, y, b, c);

        // Misc-only var.
        MAKE_VAR(B, c, b);

        // Time-and-misc var.
        MAKE_VAR(C, t, b, a);

    public:

        TestMisc2dStencil(int radius=2) :
            yc_solution_with_radius_base("test_misc_2d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Define the value at t+1 using asymmetric stencil
            // with various pos & neg indices in misc dims.
            auto r = get_radius();
            yc_number_node_ptr v = A(t, x, 0, y, 1, 2) + 1.0;
            for (int i = 1; i <= r; i++)
                v += A(t, x + i, 3, y,     0, 3);
            for (int i = 1; i <= r + 1; i++)
                v += A(t, x - i, 4, y,     2, 2);
            for (int i = 1; i <= r + 2; i++)
                v += A(t, x,    -2, y + i, 2, 2);
            for (int i = 1; i <= r + 3; i++)
                v += A(t, x,     0, y - i, 0, 3);
            v += C(t, 1, 2);
            A(t+1, x, 1, y, 2, 3) EQUALS v + B(-2, 3) - B(4, -2);
        }
    };

    // Create an object of type 'TestMisc2dStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestMisc2dStencil);

    // "Stream-like" stencils that just read and write
    // with no spatial offsets.
    // The radius controls how many reads are done in the time domain.
    // Running with radius=2 should give performance comparable to
    // (but not identical to) the stream 'triad' benchmark.
    class StreamStencil1 : public yc_solution_with_radius_base {

    protected:

        // Indices & dimensions.
        MAKE_STEP_INDEX(t);                          // step in time dim.
        MAKE_DOMAIN_INDEX(x);                        // spatial dim.
 
        // Vars.
        MAKE_VAR(A, t, x); // time-varying 3D var.

    public:

        StreamStencil1(int radius=2) :
            yc_solution_with_radius_base("test_stream_1d", radius) { }
        virtual ~StreamStencil1() { }

        // Define equation to read 'get_radius()' values and write one.
        virtual void define() {

            yc_number_node_ptr v;

            // Add 'get_radius()' values from past time-steps.
            for (int r = 0; r < get_radius(); r++)
                v += A(t-r, x);

            // define the value at t+1 to be equivalent to v + 1.
            A(t+1, x) EQUALS v + 1;
        }
    };

    // Create an object of type 'StreamStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(StreamStencil1);

    class StreamStencil2 : public yc_solution_with_radius_base {

    protected:

        // Indices & dimensions.
        MAKE_STEP_INDEX(t);                          // step in time dim.
        MAKE_DOMAIN_INDEX(x);                        // spatial dim.
        MAKE_DOMAIN_INDEX(y);                        // spatial dim.
 
        // Vars.
        MAKE_VAR(A, t, x, y); // time-varying 3D var.

    public:

        StreamStencil2(int radius=2) :
            yc_solution_with_radius_base("test_stream_2d", radius) { }
        virtual ~StreamStencil2() { }

        // Define equation to read 'get_radius()' values and write one.
        virtual void define() {

            yc_number_node_ptr v;

            // Add 'get_radius()' values from past time-steps.
            for (int r = 0; r < get_radius(); r++)
                v += A(t-r, x, y);

            // define the value at t+1 to be equivalent to v + 1.
            A(t+1, x, y) EQUALS v + 1;
        }
    };

    // Create an object of type 'StreamStencil2',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(StreamStencil2);


    class StreamStencil3 : public yc_solution_with_radius_base {

    protected:

        // Indices & dimensions.
        MAKE_STEP_INDEX(t);                          // step in time dim.
        MAKE_DOMAIN_INDEX(x);                        // spatial dim.
        MAKE_DOMAIN_INDEX(y);                        // spatial dim.
        MAKE_DOMAIN_INDEX(z);                        // spatial dim.

        // Vars.
        MAKE_VAR(A, t, x, y, z); // time-varying 3D var.

    public:

        StreamStencil3(int radius=2) :
            yc_solution_with_radius_base("test_stream_3d", radius) { }
        virtual ~StreamStencil3() { }

        // Define equation to read 'get_radius()' values and write one.
        virtual void define() {

            yc_number_node_ptr v;

            // Add 'get_radius()' values from past time-steps.
            for (int r = 0; r < get_radius(); r++)
                v += A(t-r, x, y, z);

            // define the value at t+1 to be equivalent to v + 1.
            A(t+1, x, y, z) EQUALS v + 1;
        }
    };

    // Create an object of type 'StreamStencil3',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(StreamStencil3);

    // Reverse-time stencil.
    // In this test, A(t-1) depends on A(t).
    class TestReverseStencil : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x, y);

    public:

        TestReverseStencil(int radius=2) :
            TestBase("test_reverse_2d", radius) { }

        // Define equation to do simple test.
        virtual void define() {

            // Like the previous 2D test, but defines value at 't-1'.
            A(t-1, x, y) EQUALS def_t2d(A, t, x, 0, 2, y, 4, 3);
        }
    };

    // Create an object of type 'TestReverseStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestReverseStencil);

    // Test dependent equations.
    // These will create >= 2 stages that will be applied in sequence
    // for each time-step.
    class TestDepStencil1 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x);
        MAKE_VAR(B, t, x);
        MAKE_VAR(C, t, x);

    public:

        TestDepStencil1(int radius=2) :
            TestBase("test_stages_1d", radius) { }

        // Define equation to apply to all points in 'A' and 'B' vars.
        virtual void define() {

            // Define A(t+1) and B(t+1).
            A(t+1, x) EQUALS -2 * A(t, x);
            B(t+1, x) EQUALS def_t1d(B, t, x, 0, 1);

            // 'C(t+1)' depends on 'A(t+1)', creating a 2nd stage.
            C(t+1, x) EQUALS def_t1d(A, t+1, x, 1, 0) + C(t, x+1);
        }
    };

    // Create an object of type 'TestDepStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestDepStencil1);

    class TestDepStencil2 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x, y); // time-varying var.
        MAKE_VAR(B, t, x, y); // time-varying var.
        MAKE_VAR(C, t, x, y); // time-varying var.

    public:

        TestDepStencil2(int radius=2) :
            TestBase("test_stages_2d", radius) { }

        // Define equation to apply to all points in 'A' and 'B' vars.
        virtual void define() {

            // Define A(t+1) from A(t) & B(t).
            A(t+1, x, y) EQUALS A(t, x, y) - def_t2d(B, t, x, 0, 1, y, 2, 1);

            // Define B(t+1) from B(t) & A(t+1), creating a 2nd stage.
            B(t+1, x, y) EQUALS B(t, x, y) - def_t2d(A, t+1, x, 3, 2, y, 0, 1);

            // Define C(t+1) from B(t+1), creating a 3rd stage.
            C(t+1, x, y) EQUALS B(t+1, x-1, y+2);
        }
    };

    // Create an object of type 'TestDepStencil2',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestDepStencil2);

    class TestDepStencil3 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x, y, z); // time-varying var.
        MAKE_VAR(B, t, x, y, z); // time-varying var.

    public:

        TestDepStencil3(int radius=2) :
            TestBase("test_stages_3d", radius) { }

        // Define equation to apply to all points in 'A' and 'B' vars.
        virtual void define() {

            // Define A(t+1) from A(t) & stencil at B(t).
            A(t+1, x, y, z) EQUALS A(t, x, y, z) -
                def_t3d(B, t, x, 0, 1, y, 2, 1, z, 1, 0);

            // Define B(t+1) from B(t) & stencil at A(t+1).
            B(t+1, x, y, z) EQUALS B(t, x, y, z) -
                def_t3d(A, t+1, x, 1, 0, y, 0, 1, z, 2, 1);
        }
    };

    // Create an object of type 'TestDepStencil3',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestDepStencil3);

    /////// Test the use of scratch-pad vars. ////////

    class TestScratchStencil1 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x); // time-varying var.

        // Temporary storage.
        MAKE_SCRATCH_VAR(B, x);

    public:

        TestScratchStencil1(int radius=2) :
            TestBase("test_scratch_1d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Define values in scratch var 'B' based on 'A'.
            B(x) EQUALS def_t1d(A, t, x, 1, 0);

            // Set each point in 'A' from 2 scratch var values offset from x.
            A(t+1, x) EQUALS def_1d(B, x-4, 2, 3) + def_1d(B, 6+x, 0, 1);
        }
    };

    // Create an object of type 'TestScratchStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestScratchStencil1);

    class TestScratchStencil2 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x, y); // time-varying var.

        // Temporary storage.
        MAKE_SCRATCH_VAR(t1, x, y);
        MAKE_SCRATCH_VAR(t2, x, y);
        MAKE_SCRATCH_VAR(t3, x, y);

    public:

        TestScratchStencil2(int radius=2) :
            TestBase("test_scratch_2d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Set scratch var.
            t1(x, y) EQUALS def_t2d(A, t, x, 0, 1, y, 2, 1);

            // Set 2nd scratch var from 1st scratch var.
            t2(x, y) EQUALS t1(x, y+1);

            // Set 3rd scratch var from 2nd scratch var.
            // This should reuse t1's memory.
            t3(x, y) EQUALS t2(x+1, y);

            // Update A from scratch vars.
            A(t+1, x, y) EQUALS A(t, x, y) +
                def_2d(t2, x, 2, 0, y, 1, 0) +
                def_2d(t3, x, 1, 0, y, 0, 1);
        }
    };

    // Create an object of type 'TestScratchStencil2',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestScratchStencil2);

    class TestScratchStencil3 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x, y, z); // time-varying var.

        // Temporary storage.
        MAKE_SCRATCH_VAR(t1, x, y, z);
        MAKE_SCRATCH_VAR(t2, x, y, z);
        MAKE_SCRATCH_VAR(t3, x, y, z);

    public:

        TestScratchStencil3(int radius=2) :
            TestBase("test_scratch_3d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Set 2 scratch vars, dependent on 'A', but independent
            // of each other.
            t1(x, y, z) EQUALS def_t3d(A, t, x, 0, 1, y, 2, 1, z, 1, 0);
            t2(x, y, z) EQUALS def_t3d(A, t, x, 1, 0, y, 0, 2, z, 0, 1);

            // Set another scratch var from 2 other scratch vars.
            t3(x, y, z) EQUALS t1(x-1, y+1, z) + t2(x, y, z-1);

            // Update 'A' from 2 of the scratch vars.
            A(t+1, x, y, z) EQUALS A(t, x, y, z) +
                def_3d(t1, x, 2, 0, y, 0, 1, z, 1, 0) +
                def_3d(t3, x, 1, 0, y, 0, 1, z, 0, 2);
        }
    };

    // Create an object of type 'TestScratchStencil3',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestScratchStencil3);

    // Scratches in stages.
    class TestScratchStagesStencil1 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x); // time-varying var.
        MAKE_VAR(B, t, x); // time-varying var.

        // Temporary storage.
        MAKE_SCRATCH_VAR(C, x);
        MAKE_SCRATCH_VAR(D, x);
        MAKE_SCRATCH_VAR(E, x);

    public:

        TestScratchStagesStencil1(int radius=2) :
            TestBase("test_scratch_stages_1d", radius) { }

        // Define equation to apply to all points in 'A' and 'B' vars.
        virtual void define() {

            // NB: this stencil also illustrates that equations
            // don't need to be defined in "assignment order."
            
            // 'A'
            A(t+1, x) EQUALS def_1d(C, x, 1, 0);
            C(x) EQUALS def_1d(D, x, 0, 8); // Test a large RHS scratch halo.
            D(x) EQUALS def_t1d(B, t+1, x, 1, 0);

            // 'B'
            B(t+1, x) EQUALS def_1d(E, x, 0, 1);
            E(x) EQUALS def_t1d(A, t, x, 1, 0);
        }
    };
    REGISTER_SOLUTION(TestScratchStagesStencil1);
    

    // Test the use of boundary code in sub-domains.
    class TestBoundaryStencil1 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x); // time-varying var.

    public:

        TestBoundaryStencil1(int radius=2) :
            TestBase("test_boundary_1d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Define interior sub-domain.
            auto sd0 = (x >= first_domain_index(x) + 5) && (x <= last_domain_index(x) - 3);
        
            // Define interior points.
            auto u = def_t1d(A, t, x, 0, 1);
            A(t+1, x) EQUALS u IF_DOMAIN sd0;

            // Define exterior points.
            A(t+1, x) EQUALS -u IF_DOMAIN !sd0;
        }
    };

    // Create an object of type 'TestBoundaryStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestBoundaryStencil1);

    class TestBoundaryStencil2 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x, y); // time-varying var.

    public:

        TestBoundaryStencil2(int radius=2) :
            TestBase("test_boundary_2d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Sub-domain is rectangle interior.
            auto sd0 =
                (x >= first_domain_index(x) + 5) && (x <= last_domain_index(x) - 3) &&
                (y >= first_domain_index(y) + 4) && (y <= last_domain_index(y) - 6);
        
            // Set A w/different stencils depending on condition.
            A(t+1, x, y) EQUALS def_t2d(A, t, x, 0, 2, y, 1, 0) IF_DOMAIN sd0;
            A(t+1, x, y) EQUALS def_t2d(A, t, x, 1, 0, y, 0, 2) IF_DOMAIN !sd0;
        }
    };

    // Create an object of type 'TestBoundaryStencil2',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestBoundaryStencil2);

    class TestBoundaryStencil3 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x, y, z); // time-varying var.

    public:

        TestBoundaryStencil3(int radius=2) :
            TestBase("test_boundary_3d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Sub-domain is rectangle interior.
            auto sd0 =
                (x >= first_domain_index(x) + 5) && (x <= last_domain_index(x) - 3) &&
                (y >= first_domain_index(y) + 4) && (y <= last_domain_index(y) - 6) &&
                (z >= first_domain_index(z) + 6) && (z <= last_domain_index(z) - 4);
        
            // Set A w/different stencils depending on condition.
            A(t+1, x, y, z) EQUALS def_t3d(A, t, x, 0, 2, y, 1, 0, z, 0, 1) IF_DOMAIN sd0;
            A(t+1, x, y, z) EQUALS def_t3d(A, t, x, 1, 0, y, 0, 2, z, 1, 0) IF_DOMAIN !sd0;
        }
    };

    // Create an object of type 'TestBoundaryStencil3',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestBoundaryStencil3);

    // Test step condition.
    class TestStepCondStencil1 : public TestBase {

    protected:

        // Indices.
        MAKE_MISC_INDEX(b);

        // Vars.
        MAKE_VAR(A, t, x); // time-varying var.
        MAKE_VAR(B, b);

    public:

        TestStepCondStencil1(int radius=2) :
            TestBase("test_step_cond_1d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Condition based on step value.
            auto tc0 = (t % 2 == 0);

            // Condition based on misc-var contents.
            auto vc0 = (B(0) > B(1));

            // Update A w/different stencils depending on the conditions.  It is
            // the programmer's responsibility to ensure that the conditions are
            // exclusive when necessary. It is not checked at compile or
            // run-time.

            // Use this equation when t is even.
            A(t+1, x) EQUALS def_t1d(A, t, x, 0, 0) IF_STEP tc0;

            // Use this equation when t is odd and B(0) > B(1).
            A(t+1, x) EQUALS def_t1d(A, t, x, 1, 2) IF_STEP !tc0 && vc0;

            // Use this equation when t is even and B(0) <= B(1).
            // Also use a domain condition.
            // When adding both step and domain conditions, need to use
            // parens as shown.
            (A(t+1, x) EQUALS def_t1d(A, t, x, 2, 0) IF_STEP !tc0 && !vc0)
                IF_DOMAIN (x > first_domain_index(x) + 5);
        }
    };

    // Create an object of type 'TestStepCondStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestStepCondStencil1);

    // Test the use of conditional updates with scratch-pad vars.
    class TestScratchBoundaryStencil1 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x); // time-varying var.

        // Temporary storage.
        MAKE_SCRATCH_VAR(T1, x);

    public:

        TestScratchBoundaryStencil1(int radius=2) :
            TestBase("test_scratch_boundary_1d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Define sub-domains.
            auto sd0 = (x >= first_domain_index(x) + 5) && (x <= last_domain_index(x) - 3);
            auto sd1 = (x >= first_domain_index(x) + 3) && (x <= last_domain_index(x) - 2);
        
            // Define values in scratch var 'B' using current values from 'A'.
            auto b0 = def_t1d(A, t, x, 1, 0);
            T1(x) EQUALS  b0 IF_DOMAIN sd0;
            T1(x) EQUALS -b0 IF_DOMAIN !sd0;

            // Define next values for 'A' from scratch var values.
            auto a1 = def_1d(T1, x-6, 2, 3) - def_1d(T1, x+7, 0, 2);
            A(t+1, x) EQUALS  a1 IF_DOMAIN sd1;
            A(t+1, x) EQUALS -a1 IF_DOMAIN !sd1;
        }
    };

    // Create an object of type 'TestScratchBoundaryStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestScratchBoundaryStencil1);

    // A stencil that uses math functions.
    // This stencil is an exception to the integer-result calculations
    // used in most test stencils.
    class TestFuncStencil1 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x);
        MAKE_VAR(B, t, x);
        MAKE_VAR(C, t, x);

    public:

        TestFuncStencil1(int radius=1) :
            TestBase("test_func_1d", radius) { }

        virtual void define() {

            // Define 'A(t+1)' and 'B(t+1)' based on values at 't'.
            A(t+1, x) EQUALS cos(A(t, x)) - 2 * sin(A(t, x));
            B(t+1, x) EQUALS max(def_t1d(B, t, x, 0, 1), 2.5);

            // 'C(t+1)' depends on 'A(t+1)', creating a 2nd stage.
            C(t+1, x) EQUALS atan(def_t1d(A, t+1, x, 1, 0) / cbrt(C(t, x+1)));
        }
    };

    // Create an object of type 'TestFuncStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestFuncStencil1);

    // A stencil that no vars and no stencil equation.
    // Kernel must be built with domain_dims and step_dim options.
    class TestEmptyStencil0: public TestBase {

    protected:

    public:

        TestEmptyStencil0(int radius=1) :
            TestBase("test_empty", radius) { }

        virtual void define() { }
    };

    // Create an object of type 'TestEmptyStencil0',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestEmptyStencil0);

    // A stencil that has vars but no stencil equation.
    class TestEmptyStencil2 : public TestBase {

    protected:

        // Vars.
        MAKE_VAR(A, t, x, y); // time-varying var.

    public:

        TestEmptyStencil2(int radius=1) :
            TestBase("test_empty_2d", radius) { }

        virtual void define() { }
    };

    // Create an object of type 'TestEmptyStencil2',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    REGISTER_SOLUTION(TestEmptyStencil2);

} // namespace.
