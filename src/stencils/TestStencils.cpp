/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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
        yc_index_node_ptr t = new_step_index("t");           // step in time dim.
        yc_index_node_ptr w = new_domain_index("w");         // spatial dim.
        yc_index_node_ptr x = new_domain_index("x");         // spatial dim.
        yc_index_node_ptr y = new_domain_index("y");         // spatial dim.
        yc_index_node_ptr z = new_domain_index("z");         // spatial dim.

        // Define some stencils in different dimensions.
        // The size is based on the 'radius' option.
        // These will be asymmetrical if any of the '*_ext' params are not the same;
    
        // Define simple stencil from var 'V' at 't0' centered around 'x0'.
        // Extend given radius left and/or right w/'*_ext'.
        virtual yc_number_node_ptr def_1d(yc_var_proxy& V, const yc_number_node_ptr& t0, const yc_number_node_ptr& x0,
                                          int left_ext, int right_ext) {
            yc_number_node_ptr v;
            for (int i = -get_radius() - left_ext; i <= get_radius() + right_ext; i++)
                v += V(t0, x0+i);
            return v;
        }

        // Define simple stencil from scratch or read-only var 'V' centered
        // around 'x0'.  Similar to 'def_1d()', but doesn't use step var.
        virtual yc_number_node_ptr def_no_t_1d(yc_var_proxy& V, const yc_number_node_ptr& x0,
                                               int left_ext, int right_ext) {
            yc_number_node_ptr v;
            for (int i = -get_radius() - left_ext; i <= get_radius() + right_ext; i++)
                v += V(x0+i);
            return v;
        }

        // Define simple stencil from var 'V' at 't0' centered around 'x0', 'y0'.
        // Extend given radius left and/or right w/'*_ext'.
        // Use some points from the entire rectangle, not just on the axes.
        virtual yc_number_node_ptr def_2d(yc_var_proxy& V, const yc_number_node_ptr& t0,
                                          const yc_number_node_ptr& x0,
                                          int x_left_ext, int x_right_ext,
                                          const yc_number_node_ptr& y0,
                                          int y_left_ext, int y_right_ext) {
            yc_number_node_ptr v;
            for (int i : { -get_radius() - x_left_ext, 0, get_radius() + x_right_ext })
                for (int j : { -get_radius() - y_left_ext, 0, get_radius() + y_right_ext })
                    v += V(t0, x0+i, y0+j);
            return v;
        }    

        // Define simple stencil from scratch or read-only var 'V' at 't0'
        // centered around 'x0', 'y0'.  Extend given radius left and/or right
        // w/'*_ext'.
        virtual yc_number_node_ptr def_no_t_2d(yc_var_proxy& V,
                                               const yc_number_node_ptr& x0,
                                               int x_left_ext, int x_right_ext,
                                               const yc_number_node_ptr& y0,
                                               int y_left_ext, int y_right_ext) {
            yc_number_node_ptr v;
            for (int i : { -get_radius() - x_left_ext, 0, get_radius() + x_right_ext })
                for (int j : { -get_radius() - y_left_ext, 0, get_radius() + y_right_ext })
                    v += V(x0+i, y0+j);
            return v;
        }    

        // Define simple stencil from var 'V' at 't0' centered around 'x0', 'y0', 'z0'.
        // Extend given radius left and/or right w/'*_ext'.
        // Use some points from the entire rectangular polytope, not just on the axes.
        virtual yc_number_node_ptr def_3d(yc_var_proxy& V, const yc_number_node_ptr& t0,
                                          const yc_number_node_ptr& x0,
                                          int x_left_ext, int x_right_ext,
                                          const yc_number_node_ptr& y0,
                                          int y_left_ext, int y_right_ext,
                                          const yc_number_node_ptr& z0,
                                          int z_left_ext, int z_right_ext) {
            yc_number_node_ptr v;
            for (int i : { -get_radius() - x_left_ext, 0, get_radius() + x_right_ext })
                for (int j : { -get_radius() - y_left_ext, 0, get_radius() + y_right_ext })
                    for (int k : { -get_radius() - z_left_ext, 0, get_radius() + z_right_ext })
                        v += V(t0, x0+i, y0+j, z0+k);
            return v;
        }

        // Define simple stencil from scratch or read-only var 'V' centered
        // around 'x0', 'y0', 'z0'.  Extend given radius left and/or right
        // w/'*_ext'.
        virtual yc_number_node_ptr def_no_t_3d(yc_var_proxy& V,
                                               const yc_number_node_ptr& x0,
                                               int x_left_ext, int x_right_ext,
                                               const yc_number_node_ptr& y0,
                                               int y_left_ext, int y_right_ext,
                                               const yc_number_node_ptr& z0,
                                               int z_left_ext, int z_right_ext) {
            yc_number_node_ptr v;
            for (int i : { -get_radius() - x_left_ext, 0, get_radius() + x_right_ext })
                for (int j : { -get_radius() - y_left_ext, 0, get_radius() + y_right_ext })
                    for (int k : { -get_radius() - z_left_ext, 0, get_radius() + z_right_ext })
                        v += V(x0+i, y0+j, z0+k);
            return v;
        }    

        // Define simple stencil from var 'V' at 't0' centered around 'w0', 'x0', 'y0', 'z0'.
        // Extend given radius left and/or right w/'*_ext'.
        // Use some points from the entire rectangular polytope, not just on the axes.
        virtual yc_number_node_ptr def_4d(yc_var_proxy& V, const yc_number_node_ptr& t0,
                                          const yc_number_node_ptr& w0,
                                          int w_left_ext, int w_right_ext,
                                          const yc_number_node_ptr& x0,
                                          int x_left_ext, int x_right_ext,
                                          const yc_number_node_ptr& y0,
                                          int y_left_ext, int y_right_ext,
                                          const yc_number_node_ptr& z0,
                                          int z_left_ext, int z_right_ext) {
            yc_number_node_ptr v;
            for (int h : { -get_radius() - w_left_ext, 0, get_radius() + w_right_ext })
                for (int i : { -get_radius() - x_left_ext, 0, get_radius() + x_right_ext })
                    for (int j : { -get_radius() - y_left_ext, 0, get_radius() + y_right_ext })
                        for (int k : { -get_radius() - z_left_ext, 0, get_radius() + z_right_ext })
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
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x }); // time-varying var.

    public:

        Test1dStencil(int radius=2) :
            TestBase("test_1d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // define the value at t+1 using asymmetric stencil.
            A(t+1, x) EQUALS def_1d(A, t, x, 0, 2);
        }
    };

    // Create an object of type 'Test1dStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static Test1dStencil Test1dStencil_instance;

    // Simple 2D test.
    class Test2dStencil : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y }); // time-varying var.

    public:

        Test2dStencil(int radius=2) :
            TestBase("test_2d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // define the value at t+1 using asymmetric stencil.
            A(t+1, x, y) EQUALS def_2d(A, t, x, 0, 2, y, 4, 3);
        }
    };

    // Create an object of type 'Test2dStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static Test2dStencil Test2dStencil_instance;

    // Simple 3D test.
    class Test3dStencil : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y, z }); // time-varying var.

    public:

        Test3dStencil(int radius=2) :
            TestBase("test_3d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // define the value at t+1 using asymmetric stencil.
            A(t+1, x, y, z) EQUALS def_3d(A, t, x, 0, 2, y, 4, 3, z, 2, 1);
        }
    };

    // Create an object of type 'Test3dStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static Test3dStencil Test3dStencil_instance;

    // Simple 4D test.
    class Test4dStencil : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, w, x, y, z }); // time-varying var.

    public:

        Test4dStencil(int radius=1) :
            TestBase("test_4d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // define the value at t+1 using asymmetric stencil.
            A(t+1, w, x, y, z) EQUALS def_4d(A, t, w, 1, 2, x, 0, 2, y, 2, 1, z, 1, 0);
        }
    };

    // Create an object of type 'Test4dStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static Test4dStencil Test4dStencil_instance;

    // Test vars that don't cover all domain dims.
    class TestPartialStencil3 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y, z }); // time-varying var.
        yc_var_proxy B = yc_var_proxy("B", get_soln(), { x });
        yc_var_proxy C = yc_var_proxy("C", get_soln(), { y });
        yc_var_proxy D = yc_var_proxy("D", get_soln(), { z });
        yc_var_proxy E = yc_var_proxy("E", get_soln(), { x, y });
        yc_var_proxy F = yc_var_proxy("F", get_soln(), { y, z });
        yc_var_proxy G = yc_var_proxy("G", get_soln(), { z, y });
        yc_var_proxy H = yc_var_proxy("H", get_soln(), { y, z, x });      // different order.
        yc_var_proxy I = yc_var_proxy("I", get_soln(), { }); // scalar.

    public:

        TestPartialStencil3(int radius=2) :
            TestBase("test_partial_3d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // define the value at t+1 using asymmetric stencil.
            A(t+1, x, y, z) EQUALS
                def_3d(A, t, x, 0, 2, y, 4, 3, z, 2, 1) +
                def_no_t_1d(B, x, 0, 1) +
                def_no_t_1d(C, y, 1, 0) +
                def_no_t_1d(D, z, 0, 0) +
                def_no_t_2d(E, x, 0, 0, y, 1, 0) +
                def_no_t_2d(F, y, 0, 1, z, 0, 0) +
                def_no_t_2d(G, z, 1, 0, y, 0, 1) +
                def_no_t_3d(H, y, 1, 0, z, 0, 1, x, 1, 0) + I;
        }
    };

    // Create an object of type 'TestPartialStencil3',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestPartialStencil3 TestPartialStencil3_instance;

    // Test misc indices.
    class TestMisc2dStencil : public yc_solution_with_radius_base {

    protected:

        // Indices & dimensions.
        yc_index_node_ptr t = new_step_index("t");           // step in time dim.
        yc_index_node_ptr x = new_domain_index("x");         // spatial dim.
        yc_index_node_ptr y = new_domain_index("y");         // spatial dim.
        yc_index_node_ptr a = new_misc_index("a");
        yc_index_node_ptr b = new_misc_index("b");
        yc_index_node_ptr c = new_misc_index("c");
    
        // Time-varying var. Intermix last domain dim with misc dims to make
        // sure compiler creates correct layout.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, a, y, b, c }); 

    public:

        TestMisc2dStencil(int radius=2) :
            yc_solution_with_radius_base("test_misc_2d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Define the value at t+1 using asymmetric stencil
            // with various pos & neg indices in misc dims.
            yc_number_node_ptr v = A(t, x, 0, y, -1, 2) + 1.0;
            for (int r = 1; r <= get_radius(); r++)
                v += A(t, x + r, 3, y, 0, 1);
            for (int r = 1; r <= get_radius() + 1; r++)
                v += A(t, x - r, 4, y, 2, 1);
            for (int r = 1; r <= get_radius() + 2; r++)
                v += A(t, x, -2, y + r, 2, 0);
            for (int r = 1; r <= get_radius() + 3; r++)
                v += A(t, x, 0, y - r, 0, -1);
            A(t+1, x, 1, y, 2, 3) EQUALS v;
        }
    };

    // Create an object of type 'TestMisc2dStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestMisc2dStencil TestMisc2dStencil_instance;


    // A "stream-like" stencil that just reads and writes
    // with no spatial offsets.
    // The radius controls how many reads are done in the time domain.
    // Running with radius=2 should give performance comparable to
    // (but not identical to) the stream 'triad' benchmark.

    class StreamStencil : public yc_solution_with_radius_base {

    protected:

        // Indices & dimensions.
        yc_index_node_ptr t = new_step_index("t");           // step in time dim.
        yc_index_node_ptr x = new_domain_index("x");         // spatial dim.
        yc_index_node_ptr y = new_domain_index("y");         // spatial dim.
        yc_index_node_ptr z = new_domain_index("z");         // spatial dim.

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y, z }); // time-varying 3D var.

    public:

        StreamStencil(int radius=2) :
            yc_solution_with_radius_base("test_stream_3d", radius) { }
        virtual ~StreamStencil() { }

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

    // Create an object of type 'StreamStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static StreamStencil StreamStencil_instance;

    // Reverse-time stencil.
    // In this test, A(t-1) depends on A(t).

    class TestReverseStencil : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y });

    public:

        TestReverseStencil(int radius=2) :
            TestBase("test_reverse_2d", radius) { }

        // Define equation to do simple test.
        virtual void define() {

            // Like the previous 2D test, but defines value at 't-1'.
            A(t-1, x, y) EQUALS def_2d(A, t, x, 0, 2, y, 4, 3);
        }
    };

    // Create an object of type 'TestReverseStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestReverseStencil TestReverseStencil_instance;

    // Test dependent equations.
    // These will create >= 2 stages that will be applied in sequence
    // for each time-step.
    class TestDepStencil1 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x });
        yc_var_proxy B = yc_var_proxy("B", get_soln(), { t, x });
        yc_var_proxy C = yc_var_proxy("C", get_soln(), { t, x });

    public:

        TestDepStencil1(int radius=2) :
            TestBase("test_stages_1d", radius) { }

        // Define equation to apply to all points in 'A' and 'B' vars.
        virtual void define() {

            // Define A(t+1) and B(t+1).
            A(t+1, x) EQUALS -2 * A(t, x);
            B(t+1, x) EQUALS def_1d(B, t, x, 0, 1);

            // 'C' depends on 'A', creating a 2nd stage.
            C(t+1, x) EQUALS def_1d(A, t+1, x, 1, 0) + C(t, x+1);
        }
    };

    // Create an object of type 'TestDepStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestDepStencil1 TestDepStencil1_instance;

    class TestDepStencil2 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y }); // time-varying var.
        yc_var_proxy B = yc_var_proxy("B", get_soln(), { t, x, y }); // time-varying var.

    public:

        TestDepStencil2(int radius=2) :
            TestBase("test_stages_2d", radius) { }

        // Define equation to apply to all points in 'A' and 'B' vars.
        virtual void define() {

            // Define A(t+1) from A(t) & stencil at B(t).
            A(t+1, x, y) EQUALS A(t, x, y) - def_2d(B, t, x, 0, 1, y, 2, 1);

            // Define B(t+1) from B(t) & stencil at A(t+1).
            B(t+1, x, y) EQUALS B(t, x, y) - def_2d(A, t+1, x, 3, 2, y, 0, 1);
        }
    };

    // Create an object of type 'TestDepStencil2',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestDepStencil2 TestDepStencil2_instance;

    class TestDepStencil3 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y, z }); // time-varying var.
        yc_var_proxy B = yc_var_proxy("B", get_soln(), { t, x, y, z }); // time-varying var.

    public:

        TestDepStencil3(int radius=2) :
            TestBase("test_stages_3d", radius) { }

        // Define equation to apply to all points in 'A' and 'B' vars.
        virtual void define() {

            // Define A(t+1) from A(t) & stencil at B(t).
            A(t+1, x, y, z) EQUALS A(t, x, y, z) -
                def_3d(B, t, x, 0, 1, y, 2, 1, z, 1, 0);

            // Define B(t+1) from B(t) & stencil at A(t+1).
            B(t+1, x, y, z) EQUALS B(t, x, y, z) -
                def_3d(A, t+1, x, 1, 0, y, 0, 1, z, 2, 1);
        }
    };

    // Create an object of type 'TestDepStencil3',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestDepStencil3 TestDepStencil3_instance;

    // Test the use of scratch-pad vars.

    class TestScratchStencil1 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x }); // time-varying var.

        // Temporary storage.
        yc_var_proxy B = yc_var_proxy("B", get_soln(), { x }, true);

    public:

        TestScratchStencil1(int radius=2) :
            TestBase("test_scratch_1d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Define values in scratch var 'B' based on 'A'.
            B(x) EQUALS def_1d(A, t, x, 1, 0);

            // Set 'A' from scratch var values.
            A(t+1, x) EQUALS def_no_t_1d(B, x-4, 2, 3) + def_no_t_1d(B, x+6, 0, 1);
        }
    };

    // Create an object of type 'TestScratchStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestScratchStencil1 TestScratchStencil1_instance;

    class TestScratchStencil2 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y }); // time-varying var.

        // Temporary storage.
        yc_var_proxy t1 = yc_var_proxy("t1", get_soln(), { x, y }, true);
        yc_var_proxy t2 = yc_var_proxy("t2", get_soln(), { x, y }, true);

    public:

        TestScratchStencil2(int radius=2) :
            TestBase("test_scratch_2d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Set scratch var.
            t1(x, y) EQUALS def_2d(A, t, x, 0, 1, y, 2, 1);

            // Set one scratch var from other scratch var.
            t2(x, y) EQUALS t1(x, y+1);

            // Update A from scratch vars.
            A(t+1, x, y) EQUALS A(t, x, y) +
                def_no_t_2d(t1, x, 2, 0, y, 1, 0) +
                def_no_t_2d(t2, x, 1, 0, y, 0, 1);
        }
    };

    // Create an object of type 'TestScratchStencil2',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestScratchStencil2 TestScratchStencil2_instance;

    class TestScratchStencil3 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y, z }); // time-varying var.

        // Temporary storage.
        yc_var_proxy t1 = yc_var_proxy("t1", get_soln(), { x, y, z }, true);
        yc_var_proxy t2 = yc_var_proxy("t2", get_soln(), { x, y, z }, true);
        yc_var_proxy t3 = yc_var_proxy("t3", get_soln(), { x, y, z }, true);

    public:

        TestScratchStencil3(int radius=2) :
            TestBase("test_scratch_3d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Set scratch vars.
            t1(x, y, z) EQUALS def_3d(A, t, x, 0, 1, y, 2, 1, z, 1, 0);
            t2(x, y, z) EQUALS def_3d(A, t, x, 1, 0, y, 0, 2, z, 0, 1);

            // Set a scratch var from other scratch vars.
            t3(x, y, z) EQUALS t1(x-1, y+1, z) + t2(x, y, z-1);

            // Update A from scratch vars.
            A(t+1, x, y, z) EQUALS A(t, x, y, z) +
                def_no_t_3d(t1, x, 2, 0, y, 0, 1, z, 1, 0) +
                def_no_t_3d(t3, x, 1, 0, y, 0, 1, z, 0, 2);
        }
    };

    // Create an object of type 'TestScratchStencil3',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestScratchStencil3 TestScratchStencil3_instance;

    // Test the use of boundary code in sub-domains.
    class TestBoundaryStencil1 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x }); // time-varying var.

    public:

        TestBoundaryStencil1(int radius=2) :
            TestBase("test_boundary_1d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Define interior sub-domain.
            auto sd0 = (x >= first_domain_index(x) + 5) && (x <= last_domain_index(x) - 3);
        
            // Define interior points.
            auto u = def_1d(A, t, x, 0, 1);
            A(t+1, x) EQUALS u IF_DOMAIN sd0;

            // Define exterior points.
            A(t+1, x) EQUALS -u IF_DOMAIN !sd0;
        }
    };

    // Create an object of type 'TestBoundaryStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestBoundaryStencil1 TestBoundaryStencil1_instance;

    class TestBoundaryStencil2 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y }); // time-varying var.

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
            A(t+1, x, y) EQUALS def_2d(A, t, x, 0, 2, y, 1, 0) IF_DOMAIN sd0;
            A(t+1, x, y) EQUALS def_2d(A, t, x, 1, 0, y, 0, 2) IF_DOMAIN !sd0;
        }
    };

    // Create an object of type 'TestBoundaryStencil2',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestBoundaryStencil2 TestBoundaryStencil2_instance;

    class TestBoundaryStencil3 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y, z }); // time-varying var.

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
            A(t+1, x, y, z) EQUALS def_3d(A, t, x, 0, 2, y, 1, 0, z, 0, 1) IF_DOMAIN sd0;
            A(t+1, x, y, z) EQUALS def_3d(A, t, x, 1, 0, y, 0, 2, z, 1, 0) IF_DOMAIN !sd0;
        }
    };

    // Create an object of type 'TestBoundaryStencil3',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestBoundaryStencil3 TestBoundaryStencil3_instance;

    // Test step condition.
    class TestStepCondStencil1 : public TestBase {

    protected:

        // Indices.
        yc_index_node_ptr b = new_misc_index("b");

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x }); // time-varying var.
        yc_var_proxy B = yc_var_proxy("B", get_soln(), { b });

    public:

        TestStepCondStencil1(int radius=2) :
            TestBase("test_step_cond_1d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Time condition.
            auto tc0 = (t % 2 == 0);

            // Var condition.
            auto vc0 = (B(0) > B(1));
        
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

    // Create an object of type 'TestStepCondStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestStepCondStencil1 TestStepCondStencil1_instance;

    // Test the use of conditional updates with scratch-pad vars.
    class TestScratchBoundaryStencil1 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x }); // time-varying var.

        // Temporary storage.
        yc_var_proxy B = yc_var_proxy("B", get_soln(), { x }, true);

    public:

        TestScratchBoundaryStencil1(int radius=2) :
            TestBase("test_scratch_boundary_1d", radius) { }

        // Define equation to apply to all points in 'A' var.
        virtual void define() {

            // Define values in scratch var 'B' using current values from 'A'.
            B(x) EQUALS def_1d(A, t, x, 1, 0);

            // Define sub-domain.
            auto sd0 = (x >= first_domain_index(x) + 5) && (x <= last_domain_index(x) - 3);
        
            // Define next values for 'A' from scratch var values.
            auto v = def_no_t_1d(B, x-6, 2, 3) - def_no_t_1d(B, x+7, 0, 2);
            A(t+1, x) EQUALS v IF_DOMAIN sd0;
            A(t+1, x) EQUALS -v IF_DOMAIN !sd0;
        }
    };

    // Create an object of type 'TestScratchBoundaryStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestScratchBoundaryStencil1 TestScratchBoundaryStencil1_instance;

    // A stencil that uses math functions.
    // This stencil is an exception to the integer-result calculations
    // used in most test stencils.
    class TestFuncStencil1 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x });
        yc_var_proxy B = yc_var_proxy("B", get_soln(), { t, x });
        yc_var_proxy C = yc_var_proxy("C", get_soln(), { t, x });

    public:

        TestFuncStencil1(int radius=1) :
            TestBase("test_func_1d", radius) { }

        virtual void define() {

            // Define 'A(t+1)' and 'B(t+1)' based on values at 't'.
            A(t+1, x) EQUALS cos(A(t, x)) - 2 * sin(A(t, x));
            B(t+1, x) EQUALS pow(def_1d(B, t, x, 0, 1), 1.0/2.5);

            // 'C(t+1)' depends on 'A(t+1)', creating a 2nd stage.
            C(t+1, x) EQUALS atan(def_1d(A, t+1, x, 1, 0) + cbrt(C(t, x+1)));
        }
    };

    // Create an object of type 'TestFuncStencil1',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestFuncStencil1 TestFuncStencil1_instance;

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
    static TestEmptyStencil0 TestEmptyStencil0_instance;

    // A stencil that has vars but no stencil equation.
    class TestEmptyStencil2 : public TestBase {

    protected:

        // Vars.
        yc_var_proxy A = yc_var_proxy("A", get_soln(), { t, x, y }); // time-varying var.

    public:

        TestEmptyStencil2(int radius=1) :
            TestBase("test_empty_2d", radius) { }

        virtual void define() { }
    };

    // Create an object of type 'TestEmptyStencil2',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static TestEmptyStencil2 TestEmptyStencil2_instance;

} // namespace.
