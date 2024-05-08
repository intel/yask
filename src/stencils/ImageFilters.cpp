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

// Example stencil equations for 2D image filtering.

// YASK stencil solution(s) in this file will be integrated into the YASK compiler utility.
#include "yask_compiler_api.hpp"
using namespace std;
using namespace yask;

// Create an anonymous namespace to ensure that types are local.
namespace {

    using yn=yc_number_node_ptr;
    using yb=yc_bool_node_ptr;

    // A simple smoothing filter where each pixel is updated with the
    // average value of its neighbors.
    class BoxFilter : public yc_solution_with_radius_base {

    protected:
        // Indices & dimensions.
        MAKE_STEP_INDEX(n);           // step dim.
        MAKE_DOMAIN_INDEX(x);         // spatial dim.
        MAKE_DOMAIN_INDEX(y);         // spatial dim.

        // Vars.
        MAKE_VAR(A, n, x, y);
        
    public:
        BoxFilter(string name="box_filter", int radius=2) :
            yc_solution_with_radius_base(name, radius) { }

        // Define equation at n+1 based on values at n.
        virtual void define() {
            auto r = get_radius();

            // RHS of expression being created.
            yn v;

            // Add points in square from step n.
            for (int i = -r; i <= r; i++)
                for (int j = -r; j <= r; j++)
                    v += A(n, x+i, y+j);

            // Average by dividing by number of points
            // in square.
            v /= r * r;
        
            // Define the value at n+1 to be expression v.
            A(n+1, x, y) EQUALS v;
        }
    };
    REGISTER_SOLUTION(BoxFilter);

    // Isotropic Gaussian.
    // TODO: update to parameterize sigma (or perhaps parameterize
    // radius and determine good sigma value from requested radius).
    class GaussianFilter : public yc_solution_base {

    protected:
        // Indices & dimensions.
        MAKE_STEP_INDEX(n);           // step dim.
        MAKE_DOMAIN_INDEX(x);         // spatial dim.
        MAKE_DOMAIN_INDEX(y);         // spatial dim.

        // Vars.
        MAKE_VAR(A, n, x, y);
        
    public:
        GaussianFilter(string name="gaussian_filter") :
            yc_solution_base(name) { }

        // Define equation at n+1 based on values at n.
        virtual void define() {

            // G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2)) / (2 * pi * sigma^2).
            // For the given coefficients, sigma = 1.0.
            // See https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm

            // RHS of expression being created.
            yn v;

            // Add points in square from step n.
            double coeff[5][5] = { { 1.,  4.,  7.,  4., 1. },
                                   { 4., 16., 26., 16., 4. },
                                   { 7., 26., 41., 26., 7. },
                                   { 4., 16., 26., 16., 4. },
                                   { 1.,  4.,  7.,  4., 1. } };
            double sum = 273.;
            for (int i = -2; i <= 2; i++)
                for (int j = -2; j <= 2; j++) {
                    double c = coeff[i+2][j+2] / sum;
                    v += A(n, x+i, y+j) * c;
                }
        
            // Define the value at n+1 to be expression v.
            A(n+1, x, y) EQUALS v;
        }
    };
    REGISTER_SOLUTION(GaussianFilter);

}; // namespace.
