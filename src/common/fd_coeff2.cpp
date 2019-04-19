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

///////// FD coefficients implementation.

#include "yask_common_api.hpp"
#include "fd_coeff.hpp"
#include "common_utils.hpp"

using namespace std;

namespace yask {

    // C++-style interface.
    vector<double> get_arbitrary_fd_coefficients(int derivative_order, double eval_point, const vector<double> sample_points) {
        if (derivative_order < 1)
            THROW_YASK_EXCEPTION("Error: get_fd_coefficients() called with derivative-order less than 1");
        int n = int(sample_points.size());
        if (n < 2)
            THROW_YASK_EXCEPTION("Error: get_fd_coefficients() called with fewer than 2 sample points");
        vector<double> coeffs(n);
        fd_coeff(&coeffs[0], double(eval_point), derivative_order, &sample_points[0], n);
        return coeffs;
    }

    // Common FD forms for uniform grid spacing.
    vector<double> get_center_fd_coefficients(int derivative_order, int radius) {
        if (radius < 1)
            THROW_YASK_EXCEPTION("get_center_fd_coefficients() called with less than radius 1");
        vector<double> pts;
        for (int i = -radius; i <= radius; i++)
            pts.push_back(i);
        assert(sizeof(pts) == size_t(radius * 2 + 1));
        return get_arbitrary_fd_coefficients(derivative_order, 0, pts);
    }
    vector<double> get_forward_fd_coefficients(int derivative_order, int accuracy_order) {
        if (accuracy_order < 1)
            THROW_YASK_EXCEPTION("get_forward_fd_coefficients() called with less than order-of-accuracy 1");
        vector<double> pts;
        for (int i = 0; i <= accuracy_order; i++)
            pts.push_back(i);
        return get_arbitrary_fd_coefficients(derivative_order, 0, pts);
    }
    vector<double> get_backward_fd_coefficients(int derivative_order, int accuracy_order) {
        if (accuracy_order < 1)
            THROW_YASK_EXCEPTION("get_backward_fd_coefficients() called with less than order-of-accuracy 1");
        vector<double> pts;
        for (int i = -accuracy_order; i <= 0; i++)
            pts.push_back(i);
        return get_arbitrary_fd_coefficients(derivative_order, 0, pts);
    }

} // yask namespace.
