/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

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

// Finite-differences coefficients code.
// Contributed by Jeremy Tillay.

#include <cassert>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include "fd_coeff.hpp"

using namespace std;
namespace yask {

    // C-style interface with arbitrary point locations.
    /* input:coeff=empty coefficient array (or one that can be overwritten)
       eval_point=point at which the derivative is approximated order=order
       of the derivative to approximate (e.g. f'' corresponds to order = 2)
       points=array of points from which to construct the approximation of
       the derivative.  usually an equi-spaced array with points [-radius*h,
       -(radius-1)*h,...0, ... radius*h] num_points=number of elements in
       points[], e.g. the number of points used to approximate the
       derivative.  Note: if num_points < order+1, then the coefficients will
       all be 0
       
       output:void, fills the coefficient array such that
       f^(m)[eval_point] ~~ sum of coeff[i]*f[point[i]] from i = 0 to num_points-1
    */
    void fd_coeff(double *coeff, const double eval_point, const int order, const double *points, const int num_points)
    {
        double c1, c2, c3;
        double x_0 = eval_point;

        int m_idx = (order+1)*num_points;
        int n_idx = num_points;
        int dsz = m_idx * n_idx;
        double d[dsz];
    
        for (int i = 1; i < dsz; i++)
            d[i] = 0.0;
        d[0]=1.0;
        c1 = 1.0;

        for (int n=1; n<=num_points-1; ++n){
            c2=1.0;
            for(int v=0; v<=n-1; ++v){
                c3 = points[n] - points[v];
                c2 = c2*c3;
                for (int m=0; m<=min(n, order); ++m) {
                    double dval = (points[n] - x_0) * d[m * m_idx + (n-1) * n_idx + v];
                    if (m > 0)
                        dval -= m * d[(m-1) * m_idx + (n-1) * n_idx + v];
                    dval *= 1.0/c3;
                    if (dval == -0.0)
                        dval = 0.0;
                    assert(isfinite(dval));
                    d[m * m_idx + n * n_idx + v] = dval;
                }
            }
            for (int m=0; m<= min(n, order); ++m) {
                double dval = 0.0;
                if (m > 0)
                    dval += m * d[(m-1) * m_idx + (n-1) * n_idx + (n-1)];
                dval -= (points[n-1] - x_0) * d[m * m_idx + (n-1) * n_idx + n-1];
                dval *= c1/c2;
                if (dval == -0.0)
                    dval = 0.0;
                assert(isfinite(dval));
                d[m*m_idx + n*n_idx + n] = dval;
            }
            c1=c2;
        }

        for(int i=0; i<num_points; ++i){
            coeff[i] = d[order * m_idx + (num_points-1) * n_idx + i];
        }
    }
}
