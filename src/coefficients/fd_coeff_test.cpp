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

// Finite-differences coefficients code.
// Contributed by Jeremy Tillay.

#include <iostream>
#include <cstring>
#include "fd_coeff.hpp"
#define MIN(x, y) (((x) < (y)) ? (x): (y))
#define MAX(x, y) (((x) > (y)) ? (x): (y))

using namespace std;

int main()
{
    //set the order of the derivative to approximate
    //e.g. we want to approximate d^m/dx^m 
    const int order = 2;

    //set the evaluation point e.g. we want to approximate some derivative f^(m)[eval_point]
    //for most application, this is 0
    float eval_point = 0;

    //Construct a set of points (-h*radius, -h*(radius-1), .. 0, h,..., h*radius) 
    //Could pass any arbitrary array grid_points = {x_0, x_1, ... x_n} 
    const int radius = 2;
    const int num_points = 2*radius+1;
    float h = 1;
    float coeff[num_points];
    float grid_points[num_points];

    //cout << "Approximating derivative from grid points: " ;
    for(int i=0; i<num_points; i++){
        grid_points[i] = (i-(num_points-1)/2)*h;
        cout << grid_points[i]<< " ";
    }

    //cout << endl;
    
    fd_coeff(coeff, eval_point, order, grid_points, num_points);

    string suffix = (order == 1) ? "st" : (order == 2) ? "nd" : (order == 3) ? "rd" : "th";
    cout << "The " << order << suffix << " derivative of f("<< eval_point <<
        ") is approximated by this " << num_points << "-point FD formula:" << endl;
    cout << "f^(" << order << ")(" << eval_point << ") ~= ";

    for(int i=0; i<num_points; i++) {
        if (i)
            cout << " + ";
        cout << coeff[i] << "*f[" << grid_points[i] << "]";
    }
    cout << endl;

    cout << "So coefficients are: ";
    for(int i=0; i<num_points; i++) {
        cout << coeff[i] << ", ";
    }
    cout << endl;
    return 0;
}
