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

// Test the YASK stencil compiler API for C++.

#include "yask_compiler_api.hpp"
#include <iostream>

using namespace std;
using namespace yask;

int main() {

    yask_compiler_factory cfac;
    auto soln = cfac.new_stencil_solution("test_solution");
    auto g1 = soln->add_grid("test_grid", "t", "x", "y", "z");
    
    node_factory fac;

    auto n1 = fac.new_const_number_node(3.14);
    cout << n1->format_simple() << endl;

    auto n2 = fac.new_negate_node(n1);
    cout << n2->format_simple() << endl;

    auto n3 = g1->new_relative_grid_point(0, 1, 0, -2);
    cout << n3->format_simple() << endl;

    auto n4 = fac.new_add_node(n2, n3);
    cout << n4->format_simple() << endl;

    auto n5 = g1->new_relative_grid_point(0, 1, -1, 0);
    cout << n5->format_simple() << endl;

    auto n6 = fac.new_divide_node(n4, n5);
    cout << n6->format_simple() << endl;
    auto n6l = n6->get_lhs();
    cout << " LHS: " << n6l->format_simple() << endl;
    auto n6r = n6->get_rhs();
    cout << " RHS: " << n6r->format_simple() << endl;
    
    return 0;
}
