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

    yc_factory cfac;
    auto soln = cfac.new_solution("api_cxx_test");
    auto g1 = soln->new_grid("test_grid", "t", "x", "y", "z");
    
    yc_node_factory fac;

    auto n1 = fac.new_const_number_node(3.14);
    cout << n1->format_simple() << endl;

    auto n2 = fac.new_negate_node(n1);
    cout << n2->format_simple() << endl;

    auto n3 = g1->new_relative_grid_point(0, 1, 0, -2);
    cout << n3->format_simple() << endl;

    auto n4a = fac.new_add_node(n2, n3);
    auto n4b = fac.new_add_node(n4a, n1);
    cout << n4b->format_simple() << endl;

    auto n5 = g1->new_relative_grid_point(0, 1, -1, 0);
    cout << n5->format_simple() << endl;

    auto n6 = fac.new_divide_node(n4b, n5);
    cout << n6->format_simple() << endl;

    auto n7 = g1->new_relative_grid_point(1, 0, 0, 0);
    cout << n7->format_simple() << endl;

    auto n8 = fac.new_equation_node(n7, n6);
    cout << n8->format_simple() << endl;

    cout << "Solution '" << soln->get_name() << "' contains " <<
        soln->get_num_grids() << " grid(s), and " <<
        soln->get_num_equations() << " equation(s)." << endl;

    soln->set_step_dim("t");
    soln->set_elem_bytes(4);

    string dot_file = "yc-api-test-cxx.dot";
    soln->write(dot_file, "dot", true);
    cout << "DOT-format written to '" << dot_file << "'.\n";

    string yask_file = "yc-api-test-cxx.hpp";
    soln->write(yask_file, "avx", true);
    cout << "YASK-format written to '" << yask_file << "'.\n";
    
    return 0;
}
