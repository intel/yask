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

// Test the YASK stencil compiler API for C++.

#include "yask_compiler_api.hpp"
#include <iostream>

using namespace std;
using namespace yask;

int main() {

    // Counter for exception test
    int num_exception = 0;

    // Compiler 'bootstrap' factories.
    yc_factory cfac;
    yask_output_factory ofac;
    yc_node_factory fac;

    // Create a new stencil solution.
    auto soln = cfac.new_solution("api_cxx_test");
    auto stdos = ofac.new_stdout_output();
    soln->set_debug_output(stdos);

    // Define the problem dimensions.
    auto t = fac.new_step_index("t");
    auto x = fac.new_domain_index("x");
    auto y = fac.new_domain_index("y");
    auto z = fac.new_domain_index("z");

    // Explicitly set the stencil dims in the solution.
    soln->set_step_dim(t);
    soln->set_domain_dims({x, z, y});
    
    // Create a var.
    auto g1 = soln->new_var("test_var", {t, x, y, z});

    // Exception test
    cout << "Exception Test: Call 'new_var_point' with too many arguments.\n";
    try {
        auto n3 = g1->new_var_point({t, x, y, z, x});
    } catch (yask_exception& e) {
        cout << "YASK threw an expected exception.\n";
        cout << e.get_message() << endl;
        cout << "Exception Test: Caught exception correctly.\n";
        num_exception++;
    }

    // Exception test
    cout << "Exception Test: Call 'new_file_output' with invalid dir.\n";
    try {
        auto dot_file = ofac.new_file_output("/does-not-exist/foo.dot");
    } catch (yask_exception& e) {
        cout << "YASK threw an expected exception.\n";
        cout << e.get_message() << endl;
        cout << "Exception Test: Caught exception correctly.\n";
        num_exception++;
    }

    // Check whether program handles exceptions or not.
    if (num_exception != 2) {
        cerr << "Error: unexpected number of exceptions: " << num_exception << endl;
        exit(1);
    }
    else
        cout << "End of YASK compiler API test with exceptions.\n";

    return 0;
}
