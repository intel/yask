/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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

    try {

        // Compiler 'bootstrap' factories.
        yc_factory cfac;
        yask_output_factory ofac;
        yc_node_factory fac;

        // Create a new stencil solution.
        auto soln = cfac.new_solution("api_cxx_test");
        auto stdos = ofac.new_stdout_output();
        soln->set_debug_output(stdos);

        // Number of bytes in each FP value.
        soln->set_element_bytes(4);

        // Define the problem dimensions.
        auto t = fac.new_step_index("t");
        auto x = fac.new_domain_index("x");
        auto y = fac.new_domain_index("y");
        auto z = fac.new_domain_index("z");

        // Create a var.
        auto g1 = soln->new_var("test_var", {t, x, y, z});

        // Create an equation based on some values from 'g1'.
        auto n1 = g1->new_var_point({t, x, y, z});
        cout << n1->format_simple() << endl;

        auto n2 = g1->new_var_point({t, x+1, y, z-2});
        cout << n2->format_simple() << endl;

        auto n3 = n1 + n2;
        cout << n3->format_simple() << endl;

        auto n4 = n2 * -n3 * 0.9;
        cout << n4->format_simple() << endl;

        auto n5 = g1->new_var_point({t, x+2, y-1, z});
        cout << n5->format_simple() << endl;

        auto n6 = n4 / n5;
        cout << n6->format_simple() << endl;

        // Create a scratch var.
        auto sg1 = soln->new_scratch_var("scratch_var", {x, y, z});

        // Define scratch-var value based on expression 'n6' above.
        auto ns_lhs = sg1->new_var_point({x, y, z});
        auto ns_eq = fac.new_equation_node(ns_lhs, n6);
        cout << ns_eq->format_simple() << endl;

        // Define another equation based on some values from 'sg1'.
        auto n7a = sg1->new_var_point({x-1, y, z+2});
        auto n7b = sg1->new_var_point({x+1, y-1, z-2});
        auto n8 = n7a + n7b;
        cout << n8->format_simple() << endl;

        // Define main var value at t+1.
        auto n_lhs = g1->new_var_point({t+1, x, y, z});
        cout << n_lhs->format_simple() << endl;

        // Define a sub-domain in which to apply this value.
        auto sd0 = (x >= fac.new_first_domain_index(x) + 5);

        // Set equations to update the main var using
        // expression 'n8' in sub-domain 'sd0' and -'n8' otherwise.
        auto n_eq0 = fac.new_equation_node(n_lhs, n8, sd0);
        cout << n_eq0->format_simple() << endl;
        auto n_eq1 = fac.new_equation_node(n_lhs, -n8, !sd0);
        cout << n_eq1->format_simple() << endl;

        // Insert code that will register a run-time hook.
        soln->CALL_AFTER_NEW_SOLUTION
            (
             kernel_soln.call_after_prepare_solution
             ([](yk_solution& ksoln) {
                  auto vars = ksoln.get_vars();
              });
             );
                    
        cout << "Solution '" << soln->get_name() << "' contains " <<
            soln->get_num_vars() << " var(s), and " <<
            soln->get_num_equations() << " equation(s)." << endl;

        // Generate DOT output.
        auto dot_file = ofac.new_file_output("yc-api-test-cxx.dot");
        soln->format("dot", dot_file);
        cout << "DOT-format written to '" << dot_file->get_filename() << "'.\n";

        // Generate YASK output.
        auto yask_file = ofac.new_file_output("yc-api-test-cxx.hpp");
        soln->format("avx", yask_file);
        cout << "YASK-format written to '" << yask_file->get_filename() << "'.\n";

        cout << "End of YASK compiler API test.\n";
        return 0;

    } catch(yask_exception& e) {
        cerr << e.get_message() << endl;
        return 1;
    }
}
