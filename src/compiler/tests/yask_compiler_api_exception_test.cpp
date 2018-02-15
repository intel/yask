/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

    // Create a grid var.
    auto g1 = soln->new_grid("test_grid", {t, x, y, z});

    // Create an equation for the grid.

    auto n1 = fac.new_const_number_node(3.14);
    cout << n1->format_simple() << endl;

    auto n2 = fac.new_negate_node(n1);
    cout << n2->format_simple() << endl;

    // Exception test
    cout << "Exception Test: Call 'new_relative_grid_point' with wrong argument.\n";
    try {
        auto n3 = g1->new_relative_grid_point({0, +1, 0, -2, 1});
    } catch (yask_exception e) {
        cout << "YASK throws an exception.\n";
        cout << e.get_message() << endl;
        cout << "Exception Test: Catch exception correctly.\n";
        num_exception++;
    }

    auto n3 = g1->new_relative_grid_point({0, +1, 0, -2});
    cout << n3->format_simple() << endl;

    auto n4a = fac.new_add_node(n2, n3);
    auto n4b = fac.new_add_node(n4a, n1);
    cout << n4b->format_simple() << endl;

    auto n5 = g1->new_relative_grid_point({0, +1, -1, 0});
    cout << n5->format_simple() << endl;

    auto n6 = fac.new_multiply_node(n4b, n5);
    cout << n6->format_simple() << endl;

    auto n_lhs = g1->new_relative_grid_point({+1, 0, 0, 0});
    cout << n_lhs->format_simple() << endl;

    auto n_eq = fac.new_equation_node(n_lhs, n6);
    cout << n_eq->format_simple() << endl;

    cout << "Solution '" << soln->get_name() << "' contains " <<
        soln->get_num_grids() << " grid(s), and " <<
        soln->get_num_equations() << " equation(s)." << endl;

    // Number of bytes in each FP value.
    soln->set_element_bytes(4);

    // Exception test
    cout << "Exception Test: Call 'new_file_output' with read-only file name.\n";
    try {
        auto dot_file = ofac.new_file_output("yc-api-test-with-exception-cxx-readonly");
    } catch (yask_exception e) {
        cout << "YASK throws an exception.\n";
        cout << e.get_message() << endl;
        cout << "Exception Test: Catch exception correctly.\n";
        num_exception++;
    }

    // Generate DOT output.
    auto dot_file = ofac.new_file_output("yc-api-test-with-exception-cxx.dot");
    soln->format("dot", dot_file);
    cout << "DOT-format written to '" << dot_file->get_filename() << "'.\n";

    // Generate YASK output.
    auto yask_file = ofac.new_file_output("yc-api-test-with-exception-cxx.hpp");
    soln->format("avx", yask_file);
    cout << "YASK-format written to '" << yask_file->get_filename() << "'.\n";
    
    // Exception test
    cout << "Exception Test: Call 'format' with wrong format.\n";
    try {
        soln->format("wrong_format", dot_file);
    } catch (yask_exception e) {
        cout << "YASK throws an exception.\n";
        cout << e.get_message() << endl;
        cout << "Exception Test: Catch exception correctly.\n";
        num_exception++;
    }

    // TODO: better to have exception test for the methods below
    // Eqs::findDeps (<-EqGroups::makeEqGroups<-StencilSolution::analyze_solution<-StencilSolution::format())
    // EqGroups::sort (<-EqGroups::makeEqGroups<-StencilSolution::analyze_solution<-StencilSolution::format())
    // GridPoint::GridPoint
    // castExpr
    // NumExpr::getNumVal, NumExpr::getIntVal, NumExpr::getBoolVal
    // Dimensions::setDims (<-StencilSolution::analyze_solution<-StencilSolution::format)
    // ArgParser::parseKeyValuePairs
    // YASKCppPrinter::printData (<-YASKCppPrinter::print<-StencilSolution::format)

    // Check whether program handles exceptions or not.
    if (num_exception != 3) {
        cerr << "Error: unexpected number of exceptions: " << num_exception << endl;
        exit(1);
    }
    else
        cout << "End of YASK compiler API test with exception.\n";

    return 0;
}
