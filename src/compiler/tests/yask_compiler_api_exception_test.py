#!/usr/bin/env python

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2019, Intel Corporation
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## * The above copyright notice and this permission notice shall be included in
##   all copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.
##############################################################################

## Test the YASK stencil compiler API for Python.

import yask_compiler

if __name__ == "__main__":

    # Counter for exception test
    num_exception = 0;

    # Compiler 'bootstrap' factories.
    cfac = yask_compiler.yc_factory()
    ofac = yask_compiler.yask_output_factory()
    nfac = yask_compiler.yc_node_factory()

    # Create a new stencil solution.
    soln = cfac.new_solution("api_py_test")
    do = ofac.new_string_output()
    soln.set_debug_output(do)

    # Define the problem dimensions.
    t = nfac.new_step_index("t");
    x = nfac.new_domain_index("x");
    y = nfac.new_domain_index("y");
    z = nfac.new_domain_index("z");

    # Create a var.
    g1 = soln.new_var("test_var", [t, x, y, z])

    # Create an expression for the new value.
    # This will average some of the neighboring points around the
    # current stencil application point in the current timestep.
    n0 = g1.new_relative_var_point([0, 0, 0, 0])  # center-point at this timestep.

    # Exception test
    print("Exception Test: Call 'new_relative_var_point' with wrong argument.")
    try:
        n1 = nfac.new_add_node(n0, g1.new_relative_var_point([0, -1,  0,  0,  1])) # left.
    except RuntimeError as e:
        print ("YASK threw an expected exception.")
        print (format(e))
        print ("Exception Test: Catch exception correctly.")
        num_exception = num_exception + 1

    # Create an expression using points in g1.
    # This will average some of the neighboring points around the
    # current stencil application point in the current timestep.
    n1 = (g1.new_var_point([t, x,   y,   z  ]) + # center.
          g1.new_var_point([t, x-1, y,   z  ]) + # left.
          g1.new_var_point([t, x+1, y,   z  ]) + # right.
          g1.new_var_point([t, x,   y-1, z  ]) + # above.
          g1.new_var_point([t, x,   y+1, z  ]) + # below.
          g1.new_var_point([t, x,   y,   z-1]) + # in front.
          g1.new_var_point([t, x,   y,   z  ])) # behind.
    n2 = n1 / 7  # ave of the 7 points.

    # Create an equation to define the value at the next timestep.
    n3 = g1.new_relative_var_point([1, 0, 0, 0]) # center-point at next timestep.
    n4 = nfac.new_equation_node(n3, n2) # equate to expr n2.
    print("Equation before formatting: " + n4.format_simple())
    print("Solution '" + soln.get_name() + "' contains " +
          str(soln.get_num_vars()) + " var(s), and " +
          str(soln.get_num_equations()) + " equation(s).")
    for var in soln.get_vars() :
        print("Var " + var.get_name() +
              " has the following dim(s): " +
              repr(var.get_dim_names()));

    # Number of bytes in each FP value.
    soln.set_element_bytes(4)

    # Exception test
    print("Exception Test: Call 'new_file_output' with invalid dir.")
    try:
        dot_file = ofac.new_file_output("/does-not-exist/foo.dot")
    except RuntimeError as e:
        print ("YASK threw an expected exception.")
        print (format(e))
        print ("Exception Test: Catch exception correctly.")
        num_exception = num_exception + 1

    # Generate DOT output.
    dot_file = ofac.new_file_output("yc-api-test-with-exception-py.dot")
    soln.format("dot", dot_file)
    print("DOT-format written to '" + dot_file.get_filename() + "'.")

    # Generate YASK output.
    yask_file = ofac.new_file_output("yc-api-test-with-exception-py.hpp")
    soln.format("avx", yask_file)
    print("YASK-format written to '" + yask_file.get_filename() + "'.")

    # Exception test
    try:
        soln.format("wrong_format", dot_file)
    except RuntimeError as e:
        print ("YASK threw an expected exception.")
        print (format(e))
        print ("Exception Test: Catch exception correctly.")
        num_exception = num_exception + 1

    # Check whether program handles exceptions or not.
    if num_exception != 3:
        print("There is a problem in exception test.")
        exit(1)
    else:
        print("End of YASK compiler API test with exception.")
