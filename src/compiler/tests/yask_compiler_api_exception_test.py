#!/usr/bin/env python

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2022, Intel Corporation
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
    n0 = g1.new_var_point([t, x, y, z])  # center-point at this timestep.

    # Exception test
    print("Exception Test: Calling 'new_var_point' with too many arguments.")
    try:
        n1 = g1.new_var_point([t, x, y, z, x])
    except RuntimeError as e:
        print ("YASK threw an expected exception.")
        print (format(e))
        print ("Exception Test: Caught exception correctly.")
        num_exception = num_exception + 1

    # Exception test
    print("Exception Test: Call 'new_file_output' with invalid dir.")
    try:
        dot_file = ofac.new_file_output("/does-not-exist/foo.dot")
    except RuntimeError as e:
        print ("YASK threw an expected exception.")
        print (format(e))
        print ("Exception Test: Caught exception correctly.")
        num_exception = num_exception + 1

    # Exception test
    print("Exception Test: Call 'set_target' with invalid target.")
    try:
        soln.set_target("bad_target")
    except RuntimeError as e:
        print ("YASK threw an expected exception.")
        print (format(e))
        print ("Exception Test: Caught exception correctly.")
        num_exception = num_exception + 1

    # Check whether program handles exceptions or not.
    if num_exception != 3:
        print("There is a problem in exception test.")
        exit(1)
    else:
        print("End of YASK compiler API test with exceptions.")
