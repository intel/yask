#!/usr/bin/env python

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2023, Intel Corporation
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

def make_soln() :
    global soln, t, x, y, z, k, A, B, C, T1
    
    # Create a new stencil solution.
    soln = cfac.new_solution("api_py_test")

    # Define the problem dimensions.
    t = nfac.new_step_index("t");
    x = nfac.new_domain_index("x");
    y = nfac.new_domain_index("y");
    z = nfac.new_domain_index("z");
    k = nfac.new_misc_index("k");

    # Create vars.
    A = soln.new_var("A", [t, x, y, z])
    B = soln.new_var("B", [t, x, y, z])
    C = soln.new_var("C", [t, k, x, y, z])
    T1 = soln.new_scratch_var("T1", [k, x, y, z]);

if __name__ == "__main__":

    # Counter for exception test
    num_exception = 0;

    # Compiler 'bootstrap' factories.
    cfac = yask_compiler.yc_factory()
    ofac = yask_compiler.yask_output_factory()
    nfac = yask_compiler.yc_node_factory()

    yask_file = ofac.new_file_output("yc-api-test-py.hpp");
    
    num_expected = 0
    print("**** calling 'new_var_point' with too many arguments.")
    try:
        make_soln()
        n1 = A.new_var_point([t, x, y, z, x]) # caught here.
        print(n1.format_simple())
        print("***** did NOT throw expected exception *****")
    except RuntimeError as e:
        print (format(e))
        num_exception += 1
    num_expected += 1
    
    print("**** 2 equations defining same point");
    try:
        make_soln()
        n1 = A.new_var_point([t+1, x, y, z])
        n2 = nfac.new_const_number_node(6)
        eq1 = nfac.new_equation_node(n1, n2);
        print(eq1.format_simple())
        n3 = A.new_var_point([t+1, x, y, z])
        n4 = nfac.new_const_number_node(9)
        eq2 = nfac.new_equation_node(n3, n4);
        print(eq1.format_simple())

        soln.set_target("avx");
        soln.output_solution(yask_file) # caught here.
        print("***** did NOT throw expected exception *****")
    except RuntimeError as e:
        print (format(e))
        num_exception += 1
    num_expected += 1

    print("**** equation with illegal LHS (no 't' offset)")
    try:
        make_soln()
        n1 = A.new_var_point([t, x, y, z])
        n3 = A.new_var_point([t, x, y+1, z])
        ns_eq = nfac.new_equation_node(n1, n3);
        print(ns_eq.format_simple())
        soln.set_target("avx");
        soln.output_solution(yask_file) # caught here.
        print("***** did NOT throw expected exception *****")
    except RuntimeError as e:
        print (format(e))
        num_exception += 1
    num_expected += 1

    print("**** equation with illegal LHS (mixed 't' offsets)")
    try:
        make_soln()
        n1 = A.new_var_point([t+1, x, y, z])
        n2 = B.new_var_point([t, x+1, y, z])
        n3 = B.new_var_point([t-1, x, y, z])
        n4 = A.new_var_point([t, x+1, y, z])
        eq1 = nfac.new_equation_node(n1, n2);
        print(eq1.format_simple())
        eq2 = nfac.new_equation_node(n3, n4);
        print(eq2.format_simple())
        soln.set_target("avx");
        soln.output_solution(yask_file) # caught here.
        print("***** did NOT throw expected exception *****")
    except RuntimeError as e:
        print (format(e))
        num_exception += 1
    num_expected += 1

    print("**** equation with illegal dependency (exact same point)")
    try:
        make_soln()
        n0 = A.new_var_point([t+1, x, y, z])
        n1 = A.new_var_point([t+1, x, y, z]) + 3.4
        ns_eq = nfac.new_equation_node(n0, n1);
        print(ns_eq.format_simple())
        soln.set_target("avx");
        soln.output_solution(yask_file) # caught here.
        print("***** did NOT throw expected exception *****")
    except RuntimeError as e:
        print (format(e))
        num_exception += 1
    num_expected += 1

    print("**** equation with illegal dependency (same var and time)")
    try:
        make_soln()
        n0 = A.new_var_point([t+1, x, y, z])
        n4 = A.new_var_point([t+1, x+1, y+1, z])
        ns_eq = nfac.new_equation_node(n0, n4);
        print(ns_eq.format_simple())
        soln.set_target("avx");
        soln.output_solution(yask_file) # caught here.
        print("***** did NOT throw expected exception *****")
    except RuntimeError as e:
        print (format(e))
        num_exception += 1
    num_expected += 1

    print("**** equation with illegal dependency (circular)")
    try:
        make_soln()
        n1 = A.new_var_point([t+1, x, y, z])
        n2 = B.new_var_point([t+1, x, y, z])
        eq1 = nfac.new_equation_node(n1, n2);
        print(eq1.format_simple())
        eq2 = nfac.new_equation_node(n2, n1);
        print(eq2.format_simple())
        soln.set_target("avx");
        soln.output_solution(yask_file) # caught here.
        print("***** did NOT throw expected exception *****")
    except RuntimeError as e:
        print (format(e))
        num_exception += 1
    num_expected += 1

    print("**** equation with illegal dependency (scratch var)")
    try:
        make_soln()
        n1 = C.new_var_point([t+1, nfac.new_const_number_node(5), x, y, z])
        n2 = C.new_var_point([t+1, nfac.new_const_number_node(6), x, y, z])
        eq1 = nfac.new_equation_node(n1, n2); # ok dep across non-scratch vars w/diff misc indices.
        print(eq1.format_simple())
        n3 = T1.new_var_point([nfac.new_const_number_node(3), x, y, z]);
        n4 = T1.new_var_point([nfac.new_const_number_node(4), x, y, z]);
        eq2 = nfac.new_equation_node(n3, n4); # not ok dep across scratch vars w/diff misc indices.
        print(eq2.format_simple())
        eq3 = nfac.new_equation_node(n2, n4);
        print(eq3.format_simple())
        soln.set_target("avx");
        soln.output_solution(yask_file) # caught here.
        print("***** did NOT throw expected exception *****")
    except RuntimeError as e:
        print (format(e))
        num_exception += 1
    num_expected += 1

    print("**** call 'new_file_output' with invalid dir.")
    try:
        dot_file = ofac.new_file_output("/does-not-exist/foo.dot")
        print("***** did NOT throw expected exception *****")
    except RuntimeError as e:
        print (format(e))
        num_exception += 1
    num_expected += 1

    print("**** call 'set_target' with invalid target.")
    try:
        make_soln()
        soln.set_target("bad_target")
        soln.output_solution(yask_file) # caught here.
        print("***** did NOT throw expected exception *****")
    except RuntimeError as e:
        print (format(e))
        num_exception += 1
    num_expected += 1

    # Check whether program handles exceptions or not.
    print("Caught", num_exception, "of", num_expected, "expected exceptions.")
    if num_exception != num_expected:
        exit(1)
    print("End of YASK compiler API test with exceptions.")
