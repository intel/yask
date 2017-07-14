#!/usr/bin/env python

##############################################################################
## YASK: Yet Another Stencil Kernel
## Copyright (c) 2014-2017, Intel Corporation
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

import sys
sys.path.append('lib')
import yask_compiler

if __name__ == "__main__":

    # Compiler 'bootstrap' factory.
    cfac = yask_compiler.yc_factory()

    # Create a new stencil solution.
    soln = cfac.new_solution("api_py_test")
    soln.set_step_dim_name("t")
    soln.set_domain_dim_names(["x", "y", "z"])

    # Create a grid var.
    g1 = soln.new_grid("test_grid", ["t", "x", "y", "z"])

    # Create an expression for the new value.
    # This will average some of the neighboring points around the
    # current stencil application point in the current timestep.
    fac = yask_compiler.yc_node_factory()
    n0 = g1.new_relative_grid_point([0, 0, 0, 0])  # center-point at this timestep.
    n1 = fac.new_add_node(n0, g1.new_relative_grid_point([0, -1,  0,  0])) # left.
    n1 = fac.new_add_node(n1, g1.new_relative_grid_point([0,  1,  0,  0])) # right.
    n1 = fac.new_add_node(n1, g1.new_relative_grid_point([0,  0, -1,  0])) # above.
    n1 = fac.new_add_node(n1, g1.new_relative_grid_point([0,  0,  1,  0])) # below.
    n1 = fac.new_add_node(n1, g1.new_relative_grid_point([0,  0,  0, -1])) # in front.
    n1 = fac.new_add_node(n1, g1.new_relative_grid_point([0,  0,  0,  1])) # behind.
    n2 = fac.new_divide_node(n1, fac.new_const_number_node(7)) # div by 7.

    # Create an equation to define the value at the next timestep.
    n3 = g1.new_relative_grid_point([1, 0, 0, 0]) # center-point at next timestep.
    n4 = fac.new_equation_node(n3, n2) # equate to expr n2.
    print("Equation before formatting: " + n4.format_simple())
    print("Solution '" + soln.get_name() + "' contains " +
          str(soln.get_num_grids()) + " grid(s), and " +
          str(soln.get_num_equations()) + " equation(s).")
    for grid in soln.get_grids() :
        print("Grid " + grid.get_name() +
              " has the following dim(s): " +
              repr(grid.get_dim_names()));

    # Number of bytes in each FP value.
    soln.set_element_bytes(4)

    # Generate DOT output.
    dot_file = "yc-api-test-py.dot"
    soln.write(dot_file, "dot", True)
    print("DOT-format written to '" + dot_file + "'.")

    # Generate YASK output.
    yask_file = "yc-api-test-py.hpp"
    soln.write(yask_file, "avx", True)
    print("YASK-format written to '" + yask_file + "'.")

    print("Equation after formatting: " + soln.get_equation(0).format_simple())
    print("End of YASK compiler API test.")
