#!/usr/bin/env python

##############################################################################
## YASK: Yet Another Stencil Kernel
## Copyright (c) 2014-2018, Intel Corporation
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

    # Create a grid var.
    g1 = soln.new_grid("test_grid", [t, x, y, z])

    # Create a scratch-grid var.
    sg1 = soln.new_scratch_grid("scratch_grid", [x, y, z]);
    
    # Create an expression for the new value.
    # This will average some of the neighboring points around the
    # current stencil application point in the current timestep.
    n1 = (g1.new_relative_grid_point([0, 0, 0, 0]) +  # center-point at this timestep.
          g1.new_relative_grid_point([0, -1,  0,  0]) + # left.
          g1.new_relative_grid_point([0,  1,  0,  0]) + # right.
          g1.new_relative_grid_point([0,  0, -1,  0]) + # above.
          g1.new_relative_grid_point([0,  0,  1,  0]) + # below.
          g1.new_relative_grid_point([0,  0,  0, -1]) + # in front.
          g1.new_relative_grid_point([0,  0,  0,  1])) # behind.
    n2 = n1 / 7  # ave of the 7 points.

    # Define value in scratch grid to be the above equation, i.e.,
    # this is a temporary 3-D variable that holds the average
    # values of each point.
    sn0 = sg1.new_relative_grid_point([0, 0, 0]) # LHS of eq is just a point on scratch-grid
    sn1 = nfac.new_equation_node(sn0, n2) # equate to expr n2.
    print("Scratch-grid equation before formatting: " + sn1.format_simple())

    # Use values in scratch grid to make a new eq.
    sn2 = (sg1.new_relative_grid_point([1, 0, 0]) +
           sg1.new_relative_grid_point([0, 1, 0]) +
           sg1.new_relative_grid_point([0, 0, 1]))
    sn5 = -sn2 * 2.5 - 9
    
    # Create an equation to define the value at the next timestep.
    n3 = g1.new_relative_grid_point([1, 0, 0, 0]) # center-point at next timestep.
    n4 = nfac.new_equation_node(n3, sn5) # equate to expr from scratch grid.
    print("Main-grid equation before formatting: " + n4.format_simple())
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
    dot_file = ofac.new_file_output("yc-api-test-py.dot")
    soln.format("dot", dot_file)
    print("DOT-format written to '" + dot_file.get_filename() + "'.")

    # Generate YASK output.
    yask_file = ofac.new_file_output("yc-api-test-py.hpp")
    soln.format("avx", yask_file)
    print("YASK-format written to '" + yask_file.get_filename() + "'.")

    for eq in soln.get_equations() :
        print("Equation after formatting: " + eq.format_simple())

    print("Debug output captured:\n" + do.get_string())
    print("End of YASK compiler API test.")
