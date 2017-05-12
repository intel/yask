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
sys.path.append('../swig')

import yask_compiler
cfac = yask_compiler.yask_compiler_factory()
soln = cfac.new_stencil_solution("api_py_test")
g1 = soln.new_grid("test_grid", "t", "x", "y", "z")

fac = yask_compiler.node_factory()

n1 = fac.new_const_number_node(3.14)
print(n1.format_simple())

n2 = fac.new_negate_node(n1)
print(n2.format_simple())

n3 = g1.new_relative_grid_point(0, 1, 0, -2)
print(n3.format_simple())

n4a = fac.new_add_node(n2, n3)
n4b = fac.new_add_node(n4a, n1)
print(n4b.format_simple())

n5 = g1.new_relative_grid_point(0, 1, -1, 0)
print(n5.format_simple())

n6 = fac.new_divide_node(n4b, n5)
print(n6.format_simple())

n7 = g1.new_relative_grid_point(1, 0, 0, 0)
print(n7.format_simple())

n8 = fac.new_equation_node(n7, n6)
print(n8.format_simple())

print("Solution '" + soln.get_name() + "' contains " +
      str(soln.get_num_grids()) + " grid(s), and " +
      str(soln.get_num_equations()) + " equation(s).")

soln.set_step_dim("t");
soln.set_fold_len("y", 8)

dot_file = "api-py-test.dot"
soln.write(dot_file, "dot", True)
print("DOT-format written to '" + dot_file + "'.")

yask_file = "stencil_code.hpp"
soln.write(yask_file, "avx", True)
print("YASK-format written to '" + yask_file + "'.")

