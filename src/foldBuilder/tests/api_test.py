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
fac = yask_compiler.node_factory()

n1 = fac.new_const_number_node(3.14)
print(n1.format_simple())

n2 = fac.new_negate_node(n1)
print(n2.format_simple())

v1 = n1.get_value()
n3 = fac.new_const_number_node(v1 * 2.0)
print(n3.format_simple())

n4 = fac.new_add_node(n2, n3)
print(n4.format_simple())

n5 = fac.new_const_number_node(v1 * 3.0)
print(n5.format_simple())

n6 = fac.new_divide_node(n4, n5)
print(n6.format_simple())
n6l = n6.get_lhs()
print(" LHS: " + n6l.format_simple())
n6r = n6.get_rhs()
print(" RHS: " + n6r.format_simple())
