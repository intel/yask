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

## Test the YASK stencil kernel API for Python.

import sys
sys.path.append('lib')
import yask_kernel

# The factory from which all other kernel object are made.
kfac = yask_kernel.yk_factory()

# Create and init settings.
settings = kfac.new_settings()
settings.set_domain_size("x", 128)
settings.set_domain_size("y", 128)
settings.set_domain_size("z", 128)
settings.set_block_size("x", 32)
settings.set_block_size("y", 32)
settings.set_block_size("z", 64)

# Create and init solution based on settings.
soln = kfac.new_solution(settings)
soln.init_env()

# Print some info about the solution.
name = soln.get_name()
print("Created stencil-solution '" + name + "' with the following grids:")
for gi in range(soln.get_num_grids()) :
    grid = soln.get_grid(gi)
    desc = grid.get_name() + "(";
    for di in range(grid.get_num_dims()) :
        if di > 0 :
            desc += ", ";
        desc += grid.get_dim_name(di)
    desc += ")"
    print("  " + desc)

# Allocate memory for any grids that do not have storage set.
# Set other data structures needed for stencil application.
soln.prepare_solution()

print("End of YASK kernel API test.")
