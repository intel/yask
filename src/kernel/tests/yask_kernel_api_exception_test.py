#!/usr/bin/env python

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2021, Intel Corporation
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

import numpy as np
import ctypes as ct
import argparse
import yask_kernel

# Main script.
if __name__ == "__main__":

    # Counter for exception test
    num_exception = 0;

    # The factories from which all other kernel objects are made.
    kfac = yask_kernel.yk_factory()
    ofac = yask_kernel.yask_output_factory()

    # Initalize MPI, etc.
    env = kfac.new_env()

    # Create solution.
    soln = kfac.new_solution(env)
    debug_output = ofac.new_string_output()
    env.set_debug_output(debug_output)
    name = soln.get_name()

    # NB: At this point, the vars' meta-data exists, but the vars have no
    # data allocated. We need to set the size of the domain before
    # allocating data.

    # Determine the datatype.
    if soln.get_element_bytes() == 4 :
        dtype = np.float32
    else :
        dtype = np.float64
        
    # Init global settings.
    soln_dims = soln.get_domain_dim_names()
    
    for dim_name in soln_dims :

        # Set domain size in each dim.
        soln.set_overall_domain_size(dim_name, 128)

    # Exception test
    print("Exception Test: Call 'run_solution' without calling prepare_solution().")
    try:
        soln.run_solution(0)
    except RuntimeError as e:
        print ("YASK throws an exception.")
        print (format(e))
        print ("Exception Test: Catch exception correctly.")
        num_exception = num_exception + 1

    # Exception test
    print("Exception Test: Call 'run_auto-tuner_now' without calling prepare_solution().")
    try:
        soln.run_auto_tuner_now(False)
    except RuntimeError as e:
        print ("YASK throws an exception.")
        print (format(e))
        print ("Exception Test: Catch exception correctly.")
        num_exception = num_exception + 1
    
    # Allocate memory for any vars that do not have storage set.
    # Set other data structures needed for stencil application.
    soln.prepare_solution()

    # Print some info about the solution.
    print("Stencil-solution '" + name + "':")
    print("  Step dimension: " + repr(soln.get_step_dim_name()))
    print("  Domain dimensions: " + repr(soln.get_domain_dim_names()))
    print("  Vars:")
    for var in soln.get_vars() :
        print("    " + var.get_name() + repr(var.get_dim_names()))
        for dname in var.get_dim_names() :
            if dname in soln.get_domain_dim_names() :
                print("      '" + dname + "' allowed index range in this rank: " +
                      repr(var.get_first_rank_alloc_index(dname)) + " ... " +
                      repr(var.get_last_rank_alloc_index(dname)))
            elif dname in soln.get_misc_dim_names() :
                print("      '" + dname + "' allowed index range: " +
                      repr(var.get_first_misc_index(dname)) + " ... " +
                      repr(var.get_last_misc_index(dname)))

    # Init the vars.
    for var in soln.get_vars() :

        # Init all values including padding.
        var.set_all_elements_same(1.0)

    # Apply the stencil solution to the data.
    env.global_barrier()
    print("Running the solution for 1 step...")
    soln.run_solution(0)

    soln.end_solution()
    soln.get_stats()
    print("Debug output captured:\n" + debug_output.get_string())

    if num_exception != 2:
        print("There is a problem in exception test.")
        exit(1)
    else:
        print("End of YASK kernel API test with exception.")
