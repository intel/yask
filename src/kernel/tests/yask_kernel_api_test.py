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

## Test the YASK stencil kernel API for Python.

import numpy as np
import ctypes as ct
import argparse
import yask_kernel as yk

# Prepare a NumPy ndarray to hold a slice of 'var'.
def make_ndarray(var, timestep) :

    # Create indices for YASK and shape for NumPy.
    first_indices = []
    last_indices = []
    shape = []
    point = ()
    nelems = 1
    for dname in var.get_dim_names() :

        if dname == soln.get_step_dim_name() :

            # Slice in requested time.
            first_idx = timestep
            last_idx = timestep

        # Domain dim?
        elif dname in soln.get_domain_dim_names() :

            # Cover full alloc in this rank.
            first_idx = var.get_first_rank_alloc_index(dname)
            last_idx = var.get_last_rank_alloc_index(dname)

        # Misc dim?
        else :

            # Cover all misc values.
            first_idx = var.get_first_misc_index(dname)
            last_idx = var.get_last_misc_index(dname)

        # Add indices to API vars.
        first_indices += [first_idx]
        last_indices += [last_idx]
        ne = last_idx - first_idx + 1
        shape += [ne]
        nelems *= ne

        # Define first point in ndarray.
        point += (0, )

    # Create a NumPy ndarray to hold the extracted data.
    print("Creating a NumPy ndarray with shape ", shape, " and ",
          nelems, " element(s)...")
    ndarray = np.zeros(shape, dtype, 'C');
    return ndarray, first_indices, last_indices, point

# Read data from var using NumPy ndarray.
def read_var(var, timestep) :

    # Ignore with fixed-sized vars.
    if var.is_fixed_size():
        return
    print("Testing reading var '", var.get_name(), "' at time ", timestep, "...")
    ndarray, first_indices, last_indices, point = make_ndarray(var, timestep)

    print("Reading corner elements...")
    val1 = var.get_element(first_indices)
    print("Read value ", val1)
    val2 = var.get_element(last_indices)
    print("Read value ", val2)

    print("Reading element(s) in ndarray from ", first_indices, " to ", last_indices, "...")
    nread = var.get_elements_in_slice(ndarray.data, first_indices, last_indices)
    print(ndarray)

    # Raw access to this var.
    if soln.get_element_bytes() == 4 :
        ptype = ct.POINTER(ct.c_float)
    else :
        ptype = ct.POINTER(ct.c_double)
    raw_ptr = var.get_raw_storage_buffer()
    fp_ptr = ct.cast(int(raw_ptr), ptype)
    num_elems = var.get_num_storage_elements()
    print("Raw data: ", fp_ptr[0], ", ..., ", fp_ptr[num_elems-1])

# Init var using NumPy ndarray.
def init_var(var, timestep) :
    print("Initializing var '", var.get_name(), "' at time ", timestep, "...")
    ndarray, first_indices, last_indices, point = make_ndarray(var, timestep)

    # Set one point to a non-zero value.
    val1 = 21.0
    ndarray[point] = val1
    print(ndarray)

    print("Setting var from all element(s) in ndarray...")
    nset = var.set_elements_in_slice(ndarray.data, first_indices, last_indices)
    print("Set ", nset, " element(s) in rank ", env.get_rank_index())

    # Check that set worked.
    print("Reading those element(s)...")
    val2 = var.get_element(first_indices)
    assert val2 == val1
    val2 = var.get_element(last_indices)
    if nset > 1 :
        assert val2 == 0.0
    else :
        assert val2 == val1  # Only 1 val => first == last.
    ndarray2 = ndarray
    ndarray2.fill(5.0)
    nread = var.get_elements_in_slice(ndarray2.data, first_indices, last_indices)
    assert nread == ndarray2.size
    assert ndarray2[point] == val1
    assert ndarray2.sum() == val1  # One point is val1; others are zero.

    # Test element set.
    print("Testing setting 1 point at ", last_indices, "...")
    val1 += 1.0
    nset = var.set_element(val1, last_indices);
    assert nset == 1
    val2 = var.get_element(last_indices)
    assert val2 == val1

    # Test add.
    val3 = 2.0
    print("Testing adding to 1 point at ", last_indices, "...")
    nset = var.add_to_element(val3, last_indices);
    assert nset == 1
    val2 = var.get_element(last_indices)
    assert val2 == val1 + val3
    
# Main script.
if __name__ == "__main__":

    # The factories from which all other kernel objects are made.
    kfac = yk.yk_factory()
    ofac = yk.yask_output_factory()

    # Initalize MPI, etc.
    env = kfac.new_env()
    if env.get_rank_index() > 0:
        env.disable_debug_output();
    else:
        env.set_trace_enabled(True);

    # Create solution.
    soln = kfac.new_solution(env)
    #debug_output = ofac.new_string_output()
    #env.set_debug_output(debug_output)
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

        # Ensure some minimal padding on all vars.
        soln.set_min_pad_size(dim_name, 1)

        # Set block size to 64 in z dim and 32 in other dims.
        # (Not necessarily useful, just as an example.)
        if dim_name == "z" :
            soln.set_block_size(dim_name, 64)
        else :
            soln.set_block_size(dim_name, 32)

    # Make a test fixed-size var and set its NUMA preference.
    fvar_sizes = ()
    for dim_name in soln_dims :
        fvar_sizes += (5,)
    fvar = soln.new_fixed_size_var("fvar", soln_dims, fvar_sizes)
    fvar.set_numa_preferred(yk.cvar.yask_numa_local)
    fvar.alloc_storage()

    # Allocate memory for any vars that do not have storage set.
    # Set other data structures needed for stencil application.
    soln.prepare_solution()

    # Print some info about the solution.
    print("Stencil-solution '", name, "':")
    print("  Step dimension:", soln.get_step_dim_name())
    print("  Domain dimensions:", soln.get_domain_dim_names())
    print("  Vars:")
    for var in soln.get_vars() :
        print("    ", var.get_name(), var.get_dim_names())
        for dname in var.get_dim_names() :
            if dname in soln.get_domain_dim_names() :
                print("      '", dname, "' allowed domain index range in this rank: ",
                      var.get_first_rank_alloc_index(dname), "...",
                      var.get_last_rank_alloc_index(dname))
            elif dname == soln.get_step_dim_name() :
                print("      '", dname, "' allowed step index range: ",
                      var.get_first_valid_step_index(), "...",
                      var.get_last_valid_step_index())
            else :
                print("      '", dname, "' allowed misc index range: ",
                      var.get_first_misc_index(dname), "...",
                      var.get_last_misc_index(dname))
        print("      allowed range in all dims: ",
              var.get_first_local_index_vec(), "...",
              var.get_last_local_index_vec())

    # Init the vars.
    for var in soln.get_vars() :

        # Init all values including padding.
        var.set_all_elements_same(-9.0)

        # Done with fixed-sized vars.
        if var.is_fixed_size():
            continue
        
        # Init timestep 0 using NumPy.
        init_var(var, 0)

        # Print out the values.
        read_var(var, 0)

        # Simple one-index example.
        # Note that index relative to overall problem domain,
        # so it will only appear in one rank.
        one_index = 100
        one_indices = []

        # Create indices to bound a subset of domain:
        # Index 0 in time, and a small cube in center
        # of overall problem.
        # Note that indices are relative to overall problem domain,
        # so the cube may be in one rank or it may be spread over
        # more than one.
        cube_radius = 20
        first_indices = []
        last_indices = []
            
        for dname in var.get_dim_names() :

            # Step dim?
            if dname == soln.get_step_dim_name() :

                # Add index for timestep zero (0) only.
                one_indices += [0]
                first_indices += [0]
                last_indices += [0]

            # Domain dim?
            elif dname in soln.get_domain_dim_names() :

                # Simple index for one point.
                one_indices += [one_index]
                    
                # Midpoint of overall problem in this dim.
                midpt = soln.get_overall_domain_size(dname) // 2;

                # Create indices a small amount before and after the midpoint.
                first_indices += [midpt - cube_radius]
                last_indices += [midpt + cube_radius]

            # Misc dim?
            else :

                # Add indices to set all allowed values.
                # (This isn't really meaningful; it's just illustrative.)
                one_indices += [var.get_first_misc_index(dname)]
                first_indices += [var.get_first_misc_index(dname)]
                last_indices += [var.get_last_misc_index(dname)]

        # Init value at one point.
        nset = var.set_element(15.0, one_indices, False)
        print("Set ", nset, " element(s) in rank ", env.get_rank_index())

        # Init the values within the small cube.
        nset = var.set_elements_in_slice_same(0.5, first_indices, last_indices, False)
        print("Set ", nset, " element(s) in rank ", env.get_rank_index())

        # Print the initial contents of the var.
        read_var(var, 0)

    # Apply the stencil solution to the data.
    env.global_barrier()
    print("Running the solution on", env.get_num_ranks(), "rank(s)");
    print("Running for 1 step...")
    soln.run_solution(0)

    # Print result at timestep 1.
    for var in soln.get_vars() :
        read_var(var, 1)

    print("Running for 4 more steps...")
    soln.run_solution(1, 4)

    # Print final result at timestep 5, assuming last update was to t+1.
    for var in soln.get_vars() :
        read_var(var, 5)

    soln.end_solution()
    soln.get_stats()
    env.finalize()

    #print("Debug output captured:\n", debug_output.get_string())
    print("End of YASK Python kernel API test.")
