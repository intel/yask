#!/usr/bin/env python

##############################################################################
## YASK: Yet Another Stencil Kernel
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

## Test the YASK stencil kernel API for Python.

import numpy as np
import ctypes as ct
import argparse
import yask_kernel as yk

# Prepare an NymPy ndarray to hold a slice of 'grid'.
def make_ndarray(grid, timestep) :

    # Create indices for YASK and shape for NumPy.
    first_indices = []
    last_indices = []
    shape = []
    point = ()
    nelems = 1
    for dname in grid.get_dim_names() :

        if dname == soln.get_step_dim_name() :

            # Slice in requested time.
            first_idx = timestep
            last_idx = timestep

        # Domain dim?
        elif dname in soln.get_domain_dim_names() :

            # Cover full alloc in this rank.
            first_idx = grid.get_first_rank_alloc_index(dname)
            last_idx = grid.get_last_rank_alloc_index(dname)

        # Misc dim?
        else :

            # Cover all misc values.
            first_idx = grid.get_first_misc_index(dname)
            last_idx = grid.get_last_misc_index(dname)

        # Add indices to API vars.
        first_indices += [first_idx]
        last_indices += [last_idx]
        ne = last_idx - first_idx + 1
        shape += [ne]
        nelems *= ne

        # Define first point in ndarray.
        point += (0, )

    # Create a NumPy ndarray to hold the extracted data.
    print("Creating a NumPy ndarray with shape " + repr(shape) + " and " +
          repr(nelems) + " element(s)...")
    ndarray = np.zeros(shape, dtype, 'C');
    return ndarray, first_indices, last_indices, point

# Read data from grid using NumPy ndarray.
def read_grid(grid, timestep) :

    # Ignore with fixed-sized grids.
    if grid.is_fixed_size():
        return
    print("Testing reading grid '" + grid.get_name() + "' at time " + repr(timestep) + "...")
    ndarray, first_indices, last_indices, point = make_ndarray(grid, timestep)

    print("Reading 1 element...")
    val1 = grid.get_element(first_indices)
    print("Read value " + repr(val1))

    print("Reading all element(s) in ndarray...")
    nread = grid.get_elements_in_slice(ndarray.data, first_indices, last_indices)
    print(ndarray)

    # Raw access to this grid.
    if soln.get_element_bytes() == 4 :
        ptype = ct.POINTER(ct.c_float)
    else :
        ptype = ct.POINTER(ct.c_double)
    raw_ptr = grid.get_raw_storage_buffer()
    fp_ptr = ct.cast(int(raw_ptr), ptype)
    num_elems = grid.get_num_storage_elements()
    print("Raw data: " + repr(fp_ptr[0]) + ", ..., " + repr(fp_ptr[num_elems-1]))

# Init grid using NumPy ndarray.
def init_grid(grid, timestep) :
    print("Initializing grid '" + grid.get_name() + "' at time " + repr(timestep) + "...")
    ndarray, first_indices, last_indices, point = make_ndarray(grid, timestep)

    # Set one point to a non-zero value.
    val1 = 21.0
    ndarray[point] = val1
    print(ndarray)

    print("Setting grid from all element(s) in ndarray...")
    nset = grid.set_elements_in_slice(ndarray.data, first_indices, last_indices)
    print("Set " + repr(nset) + " element(s) in rank " + repr(env.get_rank_index()))

    # Check that set worked.
    print("Reading those element(s)...")
    val2 = grid.get_element(first_indices)
    assert val2 == val1
    val2 = grid.get_element(last_indices)
    if nset > 1 :
        assert val2 == 0.0
    else :
        assert val2 == val1  # Only 1 val => first == last.
    ndarray2 = ndarray
    ndarray2.fill(5.0)
    nread = grid.get_elements_in_slice(ndarray2.data, first_indices, last_indices)
    assert nread == ndarray2.size
    assert ndarray2[point] == val1
    assert ndarray2.sum() == val1  # One point is val1; others are zero.

    # Test element set.
    print("Testing setting 1 point at " + repr(last_indices) + "...")
    val1 += 1.0
    nset = grid.set_element(val1, last_indices);
    assert nset == 1
    val2 = grid.get_element(last_indices)
    assert val2 == val1

    # Test add.
    val3 = 2.0
    print("Testing adding to 1 point at " + repr(last_indices) + "...")
    nset = grid.add_to_element(val3, last_indices);
    assert nset == 1
    val2 = grid.get_element(last_indices)
    assert val2 == val1 + val3
    
# Main script.
if __name__ == "__main__":

    # The factories from which all other kernel objects are made.
    kfac = yk.yk_factory()
    ofac = yk.yask_output_factory()

    # Initalize MPI, etc.
    env = kfac.new_env()

    # Create solution.
    soln = kfac.new_solution(env)
    debug_output = ofac.new_string_output()
    soln.set_debug_output(debug_output)
    name = soln.get_name()

    # NB: At this point, the grids' meta-data exists, but the grids have no
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

        # Ensure some minimal padding on all grids.
        soln.set_min_pad_size(dim_name, 1)

        # Set block size to 64 in z dim and 32 in other dims.
        # (Not necessarily useful, just as an example.)
        if dim_name == "z" :
            soln.set_block_size(dim_name, 64)
        else :
            soln.set_block_size(dim_name, 32)

    # Make a test fixed-size grid and set its NUMA preference.
    fgrid_sizes = ()
    for dim_name in soln_dims :
        fgrid_sizes += (5,)
    fgrid = soln.new_fixed_size_grid("fgrid", soln_dims, fgrid_sizes)
    fgrid.set_numa_preferred(yk.cvar.yask_numa_local)
    fgrid.alloc_storage()

    # Allocate memory for any grids that do not have storage set.
    # Set other data structures needed for stencil application.
    soln.prepare_solution()

    # Print some info about the solution.
    print("Stencil-solution '" + name + "':")
    print("  Step dimension: " + repr(soln.get_step_dim_name()))
    print("  Domain dimensions: " + repr(soln.get_domain_dim_names()))
    print("  Grids:")
    for grid in soln.get_grids() :
        print("    " + grid.get_name() + repr(grid.get_dim_names()))
        for dname in grid.get_dim_names() :
            if dname in soln.get_domain_dim_names() :
                print("      '" + dname + "' allowed domain index range in this rank: " +
                      repr(grid.get_first_rank_alloc_index(dname)) + " ... " +
                      repr(grid.get_last_rank_alloc_index(dname)))
            elif dname == soln.get_step_dim_name() :
                print("      '" + dname + "' allowed step index range: " +
                      repr(grid.get_first_valid_step_index()) + " ... " +
                      repr(grid.get_last_valid_step_index()))
            else :
                print("      '" + dname + "' allowed misc index range: " +
                      repr(grid.get_first_misc_index(dname)) + " ... " +
                      repr(grid.get_last_misc_index(dname)))

    # Init the grids.
    for grid in soln.get_grids() :

        # Init all values including padding.
        grid.set_all_elements_same(-9.0)

        # Done with fixed-sized grids.
        if grid.is_fixed_size():
            continue
        
        # Init timestep 0 using NumPy.
        init_grid(grid, 0)

        # Print out the values.
        read_grid(grid, 0)

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
            
        for dname in grid.get_dim_names() :

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
                one_indices += [grid.get_first_misc_index(dname)]
                first_indices += [grid.get_first_misc_index(dname)]
                last_indices += [grid.get_last_misc_index(dname)]

        # Init value at one point.
        nset = grid.set_element(15.0, one_indices)
        print("Set " + repr(nset) + " element(s) in rank " + repr(env.get_rank_index()))

        # Init the values within the small cube.
        nset = grid.set_elements_in_slice_same(0.5, first_indices, last_indices)
        print("Set " + repr(nset) + " element(s) in rank " + repr(env.get_rank_index()))

        # Print the initial contents of the grid.
        read_grid(grid, 0)

    # Apply the stencil solution to the data.
    env.global_barrier()
    print("Running the solution for 1 step...")
    soln.run_solution(0)

    # Print result at timestep 1.
    for grid in soln.get_grids() :
        read_grid(grid, 1)

    print("Running the solution for 10 more steps...")
    soln.run_solution(1, 10)

    # Print final result at timestep 11, assuming update was to t+1.
    for grid in soln.get_grids() :
        read_grid(grid, 11)

    print("Debug output captured:\n" + debug_output.get_string())
    print("End of YASK kernel API test.")
