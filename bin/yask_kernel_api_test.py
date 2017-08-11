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

import numpy as np
import ctypes as ct
import argparse
import sys
sys.path.append('lib')
import yask_kernel

# Read data from grid using NumPy ndarray.
def read_grid(grid, timestep) :
    print("Reading grid '" + grid.get_name() + "' at time " + repr(timestep) + "...")

    # Create indices for YASK and shape for NumPy.
    first_indices = []
    last_indices = []
    shape = []
    nelems = 1
    for dname in grid.get_dim_names() :

        if dname == soln.get_step_dim_name() :

            # Read one timestep only.
            # So, we don't need to add a time axis in 'shape'.
            first_indices += [timestep]
            last_indices += [timestep]

        # Domain dim?
        elif dname in soln.get_domain_dim_names() :

            # Full domain in this rank.
            first_idx = soln.get_first_rank_domain_index(dname)
            last_idx = soln.get_last_rank_domain_index(dname)

            # Read one point in the halo, too.
            first_idx -= 1
            last_idx += 1

            first_indices += [first_idx]
            last_indices += [last_idx]
            shape += [last_idx - first_idx + 1]
            nelems *= last_idx - first_idx + 1

        # Misc dim?
        else :

            # Read first index only.
            first_indices += [grid.get_first_misc_index(dname)]
            last_indices += [grid.get_first_misc_index(dname)]

    # Create a NumPy ndarray to hold the extracted data.
    ndarray1 = np.empty(shape, dtype, 'C');

    print("Reading " + repr(nelems) + " element(s)...")
    nread = grid.get_elements_in_slice(ndarray1.data, first_indices, last_indices)
    print(ndarray1)

    # Raw access to this grid.
    if soln.get_element_bytes() == 4 :
        ptype = ct.POINTER(ct.c_float)
    else :
        ptype = ct.POINTER(ct.c_double)
    raw_ptr = grid.get_raw_storage_buffer()
    fp_ptr = ct.cast(int(raw_ptr), ptype)
    num_elems = grid.get_num_storage_elements()
    print("Raw data: " + repr(fp_ptr[0]) + ", ..., " + repr(fp_ptr[num_elems-1]))
    #ndarray2 = np.fromiter(fp_ptr, dtype, num_elems); print(ndarray2)

# Init grid using NumPy ndarray.
def init_grid(grid, timestep) :
    print("Initializing grid '" + grid.get_name() + "' at time " + repr(timestep) + "...")

    # Create indices for YASK, shape & point for NumPy.
    first_indices = []
    last_indices = []
    shape = []
    point = ()
    nelems = 1
    for dname in grid.get_dim_names() :

        if dname == soln.get_step_dim_name() :

            # Write one timestep only.
            # So, we don't need to add a time axis in 'shape'.
            first_indices += [timestep]
            last_indices += [timestep]

        # Domain dim?
        elif dname in soln.get_domain_dim_names() :

            # Full domain in this rank.
            first_idx = soln.get_first_rank_domain_index(dname)
            last_idx = soln.get_last_rank_domain_index(dname)

            # Write one point in the halo, too.
            first_idx -= 1
            last_idx += 1

            first_indices += [first_idx]
            last_indices += [last_idx]
            shape += [last_idx - first_idx + 1]
            nelems *= last_idx - first_idx + 1

            # Since the array covers one layer of points in the halo
            # starting at 0,..,0, 1,..,1 is the first point in the
            # computable domain.
            point += (1,)

        # Misc dim?
        else :

            # Write first index only.
            first_indices += [grid.get_first_misc_index(dname)]
            last_indices += [grid.get_first_misc_index(dname)]

    # Create a NumPy ndarray to hold the data.
    ndarray = np.zeros(shape, dtype, 'C');

    # Set one point to a non-zero value.
    ndarray[point] = 21.0;
    print(ndarray)

    print("Writing " + repr(nelems) + " element(s)...")
    nset = grid.set_elements_in_slice(ndarray.data, first_indices, last_indices)
    print("Set " + repr(nset) + " element(s) in rank " + repr(env.get_rank_index()))

# Main script.
if __name__ == "__main__":

    # The factories from which all other kernel objects are made.
    kfac = yask_kernel.yk_factory()
    ofac = yask_kernel.yask_output_factory()

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
    for dim_name in soln.get_domain_dim_names() :

        # Set domain size in each dim.
        soln.set_rank_domain_size(dim_name, 128)

        # Ensure some minimal padding on all grids.
        soln.set_min_pad_size(dim_name, 1)

        # Set block size to 64 in z dim and 32 in other dims.
        # (Not necessarily useful, just as an example.)
        if dim_name == "z" :
            soln.set_block_size(dim_name, 64)
        else :
            soln.set_block_size(dim_name, 32)

    # Simple rank configuration in 1st dim only.
    # In production runs, the ranks would be distributed along
    # all domain dimensions.
    ddim1 = soln.get_domain_dim_name(0) # name of 1st dim.
    soln.set_num_ranks(ddim1, env.get_num_ranks()) # num ranks in this dim.

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
                print("      '" + dname + "' allowed index range in this rank: " +
                      repr(grid.get_first_rank_alloc_index(dname)) + " ... " +
                      repr(grid.get_last_rank_alloc_index(dname)))
            elif dname in soln.get_misc_dim_names() :
                print("      '" + dname + "' allowed index range: " +
                      repr(grid.get_first_misc_index(dname)) + " ... " +
                      repr(grid.get_last_misc_index(dname)))

    # Init the grids.
    for grid in soln.get_grids() :
        
        # Init all values including padding.
        grid.set_all_elements_same(-9.0)

        # Init timestep 0 using NumPy.
        # This will set one point in each rank.
        init_grid(grid, 0)
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

        # Print the initial contents of the grid at timesteps 0 and 1.
        read_grid(grid, 0)
        read_grid(grid, 1)

    # Apply the stencil solution to the data.
    env.global_barrier()
    print("Running the solution for 1 step...")
    soln.run_solution(0)

    # Print result at timestep 1.
    for grid in soln.get_grids() :
        read_grid(grid, 1)

    print("Running the solution for 100 more steps...")
    soln.run_solution(1, 100)

    # Print final result at timestep 101.
    for grid in soln.get_grids() :
        read_grid(grid, 101)

    print("Debug output captured:\n" + debug_output.get_string())
    print("End of YASK kernel API test.")
