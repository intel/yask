/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

*****************************************************************************/

// Test the YASK stencil kernel API for C++.

#include "yask_kernel_api.hpp"
#include <iostream>
#include <vector>
#include <set>

using namespace std;
using namespace yask;

int main() {

    // The factory from which all other kernel objects are made.
    yk_factory kfac;

    // Initalize MPI, etc.
    auto env = kfac.new_env();

    // Create solution.
    auto soln = kfac.new_solution(env);

    // Init global settings.
    auto soln_dims = soln->get_domain_dim_names();
    for (auto dim_name : soln_dims) {

        // Set domain size in each dim.
        soln->set_rank_domain_size(dim_name, 128);

        // Ensure some minimal padding on all grids.
        soln->set_min_pad_size(dim_name, 1);

        // Set block size to 64 in z dim and 32 in other dims.
        if (dim_name == "z")
            soln->set_block_size(dim_name, 64);
        else
            soln->set_block_size(dim_name, 32);
    }

    // Make a test fixed-size grid.
    vector<idx_t> fgrid_sizes;
    for (auto dim_name : soln_dims)
        fgrid_sizes.push_back(5);
    auto fgrid = soln->new_fixed_size_grid("fgrid", soln_dims, fgrid_sizes);

    // Simple rank configuration in 1st dim only.
    auto ddim1 = soln_dims[0];
    soln->set_num_ranks(ddim1, env->get_num_ranks());

    // Allocate memory for any grids that do not have storage set.
    // Set other data structures needed for stencil application.
    soln->prepare_solution();

    // Print some info about the solution.
    auto name = soln->get_name();
    cout << "Stencil-solution '" << name << "':\n";
    cout << "  Step dimension: '" << soln->get_step_dim_name() << "'\n";
    cout << "  Domain dimensions:";
    set<string> domain_dim_set;
    for (auto dname : soln->get_domain_dim_names()) {
        cout << " '" << dname << "'";
        domain_dim_set.insert(dname);
    }
    cout << endl;

    // Print out some info about the grids and init their data.
    for (auto grid : soln->get_grids()) {
        cout << "    " << grid->get_name() << "(";
        for (auto dname : grid->get_dim_names())
            cout << " '" << dname << "'";
        cout << " )\n";
        for (auto dname : grid->get_dim_names()) {
            if (domain_dim_set.count(dname)) {
                cout << "      '" << dname << "' domain index range on this rank: " <<
                    grid->get_first_rank_domain_index(dname) << " ... " <<
                    grid->get_last_rank_domain_index(dname) << endl;
                cout << "      '" << dname << "' allowed index range on this rank: " <<
                    grid->get_first_rank_alloc_index(dname) << " ... " <<
                    grid->get_last_rank_alloc_index(dname) << endl;
            }
        }

        // First, just init all the elements to the same value.
        grid->set_all_elements_same(0.1);

        // Done with fixed-size grids.
        if (grid->is_fixed_size())
            continue;
        
        // Create indices describing a subset of the overall domain.
        vector<idx_t> first_indices, last_indices;
        for (auto dname : grid->get_dim_names()) {

            // Is this a domain dim?
            if (domain_dim_set.count(dname)) {

                // Set indices to creaete a small cube (assuming 3D)
                // in center of overall problem.
                idx_t psize = soln->get_overall_domain_size(dname);
                idx_t first_idx = psize/2 - 10;
                idx_t last_idx = psize/2 + 10;
                first_indices.push_back(first_idx);
                last_indices.push_back(last_idx);
            }

            // Step dim?
            else if (dname == soln->get_step_dim_name()) {

                // Add indices for timestep zero (0) only.
                first_indices.push_back(0); 
                last_indices.push_back(0);

            }

            // Misc dim?
            else {

                // Add indices to set all allowed values.
                // (This isn't really meaningful; it's just illustrative.)
                first_indices.push_back(grid->get_first_misc_index(dname));
                last_indices.push_back(grid->get_last_misc_index(dname));
            }
        }
        
        // Init the values using the indices created above.
        idx_t nset = grid->set_elements_in_slice_same(0.9, first_indices, last_indices);
        cout << "      " << nset << " element(s) set.\n";

        // Raw access to this grid.
        auto raw_p = grid->get_raw_storage_buffer();
        auto num_elems = grid->get_num_storage_elements();
        cout << "      " << grid->get_num_storage_bytes() <<
            " bytes of raw data at " << raw_p << ": ";
        if (soln->get_element_bytes() == 4)
            cout << ((float*)raw_p)[0] << ", ..., " << ((float*)raw_p)[num_elems-1] << "\n";
        else
            cout << ((double*)raw_p)[0] << ", ..., " << ((double*)raw_p)[num_elems-1] << "\n";
    }

    // Apply the stencil solution to the data.
    env->global_barrier();
    cout << "Running the solution for 1 step...\n";
    soln->run_solution(0);
    cout << "Running the solution for 10 more steps...\n";
    soln->run_solution(1, 10);

    cout << "End of YASK kernel API test.\n";
    return 0;
}
