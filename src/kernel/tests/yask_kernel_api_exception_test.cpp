/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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

	// Counter for exception test
	int num_exception = 0;

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
        soln->set_overall_domain_size(dim_name, 128);

        // Set block size.
        soln->set_block_size(dim_name, 32);
    }

    // Make a test fixed-size var.
    vector<idx_t> fvar_sizes;
    for (auto dim_name : soln_dims)
        fvar_sizes.push_back(5);
    auto fvar = soln->new_fixed_size_var("fvar", soln_dims, fvar_sizes);

    // Exception test
    cout << "Exception Test: Call 'run_solution' without calling prepare_solution().\n";
    try {
        soln->run_solution(0);
    } catch (yask_exception& e) {
        cout << "YASK throws an exception.\n";
        cout << e.get_message();
        cout << "Exception Test: Catch exception correctly.\n";
        num_exception++;
    }

    // Exception test
    cout << "Exception Test: Call 'run_auto_tuner_now' without calling prepare_solution().\n";
    try {
        soln->run_auto_tuner_now(false);
    } catch (yask_exception& e) {
        cout << "YASK throws an exception.\n";
        cout << e.get_message();
        cout << "Exception Test: Catch exception correctly.\n";
        num_exception++;
    }


    // Allocate memory for any vars that do not have storage set.
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

    // Print out some info about the vars and init their data.
    for (auto var : soln->get_vars()) {
        cout << "    " << var->get_name() << "(";
        for (auto dname : var->get_dim_names())
            cout << " '" << dname << "'";
        cout << " )\n";
        for (auto dname : var->get_dim_names()) {
            if (domain_dim_set.count(dname)) {
                cout << "      '" << dname << "' domain index range on this rank: " <<
                    var->get_first_rank_domain_index(dname) << " ... " <<
                    var->get_last_rank_domain_index(dname) << endl;
                cout << "      '" << dname << "' allowed index range on this rank: " <<
                    var->get_first_rank_alloc_index(dname) << " ... " <<
                    var->get_last_rank_alloc_index(dname) << endl;
            }
        }

        // First, just init all the elements to the same value.
        var->set_all_elements_same(0.1);

        // Done with fixed-size vars.
        if (var->is_fixed_size())
            continue;

        // Create indices describing a subset of the overall domain.
        vector<idx_t> first_indices, last_indices;
        for (auto dname : var->get_dim_names()) {

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
                first_indices.push_back(var->get_first_misc_index(dname));
                last_indices.push_back(var->get_last_misc_index(dname));
            }
        }

        // Init the values using the indices created above.
        idx_t nset = var->set_elements_in_slice_same(0.9, first_indices, last_indices);
        cout << "      " << nset << " element(s) set.\n";

        // Raw access to this var.
        auto raw_p = var->get_raw_storage_buffer();
        auto num_elems = var->get_num_storage_elements();
        cout << "      " << var->get_num_storage_bytes() <<
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
    soln->end_solution();
    soln->get_stats();

    // TODO: better to have exception test for the methods below
    // StencilContext::calc_region
    // StencilContext::add_var
    // StencilContext::setup_rank
    // StencilContext::prepare_solution
    // StencilContext::new_var
    // YkVarBase::get_dim_posn
    // YkVarBase::resize
    // YkVarBase::check_dim_type
    // YkVarBase::share_storage
    // YkVarBase::check_indices
    // YkVarBase::get_element
    // YkVarBase::get_elements_in_slice
    // YkVarBase::_share_data
    // YkVarBase::get_vecs_in_slice
    // real_vec_permute2
    // Dims::check_dim_type
    // KernelEnv::init_env
    // yask_aligned_alloc
    // assert_equality_over_ranks
    // CommandLineParser::OptionBase::_idx_val


    // Check whether program handles exceptions or not.
    if (num_exception != 2) {
        cout << "There is a problem in exception test.\n";
        exit(1);
    }
    else
        cout << "End of YASK kernel API test with exception.\n";

    return 0;
}
