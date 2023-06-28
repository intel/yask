/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

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
    int num_expected = 0;

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

    // Its range.
    auto first_idxs = fvar->get_first_local_index_vec();
    auto last_idxs = fvar->get_last_local_index_vec();
    
    // Exception test
    cout << "Exception Test: Call 'run_solution' without calling prepare_solution().\n";
    num_expected++;
    try {
        soln->run_solution(0);
    } catch (yask_exception& e) {
        cout << e.get_message() << endl;
        num_exception++;
    }

    // Exception test
    cout << "Exception Test: Call 'run_auto_tuner_now' without calling prepare_solution().\n";
    num_expected++;
    try {
        soln->run_auto_tuner_now(false);
    } catch (yask_exception& e) {
        cout << e.get_message() << endl;
        num_exception++;
    }

    // Allocate memory for any vars that do not have storage set.
    // Set other data structures needed for stencil application.
    soln->prepare_solution();

    // Exception test
    cout << "Exception Test: Call 'set_element' with wrong number of indices.\n";
    num_expected++;
    try {

        // Indices.
        auto idxs = first_idxs;
        idxs.push_back(10);     // Extra index.
        
        fvar->set_element(3.14, idxs);
    }  catch (yask_exception& e) {
        cout << e.get_message() << endl;
        num_exception++;
    }

    cout << "Exception Test: Call 'set_elements_in_slice' with wrong number of indices.\n";
    num_expected++;
    try {

        // Indices.
        auto first_idxs2 = first_idxs;
        first_idxs2.push_back(10); // Extra index.

        auto nelems = fvar->get_num_storage_elements();
        double* a = new double[nelems];
        fvar->set_elements_in_slice(a, nelems, first_idxs2, last_idxs);
        free(a);
    }  catch (yask_exception& e) {
        cout << e.get_message() << endl;
        num_exception++;
    }

    cout << "Exception Test: Call 'get_elements_in_slice' with insufficient buffer.\n";
    num_expected++;
    try {

        size_t nelems = fvar->get_num_storage_elements() / 2;
        double* a = new double[nelems];
        fvar->get_elements_in_slice(a, nelems, first_idxs, last_idxs);
        free(a);
    }  catch (yask_exception& e) {
        cout << e.get_message() << endl;
        num_exception++;
    }

    // Apply the stencil solution to the data.
    env->global_barrier();
    cout << "Running the solution for 1 step...\n";
    soln->run_solution(0);

    soln->end_solution();
    soln->get_stats();
    env->finalize();

    // TODO: add exception tests for the methods below:
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
    if (num_exception != num_expected) {
        cout << "Error: unexpected number of exceptions: " << num_exception << endl;
        exit(1);
    }
    else
        cout << "End of YASK kernel C++ API test with exceptions.\n";

    return 0;
}
