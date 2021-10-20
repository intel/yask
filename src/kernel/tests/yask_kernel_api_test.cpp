/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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

#include <assert.h>
#include "yask_kernel_api.hpp"
#include <iostream>
#include <vector>
#include <set>
#include <sys/types.h>
#include <unistd.h>


using namespace std;
using namespace yask;

int main() {

    // The factory from which all other kernel objects are made.
    yk_factory kfac;

    // Initalize MPI, etc.
    auto env = kfac.new_env();

    try {

        // Create solution.
        auto soln = kfac.new_solution(env);

        // Show output only from last rank.
        // This is an example of using the rank APIs,
        // the yask_output_factory, and set_debug_output().
        ostream* osp = &cout;
        int rank_num = env->get_rank_index();
        if (rank_num < env->get_num_ranks() - 1) {
            yask_output_factory ofac;
            auto null_out = ofac.new_null_output();
            soln->set_debug_output(null_out);
            osp = &null_out->get_ostream();
            cout << "Suppressing output on rank " << rank_num << ".\n";
        }
        else
            cout << "Following information from rank " << rank_num << ".\n";
        ostream& os = *osp;

        // Init solution settings.
        auto soln_dims = soln->get_domain_dim_names();
        for (auto dim_name : soln_dims) {

            // Set domain size in each dim.
            soln->set_overall_domain_size(dim_name, 128);

            // Ensure some minimal padding on all vars.
            soln->set_min_pad_size(dim_name, 1);

            // Set block size to 64 in z dim and 32 in other dims.
            // NB: just illustrative.
            if (dim_name == "z")
                soln->set_block_size(dim_name, 64);
            else
                soln->set_block_size(dim_name, 32);
        }

        // Make a test fixed-size var.
        auto fvar_dims = soln_dims;
        fvar_dims.push_back("misc1");
        vector<idx_t> fvar_sizes;
        for (auto dim_name : fvar_dims)
            fvar_sizes.push_back(5);
        auto fvar = soln->new_fixed_size_var("fvar", fvar_dims, fvar_sizes);

        // Allocate memory for any vars that do not have storage set.
        // Set other data structures needed for stencil application.
        soln->prepare_solution();

        // Print some info about the solution.
        auto name = soln->get_name();
        os << "Stencil-solution '" << name << "':\n";
        os << "  Step dimension: '" << soln->get_step_dim_name() << "'\n";
        os << "  Domain dimensions:";
        set<string> domain_dim_set;
        for (auto dname : soln->get_domain_dim_names()) {
            os << " '" << dname << "'";
            domain_dim_set.insert(dname);
        }
        os << endl;

        // Print out some info about the vars and init their data.
        for (auto var : soln->get_vars()) {
            os << "    var '" << var->get_name() << ":\n";
            for (auto dname : var->get_dim_names()) {
                os << "      '" << dname << "' dim:\n";
                os << "        alloc-size on this rank: " <<
                    var->get_alloc_size(dname) << endl;
                
                // Is this a domain dim?
                if (domain_dim_set.count(dname)) {
                    os << "        domain index range on this rank: " <<
                        var->get_first_rank_domain_index(dname) << " ... " <<
                        var->get_last_rank_domain_index(dname) << endl;
                    os << "        domain+halo index range on this rank: " <<
                        var->get_first_rank_halo_index(dname) << " ... " <<
                        var->get_last_rank_halo_index(dname) << endl;
                    os << "        allowed index range on this rank: " <<
                        var->get_first_rank_alloc_index(dname) << " ... " <<
                        var->get_last_rank_alloc_index(dname) << endl;
                }

                // Step dim?
                else if (dname == soln->get_step_dim_name()) {
                    os << "        currently-valid step index range: " <<
                        var->get_first_valid_step_index() << " ... " <<
                        var->get_last_valid_step_index() << endl;
                }

                // Misc dim?
                else {
                    os << "        misc index range: " <<
                        var->get_first_misc_index(dname) << " ... " <<
                        var->get_last_misc_index(dname) << endl;
                }
            }

            // First, just init all the elements to the same value.
            var->set_all_elements_same(0.5);

            // Done with fixed-size vars.
            if (var->is_fixed_size())
                continue;

            // Create indices describing a subset of the overall domain.
            vector<idx_t> first_indices, last_indices;
            for (auto dname : var->get_dim_names()) {
                idx_t first_idx = 0, last_idx = 0;

                // Is this a domain dim?
                if (domain_dim_set.count(dname)) {

                    // Set indices to create a small cube (assuming 3D)
                    // in center of overall problem.
                    // Using global indices.
                    idx_t psize = soln->get_overall_domain_size(dname);
                    first_idx = psize/2 - 30;
                    last_idx = psize/2 + 30;
                }

                // Step dim?
                else if (dname == soln->get_step_dim_name()) {

                    // Set indices for valid time-steps.
                    first_idx = var->get_first_valid_step_index();
                    last_idx = var->get_last_valid_step_index();
                    assert(last_idx - first_idx + 1 == var->get_alloc_size(dname));
                }

                // Misc dim?
                else {

                    // Set indices to set all allowed values.
                    first_idx = var->get_first_misc_index(dname);
                    last_idx = var->get_last_misc_index(dname);
                    assert(last_idx - first_idx + 1 == var->get_alloc_size(dname));
                }

                // Add indices to index vectors.
                first_indices.push_back(first_idx);
                last_indices.push_back(last_idx);
            }

            // Init the values using the indices created above.
            double val = 2.0;
            bool strict_indices = false; // because first/last_indices are global.
            idx_t nset = var->set_elements_in_slice_same(val, first_indices, last_indices, strict_indices);
            os << "      " << nset << " element(s) set in sub-range from " <<
                var->format_indices(first_indices) << " to " <<
                var->format_indices(last_indices) << ".\n";
            if (var->are_indices_local(first_indices)) {
                auto val2 = var->get_element(first_indices);
                os << "      first element == " << val2 << ".\n";
                assert(val2 == val);
            }
            else
                os << "      first element NOT in rank.\n";
            if (var->are_indices_local(last_indices)) {
                auto val2 = var->get_element(last_indices);
                os << "      last element == " << val2 << ".\n";
                assert(val2 == val);
            }
            else
                os << "      last element NOT in rank.\n";

            // Add to a couple of values if they're in this rank.
            nset = var->add_to_element(1.0, first_indices);
            nset += var->add_to_element(3.0, last_indices);
            os << "      " << nset << " element(s) updated.\n";
            if (var->are_indices_local(first_indices)) {
                auto val2 = var->get_element(first_indices);
                os << "      first element == " << val2 << ".\n";
                assert(val2 == val + 1.0);
            }
            if (var->are_indices_local(last_indices)) {
                auto val2 = var->get_element(last_indices);
                os << "      last element == " << val2 << ".\n";
                assert(val2 == val + 3.0);
            }

            // Raw access to this var.
            auto raw_p = var->get_raw_storage_buffer();
            auto num_elems = var->get_num_storage_elements();
            os << "      " << var->get_num_storage_bytes() <<
                " bytes of raw data at " << raw_p << ": ";
            if (soln->get_element_bytes() == 4)
                os << ((float*)raw_p)[0] << ", ..., " << ((float*)raw_p)[num_elems-1] << "\n";
            else
                os << ((double*)raw_p)[0] << ", ..., " << ((double*)raw_p)[num_elems-1] << "\n";
        }

        // Apply the stencil solution to the data.
        env->global_barrier();
        os << "Running the solution for 1 step...\n";
        soln->run_solution(0);
        os << "Running the solution for 10 more steps...\n";
        soln->run_solution(1, 10);

        soln->end_solution();

        os << "End of YASK kernel API test.\n";
        return 0;
    }
    catch (yask_exception e) {
        cerr << "YASK kernel API test: " << e.get_message() <<
            " on rank " << env->get_rank_index() << ".\n";
        return 1;
    }
}
