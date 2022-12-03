/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2022, Intel Corporation

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
#include <string.h>

using namespace std;
using namespace yask;

int main(int argc, char** argv) {

    int rank_num = -1;
    try {

        // The factory from which all other kernel objects are made.
        yk_factory kfac;
        
        // Initalize MPI, etc.
        auto env = kfac.new_env();
        rank_num = env->get_rank_index();

        // Local options.
        for (int i = 1; i < argc; i++)
            if (string(argv[i]) == "-trace")
                env->set_trace_enabled(true);
        
        // Create solution.
        auto soln = kfac.new_solution(env);

        // Apply any YASK command-line options.
        soln->apply_command_line_options(argc, argv);

        // Show output only from last rank.
        if (rank_num < env->get_num_ranks() - 1) {
            yk_env::disable_debug_output();
            cout << "Suppressing output on rank " << rank_num << ".\n";
        }
        else
            cout << "Following information from rank " << rank_num << ".\n";
        ostream& os = yk_env::get_debug_output()->get_ostream();

        // Init solution settings.
        auto soln_dims = soln->get_domain_dim_names();
        int i = 0;
        for (auto dim_name : soln_dims) {
            os << "Setting solution dim '" << dim_name << "'...\n";

            // Set domain size in each dim.
            idx_t dsize = 64 + i * 32;
            soln->set_overall_domain_size(dim_name, dsize);

            // Check that vec API returns same.
            auto dsizes = soln->get_overall_domain_size_vec();
            os << "global domain sizes:";
            for (auto ds : dsizes)
                os << " " << ds;
            os << "\n";
            assert(dsizes[i] == dsize);

            // Set with vec and check again.
            soln->set_overall_domain_size_vec(dsizes);
            auto ds = soln->get_overall_domain_size(dim_name);
            assert(ds == dsize);

            // Ensure some minimal padding on all vars.
            soln->set_min_pad_size(dim_name, 1);

            // Set block size to 64 in last dim and 32 in other dims.
            // NB: just illustrative.
            idx_t bsize = (i == soln_dims.size() - 1) ? 64 : 32;
            soln->set_block_size(dim_name, bsize);

            // Check that vec API returns same.
            auto bsizes = soln->get_block_size_vec();
            os << "block sizes:";
            for (auto bs : bsizes)
                os << " " << bs;
            os << "\n";
            assert(bsizes[i] == bsize);

            // Set with vec and check again.
            soln->set_block_size_vec(bsizes);
            auto bs = soln->get_block_size(dim_name);
            assert(bs == bsize);
            
            i++;
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

        // For each var, print out some info about it and init its data.
        for (auto var : soln->get_vars()) {
            auto ndims = var->get_num_dims();
            os << "    var '" << var->get_name() << "' has " <<
                ndims << " dimension(s):\n";
            auto dnames = var->get_dim_names();
            int dimi = 0;
            for (auto dname : dnames) {
                os << "      '" << dname << "' dim:\n";
                os << "        alloc-size on this rank: " <<
                    var->get_alloc_size(dname) << endl;
                os << "        allowed index range on this rank: " <<
                    var->get_first_local_index(dname) << " ... " <<
                    var->get_last_local_index(dname) << endl;
                assert(var->get_alloc_size(dname) ==
                       var->get_last_local_index(dname) - var->get_first_local_index(dname) + 1);
                auto asizes = var->get_alloc_size_vec();
                assert(asizes.at(dimi) == var->get_alloc_size(dname));
                
                // Is this a domain dim?
                if (domain_dim_set.count(dname)) {
                    os << "        domain index range on this rank: " <<
                        var->get_first_rank_domain_index(dname) << " ... " <<
                        var->get_last_rank_domain_index(dname) << endl;
                    os << "        domain+halo index range on this rank: " <<
                        var->get_first_rank_halo_index(dname) << " ... " <<
                        var->get_last_rank_halo_index(dname) << endl;
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

                dimi++;
            }
            auto first_indices = var->get_first_local_index_vec();
            os << "      first local indices: " << var->format_indices(first_indices) << endl;
            auto last_indices = var->get_last_local_index_vec();
            os << "      last local indices: " << var->format_indices(last_indices) << endl;
            auto asizes = var->get_alloc_size_vec();
            os << "      local allocation sizes: " << var->format_indices(asizes) << endl;

            // First, just init all the elements to the same value.
            double outer_val = 0.5;
            var->set_all_elements_same(outer_val);

            // Done with fixed-size vars.
            if (var->is_fixed_size())
                continue;

            // Create indices describing a subset of the overall domain.
            // If running on more than one rank, it's very likely
            // part of the cube will not be in this rank.
            idx_t_vec first_cube_indices, last_cube_indices;
            for (auto dname : var->get_dim_names()) {
                idx_t first_idx = 0, last_idx = 0;

                // Is this a domain dim?
                if (domain_dim_set.count(dname)) {

                    // Set indices to create a small cube (assuming 3D)
                    // in center of overall problem.
                    // Using global indices.
                    idx_t psize = soln->get_overall_domain_size(dname);
                    idx_t center = psize/2;
                    first_idx = center - 30;
                    last_idx = center + 30;
                }

                // Step dim?
                else if (dname == soln->get_step_dim_name()) {

                    // Set indices for all valid time-steps.
                    first_idx = var->get_first_valid_step_index();
                    last_idx = var->get_last_valid_step_index();
                    assert(last_idx - first_idx + 1 == var->get_alloc_size(dname));
                }

                // Misc dim?
                else {

                    // Set indices for all valid misc-index values.
                    first_idx = var->get_first_misc_index(dname);
                    last_idx = var->get_last_misc_index(dname);
                    assert(last_idx - first_idx + 1 == var->get_alloc_size(dname));
                }

                // Add indices to index vectors.
                first_cube_indices.push_back(first_idx);
                last_cube_indices.push_back(last_idx);
            }
            assert(first_cube_indices.size() == ndims);
            assert(last_cube_indices.size() == ndims);

            // Print range of inner cube.
            if (ndims == 0)
                os << "      inner-cube is a single point for this scalar var.\n";
            else
                os << "      range of inner-cube: " <<
                var->format_indices(first_cube_indices) << " to " <<
                var->format_indices(last_cube_indices) << ".\n";

            // Check 2 corners of inner-cube to make sure they were set properly.
            if (var->are_indices_local(first_cube_indices)) {
                auto val2 = var->get_element(first_cube_indices);
                os << "      first element in inner cube == " << val2 << ".\n";
                assert(val2 == outer_val);
            }
            else
                os << "      first element NOT in this rank.\n";
            if (var->are_indices_local(last_cube_indices)) {
                auto val2 = var->get_element(last_cube_indices);
                os << "      last element in inner cube == " << val2 << ".\n";
                assert(val2 == outer_val);
            }
            else
                os << "      last element NOT in this rank.\n";

            // Re-init the values in the inner cube using the indices created above.
            // Use 'strict_indices = false' because first/last_cube_indices are global,
            // so part or all of cube may be outside this rank.
            double inner_val = 2.0;
            bool strict_indices = false;
            idx_t nset = var->set_elements_in_slice_same(inner_val, first_cube_indices,
                                                         last_cube_indices, strict_indices);
            os << "      " << nset << " element(s) in inner-cube set to " << inner_val << ".\n";
            
            // Check values on 2 corners of the inner cube to make sure they
            // were set properly.  Only check if element is in this rank.
            if (var->are_indices_local(first_cube_indices)) {
                auto val2 = var->get_element(first_cube_indices);
                os << "      first element in inner cube == " << val2 << ".\n";
                assert(val2 == inner_val);
            }
            if (var->are_indices_local(last_cube_indices)) {
                auto val2 = var->get_element(last_cube_indices);
                os << "      last element in inner cube == " << val2 << ".\n";
                assert(val2 == inner_val);
            }

            // Add to value in first corner of the inner cube if it's in
            // this rank.  Here, we can use the default 'strict_indices =
            // true' parameter to add_to_element() because the call is
            // inside the check for local indices.
            if (var->are_indices_local(first_cube_indices)) {
                nset = var->add_to_element(1.0, first_cube_indices);
                os << "      " << nset << " element(s) updated.\n";
                assert(nset == 1);

                // Check to make sure they were updated.
                auto val2 = var->get_element(first_cube_indices);
                os << "      first element in inner cube == " << val2 << ".\n";
                assert(val2 == inner_val + 1.0);
            }

            // Raw access to this var.
            auto raw_p = var->get_raw_storage_buffer();
            auto num_elems = var->get_num_storage_elements();
            os << "      " << var->get_num_storage_bytes() <<
                " bytes of raw data at " << raw_p << ": ";
            assert(raw_p);
            auto esize = soln->get_element_bytes();
            if (esize == 4)
                os << ((float*)raw_p)[0] << ", ..., " << ((float*)raw_p)[num_elems-1] << "\n";
            else if (esize == 8)
                os << ((double*)raw_p)[0] << ", ..., " << ((double*)raw_p)[num_elems-1] << "\n";
            else
                assert(false);

            // Set values from a buffer.
            size_t nelems = 1;
            for (int i = 0; i < ndims; i++)
                nelems *= asizes[i];
            size_t bsize = nelems * esize;
            assert(bsize >= esize);
            os << "      allocating " << bsize << " byte(s)...\n";
            void* buf = malloc(bsize);
            assert(buf);
            memset(buf, 0, bsize); // Sets all reals to 0.0, whether floats or doubles.

            // Try with all zeros.
            nset = var->set_elements_in_slice(buf, first_indices, last_indices);
            assert(nset == nelems);
            auto v1 = var->get_element(first_indices);
            assert(v1 == 0.0);
            auto v2 = var->get_element(last_indices);
            assert(v2 == 0.0);

            // Try with first element non-zero.
            double v3 = 123.0;
            if (esize == 4)
                ((float*)buf)[0] = float(v3);
            else
                ((double*)buf)[0] = v3;
            nset = var->set_elements_in_slice(buf, first_indices, last_indices);
            assert(nset == nelems);
            v1 = var->get_element(first_indices);
            assert(v1 == v3);
            if (nelems > 1) {
                v2 = var->get_element(last_indices);
                assert(v2 == 0.0);
            }

            // Try with last element also non-zero.
            if (esize == 4)
                ((float*)buf)[nelems-1] = float(v3);
            else
                ((double*)buf)[nelems-1] = v3;
            nset = var->set_elements_in_slice(buf, first_indices, last_indices);
            assert(nset == nelems);
            v1 = var->get_element(first_indices);
            assert(v1 == v3);
            v2 = var->get_element(last_indices);
            assert(v2 == v3);

            free(buf);
            os << "      " << nelems << " element(s) set via set_elements_in_slice()\n";
        }

        // Apply the stencil solution to the data.
        soln->reset_auto_tuner(false);
        env->global_barrier();
        os << "Running the solution on " << env->get_num_ranks() << " rank(s)...\n";
        os << "Running for 1 step...\n";
        soln->run_solution(0);
        os << "Running for 4 more steps...\n";
        soln->run_solution(1, 4);

        soln->end_solution();
        soln->get_stats();
        env->finalize();
        os << "End of YASK C++ kernel API test.\n";
        return 0;
    }
    catch (yask_exception e) {
        cerr << "YASK kernel API test: " << e.get_message() <<
            " on rank " << rank_num << ".\n";
        return 1;
    }
}
