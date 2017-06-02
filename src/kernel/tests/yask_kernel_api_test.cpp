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

using namespace std;
using namespace yask;

int main() {

    // The factory from which all other kernel objects are made.
    yk_factory kfac;

    // Initalize MPI, etc.
    auto env = kfac.new_env();

    // Create settings and solution.
    auto settings = kfac.new_settings();
    auto soln = kfac.new_solution(env, settings);

    // Init global settings.
    for (int di = 0; di < soln->get_num_domain_dims(); di++) {
        auto dim_name = soln->get_domain_dim_name(di);

        // Set min. domain size in each dim.
        settings->set_domain_size(dim_name, 150);

        // Set block size to 64 in z dim and 32 in other dims.
        if (dim_name == "z")
            settings->set_block_size(dim_name, 64);
        else
            settings->set_block_size(dim_name, 32);
    }

    // Simple rank configuration in 1st dim only.
    auto ddim1 = soln->get_domain_dim_name(0);
    settings->set_num_ranks(ddim1, env->get_num_ranks());

    // Allocate memory for any grids that do not have storage set.
    // Set other data structures needed for stencil application.
    soln->prepare_solution();

    // Print some info about the solution and init the grids.
    auto name = soln->get_name();
    cout << "Created stencil-solution '" << name << "' with the following grids:\n";
    for (int gi = 0; gi < soln->get_num_grids(); gi++) {
        auto grid = soln->get_grid(gi);
        cout << "  " << grid->get_name() << "(";
        for (int di = 0; di < grid->get_num_dims(); di++) {
            if (di) cout << ", ";
            cout << grid->get_dim_name(di);
        }
        cout << ")\n";

        grid->set_all_elements(0.0);
    }

    // NB: In a real application, the data in the grids would be
    // loaded or otherwise set to meaningful values here.
    
    // Apply the stencil solution to the data.
    env->global_barrier();
    cout << "Applying the solution for 1 step...\n";
    soln->apply_solution(0);
    cout << "Applying the solution for 100 more steps...\n";
    soln->apply_solution(1, 100);

    cout << "End of YASK kernel API test.\n";
    return 0;
}
