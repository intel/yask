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

    yk_factory kfac;
    auto settings = kfac.new_settings();
    settings->set_domain_size("x", 128);
    settings->set_domain_size("y", 128);
    settings->set_domain_size("z", 128);

    auto soln = kfac.new_solution(settings);
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
    }
    
    return 0;
}
