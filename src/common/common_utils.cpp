/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

//////// Some common code shared between YASK compiler and kernel. //////////

// Include this first to assure NDEBUG is set properly.
#include "yask_assert.hpp"

#include <sstream>
#include "common_utils.hpp"

using namespace std;

namespace yask {

    // A var that behaves like OMP_NUM_THREADS.
    int yask_num_threads[yask_max_levels] = { 0 };

    // Update this version string anytime changes are
    // committed to a repository, especially when
    // affecting master or develop branches.
    // Be sure to keep 2 digits in minor and patch
    // fields to allow proper alphanumeric sorting
    // for numbers above 9 (at least up to 99).

    // Format: "major.minor.patch".
    const string version = "2.15.10";

    string yask_get_version_string() {
        return version;
    }

    // See yask_common_api.hpp for documentation.
    const char* yask_exception::what() noexcept {
        return "yask::yask_exception\n";
    }

    void yask_exception::add_message(const string& arg_msg) {
        _msg.append(arg_msg);
    }

    const char* yask_exception::get_message() const {
        return _msg.c_str();
    }
}
