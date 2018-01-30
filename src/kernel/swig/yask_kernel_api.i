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

///////// API for the YASK stencil kernel. ////////////

// This file uses SWIG markup for API generation.

%module YK_MODULE

// See http://www.swig.org/Doc3.0/Library.html
%include <std_string.i>
%include <std_shared_ptr.i>
%include <std_vector.i>
%include <pybuffer.i>

// Shared API.
%include "yask_common_api.i"

// Must declare shared_ptr for each one used in the API.
%shared_ptr(yask::yk_env)
%shared_ptr(yask::yk_settings)
%shared_ptr(yask::yk_solution)
%shared_ptr(yask::yk_grid)
%shared_ptr(yask::yk_stats)

// Mutable buffer to access raw data.
%pybuffer_mutable_string(void* buffer_ptr)

%{
#define SWIG_FILE_WITH_INIT
#include "yask_kernel_api.hpp"
%}

// All vector types used in API.
%template(vector_idx) std::vector<long int>;
%template(vector_str) std::vector<std::string>;
%template(vector_grid_ptr) std::vector<std::shared_ptr<yask::yk_grid>>;

%exception {
  try {
    $action
  } catch (yask::yask_exception &e) {
    PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.get_message()));
    SWIG_fail;
  }
}

%include "yask_common_api.hpp"
%include "yask_kernel_api.hpp"
