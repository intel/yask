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

///////// API for the YASK stencil compiler. ////////////

// This file uses SWIG markup for API generation.

%module YC_MODULE

// See http://www.swig.org/Doc3.0/Library.html
%include <std_string.i>
%include <std_shared_ptr.i>
%include <std_vector.i>

// Shared API.
%include "yask_common_api.i"

// Must declare shared_ptrs for the entire expr_node hierarchy!
%shared_ptr(yask::yc_solution)
 //%shared_ptr(yask::yc_grid)
%shared_ptr(yask::yc_expr_node)
%shared_ptr(yask::yc_index_node)
%shared_ptr(yask::yc_equation_node)
%shared_ptr(yask::yc_number_node)
%shared_ptr(yask::yc_grid_point_node)
%shared_ptr(yask::yc_const_number_node)
%shared_ptr(yask::yc_negate_node)
%shared_ptr(yask::yc_commutative_number_node)
%shared_ptr(yask::yc_add_node)
%shared_ptr(yask::yc_multiply_node)
%shared_ptr(yask::yc_subtract_node)
%shared_ptr(yask::yc_divide_node)
%shared_ptr(yask::yc_bool_node)
%shared_ptr(yask::yc_not_node)
%shared_ptr(yask::yc_equals_node)
%shared_ptr(yask::yc_not_equals_node)
%shared_ptr(yask::yc_less_than_node)
%shared_ptr(yask::yc_greater_than_node)
%shared_ptr(yask::yc_not_less_than_node)
%shared_ptr(yask::yc_not_greater_than_node)
%shared_ptr(yask::yc_and_node)
%shared_ptr(yask::yc_or_node)

%{
#define SWIG_FILE_WITH_INIT
#include "yask_compiler_api.hpp"
%}

// All vector types used in API.
%template(vector_int) std::vector<int>;
%template(vector_str) std::vector<std::string>;
%template(vector_index) std::vector<std::shared_ptr<yask::yc_index_node>>;
%template(vector_num) std::vector<std::shared_ptr<yask::yc_number_node>>;
%template(vector_eq) std::vector<std::shared_ptr<yask::yc_equation_node>>;
%template(vector_grid) std::vector<yask::yc_grid*>;

%exception {
  try {
    $action
  } catch (yask::yask_exception &e) {
    PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.get_message()));
    SWIG_fail;
  }
}

%include "yask_common_api.hpp"
%include "yask_compiler_api.hpp"
%include "yc_nodes.hpp"
