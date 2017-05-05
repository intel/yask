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

///////// API for the YASK stencil compiler. ////////////

// This file uses SWIG markup for API generation.

%module yask_compiler

// See http://www.swig.org/Doc3.0/Library.html
%include <std_string.i>
%include <std_shared_ptr.i>

// Must declare shared_ptrs for the entire expr_node hierarchy!
%shared_ptr(yask::stencil_solution)
%shared_ptr(yask::expr_node)
%shared_ptr(yask::number_node)
%shared_ptr(yask::grid_point_node)
%shared_ptr(yask::const_number_node)
%shared_ptr(yask::negate_node)
%shared_ptr(yask::commutative_number_node)
%shared_ptr(yask::add_node)
%shared_ptr(yask::multiply_node)
%shared_ptr(yask::add_node)
%shared_ptr(yask::subtract_node)
%shared_ptr(yask::divide_node)
%shared_ptr(yask::bool_node)

%{
#include "yask_compiler_api.hpp"
%}

%include "yask_compiler_api.hpp"
