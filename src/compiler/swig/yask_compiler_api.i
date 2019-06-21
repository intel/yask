/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2019, Intel Corporation

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

// See http://www.swig.org/Doc4.0/Library.html
%include <std_string.i>
%include <std_shared_ptr.i>
%include <std_vector.i>

// Shared API.
%include "yask_common_api.i"

// Must declare shared_ptrs for the entire expr_node hierarchy!
%shared_ptr(yask::yc_solution)
 //%shared_ptr(yask::yc_var)
%shared_ptr(yask::yc_expr_node)
%shared_ptr(yask::yc_index_node)
%shared_ptr(yask::yc_equation_node)
%shared_ptr(yask::yc_number_node)
%shared_ptr(yask::yc_var_point_node)
%shared_ptr(yask::yc_const_number_node)
%shared_ptr(yask::yc_negate_node)
%shared_ptr(yask::yc_commutative_number_node)
%shared_ptr(yask::yc_binary_number_node)
%shared_ptr(yask::yc_binary_bool_node)
%shared_ptr(yask::yc_binary_comparison_node)
%shared_ptr(yask::yc_add_node)
%shared_ptr(yask::yc_multiply_node)
%shared_ptr(yask::yc_subtract_node)
%shared_ptr(yask::yc_divide_node)
%shared_ptr(yask::yc_mod_node)
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
%shared_ptr(yask::yc_number_ptr_arg)
%shared_ptr(yask::yc_number_const_arg)
%shared_ptr(yask::yc_number_any_arg)

// From http://www.swig.org/Doc4.0/SWIG.html#SWIG_nn2: Everything in the %{
// ... %} block is simply copied verbatim to the resulting wrapper file
// created by SWIG. This section is almost always used to include header
// files and other declarations that are required to make the generated
// wrapper code compile. It is important to emphasize that just because you
// include a declaration in a SWIG input file, that declaration does not
// automatically appear in the generated wrapper code---therefore you need
// to make sure you include the proper header files in the %{ ... %}
// section. It should be noted that the text enclosed in %{ ... %} is not
// parsed or interpreted by SWIG.
%{
#define SWIG_FILE_WITH_INIT
#include "yask_kernel_api.hpp"
%}
%{
#define SWIG_FILE_WITH_INIT
#include "yask_compiler_api.hpp"
%}

// All vector types used in API.
%template(yc_vector_int) std::vector<int>;
%template(yc_vector_str) std::vector<std::string>;
%template(yc_vector_index) std::vector<std::shared_ptr<yask::yc_index_node>>;
%template(yc_vector_num) std::vector<std::shared_ptr<yask::yc_number_node>>;
%template(yc_vector_eq) std::vector<std::shared_ptr<yask::yc_equation_node>>;
%template(yc_vector_var) std::vector<yask::yc_var*>;

 // Tell SWIG how to catch a YASK exception and rethrow it in Python.
%exception {
  try {
    $action
  } catch (yask::yask_exception &e) {
    PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.get_message()));
    SWIG_fail;
  }
}

// Tell SWIG how to handle non-class overloaded operators in Python.
// See https://docs.python.org/3/library/operator.html.

// For numerical ops.
%extend yask::yc_number_node {
    yask::yc_number_node_ptr __neg__() {
        auto p = $self->clone_ast();
        return yask::operator-(p);
    }
 };
%extend yask::yc_index_node {
    yask::yc_number_node_ptr __neg__() {
        auto p = $self->clone_ast();
        return yask::operator-(p);
    }
 };

%define BIN_OP(py_oper, c_oper)
%extend yask::yc_number_node {
    yask::yc_number_node_ptr py_oper(yask::yc_number_node* rhs) {
        auto lp = $self->clone_ast();
        auto rp = rhs->clone_ast();
        return yask::operator c_oper(lp, rp);
    }
 }
%extend yask::yc_number_node {
    yask::yc_number_node_ptr py_oper(double rhs) {
        auto lp = $self->clone_ast();
        return yask::operator c_oper(lp, rhs);
    }
 };
%extend yask::yc_number_node {
    yask::yc_number_node_ptr py_oper(idx_t rhs) {
        auto lp = $self->clone_ast();
        return yask::operator c_oper(lp, rhs);
    }
 };
%enddef
BIN_OP(__add__, +);
BIN_OP(__sub__, -);
BIN_OP(__mul__, *);
BIN_OP(__truediv__, /);
BIN_OP(__mod__, %);

// For boolean ops.

// For 'not', 'and', and 'or', Python only allows returning
// a bool. So we have to make our own.
%extend yask::yc_bool_node {
    yask::yc_bool_node_ptr yc_not() {
        auto p = $self->clone_ast();
        return yask::operator!(p);
    }
 };
%extend yask::yc_bool_node {
    yask::yc_bool_node_ptr yc_or(yask::yc_bool_node* rhs) {
        auto lp = $self->clone_ast();
        auto rp = rhs->clone_ast();
        return yask::operator||(lp, rp);
    }
 };
%extend yask::yc_bool_node {
    yask::yc_bool_node_ptr yc_and(yask::yc_bool_node* rhs) {
        auto lp = $self->clone_ast();
        auto rp = rhs->clone_ast();
        return yask::operator&&(lp, rp);
    }
 };

%define BOOL_OP(py_oper, c_oper)
%extend yask::yc_number_node {
    yask::yc_bool_node_ptr py_oper(yask::yc_number_node* rhs) {
        auto lp = $self->clone_ast();
        auto rp = rhs->clone_ast();
        return yask::operator c_oper(lp, rp);
    }
};
%extend yask::yc_number_node {
    yask::yc_bool_node_ptr py_oper(yask::yc_index_node* rhs) {
        auto lp = $self->clone_ast();
        auto rp = rhs->clone_ast();
        return yask::operator c_oper(lp, rp);
    }
};
%extend yask::yc_index_node {
    yask::yc_bool_node_ptr py_oper(yask::yc_number_node* rhs) {
        auto lp = $self->clone_ast();
        auto rp = rhs->clone_ast();
        return yask::operator c_oper(lp, rp);
    }
};
%extend yask::yc_index_node {
    yask::yc_bool_node_ptr py_oper(yask::yc_index_node* rhs) {
        auto lp = $self->clone_ast();
        auto rp = rhs->clone_ast();
        return yask::operator c_oper(lp, rp);
    }
};
%enddef
BOOL_OP(__eq__, ==);
BOOL_OP(__ne__, !=);
BOOL_OP(__lt__, <);
BOOL_OP(__gt__, >);
BOOL_OP(__ge__, >=);
BOOL_OP(__le__, <=);

%include "yask_common_api.hpp"
%include "yask_compiler_api.hpp"
%include "aux/yc_node_api.hpp"
