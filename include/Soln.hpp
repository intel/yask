/*****************************************************************************

YASK: Yet Another Stencil Kernel
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

// This file contains convenience functions and macros for defining
// stencils to be included in the YASK compiler binary utility.
// It is to be used only for backward compatibility for old-style
// stencil DSL code prior to YASK version 2.23.00.

#pragma once

// These functions and macros are built upon the YASK compiler utility APIs.
#include "yask_compiler_utility_api.hpp"

namespace yask {

    /// Dummy type for backward-compatibility.
    class StencilList { };

    /// Dummy object from compiler utility for backward-compatibility.
    extern StencilList stub_stencils;

    /// **[Deprecated]** The class all old-style C++ stencil solutions
    /// written for the YASK compiler binary must implement.
    /**
       New DSL code should use yc_solution_base directly.
    */
    class StencilBase : public yc_solution_base {
        
    public:

        /// Backward-compatibility contructor.
        StencilBase(const std::string& name, StencilList& stencils) :
            yc_solution_base(name) { }

        /// Create a constant expression.
        /**
           Usually not needed due to operator overloading. 
           New DSL code should use new_const_number_node() or
           simply new_number_node().
        */
        virtual yc_number_node_ptr constNum(double val) {
            return _node_factory.new_const_number_node(val);
        }
    };

    /// **[Deprecated]** A base class for stencils that have a 'radius'.
    class StencilRadiusBase : public yc_solution_with_radius_base {

    public:
        
        /// Backward-compatibility constructor.
        StencilRadiusBase(const std::string& name, StencilList& stencils, int radius) :
            yc_solution_with_radius_base(name, radius) { }

        /// Create a constant expression.
        /** Usually not needed due to operator overloading. */
        virtual yc_number_node_ptr constNum(double val) {
            return _node_factory.new_const_number_node(val);
        }
    };

    // Aliases for backward-compatibility.
    typedef yc_grid_var Grid;
    typedef yc_number_node_ptr GridIndex;
    typedef yc_number_node_ptr GridValue;
    typedef yc_bool_node_ptr Condition;
    typedef yc_grid_point_node_ptr GridPointPtr;
    typedef yc_expr_node_ptr ExprPtr;
    typedef yc_number_node_ptr NumExprPtr;
    typedef yc_index_node_ptr IndexExprPtr;
    typedef yc_bool_node_ptr BoolExprPtr;

} // namespace yask.

/// Macro for backward compatibility.
#define REGISTER_STENCIL_CONTEXT_EXTENSION(...)

/// Convenience macro for declaring an object of a type derived from \ref StencilBase
/// and registering it in the list used by the default provided YASK compiler binary.
/** The derived class must implement a constructor that takes only a \ref StencilList
    reference. */
#define REGISTER_STENCIL(class_name) \
    static class_name registered_ ## class_name(stub_stencils)

/// Convenience macros for declaring dims in a class derived from \ref StencilBase.
/// The 'd' arg is the new var name and the dim name.
#define MAKE_STEP_INDEX(d)   yc_index_node_ptr d = _node_factory.new_step_index(#d);
#define MAKE_DOMAIN_INDEX(d) yc_index_node_ptr d = _node_factory.new_domain_index(#d);
#define MAKE_MISC_INDEX(d)   yc_index_node_ptr d = _node_factory.new_misc_index(#d);

/// Convenience macros for creating grids in a class implementing get_solution().
/// The 'gvar' arg is the var name and the grid name.
/// The remaining args are the dimension names.
#define MAKE_GRID(gvar, ...)                                            \
    Grid gvar = Grid(#gvar, get_solution(), { __VA_ARGS__ }, false)
#define MAKE_SCALAR(gvar)    MAKE_GRID(gvar)
#define MAKE_ARRAY(gvar, d1) MAKE_GRID(gvar, d1)
#define MAKE_SCRATCH_GRID(gvar, ...)                                    \
    Grid gvar = Grid(#gvar, get_solution(), { __VA_ARGS__ }, true)
#define MAKE_SCRATCH_SCALAR(gvar)    MAKE_SCRATCH_GRID(gvar)
#define MAKE_SCRATCH_ARRAY(gvar, d1) MAKE_SCRATCH_GRID(gvar, d1)

// Namespaces for stencil code.
#ifndef NO_NAMESPACES
using namespace yask;
using namespace std;
#endif

