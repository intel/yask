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

#pragma once

// Standard headers not provided by API.
#include <cassert>

// These functions and macros are built upon the YASK compiler APIs.
#include "yask_compiler_api.hpp"

namespace yask {

    // Forward defn.
    class StencilBase;
    
    // Function provided by the YASK compiler binary utility
    // to create a new solution and register 'base_ptr'
    // as a known StencilBase object.
    yc_solution_ptr yc_new_solution(const std::string& name,
                                    StencilBase* base_ptr);
    
    // Dummy type and object for backward-compatibility.
    class StencilList { };
    extern StencilList stub_stencils;

    // The class all C++ stencil solutions written for the YASK compiler
    // binary must implement. Mostly, this is a wrapper around a
    // 'yc_solution_ptr'. The 'define()' method must be overloaded by
    // the YASK DSL programmer to add stencil equations.
    class StencilBase {
    protected:

        // Pointer to the YASK stencil solution.
        yc_solution_ptr _soln;

        // Factory to create new nodes.
        yc_node_factory _node_factory;
        
    public:

        // Constructor. Creates a new yc_solution object and
        // registers this object in the list for the YASK compiler.
        // The 'define()' method will be called from the YASK
        // compiler for the selected solution.
        StencilBase(const std::string name) {
            _soln = yc_new_solution(name, this);
        }

        // Constructor that uses an existing StencilBase to share underlying
        // solutions. When using this version, the 'define()' method will
        // not be called directly from the YASK compiler, but it (or any
        // other method) may be called by the parent 'base' object
        // explicitly.
        StencilBase(StencilBase& base) {
            _soln = base.get_solution();
        }

        // Backward-compatibility contructor. Deprecated.
        StencilBase(const std::string name, StencilList& stencils) :
            StencilBase(name) { }

        // Destructor.
        virtual ~StencilBase() { }

        // Define grid values relative to current domain indices in each dimension.
        // This must be implemented by each concrete stencil solution.
        virtual void define() {
            std::cout << "Warning: no stencil equations are defined in solution '" <<
                _soln->get_name() << "'\n";
        }

        // Accessors.
        virtual yc_solution_ptr get_solution() {
            return _soln;
        }
        virtual yc_node_factory& get_node_factory() {
            return _node_factory;
        }

        // Create a constant expression.
        // Usually not needed due to operator overloading.
        virtual yc_number_node_ptr constNum(double val) {
            return _node_factory.new_const_number_node(val);
        }

        // Create boundary indices, e.g., 'first_index(x)'.
        virtual yc_number_node_ptr first_index(yc_index_node_ptr dim) {
            return _node_factory.new_first_domain_index(dim);
        }
        virtual yc_number_node_ptr last_index(yc_index_node_ptr dim) {
            return _node_factory.new_last_domain_index(dim);
        }

        // Radius-access methods. The default implementations indicate that
        // the solution does not use the "radius" sizing feature.
        virtual bool uses_radius() const { return false; }
        virtual bool set_radius(int radius) { return false; }
        virtual int get_radius() const { return 0; }
    };

    // A base class for stencils that have a 'radius'.
    class StencilRadiusBase : public StencilBase {
    protected:

        // A variable that controls the size the stencil, i.e., the number of
        // points that are read to calculate a new value.  In many cases,
        // this is the number of points in the spatial dimension(s) from the
        // center point of a finite-difference approximation, but it does
        // not have to be. For example, it could be the minimum or maximum
        // radius for an asymmetical stencil.
        int _radius;

    public:
        // Constructor.
        StencilRadiusBase(const std::string name, int radius) :
            StencilBase(name) {
            set_radius(radius);
        }

        // Backward-compatibility constructor.
        StencilRadiusBase(const std::string name, StencilList& stencils, int radius) :
            StencilRadiusBase(name, radius) {
        }

        // This object does use radius.
        virtual bool uses_radius() const override { return true; }

        // Set radius.
        // Return true if successful.
        virtual bool set_radius(int radius) override {
            _radius = radius;
            _soln->set_description(_soln->get_name() + " radius " + std::to_string(radius));
            return radius >= 0;  // support only non-neg. radius.
        }

        // Get radius.
        virtual int get_radius() const override { return _radius; }
    };

    // A simple wrapper to provide automatic construction
    // of a NumExpr ptr from other types.
    class NumExprArg : public yc_number_node_ptr {

        // Factory to create new nodes.
        yc_node_factory _node_factory;
        
        // Create a constant expression.
        virtual yc_number_node_ptr constNum(double val) {
            return _node_factory.new_const_number_node(val);
        }

    public:
        NumExprArg(yc_number_node_ptr p) :
            yc_number_node_ptr(p) { }
        NumExprArg(yc_index_node_ptr p) :
            yc_number_node_ptr(p) { }
        NumExprArg(idx_t i) :
            yc_number_node_ptr(constNum(i)) { }
        NumExprArg(int i) :
            yc_number_node_ptr(constNum(i)) { }
        NumExprArg(double f) :
            yc_number_node_ptr(constNum(f)) { }
    };

    // A wrapper class around a 'yc_grid_ptr', providing
    // convenience functions for declaring grid vars and
    // creating expression nodes with references to points
    // in grid vars.
    class Grid {
    protected:
        yc_grid_ptr _grid;
        typedef std::vector<yc_number_node_ptr> num_node_vec;
        
    public:

        // Contructor taking a vector of index vars.
        Grid(const std::string& name,
             bool is_scratch,
             yc_solution_ptr soln,
             const std::vector< yc_index_node_ptr > &dims) {
            if (is_scratch)
                _grid = soln->new_scratch_grid(name, dims);
            else
                _grid = soln->new_grid(name, dims);
        }

        // Contructor taking an initializer_list of index vars.
        Grid(const std::string& name,
             bool is_scratch,
             yc_solution_ptr soln,
             const std::initializer_list< yc_index_node_ptr > &dims) {
            if (is_scratch)
                _grid = soln->new_scratch_grid(name, dims);
            else
                _grid = soln->new_grid(name, dims);
        }

        // Convenience wrapper functions around 'new_grid_point()'.
        // These allow a simplified syntax for creating grid point
        // nodes. Example usage for defining equation for grid 'A':
        // 0D grid var 'B': A(t+1, x) EQUALS A(t, x) + B.
        // 0D grid var 'B': A(t+1, x) EQUALS A(t, x) + B().
        // 1D grid var 'B': A(t+1, x) EQUALS A(t, x) + B(x).
        // 1D grid var 'B': A(t+1, x) EQUALS A(t, x) + B[x].
        // 2D grid var 'B': A(t+1, x) EQUALS A(t, x) + B(x, 4).
        
        // Convenience functions for zero dimensions (scalar).
        virtual operator yc_number_node_ptr() { // implicit conversion.
            return _grid->new_grid_point({});
        }
        virtual operator yc_grid_point_node_ptr() { // implicit conversion.
            return _grid->new_grid_point({});
        }
        virtual yc_grid_point_node_ptr operator()() {
            return _grid->new_grid_point({});
        }

        // Convenience functions for one dimension (array).
        virtual yc_grid_point_node_ptr operator[](const NumExprArg i1) {
            num_node_vec args;
            args.push_back(i1);
            return _grid->new_grid_point(args);
        }
        virtual yc_grid_point_node_ptr operator()(const NumExprArg i1) {
            return operator[](i1);
        }

        // Convenience functions for more dimensions.
        virtual yc_grid_point_node_ptr operator()(const NumExprArg i1,
                                                  const NumExprArg i2) {
            num_node_vec args;
            args.push_back(i1);
            args.push_back(i2);
            return _grid->new_grid_point(args);
        }
        virtual yc_grid_point_node_ptr operator()(const NumExprArg i1,
                                                  const NumExprArg i2,
                                                  const NumExprArg i3) {
            num_node_vec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            return _grid->new_grid_point(args);
        }
        virtual yc_grid_point_node_ptr operator()(const NumExprArg i1,
                                                  const NumExprArg i2,
                                                  const NumExprArg i3,
                                                  const NumExprArg i4) {
            num_node_vec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            args.push_back(i4);
            return _grid->new_grid_point(args);
        }
        virtual yc_grid_point_node_ptr operator()(const NumExprArg i1,
                                                  const NumExprArg i2,
                                                  const NumExprArg i3,
                                                  const NumExprArg i4,
                                                  const NumExprArg i5) {
            num_node_vec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            args.push_back(i4);
            args.push_back(i5);
            return _grid->new_grid_point(args);
        }
        virtual yc_grid_point_node_ptr operator()(const NumExprArg i1,
                                                  const NumExprArg i2,
                                                  const NumExprArg i3,
                                                  const NumExprArg i4,
                                                  const NumExprArg i5,
                                                  const NumExprArg i6) {
            num_node_vec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            args.push_back(i4);
            args.push_back(i5);
            args.push_back(i6);
            return _grid->new_grid_point(args);
        }

    };

    // Aliases for expression types.
    typedef yc_number_node_ptr GridIndex;
    typedef yc_number_node_ptr GridValue;
    typedef yc_bool_node_ptr Condition;
    typedef yc_grid_point_node_ptr GridPointPtr;
    typedef yc_expr_node_ptr ExprPtr;
    typedef yc_number_node_ptr NumExprPtr;
    typedef yc_index_node_ptr IndexExprPtr;
    typedef yc_bool_node_ptr BoolExprPtr;

} // namespace yask.

// Macro for backward compatibility.
#define REGISTER_STENCIL_CONTEXT_EXTENSION(...)

// Convenience macro for declaring an instance of a stencil and registering
// it in the list used by the default provided YASK compiler binary.
#define REGISTER_STENCIL(class_name) static class_name registered_ ## class_name(stub_stencils)

// Convenience macros for declaring dims in a class implementing get_node_factory().
// The 'd' arg is the new var name and the dim name.
#define MAKE_STEP_INDEX(d)   yc_index_node_ptr d = get_node_factory().new_step_index(#d);
#define MAKE_DOMAIN_INDEX(d) yc_index_node_ptr d = get_node_factory().new_domain_index(#d);
#define MAKE_MISC_INDEX(d)   yc_index_node_ptr d = get_node_factory().new_misc_index(#d);

// Convenience macros for creating grids in a class implementing get_solution().
// The 'gvar' arg is the var name and the grid name.
// The remaining args are the dimension names.
#define MAKE_GRID(gvar, ...)                                            \
    Grid gvar = Grid(#gvar, false, get_solution(), { __VA_ARGS__ })
#define MAKE_SCALAR(gvar)    MAKE_GRID(gvar)
#define MAKE_ARRAY(gvar, d1) MAKE_GRID(gvar, d1)
#define MAKE_SCRATCH_GRID(gvar, ...)                                    \
    Grid gvar = Grid(#gvar, true,  get_solution(), { __VA_ARGS__ })
#define MAKE_SCRATCH_SCALAR(gvar)    MAKE_SCRATCH_GRID(gvar)
#define MAKE_SCRATCH_ARRAY(gvar, d1) MAKE_SCRATCH_GRID(gvar, d1)

// Namespaces for stencil code.
#ifndef NO_NAMESPACES
using namespace yask;
using namespace std;
#endif

