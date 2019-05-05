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

///////// API for the YASK stencil compiler. ////////////

// This file uses Doxygen 1.8 markup for API documentation-generation.
// See http://www.stack.nl/~dimitri/doxygen.
/** @file yask_compiler_api.hpp */

#pragma once

#include "yask_common_api.hpp"
#include <vector>

namespace yask {

    /**
     * \defgroup yc YASK Compiler
     * Types, clases, and functions used in the \ref sec_yc.
     * @{
     */

    // Forward declarations of classes and their pointers.
    // See yask_compiler_api.hpp for more.

    class yc_solution;
    /// Shared pointer to \ref yc_solution
    typedef std::shared_ptr<yc_solution> yc_solution_ptr;

    class yc_grid;
    /// Pointer to \ref yc_grid
    typedef yc_grid* yc_grid_ptr;

    // Forward declarations of expression nodes and their pointers.

    class yc_expr_node;
    /// Shared pointer to \ref yc_expr_node
    typedef std::shared_ptr<yc_expr_node> yc_expr_node_ptr;

    class yc_bool_node;
    /// Shared pointer to \ref yc_bool_node
    typedef std::shared_ptr<yc_bool_node> yc_bool_node_ptr;

    class yc_number_node;
    /// Shared pointer to \ref yc_number_node
    typedef std::shared_ptr<yc_number_node> yc_number_node_ptr;

    class yc_index_node;
    /// Shared pointer to \ref yc_index_node
    typedef std::shared_ptr<yc_index_node> yc_index_node_ptr;

    class yc_equation_node;
    /// Shared pointer to \ref yc_equation_node
    typedef std::shared_ptr<yc_equation_node> yc_equation_node_ptr;

    class yc_grid_point_node;
    /// Shared pointer to \ref yc_grid_point_node
    typedef std::shared_ptr<yc_grid_point_node> yc_grid_point_node_ptr;

    /** @}*/
}

#include "yc_node_api.hpp"

namespace yask {

    /**
     * \addtogroup yc
     * @{
     */

    /// Bootstrap factory to create objects needed to define a stencil solution.
    class yc_factory {
    public:
        virtual ~yc_factory() {}

        /// Version information.
        /**
           @returns String describing the current version.
        */
        virtual std::string
		get_version_string();

        /// Create a stencil solution.
        /**
           A stencil solution contains all the grids and equations.
           @returns Pointer to new solution object.
        */
        virtual yc_solution_ptr
        new_solution(const std::string& name /**< [in] Name of the solution;
                                                must be a valid C++ identifier. */ ) const;
    };

    /// Stencil solution.
    /**
       Objects of this type contain all the grids and equations
       that comprise a solution.
    */
    class yc_solution {
    public:
        virtual ~yc_solution() {}

        /// Set object to receive debug output.
        virtual void
        set_debug_output(yask_output_ptr debug
                         /**< [out] Pointer to object to receive debug output.
                            See \ref yask_output_factory. */ ) =0;

        /// Get the name of the solution.
        /**
           @returns String containing the solution name provided via new_solution()
           or set_name().
        */
        virtual std::string
        get_name() const =0;

        /// Set the name of the solution.
        /**
           Allows changing the name from what was provided via new_solution().
        */
        virtual void
        set_name(std::string name
                 /**< [in] Name; must be a valid C++ identifier. */ ) =0;

        /// Get the description of the solution.
        /**
           See set_description().
           @returns String containing the solution description.
        */
        virtual std::string
        get_description() const =0;

        /// Set the description of the solution.
        /**
           By default, the solution description is the same as that
           provided via new_solution() or set_name().
           This allows setting the description to any string.
        */
        virtual void
        set_description(std::string description
                        /**< [in] Any descriptive phrase. */ ) =0;

        /// Get current floating-point precision setting.
        /** @returns Number of bytes in a FP number. */
        virtual int
        get_element_bytes() const =0;

        /// Set floating-point precision.
        virtual void
        set_element_bytes(int nbytes /**< [in] Number of bytes in a FP number.
                                        Should be 4 or 8. */ ) =0;

        /// Create an n-dimensional grid variable in the solution.
        /**
           "Grid" is a generic term for any n-dimensional variable.  A 0-dim
           grid is a scalar, a 1-dim grid is a vector, a 2-dim grid is a
           matrix, etc.

           At least one grid must be defined with a step-index
           node, and it must be the first dimension listed.
           If more than one grid uses a step-index node, the step-indices
           must have the same name across all such grids.

           At least one grid must be defined with at least one domain-index node.

           @returns Pointer to the new \ref yc_grid object.
        */
        virtual yc_grid_ptr
        new_grid(const std::string& name
                 /**< [in] Name of the new grid; must be a valid C++
                    identifier and unique across grids. */,
                 const std::vector<yc_index_node_ptr>& dims
                 /**< [in] Dimensions of the grid.
                    Each dimension is identified by an associated index. */ ) =0;

#ifndef SWIG
        /// Create an n-dimensional grid variable in the solution.
        /**
           C++ initializer-list version with same semantics as
           the vector version of new_grid().
           @note This version is not available (or needed) in the Python API.
           @returns Pointer to the new \ref yc_grid object.
        */
        virtual yc_grid_ptr
        new_grid(const std::string& name /**< [in] Name of the new grid; must be
                                            a valid C++ identifier and unique
                                            across grids. */,
                 const std::initializer_list<yc_index_node_ptr>& dims
                 /**< [in] Dimensions of the grid.
                    Each dimension is identified by an associated index. */ ) =0;
#endif

        /// Create an n-dimensional scratch-grid variable in the solution.
        /**
           A scratch grid is a temporary variable used in the
           definition of a non-scratch grid.
           - Scratch grids are not accessible via kernel APIs.
           Thus, they cannot be programmatically read from or written to.
           - Scratch grid values must be defined from equations ultimately
           referencing only non-scratch grid values, optionally referencing
           other intermediate scratch-grids.
           - Scratch grids cannot use the step-index as a dimension.

           See `TestScratchStencil*` classes in
           `src/stencils/SimpleTestStencils.hpp` for usage examples.

           @returns Pointer to the new \ref yc_grid object.
        */
        virtual yc_grid_ptr
        new_scratch_grid(const std::string& name
                         /**< [in] Name of the new grid; must be a valid C++
                            identifier and unique across grids. */,
                         const std::vector<yc_index_node_ptr>& dims
                         /**< [in] Dimensions of the grid.
                            Each dimension is identified by an associated index. */ ) =0;

#ifndef SWIG
        /// Create an n-dimensional scratch-grid variable in the solution.
        /**
           C++ initializer-list version with same semantics as
           the vector version of new_scratch_grid().
           @note This version is not available (or needed) in the Python API.
           @returns Pointer to the new \ref yc_grid object.
        */
        virtual yc_grid_ptr
        new_scratch_grid(const std::string& name
                         /**< [in] Name of the new grid; must be
                            a valid C++ identifier and unique
                            across grids. */,
                         const std::initializer_list<yc_index_node_ptr>& dims
                         /**< [in] Dimensions of the grid.
                            Each dimension is identified by an associated index. */ ) =0;
#endif

        /// Get the number of grids in the solution.
        /** @returns Number of grids that have been created via new_grid(). */
        virtual int
        get_num_grids() const =0;

        /// Get all the grids in the solution.
        /** @returns Vector containing pointer to all grids. */
        virtual std::vector<yc_grid_ptr>
        get_grids() =0;

        /// Get the specified grid.
        /** @returns Pointer to the specified grid or null pointer if it does not exist. */
        virtual yc_grid_ptr
        get_grid(const std::string& name /**< [in] Name of the grid. */ ) =0;

        /// Set the vectorization length in given dimension.
        /** For YASK-code generation, the product of the fold lengths should
            be equal to the number of elements in a HW SIMD register.
            The number of elements in a HW SIMD register is
            determined by the number of bytes in an element and the print
            format.
            Example: For SP FP elements in AVX-512 vectors, the product of
            the fold lengths should be 16, e.g., x=4 and y=4.
            @note If the product
            of the fold lengths is *not* the number of elements in a HW SIMD
            register, the fold lengths will be adjusted based on an internal
            heuristic. In this heuristic, any fold length that is >1 is
            used as a hint to indicate where to apply folding.
            @note A fold can only be applied in a domain dimension.
            @note Default length is one (1) in each domain dimension. */
        virtual void
        set_fold_len(const yc_index_node_ptr dim
                     /**< [in] Dimension of fold, e.g., "x".
                      This must be an index created by new_domain_index(). */,
                     int len /**< [in] Length of vectorization in `dim` */ ) =0;

        /// Reset all vector-folding settings.
        /** All fold lengths will return to the default of one (1). */
        virtual void
        clear_folding() =0;

        /// Set the cluster multiplier (unroll factor) in given dimension.
        /** For YASK kernel-code generation, this will have the effect of creating
            N vectors of output for each equation, where N is the product of
            the cluster multipliers.

            @note A multiplier >1 cannot be applied to
            the step dimension.
            @note Default is one (1) in each dimension. */
        virtual void
        set_cluster_mult(const yc_index_node_ptr dim
                         /**< [in] Direction of unroll, e.g., "y".
                            This must be an index created by new_domain_index().  */,
                         int mult /**< [in] Number of vectors in `dim` */ ) =0;

        /// Reset all vector-clustering settings.
        /** All cluster multipliers will return to the default of one (1). */
        virtual void
        clear_clustering() =0;

        /// Get the number of equations in the solution.
        /** Equations are added when yc_node_factory::new_equation_node() is called.
            @returns Number of equations that have been created. */
        virtual int
        get_num_equations() const =0;

        /// Get a list of all the defined equations.
        /** @returns Vector of containing pointers to all
            equations that have been created. */
        virtual std::vector<yc_equation_node_ptr>
        get_equations() =0;

        /// Format the current equation(s) and write to given output object.
        /** Currently supported format types:
            Type    | Output
            --------|--------
            cpp     | YASK stencil classes for generic C++.
            avx     | YASK stencil classes for CORE AVX ISA.
            avx2    | YASK stencil classes for CORE AVX2 ISA.
            avx512  | YASK stencil classes for CORE AVX-512 & MIC AVX-512 ISAs.
            knc     | YASK stencil classes for Knights Corner ISA.
            dot     | DOT-language description.
            dot-lite| DOT-language description of grid accesses only.
            pseudo  | Human-readable pseudo-code (for debug).
            pseudo-long  | Human-readable pseudo-code with intermediate variables.

            Progress text will be written to the output stream set via set_debug_output().

            @note "avx512f" is allowed as an alias for "avx512".
            @warning *Side effect:* Applies optimizations to the equation(s), so some pointers
            to nodes in the original equations may refer to modified nodes or nodes
            that have been optimized away after calling format().
         */
        virtual void
        format(const std::string& format_type
               /**< [in] Name of type from above table. */,
               yask_output_ptr output
               /**< [out] Pointer to object to receive formatted output.
                  See \ref yask_output_factory. */) =0;

        /// **[Advanced]** Explicitly list the domain dimensions in the solution.
        /**
           In addition, domain dimension(s) are added when new_grid() or
           new_scratch_grid() is called with one or more domain dimensions.
           Either way, the last unique domain dimension specified will become the 'inner' or
           'unit-stride' dimension in memory layouts.
           Thus, this option can be used to override the default layout order.
           It also allows specification of the domain dimensions in the 
           unusual case where a solution is defined without any
           grids containing all of the domain dimensions.
         */
        virtual void
        set_domain_dims(const std::vector<yc_index_node_ptr>& dims
                        /**< [in] Domain dimensions of the solution. */ ) =0;

#ifndef SWIG
        /// **[Advanced]** Explicitly list the domain dimensions in the solution.
        /**
           C++ initializer-list version with same semantics as
           the vector version of new_grid().
           @note This version is not available (or needed) in the Python API.
        */
        virtual void
        set_domain_dims(const std::initializer_list<yc_index_node_ptr>& dims
                        /**< [in] Domain dimensions of the solution. */ ) =0;
#endif
        
        /// **[Advanced]** Explicitly identify the step dimension in the solution.
        /** 
            By default, the step dimension is defined when new_grid() or
            new_scratch_grid() is called with the step dimension.
            This allows specification of the step dimension in the 
            unusual case where a solution is defined without any
            grids containing the step dimension.
         */
        virtual void
        set_step_dim(const yc_index_node_ptr dim
                     /**< [in] Step dimension. */) =0;

        /// **[Advanced]** Enable or disable automatic dependency checker.
        /**
           Disabling the built-in dependency checker may be done when it is
           overly conservative. Currently, the provided checker does not
           allow stencils in which points in one sub-domain depend on points
           in another sub-domain within the same value of the step index.

           @warning If dependency checker is disabled, *all* dependencies
           must be set via add_flow_dependency().
         */
        virtual void
        set_dependency_checker_enabled(bool enable
                                       /**< [in] `true` to enable or `false` to disable. */) =0;

        /// **[Advanced]** Determine whether automatic dependency checker is enabled.
        /**
           @returns Current setting.
        */
        virtual bool
        is_dependency_checker_enabled() const =0;

        /// **[Advanced]** Add a dependency between two equations.
        /**
           This function adds an arc in the data-dependency graph `from` one
           equation (node) `to` another one,
           indicating that the `from` equation depends on the `to` equation.
           In other words, the `to` expression must be evaluated _before_
           the `from` equation.
           In compiler-theory terms, this is a _flow_ dependency, also
           known as a _true_ or _read-after-write_ (RAW) dependency.
           (Strictly speaking, however, equations in the YASK compiler
           are declarative instead of imperative, so they describe
           equalities rather than assignments with reads and writes.
           On the other hand, a C++ function created to implement
           one or more equations will perform analogous reads and writes.)

           Additional considerations:

           - It is not necessary to connect all the equations into a single
           graph.  For example, if **A** depends on **B** and **C** depends
           on **D**, there will be two disconnected subgraphs.  In this
           example, the YASK kernel is free to 1) schedule the functions created for
           **B** and **D** to run together in parallel followed by those for
           **A** and **C** together in parallel, 2) run a single
           function that implements both **B** and **D** simultaneously
           followed by a single
           function that implements both **A** and **C** simultaneously,
           or 3) a combination of the implementations.

           - Only _immediate_ dependencies should be added.
           In other words, each subgraph created should be a transitive reduction.
           For example, if **A** depends on **B** and **B** depends on **C**,
           it is not necessary to add the transitive dependence from **A** to **C**.

           - Only dependencies at a given step-index value should
           be added.
           For example, given
           equation **A**: `A(t+1, x) EQUALS B(t+1, x) + 5` and
           equation **B**: `B(t+1, x) EQUALS A(t, x) / 2`,
           **A** depends on **B** at some value of the step-index `t`.
           That dependency should be added if the automatic checker is disabled.
           (It is true that the next value of `B(t+2)` depends on `A(t+1)`,
           but such inter-step -- analgous to loop-carried --
           dependencies should _not_ be added with this function.)

           - The dependencies should create one or more directed
           acyclic graphs (DAGs).
           If a cycle is created, the YASK compiler
           will throw an exception containing an error message
           about a circular dependency. This exception may not be
           thrown until format() is called.

           - If using scratch grids, dependencies among scratch grids
           and between scratch-grid equations and non-scratch-grid
           equations should also be added. Each scratch grid equation
           should ultimately depend on non-scratch-grid values.

           - This function can be used in cooperation with or instead of
           the built-in automatic dependency checker.
           When used in cooperation with the built-in checker,
           both dependencies from this function and the built-in checker
           will be considered.
           When the built-in checker is diabled via
           `set_dependency_checker_enabled(false)`, only dependencies
           from this function will be considered.
           In this case, it is imperative that all immediate
           dependencies are added.
           If the dependency graph is incomplete, the resulting generated
           stencil code will contain illegal race conditions,
           and it will most likely produce incorrect results.
        */
        virtual void
        add_flow_dependency(yc_equation_node_ptr from
                            /**< [in] Equation that must be evaluated _after_ `to`. */,
                            yc_equation_node_ptr to
                            /**< [in] Equation that must be evaluated _before_ `from`. */) =0;

        /// **[Advanced]** Remove all existing dependencies.
        /**
           Removes dependencies added via add_flow_dependency().
         */
        virtual void
        clear_dependencies() =0;
    };

    /// A compile-time data variable.
    /** "Grid" is a generic term for any n-dimensional array.  A 0-dim grid
        is a scalar, a 1-dim grid is an array, etc.
        A compile-time grid is a variable used for constructing equations.
        It does not contain any data.
        Data is only stored during run-time, using a \ref yk_grid.
        Created via yc_solution::new_grid(). */
    class yc_grid {
    public:
        virtual ~yc_grid() {}

        /// Get the name of the grid.
        /** @returns String containing name provided via new_grid(). */
        virtual const std::string& get_name() const =0;

        /// Get the number of dimensions.
        /** @returns Number of dimensions created via new_grid(). */
        virtual int get_num_dims() const =0;

        /// Get all the dimensions in this grid.
        /**
           Includes step dimension if it is a dimension of this grid.
           May be different than values returned from yc_solution::get_domain_dim_names().
           @returns List of names of all the dimensions.
        */
        virtual std::vector<std::string>
        get_dim_names() const =0;

        /// Create a reference to a point in this grid.
        /**
           Each expression in `index_exprs` describes how to access
           an element in the corresponding dimension of the grid.

           Example: if a grid was created via
           `g = new_grid("A", {t, x, y, n})` with step-dimension `t`,
           domain-dimensions `x` and `y`, and misc-dimension `n`,
           `g->new_grid_point({t + 1, x - 1, y + 1, 2})` refers to the specified
           element for the values of `t`, `x`, and `y` set dynamically
           during stencil evaluation and the constant value `2` for `n`.

           @returns Pointer to AST node used to read from or write to point in grid. */
        virtual yc_grid_point_node_ptr
        new_grid_point(const std::vector<yc_number_node_ptr>& index_exprs
                       /**< [in] Index expressions.
                          These must appear in the same order as when the
                          grid was created. */ ) =0;

#ifndef SWIG
        /// Create a reference to a point in this grid.
        /**
           C++ initializer-list version with same semantics as
           the vector version of new_grid_point().
           @note This version is not available (or needed) in the Python API.
           @returns Pointer to AST node used to read or write from point in grid. */
        virtual yc_grid_point_node_ptr
        new_grid_point(const std::initializer_list<yc_number_node_ptr>& index_exprs) = 0;
#endif

        /// Create a reference to a point in this grid using relative offsets.
        /**
           A shorthand function for calling new_grid_point() when
           all index expressions are constant offsets.
           Each offset refers to the dimensions defined when the
           grid was created via stencil_solution::new_grid().

           Example: if `g = new_grid("data", {t, x, y})` with step-dimension `t`
           and domain-dimensions `x` and `y`,
           `g->new_relative_grid_point({1, -1, 0})` refers to the same point as
           `g->new_grid_point({t + 1, x - 1, y})`.

           @warning This convenience function can only be used when every
           dimension of the grid is either the step dimension or a domain dimension.
           If this is not the case, use new_grid_point().
           @returns Pointer to AST node used to read from or write to point in grid. */
        virtual yc_grid_point_node_ptr
        new_relative_grid_point(const std::vector<int>& dim_offsets
                                /**< [in] offset from evaluation index in each dim. */ ) =0;

#ifndef SWIG
        /// Create a reference to a point in this grid using relative offsets.
        /**
           C++ initializer-list version with same semantics as
           the vector version of new_relative_grid_point().
           @note This version is not available (or needed) in the Python API.
           @returns Pointer to AST node used to read or write from point in grid. */
        virtual yc_grid_point_node_ptr
        new_relative_grid_point(const std::initializer_list<int>& dim_offsets) = 0;
#endif

        /// **[Advanced]** Get whether the allocation of the step dimension of this grid can be modified at run-time.
        /**
           See yk_grid::set_alloc_size().
         */
        virtual bool
        is_dynamic_step_alloc() const =0;

        /// **[Advanced]** Set whether the allocation of the step dimension of this grid can be modified at run-time.
        /**
           See yk_grid::set_alloc_size().
         */
        virtual void
        set_dynamic_step_alloc(bool is_dynamic
                               /**< [in] `true` to enable or `false` to disable. */) =0;

        /// **[Advanced]** Get the current allocation in the step dimension of this grid.
        /**
           If set_step_alloc_size() has been called, that setting will be returned.
           If set_step_alloc_size() has not been called, the default setting determined
           by the YASK compiler will be returned.
           @returns Allocation in the step dimension.
        */
        virtual idx_t
        get_step_alloc_size() const =0;

        /// **[Advanced]** Set the current allocation in the step dimension of this grid.
        /**
           Override the default setting determined
           by the YASK compiler for allocation in the step dimension.
        */
        virtual void
        set_step_alloc_size(idx_t size
                            /**< [in] Number of elements to allocate in the step dimension. */) =0;
    };

    /// A wrapper class around a 'yc_grid_ptr', providing
    /// convenience functions for declaring grid vars and
    /// creating expression nodes with references to points
    /// in grid vars.
    class yc_grid_var {
    private:
        yc_grid_ptr _grid;
        
    public:

        /// Contructor taking a vector of index vars.
        yc_grid_var(const std::string& name
                    /**< [in] Name of the new grid; must be a valid C++
                       identifier and unique across grids. */,
                    yc_solution_ptr soln
                    /**< [in] Pointer to solution that will own the grid. */,
                    const std::vector< yc_index_node_ptr > &dims
                    /**< [in] Dimensions of the grid.
                       Each dimension is identified by an associated index. */,
                    bool is_scratch = false
                    /**< [in] Whether to make a scratch grid. */) {
            if (is_scratch)
                _grid = soln->new_scratch_grid(name, dims);
            else
                _grid = soln->new_grid(name, dims);
        }

        // Contructor taking an initializer_list of index vars.
        yc_grid_var(const std::string& name
                    /**< [in] Name of the new grid; must be a valid C++
                       identifier and unique across grids. */,
                    yc_solution_ptr soln
                    /**< [in] Pointer to solution that will own the grid. */,
                    const std::initializer_list< yc_index_node_ptr > &dims
                    /**< [in] Dimensions of the grid.
                       Each dimension is identified by an associated index. */,
                    bool is_scratch = false
                    /**< [in] Whether to make a scratch grid. */) {
            if (is_scratch)
                _grid = soln->new_scratch_grid(name, dims);
            else
                _grid = soln->new_grid(name, dims);
        }

        /// Provide a virtual destructor.
        virtual ~yc_grid_var() { }

        /// Get the underlying \ref yc_grid_ptr.
        virtual yc_grid_ptr get_grid() {
            return _grid;
        }

        /// Get the underlying \ref yc_grid_ptr.
        virtual const yc_grid_ptr get_grid() const {
            return _grid;
        }

#ifndef SWIG
        /// Create an expression for a point in a zero-dim (scalar) grid
        /// using implicit conversion.
        /**
           Example w/0D grid var `B`: `A(t+1, x) EQUALS A(t, x) + B`.
           @note Not available in Python API. Use more explicit methods.
        */
        virtual operator yc_number_ptr_arg() {
            return _grid->new_grid_point({});
        }

        /// Create an expression for a point in a one-dim (array) grid.
        /**
           Example w/1D grid var `B`: `A(t+1, x) EQUALS A(t, x) + B[x]`.
           @note Not available in Python API. Use vector version.
        */
        virtual yc_grid_point_node_ptr operator[](const yc_number_any_arg i1) {
            return _grid->new_grid_point({i1});
        }
        
        /// Create an expression for a point in a 1-6 dim grid.
        /// The number of arguments must match the dimensionality of the grid.
        /**
           Example w/2D grid var `B`: `A(t+1, x) EQUALS A(t, x) + B(x, 3)`.
           @note Not available in Python API. Use vector version.
        */
        virtual yc_grid_point_node_ptr operator()(const yc_number_any_arg i1 = nullptr,
                                                  const yc_number_any_arg i2 = nullptr,
                                                  const yc_number_any_arg i3 = nullptr,
                                                  const yc_number_any_arg i4 = nullptr,
                                                  const yc_number_any_arg i5 = nullptr,
                                                  const yc_number_any_arg i6 = nullptr) {
            std::vector<yc_number_node_ptr> args;
            if (i1)
                args.push_back(i1);
            if (i2)
                args.push_back(i2);
            if (i3)
                args.push_back(i3);
            if (i4)
                args.push_back(i4);
            if (i5)
                args.push_back(i5);
            if (i6)
                args.push_back(i6);
            return _grid->new_grid_point(args);
        }

        /// Create an expression for a point in a grid.
        /// The number of arguments must match the dimensionality of the grid.
        /**
           Example w/2D grid var `B`: `A(t+1, x) EQUALS A(t, x) + B({x, 3})`.
           @note Not available in Python API. Use vector version.
        */
        virtual yc_grid_point_node_ptr
        operator()(const std::initializer_list<yc_number_node_ptr>& index_exprs) {
            return _grid->new_grid_point(index_exprs);
        }
#endif

        /// Create an expression for a point in a grid.
        /// The number of arguments must match the dimensionality of the grid.
        /// Create an expression for a point in a 1-6 dim grid.
        /// The number of arguments must match the dimensionality of the grid.
        /**
           Example w/2D grid var `B`: `A(t+1, x) EQUALS A(t, x) + B(vec)`,
           where `vec` is a 2-element vector of \ref yc_number_node_ptr.
        */
        virtual yc_grid_point_node_ptr
        operator()(const std::vector<yc_number_node_ptr>& index_exprs) {
            return _grid->new_grid_point(index_exprs);
        }
    };
    /** @}*/

} // namespace yask.

