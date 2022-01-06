/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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
#include <functional>
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

    class yc_var;
    /// Pointer to \ref yc_var
    typedef yc_var* yc_var_ptr;

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

    class yc_var_point_node;
    /// Shared pointer to \ref yc_var_point_node
    typedef std::shared_ptr<yc_var_point_node> yc_var_point_node_ptr;

    /** @}*/
}

#include "aux/yc_node_api.hpp"

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
           A stencil solution contains all the vars and equations.
           @returns Pointer to new solution object.
        */
        virtual yc_solution_ptr
        new_solution(const std::string& name /**< [in] Name of the solution;
                                                must be a valid C++ identifier. */ ) const;
    }; // yc_factory.

    /// Stencil solution.
    /**
       Objects of this type contain all the vars and equations
       that comprise a solution.
       Must be created via yc_factory::new_solution().
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

        /// Get the current output-file format.
        /**
           @returns Current target.

           Throws an exception if the target hasn't been set via set_target().
        */
        virtual std::string
        get_target() =0;

        /// Set the output target.
        /**
           Currently supported targets:
            Type    | Output
            --------|--------
            intel64 | YASK kernel for generic 64-bit C++.
            avx     | YASK kernel for CORE AVX ISA.
            avx2    | YASK kernel for CORE AVX2 ISA.
            avx512  | YASK kernel for CORE AVX-512 ISA.
            avx512-ymm | YASK kernel for CORE AVX-512 ISA with 256-bit SIMD.
            knl     | YASK kernel for MIC AVX-512 ISA.
            knc     | YASK kernel for Knights Corner ISA.
            dot     | DOT-language description.
            dot-lite| DOT-language description of var accesses only.
            pseudo  | Human-readable pseudo-code (for debug).
            pseudo-long  | Human-readable pseudo-code with intermediate variables.
        */
        virtual void
        set_target(/** [in] Output-file format from above list */
                   const std::string& format) =0;

        /// Determine whether target has been set.
        /**
           @returns `true` if set_target() has been called with a valid format;
           `false` otherwise.
        */
        virtual bool
        is_target_set() =0;

        /// Get current floating-point precision setting.
        /** @returns Number of bytes in a FP number. */
        virtual int
        get_element_bytes() const =0;

        /// Set floating-point precision.
        virtual void
        set_element_bytes(int nbytes /**< [in] Number of bytes in a FP number.
                                        Should be 4 or 8. */ ) =0;

        /// Create an n-dimensional variable in the solution.
        /**
           "Var" is a generic term for any n-dimensional variable.  A 0-dim
           var is a scalar, a 1-dim var is a vector, a 2-dim var is a
           matrix, etc.

           The dimensions of a variable are defined by providing a list
           of indices created via yc_node_factory::new_step_index(),
           yc_node_factory::new_domain_index(), and/or 
           yc_node_factory::new_misc_index().
           When a step index is used, it must be the first index.
           If more than one var uses a step-index, the step-indices
           must have the same name. For example, you cannot have
           one var with step-index "t" and one with step-index "time".

           Example code to create a solution with an equation for a variable named "A":
           ~~~{.cpp}
           yc_factory ycfac;
           yc_node_factory nfac;
           auto my_soln = ycfac.new_solution("my_stencil");
           auto t = nfac.new_step_index("t");
           auto x = nfac.new_domain_index("x");
           auto y = nfac.new_domain_index("y");
           auto a = ycfac.new_var("A", { t, x, y }); 
           a->new_var_point({t+1, x, y}) EQUALS (a->new_var_point({t, x, y}) +
                                                  a->new_var_point({t, x+1, y}) + 
                                                  a->new_var_point({t, x, y+1})) * (1.0/3.0);
           ~~~

           @returns Pointer to the new \ref yc_var object.
        */
        virtual yc_var_ptr
        new_var(const std::string& name
                 /**< [in] Name of the new var; must be a valid C++
                    identifier and unique across vars. */,
                 const std::vector<yc_index_node_ptr>& dims
                 /**< [in] Dimensions of the var.
                    Each dimension is identified by an associated index. */ ) =0;

#ifndef SWIG
        /// Create an n-dimensional variable in the solution.
        /**
           C++ initializer-list version with same semantics as
           the vector version of new_var().
           @note Not available in the Python API. Use the vector version.
           @returns Pointer to the new \ref yc_var object.
        */
        virtual yc_var_ptr
        new_var(const std::string& name /**< [in] Name of the new var; must be
                                            a valid C++ identifier and unique
                                            across vars. */,
                 const std::initializer_list<yc_index_node_ptr>& dims
                 /**< [in] Dimensions of the var.
                    Each dimension is identified by an associated index. */ ) =0;
#endif

        /// Create an n-dimensional scratch variable in the solution.
        /**
           A scratch variable is a temporary variable used as an 
           intermediate value in an equation.
           - Scratch vars are not accessible via kernel APIs.
           Thus, they cannot be programmatically read from or written to.
           - Scratch var values must be defined from equations ultimately
           referencing only non-scratch var values, optionally referencing
           other intermediate scratch-vars.
           - Scratch vars cannot use the step-index as a dimension.

           See `TestScratchStencil*` classes in
           `src/stencils/SimpleTestStencils.hpp` for usage examples.

           @returns Pointer to the new \ref yc_var object.
        */
        virtual yc_var_ptr
        new_scratch_var(const std::string& name
                         /**< [in] Name of the new var; must be a valid C++
                            identifier and unique across vars. */,
                         const std::vector<yc_index_node_ptr>& dims
                         /**< [in] Dimensions of the var.
                            Each dimension is identified by an associated index. */ ) =0;

#ifndef SWIG
        /// Create an n-dimensional scratch variable in the solution.
        /**
           C++ initializer-list version with same semantics as
           the vector version of new_scratch_var().
           @note Not available in the Python API. Use the vector version.
           @returns Pointer to the new \ref yc_var object.
        */
        virtual yc_var_ptr
        new_scratch_var(const std::string& name
                         /**< [in] Name of the new var; must be
                            a valid C++ identifier and unique
                            across vars. */,
                         const std::initializer_list<yc_index_node_ptr>& dims
                         /**< [in] Dimensions of the var.
                            Each dimension is identified by an associated index. */ ) =0;
#endif

        /// Get the number of vars in the solution.
        /** 
            @returns Number of vars that have been created via new_var() or
            new_scratch_var(). */
        virtual int
        get_num_vars() const =0;

        /// Get all the vars in the solution.
        /** @returns Vector containing pointer to all vars. */
        virtual std::vector<yc_var_ptr>
        get_vars() =0;

        /// Get the specified var.
        /** @returns Pointer to the specified var or null pointer if it does not exist. */
        virtual yc_var_ptr
        get_var(const std::string& name /**< [in] Name of the var. */ ) =0;

        /// Set the vectorization length in given dimension.
        /** For YASK-code generation, the product of the fold lengths should
            be equal to the number of elements in a HW SIMD register.
            The number of elements in a HW SIMD register is
            determined by the number of bytes in an element and the print
            format.

            Example: For SP FP elements in AVX-512 vectors, the product of
            the fold lengths should be 16, e.g., x=4 and y=4.

            If the product of the fold lengths is *not* the number of
            elements in a HW SIMD register, the fold lengths will be
            adjusted based on an internal heuristic. In this heuristic, any
            specified fold length is used as a hint to determine the final
            folding.
        */
        virtual void
        set_fold_len(const yc_index_node_ptr dim
                     /**< [in] Dimension of fold, e.g., "x".
                      This must be an index created by new_domain_index(). */,
                     int len /**< [in] Length of vectorization in `dim` */ ) =0;

        /// Determine whether any folding has been set.
        /**
           @returns `true` if any fold length has been specified;
           `false` if not.
        */
        virtual bool
        is_folding_set() =0;
        
        /// Remove all vector-folding settings.
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

        /// Determine whether any clustering has been set.
        /**
           @returns `true` if any cluster multiple has been specified;
           `false` if not.
        */
        virtual bool
        is_clustering_set() =0;
        
        /// Remove all vector-clustering settings.
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

        /// Get the current prefetch distance for the given cache.
        /**
           @returns Prefetch distance in number of iterations
           or zero (0) if disabled.
        */
        virtual int
        get_prefetch_dist(/** [in] Cache level: 1 or 2. */
                          int level) =0;

        /// Set the prefetch distance for the given cache.
        virtual void
        set_prefetch_dist(/** [in] Cache level: 1 or 2. */
                          int level,
                          /** [in] Number of iterations ahead to prefetch data
                              or zero (0) to disable. */
                          int distance) =0;
        
        /// Optimize and the current equation(s) and write to given output object.
        /** 
            Output will be formatted according to set_target() and all other preceding
            YASK compiler API calls.

            Progress text will be written to the output stream set via set_debug_output().

            @warning *Side effect:* Applies optimizations to the equation(s), so some pointers
            to nodes in the original equations may refer to modified nodes or nodes
            that have been optimized away after calling format(). In general, do not use
            pointers to nodes across calls to format().
         */
        virtual void
        output_solution(yask_output_ptr output
                        /**< [out] Pointer to object to receive formatted output.
                           See \ref yask_output_factory. */) =0;

#ifndef SWIG
        /// **[Advanced]** Callback type for call_before_output().
        typedef std::function<void(yc_solution& soln,
                                   yask_output_ptr output)> output_hook_t;

        /// **[Advanced]** Register a function to be called before a solution is output.
        /**
           The registered functions will be called during a call to output_solution()
           after the equations optimizations have been applied but before the
           output is written.

           A reference to the solution and the parameter to output_solution() are passed to the `hook_fn`.

           If this method is called more than once, the hook functions will be
           called in the order registered.

           @note Not available in the Python API.
        */
        virtual void
        call_before_output(/** [in] callback function */
                           output_hook_t hook_fn) =0;
#endif

        /// **[Advanced]** Add block of custom C++ code to the kernel solution.
        /**
           This block of code will be executed immediately after the stencil solution 
           is constructed in the kernel, i.e., at the end of a call to
           yk_factory::new_solution().
           The code may access the new solution via the reference
           `kernel_soln` of type \ref yk_solution.

           Common uses of this facility include setting default run-time settings
           such as block sizes and registering call-back routines, e.g., via
           yk_solution::call_before_prepare_solution();

           Unlike yk_solution::call_before_prepare_solution() and similar functions
           which have `std::function` parameters,
           the parameter to this function is a string because the code is not compiled
           (or compilable) until the kernel library is built.

           Alternatively, equivalent code can be added directly to any
           custom application using the kernel library APIs.  However, this
           function is useful when using the provided YASK
           kernel-performance utility launched via `bin/yask.sh`.  It also
           provides a method to provide consistent kernel code when the
           kernel library is used in multiple applications.
         */
        virtual void
        call_after_new_solution(const std::string& code
                                /**< [in] Code to be inserted, using
                                   `kernel_soln` of type \ref yk_solution
                                   to access the kernel solution. */) =0;

        /// **[Advanced]** A convenience macro for calling yask::yc_solution::call_after_new_solution().
        /**
           Allows writing the code without the surrounding quotes, making it easier
           to format in many editors and IDEs (and perhaps look somewhat like a lambda function).
        */
#define CALL_AFTER_NEW_SOLUTION(...) call_after_new_solution(#__VA_ARGS__)
        
        /// **[Advanced]** Explicitly define and order the domain dimensions used in the solution.
        /**
           The order of domain dimensions affects memory layout, looping
           order, vector-folding, and rank layout.  This API also allows
           specification of the domain dimensions in the unusual case where
           a solution is defined without any vars containing all of the
           domain dimensions.  Whether or not this API is called, domain
           dimension(s) are added when new_var() or new_scratch_var() is
           called with one or more domain dimensions not previously seen.
         */
        virtual void
        set_domain_dims(const std::vector<yc_index_node_ptr>& dims
                        /**< [in] Domain dimensions of the solution. */ ) =0;

#ifndef SWIG
        /// **[Advanced]** Explicitly define and order the domain dimensions used in the solution.
        /**
           C++ initializer-list version with same semantics as
           the vector version of new_var().
           @note Not available in the Python API. Use the vector version.
        */
        virtual void
        set_domain_dims(const std::initializer_list<yc_index_node_ptr>& dims
                        /**< [in] Domain dimensions of the solution. */ ) =0;
#endif
        
        /// **[Advanced]** Explicitly identify the step dimension in the solution.
        /** 
            By default, the step dimension is defined when new_var()
            is called with a step index.
            This API allows specification of the step dimension in the 
            unusual case where a solution is defined without any
            vars containing the step dimension.
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

           - If using scratch vars, dependencies among scratch vars
           and between scratch equations and non-scratch
           equations should also be added. Each scratch equation
           should ultimately depend on non-scratch values.

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

        /// **[Deprecated]** Use set_target() and output_solution().
        YASK_DEPRECATED
        inline void
        format(const std::string& format_type,
               yask_output_ptr output) {
            set_target(format_type);
            output_solution(output);
        }

        /// **[Deprecated]** Use new_var().
        YASK_DEPRECATED
        inline yc_var_ptr
        new_grid(const std::string& name,
                 const std::vector<yc_index_node_ptr>& dims) {
            return new_var(name, dims);
        }

#ifndef SWIG
        /// **[Deprecated]** Use new_var().
         YASK_DEPRECATED
       inline yc_var_ptr
        new_grid(const std::string& name,
                 const std::initializer_list<yc_index_node_ptr>& dims) {
            return new_var(name, dims);
        }
#endif

        /// **[Deprecated]** Use new_scratch_var().
        YASK_DEPRECATED
        inline yc_var_ptr
        new_scratch_grid(const std::string& name,
                         const std::vector<yc_index_node_ptr>& dims) {
            return new_scratch_var(name, dims);
        }

#ifndef SWIG
        /// **[Deprecated]** Use new_scratch_var().
        YASK_DEPRECATED
        inline yc_var_ptr
        new_scratch_grid(const std::string& name,
                         const std::initializer_list<yc_index_node_ptr>& dims) {
            return new_scratch_var(name, dims);
        }
#endif

        /// **[Deprecated]** Use get_num_vars().
        YASK_DEPRECATED
        inline int
        get_num_grids() const {
            return get_num_vars();
        }

        /// **[Deprecated]** Use get_vars().
        YASK_DEPRECATED
        inline std::vector<yc_var_ptr>
        get_grids() {
            return get_vars();
        }

        /// **[Deprecated]** Use get_var().
        YASK_DEPRECATED
        inline yc_var_ptr
        get_grid(const std::string& name) {
            return get_var(name);
        }
    };                          // yc_solution.

    /// A compile-time data variable.
    /** "Var" is a generic term for any n-dimensional variable.  A 0-dim var
        is a scalar, a 1-dim var is an array, etc.
        A compile-time variable is used for constructing stencil equations.
        It does not contain any data.
        Data is only stored during run-time, using a \ref yk_var.

        Created via yc_solution::new_var() or yc_solution::new_scratch_var()
        or implicitly via the \ref yc_var_proxy constructor.
    */
    class yc_var {
    public:
        virtual ~yc_var() {}

        /// Get the name of the var.
        /** @returns String containing name provided via
            yc_solution::new_var() or yc_solution::new_scratch_var(). */
        virtual const std::string& get_name() const =0;

        /// Get the number of dimensions.
        /** @returns Number of dimensions created via 
            yc_solution::new_var() or yc_solution::new_scratch_var(). */
        virtual int get_num_dims() const =0;

        /// Get all the dimensions in this var.
        /**
           This is not necessarily a list of all the dimensions used
           in the \ref yc_solution.
           @returns List of names of all the dimensions used in this var.
        */
        virtual std::vector<std::string>
        get_dim_names() const =0;

        /// Create a reference to a point in this var.
        /**
           Each expression in `index_exprs` describes how to access
           an element in the corresponding dimension of the var.

           @returns Pointer to AST node used to read from or write to point in var. */
        virtual yc_var_point_node_ptr
        new_var_point(const std::vector<yc_number_node_ptr>& index_exprs
                       /**< [in] Index expressions.
                          These must appear in the same order as when the
                          var was created. */ ) =0;

#ifndef SWIG
        /// Create a reference to a point in this var.
        /**
           C++ initializer-list version with same semantics as
           the vector version of new_var_point().

           See example code shown in yc_solution::new_var().

           @note Not available in the Python API. Use the vector version.
           @returns Pointer to AST node used to read or write from point in var.
        */
        virtual yc_var_point_node_ptr
        new_var_point(const std::initializer_list<yc_number_node_ptr>& index_exprs) = 0;
#endif

        /// Create a reference to a point in this var using relative offsets.
        /**
           A shorthand function for calling new_var_point() when
           all index expressions are constant offsets.
           Each offset refers to the dimensions defined when the
           var was created via yc_solution::new_var().

           Example: if `g = new_var("data", {t, x, y})` with step-dimension `t`
           and domain-dimensions `x` and `y`,
           `g->new_relative_var_point({1, -1, 0})` refers to the same point as
           `g->new_var_point({t + 1, x - 1, y})`.

           @warning This convenience function can only be used when every
           dimension of the var is either the step dimension or a domain dimension.
           If this is not the case, use new_var_point().
           @returns Pointer to AST node used to read from or write to point in var. */
        virtual yc_var_point_node_ptr
        new_relative_var_point(const std::vector<int>& dim_offsets
                                /**< [in] offset from evaluation index in each dim. */ ) =0;

#ifndef SWIG
        /// Create a reference to a point in this var using relative offsets.
        /**
           C++ initializer-list version with same semantics as
           the vector version of new_relative_var_point().
           @note Not available in the Python API. Use the vector version.
           @returns Pointer to AST node used to read or write from point in var. */
        virtual yc_var_point_node_ptr
        new_relative_var_point(const std::initializer_list<int>& dim_offsets) = 0;
#endif

        /// **[Advanced]** Get whether the allocation of the step dimension of this var can be modified at run-time.
        /**
           See yk_var::set_alloc_size().
         */
        virtual bool
        is_dynamic_step_alloc() const =0;

        /// **[Advanced]** Set whether the allocation of the step dimension of this var can be modified at run-time.
        /**
           See yk_var::set_alloc_size().
         */
        virtual void
        set_dynamic_step_alloc(bool is_dynamic
                               /**< [in] `true` to enable or `false` to disable. */) =0;

        /// **[Advanced]** Get the current allocation in the step dimension of this var.
        /**
           If set_step_alloc_size() has been called, that setting will be returned.
           If set_step_alloc_size() has not been called, the default setting determined
           by the YASK compiler will be returned.
           @returns Allocation in the step dimension.
        */
        virtual idx_t
        get_step_alloc_size() const =0;

        /// **[Advanced]** Set the current allocation in the step dimension of this var.
        /**
           Override the default setting determined
           by the YASK compiler for allocation in the step dimension.
        */
        virtual void
        set_step_alloc_size(idx_t size
                            /**< [in] Number of elements to allocate in the step dimension. */) =0;

        /// **[Deprecated]** Use new_var_point().
        YASK_DEPRECATED
        inline yc_var_point_node_ptr
        new_grid_point(const std::vector<yc_number_node_ptr>& index_exprs) {
            return new_var_point(index_exprs);
        }
        /// **[Deprecated]** Use new_var_point().
        YASK_DEPRECATED
        inline yc_var_point_node_ptr
        new_grid_point(const std::initializer_list<yc_number_node_ptr>& index_exprs) {
            return new_var_point(index_exprs);
        }
        /// **[Deprecated]** Use new_relative_var_point().
        YASK_DEPRECATED
        inline yc_var_point_node_ptr
        new_relative_grid_point(const std::vector<int>& dim_offsets) {
            return new_relative_var_point(dim_offsets);
        }
        /// **[Deprecated]** Use new_relative_var_point().
        YASK_DEPRECATED
        inline yc_var_point_node_ptr
        new_relative_grid_point(const std::initializer_list<int>& dim_offsets) {
            return new_relative_var_point(dim_offsets);
        }
        
    };                      // yc_var.

    /// A wrapper or "proxy" class around a \ref yc_var pointer.
    /**
       Using this class provides a syntactic alternative to calling yc_solution::new_var()
       (or yc_solution::new_scratch_var()) followed by yc_var::new_var_point().

       To use this wrapper class, construct an object of type \ref yc_var_proxy by
       passing a \ref yc_solution pointer to it.
       Then, expressions for points in the var can be created with a more
       intuitive syntax.

       Example code to create a solution with an equation for a variable named "A":
       ~~~{.cpp}
       yc_factory ycfac;
       yc_node_factory nfac;
       auto my_soln = ycfac.new_solution("my_stencil");
       auto t = nfac.new_step_index("t");
       auto x = nfac.new_domain_index("x");
       auto y = nfac.new_domain_index("y");
       yc_var_proxy a("A", my_soln, { t, x, y });
       a({t+1, x, y}) EQUALS (a({t, x, y}) + 
                              a({t, x+1, y}) + 
                              a({t, x, y+1})) * (1.0/3.0);
       ~~~
       Compare to the example shown in yc_solution::new_var().
       
       _Scoping and lifetime:_ Since the \ref yc_var pointer in a \ref
       yc_var_proxy object is a shared pointer also owned by the \ref
       yc_solution object used to construct the \ref yc_var_proxy object, the
       underlying YASK var will not be destroyed until both the \ref yc_var_proxy
       object and the \ref yc_solution object are destroyed.
       A \ref yc_var_proxy object created from an existing \ref yc_var
       object will have the same properties.
    */
    class yc_var_proxy {
    private:
        yc_var_ptr _var;
        
    public:

        /// Contructor taking a vector of index vars.
        /**
           A wrapper around yc_solution::new_var() and
           yc_solution::new_scratch_var().
        */
        yc_var_proxy(const std::string& name
                    /**< [in] Name of the new var; must be a valid C++
                       identifier and unique across vars. */,
                    yc_solution_ptr soln
                    /**< [in] Shared pointer to solution that will share ownership of the \ref yc_var. */,
                    const std::vector< yc_index_node_ptr > &dims
                    /**< [in] Dimensions of the var.
                       Each dimension is identified by an associated index. */,
                    bool is_scratch = false
                    /**< [in] Whether to make a scratch var. */) {
            if (is_scratch)
                _var = soln->new_scratch_var(name, dims);
            else
                _var = soln->new_var(name, dims);
        }

#ifndef SWIG
        /// Contructor taking an initializer_list of index vars.
        /**
           A wrapper around yc_solution::new_var() and
           yc_solution::new_scratch_var().
           @note Not available in the Python API. Use the vector version.
        */
        yc_var_proxy(const std::string& name
                    /**< [in] Name of the new var; must be a valid C++
                       identifier and unique across vars. */,
                    yc_solution_ptr soln
                    /**< [in] Pointer to solution that will own the var. */,
                    const std::initializer_list< yc_index_node_ptr > &dims
                    /**< [in] Dimensions of the var.
                       Each dimension is identified by an associated index. */,
                    bool is_scratch = false
                    /**< [in] Whether to make a scratch var. */) {
            if (is_scratch)
                _var = soln->new_scratch_var(name, dims);
            else
                _var = soln->new_var(name, dims);
        }
#endif
        
        /// Contructor for a simple scalar value.
        /**
           A wrapper around yc_solution::new_var().
        */
        yc_var_proxy(const std::string& name
                    /**< [in] Name of the new var; must be a valid C++
                       identifier and unique across vars. */,
                    yc_solution_ptr soln
                    /**< [in] Pointer to solution that will own the var. */) {
            _var = soln->new_var(name, { });
        }

        /// Contructor taking an existing var.
        /**
           Creates a new \ref yc_var_proxy wrapper around an
           existing var.
        */
        yc_var_proxy(yc_var_ptr& var) : _var(var) { }
                    
        /// Provide a virtual destructor.
        virtual ~yc_var_proxy() { }

        /// Get the underlying \ref yc_var pointer.
        virtual yc_var_ptr get_var() {
            return _var;
        }

        /// Get the underlying \ref yc_var pointer.
        virtual yc_var_ptr get_var() const {
            return _var;
        }

        /// Create an expression for a point in a var.
        /**
           A wrapper around yc_var::new_var_point().
           The number of arguments must match the dimensionality of the var.

           Example w/2D var `B`: `A(t+1, x) EQUALS A(t, x) + B(vec)`,
        */
        virtual yc_var_point_node_ptr
        operator()(const std::vector<yc_number_node_ptr>& index_exprs) {
            return _var->new_var_point(index_exprs);
        }

#ifndef SWIG
        /// Create an expression for a point in a var.
        /**
           A wrapper around yc_var::new_var_point().
           The number of arguments must match the dimensionality of the var.

           Example w/2D var `B`: `A(t+1, x) EQUALS A(t, x) + B({x, 3})`.
           @note Not available in Python API. Use vector version.
        */
        virtual yc_var_point_node_ptr
        operator()(const std::initializer_list<yc_number_node_ptr>& index_exprs) {
            return _var->new_var_point(index_exprs);
        }

        /// Create an expression for a point in a 1-6 dim var.
        /**
           A wrapper around yc_var::new_var_point().
           The number of non-null arguments must match the dimensionality of the var.
           For more than 6 dims, use the vector or initializer-list version.

           Example w/2D var `B`: `A(t+1, x) EQUALS A(t, x) + B(x, 3)`.
           @note Not available in Python API. Use vector version.
        */
        virtual yc_var_point_node_ptr operator()(const yc_number_any_arg i1 = nullptr,
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
            return _var->new_var_point(args);
        }

        /// Create an expression for a point in a zero-dim (scalar) var using implicit conversion.
        /**
           A wrapper around yc_var::new_var_point().

           Example w/0D var `B`: `A(t+1, x) EQUALS A(t, x) + B`.
           @note Not available in Python API. 
           Use vector version with empty vector.
        */
        virtual operator yc_number_ptr_arg() {
            return _var->new_var_point({});
        }

        /// Create an expression for a point in a one-dim (array) var.
        /**
           A wrapper around yc_var::new_var_point().

           Example w/1D var `B`: `A(t+1, x) EQUALS A(t, x) + B[x]`.
           @note Not available in Python API. 
           Use vector version with 1-element vector.
        */
        virtual yc_var_point_node_ptr operator[](const yc_number_any_arg i1) {
            return _var->new_var_point({i1});
        }
        
#endif
        
    };                          // yc_var_proxy.
    /** @}*/

    /// **[Deprecated]** Use yc_var.
    YASK_DEPRECATED
    typedef yc_var yc_grid;
    /// **[Deprecated]** Use yc_var_ptr.
    YASK_DEPRECATED
    typedef yc_var_ptr yc_grid_ptr;
    /// **[Deprecated]** Use yc_var_point_node.
    YASK_DEPRECATED
    typedef yc_var_point_node yc_grid_point_node;
    /// **[Deprecated]** Use yc_var_point_node_ptr.
    YASK_DEPRECATED
    typedef yc_var_point_node_ptr yc_grid_point_node_ptr;

} // namespace yask.

// More solution-based objects.
#include "aux/yc_solution_api.hpp"
