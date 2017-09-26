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

// This file uses Doxygen 1.8 markup for API documentation-generation.
// See http://www.stack.nl/~dimitri/doxygen.

#ifndef YASK_COMPILER_API
#define YASK_COMPILER_API

#include <string>
#include <vector>
#include <memory>
#include "yask_common_api.hpp"

namespace yask {

    // Forward declarations of classes and their pointers.
    class yc_solution;
    typedef std::shared_ptr<yc_solution> yc_solution_ptr;
    class yc_grid;
    typedef yc_grid* yc_grid_ptr;

    // Forward declarations of expression nodes and their pointers.
    class yc_expr_node;
    typedef std::shared_ptr<yc_expr_node> yc_expr_node_ptr;
    class yc_index_node;
    typedef std::shared_ptr<yc_index_node> yc_index_node_ptr;
    class yc_equation_node;
    typedef std::shared_ptr<yc_equation_node> yc_equation_node_ptr;
    class yc_number_node;
    typedef std::shared_ptr<yc_number_node> yc_number_node_ptr;
    class yc_grid_point_node;
    typedef std::shared_ptr<yc_grid_point_node> yc_grid_point_node_ptr;
    class yc_const_number_node;
    typedef std::shared_ptr<yc_const_number_node> yc_const_number_node_ptr;
    class yc_negate_node;
    typedef std::shared_ptr<yc_negate_node> yc_negate_node_ptr;
    class yc_commutative_number_node;
    typedef std::shared_ptr<yc_commutative_number_node> yc_commutative_number_node_ptr;
    class yc_add_node;
    typedef std::shared_ptr<yc_add_node> yc_add_node_ptr;
    class yc_multiply_node;
    typedef std::shared_ptr<yc_multiply_node> yc_multiply_node_ptr;
    class yc_subtract_node;
    typedef std::shared_ptr<yc_subtract_node> yc_subtract_node_ptr;
    class yc_divide_node;
    typedef std::shared_ptr<yc_divide_node> yc_divide_node_ptr;
    class yc_bool_node;
    typedef std::shared_ptr<yc_bool_node> yc_bool_node_ptr;

    /// Factory to create objects needed to define a stencil solution.
    class yc_factory {
    public:
        virtual ~yc_factory() {}

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
           @returns String containing the solution name provided via new_solution().
        */
        virtual const std::string&
        get_name() const =0;

        /// Set the name of the solution.
        /**
           Allows changing the name from what was provided via new_solution(). 
        */
        virtual void
        set_name(std::string name
                 /**< [in] Name; must be a valid C++ identifier. */ ) =0;

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

           @returns Pointer to the new grid. 
        */
        virtual yc_grid_ptr
        new_grid(const std::string& name
                 /**< [in] Unique name of the grid; must be a valid C++
                    identifier and unique across grids. */,
                 const std::vector<yc_index_node_ptr>& dims
                 /**< [in] Dimensions of the grid.
                  Each dimension is identified by an associated index. */ ) =0;

#ifndef SWIG        
        /// Create an n-dimensional grid variable in the solution.
        /**
           "Grid" is a generic term for any n-dimensional variable.  A 0-dim
           grid is a scalar, a 1-dim grid is a vector, a 2-dim grid is a
           matrix, etc.
           @note This version is not available (or needed) in SWIG-based APIs, e.g., Python.
           @returns Pointer to the new grid. 
        */
        virtual yc_grid_ptr
        new_grid(const std::string& name /**< [in] Unique name of the grid; must be
                                            a valid C++ identifier and unique
                                            across grids. */,
                 const std::initializer_list<yc_index_node_ptr>& dims
                 /**< [in] Dimensions of the grid.
                    Each dimension is identified by an associated index. */ ) =0;
#endif
        
        /// Get all the grids in the solution.
        /** @returns Vector containing pointer to all grids. */
        virtual std::vector<yc_grid_ptr>
        get_grids() =0;
        
        /// Get the number of grids in the solution.
        /** @returns Number of grids that have been created via new_grid(). */
        virtual int
        get_num_grids() const =0;
        
        /// Get the specified grid.
        /** @returns Pointer to the nth grid. */
        virtual yc_grid_ptr
        get_grid(int n /**< [in] Index of grid between zero (0)
                              and get_num_grids()-1. */ ) =0;
        
        /// Get the number of equations in the solution.
        /** Equations are added when equation_nodes are created via new_equation_node().
            @returns Number of equations that have been created. */
        virtual int
        get_num_equations() const =0;

        /// Get the specified equation.
        /** @returns Pointer to equation_node of nth equation. */
        virtual yc_equation_node_ptr
        get_equation(int n /**< [in] Index of equation between zero (0)
                              and get_num_equations()-1. */ ) =0;

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
                     int len /**< [in] Length of vectorization in 'dim' */ ) =0;

        /// Reset all vector-folding settings.
        /** All fold lengths will return to the default of one (1). */
        virtual void
        clear_folding() =0;

        /// Get current floating-point precision setting.
        /** @returns Number of bytes in a FP number. */
        virtual int
        get_element_bytes() const =0;

        /// Set floating-point precision.
        virtual void
        set_element_bytes(int nbytes /**< [in] Number of bytes in a FP number.
                                        Should be 4 or 8. */ ) =0;

        /// Set the cluster multiplier (unroll factor) in given dimension.
        /** For YASK-code generation, this will have the effect of creating
            N vectors of output for each equation, where N is the product of
            the cluster multipliers. 
            @note A multiplier >1 cannot be applied to
            the step dimension. 
            @note Default is one (1) in each dimension. */
        virtual void
        set_cluster_mult(const yc_index_node_ptr dim
                         /**< [in] Direction of unroll, e.g., "y".
                            This must be an index created by new_domain_index().  */,
                         int mult /**< [in] Number of vectors in 'dim' */ ) =0;

        /// Reset all vector-clustering settings.
        /** All cluster multipliers will return to the default of one (1). */
        virtual void
        clear_clustering() =0;
        
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

            Progress text will be written to the output stream set via set_debug_output().

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
    };

    /// A compile-time grid.
    /** "Grid" is a generic term for any n-dimensional array.  A 0-dim grid
        is a scalar, a 1-dim grid is an array, etc.
        A compile-time grid is a variable used for constructing equations.
        It does not contain any data.
        Data is only stored during run-time, using a yk_grid.
        Create new grids via yc_solution::new_grid(). */
    class yc_grid {
    public:
        virtual ~yc_grid() {}

        /// Get the name of the grid.
        /** @returns String containing name provided via new_grid(). */
        virtual const std::string& get_name() const =0;

        /// Get the number of dimensions.
        /** @returns Number of dimensions created via new_grid(). */
        virtual int get_num_dims() const =0;

        /// Get the name of the specified dimension.
        /** @returns String containing name of dimension created via new_grid(). */
        virtual const std::string&
        get_dim_name(int n /**< [in] Index of dimension between zero (0)
                              and get_num_dims()-1. */ ) const =0;

        /// Get all the dimensions in this grid.
        /**
           Includes step dimension if it is a dimension of this grid.
           May be different than values returned from yc_solution::get_domain_dim_names().
           @returns List of names of all the dimensions.
        */
        virtual std::vector<std::string>
        get_dim_names() const =0;
        
        /// Create a reference to a point in a grid.
        /** The indices are specified relative to the stencil-evaluation
            index.  Each offset refers to the dimensions defined when the
            grid was created via stencil_solution::new_grid(). 
            Example: if g = new_grid("heat", {"t", "x", "y"}), then
            g->new_relative_grid_point(1, -1, 0) refers to heat(t+1, x-1, y)
            for some point t, x, y during stencil evaluation.
            @warning This convenience function can only be used when every
            dimension of the grid is either the step dimension or a domain dimension.
            @note Offsets beyond the dimensions in the grid will be ignored.
            @returns Pointer to AST node used to read from or write to point in grid. */
        virtual yc_grid_point_node_ptr
        new_relative_grid_point(std::vector<int> dim_offsets
                                /**< [in] offset from evaluation index in each dim. */ ) =0;

#ifndef SWIG        
        /// Create a reference to a point in a grid.
        /** The indices are specified relative to the stencil-evaluation
            index.  Each offset refers to the dimensions defined when the
            grid was created via stencil_solution::new_grid(). 
            Example: if g = new_grid("heat", {"t", "x", "y"}), then
            g->new_relative_grid_point({1, -1, 0}) refers to heat(t+1, x-1, y)
            for some point t, x, y during stencil evaluation.
            @note Offsets beyond the dimensions in the grid will be ignored.
            @note This version is not available (or needed) in SWIG-based APIs, e.g., Python.
            @returns Pointer to AST node used to read or write from point in grid. */
        virtual yc_grid_point_node_ptr
        new_relative_grid_point(const std::initializer_list<int>& dim_offsets) = 0;
#endif
    };
    
    /// Factory to create AST nodes.
    /** @note Grid-point reference nodes are created from a `yc_grid` object
        instead of from this factory. */
    class yc_node_factory {
    public:
        virtual ~yc_node_factory() {}

        /// Create a step-index node.
        /**
           Create a variable to be used to index grids in the
           solution-step dimension.
           The name usually describes time, e.g. "t". 
        */
        virtual yc_index_node_ptr
        new_step_index(const std::string& name
                     /**< [in] Step dimension name. */ ) =0;

        /// Create a domain-index node.
        /**
           Create a variable to be used to index grids in the
           solution-domain dimension.
           The name usually describes spatial dimensions, e.g. "x" or "y". 
           This should *not* include the step dimension, which is specified via
           new_step_index().
         */
        virtual yc_index_node_ptr
        new_domain_index(const std::string& name
                     /**< [in] Domain index name. */ ) =0;
        
        /// Create a new miscellaneous index.
        /**
           Create an variable to be used to index grids in the
           some dimension that is not the step dimension
           or a domain dimension. Example: index into an array.
         */
        virtual yc_index_node_ptr
        new_misc_index(const std::string& name
                       /**< [in] Index name. */ ) =0;
        
        /// Create an equation node.
        /** Indicates grid point on LHS is equivalent to expression on
            RHS. This is NOT a test for equality.  When an equation is
            created, it is automatically added to the list of equations for
            the yc_solution that contains the grid that is on the
            LHS.
            @returns Pointer to new node. */
        virtual yc_equation_node_ptr
        new_equation_node(yc_grid_point_node_ptr lhs /**< [in] Grid-point before EQUALS operator. */,
                        yc_number_node_ptr rhs /**< [in] Expression after EQUALS operator. */ );

        /// Create a constant numerical value node.
        /** This is unary negation.
             Use new_subtraction_node() for binary '-'.
            @returns Pointer to new node. */
        virtual yc_const_number_node_ptr
        new_const_number_node(double val /**< [in] Value to store in node. */ );

        /// Create a numerical negation operator node.
        /** @returns Pointer to new node. */
        virtual yc_negate_node_ptr
        new_negate_node(yc_number_node_ptr rhs /**< [in] Expression after '-' sign. */ );

        /// Create an addition node.
        /** Nodes must be created with at least two operands, and more can
             be added by calling add_operand() on the returned node.
            @returns Pointer to new node. */
        virtual yc_add_node_ptr
        new_add_node(yc_number_node_ptr lhs /**< [in] Expression before '+' sign. */,
                     yc_number_node_ptr rhs /**< [in] Expression after '+' sign. */ );

        /// Create a multiplication node.
        /** Nodes must be created with at least two operands, and more can
             be added by calling add_operand() on the returned node.
            @returns Pointer to new node. */
        virtual yc_multiply_node_ptr
        new_multiply_node(yc_number_node_ptr lhs /**< [in] Expression before '*' sign. */,
                          yc_number_node_ptr rhs /**< [in] Expression after '*' sign. */ );

        /// Create a subtraction node.
        /** This is binary subtraction.
             Use new_negation_node() for unary '-'.
            @returns Pointer to new node. */
        virtual yc_subtract_node_ptr
        new_subtract_node(yc_number_node_ptr lhs /**< [in] Expression before '-' sign. */,
                          yc_number_node_ptr rhs /**< [in] Expression after '-' sign. */ );

        /// Create a division node.
        /** @returns Pointer to new node. */
        virtual yc_divide_node_ptr
        new_divide_node(yc_number_node_ptr lhs /**< [in] Expression before '/' sign. */,
                        yc_number_node_ptr rhs /**< [in] Expression after '/' sign. */ );
    };

    /// Base class for all AST nodes.
    /** An object of this abstract type cannot be created. */
    class yc_expr_node {
    public:
        virtual ~yc_expr_node() {}

        /// Create a simple human-readable string.
        /** Formats the expression starting at this node.
            @returns String containing a single-line human-readable version of the expression.
         */
        virtual std::string format_simple() const =0;

        /// Count the size of the AST.
        /** @returns Number of nodes in this tree,
            including this node and all its descendants. */
        virtual int get_num_nodes() const =0;
    };

    /// Equation node.
    /** Indicates grid point on LHS is equivalent to expression
        on RHS. This is NOT a test for equality. */
    class yc_equation_node : public virtual yc_expr_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Grid-point node appearing before the EQUALS operator. */
        virtual yc_grid_point_node_ptr get_lhs() =0;
    
        /// Get the right-hand-side operand.
        /** @returns Expression node appearing after the EQUALS operator. */
        virtual yc_number_node_ptr get_rhs() =0;
    };

    /// Base class for all real or integer AST nodes.
    /** An object of this abstract type cannot be created. */
    class yc_number_node : public virtual yc_expr_node { };

    /// Base class for all boolean AST nodes.
    /** An object of this abstract type cannot be created. */
    class yc_bool_node : public virtual yc_expr_node { };

    /// A dimension or an index in that dimension.
    /** This is a leaf node in an AST.
        Use a yask_solution object to create an object of this type. */
    class yc_index_node : public virtual yc_number_node {
    public:

        /// Get the dimension's name.
        /** @returns Name given at creation. */
        virtual const std::string& get_name() const =0;
    };

    /// A reference to a point in a grid.
   class yc_grid_point_node : public virtual yc_number_node {
    public:

        /// Get the grid this point is in.
        /** @returns Pointer to grid. */
        virtual yc_grid_ptr get_grid() =0;
    };
    
    /// A constant numerical value.
    /** All values are stored as doubles.
        This is a leaf node in an AST.
        Use a yask_compiler_factory object to create an object of this type. */
    class yc_const_number_node : public virtual yc_number_node {
    public:

        /// Set the value.
        /** The value is considered "constant" only when the 
            compiler output is created. It can be changed in the AST. */
        virtual void set_value(double val /**< [in] Value to store in node. */ ) =0;

        /// Get the stored value.
        /** @returns Copy of stored value. */
        virtual double get_value() const =0;
    };

    /// A numerical negation operator.
    /** Example: used to implement -(a*b).
        Use a yask_compiler_factory object to create an object of this type. */
    class yc_negate_node : public virtual yc_number_node {
    public:

        /// Get the [only] operand.
        /**  This node implements unary negation only, not subtraction, so there is
            never a left-hand-side.
            @returns Expression node on right-hand-side of '-' sign. */
        virtual yc_number_node_ptr get_rhs() =0;
    };

    /// Base class for commutative numerical operators.
    /** This is used for operators whose arguments can be rearranged
        mathematically, e.g., add and multiply. */
    class yc_commutative_number_node : public virtual yc_number_node {
    public:

        /// Get the number of operands.
        /** If there is just one operand, the operation itself is moot.  If
            there are more than one operand, the operation applies between
            them. Example: for an add operator, if the operands are 'a',
            'b', and 'c', then the expression is 'a + b + c'.
            @returns Number of operands. */
        virtual int get_num_operands() =0;

        /// Get the specified operand.
        /** @returns Pointer to node at given position or null pointer if out of bounds. */
        virtual yc_number_node_ptr
        get_operand(int i /**< [in] Index between zero (0)
                             and get_num_operands()-1. */ ) =0;

        /// Add an operand.
        virtual void
        add_operand(yc_number_node_ptr node /**< [in] Top node of AST to add. */ ) =0;
    };

    /// An addition node.
    class yc_add_node : public virtual yc_commutative_number_node { };

    /// A multiplication node.
    class yc_multiply_node : public virtual yc_commutative_number_node { };

    /// A subtraction node.
    class yc_subtract_node : public virtual yc_number_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Pointer to expression node appearing before the '-' sign. */
        virtual yc_number_node_ptr get_lhs() =0;
    
        /// Get the right-hand-side operand.
        /** @returns Pointer to expression node appearing after the '-' sign. */
        virtual yc_number_node_ptr get_rhs() =0;
    };

    /// A division node.
    class yc_divide_node : public virtual yc_number_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Pointer to expression node appearing before the '/' sign. */
        virtual yc_number_node_ptr get_lhs() =0;
    
        /// Get the right-hand-side operand.
        /** @returns Pointer to expression node appearing after the '/' sign. */
        virtual yc_number_node_ptr get_rhs() =0;
    };

} // namespace yask.

#endif
