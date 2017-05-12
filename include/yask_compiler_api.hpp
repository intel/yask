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
#include <memory>
#include <climits>

namespace yask {

    /** @mainpage
     * @section oview Overview
     * This document provides usage information for the YASK stencil compiler
     * API (application-programmer interface).\n
     * The API is available for C++ and for Python via SWIG.
     * @section usage Typical Usage Model
     * - Create a stencil_solution via yask_compiler_factory::new_stencil_solution().
     * - Create one or more grids via stencil_solution::new_grid() as needed by
     *   the stencil kernel being implemented.
     *   Calls to new_grid() specify the name and dimensions of each grid.
     *   Grids may be read-only (constants) or read-write.
     *   Each grid will be "owned" by the stencil_solution from which it was created.
     * - Create an equation for each read-write grid.
     *   - Example equation: `u(t+1, x, y, z) EQUALS (u(t, x, y, z) + u(t, x+1, y, z)) / 2`.
     *   - Create expressions "bottom-up" by creating the leaf nodes first.
     *   - Leaf nodes may be floating-point (FP) constants or references to grid points.
     *   - Constants are created via node_factory::new_const_number_node().
     *   - References to grid points are created via grid::new_relative_grid_point(), which
     *     specifies the grid indices relative to any point within the grid domain.
     *   - Create operator nodes via calls to node_factory::new_add_node(), etc., to build up
     *     larger expressions.
     *   - To complete each equation, use node_factory::new_equation_node() to specify an expression
     *     on the right-hand side (RHS) and the grid point that is defined to be equal
     *     to it on the left-hand side (LHS).
     * - Specify the solution step dimension via stencil_solution::set_step_dim().
     *   (This is usually "t" for time.)
     * - Specify the vector-folding and vector-clustering via stencil_solution::set_fold_len() and
     *   stencil_solution::set_cluster_mult().
     * - Format the equations for additional processing via stencil_solution::format() or stencil_solution::write().
     */
    
    // Forward declarations of classes and smart pointers.
    class stencil_solution;
    typedef std::shared_ptr<stencil_solution> stencil_solution_ptr;
    class grid;
    typedef grid* grid_ptr;
    class expr_node;
    typedef std::shared_ptr<expr_node> expr_node_ptr;
    class equation_node;
    typedef std::shared_ptr<equation_node> equation_node_ptr;
    class number_node;
    typedef std::shared_ptr<number_node> number_node_ptr;
    class grid_point_node;
    typedef std::shared_ptr<grid_point_node> grid_point_node_ptr;
    class const_number_node;
    typedef std::shared_ptr<const_number_node> const_number_node_ptr;
    class negate_node;
    typedef std::shared_ptr<negate_node> negate_node_ptr;
    class commutative_number_node;
    typedef std::shared_ptr<commutative_number_node> commutative_number_node_ptr;
    class add_node;
    typedef std::shared_ptr<add_node> add_node_ptr;
    class multiply_node;
    typedef std::shared_ptr<multiply_node> multiply_node_ptr;
    class subtract_node;
    typedef std::shared_ptr<subtract_node> subtract_node_ptr;
    class divide_node;
    typedef std::shared_ptr<divide_node> divide_node_ptr;
    class bool_node;
    typedef std::shared_ptr<bool_node> bool_node_ptr;

    /// Factory to create objects needed to define a stencil solution.
    class yask_compiler_factory {
    public:
        virtual ~yask_compiler_factory() {}

        /// Create a stencil solution.
        /** A stencil solution contains all the grids and equations.
         * @returns Pointer to new solution object. */
        virtual stencil_solution_ptr
        new_stencil_solution(const std::string& name /**< [in] Name of the solution; 
                                                        must be a valid C++ identifier. */ );
    };
            
    /// Stencil solution.
    /** Objects of this type contain all the grids and equations
     * that comprise a solution. */
    class stencil_solution {
    public:
        virtual ~stencil_solution() {}

        /// Get the name of the solution.
        /** @returns String containing the solution name provided via new_stencil_solution(). */
        virtual const std::string&
        get_name() const =0;

        /// Set the name of the solution.
        /** Allows changing the name from what was provided via new_stencil_solution(). */
        virtual void
        set_name(std::string name /**< [in] Name; must be a valid C++ identifier. */ ) =0;

        /// Create an n-dimensional grid in the solution.
        /** "Grid" is a generic term for any n-dimensional tensor.  A 1-dim
         * grid is an array, a 2-dim grid is a matrix, etc.  Define the name
         * of each dimension that is needed via this interface. Example:
         * new_grid("heat", "t", "x", "y") will create a 3D grid.
         * All dimension-name strings must be valid C++ identifiers.
         * @returns Pointer to the new grid. */
        virtual grid_ptr
        new_grid(std::string name /**< [in] Unique name of the grid;
                                     must be a valid C++ identifier.. */,
                 std::string dim1 = "" /**< [in] Name of 1st dimension. */,
                 std::string dim2 = "" /**< [in] Name of 2nd dimension. */,
                 std::string dim3 = "" /**< [in] Name of 3rd dimension. */,
                 std::string dim4 = "" /**< [in] Name of 4th dimension. */,
                 std::string dim5 = "" /**< [in] Name of 5th dimension. */,
                 std::string dim6 = "" /**< [in] Name of 6th dimension. */ ) =0;

        /// Get the number of grids in the solution.
        /** @returns Number of grids that have been created via new_grid(). */
        virtual int
        get_num_grids() const =0;
        
        /// Get the specified grid.
        /** @returns Pointer to the nth grid. */
        virtual grid_ptr
        get_grid(int n /**< [in] Index of grid between zero (0)
                              and get_num_grids()-1. */ ) =0;
        
        /// Get the number of equations in the solution.
        /** Equations are added when equation_nodes are created via new_equation_node().
         * @returns Number of equations that have been created. */
        virtual int
        get_num_equations() const =0;

        /// Get the specified equation.
        /** @returns Pointer to equation_node of nth equation. */
        virtual equation_node_ptr
        get_equation(int n /**< [in] Index of equation between zero (0)
                              and get_num_equations()-1. */ ) =0;

        /// Set the solution step dimension name.
        /** Default is "t" for time. */
        virtual void
        set_step_dim(const std::string& dim /**< [in] Step dimension, e.g., "t". */ ) =0;

        /// Get the solution step dimension.
        /** @returns String containing the current step dimension name. */
        virtual const std::string&
        get_step_dim() const =0;

        /// Set the vectorization length in given dimension.
        /** For YASK-code generation, the product of the fold lengths should
         * be equal to the number of elements in a HW SIMD register. For
         * example, for SP FP elements in AVX-512 vectors, the product of
         * the fold lengths should be 16, e.g., x=4 and y=4. This is not
         * checked by the compiler, since it does not know the FP precision
         * that will be used. A fold length >1 cannot be applied to the step
         * dimension. Default is one (1) in each dimension. */
        virtual void
        set_fold_len(const std::string& dim /**< [in] Dimension of fold, e.g., "x". */,
                     int len /**< [in] Length of vectorization in 'dim' */ ) =0;

        /// Reset all vector-folding settings.
        /** All fold lengths will return to the default of one (1). */
        virtual void
        clear_folding() =0;
        
        /// Set the cluster multiplier (unroll factor) in given dimension.
        /** For YASK-code generation, this will have the effect of creating
         * N vectors of output for each equation, where N is the product of
         * the cluster multipliers. A fold length >1 cannot be applied to
         * the step dimension. Default is one (1) in each dimension. */
        virtual void
        set_cluster_mult(const std::string& dim /**< [in] Direction of unroll, e.g., "y". */,
                         int mult /**< [in] Number of vectors in 'dim' */ ) =0;

        /// Reset all vector-clustering settings.
        /** All cluster multipliers will return to the default of one (1). */
        virtual void
        clear_clustering() =0;
        
        /// Format the current equation(s).
        /** Currently supported format types:
         * Type    | Output
         * --------|--------
         * cpp     | YASK stencil classes for generic C++.
         * avx     | YASK stencil classes for CORE AVX ISA. 
         * avx2    | YASK stencil classes for CORE AVX2 ISA.
         * avx512  | YASK stencil classes for CORE AVX-512 & MIC AVX-512 ISAs.
         * knc     | YASK stencil classes for Knights Corner ISA. 
         * dot     | DOT-language description.
         * dot-lite| DOT-language description of grid accesses only.
         * pseudo  | Human-readable pseudo-code (for debug).
         * @warning *Side effect:* Applies optimizations to the equation(s), so some pointers
         * to nodes in the original equations may refer to modified nodes or nodes
         * that have been optimized away after calling format().
         * @returns String containing formatted output. 
         * The YASK or DOT strings are typically then written to a file.
         */
        virtual std::string
        format(const std::string& format_type /**< [in] Name of type from above table. */,
               bool debug = false /**< [in] Print diagnostic info to stdout. */ ) =0;

        /// Format the current equation(s) and write to given file.
        /** See format() for supported format types and side-effects. */
        virtual void
        write(const std::string& filename /**< [in] Name of output file. */,
              const std::string& format_type /**< [in] Name of type from format() table. */,
              bool debug = false /**< [in] Print diagnostic info to stdout. */ ) =0;
    };

    /// A grid.
    /** "Grid" is a generic term for any n-dimensional tensor.  A 0-dim grid
     * is a scalar, a 1-dim grid is an array, etc.  Create new grids via
     * new_grid(). */
    class grid {
    public:
        virtual ~grid() {}

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

        /// Create a reference to a point in a 1D grid (array).
        /** See more detail on 3-argument version. */
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset /**< [in] offset from dim1 index. */ ) =0;

        /// Create a reference to a point in a 2D grid.
        /** See more detail on 3-argument version. */
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset /**< [in] offset from dim1 index. */,
                                int dim2_offset /**< [in] offset from dim2 index. */ ) =0;

        /// Create a reference to a point in a 3D grid.
        /** The indices are specified relative to the stencil-evaluation
         * index.  Each offset refers to the dimensions defined when the
         * grid was added to a stencil_solution. Example: if g =
         * new_grid("heat", "t", "x", "y"), then
         * g->new_relative_grid_point(1, -1, 0) refers to heat(t+1, x-1, y)
         * for some point t, x, y during stencil evaluation.
         * @returns Pointer to AST node used to read or write from point in grid. */
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset /**< [in] offset from dim1 index. */,
                                int dim2_offset /**< [in] offset from dim2 index. */,
                                int dim3_offset /**< [in] offset from dim3 index. */ ) =0;

        /// Create a reference to a point in a 4D grid.
        /** See more detail on 3-argument version. */
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset /**< [in] offset from dim1 index. */,
                                int dim2_offset /**< [in] offset from dim2 index. */,
                                int dim3_offset /**< [in] offset from dim3 index. */,
                                int dim4_offset /**< [in] offset from dim4 index. */ ) =0;

        /// Create a reference to a point in a 5D grid.
        /** See more detail on 3-argument version. */
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset /**< [in] offset from dim1 index. */,
                                int dim2_offset /**< [in] offset from dim2 index. */,
                                int dim3_offset /**< [in] offset from dim3 index. */,
                                int dim4_offset /**< [in] offset from dim4 index. */,
                                int dim5_offset /**< [in] offset from dim5 index. */ ) =0;

        /// Create a reference to a point in a 6D grid.
        /** See more detail on 3-argument version. */
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset /**< [in] offset from dim1 index. */,
                                int dim2_offset /**< [in] offset from dim2 index. */,
                                int dim3_offset /**< [in] offset from dim3 index. */,
                                int dim4_offset /**< [in] offset from dim4 index. */,
                                int dim5_offset /**< [in] offset from dim5 index. */,
                                int dim6_offset /**< [in] offset from dim6 index. */ ) =0;
    };

    /// Factory to create AST nodes.
    /** Note: Grid-point reference nodes are created from a grid object
     * instead of from this factory. */
    class node_factory {
    public:
        virtual ~node_factory() {}

        /// Create an equation node.
        /** Indicates grid point on LHS is equivalent to expression on
         * RHS. This is NOT a test for equality.  When an equation is
         * created, it is automatically added to the list of equations for
         * the stencil_solution that contains the grid that is on the
         * LHS.
         * @returns Pointer to new node. */
        virtual equation_node_ptr
        new_equation_node(grid_point_node_ptr lhs /**< [in] Grid-point before EQUALS operator. */,
                        number_node_ptr rhs /**< [in] Expression after EQUALS operator. */ );

        /// Create a constant numerical value node.
        /** This is unary negation.
         *  Use new_subtraction_node() for binary '-'.
         * @returns Pointer to new node. */
        virtual const_number_node_ptr
        new_const_number_node(double val /**< [in] Value to store in node. */ );

        /// Create a numerical negation operator node.
        /** @returns Pointer to new node. */
        virtual negate_node_ptr
        new_negate_node(number_node_ptr rhs /**< [in] Expression after '-' sign. */ );

        /// Create an addition node.
        /** Nodes must be created with at least two operands, and more can
         *  be added by calling add_operand() on the returned node.
         * @returns Pointer to new node. */
        virtual add_node_ptr
        new_add_node(number_node_ptr lhs /**< [in] Expression before '+' sign. */,
                     number_node_ptr rhs /**< [in] Expression after '+' sign. */ );

        /// Create a multiplication node.
        /** Nodes must be created with at least two operands, and more can
         *  be added by calling add_operand() on the returned node.
         * @returns Pointer to new node. */
        virtual multiply_node_ptr
        new_multiply_node(number_node_ptr lhs /**< [in] Expression before '*' sign. */,
                          number_node_ptr rhs /**< [in] Expression after '*' sign. */ );

        /// Create a subtraction node.
        /** This is binary subtraction.
         *  Use new_negation_node() for unary '-'.
         * @returns Pointer to new node. */
        virtual subtract_node_ptr
        new_subtract_node(number_node_ptr lhs /**< [in] Expression before '-' sign. */,
                          number_node_ptr rhs /**< [in] Expression after '-' sign. */ );

        /// Create a division node.
        /** @returns Pointer to new node. */
        virtual divide_node_ptr
        new_divide_node(number_node_ptr lhs /**< [in] Expression before '/' sign. */,
                        number_node_ptr rhs /**< [in] Expression after '/' sign. */ );
    };

    /// Base class for all AST nodes.
    /** An object of this abstract type cannot be created. */
    class expr_node {
    public:
        virtual ~expr_node() {}

        /// Create a simple human-readable string.
        /** Formats the expression starting at this node.
         * @returns String containing a single-line human-readable version of the expression.
         */
        virtual std::string format_simple() const =0;

        /// Count the size of the AST.
        /** @returns Number of nodes in this tree,
         * including this node and all its descendants. */
        virtual int get_num_nodes() const =0;
    };

    /// Equation node.
    /** Indicates grid point on LHS is equivalent to expression
     * on RHS. This is NOT a test for equality. */
    class equation_node : public virtual expr_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Grid-point node appearing before the EQUALS operator. */
        virtual grid_point_node_ptr get_lhs() =0;
    
        /// Get the right-hand-side operand.
        /** @returns Expression node appearing after the EQUALS operator. */
        virtual number_node_ptr get_rhs() =0;
    };

    /// Base class for all real or integer AST nodes.
    /** An object of this abstract type cannot be created. */
    class number_node : public virtual expr_node { };

    /// Base class for all boolean AST nodes.
    /** An object of this abstract type cannot be created. */
    class bool_node : public virtual expr_node { };

    /// A reference to a point in a grid.
    class grid_point_node : public virtual number_node {
    public:

        /// Get the grid this point is in.
        /** @returns Pointer to grid. */
        virtual grid_ptr get_grid() =0;
    };
    
    /// A constant numerical value.
    /** All values are stored as doubles.
     * This is a leaf node in an AST.
     * Use a yask_compiler_factory object to create an object of this type. */
    class const_number_node : public virtual number_node {
    public:

        /// Set the value.
        /** The value is considered "constant" only when the 
         * compiler output is created. It can be changed in the AST. */
        virtual void set_value(double val /**< [in] Value to store in node. */ ) =0;

        /// Get the stored value.
        /** @returns Copy of stored value. */
        virtual double get_value() const =0;
    };

    /// A numerical negation operator.
    /** Example: used to implement -(a*b).
     * Use a yask_compiler_factory object to create an object of this type. */
    class negate_node : public virtual number_node {
    public:

        /// Get the [only] operand.
        /**  This node implements unary negation only, not subtraction, so there is
         * never a left-hand-side.
         * @returns Expression node on right-hand-side of '-' sign. */
        virtual number_node_ptr get_rhs() =0;
    };

    /// Base class for commutative numerical operators.
    /** This is used for operators whose arguments can be rearranged
     * mathematically, e.g., add and multiply. */
    class commutative_number_node : public virtual number_node {
    public:

        /// Get the number of operands.
        /** If there is just one operand, the operation itself is moot.  If
         * there are more than one operand, the operation applies between
         * them. Example: for an add operator, if the operands are 'a',
         * 'b', and 'c', then the expression is 'a + b + c'.
         * @returns Number of operands. */
        virtual int get_num_operands() =0;

        /// Get the specified operand.
        /** @returns Pointer to node at given position or null pointer if out of bounds. */
        virtual number_node_ptr
        get_operand(int i /**< [in] Index between zero (0)
                             and get_num_operands()-1. */ ) =0;

        /// Add an operand.
        virtual void
        add_operand(number_node_ptr node /**< [in] Top node of AST to add. */ ) =0;
    };

    /// An addition node.
    class add_node : public virtual commutative_number_node { };

    /// A multiplication node.
    class multiply_node : public virtual commutative_number_node { };

    /// A subtraction node.
    class subtract_node : public virtual number_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Pointer to expression node appearing before the '-' sign. */
        virtual number_node_ptr get_lhs() =0;
    
        /// Get the right-hand-side operand.
        /** @returns Pointer to expression node appearing after the '-' sign. */
        virtual number_node_ptr get_rhs() =0;
    };

    /// A division node.
    class divide_node : public virtual number_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Pointer to expression node appearing before the '/' sign. */
        virtual number_node_ptr get_lhs() =0;
    
        /// Get the right-hand-side operand.
        /** @returns Pointer to expression node appearing after the '/' sign. */
        virtual number_node_ptr get_rhs() =0;
    };

} // namespace yask.

#endif
