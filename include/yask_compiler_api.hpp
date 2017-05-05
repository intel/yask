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

#ifndef YASK_COMPILER_API
#define YASK_COMPILER_API

#include <string>
#include <memory>
#include <climits>

namespace yask {

    // Forward declarations of classes and smart pointers.
    class stencil_solution;
    typedef std::shared_ptr<stencil_solution> stencil_solution_ptr;
    class grid;
    typedef grid* grid_ptr;
    class expr_node;
    typedef std::shared_ptr<expr_node> expr_node_ptr;
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
        new_stencil_solution(const std::string& name /**< [in] Name of the solution. */ );
    };
            
    /// Stencil solution.
    /** Objects of this type contain all the grids and equations
     * that comprise a solution. */
    class stencil_solution {
    public:
        virtual ~stencil_solution() {}

        /// Get the name of the solution.
        virtual const std::string& get_name() const =0;

        /// Set the name of the solution.
        virtual void set_name(std::string name) =0;

        /// Add an n-dimensional grid to the solution.
        /** "Grid" is a generic term for any n-dimensional tensor.  A 1-dim
         * grid is an array, a 2-dim grid is a matrix, etc.  Define the name
         * of each dimension that is needed via this interface. Example:
         * new_grid("heat", "t", "x", "y") will create a 3D grid.
         * @returns Pointer to the new grid. */
        virtual grid_ptr
        add_grid(std::string name /**< [in] Unique name of the grid. */,
                 std::string dim1 = "" /**< [in] Name of 1st dimension. */,
                 std::string dim2 = "" /**< [in] Name of 2nd dimension. */,
                 std::string dim3 = "" /**< [in] Name of 3rd dimension. */,
                 std::string dim4 = "" /**< [in] Name of 4th dimension. */,
                 std::string dim5 = "" /**< [in] Name of 5th dimension. */,
                 std::string dim6 = "" /**< [in] Name of 6th dimension. */ ) =0;
    };

    /// A grid.
    /** "Grid" is a generic term for any n-dimensional tensor.  A 0-dim
     * grid is a scalar, a 1-dim grid is an array, etc. 
     * Create new grids via stencil_solution::add_grid(). */
    class grid {
    public:
        virtual ~grid() {}

        /// Get the name of the grid.
        virtual const std::string& get_name() const =0;

        /// Get the number of dimensions.
        virtual int get_num_dims() const =0;

        /// Get the name of the specified dimension.
        virtual const std::string&
        get_dim_name(int n /**< [in] Index of dimension between zero (0)
                              and get_num_dims()-1. */ ) const =0;

        /// Create a reference to a point in a 1D grid.
        /** See more detail on 3-argument version. */
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset /**< [in] offset from dim1 index. */ ) =0;

        /// Create a reference to a point in a 2D grid.
        /** See more detail on 3-argument version. */
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset /**< [in] offset from dim1 index. */,
                                int dim2_offset /**< [in] offset from dim2 index. */ ) =0;

        /// Create a reference to a point in a 3D grid.
        /** The indices are specified relative to the stencil evaluation
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

        /// Create a constant numerical value node.
        /** This is unary negation.
         *  Use new_subtraction_node() for binary '-'.
         * @returns New node. */
        virtual const_number_node_ptr
        new_const_number_node(double val /**< [in] Value to store in node. */ );

        /// Create a numerical negation operator node.
        /** @returns New node. */
        virtual negate_node_ptr
        new_negate_node(number_node_ptr rhs /**< [in] Expression after '-' sign. */ );

        /// Create an addition node.
        /** Nodes must be created with at least two operands, and more can
         *  be added by calling add_operand() on the returned node.
         * @returns New node. */
        virtual add_node_ptr
        new_add_node(number_node_ptr lhs /**< [in] Expression before '+' sign. */,
                     number_node_ptr rhs /**< [in] Expression after '+' sign. */ );

        /// Create a multiplication node.
        /** Nodes must be created with at least two operands, and more can
         *  be added by calling add_operand() on the returned node.
         * @returns New node. */
        virtual multiply_node_ptr
        new_multiply_node(number_node_ptr lhs /**< [in] Expression before '*' sign. */,
                          number_node_ptr rhs /**< [in] Expression after '*' sign. */ );

        /// Create a subtraction node.
        /** This is binary subtraction.
         *  Use new_negation_node() for unary '-'.
         * @returns New node. */
        virtual subtract_node_ptr
        new_subtract_node(number_node_ptr lhs /**< [in] Expression before '-' sign. */,
                          number_node_ptr rhs /**< [in] Expression after '-' sign. */ );

        /// Create a division node.
        /** @returns New node. */
        virtual divide_node_ptr
        new_divide_node(number_node_ptr lhs /**< [in] Expression before '/' sign. */,
                        number_node_ptr rhs /**< [in] Expression after '/' sign. */ );
    };

    /// Base class for all AST nodes.
    /** An object of this abstract type cannot be created. */
    class expr_node {
    public:
        virtual ~expr_node() {}

        /// Create a simple readable string.
        /** Formats the expression starting at this node 
         * into a single-line readable string.
         */
        virtual std::string format_simple() const =0;

        /// Count the size of the AST.
        /** @return Number of nodes in this [sub]tree,
         * inclusing this node and all its descendants. */
        virtual int get_num_nodes() const =0;
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
        /** @return Copy of stored value. */
        virtual double get_value() const =0;
    };

    /// A numerical negation operator.
    /** Example: used to implement -(a*b).
     * Use a yask_compiler_factory object to create an object of this type. */
    class negate_node : public virtual number_node {
    public:

        /// Get the [only] operand.
        /** @return Expression node on right-hand-side of '-' sign.  This
         * node implements unary negation only, not subtraction, so there is
         * never a left-hand-side. */
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
         * 'b', and 'c', then the expression is 'a + b + c'. */
        virtual int get_num_operands() =0;

        /// Get the specified operand.
        /** @returns Node at given position or null pointer if out of bounds. */
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
        /** @return Expression node appearing before the opeator. */
        virtual number_node_ptr get_lhs() =0;
    
        /// Get the right-hand-side operand.
        /** @return Expression node appearing after the opeator. */
        virtual number_node_ptr get_rhs() =0;
    };

    /// A division node.
    class divide_node : public virtual number_node {
    public:

        /// Get the left-hand-side operand.
        /** @return Expression node appearing before the opeator. */
        virtual number_node_ptr get_lhs() =0;
    
        /// Get the right-hand-side operand.
        /** @return Expression node appearing after the opeator. */
        virtual number_node_ptr get_rhs() =0;
    };

} // namespace yask.

#endif
