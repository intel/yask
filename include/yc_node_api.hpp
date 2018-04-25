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

///////// API for the YASK stencil compiler node classes. ////////////

// This file uses Doxygen 1.8 markup for API documentation-generation.
// See http://www.stack.nl/~dimitri/doxygen.
/** @file yc_node_api.hpp */ 

#ifndef YC_NODES
#define YC_NODES

namespace yask {

    /**
     * \addtogroup yc
     * @{
     */

    // Forward declarations of expression nodes and their pointers.
    // See yask_compiler_api.hpp for more.

    class yc_number_node;
    /// Shared pointer to \ref yc_number_node
    typedef std::shared_ptr<yc_number_node> yc_number_node_ptr;

    class yc_const_number_node;
    /// Shared pointer to \ref yc_const_number_node
    typedef std::shared_ptr<yc_const_number_node> yc_const_number_node_ptr;

    class yc_negate_node;
    /// Shared pointer to \ref yc_negate_node
    typedef std::shared_ptr<yc_negate_node> yc_negate_node_ptr;

    class yc_commutative_number_node;
    /// Shared pointer to \ref yc_commutative_number_node
    typedef std::shared_ptr<yc_commutative_number_node> yc_commutative_number_node_ptr;

    class yc_add_node;
    /// Shared pointer to \ref yc_add_node
    typedef std::shared_ptr<yc_add_node> yc_add_node_ptr;

    class yc_multiply_node;
    /// Shared pointer to \ref yc_multiply_node
    typedef std::shared_ptr<yc_multiply_node> yc_multiply_node_ptr;

    class yc_subtract_node;
    /// Shared pointer to \ref yc_subtract_node
    typedef std::shared_ptr<yc_subtract_node> yc_subtract_node_ptr;

    class yc_divide_node;
    /// Shared pointer to \ref yc_divide_node
    typedef std::shared_ptr<yc_divide_node> yc_divide_node_ptr;

    class yc_bool_node;
    /// Shared pointer to \ref yc_bool_node
    typedef std::shared_ptr<yc_bool_node> yc_bool_node_ptr;

    class yc_not_node;
    /// Shared pointer to \ref yc_not_node
    typedef std::shared_ptr<yc_not_node> yc_not_node_ptr;

    class yc_equals_node;
    /// Shared pointer to \ref yc_equals_node
    typedef std::shared_ptr<yc_equals_node> yc_equals_node_ptr;

    class yc_not_equals_node;
    /// Shared pointer to \ref yc_not_equals_node
    typedef std::shared_ptr<yc_not_equals_node> yc_not_equals_node_ptr;

    class yc_less_than_node;
    /// Shared pointer to \ref yc_less_than_node
    typedef std::shared_ptr<yc_less_than_node> yc_less_than_node_ptr;

    class yc_greater_than_node;
    /// Shared pointer to \ref yc_greater_than_node
    typedef std::shared_ptr<yc_greater_than_node> yc_greater_than_node_ptr;

    class yc_not_less_than_node;
    /// Shared pointer to \ref yc_not_less_than_node
    typedef std::shared_ptr<yc_not_less_than_node> yc_not_less_than_node_ptr;

    class yc_not_greater_than_node;
    /// Shared pointer to \ref yc_not_greater_than_node
    typedef std::shared_ptr<yc_not_greater_than_node> yc_not_greater_than_node_ptr;

    class yc_and_node;
    /// Shared pointer to \ref yc_and_node
    typedef std::shared_ptr<yc_and_node> yc_and_node_ptr;

    class yc_or_node;
    /// Shared pointer to \ref yc_or_node
    typedef std::shared_ptr<yc_or_node> yc_or_node_ptr;

    /// Factory to create AST nodes.
    /** @note Grid-point reference nodes are created from a \ref yc_grid object
        instead of from a \ref yc_node_factory. */
    class yc_node_factory {
    public:
        virtual ~yc_node_factory() {}

        /// Create a step-index node.
        /**
           Create a variable to be used to index grids in the
           solution-step dimension.
           The name usually describes time, e.g. "t". 
           @returns Pointer to new \ref yc_index_node object.
        */
        virtual yc_index_node_ptr
        new_step_index(const std::string& name
                       /**< [in] Step dimension name. */ );

        /// Create a domain-index node.
        /**
           Create a variable to be used to index grids in the
           solution-domain dimension.
           The name usually describes spatial dimensions, e.g. "x" or "y",
           but it can be any dimension that is specified at run-time,
           such as an index into a number of parallel problems
           being solved simultaneously.

           @note This should *not* include the step dimension, which is specified via
           new_step_index().
           @returns Pointer to new \ref yc_index_node object.
        */
        virtual yc_index_node_ptr
        new_domain_index(const std::string& name
                         /**< [in] Domain index name. */ );
        
        /// Create a new miscellaneous index.
        /**
           Create an variable to be used to index grids in the
           some dimension that is not the step dimension
           or a domain dimension.
           The value of these indices are normally compile-time
           constants, e.g., a fixed index into an array.
           @returns Pointer to new \ref yc_index_node object.
        */
        virtual yc_index_node_ptr
        new_misc_index(const std::string& name
                       /**< [in] Index name. */ );
        
        /// Create an equation node.
        /** Indicates grid point on LHS is equivalent to expression on
            RHS. This is NOT a test for equality.  When an equation is
            created, it is automatically added to the list of equations for
            the yc_solution that contains the grid that is on the
            LHS.

            An optional condition may be provided to define the sub-domain
            to which this equation applies. See new_first_domain_index()
            for more information and an example.
            Conditions are always evaluated with respect to the overall
            problem domain, i.e., independent of any specific
            MPI domain decomposition that might occur at run-time.
            If a condition is not provided, the equation applies to the
            entire problem domain.
            A condition can be added to an equation after its creation
            via yc_equation_node.set_cond().

            @returns Pointer to new \ref yc_equation_node object. 
        */
        virtual yc_equation_node_ptr
        new_equation_node(yc_grid_point_node_ptr lhs
                          /**< [in] Grid-point before EQUALS operator. */,
                          yc_number_node_ptr rhs
                          /**< [in] Expression after EQUALS operator. */,
                          yc_bool_node_ptr cond = nullptr
                          /**< [in] Optional expression defining sub-domain
                             where `lhs EQUALS rhs` is valid. */ );

        /// Create a constant numerical value node.
        /** 
            This is unary negation.
            Use new_subtraction_node() for binary `-`.
            @returns Pointer to new \ref yc_const_number_node object. 
        */
        virtual yc_const_number_node_ptr
        new_const_number_node(double val /**< [in] Value to store in node. */ );

        ///
        /**
           Integer version of new_const_number_node(double).
           @returns Pointer to new \ref yc_const_number_node object. 
        */
        virtual yc_const_number_node_ptr
        new_const_number_node(idx_t val /**< [in] Value to store in node. */ );

        /// Create a numerical negation operator node.
        /**
            New negation nodes can also be created via the overloaded unary `-` operator.
            @returns Pointer to new \ref yc_negate_node object. 
        */
        virtual yc_negate_node_ptr
        new_negate_node(yc_number_node_ptr rhs /**< [in] Expression after `-` sign. */ );

        /// Create an addition node.
        /** 
            Nodes must be created with at least two operands, and more can
            be added by calling add_operand() on the returned node.

            New addition nodes can also be created via the overloaded `+` operator.
            @returns Pointer to new \ref yc_add_node object. 
        */
        virtual yc_add_node_ptr
        new_add_node(yc_number_node_ptr lhs /**< [in] Expression before `+` sign. */,
                     yc_number_node_ptr rhs /**< [in] Expression after `+` sign. */ );

        /// Create a multiplication node.
        /**
           Nodes must be created with at least two operands, and more can
           be added by calling add_operand() on the returned node.

            New multiplication nodes can also be created via the overloaded `*` operator.
           @returns Pointer to new \ref yc_multiply_node object. 
        */
        virtual yc_multiply_node_ptr
        new_multiply_node(yc_number_node_ptr lhs /**< [in] Expression before `*` sign. */,
                          yc_number_node_ptr rhs /**< [in] Expression after `*` sign. */ );

        /// Create a subtraction node.
        /**
           This is binary subtraction.
           Use new_negation_node() for unary `-`.

            New subtraction nodes can also be created via the overloaded `-` operator.
           @returns Pointer to new \ref yc_subtract_node object. 
        */
        virtual yc_subtract_node_ptr
        new_subtract_node(yc_number_node_ptr lhs /**< [in] Expression before `-` sign. */,
                          yc_number_node_ptr rhs /**< [in] Expression after `-` sign. */ );

        /// Create a division node.
        /**
            New division nodes can also be created via the overloaded `/` operator.
           @returns Pointer to new \ref yc_divide_node object. 
        */
        virtual yc_divide_node_ptr
        new_divide_node(yc_number_node_ptr lhs /**< [in] Expression before `/` sign. */,
                        yc_number_node_ptr rhs /**< [in] Expression after `/` sign. */ );

        /// Create a symbol for the first index value in a given dimension.
        /**
           Create an expression that indicates the first value in the overall problem
           domain in `dim` dimension.
           The `dim` argument is created via new_domain_index().

           Typical C++ usage:

           \code{.cpp}
           auto x = node_fac.new_domain_index("x");

           // Create boolean expression for the
           // boundary sub-domain "x < first_x + 10".
           auto first_x = node_fac.new_first_domain_index(x);
           auto left_bc_cond = node_fac.new_less_than_node(x, first_x + 10);

           // Create a new equation that is valid in this range.
           auto left_bc_eq = 
             node_fac.new_equation_node(grid_pt_expr, left_bc_expr, left_bc_cond);
           \endcode

           Specification of the "interior" part of a 2-D domain could be
           represented by an expression similar to
           `x >= new_first_domain_index(x) + 20 &&
           x <= new_last_domain_index(x) - 20 &&
           y >= new_first_domain_index(y) + 20 &&
           y <= new_last_domain_index(y) - 20`.

           @note The entire domain in dimension "x" would be represented by
           `x >= new_first_domain_index(x) && x <= new_last_domain_index(x)`, but
           that is the default condition so does not need to be specified.

           @returns Pointer to new \ref yc_index_node object.
        */
        virtual yc_number_node_ptr
        new_first_domain_index(yc_index_node_ptr idx
                               /**< [in] Domain index. */ );
        
        /// Create a symbol for the last index value in a given dimension.
        /**
           Create an expression that indicates the last value in the overall problem
           domain in `dim` dimension.
           The `dim` argument is created via new_domain_index().

           @returns Pointer to new \ref yc_index_node object.
        */
        virtual yc_number_node_ptr
        new_last_domain_index(yc_index_node_ptr idx
                              /**< [in] Domain index. */ );
        
        /// Create a binary inverse operator node.
        /**
           @returns Pointer to new \ref yc_not_node object. 
        */
        virtual yc_not_node_ptr
        new_not_node(yc_bool_node_ptr rhs /**< [in] Expression after `!` sign. */ );

        /// Create a boolean 'and' node.
        /**
           @returns Pointer to new \ref yc_and_node object.
        */
        virtual yc_and_node_ptr
        new_and_node(yc_bool_node_ptr lhs /**< [in] Expression before `&&` sign. */,
                     yc_bool_node_ptr rhs /**< [in] Expression after `&&` sign. */ );

        /// Create a boolean 'or' node.
        /**
           @returns Pointer to new \ref yc_or_node object.
        */
        virtual yc_or_node_ptr
        new_or_node(yc_bool_node_ptr lhs /**< [in] Expression before `||` sign. */,
                    yc_bool_node_ptr rhs /**< [in] Expression after `||` sign. */ );

        /// Create a numerical-comparison 'equals' node.
        /**
           @returns Pointer to new \ref yc_equals_node object.
        */
        virtual yc_equals_node_ptr
        new_equals_node(yc_number_node_ptr lhs /**< [in] Expression before `==` sign. */,
                        yc_number_node_ptr rhs /**< [in] Expression after `==` sign. */ );

        /// Create a numerical-comparison 'not-equals' node.
        /**
           @returns Pointer to new \ref yc_not_equals_node object.
        */
        virtual yc_not_equals_node_ptr
        new_not_equals_node(yc_number_node_ptr lhs /**< [in] Expression before `!=` sign. */,
                            yc_number_node_ptr rhs /**< [in] Expression after `!=` sign. */ );

        /// Create a numerical-comparison 'less-than' node.
        /**
           @returns Pointer to new \ref yc_less_than_node object.
        */
        virtual yc_less_than_node_ptr
        new_less_than_node(yc_number_node_ptr lhs /**< [in] Expression before `<` sign. */,
                           yc_number_node_ptr rhs /**< [in] Expression after `<` sign. */ );

        /// Create a numerical-comparison 'greater-than' node.
        /**
           @returns Pointer to new \ref yc_greater_than_node object.
        */
        virtual yc_greater_than_node_ptr
        new_greater_than_node(yc_number_node_ptr lhs /**< [in] Expression before `>` sign. */,
                              yc_number_node_ptr rhs /**< [in] Expression after `>` sign. */ );

        /// Create a numerical-comparison 'greater-than or equals' node.
        /**
           @returns Pointer to new \ref yc_not_less_than_node object.
        */
        virtual yc_not_less_than_node_ptr
        new_not_less_than_node(yc_number_node_ptr lhs /**< [in] Expression before `>=` sign. */,
                               yc_number_node_ptr rhs /**< [in] Expression after `>=` sign. */ );

        /// Create a numerical-comparison 'less-than or equals' node.
        /**
           @returns Pointer to new \ref yc_not_greater_than_node object.
        */
        virtual yc_not_greater_than_node_ptr
        new_not_greater_than_node(yc_number_node_ptr lhs /**< [in] Expression before `<=` sign. */,
                                  yc_number_node_ptr rhs /**< [in] Expression after `<=` sign. */ );

    };

    /// Base class for all AST nodes.
    /** An object of this abstract type cannot be created. */
    class yc_expr_node {
    public:
        virtual ~yc_expr_node() {}

        /// Create a simple human-readable string.
        /**
           Formats the expression starting at this node.
           @returns String containing a single-line human-readable version of the expression.
        */
        virtual std::string format_simple() const =0;

        /// Count the size of the AST.
        /**
           @returns Number of nodes in this tree,
           including this node and all its descendants. 
        */
        virtual int get_num_nodes() const =0;
    };

    /// Equation node.
    /** Indicates grid point on LHS is equivalent to expression
        on RHS. This is NOT a test for equality.
        Created via yc_node_factory::new_equation_node().
    */
    class yc_equation_node : public virtual yc_expr_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Grid-point node appearing before the EQUALS operator. */
        virtual yc_grid_point_node_ptr get_lhs() =0;
    
        /// Get the right-hand-side operand.
        /** @returns Expression node appearing after the EQUALS operator. */
        virtual yc_number_node_ptr get_rhs() =0;
    
        /// Get the condition describing the sub-domain.
        /** @returns Boolean expression describing sub-domain or
            `nullptr` if not defined. */
        virtual yc_bool_node_ptr get_cond() =0;
    
        /// Set the condition describing the sub-domain.
        /** See yc_node_factory::new_equation_node(). */
        virtual void set_cond(yc_bool_node_ptr cond
                              /**< [in] Boolean expression describing the sub-domain
                                 or `nullptr` to remove the condition. */ ) =0;
    };

    /// Base class for all numerical AST nodes.
    /** An object of this abstract type cannot be created. */
    class yc_number_node : public virtual yc_expr_node {
    public:
        
        /// Create a deep copy of AST starting with this node.
        virtual yc_number_node_ptr clone_ast() const =0;
    };

    /// Base class for all boolean AST nodes.
    /** An object of this abstract type cannot be created. */
    class yc_bool_node : public virtual yc_expr_node { };

    /// A dimension or an index in that dimension.
    /**
       This is a leaf node in an AST.
       Created via yc_node_factory::new_step_index(),
       yc_node_factory::new_domain_index(), and
       yc_node_factory::new_misc_index().
    */
    class yc_index_node : public virtual yc_number_node {
    public:

        /// Get the dimension's name.
        /** @returns Name given at creation. */
        virtual const std::string&
        get_name() const =0;
    };

    /// A reference to a point in a grid.
    /**
       Created via yc_grid::new_relative_grid_point().
    */
    class yc_grid_point_node : public virtual yc_number_node {
    public:

        /// Get the grid this point is in.
        /** @returns Pointer to a \ref yc_grid object. */
        virtual yc_grid_ptr
        get_grid() =0;
    };
    
    /// A constant numerical value.
    /** All values are stored as doubles.
        This is a leaf node in an AST.
        Created via yc_node_factory::new_const_number_node().
    */
    class yc_const_number_node : public virtual yc_number_node {
    public:

        /// Set the value.
        /** The value is considered "constant" only when the 
            compiler output is created. It can be changed in the AST. */
        virtual void
        set_value(double val /**< [in] Value to store in node. */ ) =0;

        /// Get the stored value.
        /** @returns Copy of stored value. */
        virtual double
        get_value() const =0;
    };

    /// A numerical negation operator.
    /** Example: used to implement -(a*b).
        Created via yc_node_factory::new_negate_node().
    */
    class yc_negate_node : public virtual yc_number_node {
    public:

        /// Get the [only] operand.
        /**  This node implements unary negation only, not subtraction, so there is
             never a left-hand-side.
             @returns Expression node on right-hand-side of `-` sign. */
        virtual yc_number_node_ptr
        get_rhs() =0;
    };

    /// Base class for commutative numerical operators.
    /** This is used for operators whose arguments can be rearranged
        mathematically, e.g., add and multiply. */
    class yc_commutative_number_node : public virtual yc_number_node {
    public:

        /// Get the number of operands.
        /** If there is just one operand, the operation itself is moot.  If
            there are more than one operand, the operation applies between
            them. Example: for an add operator, if the operands are `a`,
            `b`, and `c`, then the expression is `a + b + c`.
            @returns Number of operands. */
        virtual int
        get_num_operands() =0;

        /// Get a list of the operands.
        /** @returns Vector of pointers to all operand nodes. */
        virtual std::vector<yc_number_node_ptr>
        get_operands() =0;

        /// Add an operand.
        virtual void
        add_operand(yc_number_node_ptr node /**< [in] Top node of AST to add. */ ) =0;
    };

    /// An addition node.
    /** Created via yc_node_factory::new_negate_node(). */
    class yc_add_node : public virtual yc_commutative_number_node { };

    /// A multiplication node.
    /** Created via yc_node_factory::new_multiply_node(). */
    class yc_multiply_node : public virtual yc_commutative_number_node { };

    /// A subtraction node.
    /** Created via yc_node_factory::new_subtract_node(). */
    class yc_subtract_node : public virtual yc_number_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Pointer to expression node appearing before the `-` sign. */
        virtual yc_number_node_ptr
        get_lhs() =0;
    
        /// Get the right-hand-side operand.
        /** @returns Pointer to expression node appearing after the `-` sign. */
        virtual yc_number_node_ptr
        get_rhs() =0;
    };

    /// A division node.
    /** Created via yc_node_factory::new_divide_node(). */
    class yc_divide_node : public virtual yc_number_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Pointer to expression node appearing before the `/` sign. */
        virtual yc_number_node_ptr
        get_lhs() =0;
    
        /// Get the right-hand-side operand.
        /** @returns Pointer to expression node appearing after the `/` sign. */
        virtual yc_number_node_ptr
        get_rhs() =0;
    };

    /// A boolean inversion operator.
    /** Example: used to implement `!(a || b)`.
        Created via yc_node_factory::new_not_node().
    */
    class yc_not_node : public virtual yc_bool_node {
    public:

        /// Get the [only] operand.
        /** @returns Expression node on right-hand-side of `!` sign. */
        virtual yc_bool_node_ptr
        get_rhs() =0;
    };

    /// A boolean 'and' operator.
    /** Example: used to implement `a && b`.
        Created via yc_node_factory::new_and_node().
    */
    class yc_and_node : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Expression node on left-hand-side of `&&` sign. */
        virtual yc_bool_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** @returns Expression node on right-hand-side of `&&` sign. */
        virtual yc_bool_node_ptr
        get_rhs() =0;
    };

    /// A boolean 'or' operator.
    /** Example: used to implement `a || b`.
        Created via yc_node_factory::new_or_node().
    */
    class yc_or_node : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Expression node on left-hand-side of `||` sign. */
        virtual yc_bool_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** @returns Expression node on right-hand-side of `||` sign. */
        virtual yc_bool_node_ptr
        get_rhs() =0;
    };

    /// A numerical-comparison 'equals' operator.
    /** Example: used to implement `a == b`.
        Created via yc_node_factory::new_equals_node().
    */
    class yc_equals_node : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Expression node on left-hand-side of `==` sign. */
        virtual yc_number_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** @returns Expression node on right-hand-side of `==` sign. */
        virtual yc_number_node_ptr
        get_rhs() =0;
    };

    /// A numerical-comparison 'not_equals' operator.
    /** Example: used to implement `a != b`.
        Created via yc_node_factory::new_not_equals_node().
    */
    class yc_not_equals_node : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Expression node on left-hand-side of `!=` sign. */
        virtual yc_number_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** @returns Expression node on right-hand-side of `!=` sign. */
        virtual yc_number_node_ptr
        get_rhs() =0;
    };

    /// A numerical-comparison 'less_than' operator.
    /** Example: used to implement `a < b`.
        Created via yc_node_factory::new_less_than_node().
    */
    class yc_less_than_node : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Expression node on left-hand-side of `<` sign. */
        virtual yc_number_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** @returns Expression node on right-hand-side of `<` sign. */
        virtual yc_number_node_ptr
        get_rhs() =0;
    };

    /// A numerical-comparison 'greater_than' operator.
    /** Example: used to implement `a > b`.
        Created via yc_node_factory::new_greater_than_node().
    */
    class yc_greater_than_node : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Expression node on left-hand-side of `>` sign. */
        virtual yc_number_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** @returns Expression node on right-hand-side of `>` sign. */
        virtual yc_number_node_ptr
        get_rhs() =0;
    };


    /// A numerical-comparison 'not_less_than' operator.
    /** Example: used to implement `a >= b`.
        Created via yc_node_factory::new_not_less_than_node().
    */
    class yc_not_less_than_node : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Expression node on left-hand-side of `>=` sign. */
        virtual yc_number_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** @returns Expression node on right-hand-side of `>=` sign. */
        virtual yc_number_node_ptr
        get_rhs() =0;
    };

    /// A numerical-comparison 'not_greater_than' operator.
    /** Example: used to implement `a <= b`.
        Created via yc_node_factory::new_not_greater_than_node().
    */
    class yc_not_greater_than_node : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Expression node on left-hand-side of `<=` sign. */
        virtual yc_number_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** @returns Expression node on right-hand-side of `<=` sign. */
        virtual yc_number_node_ptr
        get_rhs() =0;
    };

    // Non-class operators.
    // These are only defined if the older "internal DSL" is not used.
    // The internal version will eventually be deprecated and
    // perhaps removed in favor of this API.
    
#ifndef USE_INTERNAL_DSL

    /// Operator version of yc_node_factory::new_negation_node().
    yc_negate_node_ptr operator-(yc_number_node_ptr rhs);

    //@{
    /// Operator version of yc_node_factory::new_addition_node().
    yc_add_node_ptr operator+(yc_number_node_ptr lhs, yc_number_node_ptr rhs);
    yc_add_node_ptr operator+(double lhs, yc_number_node_ptr rhs);
    yc_add_node_ptr operator+(yc_number_node_ptr lhs, double rhs);
    //@}

    //@{
    /// Operator version of yc_node_factory::new_division_node().
    yc_divide_node_ptr operator/(yc_number_node_ptr lhs, yc_number_node_ptr rhs);
    yc_divide_node_ptr operator/(double lhs, yc_number_node_ptr rhs);
    yc_divide_node_ptr operator/(yc_number_node_ptr lhs, double rhs);
    //@}

    //@{
    /// Operator version of yc_node_factory::new_multiplication_node().
    yc_multiply_node_ptr operator*(yc_number_node_ptr lhs, yc_number_node_ptr rhs);
    yc_multiply_node_ptr operator*(double lhs, yc_number_node_ptr rhs);
    yc_multiply_node_ptr operator*(yc_number_node_ptr lhs, double rhs);
    //@}

    //@{
    /// Operator version of yc_node_factory::new_subtraction_node().
    yc_subtract_node_ptr operator-(yc_number_node_ptr lhs, yc_number_node_ptr rhs);
    yc_subtract_node_ptr operator-(double lhs, yc_number_node_ptr rhs);
    yc_subtract_node_ptr operator-(yc_number_node_ptr lhs, double rhs);
    //@}
#endif
    
    /** @}*/

} // namespace yask.

#endif
