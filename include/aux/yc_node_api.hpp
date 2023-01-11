/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

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

// This file uses Doxygen markup for API documentation-generation.
// See https://www.doxygen.nl/manual/index.html.
/** @file yc_node_api.hpp */

#pragma once

namespace yask {

    /**
     * \addtogroup yc
     * @{
     */

    // More node types not exposed except via RTTI.

    class yc_const_number_node;
    /// Shared pointer to \ref yc_const_number_node
    typedef std::shared_ptr<yc_const_number_node> yc_const_number_node_ptr;

    class yc_negate_node;
    /// Shared pointer to \ref yc_negate_node
    typedef std::shared_ptr<yc_negate_node> yc_negate_node_ptr;

    class yc_commutative_number_node;
    /// Shared pointer to \ref yc_commutative_number_node
    typedef std::shared_ptr<yc_commutative_number_node> yc_commutative_number_node_ptr;

    class yc_binary_number_node;
    /// Shared pointer to \ref yc_binary_number_node
    typedef std::shared_ptr<yc_binary_number_node> yc_binary_number_node_ptr;

    class yc_binary_bool_node;
    /// Shared pointer to \ref yc_binary_bool_node
    typedef std::shared_ptr<yc_binary_bool_node> yc_binary_bool_node_ptr;

    class yc_binary_comparison_node;
    /// Shared pointer to \ref yc_binary_comparison_node
    typedef std::shared_ptr<yc_binary_comparison_node> yc_binary_comparison_node_ptr;

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

    class yc_mod_node;
    /// Shared pointer to \ref yc_mod_node
    typedef std::shared_ptr<yc_mod_node> yc_mod_node_ptr;

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
    /** Indicates var point on LHS is equivalent to expression
        on RHS. This is NOT a test for equality.
        Created via yc_node_factory::new_equation_node().
    */
    class yc_equation_node : public virtual yc_expr_node {
    public:

        /// Get the left-hand-side operand.
        /** @returns Var-point node appearing before the EQUALS operator. */
        virtual yc_var_point_node_ptr get_lhs() =0;

        /// Get the right-hand-side operand.
        /** @returns Expression node appearing after the EQUALS operator. */
        virtual yc_number_node_ptr get_rhs() =0;

        /// Get the condition describing the sub-domain.
        /** @returns Boolean expression describing sub-domain or
            `nullptr` if not defined. */
        virtual yc_bool_node_ptr get_cond() =0;

        /// Set the condition describing the sub-domain for this equation.
        /**
           See yc_node_factory::new_equation_node() for an overall description
           of conditions.
        
           Typical C++ usage to create a sub-domain condition:

           \code{.cpp}
           auto x = node_fac.new_domain_index("x");

           // Create boolean expression for a 10-point wide left boundary area.
           auto first_x = node_fac.new_first_domain_index(x);
           auto left_bc_cond = x < first_x + 10;

           // Indicate that an expression is valid only in this area.
           // (Assumes left_bc_expr was already defined.)
           left_bc_expr.set_cond(left_bc_cond);
           \endcode

           Specification of the "interior" part of a 2-D domain could be
           represented by an expression like
           `(x >= node_fac.new_first_domain_index(x) + 20) && (x <= node_fac.new_last_domain_index(x) - 20) && (y >= node_fac.new_first_domain_index(y) + 20) && (y <= node_fac.new_last_domain_index(y) - 20)`.

           @warning For performance, sub-domain expressions are only
           evaluated once when yk_solution::prepare_solution() is called,
           and the results are analyzed and cached internally.  Thus,
           sub-domain expressions should not include a step index or a
           reference to any other varible that might change during or
           between time-steps. See set_step_cond() for the mechanism to
           enable equations based on variables that can change between
           time-steps.

           @note The entire domain in dimension "x" would be represented by
           `(x >= node_fac.new_first_domain_index(x)) && (x <= node_fac.new_last_domain_index(x))`, but
           that is the default condition so does not need to be specified.

           @note Be sure to use an expression like `x < first_x + 10`
           instead of merely `x < 10` to avoid the assumption that
           the first index is always zero (0). More importantly, use
           an expression like `x > last_x - 10` instead of hard-coding
           the last index.
        */
        virtual void set_cond(yc_bool_node_ptr sub_domain_cond
                              /**< [in] Boolean expression describing
                                 where in the sub-domain this expression is valid
                                 or `nullptr` to remove the condition. */ ) =0;

        /// Set the condition describing when the equation is valid.
        /**
           See yc_node_factory::new_equation_node() for an overall description
           of conditions.
        
           Typical C++ usage to create a step condition:

           \code{.cpp}
           auto t = node_fac.new_step_index("t");

           // Create boolean expression that is true every third step.
           auto my_step_cond = (t % 3 == 0);

           // Indicate that an expression is valid only when step_cond is true.
           // (Assumes my_expr was already defined.)
           my_expr.set_step_cond(my_step_cond);
           \endcode

           Step conditions may also refer to elements in variables including
           scalars (1-D) and arrays (2-D). For non-scalar variables, indices
           used in a step condition _cannot_ include domain variables like `x` or `y`, but 
           constants are allowed. In this way, equations can be enabled or
           disabled programmatically by setting elements in the tested variables.
        */
        virtual void set_step_cond(yc_bool_node_ptr step_cond
                                   /**< [in] Boolean expression describing 
                                      when the expression is valid
                                      or `nullptr` to remove the condition. */ ) =0;

        /// Create a deep copy of AST starting with this node.
        virtual yc_equation_node_ptr clone_ast() const =0;
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
    class yc_bool_node : public virtual yc_expr_node {
    public:

        /// Create a deep copy of AST starting with this node.
        virtual yc_bool_node_ptr clone_ast() const =0;
    };

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

    /// A reference to a point in a var.
    /**
       Created via yc_var::new_var_point() or yc_var::new_relative_var_point().
    */
    class yc_var_point_node : public virtual yc_number_node {
    public:

        /// Get the var this point is in.
        /** @returns Pointer to a \ref yc_var object. */
        virtual yc_var_ptr
        get_var() =0;

        /// **[Deprecated]** Use get_var().
        YASK_DEPRECATED
        inline yc_var_ptr
        get_grid() {
            return get_var();
        }
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
    /** Created via yc_node_factory::new_add_node(). */
    class yc_add_node : public virtual yc_commutative_number_node { };

    /// A multiplication node.
    /** Created via yc_node_factory::new_multiply_node(). */
    class yc_multiply_node : public virtual yc_commutative_number_node { };

    /// Base class for numerical binary operators.
    /** A \ref yc_commutative_number_node is used instead for
        add and multiply. */
    class yc_binary_number_node : public virtual yc_number_node {
    public:

        /// Get the left-hand-side operand.
        virtual yc_number_node_ptr
        get_lhs() =0;

        /// Get the right-hand-side operand.
        virtual yc_number_node_ptr
        get_rhs() =0;
    };

    /// A subtraction node.
    /** Created via yc_node_factory::new_subtract_node(). */
    class yc_subtract_node : public virtual yc_binary_number_node { };

    /// A division node.
    /** Created via yc_node_factory::new_divide_node(). */
    class yc_divide_node : public virtual yc_binary_number_node { };

    /// A modulo node.
    /** Created via yc_node_factory::new_mod_node(). */
    class yc_mod_node : public virtual yc_binary_number_node { };

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

    /// Base class for boolean binary operators that take boolean inputs.
    class yc_binary_bool_node : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        virtual yc_bool_node_ptr
        get_lhs() =0;

        /// Get the right-hand-side operand.
        virtual yc_bool_node_ptr
        get_rhs() =0;
    };

    /// A boolean 'and' operator.
    /** Example: used to implement `a && b`.
        Created via yc_node_factory::new_and_node().
    */
    class yc_and_node : public virtual yc_binary_bool_node { };

    /// A boolean 'or' operator.
    /** Example: used to implement `a || b`.
        Created via yc_node_factory::new_or_node().
    */
    class yc_or_node : public virtual yc_binary_bool_node { };

    /// Base class for boolean binary operators that take numerical inputs.
    class yc_binary_comparison_node : public virtual yc_bool_node {
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

    /// A numerical-comparison 'equals' operator.
    /** Example: used to implement `a == b`.
        Created via yc_node_factory::new_equals_node().
    */
    class yc_equals_node : public virtual yc_binary_comparison_node { };

    /// A numerical-comparison 'not_equals' operator.
    /** Example: used to implement `a != b`.
        Created via yc_node_factory::new_not_equals_node().
    */
    class yc_not_equals_node : public virtual yc_binary_comparison_node { };

    /// A numerical-comparison 'less_than' operator.
    /** Example: used to implement `a < b`.
        Created via yc_node_factory::new_less_than_node().
    */
    class yc_less_than_node : public virtual yc_binary_comparison_node { };

    /// A numerical-comparison 'greater_than' operator.
    /** Example: used to implement `a > b`.
        Created via yc_node_factory::new_greater_than_node().
    */
    class yc_greater_than_node : public virtual yc_binary_comparison_node { };

    /// A numerical-comparison 'not_less_than' operator.
    /** Example: used to implement `a >= b`.
        Created via yc_node_factory::new_not_less_than_node().
    */
    class yc_not_less_than_node : public virtual yc_binary_comparison_node { };

    /// A numerical-comparison 'not_greater_than' operator.
    /** Example: used to implement `a <= b`.
        Created via yc_node_factory::new_not_greater_than_node().
    */
    class yc_not_greater_than_node : public virtual yc_binary_comparison_node { };

#ifndef SWIG
    /// Arguments that may be YASK numeric pointer types.
    /**
       A simple wrapper class to provide automatic construction of a
       'yc_number_node_ptr' from other YASK pointer types.

       Used only to provide conversions in function arguments.

       Not a virtual class.

       @note Not used in Python API.
    */
    class yc_number_ptr_arg : public yc_number_node_ptr {

    public:

        /// Arg can be a number-node pointer.
        yc_number_ptr_arg(yc_number_node_ptr p) :
            yc_number_node_ptr(p) { }

        /// Arg can be an index-node pointer.
        yc_number_ptr_arg(yc_index_node_ptr p) :
            yc_number_node_ptr(p) { }

        /// Arg can be a var-point-node pointer.
        yc_number_ptr_arg(yc_var_point_node_ptr p) :
            yc_number_node_ptr(p) { }
    };

    /// Arguments that may be non-YASK numeric types.
    /**
       A simple wrapper class to provide automatic construction of a
       'yc_number_node_ptr' from non-YASK fundamental numeric types.

       Used only to provide conversions in function arguments.

       Not a virtual class.

       @note Not used in Python API.
    */
    class yc_number_const_arg : public yc_number_node_ptr {

    protected:
        
        /// Create an argument from a constant value.
        yc_number_node_ptr _convert_const(double val) const;

    public:

        /// Arg can be an index type.
        yc_number_const_arg(idx_t i) :
            yc_number_node_ptr(_convert_const(i)) { }

        /// Arg can be an int.
        yc_number_const_arg(int i) :
            yc_number_node_ptr(_convert_const(i)) { }

        /// Arg can be a double.
        yc_number_const_arg(double f) :
            yc_number_node_ptr(_convert_const(f)) { }

        /// Arg can be a float.
        yc_number_const_arg(float f) :
            yc_number_node_ptr(_convert_const(f)) { }
    };

    /// Arguments that may be YASK or non-YASK numeric types.
    /**
       A simple wrapper class to provide automatic construction of a
       'yc_number_node_ptr' from a YASK pointer or non-YASK fundamental
       numeric types.

       Used only to provide conversions in function arguments.

       Not a virtual class.

       @note Not used in Python API.
    */
    class yc_number_any_arg : public yc_number_node_ptr {

    protected:
        
        /// Create an argument from a constant value.
        yc_number_node_ptr _convert_const(double val) const;

    public:

        /// Arg can be a number-node pointer.
        yc_number_any_arg(yc_number_node_ptr p) :
            yc_number_node_ptr(p) { }

        /// Arg can be an index-node pointer.
        yc_number_any_arg(yc_index_node_ptr p) :
            yc_number_node_ptr(p) { }

        /// Arg can be a var-point-node pointer.
        yc_number_any_arg(yc_var_point_node_ptr p) :
            yc_number_node_ptr(p) { }

        /// Arg can be an index type.
        yc_number_any_arg(idx_t i) :
            yc_number_node_ptr(_convert_const(i)) { }

        /// Arg can be an int.
        yc_number_any_arg(int i) :
            yc_number_node_ptr(_convert_const(i)) { }

        /// Arg can be a double.
        yc_number_any_arg(double f) :
            yc_number_node_ptr(_convert_const(f)) { }

        /// Arg can be a float.
        yc_number_any_arg(float f) :
            yc_number_node_ptr(_convert_const(f)) { }

        /// Arg can be a null pointer.
        yc_number_any_arg(std::nullptr_t p) :
            yc_number_node_ptr(p) { }
    };
#endif
    
    /// Factory to create AST nodes.
    /** @note Var-point reference nodes are created from a \ref yc_var object
        instead of from a \ref yc_node_factory. */
    class yc_node_factory {
    public:
        virtual ~yc_node_factory() {}

        /// Create a step-index node.
        /**
           Create a variable to be used to index vars in the
           solution-step dimension.
           The name usually describes time, e.g. "t".
           @returns Pointer to new \ref yc_index_node object.
        */
        virtual yc_index_node_ptr
        new_step_index(const std::string& name
                       /**< [in] Step dimension name. */ ) const;

        /// Create a domain-index node.
        /**
           Create a variable to be used to index vars in the
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
                         /**< [in] Domain index name. */ ) const;

        /// Create a new miscellaneous index.
        /**
           Create an variable to be used to index vars in the
           some dimension that is not the step dimension
           or a domain dimension.
           The value of these indices are normally compile-time
           constants, e.g., a fixed index into an array.
           @returns Pointer to new \ref yc_index_node object.
        */
        virtual yc_index_node_ptr
        new_misc_index(const std::string& name
                       /**< [in] Index name. */ ) const;

        /// Create an equation node.
        /** Indicates var point on LHS is equivalent to expression on
            RHS. This is NOT a test for equality.  When an equation is
            created, it is automatically added to the list of equations for
            the yc_solution that contains the var that is on the
            LHS.

            An optional domain condition may be provided to define the sub-domain
            to which this equation applies. 
            Domain conditions are always evaluated with respect to the overall
            problem domain, i.e., independent of any specific
            MPI domain decomposition that might occur at run-time.
            If a domain condition is not provided, the equation applies to the
            entire problem domain.
            A domain condition can be added to an equation after its creation
            via yc_equation_node.set_cond().
            See yc_equation_node.set_cond() for more information and an example.

            A step condition is similar to a domain condition, but
            enables or disables the entire equation based on the current step (usually time)
            and/or other values.
            A step condition can only be added to an equation after its creation
            via yc_equation_node.set_step_cond().
            See yc_equation_node.set_step_cond() for more information and an example.

            @returns Pointer to new \ref yc_equation_node object.
        */
        virtual yc_equation_node_ptr
        new_equation_node(yc_var_point_node_ptr lhs
                          /**< [in] Var-point before EQUALS operator. */,
                          yc_number_node_ptr rhs
                          /**< [in] Expression after EQUALS operator. */,
                          yc_bool_node_ptr sub_domain_cond = nullptr
                          /**< [in] Optional expression defining sub-domain
                             where `lhs EQUALS rhs` is valid. */ ) const;

#ifndef SWIG
        /// Create a numerical-value expression node.
        /**
           A generic method to create a pointer to a numerical expression
           from any type supported by \ref yc_number_any_arg constructors.
           @note Not available in Python API. Use a more explicit method.
        */
        virtual yc_number_node_ptr
        new_number_node(yc_number_any_arg arg
                        /**< [in] Argument to convert to a numerical expression. */) const {
            return std::move(arg);
        }
#endif
        
        /// Create a constant numerical-value node.
        /**
           Use to add a constant to an expression.
           The overloaded arithmetic operators allow `double` arguments,
           so in most cases, it is not necessary to call this directly.
           @returns Pointer to new \ref yc_const_number_node object.
        */
        virtual yc_number_node_ptr
        new_const_number_node(double val
                              /**< [in] Value to store in node. */ ) const;

        /// Create a constant numerical value node.
        /**
           Integer version of new_const_number_node(double).
           It may be necessary to cast other integer types to `idx_t` to
           avoid ambiguous overloading of this function.
           @returns Pointer to new \ref yc_const_number_node object.
        */
        virtual yc_number_node_ptr
        new_const_number_node(idx_t val
                              /**< [in] Value to store in node. */ ) const;

        /// Create a numerical negation operator node.
        /**
           This is the explicit form, which is usually not needed because
           new negation nodes can also be created via the overloaded unary `-` operator.
           @returns Pointer to new \ref yc_negate_node object.
        */
        virtual yc_number_node_ptr
        new_negate_node(yc_number_node_ptr rhs
                        /**< [in] Expression after `-` sign. */ ) const;

        /// Create an addition node.
        /**
           This is the explicit form, which is usually not needed because
           new addition nodes can also be created via the overloaded `+` operator.
           @returns Pointer to new \ref yc_add_node object. 
           Returns `rhs` if `lhs` is a null node pointer and vice-versa.
        */
        virtual yc_number_node_ptr
        new_add_node(yc_number_node_ptr lhs /**< [in] Expression before `+` sign. */,
                     yc_number_node_ptr rhs /**< [in] Expression after `+` sign. */ ) const;

        /// Create a multiplication node.
        /**
           This is the explicit form, which is usually not needed because
           new multiplication nodes can also be created via the overloaded `*` operator.
           @returns Pointer to new \ref yc_multiply_node object.
           Returns `rhs` if `lhs` is a null node pointer and vice-versa.
        */
        virtual yc_number_node_ptr
        new_multiply_node(yc_number_node_ptr lhs /**< [in] Expression before `*` sign. */,
                          yc_number_node_ptr rhs /**< [in] Expression after `*` sign. */ ) const;

        /// Create a subtraction node.
        /**
           This is binary subtraction.
           Use new_negate_node() for unary `-`.

           This is the explicit form, which is usually not needed because
           new subtraction nodes can also be created via the overloaded `-` operator.
           @returns Pointer to new \ref yc_subtract_node object.
           Returns `- rhs` if `lhs` is a null node pointer and
           `lhs` if `rhs` is null.
        */
        virtual yc_number_node_ptr
        new_subtract_node(yc_number_node_ptr lhs /**< [in] Expression before `-` sign. */,
                          yc_number_node_ptr rhs /**< [in] Expression after `-` sign. */ ) const;

        /// Create a division node.
        /**
           This is the explicit form, which is usually not needed because
           new division nodes can also be created via the overloaded `/` operator.
           @returns Pointer to new \ref yc_divide_node object.
           Returns `1.0 / rhs` if `lhs` is a null node pointer and
           `lhs` if `rhs` is null.
        */
        virtual yc_number_node_ptr
        new_divide_node(yc_number_node_ptr lhs /**< [in] Expression before `/` sign. */,
                        yc_number_node_ptr rhs /**< [in] Expression after `/` sign. */ ) const;

        /// Create a modulo node.
        /**
           This is the explicit form, which is usually not needed because
           new modulo nodes can also be created via the overloaded `%` operator.
           The modulo operator converts both operands to integers before performing
           the operation.
           @returns Pointer to new \ref yc_mod_node object.
        */
        virtual yc_number_node_ptr
        new_mod_node(yc_number_node_ptr lhs /**< [in] Expression before `%` sign. */,
                     yc_number_node_ptr rhs /**< [in] Expression after `%` sign. */ ) const;

        /// Create a symbol for the first index value in a given dimension.
        /**
           Create an expression that indicates the first value in the overall problem
           domain in `dim` dimension.
           The `dim` argument is created via new_domain_index().

           See yc_equation_node.set_cond() for more information and an example.

           @returns Pointer to new \ref yc_index_node object.
        */
        virtual yc_number_node_ptr
        new_first_domain_index(yc_index_node_ptr idx
                               /**< [in] Domain index. */ ) const;

        /// Create a symbol for the last index value in a given dimension.
        /**
           Create an expression that indicates the last value in the overall problem
           domain in `dim` dimension.
           The `dim` argument is created via new_domain_index().

           See yc_equation_node.set_cond() for more information and an example.

           @returns Pointer to new \ref yc_index_node object.
        */
        virtual yc_number_node_ptr
        new_last_domain_index(yc_index_node_ptr idx
                              /**< [in] Domain index. */ ) const;

        /// Create a binary inverse operator node.
        /**
           This is the explicit form, which is usually not needed because
           new "not" nodes can also be created via the overloaded `!` operator
           or the `yc_not` function in Python.
           @returns Pointer to new \ref yc_not_node object.
        */
        virtual yc_bool_node_ptr
        new_not_node(yc_bool_node_ptr rhs /**< [in] Expression after `!` sign. */ ) const;

        /// Create a boolean 'and' node.
        /**
           This is the explicit form, which is usually not needed because
           new "and" nodes can also be created via the overloaded `&&` operator
           or the `yc_and` function in Python.
           @returns Pointer to new \ref yc_and_node object.
        */
        virtual yc_bool_node_ptr
        new_and_node(yc_bool_node_ptr lhs /**< [in] Expression before `&&` sign. */,
                     yc_bool_node_ptr rhs /**< [in] Expression after `&&` sign. */ ) const;

        /// Create a boolean 'or' node.
        /**
           This is the explicit form, which is usually not needed because
           new "or" nodes can also be created via the overloaded `||` operator
           or the `yc_or` function in Python.
           @returns Pointer to new \ref yc_or_node object.
        */
        virtual yc_bool_node_ptr
        new_or_node(yc_bool_node_ptr lhs /**< [in] Expression before `||` sign. */,
                    yc_bool_node_ptr rhs /**< [in] Expression after `||` sign. */ ) const;

        /// Create a numerical-comparison 'equals' node.
        /**
           This is the explicit form, which is usually not needed because
           new "equals" nodes can also be created via the overloaded `==` operator.
           @returns Pointer to new \ref yc_equals_node object.
        */
        virtual yc_bool_node_ptr
        new_equals_node(yc_number_node_ptr lhs /**< [in] Expression before `==` sign. */,
                        yc_number_node_ptr rhs /**< [in] Expression after `==` sign. */ ) const;

        /// Create a numerical-comparison 'not-equals' node.
        /**
           This is the explicit form, which is usually not needed because
           new "not-equals" nodes can also be created via the overloaded `!=` operator.
           @returns Pointer to new \ref yc_not_equals_node object.
        */
        virtual yc_bool_node_ptr
        new_not_equals_node(yc_number_node_ptr lhs /**< [in] Expression before `!=` sign. */,
                            yc_number_node_ptr rhs /**< [in] Expression after `!=` sign. */ ) const;

        /// Create a numerical-comparison 'less-than' node.
        /**
           This is the explicit form, which is usually not needed because
           new "less-than" nodes can also be created via the overloaded `<` operator.
           @returns Pointer to new \ref yc_less_than_node object.
        */
        virtual yc_bool_node_ptr
        new_less_than_node(yc_number_node_ptr lhs /**< [in] Expression before `<` sign. */,
                           yc_number_node_ptr rhs /**< [in] Expression after `<` sign. */ ) const;

        /// Create a numerical-comparison 'greater-than' node.
        /**
           This is the explicit form, which is usually not needed because
           new "greater-than" nodes can also be created via the overloaded `>` operator.
           @returns Pointer to new \ref yc_greater_than_node object.
        */
        virtual yc_bool_node_ptr
        new_greater_than_node(yc_number_node_ptr lhs /**< [in] Expression before `>` sign. */,
                              yc_number_node_ptr rhs /**< [in] Expression after `>` sign. */ ) const;

        /// Create a numerical-comparison 'greater-than or equals' node.
        /**
           This is the explicit form, which is usually not needed because
           new "greater-than or equals" nodes can also be created via the overloaded `>=` operator.
           @returns Pointer to new \ref yc_not_less_than_node object.
        */
        virtual yc_bool_node_ptr
        new_not_less_than_node(yc_number_node_ptr lhs /**< [in] Expression before `>=` sign. */,
                               yc_number_node_ptr rhs /**< [in] Expression after `>=` sign. */ ) const;

        /// Create a numerical-comparison 'less-than or equals' node.
        /**
           This is the explicit form, which is usually not needed because
           new "less-than or equals" nodes can also be created via the overloaded `<=` operator.
           @returns Pointer to new \ref yc_not_greater_than_node object.
        */
        virtual yc_bool_node_ptr
        new_not_greater_than_node(yc_number_node_ptr lhs /**< [in] Expression before `<=` sign. */,
                                  yc_number_node_ptr rhs /**< [in] Expression after `<=` sign. */ ) const;

    };

    /// Unary math functions. Used internally to define sqrt(), sin(), etc.
#define UNARY_MATH_EXPR(fn_name) \
    yc_number_node_ptr fn_name(const yc_number_node_ptr rhs)

    /// Create an expression node to calculate the square-root of the argument node.
    UNARY_MATH_EXPR(sqrt);
    /// Create an expression node to calculate the cube-root of the argument node.
    UNARY_MATH_EXPR(cbrt);
    /// Create an expression node to calculate the absolute-value of the argument node.
    UNARY_MATH_EXPR(fabs);
    /// Create an expression node to calculate the error function of the argument node.
    UNARY_MATH_EXPR(erf);
    /// Create an expression node to calculate the natural exponent of the argument node.
    UNARY_MATH_EXPR(exp);
    /// Create an expression node to calculate the natural log of the argument node.
    UNARY_MATH_EXPR(log);
    /// Create an expression node to calculate the sine of the argument node.
    UNARY_MATH_EXPR(sin);
    /// Create an expression node to calculate the cosine of the argument node.
    UNARY_MATH_EXPR(cos);
    /// Create an expression node to calculate the arc-tangent of the argument node.
    UNARY_MATH_EXPR(atan);
#undef UNARY_MATH_EXPR

    /// Binary math functions. Used internally to define pow().
#define BINARY_MATH_EXPR(fn_name) \
    yc_number_node_ptr fn_name(const yc_number_node_ptr arg1, const yc_number_node_ptr arg2);   \
    yc_number_node_ptr fn_name(double arg1, const yc_number_node_ptr arg2); \
    yc_number_node_ptr fn_name(const yc_number_node_ptr arg1, double arg2)

    /// Power function.
    /**
       Create an expression node to calculate the first argument node raised to
       the power of the second argument node.
    */
    BINARY_MATH_EXPR(pow);
#undef BINARY_MATH_EXPR

#if !defined SWIG

    // Non-class operators.
    // These are not defined for SWIG because
    // the Python operators are defined in the ".i" file.
    // For the binary operators, we define 3 combinations to implicitly
    // avoid the const-const combinations, which conflict with built-in
    // operators on fundamental C++ types, e.g., '5+8'.

    /// Operator version of yc_node_factory::new_negate_node().
    yc_number_node_ptr operator-(yc_number_ptr_arg rhs);

    /// Operator version of yc_node_factory::new_add_node().
    yc_number_node_ptr operator+(yc_number_ptr_arg lhs, yc_number_ptr_arg rhs);
    /// Operator version of yc_node_factory::new_add_node().
    yc_number_node_ptr operator+(yc_number_const_arg lhs, yc_number_ptr_arg rhs);
    /// Operator version of yc_node_factory::new_add_node().
    yc_number_node_ptr operator+(yc_number_ptr_arg lhs, yc_number_const_arg rhs);

    /// Operator version of yc_node_factory::new_divide_node().
    yc_number_node_ptr operator/(yc_number_ptr_arg lhs, yc_number_ptr_arg rhs);
    /// Operator version of yc_node_factory::new_divide_node().
    yc_number_node_ptr operator/(yc_number_const_arg lhs, yc_number_ptr_arg rhs);
    /// Operator version of yc_node_factory::new_divide_node().
    yc_number_node_ptr operator/(yc_number_ptr_arg lhs, yc_number_const_arg rhs);

    /// Operator version of yc_node_factory::new_mod_node().
    yc_number_node_ptr operator%(yc_number_ptr_arg lhs, yc_number_ptr_arg rhs);
    /// Operator version of yc_node_factory::new_mod_node().
    yc_number_node_ptr operator%(yc_number_const_arg lhs, yc_number_ptr_arg rhs);
    /// Operator version of yc_node_factory::new_mod_node().
    yc_number_node_ptr operator%(yc_number_ptr_arg lhs, yc_number_const_arg rhs);

    /// Operator version of yc_node_factory::new_multiply_node().
    yc_number_node_ptr operator*(yc_number_ptr_arg lhs, yc_number_ptr_arg rhs);
    /// Operator version of yc_node_factory::new_multiply_node().
    yc_number_node_ptr operator*(yc_number_const_arg lhs, yc_number_ptr_arg rhs);
    /// Operator version of yc_node_factory::new_multiply_node().
    yc_number_node_ptr operator*(yc_number_ptr_arg lhs, yc_number_const_arg rhs);

    /// Operator version of yc_node_factory::new_subtract_node().
    yc_number_node_ptr operator-(yc_number_ptr_arg lhs, yc_number_ptr_arg rhs);
    /// Operator version of yc_node_factory::new_subtract_node().
    yc_number_node_ptr operator-(yc_number_const_arg lhs, yc_number_ptr_arg rhs);
    /// Operator version of yc_node_factory::new_subtract_node().
    yc_number_node_ptr operator-(yc_number_ptr_arg lhs, yc_number_const_arg rhs);

    /// Shortcut for creating expression A = A + B.
    void operator+=(yc_number_node_ptr& lhs, yc_number_node_ptr rhs);
    /// Shortcut for creating expression A = A + B.
    void operator+=(yc_number_node_ptr& lhs, yc_number_const_arg rhs);

    /// Shortcut for creating expression A = A - B.
    void operator-=(yc_number_node_ptr& lhs, yc_number_node_ptr rhs);
    /// Shortcut for creating expression A = A - B.
    void operator-=(yc_number_node_ptr& lhs, yc_number_const_arg rhs);

    /// Shortcut for creating expression A = A * B.
    void operator*=(yc_number_node_ptr& lhs, yc_number_node_ptr rhs);
    /// Shortcut for creating expression A = A * B.
    void operator*=(yc_number_node_ptr& lhs, yc_number_const_arg rhs);

    /// Shortcut for creating expression A = A / B.
    void operator/=(yc_number_node_ptr& lhs, yc_number_node_ptr rhs);
    /// Shortcut for creating expression A = A / B.
    void operator/=(yc_number_node_ptr& lhs, yc_number_const_arg rhs);

    /// Operator version of yc_node_factory::new_not_node().
    /** For Python, use `rhs.yc_not()` */
    yc_bool_node_ptr operator!(yc_bool_node_ptr rhs);

    /// Operator version of yc_node_factory::new_or_node().
    /** For Python, use `lhs.yc_or(rhs)` */
    yc_bool_node_ptr operator||(yc_bool_node_ptr lhs, yc_bool_node_ptr rhs);

    /// Operator version of yc_node_factory::new_and_node().
    /** For Python, use `lhs.yc_and(rhs)` */
    yc_bool_node_ptr operator&&(yc_bool_node_ptr lhs, yc_bool_node_ptr rhs);

    /// Binary numerical-to-boolean operators. Used internally to define `==`, `<`, etc.
    /**
       Must provide more explicit ptr-type operands than used with math
       operators to keep compiler from using built-in pointer comparison.
       Const values must be on RHS of operator, e.g., 'x > 5' is ok, but 
       '5 < x' is not.
    */
#define BOOL_OPER(oper, fn)                                             \
    inline yc_bool_node_ptr operator oper(const yc_number_node_ptr lhs, const yc_number_node_ptr rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, rhs); }               \
    inline yc_bool_node_ptr operator oper(const yc_number_node_ptr lhs, const yc_index_node_ptr rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, rhs); }               \
    inline yc_bool_node_ptr operator oper(const yc_number_node_ptr lhs, const yc_var_point_node_ptr rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, rhs); }               \
    inline yc_bool_node_ptr operator oper(const yc_index_node_ptr lhs, const yc_number_node_ptr rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, rhs); }               \
    inline yc_bool_node_ptr operator oper(const yc_index_node_ptr lhs, const yc_index_node_ptr rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, rhs); }               \
    inline yc_bool_node_ptr operator oper(const yc_index_node_ptr lhs, const yc_var_point_node_ptr rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, rhs); }               \
    inline yc_bool_node_ptr operator oper(const yc_var_point_node_ptr lhs, const yc_number_node_ptr rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, rhs); }               \
    inline yc_bool_node_ptr operator oper(const yc_var_point_node_ptr lhs, const yc_index_node_ptr rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, rhs); }               \
    inline yc_bool_node_ptr operator oper(const yc_var_point_node_ptr lhs, const yc_var_point_node_ptr rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, rhs); }               \
    inline yc_bool_node_ptr operator oper(const yc_number_node_ptr lhs, double rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, nfac.new_number_node(rhs)); } \
    inline yc_bool_node_ptr operator oper(const yc_index_node_ptr lhs, double rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, nfac.new_number_node(rhs)); } \
    inline yc_bool_node_ptr operator oper(const yc_var_point_node_ptr lhs, double rhs) { \
        yc_node_factory nfac; return nfac.fn(lhs, nfac.new_number_node(rhs)); }

    BOOL_OPER(==, new_equals_node)
    BOOL_OPER(!=, new_not_equals_node)
    BOOL_OPER(<, new_less_than_node)
    BOOL_OPER(>, new_greater_than_node)
    BOOL_OPER(<=, new_not_greater_than_node)
    BOOL_OPER(>=, new_not_less_than_node)
#undef BOOL_OPER

    /// Recommended macro to make the "equality" operator readable and self-explanatory.
    /**
       Uses an otherwise unneeded binary operator that has a lower priority
       than the math ops and a higher priority than the `IF_*` operators.
       See http://en.cppreference.com/w/cpp/language/operator_precedence.
       It is also required that this not be an operator that is defined for
       shared pointers.  See
       https://en.cppreference.com/w/cpp/memory/shared_ptr.
    */
#define EQUALS <<

    /// The operator version of yc_node_factory::new_equation_node() used for defining a var-point value.
    yc_equation_node_ptr operator EQUALS(yc_var_point_node_ptr gpp, const yc_number_any_arg rhs);

    /// Recommended macro to make the domain-condition operator readable and self-explanatory.
    /**
       Uses an otherwise unneeded binary operator that has a low priority.
       See http://en.cppreference.com/w/cpp/language/operator_precedence.
    */
#define IF_DOMAIN ^=

    /// The operator version of yc_equation_node::set_cond() to add a domain condition.
    yc_equation_node_ptr operator IF_DOMAIN(yc_equation_node_ptr expr,
                                            const yc_bool_node_ptr cond);

    /// Recommended macro to make the step-condition operator readable and self-explanatory.
    /**
       Uses an otherwise unneeded binary operator that has a low priority.
       See http://en.cppreference.com/w/cpp/language/operator_precedence.
    */
#define IF_STEP |=

    /// The operator version of yc_equation_node::set_step_cond() to add a domain condition.
    yc_equation_node_ptr operator IF_STEP(yc_equation_node_ptr expr,
                                          const yc_bool_node_ptr cond);

#endif  // !SWIG.

    /** @}*/

} // namespace yask.
