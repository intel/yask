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

namespace yask {

    // Forward declarations of AST nodes and pointers to them.
    class expr_node;
    typedef std::shared_ptr<expr_node> expr_node_ptr;
    class number_node;
    typedef std::shared_ptr<number_node> number_node_ptr;
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

    /// Factory to create AST nodes.
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

    /// A constant numerical value.
    /** All values are stored as doubles.
     * This is a leaf node in an AST.
     * Use a node_factory object to create an object of this type. */
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
     * Use a node_factory object to create an object of this type. */
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
