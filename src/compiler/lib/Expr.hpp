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

///////// AST Expressions ////////////

// TODO: break this up into several smaller files.

#pragma once

#include "yask_compiler_api.hpp"

#include <utility>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <cstdarg>
#include <fstream>

// Need g++ >= 4.9 for regex.
#if defined(__GNUC__) && !defined(__clang__)
#define GCC_VERSION (__GNUC__ * 10000       \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)
#if GCC_VERSION < 40900
#error G++ 4.9.0 or later is required
#endif
#endif
#include <regex>

// Common utilities.
#define CHECK
#include "common_utils.hpp"
#include "tuple.hpp"
#include "idiv.hpp"

using namespace std;

namespace yask {

    // Forward-decls of expressions.
    class Expr;
    typedef shared_ptr<Expr> expr_ptr;
    class NumExpr;
    typedef shared_ptr<NumExpr> num_expr_ptr;
    typedef vector<num_expr_ptr> num_expr_ptr_vec;
    class VarPoint;
    typedef shared_ptr<VarPoint> var_point_ptr;
    class IndexExpr;
    typedef shared_ptr<IndexExpr> index_expr_ptr;
    typedef vector<index_expr_ptr> index_expr_ptr_vec;
    class BoolExpr;
    typedef shared_ptr<BoolExpr> bool_expr_ptr;
    class EqualsExpr;
    typedef shared_ptr<EqualsExpr> equals_expr_ptr;

    // More forward-decls.
    class ExprVisitor;
    class Var;
    class StencilSolution;
    struct Dimensions;

    typedef map<string, string> VarMap; // map used when substituting vars.

    //// Classes to implement parts of expressions.

    // The base class for all expression nodes.
    // NB: there is no clone() method defined here; they
    // are defined on immediate derived types:
    // NumExpr, BoolExpr, and EqualsExpr.
    class Expr : public virtual yc_expr_node {

    public:
        Expr() { }
        virtual ~Expr() { }

        // For visitors.
        virtual string accept(ExprVisitor* ev) =0;
        virtual string accept(ExprVisitor* ev) const;

        // check for expression equivalency.
        // Does *not* check value equivalency except for
        // constants.
        virtual bool is_same(const Expr* other) const =0;
        virtual bool is_same(const expr_ptr& other) const {
            return is_same(other.get());
        }

        // Make pair if possible.
        // Return whether pair made.
        virtual bool make_pair(Expr* other) {
            return false;
        }
        virtual bool make_pair(expr_ptr other) {
            return make_pair(other.get());
        }

        // Return a formatted expr.
        virtual string make_str(const VarMap* var_map = 0) const;
        virtual string make_quoted_str(string quote = "'",
                                     const VarMap* var_map = 0) const {
            return quote + make_str(var_map) + quote;
        }
        virtual string get_descr() const {
            return make_quoted_str();
        }

        // Count and return number of nodes at and below this.
        virtual int _get_num_nodes() const;

        // Use addr of this as a unique ID for this object.
        virtual size_t get_id() const {
            return size_t(this);
        }
        virtual string get_id_str() const {
            return to_string(idx_t(this));
        }
        virtual string get_quoted_id() const {
            return "\"" + to_string(idx_t(this)) + "\"";
        }

        // APIs.
        virtual string format_simple() const {
            return make_str();
        }
        virtual int get_num_nodes() const {
            return _get_num_nodes();
        }
    };

    // Convert pointer to the given ptr type or die trying.
    template<typename T> shared_ptr<T> cast_expr(expr_ptr ep, const string& descrip) {
        auto tp = dynamic_pointer_cast<T>(ep);
        if (!tp) {
            THROW_YASK_EXCEPTION("Error: expression '" + ep->make_str() +
                                 "' is not a " + descrip);
        }
        return tp;
    }

    // Compare 2 expr pointers and return whether the expressions are
    // equivalent.
    bool are_exprs_same(const Expr* e1, const Expr* e2);
    inline bool are_exprs_same(const expr_ptr e1, const Expr* e2) {
        return are_exprs_same(e1.get(), e2);
    }
    inline bool are_exprs_same(const Expr* e1, const expr_ptr e2) {
        return are_exprs_same(e1, e2.get());
    }
    inline bool are_exprs_same(const expr_ptr e1, const expr_ptr e2) {
        return are_exprs_same(e1.get(), e2.get());
    }

    // Real or int value.
    class NumExpr : public Expr,
                    public virtual yc_number_node {
    public:

        // Return 'true' if this is a compile-time constant.
        virtual bool is_const_val() const {
            return false;
        }

        // Get the current value.
        // Exit with error if not known.
        virtual double get_num_val() const {
            THROW_YASK_EXCEPTION("Error: cannot evaluate '" + make_str() +
                "' for a known numerical value");
        }

        // Get the value as an integer.
        // Exits with error if not an integer.
        virtual int get_int_val() const {
            double val = get_num_val();
            int ival = int(val);
            if (val != double(ival)) {
                THROW_YASK_EXCEPTION("Error: '" + make_str() +
                    "' does not evaluate to an integer");
            }
            return ival;
        }

        // Return 'true' and set offset if this expr is of the form 'dim',
        // 'dim+const', or 'dim-const'.
        // TODO: make this more robust.
        virtual bool is_offset_from(string dim, int& offset) {
            return false;
        }

        // Create a deep copy of this expression.
        // For this to work properly, each derived type
        // should also implement a deep-copy copy ctor.
        virtual num_expr_ptr clone() const =0;
        virtual yc_number_node_ptr clone_ast() const {
            return clone();
        }
    };

    // Var index types.
    enum IndexType {
        STEP_INDEX,             // the step dim.
        DOMAIN_INDEX,           // a domain dim.
        MISC_INDEX,             // any other dim.
        FIRST_INDEX,            // first index value in domain.
        LAST_INDEX              // last index value in domain.
    };

    // Expression based on a dimension index.
    // This is an expression leaf-node.
    class IndexExpr : public NumExpr,
                      public virtual yc_index_node {
    protected:
        string _dim_name;
        IndexType _type;

    public:
        IndexExpr(string dim, IndexType type) :
            _dim_name(dim), _type(type) { }
        virtual ~IndexExpr() { }

        const string& _get_name() const { return _dim_name; }
        IndexType get_type() const { return _type; }
        string format(const VarMap* var_map = 0) const {
            switch (_type) {
            case FIRST_INDEX:
                return "FIRST_INDEX(" + _dim_name + ")";
            case LAST_INDEX:
                return "LAST_INDEX(" + _dim_name + ")";
            default:
                if (var_map && var_map->count(_dim_name))
                    return var_map->at(_dim_name);
                else
                    return _dim_name;
            }
        }
        virtual string accept(ExprVisitor* ev);

        // Simple offset?
        virtual bool is_offset_from(string dim, int& offset);

        // Check for equivalency.
        virtual bool is_same(const Expr* other) const {
            auto p = dynamic_cast<const IndexExpr*>(other);
            return p && _dim_name == p->_dim_name && _type == p->_type;
        }

        // Create a deep copy of this expression.
        virtual num_expr_ptr clone() const { return make_shared<IndexExpr>(*this); }

        // APIs.
        virtual const string& get_name() const {
            return _dim_name;
        }
    };

    // Boolean value.
    class BoolExpr : public Expr,
                     public virtual yc_bool_node  {
    public:

        // Get the current value.
        // Exit with error if not known.
        virtual bool get_bool_val() const {
            THROW_YASK_EXCEPTION("Error: cannot evaluate '" + make_str() +
                                 "' for a known boolean value");
        }

        // Create a deep copy of this expression.
        // For this to work properly, each derived type
        // should also implement a copy ctor.
        virtual bool_expr_ptr clone() const =0;
        virtual yc_bool_node_ptr clone_ast() const {
            return clone();
        }
    };

    // A simple constant value.
    // This is an expression leaf-node.
    class ConstExpr : public NumExpr,
                      public virtual yc_const_number_node {
    protected:
        double _f = 0.0;

    public:
        ConstExpr(double f) : _f(f) { }
        ConstExpr(idx_t i) : _f(i) {
            if (idx_t(_f) != i)
                FORMAT_AND_THROW_YASK_EXCEPTION("Error: integer value " << i <<
                                     " cannot be stored accurately as a double");
        }
        ConstExpr(int i) : ConstExpr(idx_t(i)) { }
        ConstExpr(const ConstExpr& src) : _f(src._f) { }
        virtual ~ConstExpr() { }

        virtual bool is_const_val() const { return true; }
        double get_num_val() const { return _f; }

        virtual string accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool is_same(const Expr* other) const {
            auto p = dynamic_cast<const ConstExpr*>(other);
            return p && _f == p->_f;
        }

        // Create a deep copy of this expression.
        virtual num_expr_ptr clone() const { return make_shared<ConstExpr>(*this); }

        // APIs.
        virtual void set_value(double val) { _f = val; }
        virtual double get_value() const { return _f; }
    };

    // Any expression that returns a real (not from a YASK var).
    // This is an expression leaf-node.
    class CodeExpr : public NumExpr {
    protected:
        string _code;

    public:
        CodeExpr(const string& code) :
            _code(code) { }
        CodeExpr(const CodeExpr& src) :
            _code(src._code) { }
        virtual ~CodeExpr() { }

        const string& get_code() const {
            return _code;
        }

        virtual string accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool is_same(const Expr* other) const {
            auto p = dynamic_cast<const CodeExpr*>(other);
            return p && _code == p->_code;
        }

        // Create a deep copy of this expression.
        virtual num_expr_ptr clone() const { return make_shared<CodeExpr>(*this); }
    };

    // Base class for any generic unary operator.
    // Also extended for binary operators by adding a LHS.
    // Still pure virtual because clone() not implemented.
    template <typename BaseT,
              typename ArgT, typename ArgApiT>
    class UnaryExpr : public BaseT {
    protected:
        ArgT _rhs;
        string _op_str;

    public:
        UnaryExpr(const string& op_str, ArgT rhs) :
            _rhs(rhs),
            _op_str(op_str) { }
        UnaryExpr(const UnaryExpr& src) :
            _rhs(src._rhs->clone()),
            _op_str(src._op_str) { }

        ArgT& _get_rhs() { return _rhs; }
        const ArgT& _get_rhs() const { return _rhs; }
        const string& get_op_str() const { return _op_str; }

        virtual string accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool is_same(const Expr* other) const {
            auto p = dynamic_cast<const UnaryExpr*>(other);
            return p && _op_str == p->_op_str &&
                _rhs && _rhs->is_same(p->_rhs.get());
        }
    };

    // Various types of unary operators depending on input and output types.
    typedef UnaryExpr<NumExpr, num_expr_ptr, yc_number_node_ptr> UnaryNumExpr;
    typedef UnaryExpr<BoolExpr, num_expr_ptr, yc_number_node_ptr> UnaryNum2BoolExpr;
    typedef UnaryExpr<BoolExpr, bool_expr_ptr, yc_bool_node_ptr> UnaryBoolExpr;

    // Negate operator.
    class NegExpr : public UnaryNumExpr,
                    public virtual yc_negate_node {
    public:
        NegExpr(num_expr_ptr rhs) :
            UnaryNumExpr(op_str(), rhs) { }
        NegExpr(const NegExpr& src) :
            UnaryExpr(src) { }

        static string op_str() { return "-"; }
        virtual bool is_const_val() const {
            return _rhs->is_const_val();
        }
        virtual double get_num_val() const {
            double rhs = _rhs->get_num_val();
            return -rhs;
        }
        virtual num_expr_ptr clone() const {
            return make_shared<NegExpr>(*this);
        }

        // APIs.
        virtual yc_number_node_ptr get_rhs() {
            return _rhs;
        }
    };

    // Boolean inverse operator.
    class NotExpr : public UnaryBoolExpr,
                    public virtual yc_not_node {
    public:
        NotExpr(bool_expr_ptr rhs) :
            UnaryBoolExpr(op_str(), rhs) { }
        NotExpr(const NotExpr& src) :
            UnaryBoolExpr(src) { }

        static string op_str() { return "!"; }
        virtual bool get_bool_val() const {
            bool rhs = _rhs->get_bool_val();
            return !rhs;
        }
        virtual bool_expr_ptr clone() const {
            return make_shared<NotExpr>(*this);
        }
        virtual yc_bool_node_ptr get_rhs() {
            return _get_rhs();
        }
    };


    // Base class for any generic binary operator.
    // Still pure virtual because clone() not implemented.
    template <typename BaseT, typename BaseApiT,
              typename ArgT, typename ArgApiT>
    class BinaryExpr : public BaseT,
                       public virtual BaseApiT {
    protected:
        ArgT _lhs;              // RHS in BaseT which must be a UnaryExpr.

    public:
        BinaryExpr(ArgT lhs, const string& op_str, ArgT rhs) :
            BaseT(op_str, rhs),
            _lhs(lhs) { }
        BinaryExpr(const BinaryExpr& src) :
            BaseT(src._op_str, src._rhs->clone()),
            _lhs(src._lhs->clone()) { }

        ArgT& _get_lhs() { return _lhs; }
        const ArgT& _get_lhs() const { return _lhs; }
        virtual string accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool is_same(const Expr* other) const {
            auto p = dynamic_cast<const BinaryExpr*>(other);
            return p && BaseT::_op_str == p->_op_str &&
                _lhs->is_same(p->_lhs.get()) &&
                BaseT::_rhs->is_same(p->_rhs.get());
        }

        // APIs.
        virtual ArgApiT get_lhs() {
            return _get_lhs();
        }
        virtual ArgApiT get_rhs() {
            return BaseT::_get_rhs();
        }
    };

    // Various types of binary operators depending on input and output types.
    typedef BinaryExpr<UnaryNumExpr, yc_binary_number_node,
                       num_expr_ptr, yc_number_node_ptr> BinaryNumExpr; // fn(num, num) -> num.
    typedef BinaryExpr<UnaryBoolExpr, yc_binary_bool_node,
                       bool_expr_ptr, yc_bool_node_ptr> BinaryBoolExpr; // fn(bool, bool) -> bool.
    typedef BinaryExpr<UnaryNum2BoolExpr, yc_binary_comparison_node,
                       num_expr_ptr, yc_number_node_ptr> BinaryNum2BoolExpr; // fn(num, num) -> bool.

    // Numerical binary operators.
    // TODO: redo this with a template.
#define BIN_NUM_EXPR(type, impl_type, opstr, oper)       \
    class type : public BinaryNumExpr,                  \
                 public virtual impl_type {              \
    public:                                             \
        type(num_expr_ptr lhs, num_expr_ptr rhs) :          \
            BinaryNumExpr(lhs, op_str(), rhs) { }        \
        type(const type& src) :                         \
            BinaryNumExpr(src) { }                      \
        static string op_str() { return opstr; }         \
        virtual bool is_offset_from(string dim, int& offset); \
        virtual bool is_const_val() const {               \
            return _lhs->is_const_val() &&                \
                _rhs->is_const_val();                     \
        }                                               \
        virtual double get_num_val() const {              \
            double lhs = _lhs->get_num_val();             \
            double rhs = _rhs->get_num_val();             \
            return oper;                                \
        }                                               \
        virtual num_expr_ptr clone() const {              \
            return make_shared<type>(*this);            \
        }                                               \
    }
    BIN_NUM_EXPR(SubExpr, yc_subtract_node, "-", lhs - rhs);

    // TODO: add check for div-by-0.
    // TODO: handle division properly for integer indices.
    BIN_NUM_EXPR(DivExpr, yc_divide_node, "/", lhs / rhs);

    // TODO: add check for mod-by-0.
    BIN_NUM_EXPR(ModExpr, yc_mod_node, "%", imod_flr(idx_t(lhs), idx_t(rhs)));
#undef BIN_NUM_EXPR

// Boolean binary operators with numerical inputs.
// TODO: redo this with a template.
#define BIN_NUM2BOOL_EXPR(type, impl_type, opstr, oper)  \
    class type : public BinaryNum2BoolExpr,             \
                 public virtual impl_type {              \
    public:                                             \
    type(num_expr_ptr lhs, num_expr_ptr rhs) :              \
        BinaryNum2BoolExpr(lhs, op_str(), rhs) { }       \
    type(const type& src) :                             \
        BinaryNum2BoolExpr(src) { }                     \
    static string op_str() { return opstr; }             \
    virtual bool get_bool_val() const {                   \
        double lhs = _lhs->get_num_val();                 \
        double rhs = _rhs->get_num_val();                 \
        return oper;                                    \
    }                                                   \
    virtual bool_expr_ptr clone() const {                 \
        return make_shared<type>(*this);                \
    }                                                   \
    virtual yc_number_node_ptr get_lhs() {              \
        return _get_lhs();                                \
    }                                                   \
    virtual yc_number_node_ptr get_rhs() {              \
        return _get_rhs();                                \
    }                                                   \
    }
    BIN_NUM2BOOL_EXPR(IsEqualExpr, yc_equals_node, "==", lhs == rhs);
    BIN_NUM2BOOL_EXPR(NotEqualExpr, yc_not_equals_node, "!=", lhs != rhs);
    BIN_NUM2BOOL_EXPR(IsLessExpr, yc_less_than_node, "<", lhs < rhs);
    BIN_NUM2BOOL_EXPR(NotLessExpr, yc_not_less_than_node, ">=", lhs >= rhs);
    BIN_NUM2BOOL_EXPR(IsGreaterExpr, yc_greater_than_node, ">", lhs > rhs);
    BIN_NUM2BOOL_EXPR(NotGreaterExpr, yc_not_greater_than_node, "<=", lhs <= rhs);
#undef BIN_NUM2BOOL_EXPR

    // Boolean binary operators with boolean inputs.
    // TODO: redo this with a template.
#define BIN_BOOL_EXPR(type, impl_type, opstr, oper)      \
    class type : public BinaryBoolExpr, \
                 public virtual impl_type {              \
    public:                                             \
    type(bool_expr_ptr lhs, bool_expr_ptr rhs) :            \
        BinaryBoolExpr(lhs, op_str(), rhs) { }           \
    type(const type& src) :                             \
        BinaryBoolExpr(src) { }                         \
    static string op_str() { return opstr; }             \
    virtual bool get_bool_val() const {                   \
        bool lhs = _lhs->get_bool_val();                  \
        bool rhs = _rhs->get_bool_val();                  \
        return oper;                                    \
    }                                                   \
    virtual bool_expr_ptr clone() const {                 \
        return make_shared<type>(*this);                \
    }                                                   \
    virtual yc_bool_node_ptr get_lhs() {                \
        return _get_lhs();                                \
    }                                                   \
    virtual yc_bool_node_ptr get_rhs() {                \
        return _get_rhs();                                \
    }                                                   \
    }
    BIN_BOOL_EXPR(AndExpr, yc_and_node, "&&", lhs && rhs);
    BIN_BOOL_EXPR(OrExpr, yc_or_node, "||", lhs || rhs);
#undef BIN_BOOL_EXPR

    // A list of exprs with a common operator that can be rearranged,
    // e.g., 'a * b * c' or 'a + b + c'.
    // Still pure virtual because clone() not implemented.
    class CommutativeExpr : public NumExpr,
                            public virtual yc_commutative_number_node {
    protected:
        num_expr_ptr_vec _ops;
        string _op_str;

    public:
        CommutativeExpr(const string& op_str) :
            _op_str(op_str) {
        }
        CommutativeExpr(num_expr_ptr lhs, const string& op_str, num_expr_ptr rhs) :
            _op_str(op_str) {
            _ops.push_back(lhs->clone());
            _ops.push_back(rhs->clone());
        }
        CommutativeExpr(const CommutativeExpr& src) :
            _op_str(src._op_str) {
            for(auto op : src._ops)
                _ops.push_back(op->clone());
        }
        virtual ~CommutativeExpr() { }

        // Accessors.
        num_expr_ptr_vec& get_ops() { return _ops; }
        const num_expr_ptr_vec& get_ops() const { return _ops; }
        const string& get_op_str() const { return _op_str; }

        // Clone and add an operand.
        virtual void append_op(num_expr_ptr op) {
            _ops.push_back(op->clone());
        }

        // If op is another CommutativeExpr with the
        // same operator, add its operands to this.
        // Otherwise, just add the whole node.
        // Example: if 'this' is 'A+B', 'merge_expr(C+D)'
        // returns 'A+B+C+D', and 'merge_expr(E*F)'
        // returns 'A+B+(E*F)'.
        virtual void merge_expr(num_expr_ptr op) {
            auto opp = dynamic_pointer_cast<CommutativeExpr>(op);
            if (opp && opp->get_op_str() == _op_str) {
                for(auto op2 : opp->_ops)
                    append_op(op2);
            }
            else
                append_op(op);
        }

        // Swap the contents w/another.
        virtual void swap(CommutativeExpr* ce) {
            _ops.swap(ce->_ops);
            _op_str.swap(ce->_op_str);
        }

        virtual string accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool is_same(const Expr* other) const;

        virtual bool is_const_val() const {
            for(auto op : _ops) {
                if (!op->is_const_val())
                    return false;
            }
            return true;
        }

        // APIs.
        virtual int get_num_operands() {
            return _ops.size();
        }
        virtual std::vector<yc_number_node_ptr> get_operands() {
            std::vector<yc_number_node_ptr> nv;
            for (int i = 0; i < get_num_operands(); i++)
                nv.push_back(_ops.at(i));
            return nv;
        }
        virtual void add_operand(yc_number_node_ptr node) {
            auto p = dynamic_pointer_cast<NumExpr>(node);
            assert(p);
            append_op(p);
        }
    };

    // Commutative operators.
    // TODO: redo this with a template.
#define COMM_EXPR(type, impl_type, opstr, base_val, oper)                 \
    class type : public CommutativeExpr, public virtual impl_type  {     \
    public:                                                             \
    type()  :                                                           \
        CommutativeExpr(op_str()) { }                                    \
    type(num_expr_ptr lhs, num_expr_ptr rhs) :                              \
        CommutativeExpr(lhs, op_str(), rhs) { }                          \
    type(const type& src) :                                             \
        CommutativeExpr(src) { }                                        \
    virtual ~type() { }                                                 \
    static string op_str() { return opstr; }                             \
    virtual bool is_offset_from(string dim, int& offset);                 \
    virtual double get_num_val() const {                                  \
        double val = base_val;                                           \
        for(auto op : _ops) {                                           \
            double lhs = val;                                           \
            double rhs = op->get_num_val();                               \
            val = oper;                                                 \
        }                                                               \
        return val;                                                     \
    }                                                                   \
    virtual num_expr_ptr clone() const { return make_shared<type>(*this); } \
    };
    COMM_EXPR(MultExpr, yc_multiply_node, "*", 1.0, lhs * rhs)
    COMM_EXPR(AddExpr, yc_add_node, "+", 0.0, lhs + rhs)
#undef COMM_EXPR

    // An FP function call with an arbitrary number of FP args.
    // e.g., sin(a).
    // TODO: add APIs.
    class FuncExpr : public NumExpr {
    protected:
        string _op_str;          // name of function.
        num_expr_ptr_vec _ops;     // args to function.

        // Special handler for pairable functions like sincos().
        FuncExpr* _paired = nullptr;     // ptr to counterpart.

    public:
        FuncExpr(const string& op_str, const std::initializer_list< const num_expr_ptr > & ops) :
            _op_str(op_str) {
            for (auto& op : ops)
                _ops.push_back(op->clone());
        }
        FuncExpr(const FuncExpr& src) :
            _op_str(src._op_str, {}) {

            // Deep copy.
            for (auto& op : src._ops)
                _ops.push_back(op->clone());
        }

        // Accessors.
        num_expr_ptr_vec& get_ops() { return _ops; }
        const num_expr_ptr_vec& get_ops() const { return _ops; }
        const string& get_op_str() const { return _op_str; }

        virtual string accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool is_same(const Expr* other) const;

        virtual bool make_pair(Expr* other);
        virtual FuncExpr* get_pair() { return _paired; }

        virtual bool is_const_val() const {
            for(auto op : _ops) {
                if (!op->is_const_val())
                    return false;
            }
            return true;
        }
        virtual num_expr_ptr clone() const {
            return make_shared<FuncExpr>(*this);
        }

        // APIs.
        virtual int get_num_operands() {
            return _ops.size();
        }
        virtual std::vector<yc_number_node_ptr> get_operands() {
            std::vector<yc_number_node_ptr> nv;
            for (int i = 0; i < get_num_operands(); i++)
                nv.push_back(_ops.at(i));
            return nv;
        }
    };

    // One specific point in a var.
    // This is an expression leaf-node.
    class VarPoint : public NumExpr,
                      public virtual yc_var_point_node {

    public:

        // What kind of vectorization can be done on this point.
        // Set via Eqs::analyze_vec().
        enum VecType { VEC_UNSET,
                       VEC_FULL, // vectorizable in all folded dims.
                       VEC_PARTIAL, // vectorizable in some folded dims.
                       VEC_NONE  // vectorizable in no folded dims.
        };

        // Analysis of this point for accesses via loops through the inner dim.
        // Set via Eqs::analyze_loop().
        enum LoopType { LOOP_UNSET,
                        LOOP_INVARIANT, // not dependent on a domain dim.
                        LOOP_DEPENDENT  // dep on one or more domain dims.
        };

    protected:
        Var* _var = 0;        // the var this point is from.

        // Index exprs for each dim, e.g.,
        // "3, x-5, y*2, z+4" for dims "n, x, y, z".
        num_expr_ptr_vec _args;

        // Vars below are calculated from above.
        
        // Simple offset for each expr that is dim +/- offset, e.g.,
        // "x=-5, z=4" from above example.
        // Includes zero offsets.
        // Set in ctor and modified via set_arg_offset/Const().
        IntTuple _offsets;

        // Simple value for each expr that is a const, e.g.,
        // "n=3" from above example.
        // Set in ctor and modified via set_arg_offset/Const().
        IntTuple _consts;

        VecType _vec_type = VEC_UNSET; // allowed vectorization.
        LoopType _loop_type = LOOP_UNSET; // analysis for looping.

        // Cache the string repr.
        string _def_str;
        void _update_str() {
            _def_str = _make_str();
        }
        string _make_str(const VarMap* var_map = 0) const;

    public:

        // Construct a point given a var and an arg for each dim.
        VarPoint(Var* var, const num_expr_ptr_vec& args);

        // Dtor.
        virtual ~VarPoint() {}

        // Get parent var info.
        const Var* _get_var() const { return _var; }
        Var* _get_var() { return _var; }
        virtual const string& get_var_name() const;
        virtual bool is_var_foldable() const;
        virtual const index_expr_ptr_vec& get_dims() const;

        // Accessors.
        virtual const num_expr_ptr_vec& get_args() const { return _args; }
        virtual const IntTuple& get_arg_offsets() const { return _offsets; }
        virtual const IntTuple& get_arg_consts() const { return _consts; }
        virtual VecType get_vec_type() const {
            assert(_vec_type != VEC_UNSET);
            return _vec_type;
        }
        virtual void set_vec_type(VecType vt) {
            _vec_type = vt;
        }
        virtual LoopType get_loop_type() const {
            assert(_loop_type != LOOP_UNSET);
            return _loop_type;
        }
        virtual void set_loop_type(LoopType vt) {
            _loop_type = vt;
        }

        // Get arg for 'dim' or return null if none.
        virtual const num_expr_ptr get_arg(const string& dim) const;
        
        // Set given arg to given offset; ignore if not in step or domain var dims.
        virtual void set_arg_offset(const IntScalar& offset);

        // Set given args to be given offsets.
        virtual void set_arg_offsets(const IntTuple& offsets) {
            for (auto ofs : offsets)
                set_arg_offset(ofs);
        }

        // Set given arg to given const.
        virtual void set_arg_const(const IntScalar& val);

        // Set given arg to given expr.
        virtual void set_arg_expr(const string& expr_dim, const string& expr);
        
        // Some comparisons.
        bool operator==(const VarPoint& rhs) const {
            return _def_str == rhs._def_str;
        }
        bool operator<(const VarPoint& rhs) const {
            return _def_str < rhs._def_str;
        }

        // Take ev to each value.
        virtual string accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool is_same(const Expr* other) const {
            auto p = dynamic_cast<const VarPoint*>(other);
            return p && *this == *p;
        }

        // Check for same logical var.
        // A logical var is defined by the var itself
        // and any const indices.
        virtual bool is_same_logical_var(const VarPoint& rhs) const {
            return _var == rhs._var && _consts == rhs._consts;
        }

        // String w/name and parens around args, e.g., 'u(x, y+2)'.
        // Apply substitutions to indices using 'var_map' if provided.
        virtual string make_str(const VarMap* var_map = 0) const {
            if (var_map)
                return _make_str(var_map);
            return _def_str;
        }

        // String w/name and parens around const args, e.g., 'u(n=4)'.
        // Apply substitutions to indices using 'var_map' if provided.
        virtual string make_logical_var_str(const VarMap* var_map = 0) const;

        // String w/just comma-sep args, e.g., 'x, y+2'.
        // Apply substitutions to indices using 'var_map' if provided.
        virtual string make_arg_str(const VarMap* var_map = 0) const;

        // String v/vec-normalized args, e.g., 'x, y+(2/VLEN_Y)'.
        // This object has numerators; 'fold' object has denominators.
        // Apply substitutions to indices using 'var_map' if provided.
        virtual string make_norm_arg_str(const Dimensions& dims,
                                      const VarMap* var_map = 0) const;

        // Make string like "x+(4/VLEN_X)" from original arg "x+4" in 'dname' dim.
        // This object has numerators; 'fold' object has denominators.
        // Apply substitutions to indices using 'var_map' if provided.
        virtual string make_norm_arg_str(const string& dname,
                                      const Dimensions& dims,
                                      const VarMap* var_map = 0) const;

        // Make string like "g->_wrap_step(t+1)" from original arg "t+1"
        // if var uses step dim, "0" otherwise.
        virtual string make_step_arg_str(const string& var_ptr, const Dimensions& dims) const;

        // Create a deep copy of this expression,
        // except pointed-to var is not copied.
        virtual num_expr_ptr clone() const { return make_shared<VarPoint>(*this); }
        virtual var_point_ptr clone_var_point() const { return make_shared<VarPoint>(*this); }

        // APIs.
        virtual yc_var* get_var();
    };

    // Equality operator for a var point.
    // This defines the LHS as equal to the RHS; it is NOT
    // a comparison operator; it is NOT an assignment operator.
    // It also holds an optional condition.
    class EqualsExpr : public Expr,
                       public virtual yc_equation_node {
    protected:
        var_point_ptr _lhs;
        num_expr_ptr _rhs;
        bool_expr_ptr _cond;
        bool_expr_ptr _step_cond;

    public:
        EqualsExpr(var_point_ptr lhs, num_expr_ptr rhs,
                   bool_expr_ptr cond = nullptr,
                   bool_expr_ptr step_cond = nullptr) :
            _lhs(lhs), _rhs(rhs), _cond(cond), _step_cond(step_cond) { }
        EqualsExpr(const EqualsExpr& src) :
            _lhs(src._lhs->clone_var_point()),
            _rhs(src._rhs->clone()) {
            if (src._cond)
                _cond = src._cond->clone();
            else
                _cond = nullptr;
            if (src._step_cond)
                _step_cond = src._step_cond->clone();
            else
                _step_cond = nullptr;
        }

        var_point_ptr& _get_lhs() { return _lhs; }
        const var_point_ptr& _get_lhs() const { return _lhs; }
        num_expr_ptr& _get_rhs() { return _rhs; }
        const num_expr_ptr& _get_rhs() const { return _rhs; }
        bool_expr_ptr& _get_cond() { return _cond; }
        const bool_expr_ptr& _get_cond() const { return _cond; }
        void _set_cond(bool_expr_ptr cond) { _cond = cond; }
        bool_expr_ptr& _get_step_cond() { return _step_cond; }
        const bool_expr_ptr& _get_step_cond() const { return _step_cond; }
        void _set_step_cond(bool_expr_ptr step_cond) { _step_cond = step_cond; }
        virtual string accept(ExprVisitor* ev);
        static string expr_op_str() { return "EQUALS"; }
        static string cond_op_str() { return "IF_DOMAIN"; }
        static string step_cond_op_str() { return "IF_STEP"; }

        // Get pointer to var on LHS or NULL if not set.
        virtual Var* _get_var() {
            if (_lhs.get())
                return _lhs->_get_var();
            return NULL;
        }

        // LHS is scratch var.
        virtual bool is_scratch();

        // Check for equivalency.
        virtual bool is_same(const Expr* other) const;

        // Create a deep copy of this expression.
        virtual equals_expr_ptr clone() const { return make_shared<EqualsExpr>(*this); }
        virtual yc_equation_node_ptr clone_ast() const {
            return clone();
        }

        // APIs.
        virtual yc_var_point_node_ptr get_lhs() { return _lhs; }
        virtual yc_number_node_ptr get_rhs() { return _rhs; }
        virtual yc_bool_node_ptr get_cond() { return _cond; }
        virtual yc_bool_node_ptr get_step_cond() { return _step_cond; }
        virtual void set_cond(yc_bool_node_ptr cond) {
            if (cond) {
                auto p = dynamic_pointer_cast<BoolExpr>(cond);
                assert(p);
                _cond = p;
            } else
                _cond = nullptr;
        }
        virtual void set_step_cond(yc_bool_node_ptr step_cond) {
            if (step_cond) {
                auto p = dynamic_pointer_cast<BoolExpr>(step_cond);
                assert(p);
                _step_cond = p;
            } else
                _step_cond = nullptr;
        }
    };

    typedef set<VarPoint> VarPointSet;
    typedef set<var_point_ptr> var_point_ptr_set;
    typedef vector<VarPoint> VarPointVec;

} // namespace yask.

// Define hash function for VarPoint for unordered_{set,map}.
namespace std {
    using namespace yask;

    template <> struct hash<VarPoint> {
        size_t operator()(const VarPoint& k) const {
            return hash<string>{}(k.make_str());
        }
    };
}
