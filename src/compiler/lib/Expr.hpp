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

///////// AST Expressions ////////////

// TODO: break this up into several smaller files.

#ifndef EXPR_HPP
#define EXPR_HPP

#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <cstdarg>
#include <assert.h>
#include <fstream>

#include "tuple.hpp"
#include "yask_compiler_api.hpp"

using namespace std;

namespace yask {

    // Collections of integers, used for dimensions, indices, etc.
    typedef Scalar<int> IntScalar;
    typedef Tuple<int> IntTuple;

    // Forward-decls of expressions.
    class Expr;
    typedef shared_ptr<Expr> ExprPtr;
    class NumExpr;
    typedef shared_ptr<NumExpr> NumExprPtr;
    typedef vector<NumExprPtr> NumExprPtrVec;
    class IndexExpr;
    typedef shared_ptr<IndexExpr> IndexExprPtr;
    typedef vector<IndexExprPtr> IndexExprPtrVec;
    class BoolExpr;
    typedef shared_ptr<BoolExpr> BoolExprPtr;
    class EqualsExpr;
    typedef shared_ptr<EqualsExpr> EqualsExprPtr;
    class IfExpr;
    typedef shared_ptr<IfExpr> IfExprPtr;

    // More forward-decls.
    class ExprVisitor;
    class Grid;
    class GridPoint;
    typedef shared_ptr<GridPoint> GridPointPtr;
    class StencilSolution;

    //// Classes to implement parts of expressions.
    // The expressions are constructed at run-time when the
    // StencilSolution::define() method is called.

    // The base class for all expression nodes.
    class Expr : public virtual yc_expr_node {

    public:
        Expr() { }
        virtual ~Expr() { }

        // For visitors.
        virtual void accept(ExprVisitor* ev) =0;
        virtual void accept(ExprVisitor* ev) const;

        // check for expression equivalency.
        // Does *not* check value equivalency except for
        // constants.
        virtual bool isSame(const Expr* other) const =0;

        // Return a simple string expr.
        virtual string makeStr() const;
        virtual string makeQuotedStr(string quote = "'") const {
            ostringstream oss;
            oss << quote << makeStr() << quote;
            return oss.str();
        }

        // Count and return number of nodes at and below this.
        virtual int getNumNodes() const;

        // Use addr of this as a unique ID for this object.
        virtual size_t getId() const {
            return size_t(this);
        }
        virtual string getIdStr() const {
            ostringstream oss;
            oss << this;
            return oss.str();
        }
        virtual string getQuotedId() const {
            ostringstream oss;
            oss << "\"" << this << "\"";
            return oss.str();
        }

        // APIs.
        virtual string format_simple() const {
            return makeStr();
        }
        virtual int get_num_nodes() const {
            return getNumNodes();
        }
    };

    // Convert pointer to the given ptr type or die w/an error.
    template<typename T> shared_ptr<T> castExpr(ExprPtr ep, const string& descrip) {
        auto tp = dynamic_pointer_cast<T>(ep);
        if (!tp) {
            cerr << "Error: expression '" << ep->makeStr() << "' is not a " <<
                descrip << "." << endl;
            exit(1);
        }
        return tp;
    }

    // Compare 2 expr pointers and return whether the expressions are
    // equivalent.
    bool areExprsSame(const Expr* e1, const Expr* e2);
    inline bool areExprsSame(const ExprPtr e1, const Expr* e2) {
        return areExprsSame(e1.get(), e2);
    }
    inline bool areExprsSame(const Expr* e1, const ExprPtr e2) {
        return areExprsSame(e1, e2.get());
    }
    inline bool areExprsSame(const ExprPtr e1, const ExprPtr e2) {
        return areExprsSame(e1.get(), e2.get());
    }

    // Real or int value.
    class NumExpr : public Expr, public virtual yc_number_node {
    public:

        // Return 'true' if this is a compile-time constant.
        virtual bool isConstVal() const {
            return false;
        }
        
        // Get the current value.
        // Exit with error if not known.
        virtual double getNumVal() const {
            cerr << "Error: cannot evaluate '" << makeStr() <<
                "' for a known numerical value.\n";
            exit(1);
        }

        // Get the value as an integer.
        // Exits with error if not an integer.
        virtual int getIntVal() const {
            double val = getNumVal();
            int ival = int(val);
            if (val != double(ival)) {
                cerr << "Error: '" << makeStr() <<
                    "' does not evaluate to an integer.\n";
                exit(1);
            }
            return ival;
        }

        // Return 'true' and set offset if this expr is of the form 'dim',
        // 'dim+const', or 'dim-const'.
        virtual bool isOffsetFrom(string dim, int& offset) {
            return false;
        }
    
        // Create a deep copy of this expression.
        // For this to work properly, each derived type
        // should also implement a deep-copy copy ctor.
        virtual NumExprPtr clone() const =0;
    };

    // Grid index types.
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
        string _dimName;
        IndexType _type;

    public:
        IndexExpr(string dim, IndexType type) :
            _dimName(dim), _type(type) { }
        virtual ~IndexExpr() { }

        const string& getName() const { return _dimName; }
        IndexType getType() const { return _type; }
        string format() const {
            switch (_type) {
            case FIRST_INDEX:
                return "FIRST_INDEX(" + _dimName + ")";
            case LAST_INDEX:
                return "LAST_INDEX(" + _dimName + ")";
            default:
                return _dimName;
            }
        }
        virtual void accept(ExprVisitor* ev);

        // Simple offset?
        virtual bool isOffsetFrom(string dim, int& offset);
        
        // Check for equivalency.
        virtual bool isSame(const Expr* other) const {
            auto p = dynamic_cast<const IndexExpr*>(other);
            return p && _dimName == p->_dimName && _type == p->_type;
        }
   
        // Create a deep copy of this expression.
        virtual NumExprPtr clone() const { return make_shared<IndexExpr>(*this); }

        // APIs.
        virtual const string& get_name() const {
            return _dimName;
        }
    };

    // A free function to create a constant expression.
    // Usually not needed due to operator overloading.
    NumExprPtr constNum(double rhs);

    // Free functions to create boundary indices, e.g., 'first_index(x)'.
    NumExprPtr first_index(IndexExprPtr dim);
    NumExprPtr last_index(IndexExprPtr dim);

    // A simple wrapper to provide automatic construction
    // of a NumExpr ptr from other types.
    class NumExprArg : public NumExprPtr {

    public:
        NumExprArg(NumExprPtr p) :
            NumExprPtr(p) { }
        NumExprArg(IndexExprPtr p) :
            NumExprPtr(p) { }
        NumExprArg(int i) :
            NumExprPtr(constNum(i)) { }
        NumExprArg(double f) :
            NumExprPtr(constNum(f)) { }
    };
    
    // Boolean value.
    class BoolExpr : public Expr, public virtual yc_bool_node  {
    public:

        // Get the current value.
        // Exit with error if not known.
        virtual bool getBoolVal() const {
            cerr << "Error: cannot evaluate '" << makeStr() <<
                "' for a known boolean value.\n";
            exit(1);
        }

        // Create a deep copy of this expression.
        // For this to work properly, each derived type
        // should also implement a copy ctor.
        virtual BoolExprPtr clone() const =0;
    };

    // A simple constant value.
    // This is an expression leaf-node.
    class ConstExpr : public NumExpr, public virtual yc_const_number_node {
    protected:
        double _f = 0.0;

    public:
        ConstExpr(double f) : _f(f) { }
        ConstExpr(const ConstExpr& src) : _f(src._f) { }
        virtual ~ConstExpr() { }

        virtual bool isConstVal() const { return true; }
        double getNumVal() const { return _f; }

        virtual void accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool isSame(const Expr* other) const {
            auto p = dynamic_cast<const ConstExpr*>(other);
            return p && _f == p->_f;
        }
   
        // Create a deep copy of this expression.
        virtual NumExprPtr clone() const { return make_shared<ConstExpr>(*this); }

        // APIs.
        virtual void set_value(double val) { _f = val; }
        virtual double get_value() const { return _f; }
    };

    // Any expression that returns a real (not from a grid).
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

        const string& getCode() const {
            return _code;
        }

        virtual void accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool isSame(const Expr* other) const {
            auto p = dynamic_cast<const CodeExpr*>(other);
            return p && _code == p->_code;
        }

        // Create a deep copy of this expression.
        virtual NumExprPtr clone() const { return make_shared<CodeExpr>(*this); }
    };

    // Base class for any generic unary operator.
    // Still pure virtual because clone() not implemented.
    template <typename BaseT, typename ArgT>
    class UnaryExpr : public BaseT {
    protected:
        ArgT _rhs;
        string _opStr;

    public:
        UnaryExpr(const string& opStr, ArgT rhs) :
            _rhs(rhs),
            _opStr(opStr) { }
        UnaryExpr(const UnaryExpr& src) :
            _rhs(src._rhs->clone()),
            _opStr(src._opStr) { }

        ArgT& getRhs() { return _rhs; }
        const ArgT& getRhs() const { return _rhs; }
        const string& getOpStr() const { return _opStr; }

        virtual void accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool isSame(const Expr* other) const {
            auto p = dynamic_cast<const UnaryExpr*>(other);
            return p && _opStr == p->_opStr &&
                _rhs && _rhs->isSame(p->_rhs.get());
        }
    };

    // Various types of unary operators depending on input and output types.
    typedef UnaryExpr<NumExpr, NumExprPtr> UnaryNumExpr;
    typedef UnaryExpr<BoolExpr, BoolExprPtr> UnaryBoolExpr;
    typedef UnaryExpr<BoolExpr, NumExprPtr> UnaryNum2BoolExpr;

    // Negate operator.
    class NegExpr : public UnaryNumExpr,
                    public virtual yc_negate_node {
    public:
        NegExpr(NumExprPtr rhs) :
            UnaryNumExpr(opStr(), rhs) { }
        NegExpr(const NegExpr& src) :
            UnaryExpr(src) { }

        static string opStr() { return "-"; }
        virtual bool isConstVal() const {
            return _rhs->isConstVal();
        }
        virtual double getNumVal() const {
            double rhs = _rhs->getNumVal();
            return -rhs;
        }
        virtual NumExprPtr clone() const {
            return make_shared<NegExpr>(*this);
        }

        // APIs.
        virtual yc_number_node_ptr get_rhs() {
            return _rhs;
        }
    };

    // Boolean inverse operator.
    class NotExpr : public UnaryBoolExpr {
    public:
        NotExpr(BoolExprPtr rhs) :
            UnaryBoolExpr(opStr(), rhs) { }
        NotExpr(const NotExpr& src) :
            UnaryBoolExpr(src) { }

        static string opStr() { return "!"; }
        virtual bool getBoolVal() const {
            bool rhs = _rhs->getBoolVal();
            return !rhs;
        }
        virtual BoolExprPtr clone() const {
            return make_shared<NotExpr>(*this);
        }
    };

    // Base class for any generic binary operator.
    // Still pure virtual because clone() not implemented.
    template <typename BaseT, typename ArgT>
    class BinaryExpr : public BaseT {
    protected:
        ArgT _lhs;              // RHS in BaseT which must be a UnaryExpr.

    public:
        BinaryExpr(ArgT lhs, const string& opStr, ArgT rhs) :
            BaseT(opStr, rhs),
            _lhs(lhs) { }
        BinaryExpr(const BinaryExpr& src) :
            BaseT(src._opStr, src._rhs->clone()),
            _lhs(src._lhs->clone()) { }  

        ArgT& getLhs() { return _lhs; }
        const ArgT& getLhs() const { return _lhs; }
        virtual void accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool isSame(const Expr* other) const {
            auto p = dynamic_cast<const BinaryExpr*>(other);
            return p && BaseT::_opStr == p->_opStr &&
                _lhs->isSame(p->_lhs.get()) &&
                BaseT::_rhs->isSame(p->_rhs.get());
        }
    };

    // Various types of binary operators depending on input and output types.
    typedef BinaryExpr<UnaryNumExpr, NumExprPtr> BinaryNumExpr; // fn(num, num) -> num.
    typedef BinaryExpr<UnaryNum2BoolExpr, NumExprPtr> BinaryNum2BoolExpr; // fn(num, num) -> bool.
    typedef BinaryExpr<UnaryBoolExpr, BoolExprPtr> BinaryBoolExpr; // fn(bool, bool) -> bool.

    // Numerical binary operators.
    // TODO: redo this with a template.
#define BIN_NUM_EXPR(type, implType, opstr, oper)       \
    class type : public BinaryNumExpr,                  \
                 public virtual implType {              \
    public:                                             \
        type(NumExprPtr lhs, NumExprPtr rhs) :          \
            BinaryNumExpr(lhs, opStr(), rhs) { }        \
        type(const type& src) :                         \
            BinaryNumExpr(src) { }                      \
        static string opStr() { return opstr; }         \
        virtual bool isOffsetFrom(string dim, int& offset); \
        virtual bool isConstVal() const {               \
            return _lhs->isConstVal() &&                \
                _rhs->isConstVal();                     \
        }                                               \
        virtual double getNumVal() const {              \
            double lhs = _lhs->getNumVal();             \
            double rhs = _rhs->getNumVal();             \
            return oper;                                \
        }                                               \
        virtual NumExprPtr clone() const {              \
            return make_shared<type>(*this);            \
        }                                               \
        virtual yc_number_node_ptr get_lhs() {          \
            return getLhs();                            \
        }                                               \
        virtual yc_number_node_ptr get_rhs() {          \
            return getRhs();                            \
        }                                               \
    }
    BIN_NUM_EXPR(SubExpr, yc_subtract_node, "-", lhs - rhs);
    BIN_NUM_EXPR(DivExpr, yc_divide_node, "/", lhs / rhs); // TODO: add check for div-by-0.
#undef BIN_NUM_EXPR

// Boolean binary operators with numerical inputs.
// TODO: redo this with a template.
#define BIN_NUM2BOOL_EXPR(type, opstr, oper)            \
    class type : public BinaryNum2BoolExpr {            \
    public:                                             \
    type(NumExprPtr lhs, NumExprPtr rhs) :              \
        BinaryNum2BoolExpr(lhs, opStr(), rhs) { }       \
    type(const type& src) :                             \
        BinaryNum2BoolExpr(src) { }                     \
    static string opStr() { return opstr; }             \
    virtual bool getBoolVal() const {                   \
        double lhs = _lhs->getNumVal();                 \
        double rhs = _rhs->getNumVal();                 \
        return oper;                                    \
    }                                                   \
    virtual BoolExprPtr clone() const {                 \
        return make_shared<type>(*this);                \
    }                                                   \
    }
    BIN_NUM2BOOL_EXPR(IsEqualExpr, "==", lhs == rhs);
    BIN_NUM2BOOL_EXPR(NotEqualExpr, "!=", lhs != rhs);
    BIN_NUM2BOOL_EXPR(IsLessExpr, "<", lhs < rhs);
    BIN_NUM2BOOL_EXPR(NotLessExpr, ">=", lhs >= rhs);
    BIN_NUM2BOOL_EXPR(IsGreaterExpr, ">", lhs > rhs);
    BIN_NUM2BOOL_EXPR(NotGreaterExpr, "<=", lhs <= rhs);
#undef BIN_NUM2BOOL_EXPR

    // Boolean binary operators with boolean inputs.
    // TODO: redo this with a template.
#define BIN_BOOL_EXPR(type, opstr, oper)                \
    class type : public BinaryBoolExpr {                \
    public:                                             \
    type(BoolExprPtr lhs, BoolExprPtr rhs) :            \
        BinaryBoolExpr(lhs, opStr(), rhs) { }           \
    type(const type& src) :                             \
        BinaryBoolExpr(src) { }                         \
    static string opStr() { return opstr; }             \
    virtual bool getBoolVal() const {                   \
        bool lhs = _lhs->getBoolVal();                  \
        bool rhs = _rhs->getBoolVal();                  \
        return oper;                                    \
    }                                                   \
    virtual BoolExprPtr clone() const {                 \
        return make_shared<type>(*this);                \
    }                                                   \
    }
    BIN_BOOL_EXPR(AndExpr, "&&", lhs && rhs);
    BIN_BOOL_EXPR(OrExpr, "||", lhs || rhs);
#undef BIN_BOOL_EXPR

    // A list of exprs with a common operator that can be rearranged,
    // e.g., 'a * b * c' or 'a + b + c'.
    // Still pure virtual because clone() not implemented.
    class CommutativeExpr : public NumExpr, public virtual yc_commutative_number_node {
    protected:
        NumExprPtrVec _ops;
        string _opStr;

    public:
        CommutativeExpr(const string& opStr) :
            _opStr(opStr) {
        }
        CommutativeExpr(NumExprPtr lhs, const string& opStr, NumExprPtr rhs) :
            _opStr(opStr) {
            _ops.push_back(lhs->clone());
            _ops.push_back(rhs->clone());
        }
        CommutativeExpr(const CommutativeExpr& src) :
            _opStr(src._opStr) {
            for(auto op : src._ops)
                _ops.push_back(op->clone());
        }
        virtual ~CommutativeExpr() { }

        // Accessors.
        NumExprPtrVec& getOps() { return _ops; }
        const NumExprPtrVec& getOps() const { return _ops; }
        const string& getOpStr() const { return _opStr; }

        // Clone and add an operand.
        virtual void appendOp(NumExprPtr op) {
            _ops.push_back(op->clone());
        }

        // If op is another CommutativeExpr with the
        // same operator, add its operands to this.
        // Otherwise, just clone and add the whole op.
        virtual void mergeExpr(NumExprPtr op) {
            auto opp = dynamic_pointer_cast<CommutativeExpr>(op);
            if (opp && opp->getOpStr() == _opStr) {
                for(auto op2 : opp->_ops)
                    appendOp(op2);
            }
            else
                appendOp(op);
        }

        // Swap the contents w/another.
        virtual void swap(CommutativeExpr* ce) {
            _ops.swap(ce->_ops);
            _opStr.swap(ce->_opStr);
        }

        virtual void accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool isSame(const Expr* other) const;

        // APIs.
        virtual int get_num_operands() {
            return _ops.size();
        }
        virtual yc_number_node_ptr get_operand(int i) {
            if (i >= 0 &&
                size_t(i) < _ops.size())
                return _ops.at(size_t(i));
            else
                return nullptr;
        }
        virtual void add_operand(yc_number_node_ptr node) {
            auto p = dynamic_pointer_cast<NumExpr>(node);
            assert(p);
            appendOp(p);
        }
    };

    // Commutative operators.
    // TODO: redo this with a template.
#define COMM_EXPR(type, implType, opstr, baseVal, oper)                 \
    class type : public CommutativeExpr, public virtual implType  {     \
    public:                                                             \
    type()  :                                                           \
        CommutativeExpr(opStr()) { }                                    \
    type(NumExprPtr lhs, NumExprPtr rhs) :                              \
        CommutativeExpr(lhs, opStr(), rhs) { }                          \
    type(const type& src) :                                             \
        CommutativeExpr(src) { }                                        \
    virtual ~type() { }                                                 \
    static string opStr() { return opstr; }                             \
    virtual bool isOffsetFrom(string dim, int& offset);                 \
    virtual bool isConstVal() const {                                   \
        bool is_const = true;                                           \
        for(auto op : _ops) {                                           \
            bool rhs = op->isConstVal();                                \
            is_const &= rhs;                                            \
        }                                                               \
        return is_const;                                                \
    }                                                                   \
    virtual double getNumVal() const {                                  \
        double val = baseVal;                                           \
        for(auto op : _ops) {                                           \
            double lhs = val;                                           \
            double rhs = op->getNumVal();                               \
            val = oper;                                                 \
        }                                                               \
        return val;                                                     \
    }                                                                   \
    virtual NumExprPtr clone() const { return make_shared<type>(*this); } \
    }
    COMM_EXPR(MultExpr, yc_multiply_node, "*", 1.0, lhs * rhs);
    COMM_EXPR(AddExpr, yc_add_node, "+", 0.0, lhs + rhs);
#undef COMM_EXPR

    // One specific point in a grid.
    // This is an expression leaf-node.
    class GridPoint : public NumExpr,
                      public virtual yc_grid_point_node {

    protected:
        Grid* _grid = 0;        // the grid this point is from.
        NumExprPtrVec _args;    // index exprs for each dim.
        IntTuple _offsets;      // simple offset for each expr that is dim +/- offset.
        IntTuple _consts;       // simple value for each expr that is a const.

    public:

        // Construct a point given a grid and an arg for each dim.
        GridPoint(Grid* grid, const NumExprPtrVec& args);

        // Dtor.
        virtual ~GridPoint() {}

        // Get parent grid info.
        const Grid* getGrid() const { return _grid; }
        Grid* getGrid() { return _grid; }
        virtual const string& getName() const;

        // Accessors.
        virtual const NumExprPtrVec& getArgs() const { return _args; }
        virtual const IntTuple& getArgOffsets() const { return _offsets; }
        virtual const IntTuple& getArgConsts() const { return _consts; }

        // Set given arg to given offset; ignore if not in step or domain grid dims.
        virtual void setArgOffset(const IntScalar& offset);
        
        // Set given args to be given offsets.
        virtual void setArgOffsets(const IntTuple& offsets) {
            for (auto ofs : offsets.getDims())
                setArgOffset(ofs);
        }

        // Some comparisons.
        bool operator==(const GridPoint& rhs) const;
        bool operator<(const GridPoint& rhs) const;

        // Take ev to each value.
        virtual void accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool isSame(const Expr* other) const {
            auto p = dynamic_cast<const GridPoint*>(other);
            return p && *this == *p;
        }
    
        // Return a description based on this position.
        virtual string makeStr() const;     // with name and parens around args.
        virtual string makeArgsStr() const; // just comma-sep args.
        virtual string makeNormArgsStr(const IntTuple& fold) const; // vec-normalized args.
            
        // Create a deep copy of this expression,
        // except pointed-to grid is not copied.
        virtual NumExprPtr clone() const { return make_shared<GridPoint>(*this); }
        virtual GridPointPtr cloneGridPoint() const { return make_shared<GridPoint>(*this); }
    
        // Determine whether this is 'ahead of' rhs in given direction.
        virtual bool isAheadOfInDir(const GridPoint& rhs, const IntScalar& dir) const;

        // APIs.
        virtual yc_grid* get_grid();
    };
} // namespace yask.

// Define hash function for GridPoint for unordered_{set,map}.
// TODO: make this more efficient.
namespace std {
    using namespace yask;
    
    template <> struct hash<GridPoint> {
        size_t operator()(const GridPoint& k) const {
            return hash<string>{}(k.makeStr());
        }
    };
}

namespace yask {
    
    // Equality operator for a grid point.
    // This defines the LHS as equal to the RHS; it is NOT
    // a comparison operator.
    // (Not inherited from BinaryExpr because LHS is special.)
    class EqualsExpr : public UnaryNumExpr,
                       public virtual yc_equation_node {
    protected:
        GridPointPtr _lhs;

    public:
        EqualsExpr(GridPointPtr lhs, NumExprPtr rhs) :
            UnaryNumExpr(opStr(), rhs),
            _lhs(lhs) { }
        EqualsExpr(const EqualsExpr& src) :
            UnaryNumExpr(src),
            _lhs(src._lhs->cloneGridPoint()) { }

        GridPointPtr& getLhs() { return _lhs; }
        const GridPointPtr& getLhs() const { return _lhs; }
        static string opStr() { return "=="; }
        virtual void accept(ExprVisitor* ev);

        // Get pointer to grid on LHS or NULL if not set.
        virtual Grid* getGrid() {
            if (_lhs.get())
                return _lhs->getGrid();
            return NULL;
        }
    
        // Check for equivalency.
        virtual bool isSame(const Expr* other) const;

        // Create a deep copy of this expression.
        virtual NumExprPtr clone() const { return make_shared<EqualsExpr>(*this); }
        virtual EqualsExprPtr cloneEquals() const { return make_shared<EqualsExpr>(*this); }

        // APIs.
        virtual yc_grid_point_node_ptr get_lhs() {
            return _lhs;
        }
        virtual yc_number_node_ptr get_rhs() {
            return _rhs;
        }
    };

    // Conditional operator.
    // (Not inherited from BinaryExpr because LHS is special.)
    // Condition (RHS) will be NULL if there is no condition.
    class IfExpr : public UnaryBoolExpr {
    protected:
        EqualsExprPtr _expr;

    public:
        IfExpr(EqualsExprPtr expr, const BoolExprPtr cond) :
            UnaryBoolExpr(opStr(), cond),
            _expr(expr) { }
        IfExpr(const IfExpr& src) :
            UnaryBoolExpr(src),
            _expr(src._expr->cloneEquals()) { }

        EqualsExprPtr& getExpr() { return _expr; }
        const EqualsExprPtr& getExpr() const { return _expr; }
        BoolExprPtr& getCond() { return getRhs(); }
        const BoolExprPtr& getCond() const { return getRhs(); }
        static string opStr() { return "IF"; }
        virtual void accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool isSame(const Expr* other) const;

        // Create a deep copy of this expression.
        virtual BoolExprPtr clone() const { return make_shared<IfExpr>(*this); }
    };

    // A conditional evaluation.
    // We use an otherwise unneeded binary operator that has a low priority.
    // See http://en.cppreference.com/w/cpp/language/operator_precedence.
#define IF_OPER ^=
    IfExprPtr operator IF_OPER(EqualsExprPtr expr, const BoolExprPtr cond);
#define IF IF_OPER

    ///// The following are operators and functions used in stencil expressions.

    // Various unary operators.
    NumExprPtr operator-(const NumExprPtr rhs);

    // Various binary operators.
    NumExprPtr operator+(const NumExprPtr lhs, const NumExprPtr rhs);
    NumExprPtr operator+(double lhs, const NumExprPtr rhs);
    NumExprPtr operator+(const NumExprPtr lhs, double rhs);
    void operator+=(NumExprPtr& lhs, const NumExprPtr rhs);
    void operator+=(NumExprPtr& lhs, double rhs);

    NumExprPtr operator-(const NumExprPtr lhs, const NumExprPtr rhs);
    NumExprPtr operator-(double lhs, const NumExprPtr rhs);
    NumExprPtr operator-(const NumExprPtr lhs, double rhs);
    void operator-=(NumExprPtr& lhs, const NumExprPtr rhs);
    void operator-=(NumExprPtr& lhs, double rhs);

    NumExprPtr operator*(const NumExprPtr lhs, const NumExprPtr rhs);
    NumExprPtr operator*(double lhs, const NumExprPtr rhs);
    NumExprPtr operator*(const NumExprPtr lhs, double rhs);
    void operator*=(NumExprPtr& lhs, const NumExprPtr rhs);
    void operator*=(NumExprPtr& lhs, double rhs);

    NumExprPtr operator/(const NumExprPtr lhs, const NumExprPtr rhs);
    NumExprPtr operator/(double lhs, const NumExprPtr rhs);
    NumExprPtr operator/(const NumExprPtr lhs, double rhs);
    void operator/=(NumExprPtr& lhs, const NumExprPtr rhs);
    void operator/=(NumExprPtr& lhs, double rhs);

    // The '==' operator used for defining a grid value.
#define EQUALS_OPER ==
    EqualsExprPtr operator EQUALS_OPER(GridPointPtr gpp, const NumExprPtr rhs);
    EqualsExprPtr operator EQUALS_OPER(GridPointPtr gpp, double rhs);
#define EQUALS EQUALS_OPER
#define IS_EQUIV_TO EQUALS_OPER
#define IS_EQUIVALENT_TO EQUALS_OPER

    // Binary numerical-to-boolean operators.
    // Must provide explicit IndexExprPtr operands to keep compiler from
    // using built-in pointer comparison.
#define BOOL_OPER(oper, type) \
    inline BoolExprPtr operator oper(const NumExprPtr lhs, const NumExprPtr rhs) { \
        return make_shared<type>(lhs, rhs); } \
    inline BoolExprPtr operator oper(const IndexExprPtr lhs, const NumExprPtr rhs) { \
        return make_shared<type>(lhs, rhs); } \
    inline BoolExprPtr operator oper(const NumExprPtr lhs, const IndexExprPtr rhs) { \
        return make_shared<type>(lhs, rhs); } \
    inline BoolExprPtr operator oper(const IndexExprPtr lhs, const IndexExprPtr rhs) { \
        return make_shared<type>(lhs, rhs); }

    BOOL_OPER(==, IsEqualExpr)
    BOOL_OPER(!=, NotEqualExpr)
    BOOL_OPER(<, IsLessExpr)
    BOOL_OPER(>, IsGreaterExpr)
    BOOL_OPER(<=, NotGreaterExpr)
    BOOL_OPER(>=, NotLessExpr)

    // Logical operators.
    inline BoolExprPtr operator&&(const BoolExprPtr lhs, const BoolExprPtr rhs) {
        return make_shared<AndExpr>(lhs, rhs);
    }
    inline BoolExprPtr operator||(const BoolExprPtr lhs, const BoolExprPtr rhs) {
        return make_shared<OrExpr>(lhs, rhs);
    }
    inline BoolExprPtr operator!(const BoolExprPtr rhs) {
        return make_shared<NotExpr>(rhs);
    }

    // Set that retains order of things added.
    // Or, vector that allows insertion if element doesn't exist.
    // Keeps two copies of everything, so don't put large things in it.
    // TODO: hide vector inside class and provide proper accessor methods.
    template <typename T>
    class vector_set : public vector<T> {
        map<T, size_t> _posn;

    public:
        vector_set() {}
        virtual ~vector_set() {}

        // Copy ctor.
        vector_set(const vector_set& src) :
            vector<T>(src), _posn(src._posn) {}
    
        virtual size_t count(const T& val) const {
            return _posn.count(val);
        }
        virtual void insert(const T& val) {
            if (_posn.count(val) == 0) {
                vector<T>::push_back(val);
                _posn[val] = vector<T>::size() - 1;
            }
        }
        virtual void erase(const T& val) {
            if (_posn.count(val) > 0) {
                size_t op = _posn.at(val);
                vector<T>::erase(vector<T>::begin() + op);
                for (auto pi : _posn) {
                    auto& p = pi.second;
                    if (p > op)
                        p--;
                }
            }
        }
        virtual void clear() {
            vector<T>::clear();
            _posn.clear();
        }
    };

    // Types of dependencies.
    // Must keep this consistent with list in stencil_calc.hpp.
    // TODO: make this common code.
    enum DepType {
        certain_dep,
        possible_dep,
        num_deps
    };

    // Dependencies between equations.
    // eq_deps[A].count(B) > 0 => A depends on B.
    class EqDeps {
    protected:
        typedef unordered_map<EqualsExprPtr, unordered_set<EqualsExprPtr>> DepMap;
        typedef set<EqualsExprPtr> SeenSet;

        DepMap _deps;               // direct dependencies.
        DepMap _full_deps;          // direct and indirect dependencies.
        SeenSet _all;               // all expressions.
        bool _done;                 // indirect dependencies added?
    
        // Recursive search starting at 'a'.
        // Fill in _full_deps.
        virtual bool _analyze(EqualsExprPtr a, SeenSet* seen);
    
    public:

        EqDeps() : _done(false) {}
        virtual ~EqDeps() {}
    
        // Declare that eq a depends on b.
        virtual void set_dep_on(EqualsExprPtr a, EqualsExprPtr b) {
            _deps[a].insert(b);
            _all.insert(a);
            _all.insert(b);
            _done = false;
        }
    
        // Check whether eq a depends on b.
        virtual bool is_dep_on(EqualsExprPtr a, EqualsExprPtr b) const {
            assert(_done || _deps.size() == 0);
            return _full_deps.count(a) && _full_deps.at(a).count(b) > 0;
        }
    
        // Checks for dependencies in either direction.
        virtual bool is_dep(EqualsExprPtr a, EqualsExprPtr b) const {
            return is_dep_on(a, b) || is_dep_on(b, a);
        }

        // Does recursive analysis to turn all indirect dependencies to direct
        // ones.
        virtual void analyze() {
            if (_done)
                return;
            for (auto a : _all)
                if (_full_deps.count(a) == 0)
                    _analyze(a, NULL);
            _done = true;
        }
    };

    typedef map<DepType, EqDeps> EqDepMap;

    // A list of unique equation ptrs.
    typedef vector_set<EqualsExprPtr> EqList;

    // Map of expressions: key = expression ptr, value = if-condition ptr.
    // We use this to simplify the process of replacing statements
    //  when an if-condition is encountered.
    // Example: key: grid(t,x)==grid(t,x+1); value: x>5;
    typedef map<EqualsExpr*, BoolExprPtr> CondMap;

    // A set of equations and related data.
    class Eqs {

    protected:
    
        // Equations(s) describing how values in this grid are computed.
        EqList _eqs;          // just equations w/o conditions.
        CondMap _conds;       // map from equations to their conditions, if any.

    public:

        Eqs() {}
        virtual ~Eqs() {}

        // Equation accessors.
        virtual void addEq(EqualsExprPtr ep) {
            _eqs.insert(ep);
        }
        virtual void addCondEq(EqualsExprPtr ep, BoolExprPtr cond) {
            _eqs.insert(ep);
            _conds[ep.get()] = cond;
        }
        virtual const EqList& getEqs() const {
            return _eqs;
        }
        virtual int getNumEqs() const {
            return _eqs.size();
        }

        // Get the condition associated with an expression.
        // If there is no condition, a null pointer is returned.
        virtual const BoolExprPtr getCond(EqualsExprPtr ep) {
            return getCond(ep.get());
        }
        virtual const BoolExprPtr getCond(EqualsExpr* ep) {
            if (_conds.count(ep))
                return _conds.at(ep);
            else
                return nullptr;
        }

        // Visit all equations.
        // Will NOT visit conditions.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& ep : _eqs) {
                ep->accept(ev);
            }
        }

        // Find dependencies based on all eqs.  If 'eq_deps' is
        // set, save dependencies between eqs.
        virtual void findDeps(IntTuple& pts,
                              const string& stepDim,
                              EqDepMap* eq_deps);

        // Check for illegal dependencies in all equations.
        // Exit with error if any found.
        virtual void checkDeps(IntTuple& pts,
                               const string& stepDim) {
            findDeps(pts, stepDim, NULL);
        }
    };

    ///////// Grids ////////////

    class Grid;
    
    // A list of grids.  This holds pointers to grids defined by the stencil
    // class in the order in which they are added via the INIT_GRID_* macros.
    class Grids : public vector_set<Grid*> {
    public:
    
        Grids() {}
        virtual ~Grids() {}

        // Copy ctor.
        // Copies list of grid pointers, but not grids (shallow copy).
        Grids(const Grids& src) : vector_set<Grid*>(src) {}
    };

    typedef set<GridPoint> GridPointSet;
    typedef set<GridPointPtr> GridPointPtrSet;
    typedef vector<GridPoint> GridPointVec;

    // A 'GridIndex' is simply a pointer to a numerical expression.
    typedef NumExprPtr GridIndex;

    // A 'Condition' is simply a pointer to a binary expression.
    typedef BoolExprPtr Condition;

    // A class for a Grid.
    // This is a generic container for all variables to be accessed
    // from the kernel. A 0-D grid is a scalar, a 1-D grid is an array, etc.
    // Dims can be the step dim, a domain dim, or anything else.
    class Grid : public virtual yc_grid {

    protected:
        string _name;           // name of the grid.
        IndexExprPtrVec _dims;  // dimensions.

        // Ptr to enclosing solution.
        // This is a mechanism to get access to parent's vars.
        StencilSolution* _soln = 0;

        // Values below are computed based on equations.
        
        // Min and max const indices for each dim.
        IntTuple _minIndices, _maxIndices;
        
        // Max abs-value of non-step-index halos required by all eqs at
        // various step-index values.
        // TODO: keep separate pos and neg halos.
        // TODO: keep separate halos for each equation.
        map<int, IntTuple> _halos;  // key: step-dim offset.
    
    public:
        // Ctors.
        Grid(string name, StencilSolution* soln,
             const IndexExprPtrVec& dims);
        Grid(string name, StencilSolution* soln,
             IndexExprPtr dim1 = nullptr,
             IndexExprPtr dim2 = nullptr,
             IndexExprPtr dim3 = nullptr,
             IndexExprPtr dim4 = nullptr,
             IndexExprPtr dim5 = nullptr,
             IndexExprPtr dim6 = nullptr);

        // Dtor.
        virtual ~Grid() { }

        // Name accessors.
        const string& getName() const { return _name; }
        void setName(const string& name) { _name = name; }

        // Access dims.
        virtual const IndexExprPtrVec& getDims() const { return _dims; }

        // Step dim or null if none.
        virtual const IndexExprPtr getStepDim() const {
            for (auto d : _dims)
                if (d->getType() == STEP_INDEX)
                    return d;
            return nullptr;
        }

        // Access to solution.
        virtual StencilSolution* getSoln() { return _soln; }
        virtual void setSoln(StencilSolution* soln) { _soln = soln; }

        // Get min and max observed indices.
        virtual const IntTuple& getMinIndices() const { return _minIndices; }
        virtual const IntTuple& getMaxIndices() const { return _maxIndices; }

        // Get the max size in 'dim' of halo across all step dims.
        virtual int getHaloSize(const string& dim) const {
            int h = 0;
            for (auto i : _halos) {
                auto& hi = i.second; // halo at step-val 'i'.
                auto* p = hi.lookup(dim);
                if (p)
                    h = std::max(h, *p);
            }
            return h;
        }

        // Determine how many values in step-dim are needed.
        virtual int getStepDimSize() const;

        // Update halos based on each value in 'offsets'.
        virtual void updateHalo(const IntTuple& offsets);

        // Update const indices based on 'indices'.
        virtual void updateConstIndices(const IntTuple& indices);
    
        // Create an expression to a specific point in this grid.
        // Note that this doesn't actually 'read' or 'write' a value;
        // it's just a node in an expression.
        virtual GridPointPtr makePoint(const NumExprPtrVec& args);
        virtual GridPointPtr makePoint() {
            NumExprPtrVec args;
            return makePoint(args);
        }

        // Convenience functions for zero dimensions (scalar).
        virtual operator NumExprPtr() { // implicit conversion.
            return makePoint();
        }
        virtual operator GridPointPtr() { // implicit conversion.
            return makePoint();
        }
        virtual GridPointPtr operator()() {
            return makePoint();
        }

        // Convenience functions for one dimension (array).
        virtual GridPointPtr operator[](const NumExprArg i1) {
            NumExprPtrVec args;
            args.push_back(i1);
            return makePoint(args);
        }
        virtual GridPointPtr operator()(const NumExprArg i1) {
            return operator[](i1);
        }

        // Convenience functions for more dimensions.
        virtual GridPointPtr operator()(const NumExprArg i1, const NumExprArg i2) {
            NumExprPtrVec args;
            args.push_back(i1);
            args.push_back(i2);
            return makePoint(args);
        }
        virtual GridPointPtr operator()(const NumExprArg i1, const NumExprArg i2,
                                        const NumExprArg i3) {
            NumExprPtrVec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            return makePoint(args);
        }
        virtual GridPointPtr operator()(const NumExprArg i1, const NumExprArg i2,
                                        const NumExprArg i3, const NumExprArg i4) {
            NumExprPtrVec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            args.push_back(i4);
            return makePoint(args);
        }
        virtual GridPointPtr operator()(const NumExprArg i1, const NumExprArg i2,
                                        const NumExprArg i3, const NumExprArg i4,
                                        const NumExprArg i5) {
            NumExprPtrVec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            args.push_back(i4);
            args.push_back(i5);
            return makePoint(args);
        }
        virtual GridPointPtr operator()(const NumExprArg i1, const NumExprArg i2,
                                        const NumExprArg i3, const NumExprArg i4,
                                        const NumExprArg i5, const NumExprArg i6) {
            NumExprPtrVec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            args.push_back(i4);
            args.push_back(i5);
            args.push_back(i6);
            return makePoint(args);
        }

        // APIs.
        virtual const string& get_name() const {
            return _name;
        }
        virtual int get_num_dims() const {
            return int(_dims.size());
        }
        virtual const string& get_dim_name(int n) const {
            assert(n >= 0);
            assert(n < get_num_dims());
            auto dp = _dims.at(n);
            assert(dp);
            return dp->getName();
        }
        virtual std::vector<std::string> get_dim_names() const;
        virtual yc_grid_point_node_ptr
        new_relative_grid_point(int dim1_offset,
                                int dim2_offset,
                                int dim3_offset,
                                int dim4_offset,
                                int dim5_offset,
                                int dim6_offset);
        virtual yc_grid_point_node_ptr
        new_relative_grid_point(std::vector<int> dim_offsets);
    };

    // Settings for the compiler.
    // May be provided via cmd-line or API.
    class CompilerSettings {
    public:
        int _elem_bytes = 4;    // bytes in an FP element.
        IntTuple _foldOptions;    // vector fold.
        IntTuple _clusterOptions; // cluster multipliers.
        bool _firstInner = true; // first dimension of fold is unit step.
        string _eq_group_basename_default = "stencil";
        bool _allowUnalignedLoads = false;
        int _haloSize = 0;      // 0 => calculate each halo separately and automatically.
        int _stepAlloc = 0;     // 0 => calculate step allocation automatically.
        int _maxExprSize = 50;
        int _minExprSize = 2;
        bool _doCse = true;      // do common-subexpr elim.
        bool _doComb = true;    // combine commutative operations.
        bool _doOptCluster = true; // apply optimizations also to cluster.
        bool _find_deps = true;  // find dependencies between equations.
        string _eqGroupTargets;  // how to group equations.
    };
    
    // Stencil dimensions.
    struct Dimensions {
        string _stepDim;         // step dimension.
        IntTuple _domainDims;    // domain dims with zero value.
        IntTuple _stencilDims;       // step and domain dims.
        IntTuple _miscDims;      // misc dims that are not the step or domain.

        // Following contain only domain dims.
        IntTuple _scalar;       // points in scalar (value 1 in each).
        IntTuple _fold;         // points in fold.
        IntTuple _clusterPts;    // cluster size in points.
        IntTuple _clusterMults;  // cluster size in vectors.

        Dimensions() {}
        virtual ~Dimensions() {}
    
        // Find the dimensions to be used.
        void setDims(Grids& grids,
                     CompilerSettings& settings,
                     int vlen,
                     bool is_folding_efficient,
                     ostream& os);
    };

    // A named equation group, which contains one or more grid-update equations.
    // All equations in a group must have the same condition.
    // Equations should not have inter-dependencies because they will be
    // combined into a single expression.  TODO: make this a proper class, e.g.,
    // encapsulate the fields.
    class EqGroup {
    protected:
        EqList _eqs; // expressions in this eqGroup (not including conditions).
        Grids _outGrids;          // grids updated by this eqGroup.
        Grids _inGrids;          // grids read from by this eqGroup.
        const Dimensions* _dims = 0;

        // Other eq-groups that this group depends on. This means that an
        // equation in this group has a grid value on the RHS that appears in
        // the LHS of the dependency.
        map<DepType, set<string>> _dep_on;

    public:
        string baseName;            // base name of this eqGroup.
        int index;                  // index to distinguish repeated names.
        BoolExprPtr cond;           // condition (default is null).

        // Ctor.
        EqGroup(const Dimensions& dims) : _dims(&dims) {

            // Create empty map entries.
            for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1)) {
                _dep_on[dt];
            }
        }
        virtual ~EqGroup() {}

        // Add an equation.
        // If 'update_stats', update grid and halo data.
        virtual void addEq(EqualsExprPtr ee, bool update_stats = true);
    
        // Visit all the equations.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& ep : _eqs) {
#ifdef DEBUG_EQ_GROUP
                cout << "EqGroup: visiting " << ep->makeQuotedStr() << endl;
#endif
                ep->accept(ev);
            }
        }

        // Get the list of all equations.
        // Does NOT return condition.
        virtual const EqList& getEqs() const {
            return _eqs;
        }

        // Visit the condition.
        // Return true if there was one to visit.
        virtual bool visitCond(ExprVisitor* ev) {
            if (cond.get()) {
                cond->accept(ev);
                return true;
            }
            return false;
        }

        // Get the full name.
        virtual string getName() const;

        // Get a string description.
        virtual string getDescription(bool show_cond = true,
                                      string quote = "'") const;

        // Get number of equations.
        virtual int getNumEqs() const {
            return _eqs.size();
        }

        // Get grids output and input.
        virtual const Grids& getOutputGrids() const {
            return _outGrids;
        }
        virtual const Grids& getInputGrids() const {
            return _inGrids;
        }

        // Get whether this eq-group depends on eg2.
        // Must have already been set via setDepOn().
        virtual bool isDepOn(DepType dt, const EqGroup& eq2) const {
            return _dep_on.at(dt).count(eq2.getName()) > 0;
        }

        // Get dependencies on this eq-group.
        virtual const set<string>& getDeps(DepType dt) const {
            return _dep_on.at(dt);
        }
    
        // Set dependency on eg2 if this eq-group is dependent on it.
        // Return whether dependent.
        virtual bool setDepOn(DepType dt, EqDepMap& eq_deps, const EqGroup& eg2);

        // Replicate each equation at the non-zero offsets for
        // each vector in a cluster.
        virtual void replicateEqsInCluster(Dimensions& dims);
        
        // Print stats for the equation(s) in this group.
        virtual void printStats(ostream& os, const string& msg);
    };

    // Container for multiple equation groups.
    class EqGroups : public vector<EqGroup> {
    protected:

        // Copy of some global data.
        string _basename_default;
        const Dimensions* _dims = 0;

        // Track grids that are udpated.
        Grids _outGrids;

        // Map to track indices per eq-group name.
        map<string, int> _indices;

        // Track equations that have been added already.
        set<EqualsExprPtr> _eqs_in_groups;
    
        // Add expression 'eq' with condition 'cond' to eq-group with 'baseName'
        // unless alread added.  The corresponding index in '_indices' will be
        // incremented if a new group is created.
        // 'eq_deps': pre-computed dependencies between equations.
        // Returns whether a new group was created.
        virtual bool addExprToGroup(EqualsExprPtr eq,
                                    BoolExprPtr cond, // may be nullptr.
                                    const string& baseName,
                                    EqDepMap& eq_deps);

    public:
        EqGroups() {}
        EqGroups(const string& basename_default, const Dimensions& dims) :
            _basename_default(basename_default),
            _dims(&dims) {}
        virtual ~EqGroups() {}

        virtual void set_basename_default(const string& basename_default) {
            _basename_default = basename_default;
        }
        virtual void set_dims(const Dimensions& dims) {
            _dims = &dims;
        }
        
        // Separate a set of equations into eqGroups based
        // on the target string.
        // Target string is a comma-separated list of key-value pairs, e.g.,
        // "eqGroup1=foo,eqGroup2=bar".
        // In this example, all eqs updating grid names containing 'foo' go in eqGroup1,
        // all eqs updating grid names containing 'bar' go in eqGroup2, and
        // each remaining eq goes in an eqGroup named after its grid.
        void makeEqGroups(Eqs& eqs,
                          const string& targets,
                          EqDepMap& eq_deps);
        void makeEqGroups(Eqs& eqs,
                          const string& targets,
                          IntTuple& pts,
                          bool find_deps) {
            EqDepMap eq_deps;
            if (find_deps)
                eqs.findDeps(pts, _dims->_stepDim, &eq_deps);
            makeEqGroups(eqs, targets, eq_deps);
        }

        virtual const Grids& getOutputGrids() const {
            return _outGrids;
        }

        // Visit all the equations in all eqGroups.
        // This will not visit the conditions.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& eg : *this)
                eg.visitEqs(ev);
        }

        // Replicate each equation at the non-zero offsets for
        // each vector in a cluster.
        virtual void replicateEqsInCluster(Dimensions& dims) {
            for (auto& eg : *this)
                eg.replicateEqsInCluster(dims);
        }

        // Reorder groups based on dependencies.
        virtual void sort();
    
        // Print a list of eqGroups.
        virtual void printInfo(ostream& os) const {
            os << "Identified stencil equation-groups:" << endl;
            for (auto& eq : *this) {
                for (auto gp : eq.getOutputGrids()) {
                    string eqName = eq.getName();
                    os << "  Equation group '" << eqName << "' updates grid '" <<
                        gp->getName() << "'." << endl;
                }
            }
        }

        // Print stats for the equation(s) in all groups.
        virtual void printStats(ostream& os, const string& msg);

        // Apply optimizations requested in settings.
        void optimizeEqGroups(CompilerSettings& settings,
                              const string& descr,
                              bool printSets,
                              ostream& os);
    };

    // A 'GridValue' is simply a pointer to an expression.
    typedef NumExprPtr GridValue;
 
    // Use SET_VALUE_FROM_EXPR for creating a string to insert any C++ code
    // that evaluates to a real_t.
    // The 1st arg must be the LHS of an assignment statement.
    // The 2nd arg must evaluate to a real_t (float or double) expression,
    // but it must NOT include access to a grid.
    // The code string is constructed as if writing to an ostream,
    // so '<<' operators may be used to evaluate local variables.
    // Floating-point variables will be printed w/o loss of precision.
    // The code may access the following:
    // - Any parameter to the 'calc_stencil_{cluster,scalar}' generated functions,
    //   including fields of the user-defined 'context' object.
    // - A variable within the global or current namespace where it will be used.
    // - A local variable in the 'value' method; in this case, the value
    //   of the local var must be evaluated and inserted in the expr.
    // Example code:
    //   GridValue v;
    //   SET_VALUE_FROM_EXPR(v =, "_context->temp * " << 0.2);
    //   SET_VALUE_FROM_EXPR(v +=, "_context->coeff[" << r << "]");
    // This example would generate the following partial expression (when r=9):
    //   (_context->temp * 2.00000000000000000e-01) + (_context->coeff[9])
#define SET_VALUE_FROM_EXPR(lhs, rhs) do {      \
        ostringstream oss;                      \
        oss << setprecision(17) << scientific;  \
        oss << "(" << rhs << ")";               \
        lhs  make_shared<CodeExpr>(oss.str());  \
    } while(0)
    
} // namespace yask.
    
#endif
