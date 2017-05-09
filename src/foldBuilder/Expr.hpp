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

#ifndef EXPR_HPP
#define EXPR_HPP

#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <cstdarg>
#include <assert.h>

#include "Tuple.hpp"

using namespace std;

namespace yask {

    // A tuple of integers, used for dimensions, indices, etc.
    typedef Tuple<int> IntTuple;

    // Forward-declarations of expressions.
    class Expr;
    typedef shared_ptr<Expr> ExprPtr;
    class NumExpr;
    typedef shared_ptr<NumExpr> NumExprPtr;
    class BoolExpr;
    typedef shared_ptr<BoolExpr> BoolExprPtr;
    class EqualsExpr;
    typedef shared_ptr<EqualsExpr> EqualsExprPtr;
    class IfExpr;
    typedef shared_ptr<IfExpr> IfExprPtr;
    class IntTupleExpr;
    typedef shared_ptr<IntTupleExpr> IntTupleExprPtr;
    typedef vector<NumExprPtr> NumExprPtrVec;

    // Forward-declare expression visitor.
    class ExprVisitor;

    // Forward-declaration of a grid and a grid point expression.
    class Grid;
    class GridPoint;
    typedef shared_ptr<GridPoint> GridPointPtr;

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

    // The '==' operator for comparing values.
    BoolExprPtr operator==(const NumExprPtr lhs, const NumExprPtr rhs);

    // Other comparison operators can only be used for comparing.
    BoolExprPtr operator!=(const NumExprPtr lhs, const NumExprPtr rhs);
    BoolExprPtr operator<(const NumExprPtr lhs, const NumExprPtr rhs);
    BoolExprPtr operator>(const NumExprPtr lhs, const NumExprPtr rhs);
    BoolExprPtr operator<=(const NumExprPtr lhs, const NumExprPtr rhs);
    BoolExprPtr operator>=(const NumExprPtr lhs, const NumExprPtr rhs);

    // Logical operators.
    BoolExprPtr operator&&(const BoolExprPtr lhs, const BoolExprPtr rhs);
    //BoolExprPtr operator&&(bool lhs, const BoolExprPtr rhs);
    //BoolExprPtr operator&&(const BoolExprPtr lhs, bool rhs);
    BoolExprPtr operator||(const BoolExprPtr lhs, const BoolExprPtr rhs);
    //BoolExprPtr operator||(bool lhs, const BoolExprPtr rhs);
    //BoolExprPtr operator||(const BoolExprPtr lhs, bool rhs);
    BoolExprPtr operator!(const BoolExprPtr rhs);

    // Boundary indices.
    NumExprPtr first_index(const NumExprPtr dim);
    NumExprPtr last_index(const NumExprPtr dim);

    // A function to create a constant double expression.
    // Usually not needed due to operator overloading.
    NumExprPtr constNum(double rhs);

    //// Classes to implement parts of expressions.
    // The expressions are constructed at run-time when the
    // StencilBase::define() method is called.

    // The base class for all expression nodes.
    class Expr : public virtual expr_node {

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
    class NumExpr : public Expr, public virtual number_node {
    public:
    
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
    
        // Create a deep copy of this expression.
        // For this to work properly, each derived type
        // should also implement a deep-copy copy ctor.
        virtual NumExprPtr clone() const =0;
    };

    // Boolean value.
    class BoolExpr : public Expr, public virtual bool_node  {
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
    class ConstExpr : public NumExpr, public virtual const_number_node {
    protected:
        double _f = 0.0;

    public:
        ConstExpr(double f) : _f(f) { }
        ConstExpr(const ConstExpr& src) : _f(src._f) { }
        virtual ~ConstExpr() { }

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

    // Special grid index values.
    enum IndexType {
        FIRST_INDEX,
        LAST_INDEX
    };

    // Begin/end of problem domain.
    // This is an expression leaf-node.
    class IndexExpr : public NumExpr {
    protected:
        string _dirName;
        IndexType _type;

    public:
        IndexExpr(NumExprPtr dim, IndexType type);
        IndexExpr(const IndexExpr& src) :
            _dirName(src._dirName),
            _type(src._type) { }
        virtual ~IndexExpr() { }

        const string& getDirName() const { return _dirName; }
        IndexType getType() const { return _type; }
        string getFnName() const {
            if (_type == FIRST_INDEX)
                return "FIRST_INDEX";
            else if (_type == LAST_INDEX)
                return "LAST_INDEX";
            else {
                cerr << "Error: internal error in IndexExpr\n";
                exit(1);
            }
        }
        virtual void accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool isSame(const Expr* other) const {
            auto p = dynamic_cast<const IndexExpr*>(other);
            return p && _dirName == p->_dirName && _type == p->_type;
        }
   
        // Create a deep copy of this expression.
        virtual NumExprPtr clone() const { return make_shared<IndexExpr>(*this); }
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
    class NegExpr : public UnaryNumExpr, public virtual negate_node {
    public:
        NegExpr(NumExprPtr rhs) :
            UnaryNumExpr(opStr(), rhs) { }
        NegExpr(const NegExpr& src) :
            UnaryExpr(src) { }

        static string opStr() { return "-"; }
        virtual double getNumVal() const {
            double rhs = _rhs->getNumVal();
            return -rhs;
        }
        virtual NumExprPtr clone() const {
            return make_shared<NegExpr>(*this);
        }

        // APIs.
        virtual number_node_ptr get_rhs() {
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
        virtual double getNumVal() const {              \
            double lhs = _lhs->getNumVal();             \
            double rhs = _rhs->getNumVal();             \
            return oper;                                \
        }                                               \
        virtual NumExprPtr clone() const {              \
            return make_shared<type>(*this);            \
        }                                               \
        virtual number_node_ptr get_lhs() {             \
            return getLhs();                            \
        }                                               \
        virtual number_node_ptr get_rhs() {             \
            return getRhs();                            \
        }                                               \
    }
    BIN_NUM_EXPR(SubExpr, subtract_node, "-", lhs - rhs);
    BIN_NUM_EXPR(DivExpr, divide_node, "/", lhs / rhs); // TODO: add check for div-by-0.
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
    class type : public BinaryBoolExpr {        \
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
    class CommutativeExpr : public NumExpr, public virtual commutative_number_node {
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
        virtual number_node_ptr get_operand(int i) {
            if (i >= 0 &&
                size_t(i) < _ops.size())
                return _ops.at(size_t(i));
            else
                return nullptr;
        }
        virtual void add_operand(number_node_ptr node) {
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
    COMM_EXPR(MultExpr, multiply_node, "*", 1.0, lhs * rhs);
    COMM_EXPR(AddExpr, add_node, "+", 0.0, lhs + rhs);
#undef COMM_EXPR

    // A tuple expression.
    // This is an expression leaf-node.
    class IntTupleExpr : public NumExpr, public IntTuple {

    public:
        IntTupleExpr(const IntTuple& tuple) :
            IntTuple(tuple) { }
        IntTupleExpr(const IntTupleExpr& src) :
            IntTuple(src) { }
        IntTupleExpr(const IntTupleExprPtr src) :
            IntTuple(*src) { }
    
        virtual ~IntTupleExpr() { }

        virtual double getNumVal() const {
            return double(getDirVal());
        }
        virtual int getIntVal() const {
            return getDirVal();
        }

        // Some comparisons.
        bool operator==(const IntTupleExpr& rhs) const {
            return IntTuple::operator==(rhs);
        }
        bool operator<(const IntTupleExpr& rhs) const {
            return IntTuple::operator<(rhs);
        }

        // Take ev to each value.
        virtual void accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool isSame(const Expr* other) const {
            auto p = dynamic_cast<const IntTupleExpr*>(other);

            // Only compare dimensions, not values.
            return p && areDimsSame(*p);
        }
    
        // Create a deep copy of this expression.
        virtual NumExprPtr clone() const { return make_shared<IntTupleExpr>(*this); }
    };

    // One specific point in a grid.
    // This is an expression leaf-node.
    class GridPoint : public NumExpr,
                      public IntTuple,
                      public virtual grid_point_node {

    protected:
        Grid* _grid;          // the grid this point is from.

    public:

        // Construct an n-D point given a grid and offset.
        GridPoint(Grid* grid, const IntTuple& offsets) :
            IntTuple(offsets),
            _grid(grid)
        {
            assert(areDimsSame(offsets));
        }

        // Copy ctor.
        // Note that _grid is a shallow copy!
        GridPoint(const GridPoint& src) :
            IntTuple(src), _grid(src._grid) { }
        GridPoint(const GridPointPtr src) :
            GridPoint(*src) { }

        // Construct from another point, but change location.
        GridPoint(GridPoint* gp, const IntTuple& offsets) :
            IntTuple(offsets), _grid(gp->_grid)
        {
            assert(areDimsSame(offsets));
        }

        // Dtor.
        virtual ~GridPoint() {}

        // Get parent grid info.
        const Grid* getGrid() const { return _grid; }
        Grid* getGrid() { return _grid; }
        virtual const string& getName() const;
        virtual bool isParam() const;

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
        virtual string makeStr() const;
        
        // Create a deep copy of this expression,
        // except pointed-to grid is not copied.
        virtual NumExprPtr clone() const { return make_shared<GridPoint>(*this); }
        virtual GridPointPtr cloneGridPoint() const { return make_shared<GridPoint>(*this); }
    
        // Determine whether this is 'ahead of' rhs in given direction.
        virtual bool isAheadOfInDir(const GridPoint& rhs, const IntTuple& dir) const;

        // APIs.
        virtual grid* get_grid();
    };
} // namespace yask.

// Define hash function for GridPoint for unordered_{set,map}.
namespace std {
    using namespace yask;
    
    template <> struct hash<GridPoint> {
        size_t operator()(const GridPoint& k) const {
            size_t h1 = hash<string>{}(k.getName());
            size_t h2 = hash<string>{}(k.makeDimValStr());
            return h1 ^ (h2 << 1);
        }
    };
}

namespace yask {
    
    // Equality operator for a grid point.
    // This defines the LHS as equal to the RHS; it is NOT
    // a comparison operator.
    // (Not inherited from BinaryExpr because LHS is special.)
    class EqualsExpr : public UnaryNumExpr,
                       public virtual equation_node {
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
        virtual grid_point_node_ptr get_lhs() {
            return _lhs;
        }
        virtual number_node_ptr get_rhs() {
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

    typedef set<GridPoint> GridPointSet;
    typedef set<GridPointPtr> GridPointPtrSet;
    typedef vector<GridPoint> GridPointVec;

    // A 'GridIndex' is simply a pointer to a numerical expression.
    typedef NumExprPtr GridIndex;

    // A 'Condition' is simply a pointer to a binary expression.
    typedef BoolExprPtr Condition;

    // A class for a Grid or a Parameter.
    // Dims in the IntTuple describe the grid or param.
    // For grids, values in the IntTuple are ignored (grids are sized at run-time).
    // For params, values in the IntTuple define the sizes.
    class Grid : public IntTuple,
                 public virtual grid {

        // Should not be copying grids.
        Grid(const Grid& src) { assert(0); }
    
    protected:
        string _name;               // name of the grid.

        // Note: at this time, a Grid is either indexed only by stencil indices,
        // and a 'parameter' is indexed only by non-stencil indices. So, scalar
        // parameter values will be broadcast to all elements of a grid
        // vector. TODO: generalize this so that a parameter is just one special
        // case of a grid.
        bool _isParam = false;              // is a parameter.

        // Ptr to object to store equations when they are encountered.
        Eqs* _eqs = 0;
    
        // Max abs-value of non-step-index halos required by all eqs at
        // various step-index values.
        // TODO: keep separate pos and neg halos.
        // TODO: keep separate halos for each equation.
        string _stepDim;            // Assumes all eqs use same step-dim.
        map<int, IntTuple> _halos;  // key: step-dim offset.

    public:
        Grid() { }
        virtual ~Grid() { }

        // Name accessors.
        const string& getName() const { return _name; }
        void setName(const string& name) { _name = name; }

        // Param-type accessors.
        bool isParam() const { return _isParam; }
        void setParam(bool isParam) { _isParam = isParam; }

        // Access to all equations in this stencil.
        virtual Eqs* getEqs() { return _eqs; }
        virtual void setEqs(Eqs* eqs) { _eqs = eqs; }
    
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

        // Update halos based on each value in 'vals' given the step-dim 'stepDim'.
        virtual void updateHalo(const string& stepDim, const IntTuple& vals);
    
        // Create an expression to a specific point in this grid.
        // Note that this doesn't actually 'read' or 'write' a value;
        // it's just a node in an expression.
        virtual GridPointPtr makePoint(int count, ...);

        // Convenience functions for zero dimensions (scalar).
        virtual operator NumExprPtr() { // implicit conversion.
            assert(_isParam);
            return makePoint(0);
        }
        virtual operator GridPointPtr() { // implicit conversion.
            assert(_isParam);
            return makePoint(0);
        }
        virtual GridPointPtr operator()() {
            assert(_isParam);
            return makePoint(0);
        }

        // Convenience functions for one dimension (array).
        // TODO: separate out ExprPtr varieties for Grid
        // and int varieties for Param.
        virtual GridPointPtr operator[](int i1) {
            assert(_isParam);
            return makePoint(1, i1);
        }
        virtual GridPointPtr operator()(int i1) {
            assert(_isParam);
            return makePoint(1, i1);
        }
        virtual GridPointPtr operator[](const NumExprPtr i1) {
            return makePoint(1, i1->getIntVal());
        }
        virtual GridPointPtr operator()(const NumExprPtr i1) {
            return makePoint(1, i1->getIntVal());
        }

        // Convenience functions for more dimensions.
        virtual GridPointPtr operator()(int i1, int i2) {
            assert(_isParam);
            return makePoint(2, i1, i2);
        }
        virtual GridPointPtr operator()(int i1, int i2, int i3) {
            assert(_isParam);
            return makePoint(3, i1, i2, i3);
        }
        virtual GridPointPtr operator()(int i1, int i2, int i3, int i4) {
            assert(_isParam);
            return makePoint(4, i1, i2, i3, i4);
        }
        virtual GridPointPtr operator()(int i1, int i2, int i3, int i4, int i5) {
            assert(_isParam);
            return makePoint(5, i1, i2, i3, i4, i5);
        }
        virtual GridPointPtr operator()(int i1, int i2, int i3, int i4, int i5, int i6) {
            assert(_isParam);
            return makePoint(6, i1, i2, i3, i4, i5, i6);
        }
        virtual GridPointPtr operator()(const NumExprPtr i1, const NumExprPtr i2) {
            return makePoint(2, i1->getIntVal(), i2->getIntVal());
        }
        virtual GridPointPtr operator()(const NumExprPtr i1, const NumExprPtr i2,
                                        const NumExprPtr i3) {
            return makePoint(3, i1->getIntVal(), i2->getIntVal(),
                             i3->getIntVal());
        }
        virtual GridPointPtr operator()(const NumExprPtr i1, const NumExprPtr i2,
                                        const NumExprPtr i3, const NumExprPtr i4) {
            return makePoint(4, i1->getIntVal(), i2->getIntVal(),
                             i3->getIntVal(), i4->getIntVal());
        }
        virtual GridPointPtr operator()(const NumExprPtr i1, const NumExprPtr i2,
                                        const NumExprPtr i3, const NumExprPtr i4,
                                        const NumExprPtr i5) {
            return makePoint(5, i1->getIntVal(), i2->getIntVal(),
                             i3->getIntVal(), i4->getIntVal(),
                             i5->getIntVal());
        }
        virtual GridPointPtr operator()(const NumExprPtr i1, const NumExprPtr i2,
                                        const NumExprPtr i3, const NumExprPtr i4,
                                        const NumExprPtr i5, const NumExprPtr i6) {
            return makePoint(6, i1->getIntVal(), i2->getIntVal(),
                             i3->getIntVal(), i4->getIntVal(),
                             i5->getIntVal(), i6->getIntVal());
        }

        // APIs.
        virtual const string& get_name() const {
            return _name;
        }
        virtual int get_num_dims() const {
            return getNumDims();
        }
        virtual const string& get_dim_name(int n) const {
            auto& dims = getDims();
            assert(n >= 0 && n < getNumDims());
            return *dims.at(n);
        }
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset) {
            return makePoint(1, dim1_offset);
        }
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset,
                                int dim2_offset) {
            return makePoint(2, dim1_offset, dim2_offset);
        }
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset,
                                int dim2_offset,
                                int dim3_offset) {
            return makePoint(3, dim1_offset, dim2_offset, dim3_offset);
        }
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset,
                                int dim2_offset,
                                int dim3_offset,
                                int dim4_offset) {
            return makePoint(4, dim1_offset, dim2_offset,
                             dim3_offset, dim4_offset);
        }
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset,
                                int dim2_offset,
                                int dim3_offset,
                                int dim4_offset,
                                int dim5_offset) {
            return makePoint(5, dim1_offset, dim2_offset,
                             dim3_offset, dim4_offset, dim5_offset);
        }
        virtual grid_point_node_ptr
        new_relative_grid_point(int dim1_offset,
                                int dim2_offset,
                                int dim3_offset,
                                int dim4_offset,
                                int dim5_offset,
                                int dim6_offset) {
            return makePoint(6, dim1_offset, dim2_offset, dim3_offset,
                             dim4_offset, dim5_offset, dim6_offset);
        }
    };

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

    // Aliases for parameters.
    // Even though these are just typedefs for now, don't interchange them.
    // TODO: make params just a special case of grids.
    typedef Grid Param;
    typedef Grids Params;

    // Stencil dimensions.
    struct Dimensions {
        IntTuple _allDims;          // all dims with zero value.
        IntTuple _dimCounts;        // how many grids use each dim.
        string _stepDim;            // step dimension.
        IntTuple _scalar, _fold;    // points in scalar and fold.
        IntTuple _clusterPts;       // cluster size in points.
        IntTuple _clusterMults;     // cluster size in vectors.
        IntTuple _miscDims;         // all other dims.

        Dimensions() {}
        virtual ~Dimensions() {}
    
        // Find the dimensions to be used.
        void setDims(Grids& grids,
                     string stepDim,
                     IntTuple& foldOptions,
                     bool firstInner,
                     IntTuple& clusterOptions,
                     bool allowUnalignedLoads,
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

        // Copy ctor.
        EqGroup(const EqGroup& src) :
            _eqs(src._eqs),
            _outGrids(src._outGrids),
            _inGrids(src._inGrids),
            _dims(src._dims),
            _dep_on(src._dep_on),
            baseName(src.baseName),
            index(src.index),
            cond(src.cond) {}

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
                                    BoolExprPtr cond,
                                    const string& baseName,
                                    EqDepMap& eq_deps);

    public:
        EqGroups(const string& basename_default, const Dimensions& dims) :
            _basename_default(basename_default),
            _dims(&dims) {}
        virtual ~EqGroups() {}

        // Copy ctor.
        EqGroups(const EqGroups& src) :
            vector<EqGroup>(src),
            _basename_default(src._basename_default),
            _dims(src._dims),
            _outGrids(src._outGrids),
            _indices(src._indices),
            _eqs_in_groups(src._eqs_in_groups)
        {}
    
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
