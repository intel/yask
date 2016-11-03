/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

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

#include <map>
#include <set>
#include <vector>
#include <cstdarg>
#include <assert.h>

#include "Tuple.hpp"

using namespace std;

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

// The '==' operator can define a grid value or be used for
// comparing values, depending on what the lhs is.
EqualsExprPtr operator==(GridPointPtr gpp, const NumExprPtr rhs);
EqualsExprPtr operator==(GridPointPtr gpp, double rhs);
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
class Expr {

public:
    Expr() { }
    virtual ~Expr() { }

    // For visitors.
    virtual void accept(ExprVisitor* ev) =0;
    virtual void accept(ExprVisitor* ev) const;

    // Check for expression equivalency.
    // Does *not* check value equivalency except for
    // constants.
    virtual bool isSame(const Expr* other) const =0;

    // Return a simple string expr.
    virtual string makeStr() const;
    virtual string makeQuotedStr() const {
        ostringstream oss;
        oss << "\"" << makeStr() << "\"";
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
};

// Convert pointer to the given ptr type or die w/an error.
template<typename T> shared_ptr<T> castExpr(ExprPtr ep, const string& descrip) {
    auto tp = dynamic_pointer_cast<T>(ep);
    if (!tp) {
        cerr << "error: expression '" << ep->makeStr() << "' is not a " <<
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

// Real or int values.
class NumExpr : public virtual Expr {
public:
    
    // Get the current value.
    // Exit with error if not known.
    virtual double getNumVal() const {
        cerr << "error: cannot evaluate '" << makeStr() <<
            "' for a known numerical value.\n";
        exit(1);
    }

    // Get the value as an integer.
    // Exits with error if not an integer.
    virtual int getIntVal() const {
        double val = getNumVal();
        int ival = int(val);
        if (val != double(ival)) {
            cerr << "error: '" << makeStr() <<
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

// Boolean values.
class BoolExpr : public virtual Expr {
public:

    // Get the current value.
    // Exit with error if not known.
    virtual bool getBoolVal() const {
        cerr << "error: cannot evaluate '" << makeStr() <<
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
class ConstExpr : public NumExpr {
protected:
    double _f;

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
};

// Types of indices.
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
            return "first_index";
        else if (_type == LAST_INDEX)
            return "last_index";
        else {
            cerr << "error: internal error in IndexExpr\n";
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
            _rhs->isSame(p->_rhs.get());
    }
};

// Various types of unary operators depending on input and output types.
typedef UnaryExpr<NumExpr, NumExprPtr> UnaryNumExpr;
typedef UnaryExpr<BoolExpr, BoolExprPtr> UnaryBoolExpr;
typedef UnaryExpr<BoolExpr, NumExprPtr> UnaryNum2BoolExpr;

// Negate operator.
class NegExpr : public UnaryNumExpr {
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
    ArgT _lhs;

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
#define BIN_NUM_EXPR(type, opstr, oper)                                 \
    class type : public BinaryNumExpr {                                 \
    public:                                                             \
    type(NumExprPtr lhs, NumExprPtr rhs) :                              \
        BinaryNumExpr(lhs, opStr(), rhs) { }                            \
    type(const type& src) :                                             \
        BinaryNumExpr(src) { }                                          \
    static string opStr() { return opstr; }                             \
    virtual double getNumVal() const {                                  \
        double lhs = _lhs->getNumVal();                                 \
        double rhs = _rhs->getNumVal();                                 \
        return oper;                                                    \
    }                                                                   \
    virtual NumExprPtr clone() const {                                  \
        return make_shared<type>(*this);                                \
    }                                                                   \
    }
BIN_NUM_EXPR(SubExpr, "-", lhs - rhs);
BIN_NUM_EXPR(DivExpr, "/", lhs / rhs); // TODO: add check for div-by-0.
#undef BIN_NUM_EXPR

// Boolean binary operators with numerical inputs.
// TODO: redo this with a template.
#define BIN_NUM2BOOL_EXPR(type, opstr, oper)                            \
    class type : public BinaryNum2BoolExpr {                            \
    public:                                                             \
    type(NumExprPtr lhs, NumExprPtr rhs) :                              \
        BinaryNum2BoolExpr(lhs, opStr(), rhs) { }                       \
    type(const type& src) :                                             \
        BinaryNum2BoolExpr(src) { }                                     \
    static string opStr() { return opstr; }                             \
    virtual bool getBoolVal() const {                                   \
        double lhs = _lhs->getNumVal();                                 \
        double rhs = _rhs->getNumVal();                                 \
        return oper;                                                    \
    }                                                                   \
    virtual BoolExprPtr clone() const {                                 \
        return make_shared<type>(*this);                                \
    }                                                                   \
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
#define BIN_BOOL_EXPR(type, opstr, oper)                           \
    class type : public BinaryBoolExpr {                           \
    public:                                                             \
    type(BoolExprPtr lhs, BoolExprPtr rhs) :                            \
        BinaryBoolExpr(lhs, opStr(), rhs) { }                           \
    type(const type& src) :                                             \
        BinaryBoolExpr(src) { }                                         \
    static string opStr() { return opstr; }                             \
    virtual bool getBoolVal() const {                                   \
        bool lhs = _lhs->getBoolVal();                                  \
        bool rhs = _rhs->getBoolVal();                                  \
        return oper;                                                    \
    }                                                                   \
    virtual BoolExprPtr clone() const {                                 \
        return make_shared<type>(*this);                                \
    }                                                                   \
    }
BIN_BOOL_EXPR(AndExpr, "&&", lhs && rhs);
BIN_BOOL_EXPR(OrExpr, "||", lhs || rhs);
#undef BIN_BOOL_EXPR

// A list of exprs with a common operator that can be rearranged,
// e.g., 'a * b * c' or 'a + b + c'.
// Still pure virtual because clone() not implemented.
class CommutativeExpr : public NumExpr {
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
};

// Commutative operators.
// TODO: redo this with a template.
#define COMM_EXPR(type, opstr, baseVal, oper)                           \
    class type : public CommutativeExpr {                               \
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
    virtual NumExprPtr clone() const { return make_shared<type>(*this); }  \
}
COMM_EXPR(MultExpr, "*", 1.0, lhs * rhs);
COMM_EXPR(AddExpr, "+", 0.0, lhs + rhs);
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
class GridPoint : public NumExpr, public IntTuple {

protected:
    Grid* _grid;          // the grid this point is from.

public:

    // Construct an n-D point.
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
    // TODO: check whether this overload is actually needed.
    virtual string makeStr() const;

    // Create a deep copy of this expression,
    // except pointed-to grid is not copied.
    virtual NumExprPtr clone() const { return make_shared<GridPoint>(*this); }
    virtual GridPointPtr cloneGridPoint() const { return make_shared<GridPoint>(*this); }
    
    // Determine whether this is 'ahead of' rhs in given direction.
    virtual bool isAheadOfInDir(const GridPoint& rhs, const IntTuple& dir) const;
};

// Equality operator for a grid point.
// (Not inherited from BinaryExpr because LHS is special.)
class EqualsExpr : public UnaryNumExpr {
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

    // Check for equivalency.
    virtual bool isSame(const Expr* other) const;

    // Create a deep copy of this expression.
    virtual NumExprPtr clone() const { return make_shared<EqualsExpr>(*this); }
    virtual EqualsExprPtr cloneEquals() const { return make_shared<EqualsExpr>(*this); }
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

///////// Grids ////////////

typedef set<GridPoint> GridPointSet;
typedef set<GridPointPtr> GridPointPtrSet;
typedef vector<GridPoint> GridPointVec;

// Map of expressions: key = expression, value = if-statement.
// We use this to simplify the process of handling equations
// that either do or do not have if-conditions.
// NB: both key and value will contain the equality.
// Example if there is an if-condition:
// key: grid(t,x)==grid(t,x+1); value: grid(t,x)==grid(t,x+1) if (x>5);
// Example if there is not an if-condition:
// key: grid(t,x)==grid(t,x+1); value: grid(t,x)==grid(t,x+1) if NULL;
typedef map<EqualsExprPtr, IfExprPtr> ExprMap;

// A 'GridIndex' is simply a pointer to an expression.
typedef NumExprPtr GridIndex;

// A 'Condition' is simply a pointer to a binary expression.
typedef BoolExprPtr Condition;

// A class for a collection of GridPoints.
// Dims in the IntTuple describe the grid or param.
// For grids, values in the IntTuple are ignored (grids are sized at run-time).
// For params, values in the IntTuple define the sizes.
class Grid : public IntTuple {
protected:
    string _name;               // name of the grid.

    // Note: at this time, a Grid is either indexed only by stencil indices,
    // and a 'parameter' is indexed only by non-stencil indices. So, scalar
    // parameter values will be broadcast to all elements of a grid
    // vector. TODO: generalize this so that grids may be projected between
    // different dimensionalities.
    bool _isParam;              // is a parameter.

    // specific points that have been created in this grid.
    GridPointPtrSet _points;

    // eqGroup(s) describing how values in this grid are computed.
    ExprMap _exprs;

    // Add a new point if needed and return pointer to it.
    // If it already exists, just return pointer.
    virtual GridPointPtr addPoint(GridPointPtr gpp) {
        auto i = _points.insert(gpp); // add if not found.
        return *i.first;
    }

public:
    Grid() { }
    virtual ~Grid() { }

    // Name accessors.
    const string& getName() const { return _name; }
    void setName(const string& name) { _name = name; }

    // Param-type accessors.
    bool isParam() const { return _isParam; }
    void setParam(bool isParam) { _isParam = isParam; }
    
    // Point accessors.
    const GridPointPtrSet& getPoints() const { return _points; }
    GridPointPtrSet& getPoints() { return _points; }

    // Expression accessors.
    virtual void addExpr(EqualsExprPtr ep, IfExprPtr cond) {
        _exprs[ep] = cond;
    }
    virtual const ExprMap& getExprs() const {
        return _exprs;
    }
    virtual ExprMap& getExprs() {
        return _exprs;
    }

    // Remove expressions and points.
    virtual void clearTemp() {
        _points.clear();
        _exprs.clear();
    }

#if 0
    // Visit all if-statements, if any are defined.
    virtual void visitExprs(ExprVisitor* ev) {
        for (auto i : _exprs) {
            i.second->accept(ev);
        }
    }
#endif
    
    // Create an expression to a specific point in this grid.
    // Note that this doesn't actually 'read' or 'write' a value;
    // it's just a node in an expression.
    virtual GridPointPtr makePoint(int count, ...) {

        // check for correct number of indices.
        if (count != size()) {
            cerr << "Error: attempt to access " << size() <<
                "-D grid '" << _name << "' with " << count << " indices.\n";
            exit(1);
        }

        // Copy the names from the grid to a new tuple.
        IntTuple pt = *this;

        // set the values in the tuple using the var args.
        va_list args;
        va_start(args, count);
        pt.setVals(count, args);
        va_end(args);

        // Create a point from the tuple.
        GridPointPtr gpp = make_shared<GridPoint>(this, pt);
        return addPoint(gpp);
    }

    // Convenience functions for zero dimensions (scalar).
    virtual operator NumExprPtr() { // implicit conversion.
        return makePoint(0);
    }
    virtual operator GridPointPtr() { // implicit conversion.
        return makePoint(0);
    }
    virtual GridPointPtr operator()() {
        return makePoint(0);
    }

    // Convenience functions for one dimension (array).
    // TODO: separate out ExprPtr varieties for Grid
    // and int varieties for Param.
    virtual GridPointPtr operator[](int i1) {
        return makePoint(1, i1);
    }
    virtual GridPointPtr operator()(int i1) {
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
        return makePoint(2, i1, i2);
    }
    virtual GridPointPtr operator()(int i1, int i2, int i3) {
        return makePoint(3, i1, i2, i3);
    }
    virtual GridPointPtr operator()(int i1, int i2, int i3, int i4) {
        return makePoint(4, i1, i2, i3, i4);
    }
    virtual GridPointPtr operator()(int i1, int i2, int i3, int i4, int i5) {
        return makePoint(5, i1, i2, i3, i4, i5);
    }
    virtual GridPointPtr operator()(int i1, int i2, int i3, int i4, int i5, int i6) {
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
};

// A list of grids.  This holds pointers to grids defined by the stencil
// class in the order in which they are added via the INIT_GRID_* macros.
class Grids : public vector<Grid*> {
    
public:

#if 0
    // Visit all expressions in all grids.
    virtual void visitExprs(ExprVisitor* ev);
#endif

    // Add a grid.
    virtual void add(Grid* ngp) {

        // Delete any matching old one.
        for (size_t i = 0; i < size(); i++) {
            auto ogp = at(i);
            if (ogp->getName() == ngp->getName()) {
                erase(begin() + i);
                break;
            }
        }

        // Add new one.
        push_back(ngp);
    }

    // Remove expressions and points in grids.
    virtual void clearTemp() {
        for (auto gp : *this)
            gp->clearTemp();
    }    
};

// Aliases for parameters.
// Even though these are just typedefs for now, don't interchange them.
// TODO: enforce the difference between grids and parameters.
typedef Grid Param;
typedef Grids Params;

// A named equation group, which contains one or more grid-update equations.
// Equations should not have inter-dependencies because they will be
// combined into a single expression.  TODO: make this a proper class, e.g.,
// encapsulate the fields.
typedef set<EqualsExprPtr> ExprSet;
struct EqGroup {
    string baseName;            // base name of this eqGroup.
    int index;                  // index to distinguish repeated names.
    BoolExprPtr cond;           // condition (default is null).
    ExprSet exprs;              // expressions in this eqGroup.
    set<Grid*> grids;           // grids updated by this eqGroup.

    // Visit all the expressions.
    virtual void visitExprs(ExprVisitor* ev) {
        for (auto& ep : exprs)
            ep->accept(ev);
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

    // Get number of points updated by the equations.
    virtual int getNumExprs() const {
        return exprs.size();
    }

    // Print stats for the equation(s) in this group.
    virtual void printStats(ostream& os, const string& msg);
};

// Container for equation groups.
class EqGroups : public vector<EqGroup> {
protected:
    set<Grid*> _eqGrids;        // all grids updated.

    // Add expressions from a grid to group(s) named groupName.
    // Returns whether a new group was created.
    virtual bool addExprsFromGrid(const string& groupName,
                                  map<string, int>& indices,
                                  Grid* gp);

public:
    EqGroups() {}
    virtual ~EqGroups() {}

    // Separate a set of grids into eqGroups based
    // on the target string.
    // Target string is a comma-separated list of key-value pairs, e.g.,
    // "eqGroup1=foo,eqGroup2=bar".
    // In this example, all grids with names containing 'foo' go in eqGroup1,
    // all grids with names containing 'bar' go in eqGroup2, and
    // each remaining grid goes in a eqGroup named after the grid.
    // Only grids with eqGroups are put in eqGroups.
    void findEqGroups(Grids& allGrids, const string& targets);

    const set<Grid*>& getEqGrids() const { return _eqGrids; }

    // Visit all the expressions in all eqGroups.
    virtual void visitExprs(ExprVisitor* ev) {
        for (auto& ep : *this)
            ep.visitExprs(ev);
    }
    
    // Print a list of eqGroups.
    virtual void printInfo(ostream& os) const {
        os << "Identified stencil eqGroups:" << endl;
        for (auto& eq : *this) {
            for (auto gp : eq.grids) {
                string eqName = eq.getName();
                os << "  Equation group '" << eqName << "' updates grid '" <<
                    gp->getName() << "'." << endl;
            }
        }
    }

    // Print stats for the equation(s) in all groups.
    virtual void printStats(ostream& os, const string& msg);
};

// Stencil dimensions.
struct Dimensions {
    IntTuple _allDims;                      // all dims with zero value.
    IntTuple _dimCounts;                    // how many grids use each dim.
    string _stepDim;                        // step dim.
    IntTuple _foldLengths, _clusterLengths; // dims in folds/clusters.
    IntTuple _miscDims;                     // all other dims.

    // Find the dimensions to be used.
    void setDims(Grids& grids,
                 string stepDim,
                 IntTuple& foldOptions,
                 IntTuple& clusterOptions,
                 bool allowUnalignedLoads,
                 ostream& os);
};

// A 'GridValue' is simply a pointer to an expression.
typedef NumExprPtr GridValue;

// Convenience macros for initializing grids in stencil ctors.
// Each names the grid according to the 'gvar' parameter and adds it
// to the default '_grids' collection.
// The dimensions are named according to the remaining parameters.
#define INIT_GRID_0D(gvar) \
    _grids.add(&gvar); gvar.setName(#gvar)
#define INIT_GRID_1D(gvar, d1) \
    INIT_GRID_0D(gvar); gvar.addDimBack(#d1, 1)
#define INIT_GRID_2D(gvar, d1, d2) \
    INIT_GRID_1D(gvar, d1); gvar.addDimBack(#d2, 1)
#define INIT_GRID_3D(gvar, d1, d2, d3) \
    INIT_GRID_2D(gvar, d1, d2); gvar.addDimBack(#d3, 1)
#define INIT_GRID_4D(gvar, d1, d2, d3, d4) \
    INIT_GRID_3D(gvar, d1, d2, d3); gvar.addDimBack(#d4, 1)
#define INIT_GRID_5D(gvar, d1, d2, d3, d4, d5) \
    INIT_GRID_4D(gvar, d1, d2, d3, d4); gvar.addDimBack(#d5, 1)
#define INIT_GRID_6D(gvar, d1, d2, d3, d4, d5, d6) \
    INIT_GRID_5D(gvar, d1, d2, d3, d4, d5); gvar.addDimBack(#d6, 1)

// Convenience macros for initializing parameters in stencil ctors.
// Each names the param according to the 'pvar' parameter and adds it
// to the default '_params' collection.
// The dimensions are named and sized according to the remaining parameters.
#define INIT_PARAM(pvar) \
    _params.add(&pvar); pvar.setName(#pvar); pvar.setParam(true)
#define INIT_PARAM_1D(pvar, d1, s1) \
    INIT_PARAM(pvar); pvar.addDimBack(#d1, s1)
#define INIT_PARAM_2D(pvar, d1, s1, d2, s2) \
    INIT_PARAM_1D(pvar, d1, s1); pvar.addDimBack(#d2, s2)
#define INIT_PARAM_3D(pvar, d1, s1, d2, s2, d3, s3) \
    INIT_PARAM_2D(pvar, d1, s1, d2, s2); pvar.addDimBack(#d3, s3)
#define INIT_PARAM_4D(pvar, d1, s1, d2, s2, d3, s3, d4, s4)             \
    INIT_PARAM_3D(pvar, d1, s1, d2, s2, d3, s3); pvar.addDimBack(#d4, d4)
#define INIT_PARAM_5D(pvar, d1, s1, d2, s2, d3, s3, d4, s4, d5, s5)     \
    INIT_PARAM_4D(pvar, d1, s1, d2, s2, d3, s3, d4, s4); pvar.addDimBack(#d5, d5)
#define INIT_PARAM_6D(pvar, d1, s1, d2, s2, d3, s3, d4, s4, d5, s5, d6, s6) \
    INIT_PARAM_4D(pvar, d1, s1, d2, s2, d3, s3, d4, s4, d5, s5); pvar.addDimBack(#d6, d6)

// Convenience macro for getting one offset from the 'offsets' tuple.
#define GET_OFFSET(ovar)                         \
    NumExprPtr ovar = make_shared<IntTupleExpr>(offsets.getDirInDim(#ovar))
 
 
// Use SET_VALUE_FROM_EXPR for creating a string to insert any C++ code
// that evaluates to a real_t.
// The 1st arg must be the LHS of an assignment statement.
// The 2nd arg must evaluate to a real_t (float or double) expression,
// but it must NOT include access to a grid.
// The code string is constructed as if writing to an ostream,
// so '<<' operators may be used to evaluate local variables.
// Floating-point variables will be printed w/o loss of precision.
// The code may access the following:
// - Any parameter to the 'calc_stencil_{vector,scalar}' generated functions,
//   including fields of the user-defined 'context' object.
// - A variable within the global or current namespace where it will be used.
// - A local variable in the 'value' method; in this case, the value
//   of the local var must be evaluated and inserted in the expr.
// Example code:
//   GridValue v;
//   SET_VALUE_FROM_EXPR(v =, "context.temp * " << 0.2);
//   SET_VALUE_FROM_EXPR(v +=, "context.coeff[" << r << "]");
// This example would generate the following partial expression (when r=9):
//   (context.temp * 2.00000000000000000e-01) + (context.coeff[9])
#define SET_VALUE_FROM_EXPR(lhs, rhs) do {              \
        ostringstream oss;                              \
        oss << setprecision(17) << scientific;          \
        oss << "(" << rhs << ")";                   \
        lhs  make_shared<CodeExpr>(oss.str());          \
    } while(0)

#endif
