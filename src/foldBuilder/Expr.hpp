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

///////// Stencil AST Expressions. ////////////

#ifndef EXPR_HPP
#define EXPR_HPP

#include <map>
#include <set>
#include <vector>
#include <assert.h>

#include "Dir.hpp"

using namespace std;

// Forward-declare expression visitor.
class ExprVisitor;

class Expr;
typedef shared_ptr<Expr> ExprPtr;

// The base class for all expression nodes.
class Expr {
public:
    virtual ~Expr() { }

    // For visitors.
    virtual void accept(ExprVisitor* ev) =0;

    // Try to add a compatible operand.
    // Return true if successful.
    virtual bool appendOp(ExprPtr rhs, const string& opStr) { return false; }

    // Count number of grid points in this expr.
    virtual int getNumPoints() const { return 0; }
};
typedef vector<ExprPtr> ExprPtrVec;
typedef ExprPtr GridValue;

// Various unary operators.
ExprPtr constGridValue(double rhs);
ExprPtr operator-(const ExprPtr& rhs);

// Various binary operators.
ExprPtr operator+(const ExprPtr& lhs, const ExprPtr& rhs);
ExprPtr operator+(double lhs, const ExprPtr& rhs);
ExprPtr operator+(const ExprPtr& lhs, double rhs);
void operator+=(GridValue& lhs, const ExprPtr& rhs);
void operator+=(GridValue& lhs, double rhs);

ExprPtr operator-(const ExprPtr& lhs, const ExprPtr& rhs);
ExprPtr operator-(double lhs, const ExprPtr& rhs);
ExprPtr operator-(const ExprPtr& lhs, double rhs);
void operator-=(GridValue& lhs, const ExprPtr& rhs);
void operator-=(GridValue& lhs, double rhs);

ExprPtr operator*(const ExprPtr& lhs, const ExprPtr& rhs);
ExprPtr operator*(double lhs, const ExprPtr& rhs);
ExprPtr operator*(const ExprPtr& lhs, double rhs);
void operator*=(GridValue& lhs, const ExprPtr& rhs);
void operator*=(GridValue& lhs, double rhs);

// A simple constant value.
class ConstExpr : public Expr {
protected:
    double _f;

public:
    ConstExpr(double f) : _f(f) { }
    virtual ~ConstExpr() { }

    double getVal() const {
        return _f;
    }

    virtual void accept(ExprVisitor* ev);
};

// Any expression that returns a real (not from a grid).
class CodeExpr : public Expr {
protected:
    string _code;

public:
    CodeExpr(const string& code) :
        _code(code) { }
    virtual ~CodeExpr() { }

    const string& getCode() const {
        return _code;
    }

    virtual void accept(ExprVisitor* ev);
};

// Base class for any unary operator.
class UnaryExpr : public Expr {
public:
    ExprPtr _rhs;
    string _opStr;

public:
    UnaryExpr(const string& opStr, ExprPtr rhs) :
        _rhs(rhs), _opStr(opStr) { }
    virtual ~UnaryExpr() { }

    ExprPtr getRhs() {
        return _rhs;
    }

    virtual void accept(ExprVisitor* ev);

    virtual int getNumPoints() const {
        return _rhs->getNumPoints();
    }
};

// A negation.
class NegExpr : public UnaryExpr {
public:
    NegExpr(ExprPtr rhs) :
        UnaryExpr(getOpStr(), rhs) { }
    virtual ~NegExpr() { }

    static string getOpStr() {
        return "-";
    }
};

// Base class for any binary operator.
class BinaryExpr : public UnaryExpr {
public:
    ExprPtr _lhs;

public:
    BinaryExpr(ExprPtr lhs, const string& opStr, ExprPtr rhs) :
        UnaryExpr(opStr, rhs), _lhs(lhs) { }
    virtual ~BinaryExpr() { }

    ExprPtr getLhs() {
        return _lhs;
    }

    virtual void accept(ExprVisitor* ev);

    virtual int getNumPoints() const {
        return _lhs->getNumPoints() + _rhs->getNumPoints();
    }
};

// Subtraction operator.
class SubExpr : public BinaryExpr {
public:
    SubExpr(ExprPtr lhs, ExprPtr rhs) :
        BinaryExpr(lhs, getOpStr(), rhs) { }
    virtual ~SubExpr() { }

    static string getOpStr() {
        return "-";
    }
};

// Division operator.
class DivExpr : public BinaryExpr {
public:
    DivExpr(ExprPtr lhs, ExprPtr rhs) :
        BinaryExpr(lhs, getOpStr(), rhs) { }
    virtual ~DivExpr() { }

    static string getOpStr() {
        return "/";
    }
};

// A list of exprs with a common operator that can be rearranged,
// e.g., 'a * b * c' or 'a + b + c'.
class CommutativeExpr : public Expr {
public:
    ExprPtrVec _ops;
    string _opStr;

public:
    CommutativeExpr(const string& opStr) :
        _opStr(opStr) {
    }

    CommutativeExpr(ExprPtr lhs, const string& opStr, ExprPtr rhs) :
        _opStr(opStr) {
        _ops.push_back(lhs);
        _ops.push_back(rhs);
    }

    virtual ~CommutativeExpr() { }

    ExprPtrVec& getOps() {
        return _ops;
    }

    // Try to add a compatible operand.
    // Return true if successful.
    virtual bool appendOp(ExprPtr rhs, const string& opStr) {
        if (opStr == _opStr) {
            _ops.push_back(rhs);
            return true;
        }
        return false;
    }

    virtual void accept(ExprVisitor* ev);

    virtual int getNumPoints() const {
        int numPoints = 0;
        for (auto i = _ops.begin(); i != _ops.end(); i++) {
            auto ep = *i;
            numPoints += ep->getNumPoints();
        }
        return numPoints;
    }
};

// One or more addition operators.
class AddExpr : public CommutativeExpr {
public:
    AddExpr(ExprPtr lhs, ExprPtr rhs) :
        CommutativeExpr(lhs, getOpStr(), rhs) { }
    virtual ~AddExpr() { }

    static string getOpStr() {
        return "+";
    }
};

// One or more multiplication operators.
class MultExpr : public CommutativeExpr {
public:
    MultExpr(ExprPtr lhs, ExprPtr rhs) :
        CommutativeExpr(lhs, getOpStr(), rhs) { }
    virtual ~MultExpr() { }
    
    static string getOpStr() {
        return "*";
    }
};

// One specific point in a grid.
class GridPoint : public Triple, public Expr {

protected:
    string _name;

public:
    int _t, _v;
    bool _tOk, _vOk;            // whether t and v are meaningful.

    // Construct a 4D point.
    GridPoint(const string& name, int t, int i, int j, int k) :
        Triple(i, j, k), _name(name), 
        _t(t), _v(0),
        _tOk(true), _vOk(true) {}

    // Construct a 3D point.
    GridPoint(const string& name, int i, int j, int k) :
        Triple(i, j, k), _name(name),
        _t(-1), _v(-1), _tOk(false), _vOk(false) {}

    GridPoint(const GridPoint* gp, int i, int j, int k) :
        Triple(i, j, k), _name(gp->_name),
        _t(gp->_t), _v(gp->_v),
        _tOk(gp->_tOk), _vOk(gp->_vOk) {}
    
    virtual ~GridPoint() {}

    virtual const string& name() const {
        return _name;
    }
    
    // Some comparison operators needed for containers.
    virtual bool operator==(const GridPoint& rhs) const {
        return (_name == rhs._name) &&
            (_t == rhs._t) && (_v == rhs._v) &&
            Triple::operator==(rhs);
    }
    virtual bool operator<(const GridPoint& rhs) const {
        return (_name < rhs._name) ? true : (_name > rhs._name) ? false :
            (_t < rhs._t) ? true : (_t > rhs._t) ? false :
            (_v < rhs._v) ? true : (_v > rhs._v) ? false :
            Triple::operator<(rhs);
    }

    virtual void accept(ExprVisitor* ev);

    virtual int getNumPoints() const { return 1; }

    // Determine whether this is ahead rhs in given dir.
    virtual bool isAheadOf(const GridPoint& rhs, const Dir& dir) const {
        return name() == rhs.name() && // must be same var.
            _t == rhs._t &&         // same time.
            _v == rhs._v &&         // same var.
            isInline(rhs, dir) &&   // must be in aligned in given direction.
            ((dir.isPos() && getVal(dir) > rhs.getVal(dir)) || // in front going forward.
             (dir.isNeg() && getVal(dir) < rhs.getVal(dir))); // behind going backward.
    }

    // Return a description based on this position.
    virtual string makeStr() const {
        ostringstream oss;
        oss << name() << "(";
        if (_tOk) oss << _t << ", ";
        if (_vOk) oss << _v << ", ";
        oss << _i << ", " << _j << ", " << _k << ")";
        return oss.str();
    }

};
typedef shared_ptr<GridPoint> GridPointPtr;
typedef set<GridPoint> GridPointSet;
typedef vector<GridPoint> GridPointVec;

// A base class for a collection of GridPoints.
class Grid {
protected:
    string _name;
    
public:
    Grid(const string& name) : _name(name) { }

    virtual ~Grid() { }

    const string& name() const {
        return _name;
    }
};

// A 4D collection of GridPoints (3D spatial plus time).
class TemporalGrid : public Grid {
public:
    TemporalGrid(const string& name) : Grid(name) { }

    // Create an expression to a specific point in the grid.
    virtual GridPointPtr operator()(int t, int i, int j, int k) const {
        return make_shared<GridPoint>(_name, t, i, j, k);
    }

};

// A 3D collection of GridPoints (no time dimension).
class StaticGrid : public Grid {
public:
    StaticGrid(const string& name) : Grid(name) { }

    // Create an expression to a specific point in the grid.
    virtual GridPointPtr operator()(int i, int j, int k) const {
        return make_shared<GridPoint>(_name, i, j, k);
    }
};

// Use SET_VALUE_FROM_EXPR for creating a string to insert any C++ code
// that evaluates to a REAL.
// The 1st arg must be the LHS of an assignment statement.
// The 2nd arg must evaluate to a REAL (float or double) expression,
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
