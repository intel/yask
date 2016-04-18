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
#include <assert.h>

#include "Tuple.hpp"

using namespace std;

// Forward-declare expression visitor.
class ExprVisitor;

// Forward-declaration of an expression.
class Expr;
typedef shared_ptr<Expr> ExprPtr;

// Forward-declaration of a grid and a grid point.
class Grid;
class GridPoint;
typedef shared_ptr<GridPoint> GridPointPtr;

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

// A 'GridValue' is simply a pointer to an expression.
// This means that calling an update() function will
// not actually evaluate the expression in the function.
// Rather, it will create an AST.
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

ExprPtr operator/(const ExprPtr& lhs, const ExprPtr& rhs);
ExprPtr operator/(double lhs, const ExprPtr& rhs);
ExprPtr operator/(const ExprPtr& lhs, double rhs);
void operator/=(GridValue& lhs, const ExprPtr& rhs);
void operator/=(GridValue& lhs, double rhs);

// Use the '==' operator to define a grid value.  Difficult to use '='
// operator because it cannot be declared as a standalone operator.  Also,
// the '=' operator implies replacement semantics in C++, and we want to
// represent a mathematical equality. Of course, the '==' is actually a
// *test* for equality instead of an assertion; perhaps one could think of
// this as similar to the argument to assert().  In theory, we could allow
// any expression on LHS, but we don't want to solve equations.
void operator==(GridPointPtr gpp, ExprPtr rhs);

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
protected:
    ExprPtr _rhs;
    string _opStr;

public:
    UnaryExpr(const string& opStr, ExprPtr rhs) :
        _rhs(rhs), _opStr(opStr) { }
    virtual ~UnaryExpr() { }

    ExprPtr getRhs() { return _rhs; }
    const ExprPtr getRhs() const { return _rhs; }
    const string& getOpStr() const { return _opStr; }
    
    virtual void accept(ExprVisitor* ev);

    virtual int getNumPoints() const {
        return _rhs->getNumPoints();
    }
};

// A negation.
class NegExpr : public UnaryExpr {
public:
    NegExpr(ExprPtr rhs) :
        UnaryExpr(opStr(), rhs) { }
    virtual ~NegExpr() { }

    static string opStr() {
        return "-";
    }
};

// Base class for any binary operator.
class BinaryExpr : public UnaryExpr {
protected:
    ExprPtr _lhs;

public:
    BinaryExpr(ExprPtr lhs, const string& opStr, ExprPtr rhs) :
        UnaryExpr(opStr, rhs), _lhs(lhs) { }
    virtual ~BinaryExpr() { }

    ExprPtr getLhs() { return _lhs; }
    const ExprPtr getLhs() const { return _lhs; }

    virtual void accept(ExprVisitor* ev);

    virtual int getNumPoints() const {
        return _lhs->getNumPoints() + _rhs->getNumPoints();
    }
};

// Subtraction operator.
class SubExpr : public BinaryExpr {
public:
    SubExpr(ExprPtr lhs, ExprPtr rhs) :
        BinaryExpr(lhs, opStr(), rhs) { }
    virtual ~SubExpr() { }

    static string opStr() {
        return "-";
    }
};

// Division operator.
class DivExpr : public BinaryExpr {
public:
    DivExpr(ExprPtr lhs, ExprPtr rhs) :
        BinaryExpr(lhs, opStr(), rhs) { }
    virtual ~DivExpr() { }

    static string opStr() {
        return "/";
    }
};

// Equality operator.
// (Not inherited from BinaryExpr because LHS is special.)
class EqualsExpr : public UnaryExpr {
protected:
    GridPointPtr _lhs;

public:
    EqualsExpr(GridPointPtr lhs, ExprPtr rhs) :
        UnaryExpr(opStr(), rhs), _lhs(lhs) { }
    virtual ~EqualsExpr() { }

    GridPointPtr getLhs() { return _lhs; }
    const GridPointPtr getLhs() const { return _lhs; }

    static string opStr() {
        return "=="; 
    }

    virtual void accept(ExprVisitor* ev);
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

    ExprPtrVec& getOps() { return _ops; }
    const ExprPtrVec& getOps() const { return _ops; }
    const string& getOpStr() const { return _opStr; }

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
        CommutativeExpr(lhs, opStr(), rhs) { }
    virtual ~AddExpr() { }

    static string opStr() {
        return "+";
    }
};

// One or more multiplication operators.
class MultExpr : public CommutativeExpr {
public:
    MultExpr(ExprPtr lhs, ExprPtr rhs) :
        CommutativeExpr(lhs, opStr(), rhs) { }
    virtual ~MultExpr() { }
    
    static string opStr() {
        return "*";
    }
};

///////// Grids ////////////

typedef Tuple<int> IntTuple;

// One specific point in a grid, which can appear in an expression.
class GridPoint : public IntTuple, public Expr {

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

    // Construct from another point, but change location.
    GridPoint(GridPoint* gp, const IntTuple& offsets) :
            IntTuple(offsets), _grid(gp->_grid)
    {
        assert(areDimsSame(offsets));
    }

    // Dtor.
    virtual ~GridPoint() {}

    // Get parent grid.
    const Grid* getGrid() const { return _grid; }
    Grid* getGrid() { return _grid; }

    // Name of parent grid.
    virtual const string& getName() const;

    // Some comparisons.
    bool operator==(const GridPoint& rhs) const;
    bool operator<(const GridPoint& rhs) const;

    // Take ev to each value.
    virtual void accept(ExprVisitor* ev);

    // Just one point here.
    // Used for couting points in an AST.
    virtual int getNumPoints() const { return 1; }

    // Determine whether this is 'ahead of' rhs in given direction.
    virtual bool isAheadOfInDir(const GridPoint& rhs, const IntTuple& dir) const;

    // Return a description based on this position.
    virtual string makeStr() const;
};
typedef set<GridPoint> GridPointSet;
typedef set<GridPointPtr> GridPointPtrSet;
typedef vector<GridPoint> GridPointVec;
typedef map<GridPoint, ExprPtr> Point2Exprs;

// An index into a grid.
// TODO: this typedef is just a placeholder; replace it
// with a class to limit use--we don't want
// to allow modification or conditional testing, etc.
typedef int GridIndex;

// A base class for a collection of GridPoints.
// Dims in the IntTuple describe the grid, but values
// in the IntTuple are ignored.
class Grid : public IntTuple {
protected:
    string _name;               // name of the grid.

    // specific points that have been created in this grid.
    GridPointPtrSet _points;

    // equation(s) describing how values in this grid are computed.
    Point2Exprs _exprs;

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

    // Point accessors.
    const GridPointPtrSet& getPoints() const { return _points; }
    GridPointPtrSet& getPoints() { return _points; }

    // Expression accessors.
    // gpp points to LHS grid point.
    // ep points to EqualsExpr.
    virtual void addExpr(GridPointPtr gpp, ExprPtr ep) {
        _exprs[*gpp] = ep;
    }
    virtual const Point2Exprs& getExprs() const {
        return _exprs;
    }
    virtual Point2Exprs& getExprs() {
        return _exprs;
    }
    
    // Visit all expressions, if any are defined.
    virtual void acceptToAll(ExprVisitor* ev) {
        for (auto i : _exprs) {
            auto ep = i.second;
            ep->accept(ev);
        }
    }
    
    // Visit only first expression, if it is defined.
    virtual void acceptToFirst(ExprVisitor* ev) {
        for (auto i : _exprs) {
            auto ep = i.second;
            ep->accept(ev);
            break;
        }
    }

    // Create an expression to a specific point in the grid.
    // Note that this doesn't actually 'read' a value; it's just an expression.
    virtual GridPointPtr operator()(const vector<int>& points) {

        // check for correct number of indices.
        size_t n = points.size();
        if (n != size()) {
            cerr << "Error: attempt to access " << size() <<
                "-D grid '" << _name << "' with " << n << " indices.\n";
            exit(1);
        }

        IntTuple pt = *this;       // to copy names.
        pt.setVals(points);
        GridPointPtr gpp = make_shared<GridPoint>(this, pt);
        return addPoint(gpp);
    }
    
    // Create an expression to a specific point in a 1D grid.
    virtual GridPointPtr operator()(int i1) {
        vector<int> vals = { i1 };
        return operator()(vals);
    }

    // Create an expression to a specific point in a 2D grid.
    virtual GridPointPtr operator()(int i1, int i2) {
        vector<int> vals = { i1, i2 };
        return operator()(vals);
    }

    // Create an expression to a specific point in a 3D grid.
    virtual GridPointPtr operator()(int i1, int i2, int i3) {
        vector<int> vals = { i1, i2, i3 };
        return Grid::operator()(vals);
    }

    // Create an expression to a specific point in a 4D grid.
    virtual GridPointPtr operator()(int i1, int i2, int i3, int i4) {
        vector<int> vals = { i1, i2, i3, i4 };
        return Grid::operator()(vals);
    }

    // Create an expression to a specific point in a 5D grid.
    virtual GridPointPtr operator()(int i1, int i2, int i3, int i4, int i5) {
        vector<int> vals = { i1, i2, i3, i4, i5 };
        return Grid::operator()(vals);
    }
};

// A list of grids.
class Grids : public vector<Grid*> {
public:

    // Visit all expressions in all grids.
    virtual void acceptToAll(ExprVisitor* ev);
};

// Convenience macros for initializing grids in stencil ctors.
// Each names the grid according to the variable name and adds it
// to the default '_grids' collection.
#define INIT_GRID(gvar) _grids.push_back(&gvar); gvar.setName(#gvar)
#define INIT_GRID_1D(gvar, d1) INIT_GRID(gvar); gvar.addDim(#d1, 1)
#define INIT_GRID_2D(gvar, d1, d2) INIT_GRID_1D(gvar, d1); gvar.addDim(#d2, 1)
#define INIT_GRID_3D(gvar, d1, d2, d3) INIT_GRID_2D(gvar, d1, d2); gvar.addDim(#d3, 1)
#define INIT_GRID_4D(gvar, d1, d2, d3, d4) INIT_GRID_3D(gvar, d1, d2, d3); gvar.addDim(#d4, 1)
#define INIT_GRID_5D(gvar, d1, d2, d3, d4, d5) INIT_GRID_4D(gvar, d1, d2, d3, d4); gvar.addDim(#d5, 1)

// Convenience macro for getting one offset from the 'offsets' tuple.
#define GET_OFFSET(ovar)                         \
    GridIndex ovar = offsets.getVal(#ovar)
 
 
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
