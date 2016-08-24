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

// A tuple of integers, used for dimensions and points.
typedef Tuple<int> IntTuple;

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
    virtual void accept(ExprVisitor* ev) const;

    // Check for equivalency.
    virtual bool isSame(const Expr* other) =0;

    // Return a simple string expr.
    virtual string makeStr() const;

    // Return number of nodes.
    virtual int getNumNodes() const;

    // Create a deep copy of this expression.
    virtual ExprPtr clone() const =0;
};
typedef vector<ExprPtr> ExprPtrVec;

// A function to create a scalar double.
// Usually not needed due to operator overloading.
ExprPtr constGridValue(double rhs);

// Various unary operators.
ExprPtr operator-(const ExprPtr& rhs);

// Various binary operators.
ExprPtr operator+(const ExprPtr& lhs, const ExprPtr& rhs);
ExprPtr operator+(double lhs, const ExprPtr& rhs);
ExprPtr operator+(const ExprPtr& lhs, double rhs);
void operator+=(ExprPtr& lhs, const ExprPtr& rhs);
void operator+=(ExprPtr& lhs, double rhs);

ExprPtr operator-(const ExprPtr& lhs, const ExprPtr& rhs);
ExprPtr operator-(double lhs, const ExprPtr& rhs);
ExprPtr operator-(const ExprPtr& lhs, double rhs);
void operator-=(ExprPtr& lhs, const ExprPtr& rhs);
void operator-=(ExprPtr& lhs, double rhs);

ExprPtr operator*(const ExprPtr& lhs, const ExprPtr& rhs);
ExprPtr operator*(double lhs, const ExprPtr& rhs);
ExprPtr operator*(const ExprPtr& lhs, double rhs);
void operator*=(ExprPtr& lhs, const ExprPtr& rhs);
void operator*=(ExprPtr& lhs, double rhs);

ExprPtr operator/(const ExprPtr& lhs, const ExprPtr& rhs);
ExprPtr operator/(double lhs, const ExprPtr& rhs);
ExprPtr operator/(const ExprPtr& lhs, double rhs);
void operator/=(ExprPtr& lhs, const ExprPtr& rhs);
void operator/=(ExprPtr& lhs, double rhs);

// Use the '==' operator to define a grid value.  Difficult to use '='
// operator because it cannot be declared as a standalone operator.  Also,
// the '=' operator implies replacement semantics in C++, and we want to
// represent a mathematical equality. Of course, the '==' is actually a
// *test* for equality instead of an assertion; perhaps one could think of
// this as similar to the argument to assert().  In theory, we could allow
// any expression on LHS, but we don't want to solve equations.
void operator==(GridPointPtr gpp, ExprPtr rhs);

// A simple constant value.
// This is an expression leaf-node.
class ConstExpr : public Expr {
protected:
    double _f;

public:
    ConstExpr(double f) : _f(f) { }
    ConstExpr(const ConstExpr& rhs) : _f(rhs._f) { }
    virtual ~ConstExpr() { }

    double getVal() const {
        return _f;
    }

    virtual void accept(ExprVisitor* ev);

    // Check for equivalency.
    virtual bool isSame(const Expr* other) {
        auto p = dynamic_cast<const ConstExpr*>(other);
        return p && _f == p->_f;
    }
   
    // Create a deep copy of this expression.
    virtual ExprPtr clone() const { return make_shared<ConstExpr>(*this); }
};

// Any expression that returns a real (not from a grid).
// This is an expression leaf-node.
class CodeExpr : public Expr {
protected:
    string _code;

public:
    CodeExpr(const string& code) :
        _code(code) { }
    CodeExpr(const CodeExpr& rhs) : _code(rhs._code) { }
    virtual ~CodeExpr() { }

    const string& getCode() const {
        return _code;
    }

    virtual void accept(ExprVisitor* ev);

    // Check for equivalency.
    virtual bool isSame(const Expr* other) {
        auto p = dynamic_cast<const CodeExpr*>(other);
        return p && _code == p->_code;
    }

    // Create a deep copy of this expression.
    virtual ExprPtr clone() const { return make_shared<CodeExpr>(*this); }
};

// Base class for any unary operator.
class UnaryExpr : public Expr {
protected:
    ExprPtr _rhs;
    string _opStr;

public:
    UnaryExpr(const string& opStr, ExprPtr rhs) :
        _rhs(rhs), _opStr(opStr) { }
    UnaryExpr(const UnaryExpr& rhs) :
        _rhs(rhs._rhs->clone()), _opStr(rhs._opStr) { }
    virtual ~UnaryExpr() { }

    ExprPtr& getRhs() { return _rhs; }
    const ExprPtr& getRhs() const { return _rhs; }
    const string& getOpStr() const { return _opStr; }
    
    virtual void accept(ExprVisitor* ev);

    // Check for equivalency.
    virtual bool isSame(const Expr* other) {
        auto p = dynamic_cast<const UnaryExpr*>(other);
        return p && _opStr == p->_opStr &&
            _rhs->isSame(p->_rhs.get());
    }

    // Create a deep copy of this expression.
    virtual ExprPtr clone() const { return make_shared<UnaryExpr>(*this); }
};

// A negation.
class NegExpr : public UnaryExpr {
public:
    NegExpr(ExprPtr rhs) :
        UnaryExpr(opStr(), rhs) { }
    NegExpr(const NegExpr& rhs) :
        UnaryExpr(rhs) { }
    virtual ~NegExpr() { }

    static string opStr() {
        return "-";
    }

    // Create a deep copy of this expression.
    virtual ExprPtr clone() const { return make_shared<NegExpr>(*this); }
};

// Base class for any binary operator.
class BinaryExpr : public UnaryExpr {
protected:
    ExprPtr _lhs;

public:
    BinaryExpr(ExprPtr lhs, const string& opStr, ExprPtr rhs) :
        UnaryExpr(opStr, rhs), _lhs(lhs) { }
    BinaryExpr(const BinaryExpr& rhs) :
        UnaryExpr(rhs._opStr, rhs._rhs->clone()), _lhs(rhs._lhs->clone()) { }  
    virtual ~BinaryExpr() { }

    ExprPtr& getLhs() { return _lhs; }
    const ExprPtr& getLhs() const { return _lhs; }

    virtual void accept(ExprVisitor* ev);

    // Check for equivalency.
    virtual bool isSame(const Expr* other) {
        auto p = dynamic_cast<const BinaryExpr*>(other);
        return p && _opStr == p->_opStr &&
            _lhs->isSame(p->_lhs.get()) &&
            _rhs->isSame(p->_rhs.get());
    }

    // Create a deep copy of this expression.
    virtual ExprPtr clone() const { return make_shared<BinaryExpr>(*this); }
};

// Subtraction operator.
class SubExpr : public BinaryExpr {
public:
    SubExpr(ExprPtr lhs, ExprPtr rhs) :
        BinaryExpr(lhs, opStr(), rhs) { }
    SubExpr(const SubExpr& rhs) :
        BinaryExpr(rhs) { }
    virtual ~SubExpr() { }

    static string opStr() {
        return "-";
    }

    // Create a deep copy of this expression.
    virtual ExprPtr clone() const { return make_shared<SubExpr>(*this); }
};

// Division operator.
class DivExpr : public BinaryExpr {
public:
    DivExpr(ExprPtr lhs, ExprPtr rhs) :
        BinaryExpr(lhs, opStr(), rhs) { }
    DivExpr(const DivExpr& rhs) :
        BinaryExpr(rhs) { }
    virtual ~DivExpr() { }

    static string opStr() {
        return "/";
    }

    // Create a deep copy of this expression.
    virtual ExprPtr clone() const { return make_shared<DivExpr>(*this); }
};

// A list of exprs with a common operator that can be rearranged,
// e.g., 'a * b * c' or 'a + b + c'.
class CommutativeExpr : public Expr {
protected:
    ExprPtrVec _ops;
    string _opStr;

public:
    CommutativeExpr(const string& opStr) :
        _opStr(opStr) {
    }
    CommutativeExpr(ExprPtr lhs, const string& opStr, ExprPtr rhs) :
        _opStr(opStr) {
        _ops.push_back(lhs->clone());
        _ops.push_back(rhs->clone());
    }
    CommutativeExpr(const CommutativeExpr& rhs) :
        _opStr(rhs._opStr) {
        for(auto op : rhs._ops)
            _ops.push_back(op->clone());
    }
    virtual ~CommutativeExpr() { }

    // Accessors.
    ExprPtrVec& getOps() { return _ops; }
    const ExprPtrVec& getOps() const { return _ops; }
    const string& getOpStr() const { return _opStr; }

    // Clone and add an operand.
    virtual void appendOp(ExprPtr op) {
        _ops.push_back(op->clone());
    }

    // If op is another CommutativeExpr with the
    // same operator, add its operands to this.
    // Otherwise, just clone and add the whole op.
    virtual void mergeExpr(ExprPtr op) {
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
    virtual bool isSame(const Expr* other);

    // Create a deep copy of this expression.
    virtual ExprPtr clone() const { return make_shared<CommutativeExpr>(*this); }
};

// One or more addition operators.
class AddExpr : public CommutativeExpr {
public:
    AddExpr()  :
        CommutativeExpr(opStr()) { }
    AddExpr(ExprPtr lhs, ExprPtr rhs) :
        CommutativeExpr(lhs, opStr(), rhs) { }
    AddExpr(const AddExpr& rhs) : CommutativeExpr(rhs) { }
    virtual ~AddExpr() { }

    static string opStr() {
        return "+";
    }
  
    // Create a deep copy of this expression.
    virtual ExprPtr clone() const { return make_shared<AddExpr>(*this); } 
};

// One or more multiplication operators.
class MultExpr : public CommutativeExpr {
public:
    MultExpr()  :
        CommutativeExpr(opStr()) { }
    MultExpr(ExprPtr lhs, ExprPtr rhs) :
        CommutativeExpr(lhs, opStr(), rhs) { }
    MultExpr(const MultExpr& rhs) : CommutativeExpr(rhs) { }
    virtual ~MultExpr() { }
    
    static string opStr() {
        return "*";
    }

    // Create a deep copy of this expression.
    virtual ExprPtr clone() const { return make_shared<MultExpr>(*this); } 
};

// One specific point in a grid.
// This is an expression leaf-node.
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

    // Copy ctor.
    // Note that _grid is a shallow copy!
    GridPoint(const GridPoint& rhs) :
        IntTuple(rhs), _grid(rhs._grid) { }
    GridPoint(const GridPointPtr& rhs) :
        GridPoint(*rhs) { }

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
    virtual bool isSame(const Expr* other) {
        auto p = dynamic_cast<const GridPoint*>(other);
        return p && *this == *p;
    }
    
    // Determine whether this is 'ahead of' rhs in given direction.
    virtual bool isAheadOfInDir(const GridPoint& rhs, const IntTuple& dir) const;

    // Return a description based on this position.
    virtual string makeStr() const;

    // Create a deep copy of this expression,
    // except pointed-to grid is not copied.
    virtual ExprPtr clone() const { return make_shared<GridPoint>(*this); }   
};

// Equality operator.
// (Not inherited from BinaryExpr because LHS is special.)
class EqualsExpr : public UnaryExpr {
protected:
    GridPointPtr _lhs;

public:
    EqualsExpr(GridPointPtr lhs, ExprPtr rhs) :
        UnaryExpr(opStr(), rhs), _lhs(lhs) { }
    EqualsExpr(const EqualsExpr& rhs) :
        UnaryExpr(rhs) {
        _lhs = make_shared<GridPoint>(rhs._lhs);
    }
    virtual ~EqualsExpr() { }

    GridPointPtr& getLhs() { return _lhs; }
    const GridPointPtr& getLhs() const { return _lhs; }

    static string opStr() {
        return "=="; 
    }

    virtual void accept(ExprVisitor* ev);

    // Check for equivalency.
    virtual bool isSame(const Expr* other);

    // Create a deep copy of this expression.
    virtual ExprPtr clone() const { return make_shared<EqualsExpr>(*this); }
};

///////// Grids ////////////

typedef set<GridPoint> GridPointSet;
typedef set<GridPointPtr> GridPointPtrSet;
typedef vector<GridPoint> GridPointVec;
typedef map<GridPoint, ExprPtr> Point2Exprs;

// An index into a grid.
// TODO: this typedef is just a placeholder; replace it
// with a class to limit use--we don't want
// to allow modification or conditional testing, etc.
typedef const int GridIndex;

// A class for a collection of GridPoints.
// Dims in the IntTuple describe the grid or param.
// For grids, values in the IntTuple are ignored.
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

    // Param-type accessors.
    bool isParam() const { return _isParam; }
    void setParam(bool isParam) { _isParam = isParam; }
    
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
    virtual operator ExprPtr() { // implicit conversion.
        return makePoint(0);
    }
    virtual operator GridPointPtr() { // implicit conversion.
        return makePoint(0);
    }
    virtual GridPointPtr operator()() {
        return makePoint(0);
    }

    // Convenience functions for one dimension (array).
    virtual GridPointPtr operator[](int i1) {
        return makePoint(1, i1);
    }
    virtual GridPointPtr operator()(int i1) {
        return makePoint(1, i1);
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
};

// A list of grids.
class Grids : public vector<Grid*> {
    
public:

    // Visit all expressions in all grids.
    virtual void acceptToAll(ExprVisitor* ev);

    // Visit first expression in each grid.
    virtual void acceptToFirst(ExprVisitor* ev);

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
};

// Aliases for parameters.
// Even though these are just typedefs for now, don't interchange them.
// TODO: enforce the difference between grids and parameters.
typedef Grid Param;
typedef Grids Params;

// A named equation and the grids updated by it.
struct Equation {
    string name;
    Grids grids;
};

// Equations:
class Equations : public vector<Equation> {
protected:
    set<Grid*> _eqGrids;        // all grids with equations.
    
public:
    Equations() {}
    virtual ~Equations() {}

    // Separate a set of grids into equations based
    // on the target string.
    // Target string is a comma-separated list of key-value pairs, e.g.,
    // "equation1=foo,equation2=bar".
    // In this example, all grids with names containing 'foo' go in equation1,
    // all grids with names containing 'bar' go in equation2, and
    // each remaining grid goes in a equation named after the grid.
    // Only grids with equations are put in equations.
    void findEquations(Grids& allGrids, const string& targets);

    const set<Grid*>& getEqGrids() const { return _eqGrids; }
    
    void printInfo(ostream& os) const {
        os << "Identified stencil equations:" << endl;
        for (auto& eq : *this) {
            for (auto gp : eq.grids) {
                os << "  Equation '" << eq.name << "' updates grid '" <<
                    gp->getName() << "'." << endl;
            }
        }
    }

};

// A 'GridValue' is simply a pointer to an expression.
// So, operations on a GridValue will create an AST, i.e.,
// it will not evaluate the value.
typedef ExprPtr GridValue;

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
    GridIndex ovar = offsets.getVal(#ovar)
 
 
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
