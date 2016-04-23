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

#include "Print.hpp"
#include "ExprUtils.hpp"

// Define static value in Tuple.
bool IntTuple::_defaultFirstInner = true;

// Unary.
ExprPtr constGridValue(double rhs) {
    return make_shared<ConstExpr>(rhs);
}
ExprPtr operator-(const ExprPtr& rhs) {
    return make_shared<NegExpr>(rhs);
}

// Commutative.
ExprPtr operator+(const ExprPtr& lhs, const ExprPtr& rhs) {

    // If one side is nothing, return other side;
    // This allows us to add to an uninitialized GridValue
    // and do the right thing.
    if (lhs == NULL)
        return rhs;
    else if (rhs == NULL)
        return lhs;

    // If adding to another add result, just append an operand.
    if (lhs->appendOp(rhs, AddExpr::opStr()))
        return lhs;
    if (rhs->appendOp(lhs, AddExpr::opStr()))
        return rhs;

    // Otherwise, make a new expression.
    else
        return make_shared<AddExpr>(lhs, rhs);
}
ExprPtr operator+(double lhs, const ExprPtr& rhs) {
    ExprPtr p = make_shared<ConstExpr>(lhs);
    return p + rhs;
}
ExprPtr operator+(const ExprPtr& lhs, double rhs) {
    ExprPtr p = make_shared<ConstExpr>(rhs);
    return lhs + p;
}

void operator+=(GridValue& lhs, const ExprPtr& rhs) {
    lhs = lhs + rhs;
}
void operator+=(GridValue& lhs, double rhs) {
    lhs = lhs + rhs;
}

ExprPtr operator*(const ExprPtr& lhs, const ExprPtr& rhs) {

    // If one side is nothing, return other side.
    if (lhs == NULL)
        return rhs;
    else if (rhs == NULL)
        return lhs;

    // If multiplying by another mul result, just append an operand.
    if (lhs->appendOp(rhs, MultExpr::opStr()))
        return lhs;
    if (rhs->appendOp(lhs, MultExpr::opStr()))
        return rhs;

    // Otherwise, make a new expression.
    else
        return make_shared<MultExpr>(lhs, rhs);
}
ExprPtr operator*(double lhs, const ExprPtr& rhs) {
    ExprPtr p = make_shared<ConstExpr>(lhs);
    return p * rhs;
}
ExprPtr operator*(const ExprPtr& lhs, double rhs) {
    ExprPtr p = make_shared<ConstExpr>(rhs);
    return lhs * p;
}

void operator*=(GridValue& lhs, const ExprPtr& rhs) {
    lhs = lhs * rhs;
}
void operator*=(GridValue& lhs, double rhs) {
    lhs = lhs * rhs;
}

// Binary.
ExprPtr operator-(const ExprPtr& lhs, const ExprPtr& rhs) {

#ifdef USE_ADD_NEG
    // Generate A + -B instead of A - B to allow easy reordering.
    ExprPtr nrhs = make_shared<NegExpr>(rhs);
    return lhs + nrhs;
#else
    return make_shared<SubExpr>(lhs, rhs);
#endif    
}
ExprPtr operator-(double lhs, const ExprPtr& rhs) {
    ExprPtr p = make_shared<ConstExpr>(lhs);
    return p - rhs;
}
ExprPtr operator-(const ExprPtr& lhs, double rhs) {
    ExprPtr p = make_shared<ConstExpr>(rhs);
    return lhs - p;
}

void operator-=(GridValue& lhs, const ExprPtr& rhs) {
    lhs = lhs - rhs;
}
void operator-=(GridValue& lhs, double rhs) {
    lhs = lhs - rhs;
}

ExprPtr operator/(const ExprPtr& lhs, const ExprPtr& rhs) {

    return make_shared<DivExpr>(lhs, rhs);
}
ExprPtr operator/(double lhs, const ExprPtr& rhs) {
    ExprPtr p = make_shared<ConstExpr>(lhs);
    return p / rhs;
}
ExprPtr operator/(const ExprPtr& lhs, double rhs) {
    ExprPtr p = make_shared<ConstExpr>(rhs);
    return lhs / p;
}

void operator/=(GridValue& lhs, const ExprPtr& rhs) {
    lhs = lhs / rhs;
}
void operator/=(GridValue& lhs, double rhs) {
    lhs = lhs / rhs;
}

// Define the value of a grid point.
// Note that the semantics are different than the 'normal'
// '==' operator, which tests for equality.
void operator==(GridPointPtr gpp, ExprPtr rhs) {
    assert(gpp != NULL);
    Grid* gp = gpp->getGrid();
    assert(gp);

    // Make expression node.
    ExprPtr p = make_shared<EqualsExpr>(gpp, rhs);
    gp->addExpr(gpp, p);
}

// Visitor acceptors.
void ConstExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void CodeExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void UnaryExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void BinaryExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void CommutativeExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void GridPoint::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void EqualsExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}

// EqualsExpr methods.
bool EqualsExpr::isSame(const Expr* other) {
        auto p = dynamic_cast<const EqualsExpr*>(other);
        return p && _opStr == p->_opStr &&
            _lhs->isSame(p->_lhs.get()) &&
            _rhs->isSame(p->_rhs.get());
    }

// GridPoint methods.
const string& GridPoint::getName() const {
    return _grid->getName();
}
bool GridPoint::operator==(const GridPoint& rhs) const {
    return (_grid->getName() == rhs._grid->getName()) &&
        IntTuple::operator==(rhs);
}
bool GridPoint::operator<(const GridPoint& rhs) const {
    return (_grid->getName() < rhs._grid->getName()) ? true :
        (_grid->getName() > rhs._grid->getName()) ? false :
        IntTuple::operator<(rhs);
}
bool GridPoint::isAheadOfInDir(const GridPoint& rhs, const IntTuple& dir) const {
    return _grid->getName() == rhs._grid->getName() && // must be same var.
        IntTuple::isAheadOfInDir(rhs, dir);
}
string GridPoint::makeStr() const {
    return _grid->getName() + "(" +
        makeDimValOffsetStr() + ")";
}

// Visit all expressions in all grids.
void Grids::acceptToAll(ExprVisitor* ev) {
    for (auto gp : *this) {
        gp->acceptToAll(ev);
    }
}

// Make a readable string from an expression.
string Expr::makeStr() const {
    ostringstream oss;
    
    // Use a print visitor to make a string.
    PrintHelper ph(NULL, "temp", "", "", "");
    PrintVisitorTopDown pv(oss, ph);
    accept(&pv);

    return oss.str() + pv.getExprStr();
}

// Return number of nodes.
int Expr::getNumNodes() const {

    // Use a counter visitor.
    CounterVisitor cv;
    accept(&cv);

    return cv.getNumNodes();
}

// Const version of accept.
void Expr::accept(ExprVisitor* ev) const {
    const_cast<Expr*>(this)->accept(ev);
}
