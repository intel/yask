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

#include "Visitor.hpp"

// Unary.
ExprPtr operator-(const ExprPtr& rhs) {
    return make_shared<NegExpr>(rhs);
}

// Binary.
ExprPtr operator+(const ExprPtr& lhs, const ExprPtr& rhs) {

    // If one side is nothing, return other side.
    if (lhs == NULL)
        return rhs;
    else if (rhs == NULL)
        return lhs;

    // If adding to another add result, just append an operand.
    if (lhs->appendOp(rhs, AddExpr::getOpStr()))
        return lhs;
    if (rhs->appendOp(lhs, AddExpr::getOpStr()))
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

ExprPtr operator-(const ExprPtr& lhs, const ExprPtr& rhs) {

    // Generate A + -B instead of A - B to allow easy reordering.
    ExprPtr nrhs = make_shared<NegExpr>(rhs);
    return lhs + nrhs;
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

ExprPtr operator*(const ExprPtr& lhs, const ExprPtr& rhs) {

    // If one side is nothing, return other side.
    if (lhs == NULL)
        return rhs;
    else if (rhs == NULL)
        return lhs;

    // If multiplying by another mul result, just append an operand.
    if (lhs->appendOp(rhs, MultExpr::getOpStr()))
        return lhs;
    if (rhs->appendOp(lhs, MultExpr::getOpStr()))
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

// Visitor acceptors.
void ConstExpr::accept(ExprVisitor* ev) {
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

