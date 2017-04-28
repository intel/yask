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

//////////// Generic expression-visitor class /////////////

#ifndef VISITOR_HPP
#define VISITOR_HPP

#include "Expr.hpp"

namespace yask {

    // Base class for an Expr-tree visitor.
    class ExprVisitor {
    public:
        virtual ~ExprVisitor() { }

        // By default, leaf-node visitors do nothing.
        virtual void visit(ConstExpr* ce) { }
        virtual void visit(CodeExpr* ce) { }
        virtual void visit(IndexExpr* ie) { }
        virtual void visit(IntTupleExpr* ite) { }
        virtual void visit(GridPoint* gp) { }

        // By default, a unary visitor just visits its operand.
        virtual void visit(UnaryNumExpr* ue) {
            ue->getRhs()->accept(this);
        }
        virtual void visit(UnaryNum2BoolExpr* ue) {
            ue->getRhs()->accept(this);
        }
        virtual void visit(UnaryBoolExpr* ue) {
            ue->getRhs()->accept(this);
        }

        // By default, a binary visitor just visits its operands.
        virtual void visit(BinaryNumExpr* be) {
            be->getLhs()->accept(this);
            be->getRhs()->accept(this);
        }
        virtual void visit(BinaryNum2BoolExpr* be) {
            be->getLhs()->accept(this);
            be->getRhs()->accept(this);
        }
        virtual void visit(BinaryBoolExpr* be) {
            be->getLhs()->accept(this);
            be->getRhs()->accept(this);
        }

        // By default, a conditional visitor just visits its operands.
        virtual void visit(IfExpr* be) {
            be->getExpr()->accept(this);
            if (be->getCond())
                be->getCond()->accept(this);
        }

        // By default, a commutative visitor just visits its operands.
        virtual void visit(CommutativeExpr* ce) {
            auto& ops = ce->getOps();
            for (auto ep : ops) {
                ep->accept(this);
            }
        }

        // By default, an equality visitor just visits its operands.
        virtual void visit(EqualsExpr* ee) {
            ee->getLhs()->accept(this);
            ee->getRhs()->accept(this);
        }
    };

} // namespace yask.

#endif
