/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

//////////// Implement methods for printing classes /////////////

#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {

    ////////////// Print visitors ///////////////

    // Declare a new temp var.
    // Set _exprStr to it.
    // Print LHS of assignment to it.
    // If 'ex' is non-null, it is used as key to save name of temp var and
    // to write a comment.
    // If 'comment' is set, use it for the comment.
    // Return stream to continue w/RHS.
    ostream& PrintVisitorBase::makeNextTempVar(Expr* ex, string comment) {
        _exprStr = _ph.makeVarName();
        if (ex) {
            _tempVars[ex] = _exprStr;
            if (comment.length() == 0)
                _os << endl << " // " << _exprStr << " = " << ex->makeStr() << "." << endl;
        }
        if (comment.length())
            _os << endl << " // " << _exprStr << " = " << comment << "." << endl;
        _os << _ph.getLinePrefix() << _ph.getVarType() << " " << _exprStr << " = ";
        return _os;
    }

    /////// Top-down

    // A grid read.
    // Uses the PrintHelper to format.
    void PrintVisitorTopDown::visit(GridPoint* gp) {
        _exprStr += _ph.readFromPoint(_os, *gp);
        _numCommon += _ph.getNumCommon(gp);
    }

    // An index expression.
    void PrintVisitorTopDown::visit(IndexExpr* ie) {
        _exprStr += ie->format(_varMap);
        _numCommon += _ph.getNumCommon(ie);
    }

    // A constant.
    // Uses the PrintHelper to format.
    void PrintVisitorTopDown::visit(ConstExpr* ce) {
        _exprStr += _ph.addConstExpr(_os, ce->getNumVal());
        _numCommon += _ph.getNumCommon(ce);
    }

    // Some hand-written code.
    // Uses the PrintHelper to format.
    void PrintVisitorTopDown::visit(CodeExpr* ce) {
        _exprStr += _ph.addCodeExpr(_os, ce->getCode());
        _numCommon += _ph.getNumCommon(ce);
    }

    // Generic unary operators.
    // Assumes unary operators have highest precedence, so no ()'s added.
    void PrintVisitorTopDown::visit(UnaryNumExpr* ue) {
        _exprStr += ue->getOpStr();
        ue->getRhs()->accept(this);
        _numCommon += _ph.getNumCommon(ue);
    }
    void PrintVisitorTopDown::visit(UnaryBoolExpr* ue) {
        _exprStr += ue->getOpStr();
        ue->getRhs()->accept(this);
        _numCommon += _ph.getNumCommon(ue);
    }
    void PrintVisitorTopDown::visit(UnaryNum2BoolExpr* ue) {
        _exprStr += ue->getOpStr();
        ue->getRhs()->accept(this);
        _numCommon += _ph.getNumCommon(ue);
    }

    // Generic binary operators.
    void PrintVisitorTopDown::visit(BinaryNumExpr* be) {
        _exprStr += "(";
        be->getLhs()->accept(this); // adds LHS to _exprStr.
        _exprStr += " " + be->getOpStr() + " ";
        be->getRhs()->accept(this); // adds RHS to _exprStr.
        _exprStr += ")";
        _numCommon += _ph.getNumCommon(be);
    }
    void PrintVisitorTopDown::visit(BinaryBoolExpr* be) {
        _exprStr += "(";
        be->getLhs()->accept(this); // adds LHS to _exprStr.
        _exprStr += " " + be->getOpStr() + " ";
        be->getRhs()->accept(this); // adds RHS to _exprStr.
        _exprStr += ")";
        _numCommon += _ph.getNumCommon(be);
    }
    void PrintVisitorTopDown::visit(BinaryNum2BoolExpr* be) {
        _exprStr += "(";
        be->getLhs()->accept(this); // adds LHS to _exprStr.
        _exprStr += " " + be->getOpStr() + " ";
        be->getRhs()->accept(this); // adds RHS to _exprStr.
        _exprStr += ")";
        _numCommon += _ph.getNumCommon(be);
    }

    // A commutative operator.
    void PrintVisitorTopDown::visit(CommutativeExpr* ce) {
        _exprStr += "(";
        auto& ops = ce->getOps();
        int opNum = 0;
        for (auto ep : ops) {
            if (opNum > 0)
                _exprStr += " " + ce->getOpStr() + " ";
            ep->accept(this);   // adds operand to _exprStr;
            opNum++;
        }
        _exprStr += ")";
        _numCommon += _ph.getNumCommon(ce);
    }

    // A conditional operator.
    void PrintVisitorTopDown::visit(IfExpr* ie) {

        // Null ptr => no condition.
        if (ie->getCond()) {
            ie->getCond()->accept(this); // sets _exprStr;
            string cond = getExprStrAndClear();

            // pseudo-code format.
            _os << _ph.getLinePrefix() << "IF (" << cond << ") THEN" << endl;
        }
    
        // Get assignment expr and clear expr.
        ie->getExpr()->accept(this); // writes to _exprStr;
        string vexpr = getExprStrAndClear();

        // note: _exprStr is now empty.
        // note: no need to update num-common.
    }

    // An equals operator.
    void PrintVisitorTopDown::visit(EqualsExpr* ee) {

        // Get RHS and clear expr.
        ee->getRhs()->accept(this); // writes to _exprStr;
        string rhs = getExprStrAndClear();

        // Write statement with embedded rhs.
        GridPointPtr gpp = ee->getLhs();
        _os << _ph.getLinePrefix() << _ph.writeToPoint(_os, *gpp, rhs) << _ph.getLineSuffix();

        // note: _exprStr is now empty.
        // note: no need to update num-common.
    }

    /////// Bottom-up

    // Try some simple printing techniques.
    // Return true if printing is done.
    // Return false if more complex method should be used.
    // TODO: the current code causes all nodes in an expr above a certain
    // point to avoid top-down printing because it looks at the original expr,
    // not the new one with temp vars. Fix this.
    bool PrintVisitorBottomUp::trySimplePrint(Expr* ex, bool force) {

        bool exprDone = false;

        // How many nodes in ex?
        int exprSize = ex->getNumNodes();
        bool tooBig = exprSize > _settings._maxExprSize;
        bool tooSmall = exprSize < _settings._minExprSize;

        // Determine whether this expr has already been evaluated
        // and a variable holds its result.
        auto p = _tempVars.find(ex);
        if (p != _tempVars.end()) {

            // if so, just use the existing var.
            _exprStr = p->second;
            exprDone = true;
        }
        
        // Consider top down if forcing or expr <= maxExprSize.
        else if (force || !tooBig) {

            // use a top-down printer to render the expr.
            PrintVisitorTopDown* topDown = newPrintVisitorTopDown();
            ex->accept(topDown);

            // were there any common subexprs found?
            int numCommon = topDown->getNumCommon();

            // if no common subexprs, use the top-down expression.
            if (numCommon == 0) {
                _exprStr = topDown->getExprStr();
                exprDone = true;
            }
            
            // if common subexprs exist, and top-down is forced, use the
            // top-down expression regardless.  If node is big enough for
            // sharing, also assign the result to a temp var so it can be used
            // later.
            else if (force) {
                if (tooSmall)
                    _exprStr = topDown->getExprStr();
                else
                    makeNextTempVar(ex) << topDown->getExprStr() << _ph.getLineSuffix();
                exprDone = true;
            }

            // otherwise, there are common subexprs, and top-down is not forced,
            // so don't do top-down.
        
            delete topDown;
        }

        if (force) assert(exprDone);
        return exprDone;
    }

    // A grid point: just set expr.
    void PrintVisitorBottomUp::visit(GridPoint* gp) {
        trySimplePrint(gp, true);
    }

    // A grid index.
    void PrintVisitorBottomUp::visit(IndexExpr* ie) {
        trySimplePrint(ie, true);
    }

    // A constant: just set expr.
    void PrintVisitorBottomUp::visit(ConstExpr* ce) {
        trySimplePrint(ce, true);
    }

    // Code: just set expr.
    void PrintVisitorBottomUp::visit(CodeExpr* ce) {
        trySimplePrint(ce, true);
    }

    // A numerical unary operator.
    void PrintVisitorBottomUp::visit(UnaryNumExpr* ue) {

        // Try top-down on whole expression.
        // Example: '-a' creates no immediate output,
        // and '-a' is saved in _exprStr.
        if (trySimplePrint(ue, false))
            return;

        // Expand the RHS, then apply operator to result.
        // Example: '-(a * b)' might output the following:
        // temp1 = a * b;
        // temp2 = -temp1;
        // with 'temp2' saved in _exprStr.
        ue->getRhs()->accept(this); // sets _exprStr.
        string rhs = getExprStrAndClear();
        makeNextTempVar(ue) << ue->getOpStr() << ' ' << rhs << _ph.getLineSuffix();
    }

    // A numerical binary operator.
    void PrintVisitorBottomUp::visit(BinaryNumExpr* be) {

        // Try top-down on whole expression.
        // Example: 'a/b' creates no immediate output,
        // and 'a/b' is saved in _exprStr.
        if (trySimplePrint(be, false))
            return;

        // Expand both sides, then apply operator to result.
        // Example: '(a * b) / (c * d)' might output the following:
        // temp1 = a * b;
        // temp2 = b * c;
        // temp3 = temp1 / temp2;
        // with 'temp3' saved in _exprStr.
        be->getLhs()->accept(this); // sets _exprStr.
        string lhs = getExprStrAndClear();
        be->getRhs()->accept(this); // sets _exprStr.
        string rhs = getExprStrAndClear();
        makeNextTempVar(be) << lhs << ' ' << be->getOpStr() << ' ' << rhs << _ph.getLineSuffix();
    }

    // Boolean unary and binary operators.
    // For now, don't try to use bottom-up for these.
    // TODO: investigate whether there is any potential
    // benefit in doing this.
    void PrintVisitorBottomUp::visit(UnaryBoolExpr* ue) {
        trySimplePrint(ue, true);
    }
    void PrintVisitorBottomUp::visit(BinaryBoolExpr* be) {
        trySimplePrint(be, true);
    }
    void PrintVisitorBottomUp::visit(BinaryNum2BoolExpr* be) {
        trySimplePrint(be, true);
    }

    // A commutative operator.
    void PrintVisitorBottomUp::visit(CommutativeExpr* ce) {

        // Try top-down on whole expression.
        // Example: 'a*b' creates no immediate output,
        // and 'a*b' is saved in _exprStr.
        if (trySimplePrint(ce, false))
            return;

        // Make separate assignment for N-1 operands.
        // Example: 'a + b + c + d' might output the following:
        // temp1 = a + b;
        // temp2 = temp1 + c;
        // temp3 = temp2 = d;
        // with 'temp3' left in _exprStr;
        auto& ops = ce->getOps();
        assert(ops.size() > 1);
        string lhs, exStr;
        int opNum = 0;
        for (auto ep : ops) {
            opNum++;

            // eval the operand; sets _exprStr.            
            ep->accept(this);
            string opStr = getExprStrAndClear();

            // first operand; just save as LHS for next iteration.
            if (opNum == 1) {
                lhs = opStr;
                exStr = ep->makeStr();
            }

            // subsequent operands.
            // makes separate assignment for each one.
            // result is kept as LHS of next one.
            else {

                // Use whole expression only for the last step.
                Expr* ex = (opNum == (int)ops.size()) ? ce : NULL;

                // Add RHS to partial-result comment.
                exStr += ' ' + ce->getOpStr() + ' ' + ep->makeStr();

                // Output this step.
                makeNextTempVar(ex, exStr) << lhs << ' ' << ce->getOpStr() << ' ' <<
                    opStr << _ph.getLineSuffix();
                lhs = getExprStr(); // result used in next iteration, if any.
            }
        }

        // note: _exprStr contains result of last operation.
    }

    // Conditional.
    void PrintVisitorBottomUp::visit(IfExpr* ie) {
        trySimplePrint(ie, true);
        // note: _exprStr is now empty.
    }

    // An equality.
    void PrintVisitorBottomUp::visit(EqualsExpr* ee) {

        // Note that we don't try top-down here.
        // We always assign the RHS to a temp var and then
        // write the temp var to the grid.
    
        // Eval RHS.
        Expr* rp = ee->getRhs().get();
        rp->accept(this); // sets _exprStr.
        string rhs = _exprStr;

        // Assign RHS to a temp var.
        makeNextTempVar(rp) << rhs << _ph.getLineSuffix(); // sets _exprStr.
        string tmp = getExprStrAndClear();

        // Write temp var to grid.
        GridPointPtr gpp = ee->getLhs();
        _os << "\n // Update value at " << gpp->makeStr() << ".\n";
        _os << _ph.getLinePrefix() << _ph.writeToPoint(_os, *gpp, tmp) << _ph.getLineSuffix();

        // note: _exprStr is now empty.
    }

    ///////// POVRay.

    // Only want to visit the RHS of an equality.
    void POVRayPrintVisitor::visit(EqualsExpr* ee) {
        ee->getRhs()->accept(this);      
    }
    
    // A point: output it.
    void POVRayPrintVisitor::visit(GridPoint* gp) {
        _numPts++;

        // Pick a color based on its distance.
        size_t ci = gp->getArgOffsets().max();
        ci %= _colors.size();
        
        _os << "point(" + _colors[ci] + ", " << gp->getArgOffsets().makeValStr() << ")" << endl;
    }

    ////// DOT-language.

    // A grid access.
    void DOTPrintVisitor::visit(GridPoint* gp) {
        string label = getLabel(gp);
        if (label.size())
            _os << label << " [ shape = box ];" << endl;
    }

    // A constant.
    // TODO: don't share node.
    void DOTPrintVisitor::visit(ConstExpr* ce) {
        string label = getLabel(ce);
        if (label.size())
            _os << label << endl;
    }

    // Some hand-written code.
    void DOTPrintVisitor::visit(CodeExpr* ce) {
        string label = getLabel(ce);
        if (label.size())
            _os << label << endl;
    }

    // Generic numeric unary operators.
    void DOTPrintVisitor::visit(UnaryNumExpr* ue) {
        string label = getLabel(ue);
        if (label.size()) {
            _os << label << " [ label = \"" << ue->getOpStr() << "\" ];" << endl;
            _os << getLabel(ue, false) << " -> " << getLabel(ue->getRhs(), false) << ";" << endl;
            ue->getRhs()->accept(this);
        }
    }

    // Generic numeric binary operators.
    void DOTPrintVisitor::visit(BinaryNumExpr* be) {
        string label = getLabel(be);
        if (label.size()) {
            _os << label << " [ label = \"" << be->getOpStr() << "\" ];" << endl;
            _os << getLabel(be, false) << " -> " << getLabel(be->getLhs(), false) << ";" << endl <<
                getLabel(be, false) << " -> " << getLabel(be->getRhs(), false) << ";" << endl;
            be->getLhs()->accept(this);
            be->getRhs()->accept(this);
        }
    }

    // A commutative operator.
    void DOTPrintVisitor::visit(CommutativeExpr* ce) {
        string label = getLabel(ce);
        if (label.size()) {
            _os << label << " [ label = \"" << ce->getOpStr() << "\" ];" << endl;
            for (auto ep : ce->getOps()) {
                _os << getLabel(ce, false) << " -> " << getLabel(ep, false) << ";" << endl;
                ep->accept(this);
            }
        }
    }

    // An equals operator.
    void DOTPrintVisitor::visit(EqualsExpr* ee) {
        string label = getLabel(ee);
        if (label.size()) {
            _os << label << " [ label = \"EQUALS\" ];" << endl;
            _os << getLabel(ee, false) << " -> " << getLabel(ee->getLhs(), false)  << ";" << endl <<
                getLabel(ee, false) << " -> " << getLabel(ee->getRhs(), false) << ";" << endl;
            ee->getLhs()->accept(this);
            ee->getRhs()->accept(this);
        }
    }

    // A grid access.
    void SimpleDOTPrintVisitor::visit(GridPoint* gp) {
        string label = getLabel(gp);
        if (label.size()) {
            _os << label << " [ shape = box ];" << endl;
            _gridsSeen.insert(label);
        }
    }

    // Generic numeric unary operators.
    void SimpleDOTPrintVisitor::visit(UnaryNumExpr* ue) {
        ue->getRhs()->accept(this);
    }

    // Generic numeric binary operators.
    void SimpleDOTPrintVisitor::visit(BinaryNumExpr* be) {
        be->getLhs()->accept(this);
        be->getRhs()->accept(this);
    }

    // A commutative operator.
    void SimpleDOTPrintVisitor::visit(CommutativeExpr* ce) {
        for (auto ep : ce->getOps())
            ep->accept(this);
    }

    // An equals operator.
    void SimpleDOTPrintVisitor::visit(EqualsExpr* ee) {

        // LHS is source.
        ee->getLhs()->accept(this);
        string label = getLabel(ee, false);
        for (auto g : _gridsSeen)
            label = g;              // really should only be one.
        _gridsSeen.clear();

        // RHS nodes are target.
        ee->getRhs()->accept(this);
        for (auto g : _gridsSeen)
            _os << label << " -> " << g  << ";" << endl;
        _gridsSeen.clear();
    }

    ////////////// Printers ///////////////

    /////// Pseudo-code.

    // Print out a stencil in human-readable form, for debug or documentation.
    void PseudoPrinter::print(ostream& os) {

        os << "Stencil '" << _stencil.getName() << "' pseudo-code:" << endl;

        // Loop through all eqBundles.
        for (auto& eq : _eqBundles) {

            string egName = eq.getName();
            os << endl << " ////// Equation bundle '" << egName <<
                "' //////" << endl;

            CounterVisitor cv;
            eq.visitEqs(&cv);
            PrintHelper ph(NULL, &cv, "temp", "real", " ", ".\n");

            if (eq.cond.get()) {
                string condStr = eq.cond->makeStr();
                os << endl << " // Valid under the following condition:" << endl <<
                    ph.getLinePrefix() << "IF " << condStr << ph.getLineSuffix();
            }
            else
                os << endl << " // Valid under the default condition." << endl;

            os << endl << " // Top-down stencil calculation:" << endl;
            PrintVisitorTopDown pv1(os, ph, _settings);
            eq.visitEqs(&pv1);
            
            os << endl << " // Bottom-up stencil calculation:" << endl;
            PrintVisitorBottomUp pv2(os, ph, _settings);
            eq.visitEqs(&pv2);
        }
    }

    ///// DOT language.

    // Print out a stencil in DOT form
    void DOTPrinter::print(ostream& os) {

        DOTPrintVisitor* pv = _isSimple ?
            new SimpleDOTPrintVisitor(os) :
            new DOTPrintVisitor(os);

        os << "digraph \"Stencil " << _stencil.getName() << "\" {\n"
            "rankdir=LR; ranksep=1.5;\n";

        // Loop through all eqBundles.
        for (auto& eq : _eqBundles) {
            os << "subgraph \"Equation-bundle " << eq.getName() << "\" {" << endl;
            eq.visitEqs(pv);
            os << "}" << endl;
        }
        os << "}" << endl;
        delete pv;
    }


    ///// POVRay.

    // Print out a stencil in POVRay form.
    void POVRayPrinter::print(ostream& os) {

        os << "#include \"stencil.inc\"" << endl;
        int cpos = 25;
        os << "camera { location <" <<
            cpos << ", " << cpos << ", " << cpos << ">" << endl <<
            "  look_at <0, 0, 0>" << endl <<
            "}" << endl;

        // Loop through all eqBundles.
        for (auto& eq : _eqBundles) {

            // TODO: separate mutiple grids.
            POVRayPrintVisitor pv(os);
            eq.visitEqs(&pv);
            os << " // " << pv.getNumPoints() << " stencil points" << endl;
        }
    }

} // namespace yask.
