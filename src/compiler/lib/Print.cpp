/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2019, Intel Corporation

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

    // Declare a new temp var and set 'res' to it.
    // Print LHS of assignment to it.
    // 'ex' is used as key to save name of temp var and to write a comment.
    // If 'comment' is set, use it for the comment.
    // Return stream to continue w/RHS.
    ostream& PrintVisitorBase::makeNextTempVar(string& res, Expr* ex, string comment) {
        res = _ph.makeVarName();
        if (ex) {
            _tempVars[ex] = res;
            if (comment.length() == 0)
                comment = ex->makeStr();
        }
        if (comment.length())
            _os << endl << " // " << res << " = " << comment << "." << endl;
        _os << _ph.getLinePrefix() << _ph.getVarType() << " " << res << " = ";
        return _os;
    }

    /////// Top-down

    // A var read.
    // Uses the PrintHelper to format.
    string PrintVisitorTopDown::visit(VarPoint* gp) {
        _numCommon += _ph.getNumCommon(gp);
        return _ph.readFromPoint(_os, *gp);
    }

    // An index expression.
    string PrintVisitorTopDown::visit(IndexExpr* ie) {
        _numCommon += _ph.getNumCommon(ie);
        return ie->format(_varMap);
    }

    // A constant.
    // Uses the PrintHelper to format.
    string PrintVisitorTopDown::visit(ConstExpr* ce) {
        _numCommon += _ph.getNumCommon(ce);
        return _ph.addConstExpr(_os, ce->getNumVal());
    }

    // Some hand-written code.
    // Uses the PrintHelper to format.
    string PrintVisitorTopDown::visit(CodeExpr* ce) {
        _numCommon += _ph.getNumCommon(ce);
        return _ph.addCodeExpr(_os, ce->getCode());
    }

    // Generic unary operators.
    string PrintVisitorTopDown::visit(UnaryNumExpr* ue) {
        _numCommon += _ph.getNumCommon(ue);
        return ue->getOpStr() + ue->getRhs()->accept(this);
    }
    string PrintVisitorTopDown::visit(UnaryBoolExpr* ue) {
        _numCommon += _ph.getNumCommon(ue);
        return ue->getOpStr() + ue->getRhs()->accept(this);
    }
    string PrintVisitorTopDown::visit(UnaryNum2BoolExpr* ue) {
        _numCommon += _ph.getNumCommon(ue);
        return ue->getOpStr() + ue->getRhs()->accept(this);
    }

    // Generic binary operators.
    string PrintVisitorTopDown::visit(BinaryNumExpr* be) {
        _numCommon += _ph.getNumCommon(be);
        return "(" + be->getLhs()->accept(this) +
            " " + be->getOpStr() + " " +
            be->getRhs()->accept(this) + ")";
    }
    string PrintVisitorTopDown::visit(BinaryBoolExpr* be) {
        _numCommon += _ph.getNumCommon(be);
        return "(" + be->getLhs()->accept(this) +
            " " + be->getOpStr() + " " +
            be->getRhs()->accept(this) + ")";
    }
    string PrintVisitorTopDown::visit(BinaryNum2BoolExpr* be) {
        _numCommon += _ph.getNumCommon(be);
        return "(" + be->getLhs()->accept(this) +
            " " + be->getOpStr() + " " +
            be->getRhs()->accept(this) + ")";
    }

    // A commutative operator.
    string PrintVisitorTopDown::visit(CommutativeExpr* ce) {
        _numCommon += _ph.getNumCommon(ce);
        string res = "(";
        auto& ops = ce->getOps();
        int opNum = 0;
        for (auto ep : ops) {
            if (opNum > 0)
                res += " " + ce->getOpStr() + " ";
            res += ep->accept(this);
            opNum++;
        }
        return res + ")";
    }

    // A function call.
    string PrintVisitorTopDown::visit(FuncExpr* fe) {
        _numCommon += _ph.getNumCommon(fe);

        // Special case: increment common node count
        // for pairs.
        if (fe->getPair())
            _numCommon++;

        string res = _funcPrefix + fe->getOpStr() + "(";
        auto& ops = fe->getOps();
        int opNum = 0;
        for (auto ep : ops) {
            if (opNum > 0)
                res += ", ";
            res += ep->accept(this);
            opNum++;
        }
        return res + ")";
    }

    // An equals operator.
    string PrintVisitorTopDown::visit(EqualsExpr* ee) {

        // Get RHS.
        string rhs = ee->getRhs()->accept(this);

        // Write statement with embedded rhs.
        varPointPtr gpp = ee->getLhs();
        _os << _ph.getLinePrefix() << _ph.writeToPoint(_os, *gpp, rhs);

        // Null ptr => no condition.
        if (ee->getCond()) {
            string cond = ee->getCond()->accept(this);

            // pseudo-code format.
            _os << " IF (" << cond << ")";
        }
        if (ee->getStepCond()) {
            string cond = ee->getStepCond()->accept(this);

            // pseudo-code format.
            _os << " IF_STEP (" << cond << ")";
        }
        _os << _ph.getLineSuffix();

        // note: no need to update num-common.
        return "";              // EQUALS doesn't return a value.
    }

    /////// Bottom-up

    // Try some simple printing techniques.
    // Return expr if printing is done.
    // Return empty string if more complex method should be used.
    // TODO: the current code causes all nodes in an expr above a certain
    // point to avoid top-down printing because it looks at the original expr,
    // not the new one with temp vars. Fix this.
    string PrintVisitorBottomUp::trySimplePrint(Expr* ex, bool force) {
        string res;

        // How many nodes in ex?
        int exprSize = ex->getNumNodes();
        bool tooBig = exprSize > getSettings()._maxExprSize;
        bool tooSmall = exprSize < getSettings()._minExprSize;

        // Determine whether this expr has already been evaluated
        // and a variable holds its result.
        auto p = _tempVars.find(ex);
        if (p != _tempVars.end()) {

            // if so, just use the existing var.
            res = p->second;
        }

        // Consider top down if forcing or expr <= maxExprSize.
        else if (force || !tooBig) {

            // use a top-down printer to render the expr.
            PrintVisitorTopDown* topDown = newPrintVisitorTopDown();
            string td_res = ex->accept(topDown);

            // were there any common subexprs found?
            int numCommon = topDown->getNumCommon();

            // if no common subexprs, use the top-down expression.
            if (numCommon == 0)
                res = td_res;

            // if common subexprs exist, and top-down is forced, use the
            // top-down expression regardless.  If node is big enough for
            // sharing, also assign the result to a temp var so it can be used
            // later.
            else if (force) {
                if (tooSmall)
                    res = td_res;
                else
                    makeNextTempVar(res, ex) << td_res << _ph.getLineSuffix();
            }

            // otherwise, there are common subexprs, and top-down is not forced,
            // so don't do top-down.

            delete topDown;
        }

        if (force) assert(res.length());
        return res;
    }

    // A var point: just set expr.
    string PrintVisitorBottomUp::visit(VarPoint* gp) {
        return trySimplePrint(gp, true);
    }

    // A var index.
    string PrintVisitorBottomUp::visit(IndexExpr* ie) {
        return trySimplePrint(ie, true);
    }

    // A constant: just set expr.
    string PrintVisitorBottomUp::visit(ConstExpr* ce) {
        return trySimplePrint(ce, true);
    }

    // Code: just set expr.
    string PrintVisitorBottomUp::visit(CodeExpr* ce) {
        return trySimplePrint(ce, true);
    }

    // A numerical unary operator.
    string PrintVisitorBottomUp::visit(UnaryNumExpr* ue) {

        // Try top-down on whole expression.
        // Example: '-a' creates no immediate output,
        // and '-(a)' is saved in _exprStr.
        string res = trySimplePrint(ue, false);
        if (res.length())
            return res;

        // Expand the RHS, then apply operator to result.
        // Example: '-(a * b)' might output the following:
        // temp1 = a * b;
        // temp2 = -temp1;
        // with 'temp2' saved in _exprStr.
        string rhs = ue->getRhs()->accept(this);
        makeNextTempVar(res, ue) << ue->getOpStr() << rhs << _ph.getLineSuffix();
        return res;
    }

    // A numerical binary operator.
    string PrintVisitorBottomUp::visit(BinaryNumExpr* be) {

        // Try top-down on whole expression.
        // Example: 'a/b' creates no immediate output,
        // and 'a/b' is saved in _exprStr.
        string res = trySimplePrint(be, false);
        if (res.length())
            return res;

        // Expand both sides, then apply operator to result.
        // Example: '(a * b) / (c * d)' might output the following:
        // temp1 = a * b;
        // temp2 = b * c;
        // temp3 = temp1 / temp2;
        // with 'temp3' saved in _exprStr.
        string lhs = be->getLhs()->accept(this);
        string rhs = be->getRhs()->accept(this);
        makeNextTempVar(res, be) << lhs << ' ' << be->getOpStr() << ' ' << rhs << _ph.getLineSuffix();
        return res;
    }

    // Boolean unary and binary operators.
    // For now, don't try to use bottom-up for these.
    // TODO: investigate whether there is any potential
    // benefit in doing this.
    string PrintVisitorBottomUp::visit(UnaryBoolExpr* ue) {
        return trySimplePrint(ue, true);
    }
    string PrintVisitorBottomUp::visit(BinaryBoolExpr* be) {
        return trySimplePrint(be, true);
    }
    string PrintVisitorBottomUp::visit(BinaryNum2BoolExpr* be) {
        return trySimplePrint(be, true);
    }

    // Function call.
    string PrintVisitorBottomUp::visit(FuncExpr* fe) {
        string res;

        // If this is a paired function, handle it specially.
        auto* paired = fe->getPair();
        if (paired) {

            // Do we already have this result?
            if (_tempVars.count(fe)) {
                
                // Just use existing result.
                res = _tempVars.at(fe);
            }

            // No result yet.
            else {

                _os << endl << " // Combining " << fe->getOpStr() << " and " << paired->getOpStr() <<
                    "...\n";

                // First, eval all the args.
                string args;
                auto& ops = fe->getOps();
                for (auto ep : ops)
                    args += ", " + ep->accept(this); // sets _exprStr;

                // Make 2 temp vars.
                string res2;
                makeNextTempVar(res, fe) << "0" << _ph.getLineSuffix();
                makeNextTempVar(res2, paired) << "0" << _ph.getLineSuffix();

                // Call function to set both.
                _os << _ph.getLinePrefix() << 
                    _funcPrefix << fe->getOpStr() << "_and_" << paired->getOpStr() << 
                    "(" << res << ", " << res2 << args <<
                    ")" << _ph.getLineSuffix();
            }
        }

        // If not paired, handle normally.
        else {
            res = trySimplePrint(fe, false);
            if (res.length())
                return res;

            string args;
            auto& ops = fe->getOps();
            for (auto ep : ops) {
                if (args.length())
                    args += ", ";
                args += ep->accept(this);
            }
            makeNextTempVar(res, fe) << _funcPrefix << fe->getOpStr() <<
                "(" << args << ")" << _ph.getLineSuffix();
        }
        return res;
    }

    // A commutative operator.
    string PrintVisitorBottomUp::visit(CommutativeExpr* ce) {

        // Try top-down on whole expression.
        // Example: 'a*b' creates no immediate output,
        // and 'a*b' is saved in _exprStr.
        string res = trySimplePrint(ce, false);
        if (res.length())
            return res;

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

            // eval the operand.
            string opStr = ep->accept(this);

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
                string tmp;
                makeNextTempVar(tmp, ex, exStr) << lhs << ' ' << ce->getOpStr() << ' ' <<
                    opStr << _ph.getLineSuffix();
                lhs = tmp; // result returned and/or used in next iteration.
            }
        }
        return lhs;
    }

    // An equality.
    string PrintVisitorBottomUp::visit(EqualsExpr* ee) {

        // Note that we don't try top-down here.
        // We always assign the RHS to the var.

        // Eval RHS.
        Expr* rp = ee->getRhs().get();
        string rhs = rp->accept(this);

        // Assign RHS to a temp var.
        string tmp;
        makeNextTempVar(tmp, rp) << rhs << _ph.getLineSuffix(); // sets _exprStr.

        // Comment about update.
        varPointPtr gpp = ee->getLhs();
        _os << "\n // Update value at " << gpp->makeStr();

        // Comment about condition.
        // Null ptr => no condition.
        if (ee->getCond()) {
            string cond = ee->getCond()->makeStr();
            _os << " IF (" << cond << ")";
        }
        if (ee->getStepCond()) {
            string cond = ee->getStepCond()->makeStr();
            _os << " IF_STEP (" << cond << ")";
        }
        _os << ".\n";

        // Write RHS expr to var.
        _os << _ph.getLinePrefix() << _ph.writeToPoint(_os, *gpp, tmp) << _ph.getLineSuffix();

        return "";              // EQUALS doesn't return a value.
    }

    ///////// POVRay.

    // Only want to visit the RHS of an equality.
    string POVRayPrintVisitor::visit(EqualsExpr* ee) {
        return ee->getRhs()->accept(this);
    }

    // A point: output it.
    string POVRayPrintVisitor::visit(VarPoint* gp) {
        _numPts++;

        // Pick a color based on its distance.
        size_t ci = gp->getArgOffsets().max();
        ci %= _colors.size();

        _os << "point(" + _colors[ci] + ", " << gp->getArgOffsets().makeValStr() << ")" << endl;
        return "";
    }

    ////// DOT-language.

    // A var access.
    string DOTPrintVisitor::visit(VarPoint* gp) {
        string label = getLabel(gp);
        if (label.size())
            _os << label << " [ shape = box ];" << endl;
        return "";
    }

    // A constant.
    // TODO: don't share node.
    string DOTPrintVisitor::visit(ConstExpr* ce) {
        string label = getLabel(ce);
        if (label.size())
            _os << label << endl;
        return "";
    }

    // Some hand-written code.
    string DOTPrintVisitor::visit(CodeExpr* ce) {
        string label = getLabel(ce);
        if (label.size())
            _os << label << endl;
        return "";
    }

    // Generic numeric unary operators.
    string DOTPrintVisitor::visit(UnaryNumExpr* ue) {
        string label = getLabel(ue);
        if (label.size()) {
            _os << label << " [ label = \"" << ue->getOpStr() << "\" ];" << endl;
            _os << getLabel(ue, false) << " -> " << getLabel(ue->getRhs(), false) << ";" << endl;
            ue->getRhs()->accept(this);
        }
        return "";
    }

    // Generic numeric binary operators.
    string DOTPrintVisitor::visit(BinaryNumExpr* be) {
        string label = getLabel(be);
        if (label.size()) {
            _os << label << " [ label = \"" << be->getOpStr() << "\" ];" << endl;
            _os << getLabel(be, false) << " -> " << getLabel(be->getLhs(), false) << ";" << endl <<
                getLabel(be, false) << " -> " << getLabel(be->getRhs(), false) << ";" << endl;
            be->getLhs()->accept(this);
            be->getRhs()->accept(this);
        }
        return "";
    }

    // A commutative operator.
    string DOTPrintVisitor::visit(CommutativeExpr* ce) {
        string label = getLabel(ce);
        if (label.size()) {
            _os << label << " [ label = \"" << ce->getOpStr() << "\" ];" << endl;
            for (auto ep : ce->getOps()) {
                _os << getLabel(ce, false) << " -> " << getLabel(ep, false) << ";" << endl;
                ep->accept(this);
            }
        }
        return "";
    }

    // A function call.
    string DOTPrintVisitor::visit(FuncExpr* fe) {
        string label = getLabel(fe);
        if (label.size()) {
            _os << label << " [ label = \"" << fe->getOpStr() << "\" ];" << endl;
            for (auto ep : fe->getOps()) {
                _os << getLabel(fe, false) << " -> " << getLabel(ep, false) << ";" << endl;
                ep->accept(this);
            }
        }
        return "";
    }

    // An equals operator.
    string DOTPrintVisitor::visit(EqualsExpr* ee) {
        string label = getLabel(ee);
        if (label.size()) {
            _os << label << " [ label = \"EQUALS\" ];" << endl;
            _os << getLabel(ee, false) << " -> " << getLabel(ee->getLhs(), false)  << ";" << endl <<
                getLabel(ee, false) << " -> " << getLabel(ee->getRhs(), false) << ";" << endl;
            ee->getLhs()->accept(this);
            ee->getRhs()->accept(this);

            // Null ptr => no condition.
            if (ee->getCond()) {
                _os << getLabel(ee, false) << " -> " << getLabel(ee->getCond(), false)  << ";" << endl;
                ee->getCond()->accept(this);
            }
        }
        return "";
    }

    // A var access.
    string SimpleDOTPrintVisitor::visit(VarPoint* gp) {
        string label = getLabel(gp);
        if (label.size()) {
            _os << label << " [ shape = box ];" << endl;
            _varsSeen.insert(label);
        }
        return "";
    }

    // Generic numeric unary operators.
    string SimpleDOTPrintVisitor::visit(UnaryNumExpr* ue) {
        ue->getRhs()->accept(this);
        return "";
    }

    // Generic numeric binary operators.
    string SimpleDOTPrintVisitor::visit(BinaryNumExpr* be) {
        be->getLhs()->accept(this);
        be->getRhs()->accept(this);
        return "";
    }

    // A commutative operator.
    string SimpleDOTPrintVisitor::visit(CommutativeExpr* ce) {
        for (auto& ep : ce->getOps())
            ep->accept(this);
        return "";
    }

    // A function call.
    string SimpleDOTPrintVisitor::visit(FuncExpr* fe) {
        for (auto& ep : fe->getOps())
            ep->accept(this);
        return "";
    }

    // An equals operator.
    string SimpleDOTPrintVisitor::visit(EqualsExpr* ee) {

        // LHS is source.
        ee->getLhs()->accept(this);
        string label = getLabel(ee, false);
        for (auto g : _varsSeen)
            label = g;              // really should only be one.
        _varsSeen.clear();

        // RHS nodes are target.
        ee->getRhs()->accept(this);
        for (auto g : _varsSeen)
            _os << label << " -> " << g  << ";" << endl;
        _varsSeen.clear();

        // Ignoring conditions.
        return "";
    }

    ////////////// Printers ///////////////

    /////// Pseudo-code.

    // Print out a stencil in human-readable form, for debug or documentation.
    void PseudoPrinter::print(ostream& os) {

        os << "Stencil '" << _stencil.getName() << "' pseudo-code:" << endl;

        // Loop through all eqBundles.
        for (auto& eq : _eqBundles.getAll()) {

            string egName = eq->getName();
            os << endl << " ////// Equation bundle '" << egName <<
                "' //////" << endl;

            CounterVisitor cv;
            eq->visitEqs(&cv);
            PrintHelper ph(_settings, _dims, &cv, "temp", "real", " ", ".\n");

            if (eq->cond.get()) {
                string condStr = eq->cond->makeStr();
                os << endl << " // Valid under the following domain condition:" << endl <<
                    ph.getLinePrefix() << "IF " << condStr << ph.getLineSuffix();
            }
            if (eq->step_cond.get()) {
                string condStr = eq->step_cond->makeStr();
                os << endl << " // Valid under the following step condition:" << endl <<
                    ph.getLinePrefix() << "IF_STEP " << condStr << ph.getLineSuffix();
            }

            if (!_long) {
                PrintVisitorTopDown pv1(os, ph);
                eq->visitEqs(&pv1);
            } else {
                PrintVisitorBottomUp pv2(os, ph);
                eq->visitEqs(&pv2);
            }
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
        for (auto& eq : _eqBundles.getAll()) {
            os << "subgraph \"Equation-bundle " << eq->getName() << "\" {" << endl;
            eq->visitEqs(pv);
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
        for (auto& eq : _eqBundles.getAll()) {

            // TODO: separate mutiple vars.
            POVRayPrintVisitor pv(os);
            eq->visitEqs(&pv);
            os << " // " << pv.getNumPoints() << " stencil points" << endl;
        }
    }

} // namespace yask.
