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

//////////// Generic Visitors and Printers /////////////

#include "Expr.hpp"

using namespace std;

// Base class for an Expr-tree visitor.
class ExprVisitor {
public:
    virtual ~ExprVisitor() { }

    // Leaf-node visitors.
    virtual void visit(ConstExpr* ce) { }
    virtual void visit(CodeExpr* ce) { }
    virtual void visit(GridPoint* gp) { }

    // By default, a unary visitor just visits its operand.
    virtual void visit(UnaryExpr* ue) {
        ue->_rhs->accept(this);
    }

    // By default, a binary visitor just visits its operands.
    virtual void visit(BinaryExpr* be) {
        be->_lhs->accept(this);
        be->_rhs->accept(this);
    }

    // By default, a commutative visitor just visits its operands.
    virtual void visit(CommutativeExpr* ce) {
        ExprPtrVec& ops = ce->getOps();
        for (auto i = ops.begin(); i != ops.end(); i++) {
            auto ep = *i;
            ep->accept(this);
        }
    }
};

// Base class to define methods for printing.
class PrintHelper {
    int _varNum;                // current var number.

protected:
    string _varPrefix;          // first part of var name.
    string _varType;            // type, if any, of var.

public:
    PrintHelper(const string& varPrefix = "var", const string& varType = "real") :
        _varNum(1), _varPrefix(varPrefix), _varType(varType) { }

    virtual ~PrintHelper() { }

    virtual const string& getVarType() const {
        return _varType;
    }
    virtual void setVarType(const string& varType) {
        _varType = varType;
    }
    
    // Make and return next var name.
    virtual string makeVarName() {
        ostringstream oss;
        oss << _varPrefix << _varNum++;
        return oss.str();
    }

    // Return any code expression.
    // The 'os' parameter is provided for derived types that
    // need to write intermediate code to a stream.
    virtual string addCodeExpr(ostream& os, const string& code) {
        return code;
    }

    // Return a constant expression.
    // The 'os' parameter is provided for derived types that
    // need to write intermediate code to a stream.
    virtual string addConstExpr(ostream& os, double v) {
        ostringstream oss;
        oss << v;
        return oss.str();
    }

    // Return a grid reference.
    // The 'os' parameter is provided for derived types that
    // need to write intermediate code to a stream.
    virtual string constructPoint(ostream& os, const GridPoint& gp) {
        return gp.makeStr();
    }
};

// Base class for a print visitor.
class PrintVisitorBase : public ExprVisitor {

protected:
    ostream& _os;
    PrintHelper& _ph;
    string _exprStr; // result of visiting an Expr.

public:
    // os is used for printing intermediate results as needed.
    PrintVisitorBase(ostream& os, PrintHelper& ph) :
        _os(os), _ph(ph) { }

    virtual ~PrintVisitorBase() { }

    // Get final result expression.
    virtual string getExprStr() =0;

    // Declare a new temp var.
    // Set _exprStr to it.
    // Print LHS of assignment to it.
    // Return stream to continue w/RHS.
    virtual ostream& makeNextTempVar() {
        _exprStr = _ph.makeVarName();
        _os << _ph.getVarType() << " " << _exprStr << " = ";
        return _os;
    }
};

// Outputs a simple, human-readable version of the AST
// in a top-down fashion on one line.
class PrintVisitorTopDown : public PrintVisitorBase {
protected:
    ostringstream _oss; // build expression in this string.

public:
    PrintVisitorTopDown(ostream& os, PrintHelper& ph) :
        PrintVisitorBase(os, ph) { }

    // Get final result expression.
    virtual string getExprStr() {
        return _oss.str();
    }

    // A point.
    virtual void visit(GridPoint* gp) {
        _oss << _ph.constructPoint(_os, *gp);
    }

    // A constant.
    virtual void visit(ConstExpr* ce) {
        _oss << _ph.addConstExpr(_os, ce->getVal());
    }

    // Some code.
    virtual void visit(CodeExpr* ce) {
        _oss << _ph.addCodeExpr(_os, ce->getCode());
    }

    // A unary operator.
    virtual void visit(UnaryExpr* ue) {
        _oss << ue->_opStr;
        ue->_rhs->accept(this);
    }

    // A binary operator.
    virtual void visit(BinaryExpr* be) {
        _oss << "(";
        be->_lhs->accept(this);
        _oss << " " << be->_opStr << " ";
        be->_rhs->accept(this);
        _oss << ")";
    }

    // A commutative operator.
    virtual void visit(CommutativeExpr* ce) {
        _oss << "(";
        ExprPtrVec& ops = ce->getOps();
        int opNum = 0;
        for (auto i = ops.begin(); i != ops.end(); i++, opNum++) {
            auto ep = *i;
            if (opNum > 0)
                _oss << " " << ce->_opStr << " ";
            ep->accept(this);
        }
        _oss << ")";
    }
};

// Outputs a simple, human-readable version of the AST in a bottom-up
// fashion with multiple expressions, each assigned to a temp var.  The
// maxPoints parameter controls the size of each separate expression. Within
// each expression, a top-down visitor is used.
class PrintVisitorBottomUp : public PrintVisitorBase {

protected:
    int _maxPoints;  // max points to print top-down before printing bottom-up.

public:
    // os is used for printing intermediate results as needed.
    PrintVisitorBottomUp(ostream& os, PrintHelper& ph, int maxPoints) :
        PrintVisitorBase(os, ph), _maxPoints(maxPoints) { }

    // Get final result expression.
    virtual string getExprStr() {
        return _exprStr;
    }

    // make a new top-down visitor.
    virtual PrintVisitorTopDown* newPrintVisitorTopDown() {
        return new PrintVisitorTopDown(_os, _ph);
    }
    
    // Use top-down method for simple exprs.
    // Return true if successful.
    virtual bool tryTopDown(Expr* ex, bool force=false) {

        // Use top down if <= maxPoints points in ex.
        if (force || ex->getNumPoints() <= _maxPoints) {
            PrintVisitorTopDown* topDown = newPrintVisitorTopDown();
            ex->accept(topDown);
            _exprStr = topDown->getExprStr();
            delete topDown;
            return true;
        }

        return false;
    }

    // A point: just set expr.
    virtual void visit(GridPoint* gp) {
        tryTopDown(gp, true);
    }

    // A constant: just set expr.
    virtual void visit(ConstExpr* ce) {
        tryTopDown(ce, true);
    }

    // Code: just set expr.
    virtual void visit(CodeExpr* ce) {
        tryTopDown(ce, true);
    }

    // A unary operator: get RHS; print statement; make new var for result.
    virtual void visit(UnaryExpr* ue) {
        if (tryTopDown(ue))
            return;

        ue->_rhs->accept(this); // sets _exprStr.
        string rhs = _exprStr;
        makeNextTempVar() << ue->_opStr << rhs << ";" << endl;
    }

    // A binary operator: get LHS and RHS; print statement; make new var for result.
    virtual void visit(BinaryExpr* be) {
        if (tryTopDown(be))
            return;

        be->_lhs->accept(this); // sets _exprStr.
        string lhs = _exprStr;
        be->_rhs->accept(this); // sets _exprStr.
        string rhs = _exprStr;
        makeNextTempVar() << lhs << ' ' << be->_opStr << ' ' << rhs << ";" << endl;
    }

    // A commutative operator: get ops; print statement(s); make new var(s) for result.
    virtual void visit(CommutativeExpr* ce) {
        if (tryTopDown(ce))
            return;

        ExprPtrVec& ops = ce->getOps();
        string lhs, rhs;
        int opNum = 0;
        for (auto i = ops.begin(); i != ops.end(); i++, opNum++) {
            auto ep = *i;
            ep->accept(this); // eval the operand; sets _exprStr.

            // first operand.
            if (opNum == 0)
                lhs = _exprStr;

            // subsequent operands.
            else {
                rhs = _exprStr;
                makeNextTempVar() << lhs << ' ' << ce->_opStr << ' ' << rhs << ";" << endl;
                lhs = _exprStr; // for next iteration.
            }
        }
    }
};


// Outputs a POV-Ray input file.
class POVRayPrintVisitor : public ExprVisitor {
protected:
    ostream& _os;
    vector<string> _colors;
    int _numPts;

public:
    POVRayPrintVisitor(ostream& os) : _os(os), _numPts(0) {

        // Make a rainbow.
        _colors.push_back("Red");
        _colors.push_back("Orange");
        _colors.push_back("Yellow");
        _colors.push_back("Green");
        _colors.push_back("Blue");
        _colors.push_back("Violet");

        // NB: could also calculate the hue and use CHSL2RGB().
    }

    virtual int getNumPoints() const {
        return _numPts;
    }

    // A point: output it.
    virtual void visit(GridPoint* gp) {
        _numPts++;

        // Pick a color based on its distance.
        size_t ci = max(max(abs(gp->_i), abs(gp->_j)), abs(gp->_k));
        ci %= _colors.size();
        
        _os << "point(" + _colors[ci] + ", " <<
            gp->_i << ", " << gp->_j << ", " << gp->_k << ")" << endl;
    }
};

