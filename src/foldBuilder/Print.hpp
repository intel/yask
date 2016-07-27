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

//////////// Generic expression-printer classes /////////////

#ifndef PRINT_HPP
#define PRINT_HPP

#include "ExprUtils.hpp"
#include "StencilBase.hpp"

using namespace std;

// A PrintHelper is used by a PrintVisitor to format certain
// common items like variables and lines.
class PrintHelper {
    int _varNum;                // current var number.

protected:
    const CounterVisitor* _cv;  // counter info.
    string _varPrefix;          // first part of var name.
    string _varType;            // type, if any, of var.
    string _linePrefix;         // prefix for each line.
    string _lineSuffix;         // suffix for each line.

public:
    PrintHelper(const CounterVisitor* cv,
                const string& varPrefix,
                const string& varType,
                const string& linePrefix,
                const string& lineSuffix) :
        _varNum(1), _cv(cv),
        _varPrefix(varPrefix), _varType(varType),
        _linePrefix(linePrefix), _lineSuffix(lineSuffix) { }

    virtual ~PrintHelper() { }

    virtual const string& getVarType() const { return _varType; }
    virtual void setVarType(const string& varType) { _varType = varType; }
    virtual const string& getLinePrefix() const { return _linePrefix; }
    virtual const string& getLineSuffix() const { return _lineSuffix; }
    const CounterVisitor* getCounters() const { return _cv; }

    // Return count from counter visitor.
    int getCount(Expr* ep) {
        if (!_cv)
            return 0;
        return _cv->getCount(ep);
    }

    // Return number of times this node is shared.
    int getNumCommon(Expr* ep) {
        if (!_cv)
            return 0;
        int c = _cv->getCount(ep);
        return (c <= 1) ? 0 : c-1;
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

    // Return a parameter reference.
    virtual string readFromParam(ostream& os, const GridPoint& pp) {
        string str = pp.getName();
        if (pp.size())
            str += "(" + pp.makeValStr() + ")";
        return str;
    }
    
    // Return a grid reference.
    // The 'os' parameter is provided for derived types that
    // need to write intermediate code to a stream.
    virtual string readFromPoint(ostream& os, const GridPoint& gp) {
        return gp.makeStr();
    }

    // Update a grid point.
    // The 'os' parameter is provided for derived types that
    // need to write intermediate code to a stream.
    virtual string writeToPoint(ostream& os, const GridPoint& gp, const string& val) {
        return gp.makeStr() + " = " + val;
    }
};

// Base class for a print visitor.
class PrintVisitorBase : public ExprVisitor {

protected:
    ostream& _os;               // used for printing intermediate results as needed.
    PrintHelper& _ph;           // used to format items for printing.

    // After visiting an expression, the part of the result not written to _os
    // is stored in _exprStr.
    string _exprStr;

public:
    // os is used for printing intermediate results as needed.
    PrintVisitorBase(ostream& os, PrintHelper& ph) :
        _os(os), _ph(ph) { }

    virtual ~PrintVisitorBase() { }

    // Get unwritten result expression, if any.
    virtual string getExprStr() const {
        return _exprStr;
    }

    // Get unwritten result expression, if any.
    // Clear result.
    virtual string getExprStrAndClear() {
        string v = _exprStr;
        _exprStr = "";
        return v;
    }
};

// Outputs a simple, human-readable version of the AST
// in a top-down fashion. Expressions will be written to 'os',
// and anything 'left over' will be left in '_exprStr'.
class PrintVisitorTopDown : public PrintVisitorBase {
    int _numCommon;
    
public:
    PrintVisitorTopDown(ostream& os, PrintHelper& ph) :
        PrintVisitorBase(os, ph), _numCommon(0) { }

    // Get the number of shared nodes found after this visitor
    // has been accepted.
    int getNumCommon() const { return _numCommon; }
    
    // A grid or parameter read.
    virtual void visit(GridPoint* gp);

    // A constant.
    virtual void visit(ConstExpr* ce);

    // Some code.
    virtual void visit(CodeExpr* ce);

    // A generic unary operator.
    virtual void visit(UnaryExpr* ue);

    // A generic binary operator.
    virtual void visit(BinaryExpr* be);

    // A commutative operator.
    virtual void visit(CommutativeExpr* ce);

    // An equals operator.
    virtual void visit(EqualsExpr* ee);
};

// Outputs a simple, human-readable version of the AST in a bottom-up
// fashion with multiple expressions, each assigned to a temp var.  The
// maxPoints parameter controls the size of each separate expression. Within
// each expression, a top-down visitor is used.
class PrintVisitorBottomUp : public PrintVisitorBase {

protected:
    // max points to print top-down before printing bottom-up.
    int _maxPoints;  

    // map sub-expressions to var names.
    map<Expr*, string> _tempVars;

    // Declare a new temp var.
    // Set _exprStr to it.
    // Print LHS of assignment to it.
    // If 'ex' is non-null, it is used as key to save name of temp var and
    // to write a comment.
    // If 'comment' is set, use it for the comment.
    // Return stream to continue w/RHS.
    virtual ostream& makeNextTempVar(Expr* ex, string comment = "");

public:
    // os is used for printing intermediate results as needed.
    PrintVisitorBottomUp(ostream& os, PrintHelper& ph,
                         int maxPoints) :
        PrintVisitorBase(os, ph), _maxPoints(maxPoints) { }

    // make a new top-down visitor with the same print helper.
    virtual PrintVisitorTopDown* newPrintVisitorTopDown() {
        return new PrintVisitorTopDown(_os, _ph);
    }

    // Look for existing var.
    // Then, use top-down method for simple exprs.
    // Return true if successful.
    virtual bool tryTopDown(Expr* ex, bool leaf);

    // A grid or param point.
    virtual void visit(GridPoint* gp);

    // A constant.
    virtual void visit(ConstExpr* ce);

    // Code.
    virtual void visit(CodeExpr* ce);

    // A unary operator.
    virtual void visit(UnaryExpr* ue);

    // A binary operator.
    virtual void visit(BinaryExpr* be);

    // A commutative operator.
    virtual void visit(CommutativeExpr* ce);

    // An equality.
    virtual void visit(EqualsExpr* ee);
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

    // Equals op.
    virtual void visit(EqualsExpr* ee);
    
    // A point.
    virtual void visit(GridPoint* gp);
};

// PrinterBase is the main class for defining how to print
// a stencil.
class PrinterBase {
protected:
    StencilBase& _stencil;
    Grids& _grids;
    Params& _params;
    Equations& _equations;
    int _exprSize;

    // Return an upper-case string.
    string allCaps(string str) {
        transform(str.begin(), str.end(), str.begin(), ::toupper);
        return str;
    }
    
public:
    PrinterBase(StencilBase& stencil, Equations& equations,
                int exprSize) :
        _stencil(stencil), 
        _grids(stencil.getGrids()),
        _params(stencil.getParams()),
        _equations(equations),
        _exprSize(exprSize)
    { }
    virtual ~PrinterBase() { }
    
};


// Print out a stencil in human-readable form, for debug or documentation.
class PseudoPrinter : public PrinterBase {
        
public:
    PseudoPrinter(StencilBase& stencil, Equations& equations,
                  int exprSize) :
        PrinterBase(stencil, equations, exprSize) { }
    virtual ~PseudoPrinter() { }

    virtual void print(ostream& os);
};

// Print out a stencil in POVRay form.
class POVRayPrinter : public PrinterBase {
        
public:
    POVRayPrinter(StencilBase& stencil, Equations& equations,
                  int exprSize) :
        PrinterBase(stencil, equations, exprSize) { }
    virtual ~POVRayPrinter() { }

    virtual void print(ostream& os);
};

#endif
