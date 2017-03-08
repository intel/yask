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

//////////// Generic expression-printer classes /////////////

#ifndef PRINT_HPP
#define PRINT_HPP

#include "ExprUtils.hpp"
#include "StencilBase.hpp"

using namespace std;

// A PrintHelper is used by a PrintVisitor to format certain
// common items like variables, reads, and writes.
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
        return gp.makeStr() + " EQUALS " + val;
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

    // An index.
    virtual void visit(IntTupleExpr* ite);
    virtual void visit(IndexExpr* ie);
    
    // A constant.
    virtual void visit(ConstExpr* ce);

    // Some hand-written code.
    virtual void visit(CodeExpr* ce);

    // Generic unary operators.
    virtual void visit(UnaryNumExpr* ue);
    virtual void visit(UnaryBoolExpr* ue);
    virtual void visit(UnaryNum2BoolExpr* ue);

    // Generic binary operators.
    virtual void visit(BinaryNumExpr* be);
    virtual void visit(BinaryBoolExpr* be);
    virtual void visit(BinaryNum2BoolExpr* be);

    // A commutative operator.
    virtual void visit(CommutativeExpr* ce);

    // A conditional operator.
    virtual void visit(IfExpr* ie);

    // An equals operator.
    virtual void visit(EqualsExpr* ee);
};

// Outputs a simple, human-readable version of the AST in a bottom-up
// fashion with multiple sub-expressions, each assigned to a temp var.  The
// min/maxExprSize parameters control when and where expressions are
// sub-divided. Within each sub-expression, a top-down visitor is used.
class PrintVisitorBottomUp : public PrintVisitorBase {

protected:
    // max size of a single expression.
    int _maxExprSize;

    // min size to use sharing.
    int _minExprSize;

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
                         int maxExprSize, int minExprSize) :
        PrintVisitorBase(os, ph),
        _maxExprSize(maxExprSize),
        _minExprSize(minExprSize) { }

    // make a new top-down visitor with the same print helper.
    virtual PrintVisitorTopDown* newPrintVisitorTopDown() {
        return new PrintVisitorTopDown(_os, _ph);
    }

    // Try some simple printing techniques.
    // Return true if printing is done.
    // Return false if more complex method should be used.
    virtual bool trySimplePrint(Expr* ex, bool force);

    // A grid or param point.
    virtual void visit(GridPoint* gp);

    // An index.
    virtual void visit(IntTupleExpr* ite);

    // A constant.
    virtual void visit(ConstExpr* ce);

    // Code.
    virtual void visit(CodeExpr* ce);

    // Unary operators.
    virtual void visit(UnaryNumExpr* ue);
    virtual void visit(UnaryBoolExpr* ue);

    // Binary operators.
    virtual void visit(BinaryNumExpr* be);
    virtual void visit(BinaryBoolExpr* be);
    virtual void visit(BinaryNum2BoolExpr* be);

    // A commutative operator.
    virtual void visit(CommutativeExpr* ce);

    // A conditional operator.
    virtual void visit(IfExpr* ie);

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

    // A conditional operator.
    // Only visit expression.
    virtual void visit(IfExpr* ie) {
        ie->getExpr()->accept(this);
    }
};

// Outputs a full GraphViz input file.
class DOTPrintVisitor : public ExprVisitor {
protected:
    ostream& _os;
    set<string> _done;

    // Get label to use.
    // Return empty string if already done.
    virtual string getLabel(Expr* ep) {
        string key = ep->makeQuotedStr();
        if (_done.count(key))
            return "";
        _done.insert(key);
        return key;
    }
    
public:
    DOTPrintVisitor(ostream& os) : _os(os) { }

    // A grid or parameter read.
    virtual void visit(GridPoint* gp);

    // A constant.
    virtual void visit(ConstExpr* ce);

    // Some hand-written code.
    virtual void visit(CodeExpr* ce);

    // Generic numeric unary operators.
    virtual void visit(UnaryNumExpr* ue);

    // Generic numeric binary operators.
    virtual void visit(BinaryNumExpr* be);

    // A commutative operator.
    virtual void visit(CommutativeExpr* ce);

    // A conditional operator.
    // Only visit expression.
    virtual void visit(IfExpr* ie) {
        ie->getExpr()->accept(this);
    }

    // An equals operator.
    virtual void visit(EqualsExpr* ee);
};

// Outputs a simple GraphViz input file.
class SimpleDOTPrintVisitor : public DOTPrintVisitor {
protected:
    set<string> _gridsSeen;
    
public:
    SimpleDOTPrintVisitor(ostream& os) :
        DOTPrintVisitor(os) { }

    // A grid or parameter read.
    virtual void visit(GridPoint* gp);

    // A constant.
    virtual void visit(ConstExpr* ce) {}

    // Some hand-written code.
    virtual void visit(CodeExpr* ce) {}

    // Generic numeric unary operators.
    virtual void visit(UnaryNumExpr* ue);
    
    // Generic numeric binary operators.
    virtual void visit(BinaryNumExpr* be);

    // A commutative operator.
    virtual void visit(CommutativeExpr* ce);

    // A conditional operator.
    // Only visit expression.
    virtual void visit(IfExpr* ie) {
        ie->getExpr()->accept(this);
    }

    // An equals operator.
    virtual void visit(EqualsExpr* ee);
};

// PrinterBase is the main class for defining how to print a stencil.
// A PrinterBase uses one or more PrintHelpers and ExprVisitors to
// do this.
class PrinterBase {
protected:
    StencilBase& _stencil;
    Grids& _grids;
    Params& _params;
    EqGroups& _eqGroups;
    int _maxExprSize;
    int _minExprSize;

    // Return an upper-case string.
    string allCaps(string str) {
        transform(str.begin(), str.end(), str.begin(), ::toupper);
        return str;
    }
    
public:
    PrinterBase(StencilBase& stencil, EqGroups& eqGroups,
                int maxExprSize, int minExprSize) :
        _stencil(stencil), 
        _grids(stencil.getGrids()),
        _params(stencil.getParams()),
        _eqGroups(eqGroups),
        _maxExprSize(maxExprSize),
        _minExprSize(minExprSize)
    { }
    virtual ~PrinterBase() { }
    
};


// Print out a stencil in human-readable form, for debug or documentation.
class PseudoPrinter : public PrinterBase {
        
public:
    PseudoPrinter(StencilBase& stencil, EqGroups& eqGroups,
                  int maxExprSize, int minExprSize) :
        PrinterBase(stencil, eqGroups, maxExprSize, minExprSize) { }
    virtual ~PseudoPrinter() { }

    virtual void print(ostream& os);
};

// Print out a stencil in DOT-language form.
class DOTPrinter : public PrinterBase {
protected:
    bool _isSimple;
        
public:
    DOTPrinter(StencilBase& stencil, EqGroups& eqGroups,
               int maxExprSize, int minExprSize, bool isSimple) :
        PrinterBase(stencil, eqGroups, maxExprSize, minExprSize),
        _isSimple(isSimple) { }
    virtual ~DOTPrinter() { }

    virtual void print(ostream& os);
};

// Print out a stencil in POVRay form.
class POVRayPrinter : public PrinterBase {
        
public:
    POVRayPrinter(StencilBase& stencil, EqGroups& eqGroups,
                  int maxExprSize, int minExprSize) :
        PrinterBase(stencil, eqGroups, maxExprSize, minExprSize) { }
    virtual ~POVRayPrinter() { }

    virtual void print(ostream& os);
};

#endif
