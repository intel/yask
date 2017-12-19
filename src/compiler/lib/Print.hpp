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
#include "Eqs.hpp"
#include "Soln.hpp"

namespace yask {

    // A PrintHelper is used by a PrintVisitor to format certain
    // common items like variables, reads, and writes.
    class PrintHelper {
        int _varNum;                // current temp-var number.

    protected:
        const Dimensions* _dims;    // problem dims.
        const CounterVisitor* _cv;  // counter info.
        string _varPrefix;          // first part of var name.
        string _varType;            // type, if any, of var.
        string _linePrefix;         // prefix for each line.
        string _lineSuffix;         // suffix for each line.
        VarMap _localVars;          // map from expression strings to local var names.

    public:
        PrintHelper(const Dimensions* dims,
                    const CounterVisitor* cv,
                    const string& varPrefix,
                    const string& varType,
                    const string& linePrefix,
                    const string& lineSuffix) :
            _varNum(1), _dims(dims), _cv(cv),
            _varPrefix(varPrefix), _varType(varType),
            _linePrefix(linePrefix), _lineSuffix(lineSuffix) { }

        virtual ~PrintHelper() { }

        virtual const string& getVarType() const { return _varType; }
        virtual void setVarType(const string& varType) { _varType = varType; }
        virtual const string& getLinePrefix() const { return _linePrefix; }
        virtual const string& getLineSuffix() const { return _lineSuffix; }
        const CounterVisitor* getCounters() const { return _cv; }
        virtual void forgetLocalVars() { _localVars.clear(); }

        // get dims.
        virtual const Dimensions* getDims() const {
            return _dims;
        }

        // Return count from counter visitor.
        int getCount(Expr* ep) {
            if (!_cv)
                return 0;
            return _cv->getCount(ep);
        }

        // Return number of times 'ep' node is shared.
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

        // If var exists for 'expr', return it.
        // If not, create var of 'type' in 'os' and return it.
        virtual string getLocalVar(ostream& os, const string& expr,
                                   string type = "") {

            if (_localVars.count(expr))
                return _localVars.at(expr);

            // Make a var.
            if (!type.length())
                type = _varType;
            string vName = makeVarName();
            os << _linePrefix << type << " " << vName <<
                " = " << expr << _lineSuffix;
            _localVars[expr] = vName;
            return vName;
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
        virtual string readFromPoint(ostream& os, const GridPoint& gp) {
            return gp.makeStr();
        }

        // Return code to update a grid point.
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

        // Ref to compiler settings.
        CompilerSettings& _settings;

        // After visiting an expression, the part of the result not written to _os
        // is stored in _exprStr.
        string _exprStr;

        // Make these substitutions to indices in expressions.
        const VarMap* _varMap = 0;
        
        // Map sub-expressions to var names.
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
        PrintVisitorBase(ostream& os,
                         PrintHelper& ph,
                         CompilerSettings& settings,
                         const VarMap* varMap = 0) :
            _os(os), _ph(ph), _settings(settings), _varMap(varMap) { }

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

    // Outputs an AST traversed in a top-down fashion. Expressions will be
    // written to 'os', and anything 'left over' will be left in '_exprStr'.
    class PrintVisitorTopDown : public PrintVisitorBase {
        int _numCommon;

    public:
        PrintVisitorTopDown(ostream& os, PrintHelper& ph,
                            CompilerSettings& settings,
                            const VarMap* varMap = 0) :
            PrintVisitorBase(os, ph, settings, varMap), _numCommon(0) { }

        // Get the number of shared nodes found after this visitor
        // has been accepted.
        int getNumCommon() const { return _numCommon; }
    
        // A grid access.
        virtual void visit(GridPoint* gp);

        // A grid index.
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

    // Outputs an AST traversed in a bottom-up fashion with multiple
    // sub-expressions, each assigned to a temp var.  The min/maxExprSize
    // vars in CompilerSettings control when and where expressions are
    // sub-divided. Within each sub-expression, a top-down visitor is used.
    class PrintVisitorBottomUp : public PrintVisitorBase {

    public:
        // os is used for printing intermediate results as needed.
        PrintVisitorBottomUp(ostream& os, PrintHelper& ph,
                             CompilerSettings& settings,
                             const VarMap* varMap = 0) :
            PrintVisitorBase(os, ph, settings, varMap) {}

        // make a new top-down visitor with the same print helper.
        virtual PrintVisitorTopDown* newPrintVisitorTopDown() {
            return new PrintVisitorTopDown(_os, _ph, _settings);
        }

        // Try some simple printing techniques.
        // Return true if printing is done.
        // Return false if more complex method should be used.
        virtual bool trySimplePrint(Expr* ex, bool force);

        // A grid point.
        virtual void visit(GridPoint* gp);

        // An index.
        virtual void visit(IndexExpr* ie);

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
        // Return empty string if already done if once == true.
        virtual string getLabel(Expr* ep, bool once = true) {
            string key = ep->makeQuotedStr("\"");
            if (once) {
                if (_done.count(key))
                    return "";
                _done.insert(key);
            }
            return key;
        }
        virtual string getLabel(ExprPtr ep, bool once = true) {
            return getLabel(ep.get(), once);
        }
    
    public:
        DOTPrintVisitor(ostream& os) : _os(os) { }

        // A grid read.
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

        // A grid read.
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
        StencilSolution& _stencil;
        Grids& _grids;
        EqGroups& _eqGroups;
        CompilerSettings& _settings;
        
    public:
        PrinterBase(StencilSolution& stencil,
                    EqGroups& eqGroups) :
            _stencil(stencil), 
            _grids(stencil.getGrids()),
            _eqGroups(eqGroups),
            _settings(stencil.getSettings())
        { }
        virtual ~PrinterBase() { }

        // Number of elements that should be in a SIMD vector.
        // 1 => scalars only.
        virtual int num_vec_elems() const { return 1; }

        // Whether multi-dim folding is efficient.
        virtual bool is_folding_efficient() const { return false; }
        
        // Output to 'os'.
        virtual void print(ostream& os) =0;

        // Output to string.
        virtual string format() {
            ostringstream oss;
            print(oss);
            return oss.str();
        }

        // Return an upper-case string.
        static string allCaps(string str) {
            transform(str.begin(), str.end(), str.begin(), ::toupper);
            return str;
        }
    };

    // Print out a stencil in human-readable form, for debug or documentation.
    class PseudoPrinter : public PrinterBase {
        
    public:
        PseudoPrinter(StencilSolution& stencil,
                      EqGroups& eqGroups) :
            PrinterBase(stencil, eqGroups) { }
        virtual ~PseudoPrinter() { }

        virtual void print(ostream& os);
    };

    // Print out a stencil in DOT-language form.
    class DOTPrinter : public PrinterBase {
    protected:
        bool _isSimple;
        
    public:
        DOTPrinter(StencilSolution& stencil, EqGroups& eqGroups,
                   bool isSimple) :
            PrinterBase(stencil, eqGroups),
            _isSimple(isSimple) { }
        virtual ~DOTPrinter() { }

        virtual void print(ostream& os);
    };

    // Print out a stencil in POVRay form.
    class POVRayPrinter : public PrinterBase {
        
    public:
        POVRayPrinter(StencilSolution& stencil, EqGroups& eqGroups) :
            PrinterBase(stencil, eqGroups) { }
        virtual ~POVRayPrinter() { }

        virtual void print(ostream& os);
    };

} // namespace yask.

#endif
