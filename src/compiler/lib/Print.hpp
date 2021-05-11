/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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

#pragma once

#include "ExprUtils.hpp"
#include "Eqs.hpp"
#include "Solution.hpp"

namespace yask {

    // A PrintHelper is used by a PrintVisitor to format certain
    // common items like variables, reads, and writes.
    class PrintHelper {
        int _var_num;                // current temp-var number.

    protected:
        const CompilerSettings& _settings; // compiler settings.
        const Dimensions& _dims;    // problem dims.
        const CounterVisitor* _cv;  // counter info.
        string _var_type;            // type, if any, of var.
        string _line_prefix;         // prefix for each line.
        string _line_suffix;         // suffix for each line.
        VarMap _local_vars;          // map from expression strings to local var names.

    public:
        PrintHelper(const CompilerSettings& settings,
                    const Dimensions& dims,
                    const CounterVisitor* cv,
                    const string& var_type,
                    const string& line_prefix,
                    const string& line_suffix) :
            _var_num(1), _settings(settings), _dims(dims), _cv(cv),
            _var_type(var_type), _line_prefix(line_prefix), _line_suffix(line_suffix) { }

        virtual ~PrintHelper() { }

        virtual const string& get_var_type() const { return _var_type; }
        virtual void set_var_type(const string& var_type) { _var_type = var_type; }
        virtual const string& get_line_prefix() const { return _line_prefix; }
        virtual const string& get_line_suffix() const { return _line_suffix; }
        const CounterVisitor* get_counters() const { return _cv; }
        virtual void forget_local_vars() { _local_vars.clear(); }

        // get dims & settings.
        const Dimensions& get_dims() const {
            return _dims;
        }
        const CompilerSettings& get_settings() const {
            return _settings;
        }

        // Return count from counter visitor.
        // Relies on counter visitor visiting all nodes.
        int get_count(Expr* ep) {
            if (!_cv)
                return 0;
            return _cv->get_count(ep);
        }

        // Return number of times 'ep' node is shared.
        // Relies on counter visitor visiting all nodes.
        int get_num_common(Expr* ep) {
            if (!_cv)
                return 0;
            int c = _cv->get_count(ep);
            return (c <= 1) ? 0 : c-1;
        }

        // Make and return next var name.
        virtual string make_var_name(string prefix) {
            return prefix + "_temp" + to_string(_var_num++);
        }

        // Determine if local var exists for 'expr'.
        virtual bool is_local_var(const string& expr) const {
            return _local_vars.count(expr) != 0;
        }

        // If var exists for 'expr', return it.
        // If not, create var of 'type' in 'os' and return it.
        virtual string get_local_var(ostream& os, const string& expr,
                                     string type, string prefix) {

            if (_local_vars.count(expr))
                return _local_vars.at(expr);

            // Make a var.
            if (!type.length())
                type = _var_type;
            string v_name = make_var_name(prefix);
            os << _line_prefix << type << " " << v_name <<
                " = " << expr << _line_suffix;
            _local_vars[expr] = v_name;
            return v_name;
        }

        // Return any code expression.
        // The 'os' parameter is provided for derived types that
        // need to write intermediate code to a stream.
        virtual string add_code_expr(ostream& os, const string& code) {
            return code;
        }

        // Return a constant expression.
        // The 'os' parameter is provided for derived types that
        // need to write intermediate code to a stream.
        virtual string add_const_expr(ostream& os, double v) {

            // Int representation equivalent?
            if (double(int(v)) == v)
                return to_string(int(v));

            // Need scientific repr?
            if (v < 1e-4 || v >= 1e5) {
                ostringstream oss;
                oss << scientific << v;
                return oss.str();
            }

            // Default fixed-point.
            return to_string(v);
        }

        // Return a var reference.
        // The 'os' parameter is provided for derived types that
        // need to write intermediate code to a stream.
        virtual string read_from_point(ostream& os, const VarPoint& gp) {
            return gp.make_str();
        }

        // Return code to update a var point.
        // The 'os' parameter is provided for derived types that
        // need to write intermediate code to a stream.
        virtual string write_to_point(ostream& os, const VarPoint& gp, const string& val) {
            return gp.make_str() + " EQUALS " + val;
        }
    };

    // Base class for a print visitor.
    class PrintVisitorBase : public ExprVisitor {

    protected:
        ostream& _os;               // used for printing non-returned results as needed.
        PrintHelper& _ph;           // used to format items for printing.

        // Prefix for function calls.
        string _func_prefix = "yask_";

        // Make these substitutions to indices in expressions.
        const VarMap* _var_map = 0;

        // Map sub-expressions to var names.
        map<Expr*, string> _temp_vars;

        // Declare a new temp var and set 'res' to it.
        // Prepend 'prefix' to name of var.
        // Print LHS of assignment to it.
        // 'ex' is used as key to save name of temp var and to write a comment.
        // If 'comment' is set, use it for the comment.
        // Return stream to continue w/RHS.
        virtual ostream& make_next_temp_var(string& res, Expr* ex,
                                            string prefix, string comment);

    public:
        // os is used for printing intermediate results as needed.
        PrintVisitorBase(ostream& os,
                         PrintHelper& ph,
                         const VarMap* var_map = 0) :
            _os(os), _ph(ph), _var_map(var_map) { }

        virtual ~PrintVisitorBase() { }

        // get dims & settings.
        const Dimensions& get_dims() const {
            return _ph.get_dims();
        }
        const CompilerSettings& get_settings() const {
            return _ph.get_settings();
        }
    };

    // Outputs an AST traversed in a top-down fashion. Expressions will be
    // written to 'os'.
    class PrintVisitorTopDown : public PrintVisitorBase {
        int _num_common;

    public:
        PrintVisitorTopDown(ostream& os, PrintHelper& ph,
                            const VarMap* var_map = 0) :
            PrintVisitorBase(os, ph, var_map), _num_common(0) { }

        // Get the number of shared nodes found after this visitor
        // has been accepted.
        int get_num_common() const { return _num_common; }

        // A var access.
        virtual string visit(VarPoint* gp);

        // A var index.
        virtual string visit(IndexExpr* ie);

        // A constant.
        virtual string visit(ConstExpr* ce);

        // Some hand-written code.
        virtual string visit(CodeExpr* ce);

        // Generic unary operators.
        virtual string visit(UnaryNumExpr* ue);
        virtual string visit(UnaryBoolExpr* ue);
        virtual string visit(UnaryNum2BoolExpr* ue);

        // Generic binary operators.
        virtual string visit(BinaryNumExpr* be);
        virtual string visit(BinaryBoolExpr* be);
        virtual string visit(BinaryNum2BoolExpr* be);

        // A commutative operator.
        virtual string visit(CommutativeExpr* ce);

        // A function call.
        virtual string visit(FuncExpr* fe);

        // An equals operator.
        virtual string visit(EqualsExpr* ee);
    };

    // Outputs an AST traversed in a bottom-up fashion with multiple
    // sub-expressions, each assigned to a temp var.  The min/max_expr_size
    // vars in CompilerSettings control when and where expressions are
    // sub-divided. Within each sub-expression, a top-down visitor is used.
    class PrintVisitorBottomUp : public PrintVisitorBase {

    public:
        // os is used for printing intermediate results as needed.
        PrintVisitorBottomUp(ostream& os, PrintHelper& ph,
                             const VarMap* var_map = 0) :
            PrintVisitorBase(os, ph, var_map) {}

        // make a new top-down visitor with the same print helper.
        virtual PrintVisitorTopDown* new_print_visitor_top_down() {
            return new PrintVisitorTopDown(_os, _ph);
        }

        // Try some simple printing techniques.
        // Return string if printing is done.
        // Return empty string if more complex method should be used.
        virtual string try_simple_print(Expr* ex, bool force);

        // A var point.
        virtual string visit(VarPoint* gp);

        // An index.
        virtual string visit(IndexExpr* ie);

        // A constant.
        virtual string visit(ConstExpr* ce);

        // Code.
        virtual string visit(CodeExpr* ce);

        // Unary operators.
        virtual string visit(UnaryNumExpr* ue);
        virtual string visit(UnaryBoolExpr* ue);

        // Binary operators.
        virtual string visit(BinaryNumExpr* be);
        virtual string visit(BinaryBoolExpr* be);
        virtual string visit(BinaryNum2BoolExpr* be);

        // A commutative operator.
        virtual string visit(CommutativeExpr* ce);

        // A function call.
        virtual string visit(FuncExpr* fe);

        // An equality.
        virtual string visit(EqualsExpr* ee);
    };


    // Outputs a POV-Ray input file.
    class POVRayPrintVisitor : public ExprVisitor {
    protected:
        ostream& _os;
        vector<string> _colors;
        int _num_pts;

    public:
        POVRayPrintVisitor(ostream& os) : _os(os), _num_pts(0) {

            // Make a rainbow.
            _colors.push_back("Red");
            _colors.push_back("Orange");
            _colors.push_back("Yellow");
            _colors.push_back("Green");
            _colors.push_back("Blue");
            _colors.push_back("Violet");

            // NB: could also calculate the hue and use CHSL2RGB().
        }

        virtual int get_num_points() const {
            return _num_pts;
        }

        // Equals op.
        virtual string visit(EqualsExpr* ee);

        // A point.
        virtual string visit(VarPoint* gp);
    };

    // Outputs a full GraphViz input file.
    class DOTPrintVisitor : public ExprVisitor {
    protected:
        ostream& _os;
        set<string> _done;

        // Get label to use.
        // Return empty string if already done if once == true.
        virtual string get_label(Expr* ep, bool once = true) {
            string key = ep->make_quoted_str("\"");
            if (once) {
                if (_done.count(key))
                    return "";
                _done.insert(key);
            }
            return key;
        }
        virtual string get_label(expr_ptr ep, bool once = true) {
            return get_label(ep.get(), once);
        }

    public:
        DOTPrintVisitor(ostream& os) : _os(os) { }

        // A var read.
        virtual string visit(VarPoint* gp);

        // A constant.
        virtual string visit(ConstExpr* ce);

        // Some hand-written code.
        virtual string visit(CodeExpr* ce);

        // Generic numeric unary operators.
        virtual string visit(UnaryNumExpr* ue);

        // Generic numeric binary operators.
        virtual string visit(BinaryNumExpr* be);

        // A commutative operator.
        virtual string visit(CommutativeExpr* ce);

        // A function call.
        virtual string visit(FuncExpr* fe);

        // An equals operator.
        virtual string visit(EqualsExpr* ee);
    };

    // Outputs a simple GraphViz input file.
    class SimpleDOTPrintVisitor : public DOTPrintVisitor {
    protected:
        set<string> _vars_seen;

    public:
        SimpleDOTPrintVisitor(ostream& os) :
            DOTPrintVisitor(os) { }

        // A var read.
        virtual string visit(VarPoint* gp);

        // A constant.
        virtual string visit(ConstExpr* ce) { return ""; }

        // Some hand-written code.
        virtual string visit(CodeExpr* ce) { return ""; }

        // Generic numeric unary operators.
        virtual string visit(UnaryNumExpr* ue);

        // Generic numeric binary operators.
        virtual string visit(BinaryNumExpr* be);

        // A commutative operator.
        virtual string visit(CommutativeExpr* ce);

        // A function call.
        virtual string visit(FuncExpr* fe);

        // An equals operator.
        virtual string visit(EqualsExpr* ee);
    };

    // PrinterBase is the main class for defining how to print a stencil.
    // A PrinterBase uses one or more PrintHelpers and ExprVisitors to
    // do this.
    class PrinterBase {

    protected:
        StencilSolution& _stencil;
        const CompilerSettings& _settings;
        const Dimensions& _dims;
        Vars& _vars;
        EqBundles& _eq_bundles;

    public:
        PrinterBase(StencilSolution& stencil,
                    EqBundles& eq_bundles) :
            _stencil(stencil),
            _settings(stencil.get_settings()),
            _dims(stencil.get_dims()),
            _vars(stencil._get_vars()),
            _eq_bundles(eq_bundles)
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
        static string all_caps(string str) {
            transform(str.begin(), str.end(), str.begin(), ::toupper);
            return str;
        }
    };

    // Print out a stencil in human-readable form, for debug or documentation.
    class PseudoPrinter : public PrinterBase {
    protected:
        bool _long = false;

    public:
        PseudoPrinter(StencilSolution& stencil,
                      EqBundles& eq_bundles,
                      bool is_long) :
            PrinterBase(stencil, eq_bundles), _long(is_long) { }
        virtual ~PseudoPrinter() { }

        virtual void print(ostream& os);
    };

    // Print out a stencil in DOT-language form.
    class DOTPrinter : public PrinterBase {
    protected:
        bool _is_simple;

    public:
        DOTPrinter(StencilSolution& stencil, EqBundles& eq_bundles,
                   bool is_simple) :
            PrinterBase(stencil, eq_bundles),
            _is_simple(is_simple) { }
        virtual ~DOTPrinter() { }

        virtual void print(ostream& os);
    };

    // Print out a stencil in POVRay form.
    class POVRayPrinter : public PrinterBase {

    public:
        POVRayPrinter(StencilSolution& stencil, EqBundles& eq_bundles) :
            PrinterBase(stencil, eq_bundles) { }
        virtual ~POVRayPrinter() { }

        virtual void print(ostream& os);
    };

} // namespace yask.

