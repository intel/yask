/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

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
    // Prepend 'prefix' to name of var.
    // Print LHS of assignment to it.
    // 'ex' is used as key to save name of temp var and to write a comment.
    // If 'comment' is set, use it for the comment.
    // Return stream to continue w/RHS.
    ostream& PrintVisitorBase::make_next_temp_var(string& res, Expr* ex,
                                                  string prefix, string comment) {
        res = _ph.make_var_name(prefix);
        if (ex) {
            _temp_vars[ex] = res;
            if (comment.length() == 0)
                comment = ex->make_str();
        }
        if (comment.length())
            _os << endl << " // " << res << " = " << comment << "." << endl;
        _os << _ph.get_line_prefix() << _ph.get_var_type() << " " << res << " = ";
        return _os;
    }

    /////// Top-down

    // A var read.
    // Uses the PrintHelper to format.
    string PrintVisitorTopDown::visit(VarPoint* gp) {
        _num_common += _ph.get_num_common(gp);
        return _ph.read_from_point(_os, *gp);
    }

    // An index expression.
    string PrintVisitorTopDown::visit(IndexExpr* ie) {
        _num_common += _ph.get_num_common(ie);
        return ie->format(_var_map);
    }

    // A constant.
    // Uses the PrintHelper to format.
    string PrintVisitorTopDown::visit(ConstExpr* ce) {
        _num_common += _ph.get_num_common(ce);
        return _ph.add_const_expr(_os, ce->get_num_val());
    }

    // Some hand-written code.
    // Uses the PrintHelper to format.
    string PrintVisitorTopDown::visit(CodeExpr* ce) {
        _num_common += _ph.get_num_common(ce);
        return _ph.add_code_expr(_os, ce->get_code());
    }

    // Generic unary operators.
    string PrintVisitorTopDown::visit(UnaryNumExpr* ue) {
        _num_common += _ph.get_num_common(ue);
        return ue->get_op_str() + ue->_get_rhs()->accept(this);
    }
    string PrintVisitorTopDown::visit(UnaryBoolExpr* ue) {
        _num_common += _ph.get_num_common(ue);
        return ue->get_op_str() + ue->_get_rhs()->accept(this);
    }
    string PrintVisitorTopDown::visit(UnaryNum2BoolExpr* ue) {
        _num_common += _ph.get_num_common(ue);
        return ue->get_op_str() + ue->_get_rhs()->accept(this);
    }

    // Generic binary operators.
    string PrintVisitorTopDown::visit(BinaryNumExpr* be) {
        _num_common += _ph.get_num_common(be);
        return "(" + be->_get_lhs()->accept(this) +
            " " + be->get_op_str() + " " +
            be->_get_rhs()->accept(this) + ")";
    }
    string PrintVisitorTopDown::visit(BinaryBoolExpr* be) {
        _num_common += _ph.get_num_common(be);
        return "(" + be->_get_lhs()->accept(this) +
            " " + be->get_op_str() + " " +
            be->_get_rhs()->accept(this) + ")";
    }
    string PrintVisitorTopDown::visit(BinaryNum2BoolExpr* be) {
        _num_common += _ph.get_num_common(be);
        return "(" + be->_get_lhs()->accept(this) +
            " " + be->get_op_str() + " " +
            be->_get_rhs()->accept(this) + ")";
    }

    // A commutative operator.
    string PrintVisitorTopDown::visit(CommutativeExpr* ce) {
        _num_common += _ph.get_num_common(ce);
        string res = "(";
        auto& ops = ce->get_ops();
        int op_num = 0;
        for (auto ep : ops) {
            if (op_num > 0)
                res += " " + ce->get_op_str() + " ";
            res += ep->accept(this);
            op_num++;
        }
        return res + ")";
    }

    // A function call.
    string PrintVisitorTopDown::visit(FuncExpr* fe) {
        _num_common += _ph.get_num_common(fe);

        // Special case: increment common node count
        // for pairs.
        if (fe->get_pair())
            _num_common++;

        string res = _func_prefix + fe->get_op_str() + "(";
        auto& ops = fe->get_ops();
        int op_num = 0;
        for (auto ep : ops) {
            if (op_num > 0)
                res += ", ";
            res += ep->accept(this);
            op_num++;
        }
        return res + ")";
    }

    // An equals operator.
    string PrintVisitorTopDown::visit(EqualsExpr* ee) {

        // Get RHS.
        string rhs = ee->_get_rhs()->accept(this);

        // Write statement with embedded rhs.
        var_point_ptr gpp = ee->_get_lhs();
        _os << _ph.get_line_prefix() << _ph.write_to_point(_os, *gpp, rhs);

        // Null ptr => no condition.
        if (ee->_get_cond()) {
            string cond = ee->_get_cond()->accept(this);

            // pseudo-code format.
            _os << " IF_DOMAIN (" << cond << ")";
        }
        if (ee->_get_step_cond()) {
            string cond = ee->_get_step_cond()->accept(this);

            // pseudo-code format.
            _os << " IF_STEP (" << cond << ")";
        }
        _os << _ph.get_line_suffix();

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
    string PrintVisitorBottomUp::try_simple_print(Expr* ex, bool force) {
        string res;

        // How many nodes in ex?
        int expr_size = ex->_get_num_nodes();
        bool too_big = expr_size > get_settings()._max_expr_size;
        bool too_small = expr_size < get_settings()._min_expr_size;

        // Determine whether this expr has already been evaluated
        // and a variable holds its result.
        auto p = _temp_vars.find(ex);
        if (p != _temp_vars.end()) {

            // if so, just use the existing var.
            res = p->second;
        }

        // Consider top down if forcing or expr <= max_expr_size.
        else if (force || !too_big) {

            // use a top-down printer to render the expr.
            PrintVisitorTopDown* top_down = new_print_visitor_top_down();
            string td_res = ex->accept(top_down);

            // were there any common subexprs found?
            int num_common = top_down->get_num_common();

            // if no common subexprs, use the top-down expression.
            if (num_common == 0)
                res = td_res;

            // if common subexprs exist, and top-down is forced, use the
            // top-down expression regardless.  If node is big enough for
            // sharing, also assign the result to a temp var so it can be used
            // later.
            else if (force) {
                if (too_small)
                    res = td_res;
                else
                    make_next_temp_var(res, ex, "expr", "") << td_res << _ph.get_line_suffix();
            }

            // otherwise, there are common subexprs, and top-down is not forced,
            // so don't do top-down.

            delete top_down;
        }

        if (force) assert(res.length());
        return res;
    }

    // A var point: just set expr.
    string PrintVisitorBottomUp::visit(VarPoint* gp) {
        return try_simple_print(gp, true);
    }

    // A var index.
    string PrintVisitorBottomUp::visit(IndexExpr* ie) {
        return try_simple_print(ie, true);
    }

    // A constant: just set expr.
    string PrintVisitorBottomUp::visit(ConstExpr* ce) {
        return try_simple_print(ce, true);
    }

    // Code: just set expr.
    string PrintVisitorBottomUp::visit(CodeExpr* ce) {
        return try_simple_print(ce, true);
    }

    // A numerical unary operator.
    string PrintVisitorBottomUp::visit(UnaryNumExpr* ue) {

        // Try top-down on whole expression.
        // Example: '-a' creates no immediate output,
        // and '-(a)' is returned.
        string res = try_simple_print(ue, false);
        if (res.length())
            return res;

        // Expand the RHS, then apply operator to result.
        // Example: '-(a * b)' might output the following:
        // temp1 = a * b;
        // temp2 = -temp1;
        // with 'temp2' returned.
        string rhs = ue->_get_rhs()->accept(this);
        make_next_temp_var(res, ue, "expr", "") << ue->get_op_str() << rhs << _ph.get_line_suffix();
        return res;
    }

    // A numerical binary operator.
    string PrintVisitorBottomUp::visit(BinaryNumExpr* be) {

        // Try top-down on whole expression.
        // Example: 'a/b' creates no immediate output,
        // and 'a/b' is returned.
        string res = try_simple_print(be, false);
        if (res.length())
            return res;

        // Expand both sides, then apply operator to result.
        // Example: '(a * b) / (c * d)' might output the following:
        // expr1 = a * b;
        // expr2 = b * c;
        // expr3 = expr1 / expr2;
        // with 'expr3' returned.
        string lhs = be->_get_lhs()->accept(this);
        string rhs = be->_get_rhs()->accept(this);
        make_next_temp_var(res, be, "expr", "") <<
            lhs << ' ' << be->get_op_str() << ' ' << rhs << _ph.get_line_suffix();
        return res;
    }

    // Boolean unary and binary operators.
    // For now, don't try to use bottom-up for these.
    // TODO: investigate whether there is any potential
    // benefit in doing this.
    string PrintVisitorBottomUp::visit(UnaryBoolExpr* ue) {
        return try_simple_print(ue, true);
    }
    string PrintVisitorBottomUp::visit(BinaryBoolExpr* be) {
        return try_simple_print(be, true);
    }
    string PrintVisitorBottomUp::visit(BinaryNum2BoolExpr* be) {
        return try_simple_print(be, true);
    }

    // Function call.
    string PrintVisitorBottomUp::visit(FuncExpr* fe) {
        string res;

        // If this is a paired function, handle it specially.
        auto* paired = fe->get_pair();
        if (paired) {

            // Do we already have this result?
            if (_temp_vars.count(fe)) {
                
                // Just use existing result.
                res = _temp_vars.at(fe);
            }

            // No result yet.
            else {

                _os << endl << " // Combining " << fe->get_op_str() << " and " << paired->get_op_str() <<
                    "...\n";

                // First, eval all the args.
                string args;
                auto& ops = fe->get_ops();
                for (auto ep : ops)
                    args += ", " + ep->accept(this);

                // Make 2 temp vars.
                string res2;
                make_next_temp_var(res, fe, "arg0", "") << "0" << _ph.get_line_suffix();
                make_next_temp_var(res2, paired, "arg1", "") << "0" << _ph.get_line_suffix();

                // Call function to set both.
                _os << _ph.get_line_prefix() << 
                    _func_prefix << fe->get_op_str() << "_and_" << paired->get_op_str() << 
                    "(" << res << ", " << res2 << args <<
                    ")" << _ph.get_line_suffix();
            }
        }

        // If not paired, handle normally.
        else {
            res = try_simple_print(fe, false);
            if (res.length())
                return res;

            string args;
            auto& ops = fe->get_ops();
            for (auto ep : ops) {
                if (args.length())
                    args += ", ";
                args += ep->accept(this);
            }
            make_next_temp_var(res, fe, "res", "") << _func_prefix << fe->get_op_str() <<
                "(" << args << ")" << _ph.get_line_suffix();
        }
        return res;
    }

    // A commutative operator.
    string PrintVisitorBottomUp::visit(CommutativeExpr* ce) {

        // Try top-down on whole expression.
        // Example: 'a*b' creates no immediate output,
        // and 'a*b' is returned.
        string res = try_simple_print(ce, false);
        if (res.length())
            return res;

        // Make separate assignment for N-1 operands.
        // Example: 'a + b + c + d' might output the following:
        // temp1 = a + b;
        // temp2 = temp1 + c;
        // temp3 = temp2 = d;
        // with 'temp3' returned.
        auto& ops = ce->get_ops();
        assert(ops.size() > 1);
        string lhs, ex_str;
        int op_num = 0;
        for (auto ep : ops) {
            op_num++;

            // eval the operand.
            string op_str = ep->accept(this);

            // first operand; just save as LHS for next iteration.
            if (op_num == 1) {
                lhs = op_str;
                ex_str = ep->make_str();
            }

            // subsequent operands.
            // makes separate assignment for each one.
            // result is kept as LHS of next one.
            else {

                // Use whole expression only for the last step.
                Expr* ex = (op_num == (int)ops.size()) ? ce : NULL;

                // Add RHS to partial-result comment.
                ex_str += ' ' + ce->get_op_str() + ' ' + ep->make_str();

                // Output this step.
                string tmp;
                make_next_temp_var(tmp, ex, "expr", ex_str) << lhs << ' ' << ce->get_op_str() << ' ' <<
                    op_str << _ph.get_line_suffix();
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
        Expr* rp = ee->_get_rhs().get();
        string rhs = rp->accept(this);

        // Assign RHS to a temp var.
        string tmp;
        make_next_temp_var(tmp, rp, "expr", "") << rhs << _ph.get_line_suffix();

        // Comment about update.
        var_point_ptr gpp = ee->_get_lhs();
        _os << "\n // Update value at " << gpp->make_str();

        // Comment about condition.
        // Null ptr => no condition.
        if (ee->_get_cond()) {
            string cond = ee->_get_cond()->make_str();
            _os << " IF_DOMAIN (" << cond << ")";
        }
        if (ee->_get_step_cond()) {
            string cond = ee->_get_step_cond()->make_str();
            _os << " IF_STEP (" << cond << ")";
        }
        _os << ".\n";

        // Write RHS expr to var.
        _os << _ph.get_line_prefix() << _ph.write_to_point(_os, *gpp, tmp) << _ph.get_line_suffix();

        return "";              // EQUALS doesn't return a value.
    }

    ///////// POVRay.

    // Only want to visit the RHS of an equality.
    string POVRayPrintVisitor::visit(EqualsExpr* ee) {
        return ee->_get_rhs()->accept(this);
    }

    // A point: output it.
    string POVRayPrintVisitor::visit(VarPoint* gp) {
        _num_pts++;

        // Pick a color based on its distance.
        size_t ci = gp->get_arg_offsets().max();
        ci %= _colors.size();

        _os << "point(" + _colors[ci] + ", " << gp->get_arg_offsets().make_val_str() << ")" << endl;
        return "";
    }

    ////// DOT-language.

    // A var access.
    string DOTPrintVisitor::visit(VarPoint* gp) {
        string label = get_label(gp);
        if (label.size())
            _os << label << " [ shape = box ];" << endl;
        return "";
    }

    // A constant.
    // TODO: don't share node.
    string DOTPrintVisitor::visit(ConstExpr* ce) {
        string label = get_label(ce);
        if (label.size())
            _os << label << endl;
        return "";
    }

    // Some hand-written code.
    string DOTPrintVisitor::visit(CodeExpr* ce) {
        string label = get_label(ce);
        if (label.size())
            _os << label << endl;
        return "";
    }

    // Generic numeric unary operators.
    string DOTPrintVisitor::visit(UnaryNumExpr* ue) {
        string label = get_label(ue);
        if (label.size()) {
            _os << label << " [ label = \"" << ue->get_op_str() << "\" ];" << endl;
            _os << get_label(ue, false) << " -> " << get_label(ue->_get_rhs(), false) << ";" << endl;
            ue->_get_rhs()->accept(this);
        }
        return "";
    }

    // Generic numeric binary operators.
    string DOTPrintVisitor::visit(BinaryNumExpr* be) {
        string label = get_label(be);
        if (label.size()) {
            _os << label << " [ label = \"" << be->get_op_str() << "\" ];" << endl;
            _os << get_label(be, false) << " -> " << get_label(be->_get_lhs(), false) << ";" << endl <<
                get_label(be, false) << " -> " << get_label(be->_get_rhs(), false) << ";" << endl;
            be->_get_lhs()->accept(this);
            be->_get_rhs()->accept(this);
        }
        return "";
    }

    // A commutative operator.
    string DOTPrintVisitor::visit(CommutativeExpr* ce) {
        string label = get_label(ce);
        if (label.size()) {
            _os << label << " [ label = \"" << ce->get_op_str() << "\" ];" << endl;
            for (auto ep : ce->get_ops()) {
                _os << get_label(ce, false) << " -> " << get_label(ep, false) << ";" << endl;
                ep->accept(this);
            }
        }
        return "";
    }

    // A function call.
    string DOTPrintVisitor::visit(FuncExpr* fe) {
        string label = get_label(fe);
        if (label.size()) {
            _os << label << " [ label = \"" << fe->get_op_str() << "\" ];" << endl;
            for (auto ep : fe->get_ops()) {
                _os << get_label(fe, false) << " -> " << get_label(ep, false) << ";" << endl;
                ep->accept(this);
            }
        }
        return "";
    }

    // An equals operator.
    string DOTPrintVisitor::visit(EqualsExpr* ee) {
        string label = get_label(ee);
        if (label.size()) {
            _os << label << " [ label = \"EQUALS\" ];" << endl;
            _os << get_label(ee, false) << " -> " << get_label(ee->_get_lhs(), false)  << ";" << endl <<
                get_label(ee, false) << " -> " << get_label(ee->_get_rhs(), false) << ";" << endl;
            ee->_get_lhs()->accept(this);
            ee->_get_rhs()->accept(this);

            // Null ptr => no condition.
            if (ee->_get_cond()) {
                _os << get_label(ee, false) << " -> " << get_label(ee->_get_cond(), false)  << ";" << endl;
                ee->_get_cond()->accept(this);
            }
        }
        return "";
    }

    // A var access.
    string SimpleDOTPrintVisitor::visit(VarPoint* gp) {
        string label = get_label(gp);
        if (label.size()) {
            _os << label << " [ shape = box ];" << endl;
            _vars_seen.insert(label);
        }
        return "";
    }

    // Generic numeric unary operators.
    string SimpleDOTPrintVisitor::visit(UnaryNumExpr* ue) {
        ue->_get_rhs()->accept(this);
        return "";
    }

    // Generic numeric binary operators.
    string SimpleDOTPrintVisitor::visit(BinaryNumExpr* be) {
        be->_get_lhs()->accept(this);
        be->_get_rhs()->accept(this);
        return "";
    }

    // A commutative operator.
    string SimpleDOTPrintVisitor::visit(CommutativeExpr* ce) {
        for (auto& ep : ce->get_ops())
            ep->accept(this);
        return "";
    }

    // A function call.
    string SimpleDOTPrintVisitor::visit(FuncExpr* fe) {
        for (auto& ep : fe->get_ops())
            ep->accept(this);
        return "";
    }

    // An equals operator.
    string SimpleDOTPrintVisitor::visit(EqualsExpr* ee) {

        // LHS is source.
        ee->_get_lhs()->accept(this);
        string label = get_label(ee, false);
        for (auto g : _vars_seen)
            label = g;              // really should only be one.
        _vars_seen.clear();

        // RHS nodes are target.
        ee->_get_rhs()->accept(this);
        for (auto g : _vars_seen)
            _os << label << " -> " << g  << ";" << endl;
        _vars_seen.clear();

        // Ignoring conditions.
        return "";
    }

    ////////////// Printers ///////////////

    /////// Pseudo-code.

    // Print out a stencil in human-readable form, for debug or documentation.
    void PseudoPrinter::print(ostream& os) {

        os << "Stencil '" << _stencil._get_name() << "' pseudo-code:" << endl;

        // Loop through all eq_bundles.
        for (auto& eq : _eq_bundles.get_all()) {

            string eg_name = eq->_get_name();
            os << endl << " ////// Equation bundle '" << eg_name <<
                "' //////" << endl;

            CounterVisitor cv;
            eq->visit_eqs(&cv);
            PrintHelper ph(_settings, _dims, &cv, "real", " ", ".\n");

            if (eq->cond.get()) {
                string cond_str = eq->cond->make_str();
                os << endl << " // Valid under the following domain condition:" << endl <<
                    ph.get_line_prefix() << "IF_DOMAIN " << cond_str << ph.get_line_suffix();
            }
            if (eq->step_cond.get()) {
                string cond_str = eq->step_cond->make_str();
                os << endl << " // Valid under the following step condition:" << endl <<
                    ph.get_line_prefix() << "IF_STEP " << cond_str << ph.get_line_suffix();
            }

            if (!_long) {
                PrintVisitorTopDown pv1(os, ph);
                eq->visit_eqs(&pv1);
            } else {
                PrintVisitorBottomUp pv2(os, ph);
                eq->visit_eqs(&pv2);
            }
        }
    }

    ///// DOT language.

    // Print out a stencil in DOT form
    void DOTPrinter::print(ostream& os) {

        DOTPrintVisitor* pv = _is_simple ?
            new SimpleDOTPrintVisitor(os) :
            new DOTPrintVisitor(os);

        os << "digraph \"Stencil " << _stencil._get_name() << "\" {\n"
            "rankdir=LR; ranksep=1.5;\n";

        // Loop through all eq_bundles.
        for (auto& eq : _eq_bundles.get_all()) {
            os << "subgraph \"Equation-bundle " << eq->_get_name() << "\" {" << endl;
            eq->visit_eqs(pv);
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

        // Loop through all eq_bundles.
        for (auto& eq : _eq_bundles.get_all()) {

            // TODO: separate mutiple vars.
            POVRayPrintVisitor pv(os);
            eq->visit_eqs(&pv);
            os << " // " << pv.get_num_points() << " stencil points" << endl;
        }
    }

} // namespace yask.
