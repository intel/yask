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

///////// Stencil AST Expressions. ////////////

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {

    // var_point APIs.
    yc_var* VarPoint::get_var() {
        return _var;
    }
    const yc_var* VarPoint::get_var() const {
        return _var;
    }

    //node_factory API methods.
    yc_index_node_ptr
    yc_node_factory::new_step_index(const std::string& name) const {
        return make_shared<IndexExpr>(name, STEP_INDEX);
    }
    yc_index_node_ptr
    yc_node_factory::new_domain_index(const std::string& name) const {
        return make_shared<IndexExpr>(name, DOMAIN_INDEX);
    }
    yc_index_node_ptr
    yc_node_factory::new_misc_index(const std::string& name) const {
        return make_shared<IndexExpr>(name, MISC_INDEX);
    }

    yc_equation_node_ptr
    yc_node_factory::new_equation_node(yc_var_point_node_ptr lhs,
                                       yc_number_node_ptr rexpr,
                                       yc_bool_node_ptr cond) const {
        if (!lhs)
            THROW_YASK_EXCEPTION("empty LHS of equation");
        auto gpp = dynamic_pointer_cast<VarPoint>(lhs);
        assert(gpp);
        if (!rexpr)
            THROW_YASK_EXCEPTION("empty RHS of " +
                                 gpp->make_quoted_str() + " equation");
        auto rhs = dynamic_pointer_cast<NumExpr>(rexpr);
        assert(rhs);

        // Get to list of equations in soln indirectly thru var.
        Var* gp = gpp->_get_var();
        assert(gp);
        auto* soln = gp->_get_soln();
        assert(soln);
        auto& eqs = soln->get_eqs();
        auto& settings = soln->get_settings();

        // Make expression node.
        auto expr = make_shared<EqualsExpr>(gpp, rhs);
        expr->set_cond(cond);

        // Save the expression in list of equations.
        eqs.add_item(expr);

        return expr;
    }
    yc_number_node_ptr
    yc_node_factory::new_const_number_node(double val) const {
        return make_shared<ConstExpr>(val);
    }
    yc_number_node_ptr
    yc_node_factory::new_const_number_node(idx_t val) const {
        return make_shared<ConstExpr>(val);
    }
    yc_number_node_ptr
    yc_node_factory::new_negate_node(yc_number_node_ptr rhs) const {
        auto p = dynamic_pointer_cast<NumExpr>(rhs);
        assert(p);
        return make_shared<NegExpr>(p);
    }
    yc_number_node_ptr
    yc_node_factory::new_add_node(yc_number_node_ptr lhs,
                                  yc_number_node_ptr rhs) const {
        if (!lhs)
            return rhs;
        if (!rhs)
            return lhs;
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);

        // Add to commutative operand list.
        auto ex = make_shared<AddExpr>();
        ex->merge_expr(lp);
        ex->merge_expr(rp);
        return ex;
    }
    yc_number_node_ptr
    yc_node_factory::new_multiply_node(yc_number_node_ptr lhs,
                                       yc_number_node_ptr rhs) const {
        if (!lhs)
            return rhs;
        if (!rhs)
            return lhs;
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);

        // Add to commutative operand list.
        auto ex = make_shared<MultExpr>();
        ex->merge_expr(lp);
        ex->merge_expr(rp);
        return ex;
    }
    yc_number_node_ptr
    yc_node_factory::new_subtract_node(yc_number_node_ptr lhs,
                                       yc_number_node_ptr rhs) const {
        yc_node_factory nfac;
        if (!lhs)
            return nfac.new_negate_node(rhs);
        if (!rhs)
            return lhs;
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<SubExpr>(lp, rp);
    }
    yc_number_node_ptr
    yc_node_factory::new_divide_node(yc_number_node_ptr lhs,
                                     yc_number_node_ptr rhs) const {
        yc_node_factory nfac;
        if (!lhs)
            return nfac.new_divide_node(nfac.new_const_number_node(1.0), rhs);
        if (!rhs)
            return lhs;
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<DivExpr>(lp, rp);
    }
    yc_number_node_ptr
    yc_node_factory::new_mod_node(yc_number_node_ptr lhs,
                                  yc_number_node_ptr rhs) const {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<ModExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_not_node(yc_bool_node_ptr rhs) const {
        auto p = dynamic_pointer_cast<BoolExpr>(rhs);
        assert(p);
        return make_shared<NotExpr>(p);
    }
    yc_bool_node_ptr
    yc_node_factory::new_and_node(yc_bool_node_ptr lhs,
                                      yc_bool_node_ptr rhs) const {
        auto lp = dynamic_pointer_cast<BoolExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<BoolExpr>(rhs);
        assert(rp);
        return make_shared<AndExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_or_node(yc_bool_node_ptr lhs,
                                 yc_bool_node_ptr rhs) const {
        auto lp = dynamic_pointer_cast<BoolExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<BoolExpr>(rhs);
        assert(rp);
        return make_shared<OrExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_equals_node(yc_number_node_ptr lhs,
                                      yc_number_node_ptr rhs) const {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<IsEqualExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_not_equals_node(yc_number_node_ptr lhs,
                                         yc_number_node_ptr rhs) const {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<NotEqualExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_less_than_node(yc_number_node_ptr lhs,
                                      yc_number_node_ptr rhs) const {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<IsLessExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_greater_than_node(yc_number_node_ptr lhs,
                                      yc_number_node_ptr rhs) const {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<IsGreaterExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_not_less_than_node(yc_number_node_ptr lhs,
                                      yc_number_node_ptr rhs) const {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<NotLessExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_not_greater_than_node(yc_number_node_ptr lhs,
                                               yc_number_node_ptr rhs) const {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<NotGreaterExpr>(lp, rp);
    }
    yc_index_node_ptr
    yc_node_factory::new_first_domain_index(yc_index_node_ptr idx) const {
        auto p = dynamic_pointer_cast<IndexExpr>(idx);
        if (!p)
            THROW_YASK_EXCEPTION("new_first_domain_index() called without index-node argument");
        if (p->get_type() != DOMAIN_INDEX)
            THROW_YASK_EXCEPTION("new_first_domain_index() called without domain-index-node argument");
        return make_shared<IndexExpr>(p->_get_name(), FIRST_INDEX);
    }
    yc_index_node_ptr
    yc_node_factory::new_last_domain_index(yc_index_node_ptr idx) const {
        auto p = dynamic_pointer_cast<IndexExpr>(idx);
        if (!p)
            THROW_YASK_EXCEPTION("new_last_domain_index() called without index-node argument");
        if (p->get_type() != DOMAIN_INDEX)
            THROW_YASK_EXCEPTION("new_last_domain_index() called without domain-index-node argument");
        return make_shared<IndexExpr>(p->_get_name(), LAST_INDEX);
    }

    yc_number_node_ptr yc_number_const_arg::_convert_const(double val) const {
        yc_node_factory nfac;
        return nfac.new_const_number_node(val);
    }
    yc_number_node_ptr yc_number_any_arg::_convert_const(double val) const {
        yc_node_factory nfac;
        return nfac.new_const_number_node(val);
    }
    
    // Unary.
    yc_number_node_ptr operator-(yc_number_ptr_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_negate_node(rhs);
    }

    // Binary.
    yc_number_node_ptr operator+(yc_number_ptr_arg lhs, yc_number_ptr_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_add_node(lhs, rhs);
    }
    yc_number_node_ptr operator+(yc_number_const_arg lhs, yc_number_ptr_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_add_node(lhs, rhs);
    }
    yc_number_node_ptr operator+(yc_number_ptr_arg lhs, yc_number_const_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_add_node(lhs, rhs);
    }
    yc_number_node_ptr operator/(yc_number_ptr_arg lhs, yc_number_ptr_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_divide_node(lhs, rhs);
    }
    yc_number_node_ptr operator/(yc_number_const_arg lhs, yc_number_ptr_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_divide_node(lhs, rhs);
    }
    yc_number_node_ptr operator/(yc_number_ptr_arg lhs, yc_number_const_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_divide_node(lhs, rhs);
    }
    yc_number_node_ptr operator%(yc_number_ptr_arg lhs, yc_number_ptr_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_mod_node(lhs, rhs);
    }
    yc_number_node_ptr operator%(yc_number_const_arg lhs, yc_number_ptr_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_mod_node(lhs, rhs);
    }
    yc_number_node_ptr operator%(yc_number_ptr_arg lhs, yc_number_const_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_mod_node(lhs, rhs);
    }
    yc_number_node_ptr operator*(yc_number_ptr_arg lhs, yc_number_ptr_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_multiply_node(lhs, rhs);
    }
    yc_number_node_ptr operator*(yc_number_const_arg lhs, yc_number_ptr_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_multiply_node(lhs, rhs);
    }
    yc_number_node_ptr operator*(yc_number_ptr_arg lhs, yc_number_const_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_multiply_node(lhs, rhs);
    }
    yc_number_node_ptr operator-(yc_number_ptr_arg lhs, yc_number_ptr_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_subtract_node(lhs, rhs);
    }
    yc_number_node_ptr operator-(yc_number_const_arg lhs, yc_number_ptr_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_subtract_node(lhs, rhs);
    }
    yc_number_node_ptr operator-(yc_number_ptr_arg lhs, yc_number_const_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_subtract_node(lhs, rhs);
    }

    // Modifiers.
    void operator+=(yc_number_node_ptr& lhs, yc_number_node_ptr rhs) {
        lhs = lhs + rhs;
    }
    void operator+=(yc_number_node_ptr& lhs, yc_number_const_arg rhs) {
        lhs = lhs + rhs;
    }
    void operator-=(yc_number_node_ptr& lhs, yc_number_node_ptr rhs) {
        lhs = lhs - rhs;
    }
    void operator-=(yc_number_node_ptr& lhs, yc_number_const_arg rhs) {
        lhs = lhs - rhs;
    }
    void operator*=(yc_number_node_ptr& lhs, yc_number_node_ptr rhs) {
        lhs = lhs * rhs;
    }
    void operator*=(yc_number_node_ptr& lhs, yc_number_const_arg rhs) {
        lhs = lhs * rhs;
    }
    void operator/=(yc_number_node_ptr& lhs, yc_number_node_ptr rhs) {
        lhs = lhs / rhs;
    }
    void operator/=(yc_number_node_ptr& lhs, yc_number_const_arg rhs) {
        lhs = lhs / rhs;
    }

    // Boolean unary.
    yc_bool_node_ptr operator!(yc_bool_node_ptr rhs) {
        yc_node_factory nfac;
        return nfac.new_not_node(rhs);
    }

    // Boolean binary.
    yc_bool_node_ptr operator||(yc_bool_node_ptr lhs, yc_bool_node_ptr rhs) {
        yc_node_factory nfac;
        return nfac.new_or_node(lhs, rhs);
    }
    yc_bool_node_ptr operator&&(yc_bool_node_ptr lhs, yc_bool_node_ptr rhs) {
        yc_node_factory nfac;
        return nfac.new_and_node(lhs, rhs);
    }

    // Compare 2 expr pointers and return whether the expressions are
    // equivalent.
    // TODO: Be much smarter about matching symbolically-equivalent exprs.
    bool are_exprs_same(const Expr* e1, const Expr* e2) {

        // Handle null pointers.
        if (e1 == NULL && e2 == NULL)
            return true;
        if (e1 == NULL || e2 == NULL)
            return false;

        // Neither are null, so compare contents.
        return e1->is_same(e2);
    }

    // Function calls.
#define FUNC_EXPR(fn) \
    yc_number_node_ptr fn(const yc_number_node_ptr rhs) {               \
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);                   \
        assert(rp);                                                     \
        return make_shared<FuncExpr>(#fn, std::initializer_list< const num_expr_ptr >{ rp });   \
    }
    FUNC_EXPR(sqrt)
    FUNC_EXPR(cbrt)
    FUNC_EXPR(fabs)
    FUNC_EXPR(erf)
    FUNC_EXPR(exp)
    FUNC_EXPR(log)
    FUNC_EXPR(atan)
    FUNC_EXPR(sin)
    FUNC_EXPR(cos)
#undef FUNC_EXPR
#define FUNC_EXPR(fn) \
    yc_number_node_ptr fn(const yc_number_node_ptr arg1, const yc_number_node_ptr arg2) {   \
        auto p1 = dynamic_pointer_cast<NumExpr>(arg1);                  \
        assert(p1);                                                     \
        auto p2 = dynamic_pointer_cast<NumExpr>(arg2);                  \
        assert(p2);                                                     \
        return make_shared<FuncExpr>(#fn, std::initializer_list< const num_expr_ptr >{ p1, p2 }); \
    }                                                                   \
    yc_number_node_ptr fn(const yc_number_node_ptr arg1, double arg2) { \
        yc_node_factory nfac;                                           \
        return fn(arg1, nfac.new_const_number_node(arg2));              \
    }                                                                   \
    yc_number_node_ptr fn(double arg1, const yc_number_node_ptr arg2) { \
        yc_node_factory nfac;                                           \
        return fn(nfac.new_const_number_node(arg1), arg2);              \
    }
    FUNC_EXPR(pow)
#undef FUNC_EXPR

    // Define a conditional.
    yc_equation_node_ptr operator IF_DOMAIN(yc_equation_node_ptr expr,
                                            const yc_bool_node_ptr cond) {
        auto ep = dynamic_pointer_cast<EqualsExpr>(expr);
        assert(ep);
        auto cp = dynamic_pointer_cast<BoolExpr>(cond);
        assert(cp);

        // Add cond to expr.
        ep->_set_cond(cp);
        return ep;
    }
    yc_equation_node_ptr operator IF_STEP(yc_equation_node_ptr expr,
                                          const yc_bool_node_ptr cond) {
        auto ep = dynamic_pointer_cast<EqualsExpr>(expr);
        assert(ep);
        auto cp = dynamic_pointer_cast<BoolExpr>(cond);
        assert(cp);

        // Add cond to expr.
        ep->_set_step_cond(cp);
        return ep;
    }

    // Define the value of a var point.
    // Add this equation to the list of eqs for this stencil.
    yc_equation_node_ptr operator EQUALS(yc_var_point_node_ptr lhs,
                                         const yc_number_any_arg rhs) {
        yc_node_factory nfac;
        return nfac.new_equation_node(lhs, rhs);
    }

    // Visitor acceptors.
    string ConstExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    string CodeExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    template<>
    string UnaryNumExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    template<>
    string UnaryBoolExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    template<>
    string UnaryNum2BoolExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    template<>
    string BinaryNumExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    template<>
    string BinaryBoolExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    template<>
    string BinaryNum2BoolExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    string CommutativeExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    string FuncExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    string VarPoint::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    string EqualsExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    string IndexExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }

    // Commutative methods.
    bool CommutativeExpr::is_same(const Expr* other) const {
        auto p = dynamic_cast<const CommutativeExpr*>(other);
        if (!p || _op_str != p->_op_str)
            return false;
        if (_ops.size() != p->_ops.size())
            return false;

        // Operands must be the same, but not necessarily in same order.  This
        // tracks the indices in 'other' that have already been matched.
        // Use a set to correctly handle repeating ops, e.g., a + b + a.
        set<size_t> matches;

        // Loop through this set of ops.
        for (auto op : _ops) {

            // Loop through other set of ops, looking for match.
            bool found = false;
            for (size_t i = 0; i < p->_ops.size(); i++) {
                auto oop = p->_ops[i];

                // check unless already matched.
                if (matches.count(i) == 0 && op->is_same(oop.get())) {
                    matches.insert(i);
                    found = true;
                    break;
                }
            }
            if (!found)
                return false;
        }

        // Do all match?
        return matches.size() == _ops.size();
    }

    // FuncExpr methods.
    bool FuncExpr::is_same(const Expr* other) const {
        auto p = dynamic_cast<const FuncExpr*>(other);
        if (!p || _op_str != p->_op_str)
            return false;
        if (_ops.size() != p->_ops.size())
            return false;
        for (size_t i = 0; i < p->_ops.size(); i++) {
            if (!_ops[i]->is_same(p->_ops[i]))
                return false;
        }
        return true;
    }
    bool FuncExpr::make_pair(Expr* other) {
        auto p = dynamic_cast<FuncExpr*>(other);

        // Must be another FuncExpr w/all the same operands.
        if (!p || _ops.size() != p->_ops.size())
            return false;
        for (size_t i = 0; i < p->_ops.size(); i++) {
            if (!_ops[i]->is_same(p->_ops[i]))
                return false;
        }

        // Possible pairs.
        // TODO: make list of other options.
        string f1 = "sin";
        string f2 = "cos";
        if ((_op_str == f1 && p->_op_str == f2) ||
            (_op_str == f2 && p->_op_str == f1)) {
            _paired = p;
            p->_paired = this;
            return true;
        }
        return false;
    }

    // Is this expr a simple offset?
    bool IndexExpr::is_offset_from(string dim, int& offset) {

        // An index expr is an offset if it's a step or domain dim and the
        // dims are the same.
        if ((_type == DOMAIN_INDEX || _type == STEP_INDEX) && _dim_name == dim) {
            offset = 0;
            return true;
        }
        return false;
    }
    bool DivExpr::is_offset_from(string dim, int& offset) {

        // Could allow 'dim / 1', but seems silly.
        return false;
    }
    bool ModExpr::is_offset_from(string dim, int& offset) {
        return false;
    }
    bool MultExpr::is_offset_from(string dim, int& offset) {

        // Could allow 'dim * 1', but seems silly.
        return false;
    }
    bool SubExpr::is_offset_from(string dim, int& offset) {

        // Is this of the form 'dim - offset'?
        // Allow any similar form, e.g., '(dim + 4) - 2',
        // Could allow '0 - dim', but seems silly.
        int tmp = 0;
        if (_lhs->is_offset_from(dim, tmp) &&
            _rhs->is_const_val()) {
            offset = tmp - _rhs->get_int_val();
            return true;
        }
        return false;
    }
    bool AddExpr::is_offset_from(string dim, int& offset) {

        // Is this of the form 'dim + offset'?
        // Allow any similar form, e.g., '-5 + dim + 2',
        // (dim + 3) + 7.
        int sum = 0;
        int num_dims = 0;
        int tmp = 0;
        for (auto op : _ops) {

            // Is this operand 'dim' or an offset from 'dim'?
            if (op->is_offset_from(dim, tmp))
                num_dims++;

            // Is this operand a const int?
            else if (op->is_const_val())
                sum += op->get_int_val();

            // Anything else isn't allowed.
            else
                return false;
        }
        // Must be exactly one 'dim' in the expr.
        // Don't allow silly forms like 'dim - dim + dim + 2'.
        if (num_dims == 1) {
            offset = tmp + sum;
            return true;
        }
        return false;
    }

    // Make a readable string from an expression.
    string Expr::make_str(const VarMap* var_map) const {

        // Use a print visitor to make a string.
        ostringstream oss;
        CompilerSettings _dummy_settings;
        Dimensions _dummy_dims;
        PrintHelper ph(_dummy_settings, _dummy_dims, NULL, "", "", ""); // default helper.
        CompilerSettings settings; // default settings.
        PrintVisitorTopDown pv(oss, ph, var_map);
        string res = accept(&pv);

        // Return anything written to the stream
        // concatenated with anything left in the
        // PrintVisitor.
        return oss.str() + res;
    }

    // Return number of nodes.
    int Expr::_get_num_nodes() const {

        // Use a counter visitor.
        CounterVisitor cv;
        accept(&cv);

        return cv._get_num_nodes();
    }

    // Const version of accept.
    string Expr::accept(ExprVisitor* ev) const {
        return const_cast<Expr*>(this)->accept(ev);
    }

} // namespace yask.
