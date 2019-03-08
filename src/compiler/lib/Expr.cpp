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

///////// Stencil AST Expressions. ////////////

// TODO: break this up into several smaller files.

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Parse.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {

    // grid_point APIs.
    yc_grid* GridPoint::get_grid() {
        return _grid;
    }

    //node_factory API methods.
    yc_index_node_ptr
    yc_node_factory::new_step_index(const std::string& name) {
        return make_shared<IndexExpr>(name, STEP_INDEX);
    }
    yc_index_node_ptr
    yc_node_factory::new_domain_index(const std::string& name) {
        return make_shared<IndexExpr>(name, DOMAIN_INDEX);
    }
    yc_index_node_ptr
    yc_node_factory::new_misc_index(const std::string& name) {
        return make_shared<IndexExpr>(name, MISC_INDEX);
    }

    yc_equation_node_ptr
    yc_node_factory::new_equation_node(yc_grid_point_node_ptr lhs,
                                       yc_number_node_ptr rhs,
                                       yc_bool_node_ptr cond) {
        auto lp = dynamic_pointer_cast<GridPoint>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        auto ep = operator EQUALS_OPER(lp, rp);
        ep->set_cond(cond);
        return ep;
    }
    yc_number_node_ptr
    yc_node_factory::new_const_number_node(double val) {
        return make_shared<ConstExpr>(val);
    }
    yc_number_node_ptr
    yc_node_factory::new_const_number_node(idx_t val) {
        return make_shared<ConstExpr>(val);
    }
    yc_number_node_ptr
    yc_node_factory::new_negate_node(yc_number_node_ptr rhs) {
        auto p = dynamic_pointer_cast<NumExpr>(rhs);
        assert(p);
        return make_shared<NegExpr>(p);
    }
    yc_number_node_ptr
    yc_node_factory::new_add_node(yc_number_node_ptr lhs,
                                  yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<AddExpr>(lp, rp);
    }
    yc_number_node_ptr
    yc_node_factory::new_multiply_node(yc_number_node_ptr lhs,
                                       yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<MultExpr>(lp, rp);
    }
    yc_number_node_ptr
    yc_node_factory::new_subtract_node(yc_number_node_ptr lhs,
                                       yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<SubExpr>(lp, rp);
    }
    yc_number_node_ptr
    yc_node_factory::new_divide_node(yc_number_node_ptr lhs,
                                     yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<DivExpr>(lp, rp);
    }
    yc_number_node_ptr
    yc_node_factory::new_mod_node(yc_number_node_ptr lhs,
                                  yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<ModExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_not_node(yc_bool_node_ptr rhs) {
        auto p = dynamic_pointer_cast<BoolExpr>(rhs);
        assert(p);
        return make_shared<NotExpr>(p);
    }
    yc_bool_node_ptr
    yc_node_factory::new_and_node(yc_bool_node_ptr lhs,
                                      yc_bool_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<BoolExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<BoolExpr>(rhs);
        assert(rp);
        return make_shared<AndExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_or_node(yc_bool_node_ptr lhs,
                                      yc_bool_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<BoolExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<BoolExpr>(rhs);
        assert(rp);
        return make_shared<OrExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_equals_node(yc_number_node_ptr lhs,
                                      yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<IsEqualExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_not_equals_node(yc_number_node_ptr lhs,
                                      yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<NotEqualExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_less_than_node(yc_number_node_ptr lhs,
                                      yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<IsLessExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_greater_than_node(yc_number_node_ptr lhs,
                                      yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<IsGreaterExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_not_less_than_node(yc_number_node_ptr lhs,
                                      yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<NotLessExpr>(lp, rp);
    }
    yc_bool_node_ptr
    yc_node_factory::new_not_greater_than_node(yc_number_node_ptr lhs,
                                               yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<NotGreaterExpr>(lp, rp);
    }
    yc_number_node_ptr
    yc_node_factory::new_first_domain_index(yc_index_node_ptr idx) {
        auto p = dynamic_pointer_cast<IndexExpr>(idx);
        if (!p)
            THROW_YASK_EXCEPTION("Error: new_first_domain_index() called without index-node argument");
        return first_index(p);
    }
    yc_number_node_ptr
    yc_node_factory::new_last_domain_index(yc_index_node_ptr idx) {
        auto p = dynamic_pointer_cast<IndexExpr>(idx);
        if (!p)
            THROW_YASK_EXCEPTION("Error: new_last_domain_index() called without index-node argument");
        return last_index(p);
    }
    yc_number_node_ptr operator-(yc_number_node_ptr rhs) {
        auto p = dynamic_pointer_cast<NumExpr>(rhs);
        assert(p);
        return make_shared<NegExpr>(p);
    }
    yc_number_node_ptr operator+(yc_number_node_ptr lhs, yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<AddExpr>(lp, rp);
    }
    yc_number_node_ptr operator+(double lhs, yc_number_node_ptr rhs) {
        return operator+(constNum(lhs), rhs);
    }
    yc_number_node_ptr operator+(yc_number_node_ptr lhs, double rhs) {
        return operator+(lhs, constNum(rhs));
    }
    yc_number_node_ptr operator/(yc_number_node_ptr lhs, yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<DivExpr>(lp, rp);
    }
    yc_number_node_ptr operator/(double lhs, yc_number_node_ptr rhs) {
        return operator/(constNum(lhs), rhs);
    }
    yc_number_node_ptr operator/(yc_number_node_ptr lhs, double rhs) {
        return operator/(lhs, constNum(rhs));
    }
    yc_number_node_ptr operator%(yc_number_node_ptr lhs, yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<ModExpr>(lp, rp);
    }
    yc_number_node_ptr operator%(double lhs, yc_number_node_ptr rhs) {
        return operator%(constNum(lhs), rhs);
    }
    yc_number_node_ptr operator%(yc_number_node_ptr lhs, double rhs) {
        return operator%(lhs, constNum(rhs));
    }
    yc_number_node_ptr operator*(yc_number_node_ptr lhs, yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<MultExpr>(lp, rp);
    }
    yc_number_node_ptr operator*(double lhs, yc_number_node_ptr rhs) {
        return operator*(constNum(lhs), rhs);
    }
    yc_number_node_ptr operator*(yc_number_node_ptr lhs, double rhs) {
        return operator*(lhs, constNum(rhs));
    }
    yc_number_node_ptr operator-(yc_number_node_ptr lhs, yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<SubExpr>(lp, rp);
    }
    yc_number_node_ptr operator-(double lhs, yc_number_node_ptr rhs) {
        return operator-(constNum(lhs), rhs);
    }
    yc_number_node_ptr operator-(yc_number_node_ptr lhs, double rhs) {
        return operator-(lhs, constNum(rhs));
    }
    yc_bool_node_ptr operator!(yc_bool_node_ptr rhs) {
        auto p = dynamic_pointer_cast<BoolExpr>(rhs);
        assert(p);
        return make_shared<NotExpr>(p);
    }
    yc_bool_node_ptr operator||(yc_bool_node_ptr lhs, yc_bool_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<BoolExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<BoolExpr>(rhs);
        assert(rp);
        return make_shared<OrExpr>(lp, rp);
    }
    yc_bool_node_ptr operator&&(yc_bool_node_ptr lhs, yc_bool_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<BoolExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<BoolExpr>(rhs);
        assert(rp);
        return make_shared<AndExpr>(lp, rp);
    }

// Boolean binary operators with numerical inputs.
#define BIN_NUM2BOOL_OP(type, oper)                             \
    yc_bool_node_ptr operator oper(yc_number_node_ptr lhs,      \
                                   yc_number_node_ptr rhs) {    \
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);           \
        assert(lp);                                             \
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);           \
        assert(rp);                                                     \
        return make_shared<NotLessExpr>(lp, rp);                        \
    }                                                                   \
    yc_bool_node_ptr operator oper(yc_index_node_ptr lhs,               \
                                   yc_number_node_ptr rhs) {            \
        return operator>=(dynamic_pointer_cast<yc_number_node>(lhs), rhs); \
    }                                                                   \
    yc_bool_node_ptr operator oper(yc_number_node_ptr lhs, yc_index_node_ptr rhs) { \
        return operator>=(lhs, dynamic_pointer_cast<yc_number_node>(rhs)); \
    }                                                                   \
    yc_bool_node_ptr operator oper(yc_index_node_ptr lhs, yc_index_node_ptr rhs) { \
        return operator>=(dynamic_pointer_cast<yc_number_node>(lhs),    \
                          dynamic_pointer_cast<yc_number_node>(rhs));     \
    }
    BIN_NUM2BOOL_OP(IsEqualExpr, ==)
    BIN_NUM2BOOL_OP(NotEqualExpr, !=)
    BIN_NUM2BOOL_OP(IsLessExpr, <)
    BIN_NUM2BOOL_OP(NotLessExpr, >=)
    BIN_NUM2BOOL_OP(IsGreaterExpr, >)
    BIN_NUM2BOOL_OP(NotGreaterExpr, <=)

    // Compare 2 expr pointers and return whether the expressions are
    // equivalent.
    // TODO: Be much smarter about matching symbolically-equivalent exprs.
    bool areExprsSame(const Expr* e1, const Expr* e2) {

        // Handle null pointers.
        if (e1 == NULL && e2 == NULL)
            return true;
        if (e1 == NULL || e2 == NULL)
            return false;

        // Neither are null, so compare contents.
        return e1->isSame(e2);
    }

    // Unary.
    NumExprPtr operator-(const NumExprPtr rhs) {
        return make_shared<NegExpr>(rhs);
    }

    // Function calls.
#define FUNC_EXPR(fn) \
    NumExprPtr fn(const NumExprPtr rhs) {             \
        return make_shared<FuncExpr>(#fn, std::initializer_list< const NumExprPtr >{ rhs });   \
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
    NumExprPtr fn(const NumExprPtr arg1, const NumExprPtr arg2) {   \
        return make_shared<FuncExpr>(#fn, std::initializer_list< const NumExprPtr >{ arg1, arg2 });         \
    } \
    NumExprPtr fn(const NumExprPtr arg1, double arg2) {   \
        return make_shared<FuncExpr>(#fn, std::initializer_list< const NumExprPtr >{ arg1, constNum(arg2) }); \
    } \
    NumExprPtr fn(double arg1, const NumExprPtr arg2) {   \
        return make_shared<FuncExpr>(#fn, std::initializer_list< const NumExprPtr >{ constNum(arg1), arg2 }); \
    }
    FUNC_EXPR(pow)
#undef FUNC_EXPR

    // A free function to create a constant expression.
    NumExprPtr constNum(double rhs) {
        return make_shared<ConstExpr>(rhs);
    }

    // Free functions to create boundary indices, e.g., 'first_index(x)'.
    NumExprPtr first_index(IndexExprPtr dim) {
        assert(dim->getType() == DOMAIN_INDEX);
        return make_shared<IndexExpr>(dim->getName(), FIRST_INDEX);
    }
    NumExprPtr last_index(IndexExprPtr dim) {
        assert(dim->getType() == DOMAIN_INDEX);
        return make_shared<IndexExpr>(dim->getName(), LAST_INDEX);
    }

    // Commutative.
    // If one side is nothing, just return other side;
    // This allows us to start with an uninitialized GridValue
    // and do the right thing.
    // Start with an empty expression.
    // If LHS is the same expr type, add its operands.
    // Othewise, add the whole expr.
    // Repeat for RHS.
#define COMM_OPER(oper, exprtype)                       \
    NumExprPtr operator oper(const NumExprPtr lhs,      \
                             const NumExprPtr rhs) {    \
        if (lhs.get() == NULL)                          \
            return rhs;                                 \
        else if (rhs.get() == NULL)                     \
            return lhs;                                 \
        auto ex = make_shared<exprtype>();              \
        ex->mergeExpr(lhs);                             \
        ex->mergeExpr(rhs);                             \
        return ex;                                      \
    }                                                   \
    NumExprPtr operator oper(double lhs,                \
                             const NumExprPtr rhs) {    \
        NumExprPtr p = make_shared<ConstExpr>(lhs);     \
        return p oper rhs;                              \
    }                                                   \
    NumExprPtr operator oper(const NumExprPtr lhs,      \
                             double rhs) {              \
        NumExprPtr p = make_shared<ConstExpr>(rhs);     \
        return lhs oper p;                              \
    }
    COMM_OPER(+, AddExpr)
    COMM_OPER(*, MultExpr)

    // Self-modifying versions.
    void operator+=(NumExprPtr& lhs, const NumExprPtr rhs) {
        lhs = lhs + rhs;
    }
    void operator+=(NumExprPtr& lhs, double rhs) {
        lhs = lhs + rhs;
    }
    void operator*=(NumExprPtr& lhs, const NumExprPtr rhs) {
        lhs = lhs * rhs;
    }
    void operator*=(NumExprPtr& lhs, double rhs) {
        lhs = lhs * rhs;
    }

    // Binary.
    NumExprPtr operator-(const NumExprPtr lhs, const NumExprPtr rhs) {
#ifdef USE_ADD_NEG
        // Generate A + -B instead of A - B to allow easy reordering.
        NumExprPtr nrhs = make_shared<NegExpr>(rhs);
        return lhs + nrhs;
#else
        return make_shared<SubExpr>(lhs, rhs);
#endif
    }
    NumExprPtr operator-(double lhs, const NumExprPtr rhs) {
        NumExprPtr p = make_shared<ConstExpr>(lhs);
        return p - rhs;
    }
    NumExprPtr operator-(const NumExprPtr lhs, double rhs) {
        NumExprPtr p = make_shared<ConstExpr>(rhs);
        return lhs - p;
    }

    void operator-=(NumExprPtr& lhs, const NumExprPtr rhs) {
        lhs = lhs - rhs;
    }
    void operator-=(NumExprPtr& lhs, double rhs) {
        lhs = lhs - rhs;
    }

    NumExprPtr operator/(const NumExprPtr lhs, const NumExprPtr rhs) {
        return make_shared<DivExpr>(lhs, rhs);
    }
    NumExprPtr operator/(double lhs, const NumExprPtr rhs) {
        NumExprPtr p = make_shared<ConstExpr>(lhs);
        return p / rhs;
    }
    NumExprPtr operator/(const NumExprPtr lhs, double rhs) {
        NumExprPtr p = make_shared<ConstExpr>(rhs);
        return lhs / p;
    }

    void operator/=(NumExprPtr& lhs, const NumExprPtr rhs) {
        lhs = lhs / rhs;
    }
    void operator/=(NumExprPtr& lhs, double rhs) {
        lhs = lhs / rhs;
    }

    NumExprPtr operator%(const NumExprPtr lhs, const NumExprPtr rhs) {
        return make_shared<ModExpr>(lhs, rhs);
    }
    NumExprPtr operator%(double lhs, const NumExprPtr rhs) {
        NumExprPtr p = make_shared<ConstExpr>(lhs);
        return p % rhs;
    }
    NumExprPtr operator%(const NumExprPtr lhs, double rhs) {
        NumExprPtr p = make_shared<ConstExpr>(rhs);
        return lhs % p;
    }

    // Define a conditional.
    EqualsExprPtr operator IF_OPER(EqualsExprPtr expr, const BoolExprPtr cond) {

        // Add cond to expr.
        assert(expr);
        expr->setCond(cond);
        return expr;
    }
    EqualsExprPtr operator IF_STEP_OPER(EqualsExprPtr expr, const BoolExprPtr step_cond) {

        // Add cond to expr.
        assert(expr);
        expr->setStepCond(step_cond);
        return expr;
    }

    // Define the value of a grid point.
    // Add this equation to the list of eqs for this stencil.
    EqualsExprPtr operator EQUALS_OPER(GridPointPtr gpp, const NumExprPtr rhs) {

        if (!gpp)
            THROW_YASK_EXCEPTION("Error: empty LHS of equation");
        if (!rhs)
            THROW_YASK_EXCEPTION("Error: empty RHS of " +
                                 gpp->makeQuotedStr() + " equation");

        // Get to list of equations in soln indirectly thru grid.
        Grid* gp = gpp->getGrid();
        assert(gp);
        auto* soln = gp->getSoln();
        assert(soln);
        auto& eqs = soln->getEqs();
        auto& settings = soln->getSettings();

        // Make expression node.
        // Conditions may be added later if/when the IF or IF_STEP
        // operators are processed.
        auto expr = make_shared<EqualsExpr>(gpp, rhs);
        if (settings._printEqs)
            soln->get_ostr() << "Equation defined: " << expr->getDescr() << endl;

        // Save the expression in list of equations.
        eqs.addItem(expr);

        return expr;
    }
    EqualsExprPtr operator EQUALS_OPER(GridPointPtr gpp, double rhs) {
        return gpp EQUALS_OPER constNum(rhs);
    }
    EqualsExprPtr operator EQUALS_OPER(GridPointPtr gpp, GridPointPtr rhs) {
        auto p = dynamic_pointer_cast<NumExpr>(rhs);
        assert(p);
        return gpp EQUALS_OPER p;
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
    string GridPoint::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    string EqualsExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }
    string IndexExpr::accept(ExprVisitor* ev) {
        return ev->visit(this);
    }

    // EqualsExpr methods.
    bool EqualsExpr::isScratch() {
        Grid* gp = getGrid();
        return gp && gp->isScratch();
    }
    bool EqualsExpr::isSame(const Expr* other) const {
        auto p = dynamic_cast<const EqualsExpr*>(other);
        return p &&
            _lhs->isSame(p->_lhs.get()) &&
            _rhs->isSame(p->_rhs.get()) &&
            areExprsSame(_cond, p->_cond) && // might be null.
            areExprsSame(_step_cond, p->_step_cond); // might be null.
    }

    // Commutative methods.
    bool CommutativeExpr::isSame(const Expr* other) const {
        auto p = dynamic_cast<const CommutativeExpr*>(other);
        if (!p || _opStr != p->_opStr)
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
                if (matches.count(i) == 0 && op->isSame(oop.get())) {
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
    bool FuncExpr::isSame(const Expr* other) const {
        auto p = dynamic_cast<const FuncExpr*>(other);
        if (!p || _opStr != p->_opStr)
            return false;
        if (_ops.size() != p->_ops.size())
            return false;
        for (size_t i = 0; i < p->_ops.size(); i++) {
            if (!_ops[i]->isSame(p->_ops[i]))
                return false;
        }
        return true;
    }
    bool FuncExpr::makePair(Expr* other) {
        auto p = dynamic_cast<FuncExpr*>(other);

        // Must be another FuncExpr w/all the same operands.
        if (!p || _ops.size() != p->_ops.size())
            return false;
        for (size_t i = 0; i < p->_ops.size(); i++) {
            if (!_ops[i]->isSame(p->_ops[i]))
                return false;
        }

        // Possible pairs.
        // TODO: make list of other options.
        string f1 = "sin";
        string f2 = "cos";
        if ((_opStr == f1 && p->_opStr == f2) ||
            (_opStr == f2 && p->_opStr == f1)) {
            _paired = p;
            p->_paired = this;
            return true;
        }
        return false;
    }

    // GridPoint methods.
    GridPoint::GridPoint(Grid* grid, const NumExprPtrVec& args) :
        _grid(grid), _args(args) {

        // Check for correct number of args.
        size_t nd = grid->getDims().size();
        if (nd != args.size()) {
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: attempt to create a grid point in " <<
                nd << "-D grid '" << getGridName() << "' with " <<
                args.size() << " indices");
        }

        // Eval each arg.
#ifdef DEBUG_GP
        cout << "Creating grid point " << makeQuotedStr() << "...\n";
#endif
        auto dims = grid->getDims();
        for (size_t i = 0; i < nd; i++) {
            auto dim = dims.at(i);
            auto dname = dim->getName();
            auto arg = args.at(i);
            assert(arg);
#ifdef DEBUG_GP
            cout << " Arg " << arg->makeQuotedStr() <<
                " at dim '" << dname << "'\n";
#endif
            int offset = 0;

            // A compile-time const?
            if (arg->isConstVal()) {
                _consts.addDimBack(dname, arg->getIntVal());
#ifdef DEBUG_GP
                cout << "  is const val " << arg->getIntVal() << endl;
#endif
            }

            // A simple offset?
            else if (arg->isOffsetFrom(dname, offset)) {
                _offsets.addDimBack(dname, offset);
#ifdef DEBUG_GP
                cout << "  has offset " << offset << endl;
#endif
            }
        }
        _updateStr();
    }
    const NumExprPtr GridPoint::getArg(const string& dim) const {
        for (int di = 0; di < _grid->get_num_dims(); di++) {
            auto& dn = _grid->get_dim_name(di);  // name of this dim.
            if (dim == dn)
                return _args.at(di);
        }
        return nullptr;
    }
    const string& GridPoint::getGridName() const {
        return _grid->getName();
    }
    string GridPoint::getGridPtr() const {
        string gname = _grid->getName();
        string expr = "static_cast<_context_type::" + gname + "_type*>(_context_data->" + gname;
        if (_grid->isScratch())
            expr += "_list[region_thread_idx].get()";
        expr += ")";
        return expr;
    }
    bool GridPoint::isGridFoldable() const {
        return _grid->isFoldable();
    }
    string GridPoint::makeArgStr(const VarMap* varMap) const {
        string str;
        int i = 0;
        for (auto arg : _args) {
            if (i++) str += ", ";
            str += arg->makeStr(varMap);
        }
        return str;
    }
    string GridPoint::makeStr(const VarMap* varMap) const {
        string str = _grid->getName() + "(" +
                             makeArgStr(varMap) + ")";
        return str;
    }
    string GridPoint::makeLogicalGridStr(const VarMap* varMap) const {
        string str = _grid->getName();
        if (_consts.size())
            str += "(" + _consts.makeDimValStr() + ")";
        return str;
    }
    const IndexExprPtrVec& GridPoint::getDims() const {
        return _grid->getDims();
    }

    // Make string like "x+(4/VLEN_X)" from
    // original arg "x+4" in 'dname' dim.
    // This object has numerators; 'fold' object has denominators.
    // Args w/o simple offset are not modified.
    string GridPoint::makeNormArgStr(const string& dname,
                                     const Dimensions& dims,
                                     const VarMap* varMap) const {
        string res;

        // Non-0 const offset and dname exists in fold?
        auto* ofs = _offsets.lookup(dname);
        if (ofs && *ofs && dims._fold.lookup(dname))
            res = "(" + dname + dims.makeNormStr(*ofs, dname) + ")";

        // Otherwise, just find and format arg as-is.
        else {
            auto& gdims = _grid->getDims();
            for (size_t i = 0; i < gdims.size(); i++) {
                auto gdname = gdims[i]->getName();
                if (gdname == dname)
                    res += _args.at(i)->makeStr(varMap);
            }
        }

        return res;
    }

    // Make string like "x+(4/VLEN_X), y, z-(2/VLEN_Z)" from
    // original args "x+4, y, z-2".
    // This object has numerators; norm object has denominators.
    // Args w/o simple offset are not modified.
    string GridPoint::makeNormArgStr(const Dimensions& dims,
                                     const VarMap* varMap) const {

        string res;
        auto& gd = _grid->getDims();
        for (size_t i = 0; i < gd.size(); i++) {
            if (i)
                res += ", ";
            auto dname = gd[i]->getName();
            res += makeNormArgStr(dname, dims, varMap);
        }
        return res;
    }

    // Make string like "g->_wrap_step(t+1)" from original arg "t+1"
    // if grid uses step dim, "0" otherwise.
    // If grid doesn't allow dynamic alloc, set to fixed value.
    string GridPoint::makeStepArgStr(const string& gridPtr, const Dimensions& dims) const {

        auto& gd = _grid->getDims();
        for (size_t i = 0; i < gd.size(); i++) {
            auto dname = gd[i]->getName();
            auto& arg = _args.at(i);
            if (dname == dims._stepDim) {
                if (_grid->is_dynamic_step_alloc())
                    return gridPtr + "->_wrap_step(" + arg->makeStr() + ")";
                else {
                    auto step_alloc = _grid->get_step_alloc_size();
                    if (step_alloc == 1)
                        return "0"; // 1 alloc => always index 0.
                    else 
                        return "imod_flr<idx_t>(" + arg->makeStr() + ", " +
                            to_string(step_alloc) + ")";
                }
            }
        }
        return "0";
    }

    // Set given arg to given offset; ignore if not in step or domain grid dims.
    void GridPoint::setArgOffset(const IntScalar& offset) {

        // Find dim in grid.
        auto gdims = _grid->getDims();
        for (size_t i = 0; i < gdims.size(); i++) {
            auto gdim = gdims[i];

            // Must be domain or step dim.
            if (gdim->getType() == MISC_INDEX)
                continue;

            auto dname = gdim->getName();
            if (offset.getName() == dname) {

                // Make offset equation.
                int ofs = offset.getVal();
                auto ie = gdim->clone();
                NumExprPtr nep;
                if (ofs > 0) {
                    auto op = constNum(ofs);
                    nep = make_shared<AddExpr>(ie, op);
                }
                else if (ofs < 0) {
                    auto op = constNum(-ofs);
                    nep = make_shared<SubExpr>(ie, op);
                }
                else                // 0 offset.
                    nep = ie;

                // Replace in args.
                _args[i] = nep;

                // Set offset.
                _offsets.addDimBack(dname, ofs);

                // Remove const.
                _consts = _consts.removeDim(dname);

                break;
            }
        }
        _updateStr();
    }

    // Set given arg to given const;
    void GridPoint::setArgConst(const IntScalar& val) {

        // Find dim in grid.
        auto gdims = _grid->getDims();
        for (size_t i = 0; i < gdims.size(); i++) {
            auto gdim = gdims[i];

            auto dname = gdim->getName();
            if (val.getName() == dname) {

                // Make const expr.
                int v = val.getVal();
                auto vp = constNum(v);

                // Replace in args.
                _args[i] = vp;

                // Set const
                _consts.addDimBack(dname, v);

                // Remove offset if it exists.
                _offsets = _offsets.removeDim(dname);

                break;
            }
        }
        _updateStr();
    }

    // Is this expr a simple offset?
    bool IndexExpr::isOffsetFrom(string dim, int& offset) {

        // An index expr is an offset if it's a step or domain dim and the
        // dims are the same.
        if (_type != MISC_INDEX && _dimName == dim) {
            offset = 0;
            return true;
        }
        return false;
    }
    bool DivExpr::isOffsetFrom(string dim, int& offset) {

        // Could allow 'dim / 1', but seems silly.
        return false;
    }
    bool ModExpr::isOffsetFrom(string dim, int& offset) {
        return false;
    }
    bool MultExpr::isOffsetFrom(string dim, int& offset) {

        // Could allow 'dim * 1', but seems silly.
        return false;
    }
    bool SubExpr::isOffsetFrom(string dim, int& offset) {

        // Is this of the form 'dim - offset'?
        int tmp = 0;
        if (_lhs->isOffsetFrom(dim, tmp) &&
            _rhs->isConstVal()) {
            offset = tmp - _rhs->getIntVal();
            return true;
        }
        return false;
    }
    bool AddExpr::isOffsetFrom(string dim, int& offset) {

        // Is this of the form 'dim + offset'?
        // Allow any similar form, e.g., '-5 + dim + 2'.
        int sum = 0;
        int num_dims = 0;
        int tmp = 0;
        for (auto op : _ops) {

            // Is this operand 'dim'?
            if (op->isOffsetFrom(dim, tmp))
                num_dims++;

            // Is this operand a const int?
            else if (op->isConstVal())
                sum += op->getIntVal();

            // Anything else isn't allowed.
            else
                return false;
        }
        // Must be exactly one 'dim'.
        // Don't allow silly forms like 'dim - dim + dim + 2'.
        if (num_dims == 1) {
            offset = tmp + sum;
            return true;
        }
        return false;
    }

    // Make a readable string from an expression.
    string Expr::makeStr(const VarMap* varMap) const {

        // Use a print visitor to make a string.
        ostringstream oss;
        CompilerSettings _dummySettings;
        Dimensions _dummyDims;
        PrintHelper ph(_dummySettings, _dummyDims, NULL, "temp", "", "", ""); // default helper.
        CompilerSettings settings; // default settings.
        PrintVisitorTopDown pv(oss, ph, varMap);
        string res = accept(&pv);

        // Return anything written to the stream
        // concatenated with anything left in the
        // PrintVisitor.
        return oss.str() + res;
    }

    // Return number of nodes.
    int Expr::getNumNodes() const {

        // Use a counter visitor.
        CounterVisitor cv;
        accept(&cv);

        return cv.getNumNodes();
    }

    // Const version of accept.
    string Expr::accept(ExprVisitor* ev) const {
        return const_cast<Expr*>(this)->accept(ev);
    }

} // namespace yask.
