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

///////// VarPoint and EqualsExpr expression nodes.

#pragma once

namespace yask {

    // One specific point in a var.
    // This is an expression leaf-node.
    class VarPoint : public NumExpr,
                     public virtual yc_var_point_node {

    public:

        // What kind of vectorization can be done on this point.
        // Set via Eqs::analyze_vec().
        enum VecType { VEC_UNSET,
                       VEC_FULL, // vectorizable in all vec dims.
                       VEC_PARTIAL, // vectorizable in some vec dims.
                       VEC_NONE  // vectorizable in no vec dims.
        };

        // Analysis of this point for accesses via loops through the domain dims.
        // Set via Eqs::analyze_loop().
        enum VarDepType { DOMAIN_VAR_UNSET, // bogus value.
                          DOMAIN_VAR_INVARIANT, // not dependent on any domain dim.
                          DOMAIN_VAR_DEPENDENT, // dependent on some domain dim, but not inner-loop dim.
                          INNER_LOOP_OFFSET, // dependent on simple offset in inner-loop dim.
                          INNER_LOOP_COMPLEX  // dependent on inner-loop dim in another way.
        };

    protected:
        Var* _var = 0;        // the var this point is from.

        // Index exprs for each dim, e.g.,
        // "3, x-5, y*2, z+4" for dims "n, x, y, z".
        num_expr_ptr_vec _args;

        // Vars below are calculated from above.
        
        // Simple offset for each expr that is dim +/- offset, e.g.,
        // "x=-5, z=4" from above example ('y*2' is not an offset expr).
        // Includes zero offsets.
        // Set in ctor and modified via set_arg_offset/Const().
        IntTuple _offsets;

        // Simple value for each expr that is a const, e.g.,
        // "n=3" from above example.
        // Set in ctor and modified via set_arg_offset/Const().
        IntTuple _consts;

        VecType _vec_type = VEC_UNSET; // allowed vectorization.
        VarDepType _var_dep = DOMAIN_VAR_UNSET; // analysis for looping.

        // Cache the string repr.
        string _def_str;
        void _update_str() {
            _def_str = _make_str();
        }
        string _make_str(const VarMap* var_map = 0) const;

    public:

        // Construct a point given a var and an arg for each dim.
        VarPoint(Var* var, const num_expr_ptr_vec& args);

        // Dtor.
        virtual ~VarPoint() {}

        // Get parent var info.
        const Var* _get_var() const { return _var; }
        Var* _get_var() { return _var; }
        virtual const string& get_var_name() const;
        virtual bool is_var_foldable() const;
        virtual const index_expr_ptr_vec& get_dims() const;
        virtual const index_expr_ptr_vec& get_layout_dims() const;

        // Accessors.
        virtual const num_expr_ptr_vec& get_args() const { return _args; }
        virtual const IntTuple& get_arg_offsets() const { return _offsets; }
        virtual const IntTuple& get_arg_consts() const { return _consts; }
        virtual VecType get_vec_type() const {
            assert(_vec_type != VEC_UNSET);
            return _vec_type;
        }
        virtual void set_vec_type(VecType vt) {
            _vec_type = vt;
        }
        virtual VarDepType get_var_dep() const {
            assert(_var_dep != DOMAIN_VAR_UNSET);
            return _var_dep;
        }
        virtual void set_var_dep(VarDepType vt) {
            _var_dep = vt;
        }

        // Get arg for 'dim' or return null if none.
        virtual const num_expr_ptr get_arg(const string& dim) const;
        
        // Set given arg to given offset; ignore if not in step or domain var dims.
        virtual void set_arg_offset(const IntScalar& offset);

        // Set given args to be given offsets.
        virtual void set_arg_offsets(const IntTuple& offsets) {
            for (auto ofs : offsets)
                set_arg_offset(ofs);
        }

        // Set given arg to given const.
        virtual void set_arg_const(const IntScalar& val);

        // Set given arg to given expr.
        virtual void set_arg_expr(const string& expr_dim, const string& expr);
        
        // Some comparisons.
        bool operator==(const VarPoint& rhs) const {
            return _def_str == rhs._def_str;
        }
        bool operator<(const VarPoint& rhs) const {
            return _def_str < rhs._def_str;
        }

        // Take ev to each value.
        virtual string accept(ExprVisitor* ev);

        // Check for equivalency.
        virtual bool is_same(const Expr* other) const {
            auto p = dynamic_cast<const VarPoint*>(other);
            return p && *this == *p;
        }

        // Check for same logical var.  A logical var is defined by the var
        // itself and any const indices.
        virtual bool is_same_logical_var(const VarPoint& rhs) const {
            return _var == rhs._var && _consts == rhs._consts;
        }

        // String w/name and parens around args, e.g., 'u(x, y+2)'.
        // Apply substitutions to indices using 'var_map' if provided.
        virtual string make_str(const VarMap* var_map = 0) const {
            if (var_map)
                return _make_str(var_map);
            return _def_str;
        }

        // String w/name and parens around const args, e.g., 'u(n=4)'.
        // Apply substitutions to indices using 'var_map' if provided.
        virtual string make_logical_var_str(const VarMap* var_map = 0) const;

        // String w/just comma-sep args, e.g., 'x, y+2'.
        // Apply substitutions to indices using 'var_map' if provided.
        virtual string make_arg_str(const VarMap* var_map = 0) const;

        // String w/just comma-sep args, e.g., 'y+2' in 'dname' dim.
        // Apply substitutions to indices using 'var_map' if provided.
        virtual string make_arg_str(const string& dname,
                                    const VarMap* var_map = 0) const;

        // String v/vec-normalized args, e.g., 'x, y+(2/VLEN_Y)'.
        // Apply substitutions to indices using 'var_map' if provided.
        virtual string make_norm_arg_str(const Dimensions& dims,
                                         const VarMap* var_map = 0) const;

        // Make string like "x+(4/VLEN_X)" from original arg "x+4" in 'dname' dim.
        // Apply substitutions to indices using 'var_map' if provided.
        virtual string make_norm_arg_str(const string& dname,
                                         const Dimensions& dims,
                                         const VarMap* var_map = 0) const;

        // Make string like "g->_wrap_step(t+1)" from original arg "t+1"
        // if var uses step dim, "" otherwise.
        virtual string make_step_arg_str(const string& var_ptr, const Dimensions& dims) const;

        // Create a deep copy of this expression,
        // except pointed-to var is not copied.
        virtual num_expr_ptr clone() const {
            return make_shared<VarPoint>(*this);
        }
        virtual var_point_ptr clone_var_point() const {
            return make_shared<VarPoint>(*this);
        }

        // APIs.
        virtual yc_var* get_var();
        virtual const yc_var* get_var() const;
    };

    // Equality operator for a var point.
    // This defines the LHS as equal to the RHS; it is NOT
    // a comparison operator; it is NOT an assignment operator.
    // It also holds optional conditions.
    class EqualsExpr : public Expr,
                       public virtual yc_equation_node {
    protected:
        var_point_ptr _lhs;
        num_expr_ptr _rhs;
        bool_expr_ptr _cond;
        bool_expr_ptr _step_cond;

    public:
        EqualsExpr(var_point_ptr lhs, num_expr_ptr rhs,
                   bool_expr_ptr cond = nullptr,
                   bool_expr_ptr step_cond = nullptr) :
            _lhs(lhs), _rhs(rhs), _cond(cond), _step_cond(step_cond) { }
        EqualsExpr(const EqualsExpr& src) :
            _lhs(src._lhs->clone_var_point()),
            _rhs(src._rhs->clone()) {
            if (src._cond)
                _cond = src._cond->clone();
            else
                _cond = nullptr;
            if (src._step_cond)
                _step_cond = src._step_cond->clone();
            else
                _step_cond = nullptr;
        }

        var_point_ptr& _get_lhs() { return _lhs; }
        const var_point_ptr& _get_lhs() const { return _lhs; }
        num_expr_ptr& _get_rhs() { return _rhs; }
        const num_expr_ptr& _get_rhs() const { return _rhs; }
        bool_expr_ptr& _get_cond() { return _cond; }
        const bool_expr_ptr& _get_cond() const { return _cond; }
        void _set_cond(bool_expr_ptr cond) { _cond = cond; }
        bool_expr_ptr& _get_step_cond() { return _step_cond; }
        const bool_expr_ptr& _get_step_cond() const { return _step_cond; }
        void _set_step_cond(bool_expr_ptr step_cond) { _step_cond = step_cond; }
        virtual string accept(ExprVisitor* ev);
        static string expr_op_str() { return "EQUALS"; }
        static string cond_op_str() { return "IF_DOMAIN"; }
        static string step_cond_op_str() { return "IF_STEP"; }

        // Get pointer to var on LHS or NULL if not set.
        virtual Var* get_lhs_var() {
            if (_lhs.get())
                return _lhs->_get_var();
            return NULL;
        }

        // LHS is scratch var.
        virtual bool is_scratch();

        // Check for equivalency.
        virtual bool is_same(const Expr* other) const;

        // Create a deep copy of this expression.
        virtual equals_expr_ptr clone() const { return make_shared<EqualsExpr>(*this); }
        virtual yc_equation_node_ptr clone_ast() const {
            return clone();
        }

        // APIs.
        virtual yc_var_point_node_ptr get_lhs() { return _lhs; }
        virtual yc_number_node_ptr get_rhs() { return _rhs; }
        virtual yc_bool_node_ptr get_cond() { return _cond; }
        virtual yc_bool_node_ptr get_step_cond() { return _step_cond; }
        virtual void set_cond(yc_bool_node_ptr cond) {
            if (cond) {
                auto p = dynamic_pointer_cast<BoolExpr>(cond);
                assert(p);
                _cond = p;
            } else
                _cond = nullptr;
        }
        virtual void set_step_cond(yc_bool_node_ptr step_cond) {
            if (step_cond) {
                auto p = dynamic_pointer_cast<BoolExpr>(step_cond);
                assert(p);
                _step_cond = p;
            } else
                _step_cond = nullptr;
        }
    };

    typedef set<VarPoint> VarPointSet;
    typedef set<var_point_ptr> var_point_ptr_set;
    typedef vector<VarPoint> VarPointVec;

} // namespace yask.

// Define hash function for VarPoint for unordered_{set,map}.
namespace std {
    using namespace yask;

    template <> struct hash<VarPoint> {
        size_t operator()(const VarPoint& k) const {
            return hash<string>{}(k.make_str());
        }
    };
}
