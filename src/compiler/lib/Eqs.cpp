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

///////// Methods for equations and equation bundles ////////////

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Eqs.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {

    /////// Some locally-defined specialized visitors.
    
    // A visitor to collect vars and points visited in a set of eqs.
    // For each eq, there are accessors for its output var and point
    // and its input vars and points.
    class PointVisitor : public ExprVisitor {

        // A type to hold a mapping of equations to a set of vars in each.
        typedef unordered_set<Var*> VarSet;
        typedef unordered_map<EqualsExpr*, Var*> VarMap;
        typedef unordered_map<EqualsExpr*, VarSet> VarSetMap;

        VarMap _lhs_vars; // outputs of eqs.
        VarSetMap _rhs_vars; // inputs of eqs.

        // A type to hold a mapping of equations to a set of points in each.
        typedef unordered_set<VarPoint*> PointSet;
        typedef unordered_map<EqualsExpr*, VarPoint*> PointMap;
        typedef unordered_map<EqualsExpr*, PointSet> PointSetMap;

        PointMap _lhs_pts; // outputs of eqs.
        PointSetMap _rhs_pts; // inputs of eqs.
        PointSetMap _cond_pts;  // sub-domain expr inputs.
        PointSetMap _step_cond_pts;  // step-cond expr inputs.
        PointSetMap _all_pts;  // all points in each eq (union of above).

        // Vars for indexing data.
        EqualsExpr* _eq = 0;   // Current equation.
        enum State { _in_lhs, _in_rhs, _in_cond, _in_step_cond } _state;

    public:

        // Ctor.
        PointVisitor() {}
        virtual ~PointVisitor() {}

        // Get access to vars per eq.
        VarMap& get_output_vars() { return _lhs_vars; }
        VarSetMap& get_input_vars() { return _rhs_vars; }

        // Get access to pts per eq.
        // Contains unique ptrs to pts, but pts may not
        // be unique.
        PointMap& get_output_pts() { return _lhs_pts; }
        PointSetMap& get_input_pts() { return _rhs_pts; }
        PointSetMap& get_cond_pts() { return _cond_pts; }
        PointSetMap& get_step_cond_pts() { return _step_cond_pts; }
        PointSetMap& get_all_pts() { return _all_pts; }

        int get_num_eqs() const { return (int)_lhs_pts.size(); }

        // Callback at an equality.
        // Visits all parts that might have var points.
        virtual string visit(EqualsExpr* ee) {

            // Set this equation as current one.
            _eq = ee;

            // Make sure all map entries exist for this eq.
            _lhs_vars[_eq];
            _rhs_vars[_eq];
            _lhs_pts[_eq];
            _rhs_pts[_eq];
            _cond_pts[_eq];
            _step_cond_pts[_eq];
            _all_pts[_eq];

            // visit LHS.
            auto& lhs = ee->_get_lhs();
            _state = _in_lhs;
            lhs->accept(this);
            
            // visit RHS.
            num_expr_ptr rhs = ee->_get_rhs();
            _state = _in_rhs;
            rhs->accept(this);

            // visit conds.
            auto& cp = ee->_get_cond();
            if (cp) {
                _state = _in_cond;
                cp->accept(this);
            }
            auto& scp = ee->_get_step_cond();
            if (scp) {
                _state = _in_step_cond;
                scp->accept(this);
            }
            return "";
        }

        // Callback at a var point.
        virtual string visit(VarPoint* gp) {
            assert(_eq);
            auto* g = gp->_get_var();
            _all_pts[_eq].insert(gp);

            // Save pt and/or var based on state.
            switch (_state) {

            case _in_lhs:
                _lhs_pts[_eq] = gp;
                _lhs_vars[_eq] = g;
                break;

            case _in_rhs:
                _rhs_pts[_eq].insert(gp);
                _rhs_vars[_eq].insert(g);
                break;

            case _in_cond:
                _cond_pts[_eq].insert(gp);
                break;

            case _in_step_cond:
                _step_cond_pts[_eq].insert(gp);
                break;

            default:
                assert(0 && "illegal state");
            }
            return "";
        }
    };

    // Visitor for determining vectorization potential of var points.
    // Vectorization depends not only on the dims of the var itself
    // but also on how the var is indexed at each point.
    class SetVecVisitor : public ExprVisitor {
        const Dimensions& _dims;

    public:
        SetVecVisitor(const Dimensions& dims) :
            _dims(dims) { 
            _visit_equals_lhs = true;
            _visit_var_point_args = true;
            _visit_conds = true;
        }

        // Check each var point in expr.
        virtual string visit(VarPoint* gp) {
            auto* var = gp->_get_var();

            // Folded dims in the solution.
            int soln_nfd = _dims._fold_gt1.size();
            
            // Folded dims in this var.
            int var_nfd = var->get_num_foldable_dims();
            assert(var_nfd <= soln_nfd);

            // Degenerate case with no folding in soln: we still mark points
            // using vars with some domain dims as vectorizable.
            if (soln_nfd == 0 && var->is_foldable())
                gp->set_vec_type(VarPoint::VEC_FULL);

            // No foldable dims.
            else if (var_nfd == 0)
                gp->set_vec_type(VarPoint::VEC_NONE);

            else {
                assert(var_nfd > 0);

                // Amount of vectorization allowed primarily depends on number
                // of folded dimensions in the var accessed at this point.
                // Vectorization is only possible if each access to a vectorized
                // dim is a simple offset.  For example, in var dim 'x', the
                // index in the corresponding posn must be 'x', 'x+n', or 'x-n'.
                // TODO: is this redundant with expr analysis?
                int fdoffsets = 0;
                for (auto fdim : _dims._fold_gt1) {
                    auto& fdname = fdim._get_name();
                    if (gp->get_arg_offsets().lookup(fdname))
                        fdoffsets++;
                }
                assert(fdoffsets <= var_nfd);

                // All folded dims are vectorizable?
                if (fdoffsets == soln_nfd) {
                    assert(var->is_foldable());
                    gp->set_vec_type(VarPoint::VEC_FULL); // all good.
                }

                // Some dims are vectorizable?
                else if (fdoffsets > 0)
                    gp->set_vec_type(VarPoint::VEC_PARTIAL);

                // No dims are vectorizable.
                else
                    gp->set_vec_type(VarPoint::VEC_NONE);

            }

            // Also check args of this var point.
            return ExprVisitor::visit(gp);
        }
    };

    // Visitor to find set of all referenced index vars.
    class FindIndicesVisitor : public ExprVisitor {

    public:
        set<string> vars_used;

        FindIndicesVisitor() {
            _visit_equals_lhs = true;
            _visit_var_point_args = true;
            _visit_conds = true;
        }
        
        // Check each index expr;
        virtual string visit(IndexExpr* ie) {
            vars_used.insert(ie->_get_name());
            return "";
        }
    };

    // Visitor for determining inner-loop accesses of var points.
    class SetLoopVisitor : public ExprVisitor {
        const Dimensions& _dims;
        const CompilerSettings& _settings;

    public:
        SetLoopVisitor(const Dimensions& dims,
                       const CompilerSettings& settings) :
            _dims(dims), _settings(settings) { 
            _visit_equals_lhs = true;
            _visit_var_point_args = true;
            _visit_conds = true;
        }

        // Check each var point in expr.
        virtual string visit(VarPoint* gp) {

            // Info from var.
            auto* var = gp->_get_var();
            auto gdims = var->get_dim_names();

            // Inner-loop var.
            auto& idim = _settings._inner_loop_dim;

            // Access type.
            // Assume invariant, then check below.
            VarPoint::VarDepType lt = VarPoint::DOMAIN_VAR_INVARIANT;

            // Check every point arg.
            auto& args = gp->get_args();
            for (size_t ai = 0; ai < args.size(); ai++) {
                auto& arg = args.at(ai);
                assert(ai < gdims.size());

                // Get set of indices used by this arg expr.
                FindIndicesVisitor fvv;
                arg->accept(&fvv);

                // Does this arg refer to any domain dim?
                if (lt == VarPoint::DOMAIN_VAR_INVARIANT) {
                    for (auto d : _dims._domain_dims) {
                        auto& dname = d._get_name();
                        
                        if (dname != idim && fvv.vars_used.count(dname)) {
                            lt = VarPoint::DOMAIN_VAR_DEPENDENT;
                            break;  // out of dim loop; no need to continue.
                        }
                    }
                }
                
                // Does this arg refer to idim?
                if (fvv.vars_used.count(idim)) {

                    // Is it in the idim posn and a simple offset?
                    int offset = 0;
                    if (gdims.at(ai) == idim &&
                        arg->is_offset_from(idim, offset)) {
                        lt = VarPoint::INNER_LOOP_OFFSET;
                    }

                    // Otherwise, this arg uses idim, but not
                    // in a simple way.
                    else {
                        lt = VarPoint::INNER_LOOP_COMPLEX;
                        break;  // out of arg loop; no need to continue.
                    }
                }
            }
            gp->set_var_dep(lt);
            return "";
        }
    };

    // Visitor that will shift each var point by an offset.
    class OffsetVisitor: public ExprVisitor {
        IntTuple _ofs;

    public:
        OffsetVisitor(const IntTuple& ofs) :
            _ofs(ofs) {
            _visit_equals_lhs = true;
            _visit_var_point_args = true;
            _visit_conds = true;
        }

        // Visit a var point.
        virtual string visit(VarPoint* gp) {

            // Shift var _ofs points.
            auto ofs0 = gp->get_arg_offsets();
            IntTuple new_loc = ofs0.add_elements(_ofs, false);
            gp->set_arg_offsets(new_loc);
            return "";
        }
    };

    ////////// Methods.
   
    // Analyze group of equations.
    // Sets _step_dir in dims.
    // Finds dependencies based on all eqs if 'settings._find_deps', setting
    // _imm_dep_on and _dep_on.
    // Throws exceptions on illegal dependencies.
    // TODO: split this into smaller functions.
    // BIG-TODO: replace dependency algorithms with integration of a polyhedral
    // library.
    void Eqs::analyze_eqs(const CompilerSettings& settings,
                          Dimensions& dims,
                          ostream& os) {
        auto& step_dim = dims._step_dim;

        // Gather initial stats from all eqs.
        PointVisitor pt_vis;
        visit_eqs(&pt_vis);
        auto& out_vars = pt_vis.get_output_vars();
        auto& in_vars = pt_vis.get_input_vars();
        auto& out_pts = pt_vis.get_output_pts();
        auto& in_pts = pt_vis.get_input_pts();
        //auto& cond_pts = pt_vis.get_cond_pts();
        //auto& step_cond_pts = pt_vis.get_step_cond_pts();

        // 1. Check each eq internally.
        os << "\nProcessing " << get_num() << " stencil equation(s)...\n";
        for (auto eq1 : get_all()) {
            auto* eq1p = eq1.get();
            assert(out_vars.count(eq1p));
            assert(in_vars.count(eq1p));
            auto* og1 = out_vars.at(eq1p);
            assert(og1 == eq1->_get_var());
            auto* op1 = out_pts.at(eq1p);
            //auto& ig1 = in_vars.at(eq1p);
            auto& ip1 = in_pts.at(eq1p);
            auto cond1 = eq1p->_get_cond();
            auto stcond1 = eq1p->_get_step_cond();
            num_expr_ptr step_expr1 = op1->get_arg(step_dim); // may be null.

#ifdef DEBUG_DEP
            cout << " Checking internal consistency of equation " <<
                eq1->make_quoted_str() << "...\n";
#endif

            // LHS must have all domain dims.
            for (auto& dd : dims._domain_dims) {
                auto& dname = dd._get_name();
                auto dexpr = op1->get_arg(dname);
                if (!dexpr)
                    THROW_YASK_EXCEPTION("var equation " + eq1->make_quoted_str() +
                                         " does not use domain-dimension '" + dname +
                                         "' on LHS");
            }

            // LHS of non-scratch must have step dim and vice-versa.
            if (!og1->is_scratch()) {
                if (!step_expr1)
                    THROW_YASK_EXCEPTION("non-scratch var equation " + eq1->make_quoted_str() +
                                         " does not use step-dimension '" + step_dim +
                                         "' on LHS");
            } else {
                if (step_expr1)
                    THROW_YASK_EXCEPTION("scratch-var equation " + eq1->make_quoted_str() +
                                         " cannot use step-dimension '" + step_dim + "'");
            }

            // Check LHS var dimensions and associated args.
            for (int di = 0; di < og1->get_num_dims(); di++) {
                auto& dn = og1->get_dim_name(di);  // name of this dim.
                auto argn = op1->get_args().at(di); // LHS arg for this dim.

                // Check based on dim type.
                if (dn == step_dim) {
                }

                // LHS must have simple indices in domain dims, e.g.,
                // 'x', 'y'.
                else if (dims._domain_dims.lookup(dn)) {

                    // Make expected arg, e.g., 'x'.
                    auto earg = make_shared<IndexExpr>(dn, DOMAIN_INDEX);

                    // Compare to actual.
                    if (!argn->is_same(earg))
                        THROW_YASK_EXCEPTION("LHS of equation " + eq1->make_quoted_str() +
                                             " contains expression " + argn->make_quoted_str() +
                                             " for domain dimension '" + dn +
                                             "' where " + earg->make_quoted_str() +
                                             " is expected");
                }

                // Misc dim must be a const.  TODO: allow non-const misc
                // dims and treat const and non-const ones separately, e.g.,
                // for interleaving.
                else {

                    if (!argn->is_const_val())
                        THROW_YASK_EXCEPTION("LHS of equation " + eq1->make_quoted_str() +
                                             " contains expression " + argn->make_quoted_str() +
                                             " for misc dimension '" + dn +
                                             "' where kernel-run-time constant integer is expected");
                    argn->get_int_val(); // throws exception if not an integer.
                }
            }
        
            // Heuristics to set the default step direction.
            // The accuracy isn't critical, because the default is only be
            // used in the standalone test utility and the auto-tuner.
            if (!og1->is_scratch()) {

                // First, see if LHS step arg is a simple offset, e.g., 'u(t+1, ...)'.
                // This is the most common case.
                auto& lofss = op1->get_arg_offsets();
                auto* lofsp = lofss.lookup(step_dim); // offset at step dim.
                if (lofsp) {
                    auto lofs = *lofsp;

                    // Scan input (RHS) points.
                    for (auto i1 : ip1) {
                        
                        // Is point a simple offset from step, e.g., 'u(t-2, ...)'?
                        auto* rsi1p = i1->get_arg_offsets().lookup(step_dim);
                        if (rsi1p) {
                            int rofs = *rsi1p;

                            // Example:
                            // forward: 'u(t+1, ...) EQUALS ... u(t, ...) ...',
                            // backward: 'u(t-1, ...) EQUALS ... u(t, ...) ...'.
                            if (lofs > rofs) {
                                dims._step_dir = 1;
                                break;
                            }
                            else if (lofs < rofs) {
                                dims._step_dir = -1;
                                break;
                            }
                        }
                    } // for all RHS points.
                    
                    // Soln step-direction heuristic used only if not set.
                    // Assume 'u(t+1, ...) EQUALS ...' implies forward,
                    // and 'u(t-1, ...) EQUALS ...' implies backward.
                    if (dims._step_dir == 0 && lofs != 0)
                        dims._step_dir = (lofs > 0) ? 1 : -1;

                }
            }

            // LHS of equation must be vectorizable.
            // TODO: relax this restriction.
            if (op1->get_vec_type() != VarPoint::VEC_FULL) {
                THROW_YASK_EXCEPTION("LHS of equation " + eq1->make_quoted_str() +
                                     " is not fully vectorizable because not all folded"
                                     " dimensions are accessed via simple offsets from their respective indices");
            }

            // Check that domain indices are simple offsets and
            // misc indices are consts on RHS.
            for (auto i1 : ip1) {
                auto* ig1 = i1->_get_var();

                for (int di = 0; di < ig1->get_num_dims(); di++) {
                    auto& dn = ig1->get_dim_name(di);  // name of this dim.
                    auto argn = i1->get_args().at(di); // arg for this dim.

                    // Check based on dim type.
                    if (dn == step_dim) {
                    }

                    // Must have simple indices in domain dims.
                    else if (dims._domain_dims.lookup(dn)) {
                        auto* rsi1p = i1->get_arg_offsets().lookup(dn);
                        if (!rsi1p)
                            THROW_YASK_EXCEPTION("RHS of equation " + eq1->make_quoted_str() +
                                                 " contains expression " + argn->make_quoted_str() +
                                                 " for domain dimension '" + dn +
                                                 "' where constant-integer offset from '" + dn +
                                                 "' is expected");
                    }

                    // Misc dim must be a const.
                    else {
                        if (!argn->is_const_val())
                            THROW_YASK_EXCEPTION("RHS of equation " + eq1->make_quoted_str() +
                                                 " contains expression " + argn->make_quoted_str() +
                                                 " for misc dimension '" + dn +
                                                 "' where constant integer is expected");
                        argn->get_int_val(); // throws exception if not an integer.
                    }
                }
            } // input pts.

            // TODO: check to make sure cond1 depends only on domain indices.
            // TODO: check to make sure stcond1 does not depend on domain indices.
        } // for all eqs.

        // 2. Check each pair of eqs.
        os << "Analyzing for dependencies...\n";
        for (auto eq1 : get_all()) {
            auto* eq1p = eq1.get();
            assert(out_vars.count(eq1p));
            assert(in_vars.count(eq1p));
            auto* og1 = out_vars.at(eq1p);
            assert(og1 == eq1->_get_var());
            auto* op1 = out_pts.at(eq1p);
            //auto& ig1 = in_vars.at(eq1p);
            //auto& ip1 = in_pts.at(eq1p);
            auto cond1 = eq1p->_get_cond();
            auto stcond1 = eq1p->_get_step_cond();
            num_expr_ptr step_expr1 = op1->get_arg(step_dim);

            // Check each 'eq2' to see if it depends on 'eq1'.
            for (auto eq2 : get_all()) {
                auto* eq2p = eq2.get();
                auto& og2 = out_vars.at(eq2p);
                assert(og2 == eq2->_get_var());
                auto& op2 = out_pts.at(eq2p);
                auto& ig2 = in_vars.at(eq2p);
                auto& ip2 = in_pts.at(eq2p);
                auto cond2 = eq2p->_get_cond();
                auto stcond2 = eq2p->_get_step_cond();

#ifdef DEBUG_DEP
                cout << " Checking eq " <<
                    eq1->make_quoted_str() << " vs " <<
                    eq2->make_quoted_str() << "...\n";
#endif

                bool same_eq = eq1 == eq2;
                bool same_op = are_exprs_same(op1, op2);
                bool same_cond = are_exprs_same(cond1, cond2);
                bool same_stcond = are_exprs_same(stcond1, stcond2);

                // A separate var is defined by its name and any const indices.
                //bool same_og = op1->is_same_logical_var(*op2);

                // If two different eqs have the same conditions, they
                // cannot have the same LHS.
                if (!same_eq && same_cond && same_stcond && same_op) {
                    string cdesc = cond1 ? "with domain condition " + cond1->make_quoted_str() :
                        "without domain conditions";
                    string stcdesc = stcond1 ? "with step condition " + stcond1->make_quoted_str() :
                        "without step conditions";
                    THROW_YASK_EXCEPTION("two equations " + cdesc +
                                         " and " + stcdesc +
                                         " have the same LHS: " +
                                         eq1->make_quoted_str() + " and " +
                                         eq2->make_quoted_str());
                }

                // First dep check: exact matches on LHS of eq1 to RHS of eq2.
                // eq2 dep on eq1 => some output of eq1 is an input to eq2.
                // If the two eqs have the same condition, detect
                // dependencies by looking for exact matches.
                // We do this check first because it's quicker than the
                // detailed scan done later if this one doesn't find a dep.
                // Also, this is always illegal, even if not finding deps.
                //
                // Example:
                //  eq1: a(t+1, x, ...) EQUALS ...
                //  eq2: b(t+1, x, ...) EQUALS a(t+1, x, ...) ...
                if (same_cond && same_stcond && ip2.count(op1)) {

                    // Eq depends on itself?
                    if (same_eq) {

                        // Exit with error.
                        THROW_YASK_EXCEPTION("illegal dependency: LHS of equation " +
                                             eq1->make_quoted_str() + " also appears on its RHS");
                    }

                    // Save dependency.
#ifdef DEBUG_DEP
                    cout << "  Exact match found to " << op1->make_quoted_str() << ".\n";
#endif
                    if (settings._find_deps)
                        _deps.set_imm_dep_on(eq2, eq1);

                    // Move along to next eq2.
                    continue;
                }

                // Don't do more conservative checks if not looking for deps.
                if (!settings._find_deps)
                    continue;

                // Next dep check: inexact matches on LHS of eq1 to RHS of eq2.
                // Does eq1 define *any* point in a var that eq2 inputs
                // at the same step index?  If so, they *might* have a
                // dependency. Some of these may not be real
                // dependencies due to conditions. Those that are real
                // may or may not be legal.
                //
                // Example:
                //  eq1: a(t+1, x, ...) EQUALS ... IF_DOMAIN ...
                //  eq2: b(t+1, x, ...) EQUALS a(t+1, x+5, ...) ... IF_DOMAIN ...
                //
                // Example:
                //  eq1: tmp(x, ...) EQUALS ...
                //  eq2: b(t+1, x, ...) EQUALS tmp(x+2, ...)
                //
                // TODO: be much smarter about this and find only real
                // dependencies--use a polyhedral library?
                if (ig2.count(og1)) {

                    // detailed check of g1 input points on RHS of eq2.
                    for (auto* i2 : ip2) {

                        // Same logical var?
                        bool same_var = i2->is_same_logical_var(*op1);

                        // If not same var, no dependency.
                        if (!same_var)
                            continue;

                        // Both points at same step?
                        bool same_step = false;
                        num_expr_ptr step_expr2 = i2->get_arg(step_dim);
                        if (step_expr1 && step_expr2 &&
                            are_exprs_same(step_expr1, step_expr2))
                            same_step = true;

                        // From same step index, e.g., same time?
                        // Or, passing data thru a temp var?
                        if (same_step || og1->is_scratch()) {

                            // Eq depends on itself?
                            if (same_eq) {

                                // Exit with error.
                                string stepmsg = same_step ? " at '" + step_expr1->make_quoted_str() + "'" : "";
                                THROW_YASK_EXCEPTION("disallowed dependency: var '" +
                                                     op1->make_logical_var_str() + "' on LHS of equation " +
                                                     eq1->make_quoted_str() + " also appears on its RHS" +
                                                     stepmsg);
                            }

                            // Save dependency.
#ifdef DEBUG_DEP
                            cout << "  Likely match found to " << op1->make_quoted_str() << ".\n";
#endif
                            _deps.set_imm_dep_on(eq2, eq1);

                            // Move along to next equation.
                            break;
                        }
                    }
                }
#ifdef DEBUG_DEP
                cout << "  No deps found.\n";
#endif

            } // for all eqs (eq2).
        } // for all eqs (eq1).

        // If the step dir wasn't set (no eqs), set it now.
        if (!dims._step_dir)
            dims._step_dir = 1;

#ifdef DEBUG_DEP
        cout << "Dependencies for all eqs:\n";
        _deps.print_deps(cout);
#endif

        // Find scratch children.
        analyze_scratch();

#ifdef DEBUG_DEP
        cout << "Dependencies from non-scratch to scratch eqs:\n";
        _scratches.print_deps(cout);
#endif

        // Resolve indirect dependencies.
        // Do this even if not finding deps because we want to
        // process deps provided by the user.
        os << "Finding transitive closure of dependencies...\n";
        find_all_deps();

        // Sort.
        os << "Topologically ordering equations...\n";
        topo_sort();
    }

    // Determine which var points can be vectorized.
    void Eqs::analyze_vec(const CompilerSettings& settings,
                          const Dimensions& dims) {

        // Send a 'SetVecVisitor' to each point in
        // the current equations.
        SetVecVisitor svv(dims);
        visit_eqs(&svv);
    }

    // Determine loop access behavior of var points.
    void Eqs::analyze_loop(const CompilerSettings& settings,
                           const Dimensions& dims) {

        // Send a 'SetLoopVisitor' to each point in
        // the current equations.
        SetLoopVisitor slv(dims, settings);
        visit_eqs(&slv);
    }

    // Update access stats for the vars.
    // For now, this us just const indices.
    // Halos are updated later, after stages are established.
    void Eqs::update_var_stats() {

        // Find all LHS and RHS points and vars for all eqs.
        PointVisitor pv;
        visit_eqs(&pv);

        // Analyze each eq.
        for (auto& eq : get_all()) {

            // Get all var points touched by this eq.
            auto& all_pts1 = pv.get_all_pts().at(eq.get());

            // Update stats of each var accessed in 'eq'.
            for (auto ap : all_pts1) {
                auto* g = ap->_get_var(); // var for point 'ap'.
                g->update_const_indices(ap->get_arg_consts());
            }
        }
    }

    // Find scratch-var eqs needed for each non-scratch eq.  These will
    // eventually be gathered into bundles and saved as the "scratch
    // children" for each non-scratch bundle.
    void Eqs::analyze_scratch() {

        // Example:
        // eq1: scr(x) EQUALS u(t,x+1);
        // eq2: u(t+1,x) EQUALS scr(x+2);
        // Direct deps: eq2 -> eq1(s).
        // eq1 is scratch child of eq2.

        // Example:
        // eq1: scr1(x) EQUALS u(t,x+1);
        // eq2: scr2(x) EQUALS scr1(x+2);
        // eq3: u(t+1,x) EQUALS scr2(x+4);
        // Direct deps: eq3 -> eq2(s) -> eq1(s).
        // eq1 and eq2 are scratch children of eq3.
        
        // Find all LHS and RHS points and vars for all eqs.
        PointVisitor pv;
        visit_eqs(&pv);

        // Analyze each eq.
        for (auto& eq1 : get_all()) {

            // Output var for this eq.
            auto* og1 = pv.get_output_vars().at(eq1.get());

            // Only need to look at dep paths starting from non-scratch eqs.
            if (og1->is_scratch())
                continue;

            // We start with each non-scratch eq and walk the dep tree to
            // find all dependent scratch eqs.  It's important to
            // then visit the eqs in dep order using 'path' to get only
            // unbroken chains of scratch vars.
            // Note that sub-paths may be visited more than once, e.g.,
            // 'eq3 -> eq2(s)' and 'eq3 -> eq2(s) -> eq1(s)' are 2 paths from
            // the second example above, but this shouldn't cause any issues,
            // just some redundant work.
            get_deps().visit_deps

                // For each 'b', 'eq1' is 'b' or depends on 'b',
                // immediately or indirectly; 'path' leads from
                // 'eq1' to 'b'.
                (eq1, [&](equals_expr_ptr b, EqList& path) {
                    //auto* ogb = pv.get_output_vars().at(b.get());

                    // Find scratch-var eqs in this dep path that are
                    // needed for 'eq1'. Walk dep path from 'eq1' to 'b'
                    // until a non-scratch var is found.
                    unordered_set<Var*> scratches_seen;
                    for (auto eq2 : path) {

                        // Don't process 'eq1', the initial non-scratch eq.
                        if (eq2 == eq1)
                            continue;
                        
                        // If this isn't a scratch eq, we are done
                        // w/this path because we only want the eqs
                        // from 'eq1' through an *unbroken* chain of
                        // scratch vars.
                        auto* og2 = pv.get_output_vars().at(eq2.get());
                        if (!og2->is_scratch())
                            break;

                        // Add 'eq2' to the set needed for 'eq1'.
                        // NB: scratch-deps are used as a map of sets.
                        get_scratch_deps().set_imm_dep_on(eq1, eq2);

                        // Check for illegal scratch path.
                        // TODO: this is only illegal because the scratch
                        // write area is stored in the var, so >1 write
                        // areas can't be shared. Need to store the write
                        // areas somewhere else, maybe in the equation or
                        // bundle. Would require changes to kernel as well.
                        if (scratches_seen.count(og2))
                            THROW_YASK_EXCEPTION("scratch-var '" +
                                                 og2->get_name() + "' depends upon itself");
                        scratches_seen.insert(og2);
                    }
                });
        }
    }

    // Get the full name of an eq-lot.
    // Must be unique.
    string EqLot::_get_name() const {

        // Add index to base name.
        return base_name + "_" + to_string(index);
    }

    // Make a human-readable description of this eq bundle.
    string EqBundle::get_descr(bool show_cond,
                              string quote) const
    {
        string des;
        if (is_scratch())
            des += "scratch ";
        des += "equation-bundle " + quote + _get_name() + quote;
        if (show_cond) {
            if (cond.get())
                des += " w/domain condition " + cond->make_quoted_str(quote);
            else
                des += " w/o domain condition";
            if (step_cond.get())
                des += " w/step condition " + step_cond->make_quoted_str(quote);
            else
                des += " w/o step condition";
        }
        return des;
    }

    // Add an equation to an EqBundle.
    void EqBundle::add_eq(equals_expr_ptr ee)
    {
#ifdef DEBUG_EQ_BUNDLE
        cout << "EqBundle: adding " << ee->make_quoted_str() << endl;
#endif
        _eqs.insert(ee);

        // Get I/O point data from eq 'ee'.
        PointVisitor pv;
        ee->accept(&pv);

        // update list of input and output vars for this bundle.
        auto* out_var = pv.get_output_vars().at(ee.get());
        _out_vars.insert(out_var);
        auto& in_vars = pv.get_input_vars().at(ee.get());
        for (auto* g : in_vars)
            _in_vars.insert(g);
    }

    // Print stats from eqs.
    void EqLot::print_stats(ostream& os, const string& msg) {
        CounterVisitor cv;
        visit_eqs(&cv);
        cv.print_stats(os, msg);
    }

    // Print stats from eqs in bundles.
    void EqBundles::print_stats(ostream& os, const string& msg) {
        CounterVisitor cv;

        // Use separate counter visitor for each bundle
        // to avoid treating repeated eqs as common sub-exprs.
        for (auto& eq : _all) {
            CounterVisitor ecv;
            eq->visit_eqs(&ecv);
            cv += ecv;
        }
        cv.print_stats(os, msg);
    }

    // Replicate each equation at the non-zero offsets for
    // each vector in a cluster.
    void EqBundle::replicate_eqs_in_cluster(Dimensions& dims)
    {
        // Make a copy of the original equations so we can iterate through
        // them while adding to the bundle.
        EqList eqs(_eqs);

        // Loop thru points in cluster.
        dims._cluster_mults.visit_all_points([&](const IntTuple& cluster_index,
                                                 size_t idx) {

                // Don't need copy of one at origin.
                if (cluster_index.sum() > 0) {

                    // Get offset of cluster, which is each cluster index multipled
                    // by corresponding vector size.  Example: for a 4x4 fold in a
                    // 1x2 cluster, the 2nd cluster index will be (0,1) and the
                    // corresponding cluster offset will be (0,4).
                    auto cluster_offset = cluster_index.mult_elements(dims._fold);

                    // Loop thru eqs.
                    for (auto eq : eqs) {
                        assert(eq.get());

                        // Make a copy.
                        auto eq2 = eq->clone();

                        // Add offsets to each var point.
                        OffsetVisitor ov(cluster_offset);
                        eq2->accept(&ov);

                        // Put new equation into bundle.
                        add_eq(eq2);
                    }
                }
                return true;
            });

        // Ensure the expected number of equations now exist.
        assert(_eqs.size() == eqs.size() * dims._cluster_mults.product());
    }

    // Add 'eq' from 'all_eqs' to eq-bundle with 'base_name' unless
    // alread added or illegal.  The corresponding index in '_indices' will
    // be incremented if a new bundle is created.  Returns whether a new
    // bundle was created.
    bool EqBundles::add_eq_to_bundle(Eqs& all_eqs,
                                  equals_expr_ptr eq,
                                  const string& base_name,
                                  const CompilerSettings& settings) {
        assert(_dims);
        auto& step_dim = _dims->_step_dim;
        
        // Equation already added?
        if (_eqs_in_bundles.count(eq))
            return false;

        // Get conditions, if any.
        auto cond = eq->_get_cond();
        auto stcond = eq->_get_step_cond();

        // Get step expr, if any.
        auto step_expr = eq->_get_lhs()->get_arg(step_dim);
        
        // Get deps between eqs.
        auto& eq_deps = all_eqs.get_deps();

        // Loop through existing bundles, looking for one that
        // 'eq' can be added to.
        EqBundle* target = 0;
        auto eqg = eq->_get_var();
        for (auto& eg : get_all()) {

            // Must be same scratch-ness.
            if (eg->is_scratch() != eq->is_scratch())
                continue;

            // Names must match.
            if (eg->base_name != base_name)
                continue;

            // Conditions must match (both may be null).
            if (!are_exprs_same(eg->cond, cond))
                continue;
            if (!are_exprs_same(eg->step_cond, stcond))
                continue;

            // LHS step exprs must match (both may be null).
            if (!are_exprs_same(eg->step_expr, step_expr))
                continue;

            // Look for any condition or dependencies that would prevent
            // adding 'eq' to 'eg'.
            bool is_ok = true;
            for (auto& eq2 : eg->get_eqs()) {
                auto eq2g = eq2->_get_var();
                assert (eqg);
                assert (eq2g);

                // If scratch, 'eq' and 'eq2' must have same halo.
                // This is because scratch halos are written to.
                // If dims are same, halos can be adjusted if allowed.
                if (eq->is_scratch()) {
                    if (!eq2g->are_dims_same(*eqg))
                        is_ok = false;
                    else if (!settings._bundle_scratch && !eq2g->is_halo_same(*eqg))
                        is_ok = false;
                }

                // Look for any dependency between 'eq' and 'eq2'.
                if (is_ok && eq_deps.is_dep(eq, eq2)) {
#if DEBUG_ADD_EXPRS
                    cout << "add_eq_from_var: not adding equation " <<
                        eq->make_quoted_str() << " to " << eg.get_descr() <<
                        " because of dependency w/equation " <<
                        eq2->make_quoted_str() << endl;
#endif
                    is_ok = false;
                }

                if (!is_ok)
                    break;
            }

            // Remember target bundle if ok and stop looking.
            if (is_ok) {
                target = eg.get();
                break;
            }
        }

        // Make new bundle if no target bundle found.
        bool new_bundle = false;
        if (!target) {
            auto ne = make_shared<EqBundle>(*_dims, eq->is_scratch());
            add_item(ne);
            target = ne.get();
            if (eq->is_scratch())
                target->base_name = string("scratch_") + base_name;
            else
                target->base_name = base_name;
            target->index = _indices[base_name]++;
            target->cond = cond;
            target->step_cond = stcond;
            target->step_expr = step_expr;
            new_bundle = true;

#if DEBUG_ADD_EXPRS
            cout << "Creating new " << target->get_descr() << endl;
#endif
        }

        // Add eq to target eq-bundle.
        assert(target);
#if DEBUG_ADD_EXPRS
        cout << "Adding " << eq->make_quoted_str() <<
            " to " << target->get_descr() << endl;
#endif
        target->add_eq(eq);

        // Remember eq and updated var.
        _eqs_in_bundles.insert(eq);
        _out_vars.insert(eq->_get_var());

        return new_bundle;
    }

    // Find halos needed for each var.
    void EqStages::calc_halos(EqBundles& all_bundles) {

        // Find all LHS and RHS points and vars for all eqs.
        PointVisitor pv;
        visit_eqs(&pv);

#ifdef DEBUG_SCRATCH
        cout << "* c_h: analyzing " << get_all().size() << " eqs...\n";
#endif
        
        // First, set halos based only on immediate accesses.
        // Loop thru stages.
        for (auto& sp : get_all()) {
            auto stage_name = sp->_get_name();

            // Loop thru equations in this stage.
            for (auto& eq : sp->get_eqs()) {

                // Get all var points touched by this eq.
                auto& all_pts1 = pv.get_all_pts().at(eq.get());
                auto& out_pt1 = pv.get_output_pts().at(eq.get());

                // Update stats of each var accessed in 'eq'.
                for (auto ap : all_pts1) {
                    auto* g = ap->_get_var(); // var for point 'ap'.
                    g->update_halo(stage_name, ap->get_arg_offsets());
                }
                auto* g = out_pt1->_get_var();
                g->update_write_points(stage_name, out_pt1->get_arg_offsets());
            }
        }

        // Next, propagate halos through scratch vars as needed.

        // Example 1:
        // eq1: scr1(x) EQUALS u(t,x+1); <-- orig halo of u = 1.
        // eq2: u(t+1,x) EQUALS scr1(x+2); <-- orig halo of scr1 = 2.
        // Direct deps: eq2 -> eq1(s).
        // Halo of u must be increased to 1 + 2 = 3 due to
        // eq1: u(t,x+1) on rhs and orig halo of scr1 on lhs,
        // i.e., because u(t+1,x) EQUALS u(t,(x+2)+1) by subst.

        // Example 2:
        // eq1: scr1(x) EQUALS u(t,x+1); <-- orig halo of u = 1.
        // eq2: scr2(x) EQUALS scr1(x+2); <-- orig halo of scr1 = 2.
        // eq3: u(t+1,x) EQUALS scr2(x+4); <-- orig halo of scr2 = 4.
        // Direct deps: eq3 -> eq2(s) -> eq1(s).
        // Halo of scr1 must be increased to 2 + 4 = 6 due to
        // eq2: scr1(x+2) on rhs and orig halo of scr2 on lhs.
        // Then, halo of u must be increased to 1 + 6 = 7 due to
        // eq1: u(t,x+1) on rhs and new halo of scr1 on lhs.
        // Or, u(t+1,x) EQUALS u(t,((x+4)+2)+1) by subst.

        // Example 3:
        // eq1: scr1(x) EQUALS u(t,x+1); <--|
        // eq2: scr2(x) EQUALS u(t,x+2); <--| orig halo of u = max(1,2) = 2.
        // eq3: u(t+1,x) EQUALS scr1(x+3) + scr2(x+4);
        // eq1 and eq2 are bundled => scr1 and scr2 halos are max(3,4) = 4.
        // Direct deps: eq3 -> eq1(s), eq3 -> eq2(s).
        // Halo of u is 4 + 2 = 6.
        // Or, u(t+1,x) EQUALS u(t,(x+3)+1) + u(t,(x+4)+2) by subst.
        
        // Algo: Keep a list of maps of shadow vars.
        // Each map: key=real-var ptr, val=shadow-var ptr.
        // These shadow vars will be used to track
        // updated halos for each path.
        // We don't want to update the real halos until
        // we've walked all the paths using the original
        // halos.
        // At the end, the real vars will be updated
        // from the shadows.
        vector< map<Var*, Var*>> shadows;

        // Stages.
        for (auto& bp : get_all()) {
            auto pname = bp->_get_name();
            auto& pbundles = bp->get_bundles(); // list of bundles.

            // Bundles with their dependency info.
            for (auto& b1 : all_bundles.get_all()) {

                // Only need bundles in this stage.
                if (pbundles.count(b1) == 0)
                    continue;

                // Only need to look at dep paths starting from non-scratch bundles.
                if (b1->is_scratch())
                    continue;

                // We start with each non-scratch bundle and walk the dep
                // tree to find all dependent scratch bundles.  It's
                // important to then visit them in dep order using 'path' to
                // get only unbroken chains of scratch bundles.
#ifdef DEBUG_SCRATCH
                cout << "* c_h: visiting deps of " << b1->get_descr() << endl;
#endif
                all_bundles.get_deps().visit_deps
                
                    // For each 'bn', 'b1' is 'bn' or depends on 'bn',
                    // immediately or indirectly; 'path' leads from
                    // 'b1' to 'bn'.
                    (b1, [&](EqBundlePtr bn, EqBundleList& path) {

                        // Create a new empty map of shadow vars for this path.
                        shadows.resize(shadows.size() + 1);
                        auto& shadow_map = shadows.back();

                        // Walk path from 'b1', stopping at end of scratch
                        // chain.
                        for (auto b2 : path) {

                            // Don't process 'b1', the initial non-scratch bundle.
                            if (b2 == b1)
                                continue;
                        
                            // If this isn't a scratch bundle, we are done
                            // w/this path because we only want the bundles
                            // from 'b1' through an *unbroken* chain of
                            // scratch bundles.
                            if (!b2->is_scratch())
                                break;

                            // Make shadow copies of all vars touched by 'eq2'.
                            // All changes will be applied to these shadow vars
                            // for the current 'path'.
                            for (auto& eq : b2->get_eqs()) {

                                // Output var.
                                auto* og = pv.get_output_vars().at(eq.get());
                                if (shadow_map.count(og) == 0)
                                    shadow_map[og] = new Var(*og);

                                // Input vars.
                                auto& in_pts = pv.get_input_pts().at(eq.get());
                                for (auto* ip : in_pts) {
                                    auto* ig = ip->_get_var();
                                    if (shadow_map.count(ig) == 0)
                                        shadow_map[ig] = new Var(*ig);
                                }
                            }
                        
                            // For each scratch bundle, set the size of all its
                            // output vars' halos to the max across its
                            // halos. We need to do this because halos are
                            // written in a scratch var.  Since they are
                            // bundled, all the writes must be over the same
                            // area.

                            // First, set first eq halo the max of all.
                            auto& eq1 = b2->get_eqs().front();
                            auto* og1 = shadow_map[eq1->_get_var()];
                            for (auto& eq2 : b2->get_eqs()) {
                                if (eq1 == eq2)
                                    continue;

                                // Adjust g1 to max(g1, g2).
                                auto* og2 = shadow_map[eq2->_get_var()];
                                og1->update_halo(*og2);
                            }

                            // Then, update all others based on first.
                            for (auto& eq2 : b2->get_eqs()) {
                                if (eq1 == eq2)
                                    continue;

                                // Adjust g2 to g1.
                                auto* og2 = shadow_map[eq2->_get_var()];
                                og2->update_halo(*og1);
                            }

                            // Get updated halos from the scratch bundle.  These
                            // are the points that are read from the dependent
                            // eq(s).  For scratch vars, the halo areas must
                            // also be written to.
                            auto left_ohalo = og1->get_halo_sizes(pname, true);
                            auto right_ohalo = og1->get_halo_sizes(pname, false);
                            auto l1_dist = og1->get_l1_dist();

#ifdef DEBUG_SCRATCH
                            cout << "** c_h: processing " << b2->get_descr() << "...\n" 
                                "*** c_h: LHS halos: " << left_ohalo.make_dim_val_str() <<
                                " & " << right_ohalo.make_dim_val_str() << endl;
#endif
                        
                            // Recalc min halos of all input vars of all
                            // scratch eqs in this bundle by adding size of
                            // output-var halos.
                            for (auto& eq : b2->get_eqs()) {
                                auto& in_pts = pv.get_input_pts().at(eq.get());

                                // Input points.
                                for (auto ip : in_pts) {
                                    auto* ig = shadow_map[ip->_get_var()];
                                    auto& ao = ip->get_arg_offsets(); // e.g., '2' for 'x+2'.

                                    // Increase range by subtracting left halos and
                                    // adding right halos.
                                    auto left_ihalo = ao.sub_elements(left_ohalo, false);
                                    ig->update_halo(pname, left_ihalo);
                                    auto right_ihalo = ao.add_elements(right_ohalo, false);
                                    ig->update_halo(pname, right_ihalo);
                                    ig->update_l1_dist(l1_dist);
#ifdef DEBUG_SCRATCH
                                    cout << "*** c_h: updated min halos of '" << ig->get_name() << "' to " <<
                                        left_ihalo.make_dim_val_str() <<
                                        " & " << right_ihalo.make_dim_val_str() << endl;
#endif
                                } // input pts.
                            } // eqs in bundle.
                        } // path.
                    }); // lambda fn.
            } // bundles.
        } // stages.

        // Apply the changes from the shadow vars.
        // This will result in the vars containing the max
        // of the shadow halos.
        for (auto& shadow_map : shadows) {
#ifdef DEBUG_SCRATCH
            cout << "* c_h: applying changes from a shadow map...\n";
#endif
            for (auto& si : shadow_map) {
                auto* orig_gp = si.first;
                auto* shadow_gp = si.second;
                assert(orig_gp);
                assert(shadow_gp);

                // Update the original.
                orig_gp->update_halo(*shadow_gp);
#ifdef DEBUG_SCRATCH
                cout << "** c_h: updated '" << orig_gp->get_name() << "'.\n";
#endif

                // Release the shadow var.
                delete shadow_gp;
                shadow_map.at(orig_gp) = NULL;
            }
        }
    } // calc_halos().

    // Divide all equations into eq_bundles.
    // Only process updates to vars in 'var_regex'.
    void EqBundles::make_eq_bundles(Eqs& all_eqs,
                                    const CompilerSettings& settings,
                                    ostream& os)
    {
        os << "\nPartitioning " << all_eqs.get_num() << " equation(s) into bundles...\n";
        //auto& step_dim = _dims->_step_dim;

        // Add scratch equations.
        for (auto eq : all_eqs.get_all()) {
            if (eq->is_scratch()) {

                // Add equation.
                add_eq_to_bundle(all_eqs, eq, _basename_default, settings);
            }
        }

        // Make a regex for the allowed vars.
        regex varx(settings._var_regex);

        // Add all remaining equations.
        for (auto eq : all_eqs.get_all()) {

            // Get name of updated var.
            auto gp = eq->_get_var();
            assert(gp);
            string gname = gp->_get_name();

            // Match to varx?
            if (!regex_search(gname, varx))
                continue;

            // Add equation.
            add_eq_to_bundle(all_eqs, eq, _basename_default, settings);
        }

        os << "Collapsing dependencies from equations and finding transitive closure...\n";
        inherit_deps_from(all_eqs);

        os << "Topologically ordering bundles...\n";
        topo_sort();

        // Dump info.
        os << "Created " << get_num() << " equation bundle(s):\n";
        for (auto& eg1 : get_all()) {
            os << " " << eg1->get_descr() << ":\n"
                "  Contains " << eg1->get_num_eqs() << " equation(s).\n"
                "  Updates the following var(s): ";
            int i = 0;
            for (auto* g : eg1->get_output_vars()) {
                if (i++)
                    os << ", ";
                os << g->_get_name();
            }
            os << ".\n";

            // Deps.
            for (auto& eg2 : _deps.get_deps_on(eg1))
                os << "  Dependent on bundle " << eg2->_get_name() << ".\n";
            for (auto& sg : _scratches.get_deps_on(eg1))
                os << "  Requires scratch bundle " << sg->_get_name() << ".\n";
        }

    }

    // Apply optimizations according to the 'settings'.
    void EqBundles::optimize_eq_bundles(CompilerSettings& settings,
                                    const string& descr,
                                    bool print_sets,
                                    ostream& os) {
        // print stats.
        os << "Stats across " << get_num() << " equation-bundle(s):\n";
        string edescr = "for " + descr + " equation-bundle(s)";
        print_stats(os, edescr);

        // Make a list of optimizations to apply to eq_bundles.
        vector<OptVisitor*> opts;

        // CSE.
        if (settings._do_cse)
            opts.push_back(new CseVisitor);

        // Operator combination.
        if (settings._do_comb) {
            opts.push_back(new CombineVisitor);

            // Do CSE again after combination.
            // TODO: do this only if the combination did something.
            if (settings._do_cse)
                opts.push_back(new CseVisitor);
        }

        // Pairs.
        if (settings._do_pairs)
            opts.push_back(new PairingVisitor);

        // Apply opts.
        for (auto optimizer : opts) {

            visit_eqs(optimizer);
            int num_changes = optimizer->get_num_changes();
            string odescr = "after applying " + optimizer->_get_name() + " to " +
                descr + " equation-bundle(s)";

            // Get new stats.
            if (num_changes)
                print_stats(os, odescr);
            else
                os << " No changes " << odescr << '.' << endl;

            delete optimizer;
            optimizer = 0;
        }

        // Reordering. TODO: make this an optimizer.
        if (settings._do_reorder) {
            
            // Create vector info for this eq_bundle.
            // The visitor is accepted at all nodes in the cluster AST;
            // for each var access node in the AST, the vectors
            // needed are determined and saved in the visitor.
            VecInfoVisitor vv(*_dims);
            visit_eqs(&vv);

            // Reorder some equations based on vector info.
            ExprReorderVisitor erv(vv);
            visit_eqs(&erv);

            // Get new stats.
            string odescr = "after applying reordering to " +
                descr + " equation-bundle(s)";
            print_stats(os, odescr);
        }

        // Final stats per equation bundle.
        if (print_sets && get_num() > 1) {
            os << "Stats per equation-bundle:\n";
            for (auto eg : get_all())
                eg->print_stats(os, "for " + eg->get_descr());
        }
    }

    // Make a human-readable description of this eq stage.
    string EqStage::get_descr(string quote) const
    {
        string des;
        if (is_scratch())
            des += "scratch ";
        des += "stage " + quote + _get_name() + quote;
        if (!is_scratch()) {
            if (step_cond.get())
                des += " w/step condition " + step_cond->make_quoted_str(quote);
            else
                des += " w/o step condition";
        }
        return des;
    }

    // Add a bundle to this stage.
    void EqStage::add_bundle(EqBundlePtr bp)
    {
        _bundles.insert(bp);
        _is_scratch = bp->is_scratch();

        // update list of eqs.
        for (auto& eq : bp->get_eqs())
            _eqs.insert(eq);

        // update list of input and output vars for this stage.
        for (auto& g : bp->get_output_vars())
            _out_vars.insert(g);
        for (auto& g : bp->get_input_vars())
            _in_vars.insert(g);
    }

    // Add 'bp' from 'all_bundles'. Create new stage if needed.  Returns
    // whether a new stage was created.
    bool EqStages::add_bundle_to_stage(EqBundles& all_bundles,
                                       EqBundlePtr bp)
    {
        // Already added?
        if (_bundles_in_stages.count(bp))
            return false;

        // Get condition, if any.
        auto stcond = bp->step_cond;
        
        // Get deps between bundles.
        auto& deps = all_bundles.get_deps();

        // Loop through existing stages, looking for one that
        // 'bp' can be added to.
        EqStage* target = 0;
        for (auto& ep : get_all()) {

            // Must be same scratch-ness.
            if (ep->is_scratch() != bp->is_scratch())
                continue;

            // Step conditions must match (both may be null).
            if (!are_exprs_same(ep->step_cond, stcond))
                continue;

            // Look for any dependencies that would prevent adding
            // 'bp' to 'ep'.
            bool is_ok = true;
            for (auto& bp2 : ep->get_bundles()) {

                // Look for any dependency between 'bp' and 'bp2'.
                if (deps.is_dep(bp, bp2)) {
                    is_ok = false;
                    break;
                }
            }

            // Remember target if ok and stop looking.
            if (is_ok) {
                target = ep.get();
                break;
            }
        }

        // Make new stage if no target stage found.
        bool new_stage = false;
        if (!target) {
            auto np = make_shared<EqStage>(bp->is_scratch());
            add_item(np);
            target = np.get();
            if (bp->is_scratch())
                target->base_name = string("scratch_") + _base_name;
            else
                target->base_name = _base_name;
            target->index = _idx++;
            target->step_cond = stcond;
            new_stage = true;
        }

        // Add bundle to target.
        assert(target);
        target->add_bundle(bp);

        // Remember stage and updated vars.
        _bundles_in_stages.insert(bp);
        for (auto& g : bp->get_output_vars())
            _out_vars.insert(g);

        return new_stage;
    }

    // Divide all bundles into stages.
    void EqStages::make_stages(EqBundles& all_bundles,
                                  ostream& os)
    {
        os << "\nPartitioning " << all_bundles.get_num() << " bundle(s) into stages...\n";

        for (auto bp : all_bundles.get_all())
            add_bundle_to_stage(all_bundles, bp);

        os << "Collapsing dependencies from bundles and finding transitive closure...\n";
        inherit_deps_from(all_bundles);

        os << "Topologically ordering stages...\n";
        topo_sort();

        // Dump info.
        os << "Created " << get_num() << " equation stage(s):\n";
        for (auto& bp1 : get_all()) {
            os << " " << bp1->get_descr() << ":\n"
                "  Contains " << bp1->get_bundles().size() << " bundle(s): ";
            int i = 0;
            for (auto b : bp1->get_bundles()) {
                if (i++)
                    os << ", ";
                os << b->_get_name();
            }
            os << ".\n";
            os << "  Updates the following var(s): ";
            i = 0;
            for (auto* g : bp1->get_output_vars()) {
                if (i++)
                    os << ", ";
                os << g->_get_name();
            }
            os << ".\n";

            // Deps.
            for (auto& bp2 : _deps.get_deps_on(bp1))
                os << "  Dependent on stage " << bp2->_get_name() << ".\n";
            for (auto& sp : _scratches.get_deps_on(bp1))
                os << "  Requires scratch stage " << sp->_get_name() << ".\n";
        }

    }

} // namespace yask.
