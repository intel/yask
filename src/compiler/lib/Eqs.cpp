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

///////// Methods for equations and stencil parts ////////////

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Eqs.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

//#define DEBUG_HALOS
//#define DEBUG_ADD_EXPRS
//#define DEBUG_ADD_PARTS

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

        PointMap _lhs_pts; // outputs of eqs (1 for each eq).
        PointSetMap _rhs_pts; // inputs of eqs (set for each eq).
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
        virtual string visit(VarPoint* vp) {
            assert(_eq);
            auto* g = vp->_get_var();
            _all_pts[_eq].insert(vp);

            // Save pt and/or var based on state.
            switch (_state) {

            case _in_lhs:
                _lhs_pts[_eq] = vp;
                _lhs_vars[_eq] = g;
                break;

            case _in_rhs:
                _rhs_pts[_eq].insert(vp);
                _rhs_vars[_eq].insert(g);
                break;

            case _in_cond:
                _cond_pts[_eq].insert(vp);
                break;

            case _in_step_cond:
                _step_cond_pts[_eq].insert(vp);
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
        virtual string visit(VarPoint* vp) {
            auto* var = vp->_get_var();

            // Folded dims in the solution.
            int soln_nfd = _dims._fold_gt1.size();
            
            // Folded dims in this var.
            int var_nfd = var->get_num_foldable_dims();
            assert(var_nfd <= soln_nfd);

            // Degenerate case with no folding in soln: we still mark points
            // using vars with some domain dims as vectorizable.
            if (soln_nfd == 0 && var->is_foldable())
                vp->set_vec_type(VarPoint::VEC_FULL);

            // No foldable dims.
            else if (var_nfd == 0)
                vp->set_vec_type(VarPoint::VEC_NONE);

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
                    if (vp->get_arg_offsets().lookup(fdname))
                        fdoffsets++;
                }
                assert(fdoffsets <= var_nfd);

                // All folded dims are vectorizable?
                if (fdoffsets == soln_nfd) {
                    assert(var->is_foldable());
                    vp->set_vec_type(VarPoint::VEC_FULL); // all good.
                }

                // Some dims are vectorizable?
                else if (fdoffsets > 0)
                    vp->set_vec_type(VarPoint::VEC_PARTIAL);

                // No dims are vectorizable.
                else
                    vp->set_vec_type(VarPoint::VEC_NONE);

            }

            // Also check args of this var point.
            return ExprVisitor::visit(vp);
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
        virtual string visit(VarPoint* vp) {

            // Info from var.
            auto* var = vp->_get_var();
            auto gdims = var->get_dim_names();

            // Inner-loop var.
            auto& idim = _settings._inner_loop_dim;

            // Access type.
            // Assume invariant, then check below.
            VarPoint::VarDepType lt = VarPoint::DOMAIN_VAR_INVARIANT;

            // Check every point arg.
            auto& args = vp->get_args();
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
            vp->set_var_dep(lt);
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
        virtual string visit(VarPoint* vp) {

            // Shift var _ofs points.
            auto ofs0 = vp->get_arg_offsets();
            IntTuple new_loc = ofs0.add_elements(_ofs, false);
            vp->set_arg_offsets(new_loc);
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
    // BIG-TODO: replace dependency graph and algorithms with a DAG library.
    // HUGE-TODO: replace dependency algorithms with integration of a polyhedral
    // library.
    void Eqs::analyze_eqs() {
        auto& dims = _soln->get_dims();
        auto& os = _soln->get_ostr();
        auto& settings = _soln->get_settings();
        auto& log_vars = _soln->get_logical_vars();
        auto& step_dim = dims._step_dim;

        // Gather initial stats from all eqs.
        PointVisitor pt_vis;
        visit_eqs(&pt_vis);
        auto& out_vars = pt_vis.get_output_vars();
        auto& in_vars = pt_vis.get_input_vars();
        auto& out_pts = pt_vis.get_output_pts();
        auto& in_pts = pt_vis.get_input_pts();
        //auto& dcond_pts = pt_vis.get_dcond_pts();
        //auto& step_cond_pts = pt_vis.get_step_cond_pts();

        // 1. Check each eq internally.
        os << "\nProcessing " << get_num() << " stencil equation(s)...\n";
        for (auto eq1 : get_all()) {

            if (eq1->is_scratch())
                os << "Scratch equation: " << eq1->get_descr() << endl;
            else
                os << "Equation: " << eq1->get_descr() << endl;
            
            auto* eq1p = eq1.get();
            assert(out_vars.count(eq1p));
            assert(in_vars.count(eq1p));
            auto* ov1 = out_vars.at(eq1p);
            assert(ov1 == eq1->get_lhs_var());
            auto* op1 = out_pts.at(eq1p);
            //auto& ivs1 = in_vars.at(eq1p);
            auto& ips1 = in_pts.at(eq1p);
            auto dcond1 = eq1p->_get_cond();
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
            if (!ov1->is_scratch()) {
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
            for (int di = 0; di < ov1->get_num_dims(); di++) {
                auto& dn = ov1->get_dim_name(di);  // name of this dim.
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
                                             "' where constant integer is expected");
                    argn->get_int_val(); // throws exception if not an integer.
                }
            }
        
            // Set and/or check the step direction.
            if (!ov1->is_scratch()) {

                // See if LHS step arg is a simple offset, e.g., 'u(t+1, ...)'.
                auto& lofss = op1->get_arg_offsets();
                auto* lofsp = lofss.lookup(step_dim); // offset at step dim.
                if (lofsp) {
                    auto lofs = *lofsp;

                    if (abs(lofs) != 1)
                        THROW_YASK_EXCEPTION("LHS of equation " + eq1->make_quoted_str() +
                                             " does not contain an offset of +/-1 from step-dimension '" +
                                             step_dim + "'");
                    
                    // Soln step-direction heuristic used only if not set.
                    // Assume 'u(t+1, ...) EQUALS ...' implies forward,
                    // and 'u(t-1, ...) EQUALS ...' implies backward.
                    if (dims._step_dir == 0 && lofs != 0)
                        dims._step_dir = (lofs > 0) ? 1 : -1;

                    if (lofs != dims._step_dir)
                        THROW_YASK_EXCEPTION("LHS of equation " + eq1->make_quoted_str() +
                                             " does not contain the same offset from step-dimension '" +
                                             step_dim + "' as previous equation(s)");
                    
                    // Scan input (RHS) points.
                    for (auto i1 : ips1) {
                        
                        // Is RHS point a simple offset from step, e.g., 'u(t-2, ...)'?
                        auto* rsi1p = i1->get_arg_offsets().lookup(step_dim);
                        if (rsi1p) {
                            int rofs = *rsi1p;

                            // Examples:
                            // forward: 'u(t+1, ...) EQUALS ... u(t, ...) ...',
                            // forward: 'u(t+1, ...) EQUALS ... u(t-2, ...) ...',
                            // bad forward: 'u(t+1, ...) EQUALS ... u(t+3, ...) ...',
                            if (dims._step_dir == 1 && rofs > lofs) {
                                THROW_YASK_EXCEPTION("equation " + eq1->make_quoted_str() +
                                                     " contains an offset of " + to_string(rofs) +
                                                     " from step-dimension '" + step_dim +
                                                     "' on the RHS, which greater than offset " + to_string(lofs) +
                                                     " on the LHS");
                            }
                            // backward: 'u(t-1, ...) EQUALS ... u(t, ...) ...'.
                            // bad backward: 'u(t-1, ...) EQUALS ... u(t-2, ...) ...'.
                            if (dims._step_dir == -11 && rofs < lofs) {
                                THROW_YASK_EXCEPTION("equation " + eq1->make_quoted_str() +
                                                     " contains an offset of " + to_string(rofs) +
                                                     " from step-dimension '" + step_dim +
                                                     "' on the RHS, which less than offset " + to_string(lofs) +
                                                     " on the LHS (not allowed for reverse-time stencils)");
                            }
                        }
                    } // for all RHS points.
                }
                else {
                    THROW_YASK_EXCEPTION("LHS of equation " + eq1->make_quoted_str() +
                                         " does not contain  an offset of +/-1 from step-dimension '" +
                                         step_dim + "'");
                }                    
            }

            // LHS of equation must be vectorizable.
            // TODO: relax this restriction.
            // TODO: is this needed? seems redundant w/above checks on LHS.
            if (op1->get_vec_type() != VarPoint::VEC_FULL) {
                THROW_YASK_EXCEPTION("LHS of equation " + eq1->make_quoted_str() +
                                     " is not fully vectorizable because not all folded"
                                     " dimensions are accessed via simple offsets from their respective indices");
            }

            // More RHS checks: step & domain indices must be simple offsets and
            // misc indices must be consts.
            for (auto i1 : ips1) {
                auto* iv1 = i1->_get_var();

                for (int di = 0; di < iv1->get_num_dims(); di++) {
                    auto& dn = iv1->get_dim_name(di);  // name of this dim.
                    auto argn = i1->get_args().at(di); // arg for this dim.

                    // Check based on dim type.
                    // Must have simple indices in step dim.
                    if (dn == step_dim) {
                        auto* rsi1p = i1->get_arg_offsets().lookup(dn);
                        if (!rsi1p)
                            THROW_YASK_EXCEPTION("RHS of equation " + eq1->make_quoted_str() +
                                                 " contains expression " + argn->make_quoted_str() +
                                                 " for step-dimension '" + dn +
                                                 "' where constant-integer offset from '" + dn +
                                                 "' is expected");
                    }

                    // Must have simple indices in domain dims.
                    else if (dims._domain_dims.lookup(dn)) {
                        auto* rsi1p = i1->get_arg_offsets().lookup(dn);
                        if (!rsi1p)
                            THROW_YASK_EXCEPTION("RHS of equation " + eq1->make_quoted_str() +
                                                 " contains expression " + argn->make_quoted_str() +
                                                 " for domain-dimension '" + dn +
                                                 "' where constant-integer offset from '" + dn +
                                                 "' is expected");
                    }

                    // Misc dim must be a const.
                    else {
                        if (!argn->is_const_val())
                            THROW_YASK_EXCEPTION("RHS of equation " + eq1->make_quoted_str() +
                                                 " contains expression " + argn->make_quoted_str() +
                                                 " for misc-dimension '" + dn +
                                                 "' where constant integer is expected");
                        argn->get_int_val(); // throws exception if not an integer.
                    }
                }
            } // input pts.

            // TODO: check to make sure dcond1 depends only on domain indices.
            // TODO: check to make sure stcond1 does not depend on domain indices.

            // Find logical var dependencies: LHS var depends on RHS vars.
            auto olv1 = log_vars.add_var_slice(op1);
            for (auto ip1 : ips1) {
                auto ilv1 = log_vars.add_var_slice(ip1);

                // Set dependence.
                log_vars.set_imm_dep_on(olv1, ilv1);
            }

            // Re-process after every equation to assist with debug:
            // exception will be thrown after printing offending eq.
            log_vars.find_all_deps();
            
        } // for all eqs.
        os << endl;

        // Finish processing and dump logical vars.
        log_vars.topo_sort();
        log_vars.print_info();

        // 2. Check each pair of eqs.
        os << "\nAnalyzing equations for dependencies...\n";
        for (auto eq1 : get_all()) {
            auto* eq1p = eq1.get();
            assert(out_vars.count(eq1p));
            assert(in_vars.count(eq1p));
            auto* ov1 = out_vars.at(eq1p);
            assert(ov1 == eq1->get_lhs_var());
            auto* op1 = out_pts.at(eq1p);
            //auto& ivs1 = in_vars.at(eq1p);
            //auto& ips1 = in_pts.at(eq1p);
            auto dcond1 = eq1p->_get_cond();
            auto stcond1 = eq1p->_get_step_cond();
            num_expr_ptr step_expr1 = op1->get_arg(step_dim);
            bool is_scratch1 = op1->_get_var()->is_scratch();

            // Check each 'eq2' to see if it depends on 'eq1'.
            for (auto eq2 : get_all()) {
                if (eq1 == eq2)
                    continue;
                
                auto* eq2p = eq2.get();
                auto& ov2 = out_vars.at(eq2p);
                assert(ov2 == eq2->get_lhs_var());
                auto& op2 = out_pts.at(eq2p);
                auto& ivs2 = in_vars.at(eq2p);
                auto& ips2 = in_pts.at(eq2p);
                auto dcond2 = eq2p->_get_cond();
                auto stcond2 = eq2p->_get_step_cond();

                #ifdef DEBUG_DEP
                cout << " Checking eq " <<
                    eq1->make_quoted_str() << " vs " <<
                    eq2->make_quoted_str() << "...\n";
                #endif

                bool same_op = are_exprs_same(op1, op2);
                bool same_dcond = are_exprs_same(dcond1, dcond2);
                bool same_stcond = are_exprs_same(stcond1, stcond2);

                // Check pairs that have same LHS.
                if (same_op) {

                    string dcdesc = dcond2 ? "with domain condition " + dcond2->make_quoted_str() :
                        "without domain conditions";
                    string stcdesc = stcond2 ? "with step condition " + stcond2->make_quoted_str() :
                        "without step conditions";
                    string cdesc = dcdesc + " and " + stcdesc;
                    
                    // If 2 different eqs have the same or no conditions, they
                    // cannot have the same LHS.
                    if (same_dcond && same_stcond) {
                        THROW_YASK_EXCEPTION("two equations, both with " + cdesc +
                                             ", have the same LHS: " +
                                             eq1->make_quoted_str() + " and " +
                                             eq2->make_quoted_str());
                    }

                    // If 1 eq has no conditions, and another has any conditions,
                    // they cannot have the same LHS.
                    if (is_expr_null(dcond1) && is_expr_null(stcond1) &&
                        (!is_expr_null(dcond2) || !is_expr_null(stcond2))) {
                        THROW_YASK_EXCEPTION("equation " + eq1->make_quoted_str() +
                                             " without domain or step conditions and equation " +
                                             eq2->make_quoted_str() + " " + cdesc +
                                             " have the same LHS");
                    }

                    // NB: if both of the preceding tests pass, we could still
                    // have two equations with different conditions updating the
                    // same logical var.  We will assume the stencil programmer
                    // has ensured they do not overlap.  TODO: check this at
                    // compile-time (difficult) or at run-time (easier, but not
                    // as nice for stencil programmer.)
                }

                // Stop here if not looking for deps.
                if (!settings._find_deps)
                    continue;

                /* Use logical-var dependencies to determine if eq2
                   depends on eq1.
                  
                   Example of non-scratch var dependency:
                    eq1: a(t+1, x, ...) EQUALS ...
                    eq2: b(t+1, x, ...) EQUALS a(t+1, x+5, ...) ...
                    eq2 depends on eq1 because b(t+1) depends on a(t+1).
                  
                   Example of dependency through scratch var:
                    eq1: tmp(x, ...) EQUALS ...
                    eq2: b(t+1, x, ...) EQUALS tmp(x+2, ...)
                    eq2 depends on eq1 because b(t+1) depends on tmp.
                */
                auto olv1 = log_vars.find_var_slice(op1);
                assert(olv1);
                auto olv2 = log_vars.find_var_slice(op2);
                assert(olv2);
                if (log_vars.get_deps().is_dep_on(olv2, olv1))
                    set_imm_dep_on(eq2, eq1);
                
            } // for all eqs (eq2).

            // Re-process dependencies after every equation to assist with
            // debug: exception will be thrown after printing offending eq.
            find_all_deps();
            
        } // for all eqs (eq1).

        // If the step dir wasn't set (no eqs), set it now.
        if (!dims._step_dir)
            dims._step_dir = 1;

        #ifdef DEBUG_DEP
        cout << "Dependencies for all eqs:\n";
        _deps.print_deps(cout);
        cout << "Dependencies from non-scratch to scratch eqs:\n";
        _scratches.print_deps(cout);

        // Resolve indirect dependencies.
        // Do this even if not finding deps because we want to
        // process deps provided by the user.
        os << "Finding transitive closure of dependencies...\n";
        find_all_deps();

        // Sort.
        os << "Topologically ordering equations...\n";
        topo_sort();
        #endif
    }

    // Determine which var points can be vectorized.
    void Eqs::analyze_vec() {
        auto& dims = _soln->get_dims();

        // Send a 'SetVecVisitor' to each point in
        // the current equations.
        SetVecVisitor svv(dims);
        visit_eqs(&svv);
    }

    // Determine loop access behavior of var points.
    void Eqs::analyze_loop() {
        auto& dims = _soln->get_dims();
        auto& settings = _soln->get_settings();

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

    void LogicalVars::print_info() const {
        auto& os = _soln->get_ostr();

        os << "Found " << get_num() << " logical var slice(s):\n";
        for (auto& lv1 : get_all()) {
            if (lv1->get_var().is_scratch())
                os << " scratch";
            if (lv1->get_var().get_num_dims() == 0)
                os << " scalar";
            else
                os << " var slice";
            os << " '" << lv1->get_descr() << "':\n";

            // Deps.
            auto& id = get_imm_deps_on(lv1);
            for (auto& lv2 : id)
                os << "  Immediately dependent on '" << lv2->get_descr() << "'.\n";
            for (auto& lv2 : get_all_deps_on(lv1))
                if (id.count(lv2) == 0)
                    os << "   Indirectly dependent on '" << lv2->get_descr() << "'.\n";
            if (get_all_deps_on(lv1).size() == 0)
                os << "  No dependencies within a step.\n";
        }
    }

    // Get the full name of an eq-lot.
    // Must be unique.
    string EqLot::_get_name() const {

        // Add index to base name.
        return base_name + "_" + to_string(index);
    }

    // Add an eq to an EqLot.
    void EqLot::add_eq(equals_expr_ptr ee) {
        _eqs.insert(ee);

        // Get I/O point data from eq 'ee'.
        PointVisitor pv;
        ee->accept(&pv);
            
        // update list of input and output vars.
        auto* out_var = pv.get_output_vars().at(ee.get());
        _out_vars.insert(out_var);
        auto& in_vars = pv.get_input_vars().at(ee.get());
        for (auto* g : in_vars)
            _in_vars.insert(g);

        // check scratch-ness.
        assert(out_var->is_scratch() == _is_scratch);
    }

    // Remove an eq from an EqLot.
    void EqLot::remove_eq(equals_expr_ptr ee) {
        _eqs.erase(ee);

        // Update list of input and output vars.
        // Must clear and re-add all because more than one
        // eq can effect the same vars.
        _out_vars.clear();
        _in_vars.clear();
        for (auto& eq : _eqs) {

            // Get I/O point data from 'eq'.
            PointVisitor pv;
            eq->accept(&pv);
            
            auto* out_var = pv.get_output_vars().at(eq.get());
            _out_vars.insert(out_var);
            auto& in_vars = pv.get_input_vars().at(eq.get());
            for (auto* g : in_vars)
                _in_vars.insert(g);
        }
    }

    // Print stats from eqs.
    void EqLot::print_stats(ostream& os, const string& msg) {
        CounterVisitor cv;
        visit_eqs(&cv);
        cv.print_stats(os, msg);
    }

    // Make a human-readable description of this part.
    string Part::get_descr(bool show_cond,
                               string quote) const
    {
        string des;
        if (is_scratch())
            des += "scratch ";
        des += "part " + quote + _get_name() + quote;
        if (show_cond) {
            if (cond.get())
                des += " w/domain condition " + cond->make_quoted_str(quote);
            else
                des += " w/o domain condition";
            des += " and";
            if (step_cond.get())
                des += " w/step condition " + step_cond->make_quoted_str(quote);
            else
                des += " w/o step condition";
        }
        return des;
    }

    // Print stats from eqs in parts.
    void Parts::print_stats(const string& msg) {
        auto& os = _soln->get_ostr();
        CounterVisitor cv;

        // Use separate counter visitor for each part
        // to avoid treating repeated eqs as common sub-exprs.
        for (auto& eq : get_all()) {
            CounterVisitor ecv;
            eq->visit_eqs(&ecv);
            cv += ecv;
        }
        cv.print_stats(os, msg);
    }

    // Add 'eq' to an existing part if possible.  If not possible,
    // create a new part and add 'eqs' to it. The index will be
    // incremented if a new part is created.  Returns whether a new part
    // was created.
    bool Parts::add_eq_to_part(Eq& eq)
    {
        auto& dims = _soln->get_dims();
        auto& all_eqs = _soln->get_eqs();
        auto& settings = _soln->get_settings();
        
        /*
          part scenarios:

          Normal sequences of non-scratch parts:
          Ex 0: eq B depends on eq A.
          - p1 updates A.
          - p2 updates B & depends on p1.
          Ex 1: P1 & P2 depend on A1 & A2.
          - p1 updates A1 & A2 b/c A1 & A2 are independent.
          - p2 updates P1 & P2 & depends on p1 b/c P1 & P2 are independent.
          Ex 2 w/partitions (sub-domains): B depends on A in both parts.
          - p1 updates A(part 1).
          - p2 updates A(part 2).
          - p3 updates B(part 1) & depends on p1 & p2.
          - p4 updates B(part 2) & depends on p1 & p2.
          Ex 3 w/partitions: B depends on A, but only in part 2.
          - p1 updates A(part 1) & B(part 1) b/c A & B are independent in part 1.
          - p2 updates A(part 2).
          - p3 updates B(part 2) & depends on p1 & p2.
          Writes to non-scratch vars are limited by domain.

          Normal sequences with scratch-parts (updating Tn):
          Ex s0: A depends on T1.
          - p1 updates T1.
          - p2 updates A & depends on p1.
          Ex s1: T1 & T2 are independent; A depends on T1 and T2.
          - p1 updates T1 & T2.
          - p2 updates A & depends on p1.
          Write halos for T1 & T2 will be made equal.
          Ex s2: T2 depends on T1; A depends on T2.
          - p1 updates T1.
          - p2 updates T2 & depends on p1.
          - p3 updates A & depends on p2.
          Write halo for T1 is typically made larger than T2 b/c T2 is updated from a 
          stencil on T1.
          Ex s3 w/partitions; T1 & T2 are independent; A depends on T1 and T2.
          - p1 updates T1(part 1) & T2(part 1) b/c T1 & T2 are independent in part 1.
          - p2 updates T1(part 2) & T2(part 2) b/c T1 & T2 are independent in part 2.
          - p3 updates A & depends on p1-2.
          Write halos for T1 & T2 will be made equal.
          This is okay b/c p2 is independent of p1.

          Special consideration:
          Need to avoid splitting partial updates to a scratch var across dependent
          parts, e.g., situation in Ex s3, except T2(only part 2) depends on T1:

          Incorrect:
          - p1 updates T1(part 1) & T2(part 1) b/c T1 & T2 are independent in part 1.
          - p2 updates T1(part 2).
          - p3 updates T2(part 2) & depends on p1 & p2.
          - p4 updates A & depends on p1-3.

          This is bad because the write halo for updates in p1 would
          typically need to be larger than that of p3, but they are both
          defined by the halo of T2.

          Correct:
          - p1 updates T1(part 1).
          - p2 updates T2(part 1): not bundled w/p1 b/c of dependency of T2(*part 2*) on T1.
          - p3 updates T1(part 2).
          - p4 updates T2(part 2) & depends on p1 & p3.
          - p5 updates A & depends on p1-4.
          Write halo for T1 is typically made larger than T2 as in Ex s2.

          Even though it's not illegal, it's also better not to bundle
          partial updates of non-scratch logical vars inconsistently. This
          avoids updating a sub-domain of a var early and then more of it [much]
          later, which is bad for cache locality.
        */

        // Equation already added?
        if (_eqs_in_parts.count(eq))
            return false;

        auto& step_dim = dims._step_dim;

        // Get deps between eqs.
        auto& eq_deps = all_eqs.get_deps();
        
        // Get conditions, if any.
        auto cond = eq->_get_cond();
        auto stcond = eq->_get_step_cond();

        // Get LHS info.
        auto eqv = eq->get_lhs_var();
        assert (eqv);
        bool eq_scratch = eq->is_scratch();
        auto eqp = eq->_get_lhs();
        assert (eqp);
        auto step_expr = eq->_get_lhs()->get_arg(step_dim); // may be null.
        #ifdef DEBUG_ADD_EXPRS
        cout << "* ae2b: bundling " << eq->make_quoted_str() << endl;
        #endif

        // Targeted part to add 'eq' to.
        PartPtr target = 0;
        if (settings._bundle) {
         
            // To avoid inconsistent bundling of var updates as described
            // above, find *all* other eqs that update the same var as 'eq'
            // if scratch or same logical var if non-scratch.
            EqList rel_eqs;
            for (auto& eq2 : all_eqs.get_all()) {
                if (eq == eq2)
                    continue;
                auto eq2v = eq2->get_lhs_var();
                auto eq2p = eq2->_get_lhs();
                if ((eq_scratch && eqv == eq2v) ||
                    (!eq_scratch && eqp->is_same_logical_var(*eq2p)))
                    rel_eqs.insert(eq2);
            }
            #ifdef DEBUG_ADD_EXPRS
            cout << "** ae2b: will check " << rel_eqs.size() << " related eqs\n";
            #endif
            
            // Loop through existing parts, looking for one that
            // 'eq' can be added to.
            for (auto& p : get_all()) {
                #ifdef DEBUG_ADD_EXPRS
                cout << "** ae2b: checking " << b->get_descr() << endl;
                #endif
                bool is_ok = true;

                // Look for any condition or dependencies that would prevent
                // adding 'eq' to 'p'.

                // Must be same scratch-ness.
                if (p->is_scratch() != eq->is_scratch())
                    is_ok = false;

                // Domain & step conditions must match (both may be null).
                else if (!are_exprs_same(p->cond, cond))
                    is_ok = false;
                else if (!are_exprs_same(p->step_cond, stcond))
                    is_ok = false;

                // LHS step exprs must match (both may be null for scratch updates).
                else if (!are_exprs_same(p->step_expr, step_expr))
                    is_ok = false;

                // Loop over all eqs already in 'b'.
                if (is_ok) {
                    for (auto& eq2 : p->get_eqs()) {
                        auto eq2v = eq2->get_lhs_var();
                        assert (eq2v);

                        // If scratch, 'eq' and 'eq2' must have same dims and
                        // same halo.  This is because scratch halos are written
                        // to.  If dims are same, halos can be adjusted if
                        // allowed.
                        if (eq->is_scratch()) {
                            if (!eq2v->are_dims_same(*eqv))
                                is_ok = false;
                            else if (!settings._bundle_scratch && !eq2v->is_halo_same(*eqv))
                                is_ok = false;
                        }

                        // Look for any dependency between 'eq' and 'eq2'.
                        if (is_ok && eq_deps.is_dep(eq, eq2)) {
                            #ifdef DEBUG_ADD_EXPRS
                            cout << "*** ae2b: NOT adding eq because of dependency w/eq " <<
                                eq2->make_quoted_str() << endl;
                            #endif
                            is_ok = false;
                        }

                        // Check against other eqs that update the same var.
                        if (is_ok) {
                            for (auto& eq3 : rel_eqs) {
                                #ifdef DEBUG_ADD_EXPRS
                                cout << "** ae2b: checking related eq " << eq3->get_descr() << endl;
                                #endif

                                // Look for any dependency between 'eq2' and 'eq3'.
                                if (eq_deps.is_dep(eq2, eq3)) {
                                    #ifdef DEBUG_ADD_EXPRS
                                    cout << "*** ae2b: NOT adding eq because of dependency of related eq w/eq " <<
                                        eq2->make_quoted_str() << endl;
                                    #endif
                                    is_ok = false;
                                    break;
                                }
                            }
                        }

                        if (!is_ok)
                            break;
                    } // eqs in part.
                } // if ok.
                    
                // Remember target part if ok and stop looking.
                // Try to add if ok.
                if (is_ok) {
                    try {

                        // Add eq.
                        p->add_eq(eq);

                        // Check dependencies between updated parts.
                        inherit_deps_from(all_eqs);

                        // If we get this far, the add was okay;
                        // remove the temp deps, and we're done.
                        clear_deps();
                        target = p;
                        #ifdef DEBUG_ADD_EXPRS
                        cout << "** ae2b: all checks passed: added to existing part\n";
                        #endif
                        break; // out of existing part loop.
                    }
                    catch (yask_exception& e) {

                        // Failure; must remove part.
                        p->remove_eq(eq);
                        
                        // Also remove the bad dependencies.
                        clear_deps();
                    }
                }
                else {
                    #ifdef DEBUG_ADD_EXPRS
                    cout << "** ae2b: NOT adding equation to existing part\n";
                    #endif
                }
                
            } // existing parts.
        } // if bundling.
        
        // Make new part if no target part found.
        bool new_part = false;
        if (!target) {
            auto ne = make_shared<Part>(_soln, eq->is_scratch());
            add_item(ne);
            target = ne;
            if (eq->is_scratch())
                target->base_name = string("scratch_") + _base_name;
            else
                target->base_name = _base_name;
            target->index = _idx++;
            target->cond = cond;
            target->step_cond = stcond;
            target->step_expr = step_expr;
            new_part = true;

            // Add eq to target part.
            #ifdef DEBUG_ADD_EXPRS
            cout << "** ae2b: adding to new part " << target->get_descr() << endl;
            #endif
            target->add_eq(eq);
        }

        // Remember eq and updated var.
        _eqs_in_parts.insert(eq);
        _out_vars.insert(eq->get_lhs_var());

        return new_part;
    }

    // Divide all equations into parts.
    // Only process updates to vars in 'var_regex'.
    void Parts::make_parts()
    {
        auto& dims = _soln->get_dims();
        auto& all_eqs = _soln->get_eqs();
        auto& settings = _soln->get_settings();
        auto& os = _soln->get_ostr();
        
        os << "\nBundling " << all_eqs.get_num() << " equation(s) into solution parts...\n";
        
        // Make a regex for the allowed vars.
        regex varx(settings._var_regex);

        // Add non-scratch, then scratch eqs.
        // This is done just to give the non-scratch ones lower indices.
        for (bool do_scratch : { false, true }) {
            for (auto eq : all_eqs.get_all()) {
                if (eq->is_scratch() != do_scratch)
                    continue;

                // Check name if not scratch.
                if (!eq->is_scratch()) {

                    // Get name of updated var.
                    auto vname = eq->get_lhs_var()->_get_name();
                    
                    // Match to varx?
                    if (!regex_search(vname, varx)) {
                        os << "Equation updating '" << vname <<
                            "' not added because it does not match regex '" << settings._var_regex << "'.\n";
                        continue;
                    }
                }

                // Add equation(s).
                add_eq_to_part(eq);
            }
        }

        os << "Collapsing dependencies from equations and finding transitive closure...\n";
        inherit_deps_from(all_eqs);

        os << "Topologically ordering parts...\n";
        topo_sort();

        // Dump info.
        os << "Created " << get_num() << " solution part(s):\n";
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
            auto& id = get_imm_deps_on(eg1);
            for (auto& eg2 : id)
                os << "  Immediately dependent on " << eg2->_get_name() << ".\n";
            for (auto& eg2 : get_all_deps_on(eg1))
                if (id.count(eg2) == 0)
                    os << "   Indirectly dependent on " << eg2->_get_name() << ".\n";
            for (auto& sg : get_all_scratch_deps_on(eg1))
                os << "  Requires " << sg->_get_name() << ".\n";
            if (get_all_deps_on(eg1).size() == 0)
                os << "  No dependencies within a step.\n";
        }
    }

    // Apply optimizations according to the 'settings'.
    void Parts::optimize_parts(const string& descr)
    {
        auto& all_eqs = _soln->get_eqs();
        auto& settings = _soln->get_settings();
        auto& os = _soln->get_ostr();
        auto& dims = _soln->get_dims();

        // print stats.
        os << "\nStats across " << get_num() << " part(s):\n";
        string edescr = "for " + descr + " part(s)";
        print_stats(edescr);

        // Make a list of optimizations to apply to parts.
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
                descr + " part(s)";

            // Get new stats.
            if (num_changes)
                print_stats(odescr);
            else
                os << " No changes " << odescr << '.' << endl;

            delete optimizer;
            optimizer = 0;
        }

        // Reordering. TODO: make this an optimizer.
        if (settings._do_reorder) {
            
            // Create vector info for this part.
            // The visitor is accepted at all nodes in the AST;
            // for each var access node in the AST, the vectors
            // needed are determined and saved in the visitor.
            VecInfoVisitor vv(dims);
            visit_eqs(&vv);

            // Reorder some equations based on vector info.
            ExprReorderVisitor erv(vv);
            visit_eqs(&erv);

            // Get new stats.
            string odescr = "after applying reordering to " +
                descr + " part(s)";
            print_stats(odescr);
        }

        // Final stats per equation part.
        if (get_num() > 1) {
            os << "Stats per part:\n";
            for (auto eg : get_all())
                eg->print_stats(os, "for " + eg->get_descr());
        }
    }

    // Make a human-readable description of this eq stage.
    string Stage::get_descr(string quote) const
    {
        string des;
        if (is_scratch())
            des += "scratch ";
        des += "stage " + quote + _get_name() + quote;
        return des;
    }

    // Add a part to this stage.
    void Stage::add_part(PartPtr pp)
    {
        _parts.insert(pp);

        // update list of eqs.
        for (auto& eq : pp->get_eqs())
            add_eq(eq);
        assert(is_scratch() == pp->is_scratch());
    }

    // Remove a part from this stage.
    void Stage::remove_part(PartPtr pp)
    {
        _parts.erase(pp);

        // update list of eqs.
        for (auto& eq : pp->get_eqs())
            remove_eq(eq);
    }

    // Add 'pps', a subset of 'all_parts'. Create new stage if needed.
    // Returns whether a new stage was created.
    bool Stages::add_parts_to_stage(Parts& all_parts,
                                    PartList& pps,
                                    bool var_grouping,
                                    bool logical_var_grouping)
    {
        // None to add?
        if (pps.size() == 0)
            return false;

        #ifdef DEBUG_ADD_PARTS
        cout << "** Adding " << pps.size() << " part(s):\n";
        for (auto& pp : pps)
            cout << "*** " << pp->get_descr() << "\n";
        cout << flush;
        #endif
            
        // Already added?
        // (All pps should be added or not by construction.)
        if (_parts_in_stages.count(pps.front()))
            return false;

       // Get scratch-ness.
        // (All pps should have same scratch-ness by construction.)
        auto is_scratch = pps.front()->is_scratch();
        
        // Get deps between parts.
        auto& deps = all_parts.get_deps();

        // Loop through existing stages, looking for one that
        // 'pps' can be added to.
        Tp<Stage> target = 0;
        for (auto& st : get_all()) {
            #ifdef DEBUG_ADD_PARTS
            cout << "*** Checking against existing " << st->get_descr() << "\n" << flush;
            #endif

            // Must be same scratch-ness.
            if (st->is_scratch() != is_scratch)
                continue;

            // Var matching. Look for any match.
            if (logical_var_grouping || var_grouping) {
                bool match_found = false;
                for (auto& pp1 : pps) {
                    for (auto& eq1 : pp1->get_eqs()) {
                        auto lhs1 = eq1->_get_lhs();
                        auto v1 = lhs1->_get_var();
                        for (auto& p2 : st->get_parts()) {
                            for (auto& eq2 : p2->get_eqs()) {
                                auto lhs2 = eq2->_get_lhs();
                                auto* v2 = lhs2->_get_var();
                                if (logical_var_grouping) {
                                    if (lhs1->is_same_logical_var(*lhs2)) {
                                        match_found = true;
                                        goto match_loop_end;
                                    }
                                }
                                else if (v1 == v2) {
                                    match_found = true;
                                    goto match_loop_end;
                                }
                            }
                        }
                    }
                }
                
                // Yes, we know gotos are considered harmful, but C++
                // doesn't have a cleaner way to exit multiple loop levels.
            match_loop_end:
                if (!match_found)
                    continue;
                #ifdef DEBUG_ADD_PARTS
                cout << "*** Passed var-matching: " << st->get_descr() << "\n" << flush;
                #endif
            }
            
            // Loop through all pps. All pps must be able to
            // be added to 'st' to use it.
            bool is_ok = true;
            for (auto& pp : pps) {
                #ifdef DEBUG_ADD_PARTS
                cout << "**** checking " << pp->get_descr() << endl << flush;
                #endif

                // Loop through all parts in 'st'.
                for (auto& p2 : st->get_parts()) {
                    #ifdef DEBUG_ADD_PARTS
                    cout << "**** checking against " << pp->get_descr() << " in " << st->get_descr() << endl << flush;
                    #endif

                    // Look for any dependency between 'pp' and 'p2'.
                    if (deps.is_dep(pp, p2)) {
                        is_ok = false;
                        break;
                    }
                }
                if (!is_ok)
                    break;

                // Try to add 'pp' to 'st'.
                try {

                    // Add it.
                    st->add_part(pp);

                    // Check dependencies between updated stages.
                    inherit_deps_from(all_parts);

                    // If we get this far, the add was okay.
                }
                catch (yask_exception& e) {

                    // A circular-dependency was detected.
                    is_ok = false;
                }

                // Remove the trial part and temp deps.
                st->remove_part(pp);
                clear_deps();

                if (!is_ok)
                    break;

            } // each pp in pps.

            // Found a viable target stage?
            if (is_ok) {
                target = st;
                #ifdef DEBUG_ADD_PARTS
                cout << "*** Adding to existing " << st->get_descr() << "\n" << flush;
                #endif
                break;
            }
            
        } // existing stages.

        // Make new stage if no target stage found.
        bool new_stage = false;
        if (!target) {
            auto np = make_shared<Stage>(_soln, is_scratch);
            add_item(np);
            target = np;
            if (is_scratch)
                target->base_name = string("scratch_") + _base_name;
            else
                target->base_name = _base_name;
            target->index = _idx++;
            new_stage = true;
            #ifdef DEBUG_ADD_PARTS
            cout << "*** Adding to new " << target->get_descr() << "\n" << flush;
            #endif
        }
        assert(target);
        
        // Add parts to target.
        for (auto& pp : pps) {
            target->add_part(pp);

            // Remember parts and updated vars.
            _parts_in_stages.insert(pp);
            for (auto& g : pp->get_output_vars())
                _out_vars.insert(g);
        }

        return new_stage;
    }

    // Divide all parts into stages.
    void Stages::make_stages(Parts& all_parts)
    {
        auto& os = _soln->get_ostr();
        
        os << "\nBundling " << all_parts.get_num() << " part(s) into stages...\n";

        // Temp stages with grouped-by-logical-var stages.
        Stages sts1(_soln);
        for (auto& b0 : all_parts.get_all()) {

            // Make trivial list with 1 part.
            PartList bl0;
            bl0.insert(b0);

            // Group by logical vars.
            sts1.add_parts_to_stage(all_parts, bl0, true, true);
        }
        os << "Found " << sts1.get_num() << " groups of part(s) by logical var(s).\n";

        // Temp stages with grouped-by-var stages.
        Stages sts2(_soln);
        for (auto& st1 : sts1.get_all()) {
            auto& bl1 = st1->get_parts();

            // Group by vars.
            sts2.add_parts_to_stage(all_parts, bl1, true, false);
        }
        os << "Found " << sts1.get_num() << " groups of part(s) by var(s).\n";

        // Finally, make stages with any non-dependency.
        // Add non-scratch, then scratch parts.
        // This is done just to give the non-scratch ones lower indices.
        for (bool do_scratch : { false, true }) {
            for (auto& st2 : sts2.get_all()) {
                if (st2->is_scratch() != do_scratch)
                    continue;
                
                auto& bl2 = st2->get_parts();
                add_parts_to_stage(all_parts, bl2, false, false);
            }
        }
        os << "Created " << get_num() << " equation stage(s).\n";

        os << "Collapsing dependencies from parts and finding transitive closure...\n";
        inherit_deps_from(all_parts);

        os << "Topologically ordering stages...\n";
        topo_sort();

        for (auto& st1 : get_all()) {
            os << " " << st1->get_descr() << ":\n"
                "  Contains " << st1->get_parts().size() << " part(s): ";
            int i = 0;
            for (auto p : st1->get_parts()) {
                if (i++)
                    os << ", ";
                os << p->_get_name();
            }
            os << ".\n";
            os << "  Updates the following var(s): ";
            i = 0;
            for (auto* v : st1->get_output_vars()) {
                if (i++)
                    os << ", ";
                os << v->_get_name();
            }
            os << ".\n";

            // Deps.
            auto& id = get_imm_deps_on(st1);
            for (auto& st2 : id)
                os << "  Immediately dependent on " << st2->_get_name() << ".\n";
            for (auto& st2 : get_all_deps_on(st1))
                if (id.count(st2) == 0)
                    os << "   Indirectly dependent on " << st2->_get_name() << ".\n";
            for (auto& st2 : get_all_scratch_deps_on(st1))
                os << "  Requires " << st2->_get_name() << ".\n";
            if (get_all_deps_on(st1).size() == 0)
                os << "  No dependencies within a step.\n";
        }

    }

    // Find halos needed for each var.
    // These are read halos for non-scratch vars and
    // read/write halos for scratch vars.
    // Halos are tracked for each left/right dim separately.
    // They are also tracked by their non-scratch stage and the
    // step offset when applicable. This info is needed to properly
    // determine whether memory can be reused.
    // Also updates write points in vars.
    void Stages::calc_halos(Parts& all_parts) {
        auto& os = _soln->get_ostr();
        os << "Calculating halos...\n";

        // Find all LHS and RHS points and vars for all eqs.
        PointVisitor pv;
        visit_eqs(&pv);

        #ifdef DEBUG_HALOS
        cout << "* c_h: analyzing " << get_all().size() << " eqs...\n";
        #endif

        /* Phase 1:
           First, set halos based only on immediate read accesses.
           Halo data is keyed by the name of the
           non-scratch stage it is accessed from.
        
           Example1:
           eq: A(t+1, x) EQUALS A(t, x+3).
           So, A's RHS 'x' halo for is 3.
           Since A is read and written in the same stage,
           we need to keep 2 steps of A in memory.

           Example2:
           eq1 in stage1: A(t+1, x) EQUALS B(t, x+3);
           eq2 in stage2: B(t+1, x) EQUALS A(t, x-2);
           B's RHS 'x' halo is 3 for stage1.
           A's LHS 'x' halo is 2 for stage2.
           Since B is read and written in different stages,
           we can optimize storage by keeping only 1 step
           of B in memory; same for A.

           The memory-optimization is done elsewhere, and
           it is based on the data collected here.
        */
        
        // Loop thru stages.
        for (auto& st : get_all()) {

            // Only need to start from non-scratch stages.
            if (st->is_scratch())
                continue;

            auto stname = st->_get_name();
            #ifdef DEBUG_HALOS
            cout << "* c_h, phase 1: analyzing stage '" << stname << "'" << endl;
            #endif

            // A list of this stage and required ones.
            vector<StagePtr> stages;
            stages.push_back(st);

            // Add required scratch stage(s).
            for (auto& ss : get_all_scratch_deps_on(st))
                stages.push_back(ss);
  
            // Loop thru reqd stages.
            for (auto& rst : stages) {
                #ifdef DEBUG_HALOS
                cout << "** c_h, phase 1: reqd stage '" << rst->_get_name() << "'" << endl;
                #endif

                // Loop thru equations in this stage.
                // Note that a scratch eq may be reqd from
                // more than one non-scratch stage.
                for (auto& eq : rst->get_eqs()) {
                    #ifdef DEBUG_HALOS
                    cout << "*** c_h: eq " << eq->make_quoted_str() << endl;
                    #endif

                    // Get all var points touched by this eq.
                    auto& all_pts1 = pv.get_all_pts().at(eq.get());
                    auto& out_pt1 = pv.get_output_pts().at(eq.get());

                    // Update stats of each var accessed in 'eq'.
                    for (auto ap : all_pts1) {
                        auto* g = ap->_get_var(); // var for point 'ap'.

                        // Track halos by non-scratch stage name.
                        auto changed = g->update_halo(stname, ap->get_arg_offsets());
                        #ifdef DEBUG_HALOS
                        if (changed) {
                            cout << "**** c_h: updated halos:\n";
                            g->print_halos(cout, "***** c_h: ");
                        }
                        #endif
                    }
                    auto* g = out_pt1->_get_var();
                    g->update_write_points(stname, out_pt1->get_arg_offsets());
                } // eqs.
            } // reqd stages.
        } // stages.

        /* Phase 2:
           Propagate halos through scratch vars as needed.  Using scratch
           vars is roughly equivalent to using intermediate values in the
           equation, so accessing neighbor cells in a scratch var will
           increase the halos in dependent vars.  This propagates through
           dependency chains until a non-scratch var is reached, affecting
           each var in the chain, including the non-scratch var(s) that the
           scratch vars depend on.

           Example 1:
           eq1: scr1(x) EQUALS u(t,x+1); <-- orig halo of u = 1.
           eq2: u(t+1,x) EQUALS scr1(x+2); <-- orig halo of scr1 = 2.
           Direct deps: eq2 -> eq1(scr).
           Halo of u must be increased to 1 + 2 = 3 due to
           eq1: u(t,x+1) on rhs and orig halo of scr1 on lhs.
           Or, because u(t+1,x) EQUALS u(t,(x+2)+1) by subst.

           Example 2:
           eq1: scr1(x) EQUALS u(t,x+1); <-- orig halo of u = 1.
           eq2: scr2(x) EQUALS scr1(x+2); <-- orig halo of scr1 = 2.
           eq3: u(t+1,x) EQUALS scr2(x+4); <-- orig halo of scr2 = 4.
           Direct deps: eq3 -> eq2(scr) -> eq1(scr).
           Halo of scr1 must be increased to 2 + 4 = 6 due to
           eq2: scr1(x+2) on rhs and orig halo of scr2 on lhs.
           Then, halo of u must be increased to 1 + 6 = 7 due to
           eq1: u(t,x+1) on rhs and new halo of scr1 on lhs.
           Or, u(t+1,x) EQUALS u(t,((x+4)+2)+1) by subst.

           Example 3:
           eq1: scr1(x) EQUALS u(t,x+1); <--|
           eq2: scr2(x) EQUALS u(t,x+2); <--| orig halo of u = max(1,2) = 2.
           eq3: u(t+1,x) EQUALS scr1(x+3) + scr2(x+4);
           eq1 and eq2 are bundled => scr1 and scr2 halos are max(3,4) = 4.
           Direct deps: eq3 -> eq1(scr), eq3 -> eq2(scr).
           Halo of u is 4 + 2 = 6.
           Or, u(t+1,x) EQUALS u(t,(x+3)+1) + u(t,(x+4)+2) by subst.

           Algo: Keep a list of maps of "shadow vars". Each list entry is a
           non-scratch stage.  Each map is key=real-var ptr ->
           val=shadow-var ptr.  These shadow vars are used to track updated
           halos for each stage.  We don't want to update the real halos
           until we've determined halo sizes using the original halos because
           doing so would update halos from already-updated ones.  At the end,
           the real vars will be updated from the shadows.
        */
        vector<map<Var*, Var*>> shadows;

        // Stages.
        for (auto& st1 : get_all()) {
            auto stname = st1->_get_name();
            auto& stparts = st1->get_parts(); // list of parts in this stage.

            // Only need to start from non-scratch stages.
            if (st1->is_scratch())
                continue;
            #ifdef DEBUG_HALOS
            cout << "* c_h, phase 2: analyzing stage '" << stname << "'" << endl;
            #endif

            // Create a new empty map of shadow vars for this stage.
            shadows.resize(shadows.size() + 1);
            auto& shadow_map = shadows.back(); // Use the new one.

            // Loop over the stages in reverse topo order, i.e., from most
            // to least dependent. This is critical to ensure that halo
            // expansions propagate in the correct order.  See
            // https://en.wikipedia.org/wiki/Longest_path_problem
            for (auto it2 = get_all().rbegin(); it2 != get_all().rend(); ++it2) {
                auto& st2 = *it2;

                // Only want scratch stages needed for 'st1'.
                if (!get_all_scratch_deps_on(st1).count(st2))
                    continue;

                // Loop through all parts in this stage. Order doesn't
                // matter because parts within a stage are independent.
                for (auto& p2 : st2->get_parts()) {
                    assert(p2->is_scratch());

                    // Make shadow copies of all vars touched
                    // by 'p2'.  All halo updates will be
                    // applied to these shadow vars for the
                    // current stage 'st1'.
                    for (auto& eq2 : p2->get_eqs()) {

                        // Output var.
                        auto* og = pv.get_output_vars().at(eq2.get());
                        if (shadow_map.count(og) == 0)
                            shadow_map[og] = new Var(*og);

                        // Input vars.
                        auto& in_pts = pv.get_input_pts().at(eq2.get());
                        for (auto* ip : in_pts) {
                            auto* ig = ip->_get_var();
                            if (shadow_map.count(ig) == 0)
                                shadow_map[ig] = new Var(*ig);
                        }
                    }

                    // For each scratch part, set the size of all its output
                    // vars' halos to the max across its halos. We need to
                    // do this because halos are written in a scratch var.
                    // Since they are bundled into a part, meaning they are
                    // all written in a single code block, all the writes
                    // will be over the same area. Get and store results
                    // using current shadow map.

                    // First, set first eq halo the max of all.
                    auto& eq1 = p2->get_eqs().front();
                    auto* ov1 = shadow_map.at(eq1->get_lhs_var());
                    for (auto& eq2 : p2->get_eqs()) {
                        if (eq1 == eq2)
                            continue;

                        // Adjust halo of ov1 to max(ov1, ov2).
                        auto* ov2 = shadow_map.at(eq2->get_lhs_var());
                        ov1->update_halo(*ov2);
                    }

                    // Then, update all others based on first.
                    for (auto& eq2 : p2->get_eqs()) {
                        if (eq1 == eq2)
                            continue;

                        // Adjust halo of ov2 to that of ov1.
                        auto* ov2 = shadow_map.at(eq2->get_lhs_var());
                        ov2->update_halo(*ov1);
                    }

                    // Get updated halos from the scratch vars.  These are the
                    // points that are read from the dependent eq(s).  For
                    // scratch vars, the halo areas must also be written to.
                    auto left_ohalo = ov1->get_halo_sizes(stname, true);
                    auto right_ohalo = ov1->get_halo_sizes(stname, false);
                    auto l1_dist = ov1->get_l1_dist();

                    #ifdef DEBUG_HALOS
                    cout << "** c_h: processing " << p2->get_descr() << "...\n"
                        "*** c_h: LHS halos " << left_ohalo.make_dim_val_str() <<
                        " & RHS halos " << right_ohalo.make_dim_val_str() << endl;
                    #endif
                        
                    // Recalc min halos of all *input* vars of all scratch
                    // eqs in this part by adding size of output-var
                    // halos. These inputs may include scratch and
                    // non-scratch vars. Get and stroe results using current
                    // shadow map.
                    for (auto& eq2 : p2->get_eqs()) {
                        assert(eq2->is_scratch());
                        auto& in_pts = pv.get_input_pts().at(eq2.get());

                        // Input points.
                        for (auto ip : in_pts) {
                            auto* iv2 = shadow_map.at(ip->_get_var());
                            auto& ao = ip->get_arg_offsets(); // e.g., '2' for 'x+2'.

                            // Increase range by subtracting left halos and
                            // adding right halos.
                            auto left_ihalo = ao.sub_elements(left_ohalo, false);
                            iv2->update_halo(stname, left_ihalo);
                            auto right_ihalo = ao.add_elements(right_ohalo, false);
                            iv2->update_halo(stname, right_ihalo);
                            iv2->update_l1_dist(l1_dist);
                            #ifdef DEBUG_HALOS
                            cout << "*** c_h: updated min halos of '" << iv2->get_name() << "' to " <<
                                left_ihalo.make_dim_val_str() <<
                                " & " << right_ihalo.make_dim_val_str() << endl;
                            #endif
                        } // input pts.
                    } // eqs in part.
                } // parts in scratch stage.
                
            } // stages in rev topo-order.
        } // all stages.                

        // Apply the changes from the shadow vars.
        // This will result in the vars containing the max
        // of the shadow halos.
        for (auto& shadow_map : shadows) {
            #ifdef DEBUG_HALOS
            cout << "* c_h: applying changes from a shadow map...\n";
            #endif
            for (auto& si : shadow_map) {
                auto* orig_vp = si.first;
                auto* shadow_vp = si.second;
                assert(orig_vp);
                assert(shadow_vp);

                // Update the original.
                auto changed = orig_vp->update_halo(*shadow_vp);
                #ifdef DEBUG_HALOS
                if (changed) {
                    cout << "** c_h: updated halos:" << endl;
                    orig_vp->print_halos(cout, "*** ");
                }
                #endif

                // Release the shadow var mem.
                delete shadow_vp;
                shadow_map.at(orig_vp) = NULL;
            }
        } // shadows.
    } // calc_halos().
    
} // namespace yask.
