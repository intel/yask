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

///////// Methods for equations and equation bundles ////////////

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Parse.hpp"
#include "Eqs.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {

    // A visitor to collect grids and points visited in a set of eqs.
    // For each eq, there are accessors for its output grid and point
    // and its input grids and points.
    class PointVisitor : public ExprVisitor {

        // A type to hold a mapping of equations to a set of grids in each.
        typedef unordered_set<GridVar*> GridSet;
        typedef unordered_map<EqualsExpr*, GridVar*> GridMap;
        typedef unordered_map<EqualsExpr*, GridSet> GridSetMap;

        GridMap _lhs_grids; // outputs of eqs.
        GridSetMap _rhs_grids; // inputs of eqs.

        // A type to hold a mapping of equations to a set of points in each.
        typedef unordered_set<GridPoint*> PointSet;
        typedef unordered_map<EqualsExpr*, GridPoint*> PointMap;
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

        // Get access to grids per eq.
        GridMap& getOutputGrids() { return _lhs_grids; }
        GridSetMap& getInputGrids() { return _rhs_grids; }

        // Get access to pts per eq.
        // Contains unique ptrs to pts, but pts may not
        // be unique.
        PointMap& getOutputPts() { return _lhs_pts; }
        PointSetMap& getInputPts() { return _rhs_pts; }
        PointSetMap& getCondPts() { return _cond_pts; }
        PointSetMap& getStepCondPts() { return _step_cond_pts; }
        PointSetMap& getAllPts() { return _all_pts; }

        int getNumEqs() const { return (int)_lhs_pts.size(); }

        // Callback at an equality.
        // Visits all parts that might have grid points.
        virtual string visit(EqualsExpr* ee) {

            // Set this equation as current one.
            _eq = ee;

            // Make sure all map entries exist for this eq.
            _lhs_grids[_eq];
            _rhs_grids[_eq];
            _lhs_pts[_eq];
            _rhs_pts[_eq];
            _cond_pts[_eq];
            _step_cond_pts[_eq];
            _all_pts[_eq];

            // visit LHS.
            auto& lhs = ee->getLhs();
            _state = _in_lhs;
            lhs->accept(this);
            
            // visit RHS.
            NumExprPtr rhs = ee->getRhs();
            _state = _in_rhs;
            rhs->accept(this);

            // visit conds.
            auto& cp = ee->getCond();
            if (cp) {
                _state = _in_cond;
                cp->accept(this);
            }
            auto& scp = ee->getStepCond();
            if (scp) {
                _state = _in_step_cond;
                scp->accept(this);
            }
            return "";
        }

        // Callback at a grid point.
        virtual string visit(GridPoint* gp) {
            assert(_eq);
            auto* g = gp->getGrid();
            _all_pts[_eq].insert(gp);

            // Save pt and/or grid based on state.
            switch (_state) {

            case _in_lhs:
                _lhs_pts[_eq] = gp;
                _lhs_grids[_eq] = g;
                break;

            case _in_rhs:
                _rhs_pts[_eq].insert(gp);
                _rhs_grids[_eq].insert(g);
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

    // Analyze group of equations.
    // Sets _stepDir in dims.
    // Finds dependencies based on all eqs if 'settings._findDeps', setting
    // _imm_dep_on and _dep_on.
    // Throws exceptions on illegal dependencies.
    // TODO: split this into smaller functions.
    // BIG-TODO: replace dependency algorithms with integration of a polyhedral
    // library.
    void Eqs::analyzeEqs(CompilerSettings& settings,
                         Dimensions& dims,
                         ostream& os) {
        auto& stepDim = dims._stepDim;

        // Gather initial stats from all eqs.
        PointVisitor pt_vis;
        visitEqs(&pt_vis);
        auto& outGrids = pt_vis.getOutputGrids();
        auto& inGrids = pt_vis.getInputGrids();
        auto& outPts = pt_vis.getOutputPts();
        auto& inPts = pt_vis.getInputPts();
        //auto& condPts = pt_vis.getCondPts();
        //auto& stepCondPts = pt_vis.getStepCondPts();

        // 1. Check each eq internally.
        os << "\nProcessing " << getNum() << " stencil equation(s)...\n";
        for (auto eq1 : getAll()) {
            auto* eq1p = eq1.get();
            assert(outGrids.count(eq1p));
            assert(inGrids.count(eq1p));
            auto* og1 = outGrids.at(eq1p);
            assert(og1 == eq1->getGrid());
            auto* op1 = outPts.at(eq1p);
            //auto& ig1 = inGrids.at(eq1p);
            auto& ip1 = inPts.at(eq1p);
            auto cond1 = eq1p->getCond();
            auto stcond1 = eq1p->getStepCond();
            NumExprPtr step_expr1 = op1->getArg(stepDim); // may be null.

#ifdef DEBUG_DEP
            cout << " Checking internal consistency of equation " <<
                eq1->makeQuotedStr() << "...\n";
#endif

            // Scratch grid must not have a condition.
            if (cond1 && og1->isScratch())
                THROW_YASK_EXCEPTION("Error: scratch-grid equation " + eq1->makeQuotedStr() +
                                     " cannot have a domain condition");
            if (stcond1 && og1->isScratch())
                THROW_YASK_EXCEPTION("Error: scratch-grid equation " + eq1->makeQuotedStr() +
                                     " cannot have a step condition");

            // LHS must have all domain dims.
            for (auto& dd : dims._domainDims.getDims()) {
                auto& dname = dd.getName();
                NumExprPtr dexpr = op1->getArg(dname);
                if (!dexpr)
                    THROW_YASK_EXCEPTION("Error: grid equation " + eq1->makeQuotedStr() +
                                         " does not use domain-dimension '" + dname +
                                         "' on LHS");
            }

            // LHS of non-scratch must have step dim and vice-versa.
            if (!og1->isScratch()) {
                if (!step_expr1)
                    THROW_YASK_EXCEPTION("Error: non-scratch grid equation " + eq1->makeQuotedStr() +
                                         " does not use step-dimension '" + stepDim +
                                         "' on LHS");
            } else {
                if (step_expr1)
                    THROW_YASK_EXCEPTION("Error: scratch-grid equation " + eq1->makeQuotedStr() +
                                         " cannot use step-dimension '" + stepDim + "'");
            }

            // Check LHS grid dimensions and associated args.
            for (int di = 0; di < og1->get_num_dims(); di++) {
                auto& dn = og1->get_dim_name(di);  // name of this dim.
                auto argn = op1->getArgs().at(di); // LHS arg for this dim.

                // Check based on dim type.
                if (dn == stepDim) {
                }

                // LHS must have simple indices in domain dims.
                else if (dims._domainDims.lookup(dn)) {

                    // Make expected arg, e.g., 'x'.
                    auto earg = make_shared<IndexExpr>(dn, DOMAIN_INDEX);

                    // Compare to actual.
                    if (!argn->isSame(earg))
                        THROW_YASK_EXCEPTION("Error: LHS of equation " + eq1->makeQuotedStr() +
                                             " contains expression " + argn->makeQuotedStr() +
                                             " for domain dimension '" + dn +
                                             "' where " + earg->makeQuotedStr() +
                                             " is expected");
                }

                // Misc dim must be a const.  TODO: allow non-const misc
                // dims and treat const and non-const ones separately, e.g.,
                // for interleaving.
                else {

                    if (!argn->isConstVal())
                        THROW_YASK_EXCEPTION("Error: LHS of equation " + eq1->makeQuotedStr() +
                                             " contains expression " + argn->makeQuotedStr() +
                                             " for misc dimension '" + dn +
                                             "' where kernel-run-time constant integer is expected");
                    argn->getIntVal(); // throws exception if not an integer.
                }
            }
        
            // Heuristics to set the default step direction.
            // The accuracy isn't critical, because the default is only be
            // used in the standalone test utility and the auto-tuner.
            if (!og1->isScratch()) {

                // First, see if LHS step arg is a simple offset, e.g., 'u(t+1, ...)'.
                // This is the most common case.
                auto& lofss = op1->getArgOffsets();
                auto* lofsp = lofss.lookup(stepDim); // offset at step dim.
                if (lofsp) {
                    auto lofs = *lofsp;

                    // Scan input (RHS) points.
                    for (auto i1 : ip1) {
                        
                        // Is point a simple offset from step, e.g., 'u(t-2, ...)'?
                        auto* rsi1p = i1->getArgOffsets().lookup(stepDim);
                        if (rsi1p) {
                            int rofs = *rsi1p;

                            // Example:
                            // forward: 'u(t+1, ...) EQUALS ... u(t, ...) ...',
                            // backward: 'u(t-1, ...) EQUALS ... u(t, ...) ...'.
                            if (lofs > rofs) {
                                dims._stepDir = 1;
                                break;
                            }
                            else if (lofs < rofs) {
                                dims._stepDir = -1;
                                break;
                            }
                        }
                    } // for all RHS points.
                    
                    // Soln step-direction heuristic used only if not set.
                    // Assume 'u(t+1, ...) EQUALS ...' implies forward,
                    // and 'u(t-1, ...) EQUALS ...' implies backward.
                    if (dims._stepDir == 0 && lofs != 0)
                        dims._stepDir = (lofs > 0) ? 1 : -1;

                }
            }

            // LHS of equation must be vectorizable.
            // TODO: relax this restriction.
            if (op1->getVecType() != GridPoint::VEC_FULL) {
                THROW_YASK_EXCEPTION("Error: LHS of equation " + eq1->makeQuotedStr() +
                                     " is not fully vectorizable because not all folded"
                                     " dimensions are accessed via simple offsets from their respective indices");
            }

            // Check that domain indices are simple offsets and
            // misc indices are consts on RHS.
            for (auto i1 : ip1) {
                auto* ig1 = i1->getGrid();

                for (int di = 0; di < ig1->get_num_dims(); di++) {
                    auto& dn = ig1->get_dim_name(di);  // name of this dim.
                    auto argn = i1->getArgs().at(di); // arg for this dim.

                    // Check based on dim type.
                    if (dn == stepDim) {
                    }

                    // Must have simple indices in domain dims.
                    else if (dims._domainDims.lookup(dn)) {
                        auto* rsi1p = i1->getArgOffsets().lookup(dn);
                        if (!rsi1p)
                            THROW_YASK_EXCEPTION("Error: RHS of equation " + eq1->makeQuotedStr() +
                                                 " contains expression " + argn->makeQuotedStr() +
                                                 " for domain dimension '" + dn +
                                                 "' where constant-integer offset from '" + dn +
                                                 "' is expected");
                    }

                    // Misc dim must be a const.
                    else {
                        if (!argn->isConstVal())
                            THROW_YASK_EXCEPTION("Error: RHS of equation " + eq1->makeQuotedStr() +
                                                 " contains expression " + argn->makeQuotedStr() +
                                                 " for misc dimension '" + dn +
                                                 "' where constant integer is expected");
                        argn->getIntVal(); // throws exception if not an integer.
                    }
                }
            } // input pts.

            // TODO: check to make sure cond1 depends only on domain indices.
            // TODO: check to make sure stcond1 does not depend on domain indices.
        } // for all eqs.

        // 2. Check each pair of eqs.
        os << "Analyzing for dependencies...\n";
        for (auto eq1 : getAll()) {
            auto* eq1p = eq1.get();
            assert(outGrids.count(eq1p));
            assert(inGrids.count(eq1p));
            auto* og1 = outGrids.at(eq1p);
            assert(og1 == eq1->getGrid());
            auto* op1 = outPts.at(eq1p);
            //auto& ig1 = inGrids.at(eq1p);
            //auto& ip1 = inPts.at(eq1p);
            auto cond1 = eq1p->getCond();
            auto stcond1 = eq1p->getStepCond();
            NumExprPtr step_expr1 = op1->getArg(stepDim);

            // Check each 'eq2' to see if it depends on 'eq1'.
            for (auto eq2 : getAll()) {
                auto* eq2p = eq2.get();
                auto& og2 = outGrids.at(eq2p);
                assert(og2 == eq2->getGrid());
                auto& op2 = outPts.at(eq2p);
                auto& ig2 = inGrids.at(eq2p);
                auto& ip2 = inPts.at(eq2p);
                auto cond2 = eq2p->getCond();
                auto stcond2 = eq2p->getStepCond();

#ifdef DEBUG_DEP
                cout << " Checking eq " <<
                    eq1->makeQuotedStr() << " vs " <<
                    eq2->makeQuotedStr() << "...\n";
#endif

                bool same_eq = eq1 == eq2;
                bool same_op = areExprsSame(op1, op2);
                bool same_cond = areExprsSame(cond1, cond2);
                bool same_stcond = areExprsSame(stcond1, stcond2);

                // A separate grid is defined by its name and any const indices.
                //bool same_og = op1->isSameLogicalGrid(*op2);

                // If two different eqs have the same conditions, they
                // cannot have the same LHS.
                if (!same_eq && same_cond && same_stcond && same_op) {
                    string cdesc = cond1 ? "with domain condition " + cond1->makeQuotedStr() :
                        "without domain conditions";
                    string stcdesc = stcond1 ? "with step condition " + stcond1->makeQuotedStr() :
                        "without step conditions";
                    THROW_YASK_EXCEPTION("Error: two equations " + cdesc +
                                         " and " + stcdesc +
                                         " have the same LHS: " +
                                         eq1->makeQuotedStr() + " and " +
                                         eq2->makeQuotedStr());
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
                        THROW_YASK_EXCEPTION("Error: illegal dependency: LHS of equation " +
                                             eq1->makeQuotedStr() + " also appears on its RHS");
                    }

                    // Save dependency.
#ifdef DEBUG_DEP
                    cout << "  Exact match found to " << op1->makeQuotedStr() << ".\n";
#endif
                    if (settings._findDeps)
                        _deps.set_imm_dep_on(eq2, eq1);

                    // Move along to next eq2.
                    continue;
                }

                // Don't do more conservative checks if not looking for deps.
                if (!settings._findDeps)
                    continue;

                // Next dep check: inexact matches on LHS of eq1 to RHS of eq2.
                // Does eq1 define *any* point in a grid that eq2 inputs
                // at the same step index?  If so, they *might* have a
                // dependency. Some of these may not be real
                // dependencies due to conditions. Those that are real
                // may or may not be legal.
                //
                // Example:
                //  eq1: a(t+1, x, ...) EQUALS ... IF ...
                //  eq2: b(t+1, x, ...) EQUALS a(t+1, x+5, ...) ... IF ...
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

                        // Same logical grid?
                        bool same_grid = i2->isSameLogicalGrid(*op1);

                        // If not same grid, no dependency.
                        if (!same_grid)
                            continue;

                        // Both points at same step?
                        bool same_step = false;
                        NumExprPtr step_expr2 = i2->getArg(stepDim);
                        if (step_expr1 && step_expr2 &&
                            areExprsSame(step_expr1, step_expr2))
                            same_step = true;

                        // From same step index, e.g., same time?
                        // Or, passing data thru a temp var?
                        if (same_step || og1->isScratch()) {

                            // Eq depends on itself?
                            if (same_eq) {

                                // Exit with error.
                                string stepmsg = same_step ? " at '" + step_expr1->makeQuotedStr() + "'" : "";
                                THROW_YASK_EXCEPTION("Error: disallowed dependency: grid '" +
                                                     op1->makeLogicalGridStr() + "' on LHS of equation " +
                                                     eq1->makeQuotedStr() + " also appears on its RHS" +
                                                     stepmsg);
                            }

                            // Save dependency.
#ifdef DEBUG_DEP
                            cout << "  Likely match found to " << op1->makeQuotedStr() << ".\n";
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
        if (!dims._stepDir)
            dims._stepDir = 1;

#ifdef DEBUG_DEP
        cout << "Dependencies for all eqs:\n";
        _deps.printDeps(cout);
#endif

        // Find scratch children.
        analyzeScratch();

#ifdef DEBUG_DEP
        cout << "Dependencies from non-scratch to scratch eqs:\n";
        _scratches.printDeps(cout);
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

    // Visitor for determining vectorization potential of grid points.
    // Vectorization depends not only on the dims of the grid itself
    // but also on how the grid is indexed at each point.
    class SetVecVisitor : public ExprVisitor {
        const Dimensions& _dims;

    public:
        SetVecVisitor(const Dimensions& dims) :
            _dims(dims) { 
            _visitEqualsLhs = true;
            _visitGridPointArgs = true;
            _visitConds = true;
        }

        // Check each grid point in expr.
        virtual string visit(GridPoint* gp) {
            auto* grid = gp->getGrid();

            // Never vectorize scalars.
            if (grid->get_num_dims() == 0) {
                gp->setVecType(GridPoint::VEC_NONE);
                return "";      // Also, no args to visit.
            }

            // Amount of vectorization allowed primarily depends on number
            // of folded dimensions in the grid accessed at this point.
            int grid_nfd = grid->getNumFoldableDims();
            int soln_nfd = _dims._foldGT1.size();
            assert(grid_nfd <= soln_nfd);

            // Vectorization is only possible if each access to a vectorized
            // dim is a simple offset.  For example, in grid dim 'x', the
            // index in the corresponding posn must be 'x', 'x+n', or 'x-n'.
            int fdoffsets = 0;
            for (auto fdim : _dims._foldGT1.getDims()) {
                auto& fdname = fdim.getName();
                if (gp->getArgOffsets().lookup(fdname))
                    fdoffsets++;
            }
            assert(fdoffsets <= grid_nfd);

            // All folded dims are vectorizable?
            // NB: this will always be the case when there is
            // no folding in the soln.
            if (fdoffsets == soln_nfd)
                gp->setVecType(GridPoint::VEC_FULL); // all good.

            // Some dims are vectorizable?
            else if (fdoffsets > 0)
                gp->setVecType(GridPoint::VEC_PARTIAL);

            // Uses no folded dims, so scalar only.
            else
                gp->setVecType(GridPoint::VEC_NONE);

            // Also check args of this grid point.
            return ExprVisitor::visit(gp);
        }
    };

    // Determine which grid points can be vectorized.
    void Eqs::analyzeVec(const Dimensions& dims) {

        // Send a 'SetVecVisitor' to each point in
        // the current equations.
        SetVecVisitor svv(dims);
        visitEqs(&svv);
    }

    // Visitor to find referenced vars.
    class FindVarsVisitor : public ExprVisitor {

    public:
        set<string> vars_used;

        // Check each index expr;
        virtual string visit(IndexExpr* ie) {
            vars_used.insert(ie->getName());
            return "";
        }
    };

    // Visitor for determining inner-loop accesses of grid points.
    class SetLoopVisitor : public ExprVisitor {
        const Dimensions& _dims;

    public:
        SetLoopVisitor(const Dimensions& dims) :
            _dims(dims) { 
            _visitEqualsLhs = true;
        }

        // Check each grid point in expr.
        virtual string visit(GridPoint* gp) {

            // Info from grid.
            auto* grid = gp->getGrid();
            auto gdims = grid->get_dim_names();

            // Check loop in this dim.
            auto idim = _dims._innerDim;

            // Access type.
            // Assume invariant, then check below.
            GridPoint::LoopType lt = GridPoint::LOOP_INVARIANT;

            // Check every point arg.
            auto& args = gp->getArgs();
            for (size_t ai = 0; ai < args.size(); ai++) {
                auto& arg = args.at(ai);
                assert(ai < gdims.size());

                // Get set of vars used.
                FindVarsVisitor fvv;
                arg->accept(&fvv);

                // Does this arg refer to idim?
                if (fvv.vars_used.count(idim)) {

                    // Is it in the idim posn and a simple offset?
                    int offset = 0;
                    if (gdims.at(ai) == idim &&
                        arg->isOffsetFrom(idim, offset)) {
                        lt = GridPoint::LOOP_OFFSET;
                    }

                    // Otherwise, this arg uses idim, but not
                    // in a simple way.
                    else {
                        lt = GridPoint::LOOP_OTHER;
                        break;  // no need to continue.
                    }
                }
            }
            gp->setLoopType(lt);
            return "";
        }
    };

    // Determine loop access behavior of grid points.
    void Eqs::analyzeLoop(const Dimensions& dims) {

        // Send a 'SetLoopVisitor' to each point in
        // the current equations.
        SetLoopVisitor slv(dims);
        visitEqs(&slv);
    }

    // Update access stats for the grids.
    // For now, this us just const indices.
    // Halos are updated later, after packs are established.
    void Eqs::updateGridStats() {

        // Find all LHS and RHS points and grids for all eqs.
        PointVisitor pv;
        visitEqs(&pv);

        // Analyze each eq.
        for (auto& eq : getAll()) {

            // Get all grid points touched by this eq.
            auto& allPts1 = pv.getAllPts().at(eq.get());

            // Update stats of each grid accessed in 'eq'.
            for (auto ap : allPts1) {
                auto* g = ap->getGrid(); // grid for point 'ap'.
                g->updateConstIndices(ap->getArgConsts());
            }
        }
    }

    // Find scratch-grid eqs needed for each non-scratch eq.  These will
    // eventually be gathered into bundles and saved as the "scratch
    // children" for each non-scratch bundles.
    void Eqs::analyzeScratch() {

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
        
        // Find all LHS and RHS points and grids for all eqs.
        PointVisitor pv;
        visitEqs(&pv);

        // Analyze each eq.
        for (auto& eq1 : getAll()) {

            // Output grid for this eq.
            auto* og1 = pv.getOutputGrids().at(eq1.get());

            // Only need to look at dep paths starting from non-scratch eqs.
            if (og1->isScratch())
                continue;

            // We start with each non-scratch eq and walk the dep tree to
            // find all dependent scratch eqs.  It's important to
            // then visit the eqs in dep order using 'path' to get only
            // unbroken chains of scratch grids.
            // Note that sub-paths may be visited more than once, e.g.,
            // 'eq3 -> eq2(s)' and 'eq3 -> eq2(s) -> eq1(s)' are 2 paths from
            // the second example above, but this shouldn't cause any issues,
            // just some redundant work.
            getDeps().visitDeps

                // For each 'b', 'eq1' is 'b' or depends on 'b',
                // immediately or indirectly; 'path' leads from
                // 'eq1' to 'b'.
                (eq1, [&](EqualsExprPtr b, EqList& path) {
                    //auto* ogb = pv.getOutputGrids().at(b.get());

                    // Find scratch-grid eqs in this dep path that are
                    // needed for 'eq1'. Walk dep path from 'eq1' to 'b'
                    // until a non-scratch grid is found.
                    unordered_set<GridVar*> scratches_seen;
                    for (auto eq2 : path) {

                        // Don't process 'eq1', the initial non-scratch eq.
                        if (eq2 == eq1)
                            continue;
                        
                        // If this isn't a scratch eq, we are done
                        // w/this path because we only want the eqs
                        // from 'eq1' through an *unbroken* chain of
                        // scratch grids.
                        auto* og2 = pv.getOutputGrids().at(eq2.get());
                        if (!og2->isScratch())
                            break;

                        // Add 'eq2' to the set needed for 'eq1'.
                        // NB: scratch-deps are used as a map of sets.
                        getScratchDeps().set_imm_dep_on(eq1, eq2);

                        // Check for illegal scratch path.
                        // TODO: this is only illegal because the scratch
                        // write area is stored in the grid, so >1 write
                        // areas can't be shared. Need to store the write
                        // areas somewhere else, maybe in the equation or
                        // bundle. Would require changes to kernel as well.
                        if (scratches_seen.count(og2))
                            THROW_YASK_EXCEPTION("Error: scratch-grid '" +
                                                 og2->get_name() + "' depends upon itself");
                        scratches_seen.insert(og2);
                    }

                });
        }
    }

    // Get the full name of an eq-lot.
    // Must be unique.
    string EqLot::getName() const {

        // Add index to base name.
        return baseName + "_" + to_string(index);
    }

    // Make a human-readable description of this eq bundle.
    string EqBundle::getDescr(bool show_cond,
                              string quote) const
    {
        string des;
        if (isScratch())
            des += "scratch ";
        des += "equation-bundle " + quote + getName() + quote;
        if (!isScratch() && show_cond) {
            if (cond.get())
                des += " w/domain condition " + cond->makeQuotedStr(quote);
            else
                des += " w/o domain condition";
            if (step_cond.get())
                des += " w/step condition " + step_cond->makeQuotedStr(quote);
            else
                des += " w/o step condition";
        }
        return des;
    }

    // Add an equation to an EqBundle.
    void EqBundle::addEq(EqualsExprPtr ee)
    {
#ifdef DEBUG_EQ_BUNDLE
        cout << "EqBundle: adding " << ee->makeQuotedStr() << endl;
#endif
        _eqs.insert(ee);

        // Get I/O point data from eq 'ee'.
        PointVisitor pv;
        ee->accept(&pv);

        // update list of input and output grids for this bundle.
        auto* outGrid = pv.getOutputGrids().at(ee.get());
        _outGrids.insert(outGrid);
        auto& inGrids = pv.getInputGrids().at(ee.get());
        for (auto* g : inGrids)
            _inGrids.insert(g);
    }

    // Print stats from eqs.
    void EqLot::printStats(ostream& os, const string& msg) {
        CounterVisitor cv;
        visitEqs(&cv);
        cv.printStats(os, msg);
    }

    // Print stats from eqs in bundles.
    void EqBundles::printStats(ostream& os, const string& msg) {
        CounterVisitor cv;

        // Use separate counter visitor for each bundle
        // to avoid treating repeated eqs as common sub-exprs.
        for (auto& eq : _all) {
            CounterVisitor ecv;
            eq->visitEqs(&ecv);
            cv += ecv;
        }
        cv.printStats(os, msg);
    }

    // Visitor that will shift each grid point by an offset.
    class OffsetVisitor: public ExprVisitor {
        IntTuple _ofs;

    public:
        OffsetVisitor(const IntTuple& ofs) :
            _ofs(ofs) {
            _visitEqualsLhs = true;
        }

        // Visit a grid point.
        virtual string visit(GridPoint* gp) {

            // Shift grid _ofs points.
            auto ofs0 = gp->getArgOffsets();
            IntTuple new_loc = ofs0.addElements(_ofs, false);
            gp->setArgOffsets(new_loc);
            return "";
        }
    };

    // Replicate each equation at the non-zero offsets for
    // each vector in a cluster.
    void EqBundle::replicateEqsInCluster(Dimensions& dims)
    {
        // Make a copy of the original equations so we can iterate through
        // them while adding to the bundle.
        EqList eqs(_eqs);

        // Loop thru points in cluster.
        dims._clusterMults.visitAllPoints([&](const IntTuple& clusterIndex,
                                              size_t idx) {

                // Don't need copy of one at origin.
                if (clusterIndex.sum() > 0) {

                    // Get offset of cluster, which is each cluster index multipled
                    // by corresponding vector size.  Example: for a 4x4 fold in a
                    // 1x2 cluster, the 2nd cluster index will be (0,1) and the
                    // corresponding cluster offset will be (0,4).
                    auto clusterOffset = clusterIndex.multElements(dims._fold);

                    // Loop thru eqs.
                    for (auto eq : eqs) {
                        assert(eq.get());

                        // Make a copy.
                        auto eq2 = eq->clone();

                        // Add offsets to each grid point.
                        OffsetVisitor ov(clusterOffset);
                        eq2->accept(&ov);

                        // Put new equation into bundle.
                        addEq(eq2);
                    }
                }
                return true;
            });

        // Ensure the expected number of equations now exist.
        assert(_eqs.size() == eqs.size() * dims._clusterMults.product());
    }

    // Add 'eq' from 'allEqs' to eq-bundle with 'baseName' unless
    // alread added or illegal.  The corresponding index in '_indices' will
    // be incremented if a new bundle is created.  Returns whether a new
    // bundle was created.
    bool EqBundles::addEqToBundle(Eqs& allEqs,
                                  EqualsExprPtr eq,
                                  const string& baseName,
                                  const CompilerSettings& settings) {
        assert(_dims);
        auto& stepDim = _dims->_stepDim;
        
        // Equation already added?
        if (_eqs_in_bundles.count(eq))
            return false;

        // Get conditions, if any.
        auto cond = eq->getCond();
        auto stcond = eq->getStepCond();

        // Get step expr, if any.
        auto step_expr = eq->getLhs()->getArg(stepDim);
        
        // Get deps between eqs.
        auto& eq_deps = allEqs.getDeps();

        // Loop through existing bundles, looking for one that
        // 'eq' can be added to.
        EqBundle* target = 0;
        auto eqg = eq->getGrid();
        for (auto& eg : getAll()) {

            // Must be same scratch-ness.
            if (eg->isScratch() != eq->isScratch())
                continue;

            // Names must match.
            if (eg->baseName != baseName)
                continue;

            // Conditions must match (both may be null).
            if (!areExprsSame(eg->cond, cond))
                continue;
            if (!areExprsSame(eg->step_cond, stcond))
                continue;

            // LHS step exprs must match (both may be null).
            if (!areExprsSame(eg->step_expr, step_expr))
                continue;

            // Look for any condition or dependencies that would prevent
            // adding 'eq' to 'eg'.
            bool is_ok = true;
            for (auto& eq2 : eg->getEqs()) {
                auto eq2g = eq2->getGrid();
                assert (eqg);
                assert (eq2g);

                // If scratch, 'eq' and 'eq2' must have same halo.
                // This is because scratch halos are written to.
                // If dims are same, halos can be adjusted if allowed.
                if (eq->isScratch()) {
                    if (!eq2g->areDimsSame(*eqg))
                        is_ok = false;
                    else if (!settings._bundleScratch && !eq2g->isHaloSame(*eqg))
                        is_ok = false;
                }

                // Look for any dependency between 'eq' and 'eq2'.
                if (is_ok && eq_deps.is_dep(eq, eq2)) {
#if DEBUG_ADD_EXPRS
                    cout << "addEqFromGrid: not adding equation " <<
                        eq->makeQuotedStr() << " to " << eg.getDescr() <<
                        " because of dependency w/equation " <<
                        eq2->makeQuotedStr() << endl;
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
        bool newBundle = false;
        if (!target) {
            auto ne = make_shared<EqBundle>(*_dims, eq->isScratch());
            addItem(ne);
            target = ne.get();
            target->baseName = baseName;
            target->index = _indices[baseName]++;
            target->cond = cond;
            target->step_cond = stcond;
            target->step_expr = step_expr;
            newBundle = true;

#if DEBUG_ADD_EXPRS
            cout << "Creating new " << target->getDescr() << endl;
#endif
        }

        // Add eq to target eq-bundle.
        assert(target);
#if DEBUG_ADD_EXPRS
        cout << "Adding " << eq->makeQuotedStr() <<
            " to " << target->getDescr() << endl;
#endif
        target->addEq(eq);

        // Remember eq and updated grid.
        _eqs_in_bundles.insert(eq);
        _outGrids.insert(eq->getGrid());

        return newBundle;
    }

    // Find halos needed for each grid.
    void EqBundlePacks::calcHalos(EqBundles& allBundles) {

        // Find all LHS and RHS points and grids for all eqs.
        PointVisitor pv;
        visitEqs(&pv);

#ifdef DEBUG_SCRATCH
        cout << "* cH: analyzing " << getAll().size() << " eqs...\n";
#endif
        
        // First, set halos based only on immediate accesses.
        for (auto& bp : getAll()) {
            auto pname = bp->getName();
            
            for (auto& eq : bp->getEqs()) {

                // Get all grid points touched by this eq.
                auto& allPts1 = pv.getAllPts().at(eq.get());

                // Update stats of each grid accessed in 'eq'.
                for (auto ap : allPts1) {
                    auto* g = ap->getGrid(); // grid for point 'ap'.
                    g->updateHalo(pname, ap->getArgOffsets());
                }
            }
        }

        // Next, propagate halos through scratch grids as needed.

        // Example:
        // eq1: scr(x) EQUALS u(t,x+1); <-- orig halo of u = 1.
        // eq2: u(t+1,x) EQUALS scr(x+2); <-- orig halo of scr = 2.
        // Direct deps: eq2 -> eq1(s).
        // Halo of u must be increased to 1 + 2 = 3 due to
        // eq1: u(t,x+1) on rhs and orig halo of scr on lhs.

        // Example:
        // eq1: scr1(x) EQUALS u(t,x+1); <-- orig halo of u = 1.
        // eq2: scr2(x) EQUALS scr1(x+2); <-- orig halo of scr1 = 2.
        // eq3: u(t+1,x) EQUALS scr2(x+4); <-- orig halo of scr2 = 4.
        // Direct deps: eq3 -> eq2(s) -> eq1(s).
        // Halo of scr1 must be increased to 2 + 4 = 6 due to
        // eq2: scr1(x+2) on rhs and orig halo of scr2 on lhs.
        // Then, halo of u must be increased to 1 + 6 = 7 due to
        // eq1: u(t,x+1) on rhs and new halo of scr1 on lhs.

        // Example:
        // eq1: scr1(x) EQUALS u(t,x+1); <--|
        // eq2: scr2(x) EQUALS u(t,x+2); <--| orig halo of u = max(1,2) = 2.
        // eq3: u(t+1,x) EQUALS scr1(x+3) + scr2(x+4);
        // eq1 and eq2 are bundled => scr1 and scr2 halos are max(3,4) = 4.
        // Direct deps: eq3 -> eq1(s), eq3 -> eq2(s).
        
        // Keep a list of maps of shadow grids.
        // Each map: key=real-grid ptr, val=shadow-grid ptr.
        // These shadow grids will be used to track
        // updated halos for each path.
        // We don't want to update the real halos until
        // we've walked all the paths using the original
        // halos.
        // At the end, the real grids will be updated
        // from the shadows.
        vector< map<GridVar*, GridVar*>> shadows;

        // Packs.
        for (auto& bp : getAll()) {
            auto pname = bp->getName();
            auto& pbundles = bp->getBundles(); // list of bundles.

            // Bundles with their dependency info.
            for (auto& b1 : allBundles.getAll()) {

                // Only need bundles in this pack.
                if (pbundles.count(b1) == 0)
                    continue;

                // Only need to look at dep paths starting from non-scratch bundles.
                if (b1->isScratch())
                    continue;

                // We start with each non-scratch bundle and walk the dep
                // tree to find all dependent scratch bundles.  It's
                // important to then visit them in dep order using 'path' to
                // get only unbroken chains of scratch bundles.
#ifdef DEBUG_SCRATCH
                cout << "* cH: visiting deps of " << b1->getDescr() << endl;
#endif
                allBundles.getDeps().visitDeps
                
                    // For each 'bn', 'b1' is 'bn' or depends on 'bn',
                    // immediately or indirectly; 'path' leads from
                    // 'b1' to 'bn'.
                    (b1, [&](EqBundlePtr bn, EqBundleList& path) {

                        // Create a new empty map of shadow grids for this path.
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
                            if (!b2->isScratch())
                                break;

                            // Make shadow copies of all grids touched by 'eq2'.
                            // All changes will be applied to these shadow grids
                            // for the current 'path'.
                            for (auto& eq : b2->getEqs()) {

                                // Output grid.
                                auto* og = pv.getOutputGrids().at(eq.get());
                                if (shadow_map.count(og) == 0)
                                    shadow_map[og] = new GridVar(*og);

                                // Input grids.
                                auto& inPts = pv.getInputPts().at(eq.get());
                                for (auto* ip : inPts) {
                                    auto* ig = ip->getGrid();
                                    if (shadow_map.count(ig) == 0)
                                        shadow_map[ig] = new GridVar(*ig);
                                }
                            }
                        
                            // For each scratch bundle, set the size of all its
                            // output grids' halos to the max across its
                            // halos. We need to do this because halos are
                            // written in a scratch grid.  Since they are
                            // bundled, all the writes must be over the same
                            // area.

                            // First, set first eq halo the max of all.
                            auto& eq1 = b2->getEqs().front();
                            auto* og1 = shadow_map[eq1->getGrid()];
                            for (auto& eq2 : b2->getEqs()) {
                                if (eq1 == eq2)
                                    continue;

                                // Adjust g1 to max(g1, g2).
                                auto* og2 = shadow_map[eq2->getGrid()];
                                og1->updateHalo(*og2);
                            }

                            // Then, update all others based on first.
                            for (auto& eq2 : b2->getEqs()) {
                                if (eq1 == eq2)
                                    continue;

                                // Adjust g2 to g1.
                                auto* og2 = shadow_map[eq2->getGrid()];
                                og2->updateHalo(*og1);
                            }

                            // Get updated halos from the scratch bundle.  These
                            // are the points that are read from the dependent
                            // eq(s).  For scratch grids, the halo areas must
                            // also be written to.
                            auto left_ohalo = og1->getHaloSizes(pname, true);
                            auto right_ohalo = og1->getHaloSizes(pname, false);

#ifdef DEBUG_SCRATCH
                            cout << "** cH: processing " << b2->getDescr() << "...\n" 
                                "*** cH: LHS halos: " << left_ohalo.makeDimValStr() <<
                                " & " << right_ohalo.makeDimValStr() << endl;
#endif
                        
                            // Recalc min halos of all input grids of all
                            // scratch eqs in this bundle by adding size of
                            // output-grid halos.
                            for (auto& eq : b2->getEqs()) {
                                auto& inPts = pv.getInputPts().at(eq.get());

                                // Input points.
                                for (auto ip : inPts) {
                                    auto* ig = shadow_map[ip->getGrid()];
                                    auto& ao = ip->getArgOffsets(); // e.g., '2' for 'x+2'.

                                    // Increase range by subtracting left halos and
                                    // adding right halos.
                                    auto left_ihalo = ao.subElements(left_ohalo, false);
                                    ig->updateHalo(pname, left_ihalo);
                                    auto right_ihalo = ao.addElements(right_ohalo, false);
                                    ig->updateHalo(pname, right_ihalo);
#ifdef DEBUG_SCRATCH
                                    cout << "*** cH: updated min halos of '" << ig->get_name() << "' to " <<
                                        left_ihalo.makeDimValStr() <<
                                        " & " << right_ihalo.makeDimValStr() << endl;
#endif
                                } // input pts.
                            } // eqs in bundle.
                        } // path.
                    }); // lambda fn.
            } // bundles.
        } // packs.

        // Apply the changes from the shadow grids.
        // This will result in the grids containing the max
        // of the shadow halos.
        for (auto& shadow_map : shadows) {
#ifdef DEBUG_SCRATCH
            cout << "* cH: applying changes from a shadow map...\n";
#endif
            for (auto& si : shadow_map) {
                auto* orig_gp = si.first;
                auto* shadow_gp = si.second;
                assert(orig_gp);
                assert(shadow_gp);

                // Update the original.
                orig_gp->updateHalo(*shadow_gp);
#ifdef DEBUG_SCRATCH
                cout << "** cH: updated '" << orig_gp->get_name() << "'.\n";
#endif

                // Release the shadow grid.
                delete shadow_gp;
                shadow_map.at(orig_gp) = NULL;
            }
        }
    } // calcHalos().

    // Divide all equations into eqBundles.
    // Only process updates to grids in 'gridRegex'.
    // 'targets': string provided by user to specify bundleing.
    void EqBundles::makeEqBundles(Eqs& allEqs,
                                  const CompilerSettings& settings,
                                  ostream& os)
    {
        os << "\nPartitioning " << allEqs.getNum() << " equation(s) into bundles...\n";
        //auto& stepDim = _dims->_stepDim;

        // Add scratch equations.
        for (auto eq : allEqs.getAll()) {
            if (eq->isScratch()) {

                // Add equation.
                addEqToBundle(allEqs, eq, _basename_default, settings);
            }
        }

        // Make a regex for the allowed grids.
        regex gridx(settings._gridRegex);

        // Handle each key-value pair in 'targets' string.
        // Key is eq-bundle name (with possible format strings); value is regex pattern.
        ArgParser ap;
        ap.parseKeyValuePairs
            (settings._eqBundleTargets,
             [&](const string& egfmt, const string& pattern) {

                // Make a regex for the pattern.
                regex patx(pattern);

                // Search allEqs for matches to current value.
                for (auto eq : allEqs.getAll()) {

                    // Get name of updated grid.
                    auto gp = eq->getGrid();
                    assert(gp);
                    string gname = gp->getName();

                    // Match to gridx?
                    if (!regex_search(gname, gridx))
                        continue;

                    // Match to patx?
                    smatch mr;
                    if (!regex_search(gname, mr, patx))
                        continue;

                    // Substitute special tokens with match.
                    string egname = mr.format(egfmt);

                    // Add equation if allowed.
                    addEqToBundle(allEqs, eq, egname, settings);
                }
            });

        // Add all remaining equations.
        for (auto eq : allEqs.getAll()) {

            // Get name of updated grid.
            auto gp = eq->getGrid();
            assert(gp);
            string gname = gp->getName();

            // Match to gridx?
            if (!regex_search(gname, gridx))
                continue;

            // Add equation.
            addEqToBundle(allEqs, eq, _basename_default, settings);
        }

        os << "Collapsing dependencies from equations and finding transitive closure...\n";
        inherit_deps_from(allEqs);

        os << "Topologically ordering bundles...\n";
        topo_sort();

        // Dump info.
        os << "Created " << getNum() << " equation bundle(s):\n";
        for (auto& eg1 : getAll()) {
            os << " " << eg1->getDescr() << ":\n"
                "  Contains " << eg1->getNumEqs() << " equation(s).\n"
                "  Updates the following grid(s): ";
            int i = 0;
            for (auto* g : eg1->getOutputGrids()) {
                if (i++)
                    os << ", ";
                os << g->getName();
            }
            os << ".\n";

            // Deps.
            for (auto& eg2 : _deps.get_deps_on(eg1))
                os << "  Dependent on bundle " << eg2->getName() << ".\n";
            for (auto& sg : _scratches.get_deps_on(eg1))
                os << "  Requires scratch bundle " << sg->getName() << ".\n";
        }

    }

    // Apply optimizations according to the 'settings'.
    void EqBundles::optimizeEqBundles(CompilerSettings& settings,
                                    const string& descr,
                                    bool printSets,
                                    ostream& os) {
        // print stats.
        os << "Stats across " << getNum() << " equation-bundle(s):\n";
        string edescr = "for " + descr + " equation-bundle(s)";
        printStats(os, edescr);

        // Make a list of optimizations to apply to eqBundles.
        vector<OptVisitor*> opts;

        // CSE.
        if (settings._doCse)
            opts.push_back(new CseVisitor);

        // Operator combination.
        if (settings._doComb) {
            opts.push_back(new CombineVisitor);

            // Do CSE again after combination.
            // TODO: do this only if the combination did something.
            if (settings._doCse)
                opts.push_back(new CseVisitor);
        }

        // Pairs.
        if (settings._doPairs)
            opts.push_back(new PairingVisitor);

        // Apply opts.
        for (auto optimizer : opts) {

            visitEqs(optimizer);
            int numChanges = optimizer->getNumChanges();
            string odescr = "after applying " + optimizer->getName() + " to " +
                descr + " equation-bundle(s)";

            // Get new stats.
            if (numChanges)
                printStats(os, odescr);
            else
                os << " No changes " << odescr << '.' << endl;
        }

        // Final stats per equation bundle.
        if (printSets && getNum() > 1) {
            os << "Stats per equation-bundle:\n";
            for (auto eg : getAll())
                eg->printStats(os, "for " + eg->getDescr());
        }
    }

    // Make a human-readable description of this eq bundle pack.
    string EqBundlePack::getDescr(string quote) const
    {
        string des;
        if (isScratch())
            des += "scratch ";
        des += "equation bundle-pack " + quote + getName() + quote;
        if (!isScratch()) {
            if (step_cond.get())
                des += " w/step condition " + step_cond->makeQuotedStr(quote);
            else
                des += " w/o step condition";
        }
        return des;
    }

    // Add a bundle to this pack.
    void EqBundlePack::addBundle(EqBundlePtr bp)
    {
        _bundles.insert(bp);
        _isScratch = bp->isScratch();

        // update list of eqs.
        for (auto& eq : bp->getEqs())
            _eqs.insert(eq);

        // update list of input and output grids for this pack.
        for (auto& g : bp->getOutputGrids())
            _outGrids.insert(g);
        for (auto& g : bp->getInputGrids())
            _inGrids.insert(g);
    }

    // Add 'bp' from 'allBundles'. Create new pack if needed.  Returns
    // whether a new pack was created.
    bool EqBundlePacks::addBundleToPack(EqBundles& allBundles,
                                        EqBundlePtr bp)
    {
        // Already added?
        if (_bundles_in_packs.count(bp))
            return false;

        // Get condition, if any.
        auto stcond = bp->step_cond;
        
        // Get deps between bundles.
        auto& deps = allBundles.getDeps();

        // Loop through existing packs, looking for one that
        // 'bp' can be added to.
        EqBundlePack* target = 0;
        for (auto& ep : getAll()) {

            // Must be same scratch-ness.
            if (ep->isScratch() != bp->isScratch())
                continue;

            // Step conditions must match (both may be null).
            if (!areExprsSame(ep->step_cond, stcond))
                continue;

            // Look for any dependencies that would prevent adding
            // 'bp' to 'ep'.
            bool is_ok = true;
            for (auto& bp2 : ep->getBundles()) {

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

        // Make new pack if no target pack found.
        bool newPack = false;
        if (!target) {
            auto np = make_shared<EqBundlePack>(bp->isScratch());
            addItem(np);
            target = np.get();
            target->baseName = _baseName;
            target->index = _idx++;
            target->step_cond = stcond;
            newPack = true;
        }

        // Add bundle to target.
        assert(target);
        target->addBundle(bp);

        // Remember pack and updated grids.
        _bundles_in_packs.insert(bp);
        for (auto& g : bp->getOutputGrids())
            _outGrids.insert(g);

        return newPack;
    }

    // Divide all bundles into packs.
    void EqBundlePacks::makePacks(EqBundles& allBundles,
                                  ostream& os)
    {
        os << "\nPartitioning " << allBundles.getNum() << " bundle(s) into packs...\n";

        for (auto bp : allBundles.getAll())
            addBundleToPack(allBundles, bp);

        os << "Collapsing dependencies from bundles and finding transitive closure...\n";
        inherit_deps_from(allBundles);

        os << "Topologically ordering packs...\n";
        topo_sort();

        // Dump info.
        os << "Created " << getNum() << " equation bundle pack(s):\n";
        for (auto& bp1 : getAll()) {
            os << " " << bp1->getDescr() << ":\n"
                "  Contains " << bp1->getBundles().size() << " bundle(s): ";
            int i = 0;
            for (auto b : bp1->getBundles()) {
                if (i++)
                    os << ", ";
                os << b->getName();
            }
            os << ".\n";
            os << "  Updates the following grid(s): ";
            i = 0;
            for (auto* g : bp1->getOutputGrids()) {
                if (i++)
                    os << ", ";
                os << g->getName();
            }
            os << ".\n";

            // Deps.
            for (auto& bp2 : _deps.get_deps_on(bp1))
                os << "  Dependent on bundle pack " << bp2->getName() << ".\n";
            for (auto& sp : _scratches.get_deps_on(bp1))
                os << "  Requires scratch pack " << sp->getName() << ".\n";
        }

    }

} // namespace yask.
