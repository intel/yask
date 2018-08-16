/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

        // A set of all points to ensure pointers to each
        // unique point have same value.
        set<GridPoint> _all_pts;

        // A type to hold a mapping of equations to a set of grids in each.
        typedef unordered_set<Grid*> GridSet;
        typedef unordered_map<EqualsExpr*, Grid*> GridMap;
        typedef unordered_map<EqualsExpr*, GridSet> GridSetMap;

        GridMap _lhs_grids; // outputs of eqs.
        GridSetMap _rhs_grids; // inputs of eqs.

        // A type to hold a mapping of equations to a set of points in each.
        typedef unordered_set<GridPoint*> PointSet;
        typedef unordered_map<EqualsExpr*, GridPoint*> PointMap;
        typedef unordered_map<EqualsExpr*, PointSet> PointSetMap;

        PointMap _lhs_pts; // outputs of eqs.
        PointSetMap _rhs_pts; // inputs of eqs.

        EqualsExpr* _eq=0;   // Current equation.

    public:

        // Ctor.
        // 'pts' contains offsets from each point to create.
        PointVisitor() {}
        virtual ~PointVisitor() {}

        GridMap& getOutputGrids() { return _lhs_grids; }
        GridSetMap& getInputGrids() { return _rhs_grids; }
        PointMap& getOutputPts() { return _lhs_pts; }
        PointSetMap& getInputPts() { return _rhs_pts; }
        int getNumEqs() const { return (int)_lhs_pts.size(); }

        // Determine whether 2 sets have any common points.
        virtual bool do_sets_intersect(const GridSet& a,
                                       const GridSet& b) {
            for (auto ai : a) {
                if (b.count(ai) > 0)
                    return true;
            }
            return false;
        }
        virtual bool do_sets_intersect(const PointSet& a,
                                       const PointSet& b) {
            for (auto ai : a) {
                if (b.count(ai) > 0)
                    return true;
            }
            return false;
        }

        // Callback at an equality.
        // Handles LHS grid pt explicitly, then visits RHS.
        virtual void visit(EqualsExpr* ee) {

            // Set this equation as current one.
            _eq = ee;

            // Make sure map entries exist for this eq.
            _lhs_grids[_eq];
            _rhs_grids[_eq];
            _lhs_pts[_eq];
            _rhs_pts[_eq];

            // Store LHS point.
            auto* lhs = ee->getLhs().get();
            _lhs_pts[_eq] = lhs;

            // Add grid.
            auto* g = lhs->getGrid();
            _lhs_grids[_eq] = g;

            // visit RHS.
            NumExprPtr rhs = ee->getRhs();
            rhs->accept(this);

            // Don't visit LHS because we've already saved it.
        }

        // Callback at a grid point on the RHS.
        virtual void visit(GridPoint* gp) {
            assert(_eq);

            // Store RHS point.
            _rhs_pts[_eq].insert(gp);

            // Add grid.
            auto* g = gp->getGrid();
            _rhs_grids[_eq].insert(g);
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
            NumExprPtr step_expr1 = op1->getArg(stepDim); // may be null.

#ifdef DEBUG_DEP
            cout << " Checking internal consistency of equation " <<
                eq1->makeQuotedStr() << "...\n";
#endif

            // Scratch grid must not have a condition.
            if (cond1 && og1->isScratch())
                THROW_YASK_EXCEPTION("Error: scratch-grid equation '" + eq1->makeQuotedStr() +
                                     "' cannot have a condition");

            // Check LHS grid dimensions and associated args.
            for (int di = 0; di < og1->get_num_dims(); di++) {
                auto& dn = og1->get_dim_name(di);  // name of this dim.
                auto argn = op1->getArgs().at(di); // LHS arg for this dim.

                // Check based on dim type.
                if (dn == stepDim) {

                    // Scratch grid must not use step dim.
                    if (og1->isScratch())
                        THROW_YASK_EXCEPTION("Error: scratch-grid '" + og1->getName() +
                                             "' cannot use '" + dn + "' dim");

                    // Validity of step-dim expression in non-scratch grids is checked later.
                }

                // LHS must have simple indices in domain dims.
                else if (dims._domainDims.lookup(dn)) {

                    // Make expected arg, e.g., 'x'.
                    auto earg = make_shared<IndexExpr>(dn, DOMAIN_INDEX);

                    // Compare to actual.
                    if (!argn->isSame(earg))
                        THROW_YASK_EXCEPTION("Error: LHS of equation " + eq1->makeQuotedStr() +
                                             " contains expression " + argn->makeQuotedStr() +
                                             " for dimension '" + dn +
                                             "' where " + earg->makeQuotedStr() +
                                             " is expected");
                }

                // Misc dim must be a const.
                else {

                    if (!argn->isConstVal())
                        THROW_YASK_EXCEPTION("Error: LHS of equation " + eq1->makeQuotedStr() +
                                             " contains expression " + argn->makeQuotedStr() +
                                             " for dimension '" + dn +
                                             "' where constant integer is expected");
                    argn->getIntVal(); // throws exception if not an integer.
                }
            }
        
            // Compare step offsets of LHS and RHS of non-scratch eq.
            if (!og1->isScratch()) {

                if (!step_expr1)
                    THROW_YASK_EXCEPTION("Error: non-scratch-grid '" + og1->getName() +
                                         "' does not use '" + stepDim + "' dim");
                
                // See if step arg is a simple offset, e.g., 'u(t+1, ...)'.
                // This is the most common case.
                auto& lofss = op1->getArgOffsets();
                auto* lofsp = lofss.lookup(stepDim); // offset at step dim.
                if (lofsp) {
                    auto lofs = *lofsp;

                    // Soln step-direction heuristic.
                    // Assume 'u(t+1, ...) EQUALS ...' implies forward,
                    // and 'u(t-1, ...) EQUALS ...' implies backward.
                    // TODO: improve this; at least handle
                    // 'u(t, ...) EQUALS ... u(t +/- 1, ...) ...',
                    // but need to look thru scratch vars to get this right.
                    if (dims._stepDir == 0 && lofs != 0)
                        dims._stepDir = (lofs > 0) ? 1 : -1;

                    // Scan input (RHS) points.
                    for (auto i1 : ip1) {
                        
                        // Is point a simple offset from step, e.g., 'u(t-2, ...)'?
                        auto* rsi1p = i1->getArgOffsets().lookup(stepDim);
                        if (rsi1p) {
                            int rsi1 = *rsi1p;

                            // Must be in proper relation to LHS, e.g.,
                            // the following are illegal given the heuristic
                            // used above.
                            // forward: 'u(t+1, ...) EQUALS ... u(t+2, ...) ...',
                            // backward: 'u(t-1, ...) EQUALS ... u(t-2, ...) ...'.
                            if ((lofs > 0 && rsi1 > lofs) ||
                                (lofs < 0 && rsi1 < lofs)) {
                                THROW_YASK_EXCEPTION("Error: RHS of equation " +
                                                     eq1->makeQuotedStr() +
                                                     " contains '" + dims.makeStepStr(rsi1) +
                                                     "', which is incompatible with '" +
                                                     dims.makeStepStr(lofs) +
                                                     "' on LHS");
                            }
                        }
                    } // for all RHS points.
                }
            }

            // LHS of equation must be vectorizable.
            // TODO: relax this restriction.
            if (op1->getVecType() != GridPoint::VEC_FULL) {
                THROW_YASK_EXCEPTION("Error: LHS of equation " + eq1->makeQuotedStr() +
                                     " is not fully vectorizable because not all folded"
                                     " dimensions are accessed via simple offsets from their respective indices");
            }

            // TODO: check that domain indices are simple offsets and
            // misc indices are consts on RHS.

            // TODO: check to make sure cond1 depends only on indices.
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

#ifdef DEBUG_DEP
                cout << " Checking eq " <<
                    eq1->makeQuotedStr() << " vs " <<
                    eq2->makeQuotedStr() << "...\n";
#endif

                bool same_eq = eq1 == eq2;
                bool same_cond = areExprsSame(cond1, cond2);

                // A separate grid is defined by its name and any const indices.
                bool same_og = op1->isSameLogicalGrid(*op2);

                // If two different eqs have the same condition, they
                // cannot update the same grid.
                if (!same_eq && same_cond && same_og) {
                    string cdesc = cond1 ? "with condition " + cond1->makeQuotedStr() :
                        "without conditions";
                    THROW_YASK_EXCEPTION("Error: two equations " + cdesc +
                                         " have the same LHS grid '" +
                                         op1->makeLogicalGridStr() + "': " +
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
                if (same_cond && ip2.count(op1)) {

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

        // Resolve indirect dependencies.
        // Do this even if not finding deps because we want to
        // resolve deps provided by the user.
        os << "Finding transitive closure...\n";
        find_all_deps();

        // Sort.
        os << "Topologically ordering...\n";
        topo_sort();
    }

    // Visitor for determining vectorization potential of grid points.
    // Vectorization depends not only on the dims of the grid itself
    // but also on how the grid is indexed at each point.
    class SetVecVisitor : public ExprVisitor {
        const Dimensions& _dims;

    public:
        SetVecVisitor(const Dimensions& dims) :
            _dims(dims) { }

        // Check each grid point in expr.
        virtual void visit(GridPoint* gp) {
            auto* grid = gp->getGrid();

            // Never vectorize scalars.
            if (grid->get_num_dims() == 0) {
                gp->setVecType(GridPoint::VEC_NONE);
                return;
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
            if (fdoffsets == soln_nfd)
                gp->setVecType(GridPoint::VEC_FULL); // all good.

            // Some dims are vectorizable?
            else if (fdoffsets > 0)
                gp->setVecType(GridPoint::VEC_PARTIAL);

            // Uses no folded dims, so scalar only.
            else
                gp->setVecType(GridPoint::VEC_NONE);
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
        virtual void visit(IndexExpr* ie) {
            vars_used.insert(ie->getName());
        }
    };

    // Visitor for determining inner-loop accesses of grid points.
    class SetLoopVisitor : public ExprVisitor {
        const Dimensions& _dims;

    public:
        SetLoopVisitor(const Dimensions& dims) :
            _dims(dims) { }

        // Check each grid point in expr.
        virtual void visit(GridPoint* gp) {

            // Info from grid.
            auto* grid = gp->getGrid();
            auto gdims = grid->get_dim_names();

            // Check loop in this dim.
            auto idim = _dims._innerDim;

            // Access type.
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
        }
    };

    // Determine loop access behavior of grid points.
    void Eqs::analyzeLoop(const Dimensions& dims) {

        // Send a 'SetLoopVisitor' to each point in
        // the current equations.
        SetLoopVisitor slv(dims);
        visitEqs(&slv);
    }

    // Update access stats for the grids, i.e., halos and const indices.
    // Also finds scratch-grid eqs needed for each non-scratch eq.
    void Eqs::updateGridStats() {

        // Find all LHS and RHS points and grids for all eqs.
        PointVisitor pv;
        visitEqs(&pv);

        // Analyze each eq.
        for (auto& eq1 : getAll()) {

            // Get sets of points for this eq.
            auto* outPt1 = pv.getOutputPts().at(eq1.get());
            auto& inPts1 = pv.getInputPts().at(eq1.get());
            auto* og1 = pv.getOutputGrids().at(eq1.get());

            // Union of input and output points.
            auto allPts1 = inPts1;
            allPts1.insert(outPt1);

            // Update stats based on explicit accesses as
            // written in original 'eq1'.
            for (auto ap : allPts1) {
                auto* g = ap->getGrid(); // grid written to.
                g->updateHalo(ap->getArgOffsets());
                g->updateConstIndices(ap->getArgConsts());
            }

            // We want to start with non-scratch eqs and walk the dep
            // tree to find all dependent scratch eqs.
            // If 'eq1' has a non-scratch output, visit all dependencies of
            // 'eq1'.  It's important to visit the eqs in dep order to
            // properly propagate halos sizes thru chains of scratch grids.
            // TODO: clean up this obfuscated, hard-to-follow, and fragile code.
            if (!og1->isScratch()) {
                getDeps().visitDeps

                    // 'eq1' is 'b' or depends on 'b', immediately or indirectly.
                    (eq1, [&](EqualsExprPtr b, EqList& path) {

                        // Does 'b' have a scratch-grid output?
                        // NB: scratch eqs don't have their own conditions, so
                        // we don't need to check them.
                        auto* og2 = pv.getOutputGrids().at(b.get());
                        if (og2->isScratch()) {

                            // Get halos from the output scratch grid.
                            // These are the points that are read from
                            // the dependent eq(s).
                            // For scratch grids, the halo areas must also be written to.
                            auto _left_ohalo = og2->getHaloSizes(true);
                            auto _right_ohalo = og2->getHaloSizes(false);

                            // Expand halos of all input grids by size of output-grid halo.
                            auto& inPts2 = pv.getInputPts().at(b.get());
                            for (auto ip2 : inPts2) {
                                auto* ig2 = ip2->getGrid();
                                auto& ao2 = ip2->getArgOffsets();

                                // Increase range by subtracing left halos and
                                // adding right halos.
                                auto _left_ihalo = ao2.subElements(_left_ohalo, false);
                                ig2->updateHalo(_left_ihalo);
                                auto _right_ihalo = ao2.addElements(_right_ohalo, false);
                                ig2->updateHalo(_right_ihalo);
                            }
                        }

                        // Find scratch-grid eqs in this dep path that are
                        // needed for 'eq1'. Walk dep path from 'eq1' to 'b'.
                        EqualsExprPtr prev;
                        for (auto eq2 : path) {

                            // Look for scratch-grid dep from 'prev' to 'eq2'.
                            if (prev) {

                                // Find any scratch-grid inputs of 'prev'.
                                set<Grid*> targets;
                                auto& inPts2 = pv.getInputPts().at(prev.get());
                                for (auto ip2 : inPts2) {
                                    auto* ig2 = ip2->getGrid();
                                    if (ig2->isScratch())
                                        targets.insert(ig2);
                                }

                                // If there are none, we are done.
                                if (targets.empty())
                                    break;

                                // If 'eq2' output-grid is one of the
                                // scratch-grid targets, add it to the set
                                // needed for 'eq1'.
                                auto* og2 = pv.getOutputGrids().at(eq2.get());
                                if (targets.count(og2))
                                    getScratches().set_imm_dep_on(eq1, eq2);
                                else
                                    break;
                            }
                            prev = eq2;
                        }

                    });
            }
        }
    }


    // Get the full name of an eq-lot.
    // Must be unique.
    string EqLot::getName() const {

        // Add index to base name.
        ostringstream oss;
        oss << baseName << "_" << index;
        return oss.str();
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
                des += " w/condition " + cond->makeQuotedStr(quote);
            else
                des += " w/o condition";
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
            _ofs(ofs) {}

        // Visit a grid point.
        virtual void visit(GridPoint* gp) {

            // Shift grid _ofs points.
            auto ofs0 = gp->getArgOffsets();
            IntTuple new_loc = ofs0.addElements(_ofs, false);
            gp->setArgOffsets(new_loc);
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
                                  const string& baseName) {

        // Equation already added?
        if (_eqs_in_bundles.count(eq))
            return false;

        // Get condition, if any.
        auto cond = eq->getCond();

        // Get deps between eqs.
        auto& eq_deps = allEqs.getDeps();

        // Loop through existing bundles, looking for one that
        // 'eq' can be added to.
        EqBundle* target = 0;
        for (auto& eg : getAll()) {

            // Must be same scratch-ness.
            if (eg->isScratch() != eq->isScratch())
                continue;

            // Must match name and condition.
            if (eg->baseName != baseName || !areExprsSame(eg->cond, cond))
                continue;

            // Look for any condition or dependencies that would prevent
            // adding 'eq' to 'eg'.
            bool is_ok = true;
            for (auto& eq2 : eg->getEqs()) {

                // If scratch, 'eq' and 'eq2' must have same halo.
                // This is because scratch halos are written to.
                if (eq->isScratch()) {
                    auto eq2g = eq2->getGrid();
                    auto eqg = eq->getGrid();
                    if (!eq2g->isHaloSame(*eqg))
                        is_ok = false;
                }

                // Look for any dependency between 'eq' and 'eq2'.
                if (eq_deps.is_dep(eq, eq2)) {
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

    // Divide all equations into eqBundles.
    // Only process updates to grids in 'gridRegex'.
    // 'targets': string provided by user to specify bundleing.
    void EqBundles::makeEqBundles(Eqs& allEqs,
                                  const string& gridRegex,
                                  const string& targets,
                                  ostream& os)
    {
        os << "\nPartitioning " << allEqs.getNum() << " equation(s) into bundles...\n";
        //auto& stepDim = _dims->_stepDim;

        // Add scratch equations.
        for (auto eq : allEqs.getAll()) {
            if (eq->isScratch()) {

                // Add equation.
                addEqToBundle(allEqs, eq, _basename_default);
            }
        }

        // Make a regex for the allowed grids.
        regex gridx(gridRegex);

        // Handle each key-value pair in 'targets' string.
        // Key is eq-bundle name (with possible format strings); value is regex pattern.
        ArgParser ap;
        ap.parseKeyValuePairs
            (targets, [&](const string& egfmt, const string& pattern) {

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

                    // Add equation.
                    addEqToBundle(allEqs, eq, egname);
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
            addEqToBundle(allEqs, eq, _basename_default);
        }

        os << "Finding transitive closure...\n";
        inherit_deps_from(allEqs);

        os << "Topologically ordering...\n";
        topo_sort();

        // Dump info.
        os << "Created " << getNum() << " equation bundle(s):\n";
        for (auto& eg1 : _all) {
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
        os << "Stats across " << getNum() << " equation-bundle(s) before optimization(s):\n";
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
                os << "No changes " << odescr << '.' << endl;
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

        // Get deps between bundles.
        auto& deps = allBundles.getDeps();

        // Loop through existing packs, looking for one that
        // 'bp' can be added to.
        EqBundlePack* target = 0;
        for (auto& ep : getAll()) {

            // Must be same scratch-ness.
            if (ep->isScratch() != bp->isScratch())
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

        os << "Finding transitive closure...\n";
        inherit_deps_from(allBundles);

        os << "Topologically ordering...\n";
        topo_sort();

        // Dump info.
        os << "Created " << getNum() << " equation bundle pack(s):\n";
        for (auto& bp1 : _all) {
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
