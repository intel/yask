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

///////// Methods for equations and equation groups ////////////

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Parse.hpp"
#include "Eqs.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {

    // A visitor to collect grids and points visited in a set of eqs.
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

    // Visit current node 'a' and recurse.
    void EqDeps::_visitDeps(EqualsExprPtr a,
                           std::function<void (EqualsExprPtr b, EqVecSet& path)> visitor,
                           EqVecSet* seen) const {
        
        // Already visited, i.e., a loop?
        bool was_seen = (seen && seen->count(a));
        if (was_seen)
            return;

        // Add 'a' to copy of path.
        EqVecSet seen1;
        if (seen)
            seen1 = *seen; // copy nodes already seen.
        seen1.insert(a);   // add this one.
        
        // Call lambda fn.
        visitor(a, seen1);
                    
        // any dependencies?
        if (_imm_deps.count(a)) {
            auto& adeps = _imm_deps.at(a);
    
            // loop thru deps of 'a', i.e., each 'b' deps on 'a'.
            for (auto b : adeps) {

                // Recurse to deps of 'b'.
                _visitDeps(b, visitor, &seen1);
            }
        }
    }

    // Does recursive analysis to turn all indirect dependencies to direct
    // ones.
    void EqDeps::find_all_deps() {
        if (_done)
            return;
        for (auto a : _all)
            if (_full_deps.count(a) == 0)
                visitDeps(a, [&](EqualsExprPtr b, EqVecSet& path) {
                            
                        // Walk path from ee to b.
                        // Every 'eq' in 'path' before 'b' depends on 'b'.
                        for (auto eq : path)
                            if (eq != b)
                                _full_deps[eq].insert(b);
                    });
        _done = true;
    }
    
    // Find dependencies based on all eqs.
    // If 'eq_deps' is set, save dependencies between eqs.
    // Side effect: sets _stepDir in dims.
    // Throws exceptions on illegal dependencies.
    // TODO: split this into smaller functions.
    // BIG-TODO: replace dependency algorithms with integration of a polyhedral
    // library.
    void Eqs::findDeps(Dimensions& dims,
                       EqDepMap* eq_deps,
                       ostream& os) {
        auto& stepDim = dims._stepDim;
        
        // Gather points from all eqs in all grids.
        PointVisitor pt_vis;

        // Gather initial stats from all eqs.
        os << "Scanning " << getEqs().size() << " equation(s) for dependencies...\n";
        for (auto eq1 : getEqs())
            eq1->accept(&pt_vis);
        auto& outGrids = pt_vis.getOutputGrids();
        auto& inGrids = pt_vis.getInputGrids();
        auto& outPts = pt_vis.getOutputPts();
        auto& inPts = pt_vis.getInputPts();
        
        // Check each eq.
        for (auto eq1 : getEqs()) {
            auto* eq1p = eq1.get();
            assert(outGrids.count(eq1p));
            assert(inGrids.count(eq1p));
            auto* og1 = outGrids.at(eq1p);
            assert(og1 == eq1->getGrid());
            auto* op1 = outPts.at(eq1p);
            //auto& ig1 = inGrids.at(eq1p);
            auto& ip1 = inPts.at(eq1p);
            auto cond1 = getCond(eq1p);

#ifdef DEBUG_DEP
            cout << " Checking internal consistency of equation " <<
                eq1->makeQuotedStr() << "...\n";
#endif
            int lofs = 0;        // step index offset for LHS of eq1.

            // Check LHS indices.
            for (int di = 0; di < og1->get_num_dims(); di++) {
                auto& dn = og1->get_dim_name(di);
                auto argn = op1->getArgs().at(di);

                if (dn == stepDim) {

                    // Scratch grid must not use step dim.
                    if (og1->isScratch())
                        THROW_YASK_EXCEPTION("Error: scratch-grid '" << og1->getName() <<
                                             "' cannot use '" << dn << "' dim");
                }

                // LHS must have simple indices in domain dims.
                else if (dims._domainDims.lookup(dn)) {

                    // Make expected arg.
                    auto earg = make_shared<IndexExpr>(dn, DOMAIN_INDEX);

                    // Compare to actual.
                    if (!argn->isSame(earg))
                        THROW_YASK_EXCEPTION("Error: LHS of equation " << eq1->makeQuotedStr() <<
                                             " contains expression " << argn->makeQuotedStr() <<
                                             " where " << earg->makeQuotedStr() <<
                                             " is expected");
                }

                // Misc dim must be a const.
                else {

                    if (!argn->isConstVal())
                        THROW_YASK_EXCEPTION("Error: LHS of equation " << eq1->makeQuotedStr() <<
                                             " contains expression " << argn->makeQuotedStr() <<
                                             " where constant integer is expected");
                    argn->getIntVal(); // throws exception if not an integer.
                }
            }

            // LHS of a non-scratch eq must use step dim w/a simple +/-1 offset.
            if (!og1->isScratch()) {
                auto& lofss = op1->getArgOffsets();
                auto* lofsp = lofss.lookup(stepDim);
                if (!lofsp || abs(*lofsp) != 1) {
                    THROW_YASK_EXCEPTION("Error: LHS of equation " << eq1->makeQuotedStr() <<
                                         " does not contain '" << dims.makeStepStr(1) <<
                                         "' or '" << dims.makeStepStr(-1) << "'");
                }
                lofs = *lofsp;

                // Step direction already set?
                if (dims._stepDir) {
                    if (dims._stepDir != lofs) {
                        THROW_YASK_EXCEPTION("Error: LHS of equation " << eq1->makeQuotedStr() <<
                                             " contains '" << dims.makeStepStr(lofs) <<
                                             "', which is different than a previous equation with '" <<
                                             dims.makeStepStr(dims._stepDir) << "'");
                    }
                } else
                    
                    // Side effect: store step direction.
                    dims._stepDir = lofs;
            }
            
            // LHS of equation must be vectorizable.
            // TODO: relax this restriction.
            if (op1->getVecType() != GridPoint::VEC_FULL) {
                THROW_YASK_EXCEPTION("Error: LHS of equation " << eq1->makeQuotedStr() <<
                                     " is not fully vectorizable because not all folded"
                                     " dimensions are accessed via simple offsets from their respective indices");
            }

            // Scan input (RHS) points.
            for (auto i1 : ip1) {

                // Check RHS of an equation that uses step index.
                auto* rsi1p = i1->getArgOffsets().lookup(stepDim);
                if (rsi1p) {
                    int rsi1 = *rsi1p;

                    // Must be in proper relation to LHS.
                    if ((lofs > 0 && rsi1 > lofs) ||
                        (lofs < 0 && rsi1 < lofs)) {
                        THROW_YASK_EXCEPTION("Error: RHS of equation " <<
                                             eq1->makeQuotedStr() <<
                                             " contains '" << dims.makeStepStr(rsi1) << 
                                             "', which is incompatible with '" << dims.makeStepStr(lofs) << 
                                             "' on LHS");
                    }
                }
            }

            // TODO: check to make sure cond1 depends only on indices.
            
#ifdef DEBUG_DEP
            cout << " Checking dependencies on equation " <<
                eq1->makeQuotedStr() << "...\n";
#endif

            // Check each 'eq2' to see if it depends on 'eq1'.
            for (auto eq2 : getEqs()) {
                auto* eq2p = eq2.get();
                auto& og2 = outGrids.at(eq2p);
                assert(og2 == eq2->getGrid());
                //auto& op2 = outPts.at(eq2p);
                auto& ig2 = inGrids.at(eq2p);
                auto& ip2 = inPts.at(eq2p);
                auto cond2 = getCond(eq2p);

                bool same_eq = eq1 == eq2;
                bool same_cond = areExprsSame(cond1, cond2);
                bool same_og = og1 == og2;

#ifdef DEBUG_DEP
                if (!same_eq)
                    cout << "  ...from equation " << eq2->makeQuotedStr() << "...\n";
#endif

                // If two different eqs have the same condition, they
                // cannot update the same grid.
                if (!same_eq && same_cond && same_og) {
                    string cdesc = cond1 ? "with condition " + cond1->makeQuotedStr() :
                        "without conditions";
                    THROW_YASK_EXCEPTION("Error: two equations " << cdesc <<
                                         " have the same LHS grid '" << og1->getName() << "': " <<
                                         eq1->makeQuotedStr() << " and " <<
                                         eq2->makeQuotedStr());
                }

                // First dep check: exact matches on LHS of eq1 to RHS of eq2.
                // eq2 dep on eq1 => some output of eq1 is an input to eq2.
                // If the two eqs have the same condition, detect
                // dependencies by looking for exact matches.
                // We do this check first because it's quicker than the
                // detailed scan done later if this one doesn't find a dep.
                //
                // Example:
                //  eq1: a(t+1, x, ...) EQUALS ...
                //  eq2: b(t+1, x, ...) EQUALS a(t+1, x, ...) ...
                if (same_cond && ip2.count(op1)) {

                    // Eq depends on itself?
                    if (same_eq) {
                                    
                        // Exit with error.
                        THROW_YASK_EXCEPTION("Error: illegal dependency: LHS of equation " <<
                                             eq1->makeQuotedStr() << " also appears on its RHS");
                    }

                    // Save dependency.
                    if (eq_deps) {
#ifdef DEBUG_DEP
                        cout << "  Exact match found to " << op1->makeQuotedStr() << ".\n";
#endif                        
                        (*eq_deps)[cur_step_dep].set_imm_dep_on(eq2, eq1);
                    }
                        
                    // Move along to next eq2.
                    continue;
                }

                // Check more only if saving dependencies.
                if (!eq_deps)
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

                        // Same grid?
                        auto* i2g = i2->getGrid();
                        if (i2g != og1) continue;

                        // From same step index, e.g., same time?
                        // Or, passing data thru a temp var?
                        // TODO: check that constant indices are same.
                        auto* si2p = i2->getArgOffsets().lookup(stepDim);
                        bool same_step = si2p && lofs && (*si2p == lofs);
                        if (same_step || og1->isScratch()) {

                            // Eq depends on itself?
                            if (same_eq) {
                                
                                // Exit with error.
                                string stepmsg = same_step ? " at '" + dims.makeStepStr(lofs) + "'" : "";
                                THROW_YASK_EXCEPTION("Error: disallowed dependency: grid on LHS of equation " <<
                                                     eq1->makeQuotedStr() << " also appears on its RHS" <<
                                                     stepmsg);
                            }

                            // Save dependency.
                            if (eq_deps) {
#ifdef DEBUG_DEP
                                cout << "  Likely match found to " << op1->makeQuotedStr() << ".\n";
#endif                        
                                (*eq_deps)[cur_step_dep].set_imm_dep_on(eq2, eq1);
                            }

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

        // Resolve indirect dependencies.
        if (eq_deps) {
            os << " Resolving indirect dependencies...\n";
            for (DepType dt = DepType(0); dt < num_deps; dt = DepType(dt+1))
                (*eq_deps)[dt].find_all_deps();
        }
        os << " Done with dependency analysis.\n";
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

    // Update access stats for the grids.
    // Also finds scratch-grid eqs needed for each non-scratch eq.
    void Eqs::updateGridStats(EqDepMap& eq_deps) {

        // Find all LHS and RHS points and grids for all eqs.
        PointVisitor pv;
        for (auto& eq : _eqs)
            eq->accept(&pv);

        // Analyze each eq.
        for (auto& eq1 : _eqs) {

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

            // If 'eq1' has a non-scratch output, visit all dependencies of
            // 'eq1'.  It's important to visit the eqs in dep order to
            // properly propagate halos sizes thru chains of scratch grids.
            if (!og1->isScratch()) {
                eq_deps[cur_step_dep].visitDeps

                    // 'eq1' is 'b' or depends on 'b', immediately or indirectly.
                    (eq1, [&](EqualsExprPtr b, EqDeps::EqVecSet& path) {

                        // Only check if conditions are same.
                        auto cond1 = getCond(eq1);
                        auto cond2 = getCond(b);
                        bool same_cond = areExprsSame(cond1, cond2);
                        
                        // Does 'b' have a scratch-grid output?
                        auto* og2 = pv.getOutputGrids().at(b.get());
                        if (same_cond && og2->isScratch()) {

                            // Get halos from the output scratch grid.
                            // These are the points that are read from
                            // in dependent eq(s).
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

                            // Only continue if conditions are same.
                            auto cond1 = getCond(eq1);
                            auto cond2 = getCond(eq2);
                            if (!areExprsSame(cond1, cond2))
                                break;

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
                                    _scratch_deps[eq1].insert(eq2);
                                else
                                    break;
                            }
                            prev = eq2;
                        }
                        
                    });
            }
        }
    }
   
   
    // Get the full name of an eq-group.
    // Must be unique.
    string EqGroup::getName() const {

        // Add index to base name.
        ostringstream oss;
        oss << baseName << "_" << index;
        return oss.str();
    }

    // Make a human-readable description of this eq group.
    string EqGroup::getDescription(bool show_cond,
                                   string quote) const
    {
        string des = "equation-group " + quote + getName() + quote;
        if (show_cond) {
            if (cond.get())
                des += " w/condition " + cond->makeQuotedStr(quote);
            else
                des += " w/no condition";
        }
        return des;
    }

    // Add an equation to an EqGroup.
    void EqGroup::addEq(EqualsExprPtr ee)
    {
#ifdef DEBUG_EQ_GROUP
        cout << "EqGroup: adding " << ee->makeQuotedStr() << endl;
#endif
        _eqs.insert(ee);

        // Get I/O point data from eq 'ee'.
        PointVisitor pv;
        ee->accept(&pv);
        
        // update list of input and output grids for this group.
        auto* outGrid = pv.getOutputGrids().at(ee.get());
        _outGrids.insert(outGrid);
        auto& inGrids = pv.getInputGrids().at(ee.get());
        for (auto* g : inGrids)
            _inGrids.insert(g);
    }

    // Check for and set dependencies on eg2.
    void EqGroup::checkDeps(Eqs& allEqs, EqDepMap& eq_deps, const EqGroup& eg2)
    {
        // Eqs in this.
        for (auto& eq1 : getEqs()) {
            auto& sdeps1 = allEqs.getScratchDeps(eq1);

            // Eqs in eg2.
            for (auto& eq2 : eg2.getEqs()) {

                for (DepType dt = DepType(0); dt < num_deps; dt = DepType(dt+1)) {

                    // Immediate dep.
                    if (eq_deps[dt].is_imm_dep_on(eq1, eq2)) {
                        _imm_dep_on[dt].insert(eg2.getName());
                        _dep_on[dt].insert(eg2.getName());
                    }

                    // Indirect dep.
                    else if (eq_deps[dt].is_dep_on(eq1, eq2)) {
                        _dep_on[dt].insert(eg2.getName());
                    }
                }

                // Scratch-grid dep.
                if (sdeps1.count(eq2))
                    _scratch_deps.insert(eg2.getName());
            }
        }
    }


    // Print stats from eqGroup.
    void EqGroup::printStats(ostream& os, const string& msg)
    {
        CounterVisitor cv;
        visitEqs(&cv);
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
    void EqGroup::replicateEqsInCluster(Dimensions& dims)
    {
        // Make a copy of the original equations so we can iterate through
        // them while adding to the group.
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
                        auto eq2 = eq->cloneEquals();

                        // Add offsets to each grid point.
                        OffsetVisitor ov(clusterOffset);
                        eq2->accept(&ov);

                        // Put new equation into group.
                        addEq(eq2);
                    }
                }
                return true;
            });

        // Ensure the expected number of equations now exist.
        assert(_eqs.size() == eqs.size() * dims._clusterMults.product());
    }

    // Reorder groups based on dependencies.
    void EqGroups::sort()
    {
        if (size() < 2)
            return;

        cout << " Sorting " << size() << " eq-group(s)...\n";

        // Want to keep original order as much as possible.
        // Only reorder if dependencies are in conflict.

        // Scan from beginning to end.
        for (size_t i = 0; i < size(); i++) {

            bool done = false;
            while (!done) {
        
                // Does eq-group[i] depend on any eq-group after it?
                auto& egi = at(i);
                for (size_t j = i+1; j < size(); j++) {
                
                    auto& egj = at(j);

                    // Must swap if dependent.
                    if (egi.isDepOn(cur_step_dep, egj)) {

                        // Error if also back-dep.
                        if (egj.isDepOn(cur_step_dep, egi)) {
                            THROW_YASK_EXCEPTION("Error: circular dependency between eq-groups " <<
                                egi.getDescription() << " and " <<
                                egj.getDescription());
                        }

                        // Swap them.
                        EqGroup temp(egi);
                        egi = egj;
                        egj = temp;

                        // Start over at index i.
                        done = false;
                        break;
                    }
                }
                done = true;
            }
        }
    }

    // Add expression 'eq' with condition 'cond' to eq-group with 'baseName'
    // unless alread added.  The corresponding index in '_indices' will be
    // incremented if a new group is created.
    // 'eq_deps': pre-computed dependencies between equations.
    // Returns whether a new group was created.
    bool EqGroups::addExprToGroup(EqualsExprPtr eq,
                                  BoolExprPtr cond,
                                  const string& baseName,
                                  bool is_scratch,
                                  EqDepMap& eq_deps)
    {
        // Equation already added?
        if (_eqs_in_groups.count(eq))
            return false;

        // Loop through existing groups, looking for one that
        // 'eq' can be added to.
        EqGroup* target = 0;
        for (auto& eg : *this) {

            // Must match name and condition.
            if (eg.baseName == baseName &&
                areExprsSame(eg.cond, cond)) {

                // Look for any dependencies that would prevent adding
                // 'eq' to 'eg'.
                bool is_dep = false;
                for (auto& eq2 : eg.getEqs()) {

                    for (DepType dt = DepType(0); dt < num_deps; dt = DepType(dt+1)) {
                        if (eq_deps[dt].is_dep(eq, eq2)) {
#if DEBUG_ADD_EXPRS
                            cout << "addExprsFromGrid: not adding equation " <<
                                eq->makeQuotedStr() << " to " << eg.getDescription() <<
                                " because of dependency w/equation " <<
                                eq2->makeQuotedStr() << endl;
#endif
                            is_dep = true;
                            break;
                        }
                    }
                    if (is_dep)
                        break;
                }

                // Remember target group if found and no deps.
                if (!is_dep) {
                    target = &eg;
                    break;
                }
            }
        }
        
        // Make new group if no target group found.
        bool newGroup = false;
        if (!target) {
            EqGroup ne(*_dims, is_scratch);
            push_back(ne);
            target = &back();
            target->baseName = baseName;
            target->index = _indices[baseName]++;
            target->cond = cond;
            newGroup = true;
        
#if DEBUG_ADD_EXPRS
            cout << "Creating new " << target->getDescription() << endl;
#endif
        }

        // Add eq to target eq-group.
        assert(target);
#if DEBUG_ADD_EXPRS
        cout << "Adding " << eq->makeQuotedStr() <<
            " to " << target->getDescription() << endl;
#endif
        target->addEq(eq);
    
        // Remember eq and updated grid.
        _eqs_in_groups.insert(eq);
        _outGrids.insert(eq->getGrid());

        return newGroup;
    }

    // Divide all equations into eqGroups.
    // Only process updates to grids in 'gridRegex'.
    // 'targets': string provided by user to specify grouping.
    // 'eq_deps': pre-computed dependencies between equations.
    void EqGroups::makeEqGroups(Eqs& allEqs,
                                const string& gridRegex,
                                const string& targets,
                                EqDepMap& eq_deps,
                                ostream& os)
    {
        os << "Partitioning " << allEqs.getNumEqs() << " equation(s) into groups...\n";
        //auto& stepDim = _dims->_stepDim;

        // Add each scratch equation to a separate group.
        // TODO: Allow multiple scratch eqs in a group with same conds & halos.
        // TODO: Only add scratch eqs that are needed by grids in 'gridRegex'.
        for (auto eq : allEqs.getEqs()) {

            // Get updated grid.
            auto gp = eq->getGrid();
            assert(gp);
            if (gp->isScratch()) {
                string gname = gp->getName();

                // Add equation.
                addExprToGroup(eq, allEqs.getCond(eq), gname, true, eq_deps);
            }
        }
        
        // Make a regex for the allowed grids.
        regex gridx(gridRegex);
    
        // Handle each key-value pair in 'targets' string.
        // Key is eq-group name (with possible format strings); value is regex pattern.
        ArgParser ap;
        ap.parseKeyValuePairs
            (targets, [&](const string& egfmt, const string& pattern) {

                // Make a regex for the pattern.
                regex patx(pattern);

                // Search allEqs for matches to current value.
                for (auto eq : allEqs.getEqs()) {

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
                    addExprToGroup(eq, allEqs.getCond(eq), egname, false, eq_deps);
                }
            });

        // Add all remaining equations.
        for (auto eq : allEqs.getEqs()) {

            // Get name of updated grid.
            auto gp = eq->getGrid();
            assert(gp);
            string gname = gp->getName();

            // Match to gridx?
            if (!regex_search(gname, gridx))
                continue;

            // Add equation.
            addExprToGroup(eq, allEqs.getCond(eq), _basename_default, false, eq_deps);
        }
        os << "Created " << size() << " equation group(s):\n";

        // Find dependencies between eq-groups based on deps between their eqs.
        for (auto& eg1 : *this) {
            os << " " << eg1.getDescription() << ":\n"
                "  Contains " << eg1.getNumEqs() << " equation(s).\n"
                "  Updates the following grid(s):";
            for (auto* g : eg1.getOutputGrids())
                os << " " << g->getName();
            os << ".\n";

            // Check to see if eg1 depends on other eq-groups.
            for (auto& eg2 : *this) {

                // Don't check against self.
                if (eg1.getName() == eg2.getName())
                    continue;

                eg1.checkDeps(allEqs, eq_deps, eg2);
                DepType dt = cur_step_dep;
                if (eg1.isImmDepOn(dt, eg2))
                    os << "  Immediately dependent on group " <<
                        eg2.getName() << ".\n";
                else if (eg1.isDepOn(dt, eg2))
                    os << "  Indirectly dependent on group " <<
                        eg2.getName() << ".\n";
            }
            auto& sdeps = eg1.getScratchDeps();
            if (sdeps.size()) {
                os << "  Requires evaluation of the following scratch-grid group(s):";
                for (auto& sname : sdeps)
                    os << " " << sname;
                os << ".\n";
            }
        }

        // Resort them based on dependencies.
        sort();
    }

    // Print stats from eqGroups.
    void EqGroups::printStats(ostream& os, const string& msg) {
        CounterVisitor cv;
        for (auto& eq : *this) {
            CounterVisitor ecv;
            eq.visitEqs(&ecv);
            cv += ecv;
        }
        cv.printStats(os, msg);
    }

    // Apply optimizations according to the 'settings'.
    void EqGroups::optimizeEqGroups(CompilerSettings& settings,
                                    const string& descr,
                                    bool printSets,
                                    ostream& os) {
        // print stats.
        string edescr = "for " + descr + " equation-group(s)";
        printStats(os, edescr);
    
        // Make a list of optimizations to apply to eqGroups.
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
                descr + " equation-group(s)";

            // Get new stats.
            if (numChanges)
                printStats(os, odescr);
            else
                os << "No changes " << odescr << '.' << endl;
        }

        // Final stats per equation group.
        if (printSets && size() > 1) {
            os << "Stats per equation-group:\n";
            for (auto eg : *this)
                eg.printStats(os, "for " + eg.getDescription());
        }
    }

} // namespace yask.
