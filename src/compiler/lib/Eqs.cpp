/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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
        typedef unordered_map<EqualsExpr*, GridSet> GridMap;

        GridMap _lhs_grids; // outputs of eqs.
        GridMap _rhs_grids; // inputs of eqs.
    
        // A type to hold a mapping of equations to a set of points in each.
        typedef unordered_set<const GridPoint*> PointSet;
        typedef unordered_map<EqualsExpr*, PointSet> PointMap;

        PointMap _lhs_pts; // outputs of eqs.
        PointMap _rhs_pts; // inputs of eqs.

        IntTuple* _pts=0;    // Points to visit from each index.
        EqualsExpr* _eq=0;   // Current equation.

        // Add a grid point to _all_pts and get a pointer to it.
        // If matching point exists, just get a pointer to existing one.
        const GridPoint* _add_pt(const GridPoint& gpt0, const IntTuple& offsets) {
            auto gpt1 = gpt0;
            gpt1.setArgOffsets(offsets);
            auto i = _all_pts.insert(gpt1);
            auto& gpt2 = *i.first;
            return &gpt2;
        }

        // Add all points with _pts offsets from gpt0 to 'pt_map'.
        // Add grid from gpt0 to 'grid_map'.
        // Maps keyed by _eq.
        void _add_pts(GridPoint* gpt0, GridMap& grid_map, PointMap& pt_map) {

            // Add grid.
            auto* g = gpt0->getGrid();
            grid_map[_eq].insert(g);

            // Visit each point in _pts.
            if (_pts) {
                _pts->visitAllPoints([&](const IntTuple& ptofs){

                        // Add offset to gpt0.
                        auto& pt0 = gpt0->getArgOffsets();
                        auto pt1 = pt0.addElements(ptofs, false);
                        auto* p = _add_pt(*gpt0, pt1);
                        pt_map[_eq].insert(p);

                        return true;
                    });
            }

            // Visit one point.
            else {
                pt_map[_eq].insert(gpt0);
            }
        }   
    
    public:
    
        // Ctor.
        // 'pts' contains offsets from each point to create.
        PointVisitor() :
            _pts(0) {}
        PointVisitor(IntTuple& pts) :
            _pts(&pts) {}
        virtual ~PointVisitor() {}

        GridMap& getOutputGrids() { return _lhs_grids; }
        GridMap& getInputGrids() { return _rhs_grids; }
        PointMap& getOutputPts() { return _lhs_pts; }
        PointMap& getInputPts() { return _rhs_pts; }
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
        virtual void visit(EqualsExpr* ee) {

            // Remember this equation.
            _eq = ee;

            // Make sure map entries exist.
            _lhs_grids[_eq];
            _rhs_grids[_eq];
            _lhs_pts[_eq];
            _rhs_pts[_eq];

            // Store all LHS points.
            auto* lhs = ee->getLhs().get();
            _add_pts(lhs, _lhs_grids, _lhs_pts);

            // visit RHS.
            NumExprPtr rhs = ee->getRhs();
            rhs->accept(this);

            // Don't visit LHS because we've already saved it.
        }

        // Callback at a grid point on the RHS.
        virtual void visit(GridPoint* gp) {
            assert(_eq);

            // Store all RHS points.
            _add_pts(gp, _rhs_grids, _rhs_pts);
        }
    };

    // Recursive search starting at 'a'.
    // Fill in _full_deps.
    bool EqDeps::_analyze(EqualsExprPtr a, SeenSet* seen)
    {
        // 'a' already visited?
        bool was_seen = (seen && seen->count(a));
        if (was_seen)
            return true;
    
        // any dependencies?
        if (_deps.count(a)) {
            auto& adeps = _deps.at(a);
    
            // make new seen-set adding 'a'.
            SeenSet seen1;
            if (seen)
                seen1 = *seen; // copy nodes already seen.
            seen1.insert(a);   // add this one.
        
            // loop thru dependences 'a' -> 'b'.
            for (auto b : adeps) {

                // whole path up to and including 'a' depends on 'b'.
                for (auto p : seen1)
                    _full_deps[p].insert(b);
            
                // follow path.
                _analyze(b, &seen1);
            }
        }

        // no dependence; make an empty entry.
        return false;
    }

    // Find dependencies based on all eqs.
    // If 'eq_deps' is set, save dependencies between eqs.
    // [BIG]TODO: replace dependency algorithms with integration of a polyhedral
    // library.
    void Eqs::findDeps(IntTuple& pts,
                       const string& stepDim,
                       EqDepMap* eq_deps,
                       ostream& os) {

        // Gather points from all eqs in all grids.
        PointVisitor pt_vis(pts);

        // Gather initial stats from all eqs.
        os << " Scanning " << getEqs().size() << " equations(s)...\n";
        for (auto eq1 : getEqs())
            eq1->accept(&pt_vis);
        auto& outGrids = pt_vis.getOutputGrids();
        auto& inGrids = pt_vis.getInputGrids();
        auto& outPts = pt_vis.getOutputPts();
        auto& inPts = pt_vis.getInputPts();
        
        // Check dependencies on all eqs.
        for (auto eq1 : getEqs()) {
            auto* eq1p = eq1.get();
            assert(outGrids.count(eq1p));
            assert(inGrids.count(eq1p));
            auto& og1 = outGrids.at(eq1p);
            //auto& ig1 = inGrids.at(eq1p);
            auto& op1 = outPts.at(eq1p);
            auto& ip1 = inPts.at(eq1p);
            auto cond1 = getCond(eq1p);

            // An equation must update one grid only.
            assert(og1.size() == 1);
            auto* g1 = eq1->getGrid();
            assert(og1.count(g1));

            // Scan output (LHS) points.
            int si1 = 0;        // step index offset for LHS of eq1.
            for (auto i1 : op1) {
            
                // LHS of an equation must use step dim w/a simple offset.
                auto* si1p = i1->getArgOffsets().lookup(stepDim);
                if (!si1p) {
                    cerr << "Error: equation " << eq1->makeQuotedStr() <<
                        " does not use simple offset from step-dimension index var '" <<
                        stepDim << "' on LHS.\n";
                    exit(1);
                }
                assert(si1p);
                si1 = *si1p;

                // LHS of an equation must be vectorizable.
                // TODO: relax this restriction.
                if (i1->getVecType() != GridPoint::VEC_FULL) {
                    cerr << "Error: equation " << eq1->makeQuotedStr() <<
                        " is not fully vectorizable on LHS because not all folded"
                        " dimensions are accessed via simple offsets from their respective indices.\n";
                    exit(1);
                }
            }

            // Scan input (RHS) points.
            for (auto i1 : ip1) {

                // Check RHS of an equation that uses step index.
                auto* rsi1p = i1->getArgOffsets().lookup(stepDim);
                if (rsi1p) {
                    int rsi1 = *rsi1p;

                    // Cannot depend on future value in this dim.
                    if (rsi1 > si1) {
                        cerr << "Error: equation " << eq1->makeQuotedStr() <<
                            " contains an illegal dependence from offset " << rsi1 <<
                            " to " << si1 << " relative to step-dimension index var '" <<
                            stepDim << "'.\n";
                        exit(1);
                    }

                    // TODO: should make some dependency checks when rsi1 == si1.
                }
            }

            // TODO: check to make sure cond1 doesn't depend on stepDim.
            
#ifdef DEBUG_DEP
            cout << " Checking dependencies *within* equation " <<
                eq1->makeQuotedStr() << "...\n";
#endif

            // Find other eqs that depend on eq1.
            for (auto eq2 : getEqs()) {
                auto* eq2p = eq2.get();
                //auto& og2 = outGrids.at(eq2p);
                auto& ig2 = inGrids.at(eq2p);
                auto& op2 = outPts.at(eq2p);
                auto& ip2 = inPts.at(eq2p);
                auto cond2 = getCond(eq2p);

                bool same_eq = eq1 == eq2;
                bool same_cond = areExprsSame(cond1, cond2);

                // If two different eqs have the same condition, they
                // cannot update the exact same point.
                if (!same_eq && same_cond &&
                    pt_vis.do_sets_intersect(op1, op2)) {
                    cerr << "Error: two equations with condition " <<
                        cond1->makeQuotedStr() << " update the same point: " <<
                        eq1->makeQuotedStr() << " and " <<
                        eq2->makeQuotedStr() << endl;
                    exit(1);
                }

                // eq2 dep on eq1 => some output of eq1 is an input to eq2.
                // If the two eqs have the same condition, detect certain
                // dependencies by looking for exact matches.
                if (same_cond &&
                    pt_vis.do_sets_intersect(op1, ip2)) {

                    // Eq depends on itself?
                    if (same_eq) {
                                    
                        // Exit with error.
                        cerr << "Error: illegal dependency between LHS and RHS of equation " <<
                            eq1->makeQuotedStr() <<
                            " within offsets in range " << pts.makeDimValStr(" * ") << ".\n";
                        exit(1);
                    }

                    // Save dependency.
                    // Flag as both certain and possible because we need to follow
                    // certain ones when resolving indirect possible ones.
                    if (eq_deps) {
                        (*eq_deps)[certain_dep].set_dep_on(eq2, eq1);
                        (*eq_deps)[possible_dep].set_dep_on(eq2, eq1);
                    }
                        
                    // Move along to next eq2.
                    continue;
                }

                // Check more only if saving dependencies.
                if (!eq_deps)
                    continue;

                // Only check between different equations.
                if (same_eq)
                    continue;

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
                // TODO: be much smarter about this and find only real
                // dependencies--use a polyhedral library?
                if (pt_vis.do_sets_intersect(og1, ig2)) {

                    // detailed check of g1 input points from eq2.
                    for (auto* i2 : ip2) {
                        if (i2->getGrid() != g1) continue;

                        // From same step index, e.g., same time?
                        auto* si2p = i2->getArgOffsets().lookup(stepDim);
                        if (si2p && (*si2p == si1)) {

                            // Save dependency.
                            if (eq_deps)
                                (*eq_deps)[possible_dep].set_dep_on(eq2, eq1);
                                
                            // Move along to next equation.
                            break;
                        }
                    }
                }
            }
        }

        // Resolve indirect dependencies.
        if (eq_deps) {
            os << "  Resolving indirect dependencies...\n";
            for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1))
                (*eq_deps)[dt].analyze();
        }
        os << " Done.\n";
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

            // Amount of vectorization allowed primarily depends on number
            // of folded dimensions in the grid accessed at this point.
            auto* grid = gp->getGrid();
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
#if 0
            cout << "GridPoint " << gp->makeStr() <<
                " has loop-access type " << lt << endl;
#endif
        }
    };
    
    // Determine loop access behavior of grid points.
    void Eqs::analyzeLoop(const Dimensions& dims) {

        // Send a 'SetLoopVisitor' to each point in
        // the current equations.
        SetLoopVisitor slv(dims);
        visitEqs(&slv);
    }

    // Get the full name of an eq-group.
    // Must be unique.
    string EqGroup::getName() const {

#if 0
        // Just use base name if zero index.
        if (!index)
            return baseName;
#endif

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

    // Add an equation to an EqGroup
    // If 'update_stats', update grid and halo data.
    void EqGroup::addEq(EqualsExprPtr ee, bool update_stats)
    {
#ifdef DEBUG_EQ_GROUP
        cout << "EqGroup: adding " << ee->makeQuotedStr() << endl;
#endif
        _eqs.insert(ee);

        if (update_stats) {
    
            // Get I/O point data.
            PointVisitor pv;
            ee->accept(&pv);

            // update list of input and output grids.
            auto& outGrids = pv.getOutputGrids().at(ee.get());
            for (auto* g : outGrids)
                _outGrids.insert(g);
            auto& inGrids = pv.getInputGrids().at(ee.get());
            for (auto* g : inGrids)
                _inGrids.insert(g);

            // update halo info in grids.
            auto& outPts = pv.getOutputPts().at(ee.get());
            auto& inPts = pv.getInputPts().at(ee.get());
            auto& stepDim = _dims->_stepDim;

            // Output points.
            for (auto* op : outPts) {
                auto* g = op->getGrid();
                auto* g2 = const_cast<Grid*>(g); // need to update grid.
                g2->updateHalo(op->getArgOffsets());
                g2->updateConstIndices(op->getArgConsts());
            }
    
            // Input points.
            for (auto* ip : inPts) {
                auto* g = ip->getGrid();
                auto* g2 = const_cast<Grid*>(g); // need to update grid.
                g2->updateHalo(ip->getArgOffsets());
                g2->updateConstIndices(ip->getArgConsts());
            }
        }
    }
    
    // Set dependency on eg2 if this eq-group is dependent on it.
    // Return whether dependent.
    bool EqGroup::setDepOn(DepType dt, EqDepMap& eq_deps, const EqGroup& eg2)
    {

        // Eqs in this.
        for (auto& eq1 : getEqs()) {

            // Eqs in eg2.
            for (auto& eq2 : eg2.getEqs()) {

                if (eq_deps[dt].is_dep_on(eq1, eq2)) {

                    _dep_on[dt].insert(eg2.getName());
                    return true;
                }
            }
        }
        return false;
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
        dims._clusterMults.visitAllPoints([&](const IntTuple& clusterIndex) {

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
                        // Don't need to update grids because eq isn't new.
                        addEq(eq2, false);
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
                    bool do_swap = false;

                    // Must swap on certain deps.
                    if (egi.isDepOn(certain_dep, egj)) {
                        if (egj.isDepOn(certain_dep, egi)) {
                            cerr << "Error: circular dependency between eq-groups " <<
                                egi.getDescription() << " and " <<
                                egj.getDescription() << endl;
                            exit(1);
                        }
                        do_swap = true;
                    }

                    // Swap on possible deps if one-way.
                    else if (egi.isDepOn(possible_dep, egj) && 
                             !egj.isDepOn(possible_dep, egi) &&
                             !egj.isDepOn(certain_dep, egi)) {
                        do_swap = true;
                    }

                    if (do_swap) {

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
                                  EqDepMap& eq_deps)
    {
        // Equation already added?
        if (_eqs_in_groups.count(eq))
            return false;

        // Look for existing group matching base-name and condition.
        EqGroup* target = 0;
        for (auto& eg : *this) {

            if (eg.baseName == baseName &&
                areExprsSame(eg.cond, cond)) {

                // Look for any dependencies that would prevent adding
                // eq to eg.
                bool is_dep = false;
                for (auto& eq2 : eg.getEqs()) {

                    for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1)) {
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
        
        // Make new group if needed.
        bool newGroup = false;
        if (!target) {
            EqGroup ne(*_dims);
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
    // 'targets': string provided by user to specify grouping.
    // 'eq_deps': pre-computed dependencies between equations.
    void EqGroups::makeEqGroups(Eqs& allEqs,
                                const string& targets,
                                EqDepMap& eq_deps,
                                ostream& os)
    {
        //auto& stepDim = _dims->_stepDim;
    
        // Handle each key-value pair in 'targets' string.
        ArgParser ap;
        ap.parseKeyValuePairs
            (targets, [&](const string& key, const string& value) {

                // Search allEqs for matches to current value.
                for (auto eq : allEqs.getEqs()) {

                    // Get name of updated grid.
                    auto gp = eq->getGrid();
                    assert(gp);
                    string gname = gp->getName();

                    // Does value appear in the grid name?
                    size_t np = gname.find(value);
                    if (np != string::npos) {

                        // Add equation.
                        addExprToGroup(eq, allEqs.getCond(eq), key, eq_deps);
                    }
                }
            });

        // Add all remaining equations.
        for (auto eq : allEqs.getEqs()) {

            // Add equation.
            addExprToGroup(eq, allEqs.getCond(eq), _basename_default, eq_deps);
        }

        // Find dependencies between eq-groups based on deps between their eqs.
        for (auto& eg1 : *this) {
            os << " Checking dependencies of " <<
                eg1.getDescription() << "...\n";
            os << "  Updating the following grid(s) with " <<
                eg1.getNumEqs() << " equation(s):";
            for (auto* g : eg1.getOutputGrids())
                os << " " << g->getName();
            os << endl;

            // Check to see if eg1 depends on other eq-groups.
            for (auto& eg2 : *this) {

                // Don't check against self.
                if (eg1.getName() == eg2.getName())
                    continue;

                if (eg1.setDepOn(certain_dep, eq_deps, eg2))
                    os << "  Is dependent on " << eg2.getDescription(false) << endl;
                else if (eg1.setDepOn(possible_dep, eq_deps, eg2))
                    os << "  May be dependent on " << eg2.getDescription(false) << endl;
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
