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

///////// Stencil AST Expressions. ////////////

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Parse.hpp"

namespace yask {

    // Compare 2 expr pointers and return whether the expressions are
    // equivalent.
    bool areExprsSame(const Expr* e1, const Expr* e2) {

        // Handle null pointers.
        if (e1 == NULL && e2 == NULL)
            return true;
        if (e1 == NULL || e2 == NULL)
            return false;

        // Neither are null, so compare contents.
        return e1->isSame(e2);
    }

    // Expr functions.
    NumExprPtr constNum(double rhs) {
        return make_shared<ConstExpr>(rhs);
    }
    NumExprPtr first_index(const NumExprPtr dim) {
        return make_shared<IndexExpr>(dim, FIRST_INDEX);
    }
    NumExprPtr last_index(const NumExprPtr dim) {
        return make_shared<IndexExpr>(dim, LAST_INDEX);
    }

    // Unary.
    NumExprPtr operator-(const NumExprPtr rhs) {
        return make_shared<NegExpr>(rhs);
    }
    BoolExprPtr operator!(const BoolExprPtr rhs) {
        return make_shared<NotExpr>(rhs);
    }

    // Commutative.
    // If one side is nothing, return other side;
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

    BoolExprPtr operator==(const NumExprPtr lhs, const NumExprPtr rhs) {
        return make_shared<IsEqualExpr>(lhs, rhs);
    }
    BoolExprPtr operator!=(const NumExprPtr lhs, const NumExprPtr rhs) {
        return make_shared<NotEqualExpr>(lhs, rhs);
    }
    BoolExprPtr operator<(const NumExprPtr lhs, const NumExprPtr rhs) {
        return make_shared<IsLessExpr>(lhs, rhs);
    }
    BoolExprPtr operator>(const NumExprPtr lhs, const NumExprPtr rhs) {
        return make_shared<IsGreaterExpr>(lhs, rhs);
    }
    BoolExprPtr operator<=(const NumExprPtr lhs, const NumExprPtr rhs) {
        return make_shared<NotGreaterExpr>(lhs, rhs);
    }
    BoolExprPtr operator>=(const NumExprPtr lhs, const NumExprPtr rhs) {
        return make_shared<NotLessExpr>(lhs, rhs);
    }
    BoolExprPtr operator&&(const BoolExprPtr lhs, const BoolExprPtr rhs) {
        return make_shared<AndExpr>(lhs, rhs);
    }
    BoolExprPtr operator||(const BoolExprPtr lhs, const BoolExprPtr rhs) {
        return make_shared<OrExpr>(lhs, rhs);
    }

    // Define a conditional.
    IfExprPtr operator IF_OPER(EqualsExprPtr expr, const BoolExprPtr cond) {

        // Get grid referenced by the expr.
        auto gpp = expr->getLhs();
        assert(gpp);
        Grid* gp = gpp->getGrid();
        assert(gp);
        auto* eqs = gp->getEqs();
        assert(eqs);
    
        // Make if-expression node.
        auto ifp = make_shared<IfExpr>(expr, cond);

        // Save expr and if-cond.
        eqs->addCondEq(expr, cond);

        return ifp;
    }

    // Define the value of a grid point.
    EqualsExprPtr operator EQUALS_OPER(GridPointPtr gpp, const NumExprPtr rhs) {

        // Get grid referenced by the expr.
        assert(gpp);
        auto* gp = gpp->getGrid();
        assert(gp);
        auto* eqs = gp->getEqs();
        assert(eqs);
    
        // Make sure this is a grid.
        if (gp->isParam()) {
            cerr << "Error: parameter '" << gpp->getName() <<
                "' cannot appear on LHS of a grid-value equation." << endl;
            exit(1);
        }
    
        // Make expression node.
        auto expr = make_shared<EqualsExpr>(gpp, rhs);

        // Save the expression.
        eqs->addEq(expr);

        return expr;
    }
    EqualsExprPtr operator EQUALS_OPER(GridPointPtr gpp, double rhs) {
        return gpp EQUALS_OPER constNum(rhs);
    }

    // Visitor acceptors.
    void ConstExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void CodeExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void UnaryNumExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void UnaryBoolExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void UnaryNum2BoolExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void BinaryNumExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void BinaryBoolExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void BinaryNum2BoolExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void CommutativeExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void GridPoint::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void IntTupleExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void EqualsExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void IfExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void IndexExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }

    // Index methods.
    IndexExpr::IndexExpr(NumExprPtr dim, IndexType type) :
        _type(type) {
        auto dp = castExpr<IntTupleExpr>(dim, "dimension");
        if (dp->size() != 1) {
            cerr << "Error: '" << dp->makeStr() <<
                "'argument to '" << getFnName() <<
                "' is not a dimension" << endl;
            exit(1);
        }
        _dirName = *dp->getDirName();
    }

    // EqualsExpr methods.
    bool EqualsExpr::isSame(const Expr* other) const {
        auto p = dynamic_cast<const EqualsExpr*>(other);
        return p && _opStr == p->_opStr &&
            _lhs->isSame(p->_lhs.get()) &&
            _rhs->isSame(p->_rhs.get());
    }

    // IfExpr methods.
    bool IfExpr::isSame(const Expr* other) const {
        auto p = dynamic_cast<const IfExpr*>(other);
        return p &&
            areExprsSame(_expr, p->_expr) &&
            areExprsSame(_rhs, p->_rhs);
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

    // GridPoint methods.
    const string& GridPoint::getName() const {
        return _grid->getName();
    }
    bool GridPoint::isParam() const {
        return _grid->isParam();
    }
    bool GridPoint::operator==(const GridPoint& rhs) const {
        return (_grid == rhs._grid) &&
            IntTuple::operator==(rhs);
    }
    bool GridPoint::operator<(const GridPoint& rhs) const {
        return (_grid < rhs._grid) ? true :
            (_grid > rhs._grid) ? false :
            IntTuple::operator<(rhs);
    }
    bool GridPoint::isAheadOfInDir(const GridPoint& rhs, const IntTuple& dir) const {
        return _grid == rhs._grid && // must be same var.
            IntTuple::isAheadOfInDir(rhs, dir);
    }
    string GridPoint::makeStr() const {
        string str = _grid->getName() + "(";
        str += isParam() ? makeValStr() : makeDimValOffsetStr();
        str += ")";
        return str;
    }

    // Make a readable string from an expression.
    string Expr::makeStr() const {
        ostringstream oss;
    
        // Use a print visitor to make a string.
        PrintHelper ph(NULL, "temp", "", "", "");
        PrintVisitorTopDown pv(oss, ph);
        accept(&pv);

        // Return anything written to the stream
        // concatenated with anything left in the
        // PrintVisitor.
        return oss.str() + pv.getExprStr();
    }

    // Return number of nodes.
    int Expr::getNumNodes() const {

        // Use a counter visitor.
        CounterVisitor cv;
        accept(&cv);

        return cv.getNumNodes();
    }

    // Const version of accept.
    void Expr::accept(ExprVisitor* ev) const {
        const_cast<Expr*>(this)->accept(ev);
    }

    // Create an expression to a specific point in this grid.
    // Note that this doesn't actually 'read' or 'write' a value;
    // it's just a node in an expression.
    GridPointPtr Grid::makePoint(int count, ...) {

        // check for correct number of indices.
        if (count != size()) {
            cerr << "Error: attempt to access " << size() <<
                "-D grid '" << _name << "' with " << count << " indices.\n";
            exit(1);
        }

        // Copy the names from the grid to a new tuple.
        IntTuple pt = *this;

        // set the values in the tuple using the var args.
        va_list args;
        va_start(args, count);
        pt.setVals(count, args);
        va_end(args);

        // Create a point from the tuple.
        GridPointPtr gpp = make_shared<GridPoint>(this, pt);
        return gpp;
    }

    // Update halos based on each value in 'vals' given the step-dim 'stepDim'.
    void Grid::updateHalo(const string& stepDim, const IntTuple& vals)
    {
        // set and/or check step dim.
        if (_stepDim.length() == 0)
            _stepDim = stepDim;
        else assert(_stepDim == stepDim);
    
        // Find step value or use 0 if none.
        int stepVal = 0;
        auto* p = vals.lookup(stepDim);
        if (p)
            stepVal = *p;
        auto& halos = _halos[stepVal];

        // Update halo vals.
        for (auto* dim : vals.getDims()) {
            if (*dim == stepDim)
                continue;

            auto* p = halos.lookup(*dim);
            int val = abs(vals.getVal(*dim));
            if (!p)
                halos.addDimBack(*dim, val);
            else if (*p < val)
                *p = val;
            // else, current value is larger than val, so don't update.
        }
    }

    // Determine how many values in step-dim are needed.
    int Grid::getStepDimSize() const
    {
        // Only need one value if no step-dim index used.
        if (_stepDim.length() == 0)
            return 1;

        // Nothing stored?
        if (_halos.size() == 0)
            return 1;

        // Find halos at min and max step-dim points.
        // These should correspond to the read and write points.
        auto first_i = _halos.cbegin(); // begin == first.
        auto last_i = _halos.crbegin(); // reverse-begin == last.
        int first_ofs = first_i->first;
        int last_ofs = last_i->first;
        auto& first_halo = first_i->second;
        auto& last_halo = last_i->second;

        // Default step-dim size is range of offsets.
        assert(last_ofs >= first_ofs);
        int sz = last_ofs - first_ofs + 1;
    
        // If first and last halos are zero, we can further optimize storage by
        // immediately reusing memory location.
        if (sz > 1 &&
            first_halo.max() == 0 &&
            last_halo.max() == 0)
            sz--;

        // TODO: recognize that reading in one equation and then writing in
        // another can also reuse storage.

        return sz;
    }

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

        // Add a point to _all_pts and get a pointer to it.
        // If matching point exists, get a pointer to existing one.
        const GridPoint* _add_pt(Grid* g, IntTuple& offsets) {
            GridPoint gp(g, offsets);
            auto i = _all_pts.insert(gp);
            auto& gp2 = *i.first;
            return &gp2;
        }

        // Add all points with _pts offsets from pt0 to 'pt_map'.
        // Add grid from pt0 to 'grid_map'.
        void _add_pts(GridPoint* pt0, GridMap& grid_map, PointMap& pt_map) {

            // Add grid.
            auto* g = pt0->getGrid();
            grid_map[_eq].insert(g);

            // Visit each point in pts.
            if (_pts) {
                _pts->visitAllPoints([&](IntTuple& pt){

                        // Add offset to pt0.
                        auto pt1 = pt0->addElements(pt, false);
                        auto* p = _add_pt(g, pt1);
                        pt_map[_eq].insert(p);
                    });
            }

            // Visit one point.
            else {
                pt_map[_eq].insert(pt0);
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
    // TODO: replace dependency algorithms with integration of a polyhedral
    // library.
    void Eqs::findDeps(IntTuple& pts,
                       const string& stepDim,
                       EqDepMap* eq_deps) {

        // Gather points from all eqs in all grids.
        PointVisitor pt_vis(pts);
        cout << " Scanning " << getEqs().size() << " equations(s)...\n";

        // Gather initial stats from all eqs.
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
            int si1 = 0;        // step index for LHS of eq1.
            for (auto i1 : op1) {
            
                // LHS of an equation must use step index.
                auto* si1p = i1->lookup(stepDim);
                if (!si1p) {
                    cerr << "Error: equation " << eq1->makeQuotedStr() <<
                        " does not reference step-dimension index '" << stepDim <<
                        "' on LHS.\n";
                    exit(1);
                }
                assert(si1p);
                si1 = *si1p;
            }

            // Scan input (RHS) points.
            for (auto i1 : ip1) {

                // Check RHS of an equation that uses step index.
                auto* rsi1p = i1->lookup(stepDim);
                if (rsi1p) {
                    int rsi1 = *rsi1p;

                    // Cannot depend on future value in this dim.
                    if (rsi1 > si1) {
                        cerr << "Error: equation " << eq1->makeQuotedStr() <<
                            " contains an illegal dependence from offset " << rsi1 <<
                            " to " << si1 << " relative to step-dimension index '" <<
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
                        auto* si2p = i2->lookup(stepDim);
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
            cout << "  Resolving indirect dependencies...\n";
            for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1))
                (*eq_deps)[dt].analyze();
        }
        cout << " Done.\n";
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

    // Make a description.
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
                g2->updateHalo(stepDim, *op);
            }
    
            // Input points.
            for (auto* ip : inPts) {
                auto* g = ip->getGrid();
                auto* g2 = const_cast<Grid*>(g); // need to update grid.
                g2->updateHalo(stepDim, *ip);
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

            IntTuple new_loc = gp->addElements(_ofs, false);
            gp->setVals(new_loc, false);
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

                    // Create a set of offsets with all dims.
                    // This will be a union of clusterOffset and allDims.
                    //IntTuple offsets = clusterOffset.makeUnionWith(dims._allDims);

                    // Loop thru eqs.
                    for (auto eq : eqs) {
                        assert(eq.get());
            
                        // Make a copy.
                        auto eq2 = eq->cloneEquals();

                        // Add offsets to each grid point.
                        OffsetVisitor ov(clusterOffset);
                        eq2->accept(&ov);

                        // Put new equation into group.
                        addEq(eq2, false);
                    }
                }
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
                                EqDepMap& eq_deps)
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
            cout << " Checking dependencies of " <<
                eg1.getDescription() << "...\n";
            cout << "  Updating the following grid(s) with " <<
                eg1.getNumEqs() << " equation(s):";
            for (auto* g : eg1.getOutputGrids())
                cout << " " << g->getName();
            cout << endl;

            // Check to see if eg1 depends on other eq-groups.
            for (auto& eg2 : *this) {

                // Don't check against self.
                if (eg1.getName() == eg2.getName())
                    continue;

                if (eg1.setDepOn(certain_dep, eq_deps, eg2))
                    cout << "  Is dependent on " << eg2.getDescription(false) << endl;
                else if (eg1.setDepOn(possible_dep, eq_deps, eg2))
                    cout << "  May be dependent on " << eg2.getDescription(false) << endl;
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

    // Find the dimensions to be used.
    void Dimensions::setDims(Grids& grids,
                             string stepDim,
                             IntTuple& foldOptions,
                             IntTuple& clusterOptions,
                             bool allowUnalignedLoads,
                             ostream& os)
    {
        _stepDim = stepDim;
                     
        // Create a tuple of all dimensions in all grids.
        // Also keep count of how many grids have each dim.
        // Note that dimensions won't be in any particular order!
        for (auto gp : grids) {

            // Count dimensions from this grid.
            for (auto dim : gp->getDims()) {
                if (_dimCounts.lookup(dim))
                    _dimCounts.setVal(dim, _dimCounts.getVal(dim) + 1);
                else {
                    _dimCounts.addDimBack(dim, 1);
                    _allDims.addDimBack(dim, 0);
                }
            }
        }
    
        // For now, there are only global specifications for vector and cluster
        // sizes. Also, vector folding and clustering is done identially for
        // every grid access. Thus, sizes > 1 must exist in all grids.  So, init
        // vector and cluster sizes based on dimensions that appear in ALL
        // grids. 
        // TODO: relax this restriction.
        for (auto* dim : _dimCounts.getDims()) {

            // Step dim cannot be folded.
            if (*dim == stepDim) {
                continue;
            }
        
            // Add this dimension to scalar/fold/cluster only if it was found in all grids.
            if (_dimCounts.getVal(dim) == (int)grids.size()) {
                _scalar.addDimBack(dim, 1);
                _fold.addDimBack(dim, 1);
                _clusterMults.addDimBack(dim, 1);
            }
            else {
                _miscDims.addDimBack(dim, 1);
            }
        }
        os << "Step dimension: " << stepDim << endl;
    
        // Create final fold lengths based on cmd-line options.
        IntTuple foldGT1;    // fold dimensions > 1.
        for (auto* dim : foldOptions.getDims()) {
            int sz = foldOptions.getVal(dim);
            int* p = _fold.lookup(dim);
            if (!p) {
                os << "Error: fold-length of " << sz << " in '" << dim <<
                    "' dimension not allowed because '" << dim << "' ";
                if (*dim == stepDim)
                    os << "is the step dimension." << endl;
                else
                    os << "doesn't exist in all grids." << endl;
                exit(1);
            }

            // Set size.
            *p = sz;
            if (sz > 1)
                foldGT1.addDimBack(dim, sz);
            
        }
        os << "Vector-fold dimension(s) and size(s): " <<
            _fold.makeDimValStr(" * ") << endl;


        // Checks for unaligned loads.
        if (allowUnalignedLoads) {
            if (foldGT1.size() > 1) {
                os << "Error: attempt to allow unaligned loads when there are " <<
                    foldGT1.size() << " dimensions in the vector-fold that are > 1." << endl;
                exit(1);
            }
            else if (foldGT1.size() > 0)
                os << "Notice: memory map MUST be with unit-stride in " <<
                    foldGT1.makeDimStr() << " dimension!" << endl;
        }

        // Create final cluster lengths based on cmd-line options.
        for (auto* dim : clusterOptions.getDims()) {
            int mult = clusterOptions.getVal(dim);
            int* p = _clusterMults.lookup(dim);
            if (!p) {
                os << "Error: cluster-multiplier of " << mult << " in '" << dim <<
                    "' dimension not allowed because '" << dim << "' ";
                if (*dim == stepDim)
                    os << "is the step dimension." << endl;
                else
                    os << "doesn't exist in all grids." << endl;
                exit(1);
            }

            // Set the size.
            *p = mult;
        }
        _clusterPts = _fold.multElements(_clusterMults);
    
        os << "Cluster dimension(s) and multiplier(s): " <<
            _clusterMults.makeDimValStr(" * ") << endl;
        os << "Cluster dimension(s) and size(s) in points: " <<
            _clusterPts.makeDimValStr(" * ") << endl;
        os << "Other dimension(s): " << _miscDims.makeDimStr(", ") << endl;
    }
} // namespace yask.
