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

#include <unordered_map>

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
#define COMM_OPER(oper, exprtype)                          \
    NumExprPtr operator oper(const NumExprPtr lhs,            \
                          const NumExprPtr rhs) {          \
        if (lhs.get() == NULL)                             \
            return rhs;                                 \
        else if (rhs.get() == NULL)                     \
            return lhs;                                 \
        auto ex = make_shared<exprtype>();              \
        ex->mergeExpr(lhs);                             \
        ex->mergeExpr(rhs);                             \
        return ex;                                      \
    }                                                   \
    NumExprPtr operator oper(double lhs,                   \
                          const NumExprPtr rhs) {       \
        NumExprPtr p = make_shared<ConstExpr>(lhs);        \
        return p oper rhs;                              \
    }                                                   \
    NumExprPtr operator oper(const NumExprPtr lhs,         \
                          double rhs) {                 \
        NumExprPtr p = make_shared<ConstExpr>(rhs);        \
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
    
    // Make if-expression node.
    auto ifp = make_shared<IfExpr>(expr, cond);

    // Save expr and if-cond in grid.
    gp->addCondEq(expr, cond);

    return ifp;
}

// Define the value of a grid point.
EqualsExprPtr operator EQUIV_OPER(GridPointPtr gpp, const NumExprPtr rhs) {

    // Get grid referenced by the expr.
    assert(gpp);
    auto* gp = gpp->getGrid();
    assert(gp);
    
    // Make sure this is a grid.
    if (gp->isParam()) {
        cerr << "error: parameter '" << gpp->getName() <<
            "' cannot appear on LHS of a grid-value equation." << endl;
        exit(1);
    }
    
    // Make expression node.
    auto expr = make_shared<EqualsExpr>(gpp, rhs);

    // Save it in the grid.
    gp->addEq(expr);

    return expr;
}
EqualsExprPtr operator EQUIV_OPER(GridPointPtr gpp, double rhs) {
    return gpp EQUIV_OPER constNum(rhs);
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
        cerr << "error: '" << dp->makeStr() <<
            "'argument to '" << getFnName() <<
            "' is not a dimension" << endl;
        exit(1);
    }
    _dirName = dp->getDirName();
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
        
    // Operands must be the same, but not in same order.
    // This tracks the indices in 'other' that have already
    // been matched.
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
    return addPoint(gpp);
}

// A visitor to collect points visited in a set of eqs.
class PointVisitor : public ExprVisitor {

    typedef map<EqualsExpr*, set<GridPoint>> PointMap;
    PointMap _lhs_set; // LHSs of eqs.
    PointMap _rhs_set; // RHSs of eqs.
    IntTuple& _pts;             // Points to visit from each index.
    EqualsExpr* _eq;            // Current equation.
    int _num_eqs;
    int _num_lhs_pts;
    int _num_rhs_pts;
    
public:
    
    // Ctor.
    // Points to create at each GridPoint defined by 'pts'.
    PointVisitor(IntTuple& pts) :
        _pts(pts), _eq(0),
        _num_eqs(0), _num_lhs_pts(0), _num_rhs_pts(0) {}
    virtual ~PointVisitor() {}

    PointMap& getLhses() { return _lhs_set; }
    PointMap& getRhses() { return _rhs_set; }
    int getNumEqs() const { return _num_eqs; }
    int getNumLhsPts() const { return _num_lhs_pts; }
    int getNumRhsPts() const { return _num_rhs_pts; }
    
    // Callback at an equality.
    virtual void visit(EqualsExpr* ee) {

        // Remember this equation.
        _eq = ee;
        _num_eqs++;

        // Make sure map entries exist.
        _lhs_set[_eq];
        _rhs_set[_eq];

        // Store all LHS points.
        auto lhs = ee->getLhs();
        auto* g = lhs->getGrid();
        _pts.visitAllPoints([&](const IntTuple& pt){

                // Offset in each dim is starting point of LHS plus
                // offset of pt.
                auto offsets = lhs->addElements(pt, false);
                GridPoint lhsp(g, offsets);
                _lhs_set[_eq].insert(lhsp);
                _num_lhs_pts++;
            });

        // visit RHS.
        NumExprPtr rhs = ee->getRhs();
        rhs->accept(this);

        // Don't visit LHS because we've already saved it.
    }

    // Callback at a grid point on the RHS.
    virtual void visit(GridPoint* gp) {
        assert(_eq);

        // Store all RHS points.
        _pts.visitAllPoints([&](const IntTuple& pt){

                // Offset in each dim is starting point of gp plus
                // offset of pt.
                auto offsets = gp->addElements(pt, false);
                GridPoint rhsp(gp, offsets);
                _rhs_set[_eq].insert(rhsp);
                _num_rhs_pts++;
            });        
    }
};

// A visitor to check for dependencies.
class DepVisitor : public ExprVisitor {

    // key: LHS of an eq visited.
    // value: One of the equations that created this LHS.
    map<GridPoint, EqualsExpr*> _lhs_set;

    EqualsExpr* _eq;            // Current equation.
    IntTuple& _pts;             // Points to visit from each index.
    DepVisitor* _other;         // Another set of points to check against.
    bool _dep;                  // Dependency found?

public:

    // Ctor.
    // If 'other' is set, will check for dependencies on it.
    // Otherwise, will check for dependencies on self.
    DepVisitor(IntTuple& pts, DepVisitor* other = 0) :
        _eq(0),
        _pts(pts),
        _other(other),
        _dep(false) {}
    virtual ~DepVisitor() {}

    virtual bool is_dep() const {
        return _dep;
    }

    // Callback at an equality.
    virtual void visit(EqualsExpr* ee) {

        // Remember this equation.
        _eq = ee;

        // Store all points.
        auto lhs = ee->getLhs();
        auto* g = lhs->getGrid();
        _pts.visitAllPoints([&](const IntTuple& pt){

                // Offset in each dim is starting point of LHS plus
                // offset of pt.
                auto offsets = lhs->addElements(pt, false);

                GridPoint gp(g, offsets);
                _lhs_set[gp] = ee;
            });

        // visit RHS.
        NumExprPtr rhs = ee->getRhs();
        rhs->accept(this);
    }

    // Callback at a grid point.
    // _lhs depends on *gp.
    virtual void visit(GridPoint* gp) {
        assert(_eq);

        // What to check against?
        auto* other = this;
        if (_other)
            other = _other;

        // Check against all points.
        _pts.visitAllPoints([&](const IntTuple& pt){

                // Offset in each dim is starting point of LHS plus
                // offset of pt.
                auto offsets = gp->addElements(pt, false);
                GridPoint rhs(gp, offsets);
                
                // Compare against all visited LHS points.
                // NB: Will need to run two passes to check dependencies
                // against LHS points in equations not yet visited.
                if (other->_lhs_set.count(rhs) > 0) {
                    _dep = true;

                    // Use an equation to give a good error message.
                    // Not necessary to capture every eq associated with
                    // this LHS.
                    if (!_other) {
                        auto* eq = _lhs_set[rhs];

                        // Exit with error.
                        cerr << "Error: illegal dependency on LHS of equation " <<
                            eq->makeQuotedStr() << " found in RHS of ";
                        if (eq == _eq)
                            cerr << "same equation";
                        else
                            cerr << "equation " << _eq->makeQuotedStr();
                        if (_pts.product() > 1)
                            cerr << " within range " << _pts.makeDimValStr(" * ");
                        cerr << ".\n";
                        exit(1);
                    }
                }
            });        
    }
};


// Find dependencies based on all eqs in all grids.
// If 'eq_deps' is set, check dependencies between eqs.
void Grids::findDeps(IntTuple& pts, EqDeps* eq_deps) {

    // Gather points from all eqs in all grids.
    PointVisitor pt_vis(pts);
    cerr << " Gathering grid-points from " << size() << " grid(s)...\n";

    // All grids.
    for (auto g1 : *this) {
        if (g1->getNumEqs() == 0)
            continue;

        // All eqs in grid g1.
        for (auto eq1 : g1->getEqs()) {
            eq1->accept(&pt_vis);
        }
    }
    auto& lhses = pt_vis.getLhses();
    auto& rhses = pt_vis.getRhses();
    cerr << "  Found " << pt_vis.getNumLhsPts() << " LHS point(s) and " <<
        pt_vis.getNumRhsPts() << " RHS point(s) from " <<
        pt_vis.getNumEqs() << " equation(s).\n";
        
    // Check dependencies on all eqs in all grids.
    for (auto g1 : *this) {
        if (g1->getNumEqs() == 0)
            continue;
        
        // All eqs in grid g1.
        for (auto eq1 : g1->getEqs()) {
            assert(lhses.count(eq1.get()));
            assert(rhses.count(eq1.get()));
            auto& lhs1 = lhses.at(eq1.get());
            auto& rhs1 = rhses.at(eq1.get());

#ifdef DEBUG_DEP
            cerr << " Checking dependencies within equation " <<
                eq1->makeQuotedStr() << "...\n";
#endif

            // Find other eqs that depend on this one.
            if (eq_deps) {
                for (auto g2 : *this) {

                    // All eqs in grid g2.
                    for (auto eq2 : g2->getEqs()) {
                        auto& lhs2 = lhses.at(eq2.get());
                        auto& rhs2 = rhses.at(eq2.get());

                        // Check to see if eq2 depends on eq1.
                        for (auto lhsp1 : lhs1) {
                            if (rhs2.count(lhsp1)) {
                                (*eq_deps)[eq2].insert(eq1);

                                // Eq depends on self?
                                if (eq1 == eq2) {
                                    
                                    // Exit with error.
                                    cerr << "Error: illegal dependency between LHS and RHS of equation " <<
                                        eq1->makeQuotedStr() <<
                                        " within range " << pts.makeDimValStr(" * ") << ".\n";
                                    exit(1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Get the full name of an eq-group.
// Must be unique.
string EqGroup::getName() const {

    // Just use base name if zero index.
    if (!index)
        return baseName;

    // Otherwise, add index to base name.
    ostringstream oss;
    oss << baseName << "_" << index;
    return oss.str();
}

// Make a description.
string EqGroup::getDescription() const {
    string des = "equation-group \"" + getName() + "\"";
    if (cond.get())
        des += " when " + cond->makeQuotedStr();
    return des;
}

// Print stats from eqGroup.
void EqGroup::printStats(ostream& os, const string& msg) {
    CounterVisitor cv;
    visitEqs(&cv);
    cv.printStats(os, msg);
}

// Add expressions from a grid to group(s) named 'baseName'.
// The corresponding index in 'indices' will be incremented
// if a new group is created.
// Returns whether a new group was created.
bool EqGroups::addExprsFromGrid(const string& baseName,
                                map<string, int>& indices,
                                Grid* gp,
                                EqDeps& eq_deps)
{
    bool newGroup = false;

    // Grid already added?
    if (_eqGrids.count(gp))
        return false;

    // Grid has no exprs?
    if (gp->getNumEqs() == 0)
        return false;

    // Loop through all equations in grid.
    for (auto& eq : gp->getEqs()) {
        auto cond = gp->getCond(eq); // condition; might be NULL.

        // Look for existing group matching base-name and condition.
        EqGroup* target = 0;
        for (auto& eg : *this) {

            if (eg.baseName == baseName &&
                areExprsSame(eg.cond, cond)) {

                // Look for any dependencies that would prevent adding
                // eq to eg.
                bool is_dep = false;
                for (auto& eq2 : eg.getEqs()) {
                    if (eq_deps.is_dep(eq, eq2)) {
                        is_dep = true;
#if DEBUG_ADD_EXPRS
                        cerr << "addExprsFromGrid: not adding equation " <<
                            eq->makeQuotedStr() << " to " << eg.getDescription() <<
                            " because of dependency on equation " <<
                            eq2->makeQuotedStr() << endl;
#endif
                        break;
                    }
                }

                if (!is_dep) {
                    target = &eg;
                    break;
                }
            }
        }
        
        // Make new group if needed.
        if (!target) {
            EqGroup ne;
            push_back(ne);
            target = &back();
            target->baseName = baseName;
            target->index = indices[baseName]++;
            target->cond = cond;
            newGroup = true;

#if DEBUG_ADD_EXPRS
            cerr << "Creating new " << target->getDescription() << endl;
#endif
        }

        // Add expr to target group.
        assert(target);
#if DEBUG_ADD_EXPRS
        cerr << "Adding " << eq->makeQuotedStr() <<
            " to " << target->getDescription() << endl;
#endif
        target->addEq(eq);
    }
    
    // Remember all grids updated.
    _eqGrids.insert(gp);

    return newGroup;
}

// Separate expressions from grids into eqGroups.
// Remove points and expressions from grids.
// TODO: do this automatically based on inter-equation dependencies.
void EqGroups::findEqGroups(Grids& allGrids,
                            const string& targets,
                            IntTuple& pts,
                            EqDeps& eq_deps)
{
    // Map to track indices per eq-group name.
    map<string, int> indices;
    
    // Handle each key-value pair in 'targets' string.
    ArgParser ap;
    ap.parseKeyValuePairs
        (targets, [&](const string& key, const string& value) {

            // Search allGrids for matches to current value.
            for (auto gp : allGrids) {
                string gname = gp->getName();

                // value doesn't appear in the grid name?
                size_t np = gname.find(value);
                if (np == string::npos)
                    continue;

                // Add equations.
                addExprsFromGrid(key, indices, gp, eq_deps);
            }
        });

    // Add all grids not matching any values in the 'targets' string.
    for (auto gp : allGrids) {

        // Add equations.
        string key = gp->getName(); // group name is just grid name.
        addExprsFromGrid(key, indices, gp, eq_deps);
    }

    // Now, all equations should be added to this object.
    // Can remove them from temp storage in grids.
    allGrids.clearTemp();

    // Check for circular dependencies on eq-groups.
    for (auto& eg1 : *this) {
        cerr << " Checking dependencies on " <<
            eg1.getDescription() << "...\n";
        cerr << "  Updating the following grid(s) with " <<
            eg1.getNumEqs() << " equation(s):\n";
        for (auto* g : eg1.getOutputGrids())
            cerr << "   " << g->getName() << endl;

        // Check for conflicts in this eq-group.
        DepVisitor dep_checker1(pts);
        eg1.visitEqs(&dep_checker1);

        // Need 2nd pass to ensure all LHSs are checked against
        // all RHSs when a group has >1 equation.
        eg1.visitEqs(&dep_checker1);
        
        // Check to see if other eq-groups depend on this one.
        for (auto& eg2 : *this) {

            // Don't check against self.
            if (eg1.getName() == eg2.getName())
                continue;

            // Check to see if eg2 depends on eg1.
            DepVisitor dep_checker2(pts, &dep_checker1);
            eg2.visitEqs(&dep_checker2);
            if (dep_checker2.is_dep())
                eg2.depends_on.insert(&eg1);
        }
    }
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
    for (auto dim : _dimCounts.getDims()) {

        // Step dim cannot be folded.
        if (dim == stepDim) {
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
    for (auto dim : foldOptions.getDims()) {
        int sz = foldOptions.getVal(dim);
        int* p = _fold.lookup(dim);
        if (!p) {
            os << "Error: fold-length of " << sz << " in '" << dim <<
                "' dimension not allowed because '" << dim << "' ";
            if (dim == stepDim)
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
    for (auto dim : clusterOptions.getDims()) {
        int mult = clusterOptions.getVal(dim);
        int* p = _clusterMults.lookup(dim);
        if (!p) {
            os << "Error: cluster-multiplier of " << mult << " in '" << dim <<
                "' dimension not allowed because '" << dim << "' ";
            if (dim == stepDim)
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
