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
    
    // Make expression node.
    auto ip = make_shared<IfExpr>(expr, cond);

    // Save expr and if-statement.
    gp->addExpr(expr, ip);

    return ip;
}

// Define the value of a grid point.
// Note that the semantics are different than the 'normal'
// '==' operator, which tests for equality.
EqualsExprPtr operator==(GridPointPtr gpp, const NumExprPtr rhs) {
    assert(gpp.get() != NULL);

    // Make sure this is a grid.
    if (gpp->isParam()) {
        cerr << "error: parameter '" << gpp->getName() <<
            "' cannot appear on LHS of a grid-value eqGroup." << endl;
        exit(1);
    }
    
    // Make expression node.
    auto expr = make_shared<EqualsExpr>(gpp, rhs);

    // Save it with a default null condition.
    operator IF_OPER(expr, NULL);

    return expr;
}
EqualsExprPtr operator==(GridPointPtr gpp, double rhs) {
    return gpp == constNum(rhs);
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
    return (_grid->getName() == rhs._grid->getName()) &&
        IntTuple::operator==(rhs);
}
bool GridPoint::operator<(const GridPoint& rhs) const {
    return (_grid->getName() < rhs._grid->getName()) ? true :
        (_grid->getName() > rhs._grid->getName()) ? false :
        IntTuple::operator<(rhs);
}
bool GridPoint::isAheadOfInDir(const GridPoint& rhs, const IntTuple& dir) const {
    return _grid->getName() == rhs._grid->getName() && // must be same var.
        IntTuple::isAheadOfInDir(rhs, dir);
}
string GridPoint::makeStr() const {
    string str = _grid->getName() + "(";
    str += isParam() ? makeValStr() : makeDimValOffsetStr();
    str += ")";
    return str;
}

#if 0
// Visit all expressions in all grids.
void Grids::visitExprs(ExprVisitor* ev) {
    for (auto gp : *this) {
        gp->visitExprs(ev);
    }
}
#endif

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

// Get the full name.
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

// Print stats from eqGroup.
void EqGroup::printStats(ostream& os, const string& msg) {
    CounterVisitor cv;
    visitExprs(&cv);
    cv.printStats(os, msg);
}

// Add expressions from a grid to group(s) named groupName.
// Returns whether a new group was created.
bool EqGroups::addExprsFromGrid(const string& groupName,
                                map<string, int>& indices,
                                Grid* gp) {

    bool newGroup = false;

    // Grid already added?
    if (_eqGrids.count(gp))
        return false;

    // Grid has no exprs?
    if (gp->getExprs().size() == 0)
        return false;

    // Expressions must be added in a consistent order.
    // To do this, use a map to sort the expressions by their
    // string representations.
    map<string, IfExprPtr> sortedExprs;
    for (auto& eq : gp->getExprs()) {
        auto& ifExpr = eq.second;          // if expression.

        // Key is string, which includes if-condition if not null.
        string key = ifExpr->makeStr();

        // If condition is null, add a space to beginning to make
        // sure it sorts first.
        if (ifExpr->getCond().get() == NULL)
            key = ' ' + key;

        // Store expression as value.
        sortedExprs[key] = ifExpr;
    }

    // Loop through all equations in grid.
    for (auto& i : sortedExprs) {
        auto& ifExpr = i.second;
        auto& expr = ifExpr->getExpr(); // expression.
        auto& cond = ifExpr->getCond(); // condition; might be NULL.

        // Look for existing group matching base-name and condition.
        EqGroup* ep = 0;
        for (auto& eqg : *this) {

            if (eqg.baseName == groupName &&
                areExprsSame(eqg.cond, cond)) {
                ep = &eqg;
                break;
            }
        }
        
        // Make new group if needed.
        if (!ep) {
            EqGroup ne;
            push_back(ne);
            ep = &back();
            ep->baseName = groupName;
            ep->index = indices[groupName]++;
            ep->cond = cond;
            newGroup = true;

#if DEBUG_ADD_EXPRS
            cerr << "Adding equation-group " << groupName <<
                " with condition #" << ep->index << ": " <<
                (cond.get() ? cond->makeQuotedStr() : "NULL") << endl;
#endif
        }
        assert(ep);

        // Add expr.
        ep->exprs.insert(expr);

        // Remember grids updated in this group.
        ep->grids.insert(gp);
    }
    
    // Remember all grids updated.
    _eqGrids.insert(gp);

    return newGroup;
}


// Separate expressions from grids into eqGroups.
// TODO: do this automatically based on inter-equation dependencies.
void EqGroups::findEqGroups(Grids& allGrids, const string& targets) {

    // Map to track indices per eq-group name.
    map<string, int> indices;
    
    // Handle each key-value pair in targets string.
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
                addExprsFromGrid(key, indices, gp);
            }
        });

    // Add all grids not matching any values.
    for (auto gp : allGrids) {

        // Add equations.
        string key = gp->getName();
        addExprsFromGrid(key, indices, gp);
    }

    // Now, all equations should be added to this object.
    // Can remove them from temp storage in grids.
    allGrids.clearTemp();
}

// Print stats from eqGroups.
void EqGroups::printStats(ostream& os, const string& msg) {
    CounterVisitor cv;
    for (auto& eq : *this) {
        CounterVisitor ecv;
        eq.visitExprs(&ecv);
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
        }
        
        // Add this dimension to fold/cluster only if it was found in all grids.
        else if (_dimCounts.getVal(dim) == (int)grids.size()) {
            _foldLengths.addDimBack(dim, 1);
            _clusterLengths.addDimBack(dim, 1);
        }
        else
            _miscDims.addDimBack(dim, 1);
    }
    os << "Step dimension: " << stepDim << endl;
    
    // Create final fold lengths based on cmd-line options.
    IntTuple foldLengthsGT1;    // fold dimensions > 1.
    for (auto dim : foldOptions.getDims()) {
        int sz = foldOptions.getVal(dim);
        int* p = _foldLengths.lookup(dim);
        if (!p) {
            os << "Error: fold-length of " << sz << " in '" << dim <<
                "' dimension not allowed because '" << dim << "' ";
            if (dim == stepDim)
                os << "is the step dimension." << endl;
            else
                os << "doesn't exist in all grids." << endl;
            exit(1);
        }
        *p = sz;
        if (sz > 1)
            foldLengthsGT1.addDimBack(dim, sz);
            
    }
    os << "Vector-fold dimension(s) and size(s): " <<
        _foldLengths.makeDimValStr(" * ") << endl;


    // Checks for unaligned loads.
    if (allowUnalignedLoads) {
        if (foldLengthsGT1.size() > 1) {
            os << "Error: attempt to allow unaligned loads when there are " <<
                foldLengthsGT1.size() << " dimensions in the vector-fold that are > 1." << endl;
            exit(1);
        }
        else if (foldLengthsGT1.size() > 0)
            os << "Notice: memory map MUST be with unit-stride in " <<
                foldLengthsGT1.makeDimStr() << " dimension!" << endl;
    }

    // Create final cluster lengths based on cmd-line options.
    for (auto dim : clusterOptions.getDims()) {
        int sz = clusterOptions.getVal(dim);
        int* p = _clusterLengths.lookup(dim);
        if (!p) {
            os << "Error: cluster-length of " << sz << " in '" << dim <<
                "' dimension not allowed because '" << dim << "' ";
            if (dim == stepDim)
                os << "is the step dimension." << endl;
            else
                os << "doesn't exist in all grids." << endl;
            exit(1);
        }
        *p = sz;
    }
    os << "Cluster dimension(s) and size(s): " <<
        _clusterLengths.makeDimValStr(" * ") << endl;
    os << "Other dimension(s): " << _miscDims.makeDimStr(", ") << endl;
}
