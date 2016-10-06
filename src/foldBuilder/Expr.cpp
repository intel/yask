/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

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

// Unary.
ExprPtr constGridValue(double rhs) {
    return make_shared<ConstExpr>(rhs);
}
ExprPtr operator-(const ExprPtr& rhs) {
    return make_shared<NegExpr>(rhs);
}

// Commutative.
ExprPtr operator+(const ExprPtr& lhs, const ExprPtr& rhs) {

    // If one side is nothing, return other side;
    // This allows us to add to an uninitialized GridValue
    // and do the right thing.
    if (lhs == NULL)
        return rhs;
    else if (rhs == NULL)
        return lhs;

    // Start with an empty expression.
    auto ex = make_shared<AddExpr>();

    // If LHS is also an AddExpr, add its operands.
    // Othewise, add the whole expr.
    ex->mergeExpr(lhs);

    // If RHS is also an AddExpr, add its operands.
    // Othewise, add the whole expr.
    ex->mergeExpr(rhs);

    return ex;
}
ExprPtr operator+(double lhs, const ExprPtr& rhs) {
    ExprPtr p = make_shared<ConstExpr>(lhs);
    return p + rhs;
}
ExprPtr operator+(const ExprPtr& lhs, double rhs) {
    ExprPtr p = make_shared<ConstExpr>(rhs);
    return lhs + p;
}

void operator+=(ExprPtr& lhs, const ExprPtr& rhs) {
    lhs = lhs + rhs;
}
void operator+=(ExprPtr& lhs, double rhs) {
    lhs = lhs + rhs;
}

ExprPtr operator*(const ExprPtr& lhs, const ExprPtr& rhs) {

    // If one side is nothing, return other side.
    if (lhs == NULL)
        return rhs;
    else if (rhs == NULL)
        return lhs;


    // Start with an empty expression.
    auto ex = make_shared<MultExpr>();

    // If LHS is also an MultExpr, add its operands.
    // Othewise, add the whole expr.
    ex->mergeExpr(lhs);

    // If RHS is also an MultExpr, add its operands.
    // Othewise, add the whole expr.
    ex->mergeExpr(rhs);

    return ex;
}
ExprPtr operator*(double lhs, const ExprPtr& rhs) {
    ExprPtr p = make_shared<ConstExpr>(lhs);
    return p * rhs;
}
ExprPtr operator*(const ExprPtr& lhs, double rhs) {
    ExprPtr p = make_shared<ConstExpr>(rhs);
    return lhs * p;
}

void operator*=(ExprPtr& lhs, const ExprPtr& rhs) {
    lhs = lhs * rhs;
}
void operator*=(ExprPtr& lhs, double rhs) {
    lhs = lhs * rhs;
}

// Binary.
ExprPtr operator-(const ExprPtr& lhs, const ExprPtr& rhs) {

#ifdef USE_ADD_NEG
    // Generate A + -B instead of A - B to allow easy reordering.
    ExprPtr nrhs = make_shared<NegExpr>(rhs);
    return lhs + nrhs;
#else
    return make_shared<SubExpr>(lhs, rhs);
#endif    
}
ExprPtr operator-(double lhs, const ExprPtr& rhs) {
    ExprPtr p = make_shared<ConstExpr>(lhs);
    return p - rhs;
}
ExprPtr operator-(const ExprPtr& lhs, double rhs) {
    ExprPtr p = make_shared<ConstExpr>(rhs);
    return lhs - p;
}

void operator-=(ExprPtr& lhs, const ExprPtr& rhs) {
    lhs = lhs - rhs;
}
void operator-=(ExprPtr& lhs, double rhs) {
    lhs = lhs - rhs;
}

ExprPtr operator/(const ExprPtr& lhs, const ExprPtr& rhs) {

    return make_shared<DivExpr>(lhs, rhs);
}
ExprPtr operator/(double lhs, const ExprPtr& rhs) {
    ExprPtr p = make_shared<ConstExpr>(lhs);
    return p / rhs;
}
ExprPtr operator/(const ExprPtr& lhs, double rhs) {
    ExprPtr p = make_shared<ConstExpr>(rhs);
    return lhs / p;
}

void operator/=(ExprPtr& lhs, const ExprPtr& rhs) {
    lhs = lhs / rhs;
}
void operator/=(ExprPtr& lhs, double rhs) {
    lhs = lhs / rhs;
}

// Define the value of a grid point.
// Note that the semantics are different than the 'normal'
// '==' operator, which tests for equality.
void operator==(GridPointPtr gpp, ExprPtr rhs) {
    assert(gpp != NULL);

    // Make sure this is a grid.
    if (gpp->isParam()) {
        cerr << "Error: parameter '" << gpp->getName() <<
            "' cannot appear on LHS of a grid-value equation." << endl;
        exit(1);
    }
    
    // Make expression node.
    ExprPtr p = make_shared<EqualsExpr>(gpp, rhs);

    // Save it in the grid.
    Grid* gp = gpp->getGrid();
    assert(gp);
    gp->addExpr(gpp, p);
}

// Visitor acceptors.
void ConstExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void CodeExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void UnaryExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void BinaryExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void CommutativeExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void GridPoint::accept(ExprVisitor* ev) {
    ev->visit(this);
}
void EqualsExpr::accept(ExprVisitor* ev) {
    ev->visit(this);
}

// EqualsExpr methods.
bool EqualsExpr::isSame(const Expr* other) {
        auto p = dynamic_cast<const EqualsExpr*>(other);
        return p && _opStr == p->_opStr &&
            _lhs->isSame(p->_lhs.get()) &&
            _rhs->isSame(p->_rhs.get());
    }

// Commutative methods.
bool CommutativeExpr::isSame(const Expr* other) {
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
    return _grid->getName() + "(" +
        makeDimValOffsetStr() + ")";
}

// Visit all expressions in all grids.
void Grids::acceptToAll(ExprVisitor* ev) {
    for (auto gp : *this) {
        gp->acceptToAll(ev);
    }
}

// Visit first expression in each grid.
void Grids::acceptToFirst(ExprVisitor* ev) {
    for (auto gp : *this) {
        gp->acceptToFirst(ev);
    }
}

// Make a readable string from an expression.
string Expr::makeStr() const {
    ostringstream oss;
    
    // Use a print visitor to make a string.
    PrintHelper ph(NULL, "temp", "", "", "");
    PrintVisitorTopDown pv(oss, ph);
    accept(&pv);

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

// Separate grids into equations.
void Equations::findEquations(Grids& allGrids, const string& targets) {

    // Handle each key-value pair in targets string.
    ArgParser ap;
    ap.parseKeyValuePairs
        (targets, [&](const string& key, const string& value) {

            // Search allGrids for matches.
            for (auto gp : allGrids) {

                // does the value appear in the grid name?
                string gname = gp->getName();
                size_t np = gname.find(value);
                if (np != string::npos) {

                    // Grid already added?
                    if (_eqGrids.count(gp))
                        continue;

                    // Grid has an equation?
                    if (gp->getExprs().size() == 0)
                        continue;

                    // Find existing equation named key.
                    Equation* ep = 0;
                    for (auto& eq : *this) {
                        if (eq.name == key) {
                            ep = &eq;
                            break;
                        }
                    }

                    // Add equation if needed.
                    if (!ep) {
                        Equation ne;
                        push_back(ne);
                        ep = &back();
                        ep->name = key;
                    }

                    // Add grid to equation.
                    assert(ep);
                    ep->grids.push_back(gp);
                    _eqGrids.insert(gp);
                }
            }
        });

    // Add all grids not already added.
    for (auto gp : allGrids) {

        // Grid already added?
        if (_eqGrids.count(gp))
            continue;
        
        // Grid has an equation?
        if (gp->getExprs().size() == 0)
            continue;
        
        // Make a new equation.
        Equation ne;
        push_back(ne);
        Equation& eq = back();
        
        // It has the name of the grid and just one grid.
        eq.name = gp->getName();
        eq.grids.push_back(gp);
        _eqGrids.insert(gp);
    }
}

// Print stats from equations.
void Equations::printStats(ostream& os, const string& msg, bool visitAll) {
    CounterVisitor cv;
    for (auto& eq : *this) {
        CounterVisitor ecv;
        if (visitAll)
            eq.grids.acceptToAll(&ecv);
        else
            eq.grids.acceptToFirst(&ecv);
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
                     
    // Create a union of all dimensions in all grids.
    // Also keep count of how many grids have each dim.
    // Note that dimensions won't be in any particular order!
    for (auto gp : grids) {

        // Count dimensions from this grid.
        for (auto dim : gp->getDims()) {
            if (_dimCounts.lookup(dim))
                _dimCounts.setVal(dim, _dimCounts.getVal(dim) + 1);
            else
                _dimCounts.addDimBack(dim, 1);
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
