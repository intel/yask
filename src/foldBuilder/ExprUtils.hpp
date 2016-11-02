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

//////////// Expression utilities /////////////

#ifndef EXPR_UTILS_HPP
#define EXPR_UTILS_HPP

#include "Visitor.hpp"

using namespace std;

// Base class for a visitor that applies an optimization.
class OptVisitor : public ExprVisitor {
protected:
    int _numChanges;
    string _name;
    
public:
    OptVisitor(const string& name) :
        _numChanges(0), _name(name) {}
    virtual ~OptVisitor() {}

    virtual int getNumChanges() const {
        return _numChanges;
    }
    virtual void setNumChanges(int n) {
        _numChanges = n;
    }

    virtual const string& getName() const {
        return _name;
    }
};

// A visitor that combines commutative exprs.
// Example: (a + b) + c => a + b + c;
class CombineVisitor : public OptVisitor {
public:
    CombineVisitor()  :
        OptVisitor("commutative recombination") {}
    virtual ~CombineVisitor() {}

    virtual int getNumChanges() {
        return _numChanges;
    }
    
    virtual void visit(CommutativeExpr* ce) {
        auto& ops = ce->getOps();

        // Visit ops first (depth-first).
        for (auto ep : ops) {
            ep->accept(this);
        }

        // Repeat until no changes.
        auto opstr = ce->getOpStr();
        bool done = false;
        while (!done) {
            done = true;        // assume done until change is made.
        
            // Scan elements of expr for more exprs of same type.
            for (size_t i = 0; i < ops.size(); i++) {
                auto& ep = ops[i];
                auto ce2 = dynamic_pointer_cast<CommutativeExpr>(ep);

                // Is ep also a commutative expr with same operator?
                if (ce2 && ce2->getOpStr() == opstr) {

                    // Delete the existing operand.
                    ops.erase(ops.begin() + i);

                    // Put ce2's operands into ce.
                    ce->mergeExpr(ce2);

                    // Bail out of for loop because we have modified ops.
                    _numChanges++;
                    done = false;
                    break;
                }
            }
        }
    }
};


// A visitor that eliminates common numerical subexprs.
// TODO: find matches to subsets of commutative operations;
// example: a+b+c * b+d+a => c+(a+b) * d+(a+b) w/expr a+b combined.
class CseVisitor : public OptVisitor {
protected:
    set<NumExprPtr> _seen;
    
    // If 'ep' has already been seen, just return true.
    // Else if 'ep' has a match, change pointer to that match, return true.
    // Else, return false.
    virtual bool findMatchTo(NumExprPtr& ep) {
#if DEBUG_CSE >= 1
        cerr << "- checking '" << ep->makeStr() << "'@" << ep << endl;
#endif
        
        // Already visited this node?
        if (_seen.count(ep)) {
#if DEBUG_CSE >= 2
            cerr << " - already seen '" << ep->makeStr() << "'@" << ep << endl;
#endif
            return true;
        }
        
        // Loop through nodes already seen.
        for (auto& oep : _seen) {
#if DEBUG_CSE >= 3
            cerr << " - comparing '" << ep->makeStr() << "'@" << ep <<
                " to '" << oep->makeStr() << "'@" << oep << endl;
#endif
            
            // Match?
            if (ep->isSame(oep.get())) {
#if DEBUG_CSE >= 1
                cerr << "  - found match: '" << ep->makeStr() << "'@" << ep <<
                    " to '" << oep->makeStr() << "'@" << oep << endl;
#endif
                
                // Redirect pointer to the matching expr.
                ep = oep;
                _numChanges++;
                return true;
            }
        }

        // Mark as seen.
#if DEBUG_CSE >= 2
        cerr << " - no match to " << ep->makeStr() << endl;
#endif
        _seen.insert(ep);
        return false;
    }
    
public:
    CseVisitor()  :
        OptVisitor("common subexpr elimination") {}
    virtual ~CseVisitor() {}

    // For each visitor w/children,
    // - For each child,
    //   - Redirect child pointer to matching node if one exists,
    //     otherwise, visit child.
    virtual void visit(UnaryNumExpr* ue) {
        auto& rhs = ue->getRhs();
        if (!findMatchTo(rhs))
            rhs->accept(this);
    }
    virtual void visit(BinaryNumExpr* be) {
        auto& lhs = be->getLhs();
        if (!findMatchTo(lhs))
            lhs->accept(this);
        auto& rhs = be->getRhs();
        if (!findMatchTo(rhs))
            rhs->accept(this);
    }
    virtual void visit(CommutativeExpr* ce) {
        auto& ops = ce->getOps();
        for (auto& ep : ops) {
            if (!findMatchTo(ep))
                ep->accept(this);
        }
        //cerr << "ce " << ce << " after:"; for (auto& ep : ops) cerr << ' ' << ep; cerr << endl;
    }
    virtual void visit(IfExpr* ie) {

        // Only process RHS of expression.
        // TODO: consider processing condition.
        auto& ee = ie->getExpr();
        visit(ee.get());        // compile-time binding ok for expr.
    }
    virtual void visit(EqualsExpr* ee) {

        // Only process RHS.
        // TODO: process LHS to find dependencies.
        auto& rhs = ee->getRhs();
        if (!findMatchTo(rhs))
            rhs->accept(this);
    }
};

// A visitor that can keep track of what's been visted.
class TrackingVisitor : public ExprVisitor {
protected:
    map<Expr*, int> _counts;
    int _visits;

    virtual bool alreadyVisited(Expr* ep) {
#if DEBUG_TRACKING >= 1
        cerr << "- tracking '" << ep->makeStr() << "'@" << ep << endl;
#endif
        bool seen = _counts.count(ep) > 0;
        _counts[ep]++;
        _visits++;
        return seen;
    }

public:
    TrackingVisitor() : _visits(0) {}
    virtual ~TrackingVisitor() {}

    virtual int getNumVisits() const {
        return _visits;
    }
    
    virtual int getCount(Expr* ep) const {
        auto it = _counts.find(ep);
        if (it == _counts.end()) return 0;
        return it->second;
    }

    virtual TrackingVisitor& operator+=(const TrackingVisitor& rhs) {
        for (auto i : rhs._counts)
            _counts[i.first] += i.second;
        _visits += rhs._visits;
        return *this;
    }
    
    virtual void printStats(ostream& os, const string& descr = "") const {
        os << "Expression stats";
        if (descr.length())
            os << " " << descr;
        os << ":" << endl <<
            "  " << _counts.size() << " node(s)." << endl <<
            "  " << (_visits - _counts.size()) << " shared node(s)." << endl;
    }
};

// A visitor that counts things and collects some other
// data on expressions.
// Doesn't count things in common subexprs.
// Doesn't count things in condition exprs.
class CounterVisitor : public TrackingVisitor {
protected:
    int _numOps, _numNodes, _numReads, _numWrites, _numParamReads;

    // Vars to track min and max points seen for every grid.
    // TODO: track all points to enable queries for halo, temporal
    // extent, and required exchanges.
    map<const Grid*, IntTuple> _maxPoints, _minPoints;
    const IntTuple* getPoints(const Grid* gp,
                              const map<const Grid*, IntTuple>& mp) const {
        auto i = mp.find(gp);
        if (i != mp.end())
            return &(i->second);
        return 0;
    }
    
public:
    CounterVisitor() :
        _numOps(0), _numNodes(0), _numReads(0), _numWrites(0), _numParamReads(0) { }
    virtual ~CounterVisitor() {}

    virtual CounterVisitor& operator+=(const CounterVisitor& rhs) {
        TrackingVisitor::operator+=(rhs);
        _numOps += rhs._numOps;
        _numNodes += rhs._numNodes;
        _numReads += rhs._numReads;
        _numWrites += rhs._numWrites;
        _numParamReads += rhs._numParamReads;
        return *this;
    }
    
    virtual void printStats(ostream& os, const string& descr = "") const {
        TrackingVisitor::printStats(os, descr);
        os << 
            "  " << getNumReads() << " grid read(s)." << endl <<
            "  " << getNumWrites() << " grid write(s)." << endl <<
            "  " << getNumParamReads() << " parameter read(s)." << endl <<
            "  " << getNumOps() << " FP math operation(s)." << endl;
    }
    
    int getNumNodes() const { return _numNodes; }
    int getNumReads() const { return _numReads; }
    int getNumWrites() const { return _numWrites; }
    int getNumParamReads() const { return _numParamReads; }
    int getNumOps() const { return _numOps; }

    // Get max/min points accessed in each direction for given grid.
    const IntTuple* getMaxPoints(const Grid* gp) const {
        return getPoints(gp, _maxPoints);
    }
    const IntTuple* getMinPoints(const Grid* gp) const {
        return getPoints(gp, _minPoints);
    }

    // Return halo needed for given grid in given dimension.
    // TODO: allow separate halos for beginning and end.
    int getHalo(const Grid* gp, const string& dim) const {
        auto* maxps = getMaxPoints(gp);
        auto* minps = getMinPoints(gp);
        const int* maxp = maxps ? maxps->lookup(dim) : 0;
        const int* minp = minps ? minps->lookup(dim) : 0;
        if (!maxp && !minp)
            return 0;
        if (!maxp)
            return abs(*minp);
        if (!minp)
            return abs(*maxp);
        return max(abs(*minp), abs(*maxp));
    }

    // Leaf nodes.
    virtual void visit(ConstExpr* ce) {
        if (alreadyVisited(ce)) return;
        _numNodes++;
    }
    virtual void visit(CodeExpr* ce) {
        if (alreadyVisited(ce)) return;
        _numNodes++;
    }
    virtual void visit(GridPoint* gp) {
        if (alreadyVisited(gp)) return;
        _numNodes++;
        if (gp->isParam())
            _numParamReads++;
        else {
            _numReads++;

            // Track max and min points accessed for this grid.
            const Grid* g = gp->getGrid();
            auto& maxp = _maxPoints[g];
            maxp = gp->maxElements(maxp, false);
            auto& minp = _minPoints[g];
            minp = gp->minElements(minp, false);
        }
    }
    
    // Unary: Count as one op if num type and visit operand.
    // TODO: simplify exprs like a + -b.
    virtual void visit(UnaryNumExpr* ue) {
        if (alreadyVisited(ue)) return;
        _numNodes++;
        _numOps++;
        ue->getRhs()->accept(this);
    }
    virtual void visit(UnaryBoolExpr* ue) {
        if (alreadyVisited(ue)) return;
        _numNodes++;
        ue->getRhs()->accept(this);
    }
    virtual void visit(UnaryNum2BoolExpr* ue) {
        if (alreadyVisited(ue)) return;
        _numNodes++;
        ue->getRhs()->accept(this);
    }

    // Binary: Count as one op if numerical and visit operands.
    virtual void visit(BinaryNumExpr* be) {
        if (alreadyVisited(be)) return;
        _numNodes++;
        _numOps++;
        be->getLhs()->accept(this);
        be->getRhs()->accept(this);
    }
    virtual void visit(BinaryBoolExpr* be) {
        if (alreadyVisited(be)) return;
        _numNodes++;
        be->getLhs()->accept(this);
        be->getRhs()->accept(this);
    }
    virtual void visit(BinaryNum2BoolExpr* be) {
        if (alreadyVisited(be)) return;
        _numNodes++;
        be->getLhs()->accept(this);
        be->getRhs()->accept(this);
    }

    // Count as one op between each operand and visit operands.
    virtual void visit(CommutativeExpr* ce) {
        if (alreadyVisited(ce)) return;
        _numNodes++;
        auto& ops = ce->getOps();
        //cerr << "counting ce " << ce << ":"; for (auto& ep : ops) cerr << ' ' << ep; cerr << endl;
        _numOps += ops.size() - 1;
        for (auto& ep : ops) {
            ep->accept(this);
        }
    }

    // Conditional: don't visit condition.
    // TODO: add separate stats for conditions.
    virtual void visit(IfExpr* ie) {
        if (alreadyVisited(ie)) return;
        //_numNodes++;
        ie->getExpr()->accept(this);
    }

    // Equality: assume LHS is a write; don't visit it.
    virtual void visit(EqualsExpr* ee) {
        if (alreadyVisited(ee)) return;
        _numNodes++;
        _numWrites++;
        ee->getRhs()->accept(this);
    }
};

#endif
