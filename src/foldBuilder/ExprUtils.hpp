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

// A visitor that eliminates CSEs.
class CseElimVisitor : public ExprVisitor {
protected:
    set<ExprPtr> _seen;
    int _numReplaced;
    
    // If 'ep' has already been seen, just return true.
    // Else if 'ep' has a match, change pointer to that match, return true.
    // Else, return false.
    virtual bool isMatchTo(ExprPtr& ep) {
#ifdef DEBUG_MATCHING
        cerr << "- checking " << ep->makeStr() << endl;
#endif
        
        // Already visited this node?
        // This can happen when code is written with
        // shared expressions.
        if (_seen.count(ep)) {
#ifdef DEBUG_MATCHING
            cerr << " - already seen " << ep->makeStr() << endl;
#endif
            return true;
        }
        
        // Loop through nodes already seen.
        for (auto oep : _seen) {
#ifdef DEBUG_MATCHING
            cerr << " - comparing " << ep->makeStr() << " to " << oep->makeStr() << endl;
#endif
            
            // Match?
            if (ep->isSame(oep.get())) {
#ifdef DEBUG_MATCHING
                cerr << "  - found match: " << ep->makeStr() << " to " << oep->makeStr() << endl;
#endif
                
                // Redirect pointer to the matching expr.
                ep = oep;
                _numReplaced++;
                return true;
            }
        }

#if 0
        // Already visited an expr just like this one?
        if (_csev) {
            auto msetp = _csev->getMatchesTo(ep.get());
            if (msetp) {
                for (auto mp : *msetp) {
                    if (_seen.count(mp)) {

                        // Redirect pointer to the matching expr.
                        ep.reset(mp);
                        _numReplaced++;
                        return true;
                    }
                }
            }
        }
#endif
        
        // Mark as seen now.
#ifdef DEBUG_MATCHING
        cerr << " - no match to " << ep->makeStr() << endl;
#endif
        _seen.insert(ep);
        return false;
    }
    
public:
    CseElimVisitor() : _numReplaced(0) {}
    virtual ~CseElimVisitor() {}

    virtual int getNumReplaced() const {
        return _numReplaced;
    }

    // For each visitor w/children,
    // - For each child,
    //   - Redirect child pointer to matching node if one exists,
    //     otherwise, visit child.
    virtual void visit(UnaryExpr* ue) {
        ExprPtr& rhs = ue->getRhs();
        if (!isMatchTo(rhs))
            rhs->accept(this);
    }
    virtual void visit(BinaryExpr* be) {
        ExprPtr& lhs = be->getLhs();
        if (!isMatchTo(lhs))
            lhs->accept(this);
        ExprPtr& rhs = be->getRhs();
        if (!isMatchTo(rhs))
            rhs->accept(this);
    }
    virtual void visit(CommutativeExpr* ce) {
        ExprPtrVec& ops = ce->getOps();
        for (ExprPtr& ep : ops) {
            if (!isMatchTo(ep))
                ep->accept(this);
        }
    }
    virtual void visit(EqualsExpr* ee) {
        // Don't do LHS.
        // TODO: check this--there should never be a match.
        ExprPtr& rhs = ee->getRhs();
        if (!isMatchTo(rhs))
            rhs->accept(this);
    }
};

// A visitor that can keep track of what's been visted.
class TrackingVisitor : public ExprVisitor {
protected:
    set<Expr*> _seen;
    map<Expr*, int> _counts;

    virtual bool isSeen(Expr* ep) {
        _counts[ep]++;

        // Already visited this node?
        if (_seen.count(ep))
            return true;

        // Mark as seen now.
        _seen.insert(ep);
        return false;
    }

public:
    TrackingVisitor() {}
    virtual ~TrackingVisitor() {}
    
    virtual int getCount(Expr* ep) const {
        auto it = _counts.find(ep);
        if (it == _counts.end()) return 0;
        return it->second;
    }
};

// A visitor that counts things.
// Doesn't count those in CSEs.
class CounterVisitor : public TrackingVisitor {
protected:
    int _numOps, _numNodes, _numReads, _numWrites, _numParamReads;
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
    
    int getNumNodes() const { return _numNodes; }
    int getNumReads() const { return _numReads; }
    int getNumWrites() const { return _numWrites; }
    int getNumParamReads() const { return _numParamReads; }
    int getNumOps() const { return _numOps; }
    const IntTuple* getMaxPoints(const Grid* gp) const {
        return getPoints(gp, _maxPoints);
    }
    const IntTuple* getMinPoints(const Grid* gp) const {
        return getPoints(gp, _minPoints);
    }

    // Return halo needed for given grid in given dimension.
    // TODO: allow separate halos for beginning and end.
    int getHalo(const Grid* gp, const string& dim) const {
        auto maxps = getMaxPoints(gp);
        auto minps = getMinPoints(gp);
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
        if (isSeen(ce)) return;
        _numNodes++;
    }
    virtual void visit(CodeExpr* ce) {
        if (isSeen(ce)) return;
        _numNodes++;
    }
    virtual void visit(GridPoint* gp) {
        if (isSeen(gp)) return;
        _numNodes++;
        if (gp->isParam())
            _numParamReads++;
        else
            _numReads++;

        // Track max and min points accessed for this grid.
        const Grid* g = gp->getGrid();
        auto& maxp = _maxPoints[g];
        maxp = gp->maxElements(maxp, false);
        auto& minp = _minPoints[g];
        minp = gp->minElements(minp, false);
    }
    
    // Unary: Count as one op and visit operand.
    // TODO: simplify exprs like a + -b.
    virtual void visit(UnaryExpr* ue) {
        if (isSeen(ue)) return;
        _numNodes++;
        _numOps++;
        ue->getRhs()->accept(this);
    }

    // Binary: Count as one op and visit operands.
    virtual void visit(BinaryExpr* be) {
        if (isSeen(be)) return;
        _numNodes++;
        _numOps++;
        be->getLhs()->accept(this);
        be->getRhs()->accept(this);
    }

    // Count as one op between each operand and visit operands.
    virtual void visit(CommutativeExpr* ce) {
        if (isSeen(ce)) return;
        _numNodes++;
        ExprPtrVec& ops = ce->getOps();
        _numOps += ops.size() - 1;
        for (auto ep : ops) {
            ep->accept(this);
        }
    }

    // Equality: assume LHS is a write; don't visit it.
    virtual void visit(EqualsExpr* ee) {
        if (isSeen(ee)) return;
        _numNodes++;
        _numWrites++;
        ee->getRhs()->accept(this);
    }
};

#endif
