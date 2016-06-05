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

// A visitor that applies an optimization.
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

// A visitor that distributes multiplication over
// addition, thereby potentially increasing FMAs.
// Examples:
//  a * (b + c)     => (a * b) + (a * c).
//  a * (b + c + d) => (a * b) + (a * c) + (a * d).
//  (a + b) * (c + d) => (a * (c + d)) + (b * (c + d)).
//  ((a - b) + (c - d)) * e => ((a - b) * e) + ((c - d) * e).
//  ((a * b) + (c * d)) * e => (a * b * e) + (c * d * e); bad?
//  TODO: a * (b + c) * d => ((a * b) + (a * c)) * d; only do one.
class FmaVisitor : public OptVisitor {
public:
    FmaVisitor() :
        OptVisitor("distribution for FMA") {}
    virtual ~FmaVisitor() {}

    virtual void visit(CommutativeExpr* ce) {
        ExprPtrVec& ops = ce->getOps();

        // Visit ops first (depth-first).
        int initChanges = _numChanges;
        for (auto ep : ops) {
            ep->accept(this);
        }

        // Don't apply recursively.
        if (_numChanges > initChanges)
            return;

        // Is this a multiply of exactly 2 items?
        auto opstr = ce->getOpStr();
        if (opstr == MultExpr::opStr() && ops.size() == 2) {

            // Repeat until no changes.
            bool done = false;
            while (!done) {
                done = true;        // assume done until change is made.

                // Look through ops.
                for (size_t i = 0; i < ops.size(); i++) {
                    ExprPtr& ep = ops[i];
                    auto ae = dynamic_pointer_cast<CommutativeExpr>(ep);

                    // Is this an add?
                    if (ae && ae->getOpStr() == AddExpr::opStr()) {
                        ExprPtrVec& oldAdd = ae->getOps();

                        // We will change ce to be an add and
                        // change its operands to be mults.
                        auto newAdd = make_shared<CommutativeExpr>(AddExpr::opStr());

                        // Make a new mult expr for each
                        // operand of the original add.
                        for (size_t j = 0; j < oldAdd.size(); j++) {
                            ExprPtr& aep = oldAdd[j];

                            // Keep all the old multiplicands from the original
                            // mult except the add being replaced (which is at i).
                            auto newMult = make_shared<CommutativeExpr>(opstr);
                            for (size_t k = 0; k < ops.size(); k++) {
                                if (k != i) {
                                    ExprPtr& ep = ops[k];
                                    bool ok = newMult->appendOp(ep, opstr);
                                    assert(ok);
                                }
                            }

                            // Append the current addend as a new multiplicand.
                            bool ok = newMult->appendOp(aep, opstr);
                            assert(ok);

                            // Add the new multiplication.
                            ok = newAdd->appendOp(newMult, AddExpr::opStr());
                            assert(ok);
                        }

                        // Finally, we can change ce from a mult of adds to an add of mults.
                        ce->swap(newAdd.get());
                        _numChanges++;
                        done = false;
                    }
                }
            } // while !done.
        } // if a mult.
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
        ExprPtrVec& ops = ce->getOps();

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
                ExprPtr& ep = ops[i];
                auto ce2 = dynamic_pointer_cast<CommutativeExpr>(ep);

                // Is ep also a commutative expr with same operator?
                if (ce2 && ce2->getOpStr() == opstr) {

                    // put ce2's operands into ce.
                    ExprPtrVec& ops2 = ce2->getOps();
                    bool isFirst = true;
                    for (ExprPtr ep2 : ops2) {

                        // First operand of ce2 *replaces* ce2 in ce.
                        if (isFirst) {
                            ops[i] = ep2;
                            isFirst = false;
                        }

                        // Remaining operands added to end.
                        else {
                            bool ok = ce->appendOp(ep2, opstr);
                            assert(ok);
                        }
                    }

                    // Bail out of for loop because we have modified ops.
                    _numChanges++;
                    done = false;
                    break;
                }
            }
        }
    }
    
};


// A visitor that eliminates common subexprs.
class CseVisitor : public OptVisitor {
protected:
    set<ExprPtr> _seen;
    
    // If 'ep' has already been seen, just return true.
    // Else if 'ep' has a match, change pointer to that match, return true.
    // Else, return false.
    virtual bool findMatchTo(ExprPtr& ep) {
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
                _numChanges++;
                return true;
            }
        }

        // Mark as seen.
#ifdef DEBUG_MATCHING
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
    virtual void visit(UnaryExpr* ue) {
        ExprPtr& rhs = ue->getRhs();
        if (!findMatchTo(rhs))
            rhs->accept(this);
    }
    virtual void visit(BinaryExpr* be) {
        ExprPtr& lhs = be->getLhs();
        if (!findMatchTo(lhs))
            lhs->accept(this);
        ExprPtr& rhs = be->getRhs();
        if (!findMatchTo(rhs))
            rhs->accept(this);
    }
    virtual void visit(CommutativeExpr* ce) {
        ExprPtrVec& ops = ce->getOps();
        for (ExprPtr& ep : ops) {
            if (!findMatchTo(ep))
                ep->accept(this);
        }
    }
    virtual void visit(EqualsExpr* ee) {
        // Don't do LHS.
        // TODO: check this--there should never be a match.
        ExprPtr& rhs = ee->getRhs();
        if (!findMatchTo(rhs))
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
// Doesn't count those in common subexprs.
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

    virtual void printStats(ostream& os, const string& descr = "") const {
        os << "Expression stats";
        if (descr.length())
            os << " " << descr;
        os << ":" << endl <<
            "  " << getNumNodes() << " nodes." << endl <<
            "  " << getNumReads() << " grid reads." << endl <<
            "  " << getNumWrites() << " grid writes." << endl <<
            "  " << getNumParamReads() << " parameter reads." << endl <<
            "  " << getNumOps() << " math operations." << endl;
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
