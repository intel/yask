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

//////////// Expression utilities /////////////

#ifndef EXPR_UTILS_HPP
#define EXPR_UTILS_HPP

#include "Visitor.hpp"

namespace yask {

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
    // TODO: simplify exprs like a + -b.
    class CombineVisitor : public OptVisitor {
    public:
        CombineVisitor()  :
            OptVisitor("commutative recombination") {}
        virtual ~CombineVisitor() {}

        virtual int getNumChanges() {
            return _numChanges;
        }
    
        virtual void visit(CommutativeExpr* ce);
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
        virtual bool findMatchTo(NumExprPtr& ep);
    
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
        }
        virtual void visit(EqualsExpr* ee) {

            // Only process RHS.
            // TODO: process LHS to find dependencies.
            // TODO: consider processing condition.
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

        // Visits are considered unique by address, not semantic equivalence.
        virtual bool alreadyVisited(Expr* ep) {
#if DEBUG_TRACKING >= 1
            cout << " //** tracking '" << ep->makeStr() << "'@" << ep << endl;
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
            os << " Expression stats";
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
        int _numOps, _numNodes, _numReads, _numWrites;

    public:
        CounterVisitor() :
            _numOps(0), _numNodes(0), _numReads(0), _numWrites(0) { }
        virtual ~CounterVisitor() {}

        virtual CounterVisitor& operator+=(const CounterVisitor& rhs) {
            TrackingVisitor::operator+=(rhs);
            _numOps += rhs._numOps;
            _numNodes += rhs._numNodes;
            _numReads += rhs._numReads;
            _numWrites += rhs._numWrites;
            return *this;
        }
    
        virtual void printStats(ostream& os, const string& descr = "") const {
            TrackingVisitor::printStats(os, descr);
            os << 
                "  " << getNumReads() << " grid read(s)." << endl <<
                "  " << getNumWrites() << " grid write(s)." << endl <<
                "  " << getNumOps() << " FP math operation(s)." << endl;
        }
    
        int getNumNodes() const { return _numNodes; }
        int getNumReads() const { return _numReads; }
        int getNumWrites() const { return _numWrites; }
        int getNumOps() const { return _numOps; }

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
            _numReads++;
        }
    
        // Unary: Count as one op if num type and visit operand.
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

        // Commutative: count as one op between each operand and visit operands.
        virtual void visit(CommutativeExpr* ce) {
            if (alreadyVisited(ce)) return;
            _numNodes++;
            auto& ops = ce->getOps();
            _numOps += ops.size() - 1;
            for (auto& ep : ops) {
                ep->accept(this);
            }
        }

        // Equality: assume LHS is a write; don't visit it, and don't count
        // equality as a node. Also, don't visit condition or count as nodes.
        virtual void visit(EqualsExpr* ee) {
            if (alreadyVisited(ee)) return;
            _numWrites++;
            ee->getRhs()->accept(this);
        }
    };

} // namespace yask.

#endif
