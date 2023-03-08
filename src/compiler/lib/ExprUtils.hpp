/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

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

#pragma once

#include "Visitor.hpp"

namespace yask {

    // Base class for a visitor that applies an optimization.
    class OptVisitor : public ExprVisitor {
    protected:
        int _num_changes;
        string _name;

    public:
        OptVisitor(const string& name) :
            _num_changes(0), _name(name) {}
        virtual ~OptVisitor() {}

        virtual int get_num_changes() const {
            return _num_changes;
        }
        virtual void set_num_changes(int n) {
            _num_changes = n;
        }

        virtual const string& _get_name() const {
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

        virtual int get_num_changes() const override {
            return _num_changes;
        }

        virtual string visit(CommutativeExpr* ce) override;
    };


    // A visitor that eliminates common numerical subexprs.
    // TODO: find matches to subsets of commutative operations;
    // example: a+b+c * b+d+a => c+(a+b) * d+(a+b) w/expr a+b combined.
    class CseVisitor : public OptVisitor {
    protected:
        set<num_expr_ptr> _seen;

        // If 'ep' has already been seen, just return true.
        // Else if 'ep' has a match, change pointer to that match, return true.
        // Else, return false.
        virtual bool find_match_to(num_expr_ptr& ep);

    public:
        CseVisitor()  :
            OptVisitor("common subexpr elimination") {}
        virtual ~CseVisitor() {}

        // For each visitor w/children,
        // - For each child,
        //   - Redirect child pointer to matching node if one exists,
        //     otherwise, visit child.
        virtual string visit(UnaryNumExpr* ue) {
            auto& rhs = ue->_get_rhs();
            if (!find_match_to(rhs))
                rhs->accept(this);
            return "";
        }
        virtual string visit(BinaryNumExpr* be) {
            auto& lhs = be->_get_lhs();
            if (!find_match_to(lhs))
                lhs->accept(this);
            auto& rhs = be->_get_rhs();
            if (!find_match_to(rhs))
                rhs->accept(this);
            return "";
        }
        virtual string visit(CommutativeExpr* ce) {
            auto& ops = ce->get_ops();
            for (auto& ep : ops) {
                if (!find_match_to(ep))
                    ep->accept(this);
            }
            return "";
        }
        virtual string visit(FuncExpr* fe) {
            auto& ops = fe->get_ops();
            for (auto& ep : ops) {
                if (!find_match_to(ep))
                    ep->accept(this);
            }
            return "";
        }
        virtual string visit(EqualsExpr* ee) {

            // Only process RHS.
            auto& rhs = ee->_get_rhs();
            if (!find_match_to(rhs))
                rhs->accept(this);
            return "";
        }
    };

    // A visitor that finds pairable functions, e.g., sin(x); cos(x) => sincos(x).
    class PairingVisitor : public OptVisitor {
    protected:
        set<FuncExpr*> _seen;

    public:
        PairingVisitor()  :
            OptVisitor("function pairing") {}

        // Only need to check func nodes.
        virtual string visit(FuncExpr* fe);

        // For other nodes, default behavior of
        // visiting children is ok.
    };


    // A visitor that can keep track of what's been visted.
    class TrackingVisitor : public ExprVisitor {
    protected:
        map<Expr*, int> _counts;
        int _visits;

        // Visits are considered unique by address, not semantic equivalence.
        virtual bool already_visited(Expr* ep) {
            #if DEBUG_TRACKING >= 1
            cout << " //** tracking '" << ep->make_str() << "'@" << ep << endl;
            #endif
            bool seen = _counts.count(ep) > 0;

            // Mark as seen for next time and count visits.
            _counts[ep]++;
            _visits++;

            // Return whether seen previously.
            return seen;
        }

    public:
        TrackingVisitor() : _visits(0) {}
        virtual ~TrackingVisitor() {}

        virtual int get_num_visits() const {
            return _visits;
        }

        virtual int get_count(Expr* ep) const {
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

    };

    // A visitor that counts things and collects some other
    // data on expressions.
    // Doesn't count things in common subexprs.
    // Doesn't count things in condition exprs.
    class CounterVisitor : public TrackingVisitor {
    protected:
        int _num_ops=0, _num_nodes=0, _num_reads=0, _num_writes=0, _num_paired=0;

    public:
        CounterVisitor() {}
        virtual ~CounterVisitor() {}

        virtual CounterVisitor& operator+=(const CounterVisitor& rhs) {
            TrackingVisitor::operator+=(rhs);
            _num_ops += rhs._num_ops;
            _num_nodes += rhs._num_nodes;
            _num_reads += rhs._num_reads;
            _num_writes += rhs._num_writes;
            _num_paired += rhs._num_paired;
            return *this;
        }

        virtual void print_stats(ostream& os, const string& descr = "") const {
            os << " Expression stats";
            if (descr.length())
                os << " " << descr;
            os << ":" << endl <<
                "  " << _get_num_nodes() << " node(s)." << endl <<
                "  " << get_num_pairs() << " node pair(s)." << endl <<
                "  " << get_num_reads() << " var read(s)." << endl <<
                "  " << get_num_writes() << " var write(s)." << endl <<
                "  " << get_num_ops() << " FP math operation(s)." << endl;
        }

        int _get_num_nodes() const { return _num_nodes; }
        int get_num_reads() const { return _num_reads; }
        int get_num_writes() const { return _num_writes; }
        int get_num_ops() const { return _num_ops; }
        int get_num_pairs() const { return _num_paired / 2; }

        // Leaf nodes.
        virtual string visit(ConstExpr* ce) {
            if (already_visited(ce)) return "";
            _num_nodes++;
            return "";
        }
        virtual string visit(CodeExpr* ce) {
            if (already_visited(ce)) return "";
            _num_nodes++;
            return "";
        }
        virtual string visit(VarPoint* gp) {
            if (already_visited(gp)) return "";
            _num_nodes++;
            _num_reads++;
            return "";
        }

        // Unary: Count as one op if num type and visit operand.
        virtual string visit(UnaryNumExpr* ue) {
            if (already_visited(ue)) return "";
            _num_nodes++;
            _num_ops++;
            ue->_get_rhs()->accept(this);
            return "";
        }
        virtual string visit(UnaryBoolExpr* ue) {
            if (already_visited(ue)) return "";
            _num_nodes++;
            ue->_get_rhs()->accept(this);
            return "";
        }
        virtual string visit(UnaryNum2BoolExpr* ue) {
            if (already_visited(ue)) return "";
            _num_nodes++;
            ue->_get_rhs()->accept(this);
            return "";
        }

        // Binary: Count as one op if numerical and visit operands.
        virtual string visit(BinaryNumExpr* be) {
            if (already_visited(be)) return "";
            _num_nodes++;
            _num_ops++;
            be->_get_lhs()->accept(this);
            be->_get_rhs()->accept(this);
            return "";
        }
        virtual string visit(BinaryBoolExpr* be) {
            if (already_visited(be)) return "";
            _num_nodes++;
            be->_get_lhs()->accept(this);
            be->_get_rhs()->accept(this);
            return "";
        }
        virtual string visit(BinaryNum2BoolExpr* be) {
            if (already_visited(be)) return "";
            _num_nodes++;
            be->_get_lhs()->accept(this);
            be->_get_rhs()->accept(this);
            return "";
        }

        // Commutative: count as one op between each operand and visit operands.
        virtual string visit(CommutativeExpr* ce) {
            if (already_visited(ce)) return "";
            _num_nodes++;
            auto& ops = ce->get_ops();
            _num_ops += ops.size() - 1;
            for (auto& ep : ops)
                ep->accept(this);
            return "";
        }

        // Function: count as one op and visit operands.
        virtual string visit(FuncExpr* fe) {
            if (already_visited(fe)) return "";
            _num_nodes++;
            _num_ops++;
            if (fe->get_pair())
                _num_paired++;
            auto& ops = fe->get_ops();
            for (auto& ep : ops)
                ep->accept(this);
            return "";
        }

        // Equality: assume LHS is a write; don't visit it, and don't count
        // equality as a node. Also, don't visit condition or count as nodes.
        virtual string visit(EqualsExpr* ee) {
            if (already_visited(ee)) return "";
            _num_writes++;
            ee->_get_rhs()->accept(this);
            return "";
        }
    };
   
} // namespace yask.

