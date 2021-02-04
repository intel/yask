/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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

#include "ExprUtils.hpp"

using namespace std;

namespace yask {

    string CombineVisitor::visit(CommutativeExpr* ce) {
        auto& ops = ce->get_ops();

        // Visit ops first (depth-first).
        for (auto ep : ops) {
            ep->accept(this);
        }

        // Repeat until no changes.
        auto opstr = ce->get_op_str();
        bool done = false;
        while (!done) {
            done = true;        // assume done until change is made.

            // Scan elements of expr for more exprs of same type.
            for (size_t i = 0; i < ops.size(); i++) {
                auto& ep = ops[i];
                auto ce2 = dynamic_pointer_cast<CommutativeExpr>(ep);

                // Is ep also a commutative expr with same operator?
                if (ce2 && ce2->get_op_str() == opstr) {

                    // Delete the existing operand.
                    ops.erase(ops.begin() + i);

                    // Put ce2's operands into ce.
                    ce->merge_expr(ce2);

                    // Bail out of for loop because we have modified ops.
                    _num_changes++;
                    done = false;
                    break;
                }
            }
        }
        return "";
    }


    // If 'ep' has already been seen, just return true.
    // Else if 'ep' has a match, change pointer to that match, return true.
    // Else, return false.
    bool CseVisitor::find_match_to(num_expr_ptr& ep) {
#if DEBUG_CSE >= 1
        cout << " //** checking '" << ep->make_str() << "'@" << ep << endl;
#endif

        // Already visited this node?
        if (_seen.count(ep)) {
#if DEBUG_CSE >= 2
            cout << "  //** already seen '" << ep->make_str() << "'@" << ep << endl;
#endif
            return true;
        }

        // Loop through nodes already seen.
        for (auto& oep : _seen) {
#if DEBUG_CSE >= 3
            cout << "  //** comparing '" << ep->make_str() << "'@" << ep <<
                " to '" << oep->make_str() << "'@" << oep << endl;
#endif

            // Match?
            if (ep->is_same(oep.get())) {
#if DEBUG_CSE >= 1
                cout << "   //** found match: '" << ep->make_str() << "'@" << ep <<
                    " to '" << oep->make_str() << "'@" << oep << endl;
#endif

                // Redirect pointer to the matching expr.
                ep = oep;
                _num_changes++;
                return true;
            }
        }

        // Mark as seen.
#if DEBUG_CSE >= 2
        cout << "  //** no match to " << ep->make_str() << endl;
#endif
        _seen.insert(ep);
        return false;
    }

    // Look for function pairs.
    string PairingVisitor::visit(FuncExpr* fe) {

        // Already visited this node?
        if (_seen.count(fe)) {
#if DEBUG_PAIR >= 2
            cout << "  //** already seen '" << ep->make_str() << "'@" << ep << endl;
#endif
            return "";
        }

        // Loop through func nodes already seen.
        for (auto& oep : _seen) {
#if DEBUG_PAIR >= 3
            cout << "  //** comparing '" << fe->make_str() << "'@" << ep <<
                " to '" << oep->make_str() << "'@" << oep << endl;
#endif

            // Pair?
            if (fe->make_pair(oep)) {
#if DEBUG_PAIR >= 1
                cout << "   //** found pair: '" << ep->make_str() << "'@" << ep <<
                    " to '" << oep->make_str() << "'@" << oep << endl;
#endif

                // Count and done.
                _num_changes++;
                break;
            }
        }

        // Mark as seen.
        _seen.insert(fe);
        return "";
    }

} // namespace yask.
