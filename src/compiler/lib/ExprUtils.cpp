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

//////////// Expression utilities /////////////

#include "ExprUtils.hpp"

using namespace std;

namespace yask {

    void CombineVisitor::visit(CommutativeExpr* ce) {
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


    // If 'ep' has already been seen, just return true.
    // Else if 'ep' has a match, change pointer to that match, return true.
    // Else, return false.
    bool CseVisitor::findMatchTo(NumExprPtr& ep) {
#if DEBUG_CSE >= 1
        cout << "- checking '" << ep->makeStr() << "'@" << ep << endl;
#endif
        
        // Already visited this node?
        if (_seen.count(ep)) {
#if DEBUG_CSE >= 2
            cout << " - already seen '" << ep->makeStr() << "'@" << ep << endl;
#endif
            return true;
        }
        
        // Loop through nodes already seen.
        for (auto& oep : _seen) {
#if DEBUG_CSE >= 3
            cout << " - comparing '" << ep->makeStr() << "'@" << ep <<
                " to '" << oep->makeStr() << "'@" << oep << endl;
#endif
            
            // Match?
            if (ep->isSame(oep.get())) {
#if DEBUG_CSE >= 1
                cout << "  - found match: '" << ep->makeStr() << "'@" << ep <<
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
        cout << " - no match to " << ep->makeStr() << endl;
#endif
        _seen.insert(ep);
        return false;
    }
} // namespace yask.
