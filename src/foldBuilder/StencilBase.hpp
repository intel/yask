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

// Base class for calculating a future stencil value.

#ifndef STENCIL_BASE_HPP
#define STENCIL_BASE_HPP

#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::ostringstream

using namespace std;

// Use this macro for creating a string to insert any C++ code
// into an expression.
// The 1st arg must be the LHS of an assignment statement.
// The 2nd arg must evaluate to a real (float or double) expression,
// but it must NOT include access to a grid.
// The code string is constructed as if writing to an ostream,
// so '<<' operators may be used to evaluate local variables.
// Floating-point variables will be printed w/o loss of precision.
// The code may access the following:
// - Any parameter to the 'calc_stencil_vector' generated function,
//   including fields of the user-defined 'context' object.
// - A variable within the global or current namespace where it will be used.
// - A local variable in the 'value' method; in this case, the value
//   of the local var must be evaluated and inserted in the expr.
// Example code:
//   GridValue v;
//   SET_VALUE_FROM_EXPR(v =,"context.temp * " << 0.2);
//   SET_VALUE_FROM_EXPR(v +=, "context.coeff[" << r << "]");
// This example would generate the following partial expression (when r=9):
//   (context.temp * 2.00000000000000000e-01) + (context.coeff[9])
#define SET_VALUE_FROM_EXPR(lhs, rhs) do {              \
        ostringstream oss;                              \
        oss << setprecision(17) << scientific << v;     \
        oss << "(" << rhs << ")";                       \
        lhs  make_shared<CodeExpr>(oss.str());          \
    } while(0)

class StencilBase {
protected:
    int _order;         // stencil order (width not including center point).

public:
    StencilBase(int order=2) :
        _order(order) { }
    
    virtual ~StencilBase() {}

    // Set order.
    // Return true if successful.
    virtual bool setOrder(int order) {
        _order = order;
        return order % 2 == 0;  // support only even orders by default.
    }
    
    // Calculate and return the value of stencil at u(timeWanted, varNum, i, j, k)
    // based on values at timeLastKnown.
    virtual GridValue value(Grid5d& u, int timeWanted, int timeLastKnown, int varNum,
                            int i, int j, int k) const =0;
};

#endif

