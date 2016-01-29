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

// Base class for calculaing the next stencil step.

#ifndef DIR_HPP
#define DIR_HPP

#include "Dir.hpp"

class StencilBase {
protected:
    int _order;         // stencil order (width not including center point).
    shared_ptr<Dir> _dir; // piping direction.

public:
    StencilBase(int order=2) :
        _order(order) {
        _dir = make_shared<NoDir>();
    }
    StencilBase(int order, shared_ptr<Dir> dir) :
        _order(order), _dir(dir) { }
    
    virtual ~StencilBase() {}

    // Set order.
    // Return true if successful.
    virtual bool setOrder(int order) {
        _order = order;
        return order % 2 == 0;  // support only even orders by default.
    }
    
    // Set direction for pipelining.
    // Return true if successful.
    virtual bool setDir(shared_ptr<Dir> dir) {
        _dir = dir;
        return dir->isNone(); // support only "no direction" by default.
    }
    
    // Calculate and return the value of stencil at u(timeWanted, varNum, i, j, k)
    // based on values at timeLastKnown.
    virtual GridValue value(Grid5d& u, int timeWanted, int timeLastKnown, int varNum,
                            int i, int j, int k) const =0;
};

#endif //  DIR_HPP
