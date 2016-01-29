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

// Simple stencil that calculates an average of the points in a cube.
// Similar to MG_STENCIL_3D27PT from miniGhost benchmark.

#include "StencilBase.hpp"

class AveStencil : public StencilBase {

public:
    AveStencil(int order=2) :
        StencilBase(order) { }
    AveStencil(int order, shared_ptr<Dir> dir) : 
        StencilBase(order, dir) { }

    // Set direction.
    // Return true if successful.
    virtual bool setDir(shared_ptr<Dir> dir) {
        _dir = dir;
        return _dir->isNone() || _dir->isPos();  // only handling pos direction.
    }

    // Calculate and return the value of stencil at u(t2, v0, i, j, k)
    // based on u(t0, v0, ...);
    virtual GridValue value(Grid5d& u, int t2, int t0, int v0, int i, int j, int k) const {
        assert(t2 >= t0);
        assert(v0 >= 0);

        // just the current value?
        if (t2 == t0)
            return u(t0, v0, i, j, k);

        // not the current value; calc t2-1 based on t0.
        int t1 = t2 - 1;

        // starting points.
        int rx0, ry0, rz0;
        rx0 = ry0 = rz0 = -_order/2; // default: all points in stencil.
        if (_dir->isX())
            rx0 = _order/2; // only do leading x face.
        else if (_dir->isY())
             ry0 = _order/2; // only do leading y face.
        else if (_dir->isZ())
            rz0 = _order/2; // only do leading z face.
        assert(_dir->isPos());  // only handling pos direction.

        // calc requested parts.
        GridValue v;
        for (int rx = rx0; rx <= _order/2; rx++)
            for (int ry = ry0; ry <= _order/2; ry++)
                for (int rz = rz0; rz <= _order/2; rz++)
                    v += value(u, t1, t0, v0, i+rx, j+ry, k+rz);

        // adjust by average of whole cube, regardless of parts calculated.
        double n = double(_order + 1);
        v *= 1.f / (n*n*n);

        return v;
    }
};
