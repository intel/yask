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

// Implement various kinds of example 3D stencils.

#include "StencilBase.hpp"

class ExampleStencil : public StencilBase {

    bool _doAxes;               // calculate values on x, y, and z axes.
    bool _doDiags;              // calculate values on x-y, x-z, and y-z diagonals.
    bool _doPlanes;             // calculate values in x-y, x-z, and y-z planes.
    bool _doCubes;              // calculate values in each octant.

public:

    enum StencilShape { SS_NULL, SS_3AXIS, SS_9AXIS, SS_3PLANE, SS_CUBE };

    ExampleStencil(StencilShape ss, int order=2) :
        StencilBase(order),
        _doAxes(false), _doDiags(false), _doPlanes(false), _doCubes(false)
    {

        // set requisite parts correctly.
        // (fall-through is intentional in this switch statement
        // because parts are cumulative.)
        switch(ss) {
        case SS_CUBE:
            _doCubes = true;
        case SS_3PLANE:
            _doPlanes = true;
        case SS_9AXIS:
            _doDiags = true;
        case SS_3AXIS:
            _doAxes = true;
        default:
            assert("invalid stencil shape");
        }
    }

    // Return a coefficient.
    // Input values are relative to the current time t=0 and center of stencil i=0, j=0, k=0.
    virtual double coeff(int dt, int di, int dj, int dk) const {

        // Note: These are completely fabricated values only for testing purposes.
        // In a "real" stencil, these values could be read from tables.
        if (dt < 0)
            return 0.8f / double(dt);
        int sumAbs = abs(di) + abs(dj) + abs(dk);
        if (sumAbs == 0)
            return 0.9f;
        int sumSq = (di*di) + (dj*dj) + (dk*dk);
        double num = (sumAbs % 2 == 0) ? -0.8f : 0.8f;
        return num / double(sumSq);
    }

    // Calculate and return the value of stencil at u(t2, v0, i, j, k)
    // based on u(t0, v0, ...);
    virtual GridValue value(Grid5d& u, int t2, int t0, int v0, int i, int j, int k) const {
        assert(t2 >= t0);

        // just the current value?
        if (t2 == t0)
            return u(t0, v0, i, j, k);

        // not the current value; calc t2-1 based on t0.
        int t1 = t2 - 1;

        // start with center value.
        GridValue v = coeff(0, 0, 0, 0) * value(u, t1, t0, v0, i, j, k);

        // add values from x, y, and z axes.
        if (_doAxes) {
            for (int r = 1; r <= _order/2; r++) {

                // x-axis.
                v += coeff(0, -r, 0, 0) * value(u, t1, t0, v0, i-r, j, k);
                v += coeff(0, +r, 0, 0) * value(u, t1, t0, v0, i+r, j, k);

                // y-axis.
                v += coeff(0, 0, -r, 0) * value(u, t1, t0, v0, i, j-r, k);
                v += coeff(0, 0, +r, 0) * value(u, t1, t0, v0, i, j+r, k);

                // z-axis.
                v += coeff(0, 0, 0, -r) * value(u, t1, t0, v0, i, j, k-r);
                v += coeff(0, 0, 0, +r) * value(u, t1, t0, v0, i, j, k+r);
            }
        }

        // add values from x-y, x-z, and y-z diagonals.
        if (_doDiags) {
            for (int r = 1; r <= _order/2; r++) {

                // x-y diagonal.
                v += coeff(0, -r, -r, 0) * value(u, t1, t0, v0, i-r, j-r, k);
                v += coeff(0, +r, -r, 0) * value(u, t1, t0, v0, i+r, j-r, k);
                v -= coeff(0, -r, +r, 0) * value(u, t1, t0, v0, i-r, j+r, k);
                v -= coeff(0, +r, +r, 0) * value(u, t1, t0, v0, i+r, j+r, k);

                // x-z diagonal.
                v += coeff(0, -r, 0, -r) * value(u, t1, t0, v0, i-r, j, k-r);
                v += coeff(0, +r, 0, +r) * value(u, t1, t0, v0, i+r, j, k+r);
                v -= coeff(0, -r, 0, +r) * value(u, t1, t0, v0, i-r, j, k+r);
                v -= coeff(0, +r, 0, -r) * value(u, t1, t0, v0, i+r, j, k-r);

                // y-z diagonal.
                v += coeff(0, 0, -r, -r) * value(u, t1, t0, v0, i, j-r, k-r);
                v += coeff(0, 0, +r, +r) * value(u, t1, t0, v0, i, j+r, k+r);
                v -= coeff(0, 0, -r, +r) * value(u, t1, t0, v0, i, j-r, k+r);
                v -= coeff(0, 0, +r, -r) * value(u, t1, t0, v0, i, j+r, k-r);
            }
        }

        // add values from x-y, x-z, and y-z planes not covered by axes or diagonals.
        if (_doPlanes) {
            for (int r = 1; r <= _order/2; r++) {
                for (int m = r+1; m <= _order/2; m++) {

                    // x-y plane.
                    v += coeff(0, -r, -m, 0) * value(u, t1, t0, v0, i-r, j-m, k);
                    v += coeff(0, -m, -r, 0) * value(u, t1, t0, v0, i-m, j-r, k);
                    v += coeff(0, +r, +m, 0) * value(u, t1, t0, v0, i+r, j+m, k);
                    v += coeff(0, +m, +r, 0) * value(u, t1, t0, v0, i+m, j+r, k);
                    v -= coeff(0, -r, +m, 0) * value(u, t1, t0, v0, i-r, j+m, k);
                    v -= coeff(0, -m, +r, 0) * value(u, t1, t0, v0, i-m, j+r, k);
                    v -= coeff(0, +r, -m, 0) * value(u, t1, t0, v0, i+r, j-m, k);
                    v -= coeff(0, +m, -r, 0) * value(u, t1, t0, v0, i+m, j-r, k);

                    // x-z plane.
                    v += coeff(0, -r, 0, -m) * value(u, t1, t0, v0, i-r, j, k-m);
                    v += coeff(0, -m, 0, -r) * value(u, t1, t0, v0, i-m, j, k-r);
                    v += coeff(0, +r, 0, +m) * value(u, t1, t0, v0, i+r, j, k+m);
                    v += coeff(0, +m, 0, +r) * value(u, t1, t0, v0, i+m, j, k+r);
                    v -= coeff(0, -r, 0, +m) * value(u, t1, t0, v0, i-r, j, k+m);
                    v -= coeff(0, -m, 0, +r) * value(u, t1, t0, v0, i-m, j, k+r);
                    v -= coeff(0, +r, 0, -m) * value(u, t1, t0, v0, i+r, j, k-m);
                    v -= coeff(0, +m, 0, -r) * value(u, t1, t0, v0, i+m, j, k-r);

                    // y-z plane.
                    v += coeff(0, 0, -r, -m) * value(u, t1, t0, v0, i, j-r, k-m);
                    v += coeff(0, 0, -m, -r) * value(u, t1, t0, v0, i, j-m, k-r);
                    v += coeff(0, 0, +r, +m) * value(u, t1, t0, v0, i, j+r, k+m);
                    v += coeff(0, 0, +m, +r) * value(u, t1, t0, v0, i, j+m, k+r);
                    v -= coeff(0, 0, -r, +m) * value(u, t1, t0, v0, i, j-r, k+m);
                    v -= coeff(0, 0, -m, +r) * value(u, t1, t0, v0, i, j-m, k+r);
                    v -= coeff(0, 0, +r, -m) * value(u, t1, t0, v0, i, j+r, k-m);
                    v -= coeff(0, 0, +m, -r) * value(u, t1, t0, v0, i, j+m, k-r);
                }
            }
        }

        // add values outside of planes.
        if (_doCubes) {
            for (int rx = 1; rx <= _order/2; rx++)
                for (int ry = 1; ry <= _order/2; ry++)
                    for (int rz = 1; rz <= _order/2; rz++) {

                        // Each quadrant.
                        v += coeff(rx, ry, rz, 0) * value(u, t1, t0, v0, i+rx, j+ry, k+rz);
                        v += coeff(rx, -ry, -rz, 0) * value(u, t1, t0, v0, i+rx, j-ry, k-rz);
                        v -= coeff(rx, ry, -rz, 0) * value(u, t1, t0, v0, i+rx, j+ry, k-rz);
                        v -= coeff(rx, -ry, rz, 0) * value(u, t1, t0, v0, i+rx, j-ry, k+rz);
                        v += coeff(-rx, ry, rz, 0) * value(u, t1, t0, v0, i-rx, j+ry, k+rz);
                        v += coeff(-rx, -ry, -rz, 0) * value(u, t1, t0, v0, i-rx, j-ry, k-rz);
                        v -= coeff(-rx, ry, -rz, 0) * value(u, t1, t0, v0, i-rx, j+ry, k-rz);
                        v -= coeff(-rx, -ry, rz, 0) * value(u, t1, t0, v0, i-rx, j-ry, k+rz);
                    }
        }

        return v;
    }
};
