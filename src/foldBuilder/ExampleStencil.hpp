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

protected:
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

    // Add additional contributions to v based on u(tm1, ...).
    virtual void valueAdd(GridValue& v, TemporalGrid& u,
                          int tm1, int t0, int i, int j, int k) const =0;
    
public:
    ExampleStencil(int order=2) : StencilBase(order) { }

    // Calculate and return the value of stencil at u(tW, i, j, k)
    // based on u(t0, ...);
    virtual GridValue value(TemporalGrid& u, int tW, int t0, int i, int j, int k) const {

        // if the wanted time (tW) is <= the last known time (t0),
        // we are done--just return the known value from the grid.
        if (tW <= t0)
            return u(tW, i, j, k);

        // not known; calc using values at tW-1 and tW-2 recursively.
        int tm1 = tW - 1;  // one timestep ago.
        
        // start with center value.
        GridValue v = coeff(0, 0, 0, 0) * value(u, tm1, t0, i, j, k);

        // Add additional values.
        valueAdd(v, u, tm1, t0, i, j, k);

        return v;
    }
};

// Add values from x, y, and z axes.
class AxisStencil : public ExampleStencil {
protected:

    // Add additional contributions to v based on u(tm1, ...).
    virtual void valueAdd(GridValue& v, TemporalGrid& u,
                          int tm1, int t0, int i, int j, int k) const
    {
        for (int r = 1; r <= _order/2; r++) {

            // On the axes, assume values are isotropic, i.e., the same
            // for all points the same distance from the origin.
            double c = coeff(0, +r, 0, 0);
            v += c * 
                (
                 // x-axis.
                 value(u, tm1, t0, i-r, j, k) +
                 value(u, tm1, t0, i+r, j, k) +
                 
                 // y-axis.
                 value(u, tm1, t0, i, j-r, k) +
                 value(u, tm1, t0, i, j+r, k) +
                 
                 // z-axis.
                 value(u, tm1, t0, i, j, k-r) +
                 value(u, tm1, t0, i, j, k+r)
                 );
        }
    }

public:
    AxisStencil(int order=2) : ExampleStencil(order) { }

};

// Add values from x-y, x-z, and y-z diagonals.
class DiagStencil : public AxisStencil {
protected:

    // Add additional contributions to v based on u(tm1, ...).
    virtual void valueAdd(GridValue& v, TemporalGrid& u,
                          int tm1, int t0, int i, int j, int k) const
    {
        AxisStencil::valueAdd(v, u, tm1, t0, i, j, k);
        
        for (int r = 1; r <= _order/2; r++) {

            // x-y diagonal.
            v += coeff(0, -r, -r, 0) * value(u, tm1, t0, i-r, j-r, k);
            v += coeff(0, +r, -r, 0) * value(u, tm1, t0, i+r, j-r, k);
            v -= coeff(0, -r, +r, 0) * value(u, tm1, t0, i-r, j+r, k);
            v -= coeff(0, +r, +r, 0) * value(u, tm1, t0, i+r, j+r, k);

            // x-z diagonal.
            v += coeff(0, -r, 0, -r) * value(u, tm1, t0, i-r, j, k-r);
            v += coeff(0, +r, 0, +r) * value(u, tm1, t0, i+r, j, k+r);
            v -= coeff(0, -r, 0, +r) * value(u, tm1, t0, i-r, j, k+r);
            v -= coeff(0, +r, 0, -r) * value(u, tm1, t0, i+r, j, k-r);

            // y-z diagonal.
            v += coeff(0, 0, -r, -r) * value(u, tm1, t0, i, j-r, k-r);
            v += coeff(0, 0, +r, +r) * value(u, tm1, t0, i, j+r, k+r);
            v -= coeff(0, 0, -r, +r) * value(u, tm1, t0, i, j-r, k+r);
            v -= coeff(0, 0, +r, -r) * value(u, tm1, t0, i, j+r, k-r);
        }
    }

public:
    DiagStencil(int order=2) : AxisStencil(order) { }

};

// Add values from x-y, x-z, and y-z planes not covered by axes or diagonals.
class PlaneStencil : public DiagStencil {
protected:
    
    // Add additional contributions to v based on u(tm1, ...).
    virtual void valueAdd(GridValue& v, TemporalGrid& u,
                          int tm1, int t0, int i, int j, int k) const
    {
        DiagStencil::valueAdd(v, u, tm1, t0, i, j, k);
        
        for (int r = 1; r <= _order/2; r++) {
            for (int m = r+1; m <= _order/2; m++) {

                // x-y plane.
                v += coeff(0, -r, -m, 0) * value(u, tm1, t0, i-r, j-m, k);
                v += coeff(0, -m, -r, 0) * value(u, tm1, t0, i-m, j-r, k);
                v += coeff(0, +r, +m, 0) * value(u, tm1, t0, i+r, j+m, k);
                v += coeff(0, +m, +r, 0) * value(u, tm1, t0, i+m, j+r, k);
                v -= coeff(0, -r, +m, 0) * value(u, tm1, t0, i-r, j+m, k);
                v -= coeff(0, -m, +r, 0) * value(u, tm1, t0, i-m, j+r, k);
                v -= coeff(0, +r, -m, 0) * value(u, tm1, t0, i+r, j-m, k);
                v -= coeff(0, +m, -r, 0) * value(u, tm1, t0, i+m, j-r, k);

                // x-z plane.
                v += coeff(0, -r, 0, -m) * value(u, tm1, t0, i-r, j, k-m);
                v += coeff(0, -m, 0, -r) * value(u, tm1, t0, i-m, j, k-r);
                v += coeff(0, +r, 0, +m) * value(u, tm1, t0, i+r, j, k+m);
                v += coeff(0, +m, 0, +r) * value(u, tm1, t0, i+m, j, k+r);
                v -= coeff(0, -r, 0, +m) * value(u, tm1, t0, i-r, j, k+m);
                v -= coeff(0, -m, 0, +r) * value(u, tm1, t0, i-m, j, k+r);
                v -= coeff(0, +r, 0, -m) * value(u, tm1, t0, i+r, j, k-m);
                v -= coeff(0, +m, 0, -r) * value(u, tm1, t0, i+m, j, k-r);

                // y-z plane.
                v += coeff(0, 0, -r, -m) * value(u, tm1, t0, i, j-r, k-m);
                v += coeff(0, 0, -m, -r) * value(u, tm1, t0, i, j-m, k-r);
                v += coeff(0, 0, +r, +m) * value(u, tm1, t0, i, j+r, k+m);
                v += coeff(0, 0, +m, +r) * value(u, tm1, t0, i, j+m, k+r);
                v -= coeff(0, 0, -r, +m) * value(u, tm1, t0, i, j-r, k+m);
                v -= coeff(0, 0, -m, +r) * value(u, tm1, t0, i, j-m, k+r);
                v -= coeff(0, 0, +r, -m) * value(u, tm1, t0, i, j+r, k-m);
                v -= coeff(0, 0, +m, -r) * value(u, tm1, t0, i, j+m, k-r);
            }
        }
    }

public:
    PlaneStencil(int order=2) : DiagStencil(order) { }

};

// Add values from rest of cube.
class CubeStencil : public PlaneStencil {
protected:

    // Add additional contributions to v based on u(tm1, ...).
    virtual void valueAdd(GridValue& v, TemporalGrid& u,
                          int tm1, int t0, int i, int j, int k) const
    {
        PlaneStencil::valueAdd(v, u, tm1, t0, i, j, k);
        
        for (int rx = 1; rx <= _order/2; rx++)
            for (int ry = 1; ry <= _order/2; ry++)
                for (int rz = 1; rz <= _order/2; rz++) {

                    // Each quadrant.
                    v += coeff(rx, ry, rz, 0) * value(u, tm1, t0, i+rx, j+ry, k+rz);
                    v += coeff(rx, -ry, -rz, 0) * value(u, tm1, t0, i+rx, j-ry, k-rz);
                    v -= coeff(rx, ry, -rz, 0) * value(u, tm1, t0, i+rx, j+ry, k-rz);
                    v -= coeff(rx, -ry, rz, 0) * value(u, tm1, t0, i+rx, j-ry, k+rz);
                    v += coeff(-rx, ry, rz, 0) * value(u, tm1, t0, i-rx, j+ry, k+rz);
                    v += coeff(-rx, -ry, -rz, 0) * value(u, tm1, t0, i-rx, j-ry, k-rz);
                    v -= coeff(-rx, ry, -rz, 0) * value(u, tm1, t0, i-rx, j+ry, k-rz);
                    v -= coeff(-rx, -ry, rz, 0) * value(u, tm1, t0, i-rx, j-ry, k+rz);
                }
    }

public:
    CubeStencil(int order=2) : PlaneStencil(order) { }

};
