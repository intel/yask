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

class ExampleStencil : public StencilOrderBase {

protected:

    // Generic time-varying spatial-3D grid.
    Grid grid;
    
    // Return a coefficient.  Note: This returns completely fabricated
    // values only for illustrative purposes; they have no mathematical
    // significance.
    virtual double coeff(int di, int dj, int dk) const {
        int sumAbs = abs(di) + abs(dj) + abs(dk);
        if (sumAbs == 0)
            return 0.9;
        double sumSq = double(di*di) + double(dj*dj) + double(dk*dk);
        double num = (sumAbs % 2 == 0) ? -0.8 : 0.8;
        return num / sumSq;
    }

    // Add additional contributions to v.
    virtual void valueAdd(GridValue& v, int t, int x, int y, int z) =0;
    
public:
    ExampleStencil(const string& name, StencilList& stencils, int order=2) :
        StencilOrderBase(name, stencils, order)
    {
        INIT_GRID_4D(grid, t, x, y, z);
    }

    // Define equation at t+1 based on values at t.
    virtual void define(const IntTuple& offsets) {
        GET_OFFSET(t);
        GET_OFFSET(x);
        GET_OFFSET(y);
        GET_OFFSET(z);

        // start with center value.
        GridValue v = coeff(0, 0, 0) * grid(t, x, y, z);

        // Add additional values.
        valueAdd(v, t, x, y, z);

        // define the value at t+1 to be equivalent to v.
        grid(t+1, x, y, z) == v;
    }
};

// Add values from x, y, and z axes.
class AxisStencil : public ExampleStencil {
protected:

    // Add additional contributions to v based on u(tm1, ...).
    virtual void valueAdd(GridValue& v, int t, int x, int y, int z)
    {
        for (int r = 1; r <= _order/2; r++) {

            // On the axes, assume values are isotropic, i.e., the same
            // for all points the same distance from the origin.
            double c = coeff(r, 0, 0);
            v += c * 
                (
                 // x-axis.
                 grid(t, x-r, y, z) +
                 grid(t, x+r, y, z) +
                 
                 // y-axis.
                 grid(t, x, y-r, z) +
                 grid(t, x, y+r, z) +
                 
                 // z-axis.
                 grid(t, x, y, z-r) +
                 grid(t, x, y, z+r)
                 );
        }
    }

public:
    AxisStencil(StencilList& stencils, int order=2) :
        ExampleStencil("3axis", stencils, order) { }
    AxisStencil(const string& name, StencilList& stencils, int order=2) :
        ExampleStencil(name, stencils, order) { }
};

REGISTER_STENCIL(AxisStencil);

// Add values from x-y, x-z, and y-z diagonals.
class DiagStencil : public AxisStencil {
protected:

    // Add additional contributions to v based on u(tm1, ...).
    virtual void valueAdd(GridValue& v, int t, int x, int y, int z)
    {
        // Get values from axes.
        AxisStencil::valueAdd(v, t, x, y, z);

        // Add values from diagonals.
        for (int r = 1; r <= _order/2; r++) {

            // x-y diagonal.
            v += coeff(-r, -r, 0) * grid(t, x-r, y-r, z);
            v += coeff(+r, -r, 0) * grid(t, x+r, y-r, z);
            v -= coeff(-r, +r, 0) * grid(t, x-r, y+r, z);
            v -= coeff(+r, +r, 0) * grid(t, x+r, y+r, z);

            // x-z diagonal.
            v += coeff(-r, 0, -r) * grid(t, x-r, y, z-r);
            v += coeff(+r, 0, +r) * grid(t, x+r, y, z+r);
            v -= coeff(-r, 0, +r) * grid(t, x-r, y, z+r);
            v -= coeff(+r, 0, -r) * grid(t, x+r, y, z-r);

            // y-z diagonal.
            v += coeff(0, -r, -r) * grid(t, x, y-r, z-r);
            v += coeff(0, +r, +r) * grid(t, x, y+r, z+r);
            v -= coeff(0, -r, +r) * grid(t, x, y-r, z+r);
            v -= coeff(0, +r, -r) * grid(t, x, y+r, z-r);
        }
    }

public:
    DiagStencil(StencilList& stencils, int order=2) :
        AxisStencil("6axis", stencils, order) { }
    DiagStencil(const string& name, StencilList& stencils, int order=2) :
        AxisStencil(name, stencils, order) { }
};

REGISTER_STENCIL(DiagStencil);

// Add values from x-y, x-z, and y-z planes not covered by axes or diagonals.
class PlaneStencil : public DiagStencil {
protected:
    
    // Add additional contributions to v based on u(tm1, ...).
    virtual void valueAdd(GridValue& v, int t, int x, int y, int z)
    {
        // Get values from axes and diagonals.
        DiagStencil::valueAdd(v, t, x, y, z);

        // Add remaining values on planes.
        for (int r = 1; r <= _order/2; r++) {
            for (int m = r+1; m <= _order/2; m++) {

                // x-y plane.
                v += coeff(-r, -m, 0) * grid(t, x-r, y-m, z);
                v += coeff(-m, -r, 0) * grid(t, x-m, y-r, z);
                v += coeff(+r, +m, 0) * grid(t, x+r, y+m, z);
                v += coeff(+m, +r, 0) * grid(t, x+m, y+r, z);
                v -= coeff(-r, +m, 0) * grid(t, x-r, y+m, z);
                v -= coeff(-m, +r, 0) * grid(t, x-m, y+r, z);
                v -= coeff(+r, -m, 0) * grid(t, x+r, y-m, z);
                v -= coeff(+m, -r, 0) * grid(t, x+m, y-r, z);

                // x-z plane.
                v += coeff(-r, 0, -m) * grid(t, x-r, y, z-m);
                v += coeff(-m, 0, -r) * grid(t, x-m, y, z-r);
                v += coeff(+r, 0, +m) * grid(t, x+r, y, z+m);
                v += coeff(+m, 0, +r) * grid(t, x+m, y, z+r);
                v -= coeff(-r, 0, +m) * grid(t, x-r, y, z+m);
                v -= coeff(-m, 0, +r) * grid(t, x-m, y, z+r);
                v -= coeff(+r, 0, -m) * grid(t, x+r, y, z-m);
                v -= coeff(+m, 0, -r) * grid(t, x+m, y, z-r);

                // y-z plane.
                v += coeff(0, -r, -m) * grid(t, x, y-r, z-m);
                v += coeff(0, -m, -r) * grid(t, x, y-m, z-r);
                v += coeff(0, +r, +m) * grid(t, x, y+r, z+m);
                v += coeff(0, +m, +r) * grid(t, x, y+m, z+r);
                v -= coeff(0, -r, +m) * grid(t, x, y-r, z+m);
                v -= coeff(0, -m, +r) * grid(t, x, y-m, z+r);
                v -= coeff(0, +r, -m) * grid(t, x, y+r, z-m);
                v -= coeff(0, +m, -r) * grid(t, x, y+m, z-r);
            }
        }
    }

public:
    PlaneStencil(StencilList& stencils, int order=2) :
        DiagStencil("3plane", stencils, order) { }
    PlaneStencil(const string& name, StencilList& stencils, int order=2) :
        DiagStencil(name, stencils, order) { }
};

REGISTER_STENCIL(PlaneStencil);

// Add values from rest of cube.
class CubeStencil : public PlaneStencil {
protected:

    // Add additional contributions to v based on u(tm1, ...).
    virtual void valueAdd(GridValue& v, int t, int x, int y, int z)
    {
        // Get values from planes.
        PlaneStencil::valueAdd(v, t, x, y, z);

        // Add values from rest of cube.
        for (int rx = 1; rx <= _order/2; rx++)
            for (int ry = 1; ry <= _order/2; ry++)
                for (int rz = 1; rz <= _order/2; rz++) {

                    // Each quadrant.
                    v += coeff(rx, ry, rz) * grid(t, x+rx, y+ry, z+rz);
                    v += coeff(rx, -ry, -rz) * grid(t, x+rx, y-ry, z-rz);
                    v -= coeff(rx, ry, -rz) * grid(t, x+rx, y+ry, z-rz);
                    v -= coeff(rx, -ry, rz) * grid(t, x+rx, y-ry, z+rz);
                    v += coeff(-rx, ry, rz) * grid(t, x-rx, y+ry, z+rz);
                    v += coeff(-rx, -ry, -rz) * grid(t, x-rx, y-ry, z-rz);
                    v -= coeff(-rx, ry, -rz) * grid(t, x-rx, y+ry, z-rz);
                    v -= coeff(-rx, -ry, rz) * grid(t, x-rx, y-ry, z+rz);
                }
    }

public:
    CubeStencil(StencilList& stencils, int order=2) :
        PlaneStencil("cube", stencils, order) { }
    CubeStencil(const string& name, StencilList& stencils, int order=2) :
        PlaneStencil(name, stencils, order) { }
};

REGISTER_STENCIL(CubeStencil);
