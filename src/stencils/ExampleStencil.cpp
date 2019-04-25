/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2019, Intel Corporation

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

// Implement various example symmetric 3D stencil shapes that read and
// write from only one 3D grid variable.
// All these stencils compute the average of the points read a la the
// heat-dissipation kernels in the miniGhost benchmark.

#include "Soln.hpp"

class AvePtsStencil : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_DOMAIN_INDEX(z);         // spatial dim.

    // Vars.
    MAKE_GRID(A, t, x, y, z); // time-varying 3D grid.

    // Add additional points to expression v.
    // Returns number of points added.
    virtual int addPoints(GridValue& v) =0;

public:
    AvePtsStencil(const string& name, StencilList& stencils, int radius) :
        StencilRadiusBase(name, stencils, radius) { }

    // Define equation at t+1 based on values at t.
    virtual void define() {

        // start with center point.
        GridValue v = A(t, x, y, z);

        // Add additional points from derived class.
        int pts = 1 + addPoints(v);

        // Average.
        if (pts > 1)
            v *= 1.0 / pts;
        
        // Define the value at t+1 to be equivalent to v.
        A(t+1, x, y, z) EQUALS v;
    }
};

// Add points from x, y, and z axes.
class AxisStencil : public AvePtsStencil {
protected:

    // Add additional points to expression v.
    virtual int addPoints(GridValue& v)
    {
        int pts = 0;
        for (int r = 1; r <= _radius; r++) {

            v +=
                // x-axis.
                A(t, x-r, y, z) +
                A(t, x+r, y, z) +

                // y-axis.
                A(t, x, y-r, z) +
                A(t, x, y+r, z) +

                // z-axis.
                A(t, x, y, z-r) +
                A(t, x, y, z+r);
            pts += 3 * 2;
        }
        return pts;
    }

public:
    AxisStencil(StencilList& stencils, int radius=4) :
        AvePtsStencil("3axis", stencils, radius) { }
    AxisStencil(const string& name, StencilList& stencils, int radius=4) :
        AvePtsStencil(name, stencils, radius) { }
};

REGISTER_STENCIL(AxisStencil);

// Add points from x-y, x-z, and y-z diagonals.
class DiagStencil : public AxisStencil {
protected:

    // Add additional points to v.
    virtual int addPoints(GridValue& v)
    {
        // Get points from axes.
        int pts = AxisStencil::addPoints(v);

        // Add points from diagonals.
        for (int r = 1; r <= _radius; r++) {

            v += 
                // x-y diagonal.
                A(t, x-r, y-r, z) + 
                A(t, x+r, y-r, z) +
                A(t, x-r, y+r, z) +
                A(t, x+r, y+r, z) +

                // x-z diagonal.
                A(t, x-r, y, z-r) +
                A(t, x+r, y, z+r) +
                A(t, x-r, y, z+r) +
                A(t, x+r, y, z-r) +

                // y-z diagonal.
                A(t, x, y-r, z-r) +
                A(t, x, y+r, z+r) +
                A(t, x, y-r, z+r) +
                A(t, x, y+r, z-r);
            pts += 3 * 4;
        }
        return pts;
    }

public:
    DiagStencil(StencilList& stencils, int radius=4) :
        AxisStencil("3axis_with_diags", stencils, radius) { }
    DiagStencil(const string& name, StencilList& stencils, int radius=4) :
        AxisStencil(name, stencils, radius) { }
};

REGISTER_STENCIL(DiagStencil);

// Add points from x-y, x-z, and y-z planes not covered by axes or diagonals.
class PlaneStencil : public DiagStencil {
protected:

    // Add additional points to v.
    virtual int addPoints(GridValue& v)
    {
        // Get points from axes and diagonals.
        int pts = DiagStencil::addPoints(v);

        // Add remaining points on planes.
        for (int r = 1; r <= _radius; r++) {
            for (int m = r+1; m <= _radius; m++) {

                v += 
                    // x-y plane.
                    A(t, x-r, y-m, z) +
                    A(t, x-m, y-r, z) +
                    A(t, x+r, y+m, z) +
                    A(t, x+m, y+r, z) +
                    A(t, x-r, y+m, z) +
                    A(t, x-m, y+r, z) +
                    A(t, x+r, y-m, z) +
                    A(t, x+m, y-r, z) +

                    // x-z plane.
                    A(t, x-r, y, z-m) +
                    A(t, x-m, y, z-r) +
                    A(t, x+r, y, z+m) +
                    A(t, x+m, y, z+r) +
                    A(t, x-r, y, z+m) +
                    A(t, x-m, y, z+r) +
                    A(t, x+r, y, z-m) +
                    A(t, x+m, y, z-r) +

                    // y-z plane.
                    A(t, x, y-r, z-m) +
                    A(t, x, y-m, z-r) +
                    A(t, x, y+r, z+m) +
                    A(t, x, y+m, z+r) +
                    A(t, x, y-r, z+m) +
                    A(t, x, y-m, z+r) +
                    A(t, x, y+r, z-m) +
                    A(t, x, y+m, z-r);
                pts += 3 * 8;
            }
        }
        return pts;
    }

public:
    PlaneStencil(StencilList& stencils, int radius=3) :
        DiagStencil("3plane", stencils, radius) { }
    PlaneStencil(const string& name, StencilList& stencils, int radius=3) :
        DiagStencil(name, stencils, radius) { }
};

REGISTER_STENCIL(PlaneStencil);

// Add points from rest of cube.
class CubeStencil : public PlaneStencil {
protected:

    // Add additional points to v.
    virtual int addPoints(GridValue& v)
    {
        // Get points from planes.
        int pts = PlaneStencil::addPoints(v);

        // Add points from rest of cube.
        for (int rx = 1; rx <= _radius; rx++)
            for (int ry = 1; ry <= _radius; ry++)
                for (int rz = 1; rz <= _radius; rz++) {

                    v +=
                        // Each quadrant.
                        A(t, x+rx, y+ry, z+rz) +
                        A(t, x+rx, y-ry, z-rz) +
                        A(t, x+rx, y+ry, z-rz) +
                        A(t, x+rx, y-ry, z+rz) +
                        A(t, x-rx, y+ry, z+rz) +
                        A(t, x-rx, y-ry, z-rz) +
                        A(t, x-rx, y+ry, z-rz) +
                        A(t, x-rx, y-ry, z+rz);
                    pts += 8;
                }
        return pts;
    }

public:
    CubeStencil(StencilList& stencils, int radius=2) :
        PlaneStencil("cube", stencils, radius) { }
    CubeStencil(const string& name, StencilList& stencils, int radius=2) :
        PlaneStencil(name, stencils, radius) { }
};

REGISTER_STENCIL(CubeStencil);
