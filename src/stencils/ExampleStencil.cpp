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

// YASK stencil solution(s) in this file will be integrated into the YASK compiler utility.
#include "yask_compiler_utility_api.hpp"
using namespace std;
using namespace yask;

class AvePtsStencil : public yc_solution_with_radius_base {

protected:

    // Indices & dimensions.
    yc_index_node_ptr t = _node_factory.new_step_index("t");           // step in time dim.
    yc_index_node_ptr x = _node_factory.new_domain_index("x");         // spatial dim.
    yc_index_node_ptr y = _node_factory.new_domain_index("y");         // spatial dim.
    yc_index_node_ptr z = _node_factory.new_domain_index("z");         // spatial dim.

    // Vars.
    yc_grid_var A = yc_grid_var("A", get_solution(), { t, x, y, z }); // time-varying 3D grid.

    // Add additional points to expression v.
    // Returns number of points added.
    virtual int addPoints(yc_number_node_ptr& v) =0;

public:
    AvePtsStencil(const string& name, int radius) :
        yc_solution_with_radius_base(name, radius) { }

    // Define equation at t+1 based on values at t.
    virtual void define() {

        // start with center point.
        yc_number_node_ptr v = A(t, x, y, z);

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
    virtual int addPoints(yc_number_node_ptr& v)
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
    AxisStencil(int radius=4) :
        AvePtsStencil("3axis", radius) { }
    AxisStencil(const string& name, int radius=4) :
        AvePtsStencil(name, radius) { }
};

// Create an object of type 'AxisStencil',
// making it available in the YASK compiler utility via the
// '-stencil' commmand-line option or the 'stencil=' build option.
static AxisStencil AxisStencil_instance;

// Add points from x-y, x-z, and y-z diagonals.
class DiagStencil : public AxisStencil {
protected:

    // Add additional points to v.
    virtual int addPoints(yc_number_node_ptr& v)
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
    DiagStencil(int radius=4) :
        AxisStencil("3axis_with_diags", radius) { }
    DiagStencil(const string& name, int radius=4) :
        AxisStencil(name, radius) { }
};

// Create an object of type 'DiagStencil',
// making it available in the YASK compiler utility via the
// '-stencil' commmand-line option or the 'stencil=' build option.
static DiagStencil DiagStencil_instance;

// Add points from x-y, x-z, and y-z planes not covered by axes or diagonals.
class PlaneStencil : public DiagStencil {
protected:

    // Add additional points to v.
    virtual int addPoints(yc_number_node_ptr& v)
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
    PlaneStencil(int radius=3) :
        DiagStencil("3plane", radius) { }
    PlaneStencil(const string& name, int radius=3) :
        DiagStencil(name, radius) { }
};

// Create an object of type 'PlaneStencil',
// making it available in the YASK compiler utility via the
// '-stencil' commmand-line option or the 'stencil=' build option.
static PlaneStencil PlaneStencil_instance;

// Add points from rest of cube.
class CubeStencil : public PlaneStencil {
protected:

    // Add additional points to v.
    virtual int addPoints(yc_number_node_ptr& v)
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
    CubeStencil(int radius=2) :
        PlaneStencil("cube", radius) { }
    CubeStencil(const string& name, int radius=2) :
        PlaneStencil(name, radius) { }
};

// Create an object of type 'CubeStencil',
// making it available in the YASK compiler utility via the
// '-stencil' commmand-line option or the 'stencil=' build option.
static CubeStencil CubeStencil_instance;
