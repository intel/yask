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

// Implement isotropic 3D finite-difference stencil.
// 2nd-order in time.

#include "StencilBase.hpp"

class Iso3dfdStencil : public StencilRadiusBase {

protected:
    Grid pressure;              // time-varying 3D pressure grid.
    Grid vel;                   // constant 3D vel grid.
    Param coeff;                // coefficients.
    
public:

    Iso3dfdStencil(StencilList& stencils, int radius=8) :
        StencilRadiusBase("iso3dfd", stencils, radius)
    {
        INIT_GRID_4D(pressure, t, x, y, z);
        INIT_GRID_3D(vel, x, y, z);
        INIT_PARAM_1D(coeff, r, radius + 1); // c0, c1 .. c<radius>.
    }
    virtual ~Iso3dfdStencil() { }

    // Set radius.
    // Return true if successful.
    virtual bool setRadius(int radius) {
        if (!StencilRadiusBase::setRadius(radius))
            return false;
        coeff.setVal("r", radius + 1); // change dimension of coeff.
        return true;
    }

    // Define equation for pressure at t+1 based on values from vel and pressure at t.
    virtual void define(const IntTuple& offsets) {
        GET_OFFSET(t);
        GET_OFFSET(x);
        GET_OFFSET(y);
        GET_OFFSET(z);

        // start with center value multiplied by coeff 0.
        GridValue v = pressure(t, x, y, z) * coeff(0);

        // add values from x, y, and z axes multiplied by the
        // coeff for the given radius.
        for (int r = 1; r <= _radius; r++) {

            // Add values from axes at radius r.
            v += (
                  // x-axis.
                  pressure(t, x-r, y, z) +
                  pressure(t, x+r, y, z) +
                
                  // y-axis.
                  pressure(t, x, y-r, z) +
                  pressure(t, x, y+r, z) +
                
                  // z-axis.
                  pressure(t, x, y, z-r) +
                  pressure(t, x, y, z+r)

                  ) * coeff(r);
        }

        // finish equation, including t-1 and velocity components.
        v = (2.0 * pressure(t, x, y, z))
            - pressure(t-1, x, y, z) // subtract pressure from t-1.
            + (v * vel(x, y, z));       // add v * velocity.

        // define the value at t+1 to be equivalent to v.
        pressure(t+1, x, y, z) == v;
    }
};

REGISTER_STENCIL(Iso3dfdStencil);
