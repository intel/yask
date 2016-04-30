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

class Iso3dfdStencil : public StencilBase {

    const int maxOrder = 16;
    
protected:
    double* _coeff;             // stencil coefficients.
    double _dxyz;               // dx, dy, dz factor.
    bool _deferCoeff;           // look up coefficients later.

    Grid pressure;              // time-varying 3D pressure grid.
    Grid vel;                   // constant 3D vel grid.

    Param coefficients;             // coefficients if they are not hard-coded.
    
public:

    Iso3dfdStencil(int order=8, double dxyz=50.0) :
        StencilBase(order), _coeff(0), _dxyz(dxyz), _deferCoeff(false)
    {
        INIT_GRID_4D(pressure, t, x, y, z);
        INIT_GRID_3D(vel, x, y, z);
        INIT_PARAM_1D(coefficients, r, order / 2 + 1);
    }

    virtual ~Iso3dfdStencil() {
        if (_coeff)
            delete[] _coeff;
    }

    // Set coefficient deferral policy.
    // false: calculate const coefficients during code generation.
    // true: generate code to calculate coefficicents at runtine.
    virtual void setDeferCoeff(bool defer) {
        _deferCoeff = defer;
    }

    // Set order.
    // Return true if successful.
    virtual bool setOrder(int order) {
        _order = order;

        // support only certain orders.
        // Only order 8 and 16 use accurate coefficients; others are for debug.
        if (order % 2 == 1 || order > maxOrder)
            return false;

        // set the coefficients.
        if (_coeff)
            delete[] _coeff;
        int n = order / 2 + 1;
        _coeff = new double[n];

        double coeff8[] = {
            -2.847222222,
            +1.6,
            -0.2,
            +2.53968e-2,
            -1.785714e-3};

        double coeff16[] = {
            -3.0548446,
            +1.7777778,
            -3.1111111e-1,
            +7.572087e-2,
            -1.76767677e-2,
            +3.480962e-3,
            -5.180005e-4,
            +5.074287e-5,
            -2.42812e-6};

        double *coeffN = (order == 8) ? coeff8 : coeff16;

        // copy the coefficients and adjust using dxyz factor.
        double dxyz2 = _dxyz * _dxyz;
        for (int i = 0; i < n; i++)
            _coeff[i] = coeffN[i] / dxyz2;
        _coeff[0] *= 3.0;

        return true;
    }

    // Get an expression for coefficient at radius r.
    virtual GridValue coeff(int r) {
        GridValue v;
        
        // if coefficients are deferred, load from parameter.
        if (_deferCoeff)
            v = coefficients(r);

        // if coefficients are not deferred, set from constant.
        else
            v = constGridValue(_coeff[r]);

        return v;
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
        for (int r = 1; r <= _order/2; r++) {

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
