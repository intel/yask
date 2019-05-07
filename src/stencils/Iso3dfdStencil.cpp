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

// Implement isotropic 3D finite-difference (FD) stencil, nth-order accurate in
// space (where n = 2 * radius) and 2nd-order accurate in time.
// See https://software.intel.com/en-us/articles/eight-optimizations-for-3-dimensional-finite-difference-3dfd-code-with-an-isotropic-iso.

// YASK stencil solution(s) in this file will be integrated into the YASK compiler utility.
#include "yask_compiler_utility_api.hpp"
using namespace std;
using namespace yask;

class Iso3dfdStencil : public yc_solution_with_radius_base {

protected:

    // Indices & dimensions.
    yc_index_node_ptr t = new_step_index("t");           // step in time dim.
    yc_index_node_ptr x = new_domain_index("x");         // spatial dim.
    yc_index_node_ptr y = new_domain_index("y");         // spatial dim.
    yc_index_node_ptr z = new_domain_index("z");         // spatial dim.

    // Grid vars.
    yc_grid_var pressure =
        yc_grid_var("pressure", get_soln(), { t, x, y, z }); // time-varying 3D pressure grid.
    yc_grid_var vel =
        yc_grid_var("vel", get_soln(), { x, y, z }); // constant 3D vel grid (c(x,y,z)^2 * delta_t^2).

public:

    // For this stencil, the 'radius' is the number of FD coefficients on
    // either side of center in each spatial dimension.  For example,
    // radius=8 implements a 16th-order accurate FD stencil.  
    // The accuracy in time is fixed at 2nd order.
    Iso3dfdStencil(string suffix="", int radius=8) :
        yc_solution_with_radius_base("iso3dfd" + suffix, radius) { }
    virtual ~Iso3dfdStencil() { }

    // Define RHS expression for pressure at t+1 based on values from vel and pressure at t.
    virtual yc_number_node_ptr get_next_p() {

        // yc_grid_var spacing.
        // In this implementation, it's a constant.
        // Could make this a YASK variable to allow setting at run-time.
        double delta_xyz = 50.0;
        double d2 = delta_xyz * delta_xyz;
        
        // Spatial FD coefficients for 2nd derivative.
        auto coeff = get_center_fd_coefficients(2, _radius);
        size_t c0i = _radius;      // index of center sample.

        for (size_t i = 0; i < coeff.size(); i++) {

            // Need 3 copies of center sample for x, y, and z FDs.
            if (i == c0i)
                coeff[i] *= 3.0;

            // Divide each by delta_xyz^2.
            coeff[i] /= d2;
        }

        // Calculate FDx + FDy + FDz.
        // Start with center value multiplied by coeff 0.
        auto fd_sum = pressure(t, x, y, z) * coeff[c0i];

        // Add values from x, y, and z axes multiplied by the
        // coeff for the given radius.
        for (int r = 1; r <= _radius; r++) {

            // Add values from axes at radius r.
            fd_sum += (
                       // x-axis.
                       pressure(t, x-r, y, z) +
                       pressure(t, x+r, y, z) +

                       // y-axis.
                       pressure(t, x, y-r, z) +
                       pressure(t, x, y+r, z) +

                       // z-axis.
                       pressure(t, x, y, z-r) +
                       pressure(t, x, y, z+r)

                       ) * coeff[c0i + r]; // R & L coeffs are identical.
        }

        // Temporal FD coefficients.
        // For this implementation, just check the known values to
        // simplify the solution.
        // But we could parameterize by accuracy-order in time as well.
        int torder = 2;
        auto tcoeff = get_forward_fd_coefficients(2, torder);
        assert(tcoeff[0] == 1.0);  // pressure(t+1).
        assert(tcoeff[1] == -2.0); // -2 * pressure(t+1).
        assert(tcoeff[2] == 1.0);  // pressure(t-1).

        // Wave equation is:
        // 2nd time derivative(p) = c^2 * laplacian(p).
        // See https://en.wikipedia.org/wiki/Wave_equation.
        
        // So, wave equation with FD approximations is:
        // (p(t+1) - 2 * p(t) + p(t-1)) / delta_t^2 = c^2 * fd_sum.

        // Solve wave equation for p(t+1):
        // p(t+1) = 2 * p(t) - p(t-1) + c^2 * fd_sum * delta_t^2.

        // Let vel = c^2 * delta_t^2 for each grid point.
        auto next_p = (2.0 * pressure(t, x, y, z)) -
            pressure(t-1, x, y, z) + (fd_sum * vel(x, y, z));

        return next_p;
    }

    // Define equation for pressure at t+1 based on values from vel and pressure at t.
    virtual void define() {

        // Get equation for RHS.
        auto next_p = get_next_p();

        // Define the value at t+1 to be equal to next_p.
        // Since this implements the finite-difference method, this
        // is actually an approximation.
        pressure(t+1, x, y, z) EQUALS next_p;
    }
};

// Create an object of type 'Iso3dfdStencil',
// making it available in the YASK compiler utility via the
// '-stencil' commmand-line option or the 'stencil=' build option.
static Iso3dfdStencil Iso3dfdStencil_instance;

// Add a sponge absorption factor.
class Iso3dfdSpongeStencil : public Iso3dfdStencil {
protected:

    // Sponge coefficients.
    // In practice, the interior values would be set to 1.0,
    // and values nearer the boundary would be set to values
    // increasingly approaching 0.0.
    yc_grid_var cr_x = yc_grid_var("cr_x", get_soln(), { x });
    yc_grid_var cr_y = yc_grid_var("cr_y", get_soln(), { y });
    yc_grid_var cr_z = yc_grid_var("cr_z", get_soln(), { z });

public:
    Iso3dfdSpongeStencil(int radius=8) :
        Iso3dfdStencil("_sponge", radius) { }
    virtual ~Iso3dfdSpongeStencil() { }

    // Define equation for pressure at t+1 based on values from vel and pressure at t.
    virtual void define() {

        // Get equation for RHS.
        auto next_p = get_next_p();

        // Apply sponge absorption.
        next_p *= cr_x(x) * cr_y(y) * cr_z(z);

        // Define the value at t+1 to be equal to next_p.
        pressure(t+1, x, y, z) EQUALS next_p;
    }
};

// Create an object of type 'Iso3dfdSpongeStencil',
// making it available in the YASK compiler utility via the
// '-stencil' commmand-line option or the 'stencil=' build option.
static Iso3dfdSpongeStencil Iso3dfdSpongeStencil_instance;
