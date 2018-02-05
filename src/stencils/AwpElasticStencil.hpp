/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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

// Stencil equations for AWP elastic* numerics.
// *This version does not contain the time-varying attenuation memory grids
// or the related attenuation constant grids.
// This version also contains some experimental code for calculating the
// free-surface boundary values.
// http://hpgeoc.sdsc.edu/AWPODC
// http://www.sdsc.edu/News%20Items/PR20160209_earthquake_center.html

// Set the following macro to use a sponge grid instead of 3 sponge arrays.
//#define FULL_SPONGE_GRID

// Set the following macro to calculate free-surface boundary values.
// This feature is currently under development.
//#define DO_SURFACE

#include "Soln.hpp"

class AwpElasticStencil : public StencilBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_DOMAIN_INDEX(z);         // spatial dim.

    // Time-varying 3D-spatial velocity grids.
    MAKE_GRID(vel_x, t, x, y, z);
    MAKE_GRID(vel_y, t, x, y, z);
    MAKE_GRID(vel_z, t, x, y, z);
        
    // Time-varying 3D-spatial Stress grids.
    MAKE_GRID(stress_xx, t, x, y, z);
    MAKE_GRID(stress_yy, t, x, y, z);
    MAKE_GRID(stress_zz, t, x, y, z);
    MAKE_GRID(stress_xy, t, x, y, z);
    MAKE_GRID(stress_xz, t, x, y, z);
    MAKE_GRID(stress_yz, t, x, y, z);

    // 3D-spatial Lame' coefficients.
    MAKE_GRID(lambda, x, y, z);
    MAKE_GRID(rho, x, y, z);
    MAKE_GRID(mu, x, y, z);

    // Sponge coefficients.
    // (Most of these will be 1.0.)
#ifdef FULL_SPONGE_GRID
    MAKE_GRID(sponge, x, y, z);
#else
    MAKE_ARRAY(cr_x, x);
    MAKE_ARRAY(cr_y, y);
    MAKE_ARRAY(cr_z, z);
#endif

    // Spatial FD coefficients.
    const double c1 = 9.0/8.0;
    const double c2 = -1.0/24.0;

    // Physical dimensions in time and space.
    MAKE_SCALAR(delta_t);
    MAKE_SCALAR(h);

public:

    AwpElasticStencil(StencilList& stencils) :
        StencilBase("awp_elastic", stencils) { }

    // Adjustment for sponge layer.
    void adjust_for_sponge(GridValue& val) {

#ifdef FULL_SPONGE_GRID
        val *= sponge(x, y, z);
#else
        val *= cr_x(x) * cr_y(y) * cr_z(z);
#endif        
    }

    // Velocity-grid define functions.  For each D in x, y, z, define vel_D
    // at t+1 based on vel_x at t and stress grids at t.  Note that the t,
    // x, y, z parameters are integer grid indices, not actual offsets in
    // time or space, so half-steps due to staggered grids are adjusted
    // appropriately.

    void define_vel_x(Condition at_last_z) {
        GridValue rho_val = (rho(x, y,   z  ) +
                             rho(x, y-1, z  ) +
                             rho(x, y,   z-1) +
                             rho(x, y-1, z-1)) * (1.0 / 4.0);
        GridValue d_val =
            c1 * (stress_xx(t, x,   y,   z  ) - stress_xx(t, x-1, y,   z  )) +
            c2 * (stress_xx(t, x+1, y,   z  ) - stress_xx(t, x-2, y,   z  )) +
            c1 * (stress_xy(t, x,   y,   z  ) - stress_xy(t, x,   y-1, z  )) +
            c2 * (stress_xy(t, x,   y+1, z  ) - stress_xy(t, x,   y-2, z  )) +
            c1 * (stress_xz(t, x,   y,   z  ) - stress_xz(t, x,   y,   z-1)) +
            c2 * (stress_xz(t, x,   y,   z+1) - stress_xz(t, x,   y,   z-2));
        GridValue next_vel_x = vel_x(t, x, y, z) + (delta_t / (h * rho_val)) * d_val;
        adjust_for_sponge(next_vel_x);

        // define the value at t+1.
        vel_x(t+1, x, y, z) EQUALS next_vel_x;
    }
    void define_vel_y(Condition at_last_z) {
        GridValue rho_val = (rho(x,   y, z  ) +
                             rho(x+1, y, z  ) +
                             rho(x,   y, z-1) +
                             rho(x+1, y, z-1)) * (1.0 / 4.0);
        GridValue d_val =
            c1 * (stress_xy(t, x+1, y,   z  ) - stress_xy(t, x,   y,   z  )) +
            c2 * (stress_xy(t, x+2, y,   z  ) - stress_xy(t, x-1, y,   z  )) +
            c1 * (stress_yy(t, x,   y+1, z  ) - stress_yy(t, x,   y,   z  )) +
            c2 * (stress_yy(t, x,   y+2, z  ) - stress_yy(t, x,   y-1, z  )) +
            c1 * (stress_yz(t, x,   y,   z  ) - stress_yz(t, x,   y,   z-1)) +
            c2 * (stress_yz(t, x,   y,   z+1) - stress_yz(t, x,   y,   z-2));
        GridValue next_vel_y = vel_y(t, x, y, z) + (delta_t / (h * rho_val)) * d_val;
        adjust_for_sponge(next_vel_y);

        // define the value at t+1.
        vel_y(t+1, x, y, z) EQUALS next_vel_y;
    }
    void define_vel_z(Condition at_last_z) {
        GridValue rho_val = (rho(x,   y,   z) +
                             rho(x+1, y,   z) +
                             rho(x,   y-1, z) +
                             rho(x+1, y-1, z)) * (1.0 / 4.0);
        GridValue d_val =
            c1 * (stress_xz(t, x+1, y,   z  ) - stress_xz(t, x,   y,   z  )) +
            c2 * (stress_xz(t, x+2, y,   z  ) - stress_xz(t, x-1, y,   z  )) +
            c1 * (stress_yz(t, x,   y,   z  ) - stress_yz(t, x,   y-1, z  )) +
            c2 * (stress_yz(t, x,   y+1, z  ) - stress_yz(t, x,   y-2, z  )) +
            c1 * (stress_zz(t, x,   y,   z+1) - stress_zz(t, x,   y,   z  )) +
            c2 * (stress_zz(t, x,   y,   z+2) - stress_zz(t, x,   y,   z-1));
        GridValue next_vel_z = vel_z(t, x, y, z) + (delta_t / (h * rho_val)) * d_val;
        adjust_for_sponge(next_vel_z);

        // define the value at t+1.
        vel_z(t+1, x, y, z) EQUALS next_vel_z;
    }

    // Free-surface boundary equations for velocity.
    void define_free_surface_vel(Condition at_last_z) {

        // Following expressions are valid only when z == last value in domain.
        // Note that values beyond the last index are updated, i.e., in the halo.
        
        // A couple of intermediate values.
        GridValue d_x_val = vel_x(t+1, x+1, y, z) -
            (vel_z(t+1, x+1, y, z) - vel_z(t+1, x, y, z));
        GridValue d_y_val = vel_y(t+1, x, y-1, z) -
            (vel_z(t+1, x, y, z) - vel_z(t+1, x, y-1, z));
        
        // Following values are valid at the free surface.
        GridValue plus1_vel_x = vel_x(t+1, x, y, z) -
            (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z));
        GridValue plus1_vel_y = vel_y(t+1, x, y, z) -
            (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z));
        GridValue plus1_vel_z = vel_z(t+1, x, y, z) -
            ((d_x_val - plus1_vel_x) +
             (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z)) +
             (plus1_vel_y - d_y_val) +
             (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) /
            ((mu(x, y, z) *
              (2.0 / mu(x, y, z) + 1.0 / lambda(x, y, z))));

        // Define equivalencies to be valid only when z == last value in domain.
        vel_x(t+1, x, y, z+1) EQUALS plus1_vel_x
            IF at_last_z;
        vel_y(t+1, x, y, z+1) EQUALS plus1_vel_y
            IF at_last_z;
        vel_z(t+1, x, y, z+1) EQUALS plus1_vel_z
            IF at_last_z;
    }
    
    // Stress-grid define functions.  For each D in xx, yy, zz, xy, xz, yz,
    // define stress_D at t+1 based on stress_D at t and vel grids at t+1.
    // This implies that the velocity-grid define functions must be called
    // before these for a given value of t.  Note that the t, x, y, z
    // parameters are integer grid indices, not actual offsets in time or
    // space, so half-steps due to staggered grids are adjusted
    // appropriately.

    void define_stress_xx(Condition at_last_z,
                          GridValue lambda_val, GridValue mu_val,
                          GridValue d_x_val, GridValue d_y_val, GridValue d_z_val) {

        GridValue next_stress_xx = stress_xx(t, x, y, z) +
            ((delta_t / h) * ((2 * mu_val * d_x_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val))));
        adjust_for_sponge(next_stress_xx);

        // define the value at t+1.
        stress_xx(t+1, x, y, z) EQUALS next_stress_xx;
    }
    void define_stress_yy(Condition at_last_z,
                          GridValue lambda_val, GridValue mu_val,
                          GridValue d_x_val, GridValue d_y_val, GridValue d_z_val) {

        GridValue next_stress_yy = stress_yy(t, x, y, z) +
            ((delta_t / h) * ((2 * mu_val * d_y_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val))));
        adjust_for_sponge(next_stress_yy);

        // define the value at t+1.
        stress_yy(t+1, x, y, z) EQUALS next_stress_yy;
    }
    void define_stress_xy(Condition at_last_z) {

        GridValue mu_val = 2.0 /
            (mu(x,   y,   z  ) + mu(x,   y,   z-1));

        // Note that we are using the velocity values at t+1.
        GridValue d_xy_val =
            c1 * (vel_x(t+1, x,   y+1, z  ) - vel_x(t+1, x,   y,   z  )) +
            c2 * (vel_x(t+1, x,   y+2, z  ) - vel_x(t+1, x,   y-1, z  ));
        GridValue d_yx_val =
            c1 * (vel_y(t+1, x,   y,   z  ) - vel_y(t+1, x-1, y,   z  )) +
            c2 * (vel_y(t+1, x+1, y,   z  ) - vel_y(t+1, x-2, y,   z  ));

        GridValue next_stress_xy = stress_xy(t, x, y, z) +
            ((mu_val * delta_t / h) * (d_xy_val + d_yx_val));
        adjust_for_sponge(next_stress_xy);

        // define the value at t+1.
        stress_xy(t+1, x, y, z) EQUALS next_stress_xy;
    }
    void define_stress_xz(Condition at_last_z) {

        GridValue mu_val = 2.0 /
            (mu(x,   y,   z  ) + mu(x,   y-1, z  ));

        // Note that we are using the velocity values at t+1.
        GridValue d_xz_val =
            c1 * (vel_x(t+1, x,   y,   z+1) - vel_x(t+1, x,   y,   z  )) +
            c2 * (vel_x(t+1, x,   y,   z+2) - vel_x(t+1, x,   y,   z-1));
        GridValue d_zx_val =
            c1 * (vel_z(t+1, x,   y,   z  ) - vel_z(t+1, x-1, y,   z  )) +
            c2 * (vel_z(t+1, x+1, y,   z  ) - vel_z(t+1, x-2, y,   z  ));

        GridValue next_stress_xz = stress_xz(t, x, y, z) +
            ((mu_val * delta_t / h) * (d_xz_val + d_zx_val));
        adjust_for_sponge(next_stress_xz);

        // define the value at t+1 (special case: zero at surface).
#ifdef DO_SURFACE
        stress_xz(t+1, x, y, z) EQUALS next_stress_xz
            IF !at_last_z;
        stress_xz(t+1, x, y, z) EQUALS 0.0
            IF at_last_z;
#else
        stress_xz(t+1, x, y, z) EQUALS next_stress_xz;
#endif
    }
    void define_stress_yz(Condition at_last_z) {

        GridValue mu_val = 2.0 /
            (mu(x,   y,   z  ) + mu(x+1, y,   z  ));

        // Note that we are using the velocity values at t+1.
        GridValue d_yz_val =
            c1 * (vel_y(t+1, x,   y,   z+1) - vel_y(t+1, x,   y,   z  )) +
            c2 * (vel_y(t+1, x,   y,   z+2) - vel_y(t+1, x,   y,   z-1));
        GridValue d_zy_val =
            c1 * (vel_z(t+1, x,   y+1, z  ) - vel_z(t+1, x,   y,   z  )) +
            c2 * (vel_z(t+1, x,   y+2, z  ) - vel_z(t+1, x,   y-1, z  ));

        GridValue next_stress_yz = stress_yz(t, x, y, z) +
            ((mu_val * delta_t / h) * (d_yz_val + d_zy_val));
        adjust_for_sponge(next_stress_yz);

        // define the value at t+1 (special case: zero at surface).
#ifdef DO_SURFACE
        stress_yz(t+1, x, y, z) EQUALS next_stress_yz
            IF !at_last_z;
        stress_yz(t+1, x, y, z) EQUALS 0.0
            IF at_last_z;
#else
        stress_yz(t+1, x, y, z) EQUALS next_stress_yz;
#endif
    }
    void define_stress_zz(Condition at_last_z,
                          GridValue lambda_val, GridValue mu_val,
                          GridValue d_x_val, GridValue d_y_val, GridValue d_z_val) {

        GridValue next_stress_zz = stress_zz(t, x, y, z) +
            ((delta_t / h) * ((2 * mu_val * d_z_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val))));
        adjust_for_sponge(next_stress_zz);

        // define the value at t+1 (special case: zero at surface).
#ifdef DO_SURFACE
        stress_zz(t+1, x, y, z) EQUALS next_stress_zz
            IF !at_last_z;
        stress_zz(t+1, x, y, z) EQUALS 0.0
            IF at_last_z;
#else
        stress_zz(t+1, x, y, z) EQUALS next_stress_zz;
#endif
    }

    // Free-surface boundary equations for stress.
    void define_free_surface_stress(Condition at_last_z) {

        // Define equivalencies to be valid only when z == last value in domain.
        // Note that values beyond the last index are updated, i.e., in the halo.

        stress_zz(t+1, x, y, z+1) EQUALS -stress_zz(t+1, x, y, z)
            IF at_last_z;
        stress_zz(t+1, x, y, z+2) EQUALS -stress_zz(t+1, x, y, z-1)
            IF at_last_z;

        stress_xz(t+1, x, y, z+1) EQUALS -stress_xz(t+1, x, y, z-1)
            IF at_last_z;
        stress_xz(t+1, x, y, z+2) EQUALS -stress_xz(t+1, x, y, z-2)
            IF at_last_z;

        stress_yz(t+1, x, y, z+1) EQUALS -stress_yz(t+1, x, y, z-1)
            IF at_last_z;
        stress_yz(t+1, x, y, z+2) EQUALS -stress_yz(t+1, x, y, z-2)
            IF at_last_z;
    }
    
    // Call all the define_* functions.
    virtual void define() {

        // A condition that is true when index 'z' is at the free-surface boundary.
        Condition at_last_z = (z == last_index(z));
        
        // Define velocity components.
        define_vel_x(at_last_z);
        define_vel_y(at_last_z);
        define_vel_z(at_last_z);

        // Boundary conditions.
#ifdef DO_SURFACE
        define_free_surface_vel(at_last_z);
#endif

        // Define some values common to the diagonal stress equations.
#ifdef PRECOMPUTED_LAMBDA
        // Use this the lambda values are pre-computed once before
        // all time-steps.
        GridValue lambda_val = lambda(x, y, z);
#else
        GridValue lambda_val = 8.0 /
            (lambda(x,   y,   z  ) + lambda(x+1, y,   z  ) +
             lambda(x,   y-1, z  ) + lambda(x+1, y-1, z  ) +
             lambda(x,   y,   z-1) + lambda(x+1, y,   z-1) +
             lambda(x,   y-1, z-1) + lambda(x+1, y-1, z-1));
#endif
        GridValue mu_val = 8.0 /
            (mu(x,   y,   z  ) + mu(x+1, y,   z  ) +
             mu(x,   y-1, z  ) + mu(x+1, y-1, z  ) +
             mu(x,   y,   z-1) + mu(x+1, y,   z-1) +
             mu(x,   y-1, z-1) + mu(x+1, y-1, z-1));

        // Note that we are using the velocity values at t+1.
        GridValue d_x_val =
            c1 * (vel_x(t+1, x+1, y,   z  ) - vel_x(t+1, x,   y,   z  )) +
            c2 * (vel_x(t+1, x+2, y,   z  ) - vel_x(t+1, x-1, y,   z  ));
        GridValue d_y_val =
            c1 * (vel_y(t+1, x,   y,   z  ) - vel_y(t+1, x,   y-1, z  )) +
            c2 * (vel_y(t+1, x,   y+1, z  ) - vel_y(t+1, x,   y-2, z  ));
        GridValue d_z_val =
            c1 * (vel_z(t+1, x,   y,   z  ) - vel_z(t+1, x,   y,   z-1)) +
            c2 * (vel_z(t+1, x,   y,   z+1) - vel_z(t+1, x,   y,   z-2));

        // Define stress components.
        define_stress_xx(at_last_z,
                         lambda_val, mu_val, d_x_val, d_y_val, d_z_val);
        define_stress_yy(at_last_z,
                         lambda_val, mu_val, d_x_val, d_y_val, d_z_val);
        define_stress_zz(at_last_z,
                         lambda_val, mu_val, d_x_val, d_y_val, d_z_val);
        define_stress_xy(at_last_z);
        define_stress_xz(at_last_z);
        define_stress_yz(at_last_z);

        // Boundary conditions.
#ifdef DO_SURFACE
        define_free_surface_stress(at_last_z);
#endif
    }
};

REGISTER_STENCIL(AwpElasticStencil);
