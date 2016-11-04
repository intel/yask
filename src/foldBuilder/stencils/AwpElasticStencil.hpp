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

// Stencil equations for AWP elastic numerics.
// http://hpgeoc.sdsc.edu/AWPODC
// http://www.sdsc.edu/News%20Items/PR20160209_earthquake_center.html

#include "StencilBase.hpp"

class AwpElasticStencil : public StencilBase {

protected:

    // Time-varying 3D-spatial velocity grids.
    Grid vel_x, vel_y, vel_z;

    // Time-varying 3D-spatial Stress grids.
    Grid stress_xx, stress_yy, stress_zz;
    Grid stress_xy, stress_xz, stress_yz;

    // 3D-spatial Lame' coefficients.
    Grid lambda, rho, mu;

    // Sponge coefficients.
    // (Most of these will be 1.0.)
    Grid sponge;

    // Spatial FD coefficients.
    const double c1 = 9.0/8.0;
    const double c2 = -1.0/24.0;

    // Physical dimensions in time and space.
    Param delta_t, h;

public:

    AwpElasticStencil(StencilList& stencils) :
        StencilBase("awp_elastic", stencils)
    {
        // Specify the dimensions of each grid.
        // (This names the dimensions; it does not specify their sizes.)
        INIT_GRID_4D(vel_x, t, x, y, z);
        INIT_GRID_4D(vel_y, t, x, y, z);
        INIT_GRID_4D(vel_z, t, x, y, z);
        INIT_GRID_4D(stress_xx, t, x, y, z);
        INIT_GRID_4D(stress_yy, t, x, y, z);
        INIT_GRID_4D(stress_zz, t, x, y, z);
        INIT_GRID_4D(stress_xy, t, x, y, z);
        INIT_GRID_4D(stress_xz, t, x, y, z);
        INIT_GRID_4D(stress_yz, t, x, y, z);
        INIT_GRID_3D(lambda, x, y, z);
        INIT_GRID_3D(rho, x, y, z);
        INIT_GRID_3D(mu, x, y, z);
        INIT_GRID_3D(sponge, x, y, z);

        // Initialize the parameters (both are scalars).
        INIT_PARAM(delta_t);
        INIT_PARAM(h);
    }

    // Adjustment for sponge layer.
    void adjust_for_sponge(GridValue& val, GridIndex x, GridIndex y, GridIndex z) {

        // TODO: It may be more efficient to skip processing interior nodes
        // because their sponge coefficients are 1.0. This would require
        // setting up sub-domains inside of and outside of the sponge area.
        // It may not be worth the added complexity, though: the cache
        // blocks at the sub-domain intervals would likely be broken into
        // smaller pieces, affecting performance.

        val *= sponge(x, y, z);
    }

    // Velocity-grid define functions.  For each D in x, y, z, define vel_D
    // at t+1 based on vel_x at t and stress grids at t.  Note that the t,
    // x, y, z parameters are integer grid indices, not actual offsets in
    // time or space, so half-steps due to staggered grids are adjusted
    // appropriately.

    void define_vel_x(GridIndex t, GridIndex x, GridIndex y, GridIndex z) {
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
        adjust_for_sponge(next_vel_x, x, y, z);

        // define the value at t+1.
        vel_x(t+1, x, y, z) == next_vel_x;
    }
    void define_vel_y(GridIndex t, GridIndex x, GridIndex y, GridIndex z) {
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
        adjust_for_sponge(next_vel_y, x, y, z);

        // define the value at t+1.
        vel_y(t+1, x, y, z) == next_vel_y;
    }
    void define_vel_z(GridIndex t, GridIndex x, GridIndex y, GridIndex z) {
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
        adjust_for_sponge(next_vel_z, x, y, z);

        // define the value at t+1.
        vel_z(t+1, x, y, z) == next_vel_z;
    }

    // Free-surface boundary equations for velocity.
    void define_free_surface_vel(GridIndex t, GridIndex x, GridIndex y, GridIndex z) {

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
        Condition at_lastz = z == last_index(z);
        vel_x(t+1, x, y, z+1) == plus1_vel_x IF at_lastz;
        vel_y(t+1, x, y, z+1) == plus1_vel_y IF at_lastz;
        vel_z(t+1, x, y, z+1) == plus1_vel_z IF at_lastz;
    }
    
    // Stress-grid define functions.  For each D in xx, yy, zz, xy, xz, yz,
    // define stress_D at t+1 based on stress_D at t and vel grids at t+1.
    // This implies that the velocity-grid define functions must be called
    // before these for a given value of t.  Note that the t, x, y, z
    // parameters are integer grid indices, not actual offsets in time or
    // space, so half-steps due to staggered grids are adjusted
    // appropriately.

    void define_stress_xx(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                          GridValue lambda_val, GridValue mu_val,
                          GridValue d_x_val, GridValue d_y_val, GridValue d_z_val) {

        GridValue next_stress_xx = stress_xx(t, x, y, z) +
            ((delta_t / h) * ((2 * mu_val * d_x_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val))));
        adjust_for_sponge(next_stress_xx, x, y, z);

        // define the value at t+1.
        stress_xx(t+1, x, y, z) == next_stress_xx;
    }
    void define_stress_yy(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                          GridValue lambda_val, GridValue mu_val,
                          GridValue d_x_val, GridValue d_y_val, GridValue d_z_val) {

        GridValue next_stress_yy = stress_yy(t, x, y, z) +
            ((delta_t / h) * ((2 * mu_val * d_y_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val))));
        adjust_for_sponge(next_stress_yy, x, y, z);

        // define the value at t+1.
        stress_yy(t+1, x, y, z) == next_stress_yy;
    }
    void define_stress_zz(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                          GridValue lambda_val, GridValue mu_val,
                          GridValue d_x_val, GridValue d_y_val, GridValue d_z_val) {

        GridValue next_stress_zz = stress_zz(t, x, y, z) +
            ((delta_t / h) * ((2 * mu_val * d_z_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val))));
        adjust_for_sponge(next_stress_zz, x, y, z);

        // define the value at t+1.
        stress_zz(t+1, x, y, z) == next_stress_zz;
    }
    void define_stress_xy(GridIndex t, GridIndex x, GridIndex y, GridIndex z) {

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
        adjust_for_sponge(next_stress_xy, x, y, z);

        // define the value at t+1.
        stress_xy(t+1, x, y, z) == next_stress_xy;
    }
    void define_stress_xz(GridIndex t, GridIndex x, GridIndex y, GridIndex z) {

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
        adjust_for_sponge(next_stress_xz, x, y, z);

        // define the value at t+1.
        stress_xz(t+1, x, y, z) == next_stress_xz;
    }
    void define_stress_yz(GridIndex t, GridIndex x, GridIndex y, GridIndex z) {

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
        adjust_for_sponge(next_stress_yz, x, y, z);

        // define the value at t+1.
        stress_yz(t+1, x, y, z) == next_stress_yz;
    }

    // Free-surface boundary equations for stress.
    void define_free_surface_stress(GridIndex t, GridIndex x, GridIndex y, GridIndex z) {

        // Define equivalencies to be valid only when z == last value in domain.
        // Note that values beyond the last index are updated, i.e., in the halo.

        Condition at_lastz = z == last_index(z);

        stress_zz(t+1, x, y, z+1) == -stress_zz(t+1, x, y, z)
            IF at_lastz;
        stress_zz(t+1, x, y, z+2) == -stress_zz(t+1, x, y, z-1)
            IF at_lastz;

        stress_xz(t+1, x, y, z) == 0.0 IF at_lastz;
        stress_xz(t+1, x, y, z+1) == -stress_xz(t+1, x, y, z-1)
            IF at_lastz;
        stress_xz(t+1, x, y, z+2) == -stress_zz(t+1, x, y, z-2)
            IF at_lastz;

        stress_yz(t+1, x, y, z) == 0.0 IF at_lastz;
        stress_yz(t+1, x, y, z+1) == -stress_yz(t+1, x, y, z-1)
            IF at_lastz;
        stress_yz(t+1, x, y, z+2) == -stress_yz(t+1, x, y, z-2)
            IF at_lastz;

        // TODO: these equations for stress_xz(t+1, x, y, z) and
        // stress_yz(t+1, x, y, z) conflict with those in define_stress_xz()
        // and define_stress_yz() when z == last_index(z). It works ok
        // because these are applied last, but they should actaully be in
        // distinct sub-domains.
    }
    
    // Call all the define_* functions.
    virtual void define(const IntTuple& offsets) {
        GET_OFFSET(t);
        GET_OFFSET(x);
        GET_OFFSET(y);
        GET_OFFSET(z);

        // Define velocity components.
        define_vel_x(t, x, y, z);
        define_vel_y(t, x, y, z);
        define_vel_z(t, x, y, z);

        // Define some values common to the diagonal stress equations.
#ifdef PRECOMPUTED_LAMBDA
        // This assumes the lambda stencil is computed once before
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
        define_stress_xx(t, x, y, z, lambda_val, mu_val, d_x_val, d_y_val, d_z_val);
        define_stress_yy(t, x, y, z, lambda_val, mu_val, d_x_val, d_y_val, d_z_val);
        define_stress_zz(t, x, y, z, lambda_val, mu_val, d_x_val, d_y_val, d_z_val);
        define_stress_xy(t, x, y, z);
        define_stress_xz(t, x, y, z);
        define_stress_yz(t, x, y, z);

        // Boundary conditions.
        define_free_surface_vel(t, x, y, z);
        define_free_surface_stress(t, x, y, z);
    }
};

REGISTER_STENCIL(AwpElasticStencil);
