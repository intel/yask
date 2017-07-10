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

// Stencil equations for AWP numerics.
// http://hpgeoc.sdsc.edu/AWPODC
// http://www.sdsc.edu/News%20Items/PR20160209_earthquake_center.html

#include "StencilBase.hpp"

class AwpStencil : public StencilBase {

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

    // Time-varying attenuation memory grids.
    MAKE_GRID(stress_mem_xx, t, x, y, z);
    MAKE_GRID(stress_mem_yy, t, x, y, z);
    MAKE_GRID(stress_mem_zz, t, x, y, z);
    MAKE_GRID(stress_mem_xy, t, x, y, z);
    MAKE_GRID(stress_mem_xz, t, x, y, z);
    MAKE_GRID(stress_mem_yz, t, x, y, z);

    // 3D grids used for anelastic attenuation
    MAKE_GRID(weight, x, y, z);
    MAKE_GRID(tau2, x, y, z);
    MAKE_GRID(anelastic_ap, x, y, z);
    MAKE_GRID(anelastic_as_diag, x, y, z);
    MAKE_GRID(anelastic_xy, x, y, z);
    MAKE_GRID(anelastic_xz, x, y, z);
    MAKE_GRID(anelastic_yz, x, y, z);
  
    // 3D-spatial Lame' coefficients.
    MAKE_GRID(lambda, x, y, z);
    MAKE_GRID(rho, x, y, z);
    MAKE_GRID(mu, x, y, z);

    // Sponge coefficients.
    // (Most of these will be 1.0.)
    MAKE_GRID(sponge, x, y, z);

    // Spatial FD coefficients.
    const double c1 = 9.0/8.0;
    const double c2 = -1.0/24.0;

    // Physical dimensions in time and space.
    MAKE_SCALAR(delta_t);
    MAKE_SCALAR(h);

public:

    AwpStencil(StencilList& stencils) :
        StencilBase("awp", stencils)
    {
    }

    // Adjustment for sponge layer.
    void adjust_for_sponge(GridValue& val, GridIndex x, GridIndex y, GridIndex z) {

        // TODO: It may be more efficient to skip processing interior nodes
        // because their sponge coefficients are 1.0.  But this would
        // necessitate handling conditionals. The branch mispredictions may
        // cost more than the overhead of the extra loads and multiplies.

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
        vel_x(t+1, x, y, z) EQUALS next_vel_x;
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
        vel_y(t+1, x, y, z) EQUALS next_vel_y;
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
        vel_z(t+1, x, y, z) EQUALS next_vel_z;
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
                          GridValue d_x_val, GridValue d_y_val, GridValue d_z_val,
                          GridValue tau1) {

        GridValue stress_mem_xx_old = stress_mem_xx(t, x, y, z);

        GridValue next_stress_mem_xx = tau2(x, y, z) * stress_mem_xx_old +
            (1.0 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_as_diag(x, y, z) * (d_y_val + d_z_val) -
             (mu_val + 0.5 * lambda_val) * anelastic_ap(x, y, z) * (d_x_val + d_y_val + d_z_val));
        
        GridValue next_stress_xx = stress_xx(t, x, y, z) +
            ((delta_t / h) * ((2. * mu_val * d_x_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val)))) +
            delta_t * (next_stress_mem_xx + stress_mem_xx_old);
        
        adjust_for_sponge(next_stress_xx, x, y, z);

        // define the value at t+1.
        stress_mem_xx(t+1, x, y, z) EQUALS next_stress_mem_xx;
        stress_xx(t+1, x, y, z) EQUALS next_stress_xx;
    }
    void define_stress_yy(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                          GridValue lambda_val, GridValue mu_val,
                          GridValue d_x_val, GridValue d_y_val, GridValue d_z_val,
                          GridValue tau1) {

        GridValue stress_mem_yy_old = stress_mem_yy(t, x, y, z);

        GridValue next_stress_mem_yy = tau2(x, y, z) * stress_mem_yy_old +
            (1 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_as_diag(x, y, z) * (d_x_val + d_z_val) -
             (mu_val + 0.5 * lambda_val) * anelastic_ap(x, y, z) * (d_x_val + d_y_val + d_z_val));
        
        GridValue next_stress_yy = stress_yy(t, x, y, z) +
            ((delta_t / h) * ((2. * mu_val * d_y_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val)))) +
            delta_t * (next_stress_mem_yy + stress_mem_yy_old);

        adjust_for_sponge(next_stress_yy, x, y, z);

        // define the value at t+1.
        stress_mem_yy(t+1, x, y, z) EQUALS next_stress_mem_yy;
        stress_yy(t+1, x, y, z) EQUALS next_stress_yy;
    }
    void define_stress_zz(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                          GridValue lambda_val, GridValue mu_val,
                          GridValue d_x_val, GridValue d_y_val, GridValue d_z_val,
                          GridValue tau1) {

        GridValue stress_mem_zz_old = stress_mem_zz(t, x, y, z);

        GridValue next_stress_mem_zz = tau2(x, y, z) * stress_mem_zz_old +
            (1 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_as_diag(x, y, z) * (d_x_val + d_y_val) -
             (mu_val + 0.5 * lambda_val) * anelastic_ap(x, y, z) * (d_x_val + d_y_val + d_z_val));

        GridValue next_stress_zz = stress_zz(t, x, y, z) +
            ((delta_t / h) * ((2. * mu_val * d_z_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val)))) +
            delta_t * (next_stress_mem_zz + stress_mem_zz_old);

        adjust_for_sponge(next_stress_zz, x, y, z);

        // define the value at t+1.
        stress_mem_zz(t+1, x, y, z) EQUALS next_stress_mem_zz;
        stress_zz(t+1, x, y, z) EQUALS next_stress_zz;
    }
    void define_stress_xy(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                          GridValue tau1) {

        GridValue mu_val = 2.0 /
            (mu(x,   y,   z  ) + mu(x,   y,   z-1));

        // Note that we are using the velocity values at t+1.
        GridValue d_xy_val =
            c1 * (vel_x(t+1, x,   y+1, z  ) - vel_x(t+1, x,   y,   z  )) +
            c2 * (vel_x(t+1, x,   y+2, z  ) - vel_x(t+1, x,   y-1, z  ));
        GridValue d_yx_val =
            c1 * (vel_y(t+1, x,   y,   z  ) - vel_y(t+1, x-1, y,   z  )) +
            c2 * (vel_y(t+1, x+1, y,   z  ) - vel_y(t+1, x-2, y,   z  ));

        GridValue stress_mem_xy_old = stress_mem_xy(t, x, y, z);

        GridValue next_stress_mem_xy = tau2(x, y, z) * stress_mem_xy_old -
            (0.5 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_xy(x, y, z) * (d_xy_val + d_yx_val));
      
        GridValue next_stress_xy = stress_xy(t, x, y, z) +
            ((mu_val * delta_t / h) * (d_xy_val + d_yx_val)) +
            delta_t * (next_stress_mem_xy + stress_mem_xy_old);

        adjust_for_sponge(next_stress_xy, x, y, z);

        // define the value at t+1.
        stress_mem_xy(t+1, x, y, z) EQUALS next_stress_mem_xy;
        stress_xy(t+1, x, y, z) EQUALS next_stress_xy;
    }
    void define_stress_xz(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                          GridValue tau1) {

        GridValue mu_val = 2.0 /
            (mu(x,   y,   z  ) + mu(x,   y-1, z  ));

        // Note that we are using the velocity values at t+1.
        GridValue d_xz_val =
            c1 * (vel_x(t+1, x,   y,   z+1) - vel_x(t+1, x,   y,   z  )) +
            c2 * (vel_x(t+1, x,   y,   z+2) - vel_x(t+1, x,   y,   z-1));
        GridValue d_zx_val =
            c1 * (vel_z(t+1, x,   y,   z  ) - vel_z(t+1, x-1, y,   z  )) +
            c2 * (vel_z(t+1, x+1, y,   z  ) - vel_z(t+1, x-2, y,   z  ));

        GridValue stress_mem_xz_old = stress_mem_xz(t, x, y, z);

        GridValue next_stress_mem_xz = tau2(x, y, z) * stress_mem_xz_old -
            (0.5 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_xz(x, y, z) * (d_xz_val + d_zx_val));

        GridValue next_stress_xz = stress_xz(t, x, y, z) +
            ((mu_val * delta_t / h) * (d_xz_val + d_zx_val)) +
            delta_t * (next_stress_mem_xz + stress_mem_xz_old);

        adjust_for_sponge(next_stress_xz, x, y, z);

        // define the value at t+1.
        stress_mem_xz(t+1, x, y, z) EQUALS next_stress_mem_xz;
        stress_xz(t+1, x, y, z) EQUALS next_stress_xz;
    }
    void define_stress_yz(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                          GridValue tau1) {

        GridValue mu_val = 2.0 /
            (mu(x,   y,   z  ) + mu(x+1, y,   z  ));

        // Note that we are using the velocity values at t+1.
        GridValue d_yz_val =
            c1 * (vel_y(t+1, x,   y,   z+1) - vel_y(t+1, x,   y,   z  )) +
            c2 * (vel_y(t+1, x,   y,   z+2) - vel_y(t+1, x,   y,   z-1));
        GridValue d_zy_val =
            c1 * (vel_z(t+1, x,   y+1, z  ) - vel_z(t+1, x,   y,   z  )) +
            c2 * (vel_z(t+1, x,   y+2, z  ) - vel_z(t+1, x,   y-1, z  ));

        GridValue stress_mem_yz_old = stress_mem_yz(t, x, y, z);

        GridValue next_stress_mem_yz = tau2(x, y, z) * stress_mem_yz_old -
            (0.5 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_yz(x, y, z) * (d_yz_val + d_zy_val));

        GridValue next_stress_yz = stress_yz(t, x, y, z) +
            ((mu_val * delta_t / h) * (d_yz_val + d_zy_val)) +
            delta_t * (next_stress_mem_yz + stress_mem_yz_old);

        adjust_for_sponge(next_stress_yz, x, y, z);

        // define the value at t+1.
        stress_mem_yz(t+1, x, y, z) EQUALS next_stress_mem_yz;
        stress_yz(t+1, x, y, z) EQUALS next_stress_yz;
    }

    // Call all the define_* functions.
    virtual void define() {

        // Define velocity components.
        define_vel_x(t, x, y, z);
        define_vel_y(t, x, y, z);
        define_vel_z(t, x, y, z);

        // Define some values common to the diagonal stress equations.
        GridValue lambda_val = 8.0 /
            (lambda(x,   y,   z  ) + lambda(x+1, y,   z  ) +
             lambda(x,   y-1, z  ) + lambda(x+1, y-1, z  ) +
             lambda(x,   y,   z-1) + lambda(x+1, y,   z-1) +
             lambda(x,   y-1, z-1) + lambda(x+1, y-1, z-1));
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

        GridValue tau1 = 1.0 - tau2(x, y, z);

        // Define stress components.
        define_stress_xx(t, x, y, z, lambda_val, mu_val, d_x_val, d_y_val, d_z_val, tau1);
        define_stress_yy(t, x, y, z, lambda_val, mu_val, d_x_val, d_y_val, d_z_val, tau1);
        define_stress_zz(t, x, y, z, lambda_val, mu_val, d_x_val, d_y_val, d_z_val, tau1);
        define_stress_xy(t, x, y, z, tau1);
        define_stress_xz(t, x, y, z, tau1);
        define_stress_yz(t, x, y, z, tau1);
    }
};

REGISTER_STENCIL(AwpStencil);
