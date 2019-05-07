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

// Stencil equations for AWP numerics.
// http://hpgeoc.sdsc.edu/AWPODC
// http://www.sdsc.edu/News%20Items/PR20160209_earthquake_center.html

// Set the following macro to use a sponge grid instead of 3 sponge arrays.
//#define FULL_SPONGE_GRID

// YASK stencil solution(s) in this file will be integrated into the YASK compiler utility.
#include "yask_compiler_utility_api.hpp"
using namespace std;
using namespace yask;

class AwpStencil : public yc_solution_base {

protected:

    // Indices & dimensions.
    yc_index_node_ptr t = new_step_index("t");           // step in time dim.
    yc_index_node_ptr x = new_domain_index("x");         // spatial dim.
    yc_index_node_ptr y = new_domain_index("y");         // spatial dim.
    yc_index_node_ptr z = new_domain_index("z");         // spatial dim.

    // Time-varying 3D-spatial velocity grids.
    yc_grid_var vel_x = yc_grid_var("vel_x", get_soln(), { t, x, y, z });
    yc_grid_var vel_y = yc_grid_var("vel_y", get_soln(), { t, x, y, z });
    yc_grid_var vel_z = yc_grid_var("vel_z", get_soln(), { t, x, y, z });

    // Time-varying 3D-spatial Stress grids.
    yc_grid_var stress_xx = yc_grid_var("stress_xx", get_soln(), { t, x, y, z });
    yc_grid_var stress_yy = yc_grid_var("stress_yy", get_soln(), { t, x, y, z });
    yc_grid_var stress_zz = yc_grid_var("stress_zz", get_soln(), { t, x, y, z });
    yc_grid_var stress_xy = yc_grid_var("stress_xy", get_soln(), { t, x, y, z });
    yc_grid_var stress_xz = yc_grid_var("stress_xz", get_soln(), { t, x, y, z });
    yc_grid_var stress_yz = yc_grid_var("stress_yz", get_soln(), { t, x, y, z });

    // Time-varying attenuation memory grids.
    yc_grid_var stress_mem_xx = yc_grid_var("stress_mem_xx", get_soln(), { t, x, y, z });
    yc_grid_var stress_mem_yy = yc_grid_var("stress_mem_yy", get_soln(), { t, x, y, z });
    yc_grid_var stress_mem_zz = yc_grid_var("stress_mem_zz", get_soln(), { t, x, y, z });
    yc_grid_var stress_mem_xy = yc_grid_var("stress_mem_xy", get_soln(), { t, x, y, z });
    yc_grid_var stress_mem_xz = yc_grid_var("stress_mem_xz", get_soln(), { t, x, y, z });
    yc_grid_var stress_mem_yz = yc_grid_var("stress_mem_yz", get_soln(), { t, x, y, z });

    // 3D grids used for anelastic attenuation
    yc_grid_var weight = yc_grid_var("weight", get_soln(), { x, y, z });
    yc_grid_var tau2 = yc_grid_var("tau2", get_soln(), { x, y, z });
    yc_grid_var anelastic_ap = yc_grid_var("anelastic_ap", get_soln(), { x, y, z });
    yc_grid_var anelastic_as_diag = yc_grid_var("anelastic_as_diag", get_soln(), { x, y, z });
    yc_grid_var anelastic_xy = yc_grid_var("anelastic_xy", get_soln(), { x, y, z });
    yc_grid_var anelastic_xz = yc_grid_var("anelastic_xz", get_soln(), { x, y, z });
    yc_grid_var anelastic_yz = yc_grid_var("anelastic_yz", get_soln(), { x, y, z });

    // 3D-spatial Lame' coefficients.
    yc_grid_var lambda = yc_grid_var("lambda", get_soln(), { x, y, z });
    yc_grid_var rho = yc_grid_var("rho", get_soln(), { x, y, z });
    yc_grid_var mu = yc_grid_var("mu", get_soln(), { x, y, z });

    // Sponge coefficients.
    // (Most of these will be 1.0.)
#ifdef FULL_SPONGE_GRID
    yc_grid_var sponge = yc_grid_var("sponge", get_soln(), { x, y, z });
#else
    yc_grid_var cr_x = yc_grid_var("cr_x", get_soln(), { x });
    yc_grid_var cr_y = yc_grid_var("cr_y", get_soln(), { y });
    yc_grid_var cr_z = yc_grid_var("cr_z", get_soln(), { z });
#endif

    // Spatial FD coefficients.
    const double c1 = 9.0/8.0;
    const double c2 = -1.0/24.0;

    // Physical dimensions in time and space.
    yc_grid_var delta_t = yc_grid_var("delta_t", get_soln(), { });
    yc_grid_var h = yc_grid_var("h", get_soln(), { });

public:

    AwpStencil() :
        yc_solution_base("awp") { }

    // Adjustment for sponge layer.
    void adjust_for_sponge(yc_number_node_ptr& val) {

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

    void define_vel_x() {
        auto rho_val = (rho(x, y,   z  ) +
                                      rho(x, y-1, z  ) +
                                      rho(x, y,   z-1) +
                                      rho(x, y-1, z-1)) * (1.0 / 4.0);
        auto d_val =
            c1 * (stress_xx(t, x,   y,   z  ) - stress_xx(t, x-1, y,   z  )) +
            c2 * (stress_xx(t, x+1, y,   z  ) - stress_xx(t, x-2, y,   z  )) +
            c1 * (stress_xy(t, x,   y,   z  ) - stress_xy(t, x,   y-1, z  )) +
            c2 * (stress_xy(t, x,   y+1, z  ) - stress_xy(t, x,   y-2, z  )) +
            c1 * (stress_xz(t, x,   y,   z  ) - stress_xz(t, x,   y,   z-1)) +
            c2 * (stress_xz(t, x,   y,   z+1) - stress_xz(t, x,   y,   z-2));
        auto next_vel_x = vel_x(t, x, y, z) + (delta_t / (h * rho_val)) * d_val;
        adjust_for_sponge(next_vel_x);

        // define the value at t+1.
        vel_x(t+1, x, y, z) EQUALS next_vel_x;
    }
    void define_vel_y() {
        auto rho_val = (rho(x,   y, z  ) +
                                      rho(x+1, y, z  ) +
                                      rho(x,   y, z-1) +
                                      rho(x+1, y, z-1)) * (1.0 / 4.0);
        auto d_val =
            c1 * (stress_xy(t, x+1, y,   z  ) - stress_xy(t, x,   y,   z  )) +
            c2 * (stress_xy(t, x+2, y,   z  ) - stress_xy(t, x-1, y,   z  )) +
            c1 * (stress_yy(t, x,   y+1, z  ) - stress_yy(t, x,   y,   z  )) +
            c2 * (stress_yy(t, x,   y+2, z  ) - stress_yy(t, x,   y-1, z  )) +
            c1 * (stress_yz(t, x,   y,   z  ) - stress_yz(t, x,   y,   z-1)) +
            c2 * (stress_yz(t, x,   y,   z+1) - stress_yz(t, x,   y,   z-2));
        auto next_vel_y = vel_y(t, x, y, z) + (delta_t / (h * rho_val)) * d_val;
        adjust_for_sponge(next_vel_y);

        // define the value at t+1.
        vel_y(t+1, x, y, z) EQUALS next_vel_y;
    }
    void define_vel_z() {
        auto rho_val = (rho(x,   y,   z) +
                                      rho(x+1, y,   z) +
                                      rho(x,   y-1, z) +
                                      rho(x+1, y-1, z)) * (1.0 / 4.0);
        auto d_val =
            c1 * (stress_xz(t, x+1, y,   z  ) - stress_xz(t, x,   y,   z  )) +
            c2 * (stress_xz(t, x+2, y,   z  ) - stress_xz(t, x-1, y,   z  )) +
            c1 * (stress_yz(t, x,   y,   z  ) - stress_yz(t, x,   y-1, z  )) +
            c2 * (stress_yz(t, x,   y+1, z  ) - stress_yz(t, x,   y-2, z  )) +
            c1 * (stress_zz(t, x,   y,   z+1) - stress_zz(t, x,   y,   z  )) +
            c2 * (stress_zz(t, x,   y,   z+2) - stress_zz(t, x,   y,   z-1));
        auto next_vel_z = vel_z(t, x, y, z) + (delta_t / (h * rho_val)) * d_val;
        adjust_for_sponge(next_vel_z);

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

    void define_stress_xx(yc_number_node_ptr lambda_val, yc_number_node_ptr mu_val,
                          yc_number_node_ptr d_x_val, yc_number_node_ptr d_y_val, yc_number_node_ptr d_z_val,
                          yc_number_node_ptr tau1) {

        auto stress_mem_xx_old = stress_mem_xx(t, x, y, z);

        auto next_stress_mem_xx = tau2(x, y, z) * stress_mem_xx_old +
            (1.0 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_as_diag(x, y, z) * (d_y_val + d_z_val) -
             (mu_val + 0.5 * lambda_val) * anelastic_ap(x, y, z) * (d_x_val + d_y_val + d_z_val));

        auto next_stress_xx = stress_xx(t, x, y, z) +
            ((delta_t / h) * ((2. * mu_val * d_x_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val)))) +
            delta_t * (next_stress_mem_xx + stress_mem_xx_old);

        adjust_for_sponge(next_stress_xx);

        // define the value at t+1.
        stress_mem_xx(t+1, x, y, z) EQUALS next_stress_mem_xx;
        stress_xx(t+1, x, y, z) EQUALS next_stress_xx;
    }
    void define_stress_yy(yc_number_node_ptr lambda_val, yc_number_node_ptr mu_val,
                          yc_number_node_ptr d_x_val, yc_number_node_ptr d_y_val, yc_number_node_ptr d_z_val,
                          yc_number_node_ptr tau1) {

        auto stress_mem_yy_old = stress_mem_yy(t, x, y, z);

        auto next_stress_mem_yy = tau2(x, y, z) * stress_mem_yy_old +
            (1 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_as_diag(x, y, z) * (d_x_val + d_z_val) -
             (mu_val + 0.5 * lambda_val) * anelastic_ap(x, y, z) * (d_x_val + d_y_val + d_z_val));

        auto next_stress_yy = stress_yy(t, x, y, z) +
            ((delta_t / h) * ((2. * mu_val * d_y_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val)))) +
            delta_t * (next_stress_mem_yy + stress_mem_yy_old);

        adjust_for_sponge(next_stress_yy);

        // define the value at t+1.
        stress_mem_yy(t+1, x, y, z) EQUALS next_stress_mem_yy;
        stress_yy(t+1, x, y, z) EQUALS next_stress_yy;
    }
    void define_stress_zz(yc_number_node_ptr lambda_val, yc_number_node_ptr mu_val,
                          yc_number_node_ptr d_x_val, yc_number_node_ptr d_y_val, yc_number_node_ptr d_z_val,
                          yc_number_node_ptr tau1) {

        auto stress_mem_zz_old = stress_mem_zz(t, x, y, z);

        auto next_stress_mem_zz = tau2(x, y, z) * stress_mem_zz_old +
            (1 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_as_diag(x, y, z) * (d_x_val + d_y_val) -
             (mu_val + 0.5 * lambda_val) * anelastic_ap(x, y, z) * (d_x_val + d_y_val + d_z_val));

        auto next_stress_zz = stress_zz(t, x, y, z) +
            ((delta_t / h) * ((2. * mu_val * d_z_val) +
                              (lambda_val * (d_x_val + d_y_val + d_z_val)))) +
            delta_t * (next_stress_mem_zz + stress_mem_zz_old);

        adjust_for_sponge(next_stress_zz);

        // define the value at t+1.
        stress_mem_zz(t+1, x, y, z) EQUALS next_stress_mem_zz;
        stress_zz(t+1, x, y, z) EQUALS next_stress_zz;
    }
    void define_stress_xy(yc_number_node_ptr tau1) {

        auto mu_val = 2.0 /
            (mu(x,   y,   z  ) + mu(x,   y,   z-1));

        // Note that we are using the velocity values at t+1.
        auto d_xy_val =
            c1 * (vel_x(t+1, x,   y+1, z  ) - vel_x(t+1, x,   y,   z  )) +
            c2 * (vel_x(t+1, x,   y+2, z  ) - vel_x(t+1, x,   y-1, z  ));
        auto d_yx_val =
            c1 * (vel_y(t+1, x,   y,   z  ) - vel_y(t+1, x-1, y,   z  )) +
            c2 * (vel_y(t+1, x+1, y,   z  ) - vel_y(t+1, x-2, y,   z  ));

        auto stress_mem_xy_old = stress_mem_xy(t, x, y, z);

        auto next_stress_mem_xy = tau2(x, y, z) * stress_mem_xy_old -
            (0.5 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_xy(x, y, z) * (d_xy_val + d_yx_val));

        auto next_stress_xy = stress_xy(t, x, y, z) +
            ((mu_val * delta_t / h) * (d_xy_val + d_yx_val)) +
            delta_t * (next_stress_mem_xy + stress_mem_xy_old);

        adjust_for_sponge(next_stress_xy);

        // define the value at t+1.
        stress_mem_xy(t+1, x, y, z) EQUALS next_stress_mem_xy;
        stress_xy(t+1, x, y, z) EQUALS next_stress_xy;
    }
    void define_stress_xz(yc_number_node_ptr tau1) {

        auto mu_val = 2.0 /
            (mu(x,   y,   z  ) + mu(x,   y-1, z  ));

        // Note that we are using the velocity values at t+1.
        auto d_xz_val =
            c1 * (vel_x(t+1, x,   y,   z+1) - vel_x(t+1, x,   y,   z  )) +
            c2 * (vel_x(t+1, x,   y,   z+2) - vel_x(t+1, x,   y,   z-1));
        auto d_zx_val =
            c1 * (vel_z(t+1, x,   y,   z  ) - vel_z(t+1, x-1, y,   z  )) +
            c2 * (vel_z(t+1, x+1, y,   z  ) - vel_z(t+1, x-2, y,   z  ));

        auto stress_mem_xz_old = stress_mem_xz(t, x, y, z);

        auto next_stress_mem_xz = tau2(x, y, z) * stress_mem_xz_old -
            (0.5 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_xz(x, y, z) * (d_xz_val + d_zx_val));

        auto next_stress_xz = stress_xz(t, x, y, z) +
            ((mu_val * delta_t / h) * (d_xz_val + d_zx_val)) +
            delta_t * (next_stress_mem_xz + stress_mem_xz_old);

        adjust_for_sponge(next_stress_xz);

        // define the value at t+1.
        stress_mem_xz(t+1, x, y, z) EQUALS next_stress_mem_xz;
        stress_xz(t+1, x, y, z) EQUALS next_stress_xz;
    }
    void define_stress_yz(yc_number_node_ptr tau1) {

        auto mu_val = 2.0 /
            (mu(x,   y,   z  ) + mu(x+1, y,   z  ));

        // Note that we are using the velocity values at t+1.
        auto d_yz_val =
            c1 * (vel_y(t+1, x,   y,   z+1) - vel_y(t+1, x,   y,   z  )) +
            c2 * (vel_y(t+1, x,   y,   z+2) - vel_y(t+1, x,   y,   z-1));
        auto d_zy_val =
            c1 * (vel_z(t+1, x,   y+1, z  ) - vel_z(t+1, x,   y,   z  )) +
            c2 * (vel_z(t+1, x,   y+2, z  ) - vel_z(t+1, x,   y-1, z  ));

        auto stress_mem_yz_old = stress_mem_yz(t, x, y, z);

        auto next_stress_mem_yz = tau2(x, y, z) * stress_mem_yz_old -
            (0.5 / h) * tau1 * weight(x, y, z) *
            (mu_val * anelastic_yz(x, y, z) * (d_yz_val + d_zy_val));

        auto next_stress_yz = stress_yz(t, x, y, z) +
            ((mu_val * delta_t / h) * (d_yz_val + d_zy_val)) +
            delta_t * (next_stress_mem_yz + stress_mem_yz_old);

        adjust_for_sponge(next_stress_yz);

        // define the value at t+1.
        stress_mem_yz(t+1, x, y, z) EQUALS next_stress_mem_yz;
        stress_yz(t+1, x, y, z) EQUALS next_stress_yz;
    }

    // Call all the define_* functions.
    virtual void define() {

        // Define velocity components.
        define_vel_x();
        define_vel_y();
        define_vel_z();

        // Define some values common to the diagonal stress equations.
        auto lambda_val = 8.0 /
            (lambda(x,   y,   z  ) + lambda(x+1, y,   z  ) +
             lambda(x,   y-1, z  ) + lambda(x+1, y-1, z  ) +
             lambda(x,   y,   z-1) + lambda(x+1, y,   z-1) +
             lambda(x,   y-1, z-1) + lambda(x+1, y-1, z-1));
        auto mu_val = 8.0 /
            (mu(x,   y,   z  ) + mu(x+1, y,   z  ) +
             mu(x,   y-1, z  ) + mu(x+1, y-1, z  ) +
             mu(x,   y,   z-1) + mu(x+1, y,   z-1) +
             mu(x,   y-1, z-1) + mu(x+1, y-1, z-1));

        // Note that we are using the velocity values at t+1.
        auto d_x_val =
            c1 * (vel_x(t+1, x+1, y,   z  ) - vel_x(t+1, x,   y,   z  )) +
            c2 * (vel_x(t+1, x+2, y,   z  ) - vel_x(t+1, x-1, y,   z  ));
        auto d_y_val =
            c1 * (vel_y(t+1, x,   y,   z  ) - vel_y(t+1, x,   y-1, z  )) +
            c2 * (vel_y(t+1, x,   y+1, z  ) - vel_y(t+1, x,   y-2, z  ));
        auto d_z_val =
            c1 * (vel_z(t+1, x,   y,   z  ) - vel_z(t+1, x,   y,   z-1)) +
            c2 * (vel_z(t+1, x,   y,   z+1) - vel_z(t+1, x,   y,   z-2));

        auto tau1 = 1.0 - tau2(x, y, z);

        // Define stress components.
        define_stress_xx(lambda_val, mu_val, d_x_val, d_y_val, d_z_val, tau1);
        define_stress_yy(lambda_val, mu_val, d_x_val, d_y_val, d_z_val, tau1);
        define_stress_zz(lambda_val, mu_val, d_x_val, d_y_val, d_z_val, tau1);
        define_stress_xy(tau1);
        define_stress_xz(tau1);
        define_stress_yz(tau1);
    }
};

// Create an object of type 'AwpStencil',
// making it available in the YASK compiler utility via the
// '-stencil' commmand-line option or the 'stencil=' build option.
static AwpStencil AwpStencil_instance;
