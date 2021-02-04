/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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

// YASK stencil solution(s) in this file will be integrated into the YASK compiler utility.
#include "yask_compiler_api.hpp"
using namespace std;
using namespace yask;

// Set the following macro to use a 3D sponge var instead of 3 sponge arrays.
//#define FULL_SPONGE_VAR

// The remainder of the macros apply only to the "abc" versions.

// Set the following macro to define all points, even those above the
// surface that are never used.
#define SET_ALL_POINTS

// Set the following macro to calculate free-surface boundary values.
#define DO_ABOVE_SURFACE

// Set the following macro to use intermediate scratch vars.
//#define USE_SCRATCH_VARS

// For the surface stress conditions, we need to write into 2 points
// above the surface.  Since we can only write into the "domain", we
// will define the surface index to be 2 points before the last domain
// index. Thus, there will be two layers in the domain above the surface.
#define SURFACE_IDX (last_domain_index(z) - 2)

// Define some sub-domains related to the surface.
#define BELOW_SURFACE       (z < SURFACE_IDX)
#define AT_SURFACE          (z == SURFACE_IDX)
#define AT_OR_BELOW_SURFACE (z <= SURFACE_IDX)
#define ONE_ABOVE_SURFACE   (z == SURFACE_IDX + 1)
#define TWO_ABOVE_SURFACE   (z == SURFACE_IDX + 2)
#define IF_BELOW_SURFACE    IF_DOMAIN BELOW_SURFACE
#define IF_AT_SURFACE       IF_DOMAIN AT_SURFACE
#define IF_AT_OR_BELOW_SURFACE IF_DOMAIN AT_OR_BELOW_SURFACE
#define IF_ONE_ABOVE_SURFACE IF_DOMAIN ONE_ABOVE_SURFACE
#define IF_TWO_ABOVE_SURFACE IF_DOMAIN TWO_ABOVE_SURFACE

// Create an anonymous namespace to ensure that types are local.
namespace {

    // Base class containing elastic equations for AWP stencils.
    class AwpElasticBase : public yc_solution_base {
    public:
        AwpElasticBase(const string& name) :
            yc_solution_base(name) { }

    protected:

        // Indices & dimensions.
        yc_index_node_ptr t = new_step_index("t");           // step in time dim.
        yc_index_node_ptr x = new_domain_index("x");         // spatial dim.
        yc_index_node_ptr y = new_domain_index("y");         // spatial dim.
        yc_index_node_ptr z = new_domain_index("z");         // spatial dim.

        // Time-varying 3D-spatial velocity vars.
        yc_var_proxy vel_x = yc_var_proxy("vel_x", get_soln(), { t, x, y, z });
        yc_var_proxy vel_y = yc_var_proxy("vel_y", get_soln(), { t, x, y, z });
        yc_var_proxy vel_z = yc_var_proxy("vel_z", get_soln(), { t, x, y, z });

        // Time-varying 3D-spatial Stress vars.
        yc_var_proxy stress_xx = yc_var_proxy("stress_xx", get_soln(), { t, x, y, z });
        yc_var_proxy stress_yy = yc_var_proxy("stress_yy", get_soln(), { t, x, y, z });
        yc_var_proxy stress_zz = yc_var_proxy("stress_zz", get_soln(), { t, x, y, z });
        yc_var_proxy stress_xy = yc_var_proxy("stress_xy", get_soln(), { t, x, y, z });
        yc_var_proxy stress_xz = yc_var_proxy("stress_xz", get_soln(), { t, x, y, z });
        yc_var_proxy stress_yz = yc_var_proxy("stress_yz", get_soln(), { t, x, y, z });

        // 3D-spatial Lame' coefficients.
        yc_var_proxy lambda = yc_var_proxy("lambda", get_soln(), { x, y, z });
        yc_var_proxy rho = yc_var_proxy("rho", get_soln(), { x, y, z });
        yc_var_proxy mu = yc_var_proxy("mu", get_soln(), { x, y, z });

        // Spatial FD coefficients.
        const double c1 = 9.0/8.0;
        const double c2 = -1.0/24.0;

        // Physical grid spacing in time and space.
        yc_var_proxy delta_t = yc_var_proxy("delta_t", get_soln());
        yc_var_proxy h = yc_var_proxy("h", get_soln());
        // Sponge coefficients.
        // (Most of these will be 1.0.)
#ifdef FULL_SPONGE_VAR
        yc_var_proxy sponge = yc_var_proxy("sponge", get_soln(), { x, y, z });
#else
        yc_var_proxy cr_x = yc_var_proxy("cr_x", get_soln(), { x });
        yc_var_proxy cr_y = yc_var_proxy("cr_y", get_soln(), { y });
        yc_var_proxy cr_z = yc_var_proxy("cr_z", get_soln(), { z });
#endif

#ifdef USE_SCRATCH_VARS
        yc_var_proxy tmp_vel_x = yc_var_proxy("tmp_vel_x", get_soln(), { x, y, z }, true);
        yc_var_proxy tmp_vel_y = yc_var_proxy("tmp_vel_y", get_soln(), { x, y, z }, true);
        yc_var_proxy tmp_vel_z = yc_var_proxy("tmp_vel_z", get_soln(), { x, y, z }, true);
#endif

        // Adjustment for sponge layer.
        void adjust_for_sponge(yc_number_node_ptr& val) {

#ifdef FULL_SPONGE_VAR
            val *= sponge(x, y, z);
#else
            val *= cr_x(x) * cr_y(y) * cr_z(z);
#endif
        }

        // Velocity functions.  For each D in x, y, z, define vel_D
        // at t+1 based on vel_x at t and stress vars at t.  Note that the t,
        // x, y, z parameters are integer var indices, not actual offsets in
        // time or space, so half-steps due to staggered vars are adjusted
        // appropriately.

        virtual yc_number_node_ptr
        get_next_vel_x(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
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

            // Return expr for the value at t+1.
            return next_vel_x;
        }
        virtual yc_number_node_ptr
        get_next_vel_y(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
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

            // Return expr for the value at t+1.
            return next_vel_y;
        }
        virtual yc_number_node_ptr
        get_next_vel_z(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
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

            // Return expr for the value at t+1.
            return next_vel_z;
        }

        // Compute inverse of average of 8 neighbors.
        yc_number_node_ptr
        ave8(yc_var_proxy& g,
             yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {

            return 8.0 /
                (g(x,   y,   z  ) + g(x+1, y,   z  ) +
                 g(x,   y-1, z  ) + g(x+1, y-1, z  ) +
                 g(x,   y,   z-1) + g(x+1, y,   z-1) +
                 g(x,   y-1, z-1) + g(x+1, y-1, z-1));
        }

        // Some common velocity FD calculations.
        // Note that these use vel values from t+1.
        yc_number_node_ptr
        d_x_val(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return
                c1 * (vel_x(t+1, x+1, y,   z  ) - vel_x(t+1, x,   y,   z  )) +
                c2 * (vel_x(t+1, x+2, y,   z  ) - vel_x(t+1, x-1, y,   z  ));
        }
        yc_number_node_ptr
        d_y_val(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return
                c1 * (vel_y(t+1, x,   y,   z  ) - vel_y(t+1, x,   y-1, z  )) +
                c2 * (vel_y(t+1, x,   y+1, z  ) - vel_y(t+1, x,   y-2, z  ));
        }
        yc_number_node_ptr
        d_z_val(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return
                c1 * (vel_z(t+1, x,   y,   z  ) - vel_z(t+1, x,   y,   z-1)) +
                c2 * (vel_z(t+1, x,   y,   z+1) - vel_z(t+1, x,   y,   z-2));
        }
        yc_number_node_ptr
        d_xy_val(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return
                c1 * (vel_x(t+1, x,   y+1, z  ) - vel_x(t+1, x,   y,   z  )) +
                c2 * (vel_x(t+1, x,   y+2, z  ) - vel_x(t+1, x,   y-1, z  ));
        }
        yc_number_node_ptr
        d_yx_val(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return
                c1 * (vel_y(t+1, x,   y,   z  ) - vel_y(t+1, x-1, y,   z  )) +
                c2 * (vel_y(t+1, x+1, y,   z  ) - vel_y(t+1, x-2, y,   z  ));
        }
        yc_number_node_ptr
        d_xz_val(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return
                c1 * (vel_x(t+1, x,   y,   z+1) - vel_x(t+1, x,   y,   z  )) +
                c2 * (vel_x(t+1, x,   y,   z+2) - vel_x(t+1, x,   y,   z-1));
        }        
        yc_number_node_ptr
        d_zx_val(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return
                c1 * (vel_z(t+1, x,   y,   z  ) - vel_z(t+1, x-1, y,   z  )) +
                c2 * (vel_z(t+1, x+1, y,   z  ) - vel_z(t+1, x-2, y,   z  ));
        }
        yc_number_node_ptr
        d_yz_val(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return
                c1 * (vel_y(t+1, x,   y,   z+1) - vel_y(t+1, x,   y,   z  )) +
                c2 * (vel_y(t+1, x,   y,   z+2) - vel_y(t+1, x,   y,   z-1));
        }
        yc_number_node_ptr
        d_zy_val(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return
                c1 * (vel_z(t+1, x,   y+1, z  ) - vel_z(t+1, x,   y,   z  )) +
                c2 * (vel_z(t+1, x,   y+2, z  ) - vel_z(t+1, x,   y-1, z  ));
        }
        
        // Stress functions.  For each D in xx, yy, zz, xy, xz, yz,
        // define stress_D at t+1 based on stress_D at t and vel vars at t+1.
        // This implies that the velocity-var define functions must be called
        // before these for a given value of t.  Note that the t, x, y, z
        // parameters are integer var indices, not actual offsets in time or
        // space, so half-steps due to staggered vars are adjusted
        // appropriately.

        virtual yc_number_node_ptr
        get_next_stress_xx(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {

            auto next_stress_xx = stress_xx(t, x, y, z) +
                ((delta_t / h) * ((2 * ave8(mu, x, y, z) * d_x_val(x, y, z)) +
                                  (ave8(lambda, x, y, z) *
                                   (d_x_val(x, y, z) + d_y_val(x, y, z) + d_z_val(x, y, z)))));
            adjust_for_sponge(next_stress_xx);

            // Return expr for the value at t+1.
            return next_stress_xx;
        }
        virtual yc_number_node_ptr
        get_next_stress_yy(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {

            auto next_stress_yy = stress_yy(t, x, y, z) +
                ((delta_t / h) * ((2 * ave8(mu, x, y, z) * d_y_val(x, y, z)) +
                                  (ave8(lambda, x, y, z) *
                                   (d_x_val(x, y, z) + d_y_val(x, y, z) + d_z_val(x, y, z)))));
            adjust_for_sponge(next_stress_yy);

            // Return expr for the value at t+1.
            return next_stress_yy;
        }
        virtual yc_number_node_ptr
        get_next_stress_zz(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {

            auto next_stress_zz = stress_zz(t, x, y, z) +
                ((delta_t / h) * ((2 * ave8(mu, x, y, z) * d_z_val(x, y, z)) +
                                  (ave8(lambda, x, y, z) *
                                   (d_x_val(x, y, z) + d_y_val(x, y, z) + d_z_val(x, y, z)))));
            adjust_for_sponge(next_stress_zz);

            // Return expr for the value at t+1.
            return next_stress_zz;
        }
        virtual yc_number_node_ptr
        get_next_stress_xy(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {

            // Compute average of 2 neighbors.
            auto mu2 = 2.0 /
                (mu(x,   y,   z  ) + mu(x,   y,   z-1));

            auto next_stress_xy = stress_xy(t, x, y, z) +
                ((mu2 * delta_t / h) * (d_xy_val(x, y, z) + d_yx_val(x, y, z)));
            adjust_for_sponge(next_stress_xy);

            // Return expr for the value at t+1.
            return next_stress_xy;
        }
        virtual yc_number_node_ptr
        get_next_stress_xz(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {

            // Compute average of 2 neighbors.
            auto mu2 = 2.0 /
                (mu(x,   y,   z  ) + mu(x,   y-1, z  ));

            auto next_stress_xz = stress_xz(t, x, y, z) +
                ((mu2 * delta_t / h) * (d_xz_val(x, y, z) + d_zx_val(x, y, z)));
            adjust_for_sponge(next_stress_xz);

            // Return expr for the value at t+1.
            return next_stress_xz;
        }
        virtual yc_number_node_ptr
        get_next_stress_yz(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {

            // Compute average of 2 neighbors.
            auto mu2 = 2.0 /
                (mu(x,   y,   z  ) + mu(x+1, y,   z  ));

            auto next_stress_yz = stress_yz(t, x, y, z) +
                ((mu2 * delta_t / h) * (d_yz_val(x, y, z) + d_zy_val(x, y, z)));
            adjust_for_sponge(next_stress_yz);

            // Return expr for the value at t+1.
            return next_stress_yz;
        }

        // Free-surface boundary equations for velocity.
        void define_free_surface_vel() {

            // Since we're defining points when z == surface + 1,
            // the surface itself will be at z - 1;
            auto surf = z - 1;

#ifdef USE_SCRATCH_VARS

            // The values for velocity at t+1 will be needed
            // in multiple free-surface calculations.
            // Thus, it will reduce the number of FP ops
            // required if we pre-compute them and store them
            // in scratch vars. The downside is more memory
            // and memory accesses.
#define VEL_X tmp_vel_x
#define VEL_Y tmp_vel_y
#define VEL_Z tmp_vel_z
            VEL_X(x, y, z) EQUALS get_next_vel_x(x, y, z);
            VEL_Y(x, y, z) EQUALS get_next_vel_y(x, y, z);
            VEL_Z(x, y, z) EQUALS get_next_vel_z(x, y, z);

#else

            // If not using scratch vars, just call the
            // functions to calculate each value of velocity
            // at t+1 every time it's needed.
#define VEL_X get_next_vel_x
#define VEL_Y get_next_vel_y
#define VEL_Z get_next_vel_z
#endif

            // A couple of intermediate values.
            auto d_x_val = VEL_X(x+1, y, surf) -
                (VEL_Z(x+1, y, surf) - VEL_Z(x, y, surf));
            auto d_y_val = VEL_Y(x, y-1, surf) -
                (VEL_Z(x, y, surf) - VEL_Z(x, y-1, surf));

            // Following values are valid one layer above the free surface.
            auto plus1_vel_x = VEL_X(x, y, surf) -
                (VEL_Z(x, y, surf) - VEL_Z(x-1, y, surf));
            auto plus1_vel_y = VEL_Y(x, y, surf) -
                (VEL_Z(x, y+1, surf) - VEL_Z(x, y, surf));
            auto plus1_vel_z = VEL_Z(x, y, surf) -
                ((d_x_val - plus1_vel_x) +
                 (VEL_X(x+1, y, surf) - VEL_X(x, y, surf)) +
                 (plus1_vel_y - d_y_val) +
                 (VEL_Y(x, y, surf) - VEL_Y(x, y-1, surf))) /
                ((mu(x, y, surf) *
                  (2.0 / mu(x, y, surf) + 1.0 / lambda(x, y, surf))));
#undef VEL_X
#undef VEL_Y
#undef VEL_Z

            // Define layer at one point above surface.
            vel_x(t+1, x, y, z) EQUALS plus1_vel_x IF_ONE_ABOVE_SURFACE;
            vel_y(t+1, x, y, z) EQUALS plus1_vel_y IF_ONE_ABOVE_SURFACE;
            vel_z(t+1, x, y, z) EQUALS plus1_vel_z IF_ONE_ABOVE_SURFACE;

#ifdef SET_ALL_POINTS
            // Define layer two points above surface for completeness, even
            // though these aren't input to any stencils.
            vel_x(t+1, x, y, z) EQUALS 0.0 IF_TWO_ABOVE_SURFACE;
            vel_y(t+1, x, y, z) EQUALS 0.0 IF_TWO_ABOVE_SURFACE;
            vel_z(t+1, x, y, z) EQUALS 0.0 IF_TWO_ABOVE_SURFACE;
#endif
        }

        // Free-surface boundary equations for stress.
        void define_free_surface_stress() {

            // When z == surface + 1, the surface will be at z - 1;
            auto surf = z - 1;

            stress_zz(t+1, x, y, z) EQUALS -get_next_stress_zz(x, y, surf) IF_ONE_ABOVE_SURFACE;
            stress_xz(t+1, x, y, z) EQUALS -get_next_stress_xz(x, y, surf-1) IF_ONE_ABOVE_SURFACE;
            stress_yz(t+1, x, y, z) EQUALS -get_next_stress_yz(x, y, surf-1) IF_ONE_ABOVE_SURFACE;

#ifdef SET_ALL_POINTS
            // Define other 3 stress values for completeness, even
            // though these aren't input to any stencils.
            stress_xx(t+1, x, y, z) EQUALS 0.0 IF_ONE_ABOVE_SURFACE;
            stress_yy(t+1, x, y, z) EQUALS 0.0 IF_ONE_ABOVE_SURFACE;
            stress_xy(t+1, x, y, z) EQUALS 0.0 IF_ONE_ABOVE_SURFACE;
#endif

            // When z == surface + 2, the surface will be at z - 2;
            surf = z - 2;

            stress_zz(t+1, x, y, z) EQUALS -get_next_stress_zz(x, y, surf-1) IF_TWO_ABOVE_SURFACE;
            stress_xz(t+1, x, y, z) EQUALS -get_next_stress_xz(x, y, surf-2) IF_TWO_ABOVE_SURFACE;
            stress_yz(t+1, x, y, z) EQUALS -get_next_stress_yz(x, y, surf-2) IF_TWO_ABOVE_SURFACE;

#ifdef SET_ALL_POINTS
            // Define other 3 stress values for completeness, even
            // though these aren't input to any stencils.
            stress_xx(t+1, x, y, z) EQUALS 0.0 IF_TWO_ABOVE_SURFACE;
            stress_yy(t+1, x, y, z) EQUALS 0.0 IF_TWO_ABOVE_SURFACE;
            stress_xy(t+1, x, y, z) EQUALS 0.0 IF_TWO_ABOVE_SURFACE;
#endif
        }

        // Define the t+1 values for all velocity and stress vars.
        virtual void define_base() {

            // Define velocity components.
            vel_x(t+1, x, y, z) EQUALS get_next_vel_x(x, y, z);
            vel_y(t+1, x, y, z) EQUALS get_next_vel_y(x, y, z);
            vel_z(t+1, x, y, z) EQUALS get_next_vel_z(x, y, z);

            // Define stress components.
            stress_xx(t+1, x, y, z) EQUALS get_next_stress_xx(x, y, z);
            stress_yy(t+1, x, y, z) EQUALS get_next_stress_yy(x, y, z);
            stress_xy(t+1, x, y, z) EQUALS get_next_stress_xy(x, y, z);
            stress_xz(t+1, x, y, z) EQUALS get_next_stress_xz(x, y, z);
            stress_yz(t+1, x, y, z) EQUALS get_next_stress_yz(x, y, z);
            stress_zz(t+1, x, y, z) EQUALS get_next_stress_zz(x, y, z);
        }
        
        // Define the t+1 values for all velocity and stress vars
        // with ABC.
        virtual void define_base_abc() {

            // Define velocity components.
            vel_x(t+1, x, y, z) EQUALS get_next_vel_x(x, y, z) IF_AT_OR_BELOW_SURFACE;
            vel_y(t+1, x, y, z) EQUALS get_next_vel_y(x, y, z) IF_AT_OR_BELOW_SURFACE;
            vel_z(t+1, x, y, z) EQUALS get_next_vel_z(x, y, z) IF_AT_OR_BELOW_SURFACE;

            // Define stress components.  Use non-overlapping sub-domains only,
            // i.e., AT and BELOW but not AT_OR_BELOW, even though there are some
            // repeated stencils. This allows the YASK compiler to bundle all
            // the stress equations together.
            stress_xx(t+1, x, y, z) EQUALS get_next_stress_xx(x, y, z) IF_BELOW_SURFACE;
            stress_yy(t+1, x, y, z) EQUALS get_next_stress_yy(x, y, z) IF_BELOW_SURFACE;
            stress_xy(t+1, x, y, z) EQUALS get_next_stress_xy(x, y, z) IF_BELOW_SURFACE;
            stress_xz(t+1, x, y, z) EQUALS get_next_stress_xz(x, y, z) IF_BELOW_SURFACE;
            stress_yz(t+1, x, y, z) EQUALS get_next_stress_yz(x, y, z) IF_BELOW_SURFACE;
            stress_zz(t+1, x, y, z) EQUALS get_next_stress_zz(x, y, z) IF_BELOW_SURFACE;

            stress_xx(t+1, x, y, z) EQUALS get_next_stress_xx(x, y, z) IF_AT_SURFACE;
            stress_yy(t+1, x, y, z) EQUALS get_next_stress_yy(x, y, z) IF_AT_SURFACE;
            stress_xy(t+1, x, y, z) EQUALS get_next_stress_xy(x, y, z) IF_AT_SURFACE;
            stress_xz(t+1, x, y, z) EQUALS 0.0 IF_AT_SURFACE;
            stress_yz(t+1, x, y, z) EQUALS 0.0 IF_AT_SURFACE;
            stress_zz(t+1, x, y, z) EQUALS get_next_stress_zz(x, y, z) IF_AT_SURFACE;

            // Boundary conditions.
#ifdef DO_ABOVE_SURFACE
            define_free_surface_vel();
            define_free_surface_stress();
#endif
        }

        // Set BKC (best-known configs) found by automated and/or manual
        // tuning. They are only applied for certain target configs.
        virtual void set_configs() {
            auto soln = get_soln(); // pointer to compile-time soln.

            // Only have BKCs for SP FP (4B).
            if (soln->get_element_bytes() == 4) {

                // Compile-time defaults.
                if (soln->is_target_set()) {
                    auto target = soln->get_target();
                    if (target == "knl") {
                        soln->set_prefetch_dist(1, 1);
                        soln->set_prefetch_dist(2, 0);
                    }
                    else if (target == "avx512") {
                        soln->set_prefetch_dist(1, 1);
                        soln->set_prefetch_dist(2, 0);
                    }
                    else if (target == "avx2") {
                        soln->set_prefetch_dist(1, 1);
                        soln->set_prefetch_dist(2, 2);
                    }
                }

                // Kernel run-time defaults, e.g., block-sizes.
                // This code is run immediately after 'kernel_soln' is created.
                soln->CALL_AFTER_NEW_SOLUTION
                    (
                     // Check target at kernel run-time.
                     auto isa = kernel_soln.get_target();
                     if (isa == "knl") {

                         // Use half the threads: 2 threads on 2 cores per block.
                         kernel_soln.apply_command_line_options("-thread_divisor 2 -block_threads 4");
                         
                         kernel_soln.set_block_size("x", 48);
                         kernel_soln.set_block_size("y", 48);
                         kernel_soln.set_block_size("z", 112);
                     }
                     else if (isa == "avx512") {
                         kernel_soln.set_block_size("x", 116);
                         kernel_soln.set_block_size("y", 8);
                         kernel_soln.set_block_size("z", 128);
                     }
                     else if (isa == "avx2") {
                         kernel_soln.set_block_size("x", 64);
                         kernel_soln.set_block_size("y", 8);
                         kernel_soln.set_block_size("z", 64);
                     }
                     );
            }
        }
    };    

    // The elastic version of AWP that does not contain the time-varying
    // attenuation memory vars or the related attenuation constant vars.
    class AwpElasticStencil : public AwpElasticBase {
    public:
        AwpElasticStencil() :
            AwpElasticBase("awp_elastic") { }

        // Define the t+1 values for all velocity and stress vars.
        virtual void define() override {
            define_base();
            set_configs();
        }
    };

    // The elastic version of AWP with free-surface absorbing boundary conditions.
    class AwpElasticABCStencil : public AwpElasticBase {
    public:
        AwpElasticABCStencil() :
            AwpElasticBase("awp_elastic_abc") { }

        // Define the t+1 values for all velocity and stress vars.
        virtual void define() override {
            define_base_abc();
            set_configs();
        }
    };

    // The base for the anelastic version of AWP with time-varying
    // attenuation memory vars and the related attenuation constant vars.
    class AwpBase : public AwpElasticBase {
    public:
        AwpBase(const string& name) :
            AwpElasticBase(name) { }

    protected:

        // Time-varying attenuation memory vars.
        yc_var_proxy stress_mem_xx = yc_var_proxy("stress_mem_xx", get_soln(), { t, x, y, z });
        yc_var_proxy stress_mem_yy = yc_var_proxy("stress_mem_yy", get_soln(), { t, x, y, z });
        yc_var_proxy stress_mem_zz = yc_var_proxy("stress_mem_zz", get_soln(), { t, x, y, z });
        yc_var_proxy stress_mem_xy = yc_var_proxy("stress_mem_xy", get_soln(), { t, x, y, z });
        yc_var_proxy stress_mem_xz = yc_var_proxy("stress_mem_xz", get_soln(), { t, x, y, z });
        yc_var_proxy stress_mem_yz = yc_var_proxy("stress_mem_yz", get_soln(), { t, x, y, z });

        // 3D vars used for anelastic attenuation
        yc_var_proxy weight = yc_var_proxy("weight", get_soln(), { x, y, z });
        yc_var_proxy tau2 = yc_var_proxy("tau2", get_soln(), { x, y, z });
        yc_var_proxy anelastic_ap = yc_var_proxy("anelastic_ap", get_soln(), { x, y, z });
        yc_var_proxy anelastic_as_diag = yc_var_proxy("anelastic_as_diag", get_soln(), { x, y, z });
        yc_var_proxy anelastic_xy = yc_var_proxy("anelastic_xy", get_soln(), { x, y, z });
        yc_var_proxy anelastic_xz = yc_var_proxy("anelastic_xz", get_soln(), { x, y, z });
        yc_var_proxy anelastic_yz = yc_var_proxy("anelastic_yz", get_soln(), { x, y, z });

    public:

        yc_number_node_ptr
        tau1(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return 1.0 - tau2(x, y, z);
        }

        // The velocity functions are the same as those defined in AwpBase.
        // Memory components are added to the stress functions.

        virtual yc_number_node_ptr
        get_next_stress_mem(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                            yc_number_node_ptr stress_mem_old,
                            yc_number_node_ptr d_val) {
            auto next_stress_mem = tau2(x, y, z) * stress_mem_old +
                (1.0 / h) * tau1(x, y, z) * weight(x, y, z) *
                (ave8(mu, x, y, z) * anelastic_as_diag(x, y, z) * d_val -
                 (ave8(mu, x, y, z) + 0.5 * ave8(lambda, x, y, z)) *
                 anelastic_ap(x, y, z) * (d_x_val(x, y, z) + d_y_val(x, y, z) + d_z_val(x, y, z)));
            return next_stress_mem;
        }
        virtual yc_number_node_ptr
        get_next_stress(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                        yc_number_node_ptr stress_old,
                        yc_number_node_ptr stress_mem_old,
                        yc_number_node_ptr next_stress_mem,
                        yc_number_node_ptr d_val) {

            auto next_stress = stress_old +
                ((delta_t / h) * ((2. * ave8(mu, x, y, z) * d_val) +
                                  (ave8(lambda, x, y, z) *
                                   (d_x_val(x, y, z) + d_y_val(x, y, z) + d_z_val(x, y, z))))) +
                delta_t * (next_stress_mem + stress_mem_old);
            adjust_for_sponge(next_stress);

            // Return expr for the value at t+1.
            return next_stress;
        }

        virtual yc_number_node_ptr
        get_next_stress_mem_xx(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return get_next_stress_mem(x, y, z,
                                       stress_mem_xx(t, x, y, z),
                                       d_y_val(x, y, z) + d_z_val(x, y, z));
        }
        virtual yc_number_node_ptr
        get_next_stress_xx(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) override {
            return get_next_stress(x, y, z,
                                   stress_xx(t, x, y, z),
                                   stress_mem_xx(t, x, y, z),
                                   get_next_stress_mem_xx(x, y, z),
                                   d_x_val(x, y, z));
        }
        virtual yc_number_node_ptr
        get_next_stress_mem_yy(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return get_next_stress_mem(x, y, z,
                                       stress_mem_yy(t, x, y, z),
                                       d_x_val(x, y, z) + d_z_val(x, y, z));
        }
        virtual yc_number_node_ptr
        get_next_stress_yy(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) override {
            return get_next_stress(x, y, z,
                                   stress_yy(t, x, y, z),
                                   stress_mem_yy(t, x, y, z),
                                   get_next_stress_mem_yy(x, y, z),
                                   d_y_val(x, y, z));
        }
        virtual yc_number_node_ptr
        get_next_stress_mem_zz(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            return get_next_stress_mem(x, y, z,
                                       stress_mem_zz(t, x, y, z),
                                       d_x_val(x, y, z) + d_y_val(x, y, z));
        }
        virtual yc_number_node_ptr
        get_next_stress_zz(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) override {
            return get_next_stress(x, y, z,
                                   stress_zz(t, x, y, z),
                                   stress_mem_zz(t, x, y, z),
                                   get_next_stress_mem_zz(x, y, z),
                                   d_z_val(x, y, z));
        }

        virtual yc_number_node_ptr
        get_next_stress_mem2(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                             yc_number_node_ptr stress_mem_old,
                             yc_number_node_ptr d_val,
                             yc_number_node_ptr mu_val,
                             yc_number_node_ptr anelastic) {
            auto next_stress_mem = tau2(x, y, z) * stress_mem_old -
                (0.5 / h) * tau1(x, y, z) * weight(x, y, z) *
                (mu_val * anelastic * d_val);
            return next_stress_mem;
        }
        virtual yc_number_node_ptr
        get_next_stress2(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                         yc_number_node_ptr stress_old,
                         yc_number_node_ptr stress_mem_old,
                         yc_number_node_ptr next_stress_mem,
                         yc_number_node_ptr d_val,
                         yc_number_node_ptr mu_val) {
            auto next_stress = stress_old +
                ((mu_val * delta_t / h) * d_val) +
                delta_t * (next_stress_mem + stress_mem_old);
            adjust_for_sponge(next_stress);
            return next_stress;
        }
        virtual yc_number_node_ptr
        get_next_stress_mem_xy(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            auto mu_val = 2.0 /
                (mu(x,   y,   z  ) + mu(x,   y,   z-1));
            return get_next_stress_mem2(x, y, z,
                                        stress_mem_xy(t, x, y, z),
                                        d_xy_val(x, y, z) + d_yx_val(x, y, z),
                                        mu_val,
                                        anelastic_xy(x, y, z));
        }
        virtual yc_number_node_ptr
        get_next_stress_xy(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) override {
            auto mu_val = 2.0 /
                (mu(x,   y,   z  ) + mu(x,   y,   z-1));
            return get_next_stress2(x, y, z,
                                    stress_xy(t, x, y, z),
                                    stress_mem_xy(t, x, y, z),
                                    get_next_stress_mem_xy(x, y, z),
                                    d_xy_val(x, y, z) + d_yx_val(x, y, z),
                                    mu_val);
        }
        virtual yc_number_node_ptr
        get_next_stress_mem_xz(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            auto mu_val = 2.0 /
                (mu(x,   y,   z  ) + mu(x,   y-1, z));
            return get_next_stress_mem2(x, y, z,
                                        stress_mem_xz(t, x, y, z),
                                        d_xz_val(x, y, z) + d_zx_val(x, y, z),
                                        mu_val,
                                        anelastic_xz(x, y, z));
        }
        virtual yc_number_node_ptr
        get_next_stress_xz(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) override {
            auto mu_val = 2.0 /
                (mu(x,   y,   z  ) + mu(x,   y-1, z));
            return get_next_stress2(x, y, z,
                                    stress_xz(t, x, y, z),
                                    stress_mem_xz(t, x, y, z),
                                    get_next_stress_mem_xz(x, y, z),
                                    d_xz_val(x, y, z) + d_zx_val(x, y, z),
                                    mu_val);
        }
        virtual yc_number_node_ptr
        get_next_stress_mem_yz(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) {
            auto mu_val = 2.0 /
                (mu(x,   y,   z  ) + mu(x+1, y, z));
            return get_next_stress_mem2(x, y, z,
                                        stress_mem_yz(t, x, y, z),
                                        d_yz_val(x, y, z) + d_zy_val(x, y, z),
                                        mu_val,
                                        anelastic_yz(x, y, z));
        }
        virtual yc_number_node_ptr
        get_next_stress_yz(yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z) override {
            auto mu_val = 2.0 /
                (mu(x,   y,   z  ) + mu(x+1, y, z));
            return get_next_stress2(x, y, z,
                                    stress_yz(t, x, y, z),
                                    stress_mem_yz(t, x, y, z),
                                    get_next_stress_mem_yz(x, y, z),
                                    d_yz_val(x, y, z) + d_zy_val(x, y, z),
                                    mu_val);
        }
    };

    // The anelastic version of AWP with time-varying attenuation memory
    // vars and the related attenuation constant vars.
    class AwpStencil : public AwpBase {
    public:
        AwpStencil() :
            AwpBase("awp") { }

        // Define the t+1 values for all velocity and stress vars.
        virtual void define() override {
            define_base();

            // Define memory components.
            stress_mem_xx(t+1, x, y, z) EQUALS get_next_stress_mem_xx(x, y, z);
            stress_mem_yy(t+1, x, y, z) EQUALS get_next_stress_mem_yy(x, y, z);
            stress_mem_xy(t+1, x, y, z) EQUALS get_next_stress_mem_xy(x, y, z);
            stress_mem_xz(t+1, x, y, z) EQUALS get_next_stress_mem_xz(x, y, z);
            stress_mem_yz(t+1, x, y, z) EQUALS get_next_stress_mem_yz(x, y, z);
            stress_mem_zz(t+1, x, y, z) EQUALS get_next_stress_mem_zz(x, y, z);

            set_configs();
        }
    };

    // The anelastic version of AWP with free-surface absorbing boundary conditions.
    class AwpABCStencil : public AwpBase {
    public:
        AwpABCStencil() :
            AwpBase("awp_abc") { }

        // Define the t+1 values for all velocity and stress vars.
        virtual void define() override {
            define_base_abc();

            // Define memory components using same sub-domains as the stress vars.
            for (auto cond : { BELOW_SURFACE, AT_SURFACE, ONE_ABOVE_SURFACE, TWO_ABOVE_SURFACE }) {
                stress_mem_xx(t+1, x, y, z) EQUALS get_next_stress_mem_xx(x, y, z) IF_DOMAIN cond;
                stress_mem_yy(t+1, x, y, z) EQUALS get_next_stress_mem_yy(x, y, z) IF_DOMAIN cond;
                stress_mem_xy(t+1, x, y, z) EQUALS get_next_stress_mem_xy(x, y, z) IF_DOMAIN cond;
                stress_mem_xz(t+1, x, y, z) EQUALS get_next_stress_mem_xz(x, y, z) IF_DOMAIN cond;
                stress_mem_yz(t+1, x, y, z) EQUALS get_next_stress_mem_yz(x, y, z) IF_DOMAIN cond;
                stress_mem_zz(t+1, x, y, z) EQUALS get_next_stress_mem_zz(x, y, z) IF_DOMAIN cond;
            }
            set_configs();
        }
    };
   
    // Create objects for all AWP stencils,
    // making them available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static AwpElasticStencil AwpElasticStencil_instance;
    static AwpElasticABCStencil AwpElasticABCStencil_instance;
    static AwpStencil AwpStencil_instance;
    static AwpABCStencil AwpABCStencil_instance;

} // namespace.
