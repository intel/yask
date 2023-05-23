/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

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

// Stencil equations for 2D linear wave equation.
// Written with variables for elevation, e, and 2d velocity, (u, v).
// Contributed by Tuomas Karna, Intel Corp.

// YASK stencil solution(s) in this file will be integrated into the YASK compiler utility.
#include "yask_compiler_api.hpp"
using namespace std;
using namespace yask;


// Create an anonymous namespace to ensure that types are local.
namespace {

    using yv=yc_var_proxy;
    using yn=yc_number_node_ptr;
    using yvp=yc_var_point_node_ptr;
    
    class Wave2dStencil : public yc_solution_base {
    public:
        Wave2dStencil(string name="wave2d") :
            yc_solution_base(name) { }

        virtual yn compute_dedt(yn t,
                                yn x, yn y,
                                yv e, yv u, yv v,
                                yn g, yn depth,
                                yn dx, yn dy) {
            auto dudx = (vpt(u, t, x+1, y) - vpt(u, t, x, y)) / dx;
            auto dvdy = (vpt(v, t, x, y+1) - vpt(v, t, x, y)) / dy;
            return -depth * (dudx + dvdy);
        }

        virtual yn compute_dudt(yn t,
                                yn x, yn y,
                                yv e, yv u, yv v,
                                yn g, yn depth,
                                yn dx, yn dy) {
            auto dedx = (vpt(e, t, x, y) - vpt(e, t, x-1, y)) / dx;
            return -g * dedx;
        }

        virtual yn compute_dvdt(yn t,
                                yn x, yn y,
                                yv e, yv u, yv v,
                                yn g, yn depth,
                                yn dx, yn dy) {
            auto dedy = (vpt(e, t, x, y) - vpt(e, t, x, y-1)) / dy;
            return -g * dedy;
        }

        // Compute state at time t+1 based on values at t.
        virtual void define() {

            // Define interior domains
            // Full domain size (nx+1, ny+1)
            auto x_min = first_domain_index(x);
            auto x_max = last_domain_index(x);
            auto y_min = first_domain_index(y);
            auto y_max = last_domain_index(y);
            // u grid (nx+1, ny); BC: u(0, :) = u(nx, :) = 0
            auto u_interior_domain = (x > x_min) && (x < x_max) &&                (y < y_max);
            // v grid (nx, ny+1); BC: v(:, 0) = v(:, nx) = 0
            auto v_interior_domain =                (x < x_max) && (y > y_min) && (y < y_max);
            // e grid (nx, ny); BC: none
            auto e_interior_domain =                (x < x_max) &&                (y < y_max);

            // SSPRK(3, 3) 3-stage Runge-Kutta time integration
            // stage 1
            auto dedt = compute_dedt(t, x, y, e, u, v, g, depth, dx, dy);
            auto dudt = compute_dudt(t, x, y, e, u, v, g, depth, dx, dy);
            auto dvdt = compute_dvdt(t, x, y, e, u, v, g, depth, dx, dy);
            e_sub1(x, y) EQUALS e(t, x, y) + dt * dedt IF_DOMAIN  e_interior_domain;
            u_sub1(x, y) EQUALS u(t, x, y) + dt * dudt IF_DOMAIN  u_interior_domain;
            v_sub1(x, y) EQUALS v(t, x, y) + dt * dvdt IF_DOMAIN  v_interior_domain;

            // Must specify values for all possible points in scratch vars.
            e_sub1(x, y) EQUALS 0.0 IF_DOMAIN !e_interior_domain;
            u_sub1(x, y) EQUALS 0.0 IF_DOMAIN !u_interior_domain;
            v_sub1(x, y) EQUALS 0.0 IF_DOMAIN !v_interior_domain;

            // stage 2
            dedt = compute_dedt(0, x, y, e_sub1, u_sub1, v_sub1, g, depth, dx, dy);
            dudt = compute_dudt(0, x, y, e_sub1, u_sub1, v_sub1, g, depth, dx, dy);
            dvdt = compute_dvdt(0, x, y, e_sub1, u_sub1, v_sub1, g, depth, dx, dy);
            e_sub2(x, y) EQUALS 0.75*e(t, x, y) + 0.25*(e_sub1(x, y) + dt * dedt) IF_DOMAIN  e_interior_domain;
            u_sub2(x, y) EQUALS 0.75*u(t, x, y) + 0.25*(u_sub1(x, y) + dt * dudt) IF_DOMAIN  u_interior_domain;
            v_sub2(x, y) EQUALS 0.75*v(t, x, y) + 0.25*(v_sub1(x, y) + dt * dvdt) IF_DOMAIN  v_interior_domain;

            // Must specify values for all possible points in scratch vars.
            e_sub2(x, y) EQUALS 0.0 IF_DOMAIN !e_interior_domain;
            u_sub2(x, y) EQUALS 0.0 IF_DOMAIN !u_interior_domain;
            v_sub2(x, y) EQUALS 0.0 IF_DOMAIN !v_interior_domain;
            
            // final solution
            dedt = compute_dedt(0, x, y, e_sub2, u_sub2, v_sub2, g, depth, dx, dy);
            dudt = compute_dudt(0, x, y, e_sub2, u_sub2, v_sub2, g, depth, dx, dy);
            dvdt = compute_dvdt(0, x, y, e_sub2, u_sub2, v_sub2, g, depth, dx, dy);
            e(t+1, x, y) EQUALS 1.0/3.0*e(t, x, y) + 2.0/3.0*(e_sub2(x, y) + dt * dedt) IF_DOMAIN e_interior_domain;
            u(t+1, x, y) EQUALS 1.0/3.0*u(t, x, y) + 2.0/3.0*(u_sub2(x, y) + dt * dudt) IF_DOMAIN u_interior_domain;
            v(t+1, x, y) EQUALS 1.0/3.0*v(t, x, y) + 2.0/3.0*(v_sub2(x, y) + dt * dvdt) IF_DOMAIN v_interior_domain;

        }
        
    protected:

        // Simple alias for referencing a point in either a scratch (2D) or
        // non-scratch (t+2D) var.
        yvp vpt(yv& var, yn t, yn x, yn y) {
            if (var.get_var()->get_num_dims() == 2)
                return var(x, y);
            else
                return var(t, x, y);
        }
        
        // Dimensions
        MAKE_STEP_INDEX(t);
        MAKE_DOMAIN_INDEX(x);
        MAKE_DOMAIN_INDEX(y);

        // Time-dependent model state variables
        MAKE_VAR(u, t, x, y);
        MAKE_VAR(v, t, x, y);
        MAKE_VAR(e, t, x, y);

        // Scratch vars for Runge-Kutta sub-stages.
        MAKE_SCRATCH_VAR(u_sub1, x, y);
        MAKE_SCRATCH_VAR(v_sub1, x, y);
        MAKE_SCRATCH_VAR(e_sub1, x, y);
        MAKE_SCRATCH_VAR(u_sub2, x, y);
        MAKE_SCRATCH_VAR(v_sub2, x, y);
        MAKE_SCRATCH_VAR(e_sub2, x, y);

        // Physical grid spacing in time and space
        MAKE_SCALAR_VAR(dt);
        MAKE_SCALAR_VAR(dx);
        MAKE_SCALAR_VAR(dy);

        // Constant coefficients
        MAKE_SCALAR_VAR(g);  // gravitational acceleration
        MAKE_SCALAR_VAR(depth);  // water depth

    };
    REGISTER_SOLUTION(Wave2dStencil);

} // namespace.
