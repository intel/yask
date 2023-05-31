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

// Stencil equations for 2D shallow water equations.
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
    using yb=yc_bool_node_ptr;

    // Options for using scratch vars vs. functions.
    // Can only do this for scratch vars that don't
    // use sub-domain conditions.
    //#define USE_H_VAR
    //#define USE_HU_HV_VARS

    // Option for using misc dims vs. separate vars.
    //#define USE_MISC_DIMS

    // Option for using inverses instead of division.
    #define USE_INV

    // Fill all unused elements in scratch vars with zeros. Another way to
    // achieve this is to use the -init_scratch_vars option, but that causes
    // ~20% slowdown because most elements of the scratch vars get written
    // twice.
    #define NEED_ZEROS

    // Two options for implementing the intermediate H value.
    #define H_eq(x, y) (vpt(e, t, x, y) + h(x, y))
    #ifdef USE_H_VAR
    // Use scratch var.
    #define H_val(x, y) H(x, y)
    #define DEFINE_H(x, y) H(x, y) EQUALS H_eq(x, y)
    #else
    // Calculate H on the fly.
    #define H_val(x, y) H_eq(x, y)
    #define DEFINE_H(x, y)
    #endif
    
    // Two options for implementing the intermediate Hu and Hv values.
    #define Hu_eq(x, y) (0.5*(H_val(x, y) + H_val(x-1, y)) * vpt(u, t, x, y))
    #define Hv_eq(x, y) (0.5*(H_val(x, y) + H_val(x, y-1)) * vpt(v, t, x, y))
    #ifdef USE_HU_HV_VARS
    // Use scratch vars.
    #define Hu_val(x, y) Hu(x, y)
    #define Hv_val(x, y) Hv(x, y)
    #define DEFINE_HU_HV(x, y) \
        Hu(x, y) EQUALS Hu_eq(x, y);            \
        Hv(x, y) EQUALS Hv_eq(x, y)
    #else
    // Calculate Hu and Hv on the fly.
    #define Hu_val(x, y) Hu_eq(x, y)
    #define Hv_val(x, y) Hv_eq(x, y)
    #define DEFINE_HU_HV(x, y)
    #endif

    // Two options for implementing the q[abgd] vars.
    #ifdef USE_MISC_DIMS
    #define qa_val(x, y) q_greeks(1, x, y)
    #define qb_val(x, y) q_greeks(2, x, y)
    #define qg_val(x, y) q_greeks(3, x, y)
    #define qd_val(x, y) q_greeks(4, x, y)
    #else
    #define qa_val(x, y) qa(x, y)
    #define qb_val(x, y) qb(x, y)
    #define qg_val(x, y) qg(x, y)
    #define qd_val(x, y) qd(x, y)
    #endif

    // Inverses vs. division.
    #ifdef USE_INV
    #define DIV_DX * inv_dx
    #define DIV_DY * inv_dy
    #else
    #define DIV_DX / dx
    #define DIV_DY / xy
    #endif
    
    class SWE2dStencil : public yc_solution_base {
    public:
        SWE2dStencil(string name="swe2d") :
            yc_solution_base(name) { }

        // An x-y var is a scratch var if it doesn't have a 't' dim.
        bool is_scratch(yv& vp) {
            return vp.get_var()->get_num_dims() == 2;
        }

        // Simple alias for referencing a point in either a scratch or
        // non-scratch var.
        yvp vpt(yv& var, yn t, yn x, yn y) {
            if (is_scratch(var))
                return var(x, y);
            else
                return var(t, x, y);
        }

        virtual yn compute_dedt(yn t, yv& e, yv& u, yv& v, yv& H, yv& Hu, yv& Hv) {
            auto dHudx = (Hu_val(x+1, y) - Hu_val(x, y)) DIV_DX;
            auto dHvdy = (Hv_val(x, y+1) - Hv_val(x, y)) DIV_DY;
            return -(dHudx + dHvdy);
        }

        virtual yn compute_dudt(yn t, yv& e, yv& u, yv& v, yv& ke, yv& qHv) {
            yn dedx, dkedx;
            dedx = (vpt(e, t, x, y) - vpt(e, t, x-1, y)) DIV_DX;

            // using 't+1' for ke because we want the new value from compute_kinetic_energy().
            dkedx = (vpt(ke, t+1, x, y) - vpt(ke, t+1, x-1, y)) DIV_DX;
            return -g * dedx - dkedx + qHv(x, y);
        }

        virtual yn compute_dvdt(yn t, yv& e, yv& u, yv& v, yv& ke, yv& qHu) {
            yn dedy, dkedy;
            dedy = (vpt(e, t, x, y) - vpt(e, t, x, y-1)) DIV_DY;

            // using 't+1' for ke because we want the new value from compute_kinetic_energy().
            dkedy = (vpt(ke, t+1, x, y) - vpt(ke, t+1, x, y-1)) DIV_DY;
            return -g * dedy - dkedy - qHu(x, y);
        }

        virtual void compute_H(yn t, yv& e, yv& u, yv& v,
                               yv& H, yv& Hf, yv& Hu, yv& Hv) {

            DEFINE_H(x, y);
            DEFINE_HU_HV(x, y);

            // average depth at F points (vertices)
            Hf(x, y) EQUALS 0.25 * (H_val(x, y) + H_val(x-1, y) +
                                    H_val(x, y-1) + H_val(x-1, y-1)) IF_DOMAIN interior_domain;
            // at boundary (vertex) compute average of existing interior (cell center) points
            Hf(x, y) EQUALS 0.5 * (H_val(x, y) + H_val(x, y-1)) IF_DOMAIN x_lo;
            Hf(x, y) EQUALS 0.5 * (H_val(x-1, y) + H_val(x-1, y-1)) IF_DOMAIN x_hi;
            Hf(x, y) EQUALS 0.5 * (H_val(x, y) + H_val(x-1, y)) IF_DOMAIN y_lo;
            Hf(x, y) EQUALS 0.5 * (H_val(x, y-1) + H_val(x-1, y-1)) IF_DOMAIN y_hi;
            Hf(x, y) EQUALS H_val(x, y) IF_DOMAIN x_lo_y_lo;
            Hf(x, y) EQUALS H_val(x-1, y) IF_DOMAIN x_hi_y_lo;
            Hf(x, y) EQUALS H_val(x, y-1) IF_DOMAIN x_lo_y_hi;
            Hf(x, y) EQUALS H_val(x-1, y-1) IF_DOMAIN x_hi_y_hi;

            #ifdef NEED_ZEROS
            // Any remaining points.
            // Must define all possible points for scratch vars, even in halo regions.
            Hf(x, y) EQUALS 0.0 IF_DOMAIN !yask_domain;
            #endif
        }

        virtual void compute_velocity_grad(yn t, yv& u, yv& v,
                                           yv& dudy, yv& dvdx) {

            // only cross terms are needed
            // defined at vertices (F points), only valid in interior points.
            dudy(x, y) EQUALS (vpt(u, t, x, y) - vpt(u, t, x, y-1)) DIV_DY IF_DOMAIN interior_domain;
            dvdx(x, y) EQUALS (vpt(v, t, x, y) - vpt(v, t, x-1, y)) DIV_DX IF_DOMAIN interior_domain;

            #ifdef NEED_ZEROS
            dudy(x, y) EQUALS 0.0 IF_DOMAIN !interior_domain;
            dvdx(x, y) EQUALS 0.0 IF_DOMAIN !interior_domain;
            #endif
        }

        virtual void compute_kinetic_energy(yn t, yv& u, yv& v, yv& ke) {

            // kinetic energy, ke = 1/2 |u|^2
            // average for U/V points to T points (cell center)
            yn keexpr = vpt(u, t, x+1, y) * vpt(u, t, x+1, y) +
                vpt(u, t, x, y) * vpt(u, t, x, y) +
                vpt(v, t, x, y+1) * vpt(v, t, x, y+1) +
                vpt(v, t, x, y) * vpt(v, t, x, y);

            // using 't+1' for ke because this is a new value.
            vpt(ke, t+1, x, y) EQUALS 0.25 * keexpr IF_DOMAIN e_interior_domain;

            #ifdef NEED_ZEROS
            if (is_scratch(ke))
                vpt(ke, t+1, x, y) EQUALS 0.0 IF_DOMAIN !e_interior_domain;
            #endif
        }

        virtual void compute_q(yn t,
                               yv& dudy, yv& dvdx, yv& Hf,
                               yv& q, yv& qa, yv& qb, yv& qg, yv& qd, yv& q_greeks) {
            double w = 1./12;
            
            // potential vorticity (at F points).
            // using 't+1' for q because this is a new value.
            vpt(q, t+1, x, y) EQUALS (coriolis - dudy(x, y) + dvdx(x, y)) / Hf(x, y) IF_DOMAIN yask_domain;

            // Define alpha, beta, gamma, delta for each cell (at T points)
            // using 't+1' for q because we want the new value from above.
            qa_val(x, y) EQUALS w * (vpt(q, t+1, x, y) + vpt(q, t+1, x, y+1) + vpt(q, t+1, x+1, y+1)) IF_DOMAIN yask_domain;
            qb_val(x, y) EQUALS w * (vpt(q, t+1, x+1, y) + vpt(q, t+1, x, y+1) + vpt(q, t+1, x+1, y+1)) IF_DOMAIN yask_domain;
            qg_val(x, y) EQUALS w * (vpt(q, t+1, x, y) + vpt(q, t+1, x+1, y) + vpt(q, t+1, x+1, y+1)) IF_DOMAIN yask_domain;
            qd_val(x, y) EQUALS w * (vpt(q, t+1, x, y) + vpt(q, t+1, x+1, y) + vpt(q, t+1, x, y+1)) IF_DOMAIN yask_domain;

            #ifdef NEED_ZEROS
            if (is_scratch(q))
                q(x, y) EQUALS 0.0 IF_DOMAIN !yask_domain;
            qa_val(x, y) EQUALS 0.0 IF_DOMAIN !yask_domain;
            qb_val(x, y) EQUALS 0.0 IF_DOMAIN !yask_domain;
            qg_val(x, y) EQUALS 0.0 IF_DOMAIN !yask_domain;
            qd_val(x, y) EQUALS 0.0 IF_DOMAIN !yask_domain;
            #endif
        }

        virtual void compute_pv_advection(yn t, yv& e, yv& u, yv& v,
                                          yv& H, yv& Hu, yv& Hv,
                                          yv& qa, yv& qb, yv& qg, yv& qd, yv& q_greeks,
                                          yv& qHu, yv& qHv) {

            // potential vorticity advection terms
            qHv(x, y) EQUALS (
                qa_val(x, y) * Hv_val(x, y+1) +
                qd_val(x, y) * Hv_val(x, y) +
                qb_val(x-1, y) * Hv_val(x-1, y+1) +
                qg_val(x-1, y) * Hv_val(x-1, y)
            ) IF_DOMAIN u_interior_domain;
            qHu(x, y) EQUALS (
                qg_val(x, y) * Hu_val(x+1, y) +
                qd_val(x, y) * Hu_val(x, y) +
                qa_val(x, y-1) * Hu_val(x, y-1) + 
                qb_val(x, y-1) * Hu_val(x+1, y-1)
            ) IF_DOMAIN v_interior_domain;

            #ifdef NEED_ZEROS
            qHv(x, y) EQUALS 0.0 IF_DOMAIN !u_interior_domain;
            qHu(x, y) EQUALS 0.0 IF_DOMAIN !v_interior_domain;
            #endif
        }

        // One stage of Runge-Kutta time integration.
        virtual void rk_stage(yn t,
                              yv& e_in, yv& u_in, yv& v_in,
                              yv& H_loc, yv& Hf_loc, yv& Hu_loc, yv& Hv_loc,
                              yv& dudy_loc, yv& dvdx_loc, yv& ke_loc,
                              yv& q_loc, yv& qa_loc, yv& qb_loc, yv& qg_loc, yv& qd_loc, yv& q_greeks_loc,
                              yv& qHu_loc, yv& qHv_loc,
                              yn& dedt, yn& dudt, yn& dvdt) {

            compute_H(t, e_in, u_in, v_in, H_loc, Hf_loc, Hu_loc, Hv_loc);
            compute_velocity_grad(t, u_in, v_in, dudy_loc, dvdx_loc);
            compute_kinetic_energy(t, u_in, v_in, ke_loc);
            compute_q(t, dudy_loc, dvdx_loc, Hf_loc,
                      q_loc, qa_loc, qb_loc, qg_loc, qd_loc, q_greeks_loc);
            compute_pv_advection(t, e_in, u_in, v_in, H_loc, Hu_loc, Hv_loc,
                                 qa_loc, qb_loc, qg_loc, qd_loc, q_greeks_loc,
                                 qHu_loc, qHv_loc);
            dedt = compute_dedt(t, e_in, u_in, v_in, H_loc, Hu_loc, Hv_loc);
            dudt = compute_dudt(t, e_in, u_in, v_in, ke_loc, qHv_loc);
            dvdt = compute_dvdt(t, e_in, u_in, v_in, ke_loc, qHu_loc);
        }
        
    protected:

        // Dimensions.
        MAKE_STEP_INDEX(t);
        MAKE_DOMAIN_INDEX(x);
        MAKE_DOMAIN_INDEX(y);
        MAKE_MISC_INDEX(greek_idx);

        // Time-dependent model state variables.
        MAKE_VAR(u, t, x, y);
        MAKE_VAR(v, t, x, y);
        MAKE_VAR(e, t, x, y);
        MAKE_VAR(q, t, x, y);
        MAKE_VAR(ke, t, x, y);
        
        // Intermediate solutions for RK sub-stages.
        MAKE_SCRATCH_VAR(u2, x, y);
        MAKE_SCRATCH_VAR(u3, x, y);
        MAKE_SCRATCH_VAR(v2, x, y);
        MAKE_SCRATCH_VAR(v3, x, y);
        MAKE_SCRATCH_VAR(e2, x, y);
        MAKE_SCRATCH_VAR(e3, x, y);
        MAKE_SCRATCH_VAR(q1, x, y);
        MAKE_SCRATCH_VAR(q2, x, y);
        MAKE_SCRATCH_VAR(q3, x, y);
        MAKE_SCRATCH_VAR(ke1, x, y);
        MAKE_SCRATCH_VAR(ke2, x, y);
        MAKE_SCRATCH_VAR(ke3, x, y);
        
        // Other temporary (scratch) variables
        MAKE_SCRATCH_VAR(dudy1, x, y);
        MAKE_SCRATCH_VAR(dvdx1, x, y);
        MAKE_SCRATCH_VAR(dudy2, x, y);
        MAKE_SCRATCH_VAR(dvdx2, x, y);
        MAKE_SCRATCH_VAR(dudy3, x, y);
        MAKE_SCRATCH_VAR(dvdx3, x, y);
        MAKE_SCRATCH_VAR(H1, x, y);
        MAKE_SCRATCH_VAR(H2, x, y);
        MAKE_SCRATCH_VAR(H3, x, y);
        MAKE_SCRATCH_VAR(Hf1, x, y);
        MAKE_SCRATCH_VAR(Hu1, x, y);
        MAKE_SCRATCH_VAR(Hv1, x, y);
        MAKE_SCRATCH_VAR(Hf2, x, y);
        MAKE_SCRATCH_VAR(Hu2, x, y);
        MAKE_SCRATCH_VAR(Hv2, x, y);
        MAKE_SCRATCH_VAR(Hf3, x, y);
        MAKE_SCRATCH_VAR(Hu3, x, y);
        MAKE_SCRATCH_VAR(Hv3, x, y);
        MAKE_SCRATCH_VAR(qHu1, x, y);
        MAKE_SCRATCH_VAR(qHv1, x, y);
        MAKE_SCRATCH_VAR(qHu2, x, y);
        MAKE_SCRATCH_VAR(qHv2, x, y);
        MAKE_SCRATCH_VAR(qHu3, x, y);
        MAKE_SCRATCH_VAR(qHv3, x, y);

        // Used if USE_MISC_DIMS is defined.
        MAKE_SCRATCH_VAR(q_greeks1, greek_idx, x, y);
        MAKE_SCRATCH_VAR(q_greeks2, greek_idx, x, y);
        MAKE_SCRATCH_VAR(q_greeks3, greek_idx, x, y);

        // Used if USE_MISC_DIMS is not defined.
        MAKE_SCRATCH_VAR(qa1, x, y);
        MAKE_SCRATCH_VAR(qb1, x, y);
        MAKE_SCRATCH_VAR(qg1, x, y);
        MAKE_SCRATCH_VAR(qd1, x, y);
        MAKE_SCRATCH_VAR(qa2, x, y);
        MAKE_SCRATCH_VAR(qb2, x, y);
        MAKE_SCRATCH_VAR(qg2, x, y);
        MAKE_SCRATCH_VAR(qd2, x, y);
        MAKE_SCRATCH_VAR(qa3, x, y);
        MAKE_SCRATCH_VAR(qb3, x, y);
        MAKE_SCRATCH_VAR(qg3, x, y);
        MAKE_SCRATCH_VAR(qd3, x, y);
        
        // Static variables.
        MAKE_VAR(h, x, y);

        // Physical grid spacing in time and space
        MAKE_SCALAR_VAR(dt);
        MAKE_SCALAR_VAR(dx);
        MAKE_SCALAR_VAR(dy);
        MAKE_SCALAR_VAR(inv_dx);
        MAKE_SCALAR_VAR(inv_dy);

        // Constant coefficients
        MAKE_SCALAR_VAR(g);  // gravitational acceleration
        MAKE_SCALAR_VAR(coriolis);  // coriolis parameter

        // Add a scalar for the data-export time index.
        MAKE_SCALAR_VAR(ti_exp);
        
        // Define interior domains.
        yn x_min = first_domain_index(x);
        yn x_max = last_domain_index(x);
        yn y_min = first_domain_index(y);
        yn y_max = last_domain_index(y);

        /* Sub-domains:

             x_min  x_max
              |      |
              v      v
             0000000000
             0733333390  <-- y_max
             0411111150
             0411111150
             0411111150
             0622222280  <-- y_min
             0000000000

             1+...+9: yask_domain
             0: !yask_domain (halos & other padding)

             1: interior_domain
             2: y_lo
             3: y_hi
             4: x_lo
             5: x_hi
             6: x_lo_y_lo
             7: x_lo_y_hi
             8: x_hi_y_lo
             9: x_hi_y_hi

             1+2: u_interior_domain
             1+4: v_interior_domain
             1+2+4+6: e_interior_domain
        */
        
        // YASK's definition of "inside the domain."
        yb yask_domain =        (x >= x_min) && (x <= x_max) && (y >= y_min) && (y <= y_max);

        // YASK domain minus a one-point boundary all around.
        yb interior_domain =    (x >  x_min) && (x <  x_max) && (y >  y_min) && (y <  y_max);

        // Edges and corners.
        yb x_lo = (x == x_min) && (y > y_min) && (y < y_max);
        yb x_hi = (x == x_max) && (y > y_min) && (y < y_max);
        yb y_lo = (y == y_min) && (x > x_min) && (x < x_max);
        yb y_hi = (y == y_max) && (x > x_min) && (x < x_max);
        yb x_lo_y_lo = (x == x_min) && (y == y_min);
        yb x_hi_y_lo = (x == x_max) && (y == y_min);
        yb x_lo_y_hi = (x == x_min) && (y == y_max);
        yb x_hi_y_hi = (x == x_max) && (y == y_max);

        // u grid (nx+1, ny); BC: u(0, :) = u(nx, :) = 0
        yb u_interior_domain =  (x >  x_min) && (x <  x_max) && (y >= y_min) && (y <  y_max);
        // v grid (nx, ny+1); BC: v(:, 0) = v(:, nx) = 0
        yb v_interior_domain =  (x >= x_min) && (x <  x_max) && (y >  y_min) && (y <  y_max);
        // e grid (nx, ny); BC: none
        yb e_interior_domain =  (x >= x_min) && (x <  x_max) && (y >= y_min) && (y <  y_max);

    public:

        // Compute state at time t+1 based on values at t.
        virtual void define() {
            yn dedt, dudt, dvdt;

            // SSPRK(3, 3) 3-stage Runge-Kutta time integration
            rk_stage(t, e, u, v,
                     H1, Hf1, Hu1, Hv1, dudy1, dvdx1, ke1,
                     q1, qa1, qb1, qg1, qd1, q_greeks1, qHu1, qHv1,
                     dedt, dudt, dvdt);
            e2(x, y) EQUALS e(t, x, y) + dt * dedt IF_DOMAIN e_interior_domain;
            u2(x, y) EQUALS u(t, x, y) + dt * dudt IF_DOMAIN u_interior_domain;
            v2(x, y) EQUALS v(t, x, y) + dt * dvdt IF_DOMAIN v_interior_domain;

            #ifdef NEED_ZEROS
            e2(x, y) EQUALS 0.0 IF_DOMAIN !e_interior_domain;
            u2(x, y) EQUALS 0.0 IF_DOMAIN !u_interior_domain;
            v2(x, y) EQUALS 0.0 IF_DOMAIN !v_interior_domain;
            #endif

            rk_stage(t, e2, u2, v2,
                     H2, Hf2, Hu2, Hv2, dudy2, dvdx2, ke2,
                     q2, qa2, qb2, qg2, qd2, q_greeks2, qHu2, qHv2,
                     dedt, dudt, dvdt);
            e3(x, y) EQUALS 0.75*e(t, x, y) + 0.25*(e2(x, y) + dt * dedt) IF_DOMAIN e_interior_domain;
            u3(x, y) EQUALS 0.75*u(t, x, y) + 0.25*(u2(x, y) + dt * dudt) IF_DOMAIN u_interior_domain;
            v3(x, y) EQUALS 0.75*v(t, x, y) + 0.25*(v2(x, y) + dt * dvdt) IF_DOMAIN v_interior_domain;

            #ifdef NEED_ZEROS
            e3(x, y) EQUALS 0.0 IF_DOMAIN !e_interior_domain;
            u3(x, y) EQUALS 0.0 IF_DOMAIN !u_interior_domain;
            v3(x, y) EQUALS 0.0 IF_DOMAIN !v_interior_domain;
            #endif

            rk_stage(t, e3, u3, v3,
                     H3, Hf3, Hu3, Hv3, dudy3, dvdx3, ke3,
                     q3, qa3, qb3, qg3, qd3, q_greeks3, qHu3, qHv3,
                     dedt, dudt, dvdt);
            e(t+1, x, y) EQUALS 1.0/3.0*e(t, x, y) + 2.0/3.0*(e3(x, y) + dt * dedt) IF_DOMAIN e_interior_domain;
            u(t+1, x, y) EQUALS 1.0/3.0*u(t, x, y) + 2.0/3.0*(u3(x, y) + dt * dudt) IF_DOMAIN u_interior_domain;
            v(t+1, x, y) EQUALS 1.0/3.0*v(t, x, y) + 2.0/3.0*(v3(x, y) + dt * dvdt) IF_DOMAIN v_interior_domain;

            // Copy the final values from the scratch vars.
            // Copy only when needed as defined by 'ti_exp'.
            ke(t+1, x, y) EQUALS ke3(x, y) IF_STEP t >= ti_exp;
            q(t+1, x, y) EQUALS q3(x, y) IF_STEP t >= ti_exp;
        }

    protected:


    };
    REGISTER_SOLUTION(SWE2dStencil);
        
} // namespace.
