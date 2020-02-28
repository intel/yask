/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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

// Common definitions for Elastic stencils.

#pragma once

#include "ElasticStencil.hpp"

// This class extends ElasticStencilBase to provide methods to make
// stencils with merged vars.
class Elastic2StencilBase : public ElasticStencilBase {

protected:

    // yc_var_proxy selectors.
    yc_index_node_ptr vidx = new_misc_index("vidx");
    yc_index_node_ptr sidx = new_misc_index("sidx");
    yc_index_node_ptr cidx = new_misc_index("cidx");

public:
    Elastic2StencilBase(const string& name,
                        ElasticBoundaryCondition *_bc = NULL) :
        ElasticStencilBase(name, _bc) { }

    yc_number_node_ptr stencil_O8_Z(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const int offset)
    {
        return
            (c0_8 * (g(t,x,y,z  +offset, gidx)  -
                     g(t,x,y,z-1+offset, gidx)) +
             c1_8 * (g(t,x,y,z+1+offset, gidx)  -
                     g(t,x,y,z-2+offset, gidx)) +
             c2_8 * (g(t,x,y,z+2+offset, gidx)  -
                     g(t,x,y,z-3+offset, gidx)) +
             c3_8 * (g(t,x,y,z+3+offset, gidx)  -
                     g(t,x,y,z-4+offset, gidx)))*dzi;
    }

    yc_number_node_ptr stencil_O8(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                         yc_var_proxy &g, yc_number_any_arg gidx, const Z, const B)
    {
        return stencil_O8_Z(t, x, y, z, g, gidx, 0);
    }

    yc_number_node_ptr stencil_O8(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                         yc_var_proxy &g, yc_number_any_arg gidx, const Z, const F)
    {
        return stencil_O8_Z(t, x, y, z, g, gidx, 1);
    }

    yc_number_node_ptr stencil_O8_Y(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const int offset)
    {
        return
            (c0_8 * (g(t,x,y  +offset,z, gidx)  -
                     g(t,x,y-1+offset,z, gidx)) +
             c1_8 * (g(t,x,y+1+offset,z, gidx)  -
                     g(t,x,y-2+offset,z, gidx)) +
             c2_8 * (g(t,x,y+2+offset,z, gidx)  -
                     g(t,x,y-3+offset,z, gidx)) +
             c3_8 * (g(t,x,y+3+offset,z, gidx)  -
                     g(t,x,y-4+offset,z, gidx)))*dyi;
    }

    yc_number_node_ptr stencil_O8(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                         yc_var_proxy &g, yc_number_any_arg gidx, const Y, const B)
    {
        return stencil_O8_Y(t, x, y, z, g, gidx, 0);
    }

    yc_number_node_ptr stencil_O8(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z, yc_var_proxy &g, yc_number_any_arg gidx, const Y, const F)
    {
        return stencil_O8_Y(t, x, y, z, g, gidx, 1);
    }

    yc_number_node_ptr stencil_O8_X(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const int offset)
    {
        return
            (c0_8 * (g(t,x  +offset,y,z, gidx)  -
                     g(t,x-1+offset,y,z, gidx)) +
             c1_8 * (g(t,x+1+offset,y,z, gidx)  -
                     g(t,x-2+offset,y,z, gidx)) +
             c2_8 * (g(t,x+2+offset,y,z, gidx)  -
                     g(t,x-3+offset,y,z, gidx)) +
             c3_8 * (g(t,x+3+offset,y,z, gidx)  -
                     g(t,x-4+offset,y,z, gidx)))*dxi;
    }

    yc_number_node_ptr stencil_O8(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                         yc_var_proxy &g, yc_number_any_arg gidx, const X, const B)
    {
        return stencil_O8_X(t, x, y, z, g, gidx, 0);
    }

    yc_number_node_ptr stencil_O8(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                         yc_var_proxy &g, yc_number_any_arg gidx,
                         const X, const F)
    {
        return stencil_O8_X(t, x, y, z, g, gidx, 1);
    }

    template<typename Dim, typename Dir>
    yc_number_node_ptr stencil_O8(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                         yc_var_proxy &g, yc_number_any_arg gidx)
    {
        return stencil_O8(t, x, y, z, g, gidx, Dim(), Dir());
    }

    // Velocity-var define functions.  For each D in x, y, z, define vel_D
    // at t+1 based on vel_x at t and stress vars at t.  Note that the t,
    // x, y, z parameters are integer var indices, not actual offsets in
    // time or space, so half-steps due to staggered vars are adjusted
    // appropriately.

    template<typename N, typename SZ, typename SX, typename SY>
    void define_vel(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                    yc_var_proxy& v, yc_number_any_arg vidx,
                    yc_var_proxy& s, yc_number_any_arg sx_idx, yc_number_any_arg sy_idx, yc_number_any_arg sz_idx) {

        yc_number_node_ptr lrho   = interp_rho<N>(x, y, z);

        yc_number_node_ptr stx    = stencil_O8<X,SX>(t, x, y, z, s, sx_idx);
        yc_number_node_ptr sty    = stencil_O8<Y,SY>(t, x, y, z, s, sy_idx);
        yc_number_node_ptr stz    = stencil_O8<Z,SZ>(t, x, y, z, s, sz_idx);

        yc_number_node_ptr next_v = v(t, x, y, z, vidx) + ((stx + sty + stz) * delta_t * lrho);

        // define the value at t+1.
        if (hasBoundaryCondition()) {
            yc_bool_node_ptr not_at_bc = bc->is_not_at_boundary();
            v(t+1, x, y, z, vidx) EQUALS next_v IF_DOMAIN not_at_bc;
        } else
            v(t+1, x, y, z, vidx) EQUALS next_v;
    }
    
    yc_number_node_ptr stencil_O2_Z(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const int offset)
    {
        return
            (g(t,x,y,z, gidx)  -
             g(t,x,y,z+offset, gidx))*dzi;
    }

    yc_number_node_ptr stencil_O2_Z(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const B)
    {
        return stencil_O2_Z(t, x, y, z, g, gidx, -1);
    }

    yc_number_node_ptr stencil_O2_Z(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const F)
    {
        return stencil_O2_Z(t, x, y, z, g, gidx, 1);
    }

    template<typename D>
    yc_number_node_ptr stencil_O2_Z(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx)
    {
        return stencil_O2_Z(t, x, y, z, g, gidx, D());
    }

    yc_number_node_ptr stencil_O2_Y(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const int offset)
    {
        return
            (g(t,x,y       ,z, gidx)  -
             g(t,x,y+offset,z, gidx))*dyi;
    }

    yc_number_node_ptr stencil_O2_Y(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const B)
    {
        return stencil_O2_Y(t, x, y, z, g, gidx,-1);
    }

    yc_number_node_ptr stencil_O2_Y(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const F)
    {
        return stencil_O2_Y(t, x, y, z, g, gidx, 1);
    }

    template<typename D>
    yc_number_node_ptr stencil_O2_Y(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx)
    {
        return stencil_O2_Y(t, x, y, z, g, gidx, D());
    }

    yc_number_node_ptr stencil_O2_X(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const int offset)
    {
        return
            (g(t,x       ,y,z, gidx)  -
             g(t,x+offset,y,z, gidx))*dxi;
    }

    yc_number_node_ptr stencil_O2_X(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const B)
    {
        return stencil_O2_X(t, x, y, z, g, gidx,-1);
    }

    yc_number_node_ptr stencil_O2_X(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx, const F)
    {
        return stencil_O2_X(t, x, y, z, g, gidx, 1);
    }

    template<typename D>
    yc_number_node_ptr stencil_O2_X(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                           yc_var_proxy &g, yc_number_any_arg gidx)
    {
        return stencil_O2_X(t, x, y, z, g, gidx, D());
    }
};

