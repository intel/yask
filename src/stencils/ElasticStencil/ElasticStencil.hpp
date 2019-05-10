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

// Common definitions for Elastic stencils.

#pragma once

// YASK stencil solution(s) in this file will be integrated into the YASK compiler utility.
#include "yask_compiler_utility_api.hpp"
using namespace std;
using namespace yask;

struct Node {};
struct TL: public Node {};
struct TR: public Node {};
struct BL: public Node {};
struct BR: public Node {};

struct StencilDirection {};
struct F: public StencilDirection {};
struct B: public StencilDirection {};

struct StencilDimension{};
struct X: public StencilDimension{};
struct Y: public StencilDimension{};
struct Z: public StencilDimension{};

// This class implements yc_solution_base but is not the main solution.
// The main solution is provided during construction.
class ElasticBoundaryCondition : public yc_solution_base
{
protected:

    // Dimensions.
    yc_index_node_ptr t = new_step_index("t");           // step in time dim.
    yc_index_node_ptr x = new_domain_index("x");         // spatial dim.
    yc_index_node_ptr y = new_domain_index("y");         // spatial dim.
    yc_index_node_ptr z = new_domain_index("z");         // spatial dim.

public:
    ElasticBoundaryCondition(yc_solution_base& base) :
        yc_solution_base(base) { }
    virtual ~ElasticBoundaryCondition() {}

    // Determine whether current indices are at boundary.
    virtual yc_bool_node_ptr is_at_boundary() =0;
    virtual yc_bool_node_ptr is_not_at_boundary() =0;
};

// This class implements yc_solution_base but is not the main solution.
// The main solution is provided during construction.
class ElasticStencilBase : public yc_solution_base {

protected:

    // Dimensions.
    yc_index_node_ptr t = new_step_index("t");           // step in time dim.
    yc_index_node_ptr x = new_domain_index("x");         // spatial dim.
    yc_index_node_ptr y = new_domain_index("y");         // spatial dim.
    yc_index_node_ptr z = new_domain_index("z");         // spatial dim.

    // 3D-spatial coefficients.
    yc_var_proxy rho = yc_var_proxy("rho", get_soln(), { x, y, z });

    // Spatial FD coefficients.
    const double c0_8 = 1.2;
    const double c1_8 = 1.4;
    const double c2_8 = 1.6;
    const double c3_8 = 1.8;

    // Physical dimensions in time and space.
    const double delta_t = 0.002452;

    // Inverse of discretization.
    const double dxi = 36.057693;
    const double dyi = 36.057693;
    const double dzi = 36.057693;

    ElasticBoundaryCondition *bc = NULL;

public:
    ElasticStencilBase(const string& name, 
                       ElasticBoundaryCondition *_bc = NULL) :
        yc_solution_base(name), bc(_bc)
    { }

    bool hasBoundaryCondition()
    {
        return bc != NULL;
    }

    yc_number_node_ptr interp_rho( yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z,
                                   const TL )
    {
        return ( 2.0/ (rho(x  , y  , z  ) +
                       rho(x+1, y  , z  )) );
    }

    yc_number_node_ptr interp_rho( yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z,
                                   const TR )
    {
        return ( 2.0/ (rho(x  , y  , z  ) +
                       rho(x  , y+1, z  )) );
    }

    yc_number_node_ptr interp_rho( yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z,
                                   const BL )
    {
        return ( 2.0/ (rho(x  , y  , z  ) +
                       rho(x  , y  , z+1)) );
    }

    yc_number_node_ptr interp_rho( yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z,
                                   const BR )
    {
        return ( 8.0/ (rho(x  , y  , z  ) +
                       rho(x  , y  , z+1) +
                       rho(x  , y+1, z  ) +
                       rho(x+1, y  , z  ) +
                       rho(x+1, y+1, z  ) +
                       rho(x  , y+1, z+1) +
                       rho(x+1, y  , z+1) +
                       rho(x+1, y+1, z+1)) );
    }

    // Call the interp_rho() function above depending on N.
    template<typename N>
    yc_number_node_ptr interp_rho( yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z )
    {
        return interp_rho( x, y, z, N() );
    }

    yc_number_node_ptr stencil_O8_Z( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const int offset )
    {
        return
            (c0_8 * (g(t,x,y,z  +offset)  -
                     g(t,x,y,z-1+offset)) +
             c1_8 * (g(t,x,y,z+1+offset)  -
                     g(t,x,y,z-2+offset)) +
             c2_8 * (g(t,x,y,z+2+offset)  -
                     g(t,x,y,z-3+offset)) +
             c3_8 * (g(t,x,y,z+3+offset)  -
                     g(t,x,y,z-4+offset)))*dzi;
    }

    yc_number_node_ptr stencil_O8( yc_number_node_ptr t,
                                   yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z,
                                   yc_var_proxy &g,
                                   const Z, const B )
    {
        return stencil_O8_Z( t, x, y, z, g, 0 );
    }

    yc_number_node_ptr stencil_O8( yc_number_node_ptr t,
                                   yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z,
                                   yc_var_proxy &g,
                                   const Z, const F )
    {
        return stencil_O8_Z( t, x, y, z, g, 1 );
    }

    yc_number_node_ptr stencil_O8_Y( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const int offset )
    {
        return
            (c0_8 * (g(t,x,y  +offset,z)  -
                     g(t,x,y-1+offset,z)) +
             c1_8 * (g(t,x,y+1+offset,z)  -
                     g(t,x,y-2+offset,z)) +
             c2_8 * (g(t,x,y+2+offset,z)  -
                     g(t,x,y-3+offset,z)) +
             c3_8 * (g(t,x,y+3+offset,z)  -
                     g(t,x,y-4+offset,z)))*dyi;
    }

    yc_number_node_ptr stencil_O8( yc_number_node_ptr t,
                                   yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z,
                                   yc_var_proxy &g,
                                   const Y, const B )
    {
        return stencil_O8_Y( t, x, y, z, g, 0 );
    }

    yc_number_node_ptr stencil_O8( yc_number_node_ptr t,
                                   yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z,
                                   yc_var_proxy &g,
                                   const Y, const F )
    {
        return stencil_O8_Y( t, x, y, z, g, 1 );
    }

    yc_number_node_ptr stencil_O8_X( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const int offset )
    {
        return
            (c0_8 * (g(t,x  +offset,y,z)  -
                     g(t,x-1+offset,y,z)) +
             c1_8 * (g(t,x+1+offset,y,z)  -
                     g(t,x-2+offset,y,z)) +
             c2_8 * (g(t,x+2+offset,y,z)  -
                     g(t,x-3+offset,y,z)) +
             c3_8 * (g(t,x+3+offset,y,z)  -
                     g(t,x-4+offset,y,z)))*dxi;
    }

    yc_number_node_ptr stencil_O8( yc_number_node_ptr t,
                                   yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z,
                                   yc_var_proxy &g,
                                   const X, const B )
    {
        return stencil_O8_X( t, x, y, z, g, 0 );
    }

    yc_number_node_ptr stencil_O8( yc_number_node_ptr t,
                                   yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z,
                                   yc_var_proxy &g,
                                   const X, const F )
    {
        return stencil_O8_X( t, x, y, z, g, 1 );
    }

    // Call the stencil_O8() function above depending on Dim & Dir.
    template<typename Dim, typename Dir>
    yc_number_node_ptr stencil_O8( yc_number_node_ptr t,
                                   yc_number_node_ptr x,
                                   yc_number_node_ptr y,
                                   yc_number_node_ptr z,
                                   yc_var_proxy &g )
    {
        return stencil_O8( t, x, y, z, g, Dim(), Dir() );
    }

    // Velocity-var define functions.  For each D in x, y, z, define vel_D
    // at t+1 based on vel_x at t and stress vars at t.  Note that the t,
    // x, y, z parameters are integer var indices, not actual offsets in
    // time or space, so half-steps due to staggered vars are adjusted
    // appropriately.

    template<typename N, typename SZ, typename SX, typename SY>
    void define_vel(yc_number_node_ptr t,
                    yc_number_node_ptr x,
                    yc_number_node_ptr y,
                    yc_number_node_ptr z,
                    yc_var_proxy &v,
                    yc_var_proxy &sx,
                    yc_var_proxy &sy,
                    yc_var_proxy &sz) {

        auto lrho   = interp_rho<N>( x, y, z );

        auto stx    = stencil_O8<X,SX>( t, x, y, z, sx );
        auto sty    = stencil_O8<Y,SY>( t, x, y, z, sy );
        auto stz    = stencil_O8<Z,SZ>( t, x, y, z, sz );

        auto next_v = v(t, x, y, z) + ((stx + sty + stz) * delta_t * lrho);

        // define the value at t+1.
        if ( hasBoundaryCondition() ) {
            yc_bool_node_ptr not_at_bc = bc->is_not_at_boundary();
            v(t+1, x, y, z) EQUALS next_v IF_DOMAIN not_at_bc;
        } else
            v(t+1, x, y, z) EQUALS next_v;
    }

    yc_number_node_ptr stencil_O2_Z( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const int offset )
    {
        return
            (g(t,x,y,z       )  -
             g(t,x,y,z+offset))*dzi;
    }

    yc_number_node_ptr stencil_O2_Z( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const B )
    {
        return stencil_O2_Z( t, x, y, z, g,-1 );
    }

    yc_number_node_ptr stencil_O2_Z( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const F )
    {
        return stencil_O2_Z( t, x, y, z, g, 1 );
    }

    template<typename D>
    yc_number_node_ptr stencil_O2_Z( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g )
    {
        return stencil_O2_Z( t, x, y, z, g, D() );
    }

    yc_number_node_ptr stencil_O2_Y( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const int offset )
    {
        return
            (g(t,x,y       ,z)  -
             g(t,x,y+offset,z))*dyi;
    }

    yc_number_node_ptr stencil_O2_Y( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const B )
    {
        return stencil_O2_Y( t, x, y, z, g,-1 );
    }

    yc_number_node_ptr stencil_O2_Y( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const F )
    {
        return stencil_O2_Y( t, x, y, z, g, 1 );
    }

    template<typename D>
    yc_number_node_ptr stencil_O2_Y( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g )
    {
        return stencil_O2_Y( t, x, y, z, g, D() );
    }

    yc_number_node_ptr stencil_O2_X( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const int offset )
    {
        return
            (g(t,x       ,y,z)  -
             g(t,x+offset,y,z))*dxi;
    }

    yc_number_node_ptr stencil_O2_X( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const B )
    {
        return stencil_O2_X( t, x, y, z, g,-1 );
    }

    yc_number_node_ptr stencil_O2_X( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g,
                                     const F )
    {
        return stencil_O2_X( t, x, y, z, g, 1 );
    }

    // Call the stencil_O2() function above depending on D.
    template<typename D>
    yc_number_node_ptr stencil_O2_X( yc_number_node_ptr t,
                                     yc_number_node_ptr x,
                                     yc_number_node_ptr y,
                                     yc_number_node_ptr z,
                                     yc_var_proxy &g )
    {
        return stencil_O2_X( t, x, y, z, g, D() );
    }
};

