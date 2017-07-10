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

// Common definitions for Elastic stencils.

#pragma once

#include "StencilBase.hpp"

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

// This class implements StencilPart but is not the main solution.
// The main solution is provided during construction.
class ElasticBoundaryCondition : public StencilPart
{
protected:
    StencilSolution& _sol;
    
    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_DOMAIN_INDEX(z);         // spatial dim.
    
    public:
    ElasticBoundaryCondition(StencilSolution& solution) :
        _sol(solution) {}
    virtual ~ElasticBoundaryCondition() {}

    // Determine whether current indices are at boundary.
    virtual Condition is_at_boundary() =0;
    virtual Condition is_not_at_boundary() =0;

    // Return a reference to the main stencil-solution object provided during construction.
    virtual StencilSolution& get_stencil_solution() {
        return _sol;
    }
};

class ElasticStencilBase : public StencilBase {

protected:

    // Dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_DOMAIN_INDEX(z);         // spatial dim.
    
    // 3D-spatial coefficients.
    MAKE_GRID(rho, x, y, z);

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
    ElasticStencilBase(const string& name, StencilList& stencils,
                       ElasticBoundaryCondition *_bc = NULL) :
        StencilBase(name, stencils), bc(_bc)
    {
        init();
    }
    
    void init() {
        // StencilContex specific code
        REGISTER_STENCIL_CONTEXT_EXTENSION(
           virtual void initData() {
               initDiff();
           }
        );
    }
    
    bool hasBoundaryCondition()
    {
        return bc != NULL;
    }
    
    GridValue interp_rho( GridIndex x, GridIndex y, GridIndex z, const TL )
    {
        return ( 2.0/ (rho(x  , y  , z  ) +
                       rho(x+1, y  , z  )) );
    }

    GridValue interp_rho( GridIndex x, GridIndex y, GridIndex z, const TR )
    {
        return ( 2.0/ (rho(x  , y  , z  ) +
                       rho(x  , y+1, z  )) );
    }

    GridValue interp_rho( GridIndex x, GridIndex y, GridIndex z, const BL )
    {
        return ( 2.0/ (rho(x  , y  , z  ) +
                       rho(x  , y  , z+1)) );
    }

    GridValue interp_rho( GridIndex x, GridIndex y, GridIndex z, const BR )
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

    template<typename N>
    GridValue interp_rho( GridIndex x, GridIndex y, GridIndex z )
    {
        return interp_rho( x, y, z, N() );
    }

    GridValue stencil_O8_Z( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const int offset )
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

    GridValue stencil_O8( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const Z, const B )
    {
        return stencil_O8_Z( t, x, y, z, g, 0 );
    }

    GridValue stencil_O8( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const Z, const F )
    {
        return stencil_O8_Z( t, x, y, z, g, 1 );
    }

    GridValue stencil_O8_Y( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const int offset )
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

    GridValue stencil_O8( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const Y, const B )
    {
        return stencil_O8_Y( t, x, y, z, g, 0 );
    }

    GridValue stencil_O8( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const Y, const F )
    {
        return stencil_O8_Y( t, x, y, z, g, 1 );
    }

    GridValue stencil_O8_X( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const int offset )
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

    GridValue stencil_O8( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const X, const B )
    {
        return stencil_O8_X( t, x, y, z, g, 0 );
    }

    GridValue stencil_O8( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const X, const F )
    {
        return stencil_O8_X( t, x, y, z, g, 1 );
    }

    template<typename Dim, typename Dir>
    GridValue stencil_O8( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g )
    {
        return stencil_O8( t, x, y, z, g, Dim(), Dir() );
    }

    // Velocity-grid define functions.  For each D in x, y, z, define vel_D
    // at t+1 based on vel_x at t and stress grids at t.  Note that the t,
    // x, y, z parameters are integer grid indices, not actual offsets in
    // time or space, so half-steps due to staggered grids are adjusted
    // appropriately.

    template<typename N, typename SZ, typename SX, typename SY>
    void define_vel(GridIndex t, GridIndex x, GridIndex y, GridIndex z, 
            Grid &v, Grid &sx, Grid &sy, Grid &sz) {

        GridValue lrho   = interp_rho<N>( x, y, z );

        GridValue stx    = stencil_O8<X,SX>( t, x, y, z, sx );
        GridValue sty    = stencil_O8<Y,SY>( t, x, y, z, sy );
        GridValue stz    = stencil_O8<Z,SZ>( t, x, y, z, sz );

        GridValue next_v = v(t, x, y, z) + ((stx + sty + stz) * delta_t * lrho);
        
        // define the value at t+1.
        if ( hasBoundaryCondition() ) {
            Condition not_at_bc = bc->is_not_at_boundary();
            v(t+1, x, y, z) EQUALS next_v IF not_at_bc;
        } else
            v(t+1, x, y, z) EQUALS next_v;
    }

    GridValue stencil_O2_Z( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const int offset )
    {
        return
            (g(t,x,y,z       )  -
             g(t,x,y,z+offset))*dzi;
    }

    GridValue stencil_O2_Z( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const B )
    {
        return stencil_O2_Z( t, x, y, z, g,-1 );
    }

    GridValue stencil_O2_Z( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const F )
    {
        return stencil_O2_Z( t, x, y, z, g, 1 );
    }

    template<typename D>
    GridValue stencil_O2_Z( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g )
    {
        return stencil_O2_Z( t, x, y, z, g, D() );
    }

    GridValue stencil_O2_Y( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const int offset )
    {
        return
            (g(t,x,y       ,z)  -
             g(t,x,y+offset,z))*dyi;
    }

    GridValue stencil_O2_Y( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const B )
    {
        return stencil_O2_Y( t, x, y, z, g,-1 );
    }

    GridValue stencil_O2_Y( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const F )
    {
        return stencil_O2_Y( t, x, y, z, g, 1 );
    }

    template<typename D>
    GridValue stencil_O2_Y( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g )
    {
        return stencil_O2_Y( t, x, y, z, g, D() );
    }

    GridValue stencil_O2_X( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const int offset )
    {
        return
            (g(t,x       ,y,z)  -
             g(t,x+offset,y,z))*dxi;
    }

    GridValue stencil_O2_X( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const B )
    {
        return stencil_O2_X( t, x, y, z, g,-1 );
    }

    GridValue stencil_O2_X( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g, const F )
    {
        return stencil_O2_X( t, x, y, z, g, 1 );
    }

    template<typename D>
    GridValue stencil_O2_X( GridIndex t, GridIndex x, GridIndex y, GridIndex z, Grid &g )
    {
        return stencil_O2_X( t, x, y, z, g, D() );
    }
};

