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

// Stencil equations for SSG elastic numerics.
// Contributed by Albert Farres from the Barcelona Supercomputing Center.
// This version varies from the original by grouping related grids into
// larger grids with an added dimension.

#include "ElasticStencil/Elastic2Stencil.hpp"

class SSGElastic2Stencil : public Elastic2StencilBase {

protected:

    // Time-varying 3D-spatial velocity grids.
    MAKE_GRID(v, t, x, y, z, vidx);
    enum VIDX { V_BL_W, V_TL_V, V_TR_U };

    // Time-varying 3D-spatial Stress grids.
    MAKE_GRID(s, t, x, y, z, sidx);
    enum SIDX { S_BL_YZ, S_BR_XZ, S_TL_XX, S_TL_YY, S_TL_ZZ, S_TR_XY };

public:

    SSGElastic2Stencil( StencilList& stencils) :
        Elastic2StencilBase("ssg2", stencils)
    {
    }

    GridValue interp_mu( GridIndex x, GridIndex y, GridIndex z, const BR)
    {
        return ( 2.0/ (coef(x  , y  , z  , C_MU) +
                       coef(x  , y+1, z  , C_MU) +
                       coef(x  , y  , z+1, C_MU) +
                       coef(x  , y+1, z+1, C_MU)) );
    }

    GridValue interp_mu( GridIndex x, GridIndex y, GridIndex z, const BL)
    {
        return ( 2.0/ (coef(x  , y  , z  , C_MU) +
                       coef(x+1, y  , z  , C_MU) +
                       coef(x  , y  , z+1, C_MU) +
                       coef(x+1, y  , z+1, C_MU)) );
    }

    GridValue interp_mu( GridIndex x, GridIndex y, GridIndex z, const TR)
    {
        return ( 2.0/ (coef(x  , y  , z  , C_MU) +
                       coef(x+1, y  , z  , C_MU) +
                       coef(x  , y+1, z  , C_MU) +
                       coef(x+1, y+1, z  , C_MU)) );
    }

    template<typename N>
    GridValue interp_mu( GridIndex x, GridIndex y, GridIndex z)
    {
        return interp_mu( x, y, z, N() );
    }


    // Stress-grid define functions.  For each D in xx, yy, zz, xy, xz, yz,
    // define stress_D at t+1 based on stress_D at t and vel grids at t+1.
    // This implies that the velocity-grid define functions must be called
    // before these for a given value of t.  Note that the t, x, y, z
    // parameters are integer grid indices, not actual offsets in time or
    // space, so half-steps due to staggered grids are adjusted
    // appropriately.

    template<typename N, typename DA, typename SA, typename DB, typename SB>
    void define_str(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                    GridIndex sidx, GridIndex va_idx, GridIndex vb_idx) {

        GridValue lcoeff = interp_mu<N>( x, y, z );

        GridValue vta    = stencil_O8<DA,SA>( t+1, x, y, z, v, va_idx );
        GridValue vtb    = stencil_O8<DB,SB>( t+1, x, y, z, v, vb_idx );

        GridValue next_s = s(t, x, y, z, sidx) + ((vta + vtb) * lcoeff) * delta_t;

        // define the value at t+1.
        s(t+1, x, y, z, sidx) EQUALS next_s;
    }
    template<typename N, typename DA, typename SA, typename DB, typename SB>
    void define_str(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                    int sidx, int va_idx, int vb_idx) {
        define_str<N, DA, SA, DB, SB>(t, x, y, z,
                                      constNum(sidx), constNum(va_idx), constNum(vb_idx));
    }

    void define_str_TL(GridIndex t, GridIndex x, GridIndex y, GridIndex z )
    {

        GridValue ilambdamu2 = 1.0 / coef(x,y,z, C_LAMBDA_MU2);
        GridValue ilambda    = 1.0 / coef(x,y,z, C_LAMBDA);

        GridValue vtx    = stencil_O8<X,F>( t+1, x, y, z, v, constNum(V_TR_U) );
        GridValue vty    = stencil_O8<Y,B>( t+1, x, y, z, v, constNum(V_TL_V) );
        GridValue vtz    = stencil_O8<Z,B>( t+1, x, y, z, v, constNum(V_BL_W) );

        GridValue next_xx = s(t, x, y, z, S_TL_XX) + ilambdamu2 * vtx * delta_t
            + ilambda    * vty * delta_t
            + ilambda    * vtz * delta_t;
        GridValue next_yy = s(t, x, y, z, S_TL_YY) + ilambda    * vtx * delta_t
            + ilambdamu2 * vty * delta_t
            + ilambda    * vtz * delta_t;
        GridValue next_zz = s(t, x, y, z, S_TL_ZZ) + ilambda    * vtx * delta_t
            + ilambda    * vty * delta_t
            + ilambdamu2 * vtz * delta_t;

        // define the value at t+1.
        s(t+1, x, y, z, S_TL_XX) EQUALS next_xx;
        s(t+1, x, y, z, S_TL_YY) EQUALS next_yy;
        s(t+1, x, y, z, S_TL_ZZ) EQUALS next_zz;
    }

    // Call all the define_* functions.
    virtual void define() {

        // Define velocity components.
        define_vel<BL, F, B, B>(t, x, y, z, v, V_BL_W, s, S_BL_YZ, S_BR_XZ, S_TL_ZZ);
        define_vel<TR, B, B, F>(t, x, y, z, v, V_TR_U, s, S_TR_XY, S_TL_XX, S_BR_XZ);
        define_vel<TL, B, F, B>(t, x, y, z, v, V_TL_V, s, S_TL_YY, S_TR_XY, S_BL_YZ);

        //// Define stresses components.
        define_str<BR, X, F, Z, F>(t, x, y, z, S_BR_XZ, V_BL_W, V_TR_U );
        define_str<TR, X, F, Y, F>(t, x, y, z, S_TR_XY, V_TL_V, V_TR_U );
        define_str<BL, Y, F, Z, F>(t, x, y, z, S_BL_YZ, V_BL_W, V_TL_V );
        define_str_TL(t, x, y, z);

    }
};

REGISTER_STENCIL(SSGElastic2Stencil);
