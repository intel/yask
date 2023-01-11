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

// Stencil equations for SSG elastic numerics.
// Contributed by Albert Farres from the Barcelona Supercomputing Center.
// This version varies from the original by grouping related vars into
// larger vars with an added dimension.

#include "ElasticStencil/Elastic2Stencil.hpp"

class SSGElastic2Stencil : public Elastic2StencilBase {

protected:

    // Time-varying 3D-spatial velocity vars.
    yc_var_proxy v = yc_var_proxy("v", get_soln(), { t, x, y, z, vidx });
    enum VIDX { V_BL_W, V_TL_V, V_TR_U };

    // Time-varying 3D-spatial Stress vars.
    yc_var_proxy s = yc_var_proxy("s", get_soln(), { t, x, y, z, sidx });
    enum SIDX { S_BL_YZ, S_BR_XZ, S_TL_XX, S_TL_YY, S_TL_ZZ, S_TR_XY };

    // 3D-spatial coefficients.
    yc_var_proxy coef = yc_var_proxy("c", get_soln(), { x, y, z, cidx });
    enum CIDX { C_MU, C_LAMBDA, C_LAMBDA_MU2 };

public:

    SSGElastic2Stencil( ) :
        Elastic2StencilBase("ssg_merged")
    {
    }

    yc_number_node_ptr interp_mu( yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z, const BR)
    {
        return ( 2.0/ (coef(x  , y  , z  , C_MU) +
                       coef(x  , y+1, z  , C_MU) +
                       coef(x  , y  , z+1, C_MU) +
                       coef(x  , y+1, z+1, C_MU)) );
    }

    yc_number_node_ptr interp_mu( yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z, const BL)
    {
        return ( 2.0/ (coef(x  , y  , z  , C_MU) +
                       coef(x+1, y  , z  , C_MU) +
                       coef(x  , y  , z+1, C_MU) +
                       coef(x+1, y  , z+1, C_MU)) );
    }

    yc_number_node_ptr interp_mu( yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z, const TR)
    {
        return ( 2.0/ (coef(x  , y  , z  , C_MU) +
                       coef(x+1, y  , z  , C_MU) +
                       coef(x  , y+1, z  , C_MU) +
                       coef(x+1, y+1, z  , C_MU)) );
    }

    template<typename N>
    yc_number_node_ptr interp_mu( yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z)
    {
        return interp_mu( x, y, z, N() );
    }


    // Stress-var define functions.  For each D in xx, yy, zz, xy, xz, yz,
    // define stress_D at t+1 based on stress_D at t and vel vars at t+1.
    // This implies that the velocity-var define functions must be called
    // before these for a given value of t.  Note that the t, x, y, z
    // parameters are integer var indices, not actual offsets in time or
    // space, so half-steps due to staggered vars are adjusted
    // appropriately.

    template<typename N, typename DA, typename SA, typename DB, typename SB>
    void define_str(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                    yc_number_any_arg sidx, yc_number_any_arg va_idx, yc_number_any_arg vb_idx) {

        auto lcoeff = interp_mu<N>( x, y, z );

        auto vta    = stencil_O8<DA,SA>( t+1, x, y, z, v, va_idx );
        auto vtb    = stencil_O8<DB,SB>( t+1, x, y, z, v, vb_idx );

        auto next_s = s(t, x, y, z, sidx) + ((vta + vtb) * lcoeff) * delta_t;

        // define the value at t+1.
        s(t+1, x, y, z, sidx) EQUALS next_s;
    }

    void define_str_TL(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z )
    {

        auto ilambdamu2 = 1.0 / coef(x,y,z, C_LAMBDA_MU2);
        auto ilambda    = 1.0 / coef(x,y,z, C_LAMBDA);

        auto vtx    = stencil_O8<X,F>( t+1, x, y, z, v, V_TR_U );
        auto vty    = stencil_O8<Y,B>( t+1, x, y, z, v, V_TL_V );
        auto vtz    = stencil_O8<Z,B>( t+1, x, y, z, v, V_BL_W );

        auto next_xx = s(t, x, y, z, S_TL_XX) + ilambdamu2 * vtx * delta_t
            + ilambda    * vty * delta_t
            + ilambda    * vtz * delta_t;
        auto next_yy = s(t, x, y, z, S_TL_YY) + ilambda    * vtx * delta_t
            + ilambdamu2 * vty * delta_t
            + ilambda    * vtz * delta_t;
        auto next_zz = s(t, x, y, z, S_TL_ZZ) + ilambda    * vtx * delta_t
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

// Create an object of type 'SSGElastic2Stencil',
// making it available in the YASK compiler utility via the
// '-stencil' commmand-line option or the 'stencil=' build option.
static SSGElastic2Stencil SSGElastic2Stencil_instance;
