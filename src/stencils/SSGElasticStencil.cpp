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

#include "ElasticStencil/ElasticStencil.hpp"

class SSGElasticStencil : public ElasticStencilBase {

protected:

    // Time-varying 3D-spatial velocity grids.
    yc_grid_var v_bl_w = yc_grid_var("v_bl_w", get_solution(), { t, x, y, z });
    yc_grid_var v_tl_v = yc_grid_var("v_tl_v", get_solution(), { t, x, y, z });
    yc_grid_var v_tr_u = yc_grid_var("v_tr_u", get_solution(), { t, x, y, z });

    // Time-varying 3D-spatial Stress grids.
    yc_grid_var s_bl_yz = yc_grid_var("s_bl_yz", get_solution(), { t, x, y, z });
    yc_grid_var s_br_xz = yc_grid_var("s_br_xz", get_solution(), { t, x, y, z });
    yc_grid_var s_tl_xx = yc_grid_var("s_tl_xx", get_solution(), { t, x, y, z });
    yc_grid_var s_tl_yy = yc_grid_var("s_tl_yy", get_solution(), { t, x, y, z });
    yc_grid_var s_tl_zz = yc_grid_var("s_tl_zz", get_solution(), { t, x, y, z });
    yc_grid_var s_tr_xy = yc_grid_var("s_tr_xy", get_solution(), { t, x, y, z });

    // 3D-spatial coefficients.
    yc_grid_var mu = yc_grid_var("mu", get_solution(), { x, y, z });
    yc_grid_var lambda = yc_grid_var("lambda", get_solution(), { x, y, z });
    yc_grid_var lambdamu2 = yc_grid_var("lambdamu2", get_solution(), { x, y, z });

public:

    SSGElasticStencil( ) :
        ElasticStencilBase("ssg")
    {
    }

    yc_number_node_ptr interp_mu( yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z, const BR )
    {
        return ( 2.0/ (mu(x  , y  , z  ) +
                       mu(x  , y+1, z  ) +
                       mu(x  , y  , z+1) +
                       mu(x  , y+1, z+1)) );
    }

    yc_number_node_ptr interp_mu( yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z, const BL )
    {
        return ( 2.0/ (mu(x  , y  , z  ) +
                       mu(x+1, y  , z  ) +
                       mu(x  , y  , z+1) +
                       mu(x+1, y  , z+1)) );
    }

    yc_number_node_ptr interp_mu( yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z, const TR )
    {
        return ( 2.0/ (mu(x  , y  , z  ) +
                       mu(x+1, y  , z  ) +
                       mu(x  , y+1, z  ) +
                       mu(x+1, y+1, z  )) );
    }

    template<typename N>
    yc_number_node_ptr interp_mu( yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z )
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
    void define_str(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                    yc_grid_var &s, yc_grid_var &va, yc_grid_var &vb) {

        yc_number_node_ptr lcoeff = interp_mu<N>( x, y, z );

        yc_number_node_ptr vta    = stencil_O8<DA,SA>( t+1, x, y, z, va );
        yc_number_node_ptr vtb    = stencil_O8<DB,SB>( t+1, x, y, z, vb );

        yc_number_node_ptr next_s = s(t, x, y, z) + ((vta + vtb) * lcoeff) * delta_t;

        // define the value at t+1.
        s(t+1, x, y, z) EQUALS next_s;
    }

    void define_str_TL(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z )
    {

        yc_number_node_ptr ilambdamu2 = 1.0 / lambdamu2(x,y,z);
        yc_number_node_ptr ilambda    = 1.0 / lambda   (x,y,z);

        yc_number_node_ptr vtx    = stencil_O8<X,F>( t+1, x, y, z, v_tr_u );
        yc_number_node_ptr vty    = stencil_O8<Y,B>( t+1, x, y, z, v_tl_v );
        yc_number_node_ptr vtz    = stencil_O8<Z,B>( t+1, x, y, z, v_bl_w );

        yc_number_node_ptr next_xx = s_tl_xx(t, x, y, z) + ilambdamu2 * vtx * delta_t
            + ilambda    * vty * delta_t
            + ilambda    * vtz * delta_t;
        yc_number_node_ptr next_yy = s_tl_yy(t, x, y, z) + ilambda    * vtx * delta_t
            + ilambdamu2 * vty * delta_t
            + ilambda    * vtz * delta_t;
        yc_number_node_ptr next_zz = s_tl_zz(t, x, y, z) + ilambda    * vtx * delta_t
            + ilambda    * vty * delta_t
            + ilambdamu2 * vtz * delta_t;

        // define the value at t+1.
        s_tl_xx(t+1, x, y, z) EQUALS next_xx;
        s_tl_yy(t+1, x, y, z) EQUALS next_yy;
        s_tl_zz(t+1, x, y, z) EQUALS next_zz;
    }

    // Call all the define_* functions.
    virtual void define() {

        // Define velocity components.
        define_vel<BL, F, B, B>(t, x, y, z, v_bl_w, s_bl_yz, s_br_xz, s_tl_zz);
        define_vel<TR, B, B, F>(t, x, y, z, v_tr_u, s_tr_xy, s_tl_xx, s_br_xz);
        define_vel<TL, B, F, B>(t, x, y, z, v_tl_v, s_tl_yy, s_tr_xy, s_bl_yz);

        //// Define stresses components.
        define_str<BR, X, F, Z, F>(t, x, y, z, s_br_xz, v_bl_w, v_tr_u );
        define_str<TR, X, F, Y, F>(t, x, y, z, s_tr_xy, v_tl_v, v_tr_u );
        define_str<BL, Y, F, Z, F>(t, x, y, z, s_bl_yz, v_bl_w, v_tl_v );
        define_str_TL(t, x, y, z);

    }
};

// Create an object of type 'SSGElasticStencil',
// making it available in the YASK compiler utility via the
// '-stencil' commmand-line option or the 'stencil=' build option.
static SSGElasticStencil SSGElasticStencil_instance;
