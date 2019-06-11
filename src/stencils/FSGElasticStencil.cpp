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

// Stencil equations for FSG elastic numerics.
// Contributed by Albert Farres from the Barcelona Supercomputing Center.

// YASK stencil solution(s) in this file will be integrated into the YASK compiler utility.
#include "yask_compiler_api.hpp"
using namespace std;
using namespace yask;
#include "ElasticStencil/ElasticStencil.hpp"

namespace fsg {

    class FSG_ABC;

    class FSGBoundaryCondition : public ElasticBoundaryCondition
    {
    public:
        FSGBoundaryCondition(yc_solution_base& base) :
            ElasticBoundaryCondition(base) {}
        virtual void velocity (yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z ) {}
        virtual void stress (yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z ) {}
    };

    class FSGElasticStencilBase : public ElasticStencilBase {
        friend FSG_ABC;

    protected:

        yc_var_proxy v_bl_u = yc_var_proxy("v_bl_u", get_soln(), { t, x, y, z });
        yc_var_proxy v_bl_v = yc_var_proxy("v_bl_v", get_soln(), { t, x, y, z });
        yc_var_proxy v_bl_w = yc_var_proxy("v_bl_w", get_soln(), { t, x, y, z });
        yc_var_proxy v_br_u = yc_var_proxy("v_br_u", get_soln(), { t, x, y, z });
        yc_var_proxy v_br_v = yc_var_proxy("v_br_v", get_soln(), { t, x, y, z });
        yc_var_proxy v_br_w = yc_var_proxy("v_br_w", get_soln(), { t, x, y, z });
        yc_var_proxy v_tl_u = yc_var_proxy("v_tl_u", get_soln(), { t, x, y, z });
        yc_var_proxy v_tl_v = yc_var_proxy("v_tl_v", get_soln(), { t, x, y, z });
        yc_var_proxy v_tl_w = yc_var_proxy("v_tl_w", get_soln(), { t, x, y, z });
        yc_var_proxy v_tr_u = yc_var_proxy("v_tr_u", get_soln(), { t, x, y, z });
        yc_var_proxy v_tr_v = yc_var_proxy("v_tr_v", get_soln(), { t, x, y, z });
        yc_var_proxy v_tr_w = yc_var_proxy("v_tr_w", get_soln(), { t, x, y, z });

        yc_var_proxy s_bl_xx = yc_var_proxy("s_bl_xx", get_soln(), { t, x, y, z });
        yc_var_proxy s_bl_yy = yc_var_proxy("s_bl_yy", get_soln(), { t, x, y, z });
        yc_var_proxy s_bl_zz = yc_var_proxy("s_bl_zz", get_soln(), { t, x, y, z });
        yc_var_proxy s_bl_yz = yc_var_proxy("s_bl_yz", get_soln(), { t, x, y, z });
        yc_var_proxy s_bl_xz = yc_var_proxy("s_bl_xz", get_soln(), { t, x, y, z });
        yc_var_proxy s_bl_xy = yc_var_proxy("s_bl_xy", get_soln(), { t, x, y, z });
        yc_var_proxy s_br_xx = yc_var_proxy("s_br_xx", get_soln(), { t, x, y, z });
        yc_var_proxy s_br_yy = yc_var_proxy("s_br_yy", get_soln(), { t, x, y, z });
        yc_var_proxy s_br_zz = yc_var_proxy("s_br_zz", get_soln(), { t, x, y, z });
        yc_var_proxy s_br_yz = yc_var_proxy("s_br_yz", get_soln(), { t, x, y, z });
        yc_var_proxy s_br_xz = yc_var_proxy("s_br_xz", get_soln(), { t, x, y, z });
        yc_var_proxy s_br_xy = yc_var_proxy("s_br_xy", get_soln(), { t, x, y, z });
        yc_var_proxy s_tl_xx = yc_var_proxy("s_tl_xx", get_soln(), { t, x, y, z });
        yc_var_proxy s_tl_yy = yc_var_proxy("s_tl_yy", get_soln(), { t, x, y, z });
        yc_var_proxy s_tl_zz = yc_var_proxy("s_tl_zz", get_soln(), { t, x, y, z });
        yc_var_proxy s_tl_yz = yc_var_proxy("s_tl_yz", get_soln(), { t, x, y, z });
        yc_var_proxy s_tl_xz = yc_var_proxy("s_tl_xz", get_soln(), { t, x, y, z });
        yc_var_proxy s_tl_xy = yc_var_proxy("s_tl_xy", get_soln(), { t, x, y, z });
        yc_var_proxy s_tr_xx = yc_var_proxy("s_tr_xx", get_soln(), { t, x, y, z });
        yc_var_proxy s_tr_yy = yc_var_proxy("s_tr_yy", get_soln(), { t, x, y, z });
        yc_var_proxy s_tr_zz = yc_var_proxy("s_tr_zz", get_soln(), { t, x, y, z });
        yc_var_proxy s_tr_yz = yc_var_proxy("s_tr_yz", get_soln(), { t, x, y, z });
        yc_var_proxy s_tr_xz = yc_var_proxy("s_tr_xz", get_soln(), { t, x, y, z });
        yc_var_proxy s_tr_xy = yc_var_proxy("s_tr_xy", get_soln(), { t, x, y, z });

        // 3D-spatial coefficients.
        yc_var_proxy c11 = yc_var_proxy("c11", get_soln(), { x, y, z });
        yc_var_proxy c12 = yc_var_proxy("c12", get_soln(), { x, y, z });
        yc_var_proxy c13 = yc_var_proxy("c13", get_soln(), { x, y, z });
        yc_var_proxy c14 = yc_var_proxy("c14", get_soln(), { x, y, z });
        yc_var_proxy c15 = yc_var_proxy("c15", get_soln(), { x, y, z });
        yc_var_proxy c16 = yc_var_proxy("c16", get_soln(), { x, y, z });
        yc_var_proxy c22 = yc_var_proxy("c22", get_soln(), { x, y, z });
        yc_var_proxy c23 = yc_var_proxy("c23", get_soln(), { x, y, z });
        yc_var_proxy c24 = yc_var_proxy("c24", get_soln(), { x, y, z });
        yc_var_proxy c25 = yc_var_proxy("c25", get_soln(), { x, y, z });
        yc_var_proxy c26 = yc_var_proxy("c26", get_soln(), { x, y, z });
        yc_var_proxy c33 = yc_var_proxy("c33", get_soln(), { x, y, z });
        yc_var_proxy c34 = yc_var_proxy("c34", get_soln(), { x, y, z });
        yc_var_proxy c35 = yc_var_proxy("c35", get_soln(), { x, y, z });
        yc_var_proxy c36 = yc_var_proxy("c36", get_soln(), { x, y, z });
        yc_var_proxy c44 = yc_var_proxy("c44", get_soln(), { x, y, z });
        yc_var_proxy c45 = yc_var_proxy("c45", get_soln(), { x, y, z });
        yc_var_proxy c46 = yc_var_proxy("c46", get_soln(), { x, y, z });
        yc_var_proxy c55 = yc_var_proxy("c55", get_soln(), { x, y, z });
        yc_var_proxy c56 = yc_var_proxy("c56", get_soln(), { x, y, z });
        yc_var_proxy c66 = yc_var_proxy("c66", get_soln(), { x, y, z });

    public:

        FSGElasticStencilBase( const string &name, 
                               FSGBoundaryCondition *bc = NULL ) :
            ElasticStencilBase ( name, bc )
        {
        }

        yc_number_node_ptr cell_coeff( const yc_number_node_ptr x, const yc_number_node_ptr y, const yc_number_node_ptr z, yc_var_proxy &c, const BR )
        {
            return  1.0 / (0.25*(c(x  , y  , z  ) +
                                 c(x  , y+1, z  ) +
                                 c(x  , y  , z+1) +
                                 c(x  , y+1, z+1)));
        }
        yc_number_node_ptr cell_coeff( const yc_number_node_ptr x, const yc_number_node_ptr y, const yc_number_node_ptr z, yc_var_proxy &c, const BL )
        {
            return  1.0 / (0.25*(c(x  , y  , z  ) +
                                 c(x+1, y  , z  ) +
                                 c(x  , y  , z+1) +
                                 c(x+1, y  , z+1)));
        }
        yc_number_node_ptr cell_coeff( const yc_number_node_ptr x, const yc_number_node_ptr y, const yc_number_node_ptr z, yc_var_proxy &c, const TR )
        {
            return  1.0 / (0.25*(c(x  , y  , z  ) +
                                 c(x  , y+1, z  ) +
                                 c(x+1, y  , z  ) +
                                 c(x+1, y+1, z  )));
        }
        yc_number_node_ptr cell_coeff( const yc_number_node_ptr x, const yc_number_node_ptr y, const yc_number_node_ptr z, yc_var_proxy &c, const TL )
        {
            return  1.0 /        c(x  , y  , z  );
        }
        template<typename N>
        yc_number_node_ptr cell_coeff( const yc_number_node_ptr x, const yc_number_node_ptr y, const yc_number_node_ptr z, yc_var_proxy &c )
        {
            return cell_coeff( x, y, z, c, N() );
        }

        yc_number_node_ptr cell_coeff_artm( const yc_number_node_ptr x, const yc_number_node_ptr y, const yc_number_node_ptr z, yc_var_proxy &c, const BR )
        {
            return 0.25 *( 1.0 / c(x  , y  , z  ) +
                           1.0 / c(x  , y+1, z  ) +
                           1.0 / c(x  , y  , z+1) +
                           1.0 / c(x  , y+1, z+1) );
        }
        yc_number_node_ptr cell_coeff_artm( const yc_number_node_ptr x, const yc_number_node_ptr y, const yc_number_node_ptr z, yc_var_proxy &c, const BL )
        {
            return 0.25 *( 1.0 / c(x  , y  , z  ) +
                           1.0 / c(x+1, y  , z  ) +
                           1.0 / c(x  , y  , z+1) +
                           1.0 / c(x+1, y  , z+1) );
        }
        yc_number_node_ptr cell_coeff_artm( const yc_number_node_ptr x, const yc_number_node_ptr y, const yc_number_node_ptr z, yc_var_proxy &c, const TR )
        {
            return 0.25 *( 1.0 / c(x  , y  , z  ) +
                           1.0 / c(x  , y+1, z  ) +
                           1.0 / c(x+1, y  , z  ) +
                           1.0 / c(x+1, y+1, z  ) );
        }
        yc_number_node_ptr cell_coeff_artm( const yc_number_node_ptr x, const yc_number_node_ptr y, const yc_number_node_ptr z, yc_var_proxy &c, const TL )
        {
            return  1.0 /        c(x  , y  , z  );
        }
        template<typename N>
        yc_number_node_ptr cell_coeff_artm( const yc_number_node_ptr x, const yc_number_node_ptr y, const yc_number_node_ptr z, yc_var_proxy &c )
        {
            return cell_coeff_artm( x, y, z, c, N() );
        }

        yc_number_node_ptr stress_update( yc_number_node_ptr c1, yc_number_node_ptr c2, yc_number_node_ptr c3,
                                 yc_number_node_ptr c4, yc_number_node_ptr c5, yc_number_node_ptr c6,
                                 yc_number_node_ptr u_z, yc_number_node_ptr u_y, yc_number_node_ptr u_x,
                                 yc_number_node_ptr v_z, yc_number_node_ptr v_y, yc_number_node_ptr v_x,
                                 yc_number_node_ptr w_z, yc_number_node_ptr w_y, yc_number_node_ptr w_x )
        {
            return delta_t * c1 * u_x
                + delta_t * c2 * v_y
                + delta_t * c3 * w_z
                + delta_t * c4 * (w_y + v_z)
                + delta_t * c5 * (w_x + u_z)
                + delta_t * c6 * (v_x + u_y);
        }

        //
        // Stress-var define functions.  For each D in xx, yy, zz, xy, xz, yz,
        // define stress_D at t+1 based on stress_D at t and vel vars at t+1.
        // This implies that the velocity-var define functions must be called
        // before these for a given value of t.  Note that the t, x, y, z
        // parameters are integer var indices, not actual offsets in time or
        // space, so half-steps due to staggered vars are adjusted
        // appropriately.

        template<typename N, typename SZ, typename SX, typename SY>
        void define_str(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                        yc_var_proxy &sxx, yc_var_proxy &syy, yc_var_proxy &szz, yc_var_proxy &sxy, yc_var_proxy &sxz, yc_var_proxy &syz,
                        yc_var_proxy &vxu, yc_var_proxy &vxv, yc_var_proxy &vxw, yc_var_proxy &vyu, yc_var_proxy &vyv, yc_var_proxy &vyw, yc_var_proxy &vzu, yc_var_proxy &vzv, yc_var_proxy &vzw ) {

            // Interpolate coeffs.
            auto ic11 = cell_coeff     <N>(x, y, z, c11);
            auto ic12 = cell_coeff     <N>(x, y, z, c12);
            auto ic13 = cell_coeff     <N>(x, y, z, c13);
            auto ic14 = cell_coeff_artm<N>(x, y, z, c14);
            auto ic15 = cell_coeff_artm<N>(x, y, z, c15);
            auto ic16 = cell_coeff_artm<N>(x, y, z, c16);
            auto ic22 = cell_coeff     <N>(x, y, z, c22);
            auto ic23 = cell_coeff     <N>(x, y, z, c23);
            auto ic24 = cell_coeff_artm<N>(x, y, z, c24);
            auto ic25 = cell_coeff_artm<N>(x, y, z, c25);
            auto ic26 = cell_coeff_artm<N>(x, y, z, c26);
            auto ic33 = cell_coeff     <N>(x, y, z, c33);
            auto ic34 = cell_coeff_artm<N>(x, y, z, c34);
            auto ic35 = cell_coeff_artm<N>(x, y, z, c35);
            auto ic36 = cell_coeff_artm<N>(x, y, z, c36);
            auto ic44 = cell_coeff     <N>(x, y, z, c44);
            auto ic45 = cell_coeff_artm<N>(x, y, z, c45);
            auto ic46 = cell_coeff_artm<N>(x, y, z, c46);
            auto ic55 = cell_coeff     <N>(x, y, z, c55);
            auto ic56 = cell_coeff_artm<N>(x, y, z, c56);
            auto ic66 = cell_coeff     <N>(x, y, z, c66);

            // Compute stencils. Note that we are using the velocity values at t+1.
            auto u_z = stencil_O8<Z,SZ>( t+1, x, y, z, vzu );
            auto v_z = stencil_O8<Z,SZ>( t+1, x, y, z, vzv );
            auto w_z = stencil_O8<Z,SZ>( t+1, x, y, z, vzw );

            auto u_x = stencil_O8<X,SX>( t+1, x, y, z, vxu );
            auto v_x = stencil_O8<X,SX>( t+1, x, y, z, vxv );
            auto w_x = stencil_O8<X,SX>( t+1, x, y, z, vxw );

            auto u_y = stencil_O8<Y,SY>( t+1, x, y, z, vyu );
            auto v_y = stencil_O8<Y,SY>( t+1, x, y, z, vyv );
            auto w_y = stencil_O8<Y,SY>( t+1, x, y, z, vyw );

            // Compute next stress value
            auto next_sxx = sxx(t, x, y, z) +
                stress_update(ic11,ic12,ic13,ic14,ic15,ic16,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);
            auto next_syy = syy(t, x, y, z) +
                stress_update(ic12,ic22,ic23,ic24,ic25,ic26,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);
            auto next_szz = szz(t, x, y, z) +
                stress_update(ic13,ic23,ic33,ic34,ic35,ic36,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);
            auto next_syz = syz(t, x, y, z) +
                stress_update(ic14,ic24,ic34,ic44,ic45,ic46,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);
            auto next_sxz = sxz(t, x, y, z) +
                stress_update(ic15,ic25,ic35,ic45,ic55,ic56,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);
            auto next_sxy = sxy(t, x, y, z) +
                stress_update(ic16,ic26,ic36,ic46,ic56,ic66,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);

            // define the value at t+1.
            if ( hasBoundaryCondition() ) {
                yc_bool_node_ptr not_at_bc = bc->is_not_at_boundary();
                sxx(t+1, x, y, z) EQUALS next_sxx IF_DOMAIN not_at_bc;
                syy(t+1, x, y, z) EQUALS next_syy IF_DOMAIN not_at_bc;
                szz(t+1, x, y, z) EQUALS next_szz IF_DOMAIN not_at_bc;
                syz(t+1, x, y, z) EQUALS next_syz IF_DOMAIN not_at_bc;
                sxz(t+1, x, y, z) EQUALS next_sxz IF_DOMAIN not_at_bc;
                sxy(t+1, x, y, z) EQUALS next_sxy IF_DOMAIN not_at_bc;
            } else {
                sxx(t+1, x, y, z) EQUALS next_sxx;
                syy(t+1, x, y, z) EQUALS next_syy;
                szz(t+1, x, y, z) EQUALS next_szz;
                syz(t+1, x, y, z) EQUALS next_syz;
                sxz(t+1, x, y, z) EQUALS next_sxz;
                sxy(t+1, x, y, z) EQUALS next_sxy;
            }
        }

        // Set BKC (best-known configs) found by automated and/or manual
        // tuning. They are only applied for certain target configs.
        virtual void set_configs() {
            auto soln = get_soln(); // pointer to compile-time soln.

            // Only have BKCs for SP FP (4B).
            if (soln->get_element_bytes() == 4) {

                // Compile-time defaults, e.g., prefetching.
                if (soln->is_target_set()) {
                    auto target = soln->get_target();
                    if (target == "knl") {
                        soln->set_prefetch_dist(1, 0);
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
                         // Use only 1 thread per core.
                         kernel_soln.apply_command_line_options("-thread_divisor 4 -block_threads 2");

                         kernel_soln.set_block_size("x", 16);
                         kernel_soln.set_block_size("y", 16);
                         kernel_soln.set_block_size("z", 16);
                     }
                     else if (isa == "avx512") {
                         kernel_soln.set_block_size("x", 188);
                         kernel_soln.set_block_size("y", 12);
                         kernel_soln.set_block_size("z", 24);
                     }
                     else {
                         kernel_soln.set_block_size("x", 48);
                         kernel_soln.set_block_size("y", 4);
                         kernel_soln.set_block_size("z", 128);
                     }
                     );
            }
        }
        
        // Call all the define_* functions.
        virtual void define() {

            FSGBoundaryCondition &fsg_bc = *static_cast<FSGBoundaryCondition *>(bc);

            // Define velocity components.
            define_vel<TL, B, F, B>(t, x, y, z, v_tl_w, s_tl_yz, s_tr_xz, s_bl_zz);
            define_vel<TR, B, B, F>(t, x, y, z, v_tr_w, s_tr_yz, s_tl_xz, s_br_zz);
            define_vel<BL, F, B, B>(t, x, y, z, v_bl_w, s_bl_yz, s_br_xz, s_tl_zz);
            define_vel<BR, F, F, F>(t, x, y, z, v_br_w, s_br_yz, s_bl_xz, s_tr_zz);
            define_vel<TL, B, F, B>(t, x, y, z, v_tl_u, s_tl_xy, s_tr_xx, s_bl_xz);
            define_vel<TR, B, B, F>(t, x, y, z, v_tr_u, s_tr_xy, s_tl_xx, s_br_xz);
            define_vel<BL, F, B, B>(t, x, y, z, v_bl_u, s_bl_xy, s_br_xx, s_tl_xz);
            define_vel<BR, F, F, F>(t, x, y, z, v_br_u, s_br_xy, s_bl_xx, s_tr_xz);
            define_vel<TL, B, F, B>(t, x, y, z, v_tl_v, s_tl_yy, s_tr_xy, s_bl_yz);
            define_vel<TR, B, B, F>(t, x, y, z, v_tr_v, s_tr_yy, s_tl_xy, s_br_yz);
            define_vel<BL, F, B, B>(t, x, y, z, v_bl_v, s_bl_yy, s_br_xy, s_tl_yz);
            define_vel<BR, F, F, F>(t, x, y, z, v_br_v, s_br_yy, s_bl_xy, s_tr_yz);

            if ( hasBoundaryCondition() )
                fsg_bc.velocity(t,x,y,z);

            //// Define stresses components.
            define_str<BR, F, B, F>(t, x, y, z, s_br_xx, s_br_yy, s_br_zz, s_br_xy, s_br_xz, s_br_yz,
                                    v_br_u, v_br_v, v_br_w, v_bl_u, v_bl_v, v_bl_w, v_tr_u, v_tr_v, v_tr_w);
            define_str<BL, F, F, B>(t, x, y, z, s_bl_xx, s_bl_yy, s_bl_zz, s_bl_xy, s_bl_xz, s_bl_yz,
                                    v_bl_u, v_bl_v, v_bl_w, v_br_u, v_br_v, v_br_w, v_tl_u, v_tl_v, v_tl_w);
            define_str<TR, B, F, F>(t, x, y, z, s_tr_xx, s_tr_yy, s_tr_zz, s_tr_xy, s_tr_xz, s_tr_yz,
                                    v_tr_u, v_tr_v, v_tr_w, v_tl_u, v_tl_v, v_tl_w, v_br_u, v_br_v, v_br_w);
            define_str<TL, B, B, B>(t, x, y, z, s_tl_xx, s_tl_yy, s_tl_zz, s_tl_xy, s_tl_xz, s_tl_yz,
                                    v_tl_u, v_tl_v, v_tl_w, v_tr_u, v_tr_v, v_tr_w, v_bl_u, v_bl_v, v_bl_w);

            if ( hasBoundaryCondition() )
                fsg_bc.stress(t,x,y,z);

            set_configs();
        }
    };

    class FSG_ABC : public FSGBoundaryCondition
    {
    protected:
        const int abc_width = 20;

        // Sponge coefficients.
        yc_var_proxy sponge_lx = yc_var_proxy("sponge_lx", get_soln(), { x, y, z });
        yc_var_proxy sponge_rx = yc_var_proxy("sponge_rx", get_soln(), { x, y, z });
        yc_var_proxy sponge_bz = yc_var_proxy("sponge_bz", get_soln(), { x, y, z });
        yc_var_proxy sponge_tz = yc_var_proxy("sponge_tz", get_soln(), { x, y, z });
        yc_var_proxy sponge_fy = yc_var_proxy("sponge_fy", get_soln(), { x, y, z });
        yc_var_proxy sponge_by = yc_var_proxy("sponge_by", get_soln(), { x, y, z });
        yc_var_proxy sponge_sq_lx = yc_var_proxy("sponge_sq_lx", get_soln(), { x, y, z });
        yc_var_proxy sponge_sq_rx = yc_var_proxy("sponge_sq_rx", get_soln(), { x, y, z });
        yc_var_proxy sponge_sq_bz = yc_var_proxy("sponge_sq_bz", get_soln(), { x, y, z });
        yc_var_proxy sponge_sq_tz = yc_var_proxy("sponge_sq_tz", get_soln(), { x, y, z });
        yc_var_proxy sponge_sq_fy = yc_var_proxy("sponge_sq_fy", get_soln(), { x, y, z });
        yc_var_proxy sponge_sq_by = yc_var_proxy("sponge_sq_by", get_soln(), { x, y, z });

        FSGElasticStencilBase &fsg;

    public:

        FSG_ABC (FSGElasticStencilBase &_fsg) :
            FSGBoundaryCondition(_fsg), fsg(_fsg)
        {
        }

        yc_bool_node_ptr is_at_boundary()
        {
            yc_bool_node_ptr bc =
                ( z < first_domain_index(z)+abc_width || z > last_domain_index(z)-abc_width ) ||
                ( y < first_domain_index(y)+abc_width || y > last_domain_index(y)-abc_width ) ||
                ( x < first_domain_index(x)+abc_width || x > last_domain_index(x)-abc_width );
            return bc;
        }
        yc_bool_node_ptr is_not_at_boundary()
        {
            return !is_at_boundary();
        }

        template<typename N, typename SZ, typename SX, typename SY>
        void define_vel_abc(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                            yc_var_proxy &v, yc_var_proxy &sx, yc_var_proxy &sy, yc_var_proxy &sz,
                            yc_var_proxy &abc_x, yc_var_proxy &abc_y, yc_var_proxy &abc_z, yc_var_proxy &abc_sq_x, yc_var_proxy &abc_sq_y, yc_var_proxy &abc_sq_z) {

            yc_bool_node_ptr at_abc = is_at_boundary();

            auto next_v = v(t, x, y, z) * abc_x(x,y,z) * abc_y(x,y,z) * abc_z(x,y,z);

            auto lrho   = fsg.interp_rho<N>( x, y, z );

            auto stx    = fsg.stencil_O2_X<SX>( t, x, y, z, sx );
            auto sty    = fsg.stencil_O2_Y<SY>( t, x, y, z, sy );
            auto stz    = fsg.stencil_O2_Z<SZ>( t, x, y, z, sz );

            next_v += ((stx + sty + stz) * fsg.delta_t * lrho);
            next_v *= abc_sq_x(x,y,z) * abc_sq_y(x,y,z) * abc_sq_z(x,y,z);

            // define the value at t+1.
            v(t+1, x, y, z) EQUALS next_v IF_DOMAIN at_abc;
        }

        void velocity (yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z )
        {
            define_vel_abc<TL, B, F, B>(t, x, y, z, fsg.v_tl_w, fsg.s_tl_yz, fsg.s_tr_xz, fsg.s_bl_zz,
                                        sponge_lx, sponge_by, sponge_tz, sponge_sq_lx, sponge_sq_by, sponge_sq_tz);
            define_vel_abc<TR, B, B, F>(t, x, y, z, fsg.v_tr_w, fsg.s_tr_yz, fsg.s_tl_xz, fsg.s_br_zz,
                                        sponge_rx, sponge_fy, sponge_tz, sponge_sq_rx, sponge_sq_fy, sponge_sq_tz);
            define_vel_abc<BL, F, B, B>(t, x, y, z, fsg.v_bl_w, fsg.s_bl_yz, fsg.s_br_xz, fsg.s_tl_zz,
                                        sponge_lx, sponge_fy, sponge_bz, sponge_sq_lx, sponge_sq_fy, sponge_sq_bz);
            define_vel_abc<BR, F, F, F>(t, x, y, z, fsg.v_br_w, fsg.s_br_yz, fsg.s_bl_xz, fsg.s_tr_zz,
                                        sponge_rx, sponge_by, sponge_bz, sponge_sq_rx, sponge_sq_by, sponge_sq_bz);
            define_vel_abc<TL, B, F, B>(t, x, y, z, fsg.v_tl_u, fsg.s_tl_xy, fsg.s_tr_xx, fsg.s_bl_xz,
                                        sponge_lx, sponge_by, sponge_tz, sponge_sq_lx, sponge_sq_by, sponge_sq_tz);
            define_vel_abc<TR, B, B, F>(t, x, y, z, fsg.v_tr_u, fsg.s_tr_xy, fsg.s_tl_xx, fsg.s_br_xz,
                                        sponge_rx, sponge_fy, sponge_tz, sponge_sq_rx, sponge_sq_fy, sponge_sq_tz);
            define_vel_abc<BL, F, B, B>(t, x, y, z, fsg.v_bl_u, fsg.s_bl_xy, fsg.s_br_xx, fsg.s_tl_xz,
                                        sponge_lx, sponge_fy, sponge_bz, sponge_sq_lx, sponge_sq_fy, sponge_sq_bz);
            define_vel_abc<BR, F, F, F>(t, x, y, z, fsg.v_br_u, fsg.s_br_xy, fsg.s_bl_xx, fsg.s_tr_xz,
                                        sponge_rx, sponge_by, sponge_bz, sponge_sq_rx, sponge_sq_by, sponge_sq_bz);
            define_vel_abc<TL, B, F, B>(t, x, y, z, fsg.v_tl_v, fsg.s_tl_yy, fsg.s_tr_xy, fsg.s_bl_yz,
                                        sponge_lx, sponge_by, sponge_tz, sponge_sq_lx, sponge_sq_by, sponge_sq_tz);
            define_vel_abc<TR, B, B, F>(t, x, y, z, fsg.v_tr_v, fsg.s_tr_yy, fsg.s_tl_xy, fsg.s_br_yz,
                                        sponge_rx, sponge_fy, sponge_tz, sponge_sq_rx, sponge_sq_fy, sponge_sq_tz);
            define_vel_abc<BL, F, B, B>(t, x, y, z, fsg.v_bl_v, fsg.s_bl_yy, fsg.s_br_xy, fsg.s_tl_yz,
                                        sponge_lx, sponge_fy, sponge_bz, sponge_sq_lx, sponge_sq_fy, sponge_sq_bz);
            define_vel_abc<BR, F, F, F>(t, x, y, z, fsg.v_br_v, fsg.s_br_yy, fsg.s_bl_xy, fsg.s_tr_yz,
                                        sponge_rx, sponge_by, sponge_bz, sponge_sq_rx, sponge_sq_by, sponge_sq_bz);
        }

        template<typename N, typename SZ, typename SX, typename SY>
        void define_str_abc(yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z,
                            yc_var_proxy &sxx, yc_var_proxy &syy, yc_var_proxy &szz, yc_var_proxy &sxy, yc_var_proxy &sxz, yc_var_proxy &syz,
                            yc_var_proxy &vxu, yc_var_proxy &vxv, yc_var_proxy &vxw, yc_var_proxy &vyu, yc_var_proxy &vyv, yc_var_proxy &vyw, yc_var_proxy &vzu, yc_var_proxy &vzv, yc_var_proxy &vzw,
                            yc_var_proxy &abc_x, yc_var_proxy &abc_y, yc_var_proxy &abc_z, yc_var_proxy &abc_sq_x, yc_var_proxy &abc_sq_y, yc_var_proxy &abc_sq_z) {

            auto abc = abc_x(x,y,z) * abc_y(x,y,z) * abc_z(x,y,z);
            auto next_sxx = sxx(t, x, y, z) * abc;
            auto next_syy = syy(t, x, y, z) * abc;
            auto next_szz = szz(t, x, y, z) * abc;
            auto next_syz = syz(t, x, y, z) * abc;
            auto next_sxz = sxz(t, x, y, z) * abc;
            auto next_sxy = sxy(t, x, y, z) * abc;

            // Interpolate coeffs.
            auto ic11 = fsg.cell_coeff     <N>(x, y, z, fsg.c11);
            auto ic12 = fsg.cell_coeff     <N>(x, y, z, fsg.c12);
            auto ic13 = fsg.cell_coeff     <N>(x, y, z, fsg.c13);
            auto ic14 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c14);
            auto ic15 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c15);
            auto ic16 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c16);
            auto ic22 = fsg.cell_coeff     <N>(x, y, z, fsg.c22);
            auto ic23 = fsg.cell_coeff     <N>(x, y, z, fsg.c23);
            auto ic24 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c24);
            auto ic25 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c25);
            auto ic26 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c26);
            auto ic33 = fsg.cell_coeff     <N>(x, y, z, fsg.c33);
            auto ic34 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c34);
            auto ic35 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c35);
            auto ic36 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c36);
            auto ic44 = fsg.cell_coeff     <N>(x, y, z, fsg.c44);
            auto ic45 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c45);
            auto ic46 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c46);
            auto ic55 = fsg.cell_coeff     <N>(x, y, z, fsg.c55);
            auto ic56 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c56);
            auto ic66 = fsg.cell_coeff     <N>(x, y, z, fsg.c66);

            // Compute stencils. Note that we are using the velocity values at t+1.
            auto u_z = fsg.stencil_O2_Z<SZ>( t+1, x, y, z, vzu );
            auto v_z = fsg.stencil_O2_Z<SZ>( t+1, x, y, z, vzv );
            auto w_z = fsg.stencil_O2_Z<SZ>( t+1, x, y, z, vzw );

            auto u_x = fsg.stencil_O2_X<SX>( t+1, x, y, z, vxu );
            auto v_x = fsg.stencil_O2_X<SX>( t+1, x, y, z, vxv );
            auto w_x = fsg.stencil_O2_X<SX>( t+1, x, y, z, vxw );

            auto u_y = fsg.stencil_O2_Y<SY>( t+1, x, y, z, vyu );
            auto v_y = fsg.stencil_O2_Y<SY>( t+1, x, y, z, vyv );
            auto w_y = fsg.stencil_O2_Y<SY>( t+1, x, y, z, vyw );

            // Compute next stress value
            auto abc_sq = abc_sq_x(x,y,z) * abc_sq_y(x,y,z) * abc_sq_z(x,y,z);
            next_sxx += fsg.stress_update(ic11,ic12,ic13,ic14,ic15,ic16,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;
            next_syy += fsg.stress_update(ic12,ic22,ic23,ic24,ic25,ic26,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;
            next_szz += fsg.stress_update(ic13,ic23,ic33,ic34,ic35,ic36,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;
            next_syz += fsg.stress_update(ic14,ic24,ic34,ic44,ic45,ic46,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;
            next_sxz += fsg.stress_update(ic15,ic25,ic35,ic45,ic55,ic56,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;
            next_sxy += fsg.stress_update(ic16,ic26,ic36,ic46,ic56,ic66,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;

            // define the value at t+1.
            yc_bool_node_ptr at_abc = is_at_boundary();
            sxx(t+1, x, y, z) EQUALS next_sxx IF_DOMAIN at_abc;
            syy(t+1, x, y, z) EQUALS next_syy IF_DOMAIN at_abc;
            szz(t+1, x, y, z) EQUALS next_szz IF_DOMAIN at_abc;
            syz(t+1, x, y, z) EQUALS next_syz IF_DOMAIN at_abc;
            sxz(t+1, x, y, z) EQUALS next_sxz IF_DOMAIN at_abc;
            sxy(t+1, x, y, z) EQUALS next_sxy IF_DOMAIN at_abc;
        }

        void stress (yc_number_node_ptr t, yc_number_node_ptr x, yc_number_node_ptr y, yc_number_node_ptr z )
        {
            define_str_abc<BR, F, B, F>(t, x, y, z, fsg.s_br_xx, fsg.s_br_yy, fsg.s_br_zz, fsg.s_br_xy, fsg.s_br_xz,
                                        fsg.s_br_yz, fsg.v_br_u,  fsg.v_br_v,  fsg.v_br_w,  fsg.v_bl_u,
                                        fsg.v_bl_v,  fsg.v_bl_w,  fsg.v_tr_u,  fsg.v_tr_v,  fsg.v_tr_w,
                                        sponge_rx, sponge_by, sponge_bz, sponge_sq_rx, sponge_sq_by, sponge_sq_bz);
            define_str_abc<BL, F, F, B>(t, x, y, z, fsg.s_bl_xx, fsg.s_bl_yy, fsg.s_bl_zz, fsg.s_bl_xy, fsg.s_bl_xz,
                                        fsg.s_bl_yz, fsg.v_bl_u,  fsg.v_bl_v,  fsg.v_bl_w,  fsg.v_br_u,
                                        fsg.v_br_v,  fsg.v_br_w,  fsg.v_tl_u,  fsg.v_tl_v,  fsg.v_tl_w,
                                        sponge_lx, sponge_fy, sponge_bz, sponge_sq_lx, sponge_sq_fy, sponge_sq_bz);
            define_str_abc<TR, B, F, F>(t, x, y, z, fsg.s_tr_xx, fsg.s_tr_yy, fsg.s_tr_zz, fsg.s_tr_xy, fsg.s_tr_xz,
                                        fsg.s_tr_yz, fsg.v_tr_u,  fsg.v_tr_v,  fsg.v_tr_w,  fsg.v_tl_u,
                                        fsg.v_tl_v,  fsg.v_tl_w,  fsg.v_br_u,  fsg.v_br_v,  fsg.v_br_w,
                                        sponge_rx, sponge_fy, sponge_tz, sponge_sq_rx, sponge_sq_fy, sponge_sq_tz);
            define_str_abc<TL, B, B, B>(t, x, y, z, fsg.s_tl_xx, fsg.s_tl_yy, fsg.s_tl_zz, fsg.s_tl_xy, fsg.s_tl_xz,
                                        fsg.s_tl_yz, fsg.v_tl_u,  fsg.v_tl_v,  fsg.v_tl_w,  fsg.v_tr_u,
                                        fsg.v_tr_v,  fsg.v_tr_w,  fsg.v_bl_u,  fsg.v_bl_v,  fsg.v_bl_w,
                                        sponge_lx, sponge_by, sponge_tz, sponge_sq_lx, sponge_sq_by, sponge_sq_tz);
        }

    };

    struct FSGElasticStencil : public FSGElasticStencilBase {
        FSGElasticStencil() :
            FSGElasticStencilBase("fsg") { }
    };

    struct FSGABCElasticStencil : public FSGElasticStencilBase {
        FSG_ABC abc; // Absorbing Boundary yc_bool_node_ptr

        FSGABCElasticStencil() :
            FSGElasticStencilBase("fsg_abc", &abc),
            abc(*this) { }
    };

// Create an object of type 'FSGElasticStencil',
// making it available in the YASK compiler utility via the
// '-stencil' commmand-line option or the 'stencil=' build option.
static FSGElasticStencil FSGElasticStencil_instance;
// Create an object of type 'FSGABCElasticStencil',
// making it available in the YASK compiler utility via the
// '-stencil' commmand-line option or the 'stencil=' build option.
static FSGABCElasticStencil FSGABCElasticStencil_instance;

}
