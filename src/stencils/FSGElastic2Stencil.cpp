/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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
// This version varies from the original by grouping related grids into
// larger grids with an added dimension.

#include "Soln.hpp"
#include "ElasticStencil/Elastic2Stencil.hpp"

namespace fsg {

    class FSG2_ABC;

    class FSG2BoundaryCondition : public Elastic2BoundaryCondition
    {
    public:
        FSG2BoundaryCondition(StencilBase& base) :
            Elastic2BoundaryCondition(base) {}
        virtual void velocity (GridIndex t, GridIndex x, GridIndex y, GridIndex z) {}
        virtual void stress (GridIndex t, GridIndex x, GridIndex y, GridIndex z) {}
    };

    class FSGElastic2StencilBase : public Elastic2StencilBase {
        friend FSG2_ABC;

    protected:

        // Velocity and stress grids.
        MAKE_GRID(v, t, x, y, z, vidx);
        enum VIDX { V_BL_U, V_BL_V, V_BL_W,
                    V_BR_U, V_BR_V, V_BR_W,
                    V_TL_U, V_TL_V, V_TL_W,
                    V_TR_U, V_TR_V, V_TR_W };

        MAKE_GRID(s, t, x, y, z, sidx);
        enum SIDX { S_BL_XX, S_BL_YY, S_BL_ZZ, S_BL_YZ, S_BL_XZ, S_BL_XY,
                    S_BR_XX, S_BR_YY, S_BR_ZZ, S_BR_YZ, S_BR_XZ, S_BR_XY,
                    S_TL_XX, S_TL_YY, S_TL_ZZ, S_TL_YZ, S_TL_XZ, S_TL_XY,
                    S_TR_XX, S_TR_YY, S_TR_ZZ, S_TR_YZ, S_TR_XZ, S_TR_XY };

        // 3D-spatial coefficients.
        MAKE_GRID(c, x, y, z, cidx);
        enum CIDX { C11, C12, C13, C14, C15, C16,
                    C22, C23, C24, C25, C26,
                    C33, C34, C35, C36,
                    C44, C45, C46,
                    C55, C56,
                    C66 };


    public:

        FSGElastic2StencilBase( const string &name, StencilList& stencils,
                               FSG2BoundaryCondition *bc = NULL) :
            Elastic2StencilBase(name, stencils, bc)
        {
        }

        GridValue cell_coeff( const GridIndex x, const GridIndex y, const GridIndex z,
                              Grid &c, GridIndex cidx, const BR)
        {
            return  1.0 / (0.25*(c(x  , y  , z, cidx) +
                                 c(x  , y+1, z, cidx) +
                                 c(x  , y  , z+1, cidx) +
                                 c(x  , y+1, z+1, cidx)));
        }
        GridValue cell_coeff( const GridIndex x, const GridIndex y, const GridIndex z,
                              Grid &c, GridIndex cidx, const BL)
        {
            return  1.0 / (0.25*(c(x  , y  , z, cidx) +
                                 c(x+1, y  , z, cidx) +
                                 c(x  , y  , z+1, cidx) +
                                 c(x+1, y  , z+1, cidx)));
        }
        GridValue cell_coeff( const GridIndex x, const GridIndex y, const GridIndex z,
                              Grid &c, GridIndex cidx, const TR)
        {
            return  1.0 / (0.25*(c(x  , y  , z, cidx) +
                                 c(x  , y+1, z, cidx) +
                                 c(x+1, y  , z, cidx) +
                                 c(x+1, y+1, z, cidx)));
        }
        GridValue cell_coeff( const GridIndex x, const GridIndex y, const GridIndex z,
                              Grid &c, GridIndex cidx, const TL)
        {
            return  1.0 / c(x  , y  , z, cidx);
        }
        template<typename N>
        GridValue cell_coeff( const GridIndex x, const GridIndex y, const GridIndex z,
                              Grid &c, GridIndex cidx)
        {
            return cell_coeff( x, y, z, c, cidx, N());
        }
        template<typename N>
        GridValue cell_coeff( const GridIndex x, const GridIndex y, const GridIndex z,
                              Grid &c, int cidx) {
            return cell_coeff<N>(x, y, z, c, constNum(cidx));
        }

        GridValue cell_coeff_artm( const GridIndex x, const GridIndex y, const GridIndex z,
                                   Grid &c, GridIndex cidx, const BR)
        {
            return 0.25 *( 1.0 / c(x  , y  , z, cidx) +
                           1.0 / c(x  , y+1, z, cidx) +
                           1.0 / c(x  , y  , z+1, cidx) +
                           1.0 / c(x  , y+1, z+1, cidx));
        }
        GridValue cell_coeff_artm( const GridIndex x, const GridIndex y, const GridIndex z,
                                   Grid &c, GridIndex cidx, const BL)
        {
            return 0.25 *( 1.0 / c(x  , y  , z, cidx) +
                           1.0 / c(x+1, y  , z, cidx) +
                           1.0 / c(x  , y  , z+1, cidx) +
                           1.0 / c(x+1, y  , z+1, cidx));
        }
        GridValue cell_coeff_artm( const GridIndex x, const GridIndex y, const GridIndex z,
                                   Grid &c, GridIndex cidx, const TR)
        {
            return 0.25 *( 1.0 / c(x  , y  , z, cidx) +
                           1.0 / c(x  , y+1, z, cidx) +
                           1.0 / c(x+1, y  , z, cidx) +
                           1.0 / c(x+1, y+1, z, cidx));
        }
        GridValue cell_coeff_artm( const GridIndex x, const GridIndex y, const GridIndex z,
                                   Grid &c, GridIndex cidx, const TL)
        {
            return  1.0 / c(x  , y  , z, cidx);
        }
        template<typename N>
        GridValue cell_coeff_artm( const GridIndex x, const GridIndex y, const GridIndex z,
                                   Grid &c, GridIndex cidx)
        {
            return cell_coeff_artm( x, y, z, c, cidx, N());
        }
        template<typename N>
        GridValue cell_coeff_artm( const GridIndex x, const GridIndex y, const GridIndex z,
                                   Grid &c, int cidx)
        {
            return cell_coeff_artm<N>( x, y, z, c, constNum(cidx));
        }

        GridValue stress_update( GridValue c1, GridValue c2, GridValue c3,
                                 GridValue c4, GridValue c5, GridValue c6,
                                 GridValue u_z, GridValue u_y, GridValue u_x,
                                 GridValue v_z, GridValue v_y, GridValue v_x,
                                 GridValue w_z, GridValue w_y, GridValue w_x)
        {
            return delta_t * c1 * u_x
                + delta_t * c2 * v_y
                + delta_t * c3 * w_z
                + delta_t * c4 * (w_y + v_z)
                + delta_t * c5 * (w_x + u_z)
                + delta_t * c6 * (v_x + u_y);
        }

        //
        // Stress-grid define functions.  For each D in xx, yy, zz, xy, xz, yz,
        // define stress_D at t+1 based on stress_D at t and vel grids at t+1.
        // This implies that the velocity-grid define functions must be called
        // before these for a given value of t.  Note that the t, x, y, z
        // parameters are integer grid indices, not actual offsets in time or
        // space, so half-steps due to staggered grids are adjusted
        // appropriately.

        template<typename N, typename SZ, typename SX, typename SY>
        void define_str(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                        GridIndex sxx_idx, GridIndex syy_idx, GridIndex szz_idx,
                        GridIndex sxy_idx, GridIndex sxz_idx, GridIndex syz_idx,
                        GridIndex vxu_idx, GridIndex vxv_idx, GridIndex vxw_idx,
                        GridIndex vyu_idx, GridIndex vyv_idx, GridIndex vyw_idx,
                        GridIndex vzu_idx, GridIndex vzv_idx, GridIndex vzw_idx) {

            // Interpolate coeffs.
            GridValue ic11 = cell_coeff     <N>(x, y, z, c, C11);
            GridValue ic12 = cell_coeff     <N>(x, y, z, c, C12);
            GridValue ic13 = cell_coeff     <N>(x, y, z, c, C13);
            GridValue ic14 = cell_coeff_artm<N>(x, y, z, c, C14);
            GridValue ic15 = cell_coeff_artm<N>(x, y, z, c, C15);
            GridValue ic16 = cell_coeff_artm<N>(x, y, z, c, C16);
            GridValue ic22 = cell_coeff     <N>(x, y, z, c, C22);
            GridValue ic23 = cell_coeff     <N>(x, y, z, c, C23);
            GridValue ic24 = cell_coeff_artm<N>(x, y, z, c, C24);
            GridValue ic25 = cell_coeff_artm<N>(x, y, z, c, C25);
            GridValue ic26 = cell_coeff_artm<N>(x, y, z, c, C26);
            GridValue ic33 = cell_coeff     <N>(x, y, z, c, C33);
            GridValue ic34 = cell_coeff_artm<N>(x, y, z, c, C34);
            GridValue ic35 = cell_coeff_artm<N>(x, y, z, c, C35);
            GridValue ic36 = cell_coeff_artm<N>(x, y, z, c, C36);
            GridValue ic44 = cell_coeff     <N>(x, y, z, c, C44);
            GridValue ic45 = cell_coeff_artm<N>(x, y, z, c, C45);
            GridValue ic46 = cell_coeff_artm<N>(x, y, z, c, C46);
            GridValue ic55 = cell_coeff     <N>(x, y, z, c, C55);
            GridValue ic56 = cell_coeff_artm<N>(x, y, z, c, C56);
            GridValue ic66 = cell_coeff     <N>(x, y, z, c, C66);

            // Compute stencils. Note that we are using the velocity values at t+1.
            GridValue u_z = stencil_O8<Z,SZ>( t+1, x, y, z, v, vzu_idx);
            GridValue v_z = stencil_O8<Z,SZ>( t+1, x, y, z, v, vzv_idx);
            GridValue w_z = stencil_O8<Z,SZ>( t+1, x, y, z, v, vzw_idx);

            GridValue u_x = stencil_O8<X,SX>( t+1, x, y, z, v, vxu_idx);
            GridValue v_x = stencil_O8<X,SX>( t+1, x, y, z, v, vxv_idx);
            GridValue w_x = stencil_O8<X,SX>( t+1, x, y, z, v, vxw_idx);

            GridValue u_y = stencil_O8<Y,SY>( t+1, x, y, z, v, vyu_idx);
            GridValue v_y = stencil_O8<Y,SY>( t+1, x, y, z, v, vyv_idx);
            GridValue w_y = stencil_O8<Y,SY>( t+1, x, y, z, v, vyw_idx);

            // Compute next stress value
            GridValue next_sxx = s(t, x, y, z, sxx_idx) +
                stress_update(ic11,ic12,ic13,ic14,ic15,ic16,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);
            GridValue next_syy = s(t, x, y, z, syy_idx) +
                stress_update(ic12,ic22,ic23,ic24,ic25,ic26,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);
            GridValue next_szz = s(t, x, y, z, szz_idx) +
                stress_update(ic13,ic23,ic33,ic34,ic35,ic36,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);
            GridValue next_syz = s(t, x, y, z, syz_idx) +
                stress_update(ic14,ic24,ic34,ic44,ic45,ic46,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);
            GridValue next_sxz = s(t, x, y, z, sxz_idx) +
                stress_update(ic15,ic25,ic35,ic45,ic55,ic56,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);
            GridValue next_sxy = s(t, x, y, z, sxy_idx) +
                stress_update(ic16,ic26,ic36,ic46,ic56,ic66,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y);

            // define the value at t+1.
            if(hasBoundaryCondition()) {
                Condition not_at_bc = bc->is_not_at_boundary();
                s(t+1, x, y, z, sxx_idx) EQUALS next_sxx IF not_at_bc;
                s(t+1, x, y, z, syy_idx) EQUALS next_syy IF not_at_bc;
                s(t+1, x, y, z, szz_idx) EQUALS next_szz IF not_at_bc;
                s(t+1, x, y, z, syz_idx) EQUALS next_syz IF not_at_bc;
                s(t+1, x, y, z, sxz_idx) EQUALS next_sxz IF not_at_bc;
                s(t+1, x, y, z, sxy_idx) EQUALS next_sxy IF not_at_bc;
            } else {
                s(t+1, x, y, z, sxx_idx) EQUALS next_sxx;
                s(t+1, x, y, z, syy_idx) EQUALS next_syy;
                s(t+1, x, y, z, szz_idx) EQUALS next_szz;
                s(t+1, x, y, z, syz_idx) EQUALS next_syz;
                s(t+1, x, y, z, sxz_idx) EQUALS next_sxz;
                s(t+1, x, y, z, sxy_idx) EQUALS next_sxy;
            }
        }
        template<typename N, typename SZ, typename SX, typename SY>
        void define_str(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                        int sxx_idx, int syy_idx, int szz_idx,
                        int sxy_idx, int sxz_idx, int syz_idx,
                        int vxu_idx, int vxv_idx, int vxw_idx,
                        int vyu_idx, int vyv_idx, int vyw_idx,
                        int vzu_idx, int vzv_idx, int vzw_idx) {
            define_str<N, SZ, SX, SY>(t, x, y, z,
                       constNum(sxx_idx), constNum(syy_idx), constNum(szz_idx),
                       constNum(sxy_idx), constNum(sxz_idx), constNum(syz_idx),
                       constNum(vxu_idx), constNum(vxv_idx), constNum(vxw_idx),
                       constNum(vyu_idx), constNum(vyv_idx), constNum(vyw_idx),
                       constNum(vzu_idx), constNum(vzv_idx), constNum(vzw_idx));
        }

        // Call all the define_* functions.
        virtual void define() {

            FSG2BoundaryCondition &fsg_bc = *static_cast<FSG2BoundaryCondition *>(bc);

            // Define velocity components.
            define_vel<TL, B, F, B>(t, x, y, z, v, V_TL_W, s, S_TL_YZ, S_TR_XZ, S_BL_ZZ);
            define_vel<TR, B, B, F>(t, x, y, z, v, V_TR_W, s, S_TR_YZ, S_TL_XZ, S_BR_ZZ);
            define_vel<BL, F, B, B>(t, x, y, z, v, V_BL_W, s, S_BL_YZ, S_BR_XZ, S_TL_ZZ);
            define_vel<BR, F, F, F>(t, x, y, z, v, V_BR_W, s, S_BR_YZ, S_BL_XZ, S_TR_ZZ);
            define_vel<TL, B, F, B>(t, x, y, z, v, V_TL_U, s, S_TL_XY, S_TR_XX, S_BL_XZ);
            define_vel<TR, B, B, F>(t, x, y, z, v, V_TR_U, s, S_TR_XY, S_TL_XX, S_BR_XZ);
            define_vel<BL, F, B, B>(t, x, y, z, v, V_BL_U, s, S_BL_XY, S_BR_XX, S_TL_XZ);
            define_vel<BR, F, F, F>(t, x, y, z, v, V_BR_U, s, S_BR_XY, S_BL_XX, S_TR_XZ);
            define_vel<TL, B, F, B>(t, x, y, z, v, V_TL_V, s, S_TL_YY, S_TR_XY, S_BL_YZ);
            define_vel<TR, B, B, F>(t, x, y, z, v, V_TR_V, s, S_TR_YY, S_TL_XY, S_BR_YZ);
            define_vel<BL, F, B, B>(t, x, y, z, v, V_BL_V, s, S_BL_YY, S_BR_XY, S_TL_YZ);
            define_vel<BR, F, F, F>(t, x, y, z, v, V_BR_V, s, S_BR_YY, S_BL_XY, S_TR_YZ);

            if(hasBoundaryCondition())
                fsg_bc.velocity(t,x,y,z);

            //// Define stresses components.
            define_str<BR, F, B, F>(t, x, y, z, S_BR_XX, S_BR_YY, S_BR_ZZ, S_BR_XY, S_BR_XZ, S_BR_YZ,
                                    V_BR_U, V_BR_V, V_BR_W, V_BL_U, V_BL_V, V_BL_W, V_TR_U, V_TR_V, V_TR_W);
            define_str<BL, F, F, B>(t, x, y, z, S_BL_XX, S_BL_YY, S_BL_ZZ, S_BL_XY, S_BL_XZ, S_BL_YZ,
                                    V_BL_U, V_BL_V, V_BL_W, V_BR_U, V_BR_V, V_BR_W, V_TL_U, V_TL_V, V_TL_W);
            define_str<TR, B, F, F>(t, x, y, z, S_TR_XX, S_TR_YY, S_TR_ZZ, S_TR_XY, S_TR_XZ, S_TR_YZ,
                                    V_TR_U, V_TR_V, V_TR_W, V_TL_U, V_TL_V, V_TL_W, V_BR_U, V_BR_V, V_BR_W);
            define_str<TL, B, B, B>(t, x, y, z, S_TL_XX, S_TL_YY, S_TL_ZZ, S_TL_XY, S_TL_XZ, S_TL_YZ,
                                    V_TL_U, V_TL_V, V_TL_W, V_TR_U, V_TR_V, V_TR_W, V_BL_U, V_BL_V, V_BL_W);

            if(hasBoundaryCondition())
                fsg_bc.stress(t,x,y,z);
        }
    };

    class FSG2_ABC : public FSG2BoundaryCondition
    {
    protected:
        const int abc_width = 20;

        // Sponge coefficients.
        MAKE_GRID(sponge, x, y, z, spidx);
        enum SPONGE_IDX { SPONGE_LX, SPONGE_RX, SPONGE_BZ,
                          SPONGE_TZ, SPONGE_FY, SPONGE_BY,
                          SPONGE_SQ_LX, SPONGE_SQ_RX, SPONGE_SQ_BZ,
                          SPONGE_SQ_TZ, SPONGE_SQ_FY, SPONGE_SQ_BY };

        FSGElastic2StencilBase &fsg;

    public:

        FSG2_ABC (FSGElastic2StencilBase &_fsg) :
            FSG2BoundaryCondition(_fsg), fsg(_fsg)
        {
        }

        Condition is_at_boundary()
        {
            Condition bc =
                (z < first_index(z)+abc_width || z > last_index(z)-abc_width) ||
                (y < first_index(y)+abc_width || y > last_index(y)-abc_width) ||
                (x < first_index(x)+abc_width || x > last_index(x)-abc_width);
            return bc;
        }
        Condition is_not_at_boundary()
        {
            return !is_at_boundary();
        }

        template<typename N, typename SZ, typename SX, typename SY>
        void define_vel_abc(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                            GridIndex vidx,
                            GridIndex sx_idx, GridIndex sy_idx, GridIndex sz_idx,
                            GridIndex abc_x_idx, GridIndex abc_y_idx, GridIndex abc_z_idx,
                            GridIndex abc_sq_x_idx, GridIndex abc_sq_y_idx, GridIndex abc_sq_z_idx) {

            Condition at_abc = is_at_boundary();

            GridValue next_v = fsg.v(t, x, y, z, vidx) * sponge(x,y,z, abc_x_idx) *
                sponge(x,y,z, abc_y_idx) * sponge(x,y,z, abc_z_idx);

            GridValue lrho   = fsg.interp_rho<N>( x, y, z);

            GridValue stx    = fsg.stencil_O2_X<SX>( t, x, y, z, fsg.s, sx_idx);
            GridValue sty    = fsg.stencil_O2_Y<SY>( t, x, y, z, fsg.s, sy_idx);
            GridValue stz    = fsg.stencil_O2_Z<SZ>( t, x, y, z, fsg.s, sz_idx);

            next_v += ((stx + sty + stz) * fsg.delta_t * lrho);
            next_v *= sponge(x,y,z, abc_sq_x_idx) * sponge(x,y,z, abc_sq_y_idx) *
                sponge(x,y,z, abc_sq_z_idx);

            // define the value at t+1.
            fsg.v(t+1, x, y, z, vidx) EQUALS next_v IF at_abc;
        }
        template<typename N, typename SZ, typename SX, typename SY>
        void define_vel_abc(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                            int vidx,
                            int sx_idx, int sy_idx, int sz_idx,
                            int abc_x_idx, int abc_y_idx, int abc_z_idx,
                            int abc_sq_x_idx, int abc_sq_y_idx, int abc_sq_z_idx) {
            define_vel_abc<N, SZ, SZ, SY>(t, x, y, z,
                                          constNum(vidx),
                                          constNum(sx_idx), constNum(sy_idx), constNum(sz_idx),
                                          constNum(abc_x_idx), constNum(abc_y_idx), constNum(abc_z_idx),
                                          constNum(abc_sq_x_idx), constNum(abc_sq_y_idx), constNum(abc_sq_z_idx));
        }

        void velocity (GridIndex t, GridIndex x, GridIndex y, GridIndex z)
        {
            define_vel_abc<TL, B, F, B>(t, x, y, z, fsg.V_TL_W, fsg.S_TL_YZ, fsg.S_TR_XZ, fsg.S_BL_ZZ,
                                        SPONGE_LX, SPONGE_BY, SPONGE_TZ, SPONGE_SQ_LX, SPONGE_SQ_BY, SPONGE_SQ_TZ);
            define_vel_abc<TR, B, B, F>(t, x, y, z, fsg.V_TR_W, fsg.S_TR_YZ, fsg.S_TL_XZ, fsg.S_BR_ZZ,
                                        SPONGE_RX, SPONGE_FY, SPONGE_TZ, SPONGE_SQ_RX, SPONGE_SQ_FY, SPONGE_SQ_TZ);
            define_vel_abc<BL, F, B, B>(t, x, y, z, fsg.V_BL_W, fsg.S_BL_YZ, fsg.S_BR_XZ, fsg.S_TL_ZZ,
                                        SPONGE_LX, SPONGE_FY, SPONGE_BZ, SPONGE_SQ_LX, SPONGE_SQ_FY, SPONGE_SQ_BZ);
            define_vel_abc<BR, F, F, F>(t, x, y, z, fsg.V_BR_W, fsg.S_BR_YZ, fsg.S_BL_XZ, fsg.S_TR_ZZ,
                                        SPONGE_RX, SPONGE_BY, SPONGE_BZ, SPONGE_SQ_RX, SPONGE_SQ_BY, SPONGE_SQ_BZ);
            define_vel_abc<TL, B, F, B>(t, x, y, z, fsg.V_TL_U, fsg.S_TL_XY, fsg.S_TR_XX, fsg.S_BL_XZ,
                                        SPONGE_LX, SPONGE_BY, SPONGE_TZ, SPONGE_SQ_LX, SPONGE_SQ_BY, SPONGE_SQ_TZ);
            define_vel_abc<TR, B, B, F>(t, x, y, z, fsg.V_TR_U, fsg.S_TR_XY, fsg.S_TL_XX, fsg.S_BR_XZ,
                                        SPONGE_RX, SPONGE_FY, SPONGE_TZ, SPONGE_SQ_RX, SPONGE_SQ_FY, SPONGE_SQ_TZ);
            define_vel_abc<BL, F, B, B>(t, x, y, z, fsg.V_BL_U, fsg.S_BL_XY, fsg.S_BR_XX, fsg.S_TL_XZ,
                                        SPONGE_LX, SPONGE_FY, SPONGE_BZ, SPONGE_SQ_LX, SPONGE_SQ_FY, SPONGE_SQ_BZ);
            define_vel_abc<BR, F, F, F>(t, x, y, z, fsg.V_BR_U, fsg.S_BR_XY, fsg.S_BL_XX, fsg.S_TR_XZ,
                                        SPONGE_RX, SPONGE_BY, SPONGE_BZ, SPONGE_SQ_RX, SPONGE_SQ_BY, SPONGE_SQ_BZ);
            define_vel_abc<TL, B, F, B>(t, x, y, z, fsg.V_TL_V, fsg.S_TL_YY, fsg.S_TR_XY, fsg.S_BL_YZ,
                                        SPONGE_LX, SPONGE_BY, SPONGE_TZ, SPONGE_SQ_LX, SPONGE_SQ_BY, SPONGE_SQ_TZ);
            define_vel_abc<TR, B, B, F>(t, x, y, z, fsg.V_TR_V, fsg.S_TR_YY, fsg.S_TL_XY, fsg.S_BR_YZ,
                                        SPONGE_RX, SPONGE_FY, SPONGE_TZ, SPONGE_SQ_RX, SPONGE_SQ_FY, SPONGE_SQ_TZ);
            define_vel_abc<BL, F, B, B>(t, x, y, z, fsg.V_BL_V, fsg.S_BL_YY, fsg.S_BR_XY, fsg.S_TL_YZ,
                                        SPONGE_LX, SPONGE_FY, SPONGE_BZ, SPONGE_SQ_LX, SPONGE_SQ_FY, SPONGE_SQ_BZ);
            define_vel_abc<BR, F, F, F>(t, x, y, z, fsg.V_BR_V, fsg.S_BR_YY, fsg.S_BL_XY, fsg.S_TR_YZ,
                                        SPONGE_RX, SPONGE_BY, SPONGE_BZ, SPONGE_SQ_RX, SPONGE_SQ_BY, SPONGE_SQ_BZ);
        }

        template<typename N, typename SZ, typename SX, typename SY>
        void define_str_abc(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                            GridIndex sxx_idx, GridIndex syy_idx, GridIndex szz_idx,
                            GridIndex sxy_idx, GridIndex sxz_idx, GridIndex syz_idx,
                            GridIndex vxu_idx, GridIndex vxv_idx, GridIndex vxw_idx,
                            GridIndex vyu_idx, GridIndex vyv_idx, GridIndex vyw_idx,
                            GridIndex vzu_idx, GridIndex vzv_idx, GridIndex vzw_idx,
                            GridIndex abc_x_idx, GridIndex abc_y_idx, GridIndex abc_z_idx,
                            GridIndex abc_sq_x_idx, GridIndex abc_sq_y_idx, GridIndex abc_sq_z_idx) {

            GridValue abc = sponge(x,y,z, abc_x_idx) * sponge(x,y,z, abc_y_idx) * sponge(x,y,z, abc_z_idx);
            GridValue next_sxx = fsg.s(t, x, y, z, sxx_idx) * abc;
            GridValue next_syy = fsg.s(t, x, y, z, syy_idx) * abc;
            GridValue next_szz = fsg.s(t, x, y, z, szz_idx) * abc;
            GridValue next_syz = fsg.s(t, x, y, z, syz_idx) * abc;
            GridValue next_sxz = fsg.s(t, x, y, z, sxz_idx) * abc;
            GridValue next_sxy = fsg.s(t, x, y, z, sxy_idx) * abc;

            // Interpolate coeffs.
            GridValue ic11 = fsg.cell_coeff     <N>(x, y, z, fsg.c, fsg.C11);
            GridValue ic12 = fsg.cell_coeff     <N>(x, y, z, fsg.c, fsg.C12);
            GridValue ic13 = fsg.cell_coeff     <N>(x, y, z, fsg.c, fsg.C13);
            GridValue ic14 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C14);
            GridValue ic15 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C15);
            GridValue ic16 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C16);
            GridValue ic22 = fsg.cell_coeff     <N>(x, y, z, fsg.c, fsg.C22);
            GridValue ic23 = fsg.cell_coeff     <N>(x, y, z, fsg.c, fsg.C23);
            GridValue ic24 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C24);
            GridValue ic25 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C25);
            GridValue ic26 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C26);
            GridValue ic33 = fsg.cell_coeff     <N>(x, y, z, fsg.c, fsg.C33);
            GridValue ic34 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C34);
            GridValue ic35 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C35);
            GridValue ic36 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C36);
            GridValue ic44 = fsg.cell_coeff     <N>(x, y, z, fsg.c, fsg.C44);
            GridValue ic45 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C45);
            GridValue ic46 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C46);
            GridValue ic55 = fsg.cell_coeff     <N>(x, y, z, fsg.c, fsg.C55);
            GridValue ic56 = fsg.cell_coeff_artm<N>(x, y, z, fsg.c, fsg.C56);
            GridValue ic66 = fsg.cell_coeff     <N>(x, y, z, fsg.c, fsg.C66);

            // Compute stencils. Note that we are using the velocity values at t+1.
            GridValue u_z = fsg.stencil_O2_Z<SZ>( t+1, x, y, z, fsg.v, vzu_idx);
            GridValue v_z = fsg.stencil_O2_Z<SZ>( t+1, x, y, z, fsg.v, vzv_idx);
            GridValue w_z = fsg.stencil_O2_Z<SZ>( t+1, x, y, z, fsg.v, vzw_idx);

            GridValue u_x = fsg.stencil_O2_X<SX>( t+1, x, y, z, fsg.v, vxu_idx);
            GridValue v_x = fsg.stencil_O2_X<SX>( t+1, x, y, z, fsg.v, vxv_idx);
            GridValue w_x = fsg.stencil_O2_X<SX>( t+1, x, y, z, fsg.v, vxw_idx);

            GridValue u_y = fsg.stencil_O2_Y<SY>( t+1, x, y, z, fsg.v, vyu_idx);
            GridValue v_y = fsg.stencil_O2_Y<SY>( t+1, x, y, z, fsg.v, vyv_idx);
            GridValue w_y = fsg.stencil_O2_Y<SY>( t+1, x, y, z, fsg.v, vyw_idx);

            // Compute next stress value
            GridValue abc_sq = sponge(x,y,z, abc_sq_x_idx) *
                sponge(x,y,z, abc_sq_y_idx) * sponge(x,y,z, abc_sq_z_idx);
            next_sxx += fsg.stress_update(ic11,ic12,ic13,ic14,ic15,ic16,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;
            next_syy += fsg.stress_update(ic12,ic22,ic23,ic24,ic25,ic26,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;
            next_szz += fsg.stress_update(ic13,ic23,ic33,ic34,ic35,ic36,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;
            next_syz += fsg.stress_update(ic14,ic24,ic34,ic44,ic45,ic46,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;
            next_sxz += fsg.stress_update(ic15,ic25,ic35,ic45,ic55,ic56,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;
            next_sxy += fsg.stress_update(ic16,ic26,ic36,ic46,ic56,ic66,u_z,u_x,u_y,v_z,v_x,v_y,w_z,w_x,w_y) * abc_sq;

            // define the value at t+1.
            Condition at_abc = is_at_boundary();
            fsg.s(t+1, x, y, z, sxx_idx) EQUALS next_sxx IF at_abc;
            fsg.s(t+1, x, y, z, syy_idx) EQUALS next_syy IF at_abc;
            fsg.s(t+1, x, y, z, szz_idx) EQUALS next_szz IF at_abc;
            fsg.s(t+1, x, y, z, syz_idx) EQUALS next_syz IF at_abc;
            fsg.s(t+1, x, y, z, sxz_idx) EQUALS next_sxz IF at_abc;
            fsg.s(t+1, x, y, z, sxy_idx) EQUALS next_sxy IF at_abc;
        }
        template<typename N, typename SZ, typename SX, typename SY>
        void define_str_abc(GridIndex t, GridIndex x, GridIndex y, GridIndex z,
                            int sxx_idx, int syy_idx, int szz_idx,
                            int sxy_idx, int sxz_idx, int syz_idx,
                            int vxu_idx, int vxv_idx, int vxw_idx,
                            int vyu_idx, int vyv_idx, int vyw_idx,
                            int vzu_idx, int vzv_idx, int vzw_idx,
                            int abc_x_idx, int abc_y_idx, int abc_z_idx,
                            int abc_sq_x_idx, int abc_sq_y_idx, int abc_sq_z_idx) {
            define_str_abc<N, SZ, SX, SY>(t, x, y, z,
                                          constNum(sxx_idx), constNum(syy_idx), constNum(szz_idx),
                                          constNum(sxy_idx), constNum(sxz_idx), constNum(syz_idx),
                                          constNum(vxu_idx), constNum(vxv_idx), constNum(vxw_idx),
                                          constNum(vyu_idx), constNum(vyv_idx), constNum(vyw_idx),
                                          constNum(vzu_idx), constNum(vzv_idx), constNum(vzw_idx),
                                          constNum(abc_x_idx), constNum(abc_y_idx), constNum(abc_z_idx),
                                          constNum(abc_sq_x_idx), constNum(abc_sq_y_idx), constNum(abc_sq_z_idx));
        }

        void stress (GridIndex t, GridIndex x, GridIndex y, GridIndex z)
        {
            define_str_abc<BR, F, B, F>(t, x, y, z, fsg.S_BR_XX, fsg.S_BR_YY, fsg.S_BR_ZZ, fsg.S_BR_XY, fsg.S_BR_XZ,
                                        fsg.S_BR_YZ, fsg.V_BR_U,  fsg.V_BR_V,  fsg.V_BR_W,  fsg.V_BL_U,
                                        fsg.V_BL_V,  fsg.V_BL_W,  fsg.V_TR_U,  fsg.V_TR_V,  fsg.V_TR_W,
                                        SPONGE_RX, SPONGE_BY, SPONGE_BZ, SPONGE_SQ_RX, SPONGE_SQ_BY, SPONGE_SQ_BZ);
            define_str_abc<BL, F, F, B>(t, x, y, z, fsg.S_BL_XX, fsg.S_BL_YY, fsg.S_BL_ZZ, fsg.S_BL_XY, fsg.S_BL_XZ,
                                        fsg.S_BL_YZ, fsg.V_BL_U,  fsg.V_BL_V,  fsg.V_BL_W,  fsg.V_BR_U,
                                        fsg.V_BR_V,  fsg.V_BR_W,  fsg.V_TL_U,  fsg.V_TL_V,  fsg.V_TL_W,
                                        SPONGE_LX, SPONGE_FY, SPONGE_BZ, SPONGE_SQ_LX, SPONGE_SQ_FY, SPONGE_SQ_BZ);
            define_str_abc<TR, B, F, F>(t, x, y, z, fsg.S_TR_XX, fsg.S_TR_YY, fsg.S_TR_ZZ, fsg.S_TR_XY, fsg.S_TR_XZ,
                                        fsg.S_TR_YZ, fsg.V_TR_U,  fsg.V_TR_V,  fsg.V_TR_W,  fsg.V_TL_U,
                                        fsg.V_TL_V,  fsg.V_TL_W,  fsg.V_BR_U,  fsg.V_BR_V,  fsg.V_BR_W,
                                        SPONGE_RX, SPONGE_FY, SPONGE_TZ, SPONGE_SQ_RX, SPONGE_SQ_FY, SPONGE_SQ_TZ);
            define_str_abc<TL, B, B, B>(t, x, y, z, fsg.S_TL_XX, fsg.S_TL_YY, fsg.S_TL_ZZ, fsg.S_TL_XY, fsg.S_TL_XZ,
                                        fsg.S_TL_YZ, fsg.V_TL_U,  fsg.V_TL_V,  fsg.V_TL_W,  fsg.V_TR_U,
                                        fsg.V_TR_V,  fsg.V_TR_W,  fsg.V_BL_U,  fsg.V_BL_V,  fsg.V_BL_W,
                                        SPONGE_LX, SPONGE_BY, SPONGE_TZ, SPONGE_SQ_LX, SPONGE_SQ_BY, SPONGE_SQ_TZ);
        }

    };


    struct FSGElastic2Stencil : public FSGElastic2StencilBase {
        FSGElastic2Stencil(StencilList& stencils) :
            FSGElastic2StencilBase("fsg2", stencils) { }
    };

    struct FSG2ABCElasticStencil : public FSGElastic2StencilBase {
        FSG2_ABC abc; // Absorbing Boundary Condition

        FSG2ABCElasticStencil(StencilList& stencils) :
            FSGElastic2StencilBase("fsg2_abc", stencils, &abc),
            abc(*this) { }
    };

    REGISTER_STENCIL(FSGElastic2Stencil);
    REGISTER_STENCIL(FSG2ABCElasticStencil);

}
