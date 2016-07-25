/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

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

// Stencil types.
#include "stencil.hpp"

// Base classes for stencil code.
#include "stencil_calc.hpp"

#include <sstream>
using namespace std;

namespace yask {

    ///// Top-level methods for evaluating reference and optimized stencils.

    // Eval stencil(s) over grid(s) using scalar code.
    void StencilEquations::calc_rank_ref(StencilContext& context)
    {
        init(context);
    
        // Start at a positive point to avoid any calculation referring
        // to negative time.
        idx_t t0 = TIME_DIM_SIZE * 2;

        TRACE_MSG("calc_problem_ref(%ld..%ld, 0..%ld, 0..%ld, 0..%ld, 0..%ld)", 
                  t0, t0 + context.dt - 1,
                  context.dn - 1,
                  context.dx - 1,
                  context.dy - 1,
                  context.dz - 1);
    
        // Time steps.
        // TODO: check that scalar version actually does CPTS_T time steps.
        // (At this point, CPTS_T == 1 for all existing stencil examples.)
        for(idx_t t = t0; t < t0 + context.dt; t += CPTS_T) {

            // equations to evaluate (only one in most stencils).
            for (auto stencil : stencils) {

                // Halo exchange for grid(s) updated by this equation.
                stencil->exchange_halos(context, t, t + CPTS_T);
            
                // grid index (only one in most stencils).
                for (idx_t n = 0; n < context.dn; n++) {

#pragma omp parallel for
                    for(idx_t ix = 0; ix < context.dx; ix++) {

                        CREW_FOR_LOOP
                            for(idx_t iy = 0; iy < context.dy; iy++) {

                                for(idx_t iz = 0; iz < context.dz; iz++) {

                                    TRACE_MSG("%s.calc_scalar(%ld, %ld, %ld, %ld, %ld)", 
                                              stencil->get_name().c_str(), t, n, ix, iy, iz);
                            
                                    // Evaluate the reference scalar code.
                                    stencil->calc_scalar(context, t, n, ix, iy, iz);
                                }
                            }
                    }
                }
            }
        } // iterations.
    }


    // Eval stencil(s) over grid(s) using optimized code.
    void StencilEquations::calc_rank_opt(StencilContext& context)
    {
        init(context);

        // Problem begin points.
        // Start at a positive time point to avoid any calculation referring
        // to negative time.
        idx_t begin_dt = TIME_DIM_SIZE * 2;
        idx_t begin_dn = 0, begin_dx = 0, begin_dy = 0, begin_dz = 0;
    
        // Problem end-points.
        idx_t end_dt = begin_dt + context.dt;
        idx_t end_dn = context.dn;
        idx_t end_dx = context.dx;
        idx_t end_dy = context.dy;
        idx_t end_dz = context.dz;

        TRACE_MSG("calc_rank_opt(%ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld)", 
                  begin_dt, end_dt-1,
                  begin_dn, end_dn-1,
                  begin_dx, end_dx-1,
                  begin_dy, end_dy-1,
                  begin_dz, end_dz-1);
    
        // Steps are based on region sizes.
        idx_t step_dt = context.rt;
        idx_t step_dn = context.rn;
        idx_t step_dx = context.rx;
        idx_t step_dy = context.ry;
        idx_t step_dz = context.rz;

        // Determine spatial skewing angles for temporal wavefronts based on the
        // halos.  This assumes the smallest granularity of calculation is
        // CPTS_* in each dim.
        // We only need non-zero angles if the region size is less than the rank size,
        // i.e., if the region covers the whole rank in a given dimension, no wave-front
        // is needed in thar dim.
        // TODO: make this grid-specific.
        context.angle_n = (context.rn < context.dn) ? ROUND_UP(context.hn, CPTS_N) : 0;
        context.angle_x = (context.rx < context.dx) ? ROUND_UP(context.hx, CPTS_X) : 0;
        context.angle_y = (context.ry < context.dy) ? ROUND_UP(context.hy, CPTS_Y) : 0;
        context.angle_z = (context.rz < context.dz) ? ROUND_UP(context.hz, CPTS_Z) : 0;
        TRACE_MSG("wavefront angles: %ld, %ld, %ld, %ld",
                  context.angle_n, context.angle_x, context.angle_y, context.angle_z);
    
        // Extend end points for overlapping regions due to wavefront angle.
        // For each subsequent time step in a region, the spatial location of
        // each block evaluation is shifted by the angle for each stencil. So,
        // the total shift in a region is the angle * num stencils * num
        // timesteps. Thus, the number of overlapping regions is ceil(total
        // shift / region size).  This assumes stencils are inter-dependent.
        // TODO: calculate stencil inter-dependency in the foldBuilder for each
        // dimension.
        idx_t nshifts = (idx_t(stencils.size()) * context.rt) - 1;
        end_dn += context.angle_n * nshifts;
        end_dx += context.angle_x * nshifts;
        end_dy += context.angle_y * nshifts;
        end_dz += context.angle_z * nshifts;
        TRACE_MSG("virtual domain after wavefront adjustment: %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld", 
                  begin_dt, end_dt-1,
                  begin_dn, end_dn-1,
                  begin_dx, end_dx-1,
                  begin_dy, end_dy-1,
                  begin_dz, end_dz-1);

        // Number of iterations to get from begin_dt to (but not including) end_dt,
        // stepping by step_dt.
        const idx_t num_dt = ((end_dt - begin_dt) + (step_dt - 1)) / step_dt;
        for (idx_t index_dt = 0; index_dt < num_dt; index_dt++)
        {
            // This value of index_dt covers dt from start_dt to stop_dt-1.
            const idx_t start_dt = begin_dt + (index_dt * step_dt);
            const idx_t stop_dt = min(start_dt + step_dt, end_dt);

            // If doing only one time step in a region (default), loop through equations here,
            // and do only one equation at a time in calc_region().
            if (step_dt == 1) {

                for (auto stencil : stencils) {

                    // Halo exchange for grid(s) updated by this equation.
                    stencil->exchange_halos(context, start_dt, stop_dt);

                    // Eval this stencil in calc_region().
                    StencilSet stencil_set;
                    stencil_set.insert(stencil);

                    // Include automatically-generated loop code that calls calc_region() for each region.
#include "stencil_rank_loops.hpp"
                }
            }

            // If doing more than one time step in a region (temporal wave-front),
            // must do all equations in calc_region().
            // TODO: allow doing all equations in region even with one time step for testing.
            else {

                StencilSet stencil_set;
                for (auto stencil : stencils) {

                    // Halo exchange for grid(s) updated by this equation.
                    stencil->exchange_halos(context, start_dt, stop_dt);
                    
                    // Make set of all equations.
                    stencil_set.insert(stencil);
                }
            
                // Include automatically-generated loop code that calls calc_region() for each region.
#include "stencil_rank_loops.hpp"
            }

        }
    }

    // Calculate results within a region.
    // Each region is typically computed in a separate OpenMP 'for' region.
    // In it, we loop over the time steps and the stencil
    // equations and evaluate the blocks in the region.
    void StencilEquations::
    calc_region(StencilContext& context, idx_t start_dt, idx_t stop_dt,
                StencilSet& stencil_set,
                idx_t start_dn, idx_t start_dx, idx_t start_dy, idx_t start_dz,
                idx_t stop_dn, idx_t stop_dx, idx_t stop_dy, idx_t stop_dz)
    {
        TRACE_MSG("calc_region(%ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld)", 
                  start_dt, stop_dt-1,
                  start_dn, stop_dn-1,
                  start_dx, stop_dx-1,
                  start_dy, stop_dy-1,
                  start_dz, stop_dz-1);

        // Steps within a region are based on block sizes.
        const idx_t step_rt = context.bt;
        const idx_t step_rn = context.bn;
        const idx_t step_rx = context.bx;
        const idx_t step_ry = context.by;
        const idx_t step_rz = context.bz;

        // Not yet supporting temporal blocking.
        if (step_rt != 1) {
            cerr << "Error: temporal blocking not yet supported." << endl;
            assert(step_rt == 1);
            exit(1);                // in case assert() is not active.
        }

        // Number of iterations to get from start_dt to (but not including) stop_dt,
        // stepping by step_rt.
        const idx_t num_rt = ((stop_dt - start_dt) + (step_rt - 1)) / step_rt;
    
        // Step through time steps in this region.
        for (idx_t index_rt = 0; index_rt < num_rt; index_rt++) {
        
            // This value of index_rt covers rt from start_rt to stop_rt-1.
            const idx_t start_rt = start_dt + (index_rt * step_rt);
            const idx_t stop_rt = min (start_rt + step_rt, stop_dt);

            // TODO: remove this when temporal blocking is implemented.
            assert(stop_rt == start_rt + 1);
            const idx_t rt = start_rt; // only one time value needed for block.

            // equations to evaluate at this time step.
            for (auto stencil : stencils) {
                if (stencil_set.count(stencil)) {

                    // Actual region boundaries must stay within rank domain.
                    idx_t begin_rn = max<idx_t>(start_dn, 0);
                    idx_t end_rn = min<idx_t>(stop_dn, context.dn);
                    idx_t begin_rx = max<idx_t>(start_dx, 0);
                    idx_t end_rx = min<idx_t>(stop_dx, context.dx);
                    idx_t begin_ry = max<idx_t>(start_dy, 0);
                    idx_t end_ry = min<idx_t>(stop_dy, context.dy);
                    idx_t begin_rz = max<idx_t>(start_dz, 0);
                    idx_t end_rz = min<idx_t>(stop_dz, context.dz);

                    // Only need to loop through the region if any of its blocks are
                    // at least partly inside the domain. For overlapping regions,
                    // they may start outside the domain but enter the domain as
                    // time progresses and their boundaries shift. So, we don't want
                    // to return if this condition isn't met.
                    if (end_rn > begin_rn &&
                        end_rx > begin_rx &&
                        end_ry > begin_ry &&
                        end_rz > begin_rz) {

                        // Set number of threads for a region.
                        context.set_region_threads();

                        // Include automatically-generated loop code that calls
                        // calc_block() for each block in this region.  Loops
                        // through n from begin_rn to end_rn-1; similar for x, y,
                        // and z.  This code typically contains OpenMP loop(s).
#include "stencil_region_loops.hpp"

                        // Reset threads back to max.
                        context.set_max_threads();
                    }
            
                    // Shift spatial region boundaries for next iteration to
                    // implement temporal wavefront.  We only shift backward, so
                    // region loops must increment. They may do so in any order.
                    start_dn -= context.angle_n;
                    stop_dn -= context.angle_n;
                    start_dx -= context.angle_x;
                    stop_dx -= context.angle_x;
                    start_dy -= context.angle_y;
                    stop_dy -= context.angle_y;
                    start_dz -= context.angle_z;
                    stop_dz -= context.angle_z;

                }            
            } // stencil equations.
        } // time.
    }

    // Exchange halo data for the given time.
    void StencilBase::exchange_halos(StencilContext& context, idx_t start_dt, idx_t stop_dt)
    {
#ifdef USE_MPI
        TRACE_MSG("exchange_halos(%ld..%ld)", start_dt, stop_dt);

        // For loops, set vars to step 1 vector always.
        const idx_t step_nv = 1;
        const idx_t step_xv = 1;
        const idx_t step_yv = 1;
        const idx_t step_zv = 1;

        // List of grids updated by this equation.
        // These are the grids that need their halos exchanged.
        auto eqGridPtrs = getEqGridPtrs();

        // TODO: put this loop inside visitNeighbors.
        for (size_t gi = 0; gi < eqGridPtrs.size(); gi++) {

            // Get pointer to generic grid and derived type.
            // TODO: Make this more general.
            auto gp = eqGridPtrs[gi];
#if USING_DIM_N
            auto gpd = dynamic_cast<Grid_TNXYZ*>(gp);
#else
            auto gpd = dynamic_cast<Grid_TXYZ*>(gp);
#endif
            assert(gpd);

            // Determine halo sizes to be exchanged for this grid;
            // context.h* contains the max value across all grids.  The grid
            // contains the halo+pad size actually allocated.
            // Since neither of these is exactly what we want, we use
            // the minimum of these values as a conservative value. TODO:
            // Store the actual halo needed in each grid and use this.
#if USING_DIM_N
            idx_t hn = min(context.hn, gpd->get_pn());
#else
            idx_t hn = 0;
#endif
            idx_t hx = min(context.hx, gpd->get_px());
            idx_t hy = min(context.hy, gpd->get_py());
            idx_t hz = min(context.hz, gpd->get_pz());
            
            // Array to store max number of request handles.
            MPI_Request reqs[StencilContext::Bufs::nBufDirs * context.neighborhood_size];
            int nreqs = 0;

            // Pack data and initiate non-blocking send/receive to/from all neighbors.
            TRACE_MSG("rank %i: exchange_halos: packing data for grid '%s'...",
                      context.my_rank, gp->get_name().c_str());
            context.bufs[gp].visitNeighbors
                (context,
                 [&](idx_t nn, idx_t nx, idx_t ny, idx_t nz,
                     int neighbor_rank,
                     Grid_NXYZ* sendBuf,
                     Grid_NXYZ* rcvBuf)
                 {
                     // Pack and send data if buffer exists.
                     if (sendBuf) {

                         // Set begin/end vars to indicate what part
                         // of main grid to read from.
                         // Init range to whole rank size (inside halos).
                         idx_t begin_n = 0;
                         idx_t begin_x = 0;
                         idx_t begin_y = 0;
                         idx_t begin_z = 0;
                         idx_t end_n = context.dn;
                         idx_t end_x = context.dx;
                         idx_t end_y = context.dy;
                         idx_t end_z = context.dz;

                         // Modify begin and/or end based on direction.
                         if (nn == idx_t(context.rank_prev)) // neighbor is prev N.
                             end_n = hn; // read first halo-width only.
                         if (nn == idx_t(context.rank_next)) // neighbor is next N.
                             begin_n = context.dn - hn; // read last halo-width only.
                         if (nx == idx_t(context.rank_prev)) // neighbor is on left.
                             end_x = hx;
                         if (nx == idx_t(context.rank_next)) // neighbor is on right.
                             begin_x = context.dx - hx;
                         if (ny == idx_t(context.rank_prev)) // neighbor is in front.
                             end_y = hy;
                         if (ny == idx_t(context.rank_next)) // neighbor is in back.
                             begin_y = context.dy - hy;
                         if (nz == idx_t(context.rank_prev)) // neighbor is above.
                             end_z = hz;
                         if (nz == idx_t(context.rank_next)) // neighbor is below.
                             begin_z = context.dz - hz;

                         // Divide indices by vector lengths.
                         // Begin/end vars shouldn't be negative, so '/' is ok.
                         idx_t begin_nv = begin_n / VLEN_N;
                         idx_t begin_xv = begin_x / VLEN_X;
                         idx_t begin_yv = begin_y / VLEN_Y;
                         idx_t begin_zv = begin_z / VLEN_Z;
                         idx_t end_nv = end_n / VLEN_N;
                         idx_t end_xv = end_x / VLEN_X;
                         idx_t end_yv = end_y / VLEN_Y;
                         idx_t end_zv = end_z / VLEN_Z;

                         // TODO: fix this when MPI + wave-front is enabled.
                         idx_t t = start_dt;
                         
                         // Define calc_halo() to copy a vector from main grid to sendBuf.
                         // Index sendBuf using index_* vars because they are zero-based.
#define calc_halo(context, t,                                           \
                  start_nv, start_xv, start_yv, start_zv,               \
                  stop_nv, stop_xv, stop_yv, stop_zv)                   \
                         real_vec_t hval = gpd->readVecNorm(t, ARG_N(start_nv) \
                                                            start_xv, start_yv, start_zv, __LINE__); \
                         sendBuf->writeVecNorm(hval, index_nv,        \
                                               index_xv, index_yv, index_zv, __LINE__)
                         
                         // Include auto-generated loops to invoke calc_halo() from
                         // begin_*v to end_*v;
#include "stencil_halo_loops.hpp"
#undef calc_halo

                         // Send filled buffer to neighbor.
                         const void* buf = (const void*)(sendBuf->getRawData());
                         MPI_Isend(buf, sendBuf->get_num_bytes(), MPI_BYTE,
                                   neighbor_rank, int(gi), context.comm, &reqs[nreqs++]);
                         
                     }

                     // Receive data from same neighbor if buffer exists.
                     if (rcvBuf) {
                         void* buf = (void*)(rcvBuf->getRawData());
                         MPI_Irecv(buf, rcvBuf->get_num_bytes(), MPI_BYTE,
                                   neighbor_rank, int(gi), context.comm, &reqs[nreqs++]);
                     }
                     
                 } );

            // Wait for all to complete.
            // TODO: process each buffer asynchronously immediately upon completion.
            TRACE_MSG("rank %i: exchange_halos: waiting for %i MPI request(s)...",
                      context.my_rank, nreqs);
            MPI_Waitall(nreqs, reqs, MPI_STATUS_IGNORE);
            TRACE_MSG("rank %i: exchange_halos: done waiting for %i MPI request(s).",
                      context.my_rank, nreqs);

            // Unpack received data from all neighbors.
            context.bufs[gp].visitNeighbors
                (context,
                 [&](idx_t nn, idx_t nx, idx_t ny, idx_t nz,
                     int neighbor_rank,
                     Grid_NXYZ* sendBuf,
                     Grid_NXYZ* rcvBuf)
                 {
                     // Unpack data if buffer exists.
                     if (rcvBuf) {

                         // Set begin/end vars to indicate what part
                         // of main grid's halo to write to.
                         // Init range to whole rank size (inside halos).
                         idx_t begin_n = 0;
                         idx_t begin_x = 0;
                         idx_t begin_y = 0;
                         idx_t begin_z = 0;
                         idx_t end_n = context.dn;
                         idx_t end_x = context.dx;
                         idx_t end_y = context.dy;
                         idx_t end_z = context.dz;
                         
                         // Modify begin and/or end based on direction.
                         if (nn == idx_t(context.rank_prev)) { // neighbor is prev N.
                             begin_n = -hn; // begin at outside of halo.
                             end_n = 0;     // end at inside of halo.
                         }
                         if (nn == idx_t(context.rank_next)) { // neighbor is next N.
                             begin_n = context.dn; // begin at inside of halo.
                             end_n = context.dn + hn; // end of outside of halo.
                         }
                         if (nx == idx_t(context.rank_prev)) { // neighbor is on left.
                             begin_x = -hx;
                             end_x = 0;
                         }
                         if (nx == idx_t(context.rank_next)) { // neighbor is on right.
                             begin_x = context.dx;
                             end_x = context.dx + hx;
                         }
                         if (ny == idx_t(context.rank_prev)) { // neighbor is in front.
                             begin_y = -hy;
                             end_y = 0;
                         }
                         if (ny == idx_t(context.rank_next)) { // neighbor is in back.
                             begin_y = context.dy;
                             end_y = context.dy + hy;
                         }
                         if (nz == idx_t(context.rank_prev)) { // neighbor is above.
                             begin_z = -hz;
                             end_z = 0;
                         }
                         if (nz == idx_t(context.rank_next)) { // neighbor is below.
                             begin_z = context.dz;
                             end_z = context.dz + hz;
                         }

                         // Divide indices by vector lengths.
                         // Begin/end vars shouldn't be negative, so '/' is ok.
                         idx_t begin_nv = begin_n / VLEN_N;
                         idx_t begin_xv = begin_x / VLEN_X;
                         idx_t begin_yv = begin_y / VLEN_Y;
                         idx_t begin_zv = begin_z / VLEN_Z;
                         idx_t end_nv = end_n / VLEN_N;
                         idx_t end_xv = end_x / VLEN_X;
                         idx_t end_yv = end_y / VLEN_Y;
                         idx_t end_zv = end_z / VLEN_Z;

                         // TODO: fix this when MPI + wave-front is enabled.
                         idx_t t = start_dt;
                         
                         // Define calc_halo to copy data from rcvBuf into main grid.
#define calc_halo(context, t,                                           \
                  start_nv, start_xv, start_yv, start_zv,               \
                  stop_nv, stop_xv, stop_yv, stop_zv)                   \
            real_vec_t hval = rcvBuf->readVecNorm(index_nv,             \
                                                  index_xv, index_yv, index_zv, __LINE__); \
            gpd->writeVecNorm(hval, t, ARG_N(start_nv)                  \
                              start_xv, start_yv, start_zv, __LINE__)

                         // Include auto-generated loops to invoke calc_halo() from
                         // begin_*v to end_*v;
#include "stencil_halo_loops.hpp"
#undef calc_halo
                     }
                 } );

        } // grids.
#endif
    }
                         
            
    ///// StencilContext functions:

    // Init MPI-related vars.
    void StencilContext::setupMPI() {

        // Determine my position in 4D.
        Layout_4321 rank_layout(nrn, nrx, nry, nrz);
        idx_t mrnn, mrnx, mrny, mrnz;
        rank_layout.unlayout((idx_t)my_rank, mrnn, mrnx, mrny, mrnz);
        cout << "Logical coordinates of rank " << my_rank << ": " <<
            mrnn << ", " << mrnx << ", " << mrny << ", " << mrnz << endl;

        // Determine who my neighbors are.
        int num_neighbors = 0;
        for (int rn = 0; rn < num_ranks; rn++) {
            if (rn != my_rank) {
                idx_t rnn, rnx, rny, rnz;
                rank_layout.unlayout((idx_t)rn, rnn, rnx, rny, rnz);

                // Distance from me: prev => -1, self => 0, next => +1.
                idx_t rdn = rnn - mrnn;
                idx_t rdx = rnx - mrnx;
                idx_t rdy = rny - mrny;
                idx_t rdz = rnz - mrnz;

                // Rank rn is my neighbor if its distance <= 1 in every dim.
                if (abs(rdn) <= 1 && abs(rdx) <= 1 && abs(rdy) <= 1 && abs(rdz) <= 1) {

                    num_neighbors++;
                    cout << "Neighbor #" << num_neighbors << " at " <<
                        rnn << ", " << rnx << ", " << rny << ", " << rnz <<
                        " is rank " << rn << endl;
                    
                    // Size of buffer in each direction:
                    // if dist to neighbor is zero (i.e., is self), use full size,
                    // otherwise, use halo size.
                    idx_t rsn = (rdn == 0) ? dn : hn;
                    idx_t rsx = (rdx == 0) ? dx : hx;
                    idx_t rsy = (rdy == 0) ? dy : hy;
                    idx_t rsz = (rdz == 0) ? dz : hz;

                    // FIXME: only alloc buffers in directions actually needed, e.g.,
                    // many simple stencils don't need diagonals.
                    
                    // Is buffer needed?
                    if (rsn * rsx * rsy * rsz == 0) {
                        cout << "No halo exchange needed between ranks " << my_rank <<
                            " and " << rn << '.' << endl;
                    }

                    else {

                        // Add one to -1..+1 dist to get 0..2 range for my_neighbors indices.
                        rdn++; rdx++; rdy++; rdz++;

                        // Save rank of this neighbor.
                        my_neighbors[rdn][rdx][rdy][rdz] = rn;
                    
                        // Alloc MPI buffers between rn and me.
                        // Need send and receive for each updated grid.
                        for (auto gp : eqGridPtrs) {
                            for (int bd = 0; bd < Bufs::nBufDirs; bd++) {
                                ostringstream oss;
                                oss << gp->get_name();
                                if (bd == Bufs::bufSend)
                                    oss << "_send_halo_from_" << my_rank << "_to_" << rn;
                                else
                                    oss << "_get_halo_by_" << my_rank << "_from_" << rn;

                                bufs[gp].allocBuf(bd, rdn, rdx, rdy, rdz,
                                                  rsn, rsx, rsy, rsz,
                                                  oss.str());
                            }
                        }
                    }
                }
            }
        }
    }

    // Get total size.
    idx_t StencilContext::get_num_bytes() {
        idx_t nbytes = 0;
        for (auto gp : gridPtrs)
            nbytes += gp->get_num_bytes();
        for (auto pp : paramPtrs)
            nbytes += pp->get_num_bytes();
        for (auto gp : eqGridPtrs) {
            bufs[gp].visitNeighbors
                (*this,
                 [&](idx_t nn, idx_t nx, idx_t ny, idx_t nz,
                     int rank,
                     Grid_NXYZ* sendBuf,
                     Grid_NXYZ* rcvBuf)
                 {
                     if (sendBuf)
                         nbytes += sendBuf->get_num_bytes();
                     if (rcvBuf)
                         nbytes += rcvBuf->get_num_bytes();
                 } );
        }
        return nbytes;
    }

    // Init all grids & params w/same value within each,
    // but different values between them.
    void StencilContext::initSame() {
        real_t v = 0.1;
        cout << "Initializing grids..." << endl;
        for (auto gp : gridPtrs) {
            gp->set_same(v);
            v += 0.01;
        }
        if (paramPtrs.size()) {
            cout << "Initializing parameters..." << endl;
            for (auto pp : paramPtrs) {
                pp->set_same(v);
                v += 0.01;
            }
        }
    }

    // Init all grids & params w/different values.
    // Better for validation, but slower.
    void StencilContext::initDiff() {
        real_t v = 0.01;
        cout << "Initializing grids..." << endl;
        for (auto gp : gridPtrs) {
            gp->set_diff(v);
            v += 0.001;
        }
        if (paramPtrs.size()) {
            cout << "Initializing parameters..." << endl;
            for (auto pp : paramPtrs) {
                pp->set_diff(v);
                v += 0.001;
            }
        }
    }

    // Compare grids in contexts.
    // Params should not be written to, so they are not compared.
    // Return number of mis-compares.
    idx_t StencilContext::compare(const StencilContext& ref) const {

        cout << "Comparing grid(s) in '" << name << "' to '" << ref.name << "'..." << endl;
        if (gridPtrs.size() != ref.gridPtrs.size()) {
            cerr << "** number of grids not equal." << endl;
            return 1;
        }
        idx_t errs = 0;
        for (size_t gi = 0; gi < gridPtrs.size(); gi++) {
            cout << "Grid '" << ref.gridPtrs[gi]->get_name() << "'..." << endl;
            errs += gridPtrs[gi]->compare(*ref.gridPtrs[gi]);
        }

        cout << "Comparing parameter(s) in '" << name << "' to '" << ref.name << "'..." << endl;
        if (paramPtrs.size() != ref.paramPtrs.size()) {
            cerr << "** number of params not equal." << endl;
            return 1;
        }
        for (size_t pi = 0; pi < paramPtrs.size(); pi++) {
            errs += paramPtrs[pi]->compare(ref.paramPtrs[pi], EPSILON);
        }

        return errs;
    }
}
