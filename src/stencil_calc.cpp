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

                // TODO: add halo exchange.
            
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

        TRACE_MSG("calc_problem_opt(%ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld)", 
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
        // We only need non-zero angles if the region size is less than the rank size.
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

        // Include automatically-generated loop code that calls calc_region() for each region.
#include "stencil_rank_loops.hpp"

    }

    // Calculate results within a region.
    // Each region is typically computed in a separate OpenMP 'for' region.
    // In it, we loop over the time steps and the stencil
    // equations and evaluate the blocks in the region.
    void StencilEquations::
    calc_region(StencilContext& context, 
                idx_t start_dt, idx_t start_dn, idx_t start_dx, idx_t start_dy, idx_t start_dz,
                idx_t stop_dt, idx_t stop_dn, idx_t stop_dx, idx_t stop_dy, idx_t stop_dz)
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

                    // Halo exchange for grid(s) updated by this stencil.
                    stencil->exchange_halos(context, rt,
                                            begin_rn, begin_rx, begin_ry, begin_rz,
                                            end_rn, end_rx, end_ry, end_rz);

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
            
            } // stencil equations.
        } // time.
    }

    // Exchange halo data for the given time.
    void StencilBase::exchange_halos(StencilContext& context, idx_t rt,
                                     idx_t begin_rn, idx_t begin_rx, idx_t begin_ry, idx_t begin_rz,
                                     idx_t end_rn, idx_t end_rx, idx_t end_ry, idx_t end_rz)
    {
#ifdef USE_MPI

        // For now, width of region must be across entire rank to use MPI.
        // TODO: generalize directions.
        if (context.num_ranks > 1 && (begin_rx != 0 || end_rx != context.dx)) {
            cerr << "MPI + wavefront tiling in X-direction is not allowed." << endl;
            exit(1);
        }
    
        // Divide indices by vector lengths.
        // Begin/end vars shouldn't be negative, so '/' is ok.
        const idx_t begin_rnv = begin_rn / VLEN_N;
        //const idx_t begin_rxv = begin_rx / VLEN_X;
        const idx_t begin_ryv = begin_ry / VLEN_Y;
        const idx_t begin_rzv = begin_rz / VLEN_Z;
        const idx_t end_rnv = end_rn / VLEN_N;
        //const idx_t end_rxv = end_rx / VLEN_X;
        const idx_t end_ryv = end_ry / VLEN_Y;
        const idx_t end_rzv = end_rz / VLEN_Z;
    
        // Step 1 vector.
        const idx_t step_rnv = 1;
        const idx_t step_rxv = 1;
        const idx_t step_ryv = 1;
        const idx_t step_rzv = 1;
    
        auto eqGridPtrs = getEqGridPtrs();

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

            // Request handles.
            MPI_Request reqs[context.nBufPos * context.nBufDir];
            int nreqs = 0;

            // Define calc_halo() to copy from main grid.
#define calc_halo(context, rt,                                          \
                  start_rnv, start_rxv, start_ryv, start_rzv,           \
                  stop_rnv, stop_rxv, stop_ryv, stop_rzv)               \
            real_vec_t hval = gpd->readVecNorm(rt, ARG_N(start_rnv)          \
                                          start_rxv, start_ryv, start_rzv, __LINE__); \
            haloGrid->writeVecNorm(hval, index_rnv,                     \
                                   index_rxv, index_ryv, index_rzv, __LINE__)
        
            // Send to left.
            if (context.left_rank != MPI_PROC_NULL) {
                auto haloGrid = context.bufs[gp](context.bufLeft, context.bufSend);

                // Pack left-edge data from main grid.
                idx_t begin_rxv = 0;
                idx_t end_rxv = CEIL_DIV(context.hx, VLEN_X);
#include "stencil_halo_loops.hpp"

                // Send data to left rank.
                const void* buf = (const void*)(haloGrid->getRawData());
                MPI_Isend(buf, haloGrid->get_num_bytes(), MPI_BYTE,
                          context.left_rank, int(gi), context.comm, &reqs[nreqs++]);
            }

            // Receive from right.
            if (context.right_rank != MPI_PROC_NULL) {
                auto haloGrid = context.bufs[gp](context.bufRight, context.bufRec);

                // Get data from right rank.
                void* buf = (void*)(haloGrid->getRawData());
                MPI_Irecv(buf, haloGrid->get_num_bytes(), MPI_BYTE,
                          context.right_rank, int(gi), context.comm, &reqs[nreqs++]);
            }

            // Send to right.
            if (context.right_rank != MPI_PROC_NULL) {
                auto haloGrid = context.bufs[gp](context.bufRight, context.bufSend);

                // Pack right-edge data from main grid.
                idx_t begin_rxv = (context.dx - context.hx) / VLEN_X;
                idx_t end_rxv = CEIL_DIV(context.dx, VLEN_X);
#include "stencil_halo_loops.hpp"

                // Send data to right rank.
                const void* buf = (const void*)(haloGrid->getRawData());
                MPI_Isend(buf, haloGrid->get_num_bytes(), MPI_BYTE,
                          context.right_rank, int(gi), context.comm, &reqs[nreqs++]);
            }

            // Receive from left.
            if (context.left_rank != MPI_PROC_NULL) {
                auto haloGrid = context.bufs[gp](context.bufLeft, context.bufRec);

                // Get data from left rank.
                void* buf = (void*)(haloGrid->getRawData());
                MPI_Irecv(buf, haloGrid->get_num_bytes(), MPI_BYTE,
                          context.left_rank, int(gi), context.comm, &reqs[nreqs++]);

            }
#undef calc_halo

            // Wait for all to complete.
            // TODO: unpack immediately after each read finishes.
            TRACE_MSG("exchange_halos: waiting for %i MPI request(s)", nreqs);
            MPI_Waitall(nreqs, reqs, MPI_STATUS_IGNORE);

            // Define calc_halo to copy data into main grid.
#define calc_halo(context, rt,                                          \
                  start_rnv, start_rxv, start_ryv, start_rzv,           \
                  stop_rnv, stop_rxv, stop_ryv, stop_rzv)               \
            real_vec_t hval = haloGrid->readVecNorm(index_rnv,               \
                                               index_rxv, index_ryv, index_rzv, __LINE__); \
            gpd->writeVecNorm(hval, rt, ARG_N(start_rnv)                \
                              start_rxv, start_ryv, start_rzv, __LINE__)
        
            // Unpack right-halo data into main grid.
            if (context.right_rank != MPI_PROC_NULL) {
                auto haloGrid = context.bufs[gp](context.bufRight, context.bufRec);

                idx_t begin_rxv = context.dx / VLEN_X;
                idx_t end_rxv = (context.dx + context.hx) / VLEN_X;
#include "stencil_halo_loops.hpp"
            }
        
            // Unpack left-halo data into main grid.
            if (context.left_rank != MPI_PROC_NULL) {
                auto haloGrid = context.bufs[gp](context.bufLeft, context.bufRec);

                idx_t begin_rxv = -(context.hx / VLEN_X);
                idx_t end_rxv = 0;
#include "stencil_halo_loops.hpp"
            }

        } // grids.

#undef calc_halo
#endif
            
    }

            
    ///// StencilContext functions:

    // Init MPI-related vars.
    void StencilContext::setupMPI() {

        // something to left?
        if (my_rank > 0)
            left_rank = my_rank - 1;
        else
            left_rank = MPI_PROC_NULL;

        // something to right?
        if (my_rank + 1 < num_ranks)
            right_rank = my_rank + 1;
        else
            right_rank = MPI_PROC_NULL;

        // alloc: width is halo size; rest are same as region.
        for (auto gp : eqGridPtrs)
            for (int bd = 0; bd < nBufDir; bd++) {
                string bds = (bd == bufSend) ? "_send" : "_receive";
                if (left_rank != MPI_PROC_NULL)
                    bufs[gp].newGrid(bufLeft, bd, rn, hx, ry, rz,
                                     gp->get_name() + bds + "_left");
                if (right_rank != MPI_PROC_NULL)
                    bufs[gp].newGrid(bufRight, bd, rn, hx, ry, rz, 
                                     gp->get_name() + bds + "_right");
            }
    }

    // Get total size.
    idx_t StencilContext::get_num_bytes() {
        idx_t nbytes = 0;
        for (auto gp : gridPtrs)
            nbytes += gp->get_num_bytes();
        for (auto pp : paramPtrs)
            nbytes += pp->get_num_bytes();
        for (auto gp : eqGridPtrs)
            for (int bp = 0; bp < nBufPos; bp++)
                for (int bd = 0; bd < nBufDir; bd++)
                    if (bufs[gp](bp, bd))
                        nbytes += bufs[gp](bp, bd)->get_num_bytes();
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
        cout << "Initializing parameters (if any)..." << endl;
        for (auto pp : paramPtrs) {
            pp->set_same(v);
            v += 0.01;
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
        cout << "Initializing parameters (if any)..." << endl;
        for (auto pp : paramPtrs) {
            pp->set_diff(v);
            v += 0.001;
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
