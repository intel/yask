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

#include "stencil.hpp"

///// Top-level methods for evaluating reference and optimized stencils.

// Eval stencil over grid using scalar code.
void StencilEquations::calc_problem_ref(StencilContext& context)
{
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

            // Actual region boundaries must stay within problem domain.
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
            // to exit if this condition isn't met.
            if (end_rn > begin_rn &&
                end_rx > begin_rx &&
                end_ry > begin_ry &&
                end_rz > begin_rz) {

            // Start an OpenMP parallel region.
#pragma omp parallel
                {
                    // Include automatically-generated loop code that calls
                    // calc_block() for each block in this region.  Loops
                    // through n from begin_rn to end_rn-1; similar for x, y,
                    // and z.  This code typically contains OpenMP loop(s).
#include "stencil_region_loops.hpp"
                }
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
    }
}

// Eval stencil over grid using optimized code.
void StencilEquations::calc_problem_opt(StencilContext& context)
{
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

    // Determine spatial skewing angles for temporal wavefronts.  This
    // assumes all spatial dimensions have the same halo and the 'n'
    // dimension has a zero halo.  TODO: calculate halos in the foldBuilder
    // for each dimension separately, including 'n'.
    context.angle_n = 0;
    context.angle_x = max<idx_t>(CPTS_X, STENCIL_ORDER/2);
    context.angle_y = max<idx_t>(CPTS_Y, STENCIL_ORDER/2);
    context.angle_z = max<idx_t>(CPTS_Z, STENCIL_ORDER/2);
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
    TRACE_MSG("domain after wavefront adjustment: %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld", 
              begin_dt, end_dt-1,
              begin_dn, end_dn-1,
              begin_dx, end_dx-1,
              begin_dy, end_dy-1,
              begin_dz, end_dz-1);
    
    // Include automatically-generated loop code that calls calc_region() for each region.
#include "stencil_outer_loops.hpp"

}

// StencilContext functions:

    // Get total size.
idx_t StencilContext::get_num_bytes() {
    idx_t nbytes = 0;
    for (auto gp : gridPtrs)
        nbytes += gp->get_num_bytes();
    for (auto pp : paramPtrs)
        nbytes += pp->get_num_bytes();
    return nbytes;
}

// Init all grids & params w/same value within each,
// but different values between them.
void StencilContext::initSame() {
    REAL v = 0.1;
    cout << "initializing grids..." << endl;
    for (auto gp : gridPtrs) {
        gp->set_same(v);
        v += 0.01;
    }
    cout << "initializing parameters (if any)..." << endl;
    for (auto pp : paramPtrs) {
        pp->set_same(v);
        v += 0.01;
    }
}

// Init all grids & params w/different values.
// Better for validation, but slower.
void StencilContext::initDiff() {
    REAL v = 0.01;
    for (auto gp : gridPtrs) {
        gp->set_diff(v);
        v += 0.001;
    }
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
