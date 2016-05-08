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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "stencil.hpp"

// Set MODEL_CACHE to 1 or 2 to model that cache level
// and create a global cache object here.
#ifdef MODEL_CACHE
Cache cache(MODEL_CACHE);
#endif

// Fix bsize, if needed, to fit into rsize and be a multiple of mult.
// Return number of blocks.
idx_t findNumSubsets(idx_t& bsize, const string& bname,
                    idx_t rsize, const string& rname,
                    idx_t mult, string dim) {
    if (bsize < 1) bsize = rsize; // 0 => use full size.
    if (bsize > rsize) bsize = rsize;
    bsize = ROUND_UP(bsize, mult);
    idx_t nblks = (rsize + bsize - 1) / bsize;
    idx_t rem = rsize % bsize;
    idx_t nfull_blks = rem ? (nblks - 1) : nblks;

    cout << "  '" << dim << "' dimension: dividing " << rname << " of size " <<
        rsize << " into " << nfull_blks << " " << bname << "(s) of size " << bsize;
    if (rem)
        cout << " plus 1 remainder " << bname << " of size " << rem;
    cout << "." << endl;
    return nblks;
}
idx_t findNumBlocks(idx_t& bsize, idx_t rsize, idx_t mult, string dim) {
    return findNumSubsets(bsize, "block", rsize, "region", mult, dim);
}
idx_t findNumRegions(idx_t& rsize, idx_t dsize, idx_t mult, string dim) {
    return findNumSubsets(rsize, "region", dsize, "problem", mult, dim);
}

// Parse command-line args, run kernel, run validation if requested.
int main(int argc, char** argv)
{
    SEP_PAUSE;

    printf("invocation:");
    for (int i = 0; i < argc; i++)
        printf(" %s", argv[i]);
    printf("\n");

#ifdef DEBUG
    printf("*** WARNING: binary compiled with DEBUG; ignore performance results.\n");
#endif
#if defined(NO_INTRINSICS) && (VLEN > 1)
    printf("*** WARNING: binary compiled with NO_INTRINSICS; ignore performance results.\n");
#endif
#ifdef MODEL_CACHE
    printf("*** WARNING: binary compiled with MODEL_CACHE; ignore performance results.\n");
#endif
#ifdef TRACE_MEM
    printf("*** WARNING: binary compiled with TRACE_MEM; ignore performance results.\n");
#endif
#ifdef TRACE_INTRINSICS
    printf("*** WARNING: binary compiled with TRACE_INTRINSICS; ignore performance results.\n");
#endif

    printf("\n"
           "┌──────────────────────────────────────────┐\n"
           "│  Y.A.S.K. -- Yet Another Stencil Kernel  │\n"
           "│            https://01.org/yask           │\n"
           "│    Intel Corporation, copyright 2016     │\n"
           "└──────────────────────────────────────────┘\n"
           "\nStencil name: " STENCIL_NAME "\n");

    // report threads.
    {
        cout << endl;
#if defined(_OPENMP)
        int numOrigThreads = omp_get_max_threads();
        cout << "Num OpenMP threads: " << numOrigThreads << endl;
        int numWorkers = numOrigThreads;
#ifdef __INTEL_CREW
        cout << "Creating crews..." << endl;
        kmp_crew_create();
        int numThreads = omp_get_max_threads();
        cout << "Num OpenMP threads after crew creation: " << numThreads << endl;
        int crewSize = kmp_crew_get_max_size();
        numWorkers = numThreads * crewSize;
        cout << "Total num crews: " << numWorkers << endl <<
            "  num crew-leader threads: " << numThreads << endl <<
            "  num threads per crew: " << crewSize << endl;
        if (numWorkers == numOrigThreads)
            cout << "Note: num crews == num OpenMP threads before creating crews" << endl;
        else
            cout << "Warning: num crews != num OpenMP threads before creating crews" << endl;
#endif
#else
        int numThreads = 1;
        cout << "Num threads: " << numThreads << endl;
#endif
        cout << endl;
    }
    
    // options and their defaults.
    idx_t num_trials = 3; // number of trials.
    idx_t dt = 50;     // number of time-steps per trial, over which performance is averaged.
    idx_t dn = 1, dx = DEF_PROBLEM_SIZE, dy = DEF_PROBLEM_SIZE, dz = DEF_PROBLEM_SIZE;
    idx_t rt = 1;                         // wavefront time steps.
    idx_t rn = 1, rx = 0, ry = 0, rz = 0;  // region sizes (0 => use problem size).
    idx_t bt = 1;                          // temporal block size.
    idx_t bn = 1, bx = DEF_BLOCK_SIZE, by = DEF_BLOCK_SIZE, bz = DEF_BLOCK_SIZE;  // size of cache blocks.
    idx_t pn = 0, px = 0, py = 0, pz = 0; // padding.
    bool validate = false;

    // parse options.
    bool help = false;
    for (int argi = 1; argi < argc; argi++) {
        if ( argv[argi][0] == '-' && argv[argi][1] ) {
            string opt = argv[argi];

            // options w/o values.
            if (opt == "-h" || opt == "-help" || opt == "--help") {
                cout << 
                    "Usage: [options]\n"
                    "Options:\n"
                    " -h:             print this help and the current settings, then exit\n"
                    " -t <n>          number of trials, default=" << num_trials << endl <<
                    " -dt <n>         problem size in temporal dimension (number of time steps), default=" << dt << endl <<
                    " -d{n,x,y,z} <n> problem size in specified spatial dimension, defaults=" <<
                    dn << '*' << dx << '*' << dy << '*' << dz << endl <<
                    " -d <n>          set same problem size in 3 {x,y,z} spatial dimensions\n" <<
                    " -rt <n>         OpenMP region time steps (for wavefront tiling), default=" << rt << endl <<
                    " -r{n,x,y,z} <n> OpenMP region size in specified spatial dimension, defaults=" <<
                    rn << '*' << rx << '*' << ry << '*' << rz << endl <<
                    " -r <n>          set same OpenMP region size in 3 {x,y,z} spatial dimensions\n"
                    " -b{n,x,y,z} <n> cache block size in specified spatial dimension, defaults=" <<
                    bn << '*' << bx << '*' << by << '*' << bz << endl <<
                    " -b <n>          set same cache block size in 3 {x,y,z} spatial dimensions\n" <<
                    " -p{n,x,y,z} <n> extra padding in specified spatial dimension, defaults=" <<
                    pn << '*' << px << '*' << py << '*' << pz << endl <<
                    " -p <n>          set same padding in 3 {x,y,z} spatial dimensions\n" <<
                    " -i <n>          equivalent to -dt, for backward compatibility\n" <<
                    " -v              validate by comparing to a scalar run, default=" << validate << endl <<
                    "Notes:\n"
                    " A block size of 0 => block size == region size in that dimension.\n"
                    " A region size of 0 => region size == problem size in that dimension.\n"
                    " Control the time steps in each temporal wavefront with -rt:\n"
                    "  1 effectively disables wavefront tiling.\n"
                    "  0 enables wavefront tiling across all time steps in one pass.\n"
                    "  Any value other than 1 also changes the region spatial-size defaults.\n"
                    " Temporal cache blocking is not yet supported => bt == 1.\n"
                    " Validation is very slow and uses 2x memory, so run with very small sizes.\n"
                    " If validation fails, it may be due to rounding error; try building with 8-byte reals.\n"
                    " Validation disables the warmup iteration and sets the default number of trials to 1.\n"
                    "Examples:\n" <<
                    " " << argv[0] << " -d 64 -t 1 -i 2 -v 2\n" <<
                    " " << argv[0] << " -d 768 -dn 4\n" <<
                    " " << argv[0] << " -dx 512 -dy 256 -dz 128\n" <<
                    " " << argv[0] << " -d 2048 -r 512 -b 16\n" <<
                    "note: 'n' dimension only applies to stencils that use that variable.\n";
                help = true;
            }

            // validation.
            else if (opt == "-v") {
                validate = true;
                num_trials = 1;
            }

            // options w/int values.
            else {

                if (argi >= argc) {
                    cerr << "error: no value for option '" << opt << "'." << endl;
                    exit(1);
                }
                int val = atoi(argv[++argi]);
                if (opt == "-t") num_trials = val;
                else if (opt == "-i") dt = val;
                else if (opt == "-dt") dt = val;
                else if (opt == "-dn") dn = val;
                else if (opt == "-dx") dx = val;
                else if (opt == "-dy") dy = val;
                else if (opt == "-dz") dz = val;
                else if (opt == "-d") dx = dy = dz = val;
                else if (opt == "-rt") rt = val;
                else if (opt == "-rn") rn = val;
                else if (opt == "-rx") rx = val;
                else if (opt == "-ry") ry = val;
                else if (opt == "-rz") rz = val;
                else if (opt == "-r") rx = ry = rz = val;
                else if (opt == "-bn") bn = val;
                else if (opt == "-bx") bx = val;
                else if (opt == "-by") by = val;
                else if (opt == "-bz") bz = val;
                else if (opt == "-b") bx = by = bz = val;
                else if (opt == "-pn") pn = val;
                else if (opt == "-px") px = val;
                else if (opt == "-py") py = val;
                else if (opt == "-pz") pz = val;
                else if (opt == "-p") px = py = pz = val;
                else {
                    cerr << "error: option '" << opt << "' not recognized." << endl;
                    exit(1);
                }
            }
        }
        else {
            cerr << "error: extraneous parameter '" <<
                argv[argi] << "'." << endl;
            exit(1);
        }
    }

    // done reading args.
#ifndef USING_DIM_N
    if (dn > 1) {
        cerr << "error: dn = " << dn << ", but stencil '"
            STENCIL_NAME "' doesn't use dimension 'n'." << endl;
        exit(1);
    }
#endif

    // Adjust defaults.
    if (rt != 1) {
        if (!rx) rx = DEF_WAVEFRONT_REGION_SIZE;
        if (!ry) ry = DEF_WAVEFRONT_REGION_SIZE;
        if (!rz) rz = DEF_WAVEFRONT_REGION_SIZE;
    }
    
    // Round up vars as needed.
    dt = roundUp(dt, CPTS_T, "problem size in t (time steps)");
    dn = roundUp(dn, CPTS_N, "problem size in n");
    dx = roundUp(dx, CPTS_X, "problem size in x");
    dy = roundUp(dy, CPTS_Y, "problem size in y");
    dz = roundUp(dz, CPTS_Z, "problem size in z");

    // Determine num regions based on region sizes.
    // Also fix up region sizes as needed.
    cout << "\nRegions:" << endl;
    idx_t nrt = findNumRegions(rt, dt, CPTS_T, "t");
    idx_t nrn = findNumRegions(rn, dn, CPTS_N, "n");
    idx_t nrx = findNumRegions(rx, dx, CPTS_X, "x");
    idx_t nry = findNumRegions(ry, dy, CPTS_Y, "y");
    idx_t nrz = findNumRegions(rz, dz, CPTS_Z, "z");
    idx_t nr = nrn * nrx * nry * nrz;
    cout << " num-regions = " << nr << endl;

    // Determine num blocks based on block sizes.
    // Also fix up block sizes as needed.
    cout << "\nBlocks:" << endl;
    idx_t nbt = findNumBlocks(bt, rt, CPTS_T, "t");
    idx_t nbn = findNumBlocks(bn, rn, CPTS_N, "n");
    idx_t nbx = findNumBlocks(bx, rx, CPTS_X, "x");
    idx_t nby = findNumBlocks(by, ry, CPTS_Y, "y");
    idx_t nbz = findNumBlocks(bz, rz, CPTS_Z, "z");
    idx_t nb = nbn * nbx * nby * nbz;
    cout << " num-blocks-per-region = " << nb << endl;

    // Round up padding as needed.
    pn = roundUp(pn, VLEN_N, "padding in n");
    px = roundUp(px, VLEN_X, "padding in x");
    py = roundUp(py, VLEN_Y, "padding in y");
    pz = roundUp(pz, VLEN_Z, "padding in z");

    // Pads must be large enough to hold extra padding plus halos.
    if (STENCIL_ORDER % 2) {
        cerr << "error: stencil-order not even." << endl;
        exit(1);
    }
    idx_t halo_size = STENCIL_ORDER/2 * CPTS_T;
    idx_t padn = ROUND_UP(halo_size, VLEN_N) + pn;
    idx_t padx = ROUND_UP(halo_size, VLEN_X) + px;
    idx_t pady = ROUND_UP(halo_size, VLEN_Y) + py;
    idx_t padz = ROUND_UP(halo_size, VLEN_Z) + pz;
    
    printf("\nSizes in points per grid (t*n*x*y*z):\n");
    printf(" vector-size = %d*%d*%d*%d*%d\n", VLEN_T, VLEN_N, VLEN_X, VLEN_Y, VLEN_Z);
    printf(" cluster-size = %d*%d*%d*%d*%d\n", CPTS_T, CPTS_N, CPTS_X, CPTS_Y, CPTS_Z);
    printf(" block-size = %ld*%ld*%ld*%ld*%ld\n", bt, bn, bx, by, bz);
    printf(" region-size = %ld*%ld*%ld*%ld*%ld\n", rt, rn, rx, ry, rz);
    printf(" problem-size = %ld*%ld*%ld*%ld*%ld\n", dt, dn, dx, dy, dz);
    cout << "\nOther settings:\n";
    printf(" stencil-order = %d\n", STENCIL_ORDER); // really just used for halo size.
    printf(" stencil-shape = " STENCIL_NAME "\n");
    printf(" time-dim = %d\n", TIME_DIM_SIZE);
    printf(" vector-len = %d\n", VLEN);
    printf(" padding = %ld+%ld+%ld+%ld\n", pn, px, py, pz);
    printf(" padding-with-halos = %ld+%ld+%ld+%ld\n", padn, padx, pady, padz);
    printf(" manual-L1-prefetch-distance = %d\n", PFDL1);
    printf(" manual-L2-prefetch-distance = %d\n", PFDL2);

    // Stencil functions.
    idx_t scalar_fp_ops = 0;
    STENCIL_EQUATIONS stencils;
    idx_t num_stencils = stencils.stencils.size();
    cout << endl << "Num stencil equations = " << num_stencils << ":" << endl;
    for (auto stencil : stencils.stencils) {
        idx_t fpos = stencil->get_scalar_fp_ops();
        cout << "  '" << stencil->get_name() << "': " <<
            fpos << " FP ops per point." << endl;
        scalar_fp_ops += fpos;
    }

    if (help) {
        cout << "Exiting due to help option." << endl;
        exit(1);
    }

    const idx_t numpts = dn*dx*dy*dz;
    printf("\nPoints to calculate per time step: %.3fM (%ld * %ld * %ld * %ld)\n",
           (float)numpts/1e6, dn, dx, dy, dz);
    const idx_t tot_numpts = dt*dn*dx*dy*dz;
    printf("\nTotal points to calculate: %.3fM (%ld * %ld * %ld * %ld * %ld)\n",
           (float)tot_numpts/1e6, dt, dn, dx, dy, dz);
    const idx_t numFpOps = numpts * scalar_fp_ops;
    const idx_t tot_numFpOps = dt * numpts * scalar_fp_ops;
    printf("FP ops: %ld per point, %.3fG per time step, %.3fG total\n",
           scalar_fp_ops, (float)numFpOps/1e9, (float)tot_numFpOps/1e9);
    printf("\n");

    // Context for evaluating results.
    STENCIL_CONTEXT context;
    
    // Save sizes in context struct.
    // - dt not used for allocation; set later.
    // - *s vars set below.
    context.dn = dn;
    context.dx = dx;
    context.dy = dy;
    context.dz = dz;
    
    context.rt = rt;
    context.rn = rn;
    context.rx = rx;
    context.ry = ry;
    context.rz = rz;

    context.bt = bt;
    context.bn = bn;
    context.bx = bx;
    context.by = by;
    context.bz = bz;

    context.padn = padn;
    context.padx = padx;
    context.pady = pady;
    context.padz = padz;

    // Alloc and init grids.
    cout << "allocating grids..." << endl;
    context.allocGrids();
    cout << "allocating parameters (if any)..." << endl;
    context.allocParams();
    idx_t nbytes = context.get_num_bytes();
    cout << "total allocation: " << (float(nbytes)/1e9) <<
        "G byte(s) in " << context.gridPtrs.size() << " grid(s)." << endl;
    context.initSame();

    // Exit if nothing to do.
    if (num_trials < 1) {
        cerr << "Exiting because no trials are specified." << endl;
        exit(1);
    }
    if (tot_numpts < 1) {
        cerr << "Exiting because there are zero points to evaluate." << endl;
        exit(1);
    }

    // warmup caches, threading, etc.
    if (!validate) {

        idx_t tmp_dt = min<idx_t>(CPTS_T, dt);
        context.dt = tmp_dt;
#ifdef MODEL_CACHE
        if (cache.isEnabled())
            printf("modeling cache...\n");
#endif
        printf("warmup...\n");  fflush(NULL);
        stencils.calc_problem_opt(context);

#ifdef MODEL_CACHE
        // print cache stats, then disable.
        if (cache.isEnabled()) {
            printf("done modeling cache...\n");
            cache.dumpStats();
            cache.disable();
        }
#endif
    }

    // variables for measuring performance.
    double wstart, wstop;
    float best_elapsed_time=0.0f, best_throughput_mpoints=0.0f, best_gflops=0.0f;

    // Performance runs.
    printf("Running %li performance trial(s)...\n", num_trials); fflush(NULL);
    context.dt = dt;
    for (idx_t tr = 0; tr < num_trials; tr++) {
        printf("-------------------------------\n");

        // init data for comparison if validating.
        if (validate)
            context.initDiff();
        
        SEP_RESUME;
        wstart = getTimeInSecs();

        // Actual work.
        stencils.calc_problem_opt(context);

        wstop =  getTimeInSecs();
        SEP_PAUSE;
            
        // report time
        float elapsed_time = (float)(wstop - wstart);
        float throughput_mpoints = float(tot_numpts)/(elapsed_time*1e6f);
        float gflops = float(tot_numFpOps)/(elapsed_time*1e9f);

        printf("-------------------------------\n");
        printf("time:       %8.3f sec\n", elapsed_time );
        printf("throughput: %8.3f MPoints/s\n", throughput_mpoints );
        printf("FP-rate:    %8.3f GFLOPS (est.)\n", gflops);

        if (throughput_mpoints > best_throughput_mpoints) {
            best_throughput_mpoints = throughput_mpoints;
            best_elapsed_time = elapsed_time;
            best_gflops = gflops;
        }
    }

    printf("-------------------------------\n");
    printf("best-time:       %8.3f sec\n", best_elapsed_time );
    printf("best-throughput: %8.3f MPoints/s\n", best_throughput_mpoints );
    printf("best-FP-rate:    %8.3f GFLOPS (est.)\n", best_gflops);

    if (validate) {

        // check the correctness of one iteration.
        printf("\n-------------------------------\n");
        printf("Running validation trial...\n");

        // Make a ref context for comparisons w/new grids:
        // Copy the settings from context, then re-alloc grids.
        STENCIL_CONTEXT ref = context;
        ref.name += "-reference";
        ref.allocGrids();
        ref.allocParams();

        // init to same value used in context.
        ref.initDiff();

#if CHECK_INIT
        {
            context.initDiff();
            idx_t errs = context.compare(ref);
            if( errs == 0 ) {
                cout << "INIT CHECK PASSED." << endl;
                exit(0);
            } else {
                cerr << "INIT CHECK FAILED: " << errs << " mismatches." << endl;
                exit(1);
            }
        }
#endif

        // Ref trial.
        stencils.calc_problem_ref(ref);

        // check for equality.
        idx_t errs = context.compare(ref);
        if( errs == 0 ) {
            cout << "TEST PASSED" << endl;
        } else {
            cerr << "TEST FAILED: " << errs << " mismatches." << endl;
            exit(1);
        }
    } else
        printf("\nRESULTS NOT VERIFIED.\n");
    printf("\n");

    return 0;
}
