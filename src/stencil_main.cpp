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

// Fix number of regions, if needed.
// Return region size.
idx_t findRegionSize(idx_t& nregs, idx_t dsize, idx_t mult, string dim) {

    // fix nblks if needed.
    if (nregs < 1) nregs = 1;
    if (nregs > dsize) nregs = dsize;
    
    // divide and round up (ceiling).
    idx_t rsize = (dsize + nregs - 1) / nregs;

    // round up to required multiple.
    rsize = ROUND_UP(rsize, mult);

    // recalc number of regions based on steps.
    nregs = (dsize + rsize - 1) / rsize;

    cout << "  dividing problem size " << dsize << " into " << nregs <<
        " region(s) in the " << dim << " dimension, each of size " <<
        rsize << " (or less)" << endl;

    return rsize;
}

// Fix block size, if needed.
// Return number of blocks.
idx_t findNumBlocks(idx_t& bsize, idx_t rsize, idx_t mult, string dim) {
    if (bsize < 1) bsize = 1;
    if (bsize > rsize) bsize = rsize;
    bsize = ROUND_UP(bsize, mult);
    idx_t nblks = (rsize + bsize - 1) / bsize;

    cout << "  dividing region size " << rsize << " into " << nblks <<
        " block(s) in the " << dim << " dimension, each of size " <<
        bsize << " (or less)" << endl;
    return nblks;
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

#ifdef __INTEL_CREW
    kmp_crew_create();
    int numThreads = omp_get_max_threads();
    int crewSize = kmp_crew_get_max_size();
    printf("num threads with crew: %d\n", numThreads * crewSize);
    printf("  num crew-leader threads: %d\n", numThreads);
    printf("  num threads per crew: %d\n", crewSize);
#elif defined(_OPENMP)
    int numThreads = omp_get_max_threads();
    printf("num OpenMP threads: %d\n", numThreads);
#else
    int numThreads = 1;
    printf("\nnum threads: %d\n", numThreads);
#endif

    // options and their defaults.
    int num_trials = 3; // number of trials.
    int nreps = 50;     // number of time-steps per trial, over which performance is averaged.
    int vreps = 0;      // number of verification steps.
    idx_t dn = 1, dx = DEF_PROB_SIZE, dy = DEF_PROB_SIZE, dz = DEF_PROB_SIZE; // problem dimensions.
    idx_t nrn = 1, nrx = 1, nry = 1, nrz = 1;  // number of regions.
    idx_t bn = 1, bx = 64, by = 64, bz = 64;  // size of cache blocks.
    idx_t pn = 0, px = 0, py = 0, pz = 0; // padding.

    // parse options.
    bool help = false;
    for (int argi = 1; argi < argc; argi++) {
        if ( argv[argi][0] == '-' && argv[argi][1] ) {
            string opt = argv[argi];

            // options w/o values.
            if (opt == "-h" || opt == "-help" || opt == "--help") {
                cout << 
                    "usage: [options]\n"
                    "options:\n"
                    " -h:          print this help and the current settings, then exit\n"
                    " -t           <number of trials>, default=" << num_trials << endl <<
                    " -i           <number of performance iterations (time steps) per trial>, default=" <<
                    nreps << endl <<
                    " -v           <number of validation iterations>, default=" << vreps << endl <<
                    " -d           <problem size in all 3 {x,y,z} dimensions>\n" <<
                    " -d{n,x,y,z}  <problem size in one {x,y,z} dimension>, defaults=" <<
                    dn << '*' << dx << '*' << dy << '*' << dz << endl <<
                    " -nregions          <number of OpenMP regions in all 3 {x,y,z} dimensions>\n"
                    " -nregions{n,x,y,z} <number of OpenMP regions in one {n,x,y,z} dimension>, defaults=" <<
                    nrn << '*' << nrx << '*' << nry << '*' << nrz << endl <<
                    " -b                <size of a computation block in all 3 {x,y,z} dimensions>\n" <<
                    " -b{n,x,y,z}       <size of a computation block in one {n,x,y,z} dimension>, defaults=" <<
                    bn << '*' << bx << '*' << by << '*' << bz << endl <<
                    " -p                <extra padding in all 3 {x,y,z} dimensions (in addition to halos)>\n" <<
                    " -p{n,x,y,z}       <extra padding in one {n,x,y,z} dimension>, defaults=" <<
                    pn << '*' << px << '*' << py << '*' << pz << endl <<
                    "examples:\n" <<
                    " " << argv[0] << " -d 64 -t 1 -i 2 -v 2\n" <<
                    " " << argv[0] << " -d 768 -dn 4\n" <<
                    " " << argv[0] << " -dx 512 -dy 256 -dz 128\n" <<
                    " " << argv[0] << " -d 768 -b 16 -nregionsz 2\n" <<
                    "note: 'n' dimension only applies to stencils that use that variable.\n";
                help = true;
            }

            // options w/int values.
            else {

                if (argi >= argc) {
                    cerr << "error: no value for option '" << opt << "'." << endl;
                    exit(1);
                }
                int val = atoi(argv[++argi]);
                if (opt == "-t") num_trials = val;
                else if (opt == "-i") nreps = val;
                else if (opt == "-v") vreps = val;
                else if (opt == "-d") dx = dy = dz = val;
                else if (opt == "-dn") dn = val;
                else if (opt == "-dx") dx = val;
                else if (opt == "-dy") dy = val;
                else if (opt == "-dz") dz = val;
                else if (opt == "-nregions") nrx = nry = nrz = val;
                else if (opt == "-nregionsn") nrn = val;
                else if (opt == "-nregionsx") nrx = val;
                else if (opt == "-nregionsy") nry = val;
                else if (opt == "-nregionsz") nrz = val;
                else if (opt == "-b") bx = by = bz = val;
                else if (opt == "-bn") bn = val;
                else if (opt == "-bx") bx = val;
                else if (opt == "-by") by = val;
                else if (opt == "-bz") bz = val;
                else if (opt == "-p") px = py = pz = val;
                else if (opt == "-pn") pn = val;
                else if (opt == "-px") px = val;
                else if (opt == "-py") py = val;
                else if (opt == "-pz") pz = val;
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

    // Round up vars as needed.
    nreps = roundUp(nreps, TIME_STEPS_PER_ITER, "number of iterations");
    vreps = roundUp(vreps, TIME_STEPS_PER_ITER, "number of validation iterations");
    dn = roundUp(dn, CPTS_N, "problem size in n");
    dx = roundUp(dx, CPTS_X, "problem size in x");
    dy = roundUp(dy, CPTS_Y, "problem size in y");
    dz = roundUp(dz, CPTS_Z, "problem size in z");

    // Determine step sizes based on number of regions.
    cout << "\nRegions:" << endl;
    idx_t rn = findRegionSize(nrn, dn, CPTS_N, "n");
    idx_t rx = findRegionSize(nrx, dx, CPTS_X, "x");
    idx_t ry = findRegionSize(nry, dy, CPTS_Y, "y");
    idx_t rz = findRegionSize(nrz, dz, CPTS_Z, "z");
    idx_t nr = idx_t(nrn) * nrx * nry * nrz;
    cout << " num-regions = " << nr << endl;

    // Determine num blocks based on block sizes.
    cout << "\nBlocks:" << endl;
    idx_t nbn = findNumBlocks(bn, rn, CPTS_N, "n");
    idx_t nbx = findNumBlocks(bx, rx, CPTS_X, "x");
    idx_t nby = findNumBlocks(by, ry, CPTS_Y, "y");
    idx_t nbz = findNumBlocks(bz, rz, CPTS_Z, "z");
    idx_t nb = idx_t(nbn) * nbx * nby * nbz;
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
    idx_t halo_size = STENCIL_ORDER/2 * TIME_STEPS_PER_ITER;
    idx_t padn = ROUND_UP(halo_size, VLEN_N) + pn;
    idx_t padx = ROUND_UP(halo_size, VLEN_X) + px;
    idx_t pady = ROUND_UP(halo_size, VLEN_Y) + py;
    idx_t padz = ROUND_UP(halo_size, VLEN_Z) + pz;
    
    printf("\nSizes in points (n*x*y*z):\n");
    printf(" vector-size = %d*%d*%d*%d\n", VLEN_N, VLEN_X, VLEN_Y, VLEN_Z);
    printf(" cluster-size = %d*%d*%d*%d\n", CPTS_N, CPTS_X, CPTS_Y, CPTS_Z);
    printf(" block-size = %ld*%ld*%ld*%ld\n", bn, bx, by, bz);
    printf(" region-size = %ld*%ld*%ld*%ld\n", rn, rx, ry, rz);
    printf(" problem-size = %ld*%ld*%ld*%ld\n", dn, dx, dy, dz);
    cout << "\nOther settings:\n";
    printf(" stencil-order = %d\n", STENCIL_ORDER); // really just used for halo size.
    printf(" stencil-shape = " STENCIL_NAME "\n");
    printf(" time-dim = %d\n", TIME_DIM_SIZE);
    printf(" vector-len = %d\n", VLEN);
    printf(" padding = %ld+%ld+%ld+%ld\n", pn, px, py, pz);
    printf(" padding-with-halos = %ld+%ld+%ld+%ld\n", padn, padx, pady, padz);
    printf(" num-trials = %d\n", num_trials);
    printf(" num-iterations = %d\n", nreps);
    printf(" time-step-granularity = %d\n", TIME_STEPS_PER_ITER);
    printf(" manual-L1-prefetch-distance = %d\n", PFDL1);
    printf(" manual-L2-prefetch-distance = %d\n", PFDL2);

    const idx_t numpts = dn*dx*dy*dz;
    printf("\nPoints to calculate: %.3fM (%ld * %ld * %ld * %ld)\n",
           (float)numpts/1e6, dn, dx, dy, dz);
    const idx_t numFpOps = numpts * STENCIL_NUM_FP_OPS_SCALAR;
    printf("FP ops: %i per point, %.3fG total\n", STENCIL_NUM_FP_OPS_SCALAR,
           (float)numFpOps/1e9);
    printf("\n");

    if (help) {
        cout << "Exiting due to help option." << endl;
        exit(1);
    }

    // Context for evaluating results.
    StencilContext context;
    
    // Save sizes.
    context.dn = dn;
    context.dx = dx;
    context.dy = dy;
    context.dz = dz;
    
    context.rn = rn;
    context.rx = rx;
    context.ry = ry;
    context.rz = rz;

    context.bn = bn;
    context.bx = bx;
    context.by = by;
    context.bz = bz;

    context.padn = padn;
    context.padx = padx;
    context.pady = pady;
    context.padz = padz;

    // Alloc and init grids.
    context.allocGrids();
    context.initSame();

    // Stencil functions.
    Stencils stencils{ NEW_STENCIL_OBJECTS };
    for (auto stencil : stencils)
        cout << "evaluating stencil function '" << stencil->get_name() << "'." << endl;

    // variables for measuring performance
    double wstart, wstop;
    float best_elapsed_time=0.0f, best_throughput_mpoints=0.0f, best_gflops=0.0f;

    // warmup caches, threading, etc.
    if (num_trials && nreps) {
        int tmp_nreps = min(TIME_STEPS_PER_ITER, nreps);
#ifdef MODEL_CACHE
        if (cache.isEnabled())
            printf("modeling cache...\n");
#endif
        printf("warmup...\n");  fflush(NULL);
        calc_steps_opt(context, stencils, tmp_nreps);

#ifdef MODEL_CACHE
        // print cache stats, then disable.
        if (cache.isEnabled()) {
            printf("done modeling cache...\n");
            cache.dumpStats();
            cache.disable();
        }
#endif
    }

    printf("running %i trial(s)...\n", num_trials); fflush(NULL);
    
    for (int tr = 0; tr < num_trials; tr++) {
        printf("-------------------------------\n");

        SEP_RESUME;
        wstart = getTimeInSecs();

        calc_steps_opt(context, stencils, nreps);

        wstop =  getTimeInSecs();
        SEP_PAUSE;
            
        // report time
        float elapsed_time = (float)(wstop - wstart);
        float normalized_time = elapsed_time/nreps;
        float throughput_mpoints = float(numpts)/(normalized_time*1e6f);
        float gflops = float(numFpOps)/(normalized_time*1e9f);

        printf("-------------------------------\n");
        printf("time:       %8.3f sec\n", elapsed_time );
        printf("throughput: %8.3f MPoints/s\n", throughput_mpoints );
        printf("FP-rate:    %8.3f GFLOPS\n", gflops);

        if (throughput_mpoints > best_throughput_mpoints) {
            best_throughput_mpoints = throughput_mpoints;
            best_elapsed_time = elapsed_time;
            best_gflops = gflops;
        }
    }

    if (num_trials) {
        printf("-------------------------------\n");
        printf("best-time:       %8.3f sec\n", best_elapsed_time );
        printf("best-throughput: %8.3f MPoints/s\n", best_throughput_mpoints );
        printf("best-FP-rate:    %8.3f GFLOPS\n", best_gflops);
    }

    if (vreps) {

        // check the correctness of one iteration.
        printf("\n-------------------------------\n");
        printf("validation...\n");

        // make a ref context for comparisons w/new grids:
        // copy the settings from context, then re-alloc grids.
        StencilContext ref = context;
        ref.allocGrids();

        // init both to same values.
        context.initDiff();
        ref.initDiff();
        idx_t errs = context.compare(ref);
        if( errs == 0 ) {
            cout << "init check passed; continuting with validation..." << endl;
        } else {
            cerr << "INIT CHECK FAILED: " << errs << " mismatches." << endl;
            exit(1);
        }

        // one vector iteration.
        printf("vector iteration(s)...\n");
        calc_steps_opt(context, stencils, vreps);

        // one ref iteration.
        printf("reference iteration(s)...\n");
        calc_steps_ref(ref, stencils, vreps);

        // check for equality.
        errs =  context.compare(ref);
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
