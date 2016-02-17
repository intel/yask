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
int findRegionSize(int& nregs, int dsize, int mult, string dim) {

    // fix nblks if needed.
    if (nregs < 1) nregs = 1;
    if (nregs > dsize) nregs = dsize;
    
    // divide and round up (ceiling).
    int rsize = (dsize + nregs - 1) / nregs;

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
int findNumBlocks(int& bsize, int rsize, int mult, string dim) {
    if (bsize < 1) bsize = 1;
    if (bsize > rsize) bsize = rsize;
    bsize = ROUND_UP(bsize, mult);
    int nblks = (rsize + bsize - 1) / bsize;

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
#if defined(EMU_INTRINSICS) && (VLEN > 1)
    printf("*** WARNING: binary compiled with EMU_INTRINSICS; ignore performance results.\n");
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

#ifdef __INTEL_CREW
    kmp_crew_create();
    int numThreads = omp_get_max_threads();
    int crewSize = kmp_crew_get_max_size();
    printf("\nnum threads with crew: %d\n", numThreads * crewSize);
    printf("  num crew-leader threads: %d\n", numThreads);
    printf("  num threads per crew: %d\n", crewSize);
#elif defined(_OPENMP)
    int numThreads = omp_get_max_threads();
    printf("\nnum OpenMP threads: %d\n", numThreads);
#else
    int numThreads = 1;
    printf("\nnum threads: %d\n", numThreads);
#endif

    // options and their defaults.
    int num_trials = 3; // number of trials.
    int nreps = 50;     // number of time-steps per trial, over which performance is averaged.
    int vreps = 0;      // number of verification steps.
    int dx = 768, dy = 768, dz = 768;             // problem dimensions.
    int nrx = 1, nry = 1, nrz = 1;  // number of regions.
    int bx = 64, by = 48, bz = 48;  // size of cache blocks.
    int px = 1, py = 0, pz = 0; // padding.

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
                    " -d           <problem size in all 3 dimensions>\n" <<
                    " -d{x,y,z}    <problem size in {x,y,z} dimension>, defaults=" <<
                    dx << "x" << dy << "x" << dz << endl <<
                    " -nregions         <number of OpenMP regions in all 3 dimensions>\n"
                    " -nregions{x,y,z}  <number of OpenMP regions in {x,y,z} dimension>, defaults=" <<
                    nrx << "x" << nry << "x" << nrz << endl <<
                    " -b                <size of a computation block in all 3 dimensions>\n" <<
                    " -b{x,y,z}         <size of a computation block in {x,y,z} dimension>, defaults=" <<
                    bx << "x" << by << "x" << bz << endl <<
                    " -p                <extra padding in all 3 dimensions>\n" <<
                    " -p{x,y,z}         <extra padding in {x,y,z} dimension>, defaults=" <<
                    px << "x" << py << "x" << pz << endl <<
                    "examples:\n" <<
                    " " << argv[0] << " -d 64 -t 1 -i 2 -v 2\n" <<
                    " " << argv[0] << " -d 768\n" <<
                    " " << argv[0] << " -dx 512 -dy 256 -dz 128\n" <<
                    " " << argv[0] << " -d 768 -b 16 -nregionsz 2\n";
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
                else if (opt == "-dx") dx = val;
                else if (opt == "-dy") dy = val;
                else if (opt == "-dz") dz = val;
                else if (opt == "-nregions") nrx = nry = nrz = val;
                else if (opt == "-nregionsx") nrx = val;
                else if (opt == "-nregionsy") nry = val;
                else if (opt == "-nregionsz") nrz = val;
                else if (opt == "-b") bx = by = bz = val;
                else if (opt == "-bx") bx = val;
                else if (opt == "-by") by = val;
                else if (opt == "-bz") bz = val;
                else if (opt == "-p") px = py = pz = val;
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
            cerr << "error: extraneous parameter '" << argv[argi] << "'." << endl;
            exit(1);
        }
    }

    // done reading args.

    // Round up vars as needed.
    nreps = roundUp(nreps, TIME_STEPS, "number of iterations");
    vreps = roundUp(vreps, TIME_STEPS, "number of validation iterations");
    dx = roundUp(dx, CPTS_X, "problem size in x");
    dy = roundUp(dy, CPTS_Y, "problem size in y");
    dz = roundUp(dz, CPTS_Z, "problem size in z");

    // Determine step sizes based on number of regions.
    cout << "\nRegions:" << endl;
    int rx = findRegionSize(nrx, dx, CPTS_X, "x");
    int ry = findRegionSize(nry, dy, CPTS_Y, "y");
    int rz = findRegionSize(nrz, dz, CPTS_Z, "z");
    size_t nr = size_t(nrx) * nry * nrz;
    cout << " num-regions = " << nr << endl;

    // Determine num blocks based on block sizes.
    cout << "\nBlocks:" << endl;
    int nbx = findNumBlocks(bx, rx, CPTS_X, "x");
    int nby = findNumBlocks(by, ry, CPTS_Y, "y");
    int nbz = findNumBlocks(bz, rz, CPTS_Z, "z");
    size_t nb = size_t(nbx) * nby * nbz;
    cout << " num-blocks-per-region = " << nb << endl;

    // Round up padding as needed.
    px = roundUp(px, VLEN_X, "padding in x");
    py = roundUp(py, VLEN_Y, "padding in y");
    pz = roundUp(pz, VLEN_Z, "padding in z");
    
    printf("\nSizes in points (in inner-to-outer nesting order):\n");
    printf(" vector-size = %dx%dx%d\n", VLEN_X, VLEN_Y, VLEN_Z);
    printf(" cluster-size = %dx%dx%d\n", CPTS_X, CPTS_Y, CPTS_Z);
    printf(" block-size = %dx%dx%d\n", bx, by, bz);
    printf(" region-size = %dx%dx%d\n", rx, ry, rz);
    printf(" problem-size = %dx%dx%d\n", dx, dy, dz);
    cout << "\nOther settings:\n";
    printf(" stencil-order = %d\n", STENCIL_ORDER);
    printf(" stencil-shape = " STENCIL_SHAPE_NAME "\n");
    printf(" num-vars = %d\n", NUM_VARS);
    printf(" num-work-grids = %d\n", NUM_WORKS);
    printf(" vector-len = %d\n", VLEN);
    printf(" padding = %d+%d+%d\n", px, py, pz);
    printf(" num-trials = %d\n", num_trials);
    printf(" num-iterations = %d\n", nreps);
    printf(" time-step-granularity = %d\n", TIME_STEPS);
    printf(" manual-L1-prefetch-distance = %d\n", PFDL1);
    printf(" manual-L2-prefetch-distance = %d\n", PFDL2);

    // Context for evaluating results.
    StencilContext context;
    
    // Save sizes by vector lengths.
    context.dx = dx / VLEN_X;
    context.dy = dy / VLEN_Y;
    context.dz = dz / VLEN_Z;
    
    context.rx = rx / VLEN_X;
    context.ry = ry / VLEN_Y;
    context.rz = rz / VLEN_Z;

    context.bx = bx / VLEN_X;
    context.by = by / VLEN_Y;
    context.bz = bz / VLEN_Z;
    
    if (STENCIL_ORDER % 2) {
        cerr << "error: stencil-order not even." << endl;
        exit(1);
    }

    // Pads must be large enough to hold extra padding plus halos.
    int halo_size = STENCIL_ORDER/2 * TIME_STEPS;
    int h1_pad = ROUND_UP(halo_size, VLEN_X) + px;
    int h2_pad = ROUND_UP(halo_size, VLEN_Y) + py;
    int h3_pad = ROUND_UP(halo_size, VLEN_Z) + pz;

    const size_t numpts = (size_t)dx*dy*dz * NUM_VARS;
    printf("\nMPoints to calculate: %.3f (%i * %i * %i * %i / 1e6)\n",
           (float)numpts/1e6, dx, dy, dz, NUM_VARS);

    if (help) {
        cout << "Exiting due to help option." << endl;
        exit(1);
    }

    // variables for measuring performance
    double wstart, wstop;
    float elapsed_time=0.0f, throughput_mpoints=0.0f;
    float best_elapsed_time=0.0f, best_throughput_mpoints=0.0f;

    printf("\nallocating matrices...\n"); fflush(NULL);
    Grid5d mainGrid(VLEN_X, VLEN_Y, VLEN_Z,
                    dx, dy, dz,
                    h1_pad, h2_pad, h3_pad, "main");
    Grid3d* velGrid = NULL;

#ifdef STENCIL_USES_VEL
    velGrid = new Grid3d(VLEN_X, VLEN_Y, VLEN_Z,
                         dx, dy, dz,
                         0, 0, 0, "vel");
#endif 
    context.grid = &mainGrid;
    context.vel = velGrid;

    printf("initializing matrices...\n"); fflush(NULL);
    mainGrid.getData().set_data();
    if (velGrid)
        velGrid->getData().set_data();

    // warmup caches, threading, etc.
    int tmp_nreps = min(TIME_STEPS, nreps);
    printf("warmup...\n");  fflush(NULL);
#ifdef MODEL_CACHE
    if (cache.isEnabled())
        printf("modeling cache...\n");
#endif
    calc_steps_opt(context, tmp_nreps);

#ifdef MODEL_CACHE
    // print cache stats, then disable.
    if (cache.isEnabled()) {
        printf("done modeling cache...\n");
        cache.dumpStats();
        cache.disable();
    }
#endif

    printf("running %i trial(s)...\n", num_trials); fflush(NULL);
    
    for (int tr = 0; tr < num_trials; tr++) {
        printf("-------------------------------\n");

        SEP_RESUME;
        wstart = getTimeInSecs();

        calc_steps_opt(context, nreps);

        wstop =  getTimeInSecs();
        SEP_PAUSE;
            
        // report time
        elapsed_time = (float)(wstop - wstart);
        float normalized_time = elapsed_time/nreps;   
        throughput_mpoints = numpts/(normalized_time*1e6f);

        printf("-------------------------------\n");
        printf("time:       %8.3f sec\n", elapsed_time );
        printf("throughput: %8.3f MPoints/s\n", throughput_mpoints );

        if (throughput_mpoints > best_throughput_mpoints) {
            best_throughput_mpoints = throughput_mpoints;
            best_elapsed_time = elapsed_time;
        }
    }

    if (num_trials) {
        printf("-------------------------------\n");
        printf("best-time:       %8.3f sec\n", best_elapsed_time );
        printf("best-throughput: %8.3f MPoints/s\n", best_throughput_mpoints );
    }

    if (vreps) {

        // check the correctness of one iteration
        printf("\n-------------------------------\n");
        printf("validation...\n");

        // make a ref matrix and init both to same values.
        Grid5d refGrid(VLEN_X, VLEN_Y, VLEN_Z,
                       dx, dy, dz,
                       h1_pad, h2_pad, h3_pad, "ref");
        refGrid.getData().init_data(55);
        mainGrid.getData().init_data(55);
        if( mainGrid.getData().within_epsilon(refGrid.getData()) ) {
            printf("init check passed; continuting with validation...\n");
        } else {
            printf("INIT CHECK FAILED!\n");
            exit(1);
        }

        // one vector iteration.
        printf("vector iteration(s)...\n");
        calc_steps_opt(context, vreps);

        // one ref iteration.
        printf("reference iteration(s)...\n");
        context.grid = &refGrid;
        calc_steps_ref(context, vreps);

        // check for equality.
        if( mainGrid.getData().within_epsilon(refGrid.getData()) ) {
            printf("\nTEST PASSED!\n");
        } else {
            printf("\nTEST FAILED!\n");
            exit(1);
        }
    } else
        printf("\nRESULTS NOT VERIFIED.\n");
    printf("\n");

    return 0;
}
