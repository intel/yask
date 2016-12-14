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
#include <fstream>

// Stencil types.
#include "stencil.hpp"

// Base classes for stencil code.
#include "stencil_calc.hpp"

// Include auto-generated stencil code.
#include "stencil_code.hpp"

using namespace std;
using namespace yask;

// Set MODEL_CACHE to 1 or 2 to model that cache level
// and create a global cache object here.
#ifdef MODEL_CACHE
Cache cache_model(MODEL_CACHE);
#endif

namespace yask {

    // cmd-line options and their defaults.
    idx_t num_trials = 3; // number of trials.
    idx_t dt = 50;     // number of time-steps per trial.
    idx_t dn = 1, dx = DEF_RANK_SIZE, dy = DEF_RANK_SIZE, dz = DEF_RANK_SIZE;
    idx_t rt = 1;                         // wavefront time steps.
    idx_t rn = 0, rx = 0, ry = 0, rz = 0;  // region sizes (0 => use rank-domain size).
    idx_t gn = 0, gx = 0, gy = 0, gz = 0;  // block-group sizes (0 => calculate).
    idx_t bt = 1;                          // temporal block size.
    idx_t bn = 1, bx = DEF_BLOCK_SIZE, by = DEF_BLOCK_SIZE, bz = DEF_BLOCK_SIZE;  // size of cache blocks.
    idx_t pn = 0, px = DEF_PAD, py = DEF_PAD, pz = DEF_PAD; // padding.
    bool validate = false;
    int block_threads = DEF_BLOCK_THREADS; // number of threads for a block.
    int max_threads = 0;                   // max number of threads to use (0 => use OMP default).
    int thread_divisor = DEF_THREAD_DIVISOR; // divide max threads by this factor.
    bool doWarmup = true;
    idx_t copy_in = 0, copy_out = 0;     // how often to copy to shadow grids.
    int pre_trial_sleep_time = 1;   // sec to sleep before each trial.
    idx_t nrn = 1, nrx = 1, nry = 1, nrz = 1; // num ranks in each dim.
    idx_t rin = 0, rix = 0, riy = 0, riz = 0; // my rank's index in each dim.
    idx_t msg_rank = 0;                       // rank to print messages.
    bool findLoc = true;                      // whether to calculate my rank's index.
    bool help = false;                        // user asked for help.

    // Output stream for messages.
    static ostream* ostr = &cout;
    static ofstream nulostr;    // null stream (unopened ofstream).

    // Round up val to a multiple of mult.
    // Print a message if rounding is done.
    idx_t roundUp(idx_t val, idx_t mult, string name)
    {
        idx_t res = val;
        if (val % mult != 0) {
            res = ROUND_UP(res, mult);
            *ostr << "Adjusting " << name << " from " << val << " to " <<
                res << " to be a multiple of " << mult << endl;
        }
        return res;
    }
    
    // Fix bsize, if needed, to fit into rsize and be a multiple of mult.
    // Return number of blocks.
    idx_t findNumSubsets(idx_t& bsize, const string& bname,
                         idx_t rsize, const string& rname,
                         idx_t mult, string dim) {
        if (bsize < 1) bsize = rsize; // 0 => use full size.
        bsize = ROUND_UP(bsize, mult);
        if (bsize > rsize) bsize = rsize;
        idx_t nblks = (rsize + bsize - 1) / bsize;
        idx_t rem = rsize % bsize;
        idx_t nfull_blks = rem ? (nblks - 1) : nblks;

        *ostr << " In '" << dim << "' dimension, " << rname << " of size " <<
            rsize << " is divided into " << nfull_blks << " " << bname << "(s) of size " << bsize;
        if (rem)
            *ostr << " plus 1 remainder " << bname << " of size " << rem;
        *ostr << "." << endl;
        return nblks;
    }
    idx_t findNumBlocks(idx_t& bsize, idx_t rsize, idx_t mult, string dim) {
        return findNumSubsets(bsize, "block", rsize, "region", mult, dim);
    }
    idx_t findNumGroups(idx_t& ssize, idx_t rsize, idx_t mult, string dim) {
        return findNumSubsets(ssize, "group", rsize, "region", mult, dim);
    }
    idx_t findNumRegions(idx_t& rsize, idx_t dsize, idx_t mult, string dim) {
        return findNumSubsets(rsize, "region", dsize, "rank-domain", mult, dim);
    }

    // Find sum of rank_vals over all ranks.
    idx_t sumOverRanks(idx_t rank_val, MPI_Comm comm) {
        idx_t sum_val = rank_val;
#ifdef USE_MPI
        MPI_Allreduce(&rank_val, &sum_val, 1, MPI_INTEGER8, MPI_SUM, comm);
#endif
        return sum_val;
    }

    // Make sure rank_val is same over all ranks.
    void checkOverRanks(idx_t rank_val, MPI_Comm comm, const string& descr) {
        idx_t min_val = rank_val;
        idx_t max_val = rank_val;
#ifdef USE_MPI
        MPI_Allreduce(&rank_val, &min_val, 1, MPI_INTEGER8, MPI_MIN, comm);
        MPI_Allreduce(&rank_val, &max_val, 1, MPI_INTEGER8, MPI_MAX, comm);
#endif

        if (min_val != rank_val || max_val != rank_val) {
            cerr << "error: " << descr << " values range from " << min_val << " to " <<
                max_val << " across the ranks. They should all be identical." << endl;
            exit_yask(1);
        }
    }
    
    // Print usage.
    void usage(const char* pgmName)
    {
        *ostr << 
            "Usage: " << pgmName << " [options]\n"
            "Options:\n"
            " -h:              print this help and the current settings, then exit\n"
            " -v               validate by comparing to a scalar run (see notes below)\n" <<
            " -t <n>           number of trials, default=" <<
            num_trials << endl <<
            " -dt <n>          domain size for this rank in temporal dimension (number of time steps), default=" <<
            dt << endl <<
            " -d{n,x,y,z} <n>  domain size for this rank in specified spatial dimension, defaults=" <<
            dn << '*' << dx << '*' << dy << '*' << dz << endl <<
            " -d <n>           shorthand for '-dx <n> -dy <n> -dz <n>\n" <<
            " -rt <n>          OpenMP region time steps (for wave-front tiling), default=" <<
            rt << endl <<
            " -r{n,x,y,z} <n>  OpenMP region size in specified spatial dimension, defaults to rank-domain size\n" <<
            " -r <n>           shorthand for '-rx <n> -ry <n> -rz <n>\n" <<
            " -s{n,x,y,z} <n>  group size in specified spatial dimension, defaults set automatically\n" <<
            " -s <n>           shorthand for '-sx <n> -sy <n> -sz <n>\n" <<
            " -b{n,x,y,z} <n>  cache block size in specified spatial dimension, defaults=" <<
            bn << '*' << bx << '*' << by << '*' << bz << endl <<
            " -b <n>           shorthand for '-bx <n> -by <n> -bz <n>\n" <<
            " -g{n,x,y,z} <n>  block-group size in specified spatial dimension, defaults=" <<
            gn << '*' << gx << '*' << gy << '*' << gz << endl <<
            " -g <n>           shorthand for '-gx <n> -gy <n> -gz <n>\n" <<
            " -p{n,x,y,z} <n>  extra memory-padding in specified spatial dimension, defaults=" <<
            pn << '+' << px << '+' << py << '+' << pz << endl <<
            " -p <n>           shorthand for '-px <n> -py <n> -pz <n>\n" <<
#ifdef USE_MPI
            " -nr{n,x,y,z} <n> num ranks in specified dimension, defaults=" <<
            nrn << '*' << nrx << '*' << nry << '*' << nrz << endl <<
            " -nr <n>          shorthand for '-nrx <n> -nry <n> -nrz <n>\n" <<
            " -ri{n,x,y,z} <n> this rank's index in specified dimension, defaults set automatically\n" <<
            " -msg_rank <n>    rank that will print informational messages, default=" << msg_rank << endl <<
#endif
#ifdef ENABLE_SHADOW_COPY
            " -copy_in <n>     copy from traditional-layout grids every n time-steps, default=" <<
            copy_in << endl <<
            " -copy_out <n>    copy to traditional-layout grids every n time-steps, default=" <<
            copy_out << endl <<
#endif
            " -sleep <n>       seconds to sleep before each trial, default=" <<
            pre_trial_sleep_time << endl <<
            " -no_warmup       skip warmup iterations\n" <<
            " -max_threads <n>     set max threads to use to <n>, default=" << max_threads << endl <<
            " -thread_divisor <n>  divide max_threads by <n>, default=" <<
            thread_divisor << endl <<
            " -block_threads <n>   set number of threads to use for each block, default=" <<
            block_threads << endl <<
            "Notes:\n"
#ifndef USE_MPI
            " This binary has not been built with MPI support.\n"
#endif
            " Using '-max_threads 0' => max_threads = OpenMP default number of threads.\n"
            " For stencil evaluation, threads are allocated using nested OpenMP:\n"
            "  Num threads across blocks: max_threads / thread_divisor / block_threads.\n"
            "  Num threads per block: block_threads.\n"
            " A block size of 0 => block size == region size in that dimension.\n"
            " A group size of 0 => group size == block size in that dimension.\n"
            " A region size of 0 => region size == rank-domain size in that dimension.\n"
            " Control the time steps in each temporal wave-front with -rt:\n"
            "  Using '-rt 1' effectively disables wave-front tiling.\n"
            "  Any value other than 1 changes the region spatial-size defaults.\n"
            " Temporal cache blocking is not yet supported, so bt == 1.\n"
            " Validation is very slow and uses 2x memory,\n"
            "  so run with very small sizes and number of time-steps.\n"
            "  Using '-v' is shorthand for these settings: '-no_warmup -d 64 -dt 1 -t 1',\n"
            "  which may be overridden by options *after* '-v'.\n"
            "  If validation fails, it may be due to rounding error; try building with 8-byte reals.\n"
            " The 'n' dimension only applies to stencils that use that variable.\n"
            "Examples:\n" <<
            " " << pgmName << " -d 768 -dt 4\n" <<
            " " << pgmName << " -dx 512 -dy 256 -dz 128\n" <<
            " " << pgmName << " -d 2048 -dt 20 -r 512 -rt 10  # temporal tiling.\n" <<
            " " << pgmName << " -d 512 -npx 2 -npy 1 -npz 2   # multi-rank.\n" <<
            " " << pgmName << " -v                            # validation.\n" <<
            flush;
    }

    // Parse options.
    void parseOpts(int argc, char** argv)
    {
        // parse options.
        for (int argi = 1; argi < argc; argi++) {
            if ( argv[argi][0] == '-' && argv[argi][1] ) {
                string opt = argv[argi];

                // options w/o values.
                if (opt == "-h" || opt == "-help" || opt == "--help") {
                    help = true;
                }

                else if (opt == "-no_warmup")
                    doWarmup = false;

                else if (opt == "-v") {
                    validate = true;
                    doWarmup = false;
                    dx = dy = dz = 64;
                    dt = 1;
                    num_trials = 1;
                }

                // options w/int values.
                else {

                    if (argi >= argc) {
                        cerr << "error: no value for option '" << opt << "'." << endl;
                        exit_yask(1);
                    }
                    int val = atoi(argv[++argi]);
                    if (opt == "-t") num_trials = val;
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
                    else if (opt == "-gn") gn = val;
                    else if (opt == "-gx") gx = val;
                    else if (opt == "-gy") gy = val;
                    else if (opt == "-gz") gz = val;
                    else if (opt == "-g") gx = gy = gz = val;
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
#ifdef USE_MPI
                    else if (opt == "-nrn") nrn = val;
                    else if (opt == "-nrx") nrx = val;
                    else if (opt == "-nry") nry = val;
                    else if (opt == "-nrz") nrz = val;
                    else if (opt == "-nr") nrx = nry = nrz = val;
                    else if (opt == "-rin") { rin = val; findLoc = false; }
                    else if (opt == "-rix") { rix = val; findLoc = false; }
                    else if (opt == "-riy") { riy = val; findLoc = false; }
                    else if (opt == "-riz") { riz = val; findLoc = false; }
                    else if (opt == "-msg_rank") msg_rank = val;
#endif
                    else if (opt == "-block_threads") block_threads = val;
                    else if (opt == "-max_threads") max_threads = val;
                    else if (opt == "-thread_divisor") thread_divisor = val;
#ifdef ENABLE_SHADOW_COPY
                    else if (opt == "-copy_in") copy_in = val;
                    else if (opt == "-copy_out") copy_out = val;
#endif
                    else if (opt == "-sleep") pre_trial_sleep_time = val;
            
                    else {
                        cerr << "error: option '" << opt << "' not recognized." << endl;
                        exit_yask(1);
                    }
                }
            }
            else {
                cerr << "error: extraneous parameter '" <<
                    argv[argi] << "'." << endl;
                exit_yask(1);
            }
        }

    }
}

// Parse command-line args, run kernel, run validation if requested.
int main(int argc, char** argv)
{
    // Stop collecting VTune data.
    // Even better to use -start-paused option.
    VTUNE_PAUSE;

    // MPI init.
    int my_rank = 0;
    int num_ranks = 1;
#ifdef USE_MPI
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        cerr << "error: MPI_THREAD_SERIALIZED not provided.\n";
        exit_yask(1);
    }
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &num_ranks);
#else
    MPI_Comm comm = 0;
#endif
    bool is_leader = my_rank == 0;

    // OpenMP init.
#if defined(_OPENMP)
    int omp_num_procs = omp_get_num_procs();
#endif    

    // Make sure all init data is dumped before continuing.
    MPI_Barrier(comm);
    
    // Parse options.
    parseOpts(argc, argv);

    // Enable output?
    if (my_rank != msg_rank)
        ostr = &nulostr;

    // Splash banner.
    *ostr <<
        "┌──────────────────────────────────────────┐\n"
        "│  Y.A.S.K. ── Yet Another Stencil Kernel  │\n"
        "│            https://01.org/yask           │\n"
        "│    Intel Corporation, copyright 2016     │\n"
        "└──────────────────────────────────────────┘\n"
        "\nStencil name: " STENCIL_NAME << endl;
    *ostr << "Invocation:";
    for (int i = 0; i < argc; i++)
        *ostr << " " << argv[i];
    *ostr << endl;

    // Help msg.
    if (help) {
        usage(argv[0]);
        cerr << "\nExiting due to help option." << endl;
        exit_yask(1);
    }


#ifdef DEBUG
    *ostr << "*** WARNING: binary compiled with DEBUG; ignore performance results.\n";
#endif
#if defined(NO_INTRINSICS) && (VLEN > 1)
    *ostr << "*** WARNING: binary compiled with NO_INTRINSICS; ignore performance results.\n";
#endif
#ifdef MODEL_CACHE
    *ostr << "*** WARNING: binary compiled with MODEL_CACHE; ignore performance results.\n";
#endif
#ifdef TRACE_MEM
    *ostr << "*** WARNING: binary compiled with TRACE_MEM; ignore performance results.\n";
#endif
#ifdef TRACE_INTRINSICS
    *ostr << "*** WARNING: binary compiled with TRACE_INTRINSICS; ignore performance results.\n";
#endif

    // TODO: check all dims.
#ifndef USING_DIM_N
    if (dn > 1) {
        cerr << "error: dn = " << dn << ", but stencil '"
            STENCIL_NAME "' doesn't use dimension 'n'." << endl;
        exit_yask(1);
    }
#endif

    // Round up domain size as needed.
    dt = roundUp(dt, CPTS_T, "rank domain size in t (time steps)");
    dn = roundUp(dn, CPTS_N, "rank domain size in n");
    dx = roundUp(dx, CPTS_X, "rank domain size in x");
    dy = roundUp(dy, CPTS_Y, "rank domain size in y");
    dz = roundUp(dz, CPTS_Z, "rank domain size in z");

    // Report ranks.
#ifdef USE_MPI
    *ostr << "Num MPI ranks: " << num_ranks << endl;
    *ostr << "This MPI rank: " << my_rank << endl;
#else
    *ostr << "MPI not enabled." << endl;
#endif

    // Check ranks.
    idx_t req_ranks = nrn * nrx * nry * nrz;
    if (req_ranks != num_ranks) {
        cerr << "error: " << req_ranks << " rank(s) requested, but MPI reports " <<
            num_ranks << " rank(s) are active." << endl;
        exit_yask(1);
    }
    checkOverRanks(dt, comm, "time-step");
            
    // Context for evaluating results.
    STENCIL_CONTEXT context;
    context.ostr = ostr;
    context.num_ranks = num_ranks;
    context.my_rank = my_rank;
    context.comm = comm;
#ifdef ENABLE_SHADOW_COPY
    context.shadow_in_freq = copy_in;
    context.shadow_out_freq = copy_out;
#endif

    // report threads.
    int region_threads = 1;
    {
#if defined(_OPENMP)
        *ostr << "Num OpenMP procs: " << omp_num_procs << endl;
        if (max_threads == 0)
            max_threads = omp_get_max_threads();
        context.orig_max_threads = max_threads / thread_divisor;
        *ostr << "Num OpenMP threads: " << context.orig_max_threads << endl;

#if USE_CREW
        // Init Crew.
        *ostr << "Creating crews..." << endl;
        kmp_crew_create();
        int numThreads = omp_get_max_threads();
        *ostr << "Num OpenMP threads after crew creation: " << numThreads << endl;
        int crewSize = kmp_crew_get_max_size();
        int numWorkers = numThreads * crewSize;
        *ostr << "Total num crews: " << numWorkers << endl <<
            "  Num crew-leader threads: " << numThreads << endl <<
            "  Num threads per crew: " << crewSize << endl;
        if (numWorkers == context.orig_max_threads)
            *ostr << "Note: sanity check passed: num crews == num OpenMP threads before creating crews." << endl;
        else {
            cerr << "Error: sanity check failed: num crews != num OpenMP threads before creating crews.\n"
                "This usually indicates your OpenMP library has a crew-initialization issue.\n"
                "Please update your OpenMP library or rebuild with crew disabled (make crew=0 ...).\n";
            exit_yask(1);
        }
#else

        // Enable nesting and report nesting threads.
        assert(block_threads > 0);
        if (block_threads > 1)
            omp_set_nested(1);
        context.num_block_threads = block_threads;
        int rt = context.set_region_threads(); // Temporary; just for reporting.
        region_threads = omp_get_max_threads();
        *ostr << "  Num threads per region: " << region_threads << endl;
        *ostr << "  Num threads per block: " << block_threads << endl;
        context.set_max_threads(); // Back to normal.
#endif
#else
        int numThreads = 1;
        *ostr << "Num threads: " << numThreads << endl;
#endif
    }

    // Adjust defaults for wavefronts.
    if (rt > 1) {
        if (!rn) rn = 1;
        if (!rx) rx = dx / 2;
        if (!ry) ry = dy / 2;
        if (!rz) rz = dz / 2;

        // TODO: enable this.
        if (num_ranks > 1) {
            cerr << "MPI communication is not currently enabled with wave-front tiling." << endl;
            exit_yask(1);
        }
    }

    // Determine num regions.
    // Also fix up region sizes as needed.
    *ostr << "\nRegions:" << endl;
    idx_t nrgt = findNumRegions(rt, dt, CPTS_T, "t");
    idx_t nrgn = findNumRegions(rn, dn, CPTS_N, "n");
    idx_t nrgx = findNumRegions(rx, dx, CPTS_X, "x");
    idx_t nrgy = findNumRegions(ry, dy, CPTS_Y, "y");
    idx_t nrgz = findNumRegions(rz, dz, CPTS_Z, "z");
    idx_t nrg = nrgt * nrgn * nrgx * nrgy * nrgz;
    *ostr << " num-regions-per-rank: " << nrg << endl;

    // Determine num blocks.
    // Also fix up block sizes as needed.
    *ostr << "\nBlocks:" << endl;
    idx_t nbt = findNumBlocks(bt, rt, CPTS_T, "t");
    idx_t nbn = findNumBlocks(bn, rn, CPTS_N, "n");
    idx_t nbx = findNumBlocks(bx, rx, CPTS_X, "x");
    idx_t nby = findNumBlocks(by, ry, CPTS_Y, "y");
    idx_t nbz = findNumBlocks(bz, rz, CPTS_Z, "z");
    idx_t nb = nbt * nbn * nbx * nby * nbz;
    *ostr << " num-blocks-per-region: " << nb << endl;

    // Adjust defaults for block-groups.
    // Assumes inner block loop in z dimension and
    // layout in n,x,y,z order.
    // TODO: check and adjust this accordingly.
    if (!gn) gn = bn;
    if (!gx) gx = bx;
    if (!gy) gy = by;
    if (!gz) gz = bz;

    // Determine num groups.
    // Also fix up group sizes as needed.
    *ostr << "\nBlock-groups:" << endl;
    idx_t ngn = findNumGroups(gn, rn, bn, "n");
    idx_t ngx = findNumGroups(gx, rx, bx, "x");
    idx_t ngy = findNumGroups(gy, ry, by, "y");
    idx_t ngz = findNumGroups(gz, rz, bz, "z");
    idx_t ng = ngn * ngx * ngy * ngz;
    *ostr << " num-block-groups-per-region: " << ng << endl;

    // Round up padding as needed.
    pn = roundUp(pn, VLEN_N, "extra padding in n");
    px = roundUp(px, VLEN_X, "extra padding in x");
    py = roundUp(py, VLEN_Y, "extra padding in y");
    pz = roundUp(pz, VLEN_Z, "extra padding in z");

    // Round up halos as needed.
    // TODO: get rid of this when grid-specific halos
    // are used throughout.
#ifdef USING_DIM_N
    idx_t hn = ROUND_UP(context.max_halo_n, VLEN_N);
#else
    idx_t hn = 0;
#endif
    idx_t hx = ROUND_UP(context.max_halo_x, VLEN_X);
    idx_t hy = ROUND_UP(context.max_halo_y, VLEN_Y);
    idx_t hz = ROUND_UP(context.max_halo_z, VLEN_Z);
    
    *ostr << "\nSizes in points per grid (t*n*x*y*z):\n"
        " vector-size:      " << VLEN_T << '*' << VLEN_N << '*' << VLEN_X << '*' << VLEN_Y << '*' << VLEN_Z << endl <<
        " cluster-size:     " << CPTS_T << '*' << CPTS_N << '*' << CPTS_X << '*' << CPTS_Y << '*' << CPTS_Z << endl <<
        " block-size:       " << bt << '*' << bn << '*' << bx << '*' << by << '*' << bz << endl <<
        " block-group-size: 1*" << gn << '*' << gx << '*' << gy << '*' << gz << endl <<
        " region-size:      " << rt << '*' << rn << '*' << rx << '*' << ry << '*' << rz << endl <<
        " rank-domain-size: " << dt << '*' << dn << '*' << dx << '*' << dy << '*' << dz << endl <<
        endl <<
        "Other settings:\n"
        " num-ranks: " << nrn << '*' << nrx << '*' << nry << '*' << nrz << endl <<
        " stencil-name: " STENCIL_NAME << endl << 
        " time-dim-size: " << TIME_DIM_SIZE << endl <<
        " vector-len: " << VLEN << endl <<
        " padding: " << pn << '+' << px << '+' << py << '+' << pz << endl <<
        " max-halos: " << hn << '+' << hx << '+' << hy << '+' << hz << endl <<
#ifdef ENABLE_SHADOW_COPY
        " shadow-copy-in-frequency: " << copy_in << endl <<
        " shadow-copy-out-frequency: " << copy_out << endl <<
#endif
        " manual-L1-prefetch-distance: " << PFDL1 << endl <<
        " manual-L2-prefetch-distance: " << PFDL2 << endl <<
        endl;

    // Save sizes in context struct.
    context.dt = dt;
    context.dn = dn;
    context.dx = dx;
    context.dy = dy;
    context.dz = dz;
    
    context.rt = rt;
    context.rn = rn;
    context.rx = rx;
    context.ry = ry;
    context.rz = rz;

    context.gn = gn;
    context.gx = gx;
    context.gy = gy;
    context.gz = gz;

    context.bt = bt;
    context.bn = bn;
    context.bx = bx;
    context.by = by;
    context.bz = bz;

    context.pn = pn;
    context.px = px;
    context.py = py;
    context.pz = pz;

    context.hn = hn;
    context.hx = hx;
    context.hy = hy;
    context.hz = hz;

    context.nrn = nrn;
    context.nrx = nrx;
    context.nry = nry;
    context.nrz = nrz;

    context.rin = rin;
    context.rix = rix;
    context.riy = riy;
    context.riz = riz;

    // Alloc memory, create lists of grids, etc.
    // NB: this contains MPI exchanges of rank indices.
    idx_t rank_numpts_1t, rank_numFpOps_1t; // sums across eqs for this rank.
    context.allocAll(findLoc, &rank_numpts_1t, &rank_numFpOps_1t);

    // Report total allocation.
    idx_t rank_nbytes = context.get_num_bytes();
    *ostr << "Total allocation in this rank (bytes): " <<
        printWithPow2Multiplier(rank_nbytes) << endl;
    idx_t tot_nbytes = sumOverRanks(rank_nbytes, comm);
    *ostr << "Total overall allocation in " << num_ranks << " rank(s) (bytes): " <<
        printWithPow2Multiplier(tot_nbytes) << endl;
    
    // Various metrics for amount of work.
    // 'rank_' prefix indicates for this rank.
    // 'tot_' prefix indicates over all ranks.
    // 'numpts' indicates points actually calculated in sub-domains.
    // 'domain' indicates points in domain-size specified on cmd-line.
    // 'numFpOps' indicates est. number of FP ops.
    // '_1t' suffix indicates one time-step.
    // '_dt' suffix indicates all time-steps.

    idx_t rank_numpts_dt = rank_numpts_1t * dt;
    idx_t tot_numpts_1t = sumOverRanks(rank_numpts_1t, comm);
    idx_t tot_numpts_dt = tot_numpts_1t * dt;

    idx_t rank_numFpOps_dt = rank_numFpOps_1t * dt;
    idx_t tot_numFpOps_1t = sumOverRanks(rank_numFpOps_1t, comm);
    idx_t tot_numFpOps_dt = tot_numFpOps_1t * dt;

    idx_t rank_domain_1t = dn * dx * dy * dz;
    idx_t rank_domain_dt = rank_domain_1t * dt;
    idx_t tot_domain_1t = sumOverRanks(rank_domain_1t, comm);
    idx_t tot_domain_dt = tot_domain_1t * dt;
    
    // Print some more stats.
    *ostr << endl <<
        "Amount-of-work stats:\n" <<
        " problem-size in this rank, for one time-step: " <<
        printWithPow10Multiplier(rank_domain_1t) << endl <<
        " problem-size in all ranks, for one time-step: " <<
        printWithPow10Multiplier(tot_domain_1t) << endl <<
        " problem-size in this rank, for all time-steps: " <<
        printWithPow10Multiplier(rank_domain_dt) << endl <<
        " problem-size in all ranks, for all time-steps: " <<
        printWithPow10Multiplier(tot_domain_dt) << endl <<
        endl <<
        " grid-points-updated in this rank, for one time-step: " <<
        printWithPow10Multiplier(rank_numpts_1t) << endl <<
        " grid-points-updated in all ranks, for one time-step: " <<
        printWithPow10Multiplier(tot_numpts_1t) << endl <<
        " grid-points-updated in this rank, for all time-steps: " <<
        printWithPow10Multiplier(rank_numpts_dt) << endl <<
        " grid-points-updated in all ranks, for all time-steps: " <<
        printWithPow10Multiplier(tot_numpts_dt) << endl <<
        endl <<
        " est-FP-ops in this rank, for one time-step: " <<
        printWithPow10Multiplier(rank_numFpOps_1t) << endl <<
        " est-FP-ops in all ranks, for one time-step: " <<
        printWithPow10Multiplier(tot_numFpOps_1t) << endl <<
        " est-FP-ops in this rank, for all time-steps: " <<
        printWithPow10Multiplier(rank_numFpOps_dt) << endl <<
        " est-FP-ops in all ranks, for all time-steps: " <<
        printWithPow10Multiplier(tot_numFpOps_dt) << endl <<
        endl << 
        "Notes:\n" <<
        " problem-size is based on rank-domain sizes specified in command-line (dn * dx * dy * dz).\n" <<
        " grid-points-updated is based sum of grid-updates-in-sub-domain across equation-group(s).\n" <<
        " est-FP-ops is based on sum of est-FP-ops-in-sub-domain across equation-group(s).\n" <<
        endl;

    // Exit if nothing to do.
    if (num_trials < 1) {
        cerr << "Exiting because no trials are specified." << endl;
        exit_yask(1);
    }
    if (tot_numpts_dt < 1) {
        cerr << "Exiting because there are zero points to evaluate." << endl;
        exit_yask(1);
    }
    *ostr << flush;

    // warmup caches, threading, etc.
    if (doWarmup) {
        *ostr << endl;

        // Temporarily set dt to a temp value for warmup.
        idx_t tmp_dt = min<idx_t>(dt, TIME_DIM_SIZE);
        context.dt = tmp_dt;

#ifdef MODEL_CACHE
        if (!is_leader)
            cache_model.disable();
        if (cache_model.isEnabled())
            *ostr << "Modeling cache...\n";
#endif
        *ostr << "Warmup of " << context.dt << " time step(s)...\n" << flush;
        context.calc_rank_opt();

#ifdef MODEL_CACHE
        // print cache stats, then disable.
        if (cache_model.isEnabled()) {
            *ostr << "Done modeling cache...\n";
            cache_model.dumpStats();
            cache_model.disable();
        }
#endif

        // Replace temp setting with correct value.
        context.dt = dt;
        *ostr << flush;
        MPI_Barrier(comm);
    }

    // variables for measuring performance.
    double wstart, wstop;
    float best_elapsed_time=0.0f, best_apps=0.0f, best_dpps=0.0f, best_flops=0.0f;

    // Performance runs.
    string divLine = "───────────────────────────────────────────────────────";
    *ostr << "\nRunning " << num_trials << " performance trial(s) of " <<
        context.dt << " time step(s) each...\n" << flush;
    for (idx_t tr = 0; tr < num_trials; tr++) {

        // init data for comparison if validating.
        if (validate)
            context.initDiff();

        // Stabilize.
        if (pre_trial_sleep_time)
            sleep(pre_trial_sleep_time);
        MPI_Barrier(comm);

        // Start timing.
        VTUNE_RESUME;
        context.mpi_time = 0.0;
#ifdef ENABLE_SHADOW_COPY
        context.shadow_time = 0.0;
#endif
        wstart = getTimeInSecs();

        // Actual work (must wait until all ranks are done).
        context.calc_rank_opt();
        MPI_Barrier(comm);

        // Stop timing.
        wstop =  getTimeInSecs();
        VTUNE_PAUSE;
            
        // calc and report perf.
        float elapsed_time = (float)(wstop - wstart);
        float apps = float(tot_numpts_dt)/elapsed_time;
        float dpps = float(tot_domain_dt)/elapsed_time;
        float flops = float(tot_numFpOps_dt)/elapsed_time;
        *ostr << divLine << endl <<
            "time (sec):                             " << printWithPow10Multiplier(elapsed_time) << endl <<
            "throughput (prob-size-points/sec):      " << printWithPow10Multiplier(dpps) << endl <<
            "throughput (points-updated/sec):        " << printWithPow10Multiplier(apps) << endl <<
            "throughput (est-FLOPS):                 " << printWithPow10Multiplier(flops) << endl;
#ifdef USE_MPI
        *ostr <<
            "time in halo exch (sec):                " << printWithPow10Multiplier(context.mpi_time) << endl;
#endif
#ifdef ENABLE_SHADOW_COPY
        if (copy_in || copy_out)
            *ostr <<
                "time in shadow copy (sec):              " << printWithPow10Multiplier(context.shadow_time) << endl;
#endif

        if (apps > best_apps) {
            best_apps = apps;
            best_dpps = dpps;
            best_elapsed_time = elapsed_time;
            best_flops = flops;
        }
    }

    *ostr << divLine << endl <<
        "best-time (sec):                        " << printWithPow10Multiplier(best_elapsed_time) << endl <<
        "best-throughput (prob-size-points/sec): " << printWithPow10Multiplier(best_dpps) << endl <<
        "best-throughput (points-updated/sec):   " << printWithPow10Multiplier(best_apps) << endl <<
        "best-throughput (est-FLOPS):            " << printWithPow10Multiplier(best_flops) << endl <<
        divLine << endl <<
        "Notes:\n" <<
        " prob-size-points/sec is based on problem-size as described above.\n" <<
        " points-updated/sec is based on grid-points-updated as described above.\n" <<
        " est-FLOPS is based on est-FP-ops as described above.\n" <<
        endl;
    
    if (validate) {
        MPI_Barrier(comm);
        *ostr << "Running validation trial...\n";

        // Make a reference context for comparisons w/new grids:
        // Copy the settings from context, then re-alloc grids.
        STENCIL_CONTEXT ref_context = context;
        ref_context.name += "-reference";
        ref_context.allocAll(false); // do not need to re-calc locations.

        // init to same value used in context.
        ref_context.initDiff();

#if CHECK_INIT
        {
            context.initDiff();
            idx_t errs = context.compare(ref_context);
            if( errs == 0 ) {
                *ostr << "INIT CHECK PASSED." << endl;
                exit_yask(0);
            } else {
                cerr << "INIT CHECK FAILED: " << errs << " mismatch(es)." << endl;
                exit_yask(1);
            }
        }
#endif

        // Ref trial.
        ref_context.calc_rank_ref();

        // check for equality.
#ifdef USE_MPI
        MPI_Barrier(comm);
#endif
        *ostr << "Checking results on rank " << my_rank << "..." << endl;
        idx_t errs = context.compare(ref_context);
        if( errs == 0 ) {
            *ostr << "TEST PASSED." << endl;
        } else {
            cerr << "TEST FAILED: " << errs << " mismatch(es)." << endl;
            if (REAL_BYTES < 8)
                cerr << "This is not uncommon for low-precision FP; try with 8-byte reals." << endl;
            exit_yask(1);
        }
    }
    else
        *ostr << "\nRESULTS NOT VERIFIED.\n";

#ifdef USE_MPI
    MPI_Barrier(comm);
    MPI_Finalize();
#endif
    *ostr << "YASK DONE." << endl;
    
    return 0;
}
