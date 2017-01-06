/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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

#include <stddef.h>
#include <time.h>
#include <math.h>
#include <sstream>
#include "stencil.hpp"

// Set MODEL_CACHE to 1 or 2 to model that cache level
// and create a global cache object here.
#ifdef MODEL_CACHE
Cache cache_model(MODEL_CACHE);
#endif

using namespace std;
namespace yask {

    double getTimeInSecs() {

#if defined(_OPENMP)
        return omp_get_wtime();

#elif defined(WIN32)
        return 1e-3 * (double)GetTickCount64();

#else
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);

        return ((double)(ts.tv_sec) + 
                1e-9 * (double)(ts.tv_nsec));
#endif
    }

    // Return num with SI multiplier.
    // Use this one for bytes, etc.
    string printWithPow2Multiplier(double num)
    {
        ostringstream os;
        const double oneK = 1024.;
        const double oneM = oneK * oneK;
        const double oneG = oneK * oneM;
        const double oneT = oneK * oneG;
        if (num > oneT)
            os << (num / oneT) << "Ti";
        else if (num > oneG)
            os << (num / oneG) << "Gi";
        else if (num > oneM)
            os << (num / oneM) << "Mi";
        else if (num > oneK)
            os << (num / oneK) << "Ki";
        else
            os << num;
        return os.str();
    }

    // Return num with SI multiplier.
    // Use this one for rates, etc.
    string printWithPow10Multiplier(double num)
    {
        ostringstream os;
        const double oneK = 1e3;
        const double oneM = 1e6;
        const double oneG = 1e9;
        const double oneT = 1e12;
        if (num > oneT)
            os << (num / oneT) << "T";
        else if (num > oneG)
            os << (num / oneG) << "G";
        else if (num > oneM)
            os << (num / oneM) << "M";
        else if (num > oneK)
            os << (num / oneK) << "K";
        else
            os << num;
        return os.str();
    }

    // Round up val to a multiple of mult.
    // Print a message if rounding is done.
    idx_t roundUp(ostream& os, idx_t val, idx_t mult, const string& name)
    {
        assert(mult > 0);
        idx_t res = val;
        if (val % mult != 0) {
            res = ROUND_UP(res, mult);
            os << "Adjusting " << name << " from " << val << " to " <<
                res << " to be a multiple of " << mult << endl;
        }
        return res;
    }
    
    // Fix bsize, if needed, to fit into rsize and be a multiple of mult.
    // Return number of blocks.
    idx_t findNumSubsets(ostream& os,
                         idx_t& bsize, const string& bname,
                         idx_t rsize, const string& rname,
                         idx_t mult, const string& dim) {
        if (bsize < 1) bsize = rsize; // 0 => use full size.
        bsize = ROUND_UP(bsize, mult);
        if (bsize > rsize) bsize = rsize;
        idx_t nblks = (rsize + bsize - 1) / bsize;
        idx_t rem = rsize % bsize;
        idx_t nfull_blks = rem ? (nblks - 1) : nblks;

        os << " In '" << dim << "' dimension, " << rname << " of size " <<
            rsize << " is divided into " << nfull_blks << " " << bname << "(s) of size " << bsize;
        if (rem)
            os << " plus 1 remainder " << bname << " of size " << rem;
        os << "." << endl;
        return nblks;
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
    void assertEqualityOverRanks(idx_t rank_val, MPI_Comm comm, const string& descr) {
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

}
