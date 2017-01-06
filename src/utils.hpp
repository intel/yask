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

#ifndef YASK_UTILS
#define YASK_UTILS

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdexcept>

#include <string>
#include <iostream>

// rounding macros for integer types.
#define CEIL_DIV(numer, denom) (((numer) + (denom) - 1) / (denom))
#define ROUND_UP(n, mult) (CEIL_DIV(n, mult) * (mult))

namespace yask {

    // Some utility functions.
    extern double getTimeInSecs();
    extern std::string printWithPow2Multiplier(double num);
    extern std::string printWithPow10Multiplier(double num);

    // Find sum of rank_vals over all ranks.
    extern idx_t sumOverRanks(idx_t rank_val, MPI_Comm comm);

    // Make sure rank_val is same over all ranks.
    void assertEqualityOverRanks(idx_t rank_val, MPI_Comm comm,
                                 const std::string& descr);
    
    // Round up val to a multiple of mult.
    // Print a message if rounding is done.
    extern idx_t roundUp(std::ostream& os,
                         idx_t val, idx_t mult, const std::string& name);

    // Fix bsize, if needed, to fit into rsize and be a multiple of mult.
    // Return number of blocks.
    extern idx_t findNumSubsets(std::ostream& os,
                                idx_t& bsize, const std::string& bname,
                                idx_t rsize, const std::string& rname,
                                idx_t mult, const std::string& dim);
    inline idx_t findNumBlocks(std::ostream& os, idx_t& bsize, idx_t rsize,
                               idx_t mult, const std::string& dim) {
        return findNumSubsets(os, bsize, "block", rsize, "region", mult, dim);
    }
    inline idx_t findNumGroups(std::ostream& os, idx_t& ssize, idx_t rsize,
                               idx_t mult, const std::string& dim) {
        return findNumSubsets(os, ssize, "group", rsize, "region", mult, dim);
    }
    inline idx_t findNumRegions(std::ostream& os, idx_t& rsize, idx_t dsize,
                                idx_t mult, const std::string& dim) {
        return findNumSubsets(os, rsize, "region", dsize, "rank-domain", mult, dim);
    }
}

#endif
