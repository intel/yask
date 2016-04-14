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

#include <stddef.h>
#include <time.h>
#include <math.h>
#include "stencil.hpp"

#ifdef WIN32
double getTimeInSecs() { return 0.0; }

#else
double getTimeInSecs()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    return ((double)(ts.tv_sec) + 
            1e-09 * (double)(ts.tv_nsec));
}
#endif

// Round up val to a multiple of mult.
// Print a message is rounding is done.
idx_t roundUp(idx_t val, idx_t mult, string name)
{
    idx_t res = val;
    if (val % mult != 0) {
        res = ROUND_UP(res, mult);
        cout << "Adjusting " << name << " from " << val << " to " <<
            res << " to be a multiple of " << mult << endl;
    }
    return res;
}


