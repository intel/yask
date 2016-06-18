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
#include <sstream>
#include "stencil.hpp"

using namespace std;
namespace yask {

#if defined(_OPENMP)
    double getTimeInSecs() {
        return omp_get_wtime();
    }

#elif defined(WIN32)

    // FIXME.
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
    // Print a message if rounding is done.
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

    // Return num with SI multiplier.
    // Use this one for bytes, etc.
    string printWithPow2Multiplier(double num)
    {
        ostringstream os;
        const double oneK = 1024.;
        const double oneM = oneK * oneK;
        const double oneG = oneK * oneM;
        if (num > oneG)
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
        if (num > oneG)
            os << (num / oneG) << "G";
        else if (num > oneM)
            os << (num / oneM) << "M";
        else if (num > oneK)
            os << (num / oneK) << "K";
        else
            os << num;
        return os.str();
    }

}
