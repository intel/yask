/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

#include "combo.hpp"
#include <iostream>

using namespace std;

namespace yask {

    // https://stackoverflow.com/questions/9330915/number-of-combinations-n-choose-r-in-c
    int choose(int n, int k) {
        if (k > n)
            return 0;
        if (k * 2 > n)
            k = n-k;
        if (k == 0)
            return 1;

        int result = n;
        for( int i = 2; i <= k; ++i ) {
            result *= (n-i+1);
            result /= i;
        }
        return result;    
    }

    // https://stackoverflow.com/questions/561/how-to-use-combinations-of-sets-as-test-data#794
    // Fixed to handle p==0 and p==1.

    // Get the 'x'th lexicographically ordered set of 'p' elements in 'n'.
    // Returns 'p' values in 'c'.
    // 'x' and values in 'c' are 1-based.
    void combination(int* c, int n, int p, int x) {
        assert(n > 0);
        assert(p >= 0);
        assert(p <= n);
        assert(x >= 1);
        assert(x <= choose(n, p));
        if (p == 0)
            return;
        if (p == 1) {
            c[0] = x;
            return;
        }
        int r, k = 0;
        for(int i=0; i<p-1; i++) {
            c[i] = (i != 0) ? c[i-1] : 0;
            do {
                c[i]++;
                r = choose(n-c[i],p-(i+1));
                k += r;
            } while(k < x);
            k -= r;
        }
        c[p-1] = c[p-2] + x - k;
    }

    void test_combo() {
        int n = 5;
        for (int p = 0; p <= n; p++) {
            int nx = choose(n,p);
            cout << "choose(" << n << ", " << p << ") = " << nx << endl;
            int c[p];
            for (int x = 1; x <= nx; x++) {
                cout << "combo(" << c << ", "  << n << ", "  << p << ", "  << x << ") = ";
                combination(c, n, p, x);
                for (int i = 0; i < p; i++)
                    cout << " " << c[i];
                cout << endl;
            }
        }
    }

}
