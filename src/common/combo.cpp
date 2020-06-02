/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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

    // Compute binomial coefficient, i.e., the number of combos of n things taken k at a time
    // w/o repetition.
    // Returns n!/(k!*(n-k)!).
    // https://en.wikipedia.org/wiki/Combination.
    int n_choose_k(int n, int k) {
        assert(n >= 0);
        assert(k >= 0);

        // Pick none or all.
        if (k <= 0 || k == n)
            return 1;

        // Special case for k > n.
        if (k > n)
            return 0;

        // Symmetry around center because
        // n_choose_k == n_choose_n-k when k > n/2.
        if (k * 2 > n)
            k = n - k;

        // nc = (n * (n-1) * (n-2) *...* (n-k+1)) / (1 * 2 * 3 *...* k).
        //    = n * (n-2+1) / 2 * (n-3+1) / 3 *...* (n-k+1) / k.
        // See "Another alternative computation..." on wikipedia page and
        // comment that "this can be computed using only integer arithmetic."
        int nc = n;
        for( int i = 2; i <= k; i++ ) {
            nc *= (n - i + 1);
            nc /= i;
        }
        return nc;
    }

    // Get the 'r'th set of 'k' elements from set of integers between '0' and 'n-1'.
    // Returns vector of 'k' values.
    // 'r' must be between '0' and 'n_choose_k(n, k)-1'.
    vector<int> n_choose_k_set(int n, int k, int r) {
        assert(n >= 0);
        assert(k >= 0);
        assert(r >= 0);
        assert(r < n_choose_k(n, k));

        vector<int> c;

        // Empty set.
        if (n <= 0 || k <= 0)
            return c;

        // Pick one item.
        if (k == 1) {
            c.push_back(r);
            return c;
        }

        // Pick k items.
        int j = 0;
        for (int i = 0; i < k-1; i++) {
            c.push_back((i == 0) ? -1 : c[i-1]);
            while (true) {
                c[i]++;
                int nc = n_choose_k(n - (c[i]+1), k - (i+1));
                if (j + nc >= r + 1)
                    break;
                j += nc;
            }
        }
        c.push_back(c[k-2] + r - j + 1);
        return c;
    }

}
