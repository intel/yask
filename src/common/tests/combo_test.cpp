/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2024, Intel Corporation

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

// Test the combinatorial functions.

// enable assert().
#define CHECK

#include "combo.hpp"
#include <iostream>
#include <vector>

using namespace std;
using namespace yask;

int main(int argc, char** argv) {
    constexpr int n = 5;
    int exnc[n+2] = { 1, 5, 10, 10, 5, 1, 0 }; // expected num combos.
    
    for (int k = 0; k <= n+1; k++) {

        // Num combos.
        int nc = n_choose_k(n, k);
        cout << "choose(" << n << ", " << k << ") = " << nc << endl;
        assert(nc == exnc[k]);

        // Each combo.
        vector<size_t> cv;
        for (int r = 0; r < nc; r++) {
            auto cmask = n_choose_k_set(n, k, r);
            cout << " combo #" << r << " =";
            int nset = 0;
            for (int i = 0; i < n; i++) {
                if (is_bit_set(cmask, i)) {
                    cout << " " << i;
                    nset++;
                }
            }
            cout << " (0x" << hex << cmask << dec << ")\n";
            assert(nset == k);

            // Make sure this set is unique.
            for (size_t i = 0; i < cv.size(); i++) {
                auto& cvi = cv[i];
                assert(cmask != cvi);
            }
            cv.push_back(cmask);
        }
    }
    return 0;
}
