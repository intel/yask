/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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

using namespace std;
using namespace yask;

int main(int argc, char** argv) {
    int n = 5;

    int exnc[n+2] = { 1, 5, 10, 10, 5, 1, 0 };
    
    for (int k = 0; k <= n+1; k++) {
        int nc = n_choose_k(n,k);
        cout << "choose(" << n << ", " << k << ") = " << nc << endl;
        assert(nc == exnc[k]);
        
        vector<vector<int>> cvv;
        for (int r = 0; r < nc; r++) {
            auto cv = n_choose_k_set(n, k, r);
            cout << " combo #" << r << " = ";
            assert(cv.size() == (size_t)k);
            for (int i = 0; i < k; i++) {
                cout << " " << cv[i];
                assert(cv[i] >= 0);
                assert(cv[i] < n);

                // Make sure this element is unique and in order.
                if (i > 0)
                    assert(cv[i] > cv[i-1]);
            }
            cout << endl;

            // Make sure this set is unique.
            for (size_t i = 0; i < cvv.size(); i++) {
                auto& cvi = cvv[i];
                assert(cv != cvi);
            }
            cvv.push_back(cv);
        }
    }
    return 0;
}
