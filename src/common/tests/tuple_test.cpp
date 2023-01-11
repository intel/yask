/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

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

// Test the YASK tuples.

// enable assert().
#define CHECK

#include "tuple.hpp"
using namespace std;
using namespace yask;

void ttest(bool first_inner) {

    ostream& os = cout;

    IntTuple t1;
    t1.set_first_inner(first_inner);
    t1.add_dim_back("x", 3);
    t1.add_dim_back("y", 4);
    assert(t1.get_num_dims() == 2);
    assert(t1[0] == 3);
    assert(t1[1] == 4);
    assert(t1["x"] == 3);
    assert(t1["y"] == 4);

    os << "space: " << t1.make_dim_val_str() << ", is ";
    if (!t1.is_first_inner()) os << "NOT ";
    os << "first-inner layout.\n";

    IntTuple t2(t1);
    assert(t2 == t1);
    t2["x"] = 2;
    assert(t2 < t1);
    t2["x"] = 4;
    assert(t2 > t1);

    IntTuple t3(t1);
    assert(t3 == t1);
    t3.add_dim_front("a", 1);
    assert(t3 > t1);            // more dims.

    IntTuple t4;
    t4.add_dim_back("x", 3);
    t4.add_dim_back("z", 4);
    assert(t4 > t1);

    os << "loop test...\n";
    size_t j = 0;
    for (int y = 0; y < t1["y"]; y++) {
        for (int x = 0; x < t1["x"]; x++, j++) {

            IntTuple ofs;
            ofs.add_dim_back("x", x);
            ofs.add_dim_back("y", y);

            auto i = t1.layout(ofs);
            os << " offset at " << ofs.make_dim_val_str() << " = " << i << endl;

            IntTuple ofs2 = t1.unlayout(i);
            assert(ofs == ofs2);

            if (first_inner)
                assert(i == j);
        }
    }

    // Test w/increasing number of dims.
    for (int d = 0; d <= 3; d++) {

        IntTuple t2;
        if (d > 0) t2.add_dim_back("x", 3);
        if (d > 1) t2.add_dim_back("y", 4);
        if (d > 2) t2.add_dim_back("z", 5);
        auto n = t2.product();

        os << d << "-d sequential visit test...\n";
        j = 0;
        size_t sumk = 0;
        t2.visit_all_points
            ([&](const IntTuple& ofs, size_t k) {

                 os << " offset at " << ofs.make_dim_val_str() << flush;
                 for (int d1 = 0; d1 < d; d1++) {
                     assert(ofs[d1] >= 0);
                     assert(ofs[d1] < t2[d1]);
                 }
                 auto i = t2.layout(ofs);
                 os << " = " << i << endl;
                 assert(int(k) < n);
                 assert(i == j);
                 assert(i == k);
                 j++;
                 sumk += k;
                 return true; // continue.
             });
        assert(int(j) == n);
        assert(int(sumk) == n * (n-1) / 2);

        // Test w/different num threads.
        for (int t0 : {1, 2, 3}) {
            for (int t1 : {1, 2}) {
        
                os << d << "-d parallel visit test...\n";
                omp_set_max_active_levels(2);
                yask_num_threads[0] = t0;
                yask_num_threads[1] = t1;
                os << "using " << t0 << " * " << t1 << " thread(s)\n";
                assert(t0 * t1 == yask_get_num_threads());
                j = 0;
                sumk = 0;
                t2.visit_all_points_in_parallel
                    ([&](const IntTuple& ofs, size_t k) {

                         assert(int(k) < n);
                         auto i = t2.layout(ofs);
                         #pragma omp critical
                         {
                             os << " offset at " << ofs.make_dim_val_str() << " = " << i << endl;
                             j++;
                             sumk += k;
                         }
                         assert(i == k);
                         return true;
                     });
                assert(int(j) == n);
                assert(int(sumk) == n * (n-1) / 2);
            }
        }
    }
}

int main(int argc, char** argv) {

    // Test some functions that tuples depend on.
    assert(div_equally_size_n(6, 4, 0) == 2);
    assert(div_equally_size_n(6, 4, 1) == 2);
    assert(div_equally_size_n(6, 4, 2) == 1);
    assert(div_equally_size_n(6, 4, 3) == 1);
    assert(div_equally_cumu_size_n(6, 4, -1) == 0);
    assert(div_equally_cumu_size_n(6, 4, 0) == 2);
    assert(div_equally_cumu_size_n(6, 4, 1) == 4);
    assert(div_equally_cumu_size_n(6, 4, 2) == 5);
    assert(div_equally_cumu_size_n(6, 4, 3) == 6);

    // Test tuples.
    ttest(true);
    ttest(false);
    cout << "End of YASK tuple test.\n";
    return 0;
}
