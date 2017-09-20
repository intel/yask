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

// Test the YASK tuples.

// enable assert().
#define DEBUG

#include "yask.hpp"
using namespace std;
using namespace yask;

int main(int argc, char** argv) {

    ostream& os = cout;
    
    IdxTuple t1;
    t1.setFirstInner(true);
    t1.addDimBack("x", 3);
    t1.addDimBack("y", 4);
    os << "space: " << t1.makeDimValStr() << ", is ";
    if (!t1.isFirstInner()) os << "NOT ";
    os << "first-inner layout.\n";

    for (int y = 0; y < t1["y"]; y++) {
        for (int x = 0; x < t1["x"]; x++) {

            IdxTuple ofs;
            ofs.addDimBack("x", x);
            ofs.addDimBack("y", y);

            auto i = t1.layout(ofs);
            os << " offset at " << ofs.makeDimValStr() << " = " << i << endl;

            IdxTuple ofs2 = t1.unlayout(i);
            assert(ofs == ofs2);
        }
    }
    
    os << "End of YASK tuple test.\n";
    return 0;
}
