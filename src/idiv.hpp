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

#ifndef IDIV_HPP
#define IDIV_HPP

namespace yask {

    // Floored integer modulo and division.
    // (Same as euclidian math when denominator b is positive.)
    // See https://en.wikipedia.org/wiki/Modulo_operation.
    // These maintain a repeating pattern even with negative a's:
    // -6 / +3 = -2, -6 % +3 = +0, idiv(-6, +3) = -2, imod(-6, +3) = +0
    // -5 / +3 = -1, -5 % +3 = -2, idiv(-5, +3) = -2, imod(-5, +3) = +1
    // -4 / +3 = -1, -4 % +3 = -1, idiv(-4, +3) = -2, imod(-4, +3) = +2
    // -3 / +3 = -1, -3 % +3 = +0, idiv(-3, +3) = -1, imod(-3, +3) = +0
    // -2 / +3 = +0, -2 % +3 = -2, idiv(-2, +3) = -1, imod(-2, +3) = +1
    // -1 / +3 = +0, -1 % +3 = -1, idiv(-1, +3) = -1, imod(-1, +3) = +2
    // +0 / +3 = +0, +0 % +3 = +0, idiv(+0, +3) = +0, imod(+0, +3) = +0
    // +1 / +3 = +0, +1 % +3 = +1, idiv(+1, +3) = +0, imod(+1, +3) = +1
    // +2 / +3 = +0, +2 % +3 = +2, idiv(+2, +3) = +0, imod(+2, +3) = +2
    // +3 / +3 = +1, +3 % +3 = +0, idiv(+3, +3) = +1, imod(+3, +3) = +0
    // +4 / +3 = +1, +4 % +3 = +1, idiv(+4, +3) = +1, imod(+4, +3) = +1
    // +5 / +3 = +1, +5 % +3 = +2, idiv(+5, +3) = +1, imod(+5, +3) = +2
    // +6 / +3 = +2, +6 % +3 = +0, idiv(+6, +3) = +2, imod(+6, +3) = +0
    // +7 / +3 = +2, +7 % +3 = +1, idiv(+7, +3) = +2, imod(+7, +3) = +1
    // +8 / +3 = +2, +8 % +3 = +2, idiv(+8, +3) = +2, imod(+8, +3) = +2
    template<typename T>
    inline T idiv(T a, T b) {
        //return (a<0 ? a-(b-1) : a) / b;
        //return (a - (a<0 ? b-1 : 0)) / b;
        return (a + (a>>(sizeof(a)*8-1)) * (b-1)) / b;
    }
    template<typename T>
    inline T imod(T a, T b) {
        //return ((a % b) + b) % b;
        //return ((a < 0) ? ((a % b) + b) : a) % b;
        //T c = a % b; return (c < 0) ? c + b : c;
        T c = a % b; return c - ((c>>(sizeof(c)*8-1)) * b);
    }
}

#endif
