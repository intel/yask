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

#pragma once

// Include this first to assure NDEBUG is set properly.
#include "yask_assert.hpp"

#include <cstddef>

namespace yask {

    // Return the number of ways to choose 'k' things from a set of 'n'.
    int n_choose_k(int n, int k);

    // Get the 'r'th set of 'k' elements from set of integers between '0' and 'n-1'.
    // Returns bitmask with 'k' bits set, where bit 'b' = 1 indicates integer 'b' is in set.
    // 'r' must be between '0' and 'n_choose_k(n, k)-1'.
    size_t n_choose_k_set(int n, int k, int r);

    // Handle bits in a bitset.
    template<typename T>
    inline bool is_bit_set(const T set, int i) {
        assert(i >= 0);
        assert(size_t(i) <= sizeof(T) * 8);
        return (set & (T(1) << i)) != 0;
    }
    template<typename T>
    inline void set_bit(T& set, int i) {
        assert(i >= 0);
        assert(size_t(i) <= sizeof(T) * 8);
        set |= (T(1) << i);
    }
    template<typename T>
    inline void clear_bit(T& set, int i) {
        assert(i >= 0);
        assert(size_t(i) <= sizeof(T) * 8);
        set &= ~(T(1) << i);
    }
}
