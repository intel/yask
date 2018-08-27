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
#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

//////// Some common code shared between YASK compiler and kernel. //////////

#include <set>
#include <vector>
#include <map>

// Macro for throwing yask_exception with a string.
// Example: THROW_YASK_EXCEPTION("all your base are belong to us");
#define THROW_YASK_EXCEPTION(message) do {                          \
        yask_exception e(message);                                  \
        throw e;                                                    \
    } while(0)

// Macro for creating a string and throwing yask_exception with it.
// Example: FORMAT_AND_THROW_YASK_EXCEPTION("bad value: x = " << x);
#define FORMAT_AND_THROW_YASK_EXCEPTION(message) do {               \
        yask_exception e;                                           \
        std::stringstream err;                                      \
        err << message;                                             \
        e.add_message(err.str());                                   \
        throw e;                                                    \
    } while(0)

namespace yask {
    
    // Set that retains order of things added.
    // Or, vector that allows insertion if element doesn't exist.
    // TODO: hide vector inside class and provide proper accessor methods.
    template <typename T>
    class vector_set : public std::vector<T> {
        std::map<T, size_t> _posn;

    private:
        virtual void push_front(const T& val) {
            THROW_YASK_EXCEPTION("push_front() not allowed");
        }

    public:
        vector_set() {}
        virtual ~vector_set() {}

        // Copy ctor.
        vector_set(const vector_set& src) :
            std::vector<T>(src), _posn(src._posn) {}

        virtual size_t count(const T& val) const {
            return _posn.count(val);
        }
        virtual void insert(const T& val) {
            if (_posn.count(val) == 0) {
                std::vector<T>::push_back(val);
                _posn[val] = std::vector<T>::size() - 1;
            }
        }
        virtual void push_back(const T& val) {
            insert(val);
        }
        virtual void erase(const T& val) {
            if (_posn.count(val) > 0) {
                size_t op = _posn.at(val);
                std::vector<T>::erase(std::vector<T>::begin() + op);
                for (auto pi : _posn) {
                    auto& p = pi.second;
                    if (p > op)
                        p--;
                }
            }
        }
        virtual void clear() {
            std::vector<T>::clear();
            _posn.clear();
        }
    };

} // namespace.

#endif /* SRC_COMMON_COMMON_UTILS_HPP_ */
