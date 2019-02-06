/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2019, Intel Corporation

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

///////// Tuple implementation.

#include "yask_common_api.hpp"
#include "tuple.hpp"

using namespace std;

namespace yask {

    // Declare static members.
    template <typename T>
    std::list<std::string> Scalar<T>::_allNames;

    // Implementations.

    template <typename T>
    const std::vector<std::string> Tuple<T>::getDimNames() const {
        std::vector<std::string> names;
        for (auto& i : _q)
            names.push_back(i.getName());
        return names;
    }

    template <typename T>
    void Tuple<T>::addDimBack(const std::string& dim, const T& val) {
        auto* p = lookup(dim);
        if (p)
            *p = val;
        else {
            Scalar<T> sv(dim, val);
            _q.push_back(sv);
        }
    }
    template <typename T>
    void Tuple<T>::addDimFront(const std::string& dim, const T& val) {
        auto* p = lookup(dim);
        if (p)
            *p = val;
        else {
            Scalar<T> sv(dim, val);
            _q.insert(_q.begin(), sv);
        }
    }

    template <typename T>
    void Tuple<T>::setVals(const Tuple& src, bool addMissing) {
        for (auto& i : src.getDims()) {
            auto& dim = i.getName();
            auto& val = i.getVal();
            auto* p = lookup(dim);
            if (p)
                *p = val;
            else if (addMissing)
                addDimBack(dim, val);
        }
    }

    template <typename T>
    Tuple<T> Tuple<T>::makeUnionWith(const Tuple& rhs) const {
        Tuple u = *this;    // copy.
        for (auto& i : rhs._q) {
            auto& dim = i.getName();
            auto& val = i.getVal();
            auto* p = u.lookup(dim);
            if (!p)
                u.addDimBack(dim, val);
        }
        return u;
    }

    template <typename T>
    bool Tuple<T>::areDimsSame(const Tuple& rhs, bool sameOrder) const {
        if (size() != rhs.size())
            return false;

        // Dims must be in same order.
        if (sameOrder) {
            for (size_t i = 0; i < _q.size(); i++) {
                auto& n = _q[i].getName();
                auto& rn = rhs._q[i].getName();
                if (n != rn)
                    return false;
            }
        }

        // Dims can be in any order.
        else {
            for (auto& i : _q) {
                auto& dim = i.getName();
                if (!rhs.lookup(dim))
                    return false;
            }
        }
        return true;
    }

    template <typename T>
    bool Tuple<T>::operator==(const Tuple& rhs) const {

        // Check dims.
        if (!areDimsSame(rhs, true))
            return false;

        // Check values.
        for (size_t i = 0; i < _q.size(); i++) {
            if (getVal(i) != rhs.getVal(i))
                return false;
        }
        return true;
    }

    template <typename T>
    bool Tuple<T>::operator<(const Tuple& rhs) const {
        if (size() < rhs.size()) return true;
        else if (size() > rhs.size()) return false;

        // compare vals.
        for (size_t i = 0; i < _q.size(); i++) {
            auto v = getVal(i);
            auto rv = rhs.getVal(i);
            if (v < rv)
                return true;
            else if (v > rv)
                return false;
        }

        // compare dims.
        for (size_t i = 0; i < _q.size(); i++) {
            auto& n = _q[i].getName();
            auto& rn = rhs._q[i].getName();
            if (n < rn)
                return true;
            else if (n > rn)
                return false;
        }
        return false;
    }

    template <typename T>
    size_t Tuple<T>::layout(const Tuple& offsets, bool strictRhs) const {
        if (strictRhs)
            assert(areDimsSame(offsets, true));
        size_t idx = 0;
        size_t prevSize = 1;

        // Loop thru dims.
        int startDim = _firstInner ? 0 : size()-1;
        int endDim = _firstInner ? size() : -1;
        int stepDim = _firstInner ? 1 : -1;
        for (int di = startDim; di != endDim; di += stepDim) {
            auto& i = _q.at(di);
            assert(i.getVal() >= 0);
            size_t dsize = size_t(i.getVal());

            // offset into this dim.
            size_t offset = 0;
            if (strictRhs) {
                assert(offsets[di] >= 0);
                offset = size_t(offsets[di]);
            } else {
                auto& dim = i.getName();
                auto* op = offsets.lookup(dim);
                if (op) {
                    assert(*op >= 0);
                    offset = size_t(*op);
                }
            }
            assert(offset < dsize);

            // mult offset by product of previous dims.
            idx += (offset * prevSize);
            assert(idx >= 0);
            assert(idx < size_t(product()));

            prevSize *= dsize;
            assert(prevSize <= size_t(product()));
        }
        return idx;
    }

    template <typename T>
    Tuple<T> Tuple<T>::unlayout(size_t offset) const {
        Tuple res = *this;
        size_t prevSize = 1;

        // Loop thru dims.
        int startDim = _firstInner ? 0 : size()-1;
        int endDim = _firstInner ? size() : -1;
        int stepDim = _firstInner ? 1 : -1;
        for (int di = startDim; di != endDim; di += stepDim) {
            auto& i = _q.at(di);
            //auto& dim = i.getName();
            size_t dsize = size_t(i.getVal());
            assert (dsize >= 0);

            // Div offset by product of previous dims.
            size_t dofs = offset / prevSize;

            // Wrap within size of this dim.
            dofs %= dsize;

            // Save in result.
            res[di] = dofs;

            prevSize *= dsize;
            assert(prevSize <= size_t(product()));
        }
        return res;
    }

    template <typename T>
    Tuple<T> Tuple<T>::removeDim(int posn) const {

        // For some reason, copying *this and erasing
        // the element in newt._q causes an exception.
        Tuple newt;
        for (int i = 0; i < size(); i++) {
            if (i != posn)
                newt.addDimBack(getDimName(i), getVal(i));
        }
        return newt;
    }

    template <typename T>
    std::string Tuple<T>::makeValStr(std::string separator,
                                     std::string prefix,
                                     std::string suffix) const {
        std::ostringstream oss;
        int n = 0;
        for (auto i : _q) {
            //auto& tdim = i.getName();
            auto& tval = i.getVal();
            if (n) oss << separator;
            oss << prefix << tval << suffix;
            n++;
        }
        return oss.str();
    }

    template <typename T>
    std::string Tuple<T>::makeDimStr(std::string separator,
                                     std::string prefix,
                                     std::string suffix) const {
        std::ostringstream oss;
        int n = 0;
        for (auto i : _q) {
            auto& tdim = i.getName();
            //auto& tval = i.getVal();
            if (n) oss << separator;
            oss << prefix << tdim << suffix;
            n++;
        }
        return oss.str();
    }

    template <typename T>
    std::string Tuple<T>::makeDimValStr(std::string separator,
                                        std::string infix,
                                        std::string prefix,
                                        std::string suffix) const {
        std::ostringstream oss;
        int n = 0;
        for (auto i : _q) {
            auto& tdim = i.getName();
            auto& tval = i.getVal();
            if (n) oss << separator;
            oss << prefix << tdim << infix << tval << suffix;
            n++;
        }
        return oss.str();
    }

    template <typename T>
    std::string Tuple<T>::makeDimValOffsetStr(std::string separator,
                                              std::string prefix,
                                              std::string suffix) const {
        std::ostringstream oss;
        int n = 0;
        for (auto i : _q) {
            auto& tdim = i.getName();
            auto& tval = i.getVal();
            if (n) oss << separator;
            oss << prefix << tdim;
            if (tval > 0)
                oss << "+" << tval;
            else if (tval < 0)
                oss << tval; // includes '-';
            oss << suffix;

            n++;
        }
        return oss.str();
    }

    // Explicitly allowed instantiations.
    template class Tuple<int>;
    template class Tuple<idx_t>;

} // namespace yask.
