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

///////// Stencil support.

#pragma once

// Include this first to assure NDEBUG is set properly.
#include "yask_assert.hpp"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <sstream>
#include <functional>
#include <unordered_map>
#include <list>
#include <deque>
#include <vector>
#include <cstdarg>
#include <string.h>

#include "common_utils.hpp"

namespace yask {

    template <typename T>
    class Tuple;

    // One named arithmetic value.
    // Class is not virtual for performance.
    template <typename T>
    class Scalar {
        friend Tuple<T>;

    protected:

        // A shared global pool for names.
        // Using a list so addrs won't change.
        static std::list<std::string> _allNames;

        // Look up names in the pool.
        // Should only need to do this when we're adding a new dim.
        static const std::string* _getPoolPtr(const std::string& name);

        // Name and value for this object.
        const std::string* _namep = 0; // Ptr from the _allNames pool.
        T _val = 0;

        Scalar(const std::string* namep, const T& val) {
            _namep = namep;
            _val = val;
        }

    public:
        Scalar(const std::string& name, const T& val) {
            assert(name.length() > 0);
            _namep = _getPoolPtr(name);
            _val = val;
        }
        Scalar(const std::string& name) : Scalar(name, 0) { }
        ~Scalar() { }

        // Access name.
        // (Changing it is not allowed.)
        const std::string& getName() const {
            return *_namep;
        }
        const std::string* getNamePtr() const {
            return _namep;
        }

        // Access value.
        const T& getVal() const { return _val; }
        T& getVal() { return _val; }
        const T* getValPtr() const { return &_val; }
        T* getValPtr() { return &_val; }
        void setVal(const T& val) { _val = val; }

        // Comparison ops.
        // Compare name pointers and actual names in case there
        // is more than one pool, which can happen when loading
        // more than one dynamic lib.
        bool operator==(const Scalar& rhs) const {
            return _val == rhs._val &&
                (_namep == rhs._namep || *_namep == *rhs._namep);
        }
        bool operator<(const Scalar& rhs) const {
            return (_val < rhs._val) ? true :
                (_val > rhs._val) ? false :
                (_namep == rhs._namep) ? false :
                (*_namep < *rhs._namep);
        }
    };

    // Collection of named items of one arithmetic type.
    // Can represent:
    // - an n-D space with given sizes.
    // - a point in an n-D space.
    // - a vector from (0,...,0) in n-D space.
    // - values at a point in n-D space.
    // - etc.
    // Class is not virtual for performance.
    template <typename T>
    class Tuple {
    protected:

        // Dimensions and values for this Tuple.
        // Adding to front is unusual, so using a vector instead of a deque,
        // which is less efficient.
        std::vector<Scalar<T>> _q;

        // First-inner vars control ordering. Example: dims x, y, z.
        // If _firstInner == true, x is unit stride (col major).
        // If _firstInner == false, z is unit stride (row major).
        // This setting affects [un]layout() and visitAllPoints().
        bool _firstInner = true; // whether first dim is used for inner loop.

    public:
        Tuple() {}
        ~Tuple() {}             // NOT a virtual class!

        // first-inner (first dim is unit stride) accessors.
        bool isFirstInner() const { return _firstInner; }
        void setFirstInner(bool fi) { _firstInner = fi; }

        // Query number of dims.
        size_t size() const {
            return _q.size();
        }
        int getNumDims() const {
            return int(_q.size());
        }

        // Return all dim names.
        const std::vector<std::string> getDimNames() const;

        // Get iteratable contents.
        const std::vector<Scalar<T>>& getDims() const {
            return _q;
        }
        typename std::vector<Scalar<T>>::const_iterator begin() const {
            return _q.begin();
        }
        typename std::vector<Scalar<T>>::const_iterator end() const {
            return _q.end();
        }

        // Clear data.
        void clear() {
            _q.clear();
        }

        ////// Methods to get things by position.

        // Return pointer to scalar pair or null if it doesn't exist.
        // Lookup by dim posn.
        // No non-const version because name shouldn't be modified
        // outside of this class.
        const Scalar<T>* getDimPtr(int i) const {
            return (i >= 0 && i < int(_q.size())) ?
                &_q.at(i) : NULL;
        }

        // Return dim name at index (must exist).
        const std::string& getDimName(int i) const {
            auto* p = getDimPtr(i);
            assert(p);
            return p->getName();
        }

        // Return pointer to value or null if it doesn't exist.
        // Lookup by dim posn.
        const T* lookup(int i) const {
            return (i >= 0 && i < int(_q.size())) ?
                _q.at(i).getValPtr() : NULL;
        }
        T* lookup(int i) {
            return (i >= 0 && i < int(_q.size())) ?
                _q.at(i).getValPtr() : NULL;
        }

        // Return scalar pair at index (must exist).
        const Scalar<T>& getDim(int i) const {
            auto* p = getDimPtr(i);
            assert(p);
            return *p;
        }
        const Scalar<T>& operator()(int i) const {
            return getDim(i);
        }

        // Lookup and return value by dim posn (must exist).
        const T& getVal(int i) const {
            auto* p = lookup(i);
            assert(p);
            return *p;
        }
        T& getVal(int i) {
            auto* p = lookup(i);
            assert(p);
            return *p;
        }
        const T& operator[](int i) const {
            return getVal(i);
        }
        T& operator[](int i) {
            return getVal(i);
        }

        ////// Methods to get things by name.

        // Return dim posn or -1 if it doesn't exist.
        // Lookup by name.
        int lookup_posn(const std::string& dim) const {

            // First check pointers.
            for (size_t i = 0; i < _q.size(); i++) {
                auto& s = _q[i];
                if (s.getNamePtr() == &dim)
                    return int(i);
            }

            // Then check full strings.
            for (size_t i = 0; i < _q.size(); i++) {
                auto& s = _q[i];
                if (s.getName() == dim)
                    return int(i);
            }
            return -1;
        }

        // Return scalar pair by name (must exist).
        const Scalar<T>& getDim(const std::string& dim) const {
            int i = lookup_posn(dim);
            assert(i >= 0);
            return _q.at(i);
        }

        // Return pointer to value or null if it doesn't exist.
        // Lookup by name.
        const T* lookup(const std::string& dim) const {
            int i = lookup_posn(dim);
            return (i >= 0) ? &_q[i]._val : NULL;
        }
        T* lookup(const std::string& dim) {
            int i = lookup_posn(dim);
            return (i >= 0) ? &_q[i]._val : NULL;
        }

        // Lookup and return value by dim name (must exist).
        const T& getVal(const std::string& dim) const {
            auto* p = lookup(dim);
            assert(p);
            return *p;
        }
        T& getVal(const std::string& dim) {
            auto* p = lookup(dim);
            assert(p);
            return *p;
        }
        const T& operator[](const std::string& dim) const {
            return getVal(dim);
        }
        T& operator[](const std::string& dim) {
            return getVal(dim);
        }

        ////// Other methods.

        // Add dimension to tuple (or update if it exists).
        void addDimBack(const std::string& dim, const T& val);
        void addDimBack(const Scalar<T>& sc) {
            addDimBack(sc.getName(), sc.getVal());
        }
        void addDimFront(const std::string& dim, const T& val);
        void addDimFront(const Scalar<T>& sc) {
            addDimFront(sc.getName(), sc.getVal());
        }

        // Set value by dim posn (posn i must exist).
        void setVal(int i, const T& val) {
            T* p = lookup(i);
            assert(p);
            *p = val;
        }

        // Set value by dim name (dim must already exist).
        void setVal(const std::string& dim, const T& val) {
            T* p = lookup(dim);
            assert(p);
            *p = val;
        }

        // Set multiple values.  Assumes values are in same order as
        // existing names.  If there are more values in 'vals' than 'this',
        // extra values are ignored.  If there are fewer values in 'vals'
        // than 'this', only the number of values supplied will be updated.
        void setVals(int numVals, const T vals[]) {
            int end = std::min(numVals, int(_q.size()));
            for (int i = 0; i < end; i++)
                setVal(i, vals[i]);
        }
        void setVals(const std::vector<T>& vals) {
            int end = int(std::min(vals.size(), _q.size()));
            for (int i = 0; i < end; i++)
                setVal(i, vals.at(i));
        }
        void setVals(const std::deque<T>& vals) {
            int end = int(std::min(vals.size(), _q.size()));
            for (int i = 0; i < end; i++)
                setVal(i, vals.at(i));
        }

        // Set all values to the same value.
        void setValsSame(const T& val) {
            for (auto& i : _q)
                i.setVal(val);
        }

        // Set values from 'src' Tuple, leaving non-matching ones in this
        // unchanged. Add dimensions in 'src' that are not in 'this' iff
        // addMissing==true.
        // Different than the copy operator because this method does
        // not change the order of *this or remove any existing dims.
        void setVals(const Tuple& src, bool addMissing);

        // This version is similar to vprintf.
        // 'args' must have been initialized with va_start
        // and must contain values of of type T2.
        template<typename T2>
        void setVals(int numVals, va_list args) {
            assert(size() == numVals);

            // process the var args.
            for (int i = 0; i < numVals; i++) {
                T2 n = va_arg(args, T2);
                setVal(i, T(n));
            }
        }
        template<typename T2>
        void setVals(int numVals, ...) {

            // pass the var args.
            va_list args;
            va_start(args, numVals);
            setVals<T2>(numVals, args);
            va_end(args);
        }

        // Copy 'this', then add dims and values from 'rhs' that are NOT
        // in 'this'. Return resulting union.
        // Similar to setVals(rhs, true), but does not change existing
        // values and makes a new Tuple.
        Tuple makeUnionWith(const Tuple& rhs) const;

        // Check whether dims are the same.
        // Don't have to be in same order unless 'sameOrder' is true.
        bool areDimsSame(const Tuple& rhs, bool sameOrder = false) const;

        // Equality is true if all dimensions and values are same.
        // Dimensions must be in same order.
        bool operator==(const Tuple& rhs) const;

        // Less-than is true if first value that is different
        // from corresponding value in 'rhs' is less than it.
        // If all values are same, compares dims.
        bool operator<(const Tuple& rhs) const;

        // Other comparisons derived from above.
        bool operator!=(const Tuple& rhs) const {
            return !((*this) == rhs);
        }
        bool operator <=(const Tuple& rhs) const {
            return ((*this) == rhs) || ((*this) < rhs);
        }
        bool operator>(const Tuple& rhs) const {
            return !((*this) <= rhs);
        }
        bool operator >=(const Tuple& rhs) const {
            return !((*this) < rhs);
        }

        // Convert nD 'offsets' to 1D offset using values in 'this' as sizes of nD space.
        // If 'strictRhs', RHS dims must be same and in same order as this;
        // else, only matching ones are considered and missing offsets are zero (0).
        // If '_firstInner', first dim varies most quickly; else last dim does.
        size_t layout(const Tuple& offsets, bool strictRhs=true) const;

        // Convert 1D 'offset' to nD offsets using values in 'this' as sizes of nD space.
        Tuple unlayout(size_t offset) const;

        // Create a new Tuple with the given dimension removed.
        // if dim is found, new Tuple will have one fewer dim than 'this'.
        // If dim is not found, it will be a copy of 'this'.
        Tuple removeDim(const std::string& dim) const {
            auto p = lookup_posn(dim);
            Tuple newt = removeDim(p);
            return newt;
        }

        // Create a new Tuple with the given dimension removed.
        Tuple removeDim(int posn) const;

        // reductions.
        // Apply function over all elements, returning one value.
        T reduce(std::function<T (T lhs, T rhs)> reducer) const {
            T result = 0;
            int n = 0;
            for (auto i : _q) {
                auto& tval = i.getVal();
                result = (n == 0) ? tval : reducer(result, tval);
                n++;
            }
            return result;
        }
        T sum() const {
            return reduce([&](T lhs, T rhs){ return lhs + rhs; });
        }
        T product() const {
            return _q.size() ?
                reduce([&](T lhs, T rhs){ return lhs * rhs; }) : 1;
        }
        T max() const {
            return reduce([&](T lhs, T rhs){ return std::max(lhs, rhs); });
        }
        T min() const {
            return reduce([&](T lhs, T rhs){ return std::min(lhs, rhs); });
        }

        // pair-wise functions.
        // Apply function to each pair, creating a new Tuple.
        // if strictRhs==true, RHS elements must be same as this;
        // else, only matching ones are considered.
        Tuple combineElements(std::function<T (T lhs, T rhs)> combiner,
                              const Tuple& rhs,
                              bool strictRhs=true) const {
            Tuple newt = *this;
            if (strictRhs) {
                assert(areDimsSame(rhs, true));
                for (size_t i = 0; i < _q.size(); i++) {
                    auto& tval = _q[i].getVal();
                    auto& rval = rhs[i];
                    T newv = combiner(tval, rval);
                    newt[i] = newv;
                }
            }
            else {
                for (auto& i : _q) {
                    auto& tdim = i.getName();
                    auto& tval = i.getVal();
                    auto* rp = rhs.lookup(tdim);
                    if (rp) {
                        T newv = combiner(tval, *rp);
                        newt.setVal(tdim, newv);
                    }
                }
            }
            return newt;
        }
        Tuple addElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return lhs + rhs; },
                                   rhs, strictRhs);
        }
        Tuple subElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return lhs - rhs; },
                                   rhs, strictRhs);
        }
        Tuple multElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return lhs * rhs; },
                                   rhs, strictRhs);
        }
        Tuple maxElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return std::max(lhs, rhs); },
                                   rhs, strictRhs);
        }
        Tuple minElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return std::min(lhs, rhs); },
                                   rhs, strictRhs);
        }

        // Apply func to each element, creating a new Tuple.
        Tuple mapElements(std::function<T (T lhs, T rhs)> func,
                                 T rhs) const {
            Tuple newt = *this;
            for (size_t i = 0; i < _q.size(); i++) {
                auto& tval = _q[i].getVal();
                T newv = func(tval, rhs);
                newt[i] = newv;
            }
            return newt;
        }
        Tuple mapElements(std::function<T (T in)> func) const {
            Tuple newt = *this;
            for (size_t i = 0; i < _q.size(); i++) {
                auto& tval = _q[i].getVal();
                T newv = func(tval);
                newt[i] = newv;
            }
            return newt;
        }
        Tuple addElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return lhs + rhs; },
                               rhs);
        }
        Tuple subElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return lhs - rhs; },
                               rhs);
        }
        Tuple multElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return lhs * rhs; },
                               rhs);
        }
        Tuple maxElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return std::max(lhs, rhs); },
                               rhs);
        }
        Tuple minElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return std::min(lhs, rhs); },
                               rhs);
        }
        Tuple negElements() const {
            return mapElements([&](T in){ return -in; });
        }
        Tuple absElements() const {
            return mapElements([&](T in){ return T(llabs(in)); });
        }

        // make string like "4x3x2" or "4, 3, 2".
        std::string makeValStr(std::string separator=", ",
                                       std::string prefix="",
                                      std::string suffix="") const;

        // make string like "x, y, z" or "int x, int y, int z".
        std::string makeDimStr(std::string separator=", ",
                                       std::string prefix="",
                                      std::string suffix="") const;

        // make string like "x=4, y=3, z=2".
        std::string makeDimValStr(std::string separator=", ",
                                          std::string infix="=",
                                          std::string prefix="",
                                  std::string suffix="") const;

        // make string like "x+4, y, z-2".
        std::string makeDimValOffsetStr(std::string separator=", ",
                                                std::string prefix="",
                                               std::string suffix="") const;

        // Return a "compact" set of K factors of N,
        // a set of factors with largest factor as small as possible,
        // where K is the size of 'this'.
        // Any non-zero numbers in 'this' will be kept if possible.
        Tuple get_compact_factors(idx_t N) const;

        // Call the 'visitor' lambda function at every point in the space defined by 'this'.
        // 'idx' parameter contains sequentially-numbered index.
        // Visitation order is with first dimension in unit stride, i.e., a conceptual
        // "outer loop" iterates through last dimension, ..., and an "inner loop" iterates
        // through first dimension. If '_firstInner' is false, it is done the opposite way.
        // Visitor should return 'true' to keep going or 'false' to stop.
        void visitAllPoints(std::function<bool (const Tuple&,
                                                size_t idx)> visitor) const {

            // Init lambda fn arg with *this to get dim names.
            // Values will get set during scan.
            Tuple tp(*this);

            // 0-D?
            if (!_q.size())
                visitor(tp, 0);

            // Call recursive version.
            // Set begin/step dims depending on nesting.
            else if (_firstInner)
                _visitAllPoints(visitor, size()-1, -1, tp);
            else
                _visitAllPoints(visitor, 0, 1, tp);
        }

        // Call the 'visitor' lambda function at every point in the space defined by 'this'.
        // 'idx' parameter contains sequentially-numbered index.
        // Visitation order is not predictable.
        // Visitor return value only stops visit on one thread.
        void visitAllPointsInParallel(std::function<bool (const Tuple&,
                                                          size_t idx)> visitor) const {

            // 0-D?
            if (!_q.size()) {
                Tuple tp(*this);
                visitor(tp, 0);
            }

            // Call order-independent version.
            // Set begin/end/step dims depending on nesting.
            // TODO: set this depending on dim sizes.
            else if (_firstInner)
                _visitAllPointsInPar(visitor, size()-1, -1);
            else
                _visitAllPointsInPar(visitor, 0, 1);
        }

    protected:

        // Visit elements recursively.
        bool _visitAllPoints(std::function<bool (const Tuple&, size_t idx)> visitor,
                             int curDimNum, int step, Tuple& tp) const {
            auto& sc = _q.at(curDimNum);
            auto dsize = sc.getVal();
            int lastDimNum = (step > 0) ? size()-1 : 0;

            // If no more dims, iterate along current dimension and call
            // visitor.
            if (curDimNum == lastDimNum) {

                // Get unique index to first position.
                tp.setVal(curDimNum, 0);
                size_t idx0 = layout(tp);

                // Loop through points.
                for (T i = 0; i < dsize; i++) {
                    tp.setVal(curDimNum, i);
                    bool ok = visitor(tp, idx0 + i);

                    // Leave if visitor returns false.
                    if (!ok)
                        return false;
                }
            }

            // Else, iterate along current dimension and recurse to
            // next/prev dimension.
            else {
                for (T i = 0; i < dsize; i++) {
                    tp.setVal(curDimNum, i);

                    // Recurse.
                    bool ok = _visitAllPoints(visitor, curDimNum + step, step, tp);

                    // Leave if visitor returns false.
                    if (!ok)
                        return false;
                }
            }
            return true;
        }

        // First call from public visitAllPointsInParallel(visitor).
        bool _visitAllPointsInPar(std::function<bool (const Tuple&, size_t idx)> visitor,
                                  int curDimNum, int step) const {
#ifdef _OPENMP
            auto nd = getNumDims();

            // If one dim, parallelize across it.
            if (nd == 1) {
                assert(curDimNum == 0);
                auto dsize = getVal(curDimNum);
                Tuple tp(*this);

                // Loop through points.
                // Each thread gets its own copy of 'tp', which
                // gets updated with the loop index.
                // TODO: convert to yask_parallel_for().
#pragma omp parallel for firstprivate(tp)
                for (T i = 0; i < dsize; i++) {
                    tp.setVal(curDimNum, i);
                    visitor(tp, i);
                }
            }

            // If >1 dim, parallelize over outer dims only,
            // streaming across inner dim in each thread.
            // This is to maximize HW prefetch benefit.
            else {

                // Total number of elements to visit.
                T ne = product();

                // Number of elements in last dim.
                int lastDimNum = (step > 0) ? nd-1 : 0;
                T nel = getVal(lastDimNum);

                // Parallel loop over elements w/stride = size of
                // last dim.
                yask_parallel_for(0, ne, nel,
                                  [&](idx_t start, idx_t stop, idx_t thread_num) {
                                      
                                      // Convert linear index to n-dimensional tuple.
                                      Tuple tp = unlayout(start);
                                      
                                      // Visit points in last dim.
                                      _visitAllPoints(visitor, lastDimNum, step, tp);
                                  });
            }
            return true;
#else

            // Call recursive version to handle all dims.
            Tuple tp(*this);
            return _visitAllPoints(visitor, curDimNum, step, tp);
#endif
        }
            
    }; // Tuple.

    // Explicit types.
    typedef Scalar<int> IntScalar;
    typedef Scalar<idx_t> IdxScalar;
    typedef Tuple<int> IntTuple;
    typedef Tuple<idx_t> IdxTuple;

} // namespace yask.

// Provide a hash operator for a Tuple.
// Needed for unordered_map.
// This needs to be in the 'std' namespace.
namespace std {
    template <typename T>
    class hash<yask::Tuple<T>>{
    public :
        size_t operator()(const yask::Tuple<T> &x ) const {
            size_t h = 0;
            for (int i = 0; i < x.getNumDims(); i++) {
                h ^= size_t(i) ^ std::hash<T>()(x.getVal(i)) ^ std::hash<std::string>()(x.getDimName(i));
            }
            return h;
        }
    };
}
