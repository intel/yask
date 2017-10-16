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

///////// Stencil support.

#pragma once

#include <assert.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <sstream>
#include <functional>
#include <map>
#include <set>
#include <deque>
#include <vector>
#include <cstdarg>
#include <string.h>

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
        // Use a map instead of a set to have reliable pointers
        // for the string values.
        static std::map<std::string, std::string> _allNames;

        // Look up names in the pool.
        static const std::string* _getPoolPtr(const std::string& name) {

            // Get existing entry or add.
            // Add the value to be the same as the key because
            // addr of key might change, but addr of value won't.
            auto i = _allNames.emplace(name, name); // returns iterator + bool pair.
            auto& i2 = *i.first; // iterator to element pair.
            auto& i3 = i2.second; // ref to value.
            return &i3;           // ptr to value.
        }

        // Name and value for this object.
        const std::string* _namep = 0; // Ptr from the _allNames pool.
        T _val = 0;

        inline const char* _getCStr() const {
            return _namep->c_str();
        }
        
    public:
        Scalar(const std::string& name, const T& val) {
            assert(name.length() > 0);
            _namep = _getPoolPtr(name);
            _val = val;
        }
        Scalar(const std::string& name) : Scalar(name, 0) { }
        Scalar() : Scalar("", 0) { }
        inline ~Scalar() { }

        // Access name.
        // (Changing it is not allowed.)
        inline const std::string& getName() const {
            return *_namep;
        }

        // Access value.
        inline const T& getVal() const { return _val; }
        inline T& getVal() { return _val; }
        inline const T* getValPtr() const { return &_val; }
        inline T* getValPtr() { return &_val; }
        inline void setVal(const T& val) { _val = val; }

        // Comparison ops.
        inline bool operator==(const Scalar& rhs) const {
            return _val == rhs._val &&
                (_namep == rhs._namep || *_namep == *rhs._namep);
        }
        inline bool operator<(const Scalar& rhs) const {
            return (_val < rhs._val) ? true :
                (_val > rhs._val) ? false :
                (_namep == rhs._namep) ? false :
                (*_namep < *rhs._namep) ? true : false;
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
        // Adding to front is unusual, so using a vector instead of a deque.
        std::vector<Scalar<T>> _q;

        // First-inner vars control ordering. Example: dims x, y, z.
        // If _firstInner == true, x is unit stride (col major).
        // If _firstInner == false, z is unit stride (row major).
        // This setting affects [un]layout() and visitAllPoints().
        bool _firstInner = true; // whether first dim is used for inner loop.

    public:
        Tuple() {}
        inline ~Tuple() {}

        // first-inner (first dim is unit stride) accessors.
        inline bool isFirstInner() const { return _firstInner; }
        inline void setFirstInner(bool fi) { _firstInner = fi; }

        // Query number of dims.
        inline int size() const {
            return int(_q.size());
        }
        inline int getNumDims() const {
            return int(_q.size());
        }

        // Return all dim names.
        inline const std::vector<std::string> getDimNames() const {
            std::vector<std::string> names;
            for (auto& i : _q)
                names.push_back(i.getName());
            return names;
        }
        
        // Get iteratable contents.
        inline const std::vector<Scalar<T>>& getDims() const {
            return _q;
        }

        // Clear data.
        inline void clear() {
            _q.clear();
        }

        ////// Methods to get things by position.
        
        // Return pointer to scalar pair or null if it doesn't exist.
        // Lookup by dim posn.
        // No non-const version because name shouldn't be modified
        // outside of this class.
        inline const Scalar<T>* getDimPtr(int i) const {
            return (i >= 0 && i < int(_q.size())) ?
                &_q.at(i) : NULL;
        }

        // Return dim name at index (must exist).
        inline const std::string& getDimName(int i) const {
            auto* p = getDimPtr(i);
            assert(p);
            return p->getName();
        }
        
        // Return pointer to value or null if it doesn't exist.
        // Lookup by dim posn.
        inline const T* lookup(int i) const {
            return (i >= 0 && i < int(_q.size())) ?
                _q.at(i).getValPtr() : NULL;
        }
        inline T* lookup(int i) {
            return (i >= 0 && i < int(_q.size())) ?
                _q.at(i).getValPtr() : NULL;
        }

        // Return scalar pair at index (must exist).
        inline const Scalar<T>& getDim(int i) const {
            auto* p = getDimPtr(i);
            assert(p);
            return *p;
        }
        inline const Scalar<T>& operator()(int i) const {
            return getDim(i);
        }

        // Lookup and return value by dim posn (must exist).
        inline const T& getVal(int i) const {
            auto* p = lookup(i);
            assert(p);
            return *p;
        }
        inline T& getVal(int i) {
            auto* p = lookup(i);
            assert(p);
            return *p;
        }
        inline const T& operator[](int i) const {
            return getVal(i);
        }
        inline T& operator[](int i) {
            return getVal(i);
        }

        ////// Methods to get things by name.

        // Return dim posn or -1 if it doesn't exist.
        // Lookup by name.
        inline int lookup_posn(const std::string& dim) const {

            // Get a pointer from the pool for this dim, adding
            // it if needed.
            auto* dp = Scalar<T>::_getPoolPtr(dim)->c_str();

            // First, just check pointers.
            for (size_t i = 0; i < _q.size(); i++) {
                auto& s = _q[i];
                auto* sp = s._getCStr();

                // If pointers match, must the same string.
                if (sp == dp)
                    return int(i);
            }

            // If not found, it could be an indentical string from
            // another pool.
            for (size_t i = 0; i < _q.size(); i++) {
                auto& s = _q[i];
                auto* sp = s._getCStr();

                // Check for match of value.
                if (strcmp(sp, dp) == 0)
                    return int(i);
            }
            return -1;
        }

        // Return scalar pair by name (must exist).
        inline const Scalar<T>& getDim(const std::string& dim) const {
            int i = lookup_posn(dim);
            assert(i >= 0);
            return _q.at(i);
        }
        
        // Return pointer to value or null if it doesn't exist.
        // Lookup by name.
        inline const T* lookup(const std::string& dim) const {
            int i = lookup_posn(dim);
            return (i >= 0) ? &_q[i]._val : NULL;
        }
        inline T* lookup(const std::string& dim) {
            int i = lookup_posn(dim);
            return (i >= 0) ? &_q[i]._val : NULL;
        }

        // Lookup and return value by dim name (must exist).
        inline const T& getVal(const std::string& dim) const {
            auto* p = lookup(dim);
            assert(p);
            return *p;
        }
        inline T& getVal(const std::string& dim) {
            auto* p = lookup(dim);
            assert(p);
            return *p;
        }
        inline const T& operator[](const std::string& dim) const {
            return getVal(dim);
        }
        inline T& operator[](const std::string& dim) {
            return getVal(dim);
        }

        ////// Other methods.

        // Add dimension to tuple (or update if it exists).
        void addDimBack(const std::string& dim, const T& val) {
            auto* p = lookup(dim);
            if (p)
                *p = val;
            else {
                Scalar<T> sv(dim, val);
                _q.push_back(sv);
            }
        }
        void addDimBack(const Scalar<T>& sc) {
            addDimBack(sc.getName(), sc.getVal());
        }
        void addDimFront(const std::string& dim, const T& val) {
            auto* p = lookup(dim);
            if (p)
                *p = val;
            else {
                Scalar<T> sv(dim, val);
                _q.insert(_q.begin(), sv);
            }
        }
        void addDimFront(const Scalar<T>& sc) {
            addDimFront(sc.getName(), sc.getVal());
        }
        
        // Set value by dim posn (posn i must exist).
        inline void setVal(int i, const T& val) {
            T* p = lookup(i);
            assert(p);
            *p = val;
        }
        
        // Set value by dim name (dim must already exist).
        inline void setVal(const std::string& dim, const T& val) {
            T* p = lookup(dim);
            assert(p);
            *p = val;
        }

        // Set multiple values.  Assumes values are in same order as
        // existing names.  If there are more values in 'vals' than 'this',
        // extra values are ignored.  If there are fewer values in 'vals'
        // than 'this', only the number of values supplied will be updated.
        inline void setVals(int numVals, const T vals[]) {
            int end = int(std::min(numVals, size()));
            for (int i = 0; i < end; i++)
                setVal(i, vals[i]);
        }
        inline void setVals(const std::vector<T>& vals) {
            int end = int(std::min(vals.size(), _q.size()));
            for (int i = 0; i < end; i++)
                setVal(i, vals.at(i));
        }
        inline void setVals(const std::deque<T>& vals) {
            int end = int(std::min(vals.size(), _q.size()));
            for (int i = 0; i < end; i++)
                setVal(i, vals.at(i));
        }

        // Set all values to the same value.
        inline void setValsSame(const T& val) {
            for (auto& i : _q)
                i.setVal(val);
        }
    
        // Set values from 'src' Tuple, leaving non-matching ones in this
        // unchanged. Add dimensions in 'src' that are not in 'this' iff
        // addMissing==true.
        // Different than the copy operator because this method does
        // not change the order of *this or remove any existing dims.
        inline void setVals(const Tuple& src, bool addMissing) {
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
        // Basically like setVals(rhs, true), but makes a new Tuple.
        inline Tuple makeUnionWith(const Tuple& rhs) const {
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
    
        // Check whether dims are the same.
        // Don't have to be in same order unless 'sameOrder' is true.
        inline bool areDimsSame(const Tuple& rhs, bool sameOrder = false) const {
            if (size() != rhs.size())
                return false;

            // Dims must be in same order.
            if (sameOrder) {
                for (size_t i = 0; i < _q.size(); i++) {
                    auto* p = _q[i]._getCStr();
                    auto* rp = rhs._q[i]._getCStr();
                    if (p != rp && strcmp(p, rp) != 0)
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
    
        // Equality is true if all dimensions and values are same.
        // Dimensions must be in same order.
        inline bool operator==(const Tuple& rhs) const {

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

        // Less-than is true if first value that is different
        // from corresponding value in 'rhs' is less than it.
        // If all values are same, compares dims.
        inline bool operator<(const Tuple& rhs) const {
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
                auto* p = _q[i]._getCStr();
                auto* rp = rhs._q[i]._getCStr();
                auto c = strcmp(p, rp);
                if (c < 0)
                    return true;
                else if (c > 0)
                    return false;
            }
            return false;
        }

        // Other comparisons derived from above.
        inline bool operator!=(const Tuple& rhs) const {
            return !((*this) == rhs);
        }
        inline bool operator <=(const Tuple& rhs) const {
            return ((*this) == rhs) || ((*this) < rhs);
        }
        inline bool operator>(const Tuple& rhs) const {
            return !((*this) <= rhs);
        }
        inline bool operator >=(const Tuple& rhs) const {
            return !((*this) < rhs);
        }

        // Convert nD 'offsets' to 1D offset using values in 'this' as sizes of nD space.
        // If 'strictRhs', RHS dims must be same and in same order as this;
        // else, only matching ones are considered and missing offsets are zero (0).
        // If '_firstInner', first dim varies most quickly; else last dim does.
        inline size_t layout(const Tuple& offsets, bool strictRhs=true) const {
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
                auto& dim = i.getName();
                size_t dsize = size_t(i.getVal());
                assert (dsize >= 0);

                // offset into this dim.
                auto op = strictRhs ? offsets.lookup(di) : offsets.lookup(dim);
                if (strictRhs)
                    assert(op);
                size_t offset = op ? size_t(*op) : 0; // 0 offset default.
                assert(offset >= 0);
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

        // Convert 1D 'offset' to nD offsets using values in 'this' as sizes of nD space.
        inline Tuple unlayout(size_t offset) const {
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
        
        // Create a new Tuple with the given dimension removed.
        // if dim is found, new Tuple will have one fewer dim than 'this'.
        // If dim is not found, it will be a copy of 'this'.
        inline Tuple removeDim(const std::string& dim) const {
            Tuple newt;
            for (auto i : _q) {
                auto& tdim = i.getName();
                auto& val = i.getVal();
                if (dim != tdim)
                    newt.addDimBack(tdim, val);
            }
            return newt;
        }
        inline Tuple removeDim(const Scalar<T>& dir) const {
            return removeDim(dir.getName());
        }

        // Get value from 'this' in same dim as in 'dir'.
        inline T getValInDir(const Scalar<T>& dir) const {
            auto& dim = dir.getName();
            return getVal(dim);
        }

        // Create new Scalar containing only value in given direction.
        inline Scalar<T> getDirInDim(const std::string& dim) const {
            auto* p = lookup(dim);
            assert(p);
            Scalar<T> s(dim, *p);
            return s;
        }
    
        // Determine whether this is inline with t2 along
        // given direction. This means that all values in this
        // are the same as those in t2, ignoring value in dir.
        inline bool isInlineInDir(const Tuple& t2, const Scalar<T>& dir) const {
            assert(areDimsSame(t2));
            auto& dname = dir.getName();
            for (auto i : _q) {
                auto& tdim = i.getName();
                auto& tval = i.getVal();
                auto& val2 = t2.getVal(tdim);

                // if not in given direction, values must be equal.
                if (tdim != dname && tval != val2)
                    return false;
            }
            return true;
        }

        // Determine whether this is 'ahead of' 't2' along
        // given direction in 'dir'.
        inline bool isAheadOfInDir(const Tuple& t2, const Scalar<T>& dir) const {
            assert(areDimsSame(t2));
            auto& dn = dir.getName();
            auto& dv = dir.getVal();
            auto& tv = getVal(dn);
            auto& v2 = t2.getVal(dn);
            return isInlineInDir(t2, dir) &&
                (((dv > 0) && tv > v2) ||  // in front going forward.
                 ((dv < 0) && tv < v2));   // behind going backward.
        }

        // reductions.
        inline T reduce(std::function<T (T lhs, T rhs)> reducer) const {
            T result = 0;
            int n = 0;
            for (auto i : _q) {
                //auto& tdim = i.getName();
                auto& tval = i.getVal();
                if (n == 0)
                    result = tval;
                else
                    result = reducer(result, tval);
                n++;
            }
            return result;
        }
        inline T sum() const {
            return reduce([&](T lhs, T rhs){ return lhs + rhs; });
        }
        inline T product() const {
            return _q.size() ?
                reduce([&](T lhs, T rhs){ return lhs * rhs; }) : 1;
        }
        inline T max() const {
            return reduce([&](T lhs, T rhs){ return std::max(lhs, rhs); });
        }
        inline T min() const {
            return reduce([&](T lhs, T rhs){ return std::min(lhs, rhs); });
        }

        // pair-wise functions.
        // Apply function to each pair, creating a new Tuple.
        // if strictRhs==true, RHS elements must be same as this;
        // else, only matching ones are considered.
        inline Tuple combineElements(std::function<T (T lhs, T rhs)> combiner,
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
        inline Tuple addElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return lhs + rhs; },
                                   rhs, strictRhs);
        }
        inline Tuple subElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return lhs - rhs; },
                                   rhs, strictRhs);
        }
        inline Tuple multElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return lhs * rhs; },
                                   rhs, strictRhs);
        }
        inline Tuple maxElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return std::max(lhs, rhs); },
                                   rhs, strictRhs);
        }
        inline Tuple minElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return std::min(lhs, rhs); },
                                   rhs, strictRhs);
        }

        // Apply func to each element, creating a new Tuple.
        inline Tuple mapElements(std::function<T (T lhs, T rhs)> func,
                                  T rhs) const {
            Tuple newt = *this;
            for (size_t i = 0; i < _q.size(); i++) {
                auto& tval = _q[i].getVal();
                T newv = func(tval, rhs);
                newt[i] = newv;
            }
            return newt;
        }
        inline Tuple addElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return lhs + rhs; },
                               rhs);
        }
        inline Tuple subElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return lhs - rhs; },
                               rhs);
        }
        inline Tuple multElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return lhs * rhs; },
                               rhs);
        }
        inline Tuple maxElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return std::max(lhs, rhs); },
                               rhs);
        }
        inline Tuple minElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return std::min(lhs, rhs); },
                               rhs);
        }

        // make string like "4x3x2" or "4, 3, 2".
        inline std::string makeValStr(std::string separator=", ",
                                       std::string prefix="",
                                       std::string suffix="") const {
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

        // make string like "x, y, z" or "int x, int y, int z".
        inline std::string makeDimStr(std::string separator=", ",
                                       std::string prefix="",
                                       std::string suffix="") const {
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

        // make string like "x=4, y=3, z=2".
        inline std::string makeDimValStr(std::string separator=", ",
                                          std::string infix="=",
                                          std::string prefix="",
                                          std::string suffix="") const {
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

        // make string like "x+4, y, z-2".
        inline std::string makeDimValOffsetStr(std::string separator=", ",
                                                std::string prefix="",
                                                std::string suffix="") const {
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

        // Call the 'visitor' lambda function at every point in the space defined by 'this'.
        // 'idx' parameter contains sequentially-numbered index.
        // Visitation order is with first dimension in unit stride, i.e., a conceptual
        // "outer loop" iterates through last dimension, ..., and an "inner loop" iterates
        // through first dimension. If '_firstInner' is false, it is done the opposite way.
        // Visitor should return 'true' to keep going or 'false' to stop.
        inline void
        visitAllPoints(std::function<bool (const Tuple&, size_t idx)> visitor) const {

            // Init lambda fn arg with *this to get dim names.
            // Values will get set during scan.
            Tuple tp = *this;

            // 0-D?
            if (!_q.size())
                visitor(tp, 0);
            
            // Call recursive version.
            else if (_firstInner)
                _visitAllPoints(visitor, size()-1, -1, tp);
            else
                _visitAllPoints(visitor, 0, 1, tp);
        }

        // Call the 'visitor' lambda function at every point in the space defined by 'this'.
        // 'idx' parameter contains sequentially-numbered index.
        // Visitation order is not predictable.
        // Visitor return value only stops visit on one thread.
        inline void
        visitAllPointsInParallel(std::function<bool (const Tuple&, size_t idx)> visitor) const {

            // Init lambda fn arg with *this to get dim names.
            // Values will get set during scan.
            Tuple tp = *this;

            // 0-D?
            if (!_q.size())
                visitor(tp, 0);
            
            // Call recursive version.
            else if (_firstInner)
                _visitAllPointsInPar(visitor, size()-1, -1, tp);
            else
                _visitAllPointsInPar(visitor, 0, 1, tp);
        }
    
    protected:

        // Handle recursion for public visitAllPoints(visitor).
        inline bool
        _visitAllPoints(std::function<bool (const Tuple&, size_t idx)> visitor,
                        int curDimNum, int step, Tuple& tp) const {

            auto& sc = _q.at(curDimNum);
            auto& dsize = sc.getVal();
            bool last_dim = curDimNum + step < 0 || curDimNum + step >= size();

            // If no more dims, iterate along current dimension and call
            // visitor.
            if (last_dim) {

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
                    bool ok = _visitAllPoints(visitor, curDimNum + step, step, tp);
                        
                    // Leave if visitor returns false.
                    if (!ok)
                        return false;
                }
            }
            return true;
        }

        // Handle recursion for public visitAllPointsInParallel(visitor).
        inline bool
        _visitAllPointsInPar(std::function<bool (const Tuple&, size_t idx)> visitor,
                             int curDimNum, int step, Tuple& tp) const {

#ifdef _OPENMP
            auto& sc = _q.at(curDimNum);
            auto& dsize = sc.getVal();
            bool first_dim = curDimNum - step < 0 || curDimNum - step >= size();
            bool last_dim = curDimNum + step < 0 || curDimNum + step >= size();

            // If first dim, iterate in parallel w/copies of 'tp'.
            // TODO: collapse parallelism across all but last dim.
            // TODO: provide parallelism for 1-D grids.
            if (first_dim && !last_dim) {

                Tuple tp2(tp);
#pragma omp parallel for firstprivate(tp2)
                for (T i = 0; i < dsize; i++) {
                    tp2.setVal(curDimNum, i);
                    _visitAllPoints(visitor, curDimNum + step, step, tp2);
                }
                return true;
            }

            else
#endif
                return _visitAllPoints(visitor, curDimNum, step, tp);
        }
    };

    // Declare static member.
    template <typename T>
    std::map<std::string, std::string> Scalar<T>::_allNames;
    
} // namespace yask.
