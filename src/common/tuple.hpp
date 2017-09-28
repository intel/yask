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
#include <vector>
#include <deque>
#include <cstdarg>

namespace yask {

    // A shared global pool for names.
    // This saves space and allows us to compare pointers
    // instead of strings.
    extern std::set<std::string> _allNames;
 
    template <typename T>
    class Tuple;
    
    // One named arithmetic value.
    template <typename T>
    class Scalar {
        friend Tuple<T>;
        
    protected:

        // Name and value for this object.
        const char* _name = 0;  // from the _allNames pool.
        std::string _str;       // string made from _name.
        T _val = 0;

        // A shared global pool for names.
        static std::set<std::string> _allNames;

        // Look up names in the pool.
        static const char* _getNamePtr(const std::string& name) {
            auto i = _allNames.insert(name);
            auto& i2 = *i.first;
            return i2.c_str();
        }

    public:
        Scalar(const std::string& name, const T& val) {
            auto* sp = _getNamePtr(name);
            _name = sp;
            _str = name;
            _val = val;
        }
        Scalar(const std::string& name) : Scalar(name, 0) { }
        Scalar() : Scalar("", 0) { }

        // Access name.
        inline const std::string& getName() const {
            return _str;
        }
        inline void setName(const std::string& name) {
            auto* sp = _getNamePtr(name);
            _name = sp;
        }

        // Access value.
        inline const T& getVal() const { return _val; }
        inline T& getVal() { return _val; }
        inline const T* getValPtr() const { return &_val; }
        inline T* getValPtr() { return &_val; }
        inline void setVal(const T& val) { _val = val; }

        // Comparison ops.
        virtual bool operator==(const Scalar& rhs) const {
            return _val == rhs._val &&
                _name == rhs._name;
        }
        virtual bool operator<(const Scalar& rhs) const {
            return (_val < rhs._val) ? true :
                (_val > rhs._val) ? false :
                _name < rhs._name;
        }
    };

    // Collection of named items of one arithmetic type.
    // Can represent:
    // - an n-D space with given sizes.
    // - a point in an n-D space.
    // - a vector from (0,...,0) in n-D space.
    // - values at a point in n-D space.
    // - etc.
    template <typename T>
    class Tuple {
    protected:

        // Dimensions and values for this Tuple.
        std::deque<Scalar<T>> _q;             // dirs and values in order.
        std::map<const char*, int> _map;  // positions in _q keyed by name.

        // First-inner vars control ordering. Example: dims x, y, z.
        // If _firstInner == true, x is unit stride.
        // If _firstInner == false, z is unit stride.
        // This setting affects [un]layout() and visitAllPoints().
        bool _firstInner = true; // whether first dim is used for inner loop.

        // Update _map after values are added to _q.
        void _updateMap() {
            for (size_t i = 0; i < _q.size(); i++)
                _map[_q[i]._name] = int(i);
        }

        // Completely redo the map from scratch.
        void _resetMap() {
            _map.clear();
            _updateMap();
        }

    public:
        Tuple() {}
        virtual ~Tuple() {}

        // first-inner (first dim is unit stride) accessors.
        virtual bool isFirstInner() const { return _firstInner; }
        virtual void setFirstInner(bool fi) { _firstInner = fi; }

        // Query number of dims.
        inline int size() const {
            return int(_q.size());
        }
        inline int getNumDims() const {
            return int(_q.size());
        }

        // Return pointer to scalar pair or null if it doesn't exist.
        // Lookup by dim posn.
        // No non-const version because name shouldn't be modified
        // outside of this class.
        virtual const Scalar<T>* getDimPtr(int i) const {
            return (i >= 0 && i < int(_q.size())) ?
                &_q.at(i) : NULL;
        }

        // Return dim name at index (must exist).
        virtual const std::string& getDimName(int i) const {
            auto* p = getDimPtr(i);
            assert(p);
            return p->getName();
        }
        
        // Return all dim names.
        virtual const std::vector<std::string> getDimNames() const {
            std::vector<std::string> names;
            for (auto& i : _q)
                names.push_back(i.getName());
            return names;
        }
        
        // Return scalar pair at index (must exist).
        virtual const Scalar<T>& getDim(int i) const {
            auto* p = getDimPtr(i);
            assert(p);
            return *p;
        }

        // Return scalar pair by name (must exist).
        virtual const Scalar<T>& getDim(const std::string& dim) const {
            const Scalar<T>* p = 0;
            for (auto& i : _q) {
                if (i.getName() == dim) {
                    p = &i;
                    break;
                }
            }
            assert(p);
            return *p;
        }
        
        // Return pointer to value or null if it doesn't exist.
        // Lookup by dim posn.
        virtual const T* lookup(int i) const {
            return (i >= 0 && i < int(_q.size())) ?
                _q.at(i).getValPtr() : NULL;
        }
        virtual T* lookup(int i) {
            return (i >= 0 && i < int(_q.size())) ?
                _q.at(i).getValPtr() : NULL;
        }

        // Return pointer to value or null if it doesn't exist.
        // Lookup by name.
        virtual const T* lookup(const std::string& dim) const {
            auto* sp = Scalar<T>::_getNamePtr(dim);
            auto i = _map.find(sp);
            return (i == _map.end()) ? NULL : &_q[i->second]._val;
        }
        virtual T* lookup(const std::string& dim) {
            auto* sp = Scalar<T>::_getNamePtr(dim);
            auto i = _map.find(sp);
            return (i == _map.end()) ? NULL : &_q[i->second]._val;
        }
        virtual const T* lookup(const std::string* dim) const {
            return lookup(*dim);
        }
        virtual T* lookup(const std::string* dim) {
            return lookup(*dim);
        }

        // Return dim posn or -1 if it doesn't exist.
        // Lookup by name.
        virtual int lookup_posn(const std::string& dim) const {
            auto* sp = Scalar<T>::_getNamePtr(dim);
            auto i = _map.find(sp);
            return (i == _map.end()) ? -1 : i->second;
        }
        virtual int lookup_posn(const std::string* dim) const {
            return lookup_posn(*dim);
        }

        // Lookup and return value by dim posn (must exist).
        virtual const T& getVal(int i) const {
            auto* p = lookup(i);
            assert(p);
            return *p;
        }
        virtual T& getVal(int i) {
            auto* p = lookup(i);
            assert(p);
            return *p;
        }
        virtual const T& operator[](int i) const {
            return getVal(i);
        }
        virtual T& operator[](int i) {
            return getVal(i);
        }

        // Lookup and return value by dim name (must exist).
        virtual const T& getVal(const std::string& dim) const {
            auto* p = lookup(dim);
            assert(p);
            return *p;
        }
        virtual T& getVal(const std::string& dim) {
            auto* p = lookup(dim);
            assert(p);
            return *p;
        }
        virtual const T& operator[](const std::string& dim) const {
            return getVal(dim);
        }
        virtual T& operator[](const std::string& dim) {
            return getVal(dim);
        }
        virtual const T& getVal(const std::string* dim) const {
            return getVal(*dim);
        }
        virtual T& getVal(const std::string* dim) {
            return getVal(*dim);
        }
        
        // Get iteratable contents.
        const std::deque<Scalar<T>>& getDims() const {
            return _q;
        }

        // Clear data.
        void clear() {
            _q.clear();
            _map.clear();
        }

        // Add dimension to tuple (or update if it exists).
        void addDimBack(const std::string& dim, const T& val) {
            auto* p = lookup(dim);
            if (p)
                *p = val;
            else {
                Scalar<T> sv(dim, val);
                _q.push_back(sv);
                _updateMap();
            }
        }
        void addDimBack(const std::string* dim, const T& val) {
            addDimBack(*dim, val);
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
                _q.push_front(sv);
                _updateMap();
            }
        }
        void addDimFront(const std::string* dim, const T& val) {
            addDimFront(*dim, val);
        }
        void addDimFront(const Scalar<T>& sc) {
            addDimFront(sc.getName(), sc.getVal());
        }
    
        // Set value by dim name (key(s) must already exist).
        virtual void setVal(const std::string& dim, const T& val) {
            T* p = lookup(dim);
            assert(p);
            *p = val;
        }
        virtual void setVal(const std::string* dim, const T& val) {
            setVal(*dim, val);
        }

        // Set value by dim posn (must exist).
        virtual void setVal(int i, const T& val) {
            T* p = lookup(i);
            assert(p);
            *p = val;
        }
        
        // Set multiple values.  Assumes values are in same order as
        // existing names.  If there are more values in 'vals' than 'this',
        // extra values are ignored.  If there are fewer values in 'vals'
        // than 'this', the number of values supplied will be updated.
        virtual void setVals(int numVals, const T vals[]) {
            int end = int(std::min(numVals, size()));
            for (int i = 0; i < end; i++)
                setVal(i, vals[i]);
        }
        virtual void setVals(const std::vector<T>& vals) {
            int end = int(std::min(vals.size(), _q.size()));
            for (int i = 0; i < end; i++)
                setVal(i, vals.at(i));
        }
        virtual void setVals(const std::deque<T>& vals) {
            int end = int(std::min(vals.size(), _q.size()));
            for (int i = 0; i < end; i++)
                setVal(i, vals.at(i));
        }

        // Set all values to the same value.
        virtual void setValsSame(const T& val) {
            for (auto& i : _q)
                i.setVal(val);
        }
    
        // Set values from 'src' Tuple, leaving non-matching ones in this
        // unchanged. Add dimensions in 'src' that are not in 'this' iff
        // addMissing==true.
        // Different than the copy operator because this method does
        // not change the order of *this or remove any existing dims.
        virtual void setVals(const Tuple& src, bool addMissing) {
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
        virtual Tuple makeUnionWith(const Tuple& rhs) const {
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
        virtual bool areDimsSame(const Tuple& rhs, bool sameOrder = false) const {
            if (size() != rhs.size())
                return false;

            // Dims must be in same order.
            if (sameOrder) {
                for (size_t i = 0; i < _q.size(); i++)
                    if (_q[i].getName() != rhs._q[i].getName())
                        return false;
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
        virtual bool operator==(const Tuple& rhs) const {

            // Check dims.
            if (!areDimsSame(rhs, true))
                return false;

            // Check values.
            for (auto& i : _q) {
                auto& dim = i.getName();
                auto& val = i.getVal();
                auto& rval = rhs.getVal(dim);
                if (val != rval)
                    return false;
            }
            return true;
        }

        // Less-than is true if first value that is different
        // from corresponding value in 'rhs' is less than it.
        virtual bool operator<(const Tuple& rhs) const {
            if (size() < rhs.size()) return true;
            else if (size() > rhs.size()) return false;

            // if dims are same, compare values.
            if (areDimsSame(rhs, true)) {
                for (auto i : _q) {
                    auto& dim = i.getName();
                    auto& val = i.getVal();
                    auto& rval = rhs.getVal(dim);
                    if (val < rval)
                        return true;
                    else if (val > rval)
                        return false;
                }
                return false;
            }

            // if dims are not same, compare dims themselves.
            else
                return makeDimStr() < rhs.makeDimStr();
        }

        // Other comparisons derived from above.
        virtual bool operator!=(const Tuple& rhs) const {
            return !((*this) == rhs);
        }
        virtual bool operator <=(const Tuple& rhs) const {
            return ((*this) == rhs) || ((*this) < rhs);
        }
        virtual bool operator>(const Tuple& rhs) const {
            return !((*this) <= rhs);
        }
        virtual bool operator >=(const Tuple& rhs) const {
            return !((*this) < rhs);
        }

        // Convert nD 'offsets' to 1D offset using values in 'this' as sizes of nD space.
        // If 'strictRhs', RHS elements must be same as this;
        // else, only matching ones are considered and missing offsets are zero (0).
        // If '_firstInner', first dim varies most quickly; else last dim does.
        virtual size_t layout(const Tuple& offsets, bool strictRhs=true) const {
            if (strictRhs)
                assert(areDimsSame(offsets));
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
                auto op = offsets.lookup(dim);
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
        virtual Tuple unlayout(size_t offset) const {
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
        
        // Get value from 'this' in same dim as in 'dir'.
        virtual T getValInDir(const Scalar<T>& dir) const {
            auto& dim = dir.getName();
            return getVal(dim);
        }

        // Create new Scalar containing only value in given direction.
        virtual Scalar<T> getDirInDim(const std::string& dim) const {
            auto* p = lookup(dim);
            assert(p);
            Scalar<T> s(dim, *p);
            return s;
        }

        // Create a new Tuple with the given dimension removed.
        // if dim is found, new Tuple will have one fewer dim than 'this'.
        // If dim is not found, it will be a copy of 'this'.
        virtual Tuple removeDim(const std::string& dim) const {
            Tuple newt;
            for (auto i : _q) {
                auto& tdim = i.getName();
                auto& val = i.getVal();
                if (dim != tdim)
                    newt.addDimBack(tdim, val);
            }
            return newt;
        }
        virtual Tuple removeDim(const Scalar<T>& dir) const {
            return removeDim(dir.getName());
        }
    
        // Determine whether this is inline with t2 along
        // given direction. This means that all values in this
        // are the same as those in t2, ignoring value in dir.
        virtual bool isInlineInDir(const Tuple& t2, const Scalar<T>& dir) const {
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
        virtual bool isAheadOfInDir(const Tuple& t2, const Scalar<T>& dir) const {
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
        virtual T reduce(std::function<T (T lhs, T rhs)> reducer) const {
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
        virtual T sum() const {
            return reduce([&](T lhs, T rhs){ return lhs + rhs; });
        }
        virtual T product() const {
            return _q.size() ?
                reduce([&](T lhs, T rhs){ return lhs * rhs; }) : 1;
        }
        virtual T max() const {
            return reduce([&](T lhs, T rhs){ return std::max(lhs, rhs); });
        }
        virtual T min() const {
            return reduce([&](T lhs, T rhs){ return std::min(lhs, rhs); });
        }

        // pair-wise functions.
        // Apply function to each pair, creating a new Tuple.
        // if strictRhs==true, RHS elements must be same as this;
        // else, only matching ones are considered.
        virtual Tuple combineElements(std::function<T (T lhs, T rhs)> combiner,
                                      const Tuple& rhs,
                                      bool strictRhs=true) const {
            if (strictRhs)
                assert(areDimsSame(rhs));
            Tuple newt = *this;
            for (auto& i : _q) {
                auto& tdim = i.getName();
                auto& tval = i.getVal();
                auto* rp = rhs.lookup(tdim);
                if (rp) {
                    T newv = combiner(tval, *rp);
                    newt.setVal(tdim, newv);
                }
            }
            return newt;
        }
        virtual Tuple addElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return lhs + rhs; },
                                   rhs, strictRhs);
        }
        virtual Tuple subElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return lhs - rhs; },
                                   rhs, strictRhs);
        }
        virtual Tuple multElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return lhs * rhs; },
                                   rhs, strictRhs);
        }
        virtual Tuple maxElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return std::max(lhs, rhs); },
                                   rhs, strictRhs);
        }
        virtual Tuple minElements(const Tuple& rhs, bool strictRhs=true) const {
            return combineElements([&](T lhs, T rhs){ return std::min(lhs, rhs); },
                                   rhs, strictRhs);
        }

        // Apply func to each element, creating a new Tuple.
        virtual Tuple mapElements(std::function<T (T lhs, T rhs)> func,
                                  T rhs) const {
            Tuple newt = *this;
            for (auto& i : newt._q) {
                auto& tdim = i.getName();
                auto& tval = i.getVal();
                T newv = func(tval, rhs);
                newt.setVal(tdim, newv);
            }
            return newt;
        }
        virtual Tuple addElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return lhs + rhs; },
                               rhs);
        }
        virtual Tuple subElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return lhs - rhs; },
                               rhs);
        }
        virtual Tuple multElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return lhs * rhs; },
                               rhs);
        }
        virtual Tuple maxElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return std::max(lhs, rhs); },
                               rhs);
        }
        virtual Tuple minElements(T rhs) const {
            return mapElements([&](T lhs, T rhs){ return std::min(lhs, rhs); },
                               rhs);
        }

        // make string like "4x3x2" or "4, 3, 2".
        virtual std::string makeValStr(std::string separator=", ",
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
        virtual std::string makeDimStr(std::string separator=", ",
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
        virtual std::string makeDimValStr(std::string separator=", ",
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
        virtual std::string makeDimValOffsetStr(std::string separator=", ",
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
        // Visitation order is with first dimension in unit stride, i.e., a conceptual
        // "outer loop" iterates through last dimension, ..., and an "inner loop" iterates
        // through first dimension. If '_firstInner' is false, it is done the opposite way.
        // Visitor should return 'true' to keep going or 'false' to stop.
        virtual void visitAllPoints(std::function<bool (const Tuple&)> visitor) const {

            // Init lambda fn arg with *this to get dim names.
            Tuple tp = *this;

            // 0-D?
            if (!_q.size())
                visitor(tp);
            
            // Call recursive version.
            else if (_firstInner)
                visitAllPoints(visitor, size()-1, -1, tp);
            else
                visitAllPoints(visitor, 0, 1, tp);
        }
    
    protected:

        // Handle recursion for public visitAllPoints(visitor).
        virtual bool visitAllPoints(std::function<bool (const Tuple&)> visitor,
                                    int curDimNum, int step, Tuple& tp) const {

            // Ready to call visitor, i.e., have we recursed beyond final dimension?
            if (curDimNum < 0 || curDimNum >= size())
                return visitor(tp);
        
            // Iterate along current dimension, and recurse to next/prev dimension.
            else {
                auto& sc = _q.at(curDimNum);
                auto& tdim = sc.getName();
                auto& dsize = sc.getVal();
                for (T i = 0; i < dsize; i++) {
                    tp.setVal(tdim, i);
                    bool ok = visitAllPoints(visitor, curDimNum + step, step, tp);
                    if (!ok)
                        return false;
                }
            }
            return true;
        }
    };

    // Declare static member.
    template <typename T>
    std::set<std::string> Scalar<T>::_allNames;
    
} // namespace yask.
