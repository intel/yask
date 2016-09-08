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

///////// Stencil support.

#ifndef TUPLE_HPP
#define TUPLE_HPP

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

using namespace std;

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
    map<string, T> _map;        // contents.
    deque<string> _dims;        // dirs in specified order.

    // First-inner vars control ordering. Example: dims x, y, z.
    // If _firstInner == true, x is unit stride.
    // If _firstInner == false, z is unit stride.
    // This setting affects mapTo1d() and visitAllPoints().
    bool _firstInner;           // whether first dim is used for inner loop.
    static bool _defaultFirstInner; // default for _firstInner.
    
public:

    Tuple() : _firstInner(_defaultFirstInner) { }
    Tuple(const Tuple& rhs) :
        _map(rhs._map), _dims(rhs._dims),
        _firstInner(rhs._firstInner) { }
    virtual ~Tuple() {}

    // Copy.
    virtual void operator=(const Tuple& rhs) {
        _map = rhs._map;
        _dims = rhs._dims;
    }

    // first-inner (first dim is unit stride) accessors.
    virtual bool getFirstInner() const { return _firstInner; }
    virtual void setFirstInner(bool fi) { _firstInner = fi; }
    static bool getDefaultFirstInner() { return _defaultFirstInner; }
    static void setDefaultFirstInner(bool fi) { _defaultFirstInner = fi; }
    
    // Return pointer to value or null if it doesn't exist.
    virtual const T* lookup(const string& dim) const {
        auto i = _map.find(dim);
        return (i == _map.end()) ? NULL : &(i->second);
    }
    virtual T* lookup(const string& dim) {
        auto i = _map.find(dim);
        return (i == _map.end()) ? NULL : &(i->second);
    }

    // Lookup and return value (must exist).
    virtual T getVal(const string& dim) const {
        const T* p = lookup(dim);
        assert(p);
        return *p;
    }

    // Get dimensions.
    const deque<string>& getDims() const {
        return _dims;
    }
    int size() const {
        return int(_dims.size());
    }

    // Add dimension to tuple (must NOT already exist).
    void addDimBack(const string& dim, T val) {
        assert(_map.count(dim) == 0);
        _map[dim] = val;
        _dims.push_back(dim);
    }
    void addDimFront(const string& dim, T val) {
        assert(_map.count(dim) == 0);
        _map[dim] = val;
        _dims.push_front(dim);
    }
    
    // Set value (key(s) must already exist).
    virtual void setVal(const string& dim, T val) {
        T* p = lookup(dim);
        assert(p);
        *p = val;
    }

    // Set multiple values.
    // Assumes values are in same order as keys.
    virtual void setVals(int numVals, const T vals[]) {
        assert(size() == numVals);
        for (int i = 0; i < size(); i++)
            _map[_dims.at(i)] = vals[i];
    }
    virtual void setVals(const vector<T>& vals) {
        assert(size() == (int)vals.size());
        for (int i = 0; i < size(); i++)
            _map[_dims.at(i)] = vals.at(i);
    }
    virtual void setVals(const deque<T>& vals) {
        assert(size() == (int)vals.size());
        for (int i = 0; i < size(); i++)
            _map[_dims.at(i)] = vals.at(i);
    }

    // This version is similar to vprintf.
    // 'args' must have been initialized with va_start.
    virtual void setVals(int numVals, va_list args) {
        assert(size() == numVals);
        
        // process the var args.
        for (int i = 0; i < numVals; i++) {
            int n = va_arg(args, int);
            _map[_dims.at(i)] = n;
        }
    }
    virtual void setVals(int numVals, ...) {
        
        // pass the var args.
        va_list args;
        va_start(args, numVals);
        setVals(numVals, args);
        va_end(args);
    }

    // Check whether dims are the same.
    // (Don't have to be in same order.)
    virtual bool areDimsSame(const Tuple& rhs) const {
        if (size() != rhs.size())
            return false;
        for (auto dim : _dims) {
            if (!rhs.lookup(dim))
                return false;
        }
        return true;
    }
    
    // Comparisons.
    virtual bool operator==(const Tuple& rhs) const {
        if (!areDimsSame(rhs))
            return false;
        for (auto i : _map) {
            auto dim = i.first;
            T lv = i.second;
            T rv = rhs.getVal(dim);
            if (lv != rv)
                return false;
        }
        return true;
    }
    virtual bool operator!=(const Tuple& rhs) const {
        return !operator==(rhs);
    }
    virtual bool operator<(const Tuple& rhs) const {
        if (size() < rhs.size()) return true;
        else if (size() > rhs.size()) return false;
        
        assert(areDimsSame(rhs));
        for (auto i : _map) {
            auto dim = i.first;
            T lv = i.second;
            T rv = rhs.getVal(dim);
            if (lv < rv)
                return true;
            else if (lv > rv)
                return false;
        }
        return false;
    }
    virtual bool operator>(const Tuple& rhs) const {

        // not (less-than or equal).
        return !(((*this) < rhs) || ((*this) == rhs));
    }

    // Convert offsets to 1D offset using C-style map (last index is
    // unit stride) assuming values in this are sizes.
    // if strictRhs==true, RHS elements must be same as this;
    // else, only matching ones are considered.
    virtual size_t mapTo1d(const Tuple& offsets, bool strictRhs=true) const {
        if (strictRhs)
            assert(areDimsSame(offsets));
        int idx = 0;
        int prevSize = 1;

        // Loop thru dims.
        int startDim = _firstInner ? 0 : size()-1;
        int beyondLastDim = _firstInner ? size() : -1;
        int stepDim = _firstInner ? 1 : -1;
        for (int di = startDim; di != beyondLastDim; di += stepDim) {
            auto dim = _dims[di];

            // size of this dim.
            int size = getVal(dim);
            assert (size >= 0);

            // offset into this dim.
            auto op = offsets.lookup(dim);
            int offset = op ? *op : 0; // 0 offset default.
            assert(offset >= 0);
            assert(offset < size);

            // mult offset by product of previous dims.
            idx += (offset * prevSize);
            assert(idx >= 0);
            assert(idx < product());
            
            prevSize *= size;
            assert(prevSize <= product());
        }
        //cerr << "** offsets " << offsets.makeDimValStr() << " in " <<
        //makeDimValStr() << " => " << idx << endl;
        return idx;
    }
    
    // Get value from this in given direction (ignoring sign of dir).
    // Dir must have only one dimension.
    virtual T getValInDir(const Tuple& dir) const {
        assert(dir.size() == 1);
        for (auto i : _map) {
            auto dim = i.first;
            T val = i.second;
            const T* p = dir.lookup(dim);
            if (p)
                return val;
        }
        assert("invalid dir");
        return 0;
    }

    // get name of a direction, which must be
    // 0D or 1D.
    virtual string getDirName() const {
        assert(size() <= 1);
        if (!size())
            return "none";
        return _dims[0];  // first and only key.
    }

    // get value of a direction, which must be 1D.
    virtual T getDirVal() const {
        assert(size() == 1);
        T dv = getVal(_dims[0]);  // first and only value.
        assert(dv != 0);
        return dv;
    }

    // Create a new Tuple with the dimension in dir removed.
    // Thus, retured Tuple will have one fewer dim than this.
    virtual Tuple removeDimInDir(const Tuple& dir) const {
        Tuple newt;
        string dname  = dir.getDirName();
        for (auto dim : _dims) {
            if (dim != dname)
                newt.addDimBack(dim, getVal(dim));
        }
        return newt;
    }
    
    // Determine whether this is inline with t2 along
    // given direction. This means that all values in this
    // are the same as those in t2, ignoring value in dir.
    virtual bool isInlineInDir(const Tuple& t2, const Tuple& dir) const {
        assert(areDimsSame(t2));
        if (dir.size() == 0)
            return false;       // never inline with no direction.
        string dname = dir.getDirName();
        for (auto i : _map) {
            auto dim = i.first;
            T val = i.second;
            T val2 = t2.getVal(dim);

            // if not in given direction, values must be equal.
            if (dim != dname && val != val2)
                return false;
        }
        return true;
    }

    // Determine whether this is 'ahead of' t2 along
    // given direction in dir.
    virtual bool isAheadOfInDir(const Tuple& t2, const Tuple& dir) const {
        string dn = dir.getDirName();
        T dv = dir.getDirVal();
        assert(areDimsSame(t2));
        return isInlineInDir(t2, dir) &&
            (((dv > 0) && getVal(dn) > t2.getVal(dn)) ||  // in front going forward.
             ((dv < 0) && getVal(dn) < t2.getVal(dn)));   // behind going backward.
    }

    // reductions.
    virtual T reduce(function<T (T lhs, T rhs)> reducer) const {
        T result = 0;
        int n = 0;
        for (auto i : _map) {
            T val = i.second;
            if (n == 0)
                result = val;
            else
                result = reducer(result, val);
            n++;
        }
        return result;
    }
    virtual T sum() const {
        return reduce([&](T lhs, T rhs){ return lhs + rhs; });
    }
    virtual T product() const {
        return reduce([&](T lhs, T rhs){ return lhs * rhs; });
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
    virtual Tuple combineElements(function<T (T lhs, T rhs)> combiner,
                                  const Tuple& rhs,
                                  bool strictRhs=true) const {
        if (strictRhs)
            assert(areDimsSame(rhs));
        Tuple newt = *this;
        for (auto dim : _dims) {
            auto rp = rhs.lookup(dim);
            if (rp) {
                T newv = combiner(getVal(dim), *rp);
                newt.setVal(dim, newv);
            }
        }
        return newt;
    }
    virtual Tuple addElements(const Tuple& rhs, bool strictRhs=true) const {
        return combineElements([&](T lhs, T rhs){ return lhs + rhs; },
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

    // make name like "4x3x2" or "4, 3, 2".
    virtual string makeValStr(string separator=", ",
                               string prefix="", string suffix="") const {
        ostringstream oss;
        int n = 0;
        for (auto dim : _dims) {
            if (n) oss << separator;
            oss << prefix << getVal(dim) << suffix;
            n++;
        }
        return oss.str();
    }

    // make name like "int x, int y, int z".
    virtual string makeDimStr(string separator=", ",
                               string prefix="", string suffix="") const {
        ostringstream oss;
        int n = 0;
        for (auto dim : _dims) {
            if (n) oss << separator;
            oss << prefix << dim << suffix;
            n++;
        }
        return oss.str();
    }

    // make name like "x=4, y=3, z=2".
    virtual string makeDimValStr(string separator=", ", string infix="=",
                                 string prefix="", string suffix="") const {
        ostringstream oss;
        int n = 0;
        for (auto dim : _dims) {
            if (n) oss << separator;
            oss << prefix << dim << infix << getVal(dim) << suffix;
            n++;
        }
        return oss.str();
    }

    // make name like "x+4, y, z-2".
    virtual string makeDimValOffsetStr(string separator=", ",
                                       string prefix="", string suffix="") const {
        ostringstream oss;
        int n = 0;
        for (auto dim : _dims) {

            T val = getVal(dim);

            if (n) oss << separator;
            oss << prefix << dim;
            if (val > 0) oss << "+" << val;
            else if (val < 0) oss << val; // includes '-';
            oss << suffix;

            n++;
        }
        return oss.str();
    }

    // make name like "xv+(4/2), yv, zv-(2/2)".
    // this object has numerators; norm object has denominators.
    virtual string makeDimValNormOffsetStr(const Tuple& norm,
                                           string separator=", ",
                                           string prefix="", string suffix="") const {
        ostringstream oss;
        int n = 0;
        for (auto dim : _dims) {

            // index value.
            T val = getVal(dim);

            // normalizer.
            const T* p = norm.lookup(dim);
            T d = p ? *p : 1;

            if (n) oss << separator;
            oss << prefix << dim << "v";
            if (val > 0) oss << "+(" << val << "/" << d << ")";
            else if (val < 0) oss << "-(" << -val << "/" << d << ")";
            oss << suffix;

            n++;
        }
        return oss.str();
    }

    // Call the visitor lambda function at every point in the space defined by this.
    // Visitation order is with first dimension in unit stride, i.e., a conceptual
    // "outer loop" iterates through last dimension, and an "inner loop" iterates
    // through first dimension. If '_firstInner' is false, it is done the opposite way.
    virtual void visitAllPoints(function<void (const Tuple&)> visitor) const {

        // Init lambda fn arg with *this.
        Tuple tp = *this;

        // Call recursive version.
        if (_firstInner)
            visitAllPoints(visitor, size()-1, -1, tp);
        else
            visitAllPoints(visitor, 0, 1, tp);
    }
    
protected:

    // Handle recursion for public visitAllPoints(visitor).
    virtual void visitAllPoints(function<void (const Tuple&)> visitor,
                                int curDimNum, int step, Tuple& tp) const {

        // Ready to call visitor, i.e., have we recursed beyond a dimension?
        if (curDimNum < 0 || curDimNum >= size())
            visitor(tp);
        
        // Iterate along current dimension, and recurse to prev dimension.
        else {
            auto dim = _dims.at(curDimNum);
            T dsize = getVal(dim);
            for (T i = 0; i < dsize; i++) {
                tp.setVal(dim, i);
                visitAllPoints(visitor, curDimNum + step, step, tp);
            }
        }
    }
};

// Default value.
template <typename T>
bool Tuple<T>::_defaultFirstInner = true;

#endif
