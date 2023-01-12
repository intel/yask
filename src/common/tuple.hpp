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
        static std::list<std::string> _all_names;

        // Look up names in the pool.
        // Should only need to do this when we're adding a new dim.
        static const std::string* _get_pool_ptr(const std::string& name);

        // Name and value for this object.
        const std::string* _namep = 0; // Ptr from the _all_names pool.
        T _val = 0;

        Scalar(const std::string* namep, const T& val) {
            _namep = namep;
            _val = val;
        }

    public:
        Scalar(const std::string& name, const T& val) {
            assert(name.length() > 0);
            _namep = _get_pool_ptr(name);
            _val = val;
        }
        Scalar(const std::string& name) : Scalar(name, 0) { }
        ~Scalar() { }

        // Access name.
        // (Changing it is not allowed.)
        const std::string& _get_name() const {
            return *_namep;
        }
        const std::string* get_name_ptr() const {
            return _namep;
        }

        // Access value.
        const T& get_val() const { return _val; }
        T& get_val() { return _val; }
        const T* get_val_ptr() const { return &_val; }
        T* get_val_ptr() { return &_val; }
        void set_val(const T& val) { _val = val; }

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
        // If _first_inner == true, x is unit stride (col major, like fortran).
        // If _first_inner == false, z is unit stride (row major, like C).
        // This setting affects [un]layout() and visit_all_points().
        bool _first_inner = true; // whether first dim is used for inner loop.

    public:
        Tuple() {}
        ~Tuple() {}             // NOT a virtual class!

        // first-inner (first dim is unit stride) accessors.
        bool is_first_inner() const { return _first_inner; }
        void set_first_inner(bool fi) { _first_inner = fi; }

        // Query number of dims.
        size_t size() const {
            return _q.size();
        }
        int get_num_dims() const {
            return int(_q.size());
        }

        // Return all dim names.
        const string_vec get_dim_names() const;

        // Get iteratable contents.
        const std::vector<Scalar<T>>& get_dims() const {
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
        const Scalar<T>* get_dim_ptr(int i) const {
            return (i >= 0 && i < int(_q.size())) ?
                &_q.at(i) : NULL;
        }

        // Return dim name at index (must exist).
        const std::string& get_dim_name(int i) const {
            auto* p = get_dim_ptr(i);
            assert(p);
            return p->_get_name();
        }

        // Return pointer to value or null if it doesn't exist.
        // Lookup by dim posn.
        const T* lookup(int i) const {
            return (i >= 0 && i < int(_q.size())) ?
                _q.at(i).get_val_ptr() : NULL;
        }
        T* lookup(int i) {
            return (i >= 0 && i < int(_q.size())) ?
                _q.at(i).get_val_ptr() : NULL;
        }

        // Return scalar pair at index (must exist).
        const Scalar<T>& get_dim(int i) const {
            auto* p = get_dim_ptr(i);
            assert(p);
            return *p;
        }
        const Scalar<T>& operator()(int i) const {
            return get_dim(i);
        }

        // Lookup and return value by dim posn (must exist).
        const T& get_val(int i) const {
            auto* p = lookup(i);
            assert(p);
            return *p;
        }
        T& get_val(int i) {
            auto* p = lookup(i);
            assert(p);
            return *p;
        }
        const T& operator[](int i) const {
            return get_val(i);
        }
        T& operator[](int i) {
            return get_val(i);
        }

        // Return values in range of dim posns (all must exist).
        std::vector<T> get_vals(int start, int num) const {
            std::vector<T> res(num, 0);
            for (int i = 0; i < num; i++)
                res[i] = get_val(start + i);
            return res;
        }

        ////// Methods to get things by name.

        // Return dim posn or -1 if it doesn't exist.
        // Lookup by name.
        int lookup_posn(const std::string& dim) const {

            // First check pointers.
            for (size_t i = 0; i < _q.size(); i++) {
                auto& s = _q[i];
                if (s.get_name_ptr() == &dim)
                    return int(i);
            }

            // Then check full strings.
            for (size_t i = 0; i < _q.size(); i++) {
                auto& s = _q[i];
                if (s._get_name() == dim)
                    return int(i);
            }
            return -1;
        }

        // Return scalar pair by name (must exist).
        const Scalar<T>& get_dim(const std::string& dim) const {
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
        const T& get_val(const std::string& dim) const {
            auto* p = lookup(dim);
            assert(p);
            return *p;
        }
        T& get_val(const std::string& dim) {
            auto* p = lookup(dim);
            assert(p);
            return *p;
        }
        const T& operator[](const std::string& dim) const {
            return get_val(dim);
        }
        T& operator[](const std::string& dim) {
            return get_val(dim);
        }

        ////// Other methods.

        // Add dimension to tuple (or update if it exists).
        void add_dim_back(const std::string& dim, const T& val);
        void add_dim_back(const Scalar<T>& sc) {
            add_dim_back(sc._get_name(), sc.get_val());
        }
        void add_dim_at(int posn, const std::string& dim, const T& val);
        void add_dim_at(int posn, const Scalar<T>& sc) {
            add_dim_at(posn, sc._get_name(), sc.get_val());
        }
        void add_dim_front(const std::string& dim, const T& val) {
            add_dim_at(0, dim, val);
        }
        void add_dim_front(const Scalar<T>& sc) {
            add_dim_at(0, sc._get_name(), sc.get_val());
        }

        // Set value by dim posn (posn i must exist).
        void set_val(int i, const T& val) {
            T* p = lookup(i);
            assert(p);
            *p = val;
        }

        // Set value by dim name (dim must already exist).
        void set_val(const std::string& dim, const T& val) {
            T* p = lookup(dim);
            assert(p);
            *p = val;
        }

        // Set multiple values.  Assumes values are in same order as
        // existing names.  If there are more values in 'vals' than 'this',
        // extra values are ignored.  If there are fewer values in 'vals'
        // than 'this', only the number of values supplied will be updated.
        void set_vals(int num_vals, const T vals[]) {
            int end = std::min(num_vals, int(_q.size()));
            for (int i = 0; i < end; i++)
                set_val(i, vals[i]);
        }
        void set_vals(const std::vector<T>& vals) {
            int end = int(std::min(vals.size(), _q.size()));
            for (int i = 0; i < end; i++)
                set_val(i, vals.at(i));
        }
        void set_vals(const std::deque<T>& vals) {
            int end = int(std::min(vals.size(), _q.size()));
            for (int i = 0; i < end; i++)
                set_val(i, vals.at(i));
        }

        // Set all values to the same value.
        void set_vals_same(const T& val) {
            for (auto& i : _q)
                i.set_val(val);
        }

        // Set values from 'src' Tuple, leaving non-matching ones in this
        // unchanged. Add dimensions in 'src' that are not in 'this' iff
        // add_missing==true.
        // Different than the copy operator because this method does
        // not change the order of *this or remove any existing dims.
        void set_vals(const Tuple& src, bool add_missing);

        // This version is similar to vprintf.
        // 'args' must have been initialized with va_start
        // and must contain values of of type T2.
        template<typename T2>
        void set_vals(int num_vals, va_list args) {
            assert(size() == num_vals);

            // process the var args.
            for (int i = 0; i < num_vals; i++) {
                T2 n = va_arg(args, T2);
                set_val(i, T(n));
            }
        }
        template<typename T2>
        void set_vals(int num_vals, ...) {

            // pass the var args.
            va_list args;
            va_start(args, num_vals);
            set_vals<T2>(num_vals, args);
            va_end(args);
        }

        // Set values from starting point; positions must exist.
        // Values before start or after size of vals are unchanged.
        void set_vals(int start, const std::vector<T>& vals) {
            int i = 0;
            for (auto v : vals) {
                set_val(i + start, v);
                i++;
            }
        }
        void set_vals(int start, const std::initializer_list<T>& vals) {
            int i = 0;
            for (auto v : vals) {
                set_val(i + start, v);
                i++;
            }
        }

        // Copy 'this', then add dims and values from 'rhs' that are NOT
        // in 'this'. Return resulting union.
        // Similar to set_vals(rhs, true), but does not change existing
        // values and makes a new Tuple.
        Tuple make_union_with(const Tuple& rhs) const;

        // Check whether dims are the same.
        // Don't have to be in same order unless 'same_order' is true.
        bool are_dims_same(const Tuple& rhs, bool same_order = false) const;

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

        // Convert N-d 'offsets' to 1D offset using values in 'this' as sizes of N-d space.
        // If 'strict_rhs', RHS dims must be same and in same order as this;
        // else, only matching ones are considered and missing offsets are zero (0).
        // If '_first_inner', first dim varies most quickly; else last dim does.
        size_t layout(const Tuple& offsets, bool strict_rhs=true) const;

        // Convert 1D 'offset' to N-d offsets using values in 'this' as sizes of N-d space.
        Tuple unlayout(size_t offset) const;

        // Create a new Tuple with the given dimension removed.
        // if dim is found, new Tuple will have one fewer dim than 'this'.
        // If dim is not found, it will be a copy of 'this'.
        Tuple remove_dim(const std::string& dim) const {
            auto p = lookup_posn(dim);
            Tuple newt = remove_dim(p);
            return newt;
        }

        // Create a new Tuple with the given dimension removed.
        Tuple remove_dim(int posn) const;

        // Reductions.
        // Apply 'reducer' to first pair of elements.
        // Then, apply 'reducer' result of that and next element.
        // Repeat for all elements.
        // Returns final value or 0 if no elements.
        inline T reduce(std::function<T (T lhs, T rhs)> reducer) const {
            T result = 0;
            int n = 0;
            for (auto i : _q) {
                auto& tval = i.get_val();
                result = (n == 0) ? tval : reducer(result, tval);
                n++;
            }
            return result;
        }

        // These reducers return 0 if no elements.
        T sum() const {
            return reduce([&](T lhs, T rhs){ return lhs + rhs; });
        }
        T max() const {
            return reduce([&](T lhs, T rhs){ return std::max(lhs, rhs); });
        }
        T min() const {
            return reduce([&](T lhs, T rhs){ return std::min(lhs, rhs); });
        }

        // These reducers return 1 if no elements.
        T product() const {
            return _q.size() ?
                reduce([&](T lhs, T rhs){ return lhs * rhs; }) : 1;
        }

        // Pair-wise functions.
        // Apply function to each pair, creating a new Tuple.
        // if strict_rhs==true, RHS size and element names must be same as this;
        // else, only matching ones (by name) are considered.
        inline Tuple combine_elements(std::function<T (T lhs, T rhs)> combiner,
                                      const Tuple& rhs,
                                      bool strict_rhs=true) const {
            Tuple newt = *this;
            if (strict_rhs) {
                assert(are_dims_same(rhs, true));
                for (size_t i = 0; i < _q.size(); i++) {
                    auto& tval = _q[i].get_val();
                    auto& rval = rhs[i];
                    T newv = combiner(tval, rval);
                    newt[i] = newv;
                }
            }
            else {
                for (auto& i : _q) {
                    auto& tdim = i._get_name();
                    auto& tval = i.get_val();
                    auto* rp = rhs.lookup(tdim);
                    if (rp) {
                        T newv = combiner(tval, *rp);
                        newt.set_val(tdim, newv);
                    }
                }
            }
            return newt;
        }
        Tuple add_elements(const Tuple& rhs, bool strict_rhs=true) const {
            return combine_elements([&](T lhs, T rhs){ return lhs + rhs; },
                                    rhs, strict_rhs);
        }
        Tuple sub_elements(const Tuple& rhs, bool strict_rhs=true) const {
            return combine_elements([&](T lhs, T rhs){ return lhs - rhs; },
                                    rhs, strict_rhs);
        }
        Tuple mult_elements(const Tuple& rhs, bool strict_rhs=true) const {
            return combine_elements([&](T lhs, T rhs){ return lhs * rhs; },
                                    rhs, strict_rhs);
        }
        Tuple max_elements(const Tuple& rhs, bool strict_rhs=true) const {
            return combine_elements([&](T lhs, T rhs){ return std::max(lhs, rhs); },
                                    rhs, strict_rhs);
        }
        Tuple min_elements(const Tuple& rhs, bool strict_rhs=true) const {
            return combine_elements([&](T lhs, T rhs){ return std::min(lhs, rhs); },
                                    rhs, strict_rhs);
        }

        // Apply 'func' to each element and 'rhs', creating a new Tuple.
        Tuple map_elements(std::function<T (T lhs, T rhs)> func,
                           T rhs) const {
            Tuple newt = *this;
            for (size_t i = 0; i < _q.size(); i++) {
                auto& tval = _q[i].get_val();
                T newv = func(tval, rhs);
                newt[i] = newv;
            }
            return newt;
        }
        // Apply 'func' to each element, creating a new Tuple.
        Tuple map_elements(std::function<T (T in)> func) const {
            Tuple newt = *this;
            for (size_t i = 0; i < _q.size(); i++) {
                auto& tval = _q[i].get_val();
                T newv = func(tval);
                newt[i] = newv;
            }
            return newt;
        }
        Tuple add_elements(T rhs) const {
            return map_elements([&](T lhs, T rhs){ return lhs + rhs; },
                                rhs);
        }
        Tuple sub_elements(T rhs) const {
            return map_elements([&](T lhs, T rhs){ return lhs - rhs; },
                                rhs);
        }
        Tuple mult_elements(T rhs) const {
            return map_elements([&](T lhs, T rhs){ return lhs * rhs; },
                                rhs);
        }
        Tuple max_elements(T rhs) const {
            return map_elements([&](T lhs, T rhs){ return std::max(lhs, rhs); },
                                rhs);
        }
        Tuple min_elements(T rhs) const {
            return map_elements([&](T lhs, T rhs){ return std::min(lhs, rhs); },
                                rhs);
        }
        Tuple neg_elements() const {
            return map_elements([&](T in){ return -in; });
        }
        Tuple abs_elements() const {
            return map_elements([&](T in){ return T(llabs(in)); });
        }

        // make string like "4x3x2" or "4, 3, 2".
        std::string make_val_str(std::string separator=", ",
                                 std::string prefix="",
                                 std::string suffix="") const;

        // make string like "x, y, z" or "int x, int y, int z".
        std::string make_dim_str(std::string separator=", ",
                                 std::string prefix="",
                                 std::string suffix="") const;

        // make string like "x=4, y=3, z=2".
        std::string make_dim_val_str(std::string separator=", ",
                                     std::string infix="=",
                                     std::string prefix="",
                                     std::string suffix="") const;

        // make string like "x+4, y, z-2".
        std::string make_dim_val_offset_str(std::string separator=", ",
                                            std::string prefix="",
                                            std::string suffix="") const;

        // Return a "compact" set of K factors of N,
        // a set of factors with largest factor as small as possible,
        // where K is the size of 'this'.
        // Any non-zero numbers in 'this' will be kept if possible.
        Tuple get_compact_factors(T N) const;

        // Advance Tuple 'tp' containing indices in the space defined by
        // 'this' to the next logical index.
        // Input 'tp' must contain valid indices, i.e., each value must
        // be between 0 and N-1, where N is the value in the corresponding
        // dim in 'this'.
        // If 'tp' is at last index, "wraps-around" to all zeros.
        inline void next_index(Tuple& tp) const {
            const int nd = get_num_dims();
            const int inner_dim = _first_inner ? 0 : nd-1;
            const int dim_step = _first_inner ? 1 : -1;

            // Increment inner dim.
            tp[inner_dim]++;

            // Wrap around indices as needed.
            // First test is redundant, but keeps us from entering loop most times.
            if (tp[inner_dim] >= get_val(inner_dim)) {
                for (int j = 0, k = inner_dim; j < nd; j++, k += dim_step) {
                        
                    // If too far in dim 'k', set idx to 0 and increment idx in next dim.
                    if (tp[k] >= get_val(k)) {
                        tp[k] = 0;
                        int nxt_dim = k + dim_step;
                        auto* p = tp.lookup(nxt_dim);
                        if (p)
                            (*p)++;
                    }
                    else
                        break;
                }
            }
        }
        
        // Call the 'visitor' lambda function at every point sequentially in
        // the space defined by 'this'.  'idx' parameter contains
        // sequentially-numbered index.  Visitation order is with first
        // dimension in unit stride, i.e., a conceptual "outer loop"
        // iterates through last dimension, ..., and an "inner loop"
        // iterates through first dimension. If '_first_inner' is false, it
        // is done the opposite way.  Visitor should return 'true' to keep
        // going or 'false' to stop.  Returns 'false' if any visitor
        // returned 'false' and visitation was stopped; otherwise 'true'.
        // Example:
        // sizes_tuple.visit_all_points([&](const Tuple<int>& pt, size_t idx) { ... });
        bool visit_all_points(std::function<bool (const Tuple&,
                                                  size_t idx)> visitor) const {
            Tuple tp(*this);
            tp.set_vals_same(0);

            // Total number of points to visit.
            idx_t ne = product();

            // 1 point?
            if (ne <= 1) {
                bool ok = visitor(tp, 0);
                return ok;
            }

            // Visit each point in sequential order.
            for (T i = 0; i < ne; i++) {

                // Call visitor.
                bool ok = visitor(tp, i);
                if (!ok)
                    return false;

                // Jump to next index.
                next_index(tp);
            }
            return true;
        }

        // Call the 'visitor' lambda function at every point in the space defined by 'this'.
        // 'idx' parameter contains sequentially-numbered index.
        // Visitation order is not predictable.
        // Visitation concurrency is not predicable.
        // Visitor return value is ignored.
        void visit_all_points_in_parallel(std::function<bool (const Tuple&,
                                                              size_t idx)> visitor) const {
            // Total number of points to visit.
            idx_t ne = product();

            // 1 point?
            if (ne <= 1) {
                Tuple tp(*this);
                tp.set_vals_same(0);
                visitor(tp, 0);
                return;
            }

            #ifdef _OPENMP

            // Num threads to be started.
            idx_t nthr = yask_get_num_threads();

            // Start sequential visits in parallel.
            // (Not guaranteed that each tnum will be unique in every OMP
            // impl, so don't rely on it.)
            yask_parallel_for
                (0, nthr, 1,
                 [&](idx_t n, idx_t np1, idx_t tnum) {

                     // Start and stop indices for this thread.
                     idx_t start = div_equally_cumu_size_n(ne, nthr, n - 1);
                     idx_t stop = div_equally_cumu_size_n(ne, nthr, n);
                     assert(stop >= start);
                     if (stop <= start)
                         return; // from lambda.

                     // Make tuple for this thread.
                     Tuple tp = *this;
                     
                     // Convert 1st linear index to n-dimensional tuple.
                     tp = unlayout(start);
                     
                     // Visit each point in sequential order.
                     for (T i = start; i < stop; i++) {
                         
                         // Call visitor.
                         visitor(tp, i);
                         
                         // Jump to next index.
                         next_index(tp);
                     }
                 });

            #else
            // No OMP; use sequential version.
            visit_all_points(visitor);
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
            for (int i = 0; i < x.get_num_dims(); i++) {
                h ^= size_t(i) ^
                    std::hash<T>()(x.get_val(i)) ^
                    std::hash<std::string>()(x.get_dim_name(i));
            }
            return h;
        }
    };
}
