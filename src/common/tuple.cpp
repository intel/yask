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

///////// Tuple implementation.

// See tuple.hpp for method documentation.

#include "yask_common_api.hpp"
#include "tuple.hpp"

using namespace std;

namespace yask {

    // Declare static members.
    template <typename T>
    std::list<std::string> Scalar<T>::_all_names;

    // Implementations.

    template <typename T>
    const std::string* Scalar<T>::_get_pool_ptr(const std::string& name) {
        const std::string* p = 0;

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            // Look for existing entry.
            for (auto& i : _all_names) {
                if (i == name) {
                    p = &i;
                    break;
                }
            }

            // If not found, insert.
            if (!p) {
                _all_names.push_back(name);
                auto& li = _all_names.back();
                p = &li;
            }
        }
        return p;
    }
    
    template <typename T>
    const std::vector<std::string> Tuple<T>::get_dim_names() const {
        std::vector<std::string> names;
        for (auto& i : _q)
            names.push_back(i._get_name());
        return names;
    }

    template <typename T>
    void Tuple<T>::add_dim_back(const std::string& dim, const T& val) {
        auto* p = lookup(dim);
        if (p)
            *p = val;
        else {
            Scalar<T> sv(dim, val);
            _q.push_back(sv);
        }
    }
    template <typename T>
    void Tuple<T>::add_dim_front(const std::string& dim, const T& val) {
        auto* p = lookup(dim);
        if (p)
            *p = val;
        else {
            Scalar<T> sv(dim, val);
            _q.insert(_q.begin(), sv);
        }
    }

    template <typename T>
    void Tuple<T>::set_vals(const Tuple& src, bool add_missing) {
        for (auto& i : src.get_dims()) {
            auto& dim = i._get_name();
            auto& val = i.get_val();
            auto* p = lookup(dim);
            if (p)
                *p = val;
            else if (add_missing)
                add_dim_back(dim, val);
        }
    }

    template <typename T>
    Tuple<T> Tuple<T>::make_union_with(const Tuple& rhs) const {
        Tuple u = *this;    // copy.
        for (auto& i : rhs._q) {
            auto& dim = i._get_name();
            auto& val = i.get_val();
            auto* p = u.lookup(dim);
            if (!p)
                u.add_dim_back(dim, val);
        }
        return u;
    }

    template <typename T>
    bool Tuple<T>::are_dims_same(const Tuple& rhs, bool same_order) const {
        if (size() != rhs.size())
            return false;

        // Dims must be in same order.
        if (same_order) {
            for (size_t i = 0; i < _q.size(); i++) {
                auto& n = _q[i]._get_name();
                auto& rn = rhs._q[i]._get_name();
                if (n != rn)
                    return false;
            }
        }

        // Dims can be in any order.
        else {
            for (auto& i : _q) {
                auto& dim = i._get_name();
                if (!rhs.lookup(dim))
                    return false;
            }
        }
        return true;
    }

    // Returns true only if all dims and values are same.
    template <typename T>
    bool Tuple<T>::operator==(const Tuple& rhs) const {

        // Check dims.
        if (!are_dims_same(rhs, true))
            return false;

        // Check values.
        for (size_t i = 0; i < _q.size(); i++) {
            if (get_val(i) != rhs.get_val(i))
                return false;
        }
        return true;
    }

    // Not necessarily a meaningful less-than operator, but
    // works for ordering sets, map keys, etc.
    template <typename T>
    bool Tuple<T>::operator<(const Tuple& rhs) const {
        if (size() < rhs.size()) return true;
        else if (size() > rhs.size()) return false;

        // compare vals.
        for (size_t i = 0; i < _q.size(); i++) {
            auto v = get_val(i);
            auto rv = rhs.get_val(i);
            if (v < rv)
                return true;
            else if (v > rv)
                return false;
        }

        // compare dims.
        for (size_t i = 0; i < _q.size(); i++) {
            auto& n = _q[i]._get_name();
            auto& rn = rhs._q[i]._get_name();
            if (n < rn)
                return true;
            else if (n > rn)
                return false;
        }
        return false;
    }

    template <typename T>
    size_t Tuple<T>::layout(const Tuple& offsets, bool strict_rhs) const {
        if (strict_rhs)
            assert(are_dims_same(offsets, true));
        size_t idx = 0;
        size_t prev_size = 1;

        // Loop thru dims.
        int start_dim = _first_inner ? 0 : size()-1;
        int end_dim = _first_inner ? size() : -1;
        int step_dim = _first_inner ? 1 : -1;
        for (int di = start_dim; di != end_dim; di += step_dim) {
            auto& i = _q.at(di);
            assert(i.get_val() >= 0);
            size_t dsize = size_t(i.get_val());

            // offset into this dim.
            size_t offset = 0;
            if (strict_rhs) {
                assert(offsets[di] >= 0);
                offset = size_t(offsets[di]);
            } else {
                auto& dim = i._get_name();
                auto* op = offsets.lookup(dim);
                if (op) {
                    assert(*op >= 0);
                    offset = size_t(*op);
                }
            }
            assert(offset < dsize);

            // mult offset by product of previous dims.
            idx += (offset * prev_size);
            assert(idx >= 0);
            assert(idx < size_t(product()));

            prev_size *= dsize;
            assert(prev_size <= size_t(product()));
        }
        return idx;
    }

    template <typename T>
    Tuple<T> Tuple<T>::unlayout(size_t offset) const {
        Tuple res = *this;
        size_t prev_size = 1;

        // Loop thru dims.
        int start_dim = _first_inner ? 0 : size()-1;
        int end_dim = _first_inner ? size() : -1;
        int step_dim = _first_inner ? 1 : -1;
        for (int di = start_dim; di != end_dim; di += step_dim) {
            auto& i = _q.at(di);
            //auto& dim = i._get_name();
            size_t dsize = size_t(i.get_val());
            assert (dsize >= 0);

            // Div offset by product of previous dims.
            size_t dofs = offset / prev_size;

            // Wrap within size of this dim.
            dofs %= dsize;

            // Save in result.
            res[di] = dofs;

            prev_size *= dsize;
            assert(prev_size <= size_t(product()));
        }
        return res;
    }

    template <typename T>
    Tuple<T> Tuple<T>::remove_dim(int posn) const {

        // For some reason, copying *this and erasing
        // the element in newt._q causes an exception.
        Tuple newt;
        for (int i = 0; i < _get_num_dims(); i++) {
            if (i != posn)
                newt.add_dim_back(get_dim_name(i), get_val(i));
        }
        return newt;
    }

    template <typename T>
    std::string Tuple<T>::make_val_str(std::string separator,
                                     std::string prefix,
                                     std::string suffix) const {
        std::ostringstream oss;
        int n = 0;
        for (auto i : _q) {
            //auto& tdim = i._get_name();
            auto& tval = i.get_val();
            if (n) oss << separator;
            oss << prefix << tval << suffix;
            n++;
        }
        return oss.str();
    }

    template <typename T>
    std::string Tuple<T>::make_dim_str(std::string separator,
                                     std::string prefix,
                                     std::string suffix) const {
        std::ostringstream oss;
        int n = 0;
        for (auto i : _q) {
            auto& tdim = i._get_name();
            //auto& tval = i.get_val();
            if (n) oss << separator;
            oss << prefix << tdim << suffix;
            n++;
        }
        return oss.str();
    }

    template <typename T>
    std::string Tuple<T>::make_dim_val_str(std::string separator,
                                        std::string infix,
                                        std::string prefix,
                                        std::string suffix) const {
        std::ostringstream oss;
        int n = 0;
        for (auto i : _q) {
            auto& tdim = i._get_name();
            auto& tval = i.get_val();
            if (n) oss << separator;
            oss << prefix << tdim << infix << tval << suffix;
            n++;
        }
        return oss.str();
    }

    template <typename T>
    std::string Tuple<T>::make_dim_val_offset_str(std::string separator,
                                              std::string prefix,
                                              std::string suffix) const {
        std::ostringstream oss;
        int n = 0;
        for (auto i : _q) {
            auto& tdim = i._get_name();
            auto& tval = i.get_val();
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

    // Return a "compact" set of K factors of N.
    template <typename T>
    Tuple<T> Tuple<T>::get_compact_factors(idx_t N) const {
        int K = _get_num_dims();
        
        // Keep track of "best" result, where the best is most compact.
        Tuple best;

        // Trivial cases.
        if (K == 0)
            return best;        // empty tuple.
        if (N == 0) {
            best = *this;
            best.set_vals_same(0); // tuple of all 0s.
            return best;
        }
        if (product() == N)
            return *this;       // already done.

        // Make list of factors of N.
        vector<idx_t> facts;
        for (idx_t n = 1; n <= N; n++)
            if (N % n == 0)
                facts.push_back(n);

        // Try once with keeping pre-set values and then without.
        for (bool keep : { true, false }) {
        
            // Try every combo of K-1 factors.
            // TODO: make more efficient--need algorithm to directly get
            // set of K factors that are valid.
            Tuple combos;
            for (int j = 0; j < K; j++) {
                auto& dname = get_dim_name(j);
                auto dval = get_val(j);

                // Number of factors.
                auto sz = facts.size();

                // Set first number of options 1 because it will be calculated
                // based on the other values, i.e., we don't need to search over
                // first dim.  Also don't need to search any specified value.
                if (j == 0 || (keep && dval > 0))
                    sz = 1;
                
                combos.add_dim_back(dname, sz);
            }
            combos.visit_all_points
                ([&](const Tuple& combo, size_t idx)->bool {

                     // Make candidate tuple w/factors at given indices.
                     auto can = combo.map_elements([&](T in) {
                                                      return facts.at(in);
                                                  });

                     // Override with specified values.
                     for (int j = 0; j < K; j++) {
                         auto dval = get_val(j);
                         if (keep && dval > 0)
                             can[j] = dval;
                         else if (j == 0)
                             can[j] = -1; // -1 => needs to be calculated.
                     }

                     // Replace first factor with computed value if not set.
                     if (can[0] == -1) {
                         can[0] = 1; // to calculate product of remaining ones.
                         can[0] = N / can.product();
                     }

                     // Valid?
                     if (can.product() == N) {

                         // Best so far?
                         // Layout is better if max size is smaller.
                         if (best.size() == 0 ||
                             can.max() < best.max())
                             best = can;
                     }
                     
                     return true; // keep looking.
                 });

            if (best.product() == N)
                break;          // done.

        } // keep or not.
        assert(best.size() == K);
        assert(best.product() == N);
        return best;
    }
    
    // Explicitly allowed instantiations.
    template class Scalar<int>;
    template class Scalar<idx_t>;
    template class Tuple<int>;
    template class Tuple<idx_t>;

} // namespace yask.
