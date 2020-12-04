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

#pragma once

namespace yask {

    typedef std::vector<idx_t> VarIndices;
    typedef std::vector<idx_t> VarDimSizes;
    typedef std::vector<std::string> VarDimNames;

    OMP_DECL_TARGET
    
    // Max number of indices that can be held.
    // Note use of "+max_idxs" in code below to avoid compiler
    // trying to take a reference to it, resulting in an undefined
    // symbol (sometimes).
    constexpr int max_idxs = MAX_DIMS;

    // Step dim is always in [0] of an Indices type (if it is used).
    constexpr int step_posn = 0;

    // A class to hold up to a given number of sizes or indices efficiently.
    // Similar to a Tuple, but less overhead and doesn't keep names.
    // This class is NOT virtual.
    // TODO: ultimately, combine with Tuple w/o loss of efficiency.
    class Indices {

    public:

    protected:
        idx_t _idxs[+max_idxs]; // Index values.
        int _ndims;             // Number of indices used.

    public:
        // Ctors.
        Indices() : _ndims(0) { }
        Indices(int ndims) : _ndims(ndims) { } // NB: _idxs remain uninit!
        Indices(const IdxTuple& src) {
            set_from_tuple(src);
        }
        Indices(const VarIndices& src) {
            set_from_vec(src);
        }
        Indices(const std::initializer_list<idx_t>& src) {
            set_from_init_list(src);
        }
        Indices(const idx_t src[], int ndims) {
            set_from_array(src, ndims);
        }
        Indices(idx_t src, int ndims) {
            set_from_const(src, ndims);
        }

        // Default copy ctor, copy operator should be okay.

        // Access size.
        ALWAYS_INLINE int _get_num_dims() const {
            return _ndims;
        }
        ALWAYS_INLINE void set_num_dims(int n) {
            _ndims = n;
        }

        // Access indices.
        ALWAYS_INLINE idx_t& operator[](int i) {
            host_assert(i >= 0);
            host_assert(i < _ndims);
            return _idxs[i];
        }
        ALWAYS_INLINE const idx_t& operator[](int i) const {
            host_assert(i >= 0);
            host_assert(i < _ndims);
            return _idxs[i];
        }

        // Write to an IdxTuple.
        // The 'tgt' must have the same number of dims.
        void set_tuple_vals(IdxTuple& tgt) const {
            host_assert(tgt.size() == size_t(_ndims));
            for (int i = 0; i < _ndims; i++)
                if (size_t(i) < tgt.size())
                    tgt.set_val(i, _idxs[i]);
        }

        // Read from an IdxTuple.
        void set_from_tuple(const IdxTuple& src) {
            host_assert(src.size() <= +max_idxs);
            _ndims = int(src.size());
            for (int i = 0; i < _ndims; i++)
                _idxs[i] = src.get_val(i);
        }

        // Other inits.
        void set_from_vec(const VarIndices& src) {
            host_assert(src.size() <= +max_idxs);
            int n = int(src.size());
            for (int i = 0; i < n; i++)
                _idxs[i] = src[i];
            _ndims = n;
        }

        // default n => don't change _ndims.
        void set_from_array(const idx_t src[], int n = -1) {
            if (n < 0)
                n = _ndims;
            host_assert(n <= +max_idxs);
            for (int i = 0; i < n; i++)
                _idxs[i] = src[i];
            _ndims = n;
        }
        void set_from_init_list(const std::initializer_list<idx_t>& src) {
            host_assert(src.size() <= +max_idxs);
            int i = 0;
            for (auto idx : src)
                _idxs[i++] = idx;
            _ndims = i;
        }

        // default n => don't change _ndims.
        void set_from_const(idx_t val, int n = -1) {
            if (n < 0)
                n = _ndims;
            host_assert(n <= +max_idxs);
            for (int i = 0; i < n; i++)
                _idxs[i] = val;
            _ndims = n;
        }
        void set_vals_same(idx_t val) {
            set_from_const(val);
        }

        // Some comparisons.
        // These assume all the indices are valid or
        // initialized to the same value.
        bool operator==(const Indices& rhs) const {
            if (_ndims != rhs._ndims)
                return false;
            for (int i = 0; i < _ndims; i++)
                if (_idxs[i] != rhs._idxs[i])
                    return false;
            return true;
        }
        bool operator!=(const Indices& rhs) const {
            return !operator==(rhs);
        }
        bool operator<(const Indices& rhs) const {
            if (_ndims < rhs._ndims)
                return true;
            else if (_ndims > rhs._ndims)
                return false;
            for (int i = 0; i < _ndims; i++)
                if (_idxs[i] < rhs._idxs[i])
                    return true;
                else if (_idxs[i] > rhs._idxs[i])
                    return false;
            return false;       // equal, so not less than.
        }
        bool operator>(const Indices& rhs) const {
            if (_ndims > rhs._ndims)
                return true;
            else if (_ndims < rhs._ndims)
                return false;
            for (int i = 0; i < _ndims; i++)
                if (_idxs[i] > rhs._idxs[i])
                    return true;
                else if (_idxs[i] < rhs._idxs[i])
                    return false;
            return false;       // equal, so not greater than.
        }

        // Generic element-wise operator.
        // Returns a new object.
        ALWAYS_INLINE Indices combine_elements(std::function<void (idx_t& lhs, idx_t rhs)> func,
                                       const Indices& other) const {
            Indices res(*this);

#if EXACT_INDICES
            // Use just the used elements.
            for (int i = 0; i < _ndims; i++)
                func(res._idxs[i], other._idxs[i]);
#else
            // Use all to allow unroll and avoid jumps.
            _UNROLL for (int i = 0; i < +max_idxs; i++)
                func(res._idxs[i], other._idxs[i]);
#endif
            return res;
        }

        // Some element-wise operators.
        // These all return a new set of Indices rather
        // than modifying this object.
        ALWAYS_INLINE Indices add_elements(const Indices& other) const {
            return combine_elements([&](idx_t& lhs, idx_t rhs) { lhs += rhs; },
                                   other);
        }
        ALWAYS_INLINE Indices sub_elements(const Indices& other) const {
            return combine_elements([&](idx_t& lhs, idx_t rhs) { lhs -= rhs; },
                                   other);
        }
        ALWAYS_INLINE Indices mul_elements(const Indices& other) const {
            return combine_elements([&](idx_t& lhs, idx_t rhs) { lhs *= rhs; },
                                   other);
        }
        ALWAYS_INLINE Indices min_elements(const Indices& other) const {
            return combine_elements([&](idx_t& lhs, idx_t rhs) { lhs = std::min(lhs, rhs); },
                                   other);
        }
        ALWAYS_INLINE Indices max_elements(const Indices& other) const {
            return combine_elements([&](idx_t& lhs, idx_t rhs) { lhs = std::max(lhs, rhs); },
                                   other);
        }

        // Generic element-wise operator with RHS const.
        // Returns a new object.
        ALWAYS_INLINE
        Indices map_elements(std::function<void (idx_t& lhs, idx_t rhs)> func,
                                   idx_t crhs) const {
            Indices res(*this);

#if EXACT_INDICES
            // Use just the used elements.
            for (int i = 0; i < _ndims; i++)
#else
            // Use all to allow unroll and avoid jumps.
            _UNROLL for (int i = 0; i < +max_idxs; i++)
#endif
                func(res._idxs[i], crhs);
            return res;
        }

        // Operate on all elements.
        ALWAYS_INLINE
        Indices add_const(idx_t crhs) const {
            return map_elements([&](idx_t& lhs, idx_t rhs) { lhs += rhs; },
                               crhs);
        }
        ALWAYS_INLINE
        Indices sub_const(idx_t crhs) const {
            return map_elements([&](idx_t& lhs, idx_t rhs) { lhs -= rhs; },
                               crhs);
        }
        ALWAYS_INLINE
        Indices mul_const(idx_t crhs) const {
            return map_elements([&](idx_t& lhs, idx_t rhs) { lhs *= rhs; },
                               crhs);
        }
        ALWAYS_INLINE
        Indices div_const(idx_t crhs) const {
            return map_elements([&](idx_t& lhs, idx_t rhs) { lhs /= rhs; },
                               crhs);
        }
        ALWAYS_INLINE
        Indices min_const(idx_t crhs) const {
            return map_elements([&](idx_t& lhs, idx_t rhs) { lhs = std::min(lhs, rhs); },
                               crhs);
        }
        ALWAYS_INLINE
        Indices max_const(idx_t crhs) const {
            return map_elements([&](idx_t& lhs, idx_t rhs) { lhs = std::max(lhs, rhs); },
                               crhs);
        }

        // Reduce over all elements.
        ALWAYS_INLINE
        idx_t sum() const {
            idx_t res = 0;
            for (int i = 0; i < _ndims; i++)
                res += _idxs[i];
            return res;
        }
        ALWAYS_INLINE
        idx_t product() const {
            idx_t res = 1;
            for (int i = 0; i < _ndims; i++)
                res *= _idxs[i];
            return res;
        }

        // Convert 1D 'offset' to N-d offsets using values in 'this' as sizes of N-d space.
        // If 'first_inner", '(*this)[0]' is innermost dim (fortran-like),
        // else '(*this)[_ndims-1]' is innermost dim (C-like).
        Indices unlayout(bool first_inner, size_t offset) const {
            Indices res(*this);
            size_t prev_size = 1;

            // Loop thru dims.
            int start_dim = first_inner ? 0 : _ndims-1;
            int stop_dim = first_inner ? _ndims : -1;
            int step_dim = first_inner ? 1 : -1;
            for (int di = start_dim; di != stop_dim; di += step_dim) {
                size_t dsize = size_t(_idxs[di]);
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

        // Advance 'idxs' containing indices in the N-d space defined by
        // 'this' to the next logical index.
        // Input 'idxs' must contain valid indices, i.e., each value must
        // be between 0 and N-1, where N is the value in the corresponding
        // dim in 'this'.
        // If 'idxs' is at last index, "wraps-around" to all zeros.
        // See 'unlayout()' for description of 'first_inner'.
        inline void next_index(bool first_inner, Indices& idxs) const {
            const int inner_dim = first_inner ? 0 : _ndims-1;
            const int dim_step = first_inner ? 1 : -1;

            // Increment inner dim.
            idxs[inner_dim]++;

            // Wrap around indices as needed.
            // First test is redundant, but keeps us from entering loop most times.
            if (idxs[inner_dim] >= _idxs[inner_dim]) {
                for (int j = 0, k = inner_dim; j < _ndims; j++, k += dim_step) {
                        
                    // If too far in dim 'k', set idx to 0 and increment idx in next dim.
                    if (idxs[k] >= _idxs[k]) {
                        idxs[k] = 0;
                        int nxt_dim = k + dim_step;
                        if (nxt_dim >= 0 && nxt_dim < _ndims)
                            idxs[nxt_dim]++;
                    }
                    else
                        break;
                }
            }
        }

        // Call the 'visitor' lambda function at every point sequentially in
        // the N-d space defined by 'this'. At each call, 'idxs' contains
        // next point in N-d space, and 'idx' contains sequentially-numbered
        // 1-d index.
        // Stops and returns 'false' if/when visitor returns 'false'.
        // See 'unlayout()' for description of 'first_inner'.
        bool visit_all_points(bool first_inner,
                              std::function<bool (const Indices& idxs,
                                                  size_t idx)> visitor) const {
            Indices idxs(*this);
            idxs.set_vals_same(0);

            // Total number of points to visit.
            idx_t ne = product();

            // 1 point?
            if (ne <= 1)
                return visitor(idxs, 0);

            // Visit each point in sequential order.
            for (idx_t i = 0; i < ne; i++) {

                // Call visitor.
                bool ok = visitor(idxs, i);
                if (!ok)
                    return false;

                // Jump to next index.
                next_index(first_inner, idxs);
            }
            return true;
        }

        // Same as visit_all_points(), except ranges of points are visited
        // concurrently, and return value from 'visitor' is ignored.
        void visit_all_points_in_parallel(bool first_inner,
                                          std::function<bool (const Indices& idxs,
                                                              size_t idx)> visitor) const {
            // Total number of points to visit.
            idx_t ne = product();

            // 1 point?
            if (ne <= 1) {
                Indices idxs(*this);
                idxs.set_vals_same(0);
                visitor(idxs, 0);
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

                     // Make Indices for this thread.
                     Indices idxs(*this);
                     
                     // Convert 1st linear index to n-dimensional indices.
                     idxs = unlayout(first_inner, start);
                     
                     // Visit each point in sequential order.
                     for (idx_t i = start; i < stop; i++) {
                         
                         // Call visitor.
                         visitor(idxs, i);
                         
                         // Jump to next index.
                         next_index(first_inner, idxs);
                     }
                 });

            #else
            // No OMP; use sequential version.
            visit_all_points(first_inner, visitor);
            #endif
        }

        // Make a Tuple w/given names.
        IdxTuple make_tuple(const VarDimNames& names) const {
            host_assert((int)names.size() == _ndims);

            // Make a Tuple from names.
            IdxTuple tmp;
            for (int i = 0; i < int(names.size()); i++)
                tmp.add_dim_back(names[i], _idxs[i]);
            return tmp;
        }

        // Make a Tuple w/o useful names using values in 'this'.
        IdxTuple make_tuple() const {
            IdxTuple tmp;
            for (int i = 0; i < _ndims; i++)
                tmp.add_dim_back(std::string("d") + std::to_string(i), _idxs[i]);
            return tmp;
        }

        // Make a Tuple w/names from another Tuple using values in 'this'.
        IdxTuple make_tuple(const IdxTuple& names) const {
            auto tmp = names.get_dim_names();
            return make_tuple(tmp);
        }

        // Make string like "x=4, y=8".
        std::string make_dim_val_str(const VarDimNames& names,
                                  std::string separator=", ",
                                  std::string infix="=",
                                  std::string prefix="",
                                  std::string suffix="") const {
            auto tmp = make_tuple(names);
            return tmp.make_dim_val_str(separator, infix, prefix, suffix);
        }
        std::string make_dim_val_str(const IdxTuple& names, // ignore values in 'names'.
                                  std::string separator=", ",
                                  std::string infix="=",
                                  std::string prefix="",
                                  std::string suffix="") const {
            auto tmp = make_tuple(names);
            return tmp.make_dim_val_str(separator, infix, prefix, suffix);
        }

        // Make string like "4, 3, 2".
        std::string make_val_str(std::string separator=", ",
                               std::string prefix="",
                               std::string suffix="") const {

            // Make a Tuple w/o useful names.
            auto tmp = make_tuple();
            return tmp.make_val_str(separator, prefix, suffix);
        }
    };
    static_assert(std::is_trivially_copyable<Indices>::value,
                  "Needed for OpenMP offload");

    // Define OMP reductions on Indices.
    #pragma omp declare reduction(min_idxs : Indices :                  \
                                  omp_out = omp_out.min_elements(omp_in) ) \
        initializer (omp_priv = omp_orig)
    #pragma omp declare reduction(max_idxs : Indices :                  \
                                  omp_out = omp_out.max_elements(omp_in) ) \
        initializer (omp_priv = omp_orig)

    // Layout base class.
    // This class hierarchy is NOT virtual.
    class Layout {

    protected:
        Indices _sizes;   // Size of each dimension.
        Layout(int n, const Indices& sizes) :
            _sizes(sizes) { _sizes.set_num_dims(n); }

    public:
        Layout(int nsizes) :
            _sizes(idx_t(0), nsizes) { }

        // Access sizes.
        ALWAYS_INLINE const Indices& get_sizes() const { return _sizes; }
        void set_sizes(const Indices& sizes) { _sizes = sizes; }
        ALWAYS_INLINE idx_t get_size(int i) const {
            host_assert(i >= 0);
            host_assert(i < _sizes._get_num_dims());
            return _sizes[i];
        }
        void set_size(int i, idx_t size) {
            host_assert(i >= 0);
            host_assert(i < _sizes._get_num_dims());
            _sizes[i] = size;
        }

        // Product of valid sizes.
        ALWAYS_INLINE idx_t get_num_elements() const {
            return _sizes.product();
        }
    };
    static_assert(std::is_trivially_copyable<Layout>::value,
                  "Needed for OpenMP offload");

    // 0-D <-> 1-D layout class.
    // (Trivial layout.)
    class Layout_0d : public Layout {
    public:
        Layout_0d() : Layout(0) { }
        Layout_0d(const Indices& sizes) : Layout(0, sizes) { }
        static constexpr int get_num_sizes() {
            return 0;
        }

        // Return 1-D offset from 0-D 'j' indices.
        ALWAYS_INLINE idx_t layout(const Indices& j) const {
            return 0;
        }

        // Return 0 indices based on 1-D 'ai' input.
        ALWAYS_INLINE Indices unlayout(idx_t ai) const {
            Indices j(idx_t(0), 0);
            return j;
        }
    };
    static_assert(std::is_trivially_copyable<Layout_0d>::value,
                  "Needed for OpenMP offload");

    // Auto-generated layout algorithms for >0 dims.
    #include "yask_layouts.hpp"

    // Forward defns.
    struct Dims;

    // A group of Indices needed for generated loops.
    // See the help message from gen_loops.pl for the
    // documentation of the indices.
    // This class is NOT virtual.
    struct ScanIndices {
        int ndims = 0;

        // Input values; not modified.
        Indices begin, end;     // first and end (beyond last) range of each index.
        Indices stride;         // distance between indices within [begin .. end).
        Indices align;          // alignment of indices after first one.
        Indices align_ofs;      // adjustment for alignment (see below).
        Indices group_size;     // proximity grouping within range.

        // Alignment:
        // First 'start' index is always at 'begin'.
        // Subsequent indices are at 'begin' + 'stride', 'begin' + 2*'stride', etc. if 'align'==1.
        // If 'align'>1, subsequent indices will be aligned such that
        // (('start' - 'align_ofs') % 'align') == 0.
        // Last 'start' index is always < 'end'.
        // Last 'stop' index always == 'end'.

        // Output values; set once for entire range.
        Indices num_indices;    // number of indices in each dim.
        idx_t   linear_indices = 0; // total indices over all dims (product of num_indices).

        // Output values; set for each index by loop code.
        Indices start, stop;    // first and last+1 for this sub-range.
        Indices index;          // 0-based unique index for each sub-range in each dim.
        idx_t   linear_index = 0;   // 0-based index over all dims.

        // Example w/3 sub-ranges in overall range:
        // begin                                         end
        //   |--------------------------------------------|
        //   |------------------|------------------|------|
        // start               stop                            (index = 0)
        //                    start               stop         (index = 1)
        //                                       start   stop  (index = 2)

        // Ctor.
        ScanIndices(const Dims& dims, bool use_vec_align) :
            ndims(NUM_STENCIL_DIMS),
            begin(idx_t(0), ndims),
            end(idx_t(0), ndims),
            stride(idx_t(1), ndims),
            align(idx_t(1), ndims),
            align_ofs(idx_t(0), ndims),
            group_size(idx_t(1), ndims),
            num_indices(idx_t(1), ndims),
            start(idx_t(0), ndims),
            stop(idx_t(0), ndims),
            index(idx_t(0), ndims) {
            
            // Set alignment to vector lengths.
            // i: index for stencil dims, j: index for domain dims.
            if (use_vec_align)
                DOMAIN_VAR_LOOP(i, j)
                    align[i] = fold_pts[j];
        }
        ScanIndices(const Dims& dims, bool use_vec_align, Indices* ofs) :
            ScanIndices(dims, use_vec_align) {
            if (ofs) {
                DOMAIN_VAR_LOOP(i, j) {
                    host_assert(ofs->_get_num_dims() == ndims - 1);
                    align_ofs[i] = (*ofs)[j];
                }
            }
        }
        ScanIndices(const Dims& dims, bool use_vec_align, IdxTuple* ofs) :
            ScanIndices(dims, use_vec_align) {
            if (ofs) {
                DOMAIN_VAR_LOOP(i, j) {
                    host_assert(ofs->_get_num_dims() == ndims - 1);
                    align_ofs[i] = ofs->get_val(j);
                }
            }
        }

        // Default bit-wise copy should be okay.

        // Init from outer-loop indices.
        // Start..stop from point in outer loop become begin..end
        // for this loop.
        //
        // Example:
        // begin              (outer)                    end
        //   |--------------------------------------------|
        //   |------------------|------------------|------|
        // start      |        stop
        //            V
        // begin    (this)     end
        //   |------------------|
        // start               stop  (may be sub-dividied later)
        ALWAYS_INLINE
        void init_from_outer(const ScanIndices& outer) {

            // Begin & end set from start & stop of outer loop.
            begin = start = outer.start;
            end = stop = outer.stop;

            // Pass some values through.
            align = outer.align;
            align_ofs = outer.align_ofs;

            // Leave others alone.
        }
    };
    OMP_END_DECL_TARGET

} // yask namespace.
