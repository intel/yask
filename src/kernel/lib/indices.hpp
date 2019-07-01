/*****************************************************************************

YASK: Yet Another Stencil Kit
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

#pragma once

namespace yask {

    typedef std::vector<idx_t> VarIndices;
    typedef std::vector<idx_t> VarDimSizes;
    typedef std::vector<std::string> VarDimNames;

    // A class to hold up to a given number of sizes or indices efficiently.
    // Similar to a Tuple, but less overhead and doesn't keep names.
    // Make sure this stays non-virtual.
    // TODO: make this a template with _ndims as a parameter.
    // TODO: ultimately, combine with Tuple w/o loss of efficiency.
    class Indices {

    public:

        // Max number of indices that can be held.
        // Note use of "+max_idxs" in code below to avoid compiler
        // trying to take a reference to it, resulting in an undefined
        // symbol (sometimes).
        static constexpr int max_idxs = MAX_DIMS;

        // Step dim is always in [0] of an Indices type (if it is used).
        static constexpr int step_posn = 0;

    protected:
        idx_t _idxs[max_idxs];
        int _ndims;

    public:
        // Ctors.
        Indices() : _ndims(0) { }
        Indices(int ndims) : _ndims(ndims) { } // NB: _idxs remain uninit!
        Indices(const IdxTuple& src) {
            setFromTuple(src);
        }
        Indices(const VarIndices& src) {
            setFromVec(src);
        }
        Indices(const std::initializer_list<idx_t>& src) {
            setFromInitList(src);
        }
        Indices(const idx_t src[], int ndims) {
            setFromArray(src, ndims);
        }
        Indices(idx_t src, int ndims) {
            setFromConst(src, ndims);
        }

        // Default copy ctor, copy operator should be okay.

        // Access size.
        inline int getNumDims() const {
            return _ndims;
        }
        inline void setNumDims(int n) {
            _ndims = n;
        }

        // Access indices.
        inline idx_t& operator[](int i) {
            assert(i >= 0);
            assert(i < _ndims);
            return _idxs[i];
        }
        inline const idx_t& operator[](int i) const {
            assert(i >= 0);
            assert(i < _ndims);
            return _idxs[i];
        }

        // Write to an IdxTuple.
        // The 'tgt' must have the same number of dims.
        void setTupleVals(IdxTuple& tgt) const {
            assert(tgt.size() == size_t(_ndims));
            for (int i = 0; i < _ndims; i++)
                if (size_t(i) < tgt.size())
                    tgt.setVal(i, _idxs[i]);
        }

        // Read from an IdxTuple.
        void setFromTuple(const IdxTuple& src) {
            assert(src.size() <= +max_idxs);
            int n = int(src.size());
            for (int i = 0; i < n; i++)
                _idxs[i] = src.getVal(i);
            _ndims = n;
        }

        // Other inits.
        void setFromVec(const VarIndices& src) {
            assert(src.size() <= +max_idxs);
            int n = int(src.size());
            for (int i = 0; i < n; i++)
                _idxs[i] = src[i];
            _ndims = n;
        }

        // default n => don't change _ndims.
        void setFromArray(const idx_t src[], int n = -1) {
            if (n < 0)
                n = _ndims;
            assert(n <= +max_idxs);
            for (int i = 0; i < n; i++)
                _idxs[i] = src[i];
            _ndims = n;
        }
        void setFromInitList(const std::initializer_list<idx_t>& src) {
            assert(src.size() <= +max_idxs);
            int i = 0;
            for (auto idx : src)
                _idxs[i++] = idx;
            _ndims = i;
        }

        // default n => don't change _ndims.
        void setFromConst(idx_t val, int n = -1) {
            if (n < 0)
                n = _ndims;
            assert(n <= +max_idxs);
            for (int i = 0; i < n; i++)
                _idxs[i] = val;
            _ndims = n;
        }
        void setValsSame(idx_t val) {
            setFromConst(val);
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
        inline Indices combineElements(std::function<void (idx_t& lhs, idx_t rhs)> func,
                                       const Indices& other) const {
            Indices res(*this);

#if EXACT_INDICES
            // Use just the used elements.
            for (int i = 0; i < _ndims; i++)
#else
            // Use all to allow unroll and avoid jumps.
#pragma unroll
            for (int i = 0; i < max_idxs; i++)
#endif
                func(res._idxs[i], other._idxs[i]);
            return res;
        }

        // Some element-wise operators.
        // These all return a new set of Indices rather
        // than modifying this object.
        inline Indices addElements(const Indices& other) const {
            return combineElements([&](idx_t& lhs, idx_t rhs) { lhs += rhs; },
                                   other);
        }
        inline Indices subElements(const Indices& other) const {
            return combineElements([&](idx_t& lhs, idx_t rhs) { lhs -= rhs; },
                                   other);
        }
        inline Indices mulElements(const Indices& other) const {
            return combineElements([&](idx_t& lhs, idx_t rhs) { lhs *= rhs; },
                                   other);
        }
        inline Indices divElements(const Indices& other) const {
            return combineElements([&](idx_t& lhs, idx_t rhs) { lhs /= rhs; },
                                   other);
        }
        inline Indices minElements(const Indices& other) const {
            return combineElements([&](idx_t& lhs, idx_t rhs) { lhs = std::min(lhs, rhs); },
                                   other);
        }
        inline Indices maxElements(const Indices& other) const {
            return combineElements([&](idx_t& lhs, idx_t rhs) { lhs = std::max(lhs, rhs); },
                                   other);
        }

        // Generic element-wise operator with RHS const.
        // Returns a new object.
        inline Indices mapElements(std::function<void (idx_t& lhs, idx_t rhs)> func,
                                   idx_t crhs) const {
            Indices res(*this);

#if EXACT_INDICES
            // Use just the used elements.
            for (int i = 0; i < _ndims; i++)
#else
            // Use all to allow unroll and avoid jumps.
#pragma unroll
            for (int i = 0; i < max_idxs; i++)
#endif
                func(res._idxs[i], crhs);
            return res;
        }

        // Operate on all elements.
        Indices addConst(idx_t crhs) const {
            return mapElements([&](idx_t& lhs, idx_t rhs) { lhs += rhs; },
                               crhs);
        }
        Indices subConst(idx_t crhs) const {
            return mapElements([&](idx_t& lhs, idx_t rhs) { lhs -= rhs; },
                               crhs);
        }
        Indices mulConst(idx_t crhs) const {
            return mapElements([&](idx_t& lhs, idx_t rhs) { lhs *= rhs; },
                               crhs);
        }
        Indices divConst(idx_t crhs) const {
            return mapElements([&](idx_t& lhs, idx_t rhs) { lhs /= rhs; },
                               crhs);
        }
        Indices minConst(idx_t crhs) const {
            return mapElements([&](idx_t& lhs, idx_t rhs) { lhs = std::min(lhs, rhs); },
                               crhs);
        }
        Indices maxConst(idx_t crhs) const {
            return mapElements([&](idx_t& lhs, idx_t rhs) { lhs = std::max(lhs, rhs); },
                               crhs);
        }

        // Reduce over all elements.
        idx_t sum() const {
            idx_t res = 0;
            for (int i = 0; i < _ndims; i++)
                res += _idxs[i];
            return res;
        }
        idx_t product() const {
            idx_t res = 1;
            for (int i = 0; i < _ndims; i++)
                res *= _idxs[i];
            return res;
        }

        // Make a Tuple w/given names.
        IdxTuple makeTuple(const VarDimNames& names) const {
            assert((int)names.size() == _ndims);

            // Make a Tuple from names.
            IdxTuple tmp;
            for (int i = 0; i < int(names.size()); i++)
                tmp.addDimBack(names[i], _idxs[i]);
            return tmp;
        }

        // Make a Tuple w/o useful names.
        IdxTuple makeTuple() const {
            IdxTuple tmp;
            for (int i = 0; i < _ndims; i++)
                tmp.addDimBack(std::string("d") + std::to_string(i), _idxs[i]);
            return tmp;
        }

        // Make a Tuple w/names from another Tuple.
        IdxTuple makeTuple(const IdxTuple& names) const {
            auto tmp = names.getDimNames();
            return makeTuple(tmp);
        }

        // Make string like "x=4, y=8".
        std::string makeDimValStr(const VarDimNames& names,
                                  std::string separator=", ",
                                  std::string infix="=",
                                  std::string prefix="",
                                  std::string suffix="") const {
            auto tmp = makeTuple(names);
            return tmp.makeDimValStr(separator, infix, prefix, suffix);
        }
        std::string makeDimValStr(const IdxTuple& names, // ignore values.
                                  std::string separator=", ",
                                  std::string infix="=",
                                  std::string prefix="",
                                  std::string suffix="") const {
            auto tmp = makeTuple(names);
            return tmp.makeDimValStr(separator, infix, prefix, suffix);
        }

        // Make string like "4, 3, 2".
        std::string makeValStr(std::string separator=", ",
                               std::string prefix="",
                               std::string suffix="") const {

            // Make a Tuple w/o useful names.
            auto tmp = makeTuple();
            return tmp.makeValStr(separator, prefix, suffix);
        }
    };

    // Define OMP reductions on Indices.
#pragma omp declare reduction(min_idxs : Indices : \
                              omp_out = omp_out.minElements(omp_in) )   \
    initializer (omp_priv = omp_orig)
#pragma omp declare reduction(max_idxs : Indices : \
                              omp_out = omp_out.maxElements(omp_in) )   \
    initializer (omp_priv = omp_orig)

    // Layout algorithms using Indices.
#include "yask_layouts.hpp"

    // Forward defns.
    struct Dims;

    // A group of Indices needed for generated loops.
    // See the help message from gen_loops.pl for the
    // documentation of the indices.
    // Make sure this stays non-virtual.
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
        ScanIndices(const Dims& dims, bool use_vec_align);
        ScanIndices(const Dims& dims, bool use_vec_align, Indices* ofs) :
            ScanIndices(dims, use_vec_align) {
            if (ofs) {
                DOMAIN_VAR_LOOP(i, j) {
                    assert(ofs->getNumDims() == ndims - 1);
                    align_ofs[i] = (*ofs)[j];
                }
            }
        }
        ScanIndices(const Dims& dims, bool use_vec_align, IdxTuple* ofs) :
            ScanIndices(dims, use_vec_align) {
            if (ofs) {
                DOMAIN_VAR_LOOP(i, j) {
                    assert(ofs->getNumDims() == ndims - 1);
                    align_ofs[i] = ofs->getVal(j);
                }
            }
        }

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
        void initFromOuter(const ScanIndices& outer) {

            // Begin & end set from start & stop of outer loop.
            begin = start = outer.start;
            end = stop = outer.stop;

            // Pass some values through.
            align = outer.align;
            align_ofs = outer.align_ofs;

            // Leave others alone.
        }
    };

} // yask namespace.
