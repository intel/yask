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

#pragma once

namespace yask {

    typedef Tuple<idx_t> IdxTuple;
    typedef std::vector<idx_t> GridIndices;
    typedef std::vector<idx_t> GridDimSizes;
    typedef std::vector<std::string> GridDimNames;

    // A class to hold up to a given number of sizes or indices efficiently.
    // Similar to a Tuple, but less overhead and doesn't keep names.
    // Make sure this stays non-virtual.
    // TODO: make this a template with _ndims as a parameter.
    class Indices {

    public:

        // Max number of indices that can be held.
        // Note use of "+max_idxs" in code below to avoid compiler
        // trying to take a reference to it, resulting in a undefined
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
        Indices(const GridIndices& src) {
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
            assert(tgt.size() == _ndims);
            for (int i = 0; i < _ndims; i++)
                if (i < tgt.size())
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
        void setFromVec(const GridIndices& src) {
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

        // Make string like "x=4, y=8".
        std::string makeDimValStr(const GridDimNames& names,
                                  std::string separator=", ",
                                  std::string infix="=",
                                  std::string prefix="",
                                  std::string suffix="") const {
            assert((int)names.size() == _ndims);

            // Make a Tuple from names.
            IdxTuple tmp;
            for (int i = 0; i < int(names.size()); i++)
                tmp.addDimBack(names[i], _idxs[i]);
            return tmp.makeDimValStr(separator, infix, prefix, suffix);
        }

        // Make string like "4, 3, 2".
        std::string makeValStr(int nvals,
                               std::string separator=", ",
                               std::string prefix="",
                               std::string suffix="") const {
            assert(nvals == _ndims);

            // Make a Tuple w/o useful names.
            IdxTuple tmp;
            for (int i = 0; i < nvals; i++)
                tmp.addDimBack(std::string(1, '0' + char(i)), _idxs[i]);
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
    struct StencilContext;
    class YkGridBase;

    // Some derivations from grid types.
    typedef std::shared_ptr<YkGridBase> YkGridPtr;
    typedef std::set<YkGridPtr> GridPtrSet;
    typedef std::vector<YkGridPtr> GridPtrs;
    typedef std::map<std::string, YkGridPtr> GridPtrMap;
    typedef std::vector<GridPtrs*> ScratchVecs;

    // Environmental settings.
    struct KernelEnv :
        public virtual yk_env {

        // MPI vars.
        MPI_Comm comm=0;        // communicator.
        int num_ranks=1;        // total number of ranks.
        int my_rank=0;          // MPI-assigned index.

        // OMP vars.
        int max_threads=0;      // initial value from OMP.

        virtual ~KernelEnv() {}

        // Init MPI, OMP, etc.
        // This is normally called very early in the program.
        virtual void initEnv(int* argc, char*** argv);

        // APIs.
        virtual int get_num_ranks() const {
            return num_ranks;
        }
        virtual int get_rank_index() const {
            return my_rank;
        }
        virtual void global_barrier() const {
            MPI_Barrier(comm);
        }
    };
    typedef std::shared_ptr<KernelEnv> KernelEnvPtr;

    // Dimensions for a solution.
    // Similar to the Dimensions class in the YASK compiler
    // from which these values are set.
    struct Dims {

        // Algorithm for vec dims in fold layout.
        VEC_FOLD_LAYOUT_CLASS _vec_fold_layout;

        // Dimensions with 0 values.
        std::string _step_dim;  // usually time, 't'.
        std::string _inner_dim; // the domain dim used in the inner loop.
        IdxTuple _domain_dims;
        IdxTuple _stencil_dims; // step & domain dims.
        IdxTuple _misc_dims;

        // Dimensions and sizes.
        IdxTuple _fold_pts;     // all domain dims.
        IdxTuple _vec_fold_pts; // just those with >1 pts.
        IdxTuple _cluster_pts;  // all domain dims.
        IdxTuple _cluster_mults; // all domain dims.

        // Direction of step.
        // This is a heuristic value used only for stepping the
        // perf-measuring utility and the auto-tuner.
        int _step_dir = 0;    // 0: undetermined, +1: forward, -1: backward.

        // Check whether dim exists and is of allowed type.
        // If not, abort with error, reporting 'fn_name'.
        void checkDimType(const std::string& dim,
                          const std::string& fn_name,
                          bool step_ok,
                          bool domain_ok,
                          bool misc_ok) const;

        // Get linear index into a vector given 'fold_ofs', which are
        // element offsets that must be *exactly* those in _vec_fold_pts.
        idx_t getElemIndexInVec(const Indices& fold_ofs) const {
            assert(fold_ofs.getNumDims() == NUM_VEC_FOLD_DIMS);

            // Use compiler-generated fold macro.
            idx_t i = VEC_FOLD_LAYOUT(fold_ofs);

#ifdef DEBUG_LAYOUT
            // Use compiler-generated fold layout class.
            idx_t j = _vec_fold_layout.layout(fold_ofs);
            assert(i == j);
#endif

            return i;
        }

        // Get linear index into a vector given 'elem_ofs', which are
        // element offsets that may include other dimensions.
        idx_t getElemIndexInVec(const IdxTuple& elem_ofs) const {
            assert(_vec_fold_pts.getNumDims() == NUM_VEC_FOLD_DIMS);
            if (NUM_VEC_FOLD_DIMS == 0)
                return 0;

            // Get required offsets into an Indices obj.
            IdxTuple fold_ofs(_vec_fold_pts);
            fold_ofs.setValsSame(0);
            fold_ofs.setVals(elem_ofs, false); // copy only fold offsets.
            Indices fofs(fold_ofs);

            // Call version that requires vec-fold offsets only.
            idx_t i = getElemIndexInVec(fofs);

            // Use fold layout to find element index.
#ifdef DEBUG_LAYOUT
            idx_t i2 = _vec_fold_pts.layout(fold_ofs, false);
            assert(i == i2);
#endif
            return i;
        }
    };
    typedef std::shared_ptr<Dims> DimsPtr;

    // A group of Indices needed for generated loops.
    // See the help message from gen_loops.pl for the
    // documentation of the indices.
    // Make sure this stays non-virtual.
    struct ScanIndices {
        int ndims = 0;

        // Values that remain the same for each sub-range.
        Indices begin, end;     // first and end (beyond last) range of each index.
        Indices step;           // step value within range.
        Indices align;          // alignment of steps after first one.
        Indices align_ofs;      // adjustment for alignment (see below).
        Indices group_size;     // proximity grouping within range.

        // Alignment: when possible, each step will be aligned
        // such that ((start - align_ofs) % align) == 0.

        // Values that differ for each sub-range.
        Indices start, stop;    // first and last+1 for this sub-range.
        Indices index;          // 0-based unique index for each sub-range in each dim.

        // Example w/3 sub-ranges in overall range:
        // begin                                         end
        //   |--------------------------------------------|
        //   |------------------|------------------|------|
        // start               stop                            (index = 0)
        //                    start               stop         (index = 1)
        //                                       start   stop  (index = 2)

        // Default init.
        ScanIndices(const Dims& dims, bool use_vec_align, IdxTuple* ofs) :
            ndims(dims._stencil_dims.size()),
            begin(idx_t(0), ndims),
            end(idx_t(0), ndims),
            step(idx_t(1), ndims),
            align(idx_t(1), ndims),
            align_ofs(idx_t(0), ndims),
            group_size(idx_t(1), ndims),
            start(idx_t(0), ndims),
            stop(idx_t(0), ndims),
            index(idx_t(0), ndims) {

            // i: index for stencil dims, j: index for domain dims.
            for (int i = 0, j = 0; i < ndims; i++) {
                if (i == Indices::step_posn) continue;

                // Set alignment to vector lengths.
                if (use_vec_align)
                    align[i] = dims._fold_pts[j];

                // Set alignment offset.
                if (ofs) {
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

    // MPI neighbor info.
    class MPIInfo {

    public:
        // Problem dimensions.
        DimsPtr _dims;

        // Each rank can have up to 3 neighbors in each dim, including self.
        // Example for 2D:
        //   +------+------+------+
        //   |x=prev|x=self|x=next|
        //   |y=next|y=next|y=next|
        //   +------+------+------+
        //   |x=prev|x=self|x=next|
        //   |y=self|y=self|y=self| Center rank is self.
        //   +------+------+------+
        // ^ |x=prev|x=self|x=next|
        // | |y=prev|y=prev|y=prev|
        // y +------+------+------+
        //   x-->
        enum NeighborOffset { rank_prev, rank_self, rank_next, num_offsets };

        // Max number of immediate neighbors in all domain dimensions.
        // Used to describe the n-D space of neighbors.
        // This object is effectively a constant used to convert between
        // n-D and 1-D indices.
        IdxTuple neighborhood_sizes;

        // Neighborhood size includes self.
        // Number of points in n-D space of neighbors.
        // NB: this is the *max* number of neighbors, not necessarily the actual number.
        idx_t neighborhood_size = 0;

        // What getNeighborIndex() returns for myself.
        int my_neighbor_index;

        // MPI rank of each neighbor.
        // MPI_PROC_NULL => no neighbor.
        // Vector index is per getNeighborIndex().
        typedef std::vector<int> Neighbors;
        Neighbors my_neighbors;

        // Manhattan distance to each neighbor.
        // Vector index is per getNeighborIndex().
        std::vector<int> man_dists;

        // Whether each neighbor has all its rank-domain
        // sizes as a multiple of the vector length.
        std::vector<bool> has_all_vlen_mults;

        // Ctor based on pre-set problem dimensions.
        MPIInfo(DimsPtr dims) : _dims(dims) {

            // Max neighbors.
            neighborhood_sizes = dims->_domain_dims; // copy dims from domain.
            neighborhood_sizes.setValsSame(num_offsets); // set sizes in each domain dim.
            neighborhood_size = neighborhood_sizes.product(); // neighbors in all dims.

            // Myself.
            IdxTuple noffsets(neighborhood_sizes);
            noffsets.setValsSame(rank_self);
            my_neighbor_index = getNeighborIndex(noffsets);

            // Init arrays.
            my_neighbors.resize(neighborhood_size, MPI_PROC_NULL);
            man_dists.resize(neighborhood_size, 0);
            has_all_vlen_mults.resize(neighborhood_size, false);
        }

        // Get a 1D index for a neighbor.
        // Input 'offsets': tuple of NeighborOffset vals.
        virtual idx_t getNeighborIndex(const IdxTuple& offsets) const {
            idx_t i = neighborhood_sizes.layout(offsets); // 1D index.
            assert(i >= 0);
            assert(i < neighborhood_size);
            return i;
        }

        // Visit all neighbors.
        // Does NOT visit self.
        virtual void visitNeighbors(std::function<void
                                    (const IdxTuple& offsets, // NeighborOffset vals.
                                     int rank, // MPI rank; might be MPI_PROC_NULL.
                                     int index // simple counter from 0.
                                     )> visitor);
    };
    typedef std::shared_ptr<MPIInfo> MPIInfoPtr;

    // MPI data for one buffer for one neighbor of one grid.
    struct MPIBuf {

        // Name for trace output.
        std::string name;

        // Send or receive buffer.
        std::shared_ptr<char> _base;
        real_t* _elems = 0;

        // Range to copy to/from grid.
        // NB: step index not set properly for grids with step dim.
        IdxTuple begin_pt, last_pt;

        // Number of points to copy to/from grid in each dim.
        IdxTuple num_pts;

        // Whether the number of points is a multiple of the
        // vector length in all dims and buffer is aligned.
        bool vec_copy_ok = false;

        // Number of points overall.
        idx_t get_size() const {
            if (num_pts.size() == 0)
                return 0;
            return num_pts.product();
        }
        idx_t get_bytes() const {
            return get_size() * sizeof(real_t);
        }

        // Set pointer to storage.
        // Free old storage.
        // 'base' should provide get_num_bytes() bytes at offset bytes.
        void set_storage(std::shared_ptr<char>& base, size_t offset);

        // Release storage.
        void release_storage() {
            _base.reset();
            _elems = 0;
        }

        // Reset.
        void clear() {
            name.clear();
            begin_pt.clear();
            last_pt.clear();
            num_pts.clear();
            release_storage();
        }
        ~MPIBuf() {
            clear();
        }
    };

    // MPI data for both buffers for one neighbor of one grid.
    struct MPIBufs {

        // Need one buf for send and one for receive for each neighbor.
        enum BufDir { bufSend, bufRecv, nBufDirs };

        MPIBuf bufs[nBufDirs];
    };

    // MPI data for one grid.
    // Contains a send and receive buffer for each neighbor
    // and some meta-data.
    struct MPIData {

        MPIInfoPtr _mpiInfo;

        // Buffers for all possible neighbors.
        typedef std::vector<MPIBufs> NeighborBufs;
        NeighborBufs bufs;

        // Arrays for request handles.
        // These are used for async comms.
        std::vector<MPI_Request> recv_reqs;
        std::vector<MPI_Request> send_reqs;
        
        MPIData(MPIInfoPtr mpiInfo) :
            _mpiInfo(mpiInfo) {

            // Init vector of buffers.
            auto n = _mpiInfo->neighborhood_size;
            MPIBufs emptyBufs;
            bufs.resize(n, emptyBufs);

            // Init handles.
            recv_reqs.resize(n, MPI_REQUEST_NULL);
            send_reqs.resize(n, MPI_REQUEST_NULL);
        }

        // Apply a function to each neighbor rank.
        // Called visitor function will contain the rank index of the neighbor.
        virtual void visitNeighbors(std::function<void (const IdxTuple& neighbor_offsets, // NeighborOffset.
                                                        int rank,
                                                        int index, // simple counter from 0.
                                                        MPIBufs& bufs)> visitor);

        // Access a buffer by direction and neighbor offsets.
        virtual MPIBuf& getBuf(MPIBufs::BufDir bd, const IdxTuple& neighbor_offsets);
    };

    // Application settings to control size and perf of stencil code.
    class KernelSettings {

    protected:
        idx_t def_steps = 1;
        idx_t def_rank = 128;
        idx_t def_block = 32;

        yask_output_factory yof;
        yask_output_ptr nullop = yof.new_null_output();

    public:

        // problem dimensions.
        DimsPtr _dims;

        // Sizes in elements (points).
        IdxTuple _rank_sizes;     // number of steps and this rank's domain sizes.
        IdxTuple _region_sizes;   // region size (used for wave-front tiling).
        IdxTuple _block_group_sizes; // block-group size (only used for 'grouped' region loops).
        IdxTuple _block_sizes;       // block size (used for each outer thread).
        IdxTuple _sub_block_group_sizes; // sub-block-group size (only used for 'grouped' block loops).
        IdxTuple _sub_block_sizes;       // sub-block size (used for each nested thread).
        IdxTuple _min_pad_sizes;         // minimum spatial padding.
        IdxTuple _extra_pad_sizes;       // extra spatial padding.

        // MPI settings.
        IdxTuple _num_ranks;       // number of ranks in each dim.
        IdxTuple _rank_indices;    // my rank index in each dim.
        bool find_loc = true;      // whether my rank index needs to be calculated.
        int msg_rank = 0;          // rank that prints informational messages.
        bool overlap_comms = true; // overlap comms with computation.

        // OpenMP settings.
        int max_threads = 0;      // Initial number of threads to use overall; 0=>OMP default.
        int thread_divisor = 1;   // Reduce number of threads by this amount.
        int num_block_threads = 1; // Number of threads to use for a block.

        // Prefetch distances.
        // Prefetching must be enabled via YASK_PREFETCH_L[12] macros.
        int _prefetch_L1_dist = 1;
        int _prefetch_L2_dist = 2;

        // NUMA settings.
        int _numa_pref = NUMA_PREF;

        // Ctor.
        KernelSettings(DimsPtr dims, KernelEnvPtr env) :
            _dims(dims), max_threads(env->max_threads) {

            // Use both step and domain dims for all size tuples.
            _rank_sizes = dims->_stencil_dims;
            _rank_sizes.setValsSame(def_rank);             // size of rank.
            _rank_sizes.setVal(dims->_step_dim, def_steps); // num steps.

            _region_sizes = dims->_stencil_dims;
            _region_sizes.setValsSame(0);          // 0 => full rank.
            _region_sizes.setVal(dims->_step_dim, 1); // 1 => no wave-front tiling.

            _block_group_sizes = dims->_stencil_dims;
            _block_group_sizes.setValsSame(0); // 0 => min size.

            _block_sizes = dims->_stencil_dims;
            _block_sizes.setValsSame(def_block); // size of block.
            _block_sizes.setVal(dims->_step_dim, 1); // 1 => no temporal blocking.

            _sub_block_group_sizes = dims->_stencil_dims;
            _sub_block_group_sizes.setValsSame(0); // 0 => min size.

            _sub_block_sizes = dims->_stencil_dims;
            _sub_block_sizes.setValsSame(0);            // 0 => default settings.
            _sub_block_sizes.setVal(dims->_step_dim, 1); // 1 => no temporal blocking.

            _min_pad_sizes = dims->_stencil_dims;
            _min_pad_sizes.setValsSame(0);

            _extra_pad_sizes = dims->_stencil_dims;
            _extra_pad_sizes.setValsSame(0);

            // Use only domain dims for MPI tuples.
            _num_ranks = dims->_domain_dims;
            _num_ranks.setValsSame(1);

            _rank_indices = dims->_domain_dims;
            _rank_indices.setValsSame(0);
        }
        virtual ~KernelSettings() { }

    protected:
        // Add options to set one domain var to a cmd-line parser.
        virtual void _add_domain_option(CommandLineParser& parser,
                                        const std::string& prefix,
                                        const std::string& descrip,
                                        IdxTuple& var);

        idx_t findNumSubsets(std::ostream& os,
                             IdxTuple& inner_sizes, const std::string& inner_name,
                             const IdxTuple& outer_sizes, const std::string& outer_name,
                             const IdxTuple& mults);

    public:
        // Add options to a cmd-line parser to set the settings.
        virtual void add_options(CommandLineParser& parser);

        // Print usage message.
        void print_usage(std::ostream& os,
                         CommandLineParser& parser,
                         const std::string& pgmName,
                         const std::string& appNotes,
                         const std::vector<std::string>& appExamples) const;

        // Make sure all user-provided settings are valid by rounding-up
        // values as needed.
        // Called from prepare_solution(), so it doesn't normally need to be called from user code.
        // Prints informational info to 'os'.
        virtual void adjustSettings(std::ostream& os, KernelEnvPtr env);
        virtual void adjustSettings(KernelEnvPtr env) {
            adjustSettings(nullop->get_ostream(), env);
        }

        // Determine if this is the first or last rank in given dim.
        virtual bool is_first_rank(const std::string dim) {
            return _rank_indices[dim] == 0;
        }
        virtual bool is_last_rank(const std::string dim) {
            return _rank_indices[dim] == _num_ranks[dim] - 1;
        }

        // Is WF tiling being used?
        virtual bool is_time_tiling() {
            return _region_sizes[_dims->_step_dim] > 1;
        }
    };
    typedef std::shared_ptr<KernelSettings> KernelSettingsPtr;

} // yask namespace.
