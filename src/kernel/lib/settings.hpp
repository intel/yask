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

#pragma once

namespace yask {

    typedef Tuple<idx_t> IdxTuple;
    typedef std::vector<idx_t> GridIndices;
    typedef std::vector<std::string> GridDimNames;
    
    // A class to hold up to a given number of sizes or indices efficiently.
    // Similar to a Tuple, but less overhead and doesn't keep names.
    // Make sure this stays non-virtual.
    class Indices {
        
    public:

        // Max number of indices that can be held.
        // Note use of "+max_idxs" in code below to avoid compiler
        // trying to take a reference to it, resulting in a undefined
        // symbol (sometimes).
        static const int max_idxs = MAX_DIMS;
        
    protected:
        idx_t _idxs[max_idxs];
        
    public:
        // Ctors.
        Indices() { }
        Indices(const IdxTuple& src) {
            setFromTuple(src);
        }
        Indices(const GridIndices& src) {
            setFromVec(src);
        }
        Indices(const std::initializer_list<idx_t>& src) {
            setFromInitList(src);
        }
        Indices(int nelems, const idx_t src[]) {
            setFromArray(nelems, src);
        }
        Indices(idx_t src) {
            setFromConst(src);
        }
        
        // Default copy ctor, copy operator should be okay.

        // Access indices.
        inline idx_t& operator[](int i) {
            assert(i >= 0);
            assert(i < +max_idxs);
            return _idxs[i];
        }
        inline idx_t operator[](int i) const {
            assert(i >= 0);
            assert(i < +max_idxs);
            return _idxs[i];
        }

        // Write to an IdxTuple.
        void setTupleVals(IdxTuple& tgt) const {
            assert(tgt.size() < +max_idxs);
            for (int i = 0; i < +max_idxs; i++)
                if (i < tgt.size())
                    tgt.setVal(i, _idxs[i]);
        }

        // Read from an IdxTuple.
        void setFromTuple(const IdxTuple& src) {
            assert(src.size() < +max_idxs);
            if (src.size() == 0)
                setFromConst(0);
            else {
                int n = std::min(int(src.size()), +max_idxs);
                for (int i = 0; i < n; i++)
                    _idxs[i] = src.getVal(i);
            }
        }
        
        // Other inits.
        void setFromVec(const GridIndices& src) {
            assert(src.size() < +max_idxs);
            if (src.size() == 0)
                setFromConst(0);
            else {
                int n = std::min(int(src.size()), +max_idxs);
                for (int i = 0; i < n; i++)
                    _idxs[i] = src[i];
            }
        }
        void setFromArray(int nelems, const idx_t src[]) {
            assert(nelems < +max_idxs);
            if (nelems == 0)
                setFromConst(0);
            else {
                int n = std::min(nelems, +max_idxs);
                for (int i = 0; i < n; i++)
                    _idxs[i] = src[i];
            }
        }
        void setFromInitList(const std::initializer_list<idx_t>& src) {
            assert(src.size() < +max_idxs);
            int i = 0;
            for (auto idx : src)
                _idxs[i++] = idx;
        }
        void setFromConst(idx_t val) {
            for (int i = 0; i < +max_idxs; i++)
                _idxs[i] = val;
        }
        
        // Some comparisons.
        // These assume all the indices are valid or
        // initialized to the same value.
        bool operator==(const Indices& rhs) const {
            for (int i = 0; i < +max_idxs; i++)
                if (_idxs[i] != rhs._idxs[i])
                    return false;
            return true;
        }
        bool operator!=(const Indices& rhs) const {
            return !operator==(rhs);
        }
        bool operator<(const Indices& rhs) const {
            for (int i = 0; i < +max_idxs; i++)
                if (_idxs[i] < rhs._idxs[i])
                    return true;
                else if (_idxs[i] > rhs._idxs[i])
                    return false;
            return false;       // equal, so not less than.
        }
        bool operator>(const Indices& rhs) const {
            for (int i = 0; i < +max_idxs; i++)
                if (_idxs[i] > rhs._idxs[i])
                    return true;
                else if (_idxs[i] < rhs._idxs[i])
                    return false;
            return false;       // equal, so not greater than.
        }

        // Some element-wise operators.
        // These all return a new set of Indices rather
        // than modifying this object.
        Indices addElements(const Indices& other) const {
            Indices res;
            for (int i = 0; i < max_idxs; i++)
                res._idxs[i] = _idxs[i] + other._idxs[i];
            return res;
        }
        Indices multElements(const Indices& other) const {
            Indices res;
            for (int i = 0; i < max_idxs; i++)
                res._idxs[i] = _idxs[i] * other._idxs[i];
            return res;
        }
        Indices minElements(const Indices& other) const {
            Indices res;
            for (int i = 0; i < max_idxs; i++)
                res._idxs[i] = std::min(_idxs[i], other._idxs[i]);
            return res;
        }
        Indices maxElements(const Indices& other) const {
            Indices res;
            for (int i = 0; i < max_idxs; i++)
                res._idxs[i] = std::max(_idxs[i], other._idxs[i]);
            return res;
        }

        // Operate on all elements.
        Indices addConst(idx_t n) const {
            Indices res;
            for (int i = 0; i < max_idxs; i++)
                res._idxs[i] = _idxs[i] + n;
            return res;
        }
        Indices multConst(idx_t n) const {
            Indices res;
            for (int i = 0; i < max_idxs; i++)
                res._idxs[i] = _idxs[i] * n;
            return res;
        }

        // Reduce over all elements.
        idx_t sum() const {
            idx_t res = 0;
            for (int i = 0; i < max_idxs; i++)
                res += _idxs[i];
            return res;
        }
        idx_t product() const {
            idx_t res = 1;
            for (int i = 0; i < max_idxs; i++)
                res *= _idxs[i];
            return res;
        }
        
        // Make string like "x=4, y=8".
        std::string makeDimValStr(const GridDimNames& names,
                                          std::string separator=", ",
                                          std::string infix="=",
                                          std::string prefix="",
                                          std::string suffix="") const {
            assert(names.size() <= max_idxs);

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
            assert(nvals <= max_idxs);

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
    initializer (omp_priv = idx_max)
#pragma omp declare reduction(max_idxs : Indices : \
                              omp_out = omp_out.maxElements(omp_in) )   \
    initializer (omp_priv = idx_min)

    // Layout algorithms using Indices.
#include "yask_layouts.hpp"
    
    // A group of Indices needed for generated loops.
    // See the help message from gen_loops.pl for the
    // documentation of the indices.
    // Make sure this stays non-virtual.
    struct ScanIndices {

        // Values that remain the same for each sub-range.
        Indices begin, end;     // first and last+1 range of each index.
        Indices step;           // step value within range.
        Indices group_size;     // priority grouping within range.

        // Values that differ for each sub-range.
        Indices start, stop;    // first and last+1 for this sub-range.
        Indices index;          // 0-based unique index for each sub-range.

        // Example w/3 sub-ranges in overall range:
        // begin                                         end
        //   |--------------------------------------------|
        //   |------------------|------------------|------|
        // start               stop                            (index = 0)
        //                    start               stop         (index = 1)
        //                                       start   stop  (index = 2)
        
        // Default init.
        ScanIndices() {
            begin.setFromConst(0);
            end.setFromConst(0);
            step.setFromConst(1);
            group_size.setFromConst(1);
            start.setFromConst(0);
            stop.setFromConst(0);
            index.setFromConst(0);
        }
        
        // Init from outer-loop indices.
        // Start..stop from point in outer loop become begin..end
        // for this loop.
        void initFromOuter(const ScanIndices& outer) {

            // Begin & end set from start & stop of outer loop.
            begin = outer.start;
            end = outer.stop;

            // Pass other values through by default.
            start = outer.start;
            stop = outer.stop;
            index = outer.index;
        }
    };

    // Forward defns.
    struct StencilContext;
    class YkGridBase;

    // Some derivations.
    typedef std::shared_ptr<YkGridBase> YkGridPtr;
    typedef std::vector<YkGridPtr> GridPtrs;
    typedef std::set<YkGridPtr> GridPtrSet;
    
    // Environmental settings.
    struct KernelEnv :
        public virtual yk_env {

        // MPI vars.
        MPI_Comm comm=0;        // communicator.
        int num_ranks=1;        // total number of ranks.
        int my_rank=0;          // MPI-assigned index.

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
    // Similar to that in the YASK compiler.
    struct Dims {

        // Algorithm for vec dims in fold layout.
        VEC_FOLD_LAYOUT _vec_fold_layout;

        // Dimensions with 0 values.
        std::string _step_dim;  // usually time, 't'.
        std::string _inner_dim; // the domain dim used in the inner loop.
        IdxTuple _domain_dims;
        IdxTuple _stencil_dims;
        IdxTuple _misc_dims;

        // Dimensions and sizes.
        IdxTuple _fold_pts;     // all domain dims.
        IdxTuple _vec_fold_pts; // just those with >1 pts.
        IdxTuple _cluster_pts;
        IdxTuple _cluster_mults;

        // Direction of step.
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

            // Use compiler-generated fold layout.
            idx_t i = _vec_fold_layout.layout(fold_ofs);
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
        
        // Max number of neighbors in all domain dimensions.
        // Used to describe the n-D space of neighbors.
        // This object is effectively a constant used to convert between
        // n-D and 1-D indices.
        IdxTuple neighbor_offsets;

        // Neighborhood size includes self.
        // Number of points in n-D space of neighbors.
        // NB: this is the *max* number of neighbors, not necessarily the actual number.
        idx_t neighborhood_size = 0;

        // MPI rank of each neighbor.
        // MPI_PROC_NULL => no neighbor.
        typedef std::vector<int> Neighbors;
        Neighbors my_neighbors;
        
        // Ctor based on pre-set problem dimensions.
        MPIInfo(DimsPtr dims) : _dims(dims) {

            // Max neighbors.
            neighbor_offsets = dims->_domain_dims; // copy dims from domain.
            neighbor_offsets.setValsSame(num_offsets); // set sizes in each domain dim.
            neighborhood_size = neighbor_offsets.product(); // neighbors in all dims.

            // Init array to store all rank numbers.
            my_neighbors.insert(my_neighbors.begin(), neighborhood_size, MPI_PROC_NULL);
        }

        // Get a 1D index for a neighbor.
        // Input 'offsets': tuple of NeighborOffset vals.
        virtual idx_t getNeighborIndex(const IdxTuple& offsets) const {
            idx_t i = neighbor_offsets.layout(offsets); // 1D index.
            assert(i >= 0);
            assert(i < neighborhood_size);
            return i;
        }

        // Visit all neighbors.
        virtual void visitNeighbors(std::function<void
                                    (const IdxTuple& offsets, // NeighborOffset vals.
                                     int rank, // MPI rank.
                                     int index // simple counter from 0.
                                     )> visitor);
    };
    typedef std::shared_ptr<MPIInfo> MPIInfoPtr;

    // MPI buffers for *one* grid: a send and receive buffer for each neighbor.
    struct MPIBufs {

        MPIInfoPtr _mpiInfo;
        
        // Need one buf for send and one for receive for each neighbor.
        enum BufDir { bufSend, bufRecv, nBufDirs };

        // Buffers for all possible neighbors.
        typedef std::vector<YkGridPtr> NeighborBufs;
        NeighborBufs send_bufs, recv_bufs;

        // Copy starting points.
        typedef std::vector<IdxTuple> TupleList;
        TupleList send_begins, recv_begins;

        MPIBufs(MPIInfoPtr mpiInfo) :
            _mpiInfo(mpiInfo) {

            // Init buffer pointers.
            auto n = _mpiInfo->neighborhood_size;
            send_bufs.insert(send_bufs.begin(), n, NULL);
            recv_bufs.insert(recv_bufs.begin(), n, NULL);

            // Init begin points.
            IdxTuple emptyTuple;
            send_begins.insert(send_begins.begin(), n, emptyTuple);
            recv_begins.insert(recv_begins.begin(), n, emptyTuple);
        }

        // Apply a function to each neighbor rank.
        // Called visitor function will contain the rank index of the neighbor.
        virtual void visitNeighbors(std::function<void (const IdxTuple& offsets, // NeighborOffset.
                                                        int rank,
                                                        int index, // simple counter from 0.
                                                        YkGridPtr sendBuf,
                                                        IdxTuple& sendBegin,
                                                        YkGridPtr recvBuf,
                                                        IdxTuple& recvBegin)> visitor);
            
        // Access a buffer by direction and neighbor offsets.
        virtual YkGridPtr& getBuf(BufDir bd, const IdxTuple& offsets);

        // Create new buffer in given direction and size.
        virtual YkGridPtr makeBuf(BufDir bd,
                                  const IdxTuple& offsets,
                                  const IdxTuple& sizes,
                                  const std::string& name,
                                  StencilContext& context);
    };

    // Application settings to control size and perf of stencil code.
    class KernelSettings {

    protected:
        idx_t def_steps = 1;
        idx_t def_rank = 128;
        idx_t def_block = 32;

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
        bool find_loc=true;            // whether my rank index needs to be calculated.
        int msg_rank=0;             // rank that prints informational messages.

        // OpenMP settings.
        int max_threads=0;      // Initial number of threads to use overall; 0=>OMP default.
        int thread_divisor=1;   // Reduce number of threads by this amount.
        int num_block_threads=1; // Number of threads to use for a block.

        // Ctor.
        KernelSettings(DimsPtr dims) : _dims(dims) {

            // Use both step and domain dims for all size tuples.
            _rank_sizes = dims->_stencil_dims;
            _rank_sizes.setValsSame(def_rank);             // size of rank.
            _rank_sizes.setVal(dims->_step_dim, def_steps); // num steps.

            _region_sizes = dims->_stencil_dims;
            _region_sizes.setValsSame(0);          // 0 => full rank.
            _rank_sizes.setVal(dims->_step_dim, 1); // 1 => no wave-front tiling.

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

            // Use domain dims only for MPI tuples.
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
        // Add options to set all settings to a cmd-line parser.
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
        virtual void adjustSettings(std::ostream& os);

    };
    typedef std::shared_ptr<KernelSettings> KernelSettingsPtr;
    
} // yask namespace.
