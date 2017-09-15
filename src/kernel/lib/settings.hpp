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

        static const int max_domain_dims = MAX_DIMS - 1; // 1 reserved for step dim.

        // Dimensions with unused values.
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

        // Check whether dim exists and is of allowed type.
        // If not, abort with error, reporting 'fn_name'.
        void checkDimType(const std::string& dim,
                          const std::string& fn_name,
                          bool step_ok,
                          bool domain_ok,
                          bool misc_ok) const;

        // Get linear index into a vector given 'elem_ofs', which are
        // element offsets that may include other dimensions.
        idx_t getElemIndexInVec(const IdxTuple& elem_ofs) const {
            
            // Use fold layout to find element index.
            // TODO: make this more efficient.
            auto i = _fold_pts.layout(elem_ofs, false);
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
