/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2022, Intel Corporation

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
    class StencilContext;
    class YkVarBase;
    class YkVarImpl;
    class KernelStateBase;

    // Some derivations from var types.
    typedef std::shared_ptr<YkVarImpl> YkVarPtr;
    typedef std::set<YkVarPtr> VarPtrSet;
    typedef std::vector<YkVarPtr> VarPtrs;
    typedef std::map<std::string, YkVarPtr> VarPtrMap;
    typedef std::vector<VarPtrs*> ScratchVecs;

    // Environmental settings.
    // Several member vars are static because they are considered global
    // and set/get can be called w/o an obj.
    class KernelEnv :
        public virtual yk_env {

        // An OpenMP lock to use for debug.
        static omp_lock_t _debug_lock;
        static bool _debug_lock_init_done;

    public:

        // Output stream for debug & trace messages.
        static yask_output_ptr _debug;

        // Is tracing enabled?
        static bool _trace;

        // OMP offload devices.
        static bool _use_offload;
        #ifdef USE_OFFLOAD
        static int _omp_hostn;
        static int _omp_devn;
        #endif

        // MPI vars.
        MPI_Comm comm = MPI_COMM_NULL; // global communicator.
        MPI_Group group = MPI_GROUP_NULL;
        int num_ranks = 1;        // total number of ranks.
        int my_rank = 0;          // MPI-assigned index.

        // Vars for shared-mem ranks.
        MPI_Comm shm_comm = MPI_COMM_NULL; // shm communicator.
        MPI_Group shm_group = MPI_GROUP_NULL;
        int num_shm_ranks = 1;  // ranks in shm_comm.
        int my_shm_rank = 0;    // my index in shm_comm.

        // OMP vars.
        int max_threads=0;      // initial value from OMP.

        KernelEnv() { }
        virtual ~KernelEnv() { }

        // Init MPI, OMP, etc.
        // This is normally called very early in the program.
        virtual void init_env(int* argc, char*** argv, MPI_Comm comm);

        // Lock.
        static void set_debug_lock() {
            if (!_debug_lock_init_done) {
                omp_init_lock(&_debug_lock);
                _debug_lock_init_done = true;
            }
            omp_set_lock(&_debug_lock);
        }
        static void unset_debug_lock() {
            assert(_debug_lock_init_done);
            omp_unset_lock(&_debug_lock);
        }

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
    // These are not changed after initialization.
    struct Dims {

        // Algorithm for vec dims in fold layout.
        VEC_FOLD_LAYOUT_CLASS _vec_fold_layout;

        // Special dims.
        std::string _step_dim;  // usually time, 't'.
        std::string _inner_layout_dim; // innermost index in layout.
        std::string _inner_loop_dim; // innermost index in pico loops.
        
        // Dimensions with 0 values.
        IdxTuple _domain_dims;  // e.g., 'x', 'y'.
        IdxTuple _stencil_dims; // union of step & domain dims.
        IdxTuple _misc_dims;

        // Dimensions and sizes.
        IdxTuple _fold_pts;     // all domain dims.
        IdxTuple _vec_fold_pts; // just those with >1 pts.
        IdxTuple _cluster_pts;  // all domain dims.
        IdxTuple _cluster_mults; // all domain dims.

        // Just sizes.
        Indices _fold_sizes;    // all domain dims.

        // Direction of step.
        // This is a heuristic value used only for stepping the
        // perf-measuring utility and the auto-tuner.
        int _step_dir = 0;    // 0: undetermined, +1: forward, -1: backward.

        // Check whether dim exists and is of allowed type.
        // If not, abort with error, reporting 'fn_name'.
        void check_dim_type(const std::string& dim,
                          const std::string& fn_name,
                          bool step_ok,
                          bool domain_ok,
                          bool misc_ok) const;

        // Get linear index into a vector given 'fold_ofs', which are
        // element offsets that must be *exactly* those in _vec_fold_pts.
        idx_t get_elem_index_in_vec(const Indices& fold_ofs) const {
            host_assert(fold_ofs.get_num_dims() == NUM_VEC_FOLD_DIMS);

            // Use compiler-generated fold macro.
            idx_t i = VEC_FOLD_LAYOUT(fold_ofs);

#ifdef DEBUG_LAYOUT
            // Use compiler-generated fold layout class.
            idx_t j = _vec_fold_layout.layout(fold_ofs);
            host_assert(i == j);
#endif

            return i;
        }
    };
    typedef std::shared_ptr<Dims> DimsPtr;

    // Utility to determine number of spatial points in a "sizes" var.
    inline idx_t get_num_domain_points(const IdxTuple& sizes) {
        host_assert(sizes.get_num_dims() == NUM_STENCIL_DIMS);
        idx_t pts = 1;
        DOMAIN_VAR_LOOP_FAST(i, j)
            pts *= sizes[i];
        return pts;
    }

    // Application settings to control size and perf of stencil code.  Most
    // of these vars can be set via cmd-line options and/or APIs.
    class KernelSettings {

        // Default block size on CPU.
        int def_blk_size = 0;

   public:

        // Abbreviations for sizes.
        static const std::string _mega_block_str;
        static const std::string _block_str;
        static const std::string _micro_block_str;
        static const std::string _nano_block_str;
        static const std::string _pico_block_str;

        // Ptr to problem dimensions (NOT sizes), folding, etc.
        // This is solution info from the YASK compiler.
        DimsPtr _dims;

        // Sizes in elements (points).
        // All these tuples contain step dims, even the ones that
        // don't use them, for consistency.
        IdxTuple _global_sizes;     // Overall problem domain sizes.
        IdxTuple _rank_sizes;     // This rank's domain (local) sizes.
        IdxTuple _rank_tile_sizes; // rank-tile size (only used for 'tiled' rank loops).
        IdxTuple _mega_block_sizes;   // mega-block size (used for wave-front tiling).
        IdxTuple _mega_block_tile_sizes; // mega-block-tile size (only used for 'tiled' mega-block loops).
        IdxTuple _block_sizes;       // block size (used for each outer thread).
        IdxTuple _block_tile_sizes; // block-tile size (only used for 'tiled' block loops).
        IdxTuple _micro_block_sizes;       // micro-block size (used for wave-fronts in blocks).
        IdxTuple _micro_block_tile_sizes; // micro-block-tile size (only used for 'tiled' micro-block loops).
        IdxTuple _nano_block_sizes;       // nano-block size (used for each nested thread).
        IdxTuple _nano_block_tile_sizes; // nano-block-tile size (only used for 'tiled' nano-block loops).
        IdxTuple _pico_block_sizes;       // pico-block size (used within nano-blocks, no pico-tiling).

        // Global padding applied to all vars by default.
        // These tuples contain all stencil dims, even though the step dim isn't used.
        IdxTuple _min_pad_sizes;         // minimum spatial padding (including halos).
        IdxTuple _extra_pad_sizes;       // extra spatial padding (outside of halos).

        // MPI settings.
        // These tuples contain only domain dims.
        IdxTuple _num_ranks;       // number of ranks in each domain dim.
        IdxTuple _rank_indices;    // my rank index in each domain dim.
        bool find_loc = true;      // whether my rank index needs to be calculated.
        bool overlap_comms = true; // overlap comms with computation.
        idx_t _min_exterior = 32;   // minimum size of MPI exterior to calculate.
        #ifdef USE_OFFLOAD
        bool use_device_mpi = true; // transfer data directly between devices.
        bool use_shm = false;       // transfer data using shared memory (w/o MPI calls) on same node.
        #else
        bool use_device_mpi = false;
        bool use_shm = true;
        #endif

        // OpenMP settings.
        int max_threads = 0;      // Initial number of host threads to use overall; 0=>OMP default.
        int num_outer_threads = 0; // Number of threads to use for blocks.
        int num_inner_threads = 1; // Number of threads to use within a block.
        bool bind_inner_threads = false; // Bind inner threads to global indices.
        #ifdef USE_OFFLOAD
        int thread_limit = 32;           // Offload threads per team.
        #else
        int thread_limit = 1;
        #endif

        // Var behavior, including allocation.
        bool _step_wrap = false; // Allow invalid step indices to alias to valid ones (set via APIs only).
        bool _allow_addl_pad = true; // Allow extending padding beyond what's needed for alignment.
        bool _bundle_allocs = !KernelEnv::_use_offload; // Group allocations together.
        int _numa_pref = NUMA_PREF;
        #ifdef USE_PMEM
        int _numa_pref_max = 128; // GiB to alloc before using PMEM.
        #endif

        // Stencil-dim posn in which to apply block-thread binding.
        // TODO: make this a cmd-line parameter.
        int _bind_posn = 1;

        // Tuning.
        bool _do_auto_tune = false;    // whether to do "online" auto-tuning.
        bool _allow_stage_tuners = false; // allow per-stage tuners when possible.
        double _tuner_trial_secs = 0.5;   // time to run tuner for new better setting.
        int _tuner_radius = 16;
        string_vec _tuner_targets; // things to tune from following.

        // Debug.
        bool force_scalar = false; // Do only scalar ops.
        bool do_halo_exchange = true; // False => skip halo exchanges.
        bool force_scalar_exchange = false; // Don't allow vec exchanges.

        // Ctor/dtor.
        KernelSettings(DimsPtr dims, KernelEnvPtr env);
        virtual ~KernelSettings() { }

    protected:
        // Add options to set one domain var to a cmd-line parser.
        virtual void _add_domain_option(CommandLineParser& parser,
                                        const std::string& prefix,
                                        const std::string& descrip,
                                        IdxTuple& var,
                                        bool allow_step = false);

    public:
        // Add options to a cmd-line parser to set the settings.
        virtual void add_options(CommandLineParser& parser);

        // Print informational messages.
        void print_usage(std::ostream& os);
        void print_values(std::ostream& os);
        
        // Make sure all user-provided settings are valid by rounding-up
        // values as needed.  Called from prepare_solution().
        // Prints informational info to debug output in *ksb.
        virtual void adjust_settings(KernelStateBase* ksb = 0);

        // Determine if this is the first or last rank in given dim.
        bool is_first_rank(const std::string dim) {
            return _rank_indices[dim] == 0;
        }
        bool is_last_rank(const std::string dim) {
            return _rank_indices[dim] == _num_ranks[dim] - 1;
        }
    };
    typedef std::shared_ptr<KernelSettings> KernelSettingsPtr;

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
        // Example: size = 3^3 = 27 for 3D problem.
        // NB: this is the *max* number of neighbors, not necessarily the actual number.
        idx_t neighborhood_size = 0;

        // What get_neighbor_index() returns for myself.
        // Example: trunc(3^3 / 2) = 13 for 3D problem.
        idx_t my_neighbor_index;

        // MPI rank of each neighbor.
        // MPI_PROC_NULL => no neighbor.
        // Vector index is per get_neighbor_index().
        typedef std::vector<int> Neighbors;
        Neighbors my_neighbors;

        // Manhattan distance to each neighbor.
        // Vector index is per get_neighbor_index().
        std::vector<int> man_dists;

        // Whether each neighbor has all its rank-domain
        // sizes as a multiple of the vector length.
        std::vector<bool> has_all_vlen_mults;

        // Rank number in KernelEnv::shmcomm if this neighbor
        // can communicate with shm. MPI_PROC_NULL otherwise.
        std::vector<int> shm_ranks;

        // Window for halo buffers.
        MPI_Win halo_win;

        // Shm halo buffers for each neighbor.
        std::vector<void*> halo_buf_ptrs;
        std::vector<size_t> halo_buf_sizes;

        // Ctor based on pre-set problem dimensions.
        MPIInfo(DimsPtr dims) : _dims(dims) {

            // Max neighbors.
            neighborhood_sizes = dims->_domain_dims; // copy dims from domain.
            neighborhood_sizes.set_vals_same(num_offsets); // set sizes in each domain dim.
            neighborhood_size = neighborhood_sizes.product(); // num neighbors in all dims.

            // Myself.
            IdxTuple noffsets(neighborhood_sizes);
            noffsets.set_vals_same(rank_self);
            my_neighbor_index = get_neighbor_index(noffsets);

            // Init arrays.
            my_neighbors.resize(neighborhood_size, MPI_PROC_NULL);
            man_dists.resize(neighborhood_size, 0);
            has_all_vlen_mults.resize(neighborhood_size, false);
            shm_ranks.resize(neighborhood_size, MPI_PROC_NULL);
            halo_buf_ptrs.resize(neighborhood_size, 0);
            halo_buf_sizes.resize(neighborhood_size, 0);
        }
        virtual ~MPIInfo() { }

        // Get a 1D index for a neighbor.
        // Input 'offsets': tuple of NeighborOffset vals.
        virtual idx_t get_neighbor_index(const IdxTuple& offsets) const {
            idx_t i = neighborhood_sizes.layout(offsets); // 1D index.
            host_assert(i >= 0);
            host_assert(i < neighborhood_size);
            return i;
        }

        // Visit all neighbors.
        // Does NOT visit self.
        virtual void visit_neighbors(std::function<void
                                    (const IdxTuple& offsets, // NeighborOffset vals.
                                     int rank, // MPI rank; might be MPI_PROC_NULL.
                                     int index // simple counter from 0.
                                     )> visitor);
    };
    typedef std::shared_ptr<MPIInfo> MPIInfoPtr;

    // MPI data for one buffer for one neighbor of one var.
    class MPIBuf {

        // Ptr to read/write lock when buffer is in shared mem.
        SimpleLock* _shm_lock = 0;

    public:

        // Descriptive name.
        std::string name;

        // Send or receive buffer.
        std::shared_ptr<char> _base;
        real_t* _elems = 0;

        // Range to copy to/from var.
        // NB: step index not set properly for vars with step dim.
        IdxTuple begin_pt, last_pt;

        // Number of points to copy to/from var in each dim.
        IdxTuple num_pts;

        // Whether the number of points is a multiple of the
        // vector length in all dims and buffer is aligned.
        bool vec_copy_ok = false;

        // Safe access to lock.
        void shm_lock_init() {
            if (_shm_lock)
                _shm_lock->init();
        }
        bool is_ok_to_read() const {
            if (_shm_lock)
                return _shm_lock->is_ok_to_read();
            return true;
        }
        void wait_for_ok_to_read() const {
            if (_shm_lock)
                _shm_lock->wait_for_ok_to_read();
        }
        void mark_read_done() {
            if (_shm_lock)
                _shm_lock->mark_read_done();
        }
        bool is_ok_to_write() const {
            if (_shm_lock)
                return _shm_lock->is_ok_to_write();
            return true;
        }
        void wait_for_ok_to_write() const {
            if (_shm_lock)
                _shm_lock->wait_for_ok_to_write();
        }
        void mark_write_done() {
            if (_shm_lock)
                _shm_lock->mark_write_done();
        }
        idx_t get_data() const {
            if (_shm_lock)
                return _shm_lock->get_data();
            return 0;
        }
        void set_data(idx_t v) {
            if (_shm_lock)
                _shm_lock->set_data(v);
        }

        // Number of points overall.
        idx_t get_size() const {
            if (num_pts.size() == 0)
                return 0;
            return num_pts.product();
        }
        size_t get_bytes() const {
            return get_size() * sizeof(real_t);
        }

        // Set pointer to storage.
        // Free old storage.
        // 'base' should provide get_num_bytes() bytes at offset bytes.
        // Returns raw pointer.
        void* set_storage(std::shared_ptr<char>& base, size_t offset);

        // Same as above, but does not maintain shared storage.
        void* set_storage(char* base, size_t offset);

        // Release storage.
        void release_storage() {
            _base.reset();
            _elems = 0;
            _shm_lock = 0;
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

    // MPI data for both buffers for one neighbor of one var.
    struct MPIBufs {

        // Need one buf for send and one for receive for each neighbor.
        enum BufDir { buf_send, buf_recv, n_buf_dirs };

        MPIBuf bufs[n_buf_dirs];

        // Reset lock for send buffer.
        // Another rank owns recv buffer.
        void reset_locks() {
            bufs[buf_send].shm_lock_init();
        }
    };

    // MPI data for one var.
    // Contains a send and receive buffer for each neighbor
    // and some meta-data.
    struct MPIData {

        MPIInfoPtr _mpi_info;

        // Buffers for all possible neighbors.
        typedef std::vector<MPIBufs> NeighborBufs;
        NeighborBufs bufs;

        // Arrays for request handles.
        // These are used for async comms.
        std::vector<MPI_Request> recv_reqs;
        std::vector<MPI_Request> send_reqs;
        std::vector<MPI_Status> recv_stats;
        std::vector<MPI_Status> send_stats;

        MPIData(MPIInfoPtr mpi_info) :
            _mpi_info(mpi_info) {

            // Init vector of buffers.
            auto n = _mpi_info->neighborhood_size;
            MPIBufs empty_bufs;
            bufs.resize(n, empty_bufs);

            // Init handles.
            recv_reqs.resize(n, MPI_REQUEST_NULL);
            send_reqs.resize(n, MPI_REQUEST_NULL);
            MPI_Status nullst;
            memset(&nullst, 0, sizeof(nullst));
            recv_stats.resize(n, nullst);
            send_stats.resize(n, nullst);
        }

        void reset_locks() {
            for (auto& mb : bufs)
                mb.reset_locks();
        }

        // Apply a function to each neighbor rank.
        // Called visitor function will contain the rank index of the neighbor.
        virtual void visit_neighbors(std::function<void (const IdxTuple& neighbor_offsets, // NeighborOffset.
                                                        int rank,
                                                        int index, // simple counter from 0.
                                                        MPIBufs& bufs)> visitor);

        // Access a buffer by direction and neighbor offsets.
        virtual MPIBuf& get_buf(MPIBufs::BufDir bd, const IdxTuple& neighbor_offsets);
    };

    // A collection of solution meta-data whose ownership is shared between
    // various objects.
    // This is not a virtual class.
    struct KernelState {

        // Environment (mostly MPI).
        KernelEnvPtr _env;

        // User settings.
        KernelSettingsPtr _actl_opts; // Actual settings to use.
        KernelSettingsPtr _req_opts; // Settings specified by user and/or tuner.
        bool _use_stage_tuners = false;

        // Problem dims.
        DimsPtr _dims;

        // MPI neighbor info.
        MPIInfoPtr _mpi_info;
    };
    typedef std::shared_ptr<KernelState> KernelStatePtr;

    // Macro to define and set commonly-needed state vars efficiently.
    // '_ksbp' is pointer to a 'KernelStateBase' object.
    // '*_posn' vars are positions in stencil_dims.
    // It is critical that statements here can be easily optimized
    // away by the compiler if some vars are not needed. Thus,
    // avoid accessing vars in virtual classes or calling any
    // functions with side-effects.
#define STATE_VARS0(_ksbp, pfx)                                         \
    pfx auto* ksbp = _ksbp;                                             \
    host_assert(ksbp);                                                  \
    pfx auto* state = ksbp->get_state().get();                          \
    host_assert(state);                                                 \
    pfx auto* env = state->_env.get();                                  \
    host_assert(env);                                                   \
    pfx auto* actl_opts = state->_actl_opts.get();                      \
    host_assert(actl_opts);                                             \
    pfx auto* req_opts = state->_req_opts.get();                        \
    host_assert(req_opts);                                              \
    pfx auto* dims = state->_dims.get();                                \
    host_assert(dims);                                                  \
    pfx auto* mpi_info = state->_mpi_info.get();                        \
    host_assert(mpi_info);                                              \
    const auto& step_dim = dims->_step_dim;                             \
    const auto& inner_layout_dim = dims->_inner_layout_dim;             \
    const auto& inner_loop_dim = dims->_inner_loop_dim;                 \
    const auto& domain_dims = dims->_domain_dims;                       \
    constexpr int nddims = NUM_DOMAIN_DIMS;                             \
    host_assert(nddims == domain_dims.size());                          \
    const auto& stencil_dims = dims->_stencil_dims;                     \
    constexpr int nsdims = NUM_STENCIL_DIMS;                            \
    host_assert(nsdims == stencil_dims.size());                         \
    auto& misc_dims = dims->_misc_dims;                                 \
    constexpr int step_posn = 0;                                        \
    host_assert(step_posn == +step_posn);
#define STATE_VARS(_ksbp) STATE_VARS0(_ksbp,)
#define STATE_VARS_CONST(_ksbp) STATE_VARS0(_ksbp, const)

    // A base class containing a shared pointer to a kernel state.
    // Used to ensure that the shared state object stays allocated when
    // at least one of its owners exists.
    // This is not a virtual class.
    class KernelStateBase {
    protected:

        // Common state. This is a separate object to allow
        // multiple objects to keep it alive via shared ptrs.
        KernelStatePtr _state;

    public:
        KernelStateBase(KernelStatePtr& state) :
            _state(state) {}
        KernelStateBase(KernelEnvPtr& kenv,
                        KernelSettingsPtr& kactl_opts,
                        KernelSettingsPtr& kreq_opts);
        KernelStateBase(KernelStateBase* p) :
            _state(p->_state) { }

        // Access to state.
        ALWAYS_INLINE KernelStatePtr& get_state() {
            host_assert(_state);
            return _state;
        }
        ALWAYS_INLINE const KernelStatePtr& get_state() const {
            host_assert(_state);
            return _state;
        }
        KernelSettingsPtr& get_actl_opts() { return _state->_actl_opts; }
        const KernelSettingsPtr& get_actl_opts() const { return _state->_actl_opts; }
        KernelSettingsPtr& get_req_opts() { return _state->_req_opts; }
        const KernelSettingsPtr& get_req_opts() const { return _state->_req_opts; }
        KernelEnvPtr& get_env() { return _state->_env; }
        const KernelEnvPtr& get_env() const { return _state->_env; }
        DimsPtr& get_dims() { return _state->_dims; }
        const DimsPtr& get_dims() const { return _state->_dims; }
        MPIInfoPtr& get_mpi_info() { return _state->_mpi_info; }
        const MPIInfoPtr& get_mpi_info() const { return _state->_mpi_info; }
        bool use_stage_tuners() const { return _state->_use_stage_tuners; }
        yask_output_ptr get_debug_output() const {
            return _state->_env->get_debug_output();
        }
        void set_debug_output(yask_output_ptr debug) {
            _state->_env->set_debug_output(debug);
        }
        void disable_debug_output() {
            _state->_env->disable_debug_output();
        }

        // Set number of threads w/o using thread-divisor.
        // Return number of threads.
        // Do nothing and return 0 if not properly initialized.
        int set_max_threads();

        // Get total number of computation threads to use.
        int get_num_comp_threads(int& outer_threads, int& inner_threads) const;

        // Set number of threads to use for a mega-block.
        // Enable nested OMP if there are >1 block threads,
        // disable otherwise.
        // Return number of threads.
        // Do nothing and return 0 if not properly initialized.
        int set_num_outer_threads();

        // Set number of threads for a block.
        // Must be called from within a top-level OMP parallel mega-block.
        // Return number of threads.
        // Do nothing and return 0 if not properly initialized.
        int set_num_inner_threads();
    };

    // An object that is created from a context, shares ownership of the
    // state, and keeps a pointer back to the context. However, it does not
    // share ownership of the context itself. That would create an ownership
    // loop that would not allow the context to be deleted.
    // This is not a virtual class.
    class ContextLinker :
        public KernelStateBase {

    protected:
        StencilContext* _context;

    public:
        ContextLinker(StencilContext* context);
    };

} // yask namespace.
