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

#ifndef STENCIL_CALC
#define STENCIL_CALC

namespace yask {

    // Forward defns.
    struct StencilContext;

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
        int neighborhood_size = 0;

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

        // A type to store buffers for all possible neighbors.
        typedef std::vector<YkGridPtr> NeighborBufs;
        NeighborBufs send_bufs, recv_bufs;

        MPIBufs(MPIInfoPtr mpiInfo) :
            _mpiInfo(mpiInfo) {

            // Init buffer pointers.
            send_bufs.insert(send_bufs.begin(), _mpiInfo->neighborhood_size, NULL);
            recv_bufs.insert(recv_bufs.begin(), _mpiInfo->neighborhood_size, NULL);
        }

        // Apply a function to each neighbor rank.
        // Called visitor function will contain the rank index of the neighbor.
        virtual void visitNeighbors(std::function<void (const IdxTuple& offsets, // NeighborOffset.
                                                        int rank,
                                                        int index, // simple counter from 0.
                                                        YkGridPtr sendBuf,
                                                        YkGridPtr recvBuf)> visitor);
            
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
        idx_t def_steps = 50;
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
    
    // An n-D bounding-box in domain dims.
    struct BoundingBox {

        // Boundaries around all points.
        IdxTuple bb_begin;   // first indices.
        IdxTuple bb_end;     // one past last indices.
        idx_t bb_num_points=1;  // valid points within the box.

        // Following values are calculated from the above ones.
        IdxTuple bb_len;       // size in each dim.
        idx_t bb_size=1;       // points in the entire box >= bb_num_points.
        bool bb_simple=false;  // full box with aligned vectors only.
        bool bb_valid=false;   // lengths and sizes have been calculated.

        // Calc values and set valid to true.
        // If 'force_full', set 'bb_num_points' as well as 'bb_size'.
        void update_bb(std::ostream& os,
                       const std::string& name,
                       StencilContext& context,
                       bool force_full = false);
    };

    // Collections of things in a context.
    class EqGroupBase;
    typedef std::vector<EqGroupBase*> EqGroupList;
    typedef std::set<EqGroupBase*> EqGroupSet;
    typedef std::map<std::string, YkGridPtr> GridPtrMap;
    
    // Data and hierarchical sizes.
    // This is a pure-virtual class that must be implemented
    // for a specific problem.
    // Each eq group is valid within its bounding box (BB).
    // The context's BB encompasses all eq-group BBs.
    class StencilContext :
        public BoundingBox,
        public virtual yk_solution {

    protected:

        // Output stream for messages.
        std::ostream* _ostr = 0;

        // Env.
        KernelEnvPtr _env;

        // Command-line and env parameters.
        KernelSettingsPtr _opts;

        // Problem dims.
        DimsPtr _dims;

        // MPI info.
        MPIInfoPtr _mpiInfo;
        
        // Bytes between each buffer to help avoid aliasing
        // in the HW.
        size_t _data_buf_pad = (YASK_PAD * CACHELINE_BYTES);

        // Check whether dim is appropriate type.
        virtual void checkDimType(const std::string& dim,
                                  const std::string& fn_name,
                                  bool step_ok,
                                  bool domain_ok,
                                  bool misc_ok) const {
            _dims->checkDimType(dim, fn_name, step_ok, domain_ok, misc_ok);
        }
        
    public:

        // Name.
        std::string name;

        // Step dim is always in [0] of an Indices type.
        static const int _step_posn = 0;
        
        // List of all stencil equations in the order in which
        // they should be evaluated. Current assumption is that
        // later ones are dependent on their predecessors.
        // TODO: relax this assumption, determining which eqGroups
        // are actually dependent on which others, allowing
        // more parallelism.
        EqGroupList eqGroups;

        // All grids.
        GridPtrs gridPtrs;
        GridPtrMap gridMap;

        // Only grids that are updated by the stencil equations.
        GridPtrs outputGridPtrs;
        GridPtrMap outputGridMap;

        // Some calculated domain sizes.
        IdxTuple rank_domain_offsets;       // Domain index offsets for this rank.
        IdxTuple overall_domain_sizes;       // Total of rank domains over all ranks.

        // Maximum halos and skewing angles over all grids and
        // equations. Used for calculating worst-case minimum regions.
        IdxTuple max_halos;  // spatial halos.
        IdxTuple angles;     // temporal skewing angles.

        // Various metrics calculated in prepare_solution().
        // 'rank_' prefix indicates for this rank.
        // 'tot_' prefix indicates over all ranks.
        // 'domain' indicates points in domain-size specified on cmd-line.
        // 'numpts' indicates points actually calculated in sub-domains.
        // 'reads' indicates points actually read by eq-groups.
        // 'numFpOps' indicates est. number of FP ops.
        // 'nbytes' indicates number of bytes allocated.
        // '_1t' suffix indicates work for one time-step.
        // '_dt' suffix indicates work for all time-steps.
        idx_t rank_domain_1t=0, rank_domain_dt=0, tot_domain_1t=0, tot_domain_dt=0;
        idx_t rank_numpts_1t=0, rank_numpts_dt=0, tot_numpts_1t=0, tot_numpts_dt=0;
        idx_t rank_reads_1t=0, rank_reads_dt=0, tot_reads_1t=0, tot_reads_dt=0;
        idx_t rank_numFpOps_1t=0, rank_numFpOps_dt=0, tot_numFpOps_1t=0, tot_numFpOps_dt=0;
        idx_t rank_nbytes=0, tot_nbytes=0;
        
        // MPI settings.
        double mpi_time=0.0;          // time spent doing MPI.

        // Actual MPI buffers.
        // MPI buffers are tagged by their grid names.
        std::map<std::string, MPIBufs> mpiBufs;
        
        // Constructor.
        StencilContext(KernelEnvPtr env,
                       KernelSettingsPtr settings) :
            _ostr(&std::cout),
            _env(env),
            _opts(settings),
            _dims(settings->_dims)
        {
            _mpiInfo = std::make_shared<MPIInfo>(settings->_dims);

            // Init various tuples to make sure they have the correct dims.
            rank_domain_offsets = _dims->_domain_dims;
            overall_domain_sizes = _dims->_domain_dims;
            max_halos = _dims->_domain_dims;
            angles = _dims->_domain_dims;
            
            // Set output to msg-rank per settings.
            set_ostr();
        }

        // Destructor.
        virtual ~StencilContext() { }

        // Set ostr to given stream if provided.
        // If not provided, set to cout if my_rank == msg_rank
        // or a null stream otherwise.
        virtual std::ostream& set_ostr(std::ostream* os = NULL);

        // Get the messsage output stream.
        virtual std::ostream& get_ostr() const {
            assert(_ostr);
            return *_ostr;
        }

        // Access to settings.
        virtual KernelSettingsPtr get_settings() {
            assert(_opts);
            return _opts;
        }
        virtual void set_settings(KernelSettingsPtr opts) {
            _opts = opts;
        }

        // Access to dims and MPI info.
        virtual DimsPtr get_dims() {
            return _dims;
        }
        virtual MPIInfoPtr get_mpi_info() {
            return _mpiInfo;
        }

        // Add a new grid to the containers.
        virtual void addGrid(YkGridPtr gp, bool is_output);
        
        // Set vars related to this rank's role in global problem.
        // Allocate MPI buffers as needed.
        // Called from prepare_solution(), so it doesn't normally need to be called from user code.
        virtual void setupRank();

        // Allocate grid, param, and MPI memory.
        // Called from prepare_solution(), so it doesn't normally need to be called from user code.
        virtual void allocData();

        // Allocate grids, params, MPI bufs, etc.
        // Calculate rank position in problem.
        // Initialize some other data structures.
        // Print lots of stats.
        virtual void prepare_solution();

        // Set grid sizes and offsets.
        // This should be called anytime a setting or offset is changed.
        virtual void update_grids();
        
        // Get total memory allocation required by grids.
        // Does not include MPI buffers.
        // TODO: add MPI buffers.
        virtual size_t get_num_bytes() {
            size_t sz = 0;
            for (auto gp : gridPtrs)
                if (gp)
                    sz += gp->get_num_storage_bytes() + _data_buf_pad;
            return sz;
        }

        // Init all grids & params by calling initFn.
        virtual void initValues(std::function<void (YkGridPtr gp,
                                                    real_t seed)> realInitFn);

        // Init all grids & params to same value within grids,
        // but different for each grid.
        virtual void initSame() {
            initValues([&](YkGridPtr gp, real_t seed){ gp->set_all_elements_same(seed); });
        }

        // Init all grids & params to different values within grids,
        // and different for each grid.
        virtual void initDiff() {
            initValues([&](YkGridPtr gp, real_t seed){ gp->set_all_elements_in_seq(seed); });
        }

        // Init all grids & params.
        // By default it uses the initSame initialization routine.
        virtual void initData() {
            initSame();
        }

        // Compare grids in contexts for validation.
        // Params should not be written to, so they are not compared.
        // Return number of mis-compares.
        virtual idx_t compareData(const StencilContext& ref) const;
        
        // Set number of threads to use for something other than a region.
        virtual int set_all_threads() {

            // Get max number of threads.
            int mt = _opts->max_threads;
            if (!mt)
                mt = omp_get_max_threads();
            
            // Reset number of OMP threads to max allowed.
            int nt = _opts->max_threads / _opts->thread_divisor;
            nt = std::max(nt, 1);
            TRACE_MSG("set_all_threads: omp_set_num_threads=" << nt);
            omp_set_num_threads(nt);
            return nt;
        }

        // Set number of threads to use for a region.
        // Return that number.
        virtual int set_region_threads() {

            // Start with "all" threads.
            int nt = _opts->max_threads / _opts->thread_divisor;

            // Limit outer nesting to allow num_block_threads per nested
            // block loop.
            nt /= _opts->num_block_threads;
            nt = std::max(nt, 1);
            if (_opts->num_block_threads > 1)
                omp_set_nested(1);

            TRACE_MSG("set_region_threads: omp_set_num_threads=" << nt);
            omp_set_num_threads(nt);
            return nt;
        }

        // Set number of threads for a block.
        // Return that number.
        virtual int set_block_threads() {

            // This should be a nested OMP region.
            int nt = _opts->num_block_threads;
            nt = std::max(nt, 1);
            TRACE_MSG("set_block_threads: omp_set_num_threads=" << nt);
            omp_set_num_threads(nt);
            return nt;
        }

        // Reference stencil calculations.
        virtual void calc_rank_ref();

        // Vectorized and blocked stencil calculations.
        virtual void calc_rank_opt();

        // Calculate results within a region.  Boundaries are named start_d*
        // and stop_d* because region loops are nested inside the
        // rank-domain loops; the actual begin_r* and end_r* values for the
        // region are derived from these.  TODO: create a public interface
        // w/a more logical index ordering.
        virtual void calc_region(EqGroupSet* eqGroup_set,
                                 const ScanIndices& rank_idxs);

        // Exchange halo data needed by eq-group 'eg' at the given time.
        virtual void exchange_halos(idx_t start_dt, idx_t stop_dt, EqGroupBase& eg);

        // Mark grids that have been written to by eq-group 'eg'.
        virtual void mark_grids_dirty(EqGroupBase& eg);
        
        // Set the bounding-box around all eq groups.
        virtual void find_bounding_boxes();

        virtual YkGridPtr newGrid(const std::string& name,
                                  const std::vector<std::string>& dims,
                                  bool is_visible = true);

        // APIs.
        // See yask_kernel_api.hpp.
        virtual void set_debug_output(yask_output_ptr debug) {
#warning FIXME: keep copy of shared ptr
            set_ostr(&debug->get_ostream());
        }
        virtual const std::string& get_name() const {
            return name;
        }
        virtual int get_element_bytes() const {
            return REAL_BYTES;
        }

        virtual int get_num_grids() const {
            return int(gridPtrs.size());
        }

        virtual yk_grid_ptr get_grid(int n) {
            assert (n >= 0);
            assert (n < get_num_grids());
            auto new_ptr = YkGridPtr(gridPtrs.at(n)); // shares ownership.
            return new_ptr;
        }
        virtual std::vector<yk_grid_ptr> get_grids() {
            std::vector<yk_grid_ptr> grids;
            for (int i = 0; i < get_num_grids(); i++)
                grids.push_back(get_grid(i));
            return grids;
        }
        virtual yk_grid_ptr
        new_grid(const std::string& name,
                 const std::vector<std::string>& dims) {
            return newGrid(name, dims);
        }
        virtual yk_grid_ptr
        new_grid(const std::string& name,
                 const std::string& dim1 = "",
                 const std::string& dim2 = "",
                 const std::string& dim3 = "",
                 const std::string& dim4 = "",
                 const std::string& dim5 = "",
                 const std::string& dim6 = "");

        virtual std::string get_step_dim_name() const {
            return _dims->_step_dim;
        }
        virtual int get_num_domain_dims() const {

            // TODO: remove hard-coded assumptions.
            return 3;
        }
        virtual std::string get_domain_dim_name(int n) const;
        virtual std::vector<std::string> get_domain_dim_names() const {
            std::vector<std::string> dims;
            for (auto& dim : _dims->_domain_dims.getDims())
                dims.push_back(dim.getName());
            return dims;
        }
        virtual std::vector<std::string> get_misc_dim_names() const {
            std::vector<std::string> dims;
            for (auto& dim : _dims->_misc_dims.getDims())
                dims.push_back(dim.getName());
            return dims;
        }

        virtual idx_t get_first_rank_domain_index(const std::string& dim) const;
        virtual idx_t get_last_rank_domain_index(const std::string& dim) const;
        virtual idx_t get_overall_domain_size(const std::string& dim) const;

        virtual void run_solution(idx_t first_step_index,
                                  idx_t last_step_index);
        virtual void run_solution(idx_t step_index) {
            run_solution(step_index, step_index);
        }
        virtual void share_grid_storage(yk_solution_ptr source);

        // APIs that access settings.
        virtual void set_rank_domain_size(const std::string& dim,
                                     idx_t size);
        virtual void set_min_pad_size(const std::string& dim,
                                      idx_t size);
        virtual void set_block_size(const std::string& dim,
                                    idx_t size);
        virtual void set_num_ranks(const std::string& dim,
                                   idx_t size);
        virtual idx_t get_rank_domain_size(const std::string& dim) const;
        virtual idx_t get_min_pad_size(const std::string& dim) const;
        virtual idx_t get_block_size(const std::string& dim) const;
        virtual idx_t get_num_ranks(const std::string& dim) const;
        virtual idx_t get_rank_index(const std::string& dim) const;
    };
    
    /// Classes that support evaluation of one stencil equation-group.
    /// A context contains of one or more equation-groups.

    // Types of dependencies.
    enum DepType {
        certain_dep,
        possible_dep,
        num_deps
    };

    // A pure-virtual class base for a stencil equation-set.
    class EqGroupBase : public BoundingBox {
    protected:
        StencilContext* _generic_context = 0;
        std::string _name;
        int _scalar_fp_ops = 0;
        int _scalar_points_read = 0;
        int _scalar_points_written = 0;

        // Eq-groups that this one depends on.
        std::map<DepType, EqGroupSet> _depends_on;
        
    public:

        // Grids that are written to by these stencil equations.
        GridPtrs outputGridPtrs;

        // Grids that are read by these stencil equations (not necessarify
        // read-only, i.e., a grid can be input and output).
        GridPtrs inputGridPtrs;

        // ctor, dtor.
        EqGroupBase(StencilContext* context) :
            _generic_context(context) {

            // Make sure map entries exist.
            for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1)) {
                _depends_on[dt];
            }
        }
        virtual ~EqGroupBase() { }

        // Get name of this equation set.
        virtual const std::string& get_name() { return _name; }

        // Get estimated number of FP ops done for one scalar eval.
        virtual int get_scalar_fp_ops() { return _scalar_fp_ops; }

        // Get number of points read and written for one scalar eval.
        virtual int get_scalar_points_read() const { return _scalar_points_read; }
        virtual int get_scalar_points_written() const { return _scalar_points_written; }

        // Add dependency.
        virtual void add_dep(DepType dt, EqGroupBase* eg) {
            _depends_on.at(dt).insert(eg);
        }

        // Get dependencies.
        virtual const EqGroupSet& get_deps(DepType dt) const {
            return _depends_on.at(dt);
        }
    
        // Set the bounding-box vars for this eq group in this rank.
        virtual void find_bounding_box();

        // Determine whether indices are in [sub-]domain.
        virtual bool
        is_in_valid_domain(const Indices& idxs) =0;

        // Calculate one scalar result at time t.
        virtual void
        calc_scalar(const Indices& idxs) =0;

        // Calculate results within a block.
        virtual void
        calc_block(const ScanIndices& region_idxs);

        // Calculate results within a sub-block.
        // Each block is typically computed in a separate OpenMP thread.
        virtual void
        calc_sub_block(const ScanIndices& block_idxs);

        // Calculate whole-cluster results within a sub-block.
        // Indices must be rank-relative.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void
        calc_sub_block_of_clusters(const ScanIndices& block_idxs) =0;
    };

} // yask namespace.

#endif
