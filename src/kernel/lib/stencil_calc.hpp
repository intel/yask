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
    struct KernelEnv;

    // Environmental settings.
    typedef std::shared_ptr<KernelEnv> KernelEnvPtr;
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
    
    // MPI buffers for *one* grid.
    struct MPIBufs {

        // A type to store ranks of all possible neighbors in all
        // directions, including diagonals.
        enum NeighborOffset { rank_prev, rank_self, rank_next, num_neighbors };
        typedef int Neighbors[num_neighbors][num_neighbors][num_neighbors];

        // Neighborhood size includes self.
        static const int neighborhood_size = num_neighbors * num_neighbors * num_neighbors;
        
        // Need one buf for send and one for receive for each neighbor.
        enum BufDir { bufSend, bufRec, nBufDirs };

        // A type to store buffers for all possible neighbors.
        typedef Grid_XYZ* NeighborBufs[nBufDirs][num_neighbors][num_neighbors][num_neighbors];
        NeighborBufs bufs;

        MPIBufs() {
            memset(bufs, 0, sizeof(bufs));
        }

        // Access a buffer by direction and 3D neighbor indices.
        Grid_XYZ** getBuf(int bd, idx_t nx, idx_t ny, idx_t nz) {
            assert(bd >= 0);
            assert(bd < nBufDirs);
            assert(nx >= 0);
            assert(nx < num_neighbors);
            assert(ny >= 0);
            assert(ny < num_neighbors);
            assert(nz >= 0);
            assert(nz < num_neighbors);
            return &bufs[bd][nx][ny][nz];
        }

        // Apply a function to each neighbor rank.
        // Called visitor function will contain the rank index of the neighbor.
        // The send and receive buffer pointers may be null if 'null_ok' is true.
        // TODO: remove 'null_ok' when non-symmetrical halos are supported.
        virtual void visitNeighbors(StencilContext& context,
                                    bool null_ok,
                                    std::function<void (idx_t nx, idx_t ny, idx_t nz,
                                                        int rank,
                                                        Grid_XYZ* sendBuf,
                                                        Grid_XYZ* rcvBuf)> visitor);
            
        // Create new buffer in given direction and size.
        virtual Grid_XYZ* makeBuf(int bd,
                                   idx_t nx, idx_t ny, idx_t nz,
                                   idx_t dx, idx_t dy, idx_t dz,
                                   const std::string& name);
    };

    // Application settings to control size and perf of stencil code.
    class KernelSettings {

    protected:
        const static idx_t def_steps = 50;
        const static idx_t def_rank = 128;
        const static idx_t def_block = 32;

    public:
        // Sizes in elements (points).
        // - time sizes (t) are in steps to be done.
        // - spatial sizes (x, y, z) are in elements (not vectors).
        // Sizes are the same for all grids.
        idx_t dt=def_steps;
        idx_t dx=def_rank, dy=def_rank, dz=def_rank; // rank size (without halos).
        idx_t rt=1, rx=0, ry=0, rz=0; // region size (used for wave-front tiling).
        idx_t bgx=0, bgy=0, bgz=0; // block-group size (only used for 'grouped' region loops).
        idx_t bt=1, bx=def_block, by=def_block, bz=def_block; // block size (used for each outer thread).
        idx_t sbgx=0, sbgy=0, sbgz=0; // sub-block-group size (only used for 'grouped' block loops).
        idx_t sbt=1, sbx=0, sby=0, sbz=0; // sub-block size (used for each nested thread).
        idx_t mpx=0, mpy=0, mpz=0;     // minimum spatial padding.

        // MPI settings.
        idx_t nrx=1, nry=1, nrz=1; // number of ranks in each dim.
        idx_t rix=0, riy=0, riz=0; // my rank index in each dim.
        bool find_loc=true;            // whether my rank index needs to be calculated.
        int msg_rank=0;             // rank that prints informational messages.

        // OpenMP settings.
        int max_threads=0;      // Initial number of threads to use overall; 0=>OMP default.
        int thread_divisor=1;   // Reduce number of threads by this amount.
        int num_block_threads=1; // Number of threads to use for a block.

        // Ctor.
        KernelSettings() { }
        virtual ~KernelSettings() { }
        
        // Add these settigns to a cmd-line parser.
        virtual void add_options(CommandLineParser& parser);

        // Print usage message.
        void print_usage(std::ostream& os,
                         CommandLineParser& parser,
                         const std::string& pgmName,
                         const std::string& appNotes,
                         const std::vector<std::string>& appExamples) const;
        
        // Make sure all user-provided settings are valid by rounding-up
        // values as needed.
        // If 'finalize' is true, also set defaults and print info.
        // Called from prepare_solution(), so it doesn't normally need to be called from user code.
        virtual void adjustSettings(std::ostream& os, bool finalize);

    };
    typedef std::shared_ptr<KernelSettings> KernelSettingsPtr;
    
    // A 4D bounding-box.
    struct BoundingBox {

        // Boundaries around all points.
        idx_t begin_bbx=0, begin_bby=0, begin_bbz=0;
        idx_t end_bbx=1, end_bby=1, end_bbz=1; // one past last value.
        idx_t bb_num_points=1;  // valid points within the box.

        // Following values are calculated from the above ones.
        idx_t len_bbx=1, len_bby=1, len_bbz=1;
        idx_t bb_size=1;        // points in the entire box.
        bool bb_simple=true;    // full box with aligned vectors only.
        bool bb_valid=false;    // lengths and sizes have been calculated.
        
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
    typedef std::shared_ptr<RealVecGridBase> RealVecGridPtr;
    typedef std::vector<RealVecGridPtr> GridPtrs;
    typedef std::set<RealVecGridPtr> GridPtrSet;
    typedef std::vector<RealGrid*> ParamPtrs;
    typedef std::map<std::string, RealVecGridPtr> GridPtrMap;
    typedef std::map<std::string, RealGrid*> ParamPtrMap;
    
    // Data and hierarchical sizes.
    // This is a pure-virtual class that must be implemented
    // for a specific problem.
    // Each eq group is valid within its bounding box (BB).
    // The context's BB encompasses all eq-group BBs.
    class StencilContext :
        public BoundingBox,
        public virtual yk_solution {

    private:
        // Disallow copying.
        StencilContext(const StencilContext& src) {
            exit_yask(1);
        }
        void operator=(const StencilContext& src) {
            exit_yask(1);
        }
        
    protected:
        
        // Output stream for messages.
        std::ostream* _ostr;

        // Env.
        KernelEnvPtr _env;

        // Command-line and env parameters.
        KernelSettingsPtr _opts;

        // Bytes between each buffer to help avoid aliasing
        // in the HW.
        size_t _data_buf_pad = (YASK_PAD * CACHELINE_BYTES);

        // TODO: move vars into private or protected sections and
        // add accessor methods.
    public:

        // Name.
        std::string name;

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

        // Non-grid parameters.
        ParamPtrs paramPtrs;
        ParamPtrMap paramMap;

        // Some calculated sizes.
        idx_t ofs_t=0, ofs_x=0, ofs_y=0, ofs_z=0; // Index offsets for this rank.
        idx_t tot_x=0, tot_y=0, tot_z=0; // Total of rank domains over all ranks.

        // Maximum halos and skewing angles over all grids and
        // equations. Used for calculating worst-case minimum regions.
        idx_t hx=0, hy=0, hz=0;                     // spatial halos.
        idx_t angle_x=0, angle_y=0, angle_z=0; // temporal skewing angles.

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
        MPIBufs::Neighbors my_neighbors;   // neighbor ranks.

        // Actual MPI buffers.
        // MPI buffers are tagged by their grid names.
        std::map<std::string, MPIBufs> mpiBufs;
        
        // Constructor.
        StencilContext(KernelEnvPtr env, KernelSettingsPtr settings) :
            _ostr(&std::cout),
            _env(env),
            _opts(settings)
        {
            
            // Set output to msg-rank.
            set_ostr();

            // Round-up any settings as needed at this point.
            _opts->adjustSettings(get_ostr(), false);
            
            // Init my_neighbors to indicate no neighbor.
            int *p = (int *)my_neighbors;
            for (int i = 0; i < MPIBufs::neighborhood_size; i++)
                p[i] = MPI_PROC_NULL;
        }

        // Destructor.
        virtual ~StencilContext() { }

        // Set ostr to given stream if provided.
        // If not provided, set to cout if my_rank == msg_rank
        // or a null stream otherwise.
        virtual std::ostream& set_ostr(std::ostream* os = NULL);

        // Get the default output stream.
        virtual std::ostream& get_ostr() const {
            assert(_ostr);
            return *_ostr;
        }

        // Get access to settings.
        virtual KernelSettingsPtr get_settings() {
            assert(_opts);
            return _opts;
        }
        virtual void set_settings(KernelSettingsPtr opts) {
            _opts = opts;
        }

        // Add a new grid to the containers.
        virtual void addGrid(RealVecGridPtr gp, bool is_output) {
            gridPtrs.push_back(gp);
            gridMap[gp->get_name()] = gp;
            if (is_output) {
                outputGridPtrs.push_back(gp);
                outputGridMap[gp->get_name()] = gp;
            }
        }
        
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
        virtual void set_grids();
        
        // Get total memory allocation required by grids.
        // Does not include MPI buffers.
        virtual size_t get_num_bytes() {
            size_t sz = 0;
            for (auto gp : gridPtrs)
                if (gp)
                    sz += gp->get_num_bytes() + _data_buf_pad;
            for (auto pp : paramPtrs)
                if (pp)
                    sz += pp->get_num_bytes() + _data_buf_pad;
            return sz;
        }

        // Init all grids & params by calling initFn.
        virtual void initValues(std::function<void (RealVecGridPtr gp, 
                                                    real_t seed)> realVecInitFn,
                                std::function<void (RealGrid* gp,
                                                    real_t seed)> realInitFn);

        // Init all grids & params to same value within grids,
        // but different for each grid.
        virtual void initSame() {
            initValues([&](RealVecGridPtr gp, real_t seed){ gp->set_same(seed); },
                       [&](RealGrid* gp, real_t seed){ gp->set_same(seed); });
        }

        // Init all grids & params to different values within grids,
        // and different for each grid.
        virtual void initDiff() {
            initValues([&](RealVecGridPtr gp, real_t seed){ gp->set_diff(seed); },
                       [&](RealGrid* gp, real_t seed){ gp->set_diff(seed); });
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
        virtual void calc_region(idx_t start_dt, idx_t stop_dt,
                                 EqGroupSet* eqGroup_set,
                                 idx_t start_dx, idx_t start_dy, idx_t start_dz,
                                 idx_t stop_dx, idx_t stop_dy, idx_t stop_dz);

        // Exchange halo data needed by eq-group 'eg' at the given time.
        virtual void exchange_halos(idx_t start_dt, idx_t stop_dt, EqGroupBase& eg);

        // Mark grids that have been written to by eq-group 'eg'.
        virtual void mark_grids_dirty(EqGroupBase& eg);
        
        // Set the bounding-box around all eq groups.
        virtual void find_bounding_boxes();


        // APIs.
        // See yask_kernel_api.hpp.
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
            auto new_ptr = RealVecGridPtr(gridPtrs.at(n)); // shares ownership.
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
                 const std::string& dim1 = "",
                 const std::string& dim2 = "",
                 const std::string& dim3 = "",
                 const std::string& dim4 = "",
                 const std::string& dim5 = "",
                 const std::string& dim6 = "");

        virtual std::string get_step_dim_name() const {
            return STEP_DIM;
        }
        virtual int get_num_domain_dims() const {

            // TODO: remove hard-coded assumptions.
            return 3;
        }
        virtual std::string get_domain_dim_name(int n) const;
        virtual std::vector<std::string> get_domain_dim_names() const {
            std::vector<std::string> dims;
            for (int i = 0; i < get_num_domain_dims(); i++)
                dims.push_back(get_domain_dim_name(i));
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

        // Grids that are read by these stencil equations.
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
        is_in_valid_domain(idx_t t, idx_t x, idx_t y, idx_t z) =0;

        // Calculate one scalar result at time t.
        virtual void
        calc_scalar(idx_t t, idx_t x, idx_t y, idx_t z) =0;

        // Calculate results within a block.
        virtual void
        calc_block(idx_t bt,
                   idx_t begin_bx, idx_t begin_by, idx_t begin_bz,
                   idx_t end_bx, idx_t end_by, idx_t end_bz);

        // Calculate one sub-block of results from begin to end-1 on each dimension.
        // In the 't' dimension, evaluation is at 'sbt' only.
        virtual void
        calc_sub_block(idx_t sbt,
                       idx_t begin_sbx, idx_t begin_sby, idx_t begin_sbz,
                       idx_t end_sbx, idx_t end_sby, idx_t end_sbz);

        // Calculate one sub-block of results in whole clusters from
        // 'begin_sb*v' to 'end_sb*v'-1 on each spatial dimension.  In the
        // time dimension, evaluation is at 'begin_sbtv' only.  All indices
        // are relative to the current rank, i.e., the rank offset is
        // subtracted from the overall index.  Also, all indices are in
        // vectors (hence, the 'v' suffix), i.e., element indices dividied
        // by 'VLEN_*'.
        virtual void
        calc_sub_block_of_clusters(idx_t begin_sbtv,
                                   idx_t begin_sbxv, idx_t begin_sbyv, idx_t begin_sbzv,
                                   idx_t end_sbtv,
                                   idx_t end_sbxv, idx_t end_sbyv, idx_t end_sbzv) =0;
    };

} // yask namespace.

#endif
