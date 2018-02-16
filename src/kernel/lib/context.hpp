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

    // An n-D bounding-box in domain dims.
    struct BoundingBox {

        // Boundaries around all points.
        IdxTuple bb_begin;   // first indices.
        IdxTuple bb_end;     // one past last indices.
        idx_t bb_num_points=1;  // valid points within the box.

        // Following values are calculated from the above ones.
        IdxTuple bb_len;       // size in each dim.
        idx_t bb_size=1;       // points in the entire box >= bb_num_points.
        bool bb_is_full=false; // all points in box are calculated.
        bool bb_is_aligned=false; // starting points are aligned in all dims.
        bool bb_is_cluster_mult=false; // num points are cluster multiples in all dims.
        bool bb_valid=false;   // lengths and sizes have been calculated.

        // Calc values and set valid to true.
        // If 'force_full', set 'bb_num_points' as well as 'bb_size'.
        void update_bb(std::ostream& os,
                       const std::string& name,
                       StencilContext& context,
                       bool force_full = false);
    };

    // Stats.
    class Stats : public virtual yk_stats {
    public:
        idx_t npts = 0;
        idx_t nwrites = 0;
        idx_t nfpops = 0;
        idx_t nsteps = 0;
        double run_time = 0.;
        double mpi_time = 0.;

        Stats() {}
        virtual ~Stats() {}

        void clear() {
            npts = nwrites = nfpops = nsteps = 0;
            run_time = mpi_time = 0.;
        }
        
        // APIs.
        
        /// Get the number of points in the overall domain.
        virtual idx_t
        get_num_elements() { return npts; }

        /// Get the number of points written in each step.
        virtual idx_t
        get_num_writes() { return nwrites; }

        /// Get the estimated number of floating-point operations required for each step.
        virtual idx_t
        get_est_fp_ops() { return nfpops; }

        /// Get the number of steps calculated via run_solution().
        virtual idx_t
        get_num_steps_done() { return nsteps; }

        /// Get the number of seconds elapsed during calls to run_solution().
        virtual double
        get_elapsed_run_secs() { return run_time; }
        
    };
    
    // Collections of things in a context.
    class StencilGroupBase;
    typedef std::vector<StencilGroupBase*> StencilGroupList;
    typedef std::set<StencilGroupBase*> StencilGroupSet;
    typedef std::map<std::string, YkGridPtr> GridPtrMap;
    
    // Data and hierarchical sizes.
    // This is a pure-virtual class that must be implemented
    // for a specific problem.
    // Each stencil group is valid within its bounding box (BB).
    // The context's BB encompasses all stencil-group BBs.
    class StencilContext :
        public BoundingBox,
        public virtual yk_solution {

    protected:

        // Output stream for messages.
        std::ostream* _ostr = 0;
        yask_output_ptr _debug;

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

        // List of all stencil groups in the order in which
        // they should be evaluated within a step.
        // TODO: use dependency info, allowing more parallelism.
        StencilGroupList stGroups;

        // All grids.
        GridPtrs gridPtrs;
        GridPtrMap gridMap;

        // Only grids that are updated by the stencils.
        GridPtrs outputGridPtrs;
        GridPtrMap outputGridMap;

        // Scratch grids.
        ScratchVecs scratchVecs;

        // Some calculated domain sizes.
        IdxTuple rank_domain_offsets;       // Domain index offsets for this rank.
        IdxTuple overall_domain_sizes;       // Total of rank domains over all ranks.

        // Maximum halos and skewing angles over all grids and
        // groups. Used for calculating worst-case minimum regions.
        IdxTuple max_halos;  // spatial halos.
        IdxTuple angles;     // temporal skewing angles.

        // Various amount-of-work metrics calculated in prepare_solution().
        // 'rank_' prefix indicates for this rank.
        // 'tot_' prefix indicates over all ranks.
        // 'domain' indicates points in domain-size specified on cmd-line.
        // 'numpts' indicates points actually calculated in sub-domains.
        // 'reads' indicates points actually read by stencil-groups.
        // 'numFpOps' indicates est. number of FP ops.
        // 'nbytes' indicates number of bytes allocated.
        // '_1t' suffix indicates work for one time-step.
        // '_dt' suffix indicates work for all time-steps.
        idx_t rank_domain_1t=0, rank_domain_dt=0, tot_domain_1t=0, tot_domain_dt=0;
        idx_t rank_numWrites_1t=0, rank_numWrites_dt=0, tot_numWrites_1t=0, tot_numWrites_dt=0;
        idx_t rank_reads_1t=0, rank_reads_dt=0, tot_reads_1t=0, tot_reads_dt=0;
        idx_t rank_numFpOps_1t=0, rank_numFpOps_dt=0, tot_numFpOps_1t=0, tot_numFpOps_dt=0;
        idx_t rank_nbytes=0, tot_nbytes=0;

        // Elapsed-time tracking.
        YaskTimer run_time;     // time in run_solution(), including MPI.
        YaskTimer mpi_time;     // time spent just doing MPI.
        idx_t steps_done = 0;   // number of steps that have been run.
        double domain_pts_ps = 0.; // points-per-sec in domain.
        double writes_ps = 0.;     // writes-per-sec.
        double flops = 0.;      // est. FLOPS.
        
        // MPI settings.
        // TODO: move to settings or MPI info object.
        bool allow_vec_exchange = true; // allow vectorized halo exchange.
        bool enable_halo_exchange = true;

        // MPI data for each grid.
        // Map key: grid name.
        std::map<std::string, MPIData> mpiData;

        // Auto-tuner state.
        class AT {
            StencilContext* _context = 0;
            
            // Null stream to throw away debug info.
            yask_output_ptr nullop;

            // Whether to print progress.
            bool verbose = false;
            
            // AT parameters.
            double warmup_steps = 100;
            double warmup_secs = 1.;
            idx_t min_steps = 50;
            double min_secs = 0.1; // eval when either min_steps or min_secs is reached.
            idx_t min_step = 4;
            idx_t max_radius = 64;
            idx_t min_pts = 512; // 8^3.
            idx_t min_blks = 4;

            // Results.
            std::map<IdxTuple, double> results;
            int n2big = 0, n2small = 0;

            // Best so far.
            IdxTuple best_block;
            double best_rate = 0.;

            // Current point in search.
            IdxTuple center_block;
            idx_t radius = 0;
            bool done = false;
            idx_t neigh_idx = 0;
            bool better_neigh_found = false;

            // Cumulative vars.
            double ctime = 0.;
            idx_t csteps = 0;
            bool in_warmup = true;

        public:
            AT(StencilContext* ctx) :
                _context(ctx) { }
            
            // Reset all state to beginning.
            void clear(bool mark_done, bool verbose = false);

            // Evaluate the previous run and take next auto-tuner step.
            void eval(idx_t steps, double elapsed_time);

            // Apply settings.
            void apply() {
                auto _opts = _context->_opts;
                auto _env = _context->_env;

                // Change sub-block size to 0 so adjustSettings()
                // will set it to the default.
                _opts->_sub_block_sizes.setValsSame(0);
                _opts->_sub_block_group_sizes.setValsSame(0);
                
                // Make sure everything is resized based on block size.
                _opts->adjustSettings(nullop->get_ostream(), _env);

                // Reallocate scratch data based on block size.
                _context->allocScratchData(nullop->get_ostream());
            }

            // Done?
            bool is_done() { return done; }
        };
        AT _at;
        
        // Constructor.
        StencilContext(KernelEnvPtr env,
                       KernelSettingsPtr settings) :
            _ostr(&std::cout),
            _env(env),
            _opts(settings),
            _dims(settings->_dims),
            _at(this)
        {
            yask_output_factory yof;
            set_debug_output(yof.new_stdout_output());

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
        virtual ~StencilContext() {

            // Dump stats if get_stats() hasn't been called yet.
            if (steps_done)
                get_stats();

            // Free mem, reset threads, etc.
            end_solution();
        }

        // Set debug output to cout if my_rank == msg_rank
        // or a null stream otherwise.
        virtual std::ostream& set_ostr();

        // Get the messsage output stream.
        virtual std::ostream& get_ostr() const {
            assert(_ostr);
            return *_ostr;
        }

        // Reset elapsed times to zero.
        virtual void clear_timers() {
            run_time.clear();
            mpi_time.clear();
            steps_done = 0;
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
        virtual void addScratch(GridPtrs& scratch_vec) {
            scratchVecs.push_back(&scratch_vec);
        }
        
        // Set vars related to this rank's role in global problem.
        // Allocate MPI buffers as needed.
        virtual void setupRank();

        // Allocate grid and MPI memory.
        virtual void allocData(std::ostream& os);

        // Allocate scratch-grid memory.
        virtual void allocScratchData(std::ostream& os);

        // Allocate grids, params, MPI bufs, etc.
        // Calculate rank position in problem.
        // Initialize some other data structures.
        // Print lots of stats.
        virtual void prepare_solution();


        /// Get statistics associated with preceding calls to run_solution().
        /**
           Resets all timers and step counters.
           @returns Pointer to statistics object.
        */
        virtual yk_stats_ptr get_stats();

        // Dealloc grids, etc.
        virtual void end_solution();

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
            for (auto gps : scratchVecs)
                if (gps)
                    for (auto gp : *gps)
                        if (gp)
                            sz += gp->get_num_storage_bytes() + _data_buf_pad;
            return sz;
        }

        // Init all grids & params by calling realInitFn.
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
        
        // Set number of threads w/o using thread-divisor.
        // Return number of threads.
        // Do nothing and return 0 if not properly initialized.
        virtual int set_max_threads() {

            // Get max number of threads.
            int mt = _opts->max_threads;
	    if (!mt)
	      return 0;
            
            // Reset number of OMP threads to max allowed.
            //TRACE_MSG("set_max_threads: omp_set_num_threads=" << nt);
            omp_set_num_threads(mt);
            return mt;
        }

        // Set number of threads to use for something other than a region.
        // Return number of threads.
        // Do nothing and return 0 if not properly initialized.
        virtual int set_all_threads() {

            // Get max number of threads.
            int mt = _opts->max_threads;
	    if (!mt)
	      return 0;
            int nt = mt / _opts->thread_divisor;
            nt = std::max(nt, 1);
            
            // Reset number of OMP threads to max allowed.
            //TRACE_MSG("set_all_threads: omp_set_num_threads=" << nt);
            omp_set_num_threads(nt);
            return nt;
        }

        // Set number of threads to use for a region.
        // Return number of threads.
        // Do nothing and return 0 if not properly initialized.
        virtual int set_region_threads() {

            // Start with "all" threads.
            int mt = _opts->max_threads;
	    if (!mt)
	      return 0;
            int nt = mt / _opts->thread_divisor;
            nt = std::max(nt, 1);

            // Limit outer nesting to allow num_block_threads per nested
            // block loop.
            nt /= _opts->num_block_threads;
            nt = std::max(nt, 1);
            if (_opts->num_block_threads > 1)
                omp_set_nested(1);

            //TRACE_MSG("set_region_threads: omp_set_num_threads=" << nt);
            omp_set_num_threads(nt);
            return nt;
        }

        // Set number of threads for a block.
        // Return number of threads.
        // Do nothing and return 0 if not properly initialized.
        virtual int set_block_threads() {

            // This should be a nested OMP region.
            int nt = _opts->num_block_threads;
            nt = std::max(nt, 1);
            //TRACE_MSG("set_block_threads: omp_set_num_threads=" << nt);
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
        virtual void calc_region(StencilGroupSet* stGroup_set,
                                 const ScanIndices& rank_idxs);

        // Exchange all dirty halo data.
        virtual void exchange_halos_all();

        // Exchange halo data needed by stencil-group 'sg' at the given step(s).
        virtual void exchange_halos(idx_t start, idx_t stop, StencilGroupBase& sg);

        // Mark grids that have been written to by group 'sg'.
        virtual void mark_grids_dirty(StencilGroupBase& sg, idx_t step_idx);
        
        // Set the bounding-box around all eq groups.
        virtual void find_bounding_boxes();

        // Make new scratch grids.
        virtual void makeScratchGrids (int num_threads) =0;
        
        // Make a new grid iff its dims match any in the stencil.
        // Returns pointer to the new grid or nullptr if no match.
        virtual YkGridPtr newStencilGrid (const std::string & name,
                                          const GridDimNames & dims) =0;

        // Make a new grid with 'name' and 'dims'.
        // Set sizes if 'sizes' is non-null.
        virtual YkGridPtr newGrid(const std::string& name,
                                  const GridDimNames& dims,
                                  const GridDimSizes* sizes);

        // Get output object.
        virtual yask_output_ptr get_debug_output() const {
            return _debug;
        }
        
        // APIs.
        // See yask_kernel_api.hpp.
        virtual void set_debug_output(yask_output_ptr debug) {
            _debug = debug;     // to share ownership of referent.
            _ostr = &debug->get_ostream();
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

        virtual yk_grid_ptr get_grid(const std::string& name) {
            auto i = gridMap.find(name);
            if (i != gridMap.end())
                return i->second;
            return nullptr;
        }
        virtual std::vector<yk_grid_ptr> get_grids() {
            std::vector<yk_grid_ptr> grids;
            for (int i = 0; i < get_num_grids(); i++)
                grids.push_back(gridPtrs.at(i));
            return grids;
        }
        virtual yk_grid_ptr
        new_grid(const std::string& name,
                 const GridDimNames& dims) {
            return newGrid(name, dims, NULL);
        }
        virtual yk_grid_ptr
        new_grid(const std::string& name,
                 const std::initializer_list<std::string>& dims) {
            GridDimNames dims2(dims);
            return new_grid(name, dims2);
        }
        virtual yk_grid_ptr
        new_fixed_size_grid(const std::string& name,
                             const GridDimNames& dims,
                             const GridDimSizes& dim_sizes) {
            return newGrid(name, dims, &dim_sizes);
        }
        virtual yk_grid_ptr
        new_fixed_size_grid(const std::string& name,
                             const std::initializer_list<std::string>& dims,
                             const std::initializer_list<idx_t>& dim_sizes) {
            GridDimNames dims2(dims);
            GridDimSizes sizes2(dim_sizes);
            return new_fixed_size_grid(name, dims2, sizes2);
        }

        virtual std::string get_step_dim_name() const {
            return _dims->_step_dim;
        }
        virtual int get_num_domain_dims() const {
            return _dims->_domain_dims.getNumDims();
        }
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
        virtual std::string apply_command_line_options(const std::string& args);

        virtual void reset_auto_tuner(bool enable, bool verbose = false) {
            _at.clear(!enable, verbose);
        }
        virtual void run_auto_tuner_now(bool verbose = true);
        virtual bool is_auto_tuner_enabled() {
            return !_at.is_done();
        }
    };

} // yask namespace.
