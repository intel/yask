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
        idx_t bb_size=1;       // points in the entire box; bb_size >= bb_num_points.
        bool bb_is_full=false; // all points in box are valid (bb_size == bb_num_points).
        bool bb_is_aligned=false; // starting points are vector-aligned in all dims.
        bool bb_is_cluster_mult=false; // num points are cluster multiples in all dims.
        bool bb_valid=false;   // lengths and sizes have been calculated.

        // Calc values and set valid to true.
        // If 'force_full', set 'bb_num_points' as well as 'bb_size'.
        void update_bb(const std::string& name,
                       StencilContext& context,
                       bool force_full,
                       std::ostream* os = NULL);

        // Is point in BB?
        // 'pt' must have same dims as BB.
        bool is_in_bb(const IdxTuple& pt) const {
            assert(pt.getNumDims() == bb_begin.getNumDims());
            for (int i = 0; i < pt.getNumDims(); i++) {
                if (pt[i] < bb_begin[i])
                    return false;
                if (pt[i] >= bb_end[i])
                    return false;
            }
            return true;
        }
        bool is_in_bb(const Indices& pt) const {
            assert(pt.getNumDims() == bb_begin.getNumDims());
            for (int i = 0; i < pt.getNumDims(); i++) {
                if (pt[i] < bb_begin[i])
                    return false;
                if (pt[i] >= bb_end[i])
                    return false;
            }
            return true;
        }
    };

    class BBList : public std::vector<BoundingBox> {

    public:
        BBList() {}
        virtual ~BBList() {}
    };

    // Stats.
    class Stats : public virtual yk_stats {
    public:
        // Steps done.
        idx_t nsteps = 0;

        // Points in domain.
        idx_t npts = 0;

        // Work done.
        idx_t nreads = 0;
        idx_t nwrites = 0;
        idx_t nfpops = 0;

        // Time elapsed.
        double run_time = 0.;   // overall.
        double halo_time = 0.;  // subset in halo.

        // Rates.
        double reads_ps = 0.;     // reads-per-sec.
        double writes_ps = 0.;     // writes-per-sec.
        double flops = 0.;      // est. FLOPS.
        double pts_ps = 0.; // points-per-sec in overall domain.

        Stats() {}
        virtual ~Stats() {}

        void clear() {
            npts = nwrites = nfpops = nsteps = 0;
            run_time = halo_time = 0.;
        }

        // APIs.

        /// Get the number of points in the overall domain.
        virtual idx_t
        get_num_elements() { return npts; }

        /// Get the number of points written.
        virtual idx_t
        get_num_writes_done() { return nwrites; }

        /// Get the estimated number of floating-point operations performed in each step.
        virtual idx_t
        get_est_fp_ops_done() { return nfpops; }

        /// Get the number of steps executed via run_solution().
        virtual idx_t
        get_num_steps_done() { return nsteps; }

        /// Get the number of seconds elapsed during calls to run_solution().
        virtual double
        get_elapsed_secs() { return run_time; }
    };

    // Collections of things in a context.
    class StencilBundleBase;
    class BundlePack;
    typedef std::vector<StencilBundleBase*> StencilBundleList;
    typedef std::set<StencilBundleBase*> StencilBundleSet;
    typedef std::shared_ptr<BundlePack> BundlePackPtr;
    typedef std::vector<BundlePackPtr> BundlePackList;
    typedef std::vector<bool> BridgeMask;

    // Data and hierarchical sizes.
    // This is a pure-virtual class that must be implemented
    // for a specific problem.
    class StencilContext :
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

        // Auto-tuner for global settings.
        AutoTuner _at;
        bool _use_pack_tuners = false;

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

        // Alloc given bytes on each NUMA node.
        virtual void _alloc_data(const std::map <int, size_t>& nbytes,
                                 const std::map <int, size_t>& ngrids,
                                 std::map <int, std::shared_ptr<char>>& _data_buf,
                                 const std::string& type);

    public:

        // Name.
        std::string name;

        // BB without any extensions for wave-fronts.
        // This is the BB for the domain in this rank only.
        BoundingBox rank_bb;

        // BB with any needed extensions for wave-fronts.
        // If WFs are not used, this is the same as 'rank_bb';
        BoundingBox ext_bb;

        // BB of the "interior" of this rank.
        // This is the area that does not include any data
        // that is needed for any MPI send.
        BoundingBox mpi_interior;

        // Flags to calculate the interior and/or exterior.
        // TODO: replace with function parameters.
        bool do_mpi_interior = true;
        bool do_mpi_exterior = true;
        
        // List of all non-scratch stencil bundles in the order in which
        // they should be evaluated within a step.
        StencilBundleList stBundles;

        // List of all non-scratch stencil-bundle packs in the order in
        // which they should be evaluated within a step.
        BundlePackList stPacks;

        // All non-scratch grids.
        GridPtrs gridPtrs;
        GridPtrMap gridMap;

        // Only grids that are updated by the stencils.
        GridPtrs outputGridPtrs;
        GridPtrMap outputGridMap;

        // Scratch-grid vectors.
        // Each vector contains a grid for each thread.
        ScratchVecs scratchVecs;

        // Some calculated sizes for this rank and overall.
        IdxTuple rank_domain_offsets;       // Domain index offsets for this rank.
        IdxTuple overall_domain_sizes;       // Total of rank domains over all ranks.
        idx_t rank_nbytes=0, tot_nbytes=0;
        idx_t rank_domain_pts=0, tot_domain_pts=0;

        // Elapsed-time tracking.
        YaskTimer run_time;     // time in run_solution(), including halo exchange.
        YaskTimer ext_time;     // time in exterior stencil calculation.
        YaskTimer int_time;     // time in interior stencil calculation.
        YaskTimer halo_time;     // time spent just doing halo exchange, including MPI waits.
        YaskTimer wait_time;     // time spent just doing MPI waits.
        idx_t steps_done = 0;   // number of steps that have been run.

        // Maximum halos, skewing angles, and work extensions over all grids
        // used for wave-front rank tiling (wf).
        IdxTuple max_halos;  // spatial halos.
        idx_t wf_steps = 0;  // max number of WF steps.
        IdxTuple wf_angles;  // WF skewing angles for each shift (in points).
        idx_t num_wf_shifts = 0; // number of WF shifts required in wf_steps.
        IdxTuple wf_shift_pts;    // total shifted pts (wf_angles * num_wf_shifts).
        IdxTuple left_wf_exts;    // WF extension needed on left side of rank for halo exch.
        IdxTuple right_wf_exts;    // WF extension needed on right side of rank.

        // Settings for temporal blocking and mini-blocks.
        idx_t tb_steps = 0;  // max number of TB steps (may be less than requested).
        IdxTuple tb_angles;  // TB skewing angles for each shift (in points).
        idx_t num_tb_shifts = 0; // number of TB shifts required in tb_steps.
        IdxTuple tb_widths;      // base of TB trapezoid.
        IdxTuple tb_tops;      // top of TB trapezoid.
        IdxTuple mb_angles;  // MB skewing angles for each shift (in points).

        // MPI settings.
        // TODO: move to settings or MPI info object.
#ifdef NO_VEC_EXCHANGE
        bool allow_vec_exchange = false;
#else
        bool allow_vec_exchange = true; // allow vectorized halo exchange.
#endif
#ifdef NO_HALO_EXCHANGE
        bool enable_halo_exchange = false;
#else
        bool enable_halo_exchange = true;
#endif

        // Clear this to ignore step conditions.
        bool check_step_conds = true;

        // MPI data for each grid.
        // Map key: grid name.
        std::map<std::string, MPIData> mpiData;

        // Constructor.
        StencilContext(KernelEnvPtr env,
                       KernelSettingsPtr settings);

        // Destructor.
        virtual ~StencilContext() {

            // Dump stats if get_stats() hasn't been called yet.
            if (steps_done)
                get_stats();
        }

        // Set debug output to cout if my_rank == msg_rank
        // or a null stream otherwise.
        std::ostream& set_ostr();

        // Get the messsage output stream.
        std::ostream& get_ostr() const {
            assert(_ostr);
            return *_ostr;
        }

        // Reset elapsed times to zero.
        void clear_timers();

        // Access to settings.
        KernelSettingsPtr& get_settings() {
            assert(_opts);
            return _opts;
        }
        void set_settings(KernelSettingsPtr opts) {
            _opts = opts;
            _at.set_settings(_opts.get());
        }

        // Misc accessors.
        KernelEnvPtr& get_env() { return _env; }
        DimsPtr& get_dims() { return _dims; }
        MPIInfoPtr& get_mpi_info() { return _mpiInfo; }
        AutoTuner& getAT() { return _at; }
        bool use_pack_tuners() const { return _use_pack_tuners; }

        // Add a new grid to the containers.
        virtual void addGrid(YkGridPtr gp, bool is_output);
        virtual void addScratch(GridPtrs& scratch_vec) {
            scratchVecs.push_back(&scratch_vec);
        }

        // Set vars related to this rank's role in global problem.
        // Allocate MPI buffers as needed.
        virtual void setupRank();

        // Allocate grid memory for any grids that do not
        // already have storage.
        virtual void allocGridData(std::ostream& os);

        // Determine sizes of MPI buffers and allocate MPI buffer memory.
        // Dealloc any existing MPI buffers first.
        virtual void allocMpiData(std::ostream& os);
        virtual void freeMpiData(std::ostream& os) {
            mpiData.clear();
        }

        // Alloc scratch-grid memory.
        // Dealloc any existing scratch-grids first.
        virtual void allocScratchData(std::ostream& os);
        virtual void freeScratchData(std::ostream& os) {
            makeScratchGrids(0);
        }

        // Allocate grids, params, MPI bufs, etc.
        // Calculate rank position in problem.
        // Initialize some other data structures.
        // Print lots of stats.
        virtual void prepare_solution();

        // Print info about the soln.
        virtual void print_info();
        virtual void print_temporal_tiling_info();

        /// Get statistics associated with preceding calls to run_solution().
        virtual yk_stats_ptr get_stats();

        // Dealloc grids, etc.
        virtual void end_solution();

        // Set grid sizes and offsets.
        // This should be called anytime a setting or offset is changed.
        virtual void update_grid_info();

        // Set temporal blocking data.
        // This should be called anytime a block size is changed.
        virtual void update_tb_info();

        // Adjust offsets of scratch grids based
        // on thread and scan indices.
        virtual void update_scratch_grid_info(int thread_idx,
                                          const Indices& idxs);

        // Get total memory allocation required by grids.
        // Does not include MPI buffers.
        // TODO: add MPI buffers.
        virtual size_t get_num_bytes() {
            size_t sz = 0;
            for (auto gp : gridPtrs) {
                if (gp) {
                    if (sz)
                        sz += _data_buf_pad;
                    sz += gp->get_num_storage_bytes();
                }
            }
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
        // Enable nested OMP if there are >1 block threads,
        // disable otherwise.
        // Return number of threads.
        // Do nothing and return 0 if not properly initialized.
        virtual int set_region_threads() {

            // Start with max allowed threads.
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
            else
                omp_set_nested(0);

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
        void run_ref(idx_t first_step_index,
                             idx_t last_step_index);

        // Calculate results within a region.
        void calc_region(BundlePackPtr& sel_bp,
                                 const ScanIndices& rank_idxs);

        // Calculate results within a block.
        void calc_block(BundlePackPtr& sel_bp,
                                idx_t phase,
                                const ScanIndices& region_idxs);

        // Calculate results within a mini-block.
        void calc_mini_block(BundlePackPtr& sel_bp,
                             idx_t nphases, idx_t phase,
                             idx_t nshapes, idx_t shape,
                             const BridgeMask& bridge_mask,
                             const ScanIndices& base_region_idxs,
                             const ScanIndices& base_block_idxs,
                             const ScanIndices& block_idxs);

        // Exchange all dirty halo data for all stencil bundles.
        void exchange_halos(bool test_only = false);

        // Mark grids that have been written to by bundle pack 'sel_bp'.
        // If sel_bp==null, use all bundles.
        void mark_grids_dirty(const BundlePackPtr& sel_bp,
                              idx_t start, idx_t stop);

        // Set various limits in 'idxs' based on current step in region.
        bool shift_region(const Indices& base_start, const Indices& base_stop,
                            idx_t shift_num,
                            BundlePackPtr& bp,
                            ScanIndices& idxs);

        // Set various limits in 'idxs' based on current step in block.
        bool shift_mini_block(const Indices& mb_base_start,
                              const Indices& mb_base_stop,
                              idx_t mb_shift_num,
                              const Indices& adj_block_base_start,
                              const Indices& adj_block_base_stop,
                              const Indices& block_base_start,
                              const Indices& block_base_stop,
                              idx_t block_shift_num,
                              idx_t nphases, idx_t phase,
                              idx_t nshapes, idx_t shape,
                              const BridgeMask& bridge_mask,
                              const Indices& region_base_start,
                              const Indices& region_base_stop,
                              idx_t region_shift_num,
                              BundlePackPtr& bp,
                              ScanIndices& idxs);
        
        // Set the bounding-box around all stencil bundles.
        void find_bounding_boxes();

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
        virtual void set_rank_domain_size(const std::string& dim, idx_t size);
        virtual void set_min_pad_size(const std::string& dim, idx_t size);
        virtual void set_block_size(const std::string& dim, idx_t size);
        virtual void set_region_size(const std::string& dim, idx_t size);
        virtual void set_num_ranks(const std::string& dim, idx_t size);
        virtual void set_rank_index(const std::string& dim, idx_t size);
        virtual idx_t get_rank_domain_size(const std::string& dim) const;
        virtual idx_t get_min_pad_size(const std::string& dim) const;
        virtual idx_t get_block_size(const std::string& dim) const;
        virtual idx_t get_region_size(const std::string& dim) const;
        virtual idx_t get_num_ranks(const std::string& dim) const;
        virtual idx_t get_rank_index(const std::string& dim) const;
        virtual std::string apply_command_line_options(const std::string& args);
        virtual bool set_default_numa_preferred(int numa_node) {
#ifdef USE_NUMA
            _opts->_numa_pref = numa_node;
            return true;
#else
            _opts->_numa_pref = yask_numa_none;
            return numa_node == yask_numa_none;
#endif
        }
        virtual int get_default_numa_preferred() const {
            return _opts->_numa_pref;
        }

        // Auto-tuner methods.
        virtual void eval_auto_tuner(idx_t num_steps);
        
        // Auto-tuner APIs.
        virtual void reset_auto_tuner(bool enable, bool verbose = false);
        virtual void run_auto_tuner_now(bool verbose = true);
        virtual bool is_auto_tuner_enabled() const;

    }; // StencilContext.

    // Macro to get common items for stencil calcs efficiently.
#define CONTEXT_VARS(ctx_p)                                             \
    auto* cp = ctx_p;                                                   \
    auto& os = cp->get_ostr();                                          \
    auto* opts = cp->get_settings().get();                              \
    auto* mpiInfo = cp->get_mpi_info().get();                           \
    auto* dims = cp->get_dims().get();                                  \
    const int nddims = NUM_DOMAIN_DIMS;                                 \
    assert(nddims == dims->_domain_dims.size());                        \
    const int nsdims = NUM_STENCIL_DIMS;                                \
    assert(nsdims == dims->_stencil_dims.size());                       \
    const auto& step_dim = dims->_step_dim;                             \
    const auto step_posn = 0;                                           \
    assert(step_posn == +Indices::step_posn)

} // yask namespace.
