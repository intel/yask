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

        // Make a new grid iff its dims match any in the stencil.
        // Returns pointer to the new grid or nullptr if no match.
        virtual YkGridPtr newStencilGrid (const std::string & name,
                                          const GridDimNames & dims) =0;

        // Make new grids.
        virtual YkGridPtr newGrid(const std::string& name,
                                  const std::vector<std::string>& dims,
                                  bool is_visible = true);

        // APIs.
        // See yask_kernel_api.hpp.
        virtual void set_debug_output(yask_output_ptr debug) {
            _debug = debug;     // to share ownership of referent.
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
                 const std::initializer_list<std::string>& dims) {
            std::vector<std::string> dims2(dims);
            return new_grid(name, dims2);
        }

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

} // yask namespace.
