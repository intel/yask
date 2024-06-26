/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2024, Intel Corporation

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
        Indices bb_begin;   // first indices.
        Indices bb_end;     // one past last indices.

        // Number of valid points within the box.
        idx_t bb_num_points=0;

        // Following values are calculated from the above ones.
        Indices bb_len;       // size in each dim.
        idx_t bb_size=0;       // points in the entire box; bb_size >= bb_num_points.
        bool bb_is_full=false; // all points in box are valid (bb_size == bb_num_points).
        bool bb_is_aligned=false; // starting points are vector-aligned in all dims.
        bool bb_is_vec_mult=false; // num points are fold multiples in all dims.
        bool bb_valid=false;   // lengths and sizes have been calculated.

        BoundingBox() :
            bb_begin(NUM_DOMAIN_DIMS),
            bb_end(NUM_DOMAIN_DIMS),
            bb_len(NUM_DOMAIN_DIMS) { }

        // Same?
        // Only compares indices.
        inline bool operator==(const BoundingBox& rhs) const {
            return bb_begin == rhs.bb_begin &&
                bb_end == rhs.bb_end;
        }
        inline bool operator!=(const BoundingBox& rhs) const {
            return !operator==(rhs);
        }

        // Make Tuples.
        inline IdxTuple bb_begin_tuple(const IdxTuple& ddims) const {
            return bb_begin.make_tuple(ddims);
        }
        inline IdxTuple bb_end_tuple(const IdxTuple& ddims) const {
            return bb_end.make_tuple(ddims);
        }
        inline IdxTuple bb_last_tuple(const IdxTuple& ddims) const {
            auto res = bb_end.make_tuple(ddims);
            DOMAIN_VAR_LOOP(i, j)
                res[j] = res[j] - 1;
            return res;
        }
        inline IdxTuple bb_len_tuple(const IdxTuple& ddims) const {
            return bb_len.make_tuple(ddims);
        }
        std::string make_len_string(const IdxTuple& ddims) const {
            return bb_len_tuple(ddims).make_dim_val_str(" * ");
        }
        std::string make_range_string(const IdxTuple& ddims) const {
            return bb_begin_tuple(ddims).make_dim_val_str() +
                " ... " + bb_end_tuple(ddims).sub_elements(1).make_dim_val_str();
        }
        std::string make_range_str_dbg(const IdxTuple& ddims) const {
            return std::string("(") + make_len_string(ddims) + ") = " +
                std::to_string(bb_end.sub_elements(bb_begin).max_const(0).product()) +
                " at [{" + bb_begin_tuple(ddims).make_dim_val_str() +
                "}...{" + bb_end_tuple(ddims).make_dim_val_str() + "})";
        }

        // Calc values and set valid to true.
        // If 'force_full', set 'bb_num_points' as well as 'bb_size'.
        void update_bb(const std::string& name,
                       StencilContext* context,
                       bool force_full,
                       bool print_info = false);

        // Is point in BB?
        // 'pt' must have same dims as BB.
        bool is_in_bb(const IdxTuple& pt) const {
            assert(pt.get_num_dims() == bb_begin.get_num_dims());
            for (int i = 0; i < pt.get_num_dims(); i++) {
                if (pt[i] < bb_begin[i])
                    return false;
                if (pt[i] >= bb_end[i])
                    return false;
            }
            return true;
        }
        bool is_in_bb(const Indices& pt) const {
            assert(pt.get_num_dims() == bb_begin.get_num_dims());
            for (int i = 0; i < pt.get_num_dims(); i++) {
                if (pt[i] < bb_begin[i])
                    return false;
                if (pt[i] >= bb_end[i])
                    return false;
            }
            return true;
        }

        // Intersection between 'this' and 'other'.
        BoundingBox intersection_with(const BoundingBox& other,
                                      StencilContext* context) const {
            BoundingBox bbi;
            DOMAIN_VAR_LOOP_FAST(i, j) {
                bbi.bb_begin[j] = std::max(bb_begin[j], other.bb_begin[j]);
                bbi.bb_end[j] = std::min(bb_end[j], other.bb_end[j]);
                bbi.bb_end[j] = std::max(bbi.bb_begin[j], bbi.bb_end[j]);
            }
            bbi.update_bb("", context, true);
            return bbi;
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

    // Things in a context.
    class StencilPartBase;
    class Stage;
    typedef std::vector<StencilPartBase*> StencilPartList;
    typedef std::set<StencilPartBase*> StencilPartSet;
    typedef std::unordered_set<StencilPartBase*> StencilPartUSet;
    typedef std::shared_ptr<Stage> StagePtr;
    typedef std::vector<StagePtr> StageList;

    // Common data needed in the kernel(s).
    struct CommonCoreData {

        // Copies of context info.
        Indices _global_sizes;
        Indices _rank_sizes;
        Indices _rank_domain_offsets;

        void set_core(const StencilContext *cxt);
    };

    // Base of core data needed in the kernel(s).
    // Other data is added via inheritance by the YASK compiler.
    struct StencilCoreBase {
        CommonCoreData _common_core;
    };

    // Forward decl.
    class MpiSection;

    // Data and hierarchical sizes.
    // This is a pure-virtual class that must be implemented
    // for a specific problem.
    class StencilContext :
        public KernelStateBase,
        public virtual yk_solution {

    protected:

        // Auto-tuner for global settings.
        AutoTuner _at;

        // Bytes between each buffer to help avoid aliasing
        // in the HW.
        static constexpr size_t _data_buf_pad = YASK_PAD_BYTES;

        // Alloc given bytes on each NUMA node.
        virtual void _alloc_data(AllocMap& alloc_reqs,
                                 const std::string& type);

        // Callbacks.
        typedef std::vector<hook_fn_t> hook_fn_vec;
        hook_fn_vec _before_prepare_solution_hooks;
        hook_fn_vec _after_prepare_solution_hooks;
        typedef std::vector<hook_fn_2idx_t> hook_fn_2idx_vec;
        hook_fn_2idx_vec _before_run_solution_hooks;
        hook_fn_2idx_vec _after_run_solution_hooks;

    public:

        // Name.
        std::string name, long_name;

        // BB without any extensions for wave-fronts.
        // This is the BB for the domain in this rank only.
        BoundingBox rank_bb;

        // BB with any needed extensions for wave-fronts.
        // If WFs are not used, this is the same as 'rank_bb';
        BoundingBox ext_bb;

        // BB of the "interior" of this rank.  This is the area that does
        // not include any data that is needed for any MPI send.  The size
        // is valid for the layer that needs to be exchanged, and doesn't
        // include any extensions needed for WF.
        BoundingBox mpi_interior;

        // Max write halos across all scratch parts on left and right in each dim.
        IdxTuple max_write_halo_left, max_write_halo_right;

        // Is there a non-zero exterior in the given section?
        inline bool does_exterior_exist(idx_t ddim, bool is_left) const {
            return is_left ?
                mpi_interior.bb_begin[ddim] > ext_bb.bb_begin[ddim] :
                mpi_interior.bb_end[ddim] < ext_bb.bb_end[ddim];
        }

        // List of all non-scratch stencil parts in the order in which
        // they should be evaluated within a step.
        StencilPartList st_parts;

        // List of all non-scratch stencil-stages in the order in
        // which they should be evaluated within a step.
        StageList st_stages;

        // All non-scratch vars, including those created by APIs.
        VarPtrs all_var_ptrs;
        VarPtrMap all_var_map;

        // Only non-scratch vars defined by the YASK compiler.
        VarPtrs orig_var_ptrs;
        VarPtrMap orig_var_map;

        // Only non-scratch vars defined by the YASK compiler that are updated by the stencils.
        VarPtrs output_var_ptrs;
        VarPtrMap output_var_map;

        // Scratch-var vectors.
        // Each vector contains a var for each thread.
        ScratchVecs scratch_vecs;

        // Some calculated sizes for this rank and overall.
        Indices rank_domain_offsets;       // Domain index offsets for this rank.
        idx_t rank_nbytes=0, tot_nbytes=0;
        idx_t rank_domain_pts=0, tot_domain_pts=0;

        // Vars for tracking things in a stage: vecs with one entry per outer thread.
        std::vector<StencilPartUSet> _parts_done;
        std::vector<VarPtrUSet> _vars_written;
         
        // Elapsed-time tracking.
        YaskTimer run_time;     // time in run_solution(), including halo exchange.
        YaskTimer ext_time;     // time in exterior stencil calculation.
        YaskTimer int_time;     // time in interior stencil calculation.
        YaskTimer halo_time;    // time spent in halo exchange.
        YaskTimer halo_pack_time;     // time spent on packing in halo exchange.
        YaskTimer halo_unpack_time; // time spent on unpacking in halo exchange.
        YaskTimer halo_copy_time;   // time spent on copying buffers in halo exchange.
        YaskTimer halo_wait_time;     // time spent on MPI waits in halo exchange.
        YaskTimer halo_test_time;     // time spent on MPI tests for halo exchange.
        YaskTimer halo_lock_wait_time; // time spent on shm lock waits in halo exchange.
        idx_t steps_done = 0;   // number of steps that have been run.

        // Maximum halos, skewing angles, and work extensions over all vars
        // used for wave-front rank tiling (wf).
        IdxTuple max_halos;  // spatial halos.
        idx_t wf_steps = 0;  // max number of steps in a WF. 0 => no WF.
        IdxTuple wf_angles;  // WF skewing angles for each shift (in points).
        idx_t num_wf_shifts = 0; // number of WF shifts required in wf_steps.
        IdxTuple wf_shift_pts;    // total shifted pts (wf_angles * num_wf_shifts).
        IdxTuple left_wf_exts;    // WF extension needed on left side of rank for halo exch.
        IdxTuple right_wf_exts;    // WF extension needed on right side of rank.

        // Settings for temporal blocking and micro-blocks.
        idx_t tb_steps = 0;  // max number of steps in a TB. 0 => no TB.
        IdxTuple tb_angles;  // TB skewing angles for each shift (in points).
        idx_t num_tb_shifts = 0; // number of TB shifts required in tb_steps.
        IdxTuple tb_widths;      // base of TB trapezoid.
        IdxTuple tb_tops;      // top of TB trapezoid.
        IdxTuple mb_angles;  // MB skewing angles for each shift (in points).

        // MPI buffers for each var.
        // Map key: var name.
        std::map<std::string, MPIData> mpi_data;

        // Constructor.
        StencilContext(KernelEnvPtr& kenv,
                       KernelSettingsPtr& actl_settings,
                       KernelSettingsPtr& req_settings);

        // Destructor.
        virtual ~StencilContext() {

            // Dump stats if get_stats() hasn't been called yet.
            if (steps_done)
                get_stats();
        }

        // Access core data.
        virtual StencilCoreBase* corep() =0;

        // Ready?
        bool is_prepared() const {
            return rank_bb.bb_valid;
        }
        void set_prepared(bool prep) {
            rank_bb.bb_valid = prep;
        }

        // Reset elapsed times to zero.
        void clear_timers();

        // Misc accessors.
        AutoTuner& get_at() { return _at; }

        // Add a new var to the containers.
        virtual void add_var(YkVarPtr gp, bool is_orig, bool is_output);
        virtual void add_scratch(VarPtrs& scratch_vec) {
            scratch_vecs.push_back(&scratch_vec);
        }

        // Set vars related to this rank's role in global problem.
        // Allocate MPI buffers as needed.
        virtual void setup_rank();

        // Allocate var memory for any vars that do not
        // already have storage.
        virtual void alloc_var_data();

        // Determine sizes of MPI buffers and allocate MPI buffer memory.
        // Dealloc any existing MPI buffers first.
        virtual void alloc_mpi_data();
        virtual void free_mpi_data() {
            mpi_data.clear();
        }

        // Alloc scratch-var memory.
        // Dealloc any existing scratch-vars first.
        virtual void alloc_scratch_data();
        virtual void free_scratch_data() {
            make_scratch_vars(0);
        }

        // Allocate vars, params, MPI bufs, etc.
        // Calculate rank position in problem.
        // Initialize some other data structures.
        // Print lots of stats.
        virtual void prepare_solution();

        // Init amount-of-work stats.
        virtual void init_work_stats();

        // Reset any locks, etc.
        virtual void reset_locks();

        // Print info about the soln.
        virtual void print_temporal_tiling_info(std::string prefix = "") const;
        virtual void print_sizes(std::string prefix = "") const;
        virtual void print_warnings() const;

        /// Get statistics associated with preceding calls to run_solution().
        virtual yk_stats_ptr get_stats();
        virtual void clear_stats() {
            clear_timers();
        }

        // Dealloc vars, etc.
        virtual void end_solution();

        // Set var sizes and offsets.
        // This should be called anytime a setting or offset is changed.
        virtual void update_var_info(bool force);

        // Set temporal blocking data.
        // This should be called anytime a block size is changed.
        virtual void update_tb_info();

        // Adjust offsets of scratch vars based
        // on thread and scan indices.
        virtual void update_scratch_var_info(int outer_thread_idx,
                                              const Indices& idxs);

        // Copy non-scratch vars to device as needed.
        void copy_vars_to_device() const;

        // Copy non-scratch output vars from device as needed.
        void copy_vars_from_device() const;
        
        // Get total memory allocation required by vars.
        // Does not include MPI buffers.
        // TODO: add MPI buffers.
        virtual size_t get_num_bytes() {
            size_t sz = 0;
            for (auto gp : all_var_ptrs) {
                if (gp)
                    sz += gp->get_num_storage_bytes() + _data_buf_pad;
            }
            for (auto gps : scratch_vecs)
                if (gps)
                    for (auto gp : *gps)
                        if (gp)
                            sz += gp->get_num_storage_bytes() + _data_buf_pad;
            return sz;
        }

        // Init all vars & params by calling real_init_fn.
        virtual void init_values(real_t seed0,
                                 std::function<void (YkVarPtr gp,
                                                     real_t seed)> real_init_fn);

        // Init all vars & params to same value within vars,
        // but different for each var.
        virtual void init_same(real_t seed0) {
            init_values(seed0, [&](YkVarPtr gp, real_t seed)
                               { gp->set_all_elements_same(seed); });
        }

        // Init all vars & params to different values within vars,
        // and different for each var.
        virtual void init_diff(real_t seed0) {
            init_values(seed0, [&](YkVarPtr gp, real_t seed)
                               { gp->set_all_elements_in_seq(seed); });
        }

        // Compare vars in contexts for validation.
        // Params should not be written to, so they are not compared.
        // Return number of mis-compares.
        virtual idx_t compare_data(const StencilContext& ref) const;

        // Reference stencil calculations.
        void run_ref(idx_t first_step_index,
                     idx_t last_step_index);

        // Calculate results within a mega-block.
        void calc_mega_block(StagePtr& sel_bp,
                             const ScanIndices& rank_idxs,
                             MpiSection& mpisec);

        // Calculate results within a block.
        void calc_block(StagePtr& sel_bp,
                        idx_t mega_block_shift_num,
                        idx_t nphases, idx_t phase,
                        const ScanIndices& rank_idxs,
                        const ScanIndices& mega_block_idxs,
                        MpiSection& mpisec);

        // Calculate results within a micro-block.
        void calc_micro_block(int outer_thread_idx,
                              StagePtr& sel_bp,
                              idx_t mega_block_shift_num,
                              idx_t nphases, idx_t phase,
                              idx_t nshapes, idx_t shape,
                              const bit_mask_t& bridge_mask,
                              const ScanIndices& rank_idxs,
                              const ScanIndices& base_mega_block_idxs,
                              const ScanIndices& base_block_idxs,
                              const ScanIndices& adj_block_idxs,
                              MpiSection& mpisec);

        // Exchange all dirty halo data for all stencil parts.
        void exchange_halos(MpiSection& mpisec);
        virtual void exchange_halos();

        // Call MPI_Test() on all unfinished requests to advance MPI progress.
        void adv_halo_exchange();

        // Update valid steps in vars that have been written to by stage 'sel_bp'.
        // If sel_bp==null, use all parts.
        // If 'mark_dirty', also mark as needing halo exchange.
        void update_var_info(const StagePtr& sel_bp,
                             idx_t start, idx_t stop,
                             bool mark_dirty,
                             bool mod_dev_data = true);

        // Mark all exchangable vars as possibly dirty in other ranks. This
        // should be called anytime APIs could have been called and before
        // running any steps.
        void set_all_neighbor_vars_dirty() {
            for (auto& gp : orig_var_ptrs) {
                gp->gb().set_dirty_all(YkVarBase::others, true);
            }
        }

        // Set various limits in 'idxs' based on current step in mega-block.
        bool shift_mega_block(const Indices& base_start, const Indices& base_stop,
                              idx_t shift_num,
                              StagePtr& bp,
                              ScanIndices& idxs,
                              const MpiSection& mpisec);

        // Set various limits in 'idxs' based on current step in block.
        bool shift_micro_block(const Indices& mb_base_start,
                              const Indices& mb_base_stop,
                              const Indices& adj_block_base_start,
                              const Indices& adj_block_base_stop,
                              const Indices& block_base_start,
                              const Indices& block_base_stop,
                              const Indices& mega_block_base_start,
                              const Indices& mega_block_base_stop,
                              idx_t mb_shift_num,
                              idx_t nphases, idx_t phase,
                              idx_t nshapes, idx_t shape,
                              const bit_mask_t& bridge_mask,
                              ScanIndices& idxs);

        // Set the bounding-box around all stencil parts.
        void find_bounding_boxes();

        // Determine max write halos for scratch parts.
        void find_scratch_write_halos();
        
        // Set data needed by the kernels.
        // Implemented by the YASK compiler-generated code.
        virtual void set_core() =0;
            
        // Make new scratch vars.
        // Implemented by the YASK compiler-generated code.
        virtual void make_scratch_vars(int num_threads) =0;
        
        // Make a new var iff its dims match any in the stencil.
        // Returns pointer to the new var or nullptr if no match.
        // Implemented by the YASK compiler-generated code.
        virtual VarBasePtr new_stencil_var(const std::string & name,
                                           const VarDimNames & dims) =0;

        // Make a new var with 'name' and 'dims'.
        // Set sizes if 'sizes' is non-null.
        virtual YkVarPtr new_var(const std::string& name,
                                  const VarDimNames& dims,
                                  const VarDimSizes* sizes);

        // Call user-inserted code.
        virtual void call_hooks(hook_fn_vec& hook_fns);
        virtual void call_2idx_hooks(hook_fn_2idx_vec& hook_fns,
                                     idx_t first_step_index,
                                     idx_t last_step_index);

        // APIs.
        // See yask_kernel_api.hpp.
        virtual const std::string& get_name() const {
            return name;
        }
        virtual const std::string& get_description() const {
            return long_name;
        }
        virtual bool is_offloaded() const {
            #if USE_OFFLOAD
            return true;
            #else
            return false;
            #endif
        }
        virtual void set_debug_output(yask_output_ptr debug) {
            KernelStateBase::set_debug_output(debug);
        }
        virtual void disable_debug_output() {
            KernelStateBase::disable_debug_output();
        }

        virtual int get_num_vars() const {
            return int(all_var_ptrs.size());
        }

        virtual yk_var_ptr get_var(const std::string& name) {
            auto i = all_var_map.find(name);
            if (i != all_var_map.end())
                return i->second;
            THROW_YASK_EXCEPTION("YASK variable '" + name + "' not found");
        }
        virtual std::vector<yk_var_ptr> get_vars() {
            std::vector<yk_var_ptr> vars;
            for (auto& vp : all_var_ptrs)
                vars.push_back(vp);
            return vars;
        }
        virtual yk_var_ptr
        new_var(const std::string& name,
                 const VarDimNames& dims) {
            return new_var(name, dims, NULL);
        }
        virtual yk_var_ptr
        new_var(const std::string& name,
                 const std::initializer_list<std::string>& dims) {
            VarDimNames dims2(dims);
            return new_var(name, dims2);
        }
        virtual yk_var_ptr
        new_fixed_size_var(const std::string& name,
                             const VarDimNames& dims,
                             const VarDimSizes& dim_sizes) {
            return new_var(name, dims, &dim_sizes);
        }
        virtual yk_var_ptr
        new_fixed_size_var(const std::string& name,
                             const std::initializer_list<std::string>& dims,
                             const idx_t_init_list& dim_sizes) {
            VarDimNames dims2(dims);
            VarDimSizes sizes2(dim_sizes);
            return new_fixed_size_var(name, dims2, sizes2);
        }

        virtual std::string get_step_dim_name() const {
            STATE_VARS_CONST(this);
            return dims->_step_dim;
        }
        virtual int get_num_domain_dims() const {
            STATE_VARS_CONST(this);
            return dims->_domain_dims.get_num_dims();
        }
        virtual string_vec get_domain_dim_names() const {
            STATE_VARS_CONST(this);
            return domain_dims.get_dim_names();
        }
        virtual string_vec get_misc_dim_names() const {
            STATE_VARS_CONST(this);
            return misc_dims.get_dim_names();
        }

        // Threads.
        virtual int get_num_outer_threads() const {
            STATE_VARS(this);

            // Numbers of threads.
            int othr, ithr;
            get_num_comp_threads(othr, ithr);
            return othr;
        }
        virtual int get_num_inner_threads() const {
            STATE_VARS(this);

            // Numbers of threads.
            int othr, ithr;
            get_num_comp_threads(othr, ithr);
            return ithr;
        }


        virtual void run_solution(idx_t first_step_index,
                                  idx_t last_step_index);
        virtual void run_solution(idx_t step_index) {
            run_solution(step_index, step_index);
        }
        virtual void fuse_vars(yk_solution_ptr other);

        // APIs that access settings.
        #define GET_SOLN_API(api_name) \
            virtual idx_t get_ ## api_name (const std::string& dim) const; \
            virtual idx_t_vec get_ ## api_name ## _vec() const;
        #define SET_SOLN_API(api_name) \
            virtual void set_ ## api_name (const std::string& dim, idx_t size); \
            virtual void set_ ## api_name ## _vec(const idx_t_vec& vals); \
            virtual void set_ ## api_name ## _vec(const idx_t_init_list& vals);
        #define SOLN_API(api_name) \
            GET_SOLN_API(api_name) \
            SET_SOLN_API(api_name)
        SOLN_API(num_ranks)
        SOLN_API(rank_index)
        SOLN_API(overall_domain_size)
        SOLN_API(rank_domain_size)
        SOLN_API(block_size)
        SOLN_API(min_pad_size)
        GET_SOLN_API(first_rank_domain_index)
        GET_SOLN_API(last_rank_domain_index)
        #undef SOLN_API
        #undef SET_SOLN_API
        #undef GET_SOLN_API

        virtual std::string apply_command_line_options(const std::string& args);
        virtual std::string apply_command_line_options(int argc, char* argv[]);
        virtual std::string apply_command_line_options(const string_vec& args);
        virtual std::string get_command_line_help();
        virtual std::string get_command_line_values();
        virtual bool get_step_wrap() const {
            STATE_VARS(this);
            return actl_opts->_step_wrap;
        }
        virtual void set_step_wrap(bool do_wrap) {
            STATE_VARS(this);
            req_opts->_step_wrap = do_wrap;
            actl_opts->_step_wrap = do_wrap;
        }
        virtual bool set_default_numa_preferred(int numa_node) {
            STATE_VARS(this);

            // TODO: fix this when NUMA APIs are not available.
            req_opts->_numa_pref = numa_node;
            actl_opts->_numa_pref = numa_node;
            return true;
        }
        virtual int get_default_numa_preferred() const {
            STATE_VARS_CONST(this);
            return actl_opts->_numa_pref;
        }
        virtual void
        call_before_prepare_solution(hook_fn_t hook_fn) {
            _before_prepare_solution_hooks.push_back(hook_fn);
        }
        virtual void
        call_after_prepare_solution(hook_fn_t hook_fn) {
            _after_prepare_solution_hooks.push_back(hook_fn);
        }
        virtual void
        call_before_run_solution(hook_fn_2idx_t hook_fn) {
            _before_run_solution_hooks.push_back(hook_fn);
        }
        virtual void
        call_after_run_solution(hook_fn_2idx_t hook_fn) {
            _after_run_solution_hooks.push_back(hook_fn);
        }

        // Auto-tuner methods.
        virtual void eval_auto_tuner();

        // Auto-tuner APIs.
        virtual void reset_auto_tuner(bool enable, bool verbose = false);
        virtual void run_auto_tuner_now(bool verbose = true);
        virtual bool is_auto_tuner_enabled() const;

    }; // StencilContext.

    // Flags to track state of calculating the interior and/or exterior.
    class MpiSection {
        const StencilContext* _scp;

    public:
        bool do_mpi_interior = true;
        bool do_mpi_left = true;        // left exterior in given dim.
        bool do_mpi_right = true;        // right exterior in given dim.
        idx_t mpi_exterior_dim = -1;      // which domain dim in left/right.

        MpiSection(const StencilContext* scp) :
            _scp(scp) { }

        void init() {
            do_mpi_interior = do_mpi_left = do_mpi_right = true;
            mpi_exterior_dim = -1;
        }

        // Is overlapping-comms mode currently enabled?
        // Side-effect: checks for consistency of MPI flags.
        inline bool is_overlap_active() const {
            assert(do_mpi_interior || do_mpi_left || do_mpi_right);
            if (!do_mpi_interior) {
                assert(do_mpi_left || do_mpi_right); // one or both.
                if (do_mpi_left != do_mpi_right) // one only.
                    assert(mpi_exterior_dim >= 0); // must specify dim.
            }
            else {
                assert(do_mpi_left == do_mpi_right); // both or neither.
            }
            bool active = !do_mpi_interior || !do_mpi_left || !do_mpi_right;
            if (active)
                assert(_scp->mpi_interior.bb_valid);
            return active;
        }

        // Currently doing the exterior only?
        bool is_exterior_active() const {
            return !do_mpi_interior && (do_mpi_left || do_mpi_right);
        }

        // Describe MPI flag setting.
        std::string make_descr() const {
            STATE_VARS(_scp);
            if (is_overlap_active())
                return std::string("MPI ") +
                    (do_mpi_interior ? "interior" :
                     (do_mpi_left && do_mpi_right) ? "exterior" :
                     do_mpi_left ?
                     ("exterior left-" +
                      domain_dims.get_dim_name(mpi_exterior_dim)) :
                     ("exterior right-" +
                      domain_dims.get_dim_name(mpi_exterior_dim))) +
                    " section";
            return std::string("all MPI sections");
        }
    };

} // yask namespace.
