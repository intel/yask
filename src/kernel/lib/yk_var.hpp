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

    /*
     Rank and local offsets in domain dim:
    
       | ... |        +------+       |
       |  global ofs  |      |       |
       |<------------>| var  |       |
       |     |  loc   |domain|       |
       |rank |  ofs   |      |       |
       | ofs |<------>|      |       |
       |<--->|        +------+       |
       ^     ^        ^              ^
       |     |        |              last rank-domain index
       |     |        0 index in underlying storage.
       |     first rank-domain index
       first overall-domain index
      
       Rank offset is not necessarily a vector multiple.
       Local offset must be a vector multiple.

     OOD:                                                        yk_var (API)
                                                                    ^
                                                                    |
                YkVarBaseCore <---ptr------- YkVarBase <--sh_ptr-- YkVarImpl
                  ^    ^                      ^   ^  |
                  |    |                      |   |  +---------------+
                  |    |                      |   |                  |
                  |    |               YkElemVar  YkVecVar           |
                  |    |                  |         |                |
         +-------------------------has-a--+         |                |
         |        |    |          +-----------has-a-+                |
         |        |    |          |                                  |
         v        |    |          v                                  |
    YkElemVarCore<LF> YkVecVarCore<LF>                               |
         |                        |                                  |
       has-a                    has-a                                |
         |                        |                                  |
         v                        v                                  |
    +-> GenericVarCore<real_t,LF> GenericVarCore<real_vec_t,LF> <-+  |
    |     _elems ptr                 _elems ptr                   |  |
    |                                                             |  |
    |                                                             |  |
    |                    GenericVarBase <-------------ptr------------+
    ptr                    ^      ^  _base ptr                    |
    |                      |      |                               ptr
    |  GenericVarTyped<real_t>  GenericVarTyped<real_vec_t>       |
    |    ^                        ^                               |
    |    |                        |                               |
    +- GenericVar<real_t,LF>    GenericVar<real_vec_t,LF> --------+

       "Core" types are non-virtual and can be trivially copied, e.g.,
       to an offload device; others are virtual and cannot.
       "LF" is a layout-function type.
    */
    
    ///// Yk*Var*Core types /////
    
    // Core data that is needed for computations using a var.
    // A trivially-copyable type for offloading.
    struct YkVarBaseCore {

        // The following indices have one value for each dim in the var.
        // All values are in units of reals, not underlying SIMD vectors, if different.
        // See diagram above for '_rank_offsets' and '_local_offsets'.
        // Comments show settings for domain dims | non-domain dims.
        Indices _domains;   // size of "interior" of var (i.e., not pads) | alloc size.
        Indices _req_left_epads, _req_right_epads; // requested extra space around halos | zero.
        Indices _req_left_pads, _req_right_pads; // requested space around domains | zero.
        Indices _actl_left_pads, _actl_right_pads; // actual space around domains | zero.
        Indices _left_halos, _right_halos; // space within pads for halo exchange | zero.
        Indices _left_wf_exts, _right_wf_exts; // additional halos for wave-fronts | zero.
        Indices _rank_offsets;   // offsets of this rank in global space | zero.
        Indices _local_offsets; // offsets of this var's domain in this rank | first index.
        Indices _allocs;    // actual var alloc size | same.

        // Each entry in _soln_vec_lens is same as the corresponding dim in dims->_fold_pts.
        Indices _soln_vec_lens;  // num reals in each elem in soln fold | one.

        // Each entry in _var_vec_lens may be same as dims->_fold_pts or one, depending
        // on whether var is fully vectorized.
        Indices _var_vec_lens;  // num reals in each elem in this var | one.

        // Sizes in vectors for sizes that are always vec lens.
        // These are pre-calculated to avoid division later.
        Indices _vec_left_pads; // _actl_left_pads / _var_vec_lens | zero.
        Indices _vec_allocs; // _allocs / _var_vec_lens | _allocs.
        Indices _vec_local_offsets; // _local_offsets / _var_vec_lens | first index.
        Indices _vec_strides; // num vecs between consecutive indices | one.

        // Ctor.
        YkVarBaseCore(int ndims);
        
        // Index math.
        ALWAYS_INLINE idx_t get_first_local_index(idx_t posn) const {
            return _rank_offsets[posn] + _local_offsets[posn] - _actl_left_pads[posn];
        }
        ALWAYS_INLINE idx_t get_last_local_index(idx_t posn) const {
            return _rank_offsets[posn] + _local_offsets[posn] + _domains[posn] + _actl_right_pads[posn] - 1;
        }

        // Adjust logical time index to 0-based index
        // using temporal allocation size.
        ALWAYS_INLINE idx_t _wrap_step(idx_t t) const {

            // Index wraps in tdim.
            // Examples based on tdim == 2:
            //  t => return value.
            // ---  -------------
            // -2 => 0.
            // -1 => 1.
            //  0 => 0.
            //  1 => 1.
            //  2 => 0.

            // Avoid discontinuity in dividing negative numbers
            // by using floored-mod.
            idx_t res = imod_flr(t, _domains[+step_posn]);
            return res;
        }

    }; // YkVarBaseCore.
    
    static_assert(std::is_trivially_copyable<YkVarBaseCore>::value,
                  "Needed for OpenMP offload");

    // Core data for YASK var of real elements.
    template <typename LayoutFn, bool _use_step_idx>
    struct YkElemVarCore final : public YkVarBaseCore {

        // Core for generic storage is owned here by composition.
        // We do this to reduce the number of structs that need to be
        // copied to the offload device.
        typedef GenericVarCore<real_t, LayoutFn> _data_t;
        static_assert(std::is_trivially_copyable<_data_t>::value,
                      "Needed for OpenMP offload");
        _data_t _data;

        // Ctor.
        YkElemVarCore(int ndims) :
            YkVarBaseCore(ndims) { }

    protected:

        // Calc one adjusted index and recurse to i-1.
        template <bool is_global, int i>
        void _get_adj_idx(Indices& adj_idxs,
                          const Indices& idxs,
                          idx_t alloc_step_idx) const {
            if constexpr (i < 0)
                             return;

            // Special handling for step index.
            constexpr auto sp = +step_posn;
            if constexpr (_use_step_idx && i == sp) {
                    host_assert(alloc_step_idx == _wrap_step(idxs[sp]));
                    adj_idxs[i] = alloc_step_idx;
                }

            // All other indices.
            else {

                // Adjust for offsets and padding.
                // This gives a positive 0-based local element index.
                idx_t ai = idxs[i] + _actl_left_pads[i] - _local_offsets[i];

                // Also adjust for rank offsets if using global indices.
                if constexpr (is_global)
                                 ai -= _rank_offsets[i];
                    
                host_assert(ai >= 0);
                adj_idxs[i] = uidx_t(ai);
            }

            // Recurse (during compilation) until done.
            if constexpr (i > 0)
                             _get_adj_idx<is_global, i - 1>(adj_idxs, idxs, alloc_step_idx);
        }

        // Get a pointer to given element.
        // 'alloc_step_idx' must be within allocation bounds and consistent
        // with 'idxs[step_posn]'.
        template <bool is_global>
        const real_t* _get_elem_ptr(const Indices& idxs,
                                    idx_t alloc_step_idx,
                                    bool check_bounds) const {
            constexpr auto n = LayoutFn::get_num_sizes();
            Indices adj_idxs(n);
            _get_adj_idx<is_global, n - 1>(adj_idxs, idxs, alloc_step_idx);

            // Get pointer via layout in _data.
            return _data.get_ptr(adj_idxs, check_bounds);
        }

    public:
        ALWAYS_INLINE
        const real_t* get_elem_ptr(const Indices& global_idxs,
                                   idx_t alloc_step_idx,
                                   bool check_bounds=true) const {
            return _get_elem_ptr<true>(global_idxs, alloc_step_idx, check_bounds);
        }
        ALWAYS_INLINE
        const real_t* get_elem_ptr_local(const Indices& local_idxs,
                                         idx_t alloc_step_idx,
                                         bool check_bounds=true) const {
            return _get_elem_ptr<false>(local_idxs, alloc_step_idx, check_bounds);
        }

        // Non-const versions.
        // Implemented via casting.
        ALWAYS_INLINE
        real_t* get_elem_ptr(const Indices& global_idxs,
                             idx_t alloc_step_idx,
                             bool check_bounds=true) {
            const real_t* p =
                const_cast<const YkElemVarCore*>(this)->
                get_elem_ptr(global_idxs, alloc_step_idx, check_bounds);
            return const_cast<real_t*>(p);
        }
        ALWAYS_INLINE
        real_t* get_elem_ptr_local(const Indices& local_idxs,
                                   idx_t alloc_step_idx,
                                   bool check_bounds=true) {
            const real_t* p =
                const_cast<const YkElemVarCore*>(this)->
                get_elem_ptr_local(local_idxs, alloc_step_idx, check_bounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        // Indices are global, i.e., relative to overall problem domain.
        ALWAYS_INLINE
        real_t read_elem(const Indices& idxs,
                         idx_t alloc_step_idx) const {
            const real_t* ep = get_elem_ptr(idxs, alloc_step_idx);
            return *ep;
        }

        // Write one element.
        // Indices are global, i.e., relative to overall problem domain.
        ALWAYS_INLINE
        void write_elem(real_t val,
                        const Indices& idxs,
                        idx_t alloc_step_idx) {
            real_t* ep = get_elem_ptr(idxs, alloc_step_idx);
            *ep = val;
        }


        // Read one element.
        // Indices are local.
        ALWAYS_INLINE
        real_t read_elem_local(const Indices& idxs,
                               idx_t alloc_step_idx) const {
            const real_t* ep = get_elem_ptr_local(idxs, alloc_step_idx);
            return *ep;
        }

        // Write one element.
        // Indices are local.
        ALWAYS_INLINE
        void write_elem_local(real_t val,
                              const Indices& idxs,
                              idx_t alloc_step_idx) {
            real_t* ep = get_elem_ptr_local(idxs, alloc_step_idx);
            *ep = val;
        }

    }; // YkElemVarCore.

    // Core data for YASK var of real vectors.
    template <typename LayoutFn, bool _use_step_idx, idx_t... _templ_vec_lens>
    struct YkVecVarCore final : public YkVarBaseCore {

        // Positions of var dims in vector fold dims.
        Indices _vec_fold_posns;

        // Storage core is owned here by composition.
        typedef GenericVarCore<real_vec_t, LayoutFn> _data_t;
        static_assert(std::is_trivially_copyable<_data_t>::value,
                      "Needed for OpenMP offload");
        _data_t _data;

        // Ctor.
        YkVecVarCore(int ndims) :
            YkVarBaseCore(ndims),
            _vec_fold_posns(idx_t(0), ndims) { }

    protected:
        // Calc one adjusted vec idx and offset and recurse to i-1.
        template <int i>
        void _get_adj_idx(Indices& vec_idxs,
                          Indices& elem_ofs,
                          const Indices& idxs,
                          idx_t alloc_step_idx) const {
            if constexpr (i < 0)
                             return;

            constexpr int nvls = sizeof...(_templ_vec_lens);
            constexpr uidx_t vls[nvls] { _templ_vec_lens... };

            // Special handling for step index.
            constexpr auto sp = +step_posn;
            if constexpr (_use_step_idx && i == sp) {
                    host_assert(alloc_step_idx == _wrap_step(idxs[sp]));
                    vec_idxs[sp] = alloc_step_idx;
                    elem_ofs[sp] = 0;
                }

            // All other indices.
            else {

                // Adjust for offset and padding.
                // This gives a positive 0-based local element index.
                idx_t ai = idxs[i] + _actl_left_pads[i] -
                    (_rank_offsets[i] + _local_offsets[i]);
                host_assert(ai >= 0);
                uidx_t adj_idx = uidx_t(ai);

                // Get vector index and offset.
                // Use unsigned DIV and MOD to avoid compiler having to
                // emit code for preserving sign when using shifts.
                vec_idxs[i] = idx_t(adj_idx / vls[i]);
                elem_ofs[i] = idx_t(adj_idx % vls[i]);
                host_assert(vec_idxs[i] == idx_t(adj_idx / _var_vec_lens[i]));
                host_assert(elem_ofs[i] == idx_t(adj_idx % _var_vec_lens[i]));
            }

            // Recurse (during compilation) until done.
            if constexpr (i > 0)
                             _get_adj_idx<i - 1>(vec_idxs, elem_ofs, idxs, alloc_step_idx);
        }

        // Calc one fold offset and recurse to i-1.
        template <int i>
        void _get_fold_ofs(Indices& fold_ofs,
                           const Indices& elem_ofs) const {
            if constexpr (i < 0)
                             return;

            int j = _vec_fold_posns[i];
            fold_ofs[i] = elem_ofs[j];
            
            // Recurse (during compilation) until done.
            if constexpr (i > 0)
                             _get_fold_ofs<i - 1>(fold_ofs, elem_ofs);
        }
        
    public:
        // Get a pointer to given element.
        const real_t* get_elem_ptr(const Indices& idxs,
                                   idx_t alloc_step_idx,
                                   bool check_bounds=true) const {

            // Use template vec lengths instead of run-time values for
            // efficiency.
            constexpr int nvls = sizeof...(_templ_vec_lens);
            Indices vec_idxs(nvls), elem_ofs(nvls);
            #ifdef DEBUG_LAYOUT
            constexpr auto ns = LayoutFn::get_num_sizes();
            host_assert(ns == nvls);
            #endif
            _get_adj_idx<nvls - 1>(vec_idxs, elem_ofs, idxs, alloc_step_idx);
            
            // Get only the vectorized fold offsets, i.e., those
            // with vec-lengths > 1.
            // And, they need to be in the original folding order,
            // which might be different than the var-dim order.
            Indices fold_ofs(NUM_VEC_FOLD_DIMS);
            _get_fold_ofs<NUM_VEC_FOLD_DIMS - 1>(fold_ofs, elem_ofs);

            // Get 1D element index into vector.
            //auto i = dims->get_elem_index_in_vec(fold_ofs);
            idx_t i = VEC_FOLD_LAYOUT(fold_ofs);

            // Get pointer to vector.
            const real_vec_t* vp = _data.get_ptr(vec_idxs, check_bounds);

            // Get pointer to element.
            const real_t* ep = &(*vp)[i];
            return ep;
        }

        // Non-const version.
        // Implemented with casting.
        ALWAYS_INLINE
        real_t* get_elem_ptr(const Indices& idxs,
                             idx_t alloc_step_idx,
                             bool check_bounds=true) {
            const real_t* p =
                const_cast<const YkVecVarCore*>(this)->get_elem_ptr(idxs, alloc_step_idx,
                                                                    check_bounds);
            return const_cast<real_t*>(p);
        }

        // Read one element.
        // Indices are relative to overall problem domain.
        ALWAYS_INLINE
        real_t read_elem(const Indices& idxs,
                         idx_t alloc_step_idx) const {
            const real_t* ep = get_elem_ptr(idxs, alloc_step_idx);
            return *ep;
        }

        // Write one element.
        // Indices are relative to overall problem domain.
        ALWAYS_INLINE
        void write_elem(real_t val,
                        const Indices& idxs,
                        idx_t alloc_step_idx) {
            real_t* ep = get_elem_ptr(idxs, alloc_step_idx);
            *ep = val;
        }

    protected:

        // Get one adjusted index and recurse to i-1.
        template<int i>
        void _get_adj_idx(Indices& adj_idxs,
                          const Indices& vec_idxs,
                          idx_t alloc_step_idx) const {
            if constexpr (i < 0)
                             return;

            // Special handling for step index.
            constexpr auto sp = +step_posn;
            if constexpr (_use_step_idx && i == sp) {
                    host_assert(alloc_step_idx == _wrap_step(vec_idxs[sp]));
                    adj_idxs[i] = alloc_step_idx;
                }

            // Other indices.
            else {

                // Adjust for padding.
                // Since the indices are rank-relative, subtract only
                // the local offsets.
                // This gives a 0-based local *vector* index.
                adj_idxs[i] = vec_idxs[i] + _vec_left_pads[i] - _vec_local_offsets[i];
            }

            // Recurse (during compilation) until done.
            if constexpr (i > 0)
                             _get_adj_idx<i - 1>(adj_idxs, vec_idxs, alloc_step_idx);
            
        }
        
    public:
        // Get a pointer to given vector.
        // Indices must be normalized and rank-relative.
        // It's important that this function be efficient, since
        // it's used in the stencil kernel.
        ALWAYS_INLINE
        const real_vec_t* get_vec_ptr_norm(const Indices& vec_idxs,
                                           idx_t alloc_step_idx,
                                           bool check_bounds=true) const {

            constexpr int nvls = sizeof...(_templ_vec_lens);
            Indices adj_idxs(nvls);
            _get_adj_idx<nvls - 1>(adj_idxs, vec_idxs, alloc_step_idx);

            // Get ptr via layout in _data.
            return _data.get_ptr(adj_idxs, check_bounds);
        }

        // Non-const version.
        ALWAYS_INLINE
        real_vec_t* get_vec_ptr_norm(const Indices& vec_idxs,
                                     idx_t alloc_step_idx,
                                     bool check_bounds=true) {
            const real_vec_t* p =
                const_cast<const YkVecVarCore*>(this)->
                get_vec_ptr_norm(vec_idxs, alloc_step_idx, check_bounds);
            return const_cast<real_vec_t*>(p);
        }

        // Read one vector.
        // Indices must be normalized and rank-relative.
        // 'alloc_step_idx' is pre-calculated or 0 if not used.
        ALWAYS_INLINE
        real_vec_t read_vec_norm(const Indices& vec_idxs,
                                 idx_t alloc_step_idx) const {
            const real_vec_t* vp = get_vec_ptr_norm(vec_idxs, alloc_step_idx);
            real_vec_t res;
            res.load_from(vp);
            return res;
        }

        // Write one vector.
        // Indices must be normalized and rank-relative.
        // 'alloc_step_idx' is pre-calculated or 0 if not used.
        ALWAYS_INLINE
        void write_vec_norm(real_vec_t val,
                            const Indices& vec_idxs,
                            idx_t alloc_step_idx) {
            real_vec_t* vp = get_vec_ptr_norm(vec_idxs, alloc_step_idx);
            val.store_to(vp);
        }
        ALWAYS_INLINE
        void write_vec_norm_masked(real_vec_t val,
                                   const Indices& vec_idxs,
                                   idx_t alloc_step_idx,
                                   uidx_t mask) {
            real_vec_t* vp = get_vec_ptr_norm(vec_idxs, alloc_step_idx);
            val.store_to_masked(vp, mask);
        }

        // Prefetch one vector.
        // Indices must be normalized and rank-relative.
        // 'alloc_step_idx' is pre-calculated or 0 if not used.
        template <int level>
        ALWAYS_INLINE
        void prefetch_vec_norm(const Indices& vec_idxs,
                               idx_t alloc_step_idx,
                               int line) const {
            auto p = get_vec_ptr_norm(vec_idxs, alloc_step_idx, false);
            prefetch<level>(p);
            #ifdef MODEL_CACHE
            cache_model.prefetch(p, level, line);
            #endif
        }

    }; // YkVecVarCore.

    ///// Yk*Var* types /////

    // Base class implementing all yk_var functionality. Used for
    // vars that contain either individual elements or vectors.
    // This class is pure virtual.
    class YkVarBase :
        public KernelStateBase {
        friend class YkVarImpl;

    public:

        // Index for distinguishing my var from neighbors' vars.
        enum dirty_idx { self, others };

        // Reduction result.
        class red_res : public yk_var::yk_reduction_result {

        protected:
            char _pad[CACHELINE_BYTES]; // prevent false sharing.

        public:
            int _mask = 0;
            idx_t _nred = 0;
            double _sum = 0.0;
            double _sum_sq = 0.0;
            double _prod = 1.0;
            double _max = std::numeric_limits<double>::min();
            double _min = std::numeric_limits<double>::max();

            virtual ~red_res() { }
        
            /// Get the allowed reductions.
            int get_reduction_mask() const {
                return _mask;
            }
        
            /// Get the number of elements reduced.
            idx_t get_num_elements_reduced() const {
                return _nred;
            }
        
            /// Get results
            double get_sum() const {
                if (_mask & yk_var::yk_sum_reduction)
                    return _sum;
                THROW_YASK_EXCEPTION("Sum reduction result was not requested in reduction_mask");
            }
            double get_sum_squares() const {
                if (_mask & yk_var::yk_sum_squares_reduction)
                    return _sum;
                THROW_YASK_EXCEPTION("Sum-of-squares reduction result was not requested in reduction_mask");
            }
            double get_product() const {
                if (_mask & yk_var::yk_product_reduction)
                    return _prod;
                THROW_YASK_EXCEPTION("Product reduction result was not requested in reduction_mask");
            }
            double get_max() const {
                if (_mask & yk_var::yk_max_reduction)
                    return _max;
                THROW_YASK_EXCEPTION("Max reduction result was not requested in reduction_mask");
            }
            double get_min() const {
                if (_mask & yk_var::yk_sum_reduction)
                    return _min;
                THROW_YASK_EXCEPTION("Min reduction result was not requested in reduction_mask");
            }
        };
        
    protected:

        // Ptr to the core data.
        YkVarBaseCore* _corep = 0;

        // The following masks have one bit for each dim in the var.
        idx_t _step_dim_mask;
        idx_t _domain_dim_mask;
        idx_t _misc_dim_mask;

        // Counts of each dim type.
        idx_t _num_step_dims;
        idx_t _num_domain_dims;
        idx_t _num_misc_dims;

        // Whether step dim is used.
        // If true, will always be in step_posn.
        bool _has_step_dim = false;

        // Whether certain dims can be changed.
        bool _is_dynamic_step_alloc = false;
        bool _is_dynamic_misc_alloc = false;

        // Max L1 dist of halo accesses to this var.
        int _l1_dist = 0;

        // Whether to resize this var based on solution parameters.
        bool _fixed_size = false;

        // Scratch info.
        bool _is_scratch = false;
        int _scratch_mem_slot = -1;
        int _misc_mult = 1;     // product of all misc ranges (1 if none).

        // Whether this was created via an API.
        bool _is_user_var = false;

        // Tracking flags for data modified since last halo exchange.
        // [self]: Data needs to be copied to neighbors' halos of this var.
        // [others]: Data *may* need to be copied from one or neighbors into
        // the halo of this var.
        // vector contents: If this var has the step dim, there is one flag
        // per alloc'd step.  Otherwise, only [0] is used.
        std::vector<bool> _dirty_steps[2];

        // Coherency of device data.
        Coherency _coh;

        // Convenience function to format indices like
        // "x=5, y=3".
        virtual std::string make_index_string(const Indices& idxs,
                                              std::string separator=", ",
                                              std::string infix="=",
                                              std::string prefix="",
                                              std::string suffix="") const;

        // Check whether dim exists and is of allowed type.
        virtual void check_dim_type(const std::string& dim,
                                    const std::string& fn_name,
                                    bool step_ok,
                                    bool domain_ok,
                                    bool misc_ok) const;

        // Index math.
        inline idx_t get_first_local_index(idx_t posn) const {
            return _corep->get_first_local_index(posn);
        }
        inline idx_t get_last_local_index(idx_t posn) const {
            return _corep->get_last_local_index(posn);
        }

        // Make sure indices are in range.
        // Optionally fix them to be in range and return in 'fixed_indices'.
        // If 'normalize', make rank-relative, divide by vlen and return in 'fixed_indices'.
        bool check_indices(const Indices& indices,
                           const std::string& fn,    // name for error msg.
                           bool strict_indices, // die if out-of-range.
                           bool check_step,     // check step index.
                           bool normalize = false,      // div by vec lens.
                           Indices* fixed_indices = NULL) const;

        // Resize or fail if already allocated.
        void resize();

        // Set my dirty flags in range.
        void set_dirty_in_slice(const Indices& first_indices,
                                const Indices& last_indices);

        // Find sizes needed for slicing.
        inline Indices get_slice_range(const Indices& first_indices,
                                       const Indices& last_indices) const {
            host_assert(first_indices.get_num_dims() == last_indices.get_num_dims());
            Indices range = first_indices;
            for (int i = 0; i < first_indices.get_num_dims(); i++)
                range[i] = std::max(idx_t(0), last_indices[i] + 1 - first_indices[i]);
            return range;
        }

        // Accessors to GenericVar.
        virtual GenericVarBase* get_gvbp() =0;
        virtual const GenericVarBase* get_gvbp() const =0;

        // Sync core on device.
        // Does NOT sync underlying var data; see
        // copy_data_{to,from}_device().
        virtual void sync_core() =0;
        
        // Ctor.
        // Important: *corep exists but is NOT yet constructed.
        YkVarBase(KernelStateBase& stateb,
                  YkVarBaseCore* corep,
                  const VarDimNames& dim_names);

        // Dtor.
        virtual ~YkVarBase() { }

    public:

        // Accessors to core.
        YkVarBaseCore* get_corep() {
            return _corep;
        }
        const YkVarBaseCore* get_corep() const {
            return _corep;
        }
        
        // Wrappers to GenericVar.
        const IdxTuple& get_dim_tuple() const {
            return get_gvbp()->get_dim_tuple();
        }
        const std::string& get_name() const {
            return get_gvbp()->get_name();
        }
        bool is_dim_used(const std::string& dim) const {
            return get_gvbp()->is_dim_used(dim);
        }
        const std::string& get_dim_name(int n) const {
            return get_gvbp()->get_dim_name(n);
        }
        idx_t get_dim_size(int n) const {
            return get_gvbp()->get_dim_size(n);
        }
        void set_dim_size(int n, idx_t size) {
            get_gvbp()->set_dim_size(n, size);
        }
        int get_numa_pref() const {
            return get_gvbp()->get_numa_pref();
        };
        bool set_numa_pref(int numa_node) {
            return get_gvbp()->set_numa_pref(numa_node);
        };
        void default_alloc() {
            get_gvbp()->default_alloc();
        };
        void release_storage() {
            get_gvbp()->release_storage(true);
        };
        void* get_storage() {
            return get_gvbp()->get_storage();
        };
        const void* get_storage() const {
            return get_gvbp()->get_storage();
        };
        size_t get_num_bytes() const {
            return get_gvbp()->get_num_bytes();
        };
        void set_storage(std::shared_ptr<char>& base, size_t offset) {
            return get_gvbp()->set_storage(base, offset);
        };

        // Num dims in this var.
        // Not necessarily same as stencil problem.
        inline int get_num_dims() const {
            return _corep->_domains.get_num_dims();
        }

        // Num domain dims in this var.
        inline int get_num_domain_dims() const {
            return _num_domain_dims;
        }

        // Dims same?
        bool are_dims_and_sizes_same(const YkVarBase& src) const {
            return get_num_bytes() == src.get_num_bytes() &&
                get_dim_tuple() == src.get_dim_tuple();
        }

        // Step-indices.
        void update_valid_step(idx_t t);
        inline void update_valid_step(const Indices& indices) {
            if (_has_step_dim)
                update_valid_step(indices[+step_posn]);
        }
        inline void init_valid_steps() {
            if (_has_step_dim)
                _corep->_local_offsets[+step_posn] = 0;
        }

        // Halo-exchange flag accessors.
        virtual bool is_dirty(dirty_idx whose, idx_t step_idx) const;
        virtual void set_dirty(dirty_idx whose, bool dirty, idx_t step_idx);
        inline void set_dirty_using_alloc_index(dirty_idx whose, bool dirty, idx_t alloc_idx) {
            _dirty_steps[whose][alloc_idx] = dirty;
        }
        virtual void set_dirty_all(dirty_idx whose, bool dirty);

        // Coherency.
        const Coherency& get_coh() const { return _coh; }
        Coherency& get_coh() { return _coh; }

        // Resize flag accessors.
        virtual void set_fixed_size(bool is_fixed) {
            _fixed_size = is_fixed;
            if (is_fixed) {
                _corep->_rank_offsets.set_from_const(0);
                _is_dynamic_step_alloc = true;
                _is_dynamic_misc_alloc = true;
            }
        }
        virtual void _set_dynamic_step_alloc(bool is_dynamic) {
            _is_dynamic_step_alloc = is_dynamic;
        }
        virtual void _set_dynamic_misc_alloc(bool is_dynamic) {
            _is_dynamic_misc_alloc = is_dynamic;
        }

        // Does this var cover the n-D domain?
        virtual bool is_domain_var() const;

        // Scratch accessors.
        virtual bool is_scratch() const {
            return _is_scratch;
        }
        virtual void set_scratch(bool is_scratch) {
            _is_scratch = is_scratch;
            if (is_scratch)
                _corep->_rank_offsets.set_from_const(0);
        }
        virtual int get_scratch_mem_slot() const {
            assert(_is_scratch);
            return _scratch_mem_slot;
        }
        virtual void set_scratch_mem_slot(int ms) {
            assert(_is_scratch);
            _scratch_mem_slot = ms;
        }
        virtual int get_misc_mult() const {
            return _misc_mult;
        }
        virtual void set_misc_mult(int mm) {
            _misc_mult = mm;
        }

        // New-var accessors.
        virtual bool is_user_var() const {
            return _is_user_var;
        }
        virtual void set_user_var(bool is_user_var) {
            _is_user_var = is_user_var;
            if (_is_user_var) {
                _is_dynamic_step_alloc = true;
                _is_dynamic_misc_alloc = true;
            }
        }

        // Lookup position by dim name.
        // Return -1 or die if not found, depending on flag.
        virtual int get_dim_posn(const std::string& dim,
                                 bool die_on_failure = false,
                                 const std::string& die_msg = "") const;

        // Adjust logical time index to 0-based index
        // using temporal allocation size.
        inline idx_t _wrap_step(idx_t t) const {
            return _corep->_wrap_step(t);
        }

        // Convert logical step index to index in allocated range.
        // If this var doesn't use the step dim, returns 0.
        inline idx_t get_alloc_step_index(const Indices& indices) const {
            return _has_step_dim ? _wrap_step(indices[+step_posn]) : 0;
        }

        // Get var dims with allocations in number of reals.
        virtual IdxTuple get_allocs() const {
            IdxTuple allocs = get_dim_tuple(); // make a copy.
            _corep->_allocs.set_tuple_vals(allocs);
            return allocs;
        }

        // Make a human-readable description of the var.
        virtual std::string _make_info_string() const =0;
        virtual std::string make_info_string(bool long_info = false) const;

        // Print one element.
        virtual void print_elem(const std::string& msg,
                                const Indices& idxs,
                                real_t e,
                                int line) const;

        // Print one vector.
        // Indices must be normalized and rank-relative.
        virtual void print_vec_norm(const std::string& msg,
                                    const Indices& idxs,
                                    const real_vec_t& val,
                                    int line) const;

        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const YkVarBase* ref,
                              real_t epsilon = EPSILON,
                              int max_print = 20) const;

        // Copy data to/from device.
        void copy_data_to_device();
        void copy_data_from_device();

        // Versions that lie about being 'const' so we can copy data to/from
        // the device without changing the API user's view that it has not changed.
        inline void const_copy_data_to_device() const {
            const_cast<YkVarBase*>(this)->copy_data_to_device();
        }
        inline void const_copy_data_from_device() const {
            const_cast<YkVarBase*>(this)->copy_data_from_device();
        }

        // Set elements.
        virtual void set_all_elements_in_seq(double seed) =0;
        virtual void set_all_elements_same(double val) =0;

        // Set/get_elements_in_slice().
        virtual idx_t set_elements_in_slice_same(double val,
                                                 const Indices& first_indices,
                                                 const Indices& last_indices,
                                                 bool strict_indices,
                                                 bool on_device) =0;
        virtual idx_t set_elements_in_slice(const double* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) =0;
        virtual idx_t set_elements_in_slice(const float* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) =0;
        idx_t set_elements_in_slice_void(const void* buffer_ptr,
                                         const Indices& first_indices,
                                         const Indices& last_indices,
                                         bool on_device) {
            #if REAL_BYTES == 8
            return set_elements_in_slice((double*)buffer_ptr, IDX_MAX,
                                         first_indices, last_indices,
                                         on_device);
            #elif (REAL_BYTES == 4)
            return set_elements_in_slice((float*)buffer_ptr, IDX_MAX,
                                         first_indices, last_indices,
                                         on_device);
            #else
            #error Unsupported REAL_BYTES
            #endif
        }

        virtual idx_t get_elements_in_slice(double* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) const =0;
        virtual idx_t get_elements_in_slice(float* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) const =0;
        idx_t get_elements_in_slice_void(void* buffer_ptr,
                                         const Indices& first_indices,
                                         const Indices& last_indices,
                                         bool on_device) const {
            #if REAL_BYTES == 8
            return get_elements_in_slice((double*)buffer_ptr, IDX_MAX,
                                         first_indices, last_indices,
                                         on_device);
            #elif (REAL_BYTES == 4)
            return get_elements_in_slice((float*)buffer_ptr, IDX_MAX,
                                         first_indices, last_indices,
                                         on_device);
            #else
            #error Unsupported REAL_BYTES
            #endif
        }

        // Reductions.
        virtual yk_var::yk_reduction_result_ptr
        reduce_elements_in_slice(int reduction_mask,
                                 const Indices& first_indices,
                                 const Indices& last_indices,
                                 bool strict_indices,
                                 bool on_device) const =0;

        // Possibly vectorized version of set/get_elements_in_slice().
        virtual idx_t set_vecs_in_slice(const void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices,
                                        bool on_device) =0;
        virtual idx_t get_vecs_in_slice(void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices,
                                        bool on_device) const =0;

        // Get strides in underlying storage.
        virtual Indices get_vec_strides() const =0;

        // Get a pointer to one element.
        // Indices are relative to overall problem domain.
        // 'alloc_step_idx' is the pre-computed step index "wrapped"
        // to be within the allocated space. This avoids lots of 'idiv' instrs.
        // Methods are implemented in concrete classes for efficiency.
        virtual const real_t* get_elem_ptr(const Indices& idxs,
                                           idx_t alloc_step_idx,
                                           bool check_bounds=true) const =0;
        virtual real_t* get_elem_ptr(const Indices& idxs,
                                     idx_t alloc_step_idx,
                                     bool check_bounds=true) =0;

        // Read one element.
        // Indices are relative to overall problem domain.
        virtual real_t read_elem(const Indices& idxs,
                                 idx_t alloc_step_idx,
                                 int line) const =0;

        // Write one element.
        // Indices are relative to overall problem domain.
        virtual void write_elem(real_t val,
                                const Indices& idxs,
                                idx_t alloc_step_idx,
                                int line) =0;

        // Update one element atomically.
        // Indices are relative to overall problem domain.
        inline void add_to_elem(real_t val,
                                const Indices& idxs,
                                idx_t alloc_step_idx,
                                int line) {
            real_t* ep = get_elem_ptr(idxs, alloc_step_idx);

            #pragma omp atomic update
            *ep += val;
            #ifdef TRACE_MEM
            print_elem("add_to_elem", idxs, *ep, line);
            #endif
        }

    protected:
        // Templated method to visit points in a slice.
        // Visitor should implement the following:
        //  static const char* fname();  // name of calling function.
        //  static void visit(YkVarBase* varp,  // 'this' ptr.
        //                    T* p,  // copy of 'buffer_ptr', if needed.
        //                    idx_t pofs, // offset into buffer, if needed.
        //                    T val, // some value, if needed.
        //                    const Indices& pt, // point in 'this' var.
        //                    idx_t ti,  // precomputed step index.
        //                    int thread); // thread number.
        template<typename Visitor, typename T, typename VarT>
        idx_t _visit_elements_in_slice(bool strict_indices,
                                       T* buffer_ptr,
                                       size_t buffer_size,
                                       T val,
                                       const Indices& in_first_indices,
                                       const Indices& in_last_indices,
                                       bool on_device) {
            STATE_VARS(this);
            if (get_storage() == 0) {
                THROW_YASK_EXCEPTION(std::string("call to '") +
                                     Visitor::fname() +
                                     "' with no storage allocated for var '" +
                                     get_name() + "'");
            }
            if (buffer_ptr == 0) {
                THROW_YASK_EXCEPTION(std::string("call to '") +
                                     Visitor::fname() + "' with NULL buffer pointer");
            }

            TRACE_MSG(Visitor::fname() << ": " << make_info_string() << " [" <<
                      make_index_string(in_first_indices) << " ... " <<
                      make_index_string(in_last_indices) << "] with buffer at " <<
                      buffer_ptr << " on " << (on_device ? "OMP device" : "host"));
            Indices first_indices, last_indices;
            check_indices(in_first_indices, Visitor::fname(), strict_indices,
                          true, false, &first_indices);
            check_indices(in_last_indices, Visitor::fname(), strict_indices,
                          true, false, &last_indices);
            TRACE_MSG(Visitor::fname() << ": clipped to  [" <<
                      make_index_string(first_indices) << " ... " <<
                      make_index_string(last_indices) << "]");

            // Find range.
            auto range = get_slice_range(first_indices, last_indices);
            auto ne = range.product();
            TRACE_MSG(Visitor::fname() << ": " << ne << " element(s) in shape " <<
                      make_index_string(range));
            if (ne <= 0)
                return 0;
            if (buffer_size < size_t(ne))
                THROW_YASK_EXCEPTION(std::string("call to '") +
                                     Visitor::fname() + "' with buffer of size " +
                                     std::to_string(buffer_size) + "; " +
                                     std::to_string(ne) + " needed");

            // Iterate through step index in outer loop.
            // This avoids calling _wrap_step(t) at every point.
            const auto sp = +step_posn;
            idx_t first_t = 0, last_t = 0;
            int ndims_left = get_num_dims();
            if (_has_step_dim) {
                first_t = first_indices[sp];
                last_t = last_indices[sp];
                range[sp] = 1; // Do one step per iter.
                ndims_left--;
            }

            // Amount to advance pointer each step.
            idx_t tsz = range.product();
            idx_t tofs = 0;
            TRACE_MSG(Visitor::fname() << ": " << tsz << " element(s) in shape " <<
                      make_index_string(range) << " for each step");
            if (ndims_left == 0)
                assert(tsz == 1);
            
            // Iterate through inner index in inner loop if there is enough
            // work to do.  This may enable more optimization.
            idx_t ni = 1;       // Inner iterations.
            const auto ip = get_num_dims() - 1; // Inner index.
            if (ndims_left) {
                idx_t osz = tsz / range[ip]; // Work in non-inner indices.
                if (osz >= yask_get_num_threads()) {
                    ni = range[ip]; // Do whole range in each iter.
                    range[ip] = 1;  // Visit this dim only once.
                }
            }
            TRACE_MSG(Visitor::fname() << ": " << ni <<
                      " element(s) for each starting-point in shape " <<
                      make_index_string(range) << " for each inner loop");

            // Make copy of first_indices to use as starting point
            // of each step.
            auto start_indices(first_indices);

            // Outer loop through each step.
            for (idx_t t = first_t; t <= last_t; t++) {

                // Do only this one step in this iteration.
                idx_t ti = 0;
                if (_has_step_dim) {
                    ti = _wrap_step(t);
                    start_indices[sp] = t;
                }
  
                // Visit points in slice on host in parallel.
                if (!on_device) {
                    range.visit_all_points_in_parallel
                        (false,
                         [&](const Indices& ofs, size_t idx, int thread) {
                             auto pt = start_indices.add_elements(ofs);

                             // Inner loop.
                             for (idx_t i = 0; i < ni; i++) {
                                 idx_t bofs = tofs + idx * ni + i;
                                 #ifdef DEBUG_VISIT_SLICE
                                 TRACE_MSG(Visitor::fname() << ": visting pt " <<
                                           make_index_string(pt) << " w/buf ofs " << bofs);
                                 #endif
                             
                                 // Call visitor.
                                 Visitor::visit(static_cast<VarT*>(this),
                                                buffer_ptr, bofs, val,
                                                pt, ti, thread);

                                 // Advance to next point.
                                 if (ndims_left)
                                     pt[ip]++;
                             }

                             return true;    // keep going.
                         });
                }
        
                // TBD: Visit points in slice on device.
                else {
                    THROW_YASK_EXCEPTION(std::string("(internal fault) '") +
                                         Visitor::fname() + "' for var '" +
                                         get_name() + "' not implemented for offload device");
                }

                // Skip to next step in buffer.
                tofs += tsz;

            } // steps.
            TRACE_MSG(Visitor::fname() << " returns " << ne);
            return ne;
        } // _visit_elements_in_slice();

        // Read into buffer from *this.
        template<typename T, typename VarT>
        idx_t _get_elements_in_slice(T* buffer_ptr,
                                     size_t buffer_size,
                                     const Indices& first_indices,
                                     const Indices& last_indices,
                                     bool on_device) const {

            // A specialized visitor.
            struct GetElem {
                static const char* fname() {
                    return "get_elements_in_slice";
                }

                // Copy from the var to the buffer.
                ALWAYS_INLINE
                static void visit(VarT* varp,
                                  T* p, idx_t pofs, T v,
                                  const Indices& pt, idx_t ti,
                                  int thread) {

                    // Read from var.
                    real_t val = varp->read_elem(pt, ti, __LINE__);

                    // Write to buffer at proper index, converting
                    // type if needed.
                    p[pofs] = T(val);
                }
            };

            if (on_device)
                const_copy_data_to_device();
            else
                const_copy_data_from_device();
        
            // Call the generic visit.
            auto n = dynamic_cast<VarT*>(const_cast<YkVarBase*>(this))->template
                _visit_elements_in_slice<GetElem, T, VarT>(true, buffer_ptr, buffer_size, 0,
                                                           first_indices, last_indices, on_device);

            // Return number of writes.
            return n;
        }

        // Write to *this from buffer.
        template<typename T, typename VarT>
        idx_t _set_elements_in_slice(const T* buffer_ptr,
                                     size_t buffer_size,
                                     const Indices& first_indices,
                                     const Indices& last_indices,
                                     bool on_device) {
            
            // A specialized visitor.
            struct SetElem {
                static const char* fname() {
                    return "set_elements_in_slice";
                }

                // Copy from the buffer to the var.
                ALWAYS_INLINE
                static void visit(VarT* varp,
                                  T* p, idx_t pofs, T v,
                                  const Indices& pt, idx_t ti,
                                  int thread) {

                    // Read from buffer, converting type if needed.
                    real_t val = p[pofs];

                    // Write to var
                    varp->write_elem(val, pt, ti, __LINE__);
                }
            };

            if (on_device)
                const_copy_data_to_device();
            else
                const_copy_data_from_device();
        
            // Call the generic visit.
            auto n = dynamic_cast<VarT*>(this)->template
                _visit_elements_in_slice<SetElem, T, VarT>(true, (T*)buffer_ptr, buffer_size, 0,
                                                           first_indices, last_indices, on_device);
            
            // Set appropriate dirty flags.
            if (on_device)
                _coh.mod_dev();
            else
                _coh.mod_host();
            set_dirty_in_slice(first_indices, last_indices);

            // Return number of writes.
            return n;
        }

        // Write to *this from 'val'.
        template<typename T, typename VarT>
        idx_t _set_elements_in_slice_same(T val,
                                          const Indices& first_indices,
                                          const Indices& last_indices,
                                          bool strict_indices,
                                          bool on_device) {
            // A specialized visitor.
            struct SetElem {
                static const char* fname() {
                    return "set_elements_in_slice_same";
                }

                // Set the var.
                ALWAYS_INLINE
                static void visit(VarT* varp,
                                  T* p, idx_t pofs, T val,
                                  const Indices& pt, idx_t ti,
                                  int thread) {

                    // Write val to var
                    varp->write_elem(val, pt, ti, __LINE__);
                }
            };

            if (on_device)
                const_copy_data_to_device();
            else
                const_copy_data_from_device();
        
            // Call the generic visit.
            auto n = dynamic_cast<VarT*>(this)->template
                _visit_elements_in_slice<SetElem, T, VarT>(strict_indices,
                                                           (T*)-1, IDX_MAX, val,
                                                           first_indices, last_indices,
                                                           on_device);
            
            // Set appropriate dirty flags.
            if (on_device)
                _coh.mod_dev();
            else
                _coh.mod_host();
            set_dirty_in_slice(first_indices, last_indices);

            // Return number of writes.
            return n;
        }
    
        // Perform reduction(s).
        template<typename T, typename VarT>
        yk_var::yk_reduction_result_ptr
        _reduce_elements_in_slice(int reduction_mask,
                                  const Indices& first_indices,
                                  const Indices& last_indices,
                                  bool strict_indices,
                                  bool on_device) const {
            // A specialized visitor.
            struct RedElem {
                static const char* fname() {
                    return "reduce_elements_in_slice";
                }

                // Do the reduction(s).
                ALWAYS_INLINE
                static void visit(VarT* varp,
                                  T* p, idx_t pofs, T v,
                                  const Indices& pt, idx_t ti,
                                  int thread) {

                    // Get value, converting to double if needed.
                    double val = varp->read_elem(pt, ti, __LINE__);

                    // Use p as a pointer to the result for this thread.
                    // TODO: clean up this cast.
                    red_res* resa = (red_res*)p;
                    assert(thread < yask_get_num_threads());
                    red_res* resp = resa + thread;

                    // Do desired reduction(s).
                    int mask = resp->_mask;
                    if (mask & yk_var::yk_sum_reduction)
                        resp->_sum += val;
                    if (mask & yk_var::yk_sum_squares_reduction)
                        resp->_sum_sq += val * val;
                    if (mask & yk_var::yk_product_reduction)
                        resp->_prod *= val;
                    if (mask & yk_var::yk_max_reduction)
                        resp->_max = std::max(resp->_max, val);
                    if (mask & yk_var::yk_min_reduction)
                        resp->_min = std::min(resp->_min, val);
                }
            };

            if (on_device)
                const_copy_data_to_device();
            else
                const_copy_data_from_device();

            // Make array of results, one for each thread,
            // so we don't have to use atomics or critical sections.
            int nthr = yask_get_num_threads();
            std::vector<red_res> rrv;
            rrv.resize(nthr);
            for (int i = 0; i < nthr; i++)
                rrv[i]._mask = reduction_mask;
        
            // Call the generic visit.
            // TODO: clean up ptr cast.
            auto n = dynamic_cast<VarT*>(const_cast<YkVarBase*>(this))->template
                _visit_elements_in_slice<RedElem, T, VarT>(strict_indices,
                                                           (T*)rrv.data(), IDX_MAX, 0,
                                                           first_indices, last_indices,
                                                           on_device);

            // Make final result.
            auto resp = std::make_shared<red_res>();
            resp->_mask = reduction_mask;
            resp->_nred = n;

            // Join per-thread results.
            for (int i = 0; i < nthr; i++) {
                auto* p = resp.get();
                auto* resp = &rrv.at(i);
                p->_sum += resp->_sum;
                p->_sum_sq += resp->_sum;
                p->_prod *= resp->_prod;
                p->_max = std::max(p->_max, resp->_max);
                p->_min = std::min(p->_min, resp->_min);
            }

            return resp;
        }      
        
    }; // YkVarBase.
    typedef std::shared_ptr<YkVarBase> VarBasePtr;

    // YASK var of real elements.
    // Used for vars that do not contain folded vectors.
    // If '_use_step_idx', then index to step dim will wrap around.
    template <typename LayoutFn, bool _use_step_idx>
    class YkElemVar final : public YkVarBase {

    public:
        // Type for core data.
        typedef YkElemVarCore<LayoutFn, _use_step_idx> core_t;
        static_assert(std::is_trivially_copyable<core_t>::value,
                      "Needed for OpenMP offload");

    protected:

        // Core data.
        // This also contains the storage core: _core._data.
        core_t _core;

        // Storage meta-data.
        // Owned here via composition.
        // This contains a pointer to _core._data.
        GenericVar<real_t, LayoutFn> _data;

        // Accessors to GenericVar.
        virtual GenericVarBase* get_gvbp() override final {
            return &_data;
        }
        virtual const GenericVarBase* get_gvbp() const override final {
            return &_data;
        }

        // Sync core meta-data on device.
        // Does NOT sync underlying var data; see
        // copy_data_{to,from}_device().
        void sync_core() override {
            STATE_VARS(this);
            auto* var_cp = &_core;
            offload_copy_to_device(var_cp, 1);
            _data.sync_data_ptr();
        }
        
    public:
        YkElemVar(KernelStateBase& stateb,
                  std::string name,
                  const VarDimNames& dim_names) :
            YkVarBase(stateb, &_core, dim_names),
            _core(int(dim_names.size())),
            _data(stateb, &_core._data, name, dim_names) {
            STATE_VARS(this);
            TRACE_MSG("creating element-var '" + get_name() + "'");
            _has_step_dim = _use_step_idx;

            // Init vec sizes.
            // A non-vectorized var still needs to know about
            // the solution folding of its dims for proper
            // padding.
            for (size_t i = 0; i < dim_names.size(); i++) {
                auto& dname = dim_names.at(i);
                auto* p = dims->_vec_fold_pts.lookup(dname);
                idx_t dval = p ? *p : 1;
                _core._soln_vec_lens[i] = dval;
                _core._var_vec_lens[i] = 1;
            }

            // Create core on offload device.
            auto* var_cp = &_core;
            offload_map_alloc(var_cp, 1);
             
            resize();
        }

        // Dtor.
        virtual ~YkElemVar() {
            STATE_VARS(this);

            // Release core from device.
            auto* var_cp = &_core;
            offload_map_free(var_cp, 1);
        }
        
        // Make a human-readable description.
        virtual std::string _make_info_string() const override final {
            return _data.make_info_string("FP");
        }

        // Init data.
        void set_all_elements_same(double val) override final {
            TRACE_MSG("setting all elements in '" + get_name() + "' to " << val);
            _coh._force_state(Coherency::not_init); // because all values will be written.
            _data.set_elems_same(val);
            set_dirty_all(self, true);
            _coh.mod_both();
        }
        void set_all_elements_in_seq(double seed) override final {
            TRACE_MSG("setting all elements in '" + get_name() + "' using seed " << seed);
            _coh._force_state(Coherency::not_init); // because all values will be written.
            _data.set_elems_in_seq(seed);
            set_dirty_all(self, true);
            _coh.mod_both();
        }

        // Get a pointer to given element.
        const real_t* get_elem_ptr(const Indices& idxs,
                                   idx_t alloc_step_idx,
                                   bool check_bounds=true) const override final {
            return _core.get_elem_ptr(idxs, alloc_step_idx, check_bounds);
        }

        // Non-const version.
        real_t* get_elem_ptr(const Indices& idxs,
                             idx_t alloc_step_idx,
                             bool check_bounds=true) override final {
            return _core.get_elem_ptr(idxs, alloc_step_idx, check_bounds);
        }

        // Read one element.
        // Indices are relative to overall problem domain.
        real_t read_elem(const Indices& idxs,
                         idx_t alloc_step_idx,
                         int line) const override final {
            real_t e = _core.read_elem(idxs, alloc_step_idx);
            #ifdef TRACE_MEM
            print_elem("read_elem", idxs, e, line);
            #endif
            return e;
        }

        // Write one element.
        // Indices are relative to overall problem domain.
        void write_elem(real_t val,
                        const Indices& idxs,
                        idx_t alloc_step_idx,
                        int line) override final {
            _core.write_elem(val, idxs, alloc_step_idx);
            #ifdef TRACE_MEM
            print_elem("write_elem", idxs, val, line);
            #endif
        }

        // Get strides in underlying storage.
        // This will be element strides in this class.
        virtual Indices get_vec_strides() const override {
            return _data.get_strides();
        }

        // Non-vectorized fall-back versions.
        virtual idx_t set_vecs_in_slice(const void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices,
                                        bool on_device) override {
            return YkVarBase::set_elements_in_slice_void(buffer_ptr,
                                                         first_indices, last_indices, on_device);
        }
        virtual idx_t get_vecs_in_slice(void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices,
                                        bool on_device) const override {
            return YkVarBase::get_elements_in_slice_void(buffer_ptr,
                                                         first_indices, last_indices, on_device);
        }

        // Read into buffer from *this.
        virtual idx_t get_elements_in_slice(double* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) const override {
            return _get_elements_in_slice<double, YkElemVar>(buffer_ptr, buffer_size,
                                                             first_indices, last_indices,
                                                             on_device);
        }
        virtual idx_t get_elements_in_slice(float* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) const override {
            return _get_elements_in_slice<float, YkElemVar>(buffer_ptr, buffer_size,
                                                            first_indices, last_indices,
                                                            on_device);
        }

        // Write to *this from buffer.
        virtual idx_t set_elements_in_slice(const double* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) override {
            return _set_elements_in_slice<double, YkElemVar>(buffer_ptr, buffer_size,
                                                             first_indices, last_indices,
                                                             on_device);
        }
        virtual idx_t set_elements_in_slice(const float* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) override {
            return _set_elements_in_slice<float, YkElemVar>(buffer_ptr, buffer_size,
                                                            first_indices, last_indices,
                                                            on_device);
        }

        // Write to *this from val.
        virtual idx_t set_elements_in_slice_same(double val,
                                                 const Indices& first_indices,
                                                 const Indices& last_indices,
                                                 bool strict_indices,
                                                 bool on_device) override {
            return _set_elements_in_slice_same<double, YkElemVar>(val,
                                                                  first_indices, last_indices,
                                                                  strict_indices, on_device);
        }
        
        // Reduce elements.
        virtual yk_var::yk_reduction_result_ptr
        reduce_elements_in_slice(int reduction_mask,
                                 const Indices& first_indices,
                                 const Indices& last_indices,
                                 bool strict_indices,
                                 bool on_device) const override {
            return _reduce_elements_in_slice<double, YkElemVar>(reduction_mask,
                                                                first_indices, last_indices,
                                                                strict_indices, on_device);
        }
        
    };                          // YkElemVar.

    // YASK var of real vectors.
    // Used for vars that contain all the folded dims.
    // If '_use_step_idx', then index to step dim will wrap around.
    // The '_templ_vec_lens' arguments must contain a list of vector lengths
    // corresponding to each dim in the var.
    template <typename LayoutFn, bool _use_step_idx, idx_t... _templ_vec_lens>
    class YkVecVar final : public YkVarBase {

    public:
        // Type for core data.
        typedef YkVecVarCore<LayoutFn, _use_step_idx, _templ_vec_lens...> core_t;
        static_assert(std::is_trivially_copyable<core_t>::value,
                      "Needed for OpenMP offload");
        
    protected:

        // Core data.
        // This also contains the storage core.
        core_t _core;

        // Storage meta-data.
        // Owned here via composition.
        // This contains a pointer to _core._data.
        GenericVar<real_vec_t, LayoutFn> _data;

        // Accessors to GenericVar.
        virtual GenericVarBase* get_gvbp() override final {
            return &_data;
        }
        virtual const GenericVarBase* get_gvbp() const override final {
            return &_data;
        }

        // Sync core on device.
        // Does NOT sync underlying var data; see
        // copy_data_{to,from}_device().
        void sync_core() override {
            STATE_VARS(this);
            auto* var_cp = &_core;
            offload_copy_to_device(var_cp, 1);
            _data.sync_data_ptr();
        }
        
    public:
        YkVecVar(KernelStateBase& stateb,
                 const std::string& name,
                 const VarDimNames& dim_names) :
            YkVarBase(stateb, &_core, dim_names),
            _core(int(dim_names.size())),
            _data(stateb, &_core._data, name, dim_names) {
            STATE_VARS(this);
            TRACE_MSG("creating vector-var '" + get_name() + "'");
            _has_step_dim = _use_step_idx;

            // Template vec lengths.
            const int nvls = sizeof...(_templ_vec_lens);
            const idx_t vls[nvls] { _templ_vec_lens... };
            assert((size_t)nvls == dim_names.size());

            // Init vec sizes.
            // A vectorized var must use all the vectorized
            // dims of the solution folding.
            // For each dim in the var, use the number of vector
            // fold points or 1 if not set.
            for (size_t i = 0; i < dim_names.size(); i++) {
                auto& dname = dim_names.at(i);
                auto* p = dims->_vec_fold_pts.lookup(dname);
                idx_t dval = p ? *p : 1;
                _corep->_soln_vec_lens[i] = dval;
                _corep->_var_vec_lens[i] = dval;

                // Must be same as that in template parameter pack.
                assert(dval == vls[i]);
            }

            // Init var-dim positions of fold dims.
            // TODO: figure out how to do this statically.
            assert(dims->_vec_fold_pts.get_num_dims() == NUM_VEC_FOLD_DIMS);
            for (int i = 0; i < NUM_VEC_FOLD_DIMS; i++) {
                auto& fdim = dims->_vec_fold_pts.get_dim_name(i);
                int j = get_dim_posn(fdim, true,
                                     "internal error: folded var missing folded dim");
                assert(j >= 0);
                _core._vec_fold_posns[i] = j;
            }

            // Create core on offload device.
            auto* var_cp = &_core;
            offload_map_alloc(var_cp, 1);

            resize();
        }

        // Dtor.
        virtual ~YkVecVar() {
            STATE_VARS(this);

            // Release core from device.
            auto* var_cp = &_core;
            offload_map_free(var_cp, 1);
        }
        
        // Make a human-readable description.
        std::string _make_info_string() const override final {
            return _data.make_info_string("SIMD FP");
        }

        // Init data.
        void set_all_elements_same(double val) override final {
            TRACE_MSG("setting all elements in '" + get_name() + "' to " << val);
            _coh._force_state(Coherency::not_init); // because all values will be written.
            real_vec_t valv = val; // bcast.
            _data.set_elems_same(valv);
            set_dirty_all(self, true);
            _coh.mod_both();
        }
        void set_all_elements_in_seq(double seed) override final {
            TRACE_MSG("setting all elements in '" + get_name() + "' using seed " << seed);
            _coh._force_state(Coherency::not_init); // because all values will be written.
            real_vec_t seedv;
            auto n = seedv.get_num_elems();

            // Init elements to decreasing multiples of seed.
            for (int i = 0; i < n; i++)
                seedv[i] = seed * (double(n - i));
            _data.set_elems_in_seq(seedv);
            set_dirty_all(self, true);
            _coh.mod_both();
        }

        // Get a pointer to given element.
        const real_t* get_elem_ptr(const Indices& idxs,
                                   idx_t alloc_step_idx,
                                   bool check_bounds=true) const override final {
            return _core.get_elem_ptr(idxs, alloc_step_idx, check_bounds);
        }

        // Non-const version.
        real_t* get_elem_ptr(const Indices& idxs,
                             idx_t alloc_step_idx,
                             bool check_bounds=true) override final {
            return _core.get_elem_ptr(idxs, alloc_step_idx, check_bounds);
        }

        // Read one element.
        // Indices are relative to overall problem domain.
        real_t read_elem(const Indices& idxs,
                         idx_t alloc_step_idx,
                         int line) const override final {
            auto val = _core.read_elem(idxs, alloc_step_idx);
            #ifdef TRACE_MEM
            print_elem("read_elem", idxs, val, line);
            #endif
            return val;
        }

        // Write one element.
        // Indices are relative to overall problem domain.
        void write_elem(real_t val,
                        const Indices& idxs,
                        idx_t alloc_step_idx,
                        int line) override final {
            _core.write_elem(val, idxs, alloc_step_idx);
            #ifdef TRACE_MEM
            print_elem("write_elem", idxs, val, line);
            #endif
        }

        // Get a pointer to given vector.
        // Indices must be normalized and rank-relative.
        // It's important that this function be efficient, since
        // it's indiectly used from the stencil kernel.
        const real_vec_t* get_vec_ptr_norm(const Indices& vec_idxs,
                                           idx_t alloc_step_idx,
                                           bool check_bounds=true) const {
            return _core.get_vec_ptr_norm(vec_idxs, alloc_step_idx, check_bounds);
        }

        // Non-const version.
        real_vec_t* get_vec_ptr_norm(const Indices& vec_idxs,
                                     idx_t alloc_step_idx,
                                     bool check_bounds=true) {
            return _core.get_vec_ptr_norm(vec_idxs, alloc_step_idx, check_bounds);
        }

        // Read one vector.
        // Indices must be normalized and rank-relative.
        // 'alloc_step_idx' is pre-calculated or 0 if not used.
        real_vec_t read_vec_norm(const Indices& vec_idxs,
                                 idx_t alloc_step_idx,
                                 int line) const {
            auto v = _core.read_vec_norm(vec_idxs, alloc_step_idx);
            #ifdef TRACE_MEM
            print_vec_norm("read_vec_norm", vec_idxs, v, line);
            #endif
            return v;
        }

        // Write one vector.
        // Indices must be normalized and rank-relative.
        // 'alloc_step_idx' is pre-calculated or 0 if not used.
        void write_vec_norm(real_vec_t val,
                            const Indices& vec_idxs,
                            idx_t alloc_step_idx,
                            int line) {
            _core.write_vec_norm(val, vec_idxs, alloc_step_idx);
            #ifdef TRACE_MEM
            print_vec_norm("write_vec_norm", vec_idxs, val, line);
            #endif
        }

    private:
        // Template for get/set_vecs_in_slice.
        // Input indices are global and element granularity (not rank-local or normalized).
        // This is similar to but simpler than _visit_elements_in_slice().
        template<typename Visitor>
        idx_t _copy_vecs_in_slice(void* buffer_ptr,
                                  const Indices& first_indices,
                                  const Indices& last_indices,
                                  bool on_device) {
            STATE_VARS(this);
            assert(_data.get_storage() != 0);

            // Use the core for efficiency and to allow offload.
            core_t* core_p = &_core;
            
            #ifdef USE_OFFLOAD_NO_USM
            if (on_device) {
                auto devn = KernelEnv::_omp_devn;

                // 'buffer_ptr' and 'core_p' should exist on device.
                assert(omp_target_is_present(buffer_ptr, devn));
                assert(omp_target_is_present(core_p, devn));
            }
            #endif
            
            Indices firstv, lastv;
            check_indices(first_indices, "copy_vecs_in_slice", true, false, true, &firstv);
            check_indices(last_indices, "copy_vecs_in_slice", true, false, true, &lastv);

            // Find range.
            auto vec_range = get_slice_range(firstv, lastv);
            auto nv = vec_range.product();
            auto ne = nv * VLEN;
            TRACE_MSG("copying " << nv << " vec(s) in " <<
                      make_info_string() << " [" <<
                      make_index_string(firstv) << " ... " <<
                      make_index_string(lastv) << "] with buffer at " <<
                      buffer_ptr << " on " << (on_device ? "OMP device" : "host"));
            if (nv < 1)
                return 0;

            // Iterate through step index in outer loop.
            // This avoids calling _wrap_step(t) at every point.
            auto sp = +step_posn;
            idx_t first_t = 0, last_t = 0;
            if (_has_step_dim) {
                first_t = firstv[sp];
                last_t = lastv[sp];
                vec_range[sp] = 1; // Do one step per iter.
            }

            // Amount to advance pointer each step.
            idx_t tsz = vec_range.product();
            idx_t tofs = 0;

            // Determine inner-loop dim.
            // Use last dim by default.
            auto ip = get_num_dims() - 1;

            // Use first non-step dim by default if inner loop dim doesn't match layout.
            // Remember that this var may not have either.
            if (dims->_inner_loop_dim != dims->_inner_layout_dim)
                ip = _has_step_dim ? 1 : 0;

            // Look for specified dim.
            // TODO: determine actual first or last layout dim.
            for (int i = 0; i < get_num_dims(); i++) {
                if (get_dim_name(i) == dims->_inner_loop_dim) {
                    ip = i;
                    break;
                }
            }

            // Extract inner-loop range and re-init it in 'vec_range'.
            idx_t ni = vec_range[ip];
            vec_range[ip] = 1; // Do whole range in each iter.

            // Inner-loop stride.
            idx_t si = core_p->_vec_strides[ip];

            // Outer loop through each step.
            for (idx_t t = first_t; t <= last_t; t++) {

                // Do only step 't' in this iteration.
                idx_t ti = 0;
                if (_has_step_dim) {
                    ti = _wrap_step(t);
                    firstv[sp] = t;
                }

                if (on_device) {
                    #ifdef USE_OFFLOAD
                    auto devn = KernelEnv::_omp_devn;
                    auto nj = vec_range.product();

                    // Run outer loop on device in parallel.
                    _Pragma("omp target teams distribute parallel for device(devn)")
                        for (idx_t j = 0; j < nj; j++) {

                            // Init vars for first point.
                            Indices ofs = vec_range.unlayout(false, j);
                            Indices pt = firstv.add_elements(ofs);
                            auto* vp = core_p->get_vec_ptr_norm(pt, ti);
                            idx_t bofs = tofs + j * ni;

                            // Inner loop. 
                            for (idx_t i = 0; i < ni; i++) {
                            
                                // Do the copy operation specified in visitor.
                                Visitor::do_copy(((real_vec_t*)buffer_ptr), bofs, vp);

                                // Next point in buffer and var.
                                vp += si;
                                bofs++;
                            }
                        }
                    #else
                    THROW_YASK_EXCEPTION("(internal fault) call to _copy_vecs_in_slice on device"
                                         " in non-offload build");
                    #endif
                }

                // Visit starting points in range on host in parallel.
                else {
                    vec_range.visit_all_points_in_parallel
                        (false,
                         [&](const Indices& ofs, size_t idx, int thread) {

                             // Init vars for first point.
                             auto pt = firstv.add_elements(ofs);
                             auto* vp = core_p->get_vec_ptr_norm(pt, ti);
                             idx_t bofs = tofs + idx * ni;
 
                             // Inner loop.
                             for (idx_t i = 0; i < ni; i++) {
                             
                                 // Do the copy operation specified in visitor.
                                 Visitor::do_copy(((real_vec_t*)buffer_ptr), bofs, vp);
                             
                                 // Next point in buffer and var.
                                 vp += si;
                                 bofs++;
                             }
                             return true;    // keep going.
                         });
                }

                // Skip to next step in buffer.
                tofs += tsz;
                
            } // time steps.

            assert(tofs == nv);
            return ne;
        }

    public:
        // Vectorized version of set_elements_in_slice().
        // Input indices are global and element granularity (not rank-local or normalized).
        virtual idx_t set_vecs_in_slice(const void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices,
                                        bool on_device = false) override {

            // Specialize do_copy() to copy from buffer to var.
            // Could have used a lambda, but this avoids possible conversion to std::function.
            struct SetVec {
                ALWAYS_INLINE
                static void do_copy(real_vec_t* p, idx_t pofs,
                                    real_vec_t* vp) {
                    
                    // Read vec from buffer.
                    real_vec_t val = p[pofs];
                    
                    // Write to var.
                    val.store_to(vp);
                }
            };

            if (on_device)
                const_copy_data_to_device();
            else
                const_copy_data_from_device();
            
            // Call the generic vec copier.
            auto nset = _copy_vecs_in_slice<SetVec>((void*)buffer_ptr,
                                                    first_indices, last_indices,
                                                    on_device);

            // Set appropriate dirty flag(s).
            if (on_device)
                _coh.mod_dev();
            else
                _coh.mod_host();
            set_dirty_in_slice(first_indices, last_indices);

            return nset;
        }

        // Vectorized version of get_elements_in_slice().
        // Input indices are global and element granularity (not rank-local or normalized).
        virtual idx_t get_vecs_in_slice(void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices,
                                        bool on_device) const override final {

            // Specialize do_copy() to copy to buffer from var.
            // Could have used a lambda, but this avoids possible conversion to std::function.
            struct GetVec {
                ALWAYS_INLINE
                static void do_copy(real_vec_t* p, idx_t pofs,
                                    real_vec_t* vp) {

                    // Read vec from var.
                    real_vec_t res;
                    res.load_from(vp);

                    // Write to buffer at proper index.
                    p[pofs] = res;
                }
            };

            if (on_device)
                const_copy_data_to_device();
            else
                const_copy_data_from_device();

            // Call the generic vec copier.
            auto n = const_cast<YkVecVar*>(this)->
                _copy_vecs_in_slice<GetVec>((void*)buffer_ptr,
                                            first_indices, last_indices, on_device);
            
            // Return number of writes.
            return n;
        }

        // Get strides in underlying storage.
        // This will be vector strides in this class.
        virtual Indices get_vec_strides() const override {
            return _data.get_strides();
        }

        // Read into buffer from *this.
        virtual idx_t get_elements_in_slice(double* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) const override {
            return _get_elements_in_slice<double, YkVecVar>(buffer_ptr, buffer_size,
                                                            first_indices, last_indices,
                                                            on_device);
        }
        virtual idx_t get_elements_in_slice(float* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) const override {
            return _get_elements_in_slice<float, YkVecVar>(buffer_ptr, buffer_size,
                                                           first_indices, last_indices,
                                                           on_device);
        }

        // Write to *this from buffer.
        virtual idx_t set_elements_in_slice(const double* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) override {
            return _set_elements_in_slice<double, YkVecVar>(buffer_ptr, buffer_size,
                                                            first_indices, last_indices,
                                                            on_device);
        }
        virtual idx_t set_elements_in_slice(const float* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) override {
            return _set_elements_in_slice<float, YkVecVar>(buffer_ptr, buffer_size,
                                                           first_indices, last_indices,
                                                           on_device);
        }

        // Write to *this from val.
        virtual idx_t set_elements_in_slice_same(double val,
                                                 const Indices& first_indices,
                                                 const Indices& last_indices,
                                                 bool strict_indices,
                                                 bool on_device) override {
            return _set_elements_in_slice_same<double, YkVecVar>(val,
                                                                 first_indices, last_indices,
                                                                 strict_indices, on_device);
        }
        
        // Reduce elements.
        virtual yk_var::yk_reduction_result_ptr
        reduce_elements_in_slice(int reduction_mask,
                                 const Indices& first_indices,
                                 const Indices& last_indices,
                                 bool strict_indices,
                                 bool on_device) const override {
            return _reduce_elements_in_slice<double, YkVecVar>(reduction_mask,
                                                               first_indices, last_indices,
                                                               strict_indices, on_device);
        }
        
    };                          // YkVecVar.

    // Implementation of yk_var interface.  Class contains no real data,
    // just a pointer to the underlying data and meta-data. This allows var
    // data to be shared and moved without changing pointers.
    class YkVarImpl : public virtual yk_var {
    protected:
        VarBasePtr _gbp;

    public:
        YkVarImpl() { }
        YkVarImpl(const VarBasePtr& gp) : _gbp(gp) { }
        virtual ~YkVarImpl() { }

        inline void set_gbp(const VarBasePtr& gp) {
            _gbp = gp;
        }
        inline YkVarBase& gb() {
            assert(_gbp.get());
            return *(_gbp.get());
        }
        inline const YkVarBase& gb() const {
            assert(_gbp.get());
            return *(_gbp.get());
        }
        inline YkVarBase* gbp() {
            return _gbp.get();
        }
        inline const YkVarBase* gbp() const {
            return _gbp.get();
        }
        inline YkVarBaseCore* corep() {
            return gb().get_corep();
        }
        inline const YkVarBaseCore* corep() const {
            return gb().get_corep();
        }

        // Pass-thru methods to base.
        void set_all_elements_in_seq(double seed) {
            gb().set_all_elements_in_seq(seed);
        }
        idx_t set_vecs_in_slice(const void* buffer_ptr,
                                const Indices& first_indices,
                                const Indices& last_indices,
                                bool on_device) {
            return gb().set_vecs_in_slice(buffer_ptr,
                                          first_indices, last_indices, on_device);
        }
        idx_t get_vecs_in_slice(void* buffer_ptr,
                                const Indices& first_indices,
                                const Indices& last_indices,
                                bool on_device) const {
            return gb().get_vecs_in_slice(buffer_ptr,
                                          first_indices, last_indices, on_device);
        }
        void resize() {
            gb().resize();
        }
        void sync_core() {
            gb().sync_core();
        }

        // APIs.
        // See yask_kernel_api.hpp.
        virtual const std::string& get_name() const {
            return gb().get_name();
        }
        virtual int get_num_dims() const {
            return gb().get_num_dims();
        }
        virtual int get_num_domain_dims() const {
            return gb().get_num_domain_dims();
        }
        virtual bool is_dim_used(const std::string& dim) const {
            return gb().is_dim_used(dim);
        }
        virtual const std::string& get_dim_name(int n) const {
            assert(n >= 0);
            assert(n < get_num_dims());
            return gb().get_dim_name(n);
        }
        virtual VarDimNames get_dim_names() const {
            string_vec dims(get_num_dims());
            for (int i = 0; i < get_num_dims(); i++)
                dims.at(i) = get_dim_name(i);
            return dims;
        }
        virtual bool is_fixed_size() const {
            return gb()._fixed_size;
        }
        virtual bool is_dynamic_step_alloc() const {
            return gb()._is_dynamic_step_alloc;
        }
        virtual bool is_dynamic_misc_alloc() const {
            return gb()._is_dynamic_misc_alloc;
        }
        virtual int get_numa_preferred() const {
            return gb().get_numa_pref();
        }
        virtual bool set_numa_preferred(int numa_node) {
            return gb().set_numa_pref(numa_node);
        }

        virtual idx_t get_first_valid_step_index() const {
            if (!gb()._has_step_dim)
                THROW_YASK_EXCEPTION("'get_first_valid_step_index()' called on var '" +
                                     get_name() + "' that does not use the step dimension");
            return corep()->_local_offsets[+step_posn];
        }
        virtual idx_t get_last_valid_step_index() const {
            if (!gb()._has_step_dim)
                THROW_YASK_EXCEPTION("'get_last_valid_step_index()' called on var '" +
                                     get_name() + "' that does not use the step dimension");
            return corep()->_local_offsets[+step_posn] +
                corep()->_domains[+step_posn] - 1;
        }
        virtual int
        get_halo_exchange_l1_norm() const {
            return gb()._l1_dist;
        }
        virtual void
        set_halo_exchange_l1_norm(int norm) {
            gb()._l1_dist = norm;
        }

        // See yk_var_apis.cpp for corresponding definition macros.
        #define GET_VAR_API(api_name)                                   \
            virtual idx_t api_name(const std::string& dim) const;       \
            virtual idx_t api_name(int posn) const;
        #define GET_VAR_API2(api_name)                  \
            GET_VAR_API(api_name)                       \
            virtual idx_t_vec api_name ## _vec() const;
        #define SET_VAR_API(api_name)                                   \
            virtual void api_name(const std::string& dim, idx_t n);     \
            virtual void api_name(int posn, idx_t n);
        #define SET_VAR_API2(api_name)                          \
            SET_VAR_API(api_name)                               \
            virtual void api_name ## _vec(idx_t_vec);           \
            virtual void api_name ## _vec(idx_t_init_list);

        // Settings that should never be exposed as APIs because
        // they can break the usage model.
        // They are not protected because they are used from outside
        // this class hierarchy.
        GET_VAR_API(_get_left_wf_ext)
        GET_VAR_API(_get_local_offset)
        GET_VAR_API(_get_rank_offset)
        GET_VAR_API(_get_right_wf_ext)
        GET_VAR_API(_get_soln_vec_len)
        GET_VAR_API(_get_var_vec_len)

        SET_VAR_API(_set_alloc_size)
        SET_VAR_API(_set_domain_size)
        SET_VAR_API(_set_left_pad_size)
        SET_VAR_API(_set_left_wf_ext)
        SET_VAR_API(_set_local_offset)
        SET_VAR_API(_set_rank_offset)
        SET_VAR_API(_set_right_pad_size)
        SET_VAR_API(_set_right_wf_ext)
        SET_VAR_API(update_left_min_pad_size)
        SET_VAR_API(update_right_min_pad_size)
        SET_VAR_API(update_min_pad_size)
        SET_VAR_API(update_left_extra_pad_size)
        SET_VAR_API(update_right_extra_pad_size)
        SET_VAR_API(update_extra_pad_size)

        // Exposed APIs.
        GET_VAR_API(get_first_misc_index)
        GET_VAR_API(get_last_misc_index)
        GET_VAR_API2(get_first_local_index)
        GET_VAR_API2(get_last_local_index)
        GET_VAR_API2(get_rank_domain_size)
        GET_VAR_API2(get_first_rank_domain_index)
        GET_VAR_API2(get_last_rank_domain_index)
        GET_VAR_API2(get_left_halo_size)
        GET_VAR_API2(get_right_halo_size)
        GET_VAR_API2(get_first_rank_halo_index)
        GET_VAR_API2(get_last_rank_halo_index)
        GET_VAR_API2(get_left_extra_pad_size)
        GET_VAR_API2(get_right_extra_pad_size)
        GET_VAR_API2(get_left_pad_size)
        GET_VAR_API2(get_right_pad_size)
        GET_VAR_API2(get_alloc_size)

        SET_VAR_API(set_first_misc_index)
        SET_VAR_API(set_left_halo_size)
        SET_VAR_API(set_right_halo_size)
        SET_VAR_API(set_halo_size)
        SET_VAR_API(set_left_min_pad_size)
        SET_VAR_API(set_right_min_pad_size)
        SET_VAR_API(set_min_pad_size)
        SET_VAR_API(set_left_extra_pad_size)
        SET_VAR_API(set_right_extra_pad_size)
        SET_VAR_API(set_extra_pad_size)
        SET_VAR_API(set_alloc_size)

        #undef GET_VAR_API
        #undef GET_VAR_API2
        #undef SET_VAR_API
        #undef SET_VAR_API2

        virtual std::string format_indices(const Indices& indices) const {
            gb().check_indices(indices, "format_indices", false, false);
            std::string str = get_name() + "(" + gb().make_index_string(indices) + ")";
            return str;
        }
        virtual std::string format_indices(const VarIndices& indices) const {
            const Indices indices2(indices);
            return format_indices(indices2);
        }
        virtual std::string format_indices(const idx_t_init_list& indices) const {
            const Indices indices2(indices);
            return format_indices(indices2);
        }

        virtual bool are_indices_local(const Indices& indices) const;
        virtual bool are_indices_local(const VarIndices& indices) const {
            const Indices indices2(indices);
            return are_indices_local(indices2);
        }
        virtual bool are_indices_local(const idx_t_init_list& indices) const {
            const Indices indices2(indices);
            return are_indices_local(indices2);
        }

        virtual double get_element(const Indices& indices) const;
        virtual double get_element(const VarIndices& indices) const {
            const Indices indices2(indices);
            return get_element(indices2);
        }
        virtual double get_element(const idx_t_init_list& indices) const {
            const Indices indices2(indices);
            return get_element(indices2);
        }
        virtual idx_t get_elements_in_slice(double* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) const {
            return gb().get_elements_in_slice(buffer_ptr, buffer_size,
                                              first_indices, last_indices, on_device);
        }
        virtual idx_t get_elements_in_slice(float* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) const {
            return gb().get_elements_in_slice(buffer_ptr, buffer_size,
                                              first_indices, last_indices, on_device);
        }
        virtual idx_t get_elements_in_slice(void* buffer_ptr,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) const {
            return gb().get_elements_in_slice_void(buffer_ptr,
                                                   first_indices, last_indices, on_device);
        }
        virtual idx_t get_elements_in_slice(double* buffer_ptr,
                                            size_t buffer_size,
                                            const VarIndices& first_indices,
                                            const VarIndices& last_indices) const {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return get_elements_in_slice(buffer_ptr, buffer_size,
                                         first, last, false);
        }
        virtual idx_t get_elements_in_slice(float* buffer_ptr,
                                            size_t buffer_size,
                                            const VarIndices& first_indices,
                                            const VarIndices& last_indices) const {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return get_elements_in_slice(buffer_ptr, buffer_size,
                                         first, last, false);
        }
        virtual idx_t get_elements_in_slice(void* buffer_ptr,
                                            const VarIndices& first_indices,
                                            const VarIndices& last_indices) const {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return get_elements_in_slice(buffer_ptr,
                                         first, last, false);
        }
        virtual idx_t set_element(double val,
                                  const Indices& indices,
                                  bool strict_indices);
        virtual idx_t set_element(double val,
                                  const VarIndices& indices,
                                  bool strict_indices) {
            const Indices indices2(indices);
            return set_element(val, indices2, strict_indices);
        }
        virtual idx_t set_element(double val,
                                  const idx_t_init_list& indices,
                                  bool strict_indices) {
            const Indices indices2(indices);
            return set_element(val, indices2, strict_indices);
        }
        virtual idx_t add_to_element(double val,
                                     const Indices& indices,
                                     bool strict_indices);
        virtual idx_t add_to_element(double val,
                                     const VarIndices& indices,
                                     bool strict_indices) {
            const Indices indices2(indices);
            return add_to_element(val, indices2, strict_indices);
        }
        virtual idx_t add_to_element(double val,
                                     const idx_t_init_list& indices,
                                     bool strict_indices) {
            const Indices indices2(indices);
            return add_to_element(val, indices2, strict_indices);
        }

        virtual void set_all_elements_same(double val) {
            gb().set_all_elements_same(val);
        }
        virtual idx_t set_elements_in_slice_same(double val,
                                                 const Indices& first_indices,
                                                 const Indices& last_indices,
                                                 bool strict_indices,
                                                 bool on_device) {
            return gb().set_elements_in_slice_same(val,
                                                   first_indices, last_indices,
                                                   strict_indices, on_device);
        }
        virtual yk_var::yk_reduction_result_ptr
        reduce_elements_in_slice(int reduction_mask,
                                 const Indices& first_indices,
                                 const Indices& last_indices,
                                 bool strict_indices,
                                 bool on_device) {
            return gb().reduce_elements_in_slice(reduction_mask,
                                                 first_indices, last_indices,
                                                 strict_indices, on_device);
        }
        virtual yk_var::yk_reduction_result_ptr
        reduce_elements_in_slice(int reduction_mask,
                                 const idx_t_vec& first_indices,
                                 const idx_t_vec& last_indices,
                                 bool strict_indices) {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return reduce_elements_in_slice(reduction_mask, first, last, strict_indices, false);
        }
        virtual idx_t set_elements_in_slice_same(double val,
                                                 const VarIndices& first_indices,
                                                 const VarIndices& last_indices,
                                                 bool strict_indices) {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return set_elements_in_slice_same(val, first, last, strict_indices, false);
        }

        virtual idx_t set_elements_in_slice(const double* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) {
            return gb().set_elements_in_slice(buffer_ptr, buffer_size,
                                              first_indices, last_indices, on_device);
        }
        virtual idx_t set_elements_in_slice(const float* buffer_ptr,
                                            size_t buffer_size,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) {
            return gb().set_elements_in_slice(buffer_ptr, buffer_size,
                                              first_indices, last_indices, on_device);
        }
        virtual idx_t set_elements_in_slice(const void* buffer_ptr,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) {
            return gb().set_elements_in_slice_void(buffer_ptr,
                                                   first_indices, last_indices, on_device);
        }
        virtual idx_t set_elements_in_slice(const double* buffer_ptr,
                                            size_t buffer_size,
                                            const VarIndices& first_indices,
                                            const VarIndices& last_indices) {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return set_elements_in_slice(buffer_ptr, buffer_size, first, last, false);
        }
        virtual idx_t set_elements_in_slice(const float* buffer_ptr,
                                            size_t buffer_size,
                                            const VarIndices& first_indices,
                                            const VarIndices& last_indices) {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return set_elements_in_slice(buffer_ptr, buffer_size, first, last, false);
        }
        virtual idx_t set_elements_in_slice(const void* buffer_ptr,
                                            const VarIndices& first_indices,
                                            const VarIndices& last_indices) {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return set_elements_in_slice(buffer_ptr, first, last, false);
        }

        virtual void alloc_storage() {
            STATE_VARS(gbp());
            gb().default_alloc();
            DEBUG_MSG(gb().make_info_string());
        }
        virtual void release_storage() {
            STATE_VARS(gbp());
            TRACE_MSG("release_storage(): " << gb().make_info_string());
            gb().release_storage();
            TRACE_MSG("after release_storage(): " << gb().make_info_string());
        }
        virtual bool is_storage_allocated() const {
            return gb().get_storage() != 0;
        }
        virtual idx_t get_num_storage_bytes() const {
            return idx_t(gb().get_num_bytes());
        }
        virtual idx_t get_num_storage_elements() const {
            return corep()->_allocs.product();
        }
        virtual bool is_storage_layout_identical(const YkVarImpl* other,
                                                 bool check_sizes) const;
        virtual bool is_storage_layout_identical(const yk_var_ptr other) const {
            auto op = std::dynamic_pointer_cast<YkVarImpl>(other);
            assert(op);
            return is_storage_layout_identical(op.get(), true);
        }
        virtual void fuse_vars(yk_var_ptr other);
        virtual void* get_raw_storage_buffer() {
            return gb().get_storage();
        }
        virtual void set_storage(std::shared_ptr<char> base, size_t offset) {
            gb().set_storage(base, offset);
        }
    };

}                               // namespace.
