/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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
                  |    |                      |   |  +------------+
       YkElemVarCore  YkVecVarCore     YkElemVar  YkVecVar        |
         |     ^           ^    |          |        |             |
         |     |           |    |          |        |             |
         |     |           +------ptr----- | -------+             |
         |     +--------------------ptr----+                      |
         |                      |                                 |
       has-a                  has-a                               |
         |                      |                                 |
         v                      v                                 |
    +> GenericVarCore<real_t>   GenericVarCore<real_vec_t> <-+    |
    |     _elems ptr              _elems ptr                 |    |
    |                                                        |    |
    |                                                        |    |
    |                    GenericVarBase <-------------ptr---------+
    ptr                    ^      ^  _base ptr               |
    |                      |      |                          ptr
    |  GenericVarTyped<real_t>  GenericVarTyped<real_vec_t>  |
    |    ^                        ^                          |
    |    |                        |                          |
    +- GenericVar<real_t>       GenericVar<real_vec_t> ------+

       "Core" types are non-virtual and can be trivially copied, e.g.,
       to an offload device; others are virtual and cannot.
    */
    
    ///// Yk*Var*Core types /////
    OMP_DECL_TARGET
    
    // Core data that is needed for computations using a var.
    // A trivially-copyable type for offloading.
    struct YkVarBaseCore {

        // The following indices have one value for each dim in the var.
        // All values are in units of reals, not underlying elements, if different.
        // See diagram above for '_rank_offsets' and '_local_offsets'.
        // Comments show settings for domain dims | non-domain dims.
        Indices _domains;   // size of "interior" of var | alloc size.
        Indices _req_left_epads, _req_right_epads; // requested extra space around halos | zero.
        Indices _req_left_pads, _req_right_pads; // requested extra space around domains | zero.
        Indices _actl_left_pads, _actl_right_pads; // actual extra space around domains | zero.
        Indices _left_halos, _right_halos; // space within pads for halo exchange | zero.
        Indices _left_wf_exts, _right_wf_exts; // additional halos for wave-fronts | zero.
        Indices _rank_offsets;   // offsets of this var domain in overall problem | zero.
        Indices _local_offsets; // offsets of this var domain in this rank | first index for step or misc.
        Indices _allocs;    // actual var alloc in reals | same.

        // Each entry in _soln_vec_lens is same as dims->_fold_pts.
        Indices _soln_vec_lens;  // num reals in each elem in soln fold | one.

        // Each entry in _var_vec_lens may be same as dims->_fold_pts or one, depending
        // on whether var is fully vectorized.
        Indices _var_vec_lens;  // num reals in each elem in this var | one.

        // Sizes in vectors for sizes that are always vec lens (to avoid division).
        Indices _vec_left_pads; // _actl_left_pads / _var_vec_lens.
        Indices _vec_allocs; // _allocs / _var_vec_lens.
        Indices _vec_local_offsets; // _local_offsets / _var_vec_lens.

        // The following masks have one bit for each dim in the var.
        idx_t _step_dim_mask;
        idx_t _domain_dim_mask;
        idx_t _misc_dim_mask;

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
        _data_t _data;
        static_assert(std::is_trivially_copyable<_data_t>::value,
                      "Needed for OpenMP offload");

        // Ctor.
        YkElemVarCore(int ndims) :
            YkVarBaseCore(ndims) { }

        // Get a pointer to given element.
        // 'alloc_step_idx' must be within allocation bounds and consistent
        // with 'idxs[step_posn]'.
        const real_t* get_elem_ptr(const Indices& idxs,
                                   idx_t alloc_step_idx,
                                   bool check_bounds=true) const {
            constexpr auto n = LayoutFn::get_num_sizes();
            Indices adj_idxs(n);

            // Special handling for step index.
            auto sp = +step_posn;
            if (_use_step_idx) {
                host_assert(alloc_step_idx == _wrap_step(idxs[sp]));
                adj_idxs[sp] = alloc_step_idx;
            }

            // All other indices.
            _UNROLL for (int i = 0; i < n; i++) {
                if (!(_use_step_idx && i == sp)) {

                    // Adjust for offsets and padding.
                    // This gives a positive 0-based local element index.
                    idx_t ai = idxs[i] + _actl_left_pads[i] -
                        (_rank_offsets[i] + _local_offsets[i]);
                    host_assert(ai >= 0);
                    adj_idxs[i] = uidx_t(ai);
                }
            }

            // Get pointer via layout in _data.
            return _data.get_ptr(adj_idxs, check_bounds);
        }

        // Non-const version.
        // Implemented via casting.
        ALWAYS_INLINE
        real_t* get_elem_ptr(const Indices& idxs,
                             idx_t alloc_step_idx,
                             bool check_bounds=true) {
            const real_t* p =
                const_cast<const YkElemVarCore*>(this)->
                get_elem_ptr(idxs, alloc_step_idx, check_bounds);
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

    }; // YkElemVarCore.

    // Core data for YASK var of real vectors.
    template <typename LayoutFn, bool _use_step_idx, idx_t... _templ_vec_lens>
    struct YkVecVarCore final : public YkVarBaseCore {

        // Positions of var dims in vector fold dims.
        Indices _vec_fold_posns;

        // Storage core is owned here by composition.
        typedef GenericVarCore<real_vec_t, LayoutFn> _data_t;
        _data_t _data;
        static_assert(std::is_trivially_copyable<_data_t>::value,
                      "Needed for OpenMP offload");

        // Ctor.
        YkVecVarCore(int ndims) :
            YkVarBaseCore(ndims),
            _vec_fold_posns(idx_t(0), ndims) { }
         
        // Get a pointer to given element.
        const real_t* get_elem_ptr(const Indices& idxs,
                                   idx_t alloc_step_idx,
                                   bool check_bounds=true) const {

            // Use template vec lengths instead of run-time values for
            // efficiency.
            constexpr int nvls = sizeof...(_templ_vec_lens);
            constexpr uidx_t vls[nvls] { _templ_vec_lens... };
            Indices vec_idxs(nvls), elem_ofs(nvls);
            #ifdef DEBUG_LAYOUT
            constexpr auto ns = LayoutFn::get_num_sizes();
            host_assert(ns == nvls);
            #endif

            // Special handling for step index.
            auto sp = +step_posn;
            if (_use_step_idx) {
                host_assert(alloc_step_idx == _wrap_step(idxs[sp]));
                vec_idxs[sp] = alloc_step_idx;
                elem_ofs[sp] = 0;
            }

            // Try to force compiler to use shifts instead of DIV and MOD
            // when the vec-lengths are 2^n.
            // All other indices.
            _UNROLL _NO_VECTOR
                for (int i = 0; i < nvls; i++) {
                    if (!(_use_step_idx && i == sp)) {

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
                }

            // Get only the vectorized fold offsets, i.e., those
            // with vec-lengths > 1.
            // And, they need to be in the original folding order,
            // which might be different than the var-dim order.
            Indices fold_ofs(NUM_VEC_FOLD_DIMS);
            _UNROLL for (int i = 0; i < NUM_VEC_FOLD_DIMS; i++) {
                int j = _vec_fold_posns[i];
                fold_ofs[i] = elem_ofs[j];
            }

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

        // Get a pointer to given vector.
        // Indices must be normalized and rank-relative.
        // It's important that this function be efficient, since
        // it's indiectly used from the stencil kernel.
        ALWAYS_INLINE
        const real_vec_t* get_vec_ptr_norm(const Indices& vec_idxs,
                                           idx_t alloc_step_idx,
                                           bool check_bounds=true) const {

            constexpr int nvls = sizeof...(_templ_vec_lens);
            Indices adj_idxs(nvls);

            // Special handling for step index.
            auto sp = +step_posn;
            if (_use_step_idx) {
                host_assert(alloc_step_idx == _wrap_step(vec_idxs[sp]));
                adj_idxs[sp] = alloc_step_idx;
            }

            // Domain indices.
            _UNROLL for (int i = 0; i < nvls; i++) {
                if (!(_use_step_idx && i == sp)) {

                    // Adjust for padding.
                    // Since the indices are rank-relative, subtract only
                    // the local offsets. (Compare to get_elem_ptr().)
                    // This gives a 0-based local *vector* index.
                    adj_idxs[i] = vec_idxs[i] + _vec_left_pads[i] - _vec_local_offsets[i];
                }
            }

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
            return *vp;
        }

        // Write one vector.
        // Indices must be normalized and rank-relative.
        // 'alloc_step_idx' is pre-calculated or 0 if not used.
        ALWAYS_INLINE
        void write_vec_norm(real_vec_t val,
                            const Indices& vec_idxs,
                            idx_t alloc_step_idx) {
            real_vec_t* vp = get_vec_ptr_norm(vec_idxs, alloc_step_idx);
            *vp = val;
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
    OMP_END_DECL_TARGET

    ///// Yk*Var* types /////

    // Base class implementing all yk_var functionality. Used for
    // vars that contain either individual elements or vectors.
    // This class is pure virtual.
    class YkVarBase :
        public KernelStateBase {
        friend class YkVarImpl;

    protected:

        // Ptr to the core data.
        YkVarBaseCore* _corep = 0;
        
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

        // Whether this is a scratch var.
        bool _is_scratch = false;

        // Whether this was created via an API.
        bool _is_user_var = false;

        // Data that needs to be copied to neighbor's halos if using MPI.
        // If this var has the step dim, there is one bit per alloc'd step.
        // Otherwise, only bit 0 is used.
        std::vector<bool> _dirty_steps;

        // Convenience function to format indices like
        // "x=5, y=3".
        virtual std::string make_index_string(const Indices& idxs,
                                              std::string separator=", ",
                                              std::string infix="=",
                                              std::string prefix="",
                                              std::string suffix="") const;

        // Determine required padding from halos.
        // Does not include user-specified min padding or
        // final rounding for left pad.
        virtual Indices get_reqd_pad(const Indices& halos, const Indices& wf_exts) const;

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
                           bool normalize,      // div by vec lens.
                           Indices* fixed_indices = NULL) const;

        // Resize or fail if already allocated.
        void resize();

        // Set dirty flags in range.
        void set_dirty_in_slice(const Indices& first_indices,
                                const Indices& last_indices);

        // Find sizes needed for slicing.
        inline Indices get_slice_range(const Indices& first_indices,
                                       const Indices& last_indices) const {
            return last_indices.add_const(1).sub_elements(first_indices);
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
                  const VarDimNames& dim_names) :
            KernelStateBase(stateb), _corep(corep) {
            STATE_VARS(&stateb);

            // Set masks in core.
            _corep->_step_dim_mask = 0;
            _corep->_domain_dim_mask = 0;
            _corep->_misc_dim_mask = 0;
            for (size_t i = 0; i < dim_names.size(); i++) {
                idx_t mbit = 1LL << i;
                auto& dname = dim_names[i];
                if (dname == step_dim)
                    _corep->_step_dim_mask |= mbit;
                else if (domain_dims.lookup(dname))
                    _corep->_domain_dim_mask |= mbit;
                else
                    _corep->_misc_dim_mask |= mbit;
            }
        }

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
            return _corep->_domains._get_num_dims();
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
        virtual bool is_dirty(idx_t step_idx) const;
        virtual void set_dirty(bool dirty, idx_t step_idx);
        virtual void set_dirty_all(bool dirty);
        inline void set_dirty_using_alloc_index(bool dirty, idx_t alloc_idx) {
            _dirty_steps[alloc_idx] = dirty;
        }

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

        // Does this var cover the N-D domain?
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
        void copy_data_to_device() {
            STATE_VARS(this);
            void* vp = get_storage();
            char* cp = static_cast<char*>(vp);
            auto nb = get_num_bytes();
            offload_copy_to_device(cp, nb);
        }
        void copy_data_from_device() {
            STATE_VARS(this);
            void* vp = get_storage();
            char* cp = static_cast<char*>(vp);
            auto nb = get_num_bytes();
            offload_copy_from_device(cp, nb);
        }

        // Set elements.
        virtual void set_all_elements_in_seq(double seed) =0;
        virtual void set_all_elements_same(double seed) =0;

        // Set/get_elements_in_slice().
        idx_t set_elements_in_slice_same(double val,
                                         const Indices& first_indices,
                                         const Indices& last_indices,
                                         bool strict_indices,
                                         bool on_device);
        idx_t set_elements_in_slice(const void* buffer_ptr,
                                    const Indices& first_indices,
                                    const Indices& last_indices,
                                    bool on_device);
        idx_t get_elements_in_slice(void* buffer_ptr,
                                    const Indices& first_indices,
                                    const Indices& last_indices,
                                    bool on_device) const;

        // Possibly vectorized version of set/get_elements_in_slice().
        virtual idx_t set_vecs_in_slice(const void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices,
                                        bool on_device) =0;
        virtual idx_t get_vecs_in_slice(void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices,
                                        bool on_device) const =0;

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
        //                    real_t* p,  // copy of 'buffer_ptr'.
        //                    idx_t pofs, // offset into buffer.
        //                    const Indices& pt, // point in 'this' var.
        //                    idx_t ti);  // precomputed step index.
        template<typename Visitor>
        idx_t _visit_elements_in_slice(bool strict_indices,
                                       void* buffer_ptr,
                                       const Indices& first_indices,
                                       const Indices& last_indices,
                                       bool on_device) {
            STATE_VARS(this);
            if (get_storage() == 0) {
                if (strict_indices)
                    THROW_YASK_EXCEPTION(std::string("Error: call to '") +
                                         Visitor::fname() +
                                         "' with no storage allocated for var '" +
                                         get_name() + "'");
                return 0;
            }

            TRACE_MSG(Visitor::fname() << ": " << make_info_string() << " [" <<
                      make_index_string(first_indices) << " ... " <<
                      make_index_string(last_indices) << "] with buffer at " <<
                      buffer_ptr << " on " << (on_device ? "OMP device" : "host"));
            check_indices(first_indices, Visitor::fname(), strict_indices, true, false);
            check_indices(last_indices, Visitor::fname(), strict_indices, true, false);

            // Find range.
            auto range = get_slice_range(first_indices, last_indices);
            auto ne = range.product();
            TRACE_MSG(Visitor::fname() << ": " << ne << " element(s) in shape " <<
                      make_index_string(range));
            if (ne <= 0)
                return 0;

            // Iterate through step index in outer loop.
            // This avoids calling _wrap_step(t) at every point.
            const auto sp = +step_posn;
            idx_t first_t = 0, last_t = 0;
            if (_has_step_dim) {
                first_t = first_indices[sp];
                last_t = last_indices[sp];
                range[sp] = 1; // Do one step per iter.
            }

            // Amount to advance pointer each step.
            idx_t tsz = range.product();
            idx_t tofs = 0;
            TRACE_MSG(Visitor::fname() << ": " << tsz << " element(s) in shape " <<
                      make_index_string(range) << " for each step");
            
            // Iterate through inner index in inner loop.
            // This enables more optimization.
            const auto ip = get_num_dims() - 1;
            idx_t ni = range[ip];
            range[ip] = 1; // Do whole range in each iter.
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
                         [&](const Indices& ofs, size_t idx) {
                             auto pt = start_indices.add_elements(ofs);

                             // Inner loop.
                             for (idx_t i = 0; i < ni; i++) {
                                 idx_t bofs = tofs + idx * ni + i;
                                 #if 0
                                 TRACE_MSG(Visitor::fname() << ": visting pt " <<
                                           make_index_string(pt) << " w/buf ofs " << bofs);
                                 #endif
                             
                                 // Call visitor.
                                 Visitor::visit(this, (real_t*)buffer_ptr, bofs, pt, ti);
                                 pt[ip]++;
                             }

                             return true;    // keep going.
                         });
                }
        
                // Visit points in slice on device.
                else {
                    THROW_YASK_EXCEPTION(std::string("Internal error: '") +
                                         Visitor::fname() + "' for var '" +
                                         get_name() + "' not implemented for offload device");
                }

                // Skip to next step in buffer.
                tofs += tsz;

            } // steps.
            TRACE_MSG(Visitor::fname() << " returns " << ne);
            return ne;
        }
        
    };
    typedef std::shared_ptr<YkVarBase> VarBasePtr;

    // YASK var of real elements.
    // Used for vars that do not contain folded vectors.
    // If '_use_step_idx', then index to step dim will wrap around.
    template <typename LayoutFn, bool _use_step_idx>
    class YkElemVar final : public YkVarBase {

    public:
        // Type for core data.
        typedef YkElemVarCore<LayoutFn, _use_step_idx> core_t;

    protected:

        // Core data.
        // This also contains the storage core: _core._data.
        core_t _core;
        static_assert(std::is_trivially_copyable<core_t>::value,
                      "Needed for OpenMP offload");

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
        void set_all_elements_same(double seed) override final {
            _data.set_elems_same(seed);
            set_dirty_all(true);
        }
        void set_all_elements_in_seq(double seed) override final {
            _data.set_elems_in_seq(seed);
            set_dirty_all(true);
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

        // Non-vectorized fall-back versions.
        virtual idx_t set_vecs_in_slice(const void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices,
                                        bool on_device) override {
            return set_elements_in_slice(buffer_ptr,
                                         first_indices, last_indices, on_device);
        }
        virtual idx_t get_vecs_in_slice(void* buffer_ptr,
                                        const Indices& first_indices,
                                        const Indices& last_indices,
                                        bool on_device) const override {
            return get_elements_in_slice(buffer_ptr,
                                         first_indices, last_indices, on_device);
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
        
    protected:

        // Core data.
        // This also contains the storage core.
        core_t _core;
        static_assert(std::is_trivially_copyable<core_t>::value,
                      "Needed for OpenMP offload");

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
            assert(dims->_vec_fold_pts._get_num_dims() == NUM_VEC_FOLD_DIMS);
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
        void set_all_elements_same(double seed) override final {
            real_vec_t seedv = seed; // bcast.
            _data.set_elems_same(seedv);
            set_dirty_all(true);
        }
        void set_all_elements_in_seq(double seed) override final {
            real_vec_t seedv;
            auto n = seedv.get_num_elems();

            // Init elements to values between seed and 2*seed.
            // For example if n==4, init to
            // seed * 1.0, seed * 1.25, seed * 1.5, seed * 1.75.
            for (int i = 0; i < n; i++)
                seedv[i] = seed * (1.0 + double(i) / n);
            _data.set_elems_in_seq(seedv);
            set_dirty_all(true);
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
            
            #ifdef USE_OFFLOAD
            auto devn = KernelEnv::_omp_devn;
            if (on_device) {

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
            auto nv = vec_range.product() * VLEN;
            TRACE_MSG("copy_vecs_in_slice: " << nv << " vec(s) in " <<
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

            // Iterate through inner index in inner loop.
            // This enables more optimization.
            const auto ip = get_num_dims() - 1;
            idx_t ni = vec_range[ip];
            vec_range[ip] = 1; // Do whole range in each iter.

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
                    auto nj = vec_range.product();
                    Indices ofs(vec_range);
                    ofs.set_vals_same(0);
                    Indices pt(ofs);

                    // TODO: move target parallel up to this level.
                    for (idx_t j = 0; j < nj; j++) {
                        pt = firstv.add_elements(ofs);

                        #if 0
                        TRACE_MSG("copy starting at " << pt.make_val_str() <<
                                " and buf ofs " << (tofs + j * ni));
                        #endif
                        _Pragma("omp target parallel for firstprivate(pt) device(devn)")
                            for (idx_t i = 0; i < ni; i++) {
                                idx_t bofs = tofs + j * ni + i;
                                pt[ip] = firstv[ip] + ofs[ip] + i;
                         
                                // Do the copy operation specified in visitor.
                                #if 0
                                TRACE_MSG(" copying at " << pt.make_val_str() <<
                                          " and buf ofs " << bofs << " on thread " <<
                                          omp_get_thread_num());
                                #endif
                                Visitor::do_copy(core_p,
                                                 ((real_vec_t*)buffer_ptr), bofs,
                                                 pt, ti);
                                //pt[ip]++;
                            }
                        
                        // Jump to next index.
                        vec_range.next_index(false, ofs);
                    }
                    #else
                    THROW_YASK_EXCEPTION("internal error: call to _copy_vecs_in_slice on device"
                                         " in non-offload build");
                    #endif
                }

                // Visit starting points in range on host in parallel.
                else {
                vec_range.visit_all_points_in_parallel
                    (false,
                     [&](const Indices& ofs, size_t idx) {
                         auto pt = firstv.add_elements(ofs);
                         
                         // Inner loop.
                         for (idx_t i = 0; i < ni; i++) {
                             idx_t bofs = tofs + idx * ni + i;
                             
                             // Do the copy operation specified in visitor.
                             Visitor::do_copy(core_p,
                                              ((real_vec_t*)buffer_ptr), bofs,
                                              pt, ti);
                             pt[ip]++;
                         }
                         return true;    // keep going.
                     });
                }

                // Skip to next step in buffer.
                tofs += tsz;
                
            } // time steps.

            assert(tofs * VLEN == nv);
            return nv;
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
                static void do_copy(core_t* cp,
                                    real_vec_t* p, idx_t pofs,
                                    const Indices& pt, idx_t ti) {
                    
                    // Read vec from buffer.
                    real_vec_t val = p[pofs];
                    
                    // Write to var.
                    cp->write_vec_norm(val, pt, ti);
                }
            };

            // Call the generic vec copier.
            auto nset = _copy_vecs_in_slice<SetVec>((void*)buffer_ptr,
                                                    first_indices, last_indices,
                                                    on_device);

            // Set appropriate dirty flag(s).
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
                static void do_copy(core_t* cp,
                                    real_vec_t* p, idx_t pofs,
                                    const Indices& pt, idx_t ti) {

                    // Read vec from var.
                    real_vec_t val = cp->read_vec_norm(pt, ti);

                    // Write to buffer at proper index.
                    p[pofs] = val;
                }
            };

            // Call the generic vec copier.
            auto n = const_cast<YkVecVar*>(this)->
                _copy_vecs_in_slice<GetVec>((void*)buffer_ptr,
                                            first_indices, last_indices, on_device);
            
            // Return number of writes.
            return n;
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
        virtual bool is_dim_used(const std::string& dim) const {
            return gb().is_dim_used(dim);
        }
        virtual const std::string& get_dim_name(int n) const {
            assert(n >= 0);
            assert(n < get_num_dims());
            return gb().get_dim_name(n);
        }
        virtual VarDimNames get_dim_names() const {
            std::vector<std::string> dims(get_num_dims());
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
                THROW_YASK_EXCEPTION("Error: 'get_first_valid_step_index()' called on var '" +
                                     get_name() + "' that does not use the step dimension");
            return corep()->_local_offsets[+step_posn];
        }
        virtual idx_t get_last_valid_step_index() const {
            if (!gb()._has_step_dim)
                THROW_YASK_EXCEPTION("Error: 'get_last_valid_step_index()' called on var '" +
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

        #define GET_VAR_API(api_name)                                   \
            virtual idx_t api_name(const std::string& dim) const;       \
            virtual idx_t api_name(int posn) const;
        #define SET_VAR_API(api_name)                                   \
            virtual void api_name(const std::string& dim, idx_t n);     \
            virtual void api_name(int posn, idx_t n);

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

        // Exposed APIs.
        GET_VAR_API(get_first_local_index)
        GET_VAR_API(get_last_local_index)
        GET_VAR_API(get_rank_domain_size)
        GET_VAR_API(get_first_rank_domain_index)
        GET_VAR_API(get_last_rank_domain_index)
        GET_VAR_API(get_left_halo_size)
        GET_VAR_API(get_right_halo_size)
        GET_VAR_API(get_first_rank_halo_index)
        GET_VAR_API(get_last_rank_halo_index)
        GET_VAR_API(get_left_extra_pad_size)
        GET_VAR_API(get_right_extra_pad_size)
        GET_VAR_API(get_left_pad_size)
        GET_VAR_API(get_right_pad_size)
        GET_VAR_API(get_alloc_size)
        GET_VAR_API(get_first_rank_alloc_index)
        GET_VAR_API(get_last_rank_alloc_index)
        GET_VAR_API(get_first_misc_index)
        GET_VAR_API(get_last_misc_index)

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
        SET_VAR_API(set_first_misc_index)

        #undef GET_VAR_API
        #undef SET_VAR_API

        virtual std::string format_indices(const Indices& indices) const {
            std::string str = get_name() + "(" + gb().make_index_string(indices) + ")";
            return str;
        }
        virtual std::string format_indices(const VarIndices& indices) const {
            const Indices indices2(indices);
            return format_indices(indices2);
        }
        virtual std::string format_indices(const std::initializer_list<idx_t>& indices) const {
            const Indices indices2(indices);
            return format_indices(indices2);
        }

        virtual bool are_indices_local(const Indices& indices) const;
        virtual bool are_indices_local(const VarIndices& indices) const {
            const Indices indices2(indices);
            return are_indices_local(indices2);
        }
        virtual bool are_indices_local(const std::initializer_list<idx_t>& indices) const {
            const Indices indices2(indices);
            return are_indices_local(indices2);
        }

        virtual double get_element(const Indices& indices) const;
        virtual double get_element(const VarIndices& indices) const {
            const Indices indices2(indices);
            return get_element(indices2);
        }
        virtual double get_element(const std::initializer_list<idx_t>& indices) const {
            const Indices indices2(indices);
            return get_element(indices2);
        }
        virtual idx_t get_elements_in_slice(void* buffer_ptr,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) const {
            return gb().get_elements_in_slice(buffer_ptr,
                                              first_indices, last_indices, on_device);
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
                                  bool strict_indices = false);
        virtual idx_t set_element(double val,
                                  const VarIndices& indices,
                                  bool strict_indices = false) {
            const Indices indices2(indices);
            return set_element(val, indices2, strict_indices);
        }
        virtual idx_t set_element(double val,
                                  const std::initializer_list<idx_t>& indices,
                                  bool strict_indices = false) {
            const Indices indices2(indices);
            return set_element(val, indices2, strict_indices);
        }
        virtual idx_t add_to_element(double val,
                                     const Indices& indices,
                                     bool strict_indices = false);
        virtual idx_t add_to_element(double val,
                                     const VarIndices& indices,
                                     bool strict_indices = false) {
            const Indices indices2(indices);
            return add_to_element(val, indices2, strict_indices);
        }
        virtual idx_t add_to_element(double val,
                                     const std::initializer_list<idx_t>& indices,
                                     bool strict_indices = false) {
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
        virtual idx_t set_elements_in_slice_same(double val,
                                                 const VarIndices& first_indices,
                                                 const VarIndices& last_indices,
                                                 bool strict_indices) {
            const Indices first(first_indices);
            const Indices last(last_indices);
            return set_elements_in_slice_same(val, first, last, strict_indices, false);
        }

        virtual idx_t set_elements_in_slice(const void* buffer_ptr,
                                            const Indices& first_indices,
                                            const Indices& last_indices,
                                            bool on_device) {
            return gb().set_elements_in_slice(buffer_ptr,
                                              first_indices, last_indices, on_device);
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
