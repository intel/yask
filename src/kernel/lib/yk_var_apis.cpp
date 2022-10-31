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

// Implement methods for yk_var APIs.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // APIs to get info from vars: one with name of dim with a lot
    // of checking, one with index of dim with no checking.
    #define GET_VAR_API(api_name, expr, step_ok, domain_ok, misc_ok, prep_req) \
        idx_t YkVarImpl::api_name(const string& dim) const {            \
            STATE_VARS(gbp());                                          \
            dims->check_dim_type(dim, #api_name, step_ok, domain_ok, misc_ok); \
            int posn = gb().get_dim_posn(dim, true, #api_name);         \
            if (prep_req && corep()->_rank_offsets[posn] < 0)           \
                THROW_YASK_EXCEPTION("Error: '" #api_name "()' called on var '" + \
                                     get_name() + "' before calling 'prepare_solution()'"); \
            auto cp = corep();                                          \
            auto rtn = expr;                                            \
            return rtn;                                                 \
        }                                                               \
        idx_t YkVarImpl::api_name(int posn) const {                     \
            STATE_VARS(gbp());                                          \
            auto cp = corep();                                          \
            auto rtn = expr;                                            \
            return rtn;                                                 \
        }

    // Add vector version that retuns only allowed results.
    #define GET_VAR_API2(api_name, expr, step_ok, domain_ok, misc_ok, prep_req) \
        GET_VAR_API(api_name, expr, step_ok, domain_ok, misc_ok, prep_req) \
        idx_t_vec YkVarImpl::api_name ## _vec() const {                 \
            STATE_VARS(gbp());                                          \
            if (prep_req && corep()->_rank_offsets[0] < 0)              \
                THROW_YASK_EXCEPTION("Error: '" #api_name "_vec()' called on var '" + \
                                     get_name() + "' before calling 'prepare_solution()'"); \
            auto cp = corep();                                          \
            auto nvdims = get_num_dims();                               \
            auto nadims = 0;                                            \
            if (step_ok) nadims += cp->_num_step_dims;                  \
            if (domain_ok) nadims += cp->_num_domain_dims;              \
            if (misc_ok) nadims += cp->_num_misc_dims;                  \
            idx_t_vec res(nadims, 0);                                   \
            int i = 0;                                                  \
            for (int posn = 0; posn < nvdims; posn++) {                 \
                idx_t mbit = idx_t(1) << posn;                          \
                if ((step_ok && (mbit & cp->_step_dim_mask) != 0) ||    \
                    (domain_ok && (mbit & cp->_domain_dim_mask) != 0) || \
                    (misc_ok && (mbit & cp->_misc_dim_mask) != 0)) {    \
                    res.at(i++) = expr;                                 \
                }                                                       \
            }                                                           \
            assert(i == nadims);                                        \
            return res;                                                 \
        }

    // APIs to set vars.
    #define SET_VAR_API(api_name, expr, step_ok, domain_ok, misc_ok, need_resize) \
        void YkVarImpl::api_name(const string& dim, idx_t n) {          \
            STATE_VARS(gbp());                                          \
            TRACE_MSG("var '" << get_name() << "'."                     \
                      #api_name "('" << dim << "', " << n << ")");      \
            dims->check_dim_type(dim, #api_name, step_ok, domain_ok, misc_ok); \
            int posn = gb().get_dim_posn(dim, true, #api_name);         \
            auto cp = corep();                                          \
            expr;                                                       \
            if (need_resize) resize();                                  \
            else sync_core();                                           \
        }                                                               \
        void YkVarImpl::api_name(int posn, idx_t n) {                   \
            STATE_VARS(gbp());                                          \
            TRACE_MSG("var '" << get_name() << "'."                     \
                      #api_name "('" << posn << "', " << n << ")");     \
            auto cp = corep();                                          \
            expr;                                                       \
            if (need_resize) resize();                                  \
            else sync_core();                                           \
        }

    // Add vector versions that take only domain-dim vals.
    #define SET_VAR_API2(api_name, expr, step_ok, domain_ok, misc_ok, need_resize) \
        SET_VAR_API(api_name, expr, step_ok, domain_ok, misc_ok, need_resize) \
        void YkVarImpl::api_name ## _vec(idx_t_vec vals) {              \
            STATE_VARS(gbp());                                          \
            TRACE_MSG("var '" << get_name() << "'."                     \
                      #api_name "_vec(...)");                           \
            auto cp = corep();                                          \
            auto nvdims = get_num_dims();                               \
            auto nadims = 0;                                            \
            if (step_ok) nadims += cp->_num_step_dims;                  \
            if (domain_ok) nadims += cp->_num_domain_dims;              \
            if (misc_ok) nadims += cp->_num_misc_dims;                  \
            if (vals.size() != nadims)                                  \
                THROW_YASK_EXCEPTION("Error: '" #api_name               \
                                     "_vec()' called on var '" +        \
                                     get_name() + "' without the proper number of values"); \
            int i = 0;                                                  \
            for (int posn = 0; posn < nvdims; posn++) {                 \
                idx_t mbit = idx_t(1) << posn;                          \
                if ((step_ok && (mbit & cp->_step_dim_mask) != 0) ||    \
                    (domain_ok && (mbit & cp->_domain_dim_mask) != 0) || \
                    (misc_ok && (mbit & cp->_misc_dim_mask) != 0)) {    \
                    auto n = vals.at(i++);                              \
                    expr;                                               \
                }                                                       \
            }                                                           \
            if (need_resize) resize();                                  \
            else sync_core();                                           \
        }                                                               \
        void YkVarImpl::api_name ## _vec(idx_t_init_list vals) {        \
            idx_t_vec vec(vals);                                            \
            api_name ## _vec(vec);                                      \
        }
    
    // Internal APIs.
    GET_VAR_API(_get_left_wf_ext,
                cp->_left_wf_exts[posn],
                true, true, true, false)
    GET_VAR_API(_get_right_wf_ext,
                cp->_right_wf_exts[posn],
                true, true, true, false)
    GET_VAR_API(_get_soln_vec_len,
                cp->_soln_vec_lens[posn],
                true, true, true, true)
    GET_VAR_API(_get_var_vec_len,
                cp->_var_vec_lens[posn],
                true, true, true, true)
    GET_VAR_API(_get_rank_offset,
                cp->_rank_offsets[posn],
                true, true, true, true)
    GET_VAR_API(_get_local_offset,
                cp->_local_offsets[posn],
                true, true, true, false)

    // Exposed APIs.
    GET_VAR_API(get_first_misc_index,
                cp->_local_offsets[posn],
                false, false, true, false)
    GET_VAR_API(get_last_misc_index,
                cp->_local_offsets[posn] + cp->_domains[posn] - 1,
                false, false, true, false)

    GET_VAR_API2(get_alloc_size,
                cp->_allocs[posn],
                true, true, true, false)
    GET_VAR_API2(get_first_local_index,
                cp->get_first_local_index(posn),
                true, true, true, true)
    GET_VAR_API2(get_last_local_index,
                cp->get_last_local_index(posn),
                true, true, true, true)

    GET_VAR_API2(get_left_pad_size,
                cp->_actl_left_pads[posn],
                false, true, false, false)
    GET_VAR_API2(get_right_pad_size,
                cp->_actl_right_pads[posn],
                false, true, false, false)
    GET_VAR_API2(get_left_halo_size,
                cp->_left_halos[posn],
                false, true, false, false)
    GET_VAR_API2(get_right_halo_size,
                cp->_right_halos[posn],
                false, true, false, false)
    GET_VAR_API2(get_left_extra_pad_size,
                cp->_actl_left_pads[posn] - cp->_left_halos[posn],
                false, true, false, false)
    GET_VAR_API2(get_right_extra_pad_size,
                cp->_actl_right_pads[posn] - cp->_right_halos[posn],
                false, true, false, false)

    GET_VAR_API2(get_rank_domain_size,
                cp->_domains[posn],
                false, true, false, !gb()._fixed_size)
    GET_VAR_API2(get_first_rank_domain_index,
                cp->_rank_offsets[posn],
                false, true, false, true)
    GET_VAR_API2(get_last_rank_domain_index,
                cp->_rank_offsets[posn] + cp->_domains[posn] - 1,
                false, true, false, true)
    GET_VAR_API2(get_first_rank_halo_index,
                cp->_rank_offsets[posn] - cp->_left_halos[posn],
                false, true, false, true)
    GET_VAR_API2(get_last_rank_halo_index,
                cp->_rank_offsets[posn] + cp->_domains[posn] + cp->_right_halos[posn] - 1,
                false, true, false, true)

    // These are the internal, unchecked access functions that allow
    // changes prohibited thru the APIs.
    SET_VAR_API(_set_rank_offset,
                cp->_rank_offsets[posn] = n,
                true, true, true, false)
    SET_VAR_API(_set_local_offset,
                cp->_local_offsets[posn] = n;
                assert(imod_flr(n, cp->_var_vec_lens[posn]) == 0);
                cp->_vec_local_offsets[posn] = n / cp->_var_vec_lens[posn],
                true, true, true, false)
    SET_VAR_API(_set_domain_size,
                cp->_domains[posn] = n,
                true, true, true, true)
    SET_VAR_API(_set_left_pad_size,
                cp->_actl_left_pads[posn] = n,
                true, true, true, true)
    SET_VAR_API(_set_right_pad_size,
                cp->_actl_right_pads[posn] = n,
                true, true, true, true)
    SET_VAR_API(_set_left_wf_ext,
                cp->_left_wf_exts[posn] = n,
                true, true, true, true)
    SET_VAR_API(_set_right_wf_ext,
                cp->_right_wf_exts[posn] = n,
                true, true, true, true)
    SET_VAR_API(_set_alloc_size,
                cp->_domains[posn] = n,
                true, true, true, true)
    SET_VAR_API(update_left_min_pad_size,
                cp->_req_left_pads[posn] = max(n, cp->_req_left_pads[posn]),
                false, true, false, true)
    SET_VAR_API(update_right_min_pad_size,
                cp->_req_right_pads[posn] = max(n, cp->_req_right_pads[posn]),
                false, true, false, true)
    SET_VAR_API(update_min_pad_size,
                cp->_req_left_pads[posn] = max(n, cp->_req_left_pads[posn]);
                cp->_req_right_pads[posn] = max(n, cp->_req_right_pads[posn]),
                false, true, false, true)
    SET_VAR_API(update_left_extra_pad_size,
                cp->_req_left_epads[posn] = max(n, cp->_req_left_epads[posn]),
                false, true, false, true)
    SET_VAR_API(update_right_extra_pad_size,
                cp->_req_right_epads[posn] = max (n, cp->_req_right_epads[posn]),
                false, true, false, true)
    SET_VAR_API(update_extra_pad_size,
                cp->_req_left_epads[posn] = max(n, cp->_req_left_epads[posn]);
                cp->_req_right_epads[posn] = max (n, cp->_req_right_epads[posn]),
                false, true, false, true)

    // These are the safer ones used in the APIs.
    SET_VAR_API(set_first_misc_index,
                cp->_local_offsets[posn] = n,
                false, false, gb()._is_user_var, false)
    SET_VAR_API(set_alloc_size,
                cp->_domains[posn] = n,
                gb()._is_dynamic_step_alloc, gb()._fixed_size, gb()._is_dynamic_misc_alloc, true)

    SET_VAR_API(set_left_halo_size,
                cp->_left_halos[posn] = n,
                false, true, false, true)
    SET_VAR_API(set_right_halo_size,
                cp->_right_halos[posn] = n,
                false, true, false, true)
    SET_VAR_API(set_halo_size,
                cp->_left_halos[posn] = cp->_right_halos[posn] = n,
                false, true, false, true)
    SET_VAR_API(set_left_min_pad_size,
                cp->_req_left_pads[posn] = n,
                false, true, false, true)
    SET_VAR_API(set_right_min_pad_size,
                cp->_req_right_pads[posn] = n,
                false, true, false, true)
    SET_VAR_API(set_min_pad_size,
                cp->_req_left_pads[posn] = n;
                cp->_req_right_pads[posn] = n,
                false, true, false, true)
    SET_VAR_API(set_left_extra_pad_size,
                cp->_req_left_epads[posn] = n,
                false, true, false, true)
    SET_VAR_API(set_right_extra_pad_size,
                cp->_req_right_epads[posn] = n,
                false, true, false, true)
    SET_VAR_API(set_extra_pad_size,
                cp->_req_left_epads[posn] = n;
                cp->_req_right_epads[posn] = n,
                false, true, false, true)
    #undef SET_VAR_API
    #undef SET_VAR_API2
    #undef GET_VAR_API
    #undef GET_VAR_API2

    bool YkVarImpl::is_storage_layout_identical(const YkVarImpl* op,
                                                bool check_sizes) const {

        // Same size?
        if (check_sizes && get_num_storage_bytes() != op->get_num_storage_bytes())
            return false;

        // Same num dims?
        if (get_num_dims() != op->get_num_dims())
            return false;
        for (int i = 0; i < get_num_dims(); i++) {
            auto dname = get_dim_name(i);

            // Same dims?
            if (dname != op->get_dim_name(i))
                return false;

            // Same folding?
            if (corep()->_var_vec_lens[i] != op->corep()->_var_vec_lens[i])
                return false;

            // Same dim sizes?
            if (check_sizes) {
                if (corep()->_domains[i] != op->corep()->_domains[i])
                    return false;
                if (corep()->_actl_left_pads[i] != op->corep()->_actl_left_pads[i])
                    return false;
                if (corep()->_actl_right_pads[i] != op->corep()->_actl_right_pads[i])
                    return false;
            }
        }
        return true;
    }

    void YkVarImpl::fuse_vars(yk_var_ptr src) {
        STATE_VARS(gbp());
        auto op = dynamic_pointer_cast<YkVarImpl>(src);
        assert(op);
        TRACE_MSG("fuse_vars(" << src.get() << "): this=" << gb().make_info_string() <<
                  "; source=" << op->gb().make_info_string());
        auto* sp = op.get();
        assert(!_gbp->is_scratch());

        // Check conditions for fusing into a non-user var.
        bool force_native = false;
        if (gb().is_user_var()) {
            force_native = true;
            if (!is_storage_layout_identical(sp, false))
                THROW_YASK_EXCEPTION("Error: fuse_vars(): attempt to replace meta-data"
                                     " of " + gb().make_info_string() +
                                     " used in solution with incompatible " +
                                     sp->gb().make_info_string());
        }

        // Copy shared-ptr to keep source YkVarBase active to end of method.
        VarBasePtr st_gbp = sp->_gbp;

        // Fuse meta-data.
        // After this, both YkVarImpls will point to same YkVarBase.
        _gbp = sp->_gbp;

        // Tag var as a non-user var if the original one was.
        if (force_native)
            _gbp->set_user_var(false);

        TRACE_MSG("after fuse_vars: this=" << gb().make_info_string() <<
                  "; source=" << op->gb().make_info_string());
    }

    // API get, set, etc.
    bool YkVarImpl::are_indices_local(const Indices& indices) const {
        if (!is_storage_allocated())
            return false;
        return gb().check_indices(indices, "are_indices_local", false, true, false);
    }
    double YkVarImpl::get_element(const Indices& indices) const {
        STATE_VARS(gbp());
        TRACE_MSG("get_element({" << gb().make_index_string(indices) << "}) on " <<
                  gb().make_info_string());
        if (!is_storage_allocated())
            THROW_YASK_EXCEPTION("Error: call to 'get_element' with no storage allocated for var '" +
                                 get_name() + "'");
        gb().check_indices(indices, "get_element", true, true, false);
        idx_t asi = gb().get_alloc_step_index(indices);
        
        gb().const_copy_data_from_device(); // TODO: make more efficient.
        real_t val = gb().read_elem(indices, asi, __LINE__);

        TRACE_MSG("get_element({" << gb().make_index_string(indices) << "}) on '" <<
                  get_name() + "' returns " << val);
        return double(val);
    }
    idx_t YkVarImpl::set_element(double val,
                                 const Indices& indices,
                                 bool strict_indices) {
        STATE_VARS(gbp());
        TRACE_MSG("set_element(" << val << ", {" <<
                  gb().make_index_string(indices) << "}, " <<
                  strict_indices << ") on " <<
                  gb().make_info_string());
        idx_t nup = 0;
        if (!get_raw_storage_buffer() && strict_indices)
            THROW_YASK_EXCEPTION("Error: call to 'set_element' with no storage allocated for var '" +
                                 get_name() + "'");
        if (get_raw_storage_buffer() &&

            // Don't check step index because this is a write-only API
            // that updates the step index.
            gb().check_indices(indices, "set_element", strict_indices, false, false)) {
            idx_t asi = gb().get_alloc_step_index(indices);

            gb().copy_data_from_device(); // TODO: make more efficient.
            gb().write_elem(real_t(val), indices, asi, __LINE__);
            nup++;

            // Set appropriate dirty flags.
            gb()._coh.mod_host();
            gb().set_dirty_using_alloc_index(YkVarBase::self, true, asi);
        }
        TRACE_MSG("set_element(" << val << ", {" <<
                  gb().make_index_string(indices) << "}, " <<
                  strict_indices << ") on '" <<
                  get_name() + "' returns " << nup);
        return nup;
    }
    idx_t YkVarImpl::add_to_element(double val,
                                    const Indices& indices,
                                    bool strict_indices) {
        STATE_VARS(gbp());
        TRACE_MSG("add_to_element(" << val << ", {" <<
                  gb().make_index_string(indices) <<  "}, " <<
                  strict_indices << ") on " <<
                  gb().make_info_string());
        idx_t nup = 0;
        if (!get_raw_storage_buffer() && strict_indices)
            THROW_YASK_EXCEPTION("Error: call to 'add_to_element' with no storage allocated for var '" +
                                 get_name() + "'");
        if (get_raw_storage_buffer() &&

            // Check step index because this API must read before writing.
            gb().check_indices(indices, "add_to_element", strict_indices, true, false)) {
            idx_t asi = gb().get_alloc_step_index(indices);

            #ifdef USE_OFFLOAD_NO_USM
            if (gb()._coh.need_to_update_host()) {
                #pragma omp critical
                gb().copy_data_from_device(); // TODO: make more efficient.
             }
            #endif

            gb().add_to_elem(real_t(val), indices, asi, __LINE__);
            nup++;

            // Set appropriate dirty flags.
            gb()._coh.mod_host();
            gb().set_dirty_using_alloc_index(YkVarBase::self, true, asi);
        }
        TRACE_MSG("add_to_element(" << val << ", {" <<
                  gb().make_index_string(indices) <<  "}, " <<
                  strict_indices << ") on '" <<
                  get_name() + "' returns " << nup);
        return nup;
    }

    // Read into buffer from *this.
    idx_t YkVarBase::get_elements_in_slice(void* buffer_ptr,
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
            static void visit(YkVarBase* varp,
                              real_t* p, idx_t pofs,
                              const Indices& pt, idx_t ti) {

                // Read from var.
                real_t val = varp->read_elem(pt, ti, __LINE__);

                // Write to buffer at proper index.
                p[pofs] = val;
            }
        };

        if (on_device)
            const_copy_data_to_device();
        else
            const_copy_data_from_device();
        
        // Call the generic visit.
        auto n = const_cast<YkVarBase*>(this)->
            _visit_elements_in_slice<GetElem>(true, (void*)buffer_ptr,
                                              first_indices, last_indices, on_device);

        // Return number of writes.
        return n;
    }

    // Write to *this from buffer.
    idx_t YkVarBase::set_elements_in_slice(const void* buffer_ptr,
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
            static void visit(YkVarBase* varp,
                              real_t* p, idx_t pofs,
                              const Indices& pt, idx_t ti) {

                // Read from buffer.
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
        auto n = 
            _visit_elements_in_slice<SetElem>(true, (void*)buffer_ptr,
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
    idx_t YkVarBase::set_elements_in_slice_same(double val,
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
            static void visit(YkVarBase* varp,
                              real_t* p, idx_t pofs,
                              const Indices& pt, idx_t ti) {

                // Get const value, ignoring offset.
                real_t val = *p;

                // Write to var
                varp->write_elem(val, pt, ti, __LINE__);
            }
        };

        if (on_device)
            const_copy_data_to_device();
        else
            const_copy_data_from_device();
        
        // Set up pointer to val for visitor.
        // Requires casting if real_t is a float.
        real_t v = real_t(val);
        auto* buffer_ptr = &v;
        
        // Call the generic visit.
        auto n = 
            _visit_elements_in_slice<SetElem>(strict_indices, (void*)buffer_ptr,
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
    
} // namespace.

