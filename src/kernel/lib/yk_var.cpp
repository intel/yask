/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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

// Implement methods for RealVecVarBase.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Ctor.
    // Important: _data is NOT yet constructed.
    YkVarBaseCore::YkVarBaseCore(int ndims) {

        // Init indices.
        int n = ndims;
        _domains.set_from_const(0, n);
        _req_left_pads.set_from_const(0, n);
        _req_right_pads.set_from_const(0, n);
        _req_left_epads.set_from_const(0, n);
        _req_right_epads.set_from_const(0, n);
        _actl_left_pads.set_from_const(0, n);
        _actl_right_pads.set_from_const(0, n);
        _left_halos.set_from_const(0, n);
        _right_halos.set_from_const(0, n);
        _left_wf_exts.set_from_const(0, n);
        _right_wf_exts.set_from_const(0, n);
        _rank_offsets.set_from_const(0, n);
        _local_offsets.set_from_const(0, n);
        _allocs.set_from_const(0, n);
        _soln_vec_lens.set_from_const(1, n);
        _var_vec_lens.set_from_const(1, n);
        _vec_left_pads.set_from_const(0, n);
        _vec_allocs.set_from_const(0, n);
        _vec_local_offsets.set_from_const(0, n);
        _vec_strides.set_from_const(0, n);
    }

    // Convenience function to format indices like
    // "x=5, y=3".
    std::string YkVarBase::make_index_string(const Indices& idxs,
                                            std::string separator,
                                            std::string infix,
                                            std::string prefix,
                                            std::string suffix) const {
        auto& tmp = get_dim_tuple();
        return idxs.make_dim_val_str(tmp, separator, infix, prefix, suffix);
    }

    // Does this var cover the N-D domain?
    bool YkVarBase::is_domain_var() const {
        STATE_VARS(this);

        // Check problem dims.
        for (auto& d : domain_dims) {
            auto& dname = d._get_name();
            if (!is_dim_used(dname))
                return false;
        }
        return true;
    }

    // Halo-exchange flag accessors.
    bool YkVarBase::is_dirty(idx_t step_idx) const {
        if (_dirty_steps.size() == 0)
            const_cast<YkVarBase*>(this)->resize();
        if (_has_step_dim)
            step_idx = _wrap_step(step_idx);
        else
            step_idx = 0;
        return _dirty_steps[step_idx];
    }
    void YkVarBase::set_dirty(bool dirty, idx_t step_idx) {
        if (_dirty_steps.size() == 0)
            resize();
        if (_has_step_dim) {

            // Also update valid step.
            if (dirty)
                update_valid_step(step_idx);

            // Wrap index.
            step_idx = _wrap_step(step_idx);
        }
        else
            step_idx = 0;
        set_dirty_using_alloc_index(dirty, step_idx);
    }
    void YkVarBase::set_dirty_all(bool dirty) {
        if (_dirty_steps.size() == 0)
            resize();
        for (auto i : _dirty_steps)
            i = dirty;
    }

    // Lookup position by dim name.
    // Return -1 or die if not found, depending on flag.
    int YkVarBase::get_dim_posn(const std::string& dim,
                                 bool die_on_failure,
                                 const std::string& die_msg) const {
        auto& dims = get_dim_tuple();
        int posn = dims.lookup_posn(dim);
        if (posn < 0 && die_on_failure) {
            THROW_YASK_EXCEPTION("Error: " + die_msg + ": dimension '" +
                                 dim + "' not found in " + make_info_string());
        }
        return posn;
    }

    #define IDX_STR(v) make_index_string(_corep->v)
    #define IDX_STR2(v, sep) make_index_string(_corep->v, sep)
    
    string YkVarBase::make_info_string(bool long_info) const {
            std::stringstream oss;
            if (is_scratch()) oss << "scratch ";
            if (is_user_var()) oss << "user-defined ";
            if (_fixed_size) oss << "fixed-size ";
            oss << _make_info_string() <<
                ", meta-data at " << (void*)this <<
                ", and core-data at " << (void*)_corep;
            #ifdef USE_OFFLOAD
            if (KernelEnv::_use_offload)
                oss << " (" << (void*)get_dev_ptr(_corep, false, false) <<
                    " on device)";
            #endif
            if (long_info) {
                if (_corep->_domains.get_num_dims())
                    oss <<
                        ", allocs = " << IDX_STR2(_allocs, " * ") <<
                        ", domains = " << IDX_STR2(_domains, " * ") <<
                        ", rank-offsets = " << IDX_STR(_rank_offsets) <<
                        ", local-offsets = " << IDX_STR(_local_offsets) <<
                        ", left-halos = " << IDX_STR(_left_halos) <<
                        ", right-halos = " << IDX_STR(_right_halos) <<
                        ", left-pads = " << IDX_STR(_actl_left_pads) <<
                        ", right-pads = " << IDX_STR(_actl_right_pads) <<
                        ", left-wf-exts = " << IDX_STR(_left_wf_exts) <<
                        ", right-wf-exts = " << IDX_STR(_right_wf_exts) <<
                        ", vec-strides = " << IDX_STR(_vec_strides);
                oss << ", " << _dirty_steps.size() << " dirty flag(s)";
            }
            return oss.str();
        }

    // Determine required padding from halos.
    // Does not include user-specified min padding or
    // final rounding for left pad.
    Indices YkVarBase::get_reqd_pad(const Indices& halos, const Indices& wf_exts) const {
        STATE_VARS(this);

        // Start with halos plus WF exts.
        Indices mp = halos.add_elements(wf_exts);

        // For any var, reads will be expanded to full vec-len during
        // computation.  For scratch vars, halo area must be written to.
        // Halo is sum of dependent's write halo and depender's read halo,
        // but these two components are not stored individually.  Write halo
        // will be expanded to full vec len during computation, requiring
        // load from read halo beyond full vec len.  Worst case is when
        // write halo is one and rest is read halo.  So, min padding should
        // be halos + wave-front exts + vec-len - 1. This vec-len should be
        // the solution one, not the one for this var to handle the case
        // where this var is not vectorized.
        for (int i = 0; i < get_num_dims(); i++) {
            if (mp[i])
                mp[i] += _corep->_soln_vec_lens[i] - 1;
        }
        return mp;
    }

    // Resizes the underlying generic var.
    // Updates dependent core info.
    // Fails if mem different and already alloc'd.
    void YkVarBase::resize() {
        STATE_VARS(this);

        // Original size.
        auto p = get_storage();
        IdxTuple old_allocs = get_allocs();

#ifdef TRACE
        string old_info;
        if (state->_env->_trace)
            old_info = make_info_string(true);
#endif

        // Check settings.
        for (int i = 0; i < get_num_dims(); i++) {
            if (_corep->_left_halos[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative left halo in var '" + get_name() + "'");
            if (_corep->_right_halos[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative right halo in var '" + get_name() + "'");
            if (_corep->_left_wf_exts[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative left wave-front ext in var '" + get_name() + "'");
            if (_corep->_right_wf_exts[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative right wave-front ext in var '" + get_name() + "'");
            if (_corep->_req_left_pads[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative left padding in var '" + get_name() + "'");
            if (_corep->_req_right_pads[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative right padding in var '" + get_name() + "'");
            if (_corep->_req_left_epads[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative left extra padding in var '" + get_name() + "'");
            if (_corep->_req_right_epads[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative right extra padding in var '" + get_name() + "'");
        }

        // Increase padding as needed and calculate new allocs.
        Indices new_left_pads = get_reqd_pad(_corep->_left_halos, _corep->_left_wf_exts);
        Indices new_right_pads = get_reqd_pad(_corep->_right_halos, _corep->_right_wf_exts);
        IdxTuple new_allocs(old_allocs);
        for (int i = 0; i < get_num_dims(); i++) {
            idx_t mbit = 1LL << i;

            // New allocation in each dim.
            new_allocs[i] = _corep->_domains[i];

            // Adjust padding only for domain dims.
            if (_corep->_domain_dim_mask & mbit) {

                // Add more padding.
                new_left_pads[i] += _corep->_req_left_epads[i];
                new_right_pads[i] += _corep->_req_right_epads[i];
                new_left_pads[i] = max(new_left_pads[i], _corep->_req_left_pads[i]);
                new_right_pads[i] = max(new_right_pads[i], _corep->_req_right_pads[i]);

                // If storage is allocated, get max of existing pad & new
                // pad.  This will avoid throwing an exception due to
                // decreasing requested padding after allocation.
                if (p) {
                    new_left_pads[i] = max(new_left_pads[i], _corep->_actl_left_pads[i]);
                    new_right_pads[i] = max(new_right_pads[i], _corep->_actl_right_pads[i]);
                }

                // Round left pad up to vec len.
                new_left_pads[i] = ROUND_UP(new_left_pads[i], _corep->_var_vec_lens[i]);

                // Round domain + right pad up to soln vec len by extending right pad.
                // Using soln vec len to allow reading a non-vec var in this dim
                // while calculating a vec var. (The var vec-len is always 1 or the same
                // as the soln vec-len in a given dim.)
                idx_t dprp = ROUND_UP(_corep->_domains[i] + new_right_pads[i],
                                      _corep->_soln_vec_lens[i]);
                new_right_pads[i] = dprp - _corep->_domains[i];

                // Make inner dim an odd number of vecs.
                // This reportedly helps avoid some uarch aliasing.
                if (!p &&
                    actl_opts->_allow_addl_pad &&
                    get_dim_name(i) == inner_layout_dim &&
                    (new_allocs[i] / _corep->_var_vec_lens[i]) % 2 == 0) {
                    new_right_pads[i] += _corep->_var_vec_lens[i];
                }

                // New allocation in each dim.
                new_allocs[i] += new_left_pads[i] + new_right_pads[i];
                assert(new_allocs[i] == new_left_pads[i] + _corep->_domains[i] + new_right_pads[i]);

                // Since the left pad and domain + right pad were rounded up,
                // the sum should also be a vec mult.
                assert(new_allocs[i] % _corep->_var_vec_lens[i] == 0);
            }

            // Non-domain dims.
            else {
                assert(new_allocs[i] == _corep->_domains[i]);
                assert(_corep->_var_vec_lens[i] == 1);
            }
        }

        // Attempt to change alloc with existing storage?
        if (p && old_allocs != new_allocs) {
            THROW_YASK_EXCEPTION("Error: attempt to change allocation size of var '" +
                get_name() + "' from " +
                make_index_string(old_allocs, " * ") + " to " +
                make_index_string(new_allocs, " * ") +
                " after storage has been allocated");
        }

        // Do the resize and calculate number of dirty bits needed.
        _corep->_allocs = new_allocs;
        _corep->_actl_left_pads = new_left_pads;
        _corep->_actl_right_pads = new_right_pads;
        size_t new_dirty = 1;      // default if no step dim.
        for (int i = 0; i < get_num_dims(); i++) {
            idx_t mbit = 1LL << i;

            // Calc vec-len values.
            _corep->_vec_left_pads[i] = new_left_pads[i] / _corep->_var_vec_lens[i];
            _corep->_vec_allocs[i] = _corep->_allocs[i] / _corep->_var_vec_lens[i];

            // Actual resize of underlying var.
            set_dim_size(i, _corep->_vec_allocs[i]);

            // Number of dirty bits is number of steps.
            if (_corep->_step_dim_mask & mbit)
                new_dirty = _corep->_allocs[i];
        }
        
        // Calc new strides.
        _corep->_vec_strides = get_vec_strides();

        // Resize dirty flags, too.
        size_t old_dirty = _dirty_steps.size();
        if (old_dirty != new_dirty) {

            // Resize & set all as dirty.
            _dirty_steps.assign(new_dirty, true);

            // Init range.
            init_valid_steps();
        }

        // Report changes in TRACE mode.
#ifdef TRACE
        if (state->_env->_trace) {
            string new_info = make_info_string(true);
            if (old_info != new_info)
                TRACE_MSG("resize: FROM " << old_info << " TO " << new_info);
        }
#endif

        // Copy changes to device.
        // TODO: do this only when needed.
        sync_core();
    }

    // Check whether dim is used and of allowed type.
    void YkVarBase::check_dim_type(const std::string& dim,
                                  const std::string& fn_name,
                                  bool step_ok,
                                  bool domain_ok,
                                  bool misc_ok) const {
        STATE_VARS(this);
        if (!is_dim_used(dim))
            THROW_YASK_EXCEPTION("Error in " + fn_name + "(): dimension '" +
                                 dim + "' not found in " + make_info_string());
        dims->check_dim_type(dim, fn_name, step_ok, domain_ok, misc_ok);
    }

    // Check for equality.
    // Return number of mismatches greater than epsilon.
    idx_t YkVarBase::compare(const YkVarBase* ref,
                              real_t epsilon,
                              int max_print) const {
        STATE_VARS(this);
        if (!ref) {
            DEBUG_MSG("** mismatch: no reference var.");
            return _corep->_allocs.product(); // total number of elements.
        }

        // Dims & sizes same?
        if (!are_dims_and_sizes_same(*ref)) {
            DEBUG_MSG("** mismatch due to incompatible vars: " <<
                      make_info_string() << " and " << ref->make_info_string());
            return _corep->_allocs.product(); // total number of elements.
        }

        // Compare each element.
        idx_t errs = 0;
        auto allocs = get_allocs();
        set<string> err_msgs;

        // This will loop over the entire allocation.
        // We use this as a handy way to get offsets,
        // but not all points will be used.
        allocs.visit_all_points_in_parallel
            ([&](const IdxTuple& pt, size_t idx) {

                // Adjust alloc indices to overall indices.
                IdxTuple opt(pt);
                bool ok = true;
                for (int i = 0; ok && i < pt.get_num_dims(); i++) {
                    auto val = pt.get_val(i);
                    idx_t mbit = 1LL << i;

                    // Convert to API index.
                    opt[i] = val;
                    if (!(_corep->_step_dim_mask & mbit))
                        opt[i] += _corep->_rank_offsets[i] + _corep->_local_offsets[i];

                    // Don't compare points outside the domain.
                    // TODO: check points in outermost halo.
                    auto& dname = pt.get_dim_name(i);
                    if (domain_dims.lookup(dname)) {
                        auto first_ok = _corep->_rank_offsets[i];
                        auto last_ok = first_ok + _corep->_domains[i] - 1;
                        if (opt[i] < first_ok || opt[i] > last_ok)
                            ok = false;
                    }
                }
                if (!ok)
                    return true; // stop processing this point, but keep going.

                idx_t asi = get_alloc_step_index(pt);
                auto te = read_elem(opt, asi, __LINE__);
                auto re = ref->read_elem(opt, asi, __LINE__);
                if (!within_tolerance(te, re, epsilon)) {
                    #pragma omp critical
                    {
                        errs++;
                        if (errs < max_print) {
                            err_msgs.insert(get_name() +
                                            "(" + opt.make_dim_val_str() +
                                            "): got " + to_string(te) +
                                            "; expected " + to_string(re));
                        }
                    }
                }
                return true;    // keep visiting.
            });

        for (auto& msg : err_msgs)
            DEBUG_MSG("** mismatch at " << msg);
        if (errs > max_print)
            DEBUG_MSG("** Additional errors not printed for var '" << get_name() << "'");
        TRACE_MSG("detailed compare returned " << errs);
        return errs;
    }

    // Make sure indices are in range.
    // Side-effect: If clipped_indices is not NULL,
    // 0) copy indices to *clipped_indices, and
    // 1) set them to in-range if out-of-range, and
    // 2) convert to rank-local and normalize them if 'normalize' is 'true'.
    bool YkVarBase::check_indices(const Indices& indices,
                                  const string& fn,    // name for error msg.
                                  bool strict_indices, // die if out-of-range.
                                  bool check_step,     // check step index.
                                  bool normalize,      // div by vec lens.
                                  Indices* clipped_indices) const {
        if (normalize)
            assert(clipped_indices != 0);
        STATE_VARS(this);
        bool all_ok = true;
        auto n = get_num_dims();
        if (indices.get_num_dims() != n) {
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: '" << fn << "' called with " <<
                                            indices.get_num_dims() <<
                                            " indices instead of " << n);
        }
        if (clipped_indices)
            *clipped_indices = indices;
        for (int i = 0; i < n; i++) {
            idx_t mbit = 1LL << i;
            bool is_step_dim = _corep->_step_dim_mask & mbit;
            idx_t idx = indices[i];
            bool ok = false;
            auto& dname = get_dim_name(i);

            // If this is the step dim and we're not checking
            // it, then anything is ok.
            if (is_step_dim && (!check_step || actl_opts->_step_wrap))
                ok = true;

            // Otherwise, check range.
            else {

                // First..last indices.
                auto first_ok = get_first_local_index(i);
                auto last_ok = get_last_local_index(i);
                if (idx >= first_ok && idx <= last_ok)
                    ok = true;

                // Handle outliers.
                if (!ok) {
                    if (strict_indices) {
                        THROW_YASK_EXCEPTION("Error: " + fn + ": index in dim '" + dname +
                                             "' is " + to_string(idx) + ", which is not in allowed range [" +
                                             to_string(first_ok) + "..." + to_string(last_ok) +
                                             "] of var '" + get_name() + "'");
                    }

                    // Update the output indices.
                    if (clipped_indices) {
                        if (idx < first_ok)
                            (*clipped_indices)[i] = first_ok;
                        if (idx > last_ok)
                            (*clipped_indices)[i] = last_ok;
                    }
                    all_ok = false;
                }
            } // need to check.

            // Normalize?
            if (clipped_indices && normalize) {
                if (_corep->_domain_dim_mask & mbit) {
                    (*clipped_indices)[i] -= _corep->_rank_offsets[i]; // rank-local.
                    (*clipped_indices)[i] = idiv_flr((*clipped_indices)[i], _corep->_var_vec_lens[i]);
                }
            }
        } // var dims.
        return all_ok;
    }

    // Update what steps are valid.
    void YkVarBase::update_valid_step(idx_t t) {
        STATE_VARS(this);
        if (_has_step_dim) {

            // If 't' is before first step, pull offset back.
            if (t < get_first_local_index(step_posn))
                _corep->_local_offsets[step_posn] = t;

            // If 't' is after last step, push offset out.
            else if (t > get_last_local_index(step_posn))
                _corep->_local_offsets[step_posn] = t - _corep->_domains[step_posn] + 1;

            TRACE_MSG("update_valid_step(" << t << "): valid step(s) in '" <<
                      get_name() << "' are now [" << get_first_local_index(step_posn) <<
                      " ... " << get_last_local_index(step_posn) << "]");
        }
    }


    // Set dirty flags between indices.
    void YkVarBase::set_dirty_in_slice(const Indices& first_indices,
                                        const Indices& last_indices) {
        if (_has_step_dim) {
            for (idx_t i = first_indices[+step_posn];
                 i <= last_indices[+step_posn]; i++)
                set_dirty(true, i);
        } else
            set_dirty_using_alloc_index(true, 0);
    }

    // Print one element like
    // "message: myvar[x=4, y=7] = 3.14 at line 35".
    void YkVarBase::print_elem(const std::string& msg,
                               const Indices& idxs,
                               real_t eval,
                               int line) const {
        STATE_VARS_CONST(this);
        string str;
        if (msg.length())
            str = msg + ": ";
        str += get_name() + "[" +
            make_index_string(idxs) + "] = " + to_string(eval);
        if (line)
            str += " at line " + to_string(line);
        TRACE_MEM_MSG(str);
    }

    // Print each elem in one vector.
    // Indices must be normalized and rank-relative.
    void YkVarBase::print_vec_norm(const std::string& msg,
                                   const Indices& idxs,
                                   const real_vec_t& val,
                                   int line) const {
        STATE_VARS_CONST(this);

        // Convert to elem indices.
        Indices eidxs = idxs.mul_elements(_corep->_var_vec_lens);

        // Add offsets, i.e., convert to overall indices.
        eidxs = eidxs.add_elements(_corep->_rank_offsets);

        IdxTuple idxs2 = get_dim_tuple();
        eidxs.set_tuple_vals(idxs2);      // set vals from eidxs.
        // TODO: is above correct for vars that aren't domain dims?
        Indices idxs3(idxs2);

        // Visit every point in fold.
        auto& folds = dims->_fold_sizes;
        bool first_inner = dims->_fold_pts.is_first_inner();
        folds.visit_all_points
            (first_inner,
             [&](const Indices& fofs, size_t idx) {

                 // Get element from vec val.
                 real_t ev = val[idx];

                 // Add fold offsets to elem indices for printing.
                 auto pt3 = idxs3.add_elements(fofs);

                 print_elem(msg, pt3, ev, line);
                 return true; // keep visiting.
             });
    }

} // namespace.

