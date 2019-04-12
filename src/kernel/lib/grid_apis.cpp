/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2019, Intel Corporation

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

// Implement methods for yk_grid APIs.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

#define DEPRECATED(api_name) \
    cerr << "\n*** WARNING: call to deprecated YASK API '"              \
    #api_name "' that will be removed in a future release ***\n"

    // APIs to get info from vars: one with name of dim with a lot
    // of checking, and one with index of dim with no checking.
#define GET_GRID_API(api_name, expr, step_ok, domain_ok, misc_ok, prep_req) \
    idx_t YkGridImpl::api_name(const string& dim) const {               \
        STATE_VARS(gbp());                                              \
        dims->checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok); \
        int posn = gb().get_dim_posn(dim, true, #api_name);             \
        idx_t mbit = 1LL << posn;                                       \
        if (prep_req && gb()._rank_offsets[posn] < 0)                   \
            THROW_YASK_EXCEPTION("Error: '" #api_name "()' called on grid '" + \
                                 get_name() + "' before calling 'prepare_solution()'"); \
        auto rtn = expr;                                                \
        return rtn;                                                     \
    }                                                                   \
    idx_t YkGridImpl::api_name(int posn) const {                        \
        STATE_VARS(gbp());                                              \
        idx_t mbit = 1LL << posn;                                       \
        auto rtn = expr;                                                \
        return rtn;                                                     \
    }

    // Internal APIs.
    GET_GRID_API(_get_left_wf_ext, gb()._left_wf_exts[posn], true, true, true, false)
    GET_GRID_API(_get_right_wf_ext, gb()._right_wf_exts[posn], true, true, true, false)
    GET_GRID_API(_get_vec_len, gb()._vec_lens[posn], true, true, true, true)
    GET_GRID_API(_get_rank_offset, gb()._rank_offsets[posn], true, true, true, true)
    GET_GRID_API(_get_local_offset, gb()._local_offsets[posn], true, true, true, false)

    // Exposed APIs.
    GET_GRID_API(get_first_local_index, gb().get_first_local_index(posn), true, true, true, true)
    GET_GRID_API(get_last_local_index, gb().get_last_local_index(posn), true, true, true, true)
    GET_GRID_API(get_first_misc_index, gb()._local_offsets[posn], false, false, true, false)
    GET_GRID_API(get_last_misc_index, gb()._local_offsets[posn] + gb()._domains[posn] - 1, false, false, true, false)
    GET_GRID_API(get_rank_domain_size, gb()._domains[posn], false, true, false, false)
    GET_GRID_API(get_left_pad_size, gb()._actl_left_pads[posn], false, true, false, false)
    GET_GRID_API(get_right_pad_size, gb()._actl_right_pads[posn], false, true, false, false)
    GET_GRID_API(get_left_halo_size, gb()._left_halos[posn], false, true, false, false)
    GET_GRID_API(get_right_halo_size, gb()._right_halos[posn], false, true, false, false)
    GET_GRID_API(get_left_extra_pad_size, gb()._actl_left_pads[posn] - gb()._left_halos[posn], false, true, false, false)
    GET_GRID_API(get_right_extra_pad_size, gb()._actl_right_pads[posn] - gb()._right_halos[posn], false, true, false, false)
    GET_GRID_API(get_alloc_size, gb()._allocs[posn], true, true, true, false)
    GET_GRID_API(get_first_rank_domain_index, gb()._rank_offsets[posn], false, true, false, true)
    GET_GRID_API(get_last_rank_domain_index, gb()._rank_offsets[posn] + gb()._domains[posn] - 1, false, true, false, true)
    GET_GRID_API(get_first_rank_halo_index, gb()._rank_offsets[posn] - gb()._left_halos[posn], false, true, false, true)
    GET_GRID_API(get_last_rank_halo_index, gb()._rank_offsets[posn] + gb()._domains[posn] +
                 gb()._right_halos[posn] - 1, false, true, false, true)
    GET_GRID_API(get_first_rank_alloc_index, gb().get_first_local_index(posn), false, true, false, true)
    GET_GRID_API(get_last_rank_alloc_index, gb().get_last_local_index(posn), false, true, false, true)

    // Deprecated APIs.
    GET_GRID_API(get_pad_size, gb()._actl_left_pads[posn]; DEPRECATED(get_pad_size), false, true, false, false)
    GET_GRID_API(get_halo_size, gb()._left_halos[posn]; DEPRECATED(get_halo_size), false, true, false, false)
    GET_GRID_API(get_extra_pad_size, gb()._actl_left_pads[posn] - gb()._left_halos[posn];
                 DEPRECATED(get_extra_pad_size), false, true, false, false)
#undef GET_GRID_API

    // APIs to set vars.
#define COMMA ,
#define SET_GRID_API(api_name, expr, step_ok, domain_ok, misc_ok)       \
    void YkGridImpl::api_name(const string& dim, idx_t n) {             \
        STATE_VARS(gbp());                                              \
        TRACE_MSG("grid '" << get_name() << "'."                        \
                   #api_name "('" << dim << "', " << n << ")");         \
        dims->checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok); \
        int posn = gb().get_dim_posn(dim, true, #api_name);              \
        idx_t mbit = 1LL << posn;                                       \
        expr;                                                           \
    }                                                                   \
    void YkGridImpl::api_name(int posn, idx_t n) {                      \
        STATE_VARS(gbp());                                              \
        idx_t mbit = 1LL << posn;                                       \
        int dim = posn;                                                 \
        expr;                                                           \
    }

    // These are the internal, unchecked access functions that allow
    // changes prohibited thru the APIs.
    SET_GRID_API(_set_rank_offset, gb()._rank_offsets[posn] = n, true, true, true)
    SET_GRID_API(_set_local_offset, gb()._local_offsets[posn] = n;
                 assert(imod_flr(n, gb()._vec_lens[posn]) == 0);
                 gb()._vec_local_offsets[posn] = n / gb()._vec_lens[posn], true, true, true)
    SET_GRID_API(_set_domain_size, gb()._domains[posn] = n; resize(), true, true, true)
    SET_GRID_API(_set_left_pad_size, gb()._actl_left_pads[posn] = n; resize(), true, true, true)
    SET_GRID_API(_set_right_pad_size, gb()._actl_right_pads[posn] = n; resize(), true, true, true)
    SET_GRID_API(_set_left_wf_ext, gb()._left_wf_exts[posn] = n; resize(), true, true, true)
    SET_GRID_API(_set_right_wf_ext, gb()._right_wf_exts[posn] = n; resize(), true, true, true)
    SET_GRID_API(_set_alloc_size, gb()._domains[posn] = n; resize(), true, true, true)

    // These are the safer ones used in the APIs.
    SET_GRID_API(set_left_halo_size, gb()._left_halos[posn] = n; resize(), false, true, false)
    SET_GRID_API(set_right_halo_size, gb()._right_halos[posn] = n; resize(), false, true, false)
    SET_GRID_API(set_halo_size, gb()._left_halos[posn] = gb()._right_halos[posn] = n; resize(), false, true, false)
    SET_GRID_API(set_alloc_size, gb()._domains[posn] = n; resize(),
                 gb()._is_dynamic_step_alloc, gb()._fixed_size, gb()._is_dynamic_misc_alloc)
    SET_GRID_API(set_left_min_pad_size, gb()._req_left_pads[posn] = n; resize(), false, true, false)
    SET_GRID_API(set_right_min_pad_size, gb()._req_right_pads[posn] = n; resize(), false, true, false)
    SET_GRID_API(set_min_pad_size, gb()._req_left_pads[posn] = gb()._req_right_pads[posn] = n; resize(),
                 false, true, false)
    SET_GRID_API(set_left_extra_pad_size,
                 set_left_min_pad_size(posn, gb()._left_halos[posn] + n), false, true, false)
    SET_GRID_API(set_right_extra_pad_size,
                 set_right_min_pad_size(posn, gb()._right_halos[posn] + n), false, true, false)
    SET_GRID_API(set_extra_pad_size, set_left_extra_pad_size(posn, n);
                 set_right_extra_pad_size(posn, n), false, true, false)
    SET_GRID_API(set_first_misc_index, gb()._local_offsets[posn] = n, false, false, gb()._is_user_grid)
#undef COMMA
#undef SET_GRID_API

    bool YkGridImpl::is_storage_layout_identical(const YkGridImpl* op,
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
            if (gb()._vec_lens[i] != op->gb()._vec_lens[i])
                return false;

            // Same dim sizes?
            if (check_sizes) {
                if (gb()._domains[i] != op->gb()._domains[i])
                    return false;
                if (gb()._actl_left_pads[i] != op->gb()._actl_left_pads[i])
                    return false;
                if (gb()._actl_right_pads[i] != op->gb()._actl_right_pads[i])
                    return false;
            }
        }
        return true;
    }

    void YkGridImpl::fuse_grids(yk_grid_ptr src) {
        STATE_VARS(gbp());
        auto op = dynamic_pointer_cast<YkGridImpl>(src);
        TRACE_MSG("fuse_grids(" << src.get() << "): this=" << gb().make_info_string() <<
                  "; source=" << op->gb().make_info_string());
        assert(op);
        auto* sp = op.get();
        assert(!_gbp->is_scratch());

        // Check conditions for fusing into a non-user grid.
        bool force_native = false;
        if (gb().is_user_grid()) {
            force_native = true;
            if (!is_storage_layout_identical(sp, false))
                THROW_YASK_EXCEPTION("Error: fuse_grids(): attempt to replace meta-data"
                                     " of " + gb().make_info_string() +
                                     " used in solution with incompatible " +
                                     sp->gb().make_info_string());
        }

        // Save ptr to source-storage grid before fusing meta-data.
        GridBasePtr st_gbp = sp->_gbp; // Shared-ptr to keep source active to end of method.
        GenericGridBase* st_ggb = st_gbp->_ggb;

        // Fuse meta-data.
        _gbp = sp->_gbp;

        // Tag grid as a non-user grid if the original one was.
        if (force_native)
            _gbp->set_user_grid(false);

        // Fuse storage.
        gg().share_storage(st_ggb);

        TRACE_MSG("after fuse_grids: this=" << gb().make_info_string() <<
                  "; source=" << op->gb().make_info_string());
    }

    // API get, set, etc.
    bool YkGridImpl::are_indices_local(const Indices& indices) const {
        if (!is_storage_allocated())
            return false;
        return gb().checkIndices(indices, "are_indices_local", false, true, false);
    }
    double YkGridImpl::get_element(const Indices& indices) const {
        STATE_VARS(gbp());
        TRACE_MSG("get_element({" << gb().makeIndexString(indices) << "}) on " <<
                  gb().make_info_string());
        if (!is_storage_allocated())
            THROW_YASK_EXCEPTION("Error: call to 'get_element' with no storage allocated for grid '" +
                                 get_name() + "'");
        gb().checkIndices(indices, "get_element", true, true, false);
        idx_t asi = gb().get_alloc_step_index(indices);
        real_t val = gb().readElem(indices, asi, __LINE__);
        TRACE_MSG("get_element({" << gb().makeIndexString(indices) << "}) on '" <<
                  get_name() + "' returns " << val);
        return double(val);
    }
    idx_t YkGridImpl::set_element(double val,
                                  const Indices& indices,
                                  bool strict_indices) {
        STATE_VARS(gbp());
        TRACE_MSG("set_element(" << val << ", {" <<
                  gb().makeIndexString(indices) << "}, " <<
                  strict_indices << ") on " <<
                  gb().make_info_string());
        idx_t nup = 0;
        if (!get_raw_storage_buffer() && strict_indices)
            THROW_YASK_EXCEPTION("Error: call to 'set_element' with no storage allocated for grid '" +
                                 get_name() + "'");
        if (get_raw_storage_buffer() &&

            // Don't check step index because this is a write-only API
            // that updates the step index.
            gb().checkIndices(indices, "set_element", strict_indices, false, false)) {
            idx_t asi = gb().get_alloc_step_index(indices);
            gb().writeElem(real_t(val), indices, asi, __LINE__);
            nup++;

            // Set appropriate dirty flag.
            gb().set_dirty_using_alloc_index(true, asi);
        }
        TRACE_MSG("set_element(" << val << ", {" <<
                  gb().makeIndexString(indices) << "}, " <<
                  strict_indices << ") on '" <<
                  get_name() + "' returns " << nup);
        return nup;
    }
    idx_t YkGridImpl::add_to_element(double val,
                                     const Indices& indices,
                                     bool strict_indices) {
        STATE_VARS(gbp());
        TRACE_MSG("add_to_element(" << val << ", {" <<
                  gb().makeIndexString(indices) <<  "}, " <<
                  strict_indices << ") on " <<
                  gb().make_info_string());
        idx_t nup = 0;
        if (!get_raw_storage_buffer() && strict_indices)
            THROW_YASK_EXCEPTION("Error: call to 'add_to_element' with no storage allocated for grid '" +
                                 get_name() + "'");
        if (get_raw_storage_buffer() &&

            // Check step index because this API must read before writing.
            gb().checkIndices(indices, "add_to_element", strict_indices, true, false)) {
            idx_t asi = gb().get_alloc_step_index(indices);
            gb().addToElem(real_t(val), indices, asi, __LINE__);
            nup++;

            // Set appropriate dirty flag.
            gb().set_dirty_using_alloc_index(true, asi);
        }
        TRACE_MSG("add_to_element(" << val << ", {" <<
                  gb().makeIndexString(indices) <<  "}, " <<
                  strict_indices << ") on '" <<
                  get_name() + "' returns " << nup);
        return nup;
    }

    idx_t YkGridBase::get_elements_in_slice(void* buffer_ptr,
                                            const Indices& first_indices,
                                            const Indices& last_indices) const {
        STATE_VARS(this);
        TRACE_MSG("get_elements_in_slice(" << buffer_ptr << ", {" <<
                  makeIndexString(first_indices) << "}, {" <<
                  makeIndexString(last_indices) << "}) on " <<
                  make_info_string());
        if (_ggb->get_storage() == 0)
            THROW_YASK_EXCEPTION("Error: call to 'get_elements_in_slice' with no storage allocated for grid '" +
                                 _ggb->get_name() + "'");
        checkIndices(first_indices, "get_elements_in_slice", true, true, false);
        checkIndices(last_indices, "get_elements_in_slice", true, true, false);

        // Find range.
        IdxTuple numElemsTuple = get_slice_range(first_indices, last_indices);

        // Visit points in slice.
        numElemsTuple.visitAllPointsInParallel
            ([&](const IdxTuple& ofs, size_t idx) {
                Indices pt = first_indices.addElements(ofs);

                // TODO: move this outside of loop for const step index.
                idx_t asi = get_alloc_step_index(pt);

                real_t val = readElem(pt, asi, __LINE__);
                ((real_t*)buffer_ptr)[idx] = val;
                return true;    // keep going.
            });
        auto nup = numElemsTuple.product();
        TRACE_MSG("get_elements_in_slice(" << buffer_ptr << ", {" <<
                  makeIndexString(first_indices) << "}, {" <<
                  makeIndexString(last_indices) << "}) on '" <<
                  _ggb->get_name() + "' returns " << nup);
        return nup;
    }
    idx_t YkGridBase::set_elements_in_slice_same(double val,
                                                 const Indices& first_indices,
                                                 const Indices& last_indices,
                                                 bool strict_indices) {
        STATE_VARS(this);
        TRACE_MSG("set_elements_in_slice_same(" << val << ", {" <<
                  makeIndexString(first_indices) << "}, {" <<
                  makeIndexString(last_indices) <<  "}, " <<
                  strict_indices << ") on " <<
                  make_info_string());
        if (_ggb->get_storage() == 0) {
            if (strict_indices)
                THROW_YASK_EXCEPTION("Error: call to 'set_elements_in_slice_same' with no storage allocated for grid '" +
                                     _ggb->get_name() + "'");
            return 0;
        }

        // 'Fixed' copy of indices.
        Indices first, last;
        checkIndices(first_indices, "set_elements_in_slice_same",
                     strict_indices, false, false, &first);
        checkIndices(last_indices, "set_elements_in_slice_same",
                     strict_indices, false, false, &last);

        // Find range.
        IdxTuple numElemsTuple = get_slice_range(first, last);

        // Visit points in slice.
        // TODO: optimize by setting vectors when possible.
        numElemsTuple.visitAllPointsInParallel([&](const IdxTuple& ofs,
                                                   size_t idx) {
                Indices pt = first.addElements(ofs);

                // TODO: move this outside of loop for const step index.
                idx_t asi = get_alloc_step_index(pt);

                writeElem(real_t(val), pt, asi, __LINE__);
                return true;    // keep going.
            });

        // Set appropriate dirty flag(s).
        set_dirty_in_slice(first, last);

        auto nup = numElemsTuple.product();
        TRACE_MSG("set_elements_in_slice_same(" << val << ", {" <<
                  makeIndexString(first_indices) << "}, {" <<
                  makeIndexString(last_indices) <<  "}, " <<
                  strict_indices << ") on '" <<
                  _ggb->get_name() + "' returns " << nup);
        return nup;
    }
    idx_t YkGridBase::set_elements_in_slice(const void* buffer_ptr,
                                            const Indices& first_indices,
                                            const Indices& last_indices) {
        STATE_VARS(this);
        TRACE_MSG("set_elements_in_slice(" << buffer_ptr << ", {" <<
                  makeIndexString(first_indices) << "}, {" <<
                  makeIndexString(last_indices) <<  "}) on " <<
                  make_info_string());
        if (_ggb->get_storage() == 0)
            THROW_YASK_EXCEPTION("Error: call to 'set_elements_in_slice' with no storage allocated for grid '" +
                                 _ggb->get_name() + "'");
        checkIndices(first_indices, "set_elements_in_slice", true, false, false);
        checkIndices(last_indices, "set_elements_in_slice", true, false, false);

        // Find range.
        IdxTuple numElemsTuple = get_slice_range(first_indices, last_indices);

        // Visit points in slice.
        numElemsTuple.visitAllPointsInParallel
            ([&](const IdxTuple& ofs,
                 size_t idx) {
                Indices pt = first_indices.addElements(ofs);

                // TODO: move this outside of loop for const step index.
                idx_t asi = get_alloc_step_index(pt);

                real_t val = ((real_t*)buffer_ptr)[idx];
                writeElem(val, pt, asi, __LINE__);
                return true;    // keep going.
            });

        // Set appropriate dirty flag(s).
        set_dirty_in_slice(first_indices, last_indices);

        auto nup = numElemsTuple.product();
        TRACE_MSG("set_elements_in_slice(" << buffer_ptr << ", {" <<
                  makeIndexString(first_indices) << "}, {" <<
                  makeIndexString(last_indices) <<  "}) on '" <<
                  _ggb->get_name() + "' returns " << nup);
        return nup;
    }

} // namespace.

