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

// Implement methods for yk_grid APIs.

#include "yask.hpp"
using namespace std;

namespace yask {

    // APIs to get info from vars.
#define GET_GRID_API(api_name, expr, step_ok, domain_ok, misc_ok, prep_req) \
    idx_t YkGridBase::api_name(const string& dim) const {               \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        int posn = get_dim_posn(dim, true, #api_name);                  \
        if (prep_req && _offsets[posn] < 0)                             \
            THROW_YASK_EXCEPTION("Error: '" #api_name "()' called on grid '" << \
                                 get_name() << "' before calling 'prepare_solution()'"); \
        return expr;                                                    \
    }                                                                   \
    idx_t YkGridBase::api_name(int posn) const {                        \
        return expr;                                                    \
    }
    GET_GRID_API(get_rank_domain_size, _domains[posn], false, true, false, false)
    GET_GRID_API(get_left_pad_size, _left_pads[posn], false, true, false, false) // _left_pads is actual size.
    GET_GRID_API(get_right_pad_size, _allocs[posn] - _left_pads[posn], false, true, false, false) // _right_pads is request only.
    GET_GRID_API(get_pad_size, _left_pads[posn], false, true, false, false)
    GET_GRID_API(get_left_halo_size, _left_halos[posn], false, true, false, false)
    GET_GRID_API(get_right_halo_size, _right_halos[posn], false, true, false, false)
    GET_GRID_API(get_halo_size, _left_halos[posn], false, true, false, false)
    GET_GRID_API(get_first_misc_index, _offsets[posn], false, false, true, false)
    GET_GRID_API(get_last_misc_index, _offsets[posn] + _domains[posn] - 1, false, false, true, false)
    GET_GRID_API(get_left_extra_pad_size, _left_pads[posn] - _left_halos[posn], false, true, false, false)
    GET_GRID_API(get_right_extra_pad_size, (_allocs[posn] - _left_pads[posn] - _domains[posn]) -
                 _right_halos[posn], false, true, false, false)
    GET_GRID_API(get_extra_pad_size, _left_pads[posn] - _left_halos[posn], false, true, false, false)
    GET_GRID_API(get_alloc_size, _allocs[posn], true, true, true, false)
    GET_GRID_API(get_first_rank_domain_index, _offsets[posn] - _local_offsets[posn], false, true, false, true)
    GET_GRID_API(get_last_rank_domain_index, _offsets[posn] - _local_offsets[posn] + _domains[posn] - 1;
                 assert(!_is_scratch), false, true, false, true)
    GET_GRID_API(get_first_rank_halo_index, _offsets[posn] - _left_halos[posn], false, false, true, true)
    GET_GRID_API(get_last_rank_halo_index, _offsets[posn] + _domains[posn] + _right_halos[posn] - 1, false, false, true, true)
    GET_GRID_API(get_first_rank_alloc_index, _offsets[posn] - _left_pads[posn], false, true, false, true)
    GET_GRID_API(get_last_rank_alloc_index, _offsets[posn] - _left_pads[posn] + _allocs[posn] - 1, false, true, false, true)
    GET_GRID_API(_get_left_wf_ext, _left_wf_exts[posn], true, true, true, false)
    GET_GRID_API(_get_right_wf_ext, _right_wf_exts[posn], true, true, true, false)
    GET_GRID_API(_get_offset, _offsets[posn], true, true, true, true)
    GET_GRID_API(_get_local_offset, _local_offsets[posn], true, true, true, false)
    GET_GRID_API(_get_first_alloc_index, _offsets[posn] - _left_pads[posn], true, true, true, true)
    GET_GRID_API(_get_last_alloc_index, _offsets[posn] - _left_pads[posn] + _allocs[posn] - 1, true, true, true, true)
#undef GET_GRID_API
    
    // APIs to set vars.
#define COMMA ,
#define SET_GRID_API(api_name, expr, step_ok, domain_ok, misc_ok)       \
    void YkGridBase::api_name(const string& dim, idx_t n) {             \
        TRACE_MSG0(get_ostr(), "grid '" << get_name() << "'."           \
                   #api_name "('" << dim << "', " << n << ")");          \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        int posn = get_dim_posn(dim, true, #api_name);                  \
        expr;                                                           \
    }                                                                   \
    void YkGridBase::api_name(int posn, idx_t n) {                      \
        int dim = posn;                                                 \
        expr;                                                           \
    }
    SET_GRID_API(_set_offset, _offsets[posn] = n, true, true, true)
    SET_GRID_API(_set_local_offset, _local_offsets[posn] = n;
                 _vec_local_offsets[posn] = n / _vec_lens[posn], true, true, true)
    SET_GRID_API(_set_domain_size, _domains[posn] = n; resize(), true, true, true)
    SET_GRID_API(_set_left_pad_size, _left_pads[posn] = n; resize(), true, true, true)
    SET_GRID_API(_set_right_pad_size, _right_pads[posn] = n; resize(), true, true, true)
    SET_GRID_API(_set_left_wf_ext, _left_wf_exts[posn] = n; resize(), true, true, true)
    SET_GRID_API(_set_right_wf_ext, _right_wf_exts[posn] = n; resize(), true, true, true)
    SET_GRID_API(set_left_halo_size, _left_halos[posn] = n; resize(), false, true, false)
    SET_GRID_API(set_right_halo_size, _right_halos[posn] = n; resize(), false, true, false)
    SET_GRID_API(set_halo_size, _left_halos[posn] = _right_halos[posn] = n; resize(), false, true, false)

    SET_GRID_API(set_alloc_size, _set_domain_size(posn, n), true, false, true)
    SET_GRID_API(set_left_min_pad_size,
                 if (!get_raw_storage_buffer() && n > _left_pads[posn])
                     _set_left_pad_size(posn, n),
                 false, true, false)
    SET_GRID_API(set_right_min_pad_size,
                 if (!get_raw_storage_buffer() && n > _right_pads[posn])
                     _set_right_pad_size(posn, n),
                 false, true, false)
    SET_GRID_API(set_min_pad_size,
                 if (!get_raw_storage_buffer() && n > _left_pads[posn])
                     _set_left_pad_size(posn, n);
                 if (!get_raw_storage_buffer() && n > _right_pads[posn])
                     _set_right_pad_size(posn, n),
                 false, true, false)
    SET_GRID_API(set_left_extra_pad_size,
                 set_left_min_pad_size(posn, _left_halos[posn] + _left_wf_exts[posn] + n), false, true, false)
    SET_GRID_API(set_right_extra_pad_size,
                 set_right_min_pad_size(posn, _right_halos[posn] + _right_wf_exts[posn] + n), false, true, false)
    SET_GRID_API(set_extra_pad_size, set_left_extra_pad_size(posn, n);
                 set_right_extra_pad_size(posn, n), false, true, false)
    SET_GRID_API(set_first_misc_index, _offsets[posn] = n, false, false, true)
#undef COMMA
#undef SET_GRID_API
    
    bool YkGridBase::is_storage_layout_identical(const yk_grid_ptr other) const {
        auto op = dynamic_pointer_cast<YkGridBase>(other);
        assert(op);

        // Same size?
        if (get_num_storage_bytes() != op->get_num_storage_bytes())
            return false;

        // Same dims?
        if (get_num_dims() != op->get_num_dims())
            return false;
        for (int i = 0; i < get_num_dims(); i++) {
            auto dname = get_dim_name(i);

            // Same dims?
            if (dname != op->get_dim_name(i))
                return false;

            // Same sizes?
            // NB: not checking right pads because actual values
            // are determined as function of other 3.
            if (_allocs[i] != op->_allocs[i])
                return false;
            if (_domains[i] != op->_domains[i])
                return false;
            if (_left_pads[i] != op->_left_pads[i])
                return false;
        }
        return true;
    }

    void YkGridBase::share_storage(yk_grid_ptr source) {
        auto sp = dynamic_pointer_cast<YkGridBase>(source);
        assert(sp);

        if (!sp->get_raw_storage_buffer()) {
            THROW_YASK_EXCEPTION("Error: share_storage() called without source storage allocated");
        }

        // Determine required padding from halos.
        Indices left_pads2 = getReqdPad(_left_halos, _left_wf_exts);
        Indices right_pads2 = getReqdPad(_right_halos, _left_wf_exts);

        // NB: requirements to successful share_storage() is not as strict as
        // is_storage_layout_identical(). See note on pad & halo below and API docs.
        for (int i = 0; i < get_num_dims(); i++) {
            auto dname = get_dim_name(i);

            // Same dims?
            if (sp->get_num_dims() != get_num_dims() ||
                sp->get_dim_name(i) != dname)
                THROW_YASK_EXCEPTION("Error: share_storage() called with incompatible grids: " <<
                                     make_info_string() << " and " << sp->make_info_string());


            // Check folding.
            if (_vec_lens[i] != sp->_vec_lens[i]) {
                THROW_YASK_EXCEPTION("Error: attempt to share storage from grid '" << sp->get_name() <<
                                     "' of fold-length " << sp->_vec_lens[i] << " with grid '" << get_name() <<
                                     "' of fold-length " << _vec_lens[i] << " in '" << dname << "' dim");
            }

            // Not a domain dim?
            bool is_domain = _dims->_domain_dims.lookup(dname) != 0;
            if (!is_domain) {
                auto tas = get_alloc_size(dname);
                auto sas = sp->get_alloc_size(dname);
                if (tas != sas) {
                    THROW_YASK_EXCEPTION("Error: attempt to share storage from grid '" << sp->get_name() <<
                                         "' of alloc-size " << sas << " with grid '" << get_name() <<
                                         "' of alloc-size " << tas << " in '" << dname << "' dim");
                }
            }

            // Domain dim.
            else {
                auto tdom = get_rank_domain_size(i);
                auto sdom = sp->get_rank_domain_size(i);
                if (tdom != sdom) {
                    THROW_YASK_EXCEPTION("Error: attempt to share storage from grid '" << sp->get_name() <<
                                         "' of domain-size " << sdom << " with grid '" << get_name() <<
                                         "' of domain-size " << tdom << " in '" << dname << "' dim");
                }

                // Halo and pad sizes don't have to be the same.
                // Requirement is that halo (reqd pad) of target fits inside of pad of source.
                auto spad = sp->get_left_pad_size(i);
                if (left_pads2[i] > spad) {
                    THROW_YASK_EXCEPTION("Error: attempt to share storage from grid '" << sp->get_name() <<
                                         "' of left padding-size " << spad <<
                                         ", which is insufficient for grid '" << get_name() <<
                                         "' requiring " << left_pads2[i] << " in '" << dname << "' dim");
                }
                spad = sp->get_right_pad_size(i);
                if (right_pads2[i] > spad) {
                    THROW_YASK_EXCEPTION("Error: attempt to share storage from grid '" << sp->get_name() <<
                                         "' of right padding-size " << spad <<
                                         ", which is insufficient for grid '" << get_name() <<
                                         "' requiring " << right_pads2[i] << " in '" << dname << "' dim");
                }
            }
        }

        // Copy pad sizes.
        for (int i = 0; i < get_num_dims(); i++) {
            auto dname = get_dim_name(i);
            bool is_domain = _dims->_domain_dims.lookup(dname) != 0;
            if (is_domain) {
                _left_pads[i] = sp->_left_pads[i];
                _right_pads[i] = sp->_right_pads[i];
            }
        }
        
        // Copy data.
        release_storage();
        resize();
        if (!share_data(sp.get(), true)) {
            THROW_YASK_EXCEPTION("Error: unexpected failure in data sharing");
        }
    }

    // API get, set, setc.
    bool YkGridBase::is_element_allocated(const Indices& indices) const {
        if (!is_storage_allocated())
            return false;
        return checkIndices(indices, "is_element_allocated", false, false);
    }
    double YkGridBase::get_element(const Indices& indices) const {
        if (!is_storage_allocated()) {
            THROW_YASK_EXCEPTION("Error: call to 'get_element' with no data allocated for grid '" <<
                                 get_name() << "'");
        }
        checkIndices(indices, "get_element", true, false);
        idx_t asi = get_alloc_step_index(indices);
        real_t val = readElem(indices, asi, __LINE__);
        return double(val);
    }
    idx_t YkGridBase::set_element(double val,
                                  const Indices& indices,
                                  bool strict_indices) {
        idx_t nup = 0;
        if (get_raw_storage_buffer() &&
            checkIndices(indices, "set_element", strict_indices, false)) {
            idx_t asi = get_alloc_step_index(indices);
            writeElem(real_t(val), indices, asi, __LINE__);
            nup++;

            // Set appropriate dirty flag.
            set_dirty_using_alloc_index(true, asi);
        }
        return nup;
    }
    idx_t YkGridBase::add_to_element(double val,
                                     const Indices& indices,
                                     bool strict_indices) {
        idx_t nup = 0;
        if (get_raw_storage_buffer() &&
            checkIndices(indices, "add_to_element", strict_indices, false)) {
            idx_t asi = get_alloc_step_index(indices);
            addToElem(real_t(val), indices, asi, __LINE__);
            nup++;

            // Set appropriate dirty flag.
            set_dirty_using_alloc_index(true, asi);
        }
        return nup;
    }
    
    idx_t YkGridBase::get_elements_in_slice(void* buffer_ptr,
                                            const Indices& first_indices,
                                            const Indices& last_indices) const {
        if (!is_storage_allocated()) {
            THROW_YASK_EXCEPTION("Error: call to 'get_elements_in_slice' with no data allocated for grid '" <<
                                 get_name() << "'");
        }
        checkIndices(first_indices, "get_elements_in_slice", true, false);
        checkIndices(last_indices, "get_elements_in_slice", true, false);

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
        return numElemsTuple.product();
    }
    idx_t YkGridBase::set_elements_in_slice_same(double val,
                                                 const Indices& first_indices,
                                                 const Indices& last_indices,
                                                 bool strict_indices) {
        if (!is_storage_allocated())
            return 0;
        
        // 'Fixed' copy of indices.
        Indices first, last;
        checkIndices(first_indices, "set_elements_in_slice_same",
                     strict_indices, false, &first);
        checkIndices(last_indices, "set_elements_in_slice_same",
                     strict_indices, false, &last);

        // Find range.
        IdxTuple numElemsTuple = get_slice_range(first, last);

        // Visit points in slice.
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

        return numElemsTuple.product();
    }
    idx_t YkGridBase::set_elements_in_slice(const void* buffer_ptr,
                                            const Indices& first_indices,
                                            const Indices& last_indices) {
        if (!is_storage_allocated())
            return 0;
        checkIndices(first_indices, "set_elements_in_slice", true, false);
        checkIndices(last_indices, "set_elements_in_slice", true, false);

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

        return numElemsTuple.product();
    }

} // namespace.

