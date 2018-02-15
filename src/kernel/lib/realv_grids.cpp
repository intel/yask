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

// Implement methods for RealVecGridBase.

#include "yask.hpp"
using namespace std;

namespace yask {

    // APIs to get info from vars.
#define GET_GRID_API(api_name, expr, step_ok, domain_ok, misc_ok)       \
    idx_t YkGridBase::api_name(const string& dim) const {               \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        int posn = get_dim_posn(dim, true, #api_name);                  \
        return expr;                                                    \
    }                                                                   \
    idx_t YkGridBase::api_name(int posn) const {                        \
        return expr;                                                    \
    }
    GET_GRID_API(get_rank_domain_size, _domains[posn], false, true, false)
    GET_GRID_API(get_left_pad_size, _left_pads[posn], false, true, false)
    GET_GRID_API(get_right_pad_size, _right_pads[posn], false, true, false)
    GET_GRID_API(get_pad_size, _left_pads[posn], false, true, false)
    GET_GRID_API(get_left_halo_size, _left_halos[posn], false, true, false)
    GET_GRID_API(get_right_halo_size, _right_halos[posn], false, true, false)
    GET_GRID_API(get_halo_size, _left_halos[posn], false, true, false)
    GET_GRID_API(get_first_rank_halo_index, _offsets[posn] - _left_halos[posn], false, false, true)
    GET_GRID_API(get_last_rank_halo_index, _offsets[posn] + _domains[posn] + _right_halos[posn] - 1, false, false, true)
    GET_GRID_API(get_first_misc_index, _offsets[posn], false, false, true)
    GET_GRID_API(get_last_misc_index, _offsets[posn] + _domains[posn] - 1, false, false, true)
    GET_GRID_API(get_first_rank_domain_index, _offsets[posn], false, true, false)
    GET_GRID_API(get_last_rank_domain_index, _offsets[posn] + _domains[posn] - 1, false, true, false)
    GET_GRID_API(get_first_rank_alloc_index, _offsets[posn] - _left_pads[posn], false, true, false)
    GET_GRID_API(get_last_rank_alloc_index, _offsets[posn] - _left_pads[posn] + _allocs[posn] - 1, false, true, false)
    GET_GRID_API(get_left_extra_pad_size, _left_pads[posn] - _left_halos[posn], false, true, false)
    GET_GRID_API(get_right_extra_pad_size, _right_pads[posn] - _right_halos[posn], false, true, false)
    GET_GRID_API(get_extra_pad_size, _left_pads[posn] - _left_halos[posn], false, true, false)
    GET_GRID_API(get_alloc_size, _allocs[posn], true, true, true)
    GET_GRID_API(_get_offset, _offsets[posn], true, true, true)
    GET_GRID_API(_get_first_alloc_index, _offsets[posn] - _left_pads[posn], true, true, true)
    GET_GRID_API(_get_last_alloc_index, _offsets[posn] - _left_pads[posn] + _allocs[posn] - 1, true, true, true)
#undef GET_GRID_API
    
    // APIs to set vars.
#define COMMA ,
#define SET_GRID_API(api_name, expr, step_ok, domain_ok, misc_ok)       \
    void YkGridBase::api_name(const string& dim, idx_t n) {             \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        int posn = get_dim_posn(dim, true, #api_name);                  \
        expr;                                                           \
    }                                                                   \
    void YkGridBase::api_name(int posn, idx_t n) {                      \
        int dim = posn;                                                 \
        expr;                                                           \
    }
    SET_GRID_API(set_alloc_size, _set_domain_size(dim, n), true, false, true)
    SET_GRID_API(_set_domain_size, _domains[posn] = n; resize(), true, true, true)
    SET_GRID_API(set_left_halo_size, _left_halos[posn] = n; _set_left_pad_size(dim, _left_pads[posn]),
                 false, true, false)
    SET_GRID_API(set_right_halo_size, _right_halos[posn] = n; _set_right_pad_size(dim, _right_pads[posn]),
                 false, true, false)
    SET_GRID_API(set_halo_size, set_left_halo_size(dim, n); set_right_halo_size(dim, n),
                 false, true, false)
    SET_GRID_API(set_left_min_pad_size,
                 if (!get_raw_storage_buffer() && n > _left_pads[posn])
                     _set_left_pad_size(dim, n),
                 false, true, false)
    SET_GRID_API(set_right_min_pad_size,
                 if (!get_raw_storage_buffer() && n > _right_pads[posn])
                     _set_right_pad_size(dim, n),
                 false, true, false)
    SET_GRID_API(set_min_pad_size,
                 if (!get_raw_storage_buffer() && n > _left_pads[posn])
                     _set_left_pad_size(dim, n);
                 if (!get_raw_storage_buffer() && n > _right_pads[posn])
                     _set_right_pad_size(dim, n),
                 false, true, false)
    SET_GRID_API(set_left_extra_pad_size, set_left_min_pad_size(dim, _left_halos[posn] + n), false, true, false)
    SET_GRID_API(set_right_extra_pad_size, set_right_min_pad_size(dim, _right_halos[posn] + n), false, true, false)
    SET_GRID_API(set_extra_pad_size, set_left_extra_pad_size(dim, n);
                 set_right_extra_pad_size(dim, n), false, true, false)
    SET_GRID_API(set_first_misc_index, _offsets[posn] = n, false, false, true)
    SET_GRID_API(_set_left_pad_size, _left_pads[posn] = std::max(n, _left_halos[posn]);
                 resize(), true, true, true)
    SET_GRID_API(_set_right_pad_size, _right_pads[posn] = std::max(n, _right_halos[posn]);
                 resize(), true, true, true)
    SET_GRID_API(_set_offset, _offsets[posn] = n, true, true, true)
#undef COMMA
#undef SET_GRID_API
    
    // Convenience function to format indices like
    // "x=5, y=3".
    std::string YkGridBase::makeIndexString(const Indices& idxs,
                                            std::string separator,
                                            std::string infix,
                                            std::string prefix,
                                            std::string suffix) const {
        IdxTuple tmp = get_allocs(); // get dims.
        idxs.setTupleVals(tmp);      // set vals from idxs.
        return tmp.makeDimValStr(separator, infix, prefix, suffix);
    }

    // Halo-exchange flag accessors.
    bool YkGridBase::is_dirty(idx_t step_idx) const {
        if (_dirty_steps.size() == 0)
            const_cast<YkGridBase*>(this)->resize();
        if (_has_step_dim)
            step_idx = _wrap_step(step_idx);
        else
            step_idx = 0;
        return _dirty_steps[step_idx];
    }
    void YkGridBase::set_dirty(bool dirty, idx_t step_idx) {
        if (_dirty_steps.size() == 0)
            resize();
        if (_has_step_dim)
            step_idx = _wrap_step(step_idx);
        else
            step_idx = 0;
        _dirty_steps[step_idx] = dirty;
    }
    void YkGridBase::set_dirty_all(bool dirty) {
        if (_dirty_steps.size() == 0)
            resize();
        for (auto i : _dirty_steps)
            i = dirty;
    }
    
    // Lookup position by dim name.
    // Return -1 or die if not found, depending on flag.
    int YkGridBase::get_dim_posn(const std::string& dim,
                                 bool die_on_failure,
                                 const std::string& die_msg) const {
        auto& dims = _ggb->get_dims();
        int posn = dims.lookup_posn(dim);
        if (posn < 0 && die_on_failure) {
            THROW_YASK_EXCEPTION("Error: " << die_msg << ": dimension '" <<
                                 dim << "' not found in " << make_info_string());
        }
        return posn;
    }
        
    // Resizes the underlying generic grid.
    // Modifies _pads and _allocs.
    // Fails if mem different and already alloc'd.
    void YkGridBase::resize() {
        
        // Original size.
        auto p = get_raw_storage_buffer();
        IdxTuple old_allocs = get_allocs();

        // Round up padding.
        for (int i = 0; i < get_num_dims(); i++) {
            _left_pads[i] = ROUND_UP(_left_pads[i], _vec_lens[i]);
            _right_pads[i] = ROUND_UP(_right_pads[i], _vec_lens[i]);
            _vec_left_pads[i] = _left_pads[i] / _vec_lens[i];
        }
        
        // New allocation in each dim.
        IdxTuple new_allocs(old_allocs);
        for (int i = 0; i < get_num_dims(); i++)
            new_allocs[i] = ROUND_UP(_left_pads[i] + _domains[i] + _right_pads[i], _vec_lens[i]);

        // Attempt to change alloc with existing storage?
        if (p && old_allocs != new_allocs) {
            THROW_YASK_EXCEPTION("Error: attempt to change allocation size of grid '" <<
                get_name() << "' from " << 
                makeIndexString(old_allocs, " * ") << " to " <<
                makeIndexString(new_allocs, " * ") <<
                " after storage has been allocated");
        }

        // Do the resize.
        _allocs = new_allocs;
        size_t new_dirty = 1;      // default if no step dim.
        for (int i = 0; i < get_num_dims(); i++) {
            _vec_allocs[i] = _allocs[i] / _vec_lens[i];
            _ggb->set_dim_size(i, _vec_allocs[i]);

            // Steps.
            if (get_dim_name(i) == _dims->_step_dim)
                new_dirty = _allocs[i];
        }

        // Resize dirty flags, too.
        size_t old_dirty = _dirty_steps.size();
        if (old_dirty != new_dirty)
            _dirty_steps.assign(new_dirty, true); // set all as dirty.

        if (old_allocs != new_allocs || old_dirty != new_dirty) {
            TRACE_MSG0(get_ostr(), "grid '" << get_name() << "' resized from " <<
                       makeIndexString(old_allocs, " * ") << " to " <<
                       makeIndexString(new_allocs, " * ") << " with " <<
                       _dirty_steps.size() << " dirty flags");
        }
    }
    
    // Check whether dim is used and of allowed type.
    void YkGridBase::checkDimType(const std::string& dim,
                                  const std::string& fn_name,
                                  bool step_ok,
                                  bool domain_ok,
                                  bool misc_ok) const {
        if (!is_dim_used(dim))
            THROW_YASK_EXCEPTION("Error in " << fn_name << "(): dimension '" <<
                                 dim << "' not found in " << make_info_string());
        _dims->checkDimType(dim, fn_name, step_ok, domain_ok, misc_ok);
    }
    
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
            if (_allocs[i] != op->_allocs[i])
                return false;
            if (_domains[i] != op->_domains[i])
                return false;
            if (_left_pads[i] != op->_left_pads[i])
                return false;
            if (_right_pads[i] != op->_right_pads[i])
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

        // NB: requirements to successful share_storage() is not as strict as
        // is_storage_layout_identical(). See note on pad & halo below and API docs.
        for (int i = 0; i < get_num_dims(); i++) {
            auto dname = get_dim_name(i);

            // Same dims?
            if (sp->get_num_dims() != get_num_dims() ||
                sp->get_dim_name(i) != dname)
                THROW_YASK_EXCEPTION("Error: share_storage() called with incompatible grids: " <<
                                     make_info_string() << " and " << sp->make_info_string());

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
                auto tdom = get_rank_domain_size(dname);
                auto sdom = sp->get_rank_domain_size(dname);
                if (tdom != sdom) {
                    THROW_YASK_EXCEPTION("Error: attempt to share storage from grid '" << sp->get_name() <<
                                         "' of domain-size " << sdom << " with grid '" << get_name() <<
                                         "' of domain-size " << tdom << " in '" << dname << "' dim");
                }

                // Halo and pad sizes don't have to be the same.
                // Requirement is that halo of target fits inside of pad of source.
                auto thalo = get_left_halo_size(dname);
                auto spad = sp->get_left_pad_size(dname);
                if (thalo > spad) {
                    THROW_YASK_EXCEPTION("Error: attempt to share storage from grid '" << sp->get_name() <<
                                         "' of left padding-size " << spad <<
                                         ", which is insufficient for grid '" << get_name() <<
                                         "' of left halo-size " << thalo << " in '" << dname << "' dim");
                }
                thalo = get_right_halo_size(dname);
                spad = sp->get_right_pad_size(dname);
                if (thalo > spad) {
                    THROW_YASK_EXCEPTION("Error: attempt to share storage from grid '" << sp->get_name() <<
                                         "' of right padding-size " << spad <<
                                         ", which is insufficient for grid '" << get_name() <<
                                         "' of right halo-size " << thalo << " in '" << dname << "' dim");
                }
            }

            // Check folding.
            if (_vec_lens[i] != sp->_vec_lens[i]) {
                THROW_YASK_EXCEPTION("Error: attempt to share storage from grid '" << sp->get_name() <<
                                     "' of fold-length " << sp->_vec_lens[i] << " with grid '" << get_name() <<
                                     "' of fold-length " << _vec_lens[i] << " in '" << dname << "' dim");
            }
        }

        // Copy pad sizes.
        for (int i = 0; i < get_num_dims(); i++) {
            auto dname = get_dim_name(i);
            bool is_domain = _dims->_domain_dims.lookup(dname) != 0;
            if (is_domain) {
                auto spad = sp->get_left_pad_size(dname);
                _set_left_pad_size(dname, spad);
                spad = sp->get_right_pad_size(dname);
                _set_right_pad_size(dname, spad);
            }
        }
        
        // Copy data.
        release_storage();
        if (!share_data(sp.get(), true)) {
            THROW_YASK_EXCEPTION("Error: unexpected failure in data sharing");
        }
    }

    // Check for equality.
    // Return number of mismatches greater than epsilon.
    idx_t YkGridBase::compare(const YkGridBase* ref,
                              real_t epsilon,
                              int maxPrint,
                              std::ostream& os) const {
        if (!ref) {
            os << "** mismatch: no reference grid.\n";
            return get_num_storage_elements();
        }

        // Dims & sizes same?
        if (!_ggb->are_dims_and_sizes_same(*ref->_ggb)) {
            os << "** mismatch due to incompatible grids: " <<
                make_info_string() << " and " << ref->make_info_string() << ".\n";
            return get_num_storage_elements();
        }
        
        // Quick check for errors, assuming same layout.
        // TODO: check layout.
        idx_t errs = _ggb->count_diffs(ref->_ggb, epsilon);
        TRACE_MSG0(get_ostr(), "count_diffs() returned " << errs);
        if (!errs)
            return 0;
        
        // Run detailed comparison if any errors found.
        errs = 0;
        auto allocs = get_allocs();

        // This will loop over the entire allocation.
        // Indices of 'pt' will be relative to allocation.
        allocs.visitAllPoints
            ([&](const IdxTuple& pt, size_t idx) {

                // Adjust alloc indices to overall indices.
                IdxTuple opt(pt);
                bool ok = true;
                for (int i = 0; i < pt.getNumDims(); i++) {
                    auto val = pt.getVal(i);
                    opt[i] = _offsets[i] - _left_pads[i] + val;

                    // Don't compare points in the extra padding area.
                    auto& dname = pt.getDimName(i);
                    if (_dims->_domain_dims.lookup(dname)) {
                        auto halo_sz = get_halo_size(dname);
                        auto first_ok = get_first_rank_domain_index(dname) - halo_sz;
                        auto last_ok = get_last_rank_domain_index(dname) + halo_sz;
                        if (opt[i] < first_ok || opt[i] > last_ok)
                            ok = false;
                    }
                }
                if (!ok)
                    return true; // stop processing this point, but keep going.

                idx_t asi = get_alloc_step_index(pt[Indices::step_posn]);
                auto te = readElem(opt, asi, __LINE__);
                auto re = ref->readElem(opt, asi, __LINE__);
                if (!within_tolerance(te, re, epsilon)) {
                    errs++;
                    if (errs < maxPrint) {
                        os << "** mismatch at " << get_name() <<
                            "(" << opt.makeDimValStr() << "): " <<
                            te << " != " << re << std::endl;
                    }
                    else if (errs == maxPrint)
                        os << "** Additional errors not printed." << std::endl;
                    else {
                        // errs > maxPrint.
                        return false; // stop visits.
                    }
                }
                return true;    // keep visiting.
            });
        TRACE_MSG0(get_ostr(), "detailed compare returned " << errs);
        return errs;
    }

    // Make sure indices are in range.
    // Side-effect: If fixed_indices is not NULL, set them to in-range if out-of-range.
    bool YkGridBase::checkIndices(const Indices& indices,
                                  const string& fn,
                                  bool strict_indices, // die if out-of-range.
                                  bool normalize,      // div by vec lens.
                                  Indices* fixed_indices) const {
        auto n = get_num_dims();
        if (indices.getNumDims() != n) {
            THROW_YASK_EXCEPTION("Error: '" << fn << "' called with " << indices.getNumDims() <<
                                 " indices instead of " << n);
        }
        if (fixed_indices)
            *fixed_indices = indices;
        bool ok = true;
        for (int i = 0; i < n; i++) {
            idx_t idx = indices[i];
            auto& dname = get_dim_name(i);
            bool ok = false;

            // Any step index is ok because it wraps around.
            // TODO: check that it's < magic added value in wrap_index().
            if (_has_step_dim && i == Indices::step_posn)
                ok = true;

            // Within first..last indices?
            else {
                auto first_ok = _get_first_alloc_index(i);
                auto last_ok = _get_last_alloc_index(i);
                if (idx >= first_ok && idx <= last_ok)
                    ok = true;

                // Handle outliers.
                if (!ok) {
                    if (strict_indices) {
                        THROW_YASK_EXCEPTION("Error: " << fn << ": index in dim '" << dname <<
                                             "' is " << idx << ", which is not in [" << first_ok <<
                                             "..." << last_ok << "]");
                    }
                    if (fixed_indices) {
                        if (idx < first_ok)
                            (*fixed_indices)[i] = first_ok;
                        if (idx > last_ok)
                            (*fixed_indices)[i] = last_ok;
                    }
                    ok = false;
                }
            }

            // Normalize?
            if (fixed_indices && normalize) {
                (*fixed_indices)[i] -= _offsets[i];
                (*fixed_indices)[i] = idiv_flr((*fixed_indices)[i], _vec_lens[i]);
            }
        }
        return ok;
    }

    // Set dirty flags between indices.
    void YkGridBase::set_dirty_in_slice(const Indices& first_indices,
                                        const Indices& last_indices) {
        if (_has_step_dim) {
            for (idx_t i = first_indices[Indices::step_posn];
                 i <= last_indices[Indices::step_posn]; i++)
                set_dirty(true, i);
        } else
            set_dirty(true, 0);
    }
     
    // Make tuple needed for slicing.
    IdxTuple YkGridBase::get_slice_range(const Indices& first_indices,
                                         const Indices& last_indices) const {
        // Find ranges.
        Indices numElems = last_indices.addConst(1).subElements(first_indices);
        IdxTuple numElemsTuple = get_allocs();
        numElems.setTupleVals(numElemsTuple);
        numElemsTuple.setFirstInner(_is_col_major);

        return numElemsTuple;
    }
    
    // API get/set.
    double YkGridBase::get_element(const Indices& indices) const {
        if (!is_storage_allocated()) {
            THROW_YASK_EXCEPTION("Error: call to 'get_element' with no data allocated for grid '" <<
                                 get_name() << "'");
        }
        checkIndices(indices, "get_element", true, false);
        idx_t asi = get_alloc_step_index(indices[Indices::step_posn]);
        real_t val = readElem(indices, asi, __LINE__);
        return double(val);
    }
    idx_t YkGridBase::set_element(double val,
                                  const Indices& indices,
                                  bool strict_indices) {
        idx_t nup = 0;
        if (get_raw_storage_buffer() &&
            checkIndices(indices, "set_element", strict_indices, false)) {
            idx_t asi = get_alloc_step_index(indices[Indices::step_posn]);
            writeElem(real_t(val), indices, asi, __LINE__);
            nup++;

            // Set appropriate dirty flag.
            set_dirty_in_slice(indices, indices);
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
                idx_t asi = get_alloc_step_index(pt[Indices::step_posn]);
                
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
                Indices pt = first_indices.addElements(ofs);

                // TODO: move this outside of loop for const step index.
                idx_t asi = get_alloc_step_index(pt[Indices::step_posn]);

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
                idx_t asi = get_alloc_step_index(pt[Indices::step_posn]);

                real_t val = ((real_t*)buffer_ptr)[idx];
                writeElem(val, pt, asi, __LINE__);
                return true;    // keep going.
            });

        // Set appropriate dirty flag(s).
        set_dirty_in_slice(first_indices, last_indices);

        return numElemsTuple.product();
    }

    // Print one element like
    // "message: mygrid[x=4, y=7] = 3.14 at line 35".
    void YkGridBase::printElem(const std::string& msg,
                               const Indices& idxs,
                               real_t e,
                               int line,
                               bool newline) const {
        ostream& os = _ggb->get_ostr();
        if (msg.length())
            os << msg << ": ";
        os << get_name() << "[" <<
            makeIndexString(idxs) << "] = " << e;
        if (line)
            os << " at line " << line;
        if (newline)
            os << std::endl << std::flush;
    }

    // Print one vector.
    // Indices must be normalized and rank-relative.
    void YkGridBase::printVecNorm(const std::string& msg,
                                          const Indices& idxs,
                                          const real_vec_t& val,
                                          int line,
                                          bool newline) const {

        // Convert to elem indices.
        Indices eidxs = idxs.mulElements(_vec_lens);

        // Add offsets, i.e., convert to overall indices.
        eidxs = eidxs.addElements(_offsets);

        IdxTuple idxs2 = get_allocs(); // get dims.
        eidxs.setTupleVals(idxs2);      // set vals from eidxs.

        // Visit every point in fold.
        IdxTuple folds = _dims->_fold_pts;
        folds.visitAllPoints([&](const IdxTuple& fofs,
                                 size_t idx) {
                
                // Get element from vec val.
                real_t ev = val[idx];
                
                // Add fold offsets to elem indices for printing.
                IdxTuple pt2 = idxs2.addElements(fofs, false);
                Indices pt3(pt2);
                
                printElem(msg, pt3, ev, line, newline);
                return true; // keep visiting.
            });
    }

    
} // namespace.

