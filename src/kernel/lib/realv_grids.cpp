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

    // Ctor.
    YkGridBase::YkGridBase(GenericGridBase* ggb,
                           size_t ndims,
                           DimsPtr dims) :
    _ggb(ggb), _dims(dims) {

        assert(ggb);
        assert(dims.get());
        
        // Init indices.
        int n = int(ndims);
        _domains.setFromConst(0, n);
        _left_pads.setFromConst(0, n);
        _right_pads.setFromConst(0, n);
        _left_halos.setFromConst(0, n);
        _right_halos.setFromConst(0, n);
        _left_wf_exts.setFromConst(0, n);
        _right_wf_exts.setFromConst(0, n);
        _offsets.setFromConst(0, n);
        _local_offsets.setFromConst(0, n);
        _vec_lens.setFromConst(1, n);
        _allocs.setFromConst(1, n);
        _vec_left_pads.setFromConst(1, n);
        _vec_allocs.setFromConst(1, n);
        _vec_local_offsets.setFromConst(0, n);
    }
    
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
        set_dirty_using_alloc_index(dirty, step_idx);
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

        // Check settings.
        for (int i = 0; i < get_num_dims(); i++) {
            if (_left_halos[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative left halo in grid '" << get_name() << "'");
            if (_right_halos[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative right halo in grid '" << get_name() << "'");
            if (_left_wf_exts[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative left wave-front ext in grid '" << get_name() << "'");
            if (_right_wf_exts[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative right wave-front ext in grid '" << get_name() << "'");
            if (_left_pads[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative left padding in grid '" << get_name() << "'");
            if (_right_pads[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative right padding in grid '" << get_name() << "'");
        }
        
        // Increase padding as needed.
        // _left_pads contains actual padding and is always rounded up to vec len.
        // _right_pads contains requested min padding; actual padding is calculated on-the-fly.
        // TODO: maintain requested and actual padding for left and right.
        Indices left_pads2 = getReqdPad(_left_halos, _left_wf_exts);
        Indices right_pads2 = getReqdPad(_right_halos, _right_wf_exts);
        for (int i = 0; i < get_num_dims(); i++) {

            // Get max of existing pad and reqd pad.
            left_pads2[i] = max(_left_pads[i], left_pads2[i]);
            right_pads2[i] = max(_right_pads[i], right_pads2[i]);

            // Round left pad up to vec len and store final setting.
            // Keep final padding for left.
            left_pads2[i] = ROUND_UP(left_pads2[i], _vec_lens[i]);
            _left_pads[i] = left_pads2[i];
            _vec_left_pads[i] = left_pads2[i] / _vec_lens[i];
        }
        
        // New allocation in each dim.
        IdxTuple new_allocs(old_allocs);
        for (int i = 0; i < get_num_dims(); i++)
            new_allocs[i] = ROUND_UP(_left_pads[i] + _domains[i], _vec_lens[i]) +
                ROUND_UP(right_pads2[i], _vec_lens[i]);

        // Attempt to change alloc with existing storage?
        if (p && old_allocs != new_allocs) {
            THROW_YASK_EXCEPTION("Error: attempt to change allocation size of grid '" <<
                get_name() << "' from " << 
                makeIndexString(old_allocs, " * ") << " to " <<
                makeIndexString(new_allocs, " * ") <<
                " after storage has been allocated");
        }

        // Do the resize and calculate number of dirty bits needed.
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

        // Report changes in TRACE mode.
        if (old_allocs != new_allocs || old_dirty != new_dirty) {
            Indices first_allocs = _offsets.subElements(_left_pads);
            Indices last_allocs = first_allocs.addElements(_allocs).subConst(1);
            TRACE_MSG0(get_ostr(), "grid '" << get_name() << "' resized from " <<
                       makeIndexString(old_allocs, " * ") <<
                       " to " << makeIndexString(new_allocs, " * ") <<
                       " at " << makeIndexString(first_allocs) <<
                       " ... " << makeIndexString(last_allocs) <<
                       " with " << _dirty_steps.size() << " dirty flag(s)");
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

                idx_t asi = get_alloc_step_index(pt);
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
        bool all_ok = true;
        auto n = get_num_dims();
        if (indices.getNumDims() != n) {
            THROW_YASK_EXCEPTION("Error: '" << fn << "' called with " << indices.getNumDims() <<
                                 " indices instead of " << n);
        }
        if (fixed_indices)
            *fixed_indices = indices;
        for (int i = 0; i < n; i++) {
            idx_t idx = indices[i];
            bool ok = false;
            auto& dname = get_dim_name(i);

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
                                             "' is " << idx << ", which is not in allocated range [" <<
                                             first_ok << "..." << last_ok << "] of grid '" <<
                                             get_name() << "'");
                    }

                    // Update the output indices.
                    if (fixed_indices) {
                        if (idx < first_ok)
                            (*fixed_indices)[i] = first_ok;
                        if (idx > last_ok)
                            (*fixed_indices)[i] = last_ok;
                    }
                }
            }
            if (!ok)
                all_ok = false;

            // Normalize?
            if (fixed_indices && normalize) {
                (*fixed_indices)[i] -= _offsets[i];
                (*fixed_indices)[i] = idiv_flr((*fixed_indices)[i], _vec_lens[i]);
            }
        }
        return all_ok;
    }

    // Set dirty flags between indices.
    void YkGridBase::set_dirty_in_slice(const Indices& first_indices,
                                        const Indices& last_indices) {
        if (_has_step_dim) {
            for (idx_t i = first_indices[Indices::step_posn];
                 i <= last_indices[Indices::step_posn]; i++)
                set_dirty(true, i);
        } else
            set_dirty_using_alloc_index(true, 0);
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

