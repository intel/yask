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

// Implement methods for RealVecGridBase.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Ctor.
    // Important: '*ggb' is NOT yet constructed.
    YkGridBase::YkGridBase(KernelStateBase& stateb,
                           GenericGridBase* ggb,
                           const GridDimNames& dimNames) :
        KernelStateBase(stateb),
        _ggb(ggb) {
        STATE_VARS(this);
        assert(ggb);

        // Init indices.
        int n = int(dimNames.size());
        _domains.setFromConst(0, n);
        _req_left_pads.setFromConst(0, n);
        _req_right_pads.setFromConst(0, n);
        _actl_left_pads.setFromConst(0, n);
        _actl_right_pads.setFromConst(0, n);
        _left_halos.setFromConst(0, n);
        _right_halos.setFromConst(0, n);
        _left_wf_exts.setFromConst(0, n);
        _right_wf_exts.setFromConst(0, n);
        _rank_offsets.setFromConst(0, n);
        _local_offsets.setFromConst(0, n);
        _vec_lens.setFromConst(1, n);
        _allocs.setFromConst(1, n);
        _vec_left_pads.setFromConst(1, n);
        _vec_allocs.setFromConst(1, n);
        _vec_local_offsets.setFromConst(0, n);

        // Set masks.
        for (int i = 0; i < dimNames.size(); i++) {
            idx_t mbit = 1LL << i;
            auto& dname = dimNames[i];
            if (dname == step_dim)
                _step_dim_mask |= mbit;
            else if (domain_dims.lookup(dname))
                _domain_dim_mask |= mbit;
            else
                _misc_dim_mask |= mbit;
        }
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
            THROW_YASK_EXCEPTION("Error: " + die_msg + ": dimension '" +
                                 dim + "' not found in " + make_info_string());
        }
        return posn;
    }

    // Determine required padding from halos.
    // Does not include user-specified min padding or
    // final rounding for left pad.
    Indices YkGridBase::getReqdPad(const Indices& halos, const Indices& wf_exts) const {
        STATE_VARS(this);

        // Start with halos plus WF exts.
        Indices mp = halos.addElements(wf_exts);

        // For scratch grids, halo area must be written to.  Halo is sum
        // of dependent's write halo and depender's read halo, but these
        // two components are not stored individually.  Write halo will
        // be expanded to full vec len during computation, requiring
        // load from read halo beyond full vec len.  Worst case is when
        // write halo is one and rest is read halo.  So if there is a
        // halo and/or wf-ext, padding should be that plus all but one
        // element of a vector. In addition, this vec-len should be the
        // global one, not the one for this grid to handle the case where
        // this grid is not vectorized.
        for (int i = 0; i < get_num_dims(); i++) {
            if (mp[i] >= 1) {
                auto& dname = get_dim_name(i);
                auto* p = dims->_fold_pts.lookup(dname);
                if (p) {
                    assert (*p >= 1);
                    mp[i] += *p - 1;
                }
            }
        }
        return mp;
    }

    // Resizes the underlying generic grid.
    // Modifies _pads and _allocs.
    // Fails if mem different and already alloc'd.
    void YkGridBase::resize() {
        STATE_VARS(this);

        // Original size.
        auto p = get_raw_storage_buffer();
        IdxTuple old_allocs = get_allocs();

        // Check settings.
        for (int i = 0; i < get_num_dims(); i++) {
            if (_left_halos[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative left halo in grid '" + get_name() + "'");
            if (_right_halos[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative right halo in grid '" + get_name() + "'");
            if (_left_wf_exts[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative left wave-front ext in grid '" + get_name() + "'");
            if (_right_wf_exts[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative right wave-front ext in grid '" + get_name() + "'");
            if (_req_left_pads[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative left padding in grid '" + get_name() + "'");
            if (_req_right_pads[i] < 0)
                THROW_YASK_EXCEPTION("Error: negative right padding in grid '" + get_name() + "'");
        }

        // Increase padding as needed and calculate new allocs.
        Indices new_left_pads = getReqdPad(_left_halos, _left_wf_exts);
        Indices new_right_pads = getReqdPad(_right_halos, _right_wf_exts);
        IdxTuple new_allocs(old_allocs);
        for (int i = 0; i < get_num_dims(); i++) {
            idx_t mbit = 1LL << i;

            // New allocation in each dim.
            new_allocs[i] = _domains[i];

            // Adjust padding only for domain dims.
            if (_domain_dim_mask & mbit) {

                // Get max of existing pad & new required pad.
                new_left_pads[i] = max(new_left_pads[i], _actl_left_pads[i]);
                new_right_pads[i] = max(new_right_pads[i], _actl_right_pads[i]);

                // If storage not yet allocated, also increase to requested pad.
                // This will avoid throwing an exception due to unneeded
                // extra padding after allocation.
                if (!p) {
                    new_left_pads[i] = max(new_left_pads[i], _req_left_pads[i]);
                    new_right_pads[i] = max(new_right_pads[i], _req_right_pads[i]);
                }

                // Round left pad up to vec len.
                new_left_pads[i] = ROUND_UP(new_left_pads[i], _vec_lens[i]);

                // Round domain + right pad up to vec len by extending right pad.
                idx_t dprp = ROUND_UP(_domains[i] + new_right_pads[i], _vec_lens[i]);
                new_right_pads[i] = dprp - _domains[i];

                // New allocation in each dim.
                new_allocs[i] += new_left_pads[i] + new_right_pads[i];

                // Make inner dim an odd number of vecs.
                // This reportedly helps avoid some uarch aliasing.
                if (!p && get_dim_name(i) == inner_dim &&
                    (new_allocs[i] / _vec_lens[i]) % 2 == 0) {
                    new_right_pads[i] += _vec_lens[i];
                    new_allocs[i] += _vec_lens[i];
                }
                assert(new_allocs[i] == new_left_pads[i] + _domains[i] + new_right_pads[i]);

                // Since the left pad and domain + right pad were rounded up,
                // the sum should also be a vec mult.
                assert(new_allocs[i] % _vec_lens[i] == 0);
            }
        }

        // Attempt to change alloc with existing storage?
        // TODO: restore the values before the API that called
        // resize() on failure.
        if (p && old_allocs != new_allocs) {
            THROW_YASK_EXCEPTION("Error: attempt to change allocation size of grid '" +
                get_name() + "' from " +
                makeIndexString(old_allocs, " * ") + " to " +
                makeIndexString(new_allocs, " * ") +
                " after storage has been allocated");
        }

        // Do the resize and calculate number of dirty bits needed.
        _allocs = new_allocs;
        _actl_left_pads = new_left_pads;
        _actl_right_pads = new_right_pads;
        size_t new_dirty = 1;      // default if no step dim.
        for (int i = 0; i < get_num_dims(); i++) {
            idx_t mbit = 1LL << i;

            // Calc vec-len values.
            _vec_left_pads[i] = new_left_pads[i] / _vec_lens[i];
            _vec_allocs[i] = _allocs[i] / _vec_lens[i];

            // Actual resize of underlying grid.
            _ggb->set_dim_size(i, _vec_allocs[i]);

            // Number of dirty bits is number of steps.
            if (_step_dim_mask & mbit)
                new_dirty = _allocs[i];
        }

        // Resize dirty flags, too.
        size_t old_dirty = _dirty_steps.size();
        if (old_dirty != new_dirty)
            _dirty_steps.assign(new_dirty, true); // set all as dirty.

        // Report changes in TRACE mode.
        if (old_allocs != new_allocs || old_dirty != new_dirty) {
            Indices first_allocs = _rank_offsets.subElements(_actl_left_pads);
            Indices end_allocs = first_allocs.addElements(_allocs);
            TRACE_MSG("grid '" << get_name() << "' resized from " <<
                       makeIndexString(old_allocs, " * ") << " to " <<
                       makeIndexString(new_allocs, " * ") << " at [" <<
                       makeIndexString(first_allocs) << " ... " << 
                       makeIndexString(end_allocs) << ") with left-halos " <<
                       makeIndexString(_left_halos) << ", right-halos " <<
                       makeIndexString(_right_halos) << ", left-wf-exts " <<
                       makeIndexString(_left_wf_exts) << ", right-wf-exts " <<
                       makeIndexString(_right_wf_exts) << ", and " <<
                       _dirty_steps.size() << " dirty flag(s)");
        }
    }

    // Check whether dim is used and of allowed type.
    void YkGridBase::checkDimType(const std::string& dim,
                                  const std::string& fn_name,
                                  bool step_ok,
                                  bool domain_ok,
                                  bool misc_ok) const {
        STATE_VARS(this);
        if (!is_dim_used(dim))
            THROW_YASK_EXCEPTION("Error in " + fn_name + "(): dimension '" +
                                 dim + "' not found in " + make_info_string());
        dims->checkDimType(dim, fn_name, step_ok, domain_ok, misc_ok);
    }

    // Check for equality.
    // Return number of mismatches greater than epsilon.
    idx_t YkGridBase::compare(const YkGridBase* ref,
                              real_t epsilon,
                              int maxPrint) const {
        STATE_VARS(this);
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

        // Quick check for errors, assuming same layout and
        // same values in extra-padding area.
        // TODO: check layout.
        idx_t errs = _ggb->count_diffs(ref->_ggb, epsilon);
        TRACE_MSG("count_diffs() returned " << errs);
        if (!errs)
            return 0;

        // Run detailed comparison if any errors found.
        errs = 0;
        auto allocs = get_allocs();

        // This will loop over the entire allocation.
        // We use this as a handy way to get offsets,
        // but not all will be used.
        allocs.visitAllPoints
            ([&](const IdxTuple& pt, size_t idx) {

                // Adjust alloc indices to overall indices.
                IdxTuple opt(pt);
                bool ok = true;
                for (int i = 0; ok && i < pt.getNumDims(); i++) {
                    auto val = pt.getVal(i);
                    idx_t mbit = 1LL << i;

                    // Convert to API index.
                    opt[i] = val;
                    if (!(_step_dim_mask & mbit))
                        opt[i] += _rank_offsets[i] + _local_offsets[i];

                    // Don't compare points outside the domain.
                    // TODO: check points in halo.
                    auto& dname = pt.getDimName(i);
                    if (domain_dims.lookup(dname)) {
                        auto first_ok = get_first_rank_domain_index(dname);
                        auto last_ok = get_last_rank_domain_index(dname);
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
        TRACE_MSG("detailed compare returned " << errs);
        return errs;
    }

    // Make sure indices are in range.
    // Side-effect: If clipped_indices is not NULL, set them to in-range if out-of-range.
    bool YkGridBase::checkIndices(const Indices& indices,
                                  const string& fn,
                                  bool strict_indices, // die if out-of-range.
                                  bool normalize,      // div by vec lens.
                                  Indices* clipped_indices) const {
        bool all_ok = true;
        auto n = get_num_dims();
        if (indices.getNumDims() != n) {
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: '" << fn << "' called with " <<
                                            indices.getNumDims() <<
                                            " indices instead of " << n);
        }
        if (clipped_indices)
            *clipped_indices = indices;
        for (int i = 0; i < n; i++) {
            idx_t mbit = 1LL << i;
            idx_t idx = indices[i];
            bool ok = false;
            auto& dname = get_dim_name(i);

            // Any step index is ok because it wraps around.
            // TODO: check that it's < magic added value in wrap_index().
            if (_step_dim_mask & mbit)
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
                        FORMAT_AND_THROW_YASK_EXCEPTION("Error: " + fn + ": index in dim '" + dname +
                                                        "' is " << idx << ", which is not in allocated range [" <<
                                                        first_ok << "..." << last_ok << "] of grid '" +
                                                        get_name() + "'");
                    }

                    // Update the output indices.
                    if (clipped_indices) {
                        if (idx < first_ok)
                            (*clipped_indices)[i] = first_ok;
                        if (idx > last_ok)
                            (*clipped_indices)[i] = last_ok;
                    }
                }
            }
            if (!ok)
                all_ok = false;

            // Normalize?
            if (clipped_indices && normalize) {
                if (_domain_dim_mask & mbit) {
                    (*clipped_indices)[i] -= _rank_offsets[i]; // rank-local.
                    (*clipped_indices)[i] = idiv_flr((*clipped_indices)[i], _vec_lens[i]);
                }
            }
        } // grid dims.
        return all_ok;
    }

    // Set dirty flags between indices.
    void YkGridBase::set_dirty_in_slice(const Indices& first_indices,
                                        const Indices& last_indices) {
        if (_has_step_dim) {
            for (idx_t i = first_indices[+Indices::step_posn];
                 i <= last_indices[+Indices::step_posn]; i++)
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
        STATE_VARS(this);

        // Convert to elem indices.
        Indices eidxs = idxs.mulElements(_vec_lens);

        // Add offsets, i.e., convert to overall indices.
        eidxs = eidxs.addElements(_rank_offsets);

        IdxTuple idxs2 = get_allocs(); // get dims.
        eidxs.setTupleVals(idxs2);      // set vals from eidxs.

        // Visit every point in fold.
        IdxTuple folds = dims->_fold_pts;
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

