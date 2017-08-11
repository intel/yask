/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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

    // Convenience function to format indices like
    // "x=5, y=3".
    std::string YkGridBase::makeIndexString(const Indices& idxs,
                                            std::string separator,
                                            std::string infix,
                                            std::string prefix,
                                            std::string suffix) const {
        IdxTuple tmp = _ggb->get_dims(); // copy dim names from grid.
        idxs.setTupleVals(tmp);         // set vals from idxs.
        return tmp.makeDimValStr(separator, infix, prefix, suffix);
    }
    
    // Print one element to 'os' like
    // "message: mygrid[x=4, y=7] = 3.14 at line 35".
    void YkGridBase::printElem(std::ostream& os,
                               const std::string& msg,
                               const Indices& idxs,
                               real_t e,
                               int line,
                               bool newline) const {
        if (msg.length())
            os << msg << ": ";
        os << get_name() << "[" <<
            makeIndexString(idxs) << "] = " << e;
        if (line)
            os << " at line " << line;
        if (newline)
            os << std::endl << std::flush;
    }

    // Lookup position by dim name.
    // Return -1 or die if not found, depending on flag.
    int YkGridBase::get_dim_posn(const std::string& dim,
                                 bool die_on_failure,
                                 const std::string& die_msg) const {
        auto& dims = _ggb->get_dims();
        int posn = dims.lookup_posn(dim);
        if (posn < 0 && die_on_failure) {
            cerr << "Error: " << die_msg << ": dimension '" <<
                dim << "' not used in grid '" << get_name() << ".\n";
            exit_yask(1);
        }
        return posn;
    }
        
    // Resizes the underlying generic grid.
    // Fails if mem different and already alloc'd.
    void YkGridBase::resize() {

#warning FIXME
        // TODO: add padding for cache-line alignment.
        
#if SAVE_FOR_VEC_GRID
        // Some rounding.
        _pxv = CEIL_DIV(_px, VLEN_X);
        _pyv = CEIL_DIV(_py, VLEN_Y);
        _pzv = CEIL_DIV(_pz, VLEN_Z);
        _px = _pxv * VLEN_X;
        _py = _pyv * VLEN_Y;
        _pz = _pzv * VLEN_Z;

        // Alloc.
        _axv = CEIL_DIV(_dx + 2 * _px, VLEN_X);
        _ayv = CEIL_DIV(_dy + 2 * _py, VLEN_Y);
        _azv = CEIL_DIV(_dz + 2 * _pz, VLEN_Z);
#endif

        // Original size.
        size_t old_size = get_num_storage_bytes();
        
        // New allocation in each dim.
        Indices allocs;
        size_t new_size = 1;
        for (int i = 0; i < get_num_dims(); i++) {
            allocs[i] = _domains[i] + 2 * _pads[i];
            new_size *= allocs[i];
        }

        // Attempt to change?
        auto p = get_raw_storage_buffer();
        if (p && old_size != new_size) {
            cerr << "Error: attempt to change required grid size from " <<
                makeByteStr(old_size) << " to " <<
                makeByteStr(new_size) << " after storage has been allocated.\n";
            exit_yask(1);
        }

        // Do the resize.
        for (int i = 0; i < get_num_dims(); i++)
            _ggb->set_dim_size(i, allocs[i]);
    }
    
    // Check whether dim is of allowed type.
    void YkGridBase::checkDimType(const std::string& dim,
                                  const std::string& fn_name,
                                  bool step_ok,
                                  bool domain_ok,
                                  bool misc_ok) const {
        if (!is_dim_used(dim)) {
            cerr << "Error in " << fn_name << "(): dimension '" <<
                dim << "' is not used in grid '" << get_name() << "'.\n";
            exit_yask(1);
        }
        _dims->checkDimType(dim, fn_name, step_ok, domain_ok, misc_ok);
    }
    
    // APIs to get info from vars.
#define GET_GRID_API(api_name, expr, step_ok, domain_ok, misc_ok)       \
    idx_t YkGridBase::api_name(const string& dim) const {               \
        int posn = get_dim_posn(dim, true, #api_name);                  \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        return expr;                                                    \
    }
    GET_GRID_API(get_rank_domain_size, _domains[posn], false, true, false)
    GET_GRID_API(get_pad_size, _pads[posn], false, true, false)
    GET_GRID_API(get_halo_size, _halos[posn], false, true, false)
    GET_GRID_API(get_first_misc_index, _offsets[posn], false, false, true)
    GET_GRID_API(get_last_misc_index, _offsets[posn] + _domains[posn] - 1, false, false, true)
    GET_GRID_API(get_first_rank_domain_index, _offsets[posn], false, true, false)
    GET_GRID_API(get_last_rank_domain_index, _offsets[posn] + _domains[posn] - 1, false, true, false)
    GET_GRID_API(get_first_rank_alloc_index, _offsets[posn] - _pads[posn], false, true, false)
    GET_GRID_API(get_last_rank_alloc_index, _offsets[posn] + _domains[posn] + _pads[posn] - 1, false, true, false)
    GET_GRID_API(get_extra_pad_size, _pads[posn] - _halos[posn], false, true, false)
    GET_GRID_API(get_alloc_size, _ggb->get_dim_size(posn), true, true, true)
    GET_GRID_API(_get_offset, _offsets[posn], true, true, true)
    GET_GRID_API(_get_first_allowed_index, _offsets[posn] - _pads[posn], true, true, true)
    GET_GRID_API(_get_last_allowed_index, _offsets[posn] + _domains[posn] + _pads[posn] - 1, true, true, true)

    // APIs to set vars.
#define COMMA ,
#define SET_GRID_API(api_name, expr, step_ok, domain_ok, misc_ok)       \
    void YkGridBase::api_name(const string& dim, idx_t n) {             \
        int posn = get_dim_posn(dim, true, #api_name);                  \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        expr;                                                           \
    }
    SET_GRID_API(set_halo_size, _halos[posn] = n; _set_pad_size(dim, _pads[posn]), false, true, false)
    SET_GRID_API(set_min_pad_size, if (n < _pads[posn]) _set_pad_size(dim, n), false, true, false)
    SET_GRID_API(set_extra_pad_size, _set_pad_size(dim, _halos[posn] + n), false, true, false)
    SET_GRID_API(set_first_misc_index, _offsets[posn] = n, false, false, true)
    SET_GRID_API(set_alloc_size, _set_domain_size(dim, n); resize(), true, false, true)
    SET_GRID_API(_set_domain_size, _domains[posn] = n; resize(), true, true, true)
    SET_GRID_API(_set_pad_size, _pads[posn] = std::max(n COMMA _halos[posn]); resize(), true, true, true)
    SET_GRID_API(_set_offset, _offsets[posn] = n, true, true, true)
#undef COMMA
    
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
            if (get_alloc_size(dname) != op->get_alloc_size(dname))
                return false;
            if (_domains[i] != op->_domains[i])
                return false;
            if (_pads[i] != op->_pads[i])
                return false;
        }
        return true;
    }

    void YkGridBase::share_storage(yk_grid_ptr source) {
        auto sp = dynamic_pointer_cast<YkGridBase>(source);
        assert(sp);

        if (!sp->get_raw_storage_buffer()) {
            cerr << "Error: share_storage() called without source storage allocated.\n";
            exit_yask(1);
        }

        // NB: requirements to successful share_storage() is not as strict as
        // is_storage_layout_identical(). See note on pad & halo below and API docs.
        for (int i = 0; i < get_num_dims(); i++) {
            auto dname = get_dim_name(i);

            // Same dims?
            if (sp->get_num_dims() != get_num_dims() ||
                sp->get_dim_name(i) != dname) {
                cerr << "Error: share_storage() called with incompatible grids: ";
                print_info(cerr);
                cerr << "; and ";
                sp->print_info(cerr);
                cerr << ".\n";
                exit_yask(1);
            }

            // Not a domain dim?
            if (!_dims->_domain_dims.lookup(dname)) {
                auto tas = get_alloc_size(dname);
                auto sas = sp->get_alloc_size(dname);
                if (tas != sas) {
                    cerr << "Error: attempt to share storage from grid '" << sp->get_name() <<
                        "' with alloc-size " << sas << " with grid '" << get_name() <<
                        "' with alloc-size " << tas << " in '" << dname << "' dim.\n";
                    exit_yask(1);
                }
            }

            // Domain dim.
            else {
                auto tdom = get_rank_domain_size(dname);
                auto sdom = sp->get_rank_domain_size(dname);
                if (tdom != sdom) {
                    cerr << "Error: attempt to share storage from grid '" << sp->get_name() <<
                        "' with domain-size " << sdom << " with grid '" << get_name() <<
                        "' with domain-size " << tdom << " in '" << dname << "' dim.\n";
                exit_yask(1);
                }

                // Halo and pad sizes don't have to be the same.
                // Requirement is that halo of target fits inside of pad of source.
                auto thalo = get_halo_size(dname);
                auto spad = sp->get_pad_size(dname);
                if (thalo > spad) {
                    cerr << "Error: attempt to share storage from grid '" << sp->get_name() <<
                        "' with padding-size " << spad <<
                        ", which is insufficient for grid '" << get_name() <<
                        "' with halo-size " << thalo << " in '" << dname << "' dim.\n";
                    exit_yask(1);
                }
            }
        }

        // Copy pad sizes.
        for (int i = 0; i < get_num_dims(); i++) {
            auto dname = get_dim_name(i);
            auto spad = sp->get_pad_size(dname);
            _set_pad_size(dname, spad);
        }
        
        // Copy data.
        release_storage();
        if (!share_data(sp.get())) {
            cerr << "Error: unexpected failure in data sharing.\n";
            exit_yask(1);
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
            os << "** mismatch due to incompatible grids: ";
            print_info(os);
            os << "; and ";
            ref->print_info(os);
            os << ".\n";
            return get_num_storage_elements();
        }
        
        // Quick check for errors, assuming same layout.
        // TODO: check layout.
        idx_t errs = _ggb->count_diffs(ref->_ggb, epsilon);
        if (!errs)
            return 0;
        
        // Run detailed comparison if any errors found.
        errs = 0;
        auto& allocs = _ggb->get_dims();

        // This will loop over the entire allocation.
        // Indices of 'pt' will be relative to allocation.
        allocs.visitAllPoints([&](const IdxTuple& pt) {
                auto te = readElem(pt, __LINE__);
                auto re = ref->readElem(pt, __LINE__);
                if (!within_tolerance(te, re, epsilon)) {
                    errs++;
                    if (errs < maxPrint) {

                        // Adjust alloc indices to overall indices.
                        IdxTuple opt;
                        for (int i = 0; i < pt.getNumDims(); i++) {
                            auto dname = pt.getDimName(i);
                            auto val = pt.getVal(i);
                            opt.addDimBack(dname, _offsets[i] - _pads[i] + val);
                        }
                        os << "** mismatch at " << get_name() <<
                            "(" << opt.makeDimValStr() << "): " <<
                            te << " != " << re << std::endl;
                    }
                    else if (errs == maxPrint)
                        os << "** Additional errors not printed." << std::endl;
                    else {
                        // errs > maxPrint.
                        return false;
                    }
                }
                return true;
            });
        return errs;
    }

    // Make sure indices are in range.
    // Side-effect: If fixed_indices is not NULL, set them to in-range if out-of-range.
    bool YkGridBase::checkIndices(const GridIndices& indices,
                                  const string& fn,
                                  bool strict_indices, // die if out-of-range.
                                  GridIndices* fixed_indices) const {
        if (indices.size() != size_t(get_num_dims())) {
            cerr << "Error: '" << fn << "' called with " << indices.size() <<
                " indices instead of " << get_num_dims() << ".\n";
            exit_yask(1);
        }
        if (fixed_indices)
            fixed_indices->clear();
        bool ok = true;
        for (int i = 0; i < get_num_dims(); i++) {
            idx_t idx = indices[i];
            if (fixed_indices)
                fixed_indices->push_back(idx);
            auto dname = get_dim_name(i);

            // Any step index is ok because it wraps around.
            if (dname == _dims->_step_dim)
                continue;

            // Within first..last indices?
            auto first_ok = _get_first_allowed_index(dname);
            auto last_ok = _get_last_allowed_index(dname);
            if (idx < first_ok || idx > last_ok) {
                if (strict_indices) {
                    cerr << "Error: " << fn << ": index in dim '" << dname <<
                        "' is " << idx << ", which is not in [" << first_ok <<
                        "..." << last_ok << "].\n";
                    exit_yask(1);
                }
                if (fixed_indices) {
                    if (idx < first_ok)
                        fixed_indices->at(i) = first_ok;
                    if (idx > last_ok)
                        fixed_indices->at(i) = last_ok;
                }
                ok = false;
            }
        }
        return ok;
    }

    // API get/set.
    double YkGridBase::get_element(const GridIndices& indices) const {
        if (!is_storage_allocated()) {
            cerr << "Error: call to 'get_element' with no data allocated for grid '" <<
                get_name() << "'.\n";
            exit_yask(1);
        }
        checkIndices(indices, "get_element", true);
        real_t val = readElem(indices, __LINE__);
        return double(val);
    }
    idx_t YkGridBase::set_element(double val,
                                  const GridIndices& indices,
                                  bool strict_indices) {
        idx_t nup = 0;
        if (get_raw_storage_buffer() &&
            checkIndices(indices, "set_element", strict_indices)) {
            writeElem(real_t(val), indices, __LINE__);
            nup++;
            set_updated(false);
        }
        return nup;
    }
    
    // Convenience API wrappers w/up to 6 dims.
    double YkGridBase::get_element(idx_t dim1_index, idx_t dim2_index,
                                   idx_t dim3_index, idx_t dim4_index,
                                   idx_t dim5_index, idx_t dim6_index) const {
        GridIndices idx;
        int nd = get_num_dims();
        if (nd >= 1)
            idx.push_back(dim1_index);
        if (nd >= 2)
            idx.push_back(dim2_index);
        if (nd >= 3)
            idx.push_back(dim3_index);
        if (nd >= 4)
            idx.push_back(dim4_index);
        if (nd >= 5)
            idx.push_back(dim5_index);
        if (nd >= 6)
            idx.push_back(dim6_index);
        return get_element(idx);
    }
    idx_t YkGridBase::set_element(double val,
                                  idx_t dim1_index, idx_t dim2_index,
                                  idx_t dim3_index, idx_t dim4_index,
                                  idx_t dim5_index, idx_t dim6_index) {
        GridIndices idx;
        int nd = get_num_dims();
        if (nd >= 1)
            idx.push_back(dim1_index);
        if (nd >= 2)
            idx.push_back(dim2_index);
        if (nd >= 3)
            idx.push_back(dim3_index);
        if (nd >= 4)
            idx.push_back(dim4_index);
        if (nd >= 5)
            idx.push_back(dim5_index);
        if (nd >= 6)
            idx.push_back(dim6_index);
        return set_element(val, idx, false);
    }

    idx_t YkGridBase::get_elements_in_slice(void* buffer_ptr,
                                            const GridIndices& first_indices,
                                            const GridIndices& last_indices) const {
        if (!is_storage_allocated()) {
            cerr << "Error: call to 'get_elements_in_slice' with no data allocated for grid '" <<
                get_name() << "'.\n";
            exit_yask(1);
        }
        checkIndices(first_indices, "get_elements_in_slice", true);
        checkIndices(last_indices, "get_elements_in_slice", true);

        // Find ranges.
        IdxTuple firstTuple = _ggb->get_dims();
        firstTuple.setVals(first_indices);
        IdxTuple lastTuple = _ggb->get_dims();
        lastTuple.setVals(last_indices);
        IdxTuple numElemsTuple = lastTuple.addElements(1).subElements(firstTuple);

        // Visit points in slice.
        // TODO: parallelize.
        idx_t i = 0;
        numElemsTuple.visitAllPoints([&](const IdxTuple& ofs) {
                IdxTuple pt = firstTuple.addElements(ofs);
                real_t val = readElem(pt, __LINE__);
                ((real_t*)buffer_ptr)[i] = val;
                i++;
                return true;    // keep going.
            });
        return i;
    }
    idx_t YkGridBase::set_elements_in_slice_same(double val,
                                                 const GridIndices& first_indices,
                                                 const GridIndices& last_indices,
                                                 bool strict_indices) {
        if (!is_storage_allocated())
            return 0;
        
        // 'Fixed' copy of indices.
        GridIndices first, last;
        checkIndices(first_indices, "set_elements_in_slice_same", strict_indices, &first);
        checkIndices(last_indices, "set_elements_in_slice_same", strict_indices, &last);

        // Find ranges.
        IdxTuple firstTuple = _ggb->get_dims();
        firstTuple.setVals(first_indices);
        IdxTuple lastTuple = _ggb->get_dims();
        lastTuple.setVals(last_indices);
        IdxTuple numElemsTuple = lastTuple.addElements(1).subElements(firstTuple);

        // Visit points in slice.
        // TODO: parallelize.
        idx_t i = 0;
        numElemsTuple.visitAllPoints([&](const IdxTuple& ofs) {
                IdxTuple pt = firstTuple.addElements(ofs);
                writeElem(real_t(val), pt, __LINE__);
                i++;
                return true;    // keep going.
            });
        return i;
    }
    idx_t YkGridBase::set_elements_in_slice(const void* buffer_ptr,
                                            const GridIndices& first_indices,
                                            const GridIndices& last_indices) {
        if (!is_storage_allocated())
            return 0;
        
        checkIndices(first_indices, "get_elements_in_slice", true);
        checkIndices(last_indices, "get_elements_in_slice", true);

        // Find ranges.
        IdxTuple firstTuple = _ggb->get_dims();
        firstTuple.setVals(first_indices);
        IdxTuple lastTuple = _ggb->get_dims();
        lastTuple.setVals(last_indices);
        IdxTuple numElemsTuple = lastTuple.addElements(1).subElements(firstTuple);

        // Visit points in slice.
        // TODO: parallelize.
        idx_t i = 0;
        numElemsTuple.visitAllPoints([&](const IdxTuple& ofs) {
                IdxTuple pt = firstTuple.addElements(ofs);
                real_t val = ((real_t*)buffer_ptr)[i];
                writeElem(val, pt, __LINE__);
                i++;
                return true;    // keep going.
            });
        return i;
    }

#warning FIXME
#if 0
    // Print some info.
    void YkGridBase::print_info(std::ostream& os) {
        _gp->print_info(os, "SIMD vector");
    }

    // Initialize memory to incrementing values based on val.
    void RealVecGridBase::set_diff(real_t val) {

        // make a real_vec_t pattern.
        real_vec_t rn;
        for (int i = 0; i < VLEN; i++)
            rn[i] = real_t(i * VLEN + 1) * val / VLEN;
        
        _gp->set_diff(rn);
    }

    // Print one vector at *vector* offset.
    // Indices must be relative to rank, i.e., offset is already subtracted.
    // Indices must be normalized, i.e., already divided by VLEN_*.
    void RealVecGridBase::printVecNorm_TXYZ(std::ostream& os, const std::string& m,
                                             idx_t t, idx_t xv, idx_t yv, idx_t zv,
                                             const real_vec_t& v,
                                             int line) const {
        idx_t x = xv * VLEN_X + _ox;
        idx_t y = yv * VLEN_Y + _oy;
        idx_t z = zv * VLEN_Z + _oz;

        // Print each element.
        for (int zi = 0; zi < VLEN_Z; zi++) {
            for (int yi = 0; yi < VLEN_Y; yi++) {
                for (int xi = 0; xi < VLEN_X; xi++) {
                    real_t e = v(xi, yi, zi);
#ifdef CHECK_VEC_ELEMS
                    real_t e2 = readElem_TXYZ(t, x+xi, y+yi, z+zi, line);
#endif
                    printElem_TXYZ(os, m, t, x+xi, y+yi, z+zi, e, line);
#ifdef CHECK_VEC_ELEMS
                    // compare to per-element read.
                    if (e == e2)
                        os << " (same as readElem())";
                    else
                        os << " != " << e2 << " from readElem() <<<< ERROR";
#endif
                    os << std::endl << std::flush;
                }
            }
        }
    }
#endif
}
