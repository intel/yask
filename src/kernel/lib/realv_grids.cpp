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

    // APIs.
    // TODO: remove hard-coded dimensions.
#define GET_GRID_API(api_name, fn_prefix, t_ok, t_fn)                   \
    idx_t RealVecGridBase::api_name(const string& dim) const {          \
        if (t_ok && dim == "t" && got_t()) return t_fn;                 \
        else if (dim == "x" && got_x()) return fn_prefix ## x();        \
        else if (dim == "y" && got_x()) return fn_prefix ## y();        \
        else if (dim == "z" && got_x()) return fn_prefix ## z();        \
        else {                                                          \
            cerr << "Error: " #api_name "(): bad dimension '" << dim << "'\n"; \
            exit_yask(1);                                               \
            return 0;                                                   \
        }                                                               \
    }
    GET_GRID_API(get_rank_domain_size, get_d, false, 0)
    GET_GRID_API(get_first_rank_domain_index, get_first_, false, 0)
    GET_GRID_API(get_last_rank_domain_index, get_last_, false, 0)
    GET_GRID_API(get_first_rank_alloc_index, get_first_alloc_, false, 0)
    GET_GRID_API(get_last_rank_alloc_index, get_last_alloc_, false, 0)
    GET_GRID_API(get_halo_size, get_halo_, false, 0)
    GET_GRID_API(get_extra_pad_size, get_extra_pad_, false, 0)
    GET_GRID_API(get_pad_size, get_pad_, false, 0)
    GET_GRID_API(get_alloc_size, get_alloc_, true, get_alloc_t())

#define SET_GRID_API(api_name, fn_prefix, t_ok, t_fn)                   \
    void RealVecGridBase::api_name(const string& dim, idx_t n) {        \
        if (t_ok && dim == "t" && got_t()) t_fn;                        \
        else if (dim == "x" && got_x()) fn_prefix ## x(n);              \
        else if (dim == "y" && got_x()) fn_prefix ## y(n);              \
        else if (dim == "z" && got_x()) fn_prefix ## z(n);              \
        else {                                                          \
            cerr << "Error: " #api_name "(): bad dimension '" << dim << "'\n"; \
            exit_yask(1);                                               \
        }                                                               \
    }
    SET_GRID_API(set_min_pad_size, set_min_pad_, false, (void)0)
    SET_GRID_API(set_pad_size, set_pad_, false, (void)0)
    SET_GRID_API(set_halo_size, set_halo_, false, (void)0)

    // Not using SET_GRID_API macro because *only* 't' is allowed.
    void RealVecGridBase::set_alloc_size(const string& dim, idx_t tdim) {

        // TODO: remove hard-coded dimensions.
        if (dim == "t" && got_t()) return set_alloc_t(tdim);
        else {
            cerr << "Error: set_alloc_size(): bad dim '" << dim << "'\n";
            exit_yask(1);
        }
    }
    bool RealVecGridBase::is_storage_layout_identical(const yk_grid_ptr other) const {
        auto op = dynamic_pointer_cast<RealVecGridBase>(other);
        assert(op);

        // Same size?
        if (get_num_bytes() != op->get_num_bytes())
            return false;

        // Same dims?
        if (get_num_dims() != op->get_num_dims())
            return false;
        for (int i = 0; i < get_num_dims(); i++) {
            auto dname = get_dim_name(i);
            if (dname != op->get_dim_name(i))
                return false;
            if (get_alloc_size(dname) != op->get_alloc_size(dname))
                return false;

            // TODO: remove hard-coded step-dim name.
            if (dname != "t") {
                if (get_rank_domain_size(dname) != op->get_rank_domain_size(dname))
                    return false;
                if (get_pad_size(dname) != op->get_pad_size(dname))
                    return false;
            }
        }
        return true;
    }
    void RealVecGridBase::share_storage(yk_grid_ptr source) {
        auto sp = dynamic_pointer_cast<RealVecGridBase>(source);
        assert(sp);

        if (!sp->get_storage()) {
            cerr << "Error: share_storage() called without source storage allocated.\n";
            exit_yask(1);
        }
        release_storage();

        // NB: requirements to successful share_storage() is not as strict as
        // is_storage_layout_identical(). See note on pad & halo below and API docs.
        for (int i = 0; i < get_num_dims(); i++) {
            auto dname = get_dim_name(i);
            if (sp->get_num_dims() != get_num_dims() ||
                sp->get_dim_name(i) != dname) {
                cerr << "Error: share_storage() called with incompatible grids: ";
                print_info(cerr);
                cerr << "; and ";
                sp->print_info(cerr);
                cerr << ".\n";
                exit_yask(1);
            }

            // TODO: remove hard-coded step-dim name.
            if (dname == "t") {
                auto tas = get_alloc_size(dname);
                auto sas = sp->get_alloc_size(dname);
                if (tas != sas) {
                    cerr << "Error: attempt to share storage from grid '" << sp->get_name() <<
                        "' with alloc-size " << sas << " with grid '" << get_name() <<
                        "' with alloc-size " << tas << " in '" << dname << "' dim.\n";
                    exit_yask(1);
                }
            }

            // Not step-dim.
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
                // Requirement is that halo of target fits in pad of source.
                auto thalo = get_halo_size(dname);
                auto spad = sp->get_pad_size(dname);
                if (thalo > spad) {
                    cerr << "Error: attempt to share storage from grid '" << sp->get_name() <<
                        "' with padding-size " << spad << " with grid '" << get_name() <<
                        "' with halo-size " << thalo << " in '" << dname << "' dim.\n";
                    exit_yask(1);
                }

                // Copy pad settings in this dim.
                set_pad_size(dname, spad);
            }
        }

        // Copy data.
        if (!share_data(sp.get())) {
            cerr << "Error: unexpected failure in data sharing.\n";
            exit_yask(1);
        }
    }

    double RealVecGridBase::get_element(idx_t dim1_index, idx_t dim2_index,
                                        idx_t dim3_index, idx_t dim4_index,
                                        idx_t dim5_index, idx_t dim6_index) const {
        GridIndices idx = {dim1_index, dim2_index, dim3_index};
        if (got_t())
            idx.push_back(dim4_index);
        return get_element(idx);
    }
    idx_t RealVecGridBase::set_element(double val,
                                       idx_t dim1_index, idx_t dim2_index,
                                       idx_t dim3_index, idx_t dim4_index,
                                       idx_t dim5_index, idx_t dim6_index) {
        GridIndices idx = {dim1_index, dim2_index, dim3_index};
        if (got_t())
            idx.push_back(dim4_index);
        return set_element(val, idx, false);
    }

    // Use the 'halo' loop for reading and writing elements.
#define SET_HALO_LOOP_VARS(first, last)         \
    int i = 0;                                  \
    idx_t begin_ht = 0, end_ht = 1;             \
    if (got_t()) {                              \
        begin_ht = first[i];                    \
        end_ht = last[i++] + 1;                 \
    }                                           \
    idx_t begin_hx = first[i];                  \
    idx_t end_hx = last[i++] + 1;               \
    idx_t begin_hy = first[i];                  \
    idx_t end_hy = last[i++] + 1;               \
    idx_t begin_hz = first[i];                  \
    idx_t end_hz = last[i++] + 1;               \
    const idx_t step_hx = 1;                    \
    const idx_t step_hy = 1;                    \
    const idx_t step_hz = 1;                    \
    const idx_t group_size_hx = 1;              \
    const idx_t group_size_hy = 1;              \
    const idx_t group_size_hz = 1;              \
    idx_t num_ht = end_ht - begin_ht;           \
    idx_t num_hx = end_hx - begin_hx;           \
    idx_t num_hy = end_hy - begin_hy;           \
    idx_t num_hz = end_hz - begin_hz;                           \
    idx_t num_htxyz = num_ht * num_hx * num_hy * num_hz;        \
    Layout_1234 buf_layout(num_ht, num_hx, num_hy, num_hz)
    
    double RealVecGridBase::get_element(const GridIndices& indices) const {
        if (!get_storage()) {
            cerr << "Error: call to 'get_element' with no data allocated for grid '" <<
                get_name() << "'.\n";
            exit_yask(1);
        }
        checkIndices(indices, "get_element", true);
        SET_HALO_LOOP_VARS(indices, indices);
        real_t val = readElem_TXYZ(begin_ht, begin_hx, begin_hy, begin_hz, __LINE__);
        return double(val);
    }
    idx_t RealVecGridBase::set_element(double val,
                                       const GridIndices& indices,
                                       bool strict_indices) {
        idx_t nup = 0;
        if (get_storage() &&
            checkIndices(indices, "set_element", strict_indices)) {
            SET_HALO_LOOP_VARS(indices, indices);
            writeElem_TXYZ(real_t(val),
                           begin_ht, begin_hx, begin_hy, begin_hz, __LINE__);
            nup++;
            set_updated(false);
        }
        return nup;
    }
    
    idx_t RealVecGridBase::get_elements_in_slice(void* buffer_ptr,
                                                 const GridIndices& first_indices,
                                                 const GridIndices& last_indices) const {
        if (!get_storage()) {
            cerr << "Error: call to 'get_elements_in_slice' with no data allocated for grid '" <<
                get_name() << "'.\n";
            exit_yask(1);
        }
        checkIndices(first_indices, "get_elements_in_slice", true);
        checkIndices(last_indices, "get_elements_in_slice", true);

        SET_HALO_LOOP_VARS(first_indices, last_indices);

        // Define calc func inside OMP loop.
        // 'index_h*' vars are 0-based indices for each dim.
        // Ignoring 'stop_h*' vars because all 'step_h*' vars are 1.
#define calc_halo(ht,                                                   \
                  start_hx, start_hy, start_hz,                         \
                  stop_hx, stop_hy, stop_hz)  do {                      \
            real_t v = readElem_TXYZ(ht, start_hx, start_hy, start_hz, __LINE__); \
            idx_t bi = buf_layout.layout(index_ht, index_hx, index_hy, index_hz); \
            ((real_t*)buffer_ptr)[bi] = v;                            \
        } while(0)

        // Outer time loop.
        for (idx_t ht = begin_ht; ht < end_ht; ht++) {
            idx_t index_ht = ht - begin_ht;
        
            // Include auto-generated loops to invoke calc_halo() from
            // begin_h* to end_h* by step_h*.
#include "yask_halo_loops.hpp"
#undef calc_halo
        }
        return num_htxyz;
    }
    idx_t RealVecGridBase::set_elements_in_slice_same(double val,
                                                      const GridIndices& first_indices,
                                                      const GridIndices& last_indices,
                                                      bool strict_indices) {
        if (!get_storage())
            return 0;
        
        // 'Fixed' copy of indices.
        GridIndices first, last;
        checkIndices(first_indices, "set_elements_in_slice_same", strict_indices, &first);
        checkIndices(last_indices, "set_elements_in_slice_same", strict_indices, &last);

        SET_HALO_LOOP_VARS(first, last);
        real_t v = real_t(val);
        
        // Define calc func inside OMP loop.
        // 'index_h*' vars are 0-based indices for each dim.
        // Ignoring 'stop_h*' vars because all 'step_h*' vars are 1.
#define calc_halo(ht,                                                   \
                  start_hx, start_hy, start_hz,                         \
                  stop_hx, stop_hy, stop_hz)  do {                      \
            writeElem_TXYZ(v, ht, start_hx, start_hy, start_hz, __LINE__); \
        } while(0)

        // Outer time loop.
        for (idx_t ht = begin_ht; ht < end_ht; ht++) {
            idx_t index_ht = ht - begin_ht;
        
            // Include auto-generated loops to invoke calc_halo() from
            // begin_h* to end_h* by step_h*.
#include "yask_halo_loops.hpp"
#undef calc_halo
        }
        return num_htxyz;
    }
    idx_t RealVecGridBase::set_elements_in_slice(const void* buffer_ptr,
                                                 const GridIndices& first_indices,
                                                 const GridIndices& last_indices) {
        if (!get_storage())
            return 0;
        
        checkIndices(first_indices, "get_elements_in_slice", true);
        checkIndices(last_indices, "get_elements_in_slice", true);

        SET_HALO_LOOP_VARS(first_indices, last_indices);

        // Define calc func inside OMP loop.
        // 'index_h*' vars are 0-based indices for each dim.
        // Ignoring 'stop_h*' vars because all 'step_h*' vars are 1.
#define calc_halo(ht,                                                   \
                  start_hx, start_hy, start_hz,                         \
                  stop_hx, stop_hy, stop_hz)  do {                      \
            idx_t bi = buf_layout.layout(index_ht, index_hx, index_hy, index_hz); \
            real_t v = ((real_t*)buffer_ptr)[bi];                       \
            writeElem_TXYZ(v, ht, start_hx, start_hy, start_hz, __LINE__); \
        } while(0)

        // Outer time loop.
        for (idx_t ht = begin_ht; ht < end_ht; ht++) {
            idx_t index_ht = ht - begin_ht;
        
            // Include auto-generated loops to invoke calc_halo() from
            // begin_h* to end_h* by step_h*.
#include "yask_halo_loops.hpp"
#undef calc_halo
        }
        return num_htxyz;
    }
    
    // Checked resize: fails if mem different and already alloc'd.
    void RealVecGridBase::resize() {

        // Some checking.
        assert(_tdim >= 1);
        assert(_dx >= 1);
        assert(_dy >= 1);
        assert(_dz >= 1);
        
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
        
        // Just resize grid if not alloc'd.
        auto p = get_storage();
        if (!p)
            resize_g();
            
        else {
            // Original size.
            size_t nb1 = get_num_bytes();
        
            // Resize the underlying grid.
            resize_g();

            // Changed?
            size_t nb2 = get_num_bytes();
            if (nb1 != nb2) {
                cerr << "Error: attempt to change required grid size from " <<
                    printWithPow2Multiplier(nb1) << "B to " <<
                    printWithPow2Multiplier(nb2) << "B after storage has been allocated.\n";
                exit_yask(1);
            }
        }
    }
    
    // Make sure indices are in range.
    bool RealVecGridBase::checkIndices(const GridIndices& indices,
                                       const string& fn,
                                       bool strict_indices,
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
            if (dname == "t") continue; // any time index is ok.
            auto first_ok = get_first_rank_alloc_index(dname);
            auto last_ok = get_last_rank_alloc_index(dname);
            if (idx < first_ok || idx > last_ok) {
                if (strict_indices) {
                    cerr << "Error: '" << fn << "' index in dim '" << dname <<
                        "' is " << idx << ", which is not in [" << first_ok <<
                        "..." << last_ok << "].\n";
                    exit_yask(1);
                }
                if (fixed_indices && idx < first_ok)
                    fixed_indices->at(i) = first_ok;
                if (fixed_indices && idx > last_ok)
                    fixed_indices->at(i) = last_ok;
                ok = false;
            }
        }
        return ok;
    }
        
    // Initialize memory to incrementing values based on val.
    void RealVecGridBase::set_diff(real_t val) {

        // make a real_vec_t pattern.
        real_vec_t rn;
        for (int i = 0; i < VLEN; i++)
            rn[i] = real_t(i * VLEN + 1) * val / VLEN;
        
        _gp->set_diff(rn);
    }

    // Print some info.
    void RealVecGridBase::print_info(std::ostream& os) {
        _gp->print_info(os, "SIMD vector");
    }
    
    // Check for equality.
    // Return number of mismatches greater than epsilon.
    idx_t RealVecGridBase::compare(const RealVecGridBase& ref,
                                   real_t epsilon,
                                   int maxPrint,
                                   std::ostream& os) const {
        real_vec_t ev;
        ev = epsilon;           // broadcast to real_vec_t elements.

        // Quick check for errors.
        idx_t errs = _gp->count_diffs(*ref._gp, epsilon);

        // Run detailed comparison if any errors found.
        if (errs > 0 && maxPrint) {

            // Need to recount errors by element.
            errs = 0;
            
            for (int ti = 0; ti <= get_alloc_t(); ti++) {
                for (int xi = get_first_x(); xi <= get_last_x(); xi++) {
                    for (int yi = get_first_y(); yi <= get_last_y(); yi++) {
                        for (int zi = get_first_z(); zi <= get_last_z(); zi++) {

                            real_t te = readElem_TXYZ(ti, xi, yi, zi, __LINE__);
                            real_t re = ref.readElem_TXYZ(ti, xi, yi, zi, __LINE__);

                            if (!within_tolerance(te, re, epsilon)) {
                                errs++;
                                if (errs < maxPrint) {
                                    printElem_TXYZ(os, "** mismatch",
                                                    ti, xi, yi, zi,
                                                    te, 0, false);
                                    printElem_TXYZ(os, " != reference",
                                                    ti, xi, yi, zi,
                                                    re, 0, true);
                                }
                                else if (errs == maxPrint)
                                    os << "** Additional errors not printed." << std::endl;
                            }
                        }
                    }
                }
            }
        }
        return errs;
    }

    // Print one element.
    void  RealVecGridBase::printElem_TXYZ(std::ostream& os, const std::string& m,
                                           idx_t t, idx_t x, idx_t y, idx_t z,
                                           real_t e,
                                           int line,
                                           bool newline) const {

        // TODO: make commas look ok w/o z dim.
        if (m.length())
            os << m << ": ";
        os << get_name() << "[";
        if (got_t()) os << "t=" << t << ", ";
        if (got_x()) os << "x=" << x << ", ";
        if (got_y()) os << "y=" << y << ", ";
        if (got_z()) os << "z=" << z;
        os << "] = " << e;
        if (line)
            os << " at line " << line;
        if (newline)
            os << std::endl << std::flush;
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
}
