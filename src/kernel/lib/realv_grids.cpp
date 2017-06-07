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
        else if (dim == "w" && got_w()) return fn_prefix ## w();        \
        else if (dim == "x" && got_x()) return fn_prefix ## x();        \
        else if (dim == "y" && got_x()) return fn_prefix ## y();        \
        else if (dim == "z" && got_x()) return fn_prefix ## z();        \
        else {                                                          \
            cerr << "Error: " #api_name "(): bad dimension '" << dim << "'\n"; \
            exit_yask(1);                                               \
            return 0;                                                   \
        }                                                               \
    }
    GET_GRID_API(get_domain_size, get_d, false, 0)
    GET_GRID_API(get_halo_size, get_halo_, false, 0)
    GET_GRID_API(get_extra_pad_size, get_extra_pad_, false, 0)
    GET_GRID_API(get_pad_size, get_pad_, false, 0)
    GET_GRID_API(get_alloc_size, get_alloc_, true, get_alloc_t())

#define SET_GRID_API(api_name, fn_prefix, t_ok, t_fn)                   \
    void RealVecGridBase::api_name(const string& dim, idx_t n) {        \
        if (t_ok && dim == "t" && got_t()) t_fn;                        \
        else if (dim == "w" && got_w()) fn_prefix ## w(n);              \
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
    void RealVecGridBase::share_storage(yk_grid_ptr source) {
        auto sp = dynamic_pointer_cast<RealVecGridBase>(source);
        assert(sp);

        if (!sp->get_storage()) {
            cerr << "Error: share_storage() called without source storage allocated.\n";
            exit_yask(1);
        }
        release_storage();
        
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
            auto tdom = get_domain_size(dname);
            auto sdom = sp->get_domain_size(dname);
            if (tdom != sdom) {
                cerr << "Error: attempt to share storage from grid '" << sp->get_name() <<
                    "' with domain-size " << sdom << " with grid '" << get_name() <<
                    "' with domain-size " << tdom << " in '" << dname << "' dim.\n";
                exit_yask(1);
            }
            auto thalo = get_halo_size(dname);
            auto spad = sp->get_pad_size(dname);
            if (thalo > spad) {
                cerr << "Error: attempt to share storage from grid '" << sp->get_name() <<
                    "' with padding-size " << spad << " with grid '" << get_name() <<
                    "' with halo-size " << thalo << " in '" << dname << "' dim.\n";
                exit_yask(1);
            }

            // Copy settings in this dim.
            set_pad_size(dname, spad);
        }

        // Copy data.
        if (!share_data(sp.get())) {
            cerr << "Error: unexpected failure in data sharing.\n";
            exit_yask(1);
        }
    }

    // Checked resize: fails if mem different and already alloc'd.
    void RealVecGridBase::resize() {

        // Just resize if not alloc'd.
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
                for (int wi = get_first_w(); wi <= get_last_w(); wi++) {
                    for (int xi = get_first_x(); xi <= get_last_x(); xi++) {
                        for (int yi = get_first_y(); yi <= get_last_y(); yi++) {
                            for (int zi = get_first_z(); zi <= get_last_z(); zi++) {

                                real_t te = readElem_TWXYZ(ti, wi, xi, yi, zi, __LINE__);
                                real_t re = ref.readElem_TWXYZ(ti, wi, xi, yi, zi, __LINE__);

                                if (!within_tolerance(te, re, epsilon)) {
                                    errs++;
                                    if (errs < maxPrint) {
                                        printElem_TWXYZ(os, "** mismatch",
                                                        ti, wi, xi, yi, zi,
                                                        te, 0, false);
                                        printElem_TWXYZ(os, " != reference",
                                                        ti, wi, xi, yi, zi,
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
        }
        return errs;
    }

    // Print one element.
    void  RealVecGridBase::printElem_TWXYZ(std::ostream& os, const std::string& m,
                                           idx_t t, idx_t w, idx_t x, idx_t y, idx_t z,
                                           real_t e,
                                           int line,
                                           bool newline) const {

        // TODO: make commas look ok w/o z dim.
        if (m.length())
            os << m << ": ";
        os << get_name() << "[";
        if (got_t()) os << "t=" << t << ", ";
        if (got_w()) os << "w=" << w << ", ";
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
    // Indices must be normalized, i.e., already divided by VLEN_*.
    void RealVecGridBase::printVecNorm_TWXYZ(std::ostream& os, const std::string& m,
                                             idx_t t, idx_t wv, idx_t xv, idx_t yv, idx_t zv,
                                             const real_vec_t& v,
                                             int line) const {
        idx_t w = wv * VLEN_W;
        idx_t x = xv * VLEN_X;
        idx_t y = yv * VLEN_Y;
        idx_t z = zv * VLEN_Z;

        // Print each element.
        for (int zi = 0; zi < VLEN_Z; zi++) {
            for (int yi = 0; yi < VLEN_Y; yi++) {
                for (int xi = 0; xi < VLEN_X; xi++) {
                    for (int wi = 0; wi < VLEN_W; wi++) {
                        real_t e = v(wi, xi, yi, zi);
#ifdef CHECK_VEC_ELEMS
                        real_t e2 = readElem_TWXYZ(t, w+wi, x+xi, y+yi, z+zi, line);
#endif
                        printElem_TWXYZ(os, m, t, w+wi, x+xi, y+yi, z+zi, e, line);
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

}
