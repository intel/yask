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

#include "stencil.hpp"

namespace yask {

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

        // TODO: fix '*'s w/o z dim.
        os << get_num_dims() << "D (";
        if (got_t()) os << "t=" << get_tdim() << " * ";
        if (got_n()) os << "n=" << get_dn() << " * ";
        if (got_x()) os << "x=" << get_dx() << " * ";
        if (got_y()) os << "y=" << get_dy() << " * ";
        if (got_z()) os << "z=" << get_dz();
        os << ") '" << get_name() << "' data is at " << get_storage() << ": " <<
                printWithPow10Multiplier(get_num_elems()) << " element(s) of " <<
                sizeof(real_t) << " byte(s) each, " <<
                printWithPow10Multiplier(get_num_real_vecs()) << " vector(s), " <<
                printWithPow2Multiplier(get_num_bytes()) << "B.\n";
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
            
            for (int ti = 0; ti <= get_tdim(); ti++) {
                for (int ni = get_first_n(); ni <= get_last_n(); ni++) {
                    for (int xi = get_first_x(); xi <= get_last_x(); xi++) {
                        for (int yi = get_first_y(); yi <= get_last_y(); yi++) {
                            for (int zi = get_first_z(); zi <= get_last_z(); zi++) {

                                real_t te = readElem_TNXYZ(ti, ni, xi, yi, zi, __LINE__);
                                real_t re = ref.readElem_TNXYZ(ti, ni, xi, yi, zi, __LINE__);

                                if (!within_tolerance(te, re, epsilon)) {
                                    errs++;
                                    if (errs < maxPrint) {
                                        printElem_TNXYZ(os, "** mismatch",
                                                        ti, ni, xi, yi, zi,
                                                        te, 0, false);
                                        printElem_TNXYZ(os, " != reference",
                                                        ti, ni, xi, yi, zi,
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
    void  RealVecGridBase::printElem_TNXYZ(std::ostream& os, const std::string& m,
                                           idx_t t, idx_t n, idx_t x, idx_t y, idx_t z,
                                           real_t e,
                                           int line,
                                           bool newline) const {

        // TODO: make commas look ok w/o z dim.
        if (m.length())
            os << m << ": ";
        os << _name << "[";
        if (got_t()) os << "t=" << t << ", ";
        if (got_n()) os << "n=" << n << ", ";
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
    void RealVecGridBase::printVecNorm_TNXYZ(std::ostream& os, const std::string& m,
                                             idx_t t, idx_t nv, idx_t xv, idx_t yv, idx_t zv,
                                             const real_vec_t& v,
                                             int line) const {
        idx_t n = nv * VLEN_N;
        idx_t x = xv * VLEN_X;
        idx_t y = yv * VLEN_Y;
        idx_t z = zv * VLEN_Z;

        // Print each element.
        for (int zi = 0; zi < VLEN_Z; zi++) {
            for (int yi = 0; yi < VLEN_Y; yi++) {
                for (int xi = 0; xi < VLEN_X; xi++) {
                    for (int ni = 0; ni < VLEN_N; ni++) {
                        real_t e = v(ni, xi, yi, zi);
#ifdef CHECK_VEC_ELEMS
                        real_t e2 = readElem_TNXYZ(t, n+ni, x+xi, y+yi, z+zi, line);
#endif
                        printElem_TNXYZ(os, m, t, n+ni, x+xi, y+yi, z+zi, e, line);
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
