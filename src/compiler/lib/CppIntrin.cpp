/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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

// Implementation of methods in CppIntrin.hpp.

#include "CppIntrin.hpp"

// Try to use align instruction(s) to construct nelems_target elements
// per instruction.
void CppIntrinPrintHelper::try_align(ostream& os,
                                    const string& pv_name,
                                    size_t nelems_target,
                                    const VecElemList& elems,
                                    set<size_t>& done_elems,
                                    const VarPointSet& aligned_vecs,
                                    bool mask_allowed) {
    size_t nelems = elems.size();

    // Find case(s) that can use valignd.
    // Try all possible combinations of 2 aligned vectors, including
    // each vector paired w/itself.
    for (auto mi = aligned_vecs.begin(); mi != aligned_vecs.end(); mi++) {
        auto& mv1 = *mi;
        for (auto mj = aligned_vecs.begin(); mj != aligned_vecs.end(); mj++) {
            auto& mv2 = *mj;

            // Look up existing input vars.
            // All should be ready before this function is called.
            assert(_vec_vars.count(mv1));
            string mv1_name = _vec_vars[mv1];
            assert(_vec_vars.count(mv2));
            string mv2_name = _vec_vars[mv2];

            // Try all possible shift amounts.
            for (size_t count = 1;
                 done_elems.size() < nelems && count < nelems; count++) {
#ifdef DEBUG_SHIFT
                cout << " //** Checking shift of " << mv1_name <<
                    " and " << mv2_name << " by " << count << " elements." << endl;
#endif

                // Check each position i in output to see if it's correct.
                size_t nused1 = 0, nused2 = 0;
                set<size_t> found_elems;
                unsigned mask = 0;
                for (size_t i = 0; i < nelems; i++) {
                    size_t p = i + count;

                    // Ignore elements that are already constructed.
                    if (done_elems.count(i))
                        continue;

                    // From 1st vector?
                    if (i >= nelems - count) {
                        p -= nelems;

                        // Does the ith element come from mv1?
                        auto ve = elems[i];
                        if (ve._vec == mv1) {

                            // Is the ith element mv1[p]?
                            if (ve._offset == p) {
                                nused1++;
                                mask |= (1 << i); // set proper bit.
                                found_elems.insert(i); // remember bit.
#ifdef DEBUG_SHIFT
                                cout << " // ** " << pv_name << "[" << i << "] = " <<
                                    mv1_name << "[" << p << "]" << endl;
#endif
                            }
                        }
                    }

                    // From 2nd vector.
                    else {

                        // Does the ith element come from mv2?
                        auto ve = elems[i];
                        if (ve._vec == mv2) {

                            // Is the ith element mv2[p2]?
                            if (ve._offset == p) {
                                nused2++;
                                mask |= (1 << i); // set proper bit.
                                found_elems.insert(i); // remember bit.
#ifdef DEBUG_SHIFT
                                cout << " // ** " << pv_name << "[" << i << "] = " <<
                                    mv2_name << "[" << p << "]" << endl;
#endif
                            }
                        }
                    }
                }

                // Now, nused1 is the number of elements that can be shifted from mv1,
                // and  nused2 is the number of elements that can be shifted from mv2.
                assert(nused1 + nused2 == found_elems.size());

                // Have we found the target number of bits?
                if (found_elems.size() == nelems_target) {

                    // We don't need to use the mask if no elements are already done.
                    bool need_mask = done_elems.size() > 0;

                    // If we can't use masking, must stop here.
                    if (need_mask && !mask_allowed)
                        return;

                    // 2-var shift.
                    if (nused1 && nused2) {
                        os << " // Get " << nused1 << " element(s) from " << mv1_name <<
                            " and " << nused2 << " from " << mv2_name << "." << endl;
                        os << _line_prefix << "real_vec_align";
                        if (need_mask)
                            os << "_masked";
                        os << "<" << count << ">(" << pv_name <<
                            ", " << mv1_name << ", " << mv2_name;
                        if (need_mask)
                            print_mask(os, mask);
                        os << ")" << _line_suffix;
                    }

                    // 1-var shift (rotate).
                    else if (nused1) {
                        os << " // Get " << nused1 << " element(s) from " << mv1_name <<
                            "." << endl;
                        os << _line_prefix << "real_vec_align";
                        if (need_mask)
                            os << "_masked";
                        os << "<" << count << ">(" << pv_name << ", " << mv1_name << ", " << mv1_name;
                        if (need_mask)
                            print_mask(os, mask);
                        os << ")" << _line_suffix;
                    }
                    else {
                        assert(nused2);
                        os << " // Get " << nused2 << " element(s) from " << mv2_name <<
                            "." << endl;
                        os << _line_prefix << "real_vec_align";
                        if (need_mask)
                            os << "_masked";
                        os << "<" << count << ">(" << pv_name << ", " << mv2_name << ", " << mv2_name;
                        if (need_mask)
                            print_mask(os, mask);
                        os << ")" << _line_suffix;
                    }

                    // Done w/the found elems.
                    for (auto fei = found_elems.begin(); fei != found_elems.end(); fei++) {
                        size_t fe = *fei;
                        done_elems.insert(fe);
                    }

                } // found.
            }
        }
    } // all combos of 2 aligned vecs.
}

// Try to use 1-var permute instruction(s) to construct nelems_target elements
// per instruction.
void CppIntrinPrintHelper::try_perm1(ostream& os,
                                    const string& pv_name,
                                    size_t nelems_target,
                                    const VecElemList& elems,
                                    set<size_t>& done_elems,
                                    const VarPointSet& aligned_vecs) {
    size_t nelems = elems.size();

    // Try a permute of each aligned vector.
    for (auto mi = aligned_vecs.begin(); mi != aligned_vecs.end(); mi++) {
        auto mv = *mi;

        // Look up existing input var.
        assert(_vec_vars.count(mv));
        string mv_name = _vec_vars[mv];

        // Create the permute control and mask vars.
        string name_s, ctrl_s;
        unsigned mask = 0;
        bool need_na = false;
        set<size_t> found_elems;

        // Loop through each element in the unaligned vector.
        for (size_t i = 0; i < nelems; i++) {

            // What aligned vector and offset does it come from?
            auto ve = elems[i];

            // String separators.
            if (i > 0) {
                name_s += "_";
                ctrl_s += ", ";
            }

            // Is i needed (not done) AND does i come from this mem vec?
            // If so, we want to permute it from the correct offset.
            if (done_elems.count(i) == 0 && ve._vec == mv) {
                int aligned_elem = ve._offset; // get this element.

                name_s += to_string(aligned_elem);
                ctrl_s += to_string(aligned_elem);
                mask |= (1 << i); // set proper bit.
                found_elems.insert(i); // remember element index.
            }

            // Not from this mem vec.
            else {
                name_s += "NA";
                ctrl_s += "NA";
                need_na = true;
            }
        }

        // Have we found the target number of bits?
        if (found_elems.size() == nelems_target) {

            // We don't need to use the mask if no elements are already done.
            bool need_mask = done_elems.size() > 0;

            ctrl_s = "{ .ci = { " + ctrl_s + " } }"; // assignment to 'i' array.

            // Create NA var if needed (this is just for clarity).
            if (need_na)
                make_na(os, _line_prefix, _line_suffix);

            // Create control if needed.
            if (defined_ctrls.count(name_s) == 0) {
                defined_ctrls.insert(name_s);
                os << _line_prefix << "const " << get_var_type() <<
                    "_data ctrl_data_" << name_s << " = " << ctrl_s << _line_suffix;
                os << _line_prefix << "const " << get_var_type() <<
                    " ctrl_" << name_s << "(ctrl_data_" << name_s << ")" << _line_suffix;
            }

            // Permute command.
            os << _line_prefix << "real_vec_permute";
            if (need_mask)
                os << "_masked";
            os << "(" << pv_name << ", ctrl_" << name_s << ", " << mv_name;
            if (need_mask)
                print_mask(os, mask);
            os << ")" << _line_suffix;

            // Done w/the found elems.
            for (auto fei = found_elems.begin(); fei != found_elems.end(); fei++) {
                size_t fe = *fei;
                done_elems.insert(fe);
            }
        } // found.
    } // aligned vectors.
}

// Try to use 2-var permute instruction(s) to construct nelems_target elements
// per instruction.
void CppIntrinPrintHelper::try_perm2(ostream& os,
                                    const string& pv_name,
                                    size_t nelems_target,
                                    const VecElemList& elems,
                                    set<size_t>& done_elems,
                                    const VarPointSet& aligned_vecs) {
    size_t nelems = elems.size();

    // There is no source-preserving mask version of permutex2var, so
    // we bail if any elements have already been found.
    if (done_elems.size() > 0)
        return;

    // Find case(s) that can use perm2.  Try all possible combinations
    // of 2 aligned vectors, but NOT including each vector paired
    // w/itself. (For that, we can use perm1.)
    for (auto mi = aligned_vecs.begin(); mi != aligned_vecs.end(); mi++) {
        auto& mv1 = *mi;
        for (auto mj = aligned_vecs.begin(); mj != aligned_vecs.end(); mj++) {
            auto& mv2 = *mj;

            if (mv1 == mv2) continue;

            // Look up existing input vars.
            // All should be ready before this function is called.
            assert(_vec_vars.count(mv1));
            string mv1_name = _vec_vars[mv1];
            assert(_vec_vars.count(mv2));
            string mv2_name = _vec_vars[mv2];

            // Create the permute control vars: one for 1,2 order, and one for 2,1.
            string name_s12, ctrl_s12;
            string name_s21, ctrl_s21;
            bool need_na = false;
            set<size_t> found_elems;

            // Loop through each element in the unaligned vector.
            for (size_t i = 0; i < nelems; i++) {

                // What aligned vector and offset does it come from?
                auto ve = elems[i];

                // String separators.
                if (i > 0) {
                    name_s12 += "_";
                    ctrl_s12 += ", ";
                    name_s21 += "_";
                    ctrl_s21 += ", ";
                }

                // Is i needed (not done) AND does i come from one of the mem vecs?
                // If so, we want to permute it from the correct offset.
                if (done_elems.count(i) == 0 &&
                    (ve._vec == mv1 || ve._vec == mv2)) {
                    bool use_a = (ve._vec == mv1); // first vec?
                    char aligned_vec12 = use_a ? 'A' : 'B';
                    char aligned_vec21 = !use_a ? 'A' : 'B';
                    int aligned_elem = ve._offset; // get this element.

                    name_s12 += aligned_vec12 + to_string(aligned_elem);
                    name_s21 += aligned_vec21 + to_string(aligned_elem);
                    if (!use_a)
                        ctrl_s12 += "ctrl_sel_bit |"; // set selector bit for vec B.
                    else
                        ctrl_s21 += "ctrl_sel_bit |"; // set selector bit for vec B.
                    ctrl_s12 += to_string(aligned_elem);
                    ctrl_s21 += to_string(aligned_elem);
                    found_elems.insert(i); // remember element index.
                }

                // Not from either mem vec.
                else {
                    name_s12 += "NA";
                    ctrl_s12 += "NA";
                    name_s21 += "NA";
                    ctrl_s21 += "NA";
                    need_na = true;
                }
            }

            // Have we found the target number of bits?
            if (found_elems.size() == nelems_target) {
                assert(done_elems.size() == 0);

                // Create NA var if needed (this is just for clarity).
                if (need_na)
                    make_na(os, _line_prefix, _line_suffix);

                // Var names.
                ctrl_s12 = "{ .ci = { " + ctrl_s12 + " } }";
                ctrl_s21 = "{ .ci = { " + ctrl_s21 + " } }";

                // Select 1,2 or 2,1 depending on whether ctrl var already exists.
                bool use12 = defined_ctrls.count(name_s21) == 0;
                string name_s = use12 ? name_s12 : name_s21;
                string ctrl_s = use12 ? ctrl_s12 : ctrl_s21;

                // Create control if needed.
                if (defined_ctrls.count(name_s) == 0) {
                    defined_ctrls.insert(name_s);
                    os << _line_prefix << "const " << get_var_type() <<
                        "_data ctrl_data_" << name_s << " = " << ctrl_s << _line_suffix;
                    os << _line_prefix << "const " << get_var_type() <<
                        " ctrl_" << name_s << "(ctrl_data_" << name_s << ")" << _line_suffix;
                }

                // Permute command.
                os << _line_prefix << "real_vec_permute2(" << pv_name << ", ctrl_" << name_s << ", ";
                if (use12)
                    os << mv1_name << ", " << mv2_name;
                else
                    os << mv2_name << ", " << mv1_name;
                os << ")" << _line_suffix;

                // Done w/the found elems.
                for (auto fei = found_elems.begin(); fei != found_elems.end(); fei++) {
                    size_t fe = *fei;
                    done_elems.insert(fe);
                }

                // Since there is no mask, we have to quit after one instruction.
                return;

            } // found.

        } // mj.
    } // mi.
}


// Print construction for one unaligned vector pv_name at gp.
void CppIntrinPrintHelper::print_unaligned_vec_ctor(ostream& os,
                                                 const VarPoint& gp,
                                                 const string& pv_name) {

    // Create an explanatory comment by printing the straightforward
    // code in a comment.
    print_unaligned_vec_simple(os, gp, pv_name, " // ");

    // List of elements for this vec block.
    auto& elems = _vv._vblk2elem_lists[gp];
    size_t nelems = elems.size();

    // Set of elements in gp that have been constructed.
    set<size_t> done_elems;

    // Set of aligned vec blocks that overlap with this unaligned vec block.
    const auto& aligned_vecs = _vv._vblk2avblks[gp];

    // Brute-force, greedy algorithm:
    // Want to construct this vector as efficiently as possible.
    // Try to get as many elements at once as possible.
    // For each target number of elements (most to least), try various strategies.

    // Loop through decreasing numbers of elements.
    for (size_t nelems_target = nelems;
         done_elems.size() < nelems && nelems_target > 0;
         nelems_target--) {

        try_strategies(os, pv_name,
                      nelems_target, elems, done_elems, aligned_vecs);

    } // decreasing number of target elements for next attempt.

    // Check that all elements are done and add any missing ones.
    size_t ndone = done_elems.size();
    if (ndone != nelems) {
        os << " // Note: could not create the following " << (nelems-ndone) <<
            " out of " << nelems <<
            " elements for unaligned vector with intrinsics." << endl;
        print_unaligned_vec_simple(os, gp, pv_name, _line_prefix, &done_elems);
    }
}
