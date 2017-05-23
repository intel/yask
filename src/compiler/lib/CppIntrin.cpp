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

// Implementation of methods in CppIntrin.hpp.

#include "CppIntrin.hpp"

// Try to use align instruction(s) to construct nelemsTarget elements
// per instruction.
void CppIntrinPrintHelper::tryAlign(ostream& os,
                                    const string& pvName,
                                    size_t nelemsTarget, 
                                    const VecElemList& elems, 
                                    set<size_t>& doneElems,
                                    const GridPointSet& alignedVecs,
                                    bool maskAllowed) {
    size_t nelems = elems.size();

    // Find case(s) that can use valignd.
    // Try all possible combinations of 2 aligned vectors, including
    // each vector paired w/itself.
    for (auto mi = alignedVecs.begin(); mi != alignedVecs.end(); mi++) {
        auto& mv1 = *mi;
        for (auto mj = alignedVecs.begin(); mj != alignedVecs.end(); mj++) {
            auto& mv2 = *mj;

            // Look up existing input vars.
            // All should be ready before this function is called.
            assert(_readyPoints.count(mv1));
            string mv1Name = _readyPoints[mv1];
            assert(_readyPoints.count(mv2));
            string mv2Name = _readyPoints[mv2];
                
            // Try all possible shift amounts.
            for (size_t count = 1;
                 doneElems.size() < nelems && count < nelems; count++) {
#ifdef DEBUG_SHIFT
                cout << " //** Checking shift of " << mv1Name <<
                    " and " << mv2Name << " by " << count << " elements." << endl;
#endif

                // Check each position i in output to see if it's correct.
                size_t nused1 = 0, nused2 = 0;
                set<size_t> foundElems;
                unsigned mask = 0;
                for (size_t i = 0; i < nelems; i++) {
                    size_t p = i + count;

                    // Ignore elements that are already constructed.
                    if (doneElems.count(i))
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
                                foundElems.insert(i); // remember bit.
#ifdef DEBUG_SHIFT
                                cout << " // ** " << pvName << "[" << i << "] = " <<
                                    mv1Name << "[" << p << "]" << endl;
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
                                foundElems.insert(i); // remember bit.
#ifdef DEBUG_SHIFT
                                cout << " // ** " << pvName << "[" << i << "] = " <<
                                    mv2Name << "[" << p << "]" << endl;
#endif
                            }
                        }
                    }
                }

                // Now, nused1 is the number of elements that can be shifted from mv1,
                // and  nused2 is the number of elements that can be shifted from mv2.
                assert(nused1 + nused2 == foundElems.size());

                // Have we found the target number of bits?
                if (foundElems.size() == nelemsTarget) {

                    // We don't need to use the mask if no elements are already done.
                    bool needMask = doneElems.size() > 0;

                    // If we can't use masking, must stop here.
                    if (needMask && !maskAllowed)
                        return;
                            
                    // 2-var shift.
                    if (nused1 && nused2) {
                        os << " // Get " << nused1 << " element(s) from " << mv1Name <<
                            " and " << nused2 << " from " << mv2Name << "." << endl;
                        os << _linePrefix << "real_vec_align";
                        if (needMask)
                            os << "_masked";
                        os << "<" << count << ">(" << pvName <<
                            ", " << mv1Name << ", " << mv2Name;
                        if (needMask)
                            printMask(os, mask);
                        os << ")" << _lineSuffix;
                    }

                    // 1-var shift (rotate).
                    else if (nused1) {
                        os << " // Get " << nused1 << " element(s) from " << mv1Name <<
                            "." << endl;
                        os << _linePrefix << "real_vec_align";
                        if (needMask)
                            os << "_masked";
                        os << "<" << count << ">(" << pvName << ", " << mv1Name << ", " << mv1Name;
                        if (needMask)
                            printMask(os, mask);
                        os << ")" << _lineSuffix;
                    }
                    else {
                        assert(nused2);
                        os << " // Get " << nused2 << " element(s) from " << mv2Name <<
                            "." << endl;
                        os << _linePrefix << "real_vec_align";
                        if (needMask)
                            os << "_masked";
                        os << "<" << count << ">(" << pvName << ", " << mv2Name << ", " << mv2Name;
                        if (needMask)
                            printMask(os, mask);
                        os << ")" << _lineSuffix;
                    }

                    // Done w/the found elems.
                    for (auto fei = foundElems.begin(); fei != foundElems.end(); fei++) {
                        size_t fe = *fei;
                        doneElems.insert(fe);
                    }

                } // found.
            }
        }
    } // all combos of 2 aligned vecs.
}

// Try to use 1-var permute instruction(s) to construct nelemsTarget elements
// per instruction.
void CppIntrinPrintHelper::tryPerm1(ostream& os,
                                    const string& pvName,
                                    size_t nelemsTarget, 
                                    const VecElemList& elems, 
                                    set<size_t>& doneElems,
                                    const GridPointSet& alignedVecs) {
    size_t nelems = elems.size();
    
    // Try a permute of each aligned vector.
    for (auto mi = alignedVecs.begin(); mi != alignedVecs.end(); mi++) {
        auto mv = *mi;

        // Look up existing input var.
        assert(_readyPoints.count(mv));
        string mvName = _readyPoints[mv];

        // Create the permute control and mask vars.
        ostringstream nameSS, ctrlSS;
        unsigned mask = 0;
        bool needNA = false;
        set<size_t> foundElems;

        // Loop through each element in the unaligned vector.
        for (size_t i = 0; i < nelems; i++) {

            // What aligned vector and offset does it come from?
            auto ve = elems[i];

            // String separators.
            if (i > 0) {
                nameSS << "_";
                ctrlSS << ", ";
            }
                    
            // Is i needed (not done) AND does i come from this mem vec?
            // If so, we want to permute it from the correct offset.
            if (doneElems.count(i) == 0 && ve._vec == mv) {
                int alignedElem = ve._offset; // get this element.

                nameSS << alignedElem;
                ctrlSS << alignedElem;
                mask |= (1 << i); // set proper bit.
                foundElems.insert(i); // remember element index.
            }

            // Not from this mem vec.
            else {
                nameSS << "NA";
                ctrlSS << "NA";
                needNA = true;
            }
        }

        // Have we found the target number of bits?
        if (foundElems.size() == nelemsTarget) {

            // We don't need to use the mask if no elements are already done.
            bool needMask = doneElems.size() > 0;

            string nameStr = nameSS.str();
            string ctrlStr = "{ .ci = { " + ctrlSS.str() + " } }"; // assignment to 'i' array.
                
            // Create NA var if needed (this is just for clarity).
            if (needNA)
                makeNA(os);
                
            // Create control if needed.
            if (definedCtrls.count(nameStr) == 0) {
                definedCtrls.insert(nameStr);
                os << _linePrefix << "static const " << getVarType() <<
                    " ctrl_" << nameStr << " = " << ctrlStr << _lineSuffix;
            }
                
            // Permute command.
            os << _linePrefix << "real_vec_permute";
            if (needMask)
                os << "_masked";
            os << "(" << pvName << ", ctrl_" << nameStr << ", " << mvName;
            if (needMask)
                printMask(os, mask);
            os << ")" << _lineSuffix;

            // Done w/the found elems.
            for (auto fei = foundElems.begin(); fei != foundElems.end(); fei++) {
                size_t fe = *fei;
                doneElems.insert(fe);
            }
        } // found.
    } // aligned vectors.
}

// Try to use 2-var permute instruction(s) to construct nelemsTarget elements
// per instruction.
void CppIntrinPrintHelper::tryPerm2(ostream& os,
                                    const string& pvName,
                                    size_t nelemsTarget, 
                                    const VecElemList& elems, 
                                    set<size_t>& doneElems,
                                    const GridPointSet& alignedVecs) {
    size_t nelems = elems.size();

    // There is no source-preserving mask version of permutex2var, so
    // we bail if any elements have already been found.
    if (doneElems.size() > 0)
        return;

    // Find case(s) that can use perm2.  Try all possible combinations
    // of 2 aligned vectors, but NOT including each vector paired
    // w/itself. (For that, we can use perm1.)
    for (auto mi = alignedVecs.begin(); mi != alignedVecs.end(); mi++) {
        auto& mv1 = *mi;
        for (auto mj = alignedVecs.begin(); mj != alignedVecs.end(); mj++) {
            auto& mv2 = *mj;

            if (mv1 == mv2) continue;

            // Look up existing input vars.
            // All should be ready before this function is called.
            assert(_readyPoints.count(mv1));
            string mv1Name = _readyPoints[mv1];
            assert(_readyPoints.count(mv2));
            string mv2Name = _readyPoints[mv2];
                
            // Create the permute control var.
            ostringstream nameSS, ctrlSS;
            bool needNA = false;
            set<size_t> foundElems;

            // Loop through each element in the unaligned vector.
            for (size_t i = 0; i < nelems; i++) {

                // What aligned vector and offset does it come from?
                auto ve = elems[i];

                // String separators.
                if (i > 0) {
                    nameSS << "_";
                    ctrlSS << ", ";
                }
                    
                // Is i needed (not done) AND does i come from one of the mem vecs?
                // If so, we want to permute it from the correct offset.
                if (doneElems.count(i) == 0 && 
                    (ve._vec == mv1 || ve._vec == mv2)) {
                    bool useA = (ve._vec == mv1); // first vec?
                    char alignedVec = useA ? 'A' : 'B';
                    int alignedElem = ve._offset; // get this element.
                        
                    nameSS << alignedVec << alignedElem;
                    if (!useA)
                        ctrlSS << "ctrl_sel_bit |"; // set selector bit for vec B.
                    ctrlSS << alignedElem;
                    foundElems.insert(i); // remember element index.
                }
                    
                // Not from either mem vec.
                else {
                    nameSS << "NA";
                    ctrlSS << "NA";
                    needNA = true;
                }
            }

            // Have we found the target number of bits?
            if (foundElems.size() == nelemsTarget) {
                assert(doneElems.size() == 0);

                string nameStr = nameSS.str();
                string ctrlStr = "{ .ci = { " + ctrlSS.str() + " } }";
                
                // Create NA var if needed (this is just for clarity).
                if (needNA)
                    makeNA(os);
                
                // Create control if needed.
                if (definedCtrls.count(nameStr) == 0) {
                    definedCtrls.insert(nameStr);
#if 1
                    os << _linePrefix << "const " << getVarType() <<
                        "_data ctrl_data_" << nameStr << " = " << ctrlStr << _lineSuffix;
                    os << _linePrefix << "const " << getVarType() <<
                        " ctrl_" << nameStr << "(ctrl_data_" << nameStr << ")" << _lineSuffix;
#else
                    os << _linePrefix << "const " << getVarType() <<
                        "_data ctrl_" << nameStr << " = " << ctrlStr << _lineSuffix;
#endif
                }
                
                // Permute command.
                os << _linePrefix << "real_vec_permute2(" << pvName << ", ctrl_" << nameStr << ", " <<
                    mv1Name << ", " << mv2Name << ")" << _lineSuffix;

                // Done w/the found elems.
                for (auto fei = foundElems.begin(); fei != foundElems.end(); fei++) {
                    size_t fe = *fei;
                    doneElems.insert(fe);
                }

                // Since there is no mask, we have to quit after one instruction.
                return;
                    
            } // found.
                
        } // mj.
    } // mi.
}
    
    
// Print construction for one unaligned vector pvName at gp.
void CppIntrinPrintHelper::printUnalignedVecCtor(ostream& os,
                                                 const GridPoint& gp,
                                                 const string& pvName) {

    // Create an explanatory comment by printing the straightforward
    // code in a comment.
    printUnalignedVecSimple(os, gp, pvName, " // ");

    // List of elements for this vec block.
    auto& elems = _vv._vblk2elemLists[gp];
    size_t nelems = elems.size();

    // Set of elements in gp that have been constructed.
    set<size_t> doneElems;

    // Set of aligned vec blocks that overlap with this unaligned vec block.
    const auto& alignedVecs = _vv._vblk2avblks[gp];

    // Brute-force, greedy algorithm:
    // Want to construct this vector as efficiently as possible.
    // Try to get as many elements at once as possible.
    // For each target number of elements (most to least), try various strategies.

    // Loop through decreasing numbers of elements.
    for (size_t nelemsTarget = nelems;
         doneElems.size() < nelems && nelemsTarget > 0; 
         nelemsTarget--) {

        tryStrategies(os, pvName,
                      nelemsTarget, elems, doneElems, alignedVecs);

    } // decreasing number of target elements for next attempt.

    // Check that all elements are done.
    size_t ndone = doneElems.size();
    if (ndone != nelems) {
        os << " // Note: could not create the following " << (nelems-ndone) <<
            " out of " << nelems <<
            " elements for unaligned vector with intrinsics." << endl;
        printUnalignedVecSimple(os, gp, pvName, _linePrefix, &doneElems);
    }
}
