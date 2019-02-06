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

//////////// Basic vector classes /////////////

#include "Vec.hpp"

namespace yask {

    // Determine vectorizability information about this grid point.
    // Called when a grid point is read in a stencil function.
    string VecInfoVisitor::visit(GridPoint* gp) {

        // Nothing to do if this grid-point is not vectorizable.
        if (gp->getVecType() == GridPoint::VEC_NONE) {
#ifdef DEBUG_VV
            cout << " //** cannot vectorize scalar-access " << gp->makeQuotedStr() << endl;
#endif
            _scalarPoints.insert(*gp);
            return "";
        }
        else if (gp->getVecType() == GridPoint::VEC_PARTIAL) {
#ifdef DEBUG_VV
            cout << " //** cannot vectorize non-standard-access " << gp->makeQuotedStr() << endl;
#endif
            _nonVecPoints.insert(*gp);
            return "";
        }
        assert(gp->getVecType() == GridPoint::VEC_FULL);

        // Already seen this point?
        if (_vecPoints.count(*gp) > 0) {
            assert(_vblk2elemLists.count(*gp) > 0);
            assert(_vblk2avblks.count(*gp) > 0);
            return "";
        }
        assert(_vblk2elemLists.count(*gp) == 0);
        assert(_vblk2avblks.count(*gp) == 0);

        // Vec of points to calculate.
#ifdef DEBUG_VV
        cout << " //** vec @ " << gp->makeQuotedStr() << " => " << endl;
#endif

        // Loop through all points in the vector fold.
        _dims._fold.visitAllPoints([&](const IntTuple& vecPoint,
                                       size_t pelem){

                // Final offset in each dim is offset of grid point plus
                // fold offset.
                // This works because we know this grid point is accessed
                // only by simple offsets in each foldable dim.
                // Note: there may be more or fewer dims in vecPoint than in grid point.
                auto offsets = gp->getArgOffsets().addElements(vecPoint, false);

                // Find aligned vector indices and offsets
                // for this one point.
                IntTuple vecOffsets, vecLocation;
                for (auto& dim : offsets.getDims()) {
                    auto& dname = dim.getName();

                    // length of this dimension in fold, if it exists.
                    const int* p = _dims._fold.lookup(dname);
                    int len = p ? *p : 1;

                    // convert this offset to vector index and vector offset.
                    int vecIndex, vecOffset;
                    fixIndexOffset(0, dim.getVal(), vecIndex, vecOffset, len);
                    vecOffsets.addDimBack(dname, vecOffset);
                    vecLocation.addDimBack(dname, vecIndex * len);
                }
#ifdef DEBUG_VV
                cout << "  //** element @ " << offsets.makeDimValStr() << " => " <<
                    " vec-location @ " << vecLocation.makeDimValStr() <<
                    " & vec-offsets @ " << vecOffsets.makeDimValStr() <<
                    " => " << endl;
#endif

                // Create aligned vector block that contains this point.
                GridPoint alignedVec = *gp;  // copy original.
                alignedVec.setArgOffsets(vecLocation);

                // Find linear offset within this aligned vector block.
                int alignedElem = _dims._fold.layout(vecOffsets, false);
                assert(alignedElem >= 0);
                assert(alignedElem < _vlen);
#ifdef DEBUG_VV
                cout << "   //** " << gp->makeStr() << "[" << pelem << "] = aligned-" <<
                    alignedVec.makeStr() << "[" << alignedElem << "]" << endl;
#endif

                // Update set of *all* aligned vec-blocks.
                _alignedVecs.insert(alignedVec);

                // Update set of aligned vec-blocks and elements needed for *this* vec-block element.
                _vblk2avblks[*gp].insert(alignedVec);

                // Save which aligned vec-block's element is needed for this vec-block element.
                VecElem ve(alignedVec, alignedElem, offsets);
                _vblk2elemLists[*gp].push_back(ve); // should be at pelem index.
                assert(_vblk2elemLists[*gp].size() == pelem+1); // verify at pelem index.

                return true;
            });                  // end of vector lambda-function.

        // Mark as done.
        _vecPoints.insert(*gp);
        return "";
    }                   // end of visit() method.

    // Return code containing a vector of grid points, e.g., code fragment
    // or var name.  Optionally print memory reads and/or constructions to
    // 'os' as needed.
    string VecPrintHelper::readFromPoint(ostream& os, const GridPoint& gp) {

        string codeStr;

        // Already done and saved.
        if (_reuseVars && _vecVars.count(gp))
            codeStr = _vecVars[gp]; // do nothing.

        // Scalar GP?
        else if (gp.getVecType() == GridPoint::VEC_NONE) {
#ifdef DEBUG_GP
            cout << " //** reading from point " << gp.makeStr() << " as scalar.\n";
#endif
            codeStr = readFromScalarPoint(os, gp);
        }

        // Non-scalar but non-vectorizable GP?
        else if (gp.getVecType() == GridPoint::VEC_PARTIAL) {
#ifdef DEBUG_GP
            cout << " //** reading from point " << gp.makeStr() << " as partially vectorized.\n";
#endif
            codeStr = printNonVecRead(os, gp);
        }

        // Everything below this should be VEC_FULL.

        // An aligned vector block?
        else if (_vv._alignedVecs.count(gp)) {
#ifdef DEBUG_GP
            cout << " //** reading from point " << gp.makeStr() << " as fully vectorized and aligned.\n";
#endif
            codeStr = printAlignedVecRead(os, gp);
        }

        // Unaligned loads allowed?
        else if (_settings._allowUnalignedLoads) {
#ifdef DEBUG_GP
            cout << " //** reading from point " << gp.makeStr() << " as fully vectorized and unaligned.\n";
#endif
            codeStr = printUnalignedVecRead(os, gp);
        }

        // Need to construct an unaligned vector block?
        else if (_vv._vblk2elemLists.count(gp)) {
#ifdef DEBUG_GP
            cout << " //** reading from point " << gp.makeStr() << " as fully vectorized and unaligned.\n";
#endif

            // make sure prerequisites exist by recursing.
            auto avbs = _vv._vblk2avblks[gp];
            for (auto pi = avbs.begin(); pi != avbs.end(); pi++) {
                auto& p = *pi;
                readFromPoint(os, p);
            }

            // output this construction.
            codeStr = printUnalignedVec(os, gp);
        }

        else {
            THROW_YASK_EXCEPTION("Internal error: type unknown for point " + gp.makeStr());
        }

        // Remember this point and return it.
        if (codeStr.length())
            savePointVar(gp, codeStr);
        return codeStr;
    }

    // Print any immediate memory writes to 'os'.
    // Return code to update a vector of grid points or null string
    // if all writes were printed.
    string VecPrintHelper::writeToPoint(ostream& os, const GridPoint& gp, const string& val) {

        // NB: currently, all eqs must be vectorizable on LHS,
        // so we only need to handle vectorized writes.
        // TODO: relax this restriction.

        printAlignedVecWrite(os, gp, val);
        return "";
    }

    // Sort a commutative expression.
    string ExprReorderVisitor::visit(CommutativeExpr* ce) {

        auto& oev = ce->getOps(); // old exprs.
        NumExprPtrVec nev; // new exprs.

        // Simple, greedy algorithm:
        // Select first element that needs the fewest new aligned vecs.
        // Repeat until done.
        // TODO: sort based on all reused exprs, not just grid reads.

        GridPointSet alignedVecs; // aligned vecs needed so far.
        set<size_t> usedExprs; // expressions used.
        for (size_t i = 0; i < oev.size(); i++) {

#ifdef DEBUG_SORT
            cout << "  Looking for expr #" << i << "..." << endl;
#endif

            // Scan unused exprs.
            size_t jBest = 0;
            size_t jBestCost = size_t(-1);
            GridPointSet jBestAlignedVecs;
            for (size_t j = 0; j < oev.size(); j++) {
                if (usedExprs.count(j) == 0) {

                    // This unused expr.
                    auto& expr = oev[j];

                    // Get aligned vecs needed for this expr.
                    VecInfoVisitor tmpvv(_vv);
                    expr->accept(&tmpvv);
                    auto& tmpAlignedVecs = tmpvv._alignedVecs;

                    // Calculate cost.
                    size_t cost = 0;
                    for (auto k = tmpAlignedVecs.begin(); k != tmpAlignedVecs.end(); k++) {
                        auto& av = *k;

                        // new vector needed?
                        if (alignedVecs.count(av) == 0) {
#ifdef DEBUG_SORT
                            cout << " Vec " << av.makeStr("tmp") << " is new" << endl;
#endif
                            cost++;
                        }
                    }
#ifdef DEBUG_SORT
                    cout << " Cost of expr " << j << " = " << cost << endl;
#endif
                    // Best so far?
                    if (cost < jBestCost) {
                        jBestCost = cost;
                        jBest = j;
                        jBestAlignedVecs = tmpAlignedVecs;
#ifdef DEBUG_SORT
                        cout << "  Best so far has " << jBestAlignedVecs.size() << " aligned vecs" << endl;
#endif
                    }
                }
            }

            // Must have a best one.
            assert(jBestCost != size_t(-1));

            // Add it.
            nev.push_back(oev[jBest]);
            usedExprs.insert(jBest);

            // Remember used vectors.
            for (auto k = jBestAlignedVecs.begin(); k != jBestAlignedVecs.end(); k++) {
                alignedVecs.insert(*k);
            }
        }

        // Replace the old vector w/the new one.
        assert(nev.size() == oev.size());
        oev.swap(nev);
        return "";
    }

    // TODO: fix this old code and make it available as an output.
#if 0
    // Print stats for various folding options.
    if (vlenForStats) {
        string separator(",");
        VecInfoVisitor::printStatsHeader(cout, separator);

        // Loop through all grids.
        for (auto gp : grids) {

            // Loop through possible folds of given length.
            for (int xlen = vlenForStats; xlen > 0; xlen--) {
                for (int ylen = vlenForStats / xlen; ylen > 0; ylen--) {
                    int zlen = vlenForStats / xlen / ylen;
                    if (vlenForStats == xlen * ylen * zlen) {

                        // Create vectors needed to implement RHS.
                        VecInfoVisitor vv(xlen, ylen, zlen);
                        gp->visitExprs(&vv);

                        // Print stats.
                        vv.printStats(cout, gp->getName(), separator);
                    }
                }
            }
        }
    }
#endif

} // namespace yask.
