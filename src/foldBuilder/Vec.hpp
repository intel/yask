/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

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

#include "Visitor.hpp"

using namespace std;

//#define DEBUG_VV   // debug VecInfoVisitor.
//#define DEBUG_SORT

// Dimensions of a vector.
class VecDims : public Triple {
public:
    VecDims(const Triple& triple) :
        Triple(triple) {}
    VecDims(int xlen, int ylen, int zlen) :
        Triple(xlen, ylen, zlen) {}
    virtual ~VecDims() {}

    // Get overall vector length.
    virtual int getVecLen() const {
        return getLen();
    }
};

// One element in a vector.
class VecElem {
public:
    GridPoint _vec;
    size_t _offset;

    VecElem(const GridPoint& vec, int offset) :
        _vec(vec), _offset(offset) { }

    bool operator==(const VecElem& rhs) const {
        return _vec == rhs._vec && _offset == rhs._offset;
    }
    bool operator!=(const VecElem& rhs) const {
        return !operator==(rhs);
    }
};

// STL vector of vector block elements.
// The STL vector index is the HW vector index.
// Each entry indicates in what aligned vector and offset
// the element can be found.
typedef vector<VecElem> VecElemList;

// Map of vector blocks to VecElemVecs.
typedef map<GridPoint, VecElemList> Point2VecElemLists;

// Map of vector blocks to aligned blocks.
typedef map<GridPoint, GridPointSet> Point2Vecs;

// Determines vector blocks needed to calculate the stencil.
// This doesn't actually generate code, it just collects info from the AST.
// After the AST has been visited, the object can be used to print statistics
// or used to provide data to a print visitor.
class VecInfoVisitor : public VecDims, public ExprVisitor {
public:

    // Data on vector blocks.
    GridPointSet _alignedVecs; // set of aligned vectors, i.e., ones that need to be read from memory.
    Point2Vecs _vblk2avblks; // each vec block -> its constituent aligned vec blocks.
    Point2VecElemLists _vblk2elemLists; // each vec block -> in-order list of its constituent aligned vec blocks' elements.

    // NB: the above hold much of the same info, but arranged differently:
    // _alignedVecs contains only a set of aligned blocks.
    // _vblk2avblks is used to quickly determine what aligned blocks contribute to a given block.
    // _vblk2elemLists is used to find exactly where each element comes from.
    // The keys are the same for both maps.

    VecInfoVisitor(const Triple& triple) :
        VecDims(triple) {}
    VecInfoVisitor(int xlen, int ylen, int zlen) :
        VecDims(xlen, ylen, zlen) {}

    size_t getNumPoints() const {
        return _vblk2elemLists.size();
    }

    size_t getNumAlignedVecs() const {
        return _alignedVecs.size();
    }

    // Make an index and offset canonical, i.e.,
    // find indexOut and offsetOut such that
    // (indexOut * vecLen) + offsetOut = (indexIn * vecLen) + offsetIn
    // and offsetOut is in [0..vecLen-1].
    // Makes proper adjustments for negative inputs.
    static void fixIndexOffset(int indexIn, int offsetIn,
                               int& indexOut, int& offsetOut,
                               int vecLen) {
        const int ofs = (indexIn * vecLen) + offsetIn;
        indexOut = ofs / vecLen;
        offsetOut = ofs % vecLen;
        while(offsetOut < 0) {
            offsetOut += vecLen;
            indexOut--;
        }
    }

    // Print stats header.
    static void printStatsHeader(ostream& os, string separator) {
        os << "num vector elems" <<
            separator << "fold" <<
            separator << "num points in stencil" <<
            separator << "num aligned vectors to read from memory" <<
            separator << "num blends needed" <<
            separator << "footprint in x" <<
            separator << "footprint in y" <<
            separator << "footprint in z" <<
#ifdef NORM_STATS
            separator << "reads per elem" <<
            separator << "blends per elem" <<
            separator << "footprint in x per elem" <<
            separator << "footprint in y per elem" <<
            separator << "footprint in z per elem" <<
            separator << "reads per point" <<
            separator << "blends per point" <<
            separator << "footprint in x per point" <<
            separator << "footprint in y per point" <<
            separator << "footprint in z per point" <<
#endif
            endl;
    }

    // Print some stats for the current values.
    // Pre-requisite: visitor has been accepted.
    void printStats(ostream& os, string separator) const {

        // calc num blends needed.
        int numBlends = 0;
        for (auto i = _vblk2elemLists.begin(); i != _vblk2elemLists.end(); i++) {
            //auto pt = i->first;
            auto vev = i->second;
            GridPointSet mems; // inputs for this point.
            for (size_t j = 0; j < vev.size(); j++)
                mems.insert(vev[j]._vec);
            if (mems.size() > 1) // only need to blend if >1 inputs.
                numBlends += mems.size();
        }

        // calc footprints.
        typedef pair<int, int> Proj; // projection onto one plane, e.g., x-y.
        set<Proj> xs, ys, zs;
        for (auto i = _alignedVecs.begin(); i != _alignedVecs.end(); i++) {
            auto vb = *i;
            xs.insert(Proj(vb._j, vb._k)); // x footprint in y-z plane.
            ys.insert(Proj(vb._i, vb._k)); // y footprint in x-z plane.
            zs.insert(Proj(vb._i, vb._j)); // z footprint in x-y plane.
        }

        float readsPerElem = float(getNumAlignedVecs()) / getVecLen();
        float blendsPerElem = float(numBlends) / getVecLen();

        float readsPerPoint = float(getNumAlignedVecs()) / getNumPoints();
        float blendsPerPoint = float(numBlends) / getNumPoints();

        os << getVecLen() <<
            separator << getXLen() << 'x' << getYLen() << 'x' << getZLen() <<
            separator << getNumPoints() <<
            separator << getNumAlignedVecs() <<
            separator << numBlends <<
            separator << xs.size() <<
            separator << ys.size() <<
            separator << zs.size() << 
#ifdef NORM_STATS
            separator << readsPerElem  <<
            separator << blendsPerElem <<
            separator << (float(xs.size()) / getVecLen()) <<
            separator << (float(ys.size()) / getVecLen()) <<
            separator << (float(zs.size()) / getVecLen()) <<
            separator << readsPerPoint <<
            separator << blendsPerPoint <<
            separator << (float(xs.size()) / getNumPoints()) <<
            separator << (float(ys.size()) / getNumPoints()) <<
            separator << (float(zs.size()) / getNumPoints()) <<
#endif
            endl;
    }

    // Get the set of aligned vectors on the leading edge
    // in the given direction.
    // Pre-requisite: visitor has been accepted.
    virtual void getLeadingEdge(GridPointSet& edge, const Dir& dir) const {
        edge.clear();

        // loop over aligned vectors.
        for (auto avi : _alignedVecs) {

            // compare to all others.
            bool best = true;
            for (auto avj : _alignedVecs) {
                
                // determine if avj is ahead of avi in given direction.
                if (avj.isAheadOf(avi, dir))
                    best = false;
            }

            // keep only if farthest.
            if (best)
                edge.insert(avi);
        }
    }

    // Most work done where a grid point is read.
    virtual void visit(GridPoint* gp) {

        // Already seen this point?
        if (_vblk2elemLists.count(*gp) > 0)
            return;

        // Vec of points to calculate.
#ifdef DEBUG_VV
        cout << "vec @ u(" << gp->_t << ", " <<
            gp->_i << ", " << gp->_j << ", " << gp->_k << ") => " << endl;
#endif

        // loop through one vector-block's extent starting at gp's location.
        size_t pelem = 0;
        for (int k = 0; k < getZLen(); k++) {
            int kp = k + gp->_k;
            for (int j = 0; j < getYLen(); j++) {
                int jp = j + gp->_j;
                for (int i = 0; i < getXLen(); i++) {
                    int ip = i + gp->_i;

                    // Find individual source mem vectors and offsets.
                    int xvec, yvec, zvec;
                    int xpos, ypos, zpos;
                    fixIndexOffset(0, ip, xvec, xpos, getXLen());
                    fixIndexOffset(0, jp, yvec, ypos, getYLen());
                    fixIndexOffset(0, kp, zvec, zpos, getZLen());

                    // Flatten 3D mini-vectors into one aligned vector block w/offset.
                    GridPoint alignedVec(gp, xvec * getXLen(), yvec * getYLen(), zvec * getZLen());
                    int alignedElem = map321(xpos, ypos, zpos);

                    // Update set of all aligned vec-blocks.
                    _alignedVecs.insert(alignedVec);

                    // Update set of aligned vec-blocks and elements needed for this vec-block element.
                    _vblk2avblks[*gp].insert(alignedVec);

                    // Save which aligned vec-block's element is needed for this vec-block element.
                    VecElem ve(alignedVec, alignedElem);
                    _vblk2elemLists[*gp].push_back(ve); // should be at pelem index.
                    assert(_vblk2elemLists[*gp].size() == pelem+1); // verify at pelem index.

#ifdef DEBUG_VV
                    cout << " element u(" << gp->_t << ", " <<
                        ip << ", " << jp << ", " << kp << ") => " <<
                        "x[" << xvec << "," << xpos << "], " << 
                        "y[" << yvec << "," << ypos << "], " <<
                        "z[" << zvec << "," << zpos << "] => " << 
                        gp->makeStr("unaligned_vec") << "[" << pelem << "] = " <<
                        alignedVec.makeStr("aligned_vec") << "[" << alignedElem << "]" << endl;
#endif
                    pelem++;
                }
            }
        }
    }
};

// Define methods for printing a vectorized version of the stencil.
class VecPrintHelper : public PrintHelper {
protected:
    VecInfoVisitor& _vv;
    map<GridPoint, string> _readyPoints; // points that are already constructed.
    bool _reuseVars; // if true, load to a local var; else, reload on every access.
    bool _definedNA;           // NA var defined.

    // Print access to an aligned vector block.
    // Return var name.
    virtual string printAlignedVec(ostream& os, const GridPoint& gp) =0;

    // Print conversion from existing vars to make an unaligned vector block.
    // Return var name.
    virtual string printUnalignedVec(ostream& os, const GridPoint& gp) =0;

    // Print construction for one point var pvName from elems.
    virtual void printUnalignedVecCtor(ostream& os, const GridPoint& gp, const string& pvName) =0;

public:
    VecPrintHelper(VecInfoVisitor& vv, bool reuseVars = true) :
        PrintHelper("vec"), _vv(vv), _reuseVars(reuseVars), _definedNA(false) { }

    // Add a N/A var, just for readability.
    virtual void makeNA(ostream& os) {
        if (!_definedNA) {
            os << "const int NA = 0; // indicates element not used." << endl;
            _definedNA = true;
        }
    }
    
    // Print any needed memory reads and/or constructions.
    // Return var name.
    virtual string constructPoint(ostream& os, const GridPoint& gp) {

        string varName;

        // Already done.
        if (_reuseVars && _readyPoints.count(gp))
            varName = _readyPoints[gp]; // do nothing.

        // An aligned vector block?
        else if (_vv._alignedVecs.count(gp))
            varName = printAlignedVec(os, gp);

        // An unaligned vector block?
        else if (_vv._vblk2elemLists.count(gp)) {

            // make sure prerequisites exist by recursing.
            auto avbs = _vv._vblk2avblks[gp];
            for (auto pi = avbs.begin(); pi != avbs.end(); pi++) {
                auto& p = *pi;
                constructPoint(os, p);
            }

            // output this construction.
            varName = printUnalignedVec(os, gp);
        }

        else {
            cerr << "error: on point " << gp.makeStr() << endl;
            assert("point type unknown");
        }

        // Remember this point and return its name.
        _readyPoints[gp] = varName;
        return varName;
    }

    // Sort commutative expr to a more optimized order.
    virtual void sortCommutativeExpr(CommutativeExpr& oce) const {

        ExprPtrVec& oev = oce._ops; // old exprs.
        ExprPtrVec nev; // new exprs.

        // Simple, greedy algorithm:
        // Select first element that needs the fewest new aligned vecs.
        // Repeat until done.

        GridPointSet alignedVecs; // aligned vecs needed so far.
        set<size_t> usedExprs; // expressions used.
        for (size_t i = 0; i < oev.size(); i++) {

#ifdef DEBUG_SORT
            cerr << "  Looking for expr #" << i << "..." << endl;
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
                            cerr << " Vec " << av.makeStr("tmp") << " is new" << endl;
#endif
                            cost++; 
                        }
                    }
#ifdef DEBUG_SORT
                    cerr << " Cost of expr " << j << " = " << cost << endl;
#endif
                    // Best so far?
                    if (cost < jBestCost) {
                        jBestCost = cost;
                        jBest = j;
                        jBestAlignedVecs = tmpAlignedVecs;
#ifdef DEBUG_SORT
                        cerr << "  Best so far has " << jBestAlignedVecs.size() << " aligned vecs" << endl;
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
    }
};

