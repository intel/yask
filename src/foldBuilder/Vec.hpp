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

//////////// Basic vector classes /////////////

#ifndef VEC_HPP
#define VEC_HPP

#include "Print.hpp"

using namespace std;

//#define DEBUG_VV   // debug VecInfoVisitor.

// One element in a vector.
class VecElem {
public:
    GridPoint _vec;      // starting index of vector containing this element.
    size_t _offset;            // 1-D offset in _vec.
    IntTuple _offsets;         // n-D offsets 

    VecElem(const GridPoint& vec, int offset, const IntTuple& offsets) :
        _vec(vec), _offset(offset), _offsets(offsets) { }

    virtual VecElem& operator=(const VecElem& rhs) {
        _vec = rhs._vec;
        _offset = rhs._offset;
        _offsets = rhs._offsets;
        return *this;
    }
    virtual bool operator==(const VecElem& rhs) const {
        return _vec == rhs._vec && _offset == rhs._offset;
    }
    virtual bool operator!=(const VecElem& rhs) const {
        return !operator==(rhs);
    }
};

// STL vector of vector block elements.
// The STL vector index is the HW vector index.
// Each entry indicates in what aligned vector and offset
// the element can be found.
typedef vector<VecElem> VecElemList;

// Layout of vector blocks to VecElemVecs.
typedef map<GridPoint, VecElemList> Point2VecElemLists;

// Layout of vector blocks to aligned blocks.
typedef map<GridPoint, GridPointSet> Point2Vecs;

// This visitor determines the vector blocks needed to calculate the stencil.
// It doesn't actually generate code; it just collects info from the AST.
// After the AST has been visited, the object can be used to print statistics
// or used to provide data to a print visitor.
class VecInfoVisitor : public ExprVisitor {
protected:

    const Dimensions& _dims;
    int _vlen;                   // size of one vector.
    
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

    VecInfoVisitor(const Dimensions& dims) :
        _dims(dims) {
        _vlen = dims._fold.product();
    }

    const IntTuple& getFold() const {
        return _dims._fold;
    }
    
    size_t getNumPoints() const {
        return _vblk2elemLists.size();
    }

    size_t getNumAlignedVecs() const {
        return _alignedVecs.size();
    }

    // Make an index and offset canonical, i.e.,
    // find indexOut and offsetOut such that
    // (indexOut * vecLen) + offsetOut == (indexIn * vecLen) + offsetIn
    // and offsetOut is in [0..vecLen-1].
    // Makes proper adjustments for negative inputs.
    virtual void fixIndexOffset(int indexIn, int offsetIn,
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
    virtual void printStatsHeader(ostream& os, string separator) const {
        os << "destination grid" <<
            separator << "num vector elems" <<
            separator << "fold" <<
            separator << "num points in stencil" <<
            separator << "num aligned vectors to read from memory" <<
            separator << "num blends needed" <<
            separator << _dims._fold.makeDimStr(separator, "footprint in ") <<
            endl;
    }

    // Print some stats for the current values.
    // Pre-requisite: visitor has been accepted.
    virtual void printStats(ostream& os, const string& destGrid, const string& separator) const {

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

        // calc footprint in each dim.
        map<const string*, int> footprints;
        for (auto* dim : _dims._fold.getDims()) {

            // Create direction vector in this dim.
            IntTuple dir;
            dir.addDimBack(dim, 1);

            // Make set of aligned vecs projected in this dir.
            set<IntTuple> footprint;
            for (auto vb : _alignedVecs) {
                IntTuple proj = vb.removeDimInDir(dir);
                footprint.insert(proj);
            }
            footprints[dim] = footprint.size();
        }
            
        os << destGrid <<
            separator << _vlen <<
            separator << _dims._fold.makeValStr("x") <<
            separator << getNumPoints() <<
            separator << getNumAlignedVecs() <<
            separator << numBlends;
        for (auto dim : _dims._fold.getDims())
            os << separator << footprints[dim];
        os << endl;
    }

    // Get the set of aligned vectors on the leading edge
    // in the given direction and magnitude in dir.
    // Pre-requisite: visitor has been accepted.
    virtual void getLeadingEdge(GridPointSet& edge, const IntTuple& dir) const {
        edge.clear();

        // Repeat based on magnitude (cluster step in given dir).
        for (int i = 0; i < dir.getDirVal(); i++) {
        
            // loop over aligned vectors.
            for (auto avi : _alignedVecs) {

                // ignore values already found.
                if (edge.count(avi))
                    continue;

                // ignore if this vector doesn't have a dimension in dir.
                if (!avi.lookup(dir.getDirName()))
                    continue;

                // compare to all points.
                bool best = true;
                for (auto avj : _alignedVecs) {

                    // ignore values already found.
                    if (edge.count(avj))
                        continue;

                    // Determine if avj is ahead of avi in given direction.
                    // (A point won't be ahead of itself.)
                    if (avj.isAheadOfInDir(avi, dir))
                        best = false;
                }

                // keep only if farthest.
                if (best)
                    edge.insert(avi);
            }
        }
    }

    // Only want to visit the RHS of an eqGroup.
    // Assumes LHS is aligned.
    // TODO: validate this.
    virtual void visit(EqualsExpr* ee) {
        ee->getRhs()->accept(this);      
    }
    
    // Called when a grid point is read in a stencil function.
    virtual void visit(GridPoint* gp) {

        // Don't vectorize parameters.
        if (gp->isParam())
            return;

        // Already seen this point?
        if (_vblk2elemLists.count(*gp) > 0)
            return;

        // Vec of points to calculate.
#ifdef DEBUG_VV
        cout << "vec @ " << gp->makeDimValStr() << " => " << endl;
#endif

        // Loop through all points in the vector at this cluster point.
        size_t pelem = 0;
        _dims._fold.visitAllPoints([&](const IntTuple& vecPoint){

                // Offset in each dim is starting point of grid point plus
                // offset in this vector.
                // Note: there may be more or fewer dims in vecPoint than in grid point.
                auto offsets = gp->addElements(vecPoint, false);

                // Find aligned vector indices and offsets
                // for this one point.
                IntTuple vecOffsets, vecLocation;
                for (auto dim : offsets.getDims()) {

                    // length of this dimension in fold, if it exists.
                    const int* p = _dims._fold.lookup(dim);
                    int len = p ? *p : 1;

                    // convert this offset to vector index and vector offset.
                    int vecIndex, vecOffset;
                    fixIndexOffset(0, offsets.getVal(dim), vecIndex, vecOffset, len);
                    vecOffsets.addDimBack(dim, vecOffset);
                    vecLocation.addDimBack(dim, vecIndex * len);
                }
#ifdef DEBUG_VV
                cout << " element @ " << offsets.makeDimValStr() << " => " <<
                    " vec-location @ " << vecLocation.makeDimValStr() <<
                    " & vec-offsets @ " << vecOffsets.makeDimValStr() <<
                    " => " << endl;
#endif
                    
                // Create aligned vector block that contains this point.
                GridPoint alignedVec(gp, vecLocation);

                // Find linear offset within this aligned vector block.
                int alignedElem = _dims._fold.layout(vecOffsets, false);
                assert(alignedElem >= 0);
                assert(alignedElem < _vlen);
#ifdef DEBUG_VV
                cout << "  general-" << gp->makeStr() << "[" << pelem << "] = aligned-" <<
                    alignedVec.makeStr() << "[" << alignedElem << "]" << endl;
#endif

                // Update set of all aligned vec-blocks.
                _alignedVecs.insert(alignedVec);

                // Update set of aligned vec-blocks and elements needed for this vec-block element.
                _vblk2avblks[*gp].insert(alignedVec);

                // Save which aligned vec-block's element is needed for this vec-block element.
                VecElem ve(alignedVec, alignedElem, offsets);
                _vblk2elemLists[*gp].push_back(ve); // should be at pelem index.
                assert(_vblk2elemLists[*gp].size() == pelem+1); // verify at pelem index.

                pelem++;
            });                  // end of vector lambda-function.
    }                   // end of visit() method.
};

// Define methods for printing a vectorized version of the stencil.
class VecPrintHelper : public PrintHelper {
protected:
    VecInfoVisitor& _vv;
    bool _allowUnalignedLoads;
    bool _reuseVars; // if true, load to a local var; else, reload on every access.
    bool _definedNA;           // NA var defined.
    map<GridPoint, string> _readyPoints; // points that are already constructed.

    // Print access to an aligned vector block.
    // Return var name.
    virtual string printAlignedVecRead(ostream& os, const GridPoint& gp) =0;

    // Print unaliged memory read.
    // Assumes this results in same values as printUnalignedVec().
    virtual string printUnalignedVecRead(ostream& os, const GridPoint& gp) =0;
    
    // Print write to an aligned vector block.
    // Return expression written.
    virtual string printAlignedVecWrite(ostream& os, const GridPoint& gp,
                                        const string& val) =0;

    // Print conversion from existing vars to make an unaligned vector block.
    // Return var name.
    virtual string printUnalignedVec(ostream& os, const GridPoint& gp) =0;

    // Print construction for one point var pvName from elems.
    virtual void printUnalignedVecCtor(ostream& os, const GridPoint& gp,
                                       const string& pvName) =0;

public:
    VecPrintHelper(VecInfoVisitor& vv,
                   bool allowUnalignedLoads,
                   const CounterVisitor* cv,
                   const string& varPrefix,
                   const string& varType,
                   const string& linePrefix,
                   const string& lineSuffix,
                   bool reuseVars = true) :
        PrintHelper(cv, varPrefix, varType, linePrefix, lineSuffix),
        _vv(vv), _allowUnalignedLoads(allowUnalignedLoads),
        _reuseVars(reuseVars), _definedNA(false) { }
    virtual ~VecPrintHelper() {}

    // get fold info.
    virtual const IntTuple& getFold() const {
        return _vv.getFold();
    }

    // Add a N/A var, just for readability.
    virtual void makeNA(ostream& os) {
        if (!_definedNA) {
            os << _linePrefix << "const int NA = 0; // indicates element not used." << endl;
            _definedNA = true;
        }
    }
    
    // Print any needed memory reads and/or constructions.
    // Return var name.
    virtual string readFromPoint(ostream& os, const GridPoint& gp) {

        string varName;

        // Already done.
        if (_reuseVars && _readyPoints.count(gp))
            varName = _readyPoints[gp]; // do nothing.

        // An aligned vector block?
        else if (_vv._alignedVecs.count(gp))
            varName = printAlignedVecRead(os, gp);

        // Unaligned loads allowed?
        else if (_allowUnalignedLoads)
            varName = printUnalignedVecRead(os, gp);

        // Need to construct an unaligned vector block?
        else if (_vv._vblk2elemLists.count(gp)) {

            // make sure prerequisites exist by recursing.
            auto avbs = _vv._vblk2avblks[gp];
            for (auto pi = avbs.begin(); pi != avbs.end(); pi++) {
                auto& p = *pi;
                readFromPoint(os, p);
            }

            // output this construction.
            varName = printUnalignedVec(os, gp);
        }

        else {
            cerr << "Error: on point " << gp.makeStr() << endl;
            assert("point type unknown");
        }

        // Remember this point and return its name.
        _readyPoints[gp] = varName;
        return varName;
    }

    // Update a grid point.
    // The 'os' parameter is provided for derived types that
    // need to write intermediate code to a stream.
    virtual string writeToPoint(ostream& os, const GridPoint& gp, const string& val) {
        printAlignedVecWrite(os, gp, val);
        return "";
    }
};

// A visitor that reorders exprs.
class ExprReorderVisitor : public ExprVisitor {
protected:
    VecInfoVisitor& _vv;

public:
    ExprReorderVisitor(VecInfoVisitor& vv) :
        _vv(vv) { }
    virtual ~ExprReorderVisitor() {}
                       
    // Sort a commutative expression.
    virtual void visit(CommutativeExpr* ce) {

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
    }
};

#endif
