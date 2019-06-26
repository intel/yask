/*****************************************************************************

YASK: Yet Another Stencil Kit
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

#ifndef VEC_HPP
#define VEC_HPP

#include "Print.hpp"

namespace yask {

    //#define DEBUG_VV   // debug VecInfoVisitor.

    // One element in a vector.
    class VecElem {
    public:
        VarPoint _vec;      // starting index of vector containing this element.
        size_t _offset;      // 1-D offset in _vec.
        IntTuple _offsets;   // n-D offsets.

        VecElem(const VarPoint& vec, int offset, const IntTuple& offsets) :
            _vec(vec), _offset(offset), _offsets(offsets) { }

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

    // Map of each vector block to aligned block(s) that contain its points.
    typedef map<VarPoint, VarPointSet> Point2Vecs;

    // Map of each vector block to its elements.
    typedef map<VarPoint, VecElemList> Point2VecElemLists;

    // This visitor determines the vector blocks needed to calculate the stencil.
    // It doesn't actually generate code; it just collects info from the AST.
    // After the AST has been visited, the object can be used to print statistics
    // or used to provide data to a print visitor.
    class VecInfoVisitor : public ExprVisitor {
    protected:

        const Dimensions& _dims;
        int _vlen;                   // size of one vector.

    public:

        // Data on vectorizable points.
        VarPointSet _alignedVecs; // set of aligned vectors, i.e., ones that need to be read from memory.
        Point2Vecs _vblk2avblks; // each vec block -> its constituent aligned vec blocks.
        Point2VecElemLists _vblk2elemLists; // each vec block -> in-order list of its aligned vec blocks' elements.
        VarPointSet _vecPoints;            // set of all vectorizable points read from, aligned and unaligned.
        VarPointSet _vecWrites;            // set of vectors written to.

        // NB: the above hold much of the same info, but arranged differently:
        // _alignedVecs contains only a set of aligned blocks.
        // _vblk2avblks is used to quickly determine which aligned blocks contribute to a given block.
        // _vblk2elemLists is used to find exactly where each element comes from.
        // The keys are the same for both maps: the vectorizable var points.

        // Data on non-vectorizable points.
        VarPointSet _scalarPoints; // set of points that should be read as scalars and broadcast to vectors.
        VarPointSet _nonVecPoints; // set of points that are not scalars or vectorizable.

        VecInfoVisitor(const Dimensions& dims) :
            _dims(dims) {
            _vlen = dims._fold.product();
        }

        virtual const IntTuple& getFold() const {
            return _dims._fold;
        }

        virtual size_t getNumPoints() const {
            return _vblk2elemLists.size();
        }

        virtual size_t getNumAlignedVecs() const {
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

#if 0
        // Print stats header.
        virtual void printStatsHeader(ostream& os, string separator) const {
            os << "destination var" <<
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
        virtual void printStats(ostream& os, const string& destVar, const string& separator) const {

            // calc num blends needed.
            int numBlends = 0;
            for (auto i = _vblk2elemLists.begin(); i != _vblk2elemLists.end(); i++) {
                //auto pt = i->first;
                auto vev = i->second;
                VarPointSet mems; // inputs for this point.
                for (size_t j = 0; j < vev.size(); j++)
                    mems.insert(vev[j]._vec);
                if (mems.size() > 1) // only need to blend if >1 inputs.
                    numBlends += mems.size();
            }

            // calc footprint in each dim.
            map<const string, int> footprints;
            for (auto& dim : _dims._fold.getDims()) {
                auto& dname = dim.getName();

                // Make set of aligned vecs projected in this dir.
                set<IntTuple> footprint;
                for (auto vb : _alignedVecs) {
                    IntTuple proj = vb.removeDim(dim);
                    footprint.insert(proj);
                }
                footprints[dname] = footprint.size();
            }

            os << destVar <<
                separator << _vlen <<
                separator << _dims._fold.makeValStr("*") <<
                separator << getNumPoints() <<
                separator << getNumAlignedVecs() <<
                separator << numBlends;
            for (auto& dim : _dims._fold.getDims())
                os << separator << footprints[dim.getName()];
            os << endl;
        }
#endif

        // Equality.
        virtual string visit(EqualsExpr* ee) {

            // Only want to continue visit on RHS of an eqGroup.
            ee->getRhs()->accept(this);

            // For LHS, just save point.
            auto lhs = ee->getLhs();
            _vecWrites.insert(*lhs);
            return "";
        }

        // Called when a var point is read in a stencil function.
        virtual string visit(VarPoint* gp);
    };

    // Define methods for printing a vectorized version of the stencil.
    class VecPrintHelper {
    protected:
        VecInfoVisitor& _vv;
        bool _reuseVars; // if true, load to a local var; else, reload on every access.
        bool _definedNA;           // NA var defined.
        map<VarPoint, string> _vecVars; // vecs that are already constructed.
        map<string, string> _elemVars; // elems that are already read (key is read stmt).

        // Print access to an aligned vector block.
        // Return var name.
        virtual string printAlignedVecRead(ostream& os, const VarPoint& gp) =0;

        // Print unaligned vector memory read.
        // Assumes this results in same values as printUnalignedVec().
        // Return var name.
        virtual string printUnalignedVecRead(ostream& os, const VarPoint& gp) =0;

        // Print write to an aligned vector block.
        // Return expression written.
        virtual string printAlignedVecWrite(ostream& os, const VarPoint& gp,
                                            const string& val) =0;

        // Print conversion from existing vars to make an unaligned vector block.
        // Return var name.
        virtual string printUnalignedVec(ostream& os, const VarPoint& gp) =0;

        // Print construction for one point var pvName from elems.
        virtual void printUnalignedVecCtor(ostream& os, const VarPoint& gp,
                                           const string& pvName) =0;

        // Read from a single point.
        // Return code for read.
        virtual string readFromScalarPoint(ostream& os, const VarPoint& gp,
                                           const VarMap* vMap=0) =0;

        // Read from multiple points that are not vectorizable.
        // Return var name.
        virtual string printNonVecRead(ostream& os, const VarPoint& gp) =0;

    public:
        VecPrintHelper(VecInfoVisitor& vv,
                       bool reuseVars = true) :
            _vv(vv), _reuseVars(reuseVars), _definedNA(false) { }
        virtual ~VecPrintHelper() {}

        // get fold info.
        virtual const IntTuple& getFold() const {
            return _vv.getFold();
        }

        // Add a N/A var, just for readability.
        virtual void makeNA(ostream& os, string linePrefix, string lineSuffix) {
            if (!_definedNA) {
                os << linePrefix << "const int NA = 0; // indicates element not used." << lineSuffix;
                _definedNA = true;
            }
        }

        // Return point info.
        virtual bool isAligned(const VarPoint& gp) {
            return _vv._alignedVecs.count(gp) > 0;
        }

        // Access cached values.
        virtual void savePointVar(const VarPoint& gp, string var) {
            _vecVars[gp] = var;
        }
        virtual string* lookupPointVar(const VarPoint& gp) {
            if (_vecVars.count(gp))
                return &_vecVars.at(gp);
            return 0;
        }
    };

    // A visitor that reorders exprs based on vector info.
    class ExprReorderVisitor : public ExprVisitor {
    protected:
        VecInfoVisitor& _vv;

    public:
        ExprReorderVisitor(VecInfoVisitor& vv) :
            _vv(vv) { }
        virtual ~ExprReorderVisitor() {}

        // Sort a commutative expression.
        virtual string visit(CommutativeExpr* ce);
    };

} // namespace yask.

#endif
