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

////////// Support for C++ scalar and vector-code generation //////////////

#include "Cpp.hpp"

namespace yask {

    /////////// Scalar code /////////////

    // Format a real, preserving precision.
    string CppPrintHelper::formatReal(double v) {

        // IEEE double gives 15-17 significant decimal digits of precision per
        // https://en.wikipedia.org/wiki/Double-precision_floating-point_format.
        // Some precision might be lost if/when cast to a float, but that's ok.
        ostringstream oss;
        oss << setprecision(15) << scientific << v;
        return oss.str();
    }
    
    // Make call for a point.
    // This is a utility function used for both reads and writes.
    string CppPrintHelper::makePointCall(const GridPoint& gp,
                                         const string& fname,
                                         string optArg) const {
        ostringstream oss;
        oss << "_context->" << gp.getGridName() << "->" << fname << "(";
        if (optArg.length()) oss << optArg << ", ";
        string args = gp.makeArgStr();
        oss << "{" << args << "}, "; // use Indices initializer-list ctor.
        oss << "__LINE__)";
        return oss.str();
    }
    
    // Return a grid-point reference.
    string CppPrintHelper::readFromPoint(ostream& os, const GridPoint& gp) {
        return makePointCall(gp, "readElem");
    }

    // Return code to update a grid point.
    string CppPrintHelper::writeToPoint(ostream& os, const GridPoint& gp,
                                        const string& val) {
        return makePointCall(gp, "writeElem", val);
    }

    /////////// Vector code /////////////

    // Read from a single point.
    // Return code for read.
    string CppVecPrintHelper::readFromScalarPoint(ostream& os, const GridPoint& gp,
                                                  const VarMap* vMap) {

        // Use default var-map if not provided.
        if (!vMap)
            vMap = &_varMap;
        
        // Determine type to avoid virtual call.
        bool folded = gp.isGridFoldable();
        string gtype = folded ? "YkVecGrid" : "YkElemGrid";
        
        // Assume that broadcast will be handled automatically by
        // operator overloading in kernel code.
        // Specify that any indices should use element vars.
        string str = "_context->" + gp.getGridName() + "->" + gtype + "::readElem(";
        string args = gp.makeArgStr(vMap);
        str += "{" + args + "}, ";
        str += "__LINE__)";
        return str;
    }

    // Read from multiple points that are not vectorizable.
    // Return var name.
    string CppVecPrintHelper::printNonVecRead(ostream& os, const GridPoint& gp) {
        printPointComment(os, gp, "Construct folded vector from non-folded");

        // Make a vec var.
        string mvName = makeVarName();
        os << _linePrefix << getVarType() << " " << mvName << _lineSuffix;

        // Loop through all points in the vector fold.
        getFold().visitAllPoints([&](const IntTuple& vecPoint,
                                     size_t pelem){

                // Example: vecPoint contains x=0, y=2, z=1, where
                // each val is the offset in the given fold dim.
                // We want to map y=>(y+2), z=>(z+1) in
                // grid-point index args.
                VarMap vMap;
                for (auto& dim : vecPoint.getDims()) {
                    auto& dname = dim.getName();
                    int dofs = dim.getVal();
                    if (dofs != 0) {
                        ostringstream oss;
                        oss << "(" << dname << "+" << dofs << ")";
                        vMap[dname] = oss.str();
                    }
                }

                // Read or reuse.
                string stmt = readFromScalarPoint(os, gp, &vMap);
                string varname;
                if (_elemVars.count(stmt))
                    varname = _elemVars.at(stmt);
                else {

                    // Read val into a new scalar var.
                    varname = makeVarName();
                    os << _linePrefix << "real_t " << varname <<
                        " = " << stmt << _lineSuffix;
                    _elemVars[stmt] = varname;
                }
                
                // Output translated expression for this element.
                os << _linePrefix << mvName << "[" << pelem << "] = " <<
                    varname << "; // for offset " << vecPoint.makeDimValStr() <<
                    _lineSuffix;

                return true;
            }); // end of lambda.
        return mvName;
    }

    // Print call for a point.
    // This is a utility function used for reads, writes, and prefetches.
    void CppVecPrintHelper::printVecPointCall(ostream& os,
                                              const GridPoint& gp,
                                              const string& funcName,
                                              const string& firstArg,
                                              const string& lastArg,
                                              bool isNorm) const {

        os << " _context->" << gp.getGridName() << "->" << funcName << "(";
        if (firstArg.length())
            os << firstArg << ", ";
        string args = isNorm ? gp.makeNormArgStr(_dims) : gp.makeArgStr();
        os << "{" << args << "}"; // use Indices initializer-list ctor.
        if (lastArg.length()) 
            os << ", " << lastArg;
        os << ")";
    }

    // Print code to set pointers of aligned reads.
    void CppVecPrintHelper::printBasePtrs(ostream& os) {
        const string& idim = _dims._innerDim;

        // A set for the aligned reads & writes.
        GridPointSet gps;

        // Aligned reads as determined by VecInfoVisitor.
        gps = _vv._alignedVecs;

        // Writes (assume aligned).
        gps.insert(_vv._vecWrites.begin(), _vv._vecWrites.end());

        // Loop through all aligned read & write points.
        for (auto& gp : gps) {
        
            // Make base point (inner-dim index = 0).
            auto bgp = makeBasePoint(gp);

            // Not already saved?
            if (!lookupPointPtr(*bgp)) {

                // Get temp var for ptr.
                string ptrName = makeVarName();

                // Save for future use.
                savePointPtr(*bgp, ptrName);
            }

            // Collect some stats for reads using this ptr.
            if (_vv._alignedVecs.count(gp)) {
                auto* p = lookupPointPtr(*bgp);
                assert(p);

                // Get const offsets.
                auto& offsets = gp.getArgOffsets();
                auto* ofs = offsets.lookup(idim);

                // Remember lowest one.
                if (ofs && (!_ptrOfsLo.count(*p) || _ptrOfsLo[*p] > *ofs))
                    _ptrOfsLo[*p] = *ofs;

                // Remember highest one.
                if (ofs && (!_ptrOfsHi.count(*p) || _ptrOfsHi[*p] < *ofs))
                    _ptrOfsHi[*p] = *ofs;
            }
        }

        // Loop through all aligned read & write points.
        set<string> done;
        for (auto& gp : gps) {
        
            // Make base point (inner-dim index = 0).
            auto bgp = makeBasePoint(gp);
            auto* p = lookupPointPtr(*bgp);
            assert(p);
            if (!done.count(*p)) {

                // Print pointer creation.
                printPointPtr(os, *p, *bgp);
                
                // Print prefetch(es) for this ptr if a read.
                if (_vv._alignedVecs.count(gp))
                    printPrefetches(os, false, *p);

                done.insert(*p);
            }
        }
    }

    // Print prefetches for each base pointer.
    // 'level': cache level.
    // 'ahead': prefetch PF distance ahead instead of up to PF dist.
    void CppVecPrintHelper::printPrefetches(ostream& os,
                                            bool ahead, string ptrVar) {

        // cluster mult in inner dim.
        const string& idim = _dims._innerDim;
        string imult = "CMULT_" + PrinterBase::allCaps(idim);

        for (int level = 1; level <= 2; level++) {
        
            os << "\n // Prefetch to L" << level << " cache if enabled.\n";
                os << _linePrefix << "#if PFD_L" << level << " > 0\n";

                // Loop thru vec ptrs.
                for (auto vp : _vecPtrs) {
                    auto& ptr = vp.second; // ptr var name.

                    // Filter by ptrVar if provided.
                    if (ptrVar.length() && ptrVar != ptr)
                        continue;

                    // _ptrOfs{Lo,Hi} contain first and last offsets in idim,
                    // NOT normalized to vector len.
                    string lo = _dims.makeNormStr(_ptrOfsLo[ptr], idim);
                    if (lo.length() == 0) lo = "0";
                    string hi = _dims.makeNormStr(_ptrOfsHi[ptr], idim);
            
                    // Start loop of prefetches.
                    os << "#pragma unroll\n" <<
                        _linePrefix << " for (int ofs = ";

                    // First offset.
                    if (ahead)
                        os << "(PFD_L" << level << "*" << imult << ")" << hi;
                    else
                        os << lo;

                    // End of offsets.
                    os << "; ofs < ";
                    if (ahead)
                        os << "((PFD_L" << level << "+1)*" << imult << ")" << hi;
                    else
                        os << "(PFD_L" << level << "*" << imult << ")" << hi;

                    // Continue loop.
                    os << "; ofs++)\n" <<
                        _linePrefix << "  prefetch<L" << level << "_HINT>(&" << ptr <<
                        "[" << idim << " + ofs])" << _lineSuffix;
                }
                os << _linePrefix << "#endif // L" << level << " prefetch.\n";
        }
    }
    
    // Print code to set ptrName to gp.
    void CppVecPrintHelper::printPointPtr(ostream& os, const string& ptrName,
                                          const GridPoint& gp) {
        printPointComment(os, gp, "Calculate pointer to ");

        // Calc ptr.
        os << _linePrefix << getVarType() << "* " << ptrName << " = ";
        printVecPointCall(os, gp, "getVecPtrNorm", "", "__LINE__", true);
        os << _lineSuffix;
    }
    
    // Print any needed memory reads and/or constructions to 'os'.
    // Return code containing a vector of grid points.
    string CppVecPrintHelper::readFromPoint(ostream& os, const GridPoint& gp) {
        string codeStr;

        // Already done and saved.
        if (_reuseVars && _vecVars.count(gp))
            codeStr = _vecVars[gp]; // do nothing.

        // Can we use a pointer?
        else if (gp.getLoopType() == GridPoint::LOOP_OFFSET) {
        
            // Got a pointer to the base addr?
            auto bgp = makeBasePoint(gp);
            auto* p = lookupPointPtr(*bgp);
            if (p) {

                // Offset in inner-dim direction.
                string idim = _dims._innerDim;
                string ofsStr = gp.makeNormArgStr(idim, _dims);
                
                // Output read using base addr.
                printPointComment(os, gp, "Read aligned");
                codeStr = makeVarName();
                os << _linePrefix << getVarType() << " " << codeStr << " = " <<
                    *p << "[" << ofsStr << "]" << _lineSuffix;
            }
        }

        // Remember this point and return it.
        if (codeStr.length()) {
            savePointVar(gp, codeStr);
            return codeStr;
        }

        // If not done, use parent method.
        return VecPrintHelper::readFromPoint(os, gp);
    }

    // Print any immediate memory writes to 'os'.
    // Return code to update a vector of grid points or null string
    // if all writes were printed.
    string CppVecPrintHelper::writeToPoint(ostream& os, const GridPoint& gp,
                                           const string& val) {

        // Can we use a pointer?
        if (gp.getLoopType() == GridPoint::LOOP_OFFSET) {
        
            // Got a pointer to the base addr?
            auto bgp = makeBasePoint(gp);
            auto* p = lookupPointPtr(*bgp);
            if (p) {

                // Offset.
                string idim = _dims._innerDim;
                string ofs = gp.makeNormArgStr(idim, _dims);
                
                // Output write using base addr.
                printPointComment(os, gp, "Write aligned");
                os << _linePrefix << *p << "[" << ofs << "] = " << val << _lineSuffix;
                return "";
            }
        }

        // If not done, use parent method.
        return VecPrintHelper::writeToPoint(os, gp, val);
    }

    // Print aligned memory read.
    string CppVecPrintHelper::printAlignedVecRead(ostream& os, const GridPoint& gp) {
        printPointComment(os, gp, "Read aligned");

        // Read memory.
        string mvName = makeVarName();
        os << _linePrefix << getVarType() << " " << mvName << " = ";
        printVecPointCall(os, gp, "readVecNorm", "", "__LINE__", true);
        os << _lineSuffix;
        return mvName;
    }

    // Print unaliged memory read.
    // Assumes this results in same values as printUnalignedVec().
    string CppVecPrintHelper::printUnalignedVecRead(ostream& os, const GridPoint& gp) {
        printPointComment(os, gp, "Read unaligned");
        os << " // NOTICE: Assumes constituent vectors are consecutive in memory!" << endl;
            
        // Make a var.
        string mvName = makeVarName();
        os << _linePrefix << getVarType() << " " << mvName << _lineSuffix;
        
        // Read memory.
        os << _linePrefix << mvName << ".loadUnalignedFrom((const " << getVarType() << "*)";
        printVecPointCall(os, gp, "getElemPtr", "", "true", false);
        os << ")" << _lineSuffix;
        return mvName;
    }

    // Print aligned memory write.
    string CppVecPrintHelper::printAlignedVecWrite(ostream& os, const GridPoint& gp,
                                                   const string& val) {
        printPointComment(os, gp, "Write aligned");

        // Write temp var to memory.
        printVecPointCall(os, gp, "writeVecNorm", val, "__LINE__", true);
        return val;
    }
    
    // Print conversion from memory vars to point var gp if needed.
    // This calls printUnalignedVecCtor(), which can be overloaded
    // by derived classes.
    string CppVecPrintHelper::printUnalignedVec(ostream& os, const GridPoint& gp) {
        printPointComment(os, gp, "Construct unaligned");

        // Declare var.
        string pvName = makeVarName();
        os << _linePrefix << getVarType() << " " << pvName << _lineSuffix;

        // Contruct it.
        printUnalignedVecCtor(os, gp, pvName);
        return pvName;
    }

    // Print per-element construction for one point var pvName from elems.
    void CppVecPrintHelper::printUnalignedVecSimple(ostream& os, const GridPoint& gp,
                                                    const string& pvName, string linePrefix,
                                                    const set<size_t>* doneElems) {

        // just assign each element in vector separately.
        auto& elems = _vv._vblk2elemLists[gp];
        assert(elems.size() > 0);
        for (size_t pelem = 0; pelem < elems.size(); pelem++) {

            // skip if done.
            if (doneElems && doneElems->count(pelem))
                continue;

            // one vector element from gp.
            auto& ve = elems[pelem];

            // Look up existing input var.
            assert(_vecVars.count(ve._vec));
            string mvName = _vecVars[ve._vec];

            // which element?
            int alignedElem = ve._offset; // 1-D layout of this element.
            string elemStr = ve._offsets.makeDimValOffsetStr();

            os << linePrefix << pvName << "[" << pelem << "] = " <<
                mvName << "[" << alignedElem << "];  // for " <<
                gp.getGridName() << "(" << elemStr << ")" << _lineSuffix;
        }
    }

    // Print init of element indices.
    // Fill _varMap as side-effect.
    void CppVecPrintHelper::printElemIndices(ostream& os) {
        auto& fold = getFold();
        os << endl << " // Element indices derived from vector indices." << endl;
        for (auto& dim : fold.getDims()) {
            auto& dname = dim.getName();
            string ename = dname + _elemSuffix;
            string cap_dname = PrinterBase::allCaps(dname);
            os << " const idx_t " << ename << " = " <<
                dname << " * VLEN_" << cap_dname << ";" << endl;
            _varMap[dname] = ename;
        }
    }

    // Print grid-access vars for a loop.
    void CppLoopVarPrintVisitor::visit(GridPoint* gp) {

        // Retrieve prior analysis of this grid point.
        //auto vecType = gp->getVecType();
        auto loopType = gp->getLoopType();
        
        // If invariant, we can load now.
        if (loopType == GridPoint::LOOP_INVARIANT) {

            // Not already loaded?
            if (!_cvph.lookupPointVar(*gp)) {
                string expr = _ph.readFromPoint(_os, *gp);
                makeNextTempVar(gp) << expr << _ph.getLineSuffix();
                
                // Save for future use.
                string res = getExprStrAndClear();
                _cvph.savePointVar(*gp, res);
            }
        }
    }
    
} // namespace yask.

