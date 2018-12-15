/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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
    string CppPrintHelper::makePointCall(ostream& os,
                                         const GridPoint& gp,
                                         const string& fname,
                                         string optArg) {

        // Get/set local vars.
        string gridPtr = getLocalVar(os, gp.getGridPtr(), _grid_ptr_type);
        string stepArgVar = getLocalVar(os, gp.makeStepArgStr(gridPtr, _dims), _step_val_type);

        ostringstream oss;
        oss << gridPtr << "->" << fname << "(";
        if (optArg.length()) oss << optArg << ", ";
        string args = gp.makeArgStr();
        oss << "{" << args << "}, " << stepArgVar << ", __LINE__)";
        return oss.str();
    }

    // Return a grid-point reference.
    string CppPrintHelper::readFromPoint(ostream& os, const GridPoint& gp) {
        return makePointCall(os, gp, "readElem");
    }

    // Return code to update a grid point.
    string CppPrintHelper::writeToPoint(ostream& os, const GridPoint& gp,
                                        const string& val) {
        return makePointCall(os, gp, "writeElem", val);
    }

    /////////// Vector code /////////////

    // Read from a single point.
    // Return code for read.
    string CppVecPrintHelper::readFromScalarPoint(ostream& os, const GridPoint& gp,
                                                  const VarMap* vMap) {

        // Use default var-map if not provided.
        if (!vMap)
            vMap = &_vec2elemMap;

        // Determine type to avoid virtual call.
        bool folded = gp.isGridFoldable();
        string gtype = folded ? "YkVecGrid" : "YkElemGrid";

        // Get/set local vars.
        string gridPtr = getLocalVar(os, gp.getGridPtr(), CppPrintHelper::_grid_ptr_type);
        string stepArgVar = getLocalVar(os, gp.makeStepArgStr(gridPtr, _dims),
                                        CppPrintHelper::_step_val_type);

        // Assume that broadcast will be handled automatically by
        // operator overloading in kernel code.
        // Specify that any indices should use element vars.
        string str = gridPtr + "->" + gtype + "::readElem(";
        string args = gp.makeArgStr(vMap);
        str += "{" + args + "}, " + stepArgVar + ",__LINE__)";
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

                // Example: vecPoint contains x=0, y=2, z=1, where each val
                // is the offset in the given fold dim.  We want to map
                // x=>x_elem, y=>(y_elem+2), z=>(z_elem+1) in grid-point
                // index args.
                VarMap vMap;
                for (auto& dim : vecPoint.getDims()) {
                    auto& dname = dim.getName();
                    int dofs = dim.getVal();

                    auto& ename = _vec2elemMap.at(dname);
                    if (dofs == 0)
                        vMap[dname] = ename;
                    else {
                        ostringstream oss;
                        oss << "(" << ename << "+" << dofs << ")";
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
    // This is a utility function used for reads & writes.
    string CppVecPrintHelper::printVecPointCall(ostream& os,
                                                const GridPoint& gp,
                                                const string& funcName,
                                                const string& firstArg,
                                                const string& lastArg,
                                                bool isNorm) {

        // Get/set local vars.
        string gridPtr = getLocalVar(os, gp.getGridPtr(), CppPrintHelper::_grid_ptr_type);
        string stepArgVar = getLocalVar(os, gp.makeStepArgStr(gridPtr, _dims),
                                        CppPrintHelper::_step_val_type);

        ostringstream oss;
        oss << gridPtr << "->" << funcName << "(";
        if (firstArg.length())
            oss << firstArg << ", ";
        string args = isNorm ? gp.makeNormArgStr(_dims) : gp.makeArgStr();
        oss << "{" << args << "} ," << stepArgVar;
        if (lastArg.length())
            oss << ", " << lastArg;
        oss << ")";
        return oss.str();
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

            // Can we use a pointer?
            if (gp.getLoopType() != GridPoint::LOOP_OFFSET)
                continue;

            // Make base point (misc & inner-dim indices = 0).
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

                // Get offset in inner dim.
                // E.g., A(t, x+1, y+4) => 4.
                auto* ofs = offsets.lookup(idim);

                // Remember lowest inner-dim offset from this ptr.
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

            // Got a pointer?
            auto* p = lookupPointPtr(*bgp);
            if (!p)
                continue;

            // Make code for pointer and prefetches.
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
    // TODO: add handling of misc dims.
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
                    string left = _dims.makeNormStr(_ptrOfsLo[ptr], idim);
                    if (left.length() == 0) left = "0";
                    string right = _dims.makeNormStr(_ptrOfsHi[ptr], idim);

                    // Start loop of prefetches.
                    os << "\n // For pointer '" << ptr << "'\n"
                        "#pragma unroll\n" <<
                        _linePrefix << " for (int ofs = ";

                    // First offset.
                    // If fetching ahead, only need to get those following
                    // the previous one.
                    if (ahead)
                        os << "(PFD_L" << level << "*" << imult << ")" << right;

                    // If fetching first time, need to fetch across whole range;
                    // starting at left edge.
                    else
                        os << left;

                    // End of offsets.
                    os << "; ofs < ";

                    // If fetching again, stop before next one.
                    if (ahead)
                        os << "((PFD_L" << level << "+1)*" << imult << ")" << right;

                    // If fetching first time, stop where next "ahead" one ends.
                    else
                        os << "(PFD_L" << level << "*" << imult << ")" << right;

                    // Finish loop header.
                    os << "; ofs++) {\n";

                    // Need to print prefetch for every unique grid-point read.
                    set<string> done;
                    for (auto& gp : _vv._alignedVecs) {

                        // For the current base ptr?
                        auto bgp = makeBasePoint(gp);
                        auto* p = lookupPointPtr(*bgp);
                        if (p && *p == ptr) {

                            // Expression for this offset from inner-dim var.
                            string innerExpr = idim + " + ofs";

                            // Expression for ptr offset at this point.
                            string ofsExpr = getPtrOffset(gp, innerExpr);
                            printPointComment(os, gp, "Prefetch for ");

                            // Already done?
                            if (done.count(ofsExpr))
                                os << " // Already accounted for.\n";

                            else {
                                done.insert(ofsExpr);

                                // Prefetch.
                                os << _linePrefix << "  prefetch<L" << level << "_HINT>(&" << ptr <<
                                    "[" << ofsExpr << "])" << _lineSuffix;
                            }
                        }
                    }

                    // End loop;
                    os << " }\n";
                }
                os << _linePrefix << "#endif // L" << level << " prefetch.\n";
        }
    }

    // Make base point (misc & inner-dim indices = 0).
    GridPointPtr CppVecPrintHelper::makeBasePoint(const GridPoint& gp) {
        GridPointPtr bgp = gp.cloneGridPoint();
        for (auto& dim : gp.getDims()) {
            auto& dname = dim->getName();
            auto type = dim->getType();

            // Set inner domain index to 0.
            if (dname == getDims()._innerDim) {
                IntScalar idi(dname, 0);
                bgp->setArgConst(idi);
            }

            // Set misc indices to their min value if they are inside
            // inner domain dim.
            else if (_settings._innerMisc && type == MISC_INDEX) {
                auto* grid = gp.getGrid();
                auto min_val = grid->getMinIndices()[dname];
                IntScalar idi(dname, min_val);
                bgp->setArgConst(idi);
            }
        }
        return bgp;
    }
    
    // Print code to set ptrName to gp.
    void CppVecPrintHelper::printPointPtr(ostream& os, const string& ptrName,
                                          const GridPoint& gp) {
        printPointComment(os, gp, "Calculate pointer to ");

        // Get pointer to vector using normalized indices.
        // Ignore out-of-range errors because we might get a base pointer to an
        // element before the allocated range.
        auto vp = printVecPointCall(os, gp, "getVecPtrNorm", "", "false", true);

        // Ptr will be unique if:
        // - Grid doesn't have step dim, or
        // - Grid doesn't allow dynamic step allocs and the alloc size is one (TODO), or
        // - Grid doesn't allow dynamic step allocs and all accesses are via
        //   offsets from the step dim w/compatible offsets (TODO).
        // TODO: must also share pointers during code gen in last 2 cases.
        auto* grid = gp.getGrid();
        bool is_unique = (grid->getStepDim() == nullptr);
        // || (!grid->is_dynamic_step_alloc() && grid->get_step_alloc_size() == 1);
        string type = is_unique ? "auto* restrict " : "auto* ";

        // Print type and value.
        os << _linePrefix << type << ptrName << " = " << vp << _lineSuffix;
    }

    // Get expression for offset of 'gp' from base pointer.  Base pointer
    // points to vector with outer-dims == same values as in 'gp', inner-dim
    // == 0 and misc dims == their min value.
    string CppVecPrintHelper::getPtrOffset(const GridPoint& gp, const string& innerExpr) {
        auto* grid = gp.getGrid();

        // Need to create an expression for inner-dim
        // and misc indices offsets.
                
        // Start with offset in inner-dim direction.
        // This must the dim that appears before the misc dims
        // in the grid-var layout.
        string idim = _dims._innerDim;
        string ofsStr = "(";
        if (innerExpr.length())
            ofsStr += innerExpr;
        else
            ofsStr += gp.makeNormArgStr(idim, _dims);
        ofsStr += ")";

        // Misc indices if they are inside inner-dim.
        if (_settings._innerMisc) {
            for (int i = 0; i < grid->get_num_dims(); i++) {
                auto& dimi = gp.getDims().at(i);
                auto& dni = dimi->getName();
                auto typei = dimi->getType();
                if (typei == MISC_INDEX) {

                    // Mult by size of remaining misc dims.
                    for (int j = i; j < grid->get_num_dims(); j++) {
                        auto& dimj = gp.getDims().at(j);
                        auto& dnj = dimj->getName();
                        auto typej = dimj->getType();
                        if (typej == MISC_INDEX) {
                            auto min_idx = grid->getMinIndices()[dnj];
                            auto max_idx = grid->getMaxIndices()[dnj];
                            ofsStr += " * (" + to_string(max_idx) +
                                " - " + to_string(min_idx) + " + 1)";
                        }
                    }
                        
                    // Add offset of this misc value, which must be const.
                    auto min_val = grid->getMinIndices()[dni];
                    auto val = gp.getArgConsts()[dni];
                    ofsStr += " + (" + to_string(val) + " - " +
                        to_string(min_val) + ")";
                }
            }
        }
        return ofsStr;
    }
    
    // Print any needed memory reads and/or constructions to 'os'.
    // Return code containing a vector of grid points.
    string CppVecPrintHelper::readFromPoint(ostream& os, const GridPoint& gp) {
        string codeStr;

        // Already done and saved.
        if (_reuseVars && _vecVars.count(gp))
            codeStr = _vecVars[gp]; // do nothing.

        // Can we use a vec pointer?
        // Read must be aligned, and we must have a pointer.
        else if (_vv._alignedVecs.count(gp) &&
                 gp.getVecType() == GridPoint::VEC_FULL &&
                 gp.getLoopType() == GridPoint::LOOP_OFFSET) {

            // Got a pointer to the base addr?
            auto bgp = makeBasePoint(gp);
            auto* p = lookupPointPtr(*bgp);
            if (p) {
#ifdef DEBUG_GP
                cout << " //** reading from point " << gp.makeStr() << " using pointer.\n";
#endif

                // Output read using base addr.
                auto ofsStr = getPtrOffset(gp);
                printPointComment(os, gp, "Read aligned");
                codeStr = makeVarName();
                os << _linePrefix << getVarType() << " " << codeStr << " = " <<
                    *p << "[" << ofsStr << "]" << _lineSuffix;
            }
        }

        // Did we make some code to read the point?
        if (codeStr.length()) {

            // Remember this point and return it.
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
                auto ofsStr = getPtrOffset(gp);

                // Output write using base addr.
                printPointComment(os, gp, "Write aligned");

                os << _linePrefix << val << ".storeTo_masked(" << *p << " + (" <<
                    ofsStr << "), write_mask)" << _lineSuffix;
                // without mask: os << _linePrefix << *p << "[" << ofs << "] = " << val << _lineSuffix;

                return "";
            }
        }

        // If not done, use parent method.
        return VecPrintHelper::writeToPoint(os, gp, val);
    }

    // Print aligned memory read.
    string CppVecPrintHelper::printAlignedVecRead(ostream& os, const GridPoint& gp) {

        printPointComment(os, gp, "Read aligned");
        auto rvn = printVecPointCall(os, gp, "readVecNorm", "", "__LINE__", true);

        // Read memory.
        string mvName = makeVarName();
        os << _linePrefix << getVarType() << " " << mvName << " = " << rvn << _lineSuffix;
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
        auto vp = printVecPointCall(os, gp, "getElemPtr", "", "true", false);

        // Read memory.
        os << _linePrefix << mvName <<
            ".loadUnalignedFrom((const " << getVarType() << "*)" << vp << ")" << _lineSuffix;
        return mvName;
    }

    // Print aligned memory write.
    string CppVecPrintHelper::printAlignedVecWrite(ostream& os, const GridPoint& gp,
                                                   const string& val) {
        printPointComment(os, gp, "Write aligned");
        auto vn = printVecPointCall(os, gp, "writeVecNorm_masked", val, "write_mask, __LINE__", true);
        // without mask: auto vn = printVecPointCall(os, gp, "writeVecNorm", val, "__LINE__", true);

        // Write temp var to memory.
        os << vn;
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
    // Fill _vec2elemMap as side-effect.
    void CppVecPrintHelper::printElemIndices(ostream& os) {
        auto& fold = getFold();
        os << "\n // Element indices derived from vector indices.\n";
        int i = 0;
        for (auto& dim : fold.getDims()) {
            auto& dname = dim.getName();
            string ename = dname + _elemSuffix;
            string cap_dname = PrinterBase::allCaps(dname);
            os << " idx_t " << ename <<
                " = _context->rank_domain_offsets[" << i << "] + (" <<
                dname << " * VLEN_" << cap_dname << ");\n";
            _vec2elemMap[dname] = ename;
            i++;
        }
    }

    // Print invariant grid-access vars for non-time loop(s).
    string CppStepVarPrintVisitor::visit(GridPoint* gp) {

        // Pointer to grid.
        string gridPtr = _cvph.getLocalVar(_os, gp->getGridPtr(), CppPrintHelper::_grid_ptr_type);

        // Time var.
        auto& dims = _cvph.getDims();
        _cvph.getLocalVar(_os, gp->makeStepArgStr(gridPtr, dims),
                          CppPrintHelper::_step_val_type);
        return "";
    }

    // Print invariant grid-access vars for an inner loop.
    string CppLoopVarPrintVisitor::visit(GridPoint* gp) {

        // Retrieve prior analysis of this grid point.
        auto loopType = gp->getLoopType();

        // If invariant, we can load now.
        if (loopType == GridPoint::LOOP_INVARIANT) {

            // Not already loaded?
            if (!_cvph.lookupPointVar(*gp)) {
                string expr = _ph.readFromPoint(_os, *gp);
                string res;
                makeNextTempVar(res, gp) << expr << _ph.getLineSuffix();

                // Save for future use.
                _cvph.savePointVar(*gp, res);
            }
        }
        return "";
    }

} // namespace yask.

