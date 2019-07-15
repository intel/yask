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

////////// Support for C++ scalar and vector-code generation //////////////

#include "Cpp.hpp"

namespace yask {

    /////////// Scalar code /////////////

    // Format a real, preserving precision.
    string CppPrintHelper::formatReal(double v) {

        // Int representation equivalent?
        // This is needed to properly format int expressions
        // like 'x > 5'.
        if (double(int(v)) == v)
            return to_string(int(v));
        
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
                                         const VarPoint& gp,
                                         const string& fname,
                                         string optArg) {

        // Get/set local vars.
        string varPtr = getLocalVar(os, getVarPtr(gp), _var_ptr_restrict_type);
        string stepArgVar = getLocalVar(os, gp.makeStepArgStr(varPtr, _dims), _step_val_type);

        string res = varPtr + "->" + fname + "(";
        if (optArg.length())
            res += optArg + ", ";
        string args = gp.makeArgStr();
        res += "{" + args + "}, " + stepArgVar + ", __LINE__)";
        return res;
    }

    // Return a var-point reference.
    string CppPrintHelper::readFromPoint(ostream& os, const VarPoint& gp) {
        return makePointCall(os, gp, "readElem");
    }

    // Return code to update a var point.
    string CppPrintHelper::writeToPoint(ostream& os, const VarPoint& gp,
                                        const string& val) {
        return makePointCall(os, gp, "writeElem", val);
    }

    /////////// Vector code /////////////

    // Read from a single point.
    // Return code for read.
    string CppVecPrintHelper::readFromScalarPoint(ostream& os, const VarPoint& gp,
                                                  const VarMap* vMap) {

        // Use default var-map if not provided.
        if (!vMap)
            vMap = &_vec2elemMap;

        // Determine type to avoid virtual call.
        bool folded = gp.isVarFoldable();
        string gtype = folded ? "YkVecVar" : "YkElemVar";

        // Get/set local vars.
        string varPtr = getLocalVar(os, getVarPtr(gp), _var_ptr_restrict_type);
        string stepArgVar = getLocalVar(os, gp.makeStepArgStr(varPtr, _dims),
                                        _step_val_type);

        // Assume that broadcast will be handled automatically by
        // operator overloading in kernel code.
        // Specify that any indices should use element vars.
        string str = varPtr + "->" + gtype + "::readElem(";
        string args = gp.makeArgStr(vMap);
        str += "{" + args + "}, " + stepArgVar + ",__LINE__)";
        return str;
    }

    // Read from multiple points that are not vectorizable.
    // Return var name.
    string CppVecPrintHelper::printNonVecRead(ostream& os, const VarPoint& gp) {
        printPointComment(os, gp, "Construct folded vector from non-folded");

        // Make a vec var.
        string mvName = makeVarName();
        os << _linePrefix << getVarType() << " " << mvName << _lineSuffix;

        // Loop through all points in the vector fold.
        getFold().visitAllPoints([&](const IntTuple& vecPoint,
                                     size_t pelem){

                // Example: vecPoint contains x=0, y=2, z=1, where each val
                // is the offset in the given fold dim.  We want to map
                // x=>x_elem, y=>(y_elem+2), z=>(z_elem+1) in var-point
                // index args.
                VarMap vMap;
                for (auto& dim : vecPoint) {
                    auto& dname = dim.getName();
                    int dofs = dim.getVal();

                    auto& ename = _vec2elemMap.at(dname);
                    if (dofs == 0)
                        vMap[dname] = ename;
                    else {
                        vMap[dname] = "(" + ename + "+" + to_string(dofs) + ")";
                    }
                }

                // Read or reuse.
                string stmt = readFromScalarPoint(os, gp, &vMap);
                auto* varname = lookupElemVar(stmt);
                if (!varname) {

                    // Read val into a new scalar var.
                    string vname = makeVarName();
                    os << _linePrefix << "real_t " << vname <<
                        " = " << stmt << _lineSuffix;
                    varname = saveElemVar(stmt, vname);
                }

                // Output translated expression for this element.
                os << _linePrefix << mvName << "[" << pelem << "] = " <<
                    *varname << "; // for offset " << vecPoint.makeDimValStr() <<
                    _lineSuffix;

                return true;
            }); // end of lambda.
        return mvName;
    }

    // Print call for a point.
    // This is a utility function used for reads & writes.
    string CppVecPrintHelper::printVecPointCall(ostream& os,
                                                const VarPoint& gp,
                                                const string& funcName,
                                                const string& firstArg,
                                                const string& lastArg,
                                                bool isNorm) {

        // Get/set local vars.
        string varPtr = getLocalVar(os, getVarPtr(gp), CppPrintHelper::_var_ptr_restrict_type);
        string stepArgVar = getLocalVar(os, gp.makeStepArgStr(varPtr, _dims),
                                        CppPrintHelper::_step_val_type);

        string res = varPtr + "->" + funcName + "(";
        if (firstArg.length())
            res += firstArg + ", ";
        string args = isNorm ? gp.makeNormArgStr(_dims) : gp.makeArgStr();
        res += "{" + args + "} ," + stepArgVar;
        if (lastArg.length())
            res += ", " + lastArg;
        res += ")";
        return res;
    }

    // Print code to set pointers of aligned reads.
    void CppVecPrintHelper::printBasePtrs(ostream& os) {
        const string& idim = _dims._innerDim;

        // A set for the aligned reads & writes.
        VarPointSet gps;

        // Aligned reads as determined by VecInfoVisitor.
        gps = _vv._alignedVecs;

        // Writes (assume aligned).
        gps.insert(_vv._vecWrites.begin(), _vv._vecWrites.end());

        // Loop through all aligned read & write points.
        for (auto& gp : gps) {

            // Can we use a pointer?
            if (gp.getLoopType() != VarPoint::LOOP_OFFSET)
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

                    // Loop bounds.
                    string start, stop;
                    
                    // If fetching ahead, only need to get those following
                    // the previous one.
                    if (ahead)
                        start = "(PFD_L" + to_string(level) + "*" + imult + ")" + right;

                    // If fetching first time, need to fetch across whole range;
                    // starting at left edge.
                    else
                        start = left;
                    start = "(" + start + ")";
                    
                    // If fetching again, stop before next one.
                    if (ahead)
                        stop = "((PFD_L" + to_string(level) + "+1)*" + imult + ")" + right;

                    // If fetching first time, stop where next "ahead" one ends.
                    else
                        stop = "(PFD_L" + to_string(level) + "*" + imult + ")" + right;
                    stop = "(" + stop + ")";

                    // Start loop of prefetches.
                    os << "\n // For pointer '" << ptr << "'\n"
                        "#pragma unroll(" << stop << " - " << start << ")\n" <<
                        _linePrefix << " for (int ofs = " << start <<
                        "; ofs < " << stop << "; ofs++) {\n";

                    // Need to print prefetch for every unique var-point read.
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
    varPointPtr CppVecPrintHelper::makeBasePoint(const VarPoint& gp) {
        varPointPtr bgp = gp.cloneVarPoint();
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
                auto* var = gp.getVar();
                auto min_val = var->getMinIndices()[dname];
                IntScalar idi(dname, min_val);
                bgp->setArgConst(idi);
            }
        }
        return bgp;
    }
    
    // Print code to set ptrName to gp.
    void CppVecPrintHelper::printPointPtr(ostream& os, const string& ptrName,
                                          const VarPoint& gp) {
        printPointComment(os, gp, "Calculate pointer to ");

        // Get pointer to vector using normalized indices.
        // Ignore out-of-range errors because we might get a base pointer to an
        // element before the allocated range.
        auto vp = printVecPointCall(os, gp, "getVecPtrNorm", "", "false", true);

        // Ptr will be unique if:
        // - Var doesn't have step dim, or
        // - Var doesn't allow dynamic step allocs and the alloc size is one (TODO), or
        // - Var doesn't allow dynamic step allocs and all accesses are via
        //   offsets from the step dim w/compatible offsets (TODO).
        // TODO: must also share pointers during code gen in last 2 cases.
        auto* var = gp.getVar();
        bool is_unique = false;
        //bool is_unique = (var->getStepDim() == nullptr);
        string type = is_unique ? _var_ptr_restrict_type : _var_ptr_type;

        // Print type and value.
        os << _linePrefix << type << " " << ptrName << " = " << vp << _lineSuffix;
    }

    // Get expression for offset of 'gp' from base pointer.  Base pointer
    // points to vector with outer-dims == same values as in 'gp', inner-dim
    // == 0 and misc dims == their min value.
    string CppVecPrintHelper::getPtrOffset(const VarPoint& gp, const string& innerExpr) {
        auto* var = gp.getVar();

        // Need to create an expression for inner-dim
        // and misc indices offsets.
                
        // Start with offset in inner-dim direction.
        // This must the dim that appears before the misc dims
        // in the var layout.
        string idim = _dims._innerDim;
        string ofsStr = "(";
        if (innerExpr.length())
            ofsStr += innerExpr;
        else
            ofsStr += gp.makeNormArgStr(idim, _dims);
        ofsStr += ")";

        // Misc indices if they are inside inner-dim.
        if (_settings._innerMisc) {
            for (int i = 0; i < var->get_num_dims(); i++) {
                auto& dimi = gp.getDims().at(i);
                auto& dni = dimi->getName();
                auto typei = dimi->getType();
                if (typei == MISC_INDEX) {

                    // Mult by size of remaining misc dims.
                    for (int j = i; j < var->get_num_dims(); j++) {
                        auto& dimj = gp.getDims().at(j);
                        auto& dnj = dimj->getName();
                        auto typej = dimj->getType();
                        if (typej == MISC_INDEX) {
                            auto min_idx = var->getMinIndices()[dnj];
                            auto max_idx = var->getMaxIndices()[dnj];
                            ofsStr += " * (" + to_string(max_idx) +
                                " - " + to_string(min_idx) + " + 1)";
                        }
                    }
                        
                    // Add offset of this misc value, which must be const.
                    auto min_val = var->getMinIndices()[dni];
                    auto val = gp.getArgConsts()[dni];
                    ofsStr += " + (" + to_string(val) + " - " +
                        to_string(min_val) + ")";
                }
            }
        }
        return ofsStr;
    }
    
    // Print any needed memory reads and/or constructions to 'os'.
    // Return code containing a vector of var points.
    string CppVecPrintHelper::readFromPoint(ostream& os, const VarPoint& gp) {
        string codeStr;

        // Already done and saved.
        if (_reuseVars && _vecVars.count(gp))
            codeStr = _vecVars[gp]; // do nothing.

        // Can we use a vec pointer?
        // Read must be aligned, and we must have a pointer.
        else if (_vv._alignedVecs.count(gp) &&
                 gp.getVecType() == VarPoint::VEC_FULL &&
                 gp.getLoopType() == VarPoint::LOOP_OFFSET) {

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

        // If not done, continue based on type of vectorization.
        if (!codeStr.length()) {

            // Scalar GP?
            if (gp.getVecType() == VarPoint::VEC_NONE) {
#ifdef DEBUG_GP
                cout << " //** reading from point " << gp.makeStr() << " as scalar.\n";
#endif
                codeStr = readFromScalarPoint(os, gp);
            }

            // Non-scalar but non-vectorizable GP?
            else if (gp.getVecType() == VarPoint::VEC_PARTIAL) {
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
        }

        // Remember this point and return it.
        if (codeStr.length())
            savePointVar(gp, codeStr);
        return codeStr;
    }

    // Print any immediate memory writes to 'os'.
    // Return code to update a vector of var points or null string
    // if all writes were printed.
    string CppVecPrintHelper::writeToPoint(ostream& os, const VarPoint& gp,
                                           const string& val) {

        // Can we use a pointer?
        if (gp.getLoopType() == VarPoint::LOOP_OFFSET) {

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

        // If no pointer, use vec write.
        // NB: currently, all eqs must be vectorizable on LHS,
        // so we only need to handle vectorized writes.
        // TODO: relax this restriction.
        printAlignedVecWrite(os, gp, val);

        return "";              // no returned expression.
    }


    // Print aligned memory read.
    string CppVecPrintHelper::printAlignedVecRead(ostream& os, const VarPoint& gp) {

        printPointComment(os, gp, "Read aligned");
        auto rvn = printVecPointCall(os, gp, "readVecNorm", "", "__LINE__", true);

        // Read memory.
        string mvName = makeVarName();
        os << _linePrefix << getVarType() << " " << mvName << " = " << rvn << _lineSuffix;
        return mvName;
    }

    // Print unaliged memory read.
    // Assumes this results in same values as printUnalignedVec().
    string CppVecPrintHelper::printUnalignedVecRead(ostream& os, const VarPoint& gp) {
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
    string CppVecPrintHelper::printAlignedVecWrite(ostream& os, const VarPoint& gp,
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
    string CppVecPrintHelper::printUnalignedVec(ostream& os, const VarPoint& gp) {
        printPointComment(os, gp, "Construct unaligned");

        // Declare var.
        string pvName = makeVarName();
        os << _linePrefix << getVarType() << " " << pvName << _lineSuffix;

        // Contruct it.
        printUnalignedVecCtor(os, gp, pvName);
        return pvName;
    }

    // Print per-element construction for one point var pvName from elems.
    void CppVecPrintHelper::printUnalignedVecSimple(ostream& os, const VarPoint& gp,
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
                gp.getVarName() << "(" << elemStr << ")" << _lineSuffix;
        }
    }

    // Print init of element indices.
    // Fill _vec2elemMap as side-effect.
    void CppVecPrintHelper::printElemIndices(ostream& os) {
        auto& fold = getFold();
        os << "\n // Element indices derived from vector indices.\n";
        int i = 0;
        for (auto& dim : fold) {
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

    // Print invariant var-access vars for non-time loop(s).
    string CppStepVarPrintVisitor::visit(VarPoint* gp) {

        // Pointer to var.
        string varPtr = _cvph.getLocalVar(_os, getVarPtr(*gp), CppPrintHelper::_var_ptr_restrict_type);
        
        // Time var.
        auto& dims = _cvph.getDims();
        _cvph.getLocalVar(_os, gp->makeStepArgStr(varPtr, dims),
                          CppPrintHelper::_step_val_type);
        return "";
    }

    // Print invariant var-access vars for an inner loop.
    string CppLoopVarPrintVisitor::visit(VarPoint* gp) {

        // Retrieve prior analysis of this var point.
        auto loopType = gp->getLoopType();

        // If invariant, we can load now.
        if (loopType == VarPoint::LOOP_INVARIANT) {

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

