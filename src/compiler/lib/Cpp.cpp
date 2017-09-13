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

////////// Support for YASK C++ scalar and vector-code generation //////////////

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

    // Read from a single point to be broadcast to a vector.
    // Return code for read.
    string CppVecPrintHelper::readFromScalarPoint(ostream& os, const GridPoint& gp) {

        // Assume that broadcast will be handled automatically by
        // operator overloading in kernel code.
        // Specify that any indices should use element vars.
        string str = "_context->" + gp.getGridName() + "->readElem(";
        string args = gp.makeArgStr(&_varMap);
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
        size_t pelem = 0;
        getFold().visitAllPoints([&](const IntTuple& vecPoint){

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
                string varname;
                string args = gp.makeArgStr(&vMap);
                string stmt = "_context->" + gp.getGridName() + "->readElem(";
                if (args.length())
                    stmt += args + ", ";
                stmt += "__LINE__)";
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

                pelem++;

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
        string args = isNorm ? gp.makeNormArgStr(getFold()) : gp.makeArgStr();
        os << "{" << args << "}"; // use Indices initializer-list ctor.
        if (lastArg.length()) 
            os << ", " << lastArg;
        os << ")";
    }

    // Print code to set pointers of aligned reads.
    void CppVecPrintHelper::printBasePtrs(ostream& os) {

        // Gather points.
        GridPointSet gps;

        // Aligned reads as determined by VecInfoVisitor.
        gps = _vv._alignedVecs;

        // Writes (assume aligned).
        gps.insert(_vv._vecWrites.begin(), _vv._vecWrites.end());
        
        for (auto& gp : gps) {
        
            // Make base point (inner-dim index = 0).
            auto bgp = makeBasePoint(gp);

            // Not already saved?
            if (!lookupPointPtr(*bgp)) {

                // Print code to calculate pointer.
                string expr = printPointPtr(os, *bgp);

                // Save for future use.
                savePointPtr(*bgp, expr);
            }
        }
    }

    // Print calculation of pointer to a grid point.
    // Return name of pointer.
    string CppVecPrintHelper::printPointPtr(ostream& os, const GridPoint& gp) {
        printPointComment(os, gp, "Calculate pointer to ");

        // Calc ptr.
        string mvName = makeVarName();
        os << _linePrefix << getVarType() << "* " << mvName << " = ";
        printVecPointCall(os, gp, "getVecPtrNorm", "", "__LINE__", true);
        os << _lineSuffix;
        return mvName;
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

                // Offset.
                string idim = _dims._innerDim;
                string ofs = gp.makeNormArgStr(idim, getFold());
                
                // Output read using base addr.
                printPointComment(os, gp, "Read aligned");
                codeStr = makeVarName();
                os << _linePrefix << getVarType() << " " << codeStr << " = " <<
                    *p << "[" << ofs << "]" << _lineSuffix;
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
                string ofs = gp.makeNormArgStr(idim, getFold());
                
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

    // Print body of prefetch function.
    void CppVecPrintHelper::printPrefetches(ostream& os, const IntScalar& dir) const {

        // Points to prefetch.
        GridPointSet* pfPts = NULL;

        // Prefetch leading points only if dir name is set.
        GridPointSet edge;
        if (dir.getName().length()) {
            _vv.getLeadingEdge(edge, dir);
            pfPts = &edge;
        }

        // if dir is not set, prefetch all aligned vectors.
        else
            pfPts = &_vv._alignedVecs;

        for (auto gp : *pfPts) {
            printPointComment(os, gp, "Prefetch aligned");
            
            // Prefetch memory.
            printVecPointCall(os, gp, "prefetchVecNorm<level>", "", "__LINE__", true);
            os << ";" << endl;
        }
    }

    // Print a grid-access vars for a loop.
    void CppLoopVarPrintVisitor::visit(GridPoint* gp) {

        // Retrieve prior analysis.
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
    
    // Print extraction of indices.
    void YASKCppPrinter::printIndices(ostream& os) const {
        os << endl << " // Extract individual indices.\n";
        int i = 0;
        for (auto& dim : _dims._stencilDims.getDims()) {
            auto& dname = dim.getName();
            os << " idx_t " << dname << " = idxs[" << i << "];\n";
            i++;
        }
    }

    // Print an expression as a one-line C++ comment.
    void YASKCppPrinter::addComment(ostream& os, EqGroup& eq) {
        
        // Use a simple human-readable visitor to create a comment.
        PrintHelper ph(0, "temp", "", " // ", ".\n");
        PrintVisitorTopDown commenter(os, ph, _settings);
        eq.visitEqs(&commenter);
    }

    // Print a shim function to map hard-coded YASK vars to actual dims.
    void YASKCppPrinter::printShim(ostream& os,
                                   const string& fname,
                                   bool use_template) {
    
        os << "\n // Simple shim function to use start vars and ignore others.\n";
        if (use_template)
            os << " template <int level>";
        os << " inline void " << fname <<
            "(const ScanIndices& scan_idxs) {\n"
            " " << fname;
        if (use_template)
            os << "<level>";
        os << "(scan_idxs.start);\n"
            "}\n";
    }

    // Print YASK code in new stencil context class.
    // TODO: split this into smaller methods.
    void YASKCppPrinter::print(ostream& os) {

        os << "// Automatically-generated code; do not edit.\n"
            "\n////// YASK implementation of the '" << _stencil.getName() <<
            "' stencil //////\n";

        // Macros.
        os << "\n#ifdef DEFINE_MACROS\n";
        printMacros(os);
        os << "\n#endif // DEFINE_MACROS\n";

        // Stencil-context code.
        os << "\n#ifdef DEFINE_CONTEXT\n"
            "namespace yask {" << endl;

        // First, create a class to hold the data (grids).
        {
            // get stats.
            CounterVisitor cve;
            _eqGroups.visitEqs(&cve);

            os << endl << " ////// Stencil-specific data //////" << endl <<
                "struct " << _context_base << " : public StencilContext {" << endl;

            // Grids.
            string ctorCode, ctorList;
            os << "\n ///// Grid(s)." << endl;
            for (auto gp : _grids) {
                string grid = gp->getName();
                int ndims = gp->get_num_dims();

                os << "\n // The ";
                if (ndims)
                    os << ndims << "-D grid";
                else
                    os << "scalar value";
                os << " '" << grid << "', which is ";
                if (_eqGroups.getOutputGrids().count(gp))
                    os << "updated by one or more equations.\n";
                else
                    os << "not updated by any equation (read-only).\n";

                // Type name for grid.
                string typeName;
            
                // Use vector-folded layout if possible.
                bool folded = gp->isFoldable();
                string gtype = folded ? "YkVecGrid" : "YkElemGrid";

                // Type-name in kernel is 'GRID_TYPE<LAYOUT, WRAP_1ST_IDX>'.
                ostringstream oss;
                oss << gtype << "<Layout_";
                int step_posn = 0;
                int inner_posn = 0;

                // 1-D or more.
                if (ndims) {
                    for (int dn = 0; dn < ndims; dn++) {
                        auto& dim = gp->getDims()[dn];
                        auto& dname = dim->getName();
                        auto dtype = dim->getType();
                        
                        // Step dim?
                        if (dtype == STEP_INDEX) {
                            assert(dname == _dims._stepDim);
                            step_posn = dn + 1;
                            if (dn > 0) {
                                cerr << "Error: cannot create grid '" << grid <<
                                    "' with dimension '" << dname << "' in position " <<
                                    dn << "; step dimension must be in first position.\n";
                                exit(1);
                            }
                        }

                        // Inner dim?
                        else if (dname == _dims._innerDim) {
                            assert(dtype == DOMAIN_INDEX);
                            inner_posn = dn + 1;
                        }

                        // Something else.
                        else {
                            int other_posn = dn + 1;
                            oss << other_posn;
                        }
                    }

                    // Add step and inner posns at end.
                    if (step_posn)
                        oss << step_posn;
                    if (inner_posn)
                        oss << inner_posn;
                }

                // Scalar.
                else
                    oss << "0d"; // Trivial scalar layout.

                if (step_posn)
                    oss << ", true /* wrap step index */";
                else
                    oss << ", false /* no wrap */";
                oss << ">";
                typeName = oss.str();

                // Typedef.
                string typeDef = grid + "_type";
                os << " typedef " << typeName << " " << typeDef << ";\n"; 
                
                // Actual grid declaration.
                os << " std::shared_ptr<" << typeDef << "> " << grid << "_ptr;\n" <<
                    " " << typeDef << "* " << grid << ";\n";

                // Grid init.
                ctorCode += "\n // Init grid '" + grid + "'.\n";
                ctorCode += " GridDimNames " + grid + "_dim_names = {\n";
                int ndn = 0;
                for (auto& dim : gp->getDims()) {
                    auto& dname = dim->getName();
                    if (ndn) ctorCode += ", ";
                    ctorCode += "\"" + dname + "\""; // add dim name.
                    ndn++;
                }
                ctorCode += "};\n";
                ctorCode += " " + grid + "_ptr = std::make_shared<" + typeDef +
                    ">(_dims, \"" + grid + "\", " + grid + "_dim_names, &_ostr);\n";
                ctorCode += " " + grid + " = " + grid + "_ptr.get();\n";
                ctorCode += " addGrid(" + grid + "_ptr, ";
                if (_eqGroups.getOutputGrids().count(gp))
                    ctorCode += "true /* is an output grid */";
                else
                    ctorCode += "false /* is not an output grid */";
                ctorCode += ");\n";
                
                // Alloc-setting code.
                for (auto& dim : gp->getDims()) {
                    auto& dname = dim->getName();
                    auto dtype = dim->getType();

                    // domain dimension.
                    if (dtype == DOMAIN_INDEX) {

                        // Halo for this dimension.
                        string hvar = grid + "_halo_" + dname;
                        int hval = _settings._haloSize > 0 ?
                            _settings._haloSize : gp->getHaloSize(dname);
                        os << " const idx_t " << hvar << " = " << hval <<
                            "; // default halo size in '" << dname << "' dimension.\n";
                        ctorCode += " " + grid + "->set_halo_size(\"" + dname +
                            "\", " + hvar + ");\n";
                    }

                    // non-domain dimension.
                    else {
                        string avar = grid + "_alloc_" + dname;
                        string ovar = grid + "_ofs_" + dname;
                        int aval = 1;
                        int oval = 0;
                        if (dtype == STEP_INDEX) {
                            aval = _settings._stepAlloc > 0 ?
                                _settings._stepAlloc : gp->getStepDimSize();
                        } else {
                            auto* minp = gp->getMinIndices().lookup(dname);
                            auto* maxp = gp->getMaxIndices().lookup(dname);
                            if (minp && maxp) {
                                aval = *maxp - *minp + 1;
                                oval = *minp;
                            }
                        }
                        os << " const idx_t " << avar << " = " << aval <<
                            "; // default allocation in '" << dname << "' dimension.\n";
                        ctorCode += " " + grid + "->set_alloc_size(\"" + dname +
                            "\", " + avar + ");\n";
                        if (oval) {
                            os << " const idx_t " << ovar << " = " << oval <<
                                "; // first index in '" << dname << "' dimension.\n";
                        ctorCode += " " + grid + "->set_first_misc_index(\"" + dname +
                            "\", " + ovar + ");\n";
                        }
                    }
                }
            }

            // Ctor.
            {
                os << "\n // Constructor.\n" <<
                    " " << _context_base << "(KernelEnvPtr env, KernelSettingsPtr settings) :"
                    " StencilContext(env, settings)" << ctorList <<
                    " {\n  name = \"" << _stencil.getName() << "\";\n";

                os << "\n // Create grids (but do not allocate data in them).\n" <<
                    ctorCode <<
                    "\n // Update grids with context info.\n"
                    " update_grids();\n";
            
                // end of ctor.
                os << " }" << endl;
            }
            os << "}; // " << _context_base << endl;
        }
        
        // A struct for each equation group.
        for (size_t ei = 0; ei < _eqGroups.size(); ei++) {

            // Scalar eqGroup.
            auto& eq = _eqGroups.at(ei);
            string egName = eq.getName();
            string egDesc = eq.getDescription();
            string egsName = "EqGroup_" + egName;

            os << endl << " ////// Stencil " << egDesc << " //////\n" <<
                "\n class " << egsName << " : public EqGroupBase {\n"
                " protected:\n"
                " " << _context_base << "* _context = 0;\n"
                " public:\n";

            // Stats for this eqGroup.
            CounterVisitor stats;
            eq.visitEqs(&stats);
            
            // Example computation.
            os << endl << " // " << stats.getNumOps() << " FP operation(s) per point:" << endl;
            addComment(os, eq);

            // Eq-group ctor.
            {
                os << " " << egsName << "(" << _context_base << "* context) :\n"
                    " EqGroupBase(context),\n"
                    " _context(context) {\n"
                    " _name = \"" << egName << "\";\n"
                    " _scalar_fp_ops = " << stats.getNumOps() << ";\n"
                    " _scalar_points_read = " << stats.getNumReads() << ";\n"
                    " _scalar_points_written = " << stats.getNumWrites() << ";\n";

                // I/O grids.
                if (eq.getOutputGrids().size()) {
                    os << "\n // The following grids are written by " << egsName << endl;
                    for (auto gp : eq.getOutputGrids())
                        os << "  outputGridPtrs.push_back(_context->" << gp->getName() << "_ptr);" << endl;
                }
                if (eq.getInputGrids().size()) {
                    os << "\n // The following grids are read by " << egsName << endl;
                    for (auto gp : eq.getInputGrids())
                        os << "  inputGridPtrs.push_back(_context->" << gp->getName() << "_ptr);" << endl;
                }
                os << " } // Ctor." << endl;
            }

            // Condition.
            {
                os << endl << " // Determine whether " << egsName << " is valid at the indices " <<
                    _dims._stencilDims.makeDimStr() << ".\n"
                    " // Return true if indices are within the valid sub-domain or false otherwise.\n"
                    " virtual bool is_in_valid_domain(const Indices& idxs) {\n";
                printIndices(os);
                if (eq.cond.get())
                    os << " return " << eq.cond->makeStr() << ";" << endl;
                else
                    os << " return true; // full domain." << endl;
                os << " }" << endl;
            }
        
            // Scalar code.
            {
                // Stencil-calculation code.
                // Function header.
                os << endl << " // Calculate one scalar result relative to indices " <<
                    _dims._stencilDims.makeDimStr() << ".\n"
                    " // There are approximately " << stats.getNumOps() <<
                    " FP operation(s) per invocation.\n"
                    " virtual void calc_scalar(const Indices& idxs) {\n";
                printIndices(os);

                // C++ scalar print assistant.
                CounterVisitor cv;
                eq.visitEqs(&cv);
                CppPrintHelper* sp = new CppPrintHelper(&cv, "temp", "real_t", " ", ";\n");
            
                // Generate the code.
                PrintVisitorBottomUp pcv(os, *sp, _settings);
                eq.visitEqs(&pcv);

                // End of function.
                os << "} // calc_scalar." << endl;

                delete sp;
            }

            // Cluster/Vector code.
            {
                // Cluster eqGroup at same 'ei' index.
                // This should be the same eq-group because it was copied from the
                // scalar one.
                auto& ceq = _clusterEqGroups.at(ei);
                assert(egDesc == ceq.getDescription());

                // Create vector info for this eqGroup.
                // The visitor is accepted at all nodes in the cluster AST;
                // for each grid access node in the AST, the vectors
                // needed are determined and saved in the visitor.
                VecInfoVisitor vv(_dims);
                ceq.visitEqs(&vv);

                // Reorder some equations based on vector info.
                ExprReorderVisitor erv(vv);
                ceq.visitEqs(&erv);

                // Collect stats.
                CounterVisitor cv;
                ceq.visitEqs(&cv);
                int numResults = _dims._clusterPts.product();

#if 0
                // Cluster-calculation code.
                {
                    
                    // Function header.
                    os << endl << " // Calculate " << numResults <<
                        " result(s) relative to indices " << _dims._stencilDims.makeDimStr() <<
                        " in a '" << _dims._clusterPts.makeDimValStr(" * ") <<
                        "'-point cluster containing " << _dims._clusterMults.product() << " '" <<
                        _dims._fold.makeDimValStr(" * ") << "' vector(s).\n"
                        " // Indices must be rank-relative.\n"
                        " // Indices must be normalized, i.e., already divided by VLEN_*.\n"
                        " // SIMD calculations use " << vv.getNumPoints() <<
                        " vector block(s) created from " << vv.getNumAlignedVecs() <<
                        " aligned vector-block(s).\n"
                        " // There are approximately " << (stats.getNumOps() * numResults) <<
                        " FP operation(s) per invocation.\n"
                        " inline void calc_cluster(const Indices& idxs) {\n";
                    printIndices(os);

                    // C++ vector print assistant.
                    CppVecPrintHelper* vp = newCppVecPrintHelper(vv, cv);
            
                    // Generate the code.
                    // Visit all expressions to cover the whole cluster.
                    PrintVisitorBottomUp pcv(os, *vp, _settings);
                    ceq.visitEqs(&pcv);
                
                    // End of function.
                    os << "} // calc_cluster." << endl;
                    delete vp;
                }
                
                // Insert shim function.
                printShim(os, "calc_cluster");
            
                // Generate prefetch code for no specific direction and then each
                // orthogonal direction.
                for (int diri = 0; diri < _dims._stencilDims.size(); diri++) {

                    // Create a direction object.
                    // If diri == 0, no direction.
                    // If diri > 0, add a direction.
                    IntScalar dir;
                    if (diri > 0) {
                        auto& dim = _dims._stencilDims.getDim(diri);
                        auto& dname = dim.getName();

                        // Magnitude of dimension is based on cluster.
                        const int* p = _dims._clusterMults.lookup(dname);
                        int m = p ? *p : 1;
                        dir.setName(dname);
                        dir.setVal(m);
                    }

                    // Function header.
                    os << endl << " // Prefetch cache line(s) ";
                    if (dir.getName().length())
                        os << "for leading edge of stencil advancing by " <<
                            dir.getVal() << " vector(s) in '+" <<
                            dir.getName() << "' direction ";
                    else
                        os << "for entire stencil ";
                    os << "relative to indices " << _dims._stencilDims.makeDimStr() <<
                        " in a '" << _dims._clusterPts.makeDimValStr(" * ") <<
                        "'-point cluster containing " << _dims._clusterMults.product() << " '" <<
                        _dims._fold.makeDimValStr(" * ") << "' vector(s).\n"
                        " // Indices must be rank-relative.\n"
                        " // Indices must be normalized, i.e., already divided by VLEN_*.\n";

                    ostringstream oss;
                    oss << "prefetch_cluster";
                    if (diri > 0)
                        oss << "_dir_" << diri;
                    string fname = oss.str();
                    os << " template<int level> inline void " << fname <<
                        "(const Indices& idxs) {\n";
                    printIndices(os);

                    // C++ vector print assistant.
                    CppVecPrintHelper* vp = newCppVecPrintHelper(vv, cv);
            
                    // C++ prefetch code.
                    vp->printPrefetches(os, dir);

                    // End of function.
                    os << "} // " << fname << "." << endl;

                    // Insert shim function.
                    printShim(os, fname, true);
                    delete(vp);

                } // direction.

                // Sub-block.
                {
                    os << endl <<
                        " // Calculate one sub-block of whole clusters.\n"
                        " // Indices must be rank-relative.\n"
                        " // Indices must be normalized, i.e., already divided by VLEN_*.\n"
                        " virtual void calc_sub_block_of_clusters("
                        "const ScanIndices& block_idxs) {\n"
                        "\n"
                        " ScanIndices sub_block_idxs;\n"
                        " sub_block_idxs.initFromOuter(block_idxs);\n"
                        " // Step sizes are based on cluster lengths.\n";
                    int i = 0;
                    for (auto& dim : _dims._stencilDims.getDims()) {
                        auto& dname = dim.getName();
                        string ucDim = allCaps(dname);
                        if (dname != _dims._stepDim)
                            os << " sub_block_idxs.step[" << i << "] = CMULT_" << ucDim << ";\n";
                        i++;
                    }
                    os << " #if !defined(DEBUG) && defined(__INTEL_COMPILER)\n"
                        " #pragma forceinline recursive\n"
                        " #endif\n"
                        " {\n"
                        "  // Include automatically-generated loop code that calls calc_cluster()"
                        "  and optionally, the prefetch function(s).\n"
                        "  #include \"yask_sub_block_loops.hpp\"\n"
                        " }\n"
                        "} // calc_sub_block_of_clusters\n";
                }
#endif
                
                // Loop-calculation code.
                {

                    // Function header.
                    string idim = _dims._innerDim;
                    string istart = "start_" + idim;
                    string istop = "stop_" + idim;
                    string istep = "step_" + idim;
                    os << endl << " // Calculate a series of clusters iterating in '" << idim <<
                        "' direction from " << _dims._stencilDims.makeDimStr() <<
                        " indices in 'idxs' to '" << istop << "' by '" << istep << "'.\n" <<
                        " // Each cluster calculates '" << _dims._clusterPts.makeDimValStr(" * ") <<
                        "' points containing " << _dims._clusterMults.product() << " '" <<
                        _dims._fold.makeDimValStr(" * ") << "' vector(s).\n"
                        " // Indices must be rank-relative.\n"
                        " // Indices must be normalized, i.e., already divided by VLEN_*.\n"
                        " // SIMD calculations use " << vv.getNumPoints() <<
                        " vector block(s) created from " << vv.getNumAlignedVecs() <<
                        " aligned vector-block(s).\n"
                        " // There are approximately " << (stats.getNumOps() * numResults) <<
                        " FP operation(s) per loop iteration.\n"
                        " void calc_loop_of_clusters(const Indices& idxs, idx_t " <<
                        istop << ", idx_t " << istep << ") {\n";
                    printIndices(os);
                    os << " idx_t " << istart << " = " << idim << ";\n";

                    // C++ vector print assistant.
                    CppVecPrintHelper* vp = newCppVecPrintHelper(vv, cv);

                    // Start forced-inline code.
                    os << "\n // Force inlining if possible.\n"
                        "#if !defined(DEBUG) && defined(__INTEL_COMPILER)\n"
                        "#pragma forceinline recursive\n"
                        "#endif\n"
                        " {\n";
                    
                    // Print invariants.
                    CppLoopVarPrintVisitor lvv(os, *vp, _settings);
                    ceq.visitEqs(&lvv);

                    // Print pointers.
                    vp->printBasePtrs(os);

                    // Actual Loop.
                    os << "\n // Inner loop.\n"
                        " for (idx_t " << idim << " = " << istart << "; " <<
                        idim << " < " << istop << "; " <<
                        idim << " += " << istep << ") {\n";
            
                    // Generate loop body using vars stored in print helper.
                    // Visit all expressions to cover the whole cluster.
                    PrintVisitorBottomUp pcv(os, *vp, _settings);
                    ceq.visitEqs(&pcv);

                    // End of loop.
                    os << " } // '" << idim << "' loop.\n";

                    // End forced-inline code.
                    os << " } // Forced-inline block.\n";
                    
                    // End of function.
                    os << "} // calc_loop_of_clusters.\n";
                    delete vp;
                }
            }

            os << "}; // " << egsName << ".\n"; // end of class.
            
        } // stencil eqGroups.

        // Finish the context.
        {
            os << endl << " ////// Overall stencil-specific context //////" << endl <<
                "struct " << _context << " : public " << _context_base << " {" << endl;

            // Stencil eqGroup objects.
            os << endl << " // Stencil equation-groups." << endl;
            for (auto& eg : _eqGroups) {
                string egName = eg.getName();
                string egsName = "EqGroup_" + egName;
                os << " " << egsName << " eqGroup_" << egName << ";" << endl;
            }

            // Ctor.
            os << "\n // Constructor.\n" <<
                " " << _context << "(KernelEnvPtr env, KernelSettingsPtr settings) : " <<
                _context_base << "(env, settings)";
            for (auto& eg : _eqGroups) {
                string egName = eg.getName();
                os << ",\n  eqGroup_" << egName << "(this)";
            }
            os << " {\n";
        
            // Push eq-group pointers to list.
            os << "\n // Equation groups.\n";
            for (auto& eg : _eqGroups) {
                string egName = eg.getName();
                os << "  eqGroups.push_back(&eqGroup_" << egName << ");\n";

                // Add dependencies.
                for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1)) {
                    for (auto dep : eg.getDeps(dt)) {
                        string dtName = (dt == certain_dep) ? "certain_dep" :
                            (dt == possible_dep) ? "possible_dep" :
                            "internal_error";
                        os << "  eqGroup_" << egName <<
                            ".add_dep(yask::" << dtName <<
                            ", &eqGroup_" << dep << ");\n";
                    }
                }
            }
            os << " } // Ctor.\n";

            // Dims creator.
            os << "\n  // Create Dims object.\n"
                "  static DimsPtr new_dims() {\n"
                "    auto p = std::make_shared<Dims>();\n"
                "    p->_step_dim = \"" << _dims._stepDim << "\";\n"
                "    p->_inner_dim = \"" << _dims._innerDim << "\";\n";
            for (auto& dim : _dims._domainDims.getDims()) {
                auto& dname = dim.getName();
                os << "    p->_domain_dims.addDimBack(\"" << dname << "\", 0);\n";
            }
            for (auto& dim : _dims._stencilDims.getDims()) {
                auto& dname = dim.getName();
                os << "    p->_stencil_dims.addDimBack(\"" << dname << "\", 0);\n";
            }
            for (auto& dim : _dims._miscDims.getDims()) {
                auto& dname = dim.getName();
                os << "    p->_misc_dims.addDimBack(\"" << dname << "\", 0);\n";
            }
            for (auto& dim : _dims._fold.getDims()) {
                auto& dname = dim.getName();
                auto& dval = dim.getVal();
                os << "    p->_fold_pts.addDimBack(\"" << dname << "\", " << dval << ");\n";
            }
            for (auto& dim : _dims._foldGT1.getDims()) {
                auto& dname = dim.getName();
                auto& dval = dim.getVal();
                os << "    p->_vec_fold_pts.addDimBack(\"" << dname << "\", " << dval << ");\n";
            }
            string ffi = (_dims._fold.isFirstInner()) ? "true" : "false";
            os << "    p->_fold_pts.setFirstInner(" << ffi << ");\n"
                "    p->_vec_fold_pts.setFirstInner(" << ffi << ");\n";
            for (auto& dim : _dims._clusterPts.getDims()) {
                auto& dname = dim.getName();
                auto& dval = dim.getVal();
                os << "    p->_cluster_pts.addDimBack(\"" << dname << "\", " << dval << ");\n";
            }
            for (auto& dim : _dims._clusterMults.getDims()) {
                auto& dname = dim.getName();
                auto& dval = dim.getVal();
                os << "    p->_cluster_mults.addDimBack(\"" << dname << "\", " << dval << ");\n";
            }
            
            os << "    return p;\n"
                "  }\n";
        
            // Stencil provided code for StencilContext
            CodeList *extraCode;
            if ( (extraCode = _stencil.getExtensionCode(STENCIL_CONTEXT)) != NULL )
            {
                os << "\n  // Functions provided by user.\n";
                for ( auto code : *extraCode )
                    os << code << endl;
            }
        
            os << "}; // " << _context << endl;
        }

        os << "} // namespace yask.\n"
            "#endif // DEFINE_CONTEXT\n"
            "\n//End of automatically-generated code." << endl;
    }

    // Print YASK macros.  TODO: get rid of all or most of the macros
    // in favor of consts or templates.
    void YASKCppPrinter::printMacros(ostream& os) {

        os << "// Stencil solution:\n"
            "#define YASK_STENCIL_NAME \"" << _stencil.getName() << "\"\n"
            "#define YASK_STENCIL_CONTEXT " << _context << endl;

        os << "\n// FP precision:\n"
            "#define REAL_BYTES " << _settings._elem_bytes << endl;

        os << "\n// Number of stencil dimensions (step and domain):\n"
            "#define NUM_STENCIL_DIMS " << _dims._stencilDims.size() << endl;
        
        // Vec/cluster lengths.
        os << "\n// One vector fold: " << _dims._fold.makeDimValStr(" * ") << endl;
        for (auto& dim : _dims._fold.getDims()) {
            auto& dname = dim.getName();
            string ucDim = allCaps(dname);
            os << "#define VLEN_" << ucDim << " (" << dim.getVal() << ")" << endl;
        }
        os << "#define VLEN (" << _dims._fold.product() << ")" << endl;
        os << "#define VLEN_FIRST_INDEX_IS_UNIT_STRIDE (" <<
            (_dims._fold.isFirstInner() ? 1 : 0) << ")" << endl;
        os << "#define USING_UNALIGNED_LOADS (" <<
            (_settings._allowUnalignedLoads ? 1 : 0) << ")" << endl;

        os << endl;
        os << "// Cluster multipliers of vector folds: " <<
            _dims._clusterMults.makeDimValStr(" * ") << endl;
        for (auto& dim : _dims._clusterMults.getDims()) {
            auto& dname = dim.getName();
            string ucDim = allCaps(dname);
            os << "#define CMULT_" << ucDim << " (" <<
                dim.getVal() << ")" << endl;
        }
        os << "#define CMULT (" << _dims._clusterMults.product() << ")" << endl;
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

