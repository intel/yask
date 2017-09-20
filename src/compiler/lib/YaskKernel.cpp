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
        printData(os);
        
        // A struct for each equation group.
        printEqGroups(os);

        // Finish the context.
        printContext(os);

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
        auto nvec = _dims._foldGT1.getNumDims();
        os << "\n// One vector fold: " << _dims._fold.makeDimValStr(" * ") << endl;
        for (auto& dim : _dims._fold.getDims()) {
            auto& dname = dim.getName();
            string ucDim = allCaps(dname);
            os << "#define VLEN_" << ucDim << " (" << dim.getVal() << ")" << endl;
        }
        os << "#define VLEN (" << _dims._fold.product() << ")" << endl;
        os << "#define FIRST_FOLD_INDEX_IS_UNIT_STRIDE (" <<
            (_dims._fold.isFirstInner() ? 1 : 0) << ")" << endl;
        os << "#define NUM_VEC_FOLD_DIMS (" << nvec << ")" << endl;

        // Layout for folding.
        ostringstream oss;
        oss << "Layout_";
        if (nvec) {
            if (_dims._foldGT1.isFirstInner())
                for (int i = nvec; i > 0; i--)
                    oss << i;       // e.g., 321
            else
                for (int i = 1; i <= nvec; i++)
                    oss << i;       // e.g., 123
        }
        else
            oss << "0d";         // no folding; scalar layout
        string layout = oss.str();
        os << "#define VEC_FOLD_LAYOUT " << layout << endl;

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

    // Print YASK data class.
    void YASKCppPrinter::printData(ostream& os) {

        // get stats.
        CounterVisitor cve;
        _eqGroups.visitEqs(&cve);

        os << endl << " ////// Stencil-specific data //////" << endl <<
            "struct " << _context_base << " : public StencilContext {" << endl;

        // Save data for ctor and new-grid method.
        string ctorCode, ctorList, newGridCode;
        set<string> newGridDims;
        
        // Grids.
        os << "\n ///// Grid(s)." << endl;
        for (auto gp : _grids) {
            string grid = gp->getName();
            int ndims = gp->get_num_dims();

            // Tuple version of dims.
            IntTuple gdims;
            for (int dn = 0; dn < ndims; dn++) {
                auto& dim = gp->getDims()[dn];
                auto& dname = dim->getName();
                gdims.addDimBack(dname, 0);
            }

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
                                "' with dimensions '" << gdims.makeDimStr() <<
                                "' because '" << dname << "' must be first dimension.\n";
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
                " " << typeDef << "* " << grid << ";\n"
                " GridDimNames " + grid + "_dim_names;\n";

            // Grid init.
            ctorCode += "\n // Init grid '" + grid + "'.\n";
            ctorCode += " " + grid + "_dim_names = {\n" +
                gdims.makeDimStr(", ", "\"", "\"") + "};\n";
            ctorCode += " " + grid + "_ptr = std::make_shared<" + typeDef +
                ">(_dims, \"" + grid + "\", " + grid + "_dim_names, &_ostr);\n";

            // Make new grids via API.
            string newGridKey = gdims.makeDimStr();
            if (!newGridDims.count(newGridKey)) {
                newGridDims.insert(newGridKey);
                bool firstGrid = newGridCode.length() == 0;
                if (gdims.getNumDims())
                    newGridCode += "\n // Grids with '" + newGridKey + "' dim(s).\n";
                else
                    newGridCode += "\n // Scalar grids.\n";
                if (!firstGrid)
                    newGridCode += " else";
                newGridCode += " if (dims == " + grid + "_dim_names) gp = std::make_shared<" +
                    typeDef + ">(_dims, name, dims, &_ostr);\n";
            }

            // Finish grid init.
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
            os << " } // ctor" << endl;
        }

        // New-grid method.
        {
            os << "\n // Make a new grid iff its dims match any in the stencil.\n"
                " // Returns pointer to the new grid or nullptr if no match.\n"
                " virtual YkGridPtr newStencilGrid(const std::string& name,"
                " const GridDimNames& dims) {\n"
                " YkGridPtr gp;\n" <<
                newGridCode <<
                " return gp;\n"
                " } // newStencilGrid\n";
        }
        
        os << "}; // " << _context_base << endl;
    }

    // Print YASK equation groups.
    void YASKCppPrinter::printEqGroups(ostream& os) {
        
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
                        " indices in 'idxs' to '" << istop << "'.\n" <<
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
                        istop << ") {\n";
                    printIndices(os);
                    os << " idx_t " << istart << " = " << idim << ";\n";
                    os << " idx_t " << istep << " = CMULT_" << allCaps(idim) << "; // number of vectors.\n";

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
    }

    // Print final YASK context.
    void YASKCppPrinter::printContext(ostream& os) {
        
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
            "    auto p = std::make_shared<Dims>();\n";
        for (int i = 0; i < _dims._foldGT1.getNumDims(); i++)
            os << "    p->_vec_fold_layout.set_size(" << i << ", " <<
                _dims._foldGT1[i] << "); // '" << _dims._foldGT1.getDimName(i) << "'\n";
        os <<
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

} // namespace yask.

