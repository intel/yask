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
    void YASKCppPrinter::addComment(ostream& os, EqBundle& eq) {

        // Use a simple human-readable visitor to create a comment.
        PrintHelper ph(_settings, _dims, NULL, "temp", "", " // ", ".\n");
        PrintVisitorTopDown commenter(os, ph);
        eq.visitEqs(&commenter);
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

        // A struct for each equation bundle.
        printEqBundles(os);

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
            "#define REAL_BYTES (" << _settings._elem_bytes << ")\n";

        os << "\n// Number of domain dimensions:\n"
            "#define NUM_DOMAIN_DIMS " << _dims._domainDims.size() << "\n";
        int i = 0;
        for (auto& dim : _dims._domainDims.getDims()) {
            auto& dname = dim.getName();
            os << "#define DOMAIN_DIM_IDX_" << dname << " (" << (i++) << ")\n";
        }
        os << "\n// Number of stencil dimensions (step and domain):\n"
            "#define NUM_STENCIL_DIMS " << _dims._stencilDims.size() << "\n";
        i = 0;
        for (auto& dim : _dims._stencilDims.getDims()) {
            auto& dname = dim.getName();
            os << "#define STENCIL_DIM_IDX_" << dname << " (" << (i++) << ")\n";
        }
        int gdims = 0;
        for (auto gp : _grids) {
            int ndims = gp->get_num_dims();
            gdims = max(gdims, ndims);
        }
        os << "\n// Max number of grid dimensions:\n"
            "#define NUM_GRID_DIMS " << gdims << endl;

        os << "\n// Number of stencil equations:\n"
            "#define NUM_STENCIL_EQS " << _stencil.get_num_equations() << endl;

        // Vec/cluster lengths.
        auto nvec = _dims._foldGT1.getNumDims();
        os << "\n// One vector fold: " << _dims._fold.makeDimValStr(" * ") << endl;
        for (auto& dim : _dims._fold.getDims()) {
            auto& dname = dim.getName();
            string ucDim = allCaps(dname);
            os << "#define VLEN_" << ucDim << " (" << dim.getVal() << ")" << endl;
        }
        os << "namespace yask {\n"
            " constexpr idx_t fold_pts[]{ " << _dims._fold.makeValStr() << " };\n"
            "}\n";
        os << "#define VLEN (" << _dims._fold.product() << ")" << endl;
        os << "#define FIRST_FOLD_INDEX_IS_UNIT_STRIDE (" <<
            (_dims._fold.isFirstInner() ? 1 : 0) << ")" << endl;
        os << "#define NUM_VEC_FOLD_DIMS (" << nvec << ")" << endl;

        // Layout for folding.
        // This contains only the vectorized (len > 1) dims.
        string layout;
        if (_dims._foldGT1.isFirstInner())
            for (int i = nvec; i > 0; i--)
                layout += to_string(i);       // e.g., 321
        else
            for (int i = 1; i <= nvec; i++)
                layout += to_string(i);       // e.g., 123
        os << "#define VEC_FOLD_LAYOUT_CLASS Layout_";
        if (nvec)
            os << layout << endl;
        else
            os << "0d\n";         // no folding; scalar layout
        os << "#define VEC_FOLD_LAYOUT(idxs) ";
        if (nvec) {
            os << "LAYOUT_" << layout << "(";
            for (int i = 0; i < nvec; i++) {
                if (i) os << ", ";
                os << "idxs[" << i << "]";
            }
            for (int i = 0; i < nvec; i++)
                os << ", " << _dims._foldGT1[i]; // fold lengths.
            os << ")\n";
        } else
            os << "(0)\n";

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
        _eqBundles.visitEqs(&cve);

        os << endl << " ////// Stencil-specific data //////" << endl <<
            "class " << _context_base << " : public StencilContext {\n"
            "public:\n";

        // Save data for ctor and new-grid method.
        string ctorCode, ctorList, newGridCode, scratchCode;
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
                os << ndims << "-D grid var";
            else
                os << "scalar value";
            os << " '" << grid << "', which is ";
            if (gp->isScratch())
                os << " a scratch variable.\n";
            else if (_eqBundles.getOutputGrids().count(gp))
                os << "updated by one or more equations.\n";
            else
                os << "not updated by any equation (read-only).\n";
            if (ndims) {
                os << " // Dimensions: ";
                for (int dn = 0; dn < ndims; dn++) {
                    if (dn) os << ", ";
                    auto& dim = gp->getDims()[dn];
                    auto& dname = dim->getName();
                    os << "'" << dname << "'(#" << (dn+1) << ")";
                }
                os << ".\n";
            }

            // Use vector-folded layout if possible.
            bool folded = gp->isFoldable();
            string gtype = folded ? "YkVecGrid" : "YkElemGrid";

            // Type-name in kernel is 'GRID_TYPE<LAYOUT, WRAP_1ST_IDX, VEC_LENGTHS...>'.
            string typeName = gtype + "<Layout_";
            int step_posn = 0;
            int inner_posn = 0;
            vector<int> vlens;
            vector<int> misc_posns;

            // 1-D or more.
            if (ndims) {
                for (int dn = 0; dn < ndims; dn++) {
                    auto& dim = gp->getDims()[dn];
                    auto& dname = dim->getName();
                    auto dtype = dim->getType();
                    bool defer = false; // add dim later.

                    // Step dim?
                    // If this exists, it will get placed near to the end,
                    // just before the inner & misc dims.
                    if (dtype == STEP_INDEX) {
                        assert(dname == _dims._stepDim);
                        if (dn > 0) {
                            THROW_YASK_EXCEPTION("Error: cannot create grid '" + grid +
                                                 "' with dimensions '" + gdims.makeDimStr() +
                                                 "' because '" + dname + "' must be first dimension");
                        }
                        if (folded) {
                            step_posn = dn + 1;
                            defer = true;
                        }
                    }

                    // Inner domain dim?
                    // If this exists, it will get placed at or near the end.
                    else if (dname == _dims._innerDim) {
                        assert(dtype == DOMAIN_INDEX);
                        if (folded) {
                            inner_posn = dn + 1;
                            defer = true;
                        }
                    }

                    // Misc dims? Placed after the inner domain dim if requested.
                    else if (dtype == MISC_INDEX) {
                        if (folded && _settings._innerMisc) {
                            misc_posns.push_back(dn + 1);
                            defer = true;
                        }
                    }

                    // Add index position to layout.
                    if (!defer) {
                        int other_posn = dn + 1;
                        typeName += to_string(other_posn);
                    }

                    // Add vector len to list.
                    if (folded) {
                        auto* p = _dims._fold.lookup(dname);
                        int dval = p ? *p : 1;
                        vlens.push_back(dval);
                    }
                }

                // Add deferred posns at end.
                if (step_posn)
                    typeName += to_string(step_posn);
                if (inner_posn)
                    typeName += to_string(inner_posn);
                for (auto mp : misc_posns)
                    typeName += to_string(mp);
            }

            // Scalar.
            else
                typeName += "0d"; // Trivial scalar layout.

            // Add step-dim flag.
            if (step_posn)
                typeName += ", true";
            else
                typeName += ", false";

            // Add vec lens.
            if (folded) {
                for (auto i : vlens)
                    typeName += ", " + to_string(i);
            }

            typeName += ">";

            // Typedef.
            string typeDef = grid + "_type";
            string ptrTypeDef = grid + "_ptr_type";
            os << " typedef " << typeName << " " << typeDef << ";\n" <<
                " typedef std::shared_ptr<" << typeDef << "> " << ptrTypeDef << ";\n"
                " GridDimNames " + grid + "_dim_names;\n";

            ctorCode += "\n // Grid '" + grid + "'.\n";
            ctorCode += " " + grid + "_dim_names = {" +
                gdims.makeDimStr(", ", "\"", "\"") + "};\n";
            string initCode = " " + grid + "_ptr = std::make_shared<" + typeDef +
                ">(*this, \"" + grid + "\", " + grid + "_dim_names);\n"
                " assert(" + grid + "_ptr);\n";

            // Grid vars.
            if (gp->isScratch()) {

                // Collection of scratch grids.
                os << " GridPtrs " << grid << "_list;\n";
                ctorCode += " addScratch(" + grid + "_list);\n";
            }
            else {

                // Actual grid ptr declaration.
                os << " " << ptrTypeDef << " " << grid << "_ptr;\n" <<
                    " " << typeDef << "* " << grid << ";\n";
            }

            // Alloc-setting code.
            for (auto& dim : gp->getDims()) {
                auto& dname = dim->getName();
                auto dtype = dim->getType();

                // domain dimension.
                if (dtype == DOMAIN_INDEX) {

                    // Halos for this dimension.
                    for (bool left : { true, false }) {
                        string bstr = left ? "_left_halo_" : "_right_halo_";
                        string hvar = grid + bstr + dname;
                        int hval = _settings._haloSize > 0 ?
                            _settings._haloSize : gp->getHaloSize(dname, left);
                        os << " const idx_t " << hvar << " = " << hval <<
                            "; // default halo size in '" << dname << "' dimension.\n";
                        initCode += " " + grid + "_ptr->set" + bstr + "size(\"" + dname +
                            "\", " + hvar + ");\n";
                    }
                }

                // non-domain dimension.
                else {
                    string avar = grid + "_alloc_" + dname;
                    string ovar = grid + "_ofs_" + dname;
                    int aval = 1;
                    int oval = 0;
                    if (dtype == STEP_INDEX) {
                        aval = gp->getStepDimSize();
                        initCode += " " + grid + "_ptr->_set_dynamic_step_alloc(" +
                            (gp->is_dynamic_step_alloc() ? "true" : "false") +
                            ");\n";
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
                    initCode += " " + grid + "_ptr->_set_alloc_size(\"" + dname +
                        "\", " + avar + ");\n";
                    if (oval) {
                        os << " const idx_t " << ovar << " = " << oval <<
                            "; // first index in '" << dname << "' dimension.\n";
                        initCode += " " + grid + "_ptr->_set_local_offset(\"" + dname +
                            "\", " + ovar + ");\n";
                    }
                }
            } // dims.

            // Allow dynamic misc alloc setting if not interleaved.
            initCode += " " + grid + "_ptr->_set_dynamic_misc_alloc(" +
                (_settings._innerMisc ? "false" : "true") +
                ");\n";

            // If not scratch, init grids in ctor.
            if (!gp->isScratch()) {

                // Grid init.
                ctorCode += initCode;
                ctorCode += " " + grid + " = " + grid + "_ptr.get();\n";
                ctorCode += " addGrid(" + grid + "_ptr, ";
                if (_eqBundles.getOutputGrids().count(gp))
                    ctorCode += "true /* is an output grid */";
                else
                    ctorCode += "false /* is not an output grid */";
                ctorCode += ");\n";
            }

            // For scratch, make code for one vec element.
            else {
                scratchCode += " " + grid + "_list.clear();\n"
                    " for (int i = 0; i < num_threads; i++) {\n"
                    " " + ptrTypeDef + " " + grid + "_ptr;\n" +
                    initCode +
                    " " + grid + "_ptr->set_scratch(true);\n" +
                    " " + grid + "_list.push_back(" + grid + "_ptr);\n"
                    " }\n";
            }

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
                    typeDef + ">(*this, name, dims);\n";
            }

        } // grids.

        // Ctor.
        {
            os << "\n // Constructor.\n" <<
                " " << _context_base << "(KernelEnvPtr env, KernelSettingsPtr settings) :"
                " StencilContext(env, settings)" << ctorList <<
                " {\n  name = \"" << _stencil.getName() << "\";\n"
                " long_name = \"" << _stencil.getLongName() << "\";\n";

            os << "\n // Create grids (but do not allocate data in them).\n" <<
                ctorCode <<
                "\n // Update grids with context info.\n"
                " update_grid_info();\n";

            // end of ctor.
            os << " } // ctor" << endl;
        }

        // New-grid method.
        os << "\n // Make a new grid iff its dims match any in the stencil.\n"
            " // Returns pointer to the new grid or nullptr if no match.\n"
            " virtual YkGridPtr newStencilGrid(const std::string& name,"
            " const GridDimNames& dims) {\n"
            " YkGridPtr gp;\n" <<
            newGridCode <<
            " return gp;\n"
            " } // newStencilGrid\n";

        // Scratch-grids method.
        os << "\n // Make new scratch grids.\n"
            " virtual void makeScratchGrids(int num_threads) {\n" <<
            scratchCode <<
            " } // newScratchGrids\n";

        os << "}; // " << _context_base << endl;
    }

    // Print YASK equation bundles.
    void YASKCppPrinter::printEqBundles(ostream& os) {

        for (int ei = 0; ei < _eqBundles.getNum(); ei++) {

            // Scalar eqBundle.
            auto& eq = _eqBundles.getAll().at(ei);
            string egName = eq->getName();
            string egDesc = eq->getDescr();
            string egsName = "StencilBundle_" + egName;

            os << endl << " ////// Stencil " << egDesc << " //////\n" <<
                "\n class " << egsName << " : public StencilBundleBase {\n"
                " protected:\n"
                " typedef " << _context_base << " _context_type;\n"
                " _context_type* _context_data = 0;\n"
                " public:\n";

            // Stats for this eqBundle.
            CounterVisitor stats;
            eq->visitEqs(&stats);

            // Example computation.
            os << endl << " // " << stats.getNumOps() << " FP operation(s) per point:" << endl;
            addComment(os, *eq);

            // Stencil-bundle ctor.
            {
                os << " " << egsName << "(" << _context_base << "* context) :\n"
                    " StencilBundleBase(context),\n"
                    " _context_data(context) {\n"
                    " _name = \"" << egName << "\";\n"
                    " _scalar_fp_ops = " << stats.getNumOps() << ";\n"
                    " _scalar_points_read = " << stats.getNumReads() << ";\n"
                    " _scalar_points_written = " << stats.getNumWrites() << ";\n"
                    " _is_scratch = " << (eq->isScratch() ? "true" : "false") << ";\n";

                // I/O grids.
                os << "\n // The following grid(s) are read by " << egsName << endl;
                for (auto gp : eq->getInputGrids()) {
                    if (gp->isScratch())
                        os << "  inputScratchVecs.push_back(&_context_data->" << gp->getName() << "_list);\n";
                    else
                        os << "  inputGridPtrs.push_back(_context_data->" << gp->getName() << "_ptr);\n";
                }
                os << "\n // The following grid(s) are written by " << egsName;
                if (eq->step_expr)
                    os << " at " << eq->step_expr->makeQuotedStr();
                os << ".\n";
                for (auto gp : eq->getOutputGrids()) {
                    if (gp->isScratch())
                        os << "  outputScratchVecs.push_back(&_context_data->" << gp->getName() << "_list);\n";
                    else
                        os << "  outputGridPtrs.push_back(_context_data->" << gp->getName() << "_ptr);\n";
                }
                os << " } // Ctor." << endl;
            }

            // Domain condition.
            {
                os << "\n // Determine whether " << egsName << " is valid at the domain indices " <<
                    _dims._stencilDims.makeDimStr() << ".\n"
                    " // Return true if indices are within the valid sub-domain or false otherwise.\n"
                    " virtual bool is_in_valid_domain(const Indices& idxs) const final {\n";
                printIndices(os);
                if (eq->cond)
                    os << " return " << eq->cond->makeStr() << ";\n";
                else
                    os << " return true; // full domain.\n";
                os << " }\n";

                os << "\n // Return whether there is a sub-domain expression.\n"
                    " virtual bool is_sub_domain_expr() const {\n"
                    "  return " << (eq->cond ? "true" : "false") <<
                    ";\n }\n";

                os << "\n // Return human-readable description of sub-domain.\n"
                    " virtual std::string get_domain_description() const {\n";
                if (eq->cond)
                    os << " return \"" << eq->cond->makeStr() << "\";\n";
                else
                    os << " return \"true\"; // full domain.\n";
                os << " }\n";
            }

            // Step condition.
            {
                os << endl << " // Determine whether " << egsName <<
                    " is valid at the step input_step_index.\n" <<
                    " // Return true if valid or false otherwise.\n"
                    " virtual bool is_in_valid_step(idx_t input_step_index) const final {\n";
                if (eq->step_cond) {
                    os << " idx_t " << _dims._stepDim << " = input_step_index;\n"
                        "\n // " << eq->step_cond->makeStr() << "\n";
                    
                    // C++ scalar print assistant.
                    CounterVisitor cv;
                    eq->step_cond->accept(&cv);
                    CppPrintHelper* sp = new CppPrintHelper(_settings, _dims, &cv, "temp", "real_t", " ", ";\n");

                    // Generate the code.
                    PrintVisitorTopDown pcv(os, *sp);
                    string expr = eq->step_cond->accept(&pcv);
                    os << " return " << expr << ";\n";
                }
                else
                    os << " return true; // any step.\n";
                os << " }\n";

                os << "\n // Return whether there is a step-condition expression.\n"
                    " virtual bool is_step_cond_expr() const {\n"
                    "  return " << (eq->step_cond ? "true" : "false") <<
                    ";\n }\n";

                os << "\n // Return human-readable description of step condition.\n"
                    " virtual std::string get_step_cond_description() const {\n";
                if (eq->step_cond)
                    os << " return \"" << eq->step_cond->makeStr() << "\";\n";
                else
                    os << " return \"true\"; // any step.\n";
                os << " }\n";
            }

            // LHS step index.
            {
                os << endl;
                if (eq->step_expr)
                    os << " // Set 'output_step_index' to the step that an update"
                    " occurs when calling one of the calc_*() methods with"
                    " 'input_step_index' and return 'true'.\n";
                else
                    os << "// Return 'false' because this bundle does not update"
                        " grids with the step dimension.\n";
                os << " virtual bool get_output_step_index(idx_t input_step_index,"
                    " idx_t& output_step_index) const final {\n";
                if (eq->step_expr) {
                    os << " idx_t " << _dims._stepDim << " = input_step_index;\n"
                        " output_step_index = " << eq->step_expr->makeStr() << ";\n"
                        " return true;\n";
                }
                else
                    os << " return false;\n";
                os << " }\n";
            }
            
            // Scalar code.
            {
                // Stencil-calculation code.
                // Function header.
                os << endl << " // Calculate one scalar result relative to indices " <<
                    _dims._stencilDims.makeDimStr() << ".\n"
                    " // There are approximately " << stats.getNumOps() <<
                    " FP operation(s) per invocation.\n"
                    " virtual void calc_scalar(int region_thread_idx, const Indices& idxs) {\n";
                    printIndices(os);

                // C++ scalar print assistant.
                CounterVisitor cv;
                eq->visitEqs(&cv);
                CppPrintHelper* sp = new CppPrintHelper(_settings, _dims, &cv, "temp", "real_t", " ", ";\n");

                // Generate the code.
                PrintVisitorBottomUp pcv(os, *sp);
                eq->visitEqs(&pcv);

                // End of function.
                os << "} // calc_scalar." << endl;

                delete sp;
            }

            // Vector/Cluster code.
            for (bool do_cluster : { false, true }) {

                // Cluster eqBundle at same 'ei' index.
                // This should be the same eq-bundle because it was copied from the
                // scalar one.
                auto& vceq = do_cluster ?
                    _clusterEqBundles.getAll().at(ei) : eq;
                assert(egDesc == vceq->getDescr());

                // Create vector info for this eqBundle.
                // The visitor is accepted at all nodes in the cluster AST;
                // for each grid access node in the AST, the vectors
                // needed are determined and saved in the visitor.
                VecInfoVisitor vv(_dims);
                vceq->visitEqs(&vv);

                // Reorder some equations based on vector info.
                // TODO: make a knob to control this.
                if (false) {
                    ExprReorderVisitor erv(vv);
                    vceq->visitEqs(&erv);
                }

                // Collect stats.
                CounterVisitor cv;
                vceq->visitEqs(&cv);
                int numResults = do_cluster ?
                    _dims._clusterPts.product() :
                    _dims._fold.product();

                // Vector/cluster vars.
                string idim = _dims._innerDim;
                string vcstr = do_cluster ? "cluster" : "vector";
                string funcstr = "calc_loop_of_" + vcstr + "s";
                string nvecs = do_cluster ? "CMULT_" + allCaps(idim) : "1";
                string nelems = (do_cluster ? nvecs + " * ": "") + "VLEN_" + allCaps(idim);

                // Loop-calculation code.
                // Function header.
                string istart = "start_" + idim;
                string istop = "stop_" + idim;
                string istep = "step_" + idim;
                string iestep = "step_" + idim + "_elem";
                os << endl << " // Calculate a series of " << vcstr << "s iterating in +'" << idim <<
                    "' direction from " << _dims._stencilDims.makeDimStr() <<
                    " indices in 'idxs' to '" << istop << "'.\n";
                if (do_cluster)
                    os << " // Each cluster calculates '" << _dims._clusterPts.makeDimValStr(" * ") <<
                        "' point(s) containing " << _dims._clusterMults.product() << " '" <<
                        _dims._fold.makeDimValStr(" * ") << "' vector(s).\n";
                else
                    os << " // Each vector calculates '" << _dims._fold.makeDimValStr(" * ") <<
                        "' point(s).\n";
                os << " // Indices must be rank-relative (not global).\n"
                    " // Indices must be normalized, i.e., already divided by VLEN_*.\n"
                    " // SIMD calculations use " << vv.getNumPoints() <<
                    " vector block(s) created from " << vv.getNumAlignedVecs() <<
                    " aligned vector-block(s).\n"
                    " // There are approximately " << (stats.getNumOps() * numResults) <<
                    " FP operation(s) per iteration.\n" <<
                    " void " << funcstr << "(int region_thread_idx, int block_thread_idx,"
                    " const Indices& idxs, idx_t " << istop;
                if (!do_cluster)
                    os << ", idx_t write_mask";
                os << ") {\n";
                printIndices(os);
                os << " idx_t " << istart << " = " << idim << ";\n";
                os << " idx_t " << istep << " = " << nvecs << "; // number of vectors per iter.\n";
                os << " idx_t " << iestep << " = " << nelems << "; // number of elements per iter.\n";
                if (do_cluster)
                    os << " constexpr idx_t write_mask = idx_t(-1); // no masking for clusters.\n";

                // C++ vector print assistant.
                CppVecPrintHelper* vp = newCppVecPrintHelper(vv, cv);
                vp->printElemIndices(os);

                // Start forced-inline code.
                os << "\n // Force inlining if possible.\n"
                    "#if !defined(DEBUG) && defined(__INTEL_COMPILER)\n"
                    "#pragma forceinline recursive\n"
                    "#endif\n"
                    " {\n";

                // Print time-invariants.
                os << "\n // Invariants within a step.\n";
                CppStepVarPrintVisitor svv(os, *vp);
                vceq->visitEqs(&svv);

                // Print loop-invariants.
                os << "\n // Inner-loop invariants.\n";
                CppLoopVarPrintVisitor lvv(os, *vp);
                vceq->visitEqs(&lvv);

                // Print pointers and pre-loop prefetches.
                vp->printBasePtrs(os);

                // Actual computation loop.
                os << "\n // Inner loop.\n"
                    " for (idx_t " << idim << " = " << istart << "; " <<
                    idim << " < " << istop << "; " <<
                    idim << " += " << istep << ", " <<
                    vp->getElemIndex(idim) << " += " << iestep << ") {\n";

                // Generate loop body using vars stored in print helper.
                // Visit all expressions to cover the whole vector/cluster.
                PrintVisitorBottomUp pcv(os, *vp);
                vceq->visitEqs(&pcv);

                // Insert prefetches using vars stored in print helper for next iteration.
                vp->printPrefetches(os, true);

                // End of loop.
                os << " } // '" << idim << "' loop.\n";

                // End forced-inline code.
                os << " } // Forced-inline block.\n";

                // End of function.
                os << "} // " << funcstr << ".\n";
                delete vp;
            }

            os << "}; // " << egsName << ".\n"; // end of class.

        } // stencil eqBundles.
    }

    // Print final YASK context.
    void YASKCppPrinter::printContext(ostream& os) {

        os << endl << " ////// Overall stencil-specific context //////" << endl <<
            "struct " << _context << " : public " << _context_base << " {" << endl;

        // Stencil eqBundle objects.
        os << endl << " // Stencil equation-bundles." << endl;
        for (auto& eg : _eqBundles.getAll()) {
            string egName = eg->getName();
            os << " StencilBundle_" << egName << " " << egName << ";" << endl;
        }

        // Ctor.
        os << "\n // Constructor.\n" <<
            " " << _context << "(KernelEnvPtr env, KernelSettingsPtr settings) : " <<
            _context_base << "(env, settings)";
        for (auto& eg : _eqBundles.getAll()) {
            string egName = eg->getName();
            os << ",\n  " << egName << "(this)";
        }
        os << " {\n";

        // Push eq-bundle pointers to list.
        os << "\n // Stencil bundles.\n";
        for (auto& eg : _eqBundles.getAll()) {
            string egName = eg->getName();

            // Only want non-scratch bundles in stBundles.
            // Each scratch bundles will be added to its
            // parent bundle.
            if (!eg->isScratch())
                os << "  stBundles.push_back(&" << egName << ");\n";

            // Add scratch-bundle deps in proper order.
            auto& sdeps = _eqBundles.getScratchDeps(eg);
            for (auto& eg2 : _eqBundles.getAll()) {
                if (sdeps.count(eg2)) {
                    string eg2Name = eg2->getName();
                    os << "  " << egName <<
                        ".add_scratch_child(&" << eg2Name << ");\n";
                }
            }

        } // eq-bundles.

        // Deps.
        os << "\n // Stencil bundle inter-dependencies.\n";
        for (auto& eg : _eqBundles.getAll()) {
            string egName = eg->getName();

            // Add deps between bundles.
            for (auto& dep : _eqBundles.getDeps(eg)) {
                string depName = dep->getName();
                os << "  " << egName <<
                    ".add_dep(&" << depName << ");\n";
            }
        } // bundles.

        // Packs.
        os << "\n // Stencil bundle packs.\n";
        for (auto& bp : _eqBundlePacks.getAll()) {
            if (bp->isScratch())
                continue;
            string bpName = bp->getName();
            os << "  auto " << bpName << " = std::make_shared<BundlePack>(this, \"" <<
                bpName << "\");\n";
            for (auto& eg : bp->getBundles()) {
                if (eg->isScratch())
                    continue;
                string egName = eg->getName();
                os << "  " << bpName << "->push_back(&" << egName << ");\n";
            }
            os << "  stPacks.push_back(" << bpName << ");\n";
        }

        os << " } // Ctor.\n";

        // Dims creator.
        os << "\n  // Create Dims object.\n"
            "  static DimsPtr new_dims() {\n"
            "    auto p = std::make_shared<Dims>();\n";
        for (int i = 0; i < _dims._foldGT1.getNumDims(); i++)
            os << "    p->_vec_fold_layout.set_size(" << i << ", " <<
                _dims._foldGT1[i] << "); // '" <<
                _dims._foldGT1.getDimName(i) << "'\n";
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
        os << "    p->_step_dir = " << _dims._stepDir << ";\n";

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
