/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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
        for (auto& dim : _dims._stencilDims) {
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

        // First, create a class to hold the data (vars).
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

        os << "\n// target:\n"
            "#define YASK_TARGET \"" << _settings._target << "\"\n"
            "#define REAL_BYTES (" << _settings._elem_bytes << ")\n";

        os << "\n// Number of domain dimensions:\n"
            "#define NUM_DOMAIN_DIMS " << _dims._domainDims.size() << "\n";
        int i = 0;
        for (auto& dim : _dims._domainDims) {
            auto& dname = dim.getName();
            os << "#define DOMAIN_DIM_IDX_" << dname << " (" << (i++) << ")\n";
        }
        i = 0;
        for (auto& dim : _dims._stencilDims) {
            auto& dname = dim.getName();
            os << "#define STENCIL_DIM_IDX_" << dname << " (" << (i++) << ")\n";
        }
        int gdims = 0;
        for (auto gp : _vars) {
            int ndims = gp->get_num_dims();
            gdims = max(gdims, ndims);
        }
        auto nsdims = _dims._stencilDims.size();
        os << "\n// Number of stencil dimensions (step and domain):\n"
            "#define NUM_STENCIL_DIMS " << nsdims << endl;
        os << "\n// Max number of var dimensions:\n"
            "#define NUM_VAR_DIMS " << gdims << endl;
        os << "\n// Max of stencil and var dims:\n"
            "#define NUM_STENCIL_AND_VAR_DIMS " << max<int>(gdims, nsdims) << endl;

        os << "\n// Number of stencil equations:\n"
            "#define NUM_STENCIL_EQS " << _stencil.get_num_equations() << endl;

        // Vec/cluster lengths.
        auto nvec = _dims._foldGT1.getNumDims();
        os << "\n// One vector fold: " << _dims._fold.makeDimValStr(" * ") << endl;
        for (auto& dim : _dims._fold) {
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
        for (auto& dim : _dims._clusterMults) {
            auto& dname = dim.getName();
            string ucDim = allCaps(dname);
            os << "#define CMULT_" << ucDim << " (" <<
                dim.getVal() << ")\n";
        }
        os << "#define CMULT (" << _dims._clusterMults.product() << ")\n";

        os << "\n// Prefetch distances\n";
        for (int level : { 1, 2 }) {
            os << "#define PFD_L" << level << " (" <<
                _stencil.get_prefetch_dist(level) << ")\n";
        }
    }

    // Print YASK data class.
    void YASKCppPrinter::printData(ostream& os) {

        // get stats.
        CounterVisitor cve;
        _eqBundles.visitEqs(&cve);

        os << "\n ////// Stencil-specific data //////" << endl <<
            "class " << _context_base << " : public StencilContext {\n"
            "public:\n";

        // APIs.
        os << "\n virtual std::string get_target() const override {\n"
            "  return \"" << _settings._target << "\";\n"
            " }\n"
            "\n virtual int get_element_bytes() const override {\n"
            "  return " << _settings._elem_bytes << ";\n"
            " }\n";

        // Save data for ctor and new-var method.
        string ctorCode, ctorList, newVarCode, scratchCode;
        set<string> newVarDims;

        // Vars.
        os << "\n ///// Var(s)." << endl;
        for (auto gp : _vars) {
            string var = gp->getName();
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
                os << ndims << "-D var";
            else
                os << "scalar value";
            os << " '" << var << "', which is ";
            if (gp->isScratch())
                os << " a scratch variable.\n";
            else if (_eqBundles.getOutputVars().count(gp))
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
            string gtype = folded ? "YkVecVar" : "YkElemVar";

            // Type-name in kernel is 'VAR_TYPE<LAYOUT, WRAP_1ST_IDX, VEC_LENGTHS...>'.
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
                            THROW_YASK_EXCEPTION("Error: cannot create var '" + var +
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
            string typeDef = var + "_type";
            string ptrTypeDef = var + "_ptr_type";
            os << " typedef " << typeName << " " << typeDef << ";\n" <<
                " typedef std::shared_ptr<" << typeDef << "> " << ptrTypeDef << ";\n"
                " VarDimNames " + var + "_dim_names;\n";

            ctorCode += "\n // Var '" + var + "'.\n";
            ctorCode += " " + var + "_dim_names = {" +
                gdims.makeDimStr(", ", "\"", "\"") + "};\n";
            string gbp = var + "_base_ptr";
            string initCode = " " + var + "_ptr_type " + gbp + " = std::make_shared<" + typeDef +
                ">(*this, \"" + var + "\", " + var + "_dim_names);\n"
                " assert(" + gbp + ");\n"
                " " + var + "_ptr = std::make_shared<YkVarImpl>(" + gbp + ");\n"
                " assert(" + var + "_ptr->gbp());\n";

            // Vars.
            if (gp->isScratch()) {

                // Collection of scratch vars.
                os << " VarPtrs " << var << "_list;\n";
                ctorCode += " addScratch(" + var + "_list);\n";
            }
            else {

                // Var ptr declaration.
                // Default ctor gives null ptr.
                os << " YkVarPtr " << var << "_ptr;\n";
            }

            // Alloc-setting code.
            bool gotDomain = false;
            for (auto& dim : gp->getDims()) {
                auto& dname = dim->getName();
                auto dtype = dim->getType();

                // domain dimension.
                if (dtype == DOMAIN_INDEX) {
                    gotDomain = true;

                    // Halos for this dimension.
                    for (bool left : { true, false }) {
                        string bstr = left ? "_left_halo_" : "_right_halo_";
                        string hvar = var + bstr + dname;
                        int hval = _settings._haloSize > 0 ?
                            _settings._haloSize : gp->getHaloSize(dname, left);
                        os << " const idx_t " << hvar << " = " << hval <<
                            "; // default halo size in '" << dname << "' dimension.\n";
                        initCode += " " + var + "_ptr->set" + bstr + "size(\"" + dname +
                            "\", " + hvar + ");\n";
                    }
                }

                // non-domain dimension.
                else {
                    string avar = var + "_alloc_" + dname;
                    string ovar = var + "_ofs_" + dname;
                    int aval = 1;
                    int oval = 0;
                    if (dtype == STEP_INDEX) {
                        aval = gp->getStepDimSize();
                        initCode += " " + var + "_base_ptr->_set_dynamic_step_alloc(" +
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
                    initCode += " " + var + "_ptr->_set_alloc_size(\"" + dname +
                        "\", " + avar + ");\n";
                    if (oval) {
                        os << " const idx_t " << ovar << " = " << oval <<
                            "; // first index in '" << dname << "' dimension.\n";
                        initCode += " " + var + "_ptr->_set_local_offset(\"" + dname +
                            "\", " + ovar + ");\n";
                    }
                }
            } // dims.

            // L1 dist.
            if (gotDomain) {
                auto l1var = var + "_l1_norm";
                os << " const int " << l1var << " = " << gp->getL1Dist() <<
                    "; // Max L1-norm of MPI neighbor for halo exchanges.\n";
                initCode += " " + var + "_ptr->set_halo_exchange_l1_norm(" +
                    l1var + ");\n";
            }
            
            // Allow dynamic misc alloc setting if not interleaved.
            initCode += " " + var + "_base_ptr->_set_dynamic_misc_alloc(" +
                (_settings._innerMisc ? "false" : "true") +
                ");\n";

            // If not scratch, init vars in ctor.
            if (!gp->isScratch()) {

                // Var init.
                ctorCode += initCode;
                ctorCode += " addVar(" + var + "_ptr, true, ";
                if (_eqBundles.getOutputVars().count(gp))
                    ctorCode += "true /* is an output var */";
                else
                    ctorCode += "false /* is not an output var */";
                ctorCode += ");\n";
            }

            // For scratch, make code for one vec element.
            else {
                scratchCode += " " + var + "_list.clear();\n"
                    " for (int i = 0; i < num_threads; i++) {\n"
                    " YkVarPtr " + var + "_ptr;\n" +
                    initCode +
                    " " + var + "_base_ptr->set_scratch(true);\n" +
                    " " + var + "_list.push_back(" + var + "_ptr);\n"
                    " }\n";
            }

            // Make new vars via API.
            string newVarKey = gdims.makeDimStr();
            if (!newVarDims.count(newVarKey)) {
                newVarDims.insert(newVarKey);
                bool firstVar = newVarCode.length() == 0;
                if (gdims.getNumDims())
                    newVarCode += "\n // Vars with '" + newVarKey + "' dim(s).\n";
                else
                    newVarCode += "\n // Scalar vars.\n";
                if (!firstVar)
                    newVarCode += " else";
                newVarCode += " if (dims == " + var + "_dim_names)\n"
                    " gp = std::make_shared<" + typeDef + ">(*this, name, dims);\n";
            }

        } // vars.

        // Ctor.
        {
            os << "\n // Constructor.\n" <<
                " " << _context_base << "(KernelEnvPtr env, KernelSettingsPtr settings) :"
                " StencilContext(env, settings)" << ctorList <<
                " {\n  name = \"" << _stencil.getName() << "\";\n"
                " long_name = \"" << _stencil.getLongName() << "\";\n";

            os << "\n // Create vars (but do not allocate data in them).\n" <<
                ctorCode <<
                "\n // Update vars with context info.\n"
                " update_var_info(false);\n";

            // end of ctor.
            os << " } // ctor" << endl;
        }

        // New-var method.
        os << "\n // Make a new var iff its dims match any in the stencil.\n"
            " // Returns pointer to the new var or nullptr if no match.\n"
            " virtual VarBasePtr newStencilVar(const std::string& name,"
            " const VarDimNames& dims) override {\n"
            " VarBasePtr gp;\n" <<
            newVarCode <<
            " return gp;\n"
            " } // newStencilVar\n";

        // Scratch-vars method.
        os << "\n // Make new scratch vars.\n"
            " virtual void makeScratchVars(int num_threads) override {\n" <<
            scratchCode <<
            " } // newScratchVars\n";

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

                // I/O vars.
                os << "\n // The following var(s) are read by " << egsName << endl;
                for (auto gp : eq->getInputVars()) {
                    if (gp->isScratch())
                        os << "  inputScratchVecs.push_back(&_context_data->" << gp->getName() << "_list);\n";
                    else
                        os << "  inputVarPtrs.push_back(_context_data->" << gp->getName() << "_ptr);\n";
                }
                os << "\n // The following var(s) are written by " << egsName;
                if (eq->step_expr)
                    os << " at " << eq->step_expr->makeQuotedStr();
                os << ".\n";
                for (auto gp : eq->getOutputVars()) {
                    if (gp->isScratch())
                        os << "  outputScratchVecs.push_back(&_context_data->" << gp->getName() << "_list);\n";
                    else
                        os << "  outputVarPtrs.push_back(_context_data->" << gp->getName() << "_ptr);\n";
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
                        " vars with the step dimension.\n";
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
                // for each var access node in the AST, the vectors
                // needed are determined and saved in the visitor.
                VecInfoVisitor vv(_dims);
                vceq->visitEqs(&vv);

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
 
                // C++ vector print assistant.
                CppVecPrintHelper* vp = newCppVecPrintHelper(vv, cv);
                vp->setUseMaskedWrites(!do_cluster);
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
                os << "\n // Inner loop.\n";
                if (_dims._fold.product() == 1)
                    os << " // Specifying SIMD here because there is no explicit vectorization.\n"
                        "#pragma omp simd\n";
                os << " for (idx_t " << idim << " = " << istart << "; " <<
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

        os << "\n ////// User-provided code //////" << endl <<
            "struct " << _context_hook << " {\n"
            " static void call_after_new_solution(yk_solution& kernel_soln) {\n"
            "  // Code provided by user.\n";
        for (auto& code : _stencil.getKernelCode())
            os << "  " << code << "\n";
        os << " }\n"
            "};\n";
        
        os << "\n ////// Overall stencil-specific context //////" << endl <<
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

        // Stages.
        os << "\n // Stencil stages.\n";
        for (auto& bp : _eqStages.getAll()) {
            if (bp->isScratch())
                continue;
            string bpName = bp->getName();
            os << "  auto " << bpName << " = std::make_shared<Stage>(this, \"" <<
                bpName << "\");\n";
            for (auto& eg : bp->getBundles()) {
                if (eg->isScratch())
                    continue;
                string egName = eg->getName();
                os << "  " << bpName << "->push_back(&" << egName << ");\n";
            }
            os << "  stStages.push_back(" << bpName << ");\n";
        }

        os << "\n // Call code provided by user.\n" <<
            _context_hook << "::call_after_new_solution(*this);\n";

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
        for (auto& dim : _dims._domainDims) {
            auto& dname = dim.getName();
            os << "    p->_domain_dims.addDimBack(\"" << dname << "\", 0);\n";
        }
        for (auto& dim : _dims._stencilDims) {
            auto& dname = dim.getName();
            os << "    p->_stencil_dims.addDimBack(\"" << dname << "\", 0);\n";
        }
        for (auto& dim : _dims._miscDims) {
            auto& dname = dim.getName();
            os << "    p->_misc_dims.addDimBack(\"" << dname << "\", 0);\n";
        }
        for (auto& dim : _dims._fold) {
            auto& dname = dim.getName();
            auto& dval = dim.getVal();
            os << "    p->_fold_pts.addDimBack(\"" << dname << "\", " << dval << ");\n";
        }
        for (auto& dim : _dims._foldGT1) {
            auto& dname = dim.getName();
            auto& dval = dim.getVal();
            os << "    p->_vec_fold_pts.addDimBack(\"" << dname << "\", " << dval << ");\n";
        }
        string ffi = (_dims._fold.isFirstInner()) ? "true" : "false";
        os << "    p->_fold_pts.setFirstInner(" << ffi << ");\n"
            "    p->_vec_fold_pts.setFirstInner(" << ffi << ");\n";
        for (auto& dim : _dims._clusterPts) {
            auto& dname = dim.getName();
            auto& dval = dim.getVal();
            os << "    p->_cluster_pts.addDimBack(\"" << dname << "\", " << dval << ");\n";
        }
        for (auto& dim : _dims._clusterMults) {
            auto& dname = dim.getName();
            auto& dval = dim.getVal();
            os << "    p->_cluster_mults.addDimBack(\"" << dname << "\", " << dval << ");\n";
        }
        os << "    p->_step_dir = " << _dims._stepDir << ";\n";

        os << "    return p;\n"
            "  }\n";

        os << "}; // " << _context << endl;
    }

} // namespace yask.
