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
    void YASKCppPrinter::print_indices(ostream& os) const {
        os << "\n // Extract index for each dim.\n";
        int i = 0;
        for (auto& dim : _dims._stencil_dims) {
            auto& dname = dim._get_name();
            os << " idx_t " << dname << " = idxs[" << i << "];\n";
            i++;
        }
    }

    // Print an expression as a one-line C++ comment.
    void YASKCppPrinter::add_comment(ostream& os, EqBundle& eq) {

        // Use a simple human-readable visitor to create a comment.
        PrintHelper ph(_settings, _dims, NULL, "temp", "", " // ", ".\n");
        PrintVisitorTopDown commenter(os, ph);
        eq.visit_eqs(&commenter);
    }

    // Print YASK code in new stencil context class.
    void YASKCppPrinter::print(ostream& os) {

        string sname = _stencil._get_name();
        os << "// Automatically-generated code; do not edit.\n"
            "\n////// YASK implementation of the '" << sname <<
            "' stencil //////\n";

        // Macros.
        os << "\n#ifdef DEFINE_MACROS\n";
        print_macros(os);
        os << "\n#endif // DEFINE_MACROS\n";

        // Stencil-context code.
        os << "\n#ifdef DEFINE_CONTEXT\n"
            "namespace yask {" << endl;

        // First, create a class to hold the data (vars).
        print_data(os);

        // A struct for each equation bundle.
        print_eq_bundles(os);

        // Finish the context.
        print_context(os);

        os << "} // namespace yask.\n"
            "#endif // DEFINE_CONTEXT\n"
            "\n//End of automatically-generated code." << endl;
    }

    // Print YASK macros.  TODO: get rid of all or most of the macros
    // in favor of consts or templates.
    void YASKCppPrinter::print_macros(ostream& os) {

        string sname = _stencil._get_name();
        os << "// Stencil solution:\n"
            "#define YASK_STENCIL_NAME \"" << sname << "\"\n"
            "#define YASK_STENCIL_CONTEXT " << _context << endl;

        os << "\n// target:\n"
            "#define YASK_TARGET \"" << _settings._target << "\"\n"
            "#define REAL_BYTES (" << _settings._elem_bytes << ")\n";

        os << "\n// Number of domain dimensions:\n"
            "#define NUM_DOMAIN_DIMS " << _dims._domain_dims.size() << "\n";
        int i = 0;
        for (auto& dim : _dims._domain_dims) {
            auto& dname = dim._get_name();
            os << "#define DOMAIN_DIM_IDX_" << dname << " (" << (i++) << ")\n";
        }
        i = 0;
        for (auto& dim : _dims._stencil_dims) {
            auto& dname = dim._get_name();
            os << "#define STENCIL_DIM_IDX_" << dname << " (" << (i++) << ")\n";
        }
        int gdims = 0;
        for (auto gp : _vars) {
            int ndims = gp->get_num_dims();
            gdims = max(gdims, ndims);
        }
        auto nsdims = _dims._stencil_dims.size();
        os << "\n// Number of stencil dimensions (step and domain):\n"
            "#define NUM_STENCIL_DIMS " << nsdims << endl;
        os << "\n// Max number of var dimensions:\n"
            "#define NUM_VAR_DIMS " << gdims << endl;
        os << "\n// Max of stencil and var dims:\n"
            "#define NUM_STENCIL_AND_VAR_DIMS " << max<int>(gdims, nsdims) << endl;

        os << "\n// Number of stencil equations:\n"
            "#define NUM_STENCIL_EQS " << _stencil.get_num_equations() << endl;

        // Vec/cluster lengths.
        auto nvec = _dims._fold_gt1._get_num_dims();
        os << "\n// One vector fold: " << _dims._fold.make_dim_val_str(" * ") << endl;
        for (auto& dim : _dims._fold) {
            auto& dname = dim._get_name();
            string uc_dim = all_caps(dname);
            os << "#define VLEN_" << uc_dim << " (" << dim.get_val() << ")" << endl;
        }
        os << "namespace yask {\n"
            " constexpr idx_t fold_pts[]{ " << _dims._fold.make_val_str() << " };\n"
            "}\n";
        os << "#define VLEN (" << _dims._fold.product() << ")" << endl;
        os << "#define FIRST_FOLD_INDEX_IS_UNIT_STRIDE (" <<
            (_dims._fold.is_first_inner() ? 1 : 0) << ")" << endl;
        os << "#define NUM_VEC_FOLD_DIMS (" << nvec << ")" << endl;

        // Layout for folding.
        // This contains only the vectorized (len > 1) dims.
        string layout;
        if (_dims._fold_gt1.is_first_inner())
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
                os << ", " << _dims._fold_gt1[i]; // fold lengths.
            os << ")\n";
        } else
            os << "(0)\n";

        os << "#define USING_UNALIGNED_LOADS (" <<
            (_settings._allow_unaligned_loads ? 1 : 0) << ")" << endl;

        os << endl;
        os << "// Cluster multipliers of vector folds: " <<
            _dims._cluster_mults.make_dim_val_str(" * ") << endl;
        for (auto& dim : _dims._cluster_mults) {
            auto& dname = dim._get_name();
            string uc_dim = all_caps(dname);
            os << "#define CMULT_" << uc_dim << " (" <<
                dim.get_val() << ")\n";
        }
        os << "#define CMULT (" << _dims._cluster_mults.product() << ")\n";

        os << "\n// Prefetch distances\n";
        for (int level : { 1, 2 }) {
            os << "#define PFD_L" << level << " (" <<
                _stencil.get_prefetch_dist(level) << ")\n";
        }
    }

    #define VAR_DECLS(gp) \
        int ndims = gp->get_num_dims();         \
        auto gdims = gp->get_dims_tuple();      \
        string var = gp->_get_name();           \
        string vprefix = "var_" + var;          \
        string base_t = vprefix + "_base_t";        \
        string ptr_t = vprefix + "_base_ptr_t";     \
        string core_t = vprefix + "_core_t";        \
        string base_ptr = vprefix + "_base_p";      \
        string core_ptr = vprefix + "_core_p";      \
        string var_ptr = vprefix + "_p";            \
        string var_list = vprefix + "_list";        \
        string var_dim_names = vprefix + "_dim_names"
    
    // Print YASK var types and core-data class.
    void YASKCppPrinter::print_data(ostream& os) {

        // Var types.
        os << "\n ///// Stencil var type(s)." << endl;
        for (auto gp : _vars) {
            VAR_DECLS(gp);

            os << "\n // The ";
            if (ndims)
                os << ndims << "-D var";
            else
                os << "scalar value";
            os << " '" << var << "', which is ";
            if (gp->is_scratch())
                os << " a scratch variable.\n";
            else if (_eq_bundles.get_output_vars().count(gp))
                os << "updated by one or more equations.\n";
            else
                os << "not updated by any equation (read-only).\n";
            if (ndims) {
                os << " // Dimensions: ";
                for (int dn = 0; dn < ndims; dn++) {
                    if (dn) os << ", ";
                    auto& dim = gp->get_dims()[dn];
                    auto& dname = dim->_get_name();
                    os << "'" << dname << "'(#" << (dn+1) << ")";
                }
                os << ".\n";
            }

            // Use vector-folded layout if possible.
            bool folded = gp->is_foldable();
            string vtype = folded ? "YkVecVar" : "YkElemVar";
            string ctype = vtype + "Core";

            // Create the template params.
            // Type-name in kernel is 'VAR_TYPE<LAYOUT, WRAP_1ST_IDX, VEC_LENGTHS...>'.
            {
                string templ = "<Layout_";
                int step_posn = 0;
                int inner_posn = 0;
                vector<int> vlens;
                vector<int> misc_posns;

                // 1-D or more.
                if (ndims) {
                    for (int dn = 0; dn < ndims; dn++) {
                        auto& dim = gp->get_dims()[dn];
                        auto& dname = dim->_get_name();
                        auto dtype = dim->get_type();
                        bool defer = false; // add dim later.

                        // Step dim?
                        // If this exists, it will get placed near to the end,
                        // just before the inner & misc dims.
                        if (dtype == STEP_INDEX) {
                            assert(dname == _dims._step_dim);
                            if (dn > 0) {
                                THROW_YASK_EXCEPTION("Error: cannot create var '" + var +
                                                     "' with dimensions '" + gdims.make_dim_str() +
                                                     "' because '" + dname + "' must be first dimension");
                            }
                            if (folded) {
                                step_posn = dn + 1;
                                defer = true;
                            }
                        }

                        // Inner domain dim?
                        // If this exists, it will get placed at or near the end.
                        else if (dname == _dims._inner_dim) {
                            assert(dtype == DOMAIN_INDEX);
                            if (folded) {
                                inner_posn = dn + 1;
                                defer = true;
                            }
                        }

                        // Misc dims? Placed after the inner domain dim if requested.
                        else if (dtype == MISC_INDEX) {
                            if (folded && _settings._inner_misc) {
                                misc_posns.push_back(dn + 1);
                                defer = true;
                            }
                        }

                        // Add index position to layout.
                        if (!defer) {
                            int other_posn = dn + 1;
                            templ += to_string(other_posn);
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
                        templ += to_string(step_posn);
                    if (inner_posn)
                        templ += to_string(inner_posn);
                    for (auto mp : misc_posns)
                        templ += to_string(mp);
                }

                // Scalar.
                else
                    templ += "0d"; // Trivial scalar layout.

                // Add step-dim flag.
                if (step_posn)
                    templ += ", true";
                else
                    templ += ", false";

                // Add vec lens.
                if (folded) {
                    for (auto i : vlens)
                        templ += ", " + to_string(i);
                }
                templ += ">";

                // Add templates to types.
                vtype += templ;
                ctype += templ;
            }

            // Typedefs.
            os << " typedef " << vtype << " " << base_t << ";\n" <<
                " typedef std::shared_ptr<" << base_t << "> " << ptr_t << ";\n" <<
                " typedef " << ctype << " " << core_t << ";\n";
        } // vars.

        // Type with ptrs to core data.
        {
        os << "\n // Data needed in kernel(s).\n"
            " // Will create one for each region thread.\n"
            "struct " << _core_t << " {\n";

        os << "\n // Copy of context info.\n"
            " Indices rank_domain_offsets;\n";
        
        os << "\n // Pointer(s) to var core data.\n";
        for (auto gp : _vars) {
            VAR_DECLS(gp);
            os << " " << core_t << "* " << core_ptr << ";\n";
        }
        os << "}; // " << _core_t << endl;
        }
    }
        
    // Print YASK equation bundles.
    void YASKCppPrinter::print_eq_bundles(ostream& os) {

        for (int ei = 0; ei < _eq_bundles.get_num(); ei++) {

            // Scalar eq_bundle.
            auto& eq = _eq_bundles.get_all().at(ei);
            string eg_name = eq->_get_name();
            string eg_desc = eq->get_descr();
            string egs_name = "StencilBundle_" + eg_name;

            os << endl << " ////// Stencil " << eg_desc << " //////\n" <<
                "\n struct " << egs_name << " {\n"
                "  typedef " << _core_t << " _core_t;\n"
                "  _core_t* _core_list = 0;\n"
                "  std::string _name;\n"
                "  int _scalar_fp_ops;\n"
                "  int _scalar_points_read;\n"
                "  int _scalar_points_written;\n"
                "  bool _is_scratch;\n";

            // Stats for this eq_bundle.
            CounterVisitor stats;
            eq->visit_eqs(&stats);

            // Example computation.
            os << endl << " // " << stats.get_num_ops() << " FP operation(s) per point:" << endl;
            add_comment(os, *eq);

            // Stencil-bundle ctor.
            {
                os << " " << egs_name << "() {\n"
                    " _name = \"" << eg_name << "\";\n"
                    " _scalar_fp_ops = " << stats.get_num_ops() << ";\n"
                    " _scalar_points_read = " << stats.get_num_reads() << ";\n"
                    " _scalar_points_written = " << stats.get_num_writes() << ";\n"
                    " _is_scratch = " << (eq->is_scratch() ? "true" : "false") << ";\n";

                os << " } // Ctor." << endl;
            }

            // Domain condition.
            {
                os << "\n // Determine whether " << egs_name << " is valid at the domain indices " <<
                    _dims._stencil_dims.make_dim_str() << ".\n"
                    " // Return true if indices are within the valid sub-domain or false otherwise.\n"
                    " ALWAYS_INLINE bool is_in_valid_domain(const Indices& idxs) const {\n";
                print_indices(os);
                if (eq->cond)
                    os << " return " << eq->cond->make_str() << ";\n";
                else
                    os << " return true; // full domain.\n";
                os << " }\n";

                os << "\n // Return whether there is a sub-domain expression.\n"
                    " ALWAYS_INLINE bool is_sub_domain_expr() const {\n"
                    "  return " << (eq->cond ? "true" : "false") <<
                    ";\n }\n";

                os << "\n // Return human-readable description of sub-domain.\n"
                    " inline std::string get_domain_description() const {\n";
                if (eq->cond)
                    os << " return \"" << eq->cond->make_str() << "\";\n";
                else
                    os << " return \"true\"; // full domain.\n";
                os << " }\n";
            }

            // Step condition.
            {
                os << endl << " // Determine whether " << egs_name <<
                    " is valid at the step input_step_index.\n" <<
                    " // Return true if valid or false otherwise.\n"
                    " ALWAYS_INLINE bool is_in_valid_step(idx_t input_step_index) const {\n";
                if (eq->step_cond) {
                    os << " idx_t " << _dims._step_dim << " = input_step_index;\n"
                        "\n // " << eq->step_cond->make_str() << "\n";
                    
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
                    " ALWAYS_INLINE bool is_step_cond_expr() const {\n"
                    "  return " << (eq->step_cond ? "true" : "false") <<
                    ";\n }\n";

                os << "\n // Return human-readable description of step condition.\n"
                    " inline std::string get_step_cond_description() const {\n";
                if (eq->step_cond)
                    os << " return \"" << eq->step_cond->make_str() << "\";\n";
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
                os << " ALWAYS_INLINE bool get_output_step_index(idx_t input_step_index,"
                    " idx_t& output_step_index) const {\n";
                if (eq->step_expr) {
                    os << " idx_t " << _dims._step_dim << " = input_step_index;\n"
                        " output_step_index = " << eq->step_expr->make_str() << ";\n"
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
                    _dims._stencil_dims.make_dim_str() << ".\n"
                    " // There are approximately " << stats.get_num_ops() <<
                    " FP operation(s) per invocation.\n"
                    " ALWAYS_INLINE void calc_scalar(int core_idx, const Indices& idxs) {\n"
                    " assert(_core_list);\n"
                    " auto& core_data = _core_list[core_idx];\n";
                print_indices(os);

                // C++ scalar print assistant.
                CounterVisitor cv;
                eq->visit_eqs(&cv);
                CppPrintHelper* sp = new CppPrintHelper(_settings, _dims, &cv, "temp", "real_t", " ", ";\n");

                // Generate the code.
                PrintVisitorBottomUp pcv(os, *sp);
                eq->visit_eqs(&pcv);

                // End of function.
                os << "} // calc_scalar." << endl;

                delete sp;
            }

            // Vector/Cluster code.
            for (bool do_cluster : { false, true }) {

                // Cluster eq_bundle at same 'ei' index.
                // This should be the same eq-bundle because it was copied from the
                // scalar one.
                auto& vceq = do_cluster ?
                    _cluster_eq_bundles.get_all().at(ei) : eq;
                assert(eg_desc == vceq->get_descr());

                // Create vector info for this eq_bundle.
                // The visitor is accepted at all nodes in the cluster AST;
                // for each var access node in the AST, the vectors
                // needed are determined and saved in the visitor.
                VecInfoVisitor vv(_dims);
                vceq->visit_eqs(&vv);

                // Collect stats.
                CounterVisitor cv;
                vceq->visit_eqs(&cv);
                int num_results = do_cluster ?
                    _dims._cluster_pts.product() :
                    _dims._fold.product();

                // Vector/cluster vars.
                string idim = _dims._inner_dim;
                string vcstr = do_cluster ? "cluster" : "vector";
                string funcstr = "calc_loop_of_" + vcstr + "s";
                string nvecs = do_cluster ? "CMULT_" + all_caps(idim) : "1";
                string nelems = (do_cluster ? nvecs + " * ": "") + "VLEN_" + all_caps(idim);

                // Loop-calculation code.
                // Function header.
                string istart = "start_" + idim;
                string istop = "stop_" + idim;
                string istep = "step_" + idim;
                string iestep = "step_" + idim + "_elem";
                os << endl << " // Calculate a series of " << vcstr << "s iterating in +'" << idim <<
                    "' direction from " << _dims._stencil_dims.make_dim_str() <<
                    " indices in 'idxs' to '" << istop << "'.\n";
                if (do_cluster)
                    os << " // Each cluster calculates '" << _dims._cluster_pts.make_dim_val_str(" * ") <<
                        "' point(s) containing " << _dims._cluster_mults.product() << " '" <<
                        _dims._fold.make_dim_val_str(" * ") << "' vector(s).\n";
                else
                    os << " // Each vector calculates '" << _dims._fold.make_dim_val_str(" * ") <<
                        "' point(s).\n";
                os << " // Indices must be rank-relative (not global).\n"
                    " // Indices must be normalized, i.e., already divided by VLEN_*.\n"
                    " // SIMD calculations use " << vv.get_num_points() <<
                    " vector block(s) created from " << vv.get_num_aligned_vecs() <<
                    " aligned vector-block(s).\n"
                    " // There are approximately " << (stats.get_num_ops() * num_results) <<
                    " FP operation(s) per iteration.\n" <<
                    " ALWAYS_INLINE void " << funcstr << "(int core_idx, int block_thread_idx,"
                    " const Indices& idxs, idx_t " << istop;
                if (!do_cluster)
                    os << ", idx_t write_mask";
                os << ") {\n"
                    " assert(_core_list);\n"
                    " auto& core_data = _core_list[core_idx];\n";
                print_indices(os);
                os << " idx_t " << istart << " = " << idim << ";\n";
                os << " idx_t " << istep << " = " << nvecs << "; // number of vectors per iter.\n";
                os << " idx_t " << iestep << " = " << nelems << "; // number of elements per iter.\n";
 
                // C++ vector print assistant.
                CppVecPrintHelper* vp = new_cpp_vec_print_helper(vv, cv);
                vp->set_use_masked_writes(!do_cluster);
                vp->print_elem_indices(os);

                // Start forced-inline code.
                os << "\n // Force inlining if possible.\n"
                    "#if !defined(DEBUG) && defined(__INTEL_COMPILER)\n"
                    "#pragma forceinline recursive\n"
                    "#endif\n"
                    " {\n";

                // Print time-invariants.
                os << "\n // Invariants within a step.\n";
                CppStepVarPrintVisitor svv(os, *vp);
                vceq->visit_eqs(&svv);

                // Print loop-invariants.
                os << "\n // Inner-loop invariants.\n";
                CppLoopVarPrintVisitor lvv(os, *vp);
                vceq->visit_eqs(&lvv);

                // Print pointers and pre-loop prefetches.
                vp->print_base_ptrs(os);

                // Actual computation loop.
                os << "\n // Inner loop.\n";
                if (_dims._fold.product() == 1)
                    os << " // Specifying SIMD here because there is no explicit vectorization.\n"
                        "#pragma omp simd\n";
                os << " for (idx_t " << idim << " = " << istart << "; " <<
                    idim << " < " << istop << "; " <<
                    idim << " += " << istep << ", " <<
                    vp->get_elem_index(idim) << " += " << iestep << ") {\n";

                // Generate loop body using vars stored in print helper.
                // Visit all expressions to cover the whole vector/cluster.
                PrintVisitorBottomUp pcv(os, *vp);
                vceq->visit_eqs(&pcv);

                // Insert prefetches using vars stored in print helper for next iteration.
                vp->print_prefetches(os, true);

                // End of loop.
                os << " } // '" << idim << "' loop.\n";

                // End forced-inline code.
                os << " } // Forced-inline block.\n";

                // End of function.
                os << "} // " << funcstr << ".\n";
                delete vp;
            }

            os << "}; // " << egs_name << ".\n"; // end of struct.

        } // stencil eq_bundles.
    }

    // Print derived YASK context.
    void YASKCppPrinter::print_context(ostream& os) {

        os << "\n ////// User-provided code //////" << endl <<
            "struct " << _context_hook << " {\n"
            " static void call_after_new_solution(yk_solution& kernel_soln) {\n"
            "  // Code provided by user.\n";
        for (auto& code : _stencil.get_kernel_code())
            os << "  " << code << "\n";
        os << " }\n"
            "};\n";
        
        os << "\n ////// Overall stencil-specific context //////" << endl <<
            "class " << _context << " : public StencilContext {\n"
            " protected:\n";

        // Save code to be added later.
        string ctor_code, new_var_code, scratch_code, core_code;
        set<string> new_var_dims;

        // Vars.
        os << "\n ///// Var(s)." << endl;
        for (auto gp : _vars) {
            VAR_DECLS(gp);

            string header = "\n // Var '" + var + "'.\n";
            os << header;
            ctor_code += header;

            os << " VarDimNames " << var_dim_names << ";\n";
            ctor_code += " " + var_dim_names + " = {" +
                gdims.make_dim_str(", ", "\"", "\"") + "};\n";

            // Code to create a local base ptr and set pre-defined generic ptr.
            string init_code =
                " " + ptr_t + " " + base_ptr + " = std::make_shared<" + base_t + ">"
                "(*this, \"" + var + "\", " + var_dim_names + ");\n"
                " assert(" + base_ptr + ");\n"
                " " + var_ptr + " = std::make_shared<YkVarImpl>(" + base_ptr + ");\n"
                " assert(" + var_ptr + "->gbp());\n";

            if (!gp->is_scratch()) {

                // Var ptr declaration.
                // Default ctor gives null ptr.
                os << " YkVarPtr " << var_ptr << ";\n";
            }
            else {

                // List of scratch vars, one for each thread.
                os << " VarPtrs " << var_list << ";\n";
                ctor_code += " add_scratch(" + var_list + ");\n";
            }

            // Alloc-setting code.
            bool got_domain = false;
            for (auto& dim : gp->get_dims()) {
                auto& dname = dim->_get_name();
                auto dtype = dim->get_type();

                // domain dimension.
                if (dtype == DOMAIN_INDEX) {
                    got_domain = true;

                    // Halos for this dimension.
                    for (bool left : { true, false }) {
                        string bstr = left ? "_left_halo_" : "_right_halo_";
                        string hvar = var + bstr + dname;
                        int hval = _settings._halo_size > 0 ?
                            _settings._halo_size : gp->get_halo_size(dname, left);
                        os << " const idx_t " << hvar << " = " << hval <<
                            "; // default halo size in '" << dname << "' dimension.\n";
                        init_code += " " + var_ptr + "->set" + bstr + "size(\"" + dname +
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
                        aval = gp->get_step_dim_size();
                        init_code += " " + base_ptr + "->_set_dynamic_step_alloc(" +
                            (gp->is_dynamic_step_alloc() ? "true" : "false") +
                            ");\n";
                    } else {
                        auto* minp = gp->get_min_indices().lookup(dname);
                        auto* maxp = gp->get_max_indices().lookup(dname);
                        if (minp && maxp) {
                            aval = *maxp - *minp + 1;
                            oval = *minp;
                        }
                    }
                    os << " const idx_t " << avar << " = " << aval <<
                        "; // default allocation in '" << dname << "' dimension.\n";
                    init_code += " " + var_ptr + "->_set_alloc_size(\"" + dname +
                        "\", " + avar + ");\n";
                    if (oval) {
                        os << " const idx_t " << ovar << " = " << oval <<
                            "; // first index in '" << dname << "' dimension.\n";
                        init_code += " " + var_ptr + "->_set_local_offset(\"" + dname +
                            "\", " + ovar + ");\n";
                    }
                }
            } // dims.

            // L1 dist.
            if (got_domain) {
                auto l1var = var + "_l1_norm";
                os << " const int " << l1var << " = " << gp->get_l1_dist() <<
                    "; // Max L1-norm of MPI neighbor for halo exchanges.\n";
                init_code += " " + var_ptr + "->set_halo_exchange_l1_norm(" +
                    l1var + ");\n";
            }
            
            // Allow dynamic misc alloc setting if not interleaved.
            init_code += " " + base_ptr + "->_set_dynamic_misc_alloc(" +
                (_settings._inner_misc ? "false" : "true") +
                ");\n";

            // If not scratch, init vars in ctor.
            if (!gp->is_scratch()) {

                // Var init.
                ctor_code += init_code;
                ctor_code += " add_var(" + var_ptr + ", true, ";
                if (_eq_bundles.get_output_vars().count(gp))
                    ctor_code += "true /* is an output var */";
                else
                    ctor_code += "false /* is not an output var */";
                ctor_code += ");\n";

                // Core init.
                core_code += " _core_list[i]." + core_ptr + " = static_cast<" + core_t + "*>"
                    "(" + var_ptr + "->corep());\n";
            }

            // For scratch, make code to fill vector.
            else {
                scratch_code +=
                    " " + var_list + ".clear();\n"
                    " for (int i = 0; i < num_threads; i++) {\n"
                    " YkVarPtr " + var_ptr + ";\n" +
                    init_code +
                    " " + base_ptr + "->set_scratch(true);\n" +
                    " " + var_list + ".push_back(" + var_ptr + ");\n"
                    " }\n";

                // Core init.
                core_code += " _core_list[i]." + core_ptr + " = static_cast<" + core_t + "*>"
                    "(" + var_list + ".at(i)->get_corep());\n";
            }

            // Make new vars via API.
            string new_var_key = gdims.make_dim_str();
            if (!new_var_dims.count(new_var_key)) {
                new_var_dims.insert(new_var_key);
                bool first_var = new_var_code.length() == 0;
                if (gdims._get_num_dims())
                    new_var_code += "\n // Vars with '" + new_var_key + "' dim(s).\n";
                else
                    new_var_code += "\n // Scalar vars.\n";
                if (!first_var)
                    new_var_code += " else";
                new_var_code += " if (dims == " + var_dim_names + ")\n"
                    " gp = std::make_shared<" + base_t + ">(*this, name, dims);\n";
            }
        } // vars.

        os << "\n // Core data for each thread.\n"
            " std::vector<" << _core_t << "> _core_list;\n";

        // Stencil eq_bundle objects.
        os << endl << " // Stencil equation-bundles." << endl;
        for (auto& eg : _eq_bundles.get_all()) {
            string eg_name = eg->_get_name();
            os << " StencilBundleTempl<StencilBundle_" << eg_name << ", " <<
                _core_t << "> " << eg_name << ";" << endl;
        }

        os << "\n public:\n";

        // Ctor.
        {
            os << "\n // Constructor.\n" <<
                " " << _context << "(KernelEnvPtr env, KernelSettingsPtr settings) : " <<
                " StencilContext(env, settings)";
            for (auto& eg : _eq_bundles.get_all()) {
                string eg_name = eg->_get_name();
                os << ",\n  " << eg_name << "(this)";
            }
            os << " {\n"
                " name = \"" << _stencil._get_name() << "\";\n"
                " long_name = \"" << _stencil.get_long_name() << "\";\n";

            os << "\n // Create vars (but do not allocate data in them).\n" <<
                ctor_code <<
                "\n // Update vars with context info.\n"
                " update_var_info(false);\n";

            // Push eq-bundle pointers to list.
            for (auto& eg : _eq_bundles.get_all()) {
                string eg_name = eg->_get_name();

                os << "\n // Configure '" << eg_name << "'.\n";

                // Only want non-scratch bundles in st_bundles.
                // Each scratch bundles will be added to its
                // parent bundle.
                if (!eg->is_scratch())
                    os << "  st_bundles.push_back(&" << eg_name << ");\n";

                // Add scratch-bundle deps in proper order.
                auto& sdeps = _eq_bundles.get_scratch_deps(eg);
                for (auto& eg2 : _eq_bundles.get_all()) {
                    if (sdeps.count(eg2)) {
                        string eg2_name = eg2->_get_name();
                        os << "  " << eg_name <<
                            ".add_scratch_child(&" << eg2_name << ");\n";
                    }
                }

                // Add deps between bundles.
                for (auto& dep : _eq_bundles.get_deps(eg)) {
                    string dep_name = dep->_get_name();
                    os << "  " << eg_name <<
                        ".add_dep(&" << dep_name << ");\n";
                }

                // Populate the var lists in the StencilBundleBase objs.
                // I/O vars.
                os << "\n // The following var(s) are read by '" << eg_name << "'.\n";
                for (auto gp : eg->get_input_vars()) {
                    VAR_DECLS(gp);
                    if (gp->is_scratch())
                        os << "  " << eg_name << ".input_scratch_vecs.push_back(&" << var_list << ");\n";
                    else
                        os << "  " << eg_name << ".input_var_ptrs.push_back(" << var_ptr << ");\n";
                }
                os << "\n // The following var(s) are written by '" << eg_name << "'";
                if (eg->step_expr)
                    os << " at " << eg->step_expr->make_quoted_str();
                os << ".\n";
                for (auto gp : eg->get_output_vars()) {
                    VAR_DECLS(gp);
                    if (gp->is_scratch())
                        os << "  " << eg_name << ".output_scratch_vecs.push_back(&" << var_list << ");\n";
                    else
                        os << "  " << eg_name << ".output_var_ptrs.push_back(" << var_ptr << ");\n";
                }
            } // bundles.

            // Stages.
            os << "\n // Create stencil stage(s).\n";
            for (auto& bp : _eq_stages.get_all()) {
                if (bp->is_scratch())
                    continue;
                string bp_name = bp->_get_name();
                os << "  auto " << bp_name << " = std::make_shared<Stage>(this, \"" <<
                    bp_name << "\");\n";
                for (auto& eg : bp->get_bundles()) {
                    if (eg->is_scratch())
                        continue;
                    string eg_name = eg->_get_name();
                    os << "  " << bp_name << "->push_back(&" << eg_name << ");\n";
                }
                os << "  st_stages.push_back(" << bp_name << ");\n";
            }
                
            os << "\n // Call code provided by user.\n" <<
                _context_hook << "::call_after_new_solution(*this);\n";

            // end of ctor.
            os << " } // ctor" << endl;
        }

        // New-var method.
        os << "\n // Make a new var iff its dims match any in the stencil.\n"
            " // Returns pointer to the new var or nullptr if no match.\n"
            " VarBasePtr new_stencil_var(const std::string& name,"
            " const VarDimNames& dims) override {\n"
            " VarBasePtr gp;\n" <<
            new_var_code <<
            " return gp;\n"
            " } // new_stencil_var\n";

        // Scratch-vars method.
        os << "\n // Make new scratch vars for each thread.\n"
            " void make_scratch_vars(int num_threads) override {\n" <<
            scratch_code <<
            " } // make_scratch_vars\n";

        // Core-setting method.
        {
            os << "\n // Set the core pointers.\n"
                " void set_cores(int num_threads) override {\n"
                "  _core_list.resize(num_threads);\n"
                "  assert(_core_list.data());\n"
                "  for (int i = 0; i < num_threads; i++) {\n"
                "   _core_list[i].rank_domain_offsets = rank_domain_offsets;\n"
                "   " << core_code <<
                "  }\n";

            // Set _core_list in each bundle.
            for (auto& eg : _eq_bundles.get_all()) {
                string eg_name = eg->_get_name();
                os << " " << eg_name << ".set_core_list(_core_list.data());\n";
            }
            os << " } // set_cores\n";
        }
            
        // APIs.
        os << "\n virtual std::string get_target() const override {\n"
            "  return \"" << _settings._target << "\";\n"
            " }\n"
            "\n virtual int get_element_bytes() const override {\n"
            "  return " << _settings._elem_bytes << ";\n"
            " }\n";

        // Dims creator.
        os << "\n  // Create Dims object.\n"
            "  static DimsPtr new_dims() {\n"
            "    auto p = std::make_shared<Dims>();\n";
        for (int i = 0; i < _dims._fold_gt1._get_num_dims(); i++)
            os << "    p->_vec_fold_layout.set_size(" << i << ", " <<
                _dims._fold_gt1[i] << "); // '" <<
                _dims._fold_gt1.get_dim_name(i) << "'\n";
        os <<
            "    p->_step_dim = \"" << _dims._step_dim << "\";\n"
            "    p->_inner_dim = \"" << _dims._inner_dim << "\";\n";
        for (auto& dim : _dims._domain_dims) {
            auto& dname = dim._get_name();
            os << "    p->_domain_dims.add_dim_back(\"" << dname << "\", 0);\n";
        }
        for (auto& dim : _dims._stencil_dims) {
            auto& dname = dim._get_name();
            os << "    p->_stencil_dims.add_dim_back(\"" << dname << "\", 0);\n";
        }
        for (auto& dim : _dims._misc_dims) {
            auto& dname = dim._get_name();
            os << "    p->_misc_dims.add_dim_back(\"" << dname << "\", 0);\n";
        }
        for (auto& dim : _dims._fold) {
            auto& dname = dim._get_name();
            auto& dval = dim.get_val();
            os << "    p->_fold_pts.add_dim_back(\"" << dname << "\", " << dval << ");\n";
        }
        for (auto& dim : _dims._fold_gt1) {
            auto& dname = dim._get_name();
            auto& dval = dim.get_val();
            os << "    p->_vec_fold_pts.add_dim_back(\"" << dname << "\", " << dval << ");\n";
        }
        string ffi = (_dims._fold.is_first_inner()) ? "true" : "false";
        os << "    p->_fold_pts.set_first_inner(" << ffi << ");\n"
            "    p->_vec_fold_pts.set_first_inner(" << ffi << ");\n";
        for (auto& dim : _dims._cluster_pts) {
            auto& dname = dim._get_name();
            auto& dval = dim.get_val();
            os << "    p->_cluster_pts.add_dim_back(\"" << dname << "\", " << dval << ");\n";
        }
        for (auto& dim : _dims._cluster_mults) {
            auto& dname = dim._get_name();
            auto& dval = dim.get_val();
            os << "    p->_cluster_mults.add_dim_back(\"" << dname << "\", " << dval << ");\n";
        }
        os << "    p->_step_dir = " << _dims._step_dir << ";\n";

        os << "    return p;\n"
            "  }\n";

        os << "}; // " << _context << endl;
    }

} // namespace yask.
