/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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
    void YASKCppPrinter::print_indices(ostream& os,
                                       bool print_step, bool print_domain,
                                       const string var_prefix,
                                       const string inner_var_prefix) const {
        if (print_step && print_domain)
            os << "\n // Extract index for each stencil dim.\n";
        else if (print_step)
            os << "\n // Extract index for the step dim.\n";
        else if (print_domain)
            os << "\n // Extract index for each domain dim.\n";
        else
            return;
        int i = 0;
        for (auto& dim : _dims._stencil_dims) {
            auto& dname = dim._get_name();
            bool is_step = dname == _dims._step_dim;
            bool is_inner = dname == _settings._inner_loop_dim;
            if ((print_step && is_step) ||
                (print_domain && !is_step)) {
                if (inner_var_prefix.length() && is_inner)
                    os << " idx_t " << dname << " = " << inner_var_prefix << i << ";\n";
                else if (var_prefix.length())
                    os << " idx_t " << dname << " = " << var_prefix << i << ";\n";
                else
                    os << " idx_t " << dname << " = idxs[" << i << "];\n";
            }
            i++;
        }
    }

    // Print an expression as a one-line C++ comment.
    void YASKCppPrinter::add_comment(ostream& os, EqBundle& eq) {

        // Use a simple human-readable visitor to create a comment.
        PrintHelper ph(_settings, _dims, NULL, "", " // ", ".\n");
        PrintVisitorTopDown commenter(os, ph);
        eq.visit_eqs(&commenter);
    }

    // Print YASK code in new stencil context class.
    void YASKCppPrinter::print(ostream& os) {

        string sname = _stencil.get_long_name();
        os << "// Automatically-generated code; do not edit.\n"
            "\n////// YASK implementation of the '" << sname <<
            "' stencil //////\n";

        // Macros.
        os << "\n#if defined(DEFINE_MACROS) && !defined(MACROS_DONE)\n"
            "#define MACROS_DONE\n";
        print_macros(os);
        os << "\n#endif // DEFINE_MACROS\n";

        // Stencil-context code.
        os << "\n#if defined(DEFINE_CONTEXT) && !defined(CONTEXT_DONE)\n"
            "#define CONTEXT_DONE\n"
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
        os << "\n// Target:\n"
            "#define YASK_TARGET \"" << _settings._target << "\"\n"
            "#define REAL_BYTES " << _settings._elem_bytes << endl;

        auto nsdims = _dims._stencil_dims.size();
        os << "\n// Dimensions:\n"
            "#define STEP_DIM " << _dims._step_dim << endl <<
            "#define INNER_LOOP_DIM " << _settings._inner_loop_dim << endl;
        os << "#define NUM_DOMAIN_DIMS " << _dims._domain_dims.size() << endl;
        for (int i = 0; i < _dims._domain_dims.get_num_dims(); i++) {
            auto& dim = _dims._domain_dims(i);
            auto& dname = dim._get_name();
            os << "#define DOMAIN_DIM_IDX_" << dname << " " << i << endl;
        }
        os << "#define NUM_STENCIL_DIMS " << nsdims << endl;
        for (int i = 0; i < _dims._stencil_dims.get_num_dims(); i++) {
            auto& dim = _dims._stencil_dims(i);
            auto& dname = dim._get_name();
            os << "#define STENCIL_DIM_IDX_" << dname << " " << i << endl;
        }
        os << "#define STENCIL_DIM_IDX_INNER_LOOP " << _dims._inner_loop_dim_num << endl;
        os << "#define DOMAIN_LOOP_DIMS ";
        bool need_comma = false;
        for (int i = 0; i < _dims._domain_dims.get_num_dims(); i++) {
            auto& dim = _dims._domain_dims(i);
            auto& dname = dim._get_name();
            if (need_comma)
                os << ",";
            os << (i+1);
            need_comma = true;
        }
        os << "\n#define PICO_BLOCK_OUTER_LOOP_DIMS ";
        need_comma = false;
        for (int i = 0; i < _dims._domain_dims.get_num_dims(); i++) {
            if (i+1 != _dims._inner_loop_dim_num) {
                auto& dim = _dims._domain_dims(i);
                auto& dname = dim._get_name();
                if (need_comma)
                    os << ",";
                os << (i+1);
                need_comma = true;
            }
        }
        os << "\n#define PICO_BLOCK_INNER_LOOP_DIM " << _dims._inner_loop_dim_num << "\n";
        int gdims = 0;
        for (auto gp : _vars) {
            int ndims = gp->get_num_dims();
            gdims = max(gdims, ndims);
        }
        os << "\n// Max number of var dimensions:\n"
            "#define NUM_VAR_DIMS " << gdims << endl;
        os << "\n// Max of stencil and var dims:\n"
            "#define NUM_STENCIL_AND_VAR_DIMS " << max<int>(gdims, nsdims) << endl;

        os << "\n// Number of stencil equations:\n"
            "#define NUM_STENCIL_EQS " << _stencil.get_num_equations() << endl;

        // Vec/cluster lengths.
        auto nvec = _dims._fold_gt1.get_num_dims();
        os << "\n// One vector fold: " << _dims._fold.make_dim_val_str(" * ") << endl;
        for (auto& dim : _dims._fold) {
            auto& dname = dim._get_name();
            string uc_dim = all_caps(dname);
            os << "#define VLEN_" << uc_dim << " " << dim.get_val() << endl;
        }
        os << "namespace yask {\n"
            "\n // Number of points or multipliers in domain dims.\n"
            " constexpr idx_t fold_pts[]{ " << _dims._fold.make_val_str() << " };\n"
            " constexpr idx_t cluster_pts[]{ " << _dims._cluster_pts.make_val_str() << " };\n"
            " constexpr idx_t cluster_mults[]{ " << _dims._cluster_mults.make_val_str() << " };\n"
            "\n // Number of points or multipliers in stencil dims.\n"
            " constexpr idx_t stencil_fold_pts[]{ 1, " << _dims._fold.make_val_str() << " };\n"
            " constexpr idx_t stencil_cluster_pts[]{ 1, " << _dims._cluster_pts.make_val_str() << " };\n"
            " constexpr idx_t stencil_cluster_mults[]{ 1, " << _dims._cluster_mults.make_val_str() << " };\n"
            "}\n";
        os << "#define VLEN (" << _dims._fold.product() << ")\n"
            "#define CPTS (" << _dims._cluster_pts.product() << ")\n";
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

    // A handy macro to create some local vars based on a Var ptr.
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

        // Preferred dim layout order.
        vector<string> dorder;
        for (int i = 0; i < _dims._domain_dims.get_num_dims(); i++)
            dorder.push_back(_dims._domain_dims.get_dim_name(i));
        if (_settings._inner_step && dorder.size() > 1)
            dorder.insert(dorder.begin() + 1, _dims._step_dim);
        else
            dorder.insert(dorder.begin(), _dims._step_dim);
        for (int i = 0; i < _dims._misc_dims.get_num_dims(); i++) {
            auto& dname = _dims._misc_dims.get_dim_name(i);
            if (_settings._inner_misc)
                dorder.push_back(dname);
            else
                dorder.insert(dorder.begin() + i, dname);
        }
        os << "\n ///// Stencil var type(s).\n"
            " // General array layout order (outer-to-inner): ";
        for (size_t i = 0; i < dorder.size(); i++) {
            if (i > 0)
                os << ", ";
            os << dorder[i];
        }
        os << ".\n";
        
        // Var types.
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
                os << " // Dimensions in parameter order: ";
                for (int dn = 0; dn < ndims; dn++) {
                    if (dn)
                        os << ", ";
                    auto& dim = gp->get_dims()[dn];
                    auto& dname = dim->_get_name();
                    os << "'" << dname << "'(#" << (dn+1) << ")";
                }
                os << ".\n";
            }

            // Use vector-folded layout if possible.
            // Possible when a var contains all of the dims with
            // vec-len > 1.
            bool folded = gp->is_foldable();
            string vtype = folded ? "YkVecVar" : "YkElemVar";
            string ctype = vtype + "Core";

            // Create the template params.
            // Type-name in kernel is 'VAR_TYPE<LAYOUT, WRAP_1ST_IDX, VEC_LENGTHS...>'.
            {
                string templ;
                bool got_step = false;

                // 1-D or more.
                if (ndims) {
                    int ndone = 0;
                    os << " // Dimensions in layout order: ";

                    // Preferred order.
                    for (size_t i = 0; i < dorder.size(); i++) {
                        auto& dname = dorder[i];

                        // Find in this var (if it exists).
                        for (int dn = 0; dn < ndims; dn++) {
                            auto& dim = gp->get_dims()[dn];
                            if (dname != dim->_get_name())
                                continue;
                            auto dtype = dim->get_type();

                            // Step dim?
                            if (dtype == STEP_INDEX) {
                                assert(dname == _dims._step_dim);
                                if (dn > 0) {
                                    THROW_YASK_EXCEPTION("Error: cannot create var '" + var +
                                                         "' with dimensions '" + gdims.make_dim_str() +
                                                         "' because '" + dname + "' must be first dimension");
                                }
                                got_step = true;
                            }

                            // Inner domain dim?
                            else if (dname == _dims._inner_layout_dim) {
                                assert(dtype == DOMAIN_INDEX);
                            }

                            // Add this index position to layout template.
                            templ += to_string(dn+1);
                            if (ndone)
                                os << ", ";
                            os << dname;
                            ndone++;
                        }
                    } // dims.
                    assert(ndims == ndone);
                    os << ".\n";

                } // not scalar.

                // Scalar.
                else
                    templ += "0d"; // Trivial scalar layout.

                // Add step-dim flag.
                if (got_step)
                    templ += ", true";
                else
                    templ += ", false";

                // Vector lengths.
                if (folded) {
                    for (int dn = 0; dn < ndims; dn++) {
                        auto& dim = gp->get_dims()[dn];
                        auto& dname = dim->_get_name();
                        
                        // Add vector len to list.
                        auto* p = _dims._fold.lookup(dname);
                        int dval = p ? *p : 1;
                        templ += ", " + to_string(dval);
                    }
                }

                templ.insert(0, "<Layout_");
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

        // Types with data needed in kernels.
        {
            os << "\n // Per-thread data needed in kernel(s).\n"
                "struct " << _thread_core_t << " {\n";

            bool found = false;
            for (auto gp : _vars) {
                VAR_DECLS(gp);
                if (gp->is_scratch()) {
                    if (!found)
                        os << "\n // Pointer(s) to scratch-var core data.\n";
                    os << " synced_ptr<" << core_t << "> " << core_ptr << ";\n";
                    found = true;
                }
            }
            if (!found)
                os << "\n // No per-thread data needed for this stencil.\n";
            os << "}; // " << _thread_core_t << endl;
        }
        {
            os << "\n // Data needed in kernel(s).\n"
                "struct " << _core_t << " : public StencilCoreBase {\n";

            os << "\n // Pointer(s) to var core data.\n";
            for (auto gp : _vars) {
                VAR_DECLS(gp);
                if (!gp->is_scratch())
                    os << " synced_ptr<" << core_t << "> " << core_ptr << ";\n";
            }
            os << "\n // List of pointer(s) to per-thread data.\n"
                " synced_ptr<" << _thread_core_t << "> _thread_core_list;\n"
                "}; // " << _core_t << endl;
        }
    }
        
    // Print YASK equation bundles.
    void YASKCppPrinter::print_eq_bundles(ostream& os) {

        for (auto& bp : _eq_stages.get_all()) {
            string stage_name = bp->_get_name();
            os << "\n //////// Stencil ";
            if (bp->is_scratch())
                os << "scratch-";
            os << "stage '" << stage_name << "' //////\n";

            // Bundles in this stage;
            for (auto& eq : bp->get_bundles()) {

                // Find equation index.
                // TODO: remove need for this.
                int ei = 0;
                for (; ei < _eq_bundles.get_num(); ei++) {
                    if (eq == _eq_bundles.get_all().at(ei))
                        break;
                }
                assert(ei < _eq_bundles.get_num());
                string eg_name = eq->_get_name();
                string eg_desc = eq->get_descr();
                string egs_name = _stencil_prefix + eg_name;

                // Stats for this eq_bundle.
                CounterVisitor stats;
                eq->visit_eqs(&stats);

                os << endl << " ////// Stencil " << eg_desc << " //////\n" <<
                "\n struct " << egs_name << " {\n"
                    "  const char* _name = \"" << eg_name << "\";\n"
                    "  const int _scalar_fp_ops = " << stats.get_num_ops() << ";\n"
                    "  const int _scalar_points_read = " << stats.get_num_reads() << ";\n"
                    "  const int _scalar_points_written = " << stats.get_num_writes() << ";\n"
                    "  const bool _is_scratch = " << (eq->is_scratch() ? "true" : "false") << ";\n";

                // Example computation.
                os << endl << " // " << stats.get_num_ops() << " FP operation(s) per point:\n";
                add_comment(os, *eq);

                // Domain condition.
                {
                    os << "\n // Determine whether " << egs_name << " is valid at the domain indices " <<
                        _dims._stencil_dims.make_dim_str() << ".\n"
                        " // Return true if indices are within the valid sub-domain or false otherwise.\n"
                        " ALWAYS_INLINE static bool is_in_valid_domain(const " <<
                        _core_t << "* core_data, const Indices& idxs) {"
                        " host_assert(core_data);\n";
                    print_indices(os);
                    if (eq->cond)
                        os << " return " << eq->cond->make_str() << ";\n";
                    else
                        os << " return true; // full domain.\n";
                    os << " }\n";

                    os << "\n // Return whether there is a sub-domain expression.\n"
                        " ALWAYS_INLINE static bool is_sub_domain_expr() {\n"
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
                        " ALWAYS_INLINE static bool is_in_valid_step(const " <<
                        _core_t << "* core_data, idx_t input_step_index) {"
                        " host_assert(core_data);\n";
                    if (eq->step_cond) {
                        os << " idx_t " << _dims._step_dim << " = input_step_index;\n"
                            "\n // " << eq->step_cond->make_str() << "\n";
                    
                        // C++ scalar print assistant.
                        CounterVisitor cv;
                        eq->step_cond->accept(&cv);
                        CppPrintHelper* sp = new CppPrintHelper(_settings, _dims, &cv, "real_t", " ", ";\n");

                        // Generate the code.
                        PrintVisitorTopDown pcv(os, *sp);
                        string expr = eq->step_cond->accept(&pcv);
                        os << " return " << expr << ";\n";
                    }
                    else
                        os << " return true; // any step.\n";
                    os << " }\n";

                    os << "\n // Return whether there is a step-condition expression.\n"
                        " ALWAYS_INLINE static bool is_step_cond_expr() {\n"
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
                    os << " ALWAYS_INLINE static bool get_output_step_index(idx_t input_step_index,"
                        " idx_t& output_step_index) {\n";
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
                        " static void calc_scalar(" <<
                        _core_t << "* core_data, int core_idx, const Indices& idxs) {\n"
                        " host_assert(core_data);\n"
                        " host_assert(core_data->_thread_core_list.get());\n"
                        " auto& thread_core_data = core_data->_thread_core_list[core_idx];\n";
                    print_indices(os);

                    // C++ scalar print assistant.
                    CounterVisitor cv;
                    eq->visit_eqs(&cv);
                    CppPrintHelper* sp = new CppPrintHelper(_settings, _dims, &cv, "real_t", " ", ";\n");

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

                    // Create vector info for this eq_bundle.  The visitor is
                    // accepted at all nodes in the cluster AST; for each var
                    // access node in the AST, the vectors needed are determined
                    // and saved in the visitor.
                    VecInfoVisitor vv(_dims);
                    vceq->visit_eqs(&vv);

                    // Collect stats.
                    CounterVisitor cv;
                    vceq->visit_eqs(&cv);
                    int num_results = do_cluster ?
                        _dims._cluster_pts.product() :
                        _dims._fold.product();

                    // Vector/cluster vars.
                    string idim = _settings._inner_loop_dim;
                    string vcstr = do_cluster ? "cluster" : "vector";
                    string funcstr = "calc_" + vcstr + "s";
                    string nvecs = do_cluster ? "CMULT_" + all_caps(idim) : "1";
                    string nelems = (do_cluster ? nvecs + " * ": "") + "VLEN_" + all_caps(idim);
                    string write_mask = do_cluster ? "" : "write_mask";

                    // Loop-calculation code.
                    // Function header.
                    os << endl << " // Calculate a nano-block of " << vcstr << "s bounded by 'norm_nb_idxs'.\n";
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
                        " FP operation(s) per inner-loop iteration.\n" <<
                        " static void " << funcstr << "(" <<
                        _core_t << "* core_data, int core_idx, int block_thread_idx,"
                        " int thread_limit, ScanIndices& norm_nb_idxs";
                    if (!do_cluster)
                        os << ", bit_mask_t " << write_mask;
                    os << ") {\n"
                        " FORCE_INLINE_RECURSIVE {\n"
                        " assert(core_data);\n"
                        " assert(core_data->_thread_core_list.get());\n"
                        " auto& thread_core_data = core_data->_thread_core_list[core_idx];\n"
                        " const Indices& idxs = norm_nb_idxs.start;\n";
                    print_indices(os, true, false); // Just step index.
 
                    // C++ vector print assistant.
                    auto* vp = new_cpp_vec_print_helper(vv, cv);
                    vp->set_write_mask(write_mask);
                    vp->set_using_cluster(do_cluster);
                    vp->set_stage_name(stage_name);

                    // Print loop-invariant meta values.
                    // Store them in the CppVecPrintHelper for later use in the loop body.
                    os << "\n ////// Loop-invariant meta values.\n";
                    CppPreLoopPrintMetaVisitor plpmv(os, *vp);
                    vceq->visit_eqs(&plpmv);
                    vp->print_rank_data(os);

                    // Print loop-invariant data values.
                    // Store them in the CppVecPrintHelper for later use in the loop body.
                    CppPreLoopPrintDataVisitor plpdv(os, *vp);
                    vceq->visit_eqs(&plpdv);
                
                    // Inner-loop strides.
                    // Will be 1 for vectors and cluster-mults for clusters.
                    string inner_strides = do_cluster ?
                        "stencil_cluster_mults[dn]" :
                        "idx_t(1)";

                    // Computation loops.
                    // Include generated loops.
                    os <<
                        "\n // Nano loops.\n"
                        "#define NANO_BLOCK_LOOP_INDICES norm_nb_idxs\n"
                        "\n // Start Nano loop(s).\n"
                        "#define NANO_BLOCK_USE_LOOP_PART_0\n"
                        "#include \"yask_nano_block_loops.hpp\"\n";
                    os <<
                        "\n // Pico loops inside nano loops.\n"
                        " // Use macros to get values directly from nano loops.\n"
                        "#define PICO_BLOCK_BEGIN(dn) NANO_BLOCK_BODY_START(dn)\n"
                        "#define PICO_BLOCK_END(dn) NANO_BLOCK_BODY_STOP(dn)\n"
                        "#define PICO_BLOCK_STRIDE(dn) " << inner_strides << "\n";
                    os <<
                        "\n // Start Pico outer-loop(s).\n"
                        "#define PICO_BLOCK_USE_LOOP_PART_0\n"
                        "#include \"yask_pico_block_loops.hpp\"\n";

                    // Get named domain indices directly from scalar vars.
                    print_indices(os, false, true, "pico_block_start_", "pico_block_begin_");
                    vp->print_elem_indices(os);

                    // Create inner-loop base ptrs.
                    os << "\n // Set up for inner loop.\n";
                    vp->print_inner_loop_prefix(os);
    
                    // Initial prefetches, if any.
                    vp->print_prefetches(os, false);

                    // Create and init buffers, if any.
                    vp->print_buffer_code(os, false);

                    auto& ild = _settings._inner_loop_dim;
                    os <<
                        "\n // Start Pico inner-loop for dim '" << ild << "'.\n"
                        "#define PICO_BLOCK_USE_LOOP_PART_1\n"
                        "#include \"yask_pico_block_loops.hpp\"\n";

                    // Issue loads early.
                    if (_settings._early_loads)
                        vp->print_early_loads(os);
                
                    // Generate loop body using vars stored in print helper.
                    // Visit all expressions to cover the whole vector/cluster.
                    PrintVisitorBottomUp pcv(os, *vp);
                    vceq->visit_eqs(&pcv);

                    // Insert prefetches using vars stored in print helper for next iteration.
                    vp->print_prefetches(os, true);

                    // Shift and fill buffers.
                    vp->print_buffer_code(os, true);
                
                    // Increment indices, etc.
                    vp->print_end_inner_loop(os);

                    // End of loops.
                    os <<
                        "\n ////// Loop endings.\n"
                        "#define PICO_BLOCK_USE_LOOP_PART_2\n"
                        "#include \"yask_pico_block_loops.hpp\"\n"
                        "#define NANO_BLOCK_USE_LOOP_PART_1\n"
                        "#include \"yask_nano_block_loops.hpp\"\n";

                    // End of recursive block & function.
                    os << "} } // " << funcstr << ".\n";
                    delete vp;
                } // cluster/vector

                os << "}; // " << egs_name << ".\n" // end of struct.
                    " static_assert(std::is_trivially_copyable<" << egs_name << ">::value,"
                    "\"Needed for OpenMP offload\");\n";

            } // stencil eq_bundles.
        } // stages.
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
                " host_assert(" + base_ptr + ");\n"
                " " + var_ptr + " = std::make_shared<YkVarImpl>(" + base_ptr + ");\n"
                " host_assert(" + var_ptr + "->gbp());\n";

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

                    // Extra padding for read-ahead.
                    if (dname == _settings._inner_loop_dim &&
                        gp->get_read_ahead_pad() > 0)
                        init_code += " " + var_ptr + "->update_right_extra_pad_size(\"" + dname +
                            "\", " + to_string(gp->get_read_ahead_pad()) + "); // For read-ahead.\n";
                }

                // non-domain dimension.
                else {
                    string avar = var + "_alloc_" + dname;
                    string ovar = var + "_ofs_" + dname;
                    int aval = 1;
                    int oval = 0;
                    string comment;
                    if (dtype == STEP_INDEX) {
                        auto sdi = gp->get_step_dim_info();
                        aval = sdi.step_dim_size;
                        auto& wb_ofs = sdi.writeback_ofs;
                        for (auto& i : gp->get_write_points()) {
                            auto& ws = i.first;
                            int sofs = i.second;
                            comment += " Written in stage '" + ws +
                                "' at step-offset " + to_string(sofs) +
                                " with writeback (immediate replacement)";
                            if (!wb_ofs.count(ws))
                                comment += " NOT allowed.";
                            else
                                comment += " allowed over read at step-offset " +
                                    to_string(wb_ofs.at(ws)) + ".";
                        }
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
                        "; // Default allocation in '" << dname << "' dimension." <<
                        comment << "\n";
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

                // Core init for this var.
                core_code +=
                    "  auto* " + core_ptr + " = static_cast<" + core_t + "*>(" + var_ptr + "->corep());\n"
                    "  cxt_cd->" + core_ptr + ".set_and_sync(" + core_ptr + ");\n";
            }

            // For scratch, make code to fill vector.
            else {
                scratch_code +=
                    " " + var_list + ".resize(num_threads);\n"
                    " for (int i = 0; i < num_threads; i++) {\n" +

                    // Make scratch var for 'i'th thread.
                    " YkVarPtr " + var_ptr + ";\n" +
                    init_code +
                    " " + base_ptr + "->set_scratch(true);\n" +
                    " " + var_list + "[i] = " + var_ptr + ";\n" +

                    // Init core ptr for this var.
                    " auto* cp = static_cast<" + core_t + "*>(" + var_ptr + "->corep());\n"
                    " _core_data._thread_core_list[i]." + core_ptr + ".set_and_sync(cp);\n"

                    " }\n";
            }

            // Make new vars via API.
            string new_var_key = gdims.make_dim_str();
            if (!new_var_dims.count(new_var_key)) {
                new_var_dims.insert(new_var_key);
                bool first_var = new_var_code.length() == 0;
                if (gdims.get_num_dims())
                    new_var_code += "\n // Vars with '" + new_var_key + "' dim(s).\n";
                else
                    new_var_code += "\n // Scalar vars.\n";
                if (!first_var)
                    new_var_code += " else";
                new_var_code += " if (dims == " + var_dim_names + ")\n"
                    " gp = std::make_shared<" + base_t + ">(*this, name, dims);\n";
            }
        } // vars.

        os << "\n // Core data used in kernel(s).\n"
            " " << _core_t << " _core_data;\n" <<
            " std::vector<" << _thread_core_t << "> _thread_cores;\n";

        // Stencil eq_bundle objects.
        os << endl << " // Stencil equation-bundles." << endl;
        for (auto& eg : _eq_bundles.get_all()) {
            string eg_name = eg->_get_name();
            os << " StencilBundleTempl<" << _stencil_prefix << eg_name << ", " <<
                _core_t << "> " << eg_name << ";" << endl;
        }

        os << "\n public:\n";

        // Ctor.
        {
            os << "\n // Constructor.\n" <<
                " " << _context << "(KernelEnvPtr kenv, "
                "KernelSettingsPtr ksettings, "
                "KernelSettingsPtr user_settings) : " <<
                " StencilContext(kenv, ksettings, user_settings)";
            for (auto& eg : _eq_bundles.get_all()) {
                string eg_name = eg->_get_name();
                os << ",\n  " << eg_name << "(this)";
            }
            os << " {\n"
                " STATE_VARS(this);\n"
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

            os << "\n // Alloc core on offload device.\n"
                "  auto* cxt_cd = &_core_data;\n"
                "  offload_map_alloc(cxt_cd, 1);\n";
                
            os << "\n // Call code provided by user.\n" <<
                _context_hook << "::call_after_new_solution(*this);\n";

            // end of ctor.
            os << " } // ctor" << endl;
        }

        // Dtor.
        os << "\n // Dtor.\n"
            " virtual ~" << _context << "() {\n"
            "  STATE_VARS(this);\n"
            "  auto* cxt_cd = &_core_data;\n"
            "  offload_map_free(cxt_cd, 1);\n" <<
            "  auto* tcl = _thread_cores.data();\n"
            "  auto nt = _thread_cores.size();\n"
            "  if (tcl && nt) offload_map_free(tcl, nt);\n"
            " }\n";

        // New-var method.
        os << "\n // Make a new var iff its dims match any in the stencil.\n"
            " // Returns pointer to the new var or nullptr if no match.\n"
            " VarBasePtr new_stencil_var(const std::string& name,"
            " const VarDimNames& dims) override {\n"
            " VarBasePtr gp;\n" <<
            new_var_code <<
            " return gp;\n"
            " } // new_stencil_var\n";

        // Core methods.
        {
            os << "\n // Set the core pointers of the non-scratch vars and copy some other info.\n"
                " void set_core() override {\n"
                "  auto* cxt_cd = &_core_data;\n"
                "  cxt_cd->_common_core.set_core(this);\n"
                "  offload_copy_to_device(cxt_cd, 1);\n" <<
                core_code <<
                " }\n";
            os << "\n // Access the core data.\n"
                " StencilCoreBase* corep() override {\n"
                "  return &_core_data;\n"
                " }\n";
        }

        // Scratch-vars method.
        os << "\n // Make new scratch vars for each thread and sync offload core ptr.\n"
            " // Does not allocate data for vars.\n"
            " // Must call set_core() before this.\n"
            " void make_scratch_vars(int num_threads) override {\n"
            " STATE_VARS(this);\n"
            " TRACE_MSG(\"make_scratch_vars(\" << num_threads << \")\");\n"
            "\n  // Release old device data for thread array.\n"
            "  auto* tcl = _thread_cores.data();\n"
            "  auto old_nt = _thread_cores.size();\n"
            "  if (tcl && old_nt) offload_map_free(tcl, old_nt);\n"
          "\n  // Make new array.\n"
            "  _thread_cores.resize(num_threads);\n"
            "  tcl = num_threads ? _thread_cores.data() : 0;\n"
            "  if (tcl)\n"
            "    offload_map_alloc(tcl, num_threads);\n"
            "  _core_data._thread_core_list.set_and_sync(tcl);\n"
            "\n  // Create scratch var(s) and set core ptr(s).\n" <<
            scratch_code <<
            " } // make_scratch_vars\n";
            
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
        for (int i = 0; i < _dims._fold_gt1.get_num_dims(); i++)
            os << "    p->_vec_fold_layout.set_size(" << i << ", " <<
                _dims._fold_gt1[i] << "); // '" <<
                _dims._fold_gt1.get_dim_name(i) << "'\n";
        os <<
            "    p->_step_dim = \"" << _dims._step_dim << "\";\n"
            "    p->_inner_layout_dim = \"" << _dims._inner_layout_dim << "\";\n"
            "    p->_inner_loop_dim = \"" << _settings._inner_loop_dim << "\";\n";
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
        os << "    p->_fold_sizes.set_from_tuple(p->_fold_pts);\n";
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
