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

////////// Support for C++ scalar and vector-code generation //////////////

#include "Cpp.hpp"

namespace yask {

    /////////// Scalar code /////////////

    // Format a real, preserving precision.
    // Also used for vector code, assuming automatic broadcast.
    string CppPrintHelper::format_real(double v) {

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
    string CppPrintHelper::make_point_call(ostream& os,
                                           const VarPoint& gp,
                                           const string& fname,
                                           string opt_arg) {

        // Get/set local vars.
        string var_ptr = get_local_var(os, get_var_ptr(gp), _var_ptr_restrict_type);
        string sas = gp.make_step_arg_str(var_ptr, _dims);
        string step_arg = sas.length() ? get_local_var(os, sas, _step_val_type) : "0";

        string res = var_ptr + "->" + fname + "(";
        if (opt_arg.length())
            res += opt_arg + ", ";
        string args = gp.make_arg_str();
        res += "{" + args + "}, " + step_arg + ")";
        return res;
    }

    // Return a var-point reference.
    string CppPrintHelper::read_from_point(ostream& os, const VarPoint& gp) {
        return make_point_call(os, gp, "read_elem");
    }

    // Return code to update a var point.
    string CppPrintHelper::write_to_point(ostream& os, const VarPoint& gp,
                                          const string& val) {
        return make_point_call(os, gp, "write_elem", val);
    }

    /////////// Vector code /////////////

    // Create call for a point.
    // This is a utility function used for most var accesses.
    string CppVecPrintHelper::make_point_call(ostream& os,
                                              const VarPoint& gp,
                                              const string& func_name,
                                              const string& first_arg,
                                              const string& last_arg,
                                              bool is_vec_norm,
                                              const VarMap* var_map) {

        // Vec-norm accesses must be from folded var.
        if (is_vec_norm)
            assert(gp.is_var_foldable());
        
        // Var map is required for non-vec accesses to get elem indices.
        else
            assert(var_map);

        // Get/set local vars.
        string var_ptr = get_local_var(os, get_var_ptr(gp),
                                       CppPrintHelper::_var_ptr_restrict_type);
        string sas = gp.make_step_arg_str(var_ptr, _dims);
        string step_arg = sas.length() ?
            get_local_var(os, sas, CppPrintHelper::_step_val_type) : "0";

        string res = var_ptr + "->" + func_name + "(";
        if (first_arg.length())
            res += first_arg + ", ";
        string args = is_vec_norm ?
            gp.make_norm_arg_str(_dims, var_map) : gp.make_arg_str(var_map);
        res += "{" + args + "} ," + step_arg;
        if (last_arg.length())
            res += ", " + last_arg;
        res += ")";
        return res;
    }
    
    // Make base point: same as 'gp', but misc indices and domain indices =
    // local offset (step index unchanged).
    var_point_ptr CppVecPrintHelper::make_base_point(const VarPoint& gp) {
        var_point_ptr bgp = gp.clone_var_point();
        auto* var = bgp->_get_var();
        assert(var);
        bool is_scratch = var->is_scratch();
        int dnum = 0;
        for (auto& dim : bgp->get_dims()) {
            auto& dname = dim->_get_name();
            auto type = dim->get_type();

            if (type == DOMAIN_INDEX || type == MISC_INDEX) {
                auto* lofs = lookup_local_offset(*var, dname);
                assert(lofs);
                bgp->set_arg_expr(dname, *lofs);
            }
            dnum++;
        }
        return bgp;
    }
    
    // Print base pointer of 'gp'.
    void CppVecPrintHelper::print_base_ptr(ostream& os, const VarPoint& gp) {
        if (!_settings._use_ptrs)
            return;

        // Got a pointer to it already?
        auto* p = lookup_base_point_ptr(gp);
        if (!p) {

            // Make base point (misc & domain indices set to canonical
            // value). There will be one pointer for every
            // unique var/step-arg combo.
            auto bgp = make_base_point(gp);
            auto* var = bgp->_get_var();
           
            // Make and save ptr var for future use.
            string ptr_name = make_var_name();
            _vec_ptrs[*bgp] = ptr_name;

            // Print pointer definition.
            print_point_comment(os, *bgp, "Create pointer");

            // Get pointer to var using normalized indices.
            // Ignore out-of-range errors because we might get a base pointer to an
            // element before the allocated range.
            // TODO: is this still true with local offsets?
            bool folded = var->is_foldable();
            auto vp = folded ?
                make_point_call(os, *bgp, "get_vec_ptr_norm", "", "false", true) :
                make_point_call(os, *bgp, "get_elem_ptr_local", "", "false", false, &_vec2elem_local_map);

            // Ptr should provide unique access if all accesses are through pointers.
            // TODO: check for non-ptr accesses via read/write calls.
            bool is_unique = !_settings._allow_unaligned_loads;
            string type = is_unique ? _var_ptr_restrict_type : _var_ptr_type;

            // Print type and value.
            os << _line_prefix << type << " " << ptr_name << " = " << vp << _line_suffix;
        }

        // Collect some stats for reads using this ptr.
        // These stats will be used for calculating prefetch ranges.
        // TODO: update prefetch to work with any inner-loop dim, not
        // just the one that's the same as the inner dim of the layouts.
        p = lookup_base_point_ptr(gp);
        assert(p);

        // Get const offsets from original.
        auto& offsets = gp.get_arg_offsets();

        // Get offset in inner dim.
        // E.g., A(t, x+1, y+4) => 4.
        const string& idim = _dims._inner_dim;
        auto* ofs = offsets.lookup(idim);
        if (ofs) {
            
            // Remember lowest inner-dim offset from this ptr.
            if (!_ptr_ofs_lo.count(*p) || _ptr_ofs_lo[*p] > *ofs)
                _ptr_ofs_lo[*p] = *ofs;
            
            // Remember highest one.
            if (!_ptr_ofs_hi.count(*p) || _ptr_ofs_hi[*p] < *ofs)
                _ptr_ofs_hi[*p] = *ofs;
        }
    }

    // Print creation of stride and local-offset vars.
    // Save var names for later use.
    void CppVecPrintHelper::print_strides(ostream& os, const VarPoint& gp) {
        auto* vp = gp._get_var();
        assert(vp);
        auto& var = *vp;

        for (int dnum = 0; dnum < var.get_num_dims(); dnum++) {
            auto& dim = var.get_dims().at(dnum);
            const auto& vname = var.get_name();
            const auto& dname = dim->_get_name();
            auto dtype = dim->get_type();

            // Don't need step-dim stride.
            bool is_step = dtype == STEP_INDEX;
            
            auto key = VarDimKey(vname, dname);
            if (!is_step && !_strides.count(key) ) {
                os << endl << " // Stride for var '" << vname <<
                    "' in '" << dname << "' dim.\n";

                string svar = make_var_name();
                _strides[key] = svar;
                assert(lookup_stride(var, dname));

                // Get ptr to var core.
                string var_ptr = get_local_var(os, get_var_ptr(var),
                                               CppPrintHelper::_var_ptr_restrict_type);

                // Determine stride.
                // Default is to obtain dynamic value from var.
                string slookup = var_ptr + "->_vec_strides[" + to_string(dnum) + "]";
                string stride = slookup;

                // Inner dim stride and inner misc dim strides are fixed.
                // NB: during layout, the inner dim is moved to the inner-most
                // position; the misc dims are moved after that if '_inner_misc' is true.
                bool is_inner = dname == _dims._inner_dim;
                bool is_inner_misc = dtype == MISC_INDEX && _settings._inner_misc;
                if (is_inner || is_inner_misc) {
                    os << " // This is a known fixed value.\n";

                    // Stride of inner dim will be 1 or size of misc vars.
                    stride = "1";

                    // Mult by size of following inner-misc dims, if any.
                    // If inner-misc is active, all misc dims are nested
                    // inside the inner domain dim, so their strides are
                    // fixed.
                    if (_settings._inner_misc)
                        for (int j = 0; j < var.get_num_dims(); j++) {
                            auto& dimj = var.get_dims().at(j);
                            auto& dnj = dimj->_get_name();
                            auto typej = dimj->get_type();
                            if (typej == MISC_INDEX && (is_inner || j > dnum)) {
                                auto min_idx = var.get_min_indices()[dnj];
                                auto max_idx = var.get_max_indices()[dnj];
                                auto sz = max_idx - min_idx + 1;
                                os << " // Misc indices for '" << dnj << "' range from " <<
                                    min_idx << " to " << max_idx << ": " << sz << " value(s).\n";
                                stride += " * " + to_string(sz);
                            }
                    }
                }

                // Print final assignment.
                os << _line_prefix << "const idx_t " << svar << " = " <<
                    stride << _line_suffix;
                if (stride != slookup)
                    os << _line_prefix << "host_assert(" << slookup <<
                        " == " << svar << ")" << _line_suffix;

                // Local offset.
                string lofs_lookup = var_ptr + "->_local_offsets[" + to_string(dnum) + "]";
                string lofs_var = make_var_name();
                string lofs("idx_t(0)"); // Known to be zero.
                _local_offsets[key] = lofs_var;
                assert(lookup_local_offset(var, dname));
                os << endl << " // Local offset for var '" << vname <<
                    "' in '" << dname << "' dim.\n";

                // Lookup needed for domain dim in scratch var because scratch
                // vars are "relocated" to location of current block.
                if (var.is_scratch() && dtype == DOMAIN_INDEX) {
                    os << " // This value varies because '" << vname <<
                        "' is a scratch var.\n";
                    lofs = lofs_lookup;
                }
                else
                    os << " // This is a known fixed value.\n";

                // Need min value for misc indices.
                if (dtype == MISC_INDEX)
                    lofs = "idx_t(" + to_string(var.get_min_indices()[dname]) + ")";
                
                os << _line_prefix << "const idx_t " << lofs_var << " = " <<
                    lofs << _line_suffix;
                if (lofs != lofs_lookup)
                    os << _line_prefix << "host_assert(" << lofs_lookup <<
                        " == " << lofs_var << ")" << _line_suffix;
            }
        }
    }

    // Print prefetches for each base pointer.
    // 'ahead': prefetch PF distance ahead instead of up to PF dist.
    // TODO: handle of misc dims.
    // TODO: allow non-inner-dim prefetching.
    void CppVecPrintHelper::print_prefetches(ostream& os,
                                             bool ahead, string ptr_var) {
        if (!_settings._use_ptrs)
            return;

        // cluster mult in inner dim.
        const string& idim = _dims._inner_dim;
        string imult = "CMULT_" + PrinterBase::all_caps(idim);

        // 'level': cache level.
        for (int level = 1; level <= 2; level++) {

            os << "\n // Prefetch to L" << level << " cache if enabled.\n";
            os << _line_prefix << "#if (PFD_L" << level << " > 0)";
            if (!ahead)
                os << " && (defined PREFETCH_BEFORE_LOOP)";
            os << "\n";

            // Loop thru vec ptrs.
            for (auto vp : _vec_ptrs) {
                auto& ptr = vp.second; // ptr var name.

                // Filter by ptr_var if provided.
                if (ptr_var.length() && ptr_var != ptr)
                    continue;

                // _ptr_ofs{Lo,Hi} contain first and last offsets in idim,
                // NOT normalized to vector len.
                string left = _dims.make_norm_str(_ptr_ofs_lo[ptr], idim);
                if (left.length() == 0) left = "0";
                string right = _dims.make_norm_str(_ptr_ofs_hi[ptr], idim);

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
                    _line_prefix << " for (int ofs = " << start <<
                    "; ofs < " << stop << "; ofs++) {\n";

                // Need to print prefetch for every unique var-point read.
                set<string> done;
                for (auto& gp : _vv._aligned_vecs) {

                    // For the current base ptr?
                    auto* p = lookup_base_point_ptr(gp);
                    if (p && *p == ptr) {

                        // Expression for this offset from inner-dim var.
                        string inner_ofs = "ofs";

                        // Expression for ptr offset at this point.
                        string ofs_expr = get_ptr_offset(os, gp, inner_ofs);
                        print_point_comment(os, gp, "Prefetch");

                        // Already done?
                        if (done.count(ofs_expr))
                            os << " // Already accounted for.\n";

                        else {
                            done.insert(ofs_expr);

                            // Prefetch.
                            os << _line_prefix << "  prefetch<L" << level << "_HINT>(&" << ptr <<
                                "[" << ofs_expr << "])" << _line_suffix;
                        }
                    }
                }

                // End loop;
                os << " }\n";
            }
            os << _line_prefix << "#endif // L" << level << " prefetch.\n";
        }
    }

    // Get expression for offset of 'gp' from base pointer.  Base pointer
    // points to vector with domain dims and misc dims == 0.
    string CppVecPrintHelper::get_ptr_offset(ostream& os,
                                             const VarPoint& gp,
                                             const string& inner_ofs) {
        auto* var = gp._get_var();
        assert(var);
        string ofs_str;
        int nterms = 0;

        // Construct the linear offset by adding the products of
        // each index with the var's stride in that dim.
        for (int i = 0; i < var->get_num_dims(); i++) {
            auto& dimi = gp.get_dims().at(i);
            auto typei = dimi->get_type();

            // There is a separate pointer for each value of
            // the step index, so we don't need to include
            // that index in the offset calculation.
            bool is_step = typei == STEP_INDEX;
            if (!is_step) {
                string dni = dimi->_get_name();
                auto* lofs = lookup_local_offset(*var, dni);
                assert(lofs);
                auto* stride = lookup_stride(*var, dni);
                assert(stride);
                string nas = gp.make_norm_arg_str(dni, _dims);

                // Add to offset expression.
                if (nterms)
                    ofs_str += " + ";
                nas += " - " + *lofs;
                if (dni == _dims._inner_dim && inner_ofs.length())
                    nas += " + " + inner_ofs;
                ofs_str += "((" + nas + ") * " + *stride + ")";
                nterms++;
            }
        }

        return ofs_str;
    }
    
    // Print any needed memory reads and/or constructions to 'os'.
    // Return code containing a vector of var points.
    string CppVecPrintHelper::read_from_point(ostream& os, const VarPoint& gp) {
        string code_str;

        // Already done and saved.
        if (_reuse_vars && _vec_vars.count(gp))
            code_str = _vec_vars[gp]; // do nothing.

        // If not done, continue based on type of vectorization.
        else {

            // Scalar point?
            if (gp.get_vec_type() == VarPoint::VEC_NONE) {
#ifdef DEBUG_GP
                cout << " //** reading from point " << gp.make_str() << " as scalar.\n";
#endif
                code_str = read_from_scalar_point(os, gp, &_vec2elem_global_map);
            }

            // Non-scalar but non-vectorizable point?
            else if (gp.get_vec_type() == VarPoint::VEC_PARTIAL) {
#ifdef DEBUG_GP
                cout << " //** reading from point " << gp.make_str() << " as partially vectorized.\n";
#endif
                code_str = print_partial_vec_read(os, gp);
            }

            // Everything below this should be VEC_FULL.

            // An aligned vector block?
            else if (_vv._aligned_vecs.count(gp)) {
#ifdef DEBUG_GP
                cout << " //** reading from point " << gp.make_str() << " as fully vectorized and aligned.\n";
#endif
                code_str = print_aligned_vec_read(os, gp);
            }

            // Unaligned loads allowed?
            else if (_settings._allow_unaligned_loads) {
#ifdef DEBUG_GP
                cout << " //** reading from point " << gp.make_str() << " as fully vectorized and unaligned.\n";
#endif
                code_str = print_unaligned_vec_read(os, gp);
            }

            // Need to construct an unaligned vector block?
            else if (_vv._vblk2elem_lists.count(gp)) {
#ifdef DEBUG_GP
                cout << " //** reading from point " << gp.make_str() << " as fully vectorized and unaligned.\n";
#endif

                // make sure prerequisites exist by recursing.
                auto avbs = _vv._vblk2avblks[gp];
                for (auto pi = avbs.begin(); pi != avbs.end(); pi++) {
                    auto& p = *pi;
                    read_from_point(os, p);
                }

                // output this construction.
                code_str = print_unaligned_vec(os, gp);
            }

            else {
                THROW_YASK_EXCEPTION("Internal error: type unknown for point " + gp.make_str());
            }
        }

        // Remember this point and return it.
        if (code_str.length())
            save_point_var(gp, code_str);
        return code_str;
    }

    // Print any immediate memory writes to 'os'.
    // Return code to update a vector of var points or null string
    // if all writes were printed.
    string CppVecPrintHelper::write_to_point(ostream& os, const VarPoint& gp,
                                             const string& val) {

        // Use vec write.
        // NB: currently, all eqs must be vectorizable on LHS,
        // so we only need to handle vectorized writes.
        // TODO: relax this restriction.
        print_aligned_vec_write(os, gp, val);

        return "";              // no returned expression.
    }
    
    // Print aligned memory read.
    // This should be the most common type of var read.
    string CppVecPrintHelper::print_aligned_vec_read(ostream& os, const VarPoint& gp) {

        // Make comment and function call.
        print_point_comment(os, gp, "Read aligned vector");
        string mv_name = make_var_name();

        // Do we have a pointer to the base?
        auto* p = lookup_base_point_ptr(gp);
        if (p) {

            // Ptr expression.
            auto ofs_str = get_ptr_offset(os, gp);
            string ptr_expr = *p + " + (" + ofs_str + ")";

            // Check addr.
            auto rpn = make_point_call(os, gp, "get_vec_ptr_norm", "", "", true);
            os << _line_prefix << "host_assert(" <<
                ptr_expr << " == " << rpn << ")" << _line_suffix;

            // Output load.
            // We don't use masked loads because several aligned loads might
            // be combined to make a simulated unaligned load.
            os << _line_prefix << get_var_type() << " " << mv_name << _line_suffix;
            os << _line_prefix << mv_name << ".load_from(" << ptr_expr << ")" <<
                _line_suffix;

        } else {

            // If no pointer, use function call.
            auto rvn = make_point_call(os, gp, "read_vec_norm", "", "", true);
            os << _line_prefix << get_var_type() << " " << mv_name << " = " <<
                rvn << _line_suffix;
        }
            
        return mv_name;
    }

    // Read from a single point.
    // 'gp' may or may not allow vec read.
    // Return code for read.
    string CppVecPrintHelper::read_from_scalar_point(ostream& os, const VarPoint& gp,
                                                     const VarMap* var_map) {
        assert(var_map);
        auto* var = gp._get_var();
        assert(!var->is_foldable()); // Assume all scalar reads from scalar vars.
        
        ///// TODO: use pointer when avail /////

        return make_point_call(os, gp, "read_elem", "", "", false, var_map);
    }

    // Read from multiple points that are not vectorized.
    // Return var name.
    string CppVecPrintHelper::print_partial_vec_read(ostream& os, const VarPoint& gp) {
        print_point_comment(os, gp, "Construct folded vector from non-folded data");

        // Make a vec var.
        string mv_name = make_var_name();
        os << _line_prefix << get_var_type() << " " << mv_name << _line_suffix;

        // Loop through all points in the vector fold.
        get_fold().visit_all_points([&](const IntTuple& vec_point,
                                        size_t pelem){

                // Example: vec_point contains x=0, y=2, z=1, where each val
                // is the offset in the given fold dim.  We want to map
                // x=>x_elem, y=>(y_elem+2), z=>(z_elem+1) in var-point
                // index args.
                VarMap v_map;
                for (auto& dim : vec_point) {
                    auto& dname = dim._get_name();
                    int dofs = dim.get_val();

                    auto& ename = _vec2elem_global_map.at(dname);
                    if (dofs == 0)
                        v_map[dname] = ename;
                    else {
                        v_map[dname] = "(" + ename + "+" + to_string(dofs) + ")";
                    }
                }

                // Read or reuse.
                string stmt = read_from_scalar_point(os, gp, &v_map);
                auto* varname = lookup_elem_var(stmt);
                if (!varname) {

                    // Read val into a new scalar var.
                    string vname = make_var_name();
                    os << _line_prefix << "real_t " << vname <<
                        " = " << stmt << _line_suffix;
                    varname = save_elem_var(stmt, vname);
                }

                // Output translated expression for this element.
                os << _line_prefix << mv_name << "[" << pelem << "] = " <<
                    *varname << "; // for offset " << vec_point.make_dim_val_str() <<
                    _line_suffix;

                return true;
            }); // end of lambda.
        return mv_name;
    }

    // Print unaliged memory read.
    // Assumes this results in same values as print_unaligned_vec().
    // TODO: use pointer.
    string CppVecPrintHelper::print_unaligned_vec_read(ostream& os, const VarPoint& gp) {
        print_point_comment(os, gp, "Read unaligned");
        os << " // NOTICE: Assumes constituent vectors are consecutive in memory!" << endl;

        // Make a var.
        string mv_name = make_var_name();
        os << _line_prefix << get_var_type() << " " << mv_name << _line_suffix;
        auto vp = make_point_call(os, gp, "get_elem_ptr", "", "true", false);

        // Read memory.
        os << _line_prefix << mv_name <<
            ".load_unaligned_from((const " << get_var_type() << "*)" << vp << ")" << _line_suffix;
        return mv_name;
    }

    // Print aligned memory write.
    void CppVecPrintHelper::print_aligned_vec_write(ostream& os, const VarPoint& gp,
                                                    const string& val) {

        print_point_comment(os, gp, "Write aligned vector");
        
        // Got a pointer to the base addr?
        auto* p = lookup_base_point_ptr(gp);
        if (p) {

            // Offset.
            auto ofs_str = get_ptr_offset(os, gp);
            auto ptr_expr = *p + " + (" + ofs_str + ")";

            // Check addr.
            auto rpn = make_point_call(os, gp, "get_vec_ptr_norm", "", "", true);
            os << _line_prefix << "host_assert(" <<
                ptr_expr << " == " << rpn << ")" << _line_suffix;

            // Output store.
            os << _line_prefix << val;
            if (_write_mask.length())
                os << ".store_to_masked(" << ptr_expr << ", " << _write_mask << ")";
            else
                os << ".store_to(" << ptr_expr << ")";
            os << _line_suffix;
        }

        else {

            // If no pointer, use function call.
            string fn = _write_mask.length() ? "write_vec_norm_masked" : "write_vec_norm";
            auto vn = make_point_call(os, gp, fn, val, _write_mask, true);
            os << _line_prefix << vn << _line_suffix;
        }
    }

    // Print conversion from memory vars to point var gp if needed.
    // This calls print_unaligned_vec_ctor(), which can be overloaded
    // by derived classes.
    string CppVecPrintHelper::print_unaligned_vec(ostream& os, const VarPoint& gp) {
        print_point_comment(os, gp, "Construct unaligned");

        // Declare var.
        string pv_name = make_var_name();
        os << _line_prefix << get_var_type() << " " << pv_name << _line_suffix;

        // Contruct it.
        print_unaligned_vec_ctor(os, gp, pv_name);
        return pv_name;
    }

    // Print per-element construction for one point var pv_name from elems.
    void CppVecPrintHelper::print_unaligned_vec_simple(ostream& os, const VarPoint& gp,
                                                    const string& pv_name, string line_prefix,
                                                    const set<size_t>* done_elems) {

        // just assign each element in vector separately.
        auto& elems = _vv._vblk2elem_lists[gp];
        assert(elems.size() > 0);
        for (size_t pelem = 0; pelem < elems.size(); pelem++) {

            // skip if done.
            if (done_elems && done_elems->count(pelem))
                continue;

            // one vector element from gp.
            auto& ve = elems[pelem];

            // Look up existing input var.
            assert(_vec_vars.count(ve._vec));
            string mv_name = _vec_vars[ve._vec];

            // which element?
            int aligned_elem = ve._offset; // 1-D layout of this element.
            string elem_str = ve._offsets.make_dim_val_offset_str();

            os << line_prefix << pv_name << "[" << pelem << "] = " <<
                mv_name << "[" << aligned_elem << "];  // for " <<
                gp.get_var_name() << "(" << elem_str << ")" << _line_suffix;
        }
    }

    // Print init of element indices.
    // Fill _vec2elem_*_map as side-effect.
    void CppVecPrintHelper::print_elem_indices(ostream& os) {
        auto& fold = get_fold();
        os << "\n // Element indices derived from vector indices.\n";
        int i = 0;
        for (auto& dim : fold) {
            auto& dname = dim._get_name();
            string cap_dname = PrinterBase::all_caps(dname);
            string elname = dname + _elem_suffix_local;
            os << " idx_t " << elname <<
                " = " << dname << " * VLEN_" << cap_dname << ";\n";
            _vec2elem_local_map[dname] = elname;
            string egname = dname + _elem_suffix_global;
            os << " idx_t " << egname <<
                " = core_data->_common_core._rank_domain_offsets[" << i << "] + (" <<
                dname << " * VLEN_" << cap_dname << ");\n";
            _vec2elem_global_map[dname] = egname;
            i++;
        }
    }
    
    // Print loop-invariant values for each VarPoint.
    string CppPreLoopPrintVisitor::visit(VarPoint* gp) {
        assert(gp);

        // Pointer to this var.
        string varp = get_var_ptr(*gp);
        if (!_cvph.is_local_var(varp))
            _os << "\n // Pointer to core of var '" << gp->get_var_name() << "'.\n";
        string var_ptr = _cvph.get_local_var(_os, varp,
                                             CppPrintHelper::_var_ptr_restrict_type);
        
        // Time var for this access, if any.
        auto& dims = _cvph.get_dims();
        string sas = gp->make_step_arg_str(var_ptr, dims);
        if (sas.length()) {
            if (!_cvph.is_local_var(sas))
                _os << "\n // Index for '" << sas << "'.\n";
            _cvph.get_local_var(_os, sas, CppPrintHelper::_step_val_type);
        }

        // Print strides and local offsets for this var.
        _cvph.print_strides(_os, *gp);

        // Make and print a base pointer for this access.
        _cvph.print_base_ptr(_os, *gp);

        // Retrieve prior dependence analysis of this var point.
        auto dep_type = gp->get_var_dep();

        // If invariant, we can load now.
        if (dep_type == VarPoint::DOMAIN_VAR_INVARIANT) {

            // Not already loaded?
            if (!_cvph.lookup_point_var(*gp)) {
                string expr = _ph.read_from_point(_os, *gp);
                string res;
                make_next_temp_var(res, gp) << expr << _ph.get_line_suffix();

                // Save for future use.
                _cvph.save_point_var(*gp, res);
            }
        }
        return "";
    }

} // namespace yask.

