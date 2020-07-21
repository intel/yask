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

////////// Support for C++ scalar and vector-code generation //////////////

#include "Cpp.hpp"

namespace yask {

    /////////// Scalar code /////////////

    // Format a real, preserving precision.
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
        string step_arg_var = get_local_var(os, gp.make_step_arg_str(var_ptr, _dims), _step_val_type);

        string res = var_ptr + "->" + fname + "(";
        if (opt_arg.length())
            res += opt_arg + ", ";
        string args = gp.make_arg_str();
        res += "{" + args + "}, " + step_arg_var + ", __LINE__)";
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

    // Read from a single point.
    // Return code for read.
    string CppVecPrintHelper::read_from_scalar_point(ostream& os, const VarPoint& gp,
                                                  const VarMap* v_map) {

        // Use default var-map if not provided.
        if (!v_map)
            v_map = &_vec2elem_map;

        // Get/set local vars.
        string var_ptr = get_local_var(os, get_var_ptr(gp), _var_ptr_restrict_type);
        string step_arg_var = get_local_var(os, gp.make_step_arg_str(var_ptr, _dims),
                                        _step_val_type);

        // Assume that broadcast will be handled automatically by
        // operator overloading in kernel code.
        // Specify that any indices should use element vars.
        string str = var_ptr + "->read_elem(";
        string args = gp.make_arg_str(v_map);
        str += "{" + args + "}, " + step_arg_var + ",__LINE__)";
        return str;
    }

    // Read from multiple points that are not vectorizable.
    // Return var name.
    string CppVecPrintHelper::print_non_vec_read(ostream& os, const VarPoint& gp) {
        print_point_comment(os, gp, "Construct folded vector from non-folded");

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

                    auto& ename = _vec2elem_map.at(dname);
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

    // Print call for a point.
    // This is a utility function used for reads & writes.
    string CppVecPrintHelper::print_vec_point_call(ostream& os,
                                                const VarPoint& gp,
                                                const string& func_name,
                                                const string& first_arg,
                                                const string& last_arg,
                                                bool is_norm) {

        // Get/set local vars.
        string var_ptr = get_local_var(os, get_var_ptr(gp), CppPrintHelper::_var_ptr_restrict_type);
        string step_arg_var = get_local_var(os, gp.make_step_arg_str(var_ptr, _dims),
                                        CppPrintHelper::_step_val_type);

        string res = var_ptr + "->" + func_name + "(";
        if (first_arg.length())
            res += first_arg + ", ";
        string args = is_norm ? gp.make_norm_arg_str(_dims) : gp.make_arg_str();
        res += "{" + args + "} ," + step_arg_var;
        if (last_arg.length())
            res += ", " + last_arg;
        res += ")";
        return res;
    }

    // Print code to set pointers of aligned reads.
    void CppVecPrintHelper::print_base_ptrs(ostream& os) {
        const string& idim = _dims._inner_dim;

        // A set for the aligned reads & writes.
        VarPointSet gps;

        // Aligned reads as determined by VecInfoVisitor.
        gps = _vv._aligned_vecs;

        // Writes (assume aligned).
        gps.insert(_vv._vec_writes.begin(), _vv._vec_writes.end());

        // Loop through all aligned read & write points.
        for (auto& gp : gps) {

            // Can we use a pointer?
            if (gp.get_loop_type() != VarPoint::LOOP_OFFSET)
                continue;

            // Make base point (misc & inner-dim indices = 0).
            auto bgp = make_base_point(gp);

            // Not already saved?
            if (!lookup_point_ptr(*bgp)) {

                // Get temp var for ptr.
                string ptr_name = make_var_name();

                // Save for future use.
                save_point_ptr(*bgp, ptr_name);
            }

            // Collect some stats for reads using this ptr.
            if (_vv._aligned_vecs.count(gp)) {
                auto* p = lookup_point_ptr(*bgp);
                assert(p);

                // Get const offsets.
                auto& offsets = gp.get_arg_offsets();

                // Get offset in inner dim.
                // E.g., A(t, x+1, y+4) => 4.
                auto* ofs = offsets.lookup(idim);

                // Remember lowest inner-dim offset from this ptr.
                if (ofs && (!_ptr_ofs_lo.count(*p) || _ptr_ofs_lo[*p] > *ofs))
                    _ptr_ofs_lo[*p] = *ofs;

                // Remember highest one.
                if (ofs && (!_ptr_ofs_hi.count(*p) || _ptr_ofs_hi[*p] < *ofs))
                    _ptr_ofs_hi[*p] = *ofs;
            }
        }

        // Loop through all aligned read & write points.
        set<string> done;
        for (auto& gp : gps) {

            // Make base point (inner-dim index = 0).
            auto bgp = make_base_point(gp);

            // Got a pointer?
            auto* p = lookup_point_ptr(*bgp);
            if (!p)
                continue;

            // Make code for pointer and prefetches.
            if (!done.count(*p)) {

                // Print pointer creation.
                print_point_ptr(os, *p, *bgp);

                // Print prefetch(es) for this ptr if a read.
                if (_vv._aligned_vecs.count(gp))
                    print_prefetches(os, false, *p);

                done.insert(*p);
            }
        }
    }

    // Print prefetches for each base pointer.
    // 'level': cache level.
    // 'ahead': prefetch PF distance ahead instead of up to PF dist.
    // TODO: add handling of misc dims.
    void CppVecPrintHelper::print_prefetches(ostream& os,
                                            bool ahead, string ptr_var) {

        // cluster mult in inner dim.
        const string& idim = _dims._inner_dim;
        string imult = "CMULT_" + PrinterBase::all_caps(idim);

        for (int level = 1; level <= 2; level++) {

            os << "\n // Prefetch to L" << level << " cache if enabled.\n";
                os << _line_prefix << "#if PFD_L" << level << " > 0\n";

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
                        auto bgp = make_base_point(gp);
                        auto* p = lookup_point_ptr(*bgp);
                        if (p && *p == ptr) {

                            // Expression for this offset from inner-dim var.
                            string inner_expr = idim + " + ofs";

                            // Expression for ptr offset at this point.
                            string ofs_expr = get_ptr_offset(gp, inner_expr);
                            print_point_comment(os, gp, "Prefetch for ");

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

    // Make base point (misc & inner-dim indices = 0).
    var_point_ptr CppVecPrintHelper::make_base_point(const VarPoint& gp) {
        var_point_ptr bgp = gp.clone_var_point();
        for (auto& dim : gp.get_dims()) {
            auto& dname = dim->_get_name();
            auto type = dim->get_type();

            // Set inner domain index to 0.
            if (dname == get_dims()._inner_dim) {
                IntScalar idi(dname, 0);
                bgp->set_arg_const(idi);
            }

            // Set misc indices to their min value if they are inside
            // inner domain dim.
            else if (_settings._inner_misc && type == MISC_INDEX) {
                auto* var = gp._get_var();
                auto min_val = var->get_min_indices()[dname];
                IntScalar idi(dname, min_val);
                bgp->set_arg_const(idi);
            }
        }
        return bgp;
    }
    
    // Print code to set ptr_name to gp.
    void CppVecPrintHelper::print_point_ptr(ostream& os, const string& ptr_name,
                                          const VarPoint& gp) {
        print_point_comment(os, gp, "Calculate pointer to ");

        // Get pointer to vector using normalized indices.
        // Ignore out-of-range errors because we might get a base pointer to an
        // element before the allocated range.
        auto vp = print_vec_point_call(os, gp, "get_vec_ptr_norm", "", "false", true);

        // Ptr will be unique if:
        // - Var doesn't have step dim, or
        // - Var doesn't allow dynamic step allocs and the alloc size is one (TODO), or
        // - Var doesn't allow dynamic step allocs and all accesses are via
        //   offsets from the step dim w/compatible offsets (TODO).
        // TODO: must also share pointers during code gen in last 2 cases.
        auto* var = gp._get_var();
        bool is_unique = false;
        //bool is_unique = (var->get_step_dim() == nullptr);
        string type = is_unique ? _var_ptr_restrict_type : _var_ptr_type;

        // Print type and value.
        os << _line_prefix << type << " " << ptr_name << " = " << vp << _line_suffix;
    }

    // Get expression for offset of 'gp' from base pointer.  Base pointer
    // points to vector with outer-dims == same values as in 'gp', inner-dim
    // == 0 and misc dims == their min value.
    string CppVecPrintHelper::get_ptr_offset(const VarPoint& gp, const string& inner_expr) {
        auto* var = gp._get_var();

        // Need to create an expression for inner-dim
        // and misc indices offsets.
                
        // Start with offset in inner-dim direction.
        // This must the dim that appears before the misc dims
        // in the var layout.
        string idim = _dims._inner_dim;
        string ofs_str = "(";
        if (inner_expr.length())
            ofs_str += inner_expr;
        else
            ofs_str += gp.make_norm_arg_str(idim, _dims);
        ofs_str += ")";

        // Misc indices if they are inside inner-dim.
        if (_settings._inner_misc) {
            for (int i = 0; i < var->get_num_dims(); i++) {
                auto& dimi = gp.get_dims().at(i);
                auto& dni = dimi->_get_name();
                auto typei = dimi->get_type();
                if (typei == MISC_INDEX) {

                    // Mult by size of remaining misc dims.
                    for (int j = i; j < var->get_num_dims(); j++) {
                        auto& dimj = gp.get_dims().at(j);
                        auto& dnj = dimj->_get_name();
                        auto typej = dimj->get_type();
                        if (typej == MISC_INDEX) {
                            auto min_idx = var->get_min_indices()[dnj];
                            auto max_idx = var->get_max_indices()[dnj];
                            ofs_str += " * (" + to_string(max_idx) +
                                " - " + to_string(min_idx) + " + 1)";
                        }
                    }
                        
                    // Add offset of this misc value, which must be const.
                    auto min_val = var->get_min_indices()[dni];
                    auto val = gp.get_arg_consts()[dni];
                    ofs_str += " + (" + to_string(val) + " - " +
                        to_string(min_val) + ")";
                }
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

        // Can we use a vec pointer?
        // Read must be aligned, and we must have a pointer.
        else if (_vv._aligned_vecs.count(gp) &&
                 gp.get_vec_type() == VarPoint::VEC_FULL &&
                 gp.get_loop_type() == VarPoint::LOOP_OFFSET) {

            // Got a pointer to the base addr?
            auto bgp = make_base_point(gp);
            auto* p = lookup_point_ptr(*bgp);
            if (p) {
#ifdef DEBUG_GP
                cout << " //** reading from point " << gp.make_str() << " using pointer.\n";
#endif

                // Output read using base addr.
                auto ofs_str = get_ptr_offset(gp);
                print_point_comment(os, gp, "Read aligned");
                code_str = make_var_name();
                os << _line_prefix << get_var_type() << " " << code_str << " = " <<
                    *p << "[" << ofs_str << "]" << _line_suffix;
            }
        }

        // If not done, continue based on type of vectorization.
        if (!code_str.length()) {

            // Scalar GP?
            if (gp.get_vec_type() == VarPoint::VEC_NONE) {
#ifdef DEBUG_GP
                cout << " //** reading from point " << gp.make_str() << " as scalar.\n";
#endif
                code_str = read_from_scalar_point(os, gp);
            }

            // Non-scalar but non-vectorizable GP?
            else if (gp.get_vec_type() == VarPoint::VEC_PARTIAL) {
#ifdef DEBUG_GP
                cout << " //** reading from point " << gp.make_str() << " as partially vectorized.\n";
#endif
                code_str = print_non_vec_read(os, gp);
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

        // Can we use a pointer?
        if (gp.get_loop_type() == VarPoint::LOOP_OFFSET) {

            // Got a pointer to the base addr?
            auto bgp = make_base_point(gp);
            auto* p = lookup_point_ptr(*bgp);
            if (p) {

                // Offset.
                auto ofs_str = get_ptr_offset(gp);

                // Output write using base addr.
                print_point_comment(os, gp, "Write aligned");

                if (_use_masked_writes)
                    os << _line_prefix << val << ".store_to_masked(" << *p << " + (" <<
                        ofs_str << "), write_mask)" << _line_suffix;
                else
                    os << _line_prefix << val << ".store_to(" << *p << " + (" <<
                        ofs_str << "))" << _line_suffix;

                return "";
            }
        }

        // If no pointer, use vec write.
        // NB: currently, all eqs must be vectorizable on LHS,
        // so we only need to handle vectorized writes.
        // TODO: relax this restriction.
        print_aligned_vec_write(os, gp, val);

        return "";              // no returned expression.
    }


    // Print aligned memory read.
    string CppVecPrintHelper::print_aligned_vec_read(ostream& os, const VarPoint& gp) {

        print_point_comment(os, gp, "Read aligned");
        auto rvn = print_vec_point_call(os, gp, "read_vec_norm", "", "__LINE__", true);

        // Read memory.
        string mv_name = make_var_name();
        os << _line_prefix << get_var_type() << " " << mv_name << " = " << rvn << _line_suffix;
        return mv_name;
    }

    // Print unaliged memory read.
    // Assumes this results in same values as print_unaligned_vec().
    string CppVecPrintHelper::print_unaligned_vec_read(ostream& os, const VarPoint& gp) {
        print_point_comment(os, gp, "Read unaligned");
        os << " // NOTICE: Assumes constituent vectors are consecutive in memory!" << endl;

        // Make a var.
        string mv_name = make_var_name();
        os << _line_prefix << get_var_type() << " " << mv_name << _line_suffix;
        auto vp = print_vec_point_call(os, gp, "get_elem_ptr", "", "true", false);

        // Read memory.
        os << _line_prefix << mv_name <<
            ".load_unaligned_from((const " << get_var_type() << "*)" << vp << ")" << _line_suffix;
        return mv_name;
    }

    // Print aligned memory write.
    string CppVecPrintHelper::print_aligned_vec_write(ostream& os, const VarPoint& gp,
                                                   const string& val) {
        print_point_comment(os, gp, "Write aligned");
        auto vn = print_vec_point_call(os, gp, "write_vec_norm_masked", val, "write_mask, __LINE__", true);
        // without mask: auto vn = print_vec_point_call(os, gp, "write_vec_norm", val, "__LINE__", true);

        // Write temp var to memory.
        os << vn;
        return val;
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
    // Fill _vec2elem_map as side-effect.
    void CppVecPrintHelper::print_elem_indices(ostream& os) {
        auto& fold = get_fold();
        os << "\n // Element indices derived from vector indices.\n";
        int i = 0;
        for (auto& dim : fold) {
            auto& dname = dim._get_name();
            string ename = dname + _elem_suffix;
            string cap_dname = PrinterBase::all_caps(dname);
            os << " idx_t " << ename <<
                " = _core_p->_common_core._rank_domain_offsets[" << i << "] + (" <<
                dname << " * VLEN_" << cap_dname << ");\n";
            _vec2elem_map[dname] = ename;
            i++;
        }
    }

    // Print invariant var-access vars for non-time loop(s).
    string CppStepVarPrintVisitor::visit(VarPoint* gp) {

        // Pointer to var.
        string var_ptr = _cvph.get_local_var(_os, get_var_ptr(*gp), CppPrintHelper::_var_ptr_restrict_type);
        
        // Time var.
        auto& dims = _cvph.get_dims();
        _cvph.get_local_var(_os, gp->make_step_arg_str(var_ptr, dims),
                          CppPrintHelper::_step_val_type);
        return "";
    }

    // Print invariant var-access vars for an inner loop.
    string CppLoopVarPrintVisitor::visit(VarPoint* gp) {

        // Retrieve prior analysis of this var point.
        auto loop_type = gp->get_loop_type();

        // If invariant, we can load now.
        if (loop_type == VarPoint::LOOP_INVARIANT) {

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

