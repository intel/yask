/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2022, Intel Corporation

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
        string var_ptr = get_local_var(os, get_var_ptr(gp), _var_ptr_type, "expr");
        string sas = gp.make_step_arg_str(var_ptr, _dims);
        string step_arg = sas.length() ? get_local_var(os, sas, _step_val_type, "step") : "0";

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
    string CppVecPrintHelper::make_point_call_vec(ostream& os,
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
                                       CppPrintHelper::_var_ptr_type, "expr");
        string sas = gp.make_step_arg_str(var_ptr, _dims);
        string step_arg = sas.length() ?
            get_local_var(os, sas, CppPrintHelper::_step_val_type, "step") : "0";

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
    
    // Make base point:
    //  domain indices = local-offset;
    //  misc indices = min-val (local-offset);
    //  other indices = those from 'gp'.
    var_point_ptr CppVecPrintHelper::make_var_base_point(const VarPoint& gp) {
        var_point_ptr bgp = gp.clone_var_point();
        auto* var = bgp->_get_var();
        assert(var);
        for (auto& dim : bgp->get_dims()) {
            auto& dname = dim->_get_name();
            auto type = dim->get_type();

            if (type == DOMAIN_INDEX || type == MISC_INDEX) {
                auto* lofs = lookup_offset(*var, dname);
                assert(lofs);
                bgp->set_arg_expr(dname, *lofs);
            }
        }
        return bgp;
    }

    // Make inner-loop base point:
    //  domain dim offset = 0;
    //  misc indices = min-val (local-offset);
    //  other indices = those from 'gp'.
    var_point_ptr CppVecPrintHelper::make_inner_loop_base_point(const VarPoint& gp) {
        var_point_ptr bgp = gp.clone_var_point();
        for (auto& dim : gp.get_dims()) {
            auto& dname = dim->_get_name();
            auto type = dim->get_type();
            bool use_domain = (type == DOMAIN_INDEX) &&
                (!_settings._use_many_ptrs || dname == _dims._inner_layout_dim);
            bool use_misc = type == MISC_INDEX;

            // Set domain dims to current index only,
            // i.e., no offset.
            if (use_domain) {
                IntScalar idi(dname, 0);
                bgp->set_arg_offset(idi);
            }

            // Set misc indices to their min value.
            else if (use_misc) {
                auto* var = gp._get_var();
                auto min_val = var->get_min_indices()[dname];
                IntScalar idi(dname, min_val);
                bgp->set_arg_const(idi);
            }
        }
        return bgp;
    }
    
    // Print creation of var-base pointer of 'gp'.
    void CppVecPrintHelper::print_var_base_ptr(ostream& os, const VarPoint& gp) {
        if (!_settings._use_ptrs)
            return;

        // Got a pointer to it already?
        auto* p = lookup_var_base_ptr(gp);
        if (!p) {

            // Make base point (misc & domain indices set to min
            // values). There will be one pointer for every unique
            // var/step-arg combo.
            auto bgp = make_var_base_point(gp);
            auto* var = bgp->_get_var();
            assert(var);
           
            // Make and save ptr var for future use.
            string ptr_name = make_var_name(var->_get_name() + "_var_base_ptr");
            _var_base_ptrs[*bgp] = ptr_name;

            // Print pointer definition.
            print_point_comment(os, *bgp, "Create var-base pointer");

            // Get pointer to var using normalized indices.
            // Ignore out-of-range errors because we might get a base pointer to an
            // element before the allocated range.
            // TODO: is this still true with local offsets?
            bool folded = var->is_foldable();
            auto vp = folded ?
                make_point_call_vec(os, *bgp, "get_vec_ptr_norm", "", "false", true) :
                make_point_call_vec(os, *bgp, "get_elem_ptr_local", "", "false", false, &_vec2elem_local_map);

            // Ptr should provide unique access if all accesses are through pointers.
            // TODO: check for reusing time-slots, e.g., p(t+1) aliased to p(t-1).
            // TODO: check for non-ptr accesses via read/write calls.
            bool is_unique = false; // !_settings._allow_unaligned_loads;
            string type = is_unique ? _var_ptr_restrict_type : _var_ptr_type;

            // Print type and value.
            os << _line_prefix << type << " " << ptr_name << " = " << vp << _line_suffix;
        }
    }

    // Print creation of stride and local-offset vars.
    // Save var names for later use.
    void CppVecPrintHelper::print_strides(ostream& os, const VarPoint& gp) {
        auto* vp = gp._get_var();
        assert(vp);
        auto& var = *vp;
        const auto& vname = var.get_name();

        // Already done this var?
        if (_ptr_ofs.count(vname))
            return;

        // Index-invariant pointer offset.
        string po_var = make_var_name(vname + "_ptr_ofs");
        _ptr_ofs[vname] = po_var;
        string po_expr = "idx_t(0)";
        string po_deco = "const";

        for (int dnum = 0; dnum < var.get_num_dims(); dnum++) {
            auto& dim = var.get_dims().at(dnum);
            const auto& dname = dim->_get_name();
            auto dtype = dim->get_type();
            bool is_step = dtype == STEP_INDEX;
            bool is_misc = dtype == MISC_INDEX;
            bool is_inner = dname == _dims._inner_layout_dim;
            
            auto key = VarDimKey(vname, dname);
            if (!is_step) {
                os << endl << " // Stride for var '" << vname <<
                    "' in dim '" << dname << "'.\n";

                string str_var = make_var_name(vname + "_" + dname + "_stride");
                _strides[key] = str_var;
                assert(lookup_stride(var, dname));

                // Get ptr to var core.
                string var_ptr = get_local_var(os, get_var_ptr(var),
                                               CppPrintHelper::_var_ptr_restrict_type,
                                               vname + "_core");

                // Determine stride.
                // Default is to obtain dynamic value from var.
                string slookup = var_ptr + "->_vec_strides[" + to_string(dnum) + "]";
                string stride = slookup;
                string sdeco = "const";

                // Under certain conditions, strides are known fixed values.
                // Must be last dim or followed only by fixed-size misc dims.
                bool is_fixed = true;
                string fstride;

                // Loop in layout order.
                bool dfound = false;
                for (int j = 0; j < var.get_num_dims(); j++) {
                    auto& dimj = var.get_layout_dims().at(j);
                    auto& dnj = dimj->_get_name();
                    auto typej = dimj->get_type();

                    // If misc, must also use inner-misc setting, which disallows
                    // resizing via APIs.
                    bool is_miscj = (typej == MISC_INDEX) && _settings._inner_misc;
                    
                    // Multply the strides of the dims following the current one.
                    if (dfound) {
                        if (is_miscj) {
                            auto min_idx = var.get_min_indices()[dnj];
                            auto max_idx = var.get_max_indices()[dnj];
                            auto sz = max_idx - min_idx + 1;
                            os << " // Indices for inner-dim '" << dnj << "' range from " <<
                                min_idx << " to " << max_idx << ": " << sz << " value(s).\n";
                            if (fstride.length())
                                fstride += " * ";
                            fstride += to_string(sz);
                        }
                        else {
                            os << " // Size of inner-dim '" << dnj << "' is not known at "
                                "compile-time.\n";
                            is_fixed = false; // Not a fixed value.
                        }
                    }
                    if (dnj == dname)
                        dfound = true;
                }
                if (is_fixed) {
                    sdeco = "constexpr";
                    if (fstride.length())
                        stride = fstride;
                    else
                        stride = "1";
                }

                // Print final assignment.
                os << _line_prefix << sdeco << " idx_t " << str_var << " = " <<
                    stride << _line_suffix;
                if (stride != slookup)
                    os << _line_prefix << "host_assert(" << slookup <<
                        " == " << str_var << ")" << _line_suffix;

                // Offset to be subtracted from index.
                os << endl << " // Index offset for var '" << vname <<
                    "' in dim '" << dname << "'.\n";
                string ofs_var = make_var_name(vname + "_" + dname + "_ofs");
                _offsets[key] = ofs_var;
                assert(lookup_offset(var, dname));
                string ofs_deco = "constexpr";

                string ofs; // Offset value.
                string ofs_expr = var_ptr + "->_local_offsets[" + to_string(dnum) + "]";
                if (var.is_scratch() && dtype == DOMAIN_INDEX) {

                    // Lookup needed for domain dim in scratch var because scratch
                    // vars are "relocated" to location of current block.
                    os << " // Local offset varies because '" << vname <<
                        "' is a scratch var.\n";
                    ofs = ofs_expr;
                    ofs_deco = "const";
                }
                else if (dtype == MISC_INDEX) {

                    // Need min value for misc indices.
                    os << " // Local offset is minimum misc-index.\n";
                    ofs = "idx_t(" + to_string(var.get_min_indices()[dname]) + ")";
                }
                else {
                    os << " // Local offset is zero.\n";
                    ofs = "idx_t(0)";
                }

                // Offset includes local offset and pad.
                if (_settings._use_offsets) {
                    string pad_expr = var_ptr + "->_actl_left_pads[" + to_string(dnum) + "]";
                    ofs_expr += " - " + pad_expr;
                    if (dtype != MISC_INDEX) {
                        os << " // Offset is adjusted by actual allocated padding.\n";
                        ofs += " - " + pad_expr;
                    }
                    ofs_deco = "const";
                }

                os << _line_prefix << ofs_deco << " idx_t " << ofs_var << " = " <<
                    ofs << _line_suffix;
                if (ofs != ofs_expr)
                    os << _line_prefix << "host_assert(" << ofs_expr <<
                        " == " << ofs_var << ")" << _line_suffix;

                // Build total offset expr.
                po_expr += string(" - (") + ofs_var + " * " + str_var + ")";
            }
        }

        os << "\n // Offset from base ptr to 0th position in var '" << vname << "'.\n" <<
            _line_prefix << po_deco << " idx_t " << po_var << " = " <<
            po_expr << _line_suffix;
    }

    // Print creation of inner-loop ptrs.
    // To be used before the inner-loop starts.
    void CppVecPrintHelper::print_inner_loop_prefix(ostream& os) {
        get_point_stats();
        
        // A set for both aligned reads & writes.
        VarPointSet gps = _aligned_reads;
        gps.insert(_vv._vec_writes.begin(), _vv._vec_writes.end());

        // Loop through all aligned read & write points.
        for (auto& gp : gps) {

            // Can we use a loop pointer?
            auto dep_type = gp.get_var_dep();
            if (dep_type != VarPoint::INNER_LOOP_OFFSET)
                continue;
            auto* vp = gp._get_var();
            assert(vp);
            auto& var = *vp;
            const auto& vname = var.get_name();

            // Doesn't already exist?
            if (!lookup_inner_loop_base_ptr(gp)) {
                const auto* vbp = lookup_var_base_ptr(gp);
                if (vbp) {

                    // Make base point (domain offset = 0; inner-misc indices = min-val).
                    auto bgp = make_inner_loop_base_point(gp);
                    
                    // Get temp var for ptr.
                    string ptr_name = make_var_name(vname + "_inner_loop_ptr");
                    
                    // Save for future use.
                    _inner_loop_base_ptrs[*bgp] = ptr_name;
                    
                    // Print pointer creation.
                    auto ofs_expr = get_var_base_ptr_offset(os, *bgp);
                    os << "\n // Pointer to " << bgp->make_str() << " in loop\n";
                    os << _line_prefix << _var_ptr_type << " " << ptr_name << " = " <<
                        *vbp << " + " << ofs_expr << _line_suffix;
                }
            }
        }
    }

    //#define DEBUG_BUFFERS
    
    // Collect some stats on read points.
    // These are used to create buffers and prefetches.
    void CppVecPrintHelper::get_point_stats() {

        // Done if there are at least as many reads as original.
        if (_aligned_reads.size() >= _vv._aligned_vecs.size())
            return;
        
        const string& ildim = _settings._inner_loop_dim;
        const string& sdim = _dims._step_dim;
        
        // Loop through all aligned read points.
        for (auto& gp : _vv._aligned_vecs) {
            Var* vp = const_cast<Var*>(gp._get_var()); // Need to modify var.
            assert(vp);
            auto& var = *vp;
            const auto& vname = var.get_name();

            #ifdef DEBUG_BUFFERS
            cout << "*** Getting stats for " << gp.make_str() << endl;
            #endif
            
            // Add to read set.
            _aligned_reads.insert(gp);

            // Get const offsets for this point.
            auto& offsets = gp.get_arg_offsets();

            // Get offset in step dim, if any.
            auto* sofs = offsets.lookup(sdim);

            // Is there also a write to this var that might overwrite
            // a read at this step-dim offset?
            // This would be true only with immediate replacement (writeback)
            // optimization.
            bool is_write = false;
            if (sofs) {

                auto sdi = var.get_step_dim_info();
                if (sdi.writeback_ofs.count(_stage_name) &&
                    sdi.writeback_ofs.at(_stage_name) == *sofs) {
                    is_write = true;
                    #ifdef DEBUG_BUFFERS
                    cout << "** Found writeback to " << vname << " over ofs " << *sofs << endl;
                    #endif
                }
            }

            // Get offset in inner-loop dim.
            // E.g., A(t, x+1, y+4, z-2) => 4 if ildim = 'y'.
            auto* ofs = offsets.lookup(ildim);
            if (ofs) {

                // Vec offset.
                auto vofs = *ofs / _dims._fold[ildim];

                // Make a copy of this point w/inner-loop index=0.
                // E.g., A(t, x+1, y+4, z-2, 5) => A(t, x+1, y, z-2, 5) if ildim = 'y'.
                auto key = gp.clone_var_point();
                IntScalar idi(ildim, 0);
                key->set_arg_offset(idi);

                // Remember key for this point.
                _inner_loop_key[gp] = key;

                // Remember lowest inner-loop dim offset from this key.
                if (!_pt_inner_loop_lo.count(*key) || _pt_inner_loop_lo.at(*key) > vofs)
                    _pt_inner_loop_lo[*key] = vofs;
                auto lo = _pt_inner_loop_lo.at(*key);
                
                // Remember highest one.
                if (!_pt_inner_loop_hi.count(*key) || _pt_inner_loop_hi.at(*key) < vofs)
                    _pt_inner_loop_hi[*key] = vofs;
                auto hi = _pt_inner_loop_hi.at(*key);

                // Need a buffer? (This will change as new points are
                // discovered.)  Length will cover range of vecs needed.
                // The num of vecs stepped in the inner loop is subtracted
                // because we don't need to put the vecs read in the current
                // loop iteration in the buffer (until it's shifted at the
                // end of the loop.)  Then, the length may then be increased
                // if reading ahead unless we're also writing back, in which
                // case read-ahead can't be used.
                auto len = hi - lo + 1;
                len -= _inner_loop_vec_step;
                #ifdef DEBUG_BUFFERS
                cout << "*** Buffer for " << key->make_str() <<
                    " has non-read-ahead length " << len << endl;
                #endif
                auto mbl = max(_settings._min_buffer_len, 1);

                // Add read-ahead if requested and allowed.
                auto rad = _settings._read_ahead_dist;
                auto ralv = _inner_loop_vec_step * rad;
                auto rale = _inner_loop_elem_step * rad;
                if (rad > 0 && !is_write && (len + ralv) >= mbl) {
                    #ifdef DEBUG_BUFFERS
                    cout << " *** Adding " << ralv << " vecs to buffer for read-ahead\n";
                    #endif

                    // Add more read points to read set.
                    // These may not be in the original set because they are
                    // for reading ahead.
                    // If some already exist, it will not hurt to re-add them.
                    auto ofs = lo + len + 1;
                    for (int i = 0; i < ralv; i++, ofs++) {
                        auto rap = key->clone_var_point();
                        auto eofs = ofs * _dims._fold[ildim];
                        IntScalar idi(ildim, eofs); // At end of buffer.
                        rap->set_arg_offset(idi);
                        #ifdef DEBUG_READ_AHEAD
                        cout << "  *** Adding read point " << rap->make_str() << endl;
                        #endif
                        _aligned_reads.insert(*rap); // Save new read point.
                        _inner_loop_key[*rap] = key; // Save its key.
                    }

                    // Increase buf len.
                    len += ralv;

                    // Increase var allocation for read-ahead (in elements,
                    // not vecs).  TODO: be more accurate about when to
                    // increase pad; this assumes it extends beyond halo
                    // region.
                    var.update_read_ahead_pad(rale);
                }

                // Remember buf len using key if above threshold.
                if (len >= mbl)
                    _pt_buf_len[*key] = len;
            }
        }
    }

    // Print all aligned loads.
    void CppVecPrintHelper::print_early_loads(ostream& os) {
        get_point_stats();

        os << "\n // Issuing all aligned loads early (before needed).\n";

        // Loop through all aligned read points.
        // TODO: ignore points in buffer.
        for (auto& gp : _aligned_reads) {
            read_from_point(os, gp);
        }
        os << "\n // Done issuing all aligned loads early.\n";
    }

    // Print buffer-code for each inner-loop base pointer.
    // 'in_loop': just shift and load last one.
    void CppVecPrintHelper::print_buffer_code(ostream& os, bool in_loop) {
        get_point_stats();

        set<VarPoint> done;
        const string& ildim = _settings._inner_loop_dim;
        
        // Loop through all aligned read points.
        // TODO: can we just loop thru _inner_loop_key?
        for (auto& gp : _aligned_reads) {
            if (_inner_loop_key.count(gp) == 0)
                continue;

            // Only need buffer for unique point along inner-loop.
            auto key = _inner_loop_key[gp];
            if (done.count(*key))
                continue;

            // Need a buffer?
            if (_pt_buf_len.count(*key) == 0)
                continue;
            
            auto lo = _pt_inner_loop_lo.at(*key);
            auto len = _pt_buf_len[*key];
            auto end = lo + len;
            auto* vp = gp._get_var();
            assert(vp);
            auto& var = *vp;
            const auto& vname = var.get_name();

            int start_ofs, stop_ofs, start_load;
            string bname;

            // Before end of loop.
            if (in_loop) {
                os << "\n // Update buffer for " << key->make_str() << endl;
                os << _line_prefix << "{\n";
                assert(_pt_buf_name.count(*key));
                bname = _pt_buf_name.at(*key);
                for (int i = 0; i < len - _inner_loop_vec_step; i++)
                    os << _line_prefix << bname << "[" << i << "] = " <<
                        bname << "[" << (i + _inner_loop_vec_step) << "]" << _line_suffix;
                start_ofs = end;
                stop_ofs = end + _inner_loop_vec_step;
                start_load = max(len - _inner_loop_vec_step, 0);
            }

            // Before start of loop.
            else {
                bname = make_var_name(vname + "_buf");
                os << "\n // Buffer for " << key->make_str() << " with " << ildim << " vector ";
                if (len == 1)
                    os << "offset " << lo << "\n";
                else
                    os << "offsets in [" << lo << "..." << (end-1) << "]\n";
                os << _line_prefix << _var_type << " " << bname << "[" << len << "];\n";
                os << _line_prefix << "{\n";
                start_ofs = lo;
                stop_ofs = end;
                start_load = 0;
            }

            // Load the buffer.
            int i = start_load;
            for (int vofs = start_ofs; vofs < stop_ofs && i < len; vofs++, i++) {
                auto eofs = vofs * _dims._fold[ildim]; // Vector ofs.

                // Make pt w/needed offset.
                auto ogp = gp.clone_var_point();
                const string& ildim = _settings._inner_loop_dim; // ofs dim.
                IntScalar idi(ildim, eofs);
                ogp->set_arg_offset(idi);

                // Get value at pt.
                string res;
                if (_reuse_vars && _vec_vars.count(*ogp))
                    res = _vec_vars[*ogp];
                else
                    res = print_aligned_vec_read(os, *ogp);

                // Save in buf.
                os << _line_prefix << bname << "[" << i << "] = " << res << _line_suffix;
            }
            os << _line_prefix << "} // Setting " << bname << "\n";
                
            if (!in_loop)
                _pt_buf_name[*key] = bname;
            done.insert(*key);
        }
    }
    
    // Print prefetches for each inner-loop base pointer.
    // 'in_loop' == 'true': prefetch at end of loop; otherwise before loop.
    void CppVecPrintHelper::print_prefetches(ostream& os, bool in_loop) {
        get_point_stats();

        // Not currently prefetching anything before loop starts.
        if (!in_loop)
            return;
        
        const string& ildim = _settings._inner_loop_dim;
        auto& imult = _inner_loop_vec_step;

        // 'level': cache level.
        for (int level = 1; level <= 2; level++) {

            // Distance.
            if (!_settings._prefetch_dists.count(level))
                continue;
            auto pfd = _settings._prefetch_dists.at(level);
            if (pfd < 1)
                continue;

            os << "\n // Prefetch " << pfd << " iteration(s) ahead to L" <<
                level << " cache.\n" <<
                _line_prefix << "{\n";

            // Loop thru inner-loop stats.
            for (auto i : _pt_inner_loop_hi) {
                auto& key = i.first;
                auto hi = i.second; // Furthest vec read at offset in key.

                // Pts in vec.
                int start_ofs = hi + (pfd - 1) * _inner_loop_vec_step + 1;
                int stop_ofs = start_ofs + _inner_loop_vec_step;
                for (int vofs = start_ofs; vofs < stop_ofs; vofs++) {
                    auto eofs = vofs * _dims._fold[ildim]; // Vector ofs.

                    // Make pt w/needed offset.
                    auto ogp = key.clone_var_point();
                    const string& ildim = _settings._inner_loop_dim; // ofs dim.
                    IntScalar idi(ildim, eofs);
                    ogp->set_arg_offset(idi);

                    // Get ptr to it.
                    auto* p = lookup_inner_loop_base_ptr(*ogp);
                    if (p) {
                        string ptr_expr = *p;
                        string ptr_var = ptr_expr;
                        auto ofs_str = get_inner_loop_ptr_offset(os, *ogp);
                        if (ofs_str.length()) {
                            ptr_expr += " + (" + ofs_str + ")";
                            ptr_var = make_var_name("vec_ptr");
                            os << _line_prefix << CppPrintHelper::_var_ptr_type << " " << ptr_var <<
                                " = " << ptr_expr << _line_suffix;
                        }

                        // Insert prefetch.
                        os << _line_prefix << "  prefetch<L" << level << "_HINT>(" << ptr_var <<
                            ")" << _line_suffix;
                    }

                    // TODO: handle case w/o ptr.
                }
            }
            os << _line_prefix << "} // L" << level << " prefetching\n";
        } // levels.
    }

    // print increments of indices & pointers.
    void CppVecPrintHelper::print_end_inner_loop(ostream& os) {
        get_point_stats();

        auto& ild = _settings._inner_loop_dim;
        os << "\n // Increment indices and pointers.\n" <<
            _line_prefix << ild << " += " <<
            _inner_loop_vec_step << _line_suffix <<

            _line_prefix << get_local_elem_index(ild) << " += " <<
            _inner_loop_elem_step << _line_suffix <<

            _line_prefix << get_global_elem_index(ild) << " += " <<
            _inner_loop_elem_step << _line_suffix;
        
        for (auto& i : _inner_loop_base_ptrs) {
            auto& vp = i.first;
            auto& ptr = i.second;
            auto* stride = lookup_stride(*vp._get_var(), _settings._inner_loop_dim);
            assert(stride);
            os << _line_prefix << ptr << " += " <<
               _inner_loop_vec_step << " * " << *stride << _line_suffix;
        }
    }
    
    // Get expression for offset of 'gp' from var-base pointer.
    string CppVecPrintHelper::get_var_base_ptr_offset(ostream& os,
                                                      const VarPoint& gp,
                                                      const VarMap* var_map) {
        auto* var = gp._get_var();
        assert(var);
        auto vname = var->get_name();
        string ofs_str;
        int nterms = 0;

        // Const offset.
        if (_ptr_ofs.count(vname)) {
            ofs_str += string("(") + _ptr_ofs[vname] + ")";
            nterms++;
        }

        // Construct the point-specific linear offset by adding the products
        // of each index with the var's stride in that dim.
        for (int i = 0; i < var->get_num_dims(); i++) {

            // Access in layout order.
            auto& dimi = gp.get_layout_dims().at(i);
            auto typei = dimi->get_type();
            bool is_step = typei == STEP_INDEX;

            // There is a separate pointer for each value of
            // the step index, so we don't need to include
            // that index in the offset calculation.
            if (!is_step) {
                string dni = dimi->_get_name();

                // Construct offset in this dim.
                string nas = (gp.get_vec_type() == VarPoint::VEC_FULL) ?
                    gp.make_norm_arg_str(dni, _dims, var_map) :
                    gp.make_arg_str(dni, var_map);

                // Get stride in this dim.
                auto* stride = lookup_stride(*var, dni);
                assert(stride);

                // Mult & add to offset expression.
                if (nterms)
                    ofs_str += " + ";
                ofs_str += "((" + nas + ") * (" + *stride + "))";
                nterms++;
            }
        }

        return ofs_str;
    }

    // Get expression for offset of 'gp' from inner-loop base pointer.  Base
    // pointer points to vector with domain dim w/no offset or
    // same values as in 'gp', and misc dims == their min value.
    // Return empty string if no offset.
    string CppVecPrintHelper::get_inner_loop_ptr_offset(ostream& os,
                                                        const VarPoint& gp,
                                                        const VarMap* var_map,
                                                        const string& inner_expr) {
        auto* var = gp._get_var();
        assert(var);
        auto vname = var->get_name();
        string ofs_str;
        int nterms = 0;

        // Construct the point-specific linear offset by adding the products
        // of each index with the var's stride in that dim.
        for (int i = 0; i < var->get_num_dims(); i++) {
            auto& dimi = gp.get_layout_dims().at(i);
            auto dname = dimi->_get_name();
            auto type = dimi->get_type();
            bool use_domain = (type == DOMAIN_INDEX) &&
                (!_settings._use_many_ptrs || dname == _dims._inner_layout_dim);
            bool use_misc = type == MISC_INDEX;

            // Need to create an expression for offsets.
            if (use_domain || use_misc) {

                // Construct offset in this dim.
                string nas;

                if (use_domain) {

                    // Get const offset in inner dim.
                    // E.g., if idim == 'y', A(t, x+1, y+4) => 4.
                    auto& offsets = gp.get_arg_offsets();
                    auto ofs = offsets[dname];

                    // Is non-zero?
                    if (ofs) {
                        if (_dims._fold_gt1.lookup(dname))
                            nas = _dims.make_norm_str(ofs, dname);
                        else
                            nas = to_string(ofs);
                    }

                    // Override?
                    if (dname == _dims._inner_layout_dim && inner_expr.length())
                        nas = inner_expr;
                }
                    
                // Offset from min value of this misc index.
                else {
                    assert(type == MISC_INDEX);
                    auto min_val = var->get_min_indices()[dname];
                    nas = gp.make_arg_str(dname, var_map) + " - " + to_string(min_val);
                }

                // Get stride in this dim.
                auto* stride = lookup_stride(*var, dname);
                assert(stride);

                // Mult & add to offset expression.
                if (nas.length()) {
                    if (nterms)
                        ofs_str += " + ";
                    ofs_str += "((" + nas + ") * (" + *stride + "))";
                    nterms++;
                }
            }
        }

        return ofs_str;
    }
    
    // Print any needed memory reads and/or constructions to 'os'.
    // Return code containing a vector of var points.
    string CppVecPrintHelper::read_from_point(ostream& os, const VarPoint& gp) {
        get_point_stats();
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
                code_str = read_from_scalar_point(os, gp, &_vec2elem_local_map);
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
            else if (_aligned_reads.count(gp)) {
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
                THROW_YASK_EXCEPTION("(internal fault) type unknown for point " + gp.make_str());
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
    string CppVecPrintHelper::print_aligned_vec_read(ostream& os,
                                                     const VarPoint& gp) {

        // Make comment and function call.
        print_point_comment(os, gp, "Read aligned vector");
        string mv_name = make_var_name("vec");

        // Is it already in a buffer?
        bool in_buf = false;
        if (_inner_loop_key.count(gp)) {
            auto key = _inner_loop_key.at(gp);
            if (_pt_buf_name.count(*key)) {
                auto& bname = _pt_buf_name.at(*key);

                // Offset.
                const string& ildim = _settings._inner_loop_dim; // ofs dim.
                auto& offsets = gp.get_arg_offsets();
                auto& ofs = offsets[ildim]; // Elem ofs.
                auto vofs = ofs / _dims._fold[ildim]; // Vector ofs.
                auto lo = _pt_inner_loop_lo.at(*key);
                auto len = _pt_buf_len.at(*key);
                auto end = lo + len;
                int i = vofs - lo;
                if (i >= 0 && i < len) {
                    in_buf = true;

                    // Load from buffer.
                    os << _line_prefix << get_var_type() << " " << mv_name << " = " <<
                        bname << "[" << i << "]" << _line_suffix;
                }
            }
        }
        
        // Do we have a pointer to the base?
        // TODO: handle case with pointer to var base but no ptr to inner-loop base.
        auto* p = lookup_inner_loop_base_ptr(gp);
        if (p) {

            if (in_buf)
                os << _line_prefix << "#ifdef CHECK\n";

            // Ptr expression.
            string ptr_expr = *p;
            string ptr_var = ptr_expr;
            auto ofs_str = get_inner_loop_ptr_offset(os, gp);
            if (ofs_str.length()) {
                ptr_expr += " + (" + ofs_str + ")";
                ptr_var = make_var_name("vec_ptr");
                os << _line_prefix << CppPrintHelper::_var_ptr_type << " " << ptr_var <<
                    " = " << ptr_expr << _line_suffix;
            }

            // Check addr.
            auto rpn = make_point_call_vec(os, gp, "get_vec_ptr_norm", "", "", true);
            os << _line_prefix << "host_assert(" <<
                ptr_var << " == " << rpn << ")" << _line_suffix;

            // Output load.
            // We don't use masked loads because several aligned loads might
            // be combined to make a simulated unaligned load.
            if (!in_buf) {
                os << _line_prefix << get_var_type() << " " << mv_name << _line_suffix;
                os << _line_prefix << mv_name << ".load_from(" << ptr_var << ")" <<
                    _line_suffix;
            }

            // Check value.
            else {
                os << _line_prefix << "host_assert(" << mv_name << " == *" <<
                    ptr_var << ")" << _line_suffix << 
                    _line_prefix << "#endif // CHECK\n";
            }

        } else if (!in_buf) {

            // If no buffer or pointer, use function call.
            auto rvn = make_point_call_vec(os, gp, "read_vec_norm", "", "", true);
            os << _line_prefix << get_var_type() << " " << mv_name << " = " <<
                rvn << _line_suffix;
        }
            
        return mv_name;
    }

    // Read from a single point.
    // Return code for read.
    string CppVecPrintHelper::read_from_scalar_point(ostream& os, const VarPoint& gp,
                                                     const VarMap* var_map) {
        assert(var_map);
        auto* var = gp._get_var();
        assert(!var->is_foldable()); // Assume all scalar reads are from non-vec vars.

        // Do we have a pointer to the base?
        auto* p = lookup_inner_loop_base_ptr(gp);
        if (p) {

            // Ptr expression.
            string ptr_expr = *p;
            string ptr_var = ptr_expr;
            auto ofs_str = get_inner_loop_ptr_offset(os, gp, var_map);
            if (ofs_str.length()) {
                ptr_expr += " + (" + ofs_str + ")";
                ptr_var = make_var_name("elem_ptr");
                os << _line_prefix << CppPrintHelper::_var_ptr_type << " " << ptr_var <<
                    " = " << ptr_expr << _line_suffix;
            }
            
            // Check addr.
            auto rp = make_point_call_vec(os, gp, "get_elem_ptr_local", "", "", false, var_map);
            os << _line_prefix << "host_assert(" <<
                ptr_var << " == " << rp << ")" << _line_suffix;

            // Return expr.
            return string("*(") + ptr_var + ")";
        }

        else
            return make_point_call_vec(os, gp, "read_elem_local", "", "", false, var_map);
    }

    // Read from multiple points that are not vectorized.
    // Return var name.
    string CppVecPrintHelper::print_partial_vec_read(ostream& os, const VarPoint& gp) {
        print_point_comment(os, gp, "Construct folded vector from non-folded data");

        // Make a vec var.
        string mv_name = make_var_name("vec");
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

                    auto& ename = _vec2elem_local_map.at(dname);
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
                    string vname = make_var_name("scalar");
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
        string mv_name = make_var_name("unaligned_vec");
        os << _line_prefix << get_var_type() << " " << mv_name << _line_suffix;
        auto vp = make_point_call_vec(os, gp, "get_elem_ptr", "", "true", false);

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
        auto* p = lookup_inner_loop_base_ptr(gp);
        if (p) {

            // Ptr expression.
            string ptr_expr = *p;
            string ptr_var = ptr_expr;
            auto ofs_str = get_inner_loop_ptr_offset(os, gp);
            if (ofs_str.length()) {
                ptr_expr += " + (" + ofs_str + ")";
                ptr_var = make_var_name("var_ptr");
                os << _line_prefix << CppPrintHelper::_var_ptr_type << " " << ptr_var <<
                    " = " << ptr_expr << _line_suffix;
            }

            // Check addr.
            auto rpn = make_point_call_vec(os, gp, "get_vec_ptr_norm", "", "", true);
            os << _line_prefix << "host_assert(" <<
                ptr_var << " == " << rpn << ")" << _line_suffix;

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
            auto vn = make_point_call_vec(os, gp, fn, val, _write_mask, true);
            os << _line_prefix << vn << _line_suffix;
        }
    }

    // Print conversion from memory vars to point var gp if needed.
    // This calls print_unaligned_vec_ctor(), which can be overloaded
    // by derived classes.
    string CppVecPrintHelper::print_unaligned_vec(ostream& os, const VarPoint& gp) {
        print_point_comment(os, gp, "Construct unaligned");

        // Declare var.
        string pv_name = make_var_name("unaligned_vec");
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

    // Print some rank info.
    void CppVecPrintHelper::print_rank_data(ostream& os) {
        auto& fold = get_fold();
        os << "\n // Rank data.\n";
        int i = 0;
        for (auto& dim : fold) {
            auto& dname = dim._get_name();
            string rdoname = _rank_domain_offset_prefix + dname;
            os << " const idx_t " << rdoname <<
                " = core_data->_common_core._rank_domain_offsets[" << i << "];\n";
            i++;
        }
    }
    
    // Print init of element indices.
    // Fill _vec2elem_*_map as side-effect.
    void CppVecPrintHelper::print_elem_indices(ostream& os) {
        auto& fold = get_fold();
        os << "\n // Element indices derived from vector indices"
            " (only used for non-vectorized vars).\n";
        for (auto& dim : fold) {
            auto& dname = dim._get_name();
            string cap_dname = PrinterBase::all_caps(dname);
            string elname = dname + _elem_suffix_local;
            string egname = dname + _elem_suffix_global;
            string rdoname = _rank_domain_offset_prefix + dname;
            os << " idx_t " << elname <<
                " = " << dname << " * VLEN_" << cap_dname << ";\n"
                " idx_t " << egname << " = " << rdoname << " + " << elname << ";\n";
            _vec2elem_local_map[dname] = elname;
            _vec2elem_global_map[dname] = egname;
        }
    }
    
    // Print loop-invariant meta values for each VarPoint.
    string CppPreLoopPrintMetaVisitor::visit(VarPoint* gp) {
        assert(gp);

        // Pointer to this var's core.
        string varp = get_var_ptr(*gp);
        string vname = gp->get_var_name();
        if (!_cvph.is_local_var(varp))
            _os << "\n // Pointer to core of var '" << vname << "'.\n";
        string var_ptr = _cvph.get_local_var(_os, varp,
                                             CppPrintHelper::_var_ptr_restrict_type,
                                             vname + "_core");
        
        // Step var for this access, if any.
        auto& dims = _cvph.get_dims();
        string sas = gp->make_step_arg_str(var_ptr, dims);
        if (sas.length()) {
            if (!_cvph.is_local_var(sas))
                _os << "\n // Step index for var '" << vname << "'.\n";
            _cvph.get_local_var(_os, sas, CppPrintHelper::_step_val_type,
                                vname + "_step_idx");
        }

        // Print strides and local offsets for this var.
        _cvph.print_strides(_os, *gp);

        // Make and print a var-base pointer for this access.
        _cvph.print_var_base_ptr(_os, *gp);

       return "";
    }

    // Print loop-invariant data values for each VarPoint.
    // TODO: fix warning from loading invariant real_vec_t outside of OMP device region.
    string CppPreLoopPrintDataVisitor::visit(VarPoint* gp) {
        assert(gp);

        // Retrieve prior dependence analysis of this var point.
        auto dep_type = gp->get_var_dep();

        // If invariant, we can load now.
        if (dep_type == VarPoint::DOMAIN_VAR_INVARIANT) {

            // Not already loaded?
            if (!_cvph.lookup_point_var(*gp)) {
                string expr = _ph.read_from_point(_os, *gp);
                string res;
                make_next_temp_var(res, gp, "expr", "") << expr << _ph.get_line_suffix();

                // Save for future use.
                _cvph.save_point_var(*gp, res);
            }
        }
        return "";
    }

} // namespace yask.

