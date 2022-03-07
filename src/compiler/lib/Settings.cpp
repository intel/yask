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

///////// Methods for Settings & Dimensions ////////////

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Parse.hpp"
#include "Var.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {

    // yask_compiler_factory API methods.
    // See yask_compiler_api.hpp.
    std::string yc_factory::get_version_string() {
    	return yask_get_version_string();
    }
    yc_solution_ptr yc_factory::new_solution(const std::string& name) const {
        return make_shared<StencilSolution>(name);
    }

    // Find the dimensions to be used based on the vars in
    // the solution and the settings from the cmd-line or API.
    void Dimensions::set_dims(Vars& vars,
                             CompilerSettings& settings,
                             int vlen,                  // SIMD len based on CPU arch.
                             bool is_folding_efficient, // heuristic based on CPU arch.
                             ostream& os)
    {
        _domain_dims.clear();
        _stencil_dims.clear();
        _scalar.clear();
        _fold.clear();
        _fold_gt1.clear();
        _cluster_pts.clear();
        _cluster_mults.clear();
        _misc_dims.clear();

        // Get dims from settings.
        if (settings._step_dim.length()) {
            add_step_dim(settings._step_dim);
            os << "Explicit step dimension: " << _step_dim << endl;
        }
        for (auto& dname : settings._domain_dims)
            add_domain_dim(dname);
        if (_domain_dims.size())
            os << "Explicit domain dimension(s): " << _domain_dims.make_dim_str() << endl;

        // Get dims from vars.
        for (auto& gp : vars) {
            auto& gname = gp->_get_name();
            os << "Var: " << gp->get_descr() << endl;

            // Dimensions in this var.
            for (auto dim : gp->get_dims()) {
                auto& dname = dim->_get_name();
                auto type = dim->get_type();

                switch (type) {

                case STEP_INDEX:
                    if (_step_dim.length() && _step_dim != dname) {
                        THROW_YASK_EXCEPTION("Error: step dimensions '" + _step_dim +
                                             "' and '" + dname + "' found; only one allowed");
                    }
                    add_step_dim(dname);

                    // Scratch vars cannot use step dim.
                    if (gp->is_scratch())
                        THROW_YASK_EXCEPTION("Error: scratch var '" + gname +
                                             "' cannot use step dimension '" +
                                             dname + "'.\n");
                    break;

                case DOMAIN_INDEX:
                    add_domain_dim(dname);
                    break;

                case MISC_INDEX:
                    _misc_dims.add_dim_back(dname, 0);
                    break;

                default:
                    THROW_YASK_EXCEPTION("Error: unexpected dim type " + to_string(type));
                }
            }
        }
        if (_step_dim.length() == 0) {
            THROW_YASK_EXCEPTION("Error: no step dimension defined");
        }
        if (!_domain_dims.get_num_dims()) {
            THROW_YASK_EXCEPTION("Error: no domain dimension(s) defined");
        }

        // Set specific positional dims.
        auto ndd = _domain_dims.get_num_dims();
        _outer_layout_dim = _domain_dims.get_dim_name(0);
        _inner_layout_dim = _domain_dims.get_dim_name(ndd - 1);
        string _near_inner_dim = _domain_dims.get_num_dims() >= 2 ?
            _domain_dims.get_dim_name(_domain_dims.get_num_dims() - 2) : _outer_layout_dim;
        if (settings._inner_loop_dim.length()) {
            if (isdigit(settings._inner_loop_dim[0])) {
                int dn = atoi(settings._inner_loop_dim.c_str());
                if (dn < 1) {
                    os << "Note: adjusting inner-loop-dim " << dn << " to 1.\n";
                    dn = 1;
                }
                if (dn > ndd) {
                    os << "Note: adjusting inner-loop-dim " << dn << " to " << ndd << ".\n";
                    dn = ndd;
                }
                settings._inner_loop_dim = _domain_dims.get_dim_name(dn - 1);
                _inner_loop_dim_num = dn;
            }
            int dp = _domain_dims.lookup_posn(settings._inner_loop_dim);
            if (dp < 0) {
                os << "Warning: inner-loop-dim '" << settings._inner_loop_dim <<
                    "' ignored because it's not a domain dim.\n";
                settings._inner_loop_dim.clear();
            } else
                _inner_loop_dim_num = dp + 1;
        }
        if (!settings._inner_loop_dim.length()) {
            settings._inner_loop_dim = _inner_layout_dim;
            _inner_loop_dim_num = ndd;
        }
        assert(_inner_loop_dim_num > 0);
        assert(_inner_loop_dim_num <= ndd);

        os << "Step dimension: " << _step_dim << endl;
        os << "Domain dimension(s): " << _domain_dims.make_dim_str() << endl;
        os << "Inner-loop dimension: " << settings._inner_loop_dim << endl;

        // Extract domain fold lengths based on cmd-line options.
        IntTuple fold_opts;
        for (auto& dim : _domain_dims) {
            auto& dname = dim._get_name();

            // Was folding specified for this dim?
            auto* p = settings._fold_options.lookup(dname);
            if (!p)
                continue;
            int sz = *p;
            if (sz < 1)
                continue;
            
            // Set size.
            _fold.set_val(dname, sz);
            fold_opts.add_dim_back(dname, sz);
        }
        os << " Number of SIMD elements: " << vlen << endl;
        if (fold_opts.get_num_dims())
            os << " Requested vector-fold dimension(s) and point-size(s): " <<
                _fold.make_dim_val_str(" * ") << endl;
        else
            os << " No explicitly-requested vector-folding.\n";

        // If needed, adjust folding to exactly cover vlen unless vlen is 1.
        // If vlen is 1, we will allow any folding.
        if (vlen > 1 && _fold.product() != vlen) {
            if (fold_opts.get_num_dims())
                os << "Note: adjusting requested fold to achieve SIMD length of " <<
                    vlen << ".\n";

            // If 1D, there is only one option.
            if (_domain_dims.get_num_dims() == 1)
                _fold[_inner_layout_dim] = vlen;

            // If 2D+, adjust folding.
            else {

                // Determine inner-dim size separately because
                // vector-folding works best when folding is
                // applied in non-inner-loop dims.
                int inner_sz = 1;

                // If specified dims are within vlen, try to use
                // specified inner-dim.
                if (fold_opts.product() < vlen) {

                    // Inner-dim fold-size requested and a factor of vlen?
                    auto* p = fold_opts.lookup(settings._inner_loop_dim);
                    if (p && (vlen % *p == 0))
                        inner_sz = *p;
                }

                // Remaining vlen to be split over non-inner dims.
                int upper_sz = vlen / inner_sz;

                // Tuple for non-inner dims.
                IntTuple inner_folds;
                
                // If we only want 1D folding, just set one to
                // needed value.
                if (!is_folding_efficient)
                    inner_folds.add_dim_back(_near_inner_dim, upper_sz);

                // Else, make a tuple of hints to use for setting non-inner
                // sizes.
                else {
                    IntTuple inner_opts;
                    for (auto& dim : _domain_dims) {
                        auto& dname = dim._get_name();
                        if (dname == settings._inner_loop_dim)
                            continue;
                        auto* p = fold_opts.lookup(dname);
                        int sz = p ? *p : 0; // 0 => not specified.
                        inner_opts.add_dim_front(dname, sz); // favor more inner ones.
                    }
                    assert(inner_opts.get_num_dims() == _domain_dims.get_num_dims() - 1);

                    // Get final size of non-inner dims.
                    inner_folds = inner_opts.get_compact_factors(upper_sz);
                }

                // Put them into the fold.
                for (auto& dim : _domain_dims) {
                    auto& dname = dim._get_name();
                    if (dname == settings._inner_loop_dim)
                        _fold[dname] = inner_sz;
                    else if (inner_folds.lookup(dname))
                        _fold[dname] = inner_folds[dname];
                    else
                        _fold[dname] = 1;
                }
                assert(_fold.get_num_dims() == _domain_dims.get_num_dims());
            }            

            // Check it.
            if (_fold.product() != vlen)
                THROW_YASK_EXCEPTION("Internal error: failed to set folding for VLEN " +
                                     to_string(vlen));
        }

        // Set fold_gt1.
        for (auto i : _fold) {
            auto& dname = i._get_name();
            auto& val = i.get_val();
            if (val > 1)
                _fold_gt1.add_dim_back(dname, val);
        }
        os << " Vector-fold dimension(s) and point-size(s): " <<
            _fold.make_dim_val_str(" * ") << endl;

        // Layout used inside each folded vector.
        _fold.set_first_inner(settings._first_inner);
        _fold_gt1.set_first_inner(settings._first_inner);

        // Checks for unaligned loads.
        if (settings._allow_unaligned_loads) {
            if (_fold_gt1.size() > 1) {
                FORMAT_AND_THROW_YASK_EXCEPTION("Error: attempt to allow "
                                                "unaligned loads when there are " <<
                                                _fold_gt1.size() <<
                                                " dimensions in the vector-fold that are > 1");
            }
            else if (_fold_gt1.size() > 0)
                cout << "Notice: memory layout MUST have unit-stride in " <<
                    _fold_gt1.make_dim_str() << " dimension!" << endl;
        }

        // Create final cluster lengths based on cmd-line options.
        for (auto& dim : settings._cluster_options) {
            auto& dname = dim._get_name();
            int mult = dim.get_val();

            // Nothing to do for mult < 2.
            if (mult <= 1)
                continue;

            // Does it exist anywhere?
            if (!_domain_dims.lookup(dname)) {
                os << "Warning: cluster-multiplier in '" << dname <<
                    "' dim ignored because it's not a domain dim.\n";
                continue;
            }

            // Set the size.
            _cluster_mults.add_dim_back(dname, mult);
        }
        _cluster_pts = _fold.mult_elements(_cluster_mults);

        os << " Cluster dimension(s) and multiplier(s): " <<
            _cluster_mults.make_dim_val_str(" * ") << endl;
        os << " Cluster dimension(s) and point-size(s): " <<
            _cluster_pts.make_dim_val_str(" * ") << endl;
        if (_misc_dims.get_num_dims())
            os << "Misc dimension(s): " << _misc_dims.make_dim_str() << endl;
        else
            os << "No misc dimensions used\n";
    }

    // Make string like "+(4/VLEN_X)" or "-(2/VLEN_Y)" or "" if ofs==zero.
    // given signed offset and direction.
    string Dimensions::make_norm_str(int ofs, string dname) const {

        if (ofs == 0)
            return "";

        string res;
        if (_fold.lookup(dname)) {

            // Positive offset, e.g., '+(4 / VLEN_X)'.
            if (ofs > 0)
                res += "+(" + to_string(ofs);

            // Neg offset, e.g., '-(4 / VLEN_X)'.
            // Put '-' sign outside division to fix truncated division problem.
            else
                res += "-(" + to_string(-ofs);

            // add divisor.
            string cap_dname = PrinterBase::all_caps(dname);
            res += " / VLEN_" + cap_dname + ")";
        }

        // No fold const avail.
        else
            res += to_string(ofs);

        return res;
    }

    // Make string like "t+1" or "t-1".
    string Dimensions::make_step_str(int offset) const {
        IntTuple step;
        step.add_dim_back(_step_dim, offset);
        return step.make_dim_val_offset_str();
    }

} // namespace yask.
