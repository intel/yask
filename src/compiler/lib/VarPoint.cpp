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

///////// VarPoint and EqualsExpr expression nodes.

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Parse.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {

    // VarPoint methods.
    VarPoint::VarPoint(Var* var, const num_expr_ptr_vec& args) :
        _var(var), _args(args) {
        assert(var);

        // Check for correct number of args.
        size_t nd = var->get_dims().size();
        if (nd != args.size()) {
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: attempt to create a var point in " <<
                nd << "-D var '" << get_var_name() << "' with " <<
                args.size() << " indices");
        }

        // Eval each arg.
#ifdef DEBUG_GP
        cout << "Creating var point " << make_quoted_str() << "...\n";
#endif
        auto dims = var->get_dims();
        for (size_t i = 0; i < nd; i++) {
            auto dim = dims.at(i);
            auto dname = dim->_get_name();
            auto arg = args.at(i);
            assert(arg);
#ifdef DEBUG_GP
            cout << " Arg " << arg->make_quoted_str() <<
                " at dim '" << dname << "'\n";
#endif
            int offset = 0;

            // A compile-time const?
            if (arg->is_const_val()) {
#ifdef DEBUG_GP
                cout << "  is const val " << arg->get_int_val() << endl;
#endif
                IntScalar c(dname, arg->get_int_val());
                set_arg_const(c);
            }

            // A simple offset?
            else if (arg->is_offset_from(dname, offset)) {
#ifdef DEBUG_GP
                cout << "  has offset " << offset << endl;
#endif
                IntScalar o(dname, offset);
                set_arg_offset(o);
            }
        }
        _update_str();
    }
    const num_expr_ptr VarPoint::get_arg(const string& dim) const {
        for (int di = 0; di < _var->get_num_dims(); di++) {
            auto& dn = _var->get_dim_name(di);  // name of dim at this posn.
            if (dim == dn)
                return _args.at(di);
        }
        return nullptr;
    }
    const string& VarPoint::get_var_name() const {
        return _var->_get_name();
    }
    bool VarPoint::is_var_foldable() const {
        return _var->is_foldable();
    }
    string VarPoint::make_arg_str(const VarMap* var_map) const {
        string str;
        int i = 0;
        for (auto arg : _args) {
            if (i++) str += ", ";
            str += arg->make_str(var_map);
        }
        return str;
    }
    string VarPoint::make_arg_str(const string& dim,
                                  const VarMap* var_map) const {
        int i = 0;
        auto arg = get_arg(dim);
        assert(arg.get());
        string str = arg->make_str(var_map);
        return str;
    }
    string VarPoint::_make_str(const VarMap* var_map) const {
        string str = _var->_get_name() + "(" +
                             make_arg_str(var_map) + ")";
        return str;
    }
    string VarPoint::make_logical_var_str(const VarMap* var_map) const {
        string str = _var->_get_name();
        if (_consts.size())
            str += "(" + _consts.make_dim_val_str() + ")";
        return str;
    }
    const index_expr_ptr_vec& VarPoint::get_dims() const {
        return _var->get_dims();
    }
    const index_expr_ptr_vec& VarPoint::get_layout_dims() const {
        return _var->get_layout_dims();
    }

    // Make normalized string like "x+(4/VLEN_X)" from
    // original arg "x+4" in 'dname' dim.
    // Args w/o simple offset are not modified.
    string VarPoint::make_norm_arg_str(const string& dname,
                                       const Dimensions& dims,
                                       const VarMap* var_map) const {
        string res;

        // Const offset?
        auto* ofs = _offsets.lookup(dname);

        // Zero offset?
        if (ofs && *ofs == 0)
            res = dname;
        
        // dname exists in fold?
        else if (ofs && dims._fold_gt1.lookup(dname))
            res = "(" + dname + dims.make_norm_str(*ofs, dname) + ")";
        
        // Otherwise, just find and format arg as-is.
        else {
            auto& gdims = _var->get_dims();
            for (size_t i = 0; i < gdims.size(); i++) {
                auto gdname = gdims[i]->_get_name();
                if (gdname == dname) {
                    res = _args.at(i)->make_str(var_map);
                    break;
                }
            }
        }
        assert(res.length());
        return res;
    }

    // Make string like "x+(4/VLEN_X), y, z-(2/VLEN_Z)" from
    // original args "x+4, y, z-2".
    // This object has numerators; norm object has denominators.
    // Args w/o simple offset are not modified.
    string VarPoint::make_norm_arg_str(const Dimensions& dims,
                                       const VarMap* var_map) const {

        string res;
        auto& gd = _var->get_dims();
        for (size_t i = 0; i < gd.size(); i++) {
            if (i)
                res += ", ";
            auto dname = gd[i]->_get_name();
            res += make_norm_arg_str(dname, dims, var_map);
        }
        return res;
    }

    // Make string like "g->_wrap_step(t+1)" from original arg "t+1"
    // if var uses step dim, "" otherwise.
    // If var doesn't allow dynamic alloc, set to fixed value.
    string VarPoint::make_step_arg_str(const string& var_ptr,
                                       const Dimensions& dims) const {

        auto& gd = _var->get_dims();
        for (size_t i = 0; i < gd.size(); i++) {
            auto dname = gd[i]->_get_name();
            auto& arg = _args.at(i);
            if (dname == dims._step_dim) {
                if (_var->is_dynamic_step_alloc())
                    return var_ptr + "->_wrap_step(" + arg->make_str() + ")";
                else {
                    auto step_alloc = _var->get_step_alloc_size();
                    if (step_alloc == 1)
                        return "0"; // 1 alloc => always index 0.
                    else 
                        return "imod_flr<idx_t>(" + arg->make_str() + ", " +
                            to_string(step_alloc) + ")";
                }
            }
        }
        return "";
    }

    // Set given arg to given offset; ignore if not in step or domain var dims.
    void VarPoint::set_arg_offset(const IntScalar& offset) {

        // Find dim in var.
        auto gdims = _var->get_dims();
        for (size_t i = 0; i < gdims.size(); i++) {
            auto gdim = gdims[i];

            // Must be domain or step dim.
            if (gdim->get_type() == MISC_INDEX)
                continue;

            auto dname = gdim->_get_name();
            if (offset._get_name() == dname) {

                // Make offset equation.
                int ofs = offset.get_val();
                auto ie = gdim->clone();
                num_expr_ptr nep;
                if (ofs > 0) {
                    auto op = make_shared<ConstExpr>(ofs);
                    nep = make_shared<AddExpr>(ie, op);
                }
                else if (ofs < 0) {
                    auto op = make_shared<ConstExpr>(-ofs);
                    nep = make_shared<SubExpr>(ie, op);
                }
                else                // 0 offset.
                    nep = ie;

                // Replace in args.
                _args[i] = nep;

                // Set offset.
                _offsets.add_dim_back(dname, ofs);

                // Remove const if it exists.
                _consts = _consts.remove_dim(dname);

                break;
            }
        }
        _update_str();
    }

    // Set given arg to given const;
    void VarPoint::set_arg_const(const IntScalar& val) {

        // Find dim in var.
        auto gdims = _var->get_dims();
        for (size_t i = 0; i < gdims.size(); i++) {
            auto gdim = gdims[i];

            auto dname = gdim->_get_name();
            if (val._get_name() == dname) {

                // Make const expr.
                int v = val.get_val();
                auto vp = make_shared<ConstExpr>(v);

                // Replace in args.
                _args[i] = vp;

                // Set const
                _consts.add_dim_back(dname, v);

                // Remove offset if it exists.
                _offsets = _offsets.remove_dim(dname);

                break;
            }
        }
        _update_str();
    }

    // Set given arg to given expr.
    void VarPoint::set_arg_expr(const string& expr_dim, const string& expr) {

        // Find dim in var.
        auto gdims = _var->get_dims();
        for (size_t i = 0; i < gdims.size(); i++) {
            auto gdim = gdims[i];
            auto dname = gdim->_get_name();
            if (expr_dim == dname) {

                // Make expr node.
                auto ep = make_shared<CodeExpr>(expr);

                // Replace in args.
                _args[i] = ep;

                // Remove const and/or offset if either exists.
                _consts = _consts.remove_dim(dname);
                _offsets = _offsets.remove_dim(dname);

                break;
            }
        }
        _update_str();
    }

    // EqualsExpr methods.
    bool EqualsExpr::is_scratch() {
        Var* gp = _get_var();
        return gp && gp->is_scratch();
    }
    bool EqualsExpr::is_same(const Expr* other) const {
        auto p = dynamic_cast<const EqualsExpr*>(other);
        return p &&
            _lhs->is_same(p->_lhs.get()) &&
            _rhs->is_same(p->_rhs.get()) &&
            are_exprs_same(_cond, p->_cond) && // might be null.
            are_exprs_same(_step_cond, p->_step_cond); // might be null.
    }

} // namespace yask.
