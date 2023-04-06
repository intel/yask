/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

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

///////// Classes for Vars ////////////

#pragma once

#include "Expr.hpp"
#include "VarPoint.hpp"

using namespace std;

namespace yask {

    // Fwd decl.
    struct Dimensions;
    class CompilerSettings;

    // A class for a Var.
    // This is a generic container for all variables to be accessed
    // from the kernel. A 0-D var is a scalar, a 1-D var is an array, etc.
    // Dims can be the step dim, a domain dim, or a misc dim.
    class Var : public virtual yc_var {

    protected:
        string _name;           // name of this var.
        index_expr_ptr_vec _dims;  // dimensions of this var in param order.
        index_expr_ptr_vec _layout_dims;  // dimensions of this var in layout order.
        bool _is_scratch = false; // true if a temp var.

        // Step-dim info.
        bool _is_step_alloc_fixed = true; // step alloc cannot be changed at run-time.
        idx_t _step_alloc = 0;         // step-alloc override (0 => calculate).

        // Ptr to solution that this var belongs to (its parent).
        StencilSolution* _soln = 0;

        // How many dims of various types.
        // -1 => unknown.
        int _num_step_dims = -1;
        int _num_domain_dims = -1;
        int _num_misc_dims = -1;
        int _num_foldable_dims = -1;

        // Whether this var can be vector-folded.
        bool _is_foldable = false;

        ///// Values below are computed based on VarPoint accesses in equations.

        // Min and max const indices that are used to access each dim.
        IntTuple _min_indices, _max_indices;

        // Max abs-value of domain-index halos required by all eqs at
        // various step-index values.
        // string key: name of stage.
        // bool key: true=left, false=right.
        // int key: step-dim offset or 0 if no step-dim.
        map<string, map<bool, map<int, IntTuple>>> _halos;

        // Extra padding needed for read-ahead.
        int _read_ahead_pad = 0;

        // Key: stage.
        // Val: step-index for write in stage.
        map<string, int> _write_points;

        // Greatest L1 dist of any halo point that accesses this var.
        int _l1_dist = 0;

        virtual void _check_ok() const {
            assert(_num_step_dims >= 0);
            assert(_num_step_dims <= 1);
            assert(_num_domain_dims >= 0);
            assert(_num_misc_dims >= 0);
            assert(_num_foldable_dims >= 0);
            assert(_num_foldable_dims <= _num_domain_dims);
        }

    public:
        // Ctors.
        Var(string name,
            bool is_scratch,
            StencilSolution* soln,
            const index_expr_ptr_vec& dims);

        // Dtor.
        virtual ~Var() { }

        // Name accessors.
        const string& _get_name() const { return _name; }
        void set_name(const string& name) { _name = name; }
        string get_descr() const;

        // Access dims for this var (not for soln).
        // The are returned in declaration order (not necessarily layout order).
        virtual const index_expr_ptr_vec& get_dims() const { return _dims; }
        IntTuple get_dims_tuple() const {
            IntTuple gdims;
            for (const auto& dim : _dims) {
                const auto& dname = dim->_get_name();
                gdims.add_dim_back(dname, 0);
            }
            return gdims;
        }
        virtual const index_expr_ptr_vec& get_layout_dims() const { return _layout_dims; }
        virtual index_expr_ptr_vec& get_layout_dims() { return _layout_dims; }

        // Step dim or null if none.
        virtual const index_expr_ptr get_step_dim() const {
            _check_ok();
            if (_num_step_dims > 0) {
                for (auto d : _dims)
                    if (d->get_type() == STEP_INDEX)
                        return d;
                assert("internal error: step dim not found");
            }
            return nullptr;
        }

        // Temp var?
        virtual bool is_scratch() const { return _is_scratch; }

        // Access to solution.
        virtual StencilSolution* _get_soln() { return _soln; }
        virtual void set_soln(StencilSolution* soln) { _soln = soln; }
        virtual CompilerSettings& get_settings();
        virtual const Dimensions& get_soln_dims();

        // Get dim-type counts.
        virtual int get_num_step_dims() const {
            _check_ok();
            return _num_step_dims;
        }
        virtual int get_num_domain_dims() const {
            _check_ok();
            return _num_domain_dims;
        }
        virtual int get_num_misc_dims() const {
            _check_ok();
            return _num_misc_dims;
        }
        
        // Get foldablity.
        virtual int get_num_foldable_dims() const {
            _check_ok();
            return _num_foldable_dims;
        }
        virtual bool is_foldable() const {
            _check_ok();
            return _is_foldable;
        }

        // Get min and max observed indices.
        virtual const IntTuple& get_min_indices() const { return _min_indices; }
        virtual const IntTuple& get_max_indices() const { return _max_indices; }

        // Get the max sizes of halo across all steps for given stage.
        virtual IntTuple get_halo_sizes(const string& stage_name, bool left) const {
            IntTuple halo;
            if (_halos.count(stage_name) && _halos.at(stage_name).count(left)) {
                for (auto i : _halos.at(stage_name).at(left)) {
                    auto& hs = i.second; // halo at step-val 'i'.
                    halo = halo.make_union_with(hs);
                    halo = halo.max_elements(hs, false);
                }
            }
            return halo;
        }

        // Get the max size in 'dim' of halo across all stages and steps.
        virtual int get_halo_size(const string& dim, bool left) const {
            int h = 0;
            for (auto& hi : _halos) {
                //auto& pname = hi.first;
                auto& h2 = hi.second;
                if (h2.count(left)) {
                    for (auto i : h2.at(left)) {
                        auto& hs = i.second; // halo at step-val 'i'.
                        auto* p = hs.lookup(dim);
                        if (p)
                            h = std::max(h, *p);
                    }
                }
            }
            return h;
        }

        // Print halos.
        void print_halos(ostream& os, const string& leader) const {
            bool found = false;
            for (auto& hi : _halos) {
                auto& stname = hi.first;
                auto& h2 = hi.second;
                for (auto& i0 : h2) {
                    auto& left = i0.first;
                    auto& m1 = i0.second;
                    for (auto& i1 : m1) {
                        auto& step = i1.first;
                        const IntTuple& ohalos = i1.second;
                        os << leader << "var " << get_name() << " halo[" << stname <<
                            "][" << (left ? string("left") : string("right")) <<
                            "][step " << step << "] = " << ohalos.make_dim_val_str() << endl;
                        found = true;
                    }
                }
            }
            if (!found)
                os << leader << "var " << get_name() << " has no halo data\n";
        }

        // Extra padding.
        virtual void set_read_ahead_pad(int n) {
            _read_ahead_pad = n;
        }
        virtual void update_read_ahead_pad(int n) {
            _read_ahead_pad = max(_read_ahead_pad, n);
        }
        virtual int get_read_ahead_pad() const {
            return _read_ahead_pad;
        }

        // Get max L1 dist of halos.
        virtual int get_l1_dist() const {
            return _l1_dist;
        }

        // Determine whether dims are same as 'other' var.
        virtual bool are_dims_same(const Var& other) const {
            if (_dims.size() != other._dims.size())
                return false;
            size_t i = 0;
            for (auto& dim : _dims) {
                auto d2 = other._dims[i].get();
                if (!dim->is_same(d2))
                    return false;
                i++;
            }
            return true;
        }

        // Determine how many values in step-dim are needed.
        struct StepDimInfo {
            int step_dim_size = 1;
            map<string, int> writeback_ofs;
        };
        virtual StepDimInfo get_step_dim_info() const;

        // Determine dim-type counts and whether var can be folded.
        virtual void set_dim_counts(const Dimensions& dims);

        // Determine whether halo sizes are equal.
        virtual bool is_halo_same(const Var& other) const;

        // Update halos and L1 dist based on halo in 'other' var.
        // Returns 'true' if halo changed.
        virtual bool update_halo(const Var& other);

        // Update halos and L1 dist based on each value in 'offsets'.
        // Returns 'true' if halo changed.
        virtual bool update_halo(const string& stage_name, const IntTuple& offsets);

        // Stage(s) with writes.
        virtual const map<string, int>& get_write_points() const {
            return _write_points;
        }
        virtual void update_write_points(const string& stage_name, const IntTuple& offsets);

        // Update L1 dist.
        virtual void update_l1_dist(int l1_dist) {
            _l1_dist = max(_l1_dist, l1_dist);
        }

        // Update const indices based on 'indices'.
        virtual void update_const_indices(const IntTuple& indices);

        // APIs.
        virtual const string& get_name() const {
            return _name;
        }
        virtual int get_num_dims() const {
            return int(_dims.size());
        }
        virtual const string& get_dim_name(int n) const {
            assert(n >= 0);
            assert(n < get_num_dims());
            auto dp = _dims.at(n);
            assert(dp);
            return dp->_get_name();
        }
        virtual string_vec get_dim_names() const;
        virtual bool
        is_dynamic_step_alloc() const {
            return !_is_step_alloc_fixed;
        }
        virtual void
        set_dynamic_step_alloc(bool enable) {
            _is_step_alloc_fixed = !enable;
        }
        virtual idx_t
        get_step_alloc_size() const {
            auto sdi = get_step_dim_info();
            return sdi.step_dim_size;
        }
        virtual void
        set_step_alloc_size(idx_t size) {
            _step_alloc = size;
        }
        virtual yc_var_point_node_ptr
        new_var_point(const std::vector<yc_number_node_ptr>& index_exprs);
        virtual yc_var_point_node_ptr
        new_var_point(const std::initializer_list<yc_number_node_ptr>& index_exprs) {
            std::vector<yc_number_node_ptr> idx_expr_vec(index_exprs);
            return new_var_point(idx_expr_vec);
        }
        virtual yc_var_point_node_ptr
        new_relative_var_point(const std::vector<int>& dim_offsets);
        virtual yc_var_point_node_ptr
        new_relative_var_point(const std::initializer_list<int>& dim_offsets) {
            std::vector<int> dim_ofs_vec(dim_offsets);
            return new_relative_var_point(dim_ofs_vec);
        }
    };

    // A list of vars.  This holds pointers to vars defined by the stencil
    // class in the order in which they are added via the INIT_VAR_* macros.
    class Vars {
        vector_set<Var*> _vars;

    public:

        Vars() {}
        virtual ~Vars() {}

        // Copy ctor.
        // Copies list of var pointers, but not vars (shallow copy).
        Vars(const Vars& src) : _vars(src._vars) {}

        // STL methods.
        void clear() {
            _vars.clear();
        }
        size_t size() const {
            return _vars.size();
        }
        Var* at(size_t i) {
            return _vars.at(i);
        }
        const Var* at(size_t i) const {
            return _vars.at(i);
        }
        vector<Var*>::const_iterator begin() const {
            return _vars.begin();
        }
        vector<Var*>::const_iterator end() const {
            return _vars.end();
        }
        size_t count(Var* p) const {
            return _vars.count(p);
        }
        void insert(Var* p) {
            _vars.insert(p);
        }

        // Determine dim-type counts and whether each var can be folded.
        virtual void set_dim_counts(const Dimensions& dims) {
            for (auto gp : _vars)
                gp->set_dim_counts(dims);
        }
    };

} // namespace yask.
