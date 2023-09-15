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
        Solution* _soln = 0;

        string _name;           // name of this var.
        index_expr_ptr_vec _vdims;  // dimensions of this var in param order.
        index_expr_ptr_vec _layout_dims;  // dimensions of this var in layout order.
        bool _is_scratch = false; // true if a temp var.
        int _scratch_mem_slot = -1; // mem chunk for scratch var.

        // Step-dim info.
        bool _is_step_alloc_fixed = true; // step alloc cannot be changed at run-time.
        idx_t _step_alloc = 0;         // step-alloc override (0 => calculate).

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
        Var(Solution* soln,
            string name,
            bool is_scratch,
            const index_expr_ptr_vec& vdims);

        // Dtor.
        virtual ~Var() { }

        // Name accessors.
        const string& _get_name() const { return _name; }
        void set_name(const string& name) { _name = name; }
        string get_descr() const;

        // Access dims for this var (not for soln).
        // The are returned in declaration order (not necessarily layout order).
        virtual const index_expr_ptr_vec& get_dims() const { return _vdims; }
        IntTuple get_dims_tuple() const {
            IntTuple gdims;
            for (const auto& dim : _vdims) {
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
                for (auto d : _vdims)
                    if (d->get_type() == STEP_INDEX)
                        return d;
                assert("internal error: step dim not found");
            }
            return nullptr;
        }

        // Temp var info.
        virtual bool is_scratch() const { return _is_scratch; }
        virtual int get_scratch_mem_slot() const {
            assert(_is_scratch);
            return _scratch_mem_slot;
        }
        virtual void set_scratch_mem_slot(int ms) {
            assert(_is_scratch);
            _scratch_mem_slot = ms;
        }
        virtual bool is_needed() const {
            return !is_scratch() || get_scratch_mem_slot() >= 0;
        }

        // Access to solution.
        virtual Solution* _get_soln() { return _soln; }
        virtual void set_soln(Solution* soln) { _soln = soln; }
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
        virtual int get_misc_space_size() const;

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

        // Get max L1 dist of halos.
        virtual int get_l1_dist() const {
            return _l1_dist;
        }

        // Determine whether dims are same as 'other' var.
        virtual bool are_dims_same(const Var& other) const {
            if (_vdims.size() != other._vdims.size())
                return false;
            size_t i = 0;
            for (auto& dim : _vdims) {
                auto d2 = other._vdims[i].get();
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
            _l1_dist = std::max(_l1_dist, l1_dist);
        }

        // Update const indices based on 'indices'.
        virtual void update_const_indices(const IntTuple& indices);

        // APIs.
        virtual const string& get_name() const {
            return _name;
        }
        virtual int get_num_dims() const {
            return int(_vdims.size());
        }
        virtual const string& get_dim_name(int n) const {
            assert(n >= 0);
            assert(n < get_num_dims());
            auto dp = _vdims.at(n);
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
        virtual yc_var_point_node_ptr
        new_relative_grid_point(const std::vector<int>& dim_offsets) {
            return new_relative_var_point(dim_offsets);
        }
        virtual yc_var_point_node_ptr
        new_relative_grid_point(const std::initializer_list<int>& dim_offsets) {
            return new_relative_var_point(dim_offsets);
        }
    };

    // A list of vars.  This holds pointers to vars defined by the stencil
    // class in the order in which they are added via the INIT_VAR_* macros.
    class Vars {
    protected:
        Solution* _soln;
        vector_set<Var*> _vars;

    public:
        Vars(Solution* soln) :
            _soln(soln) { }
        virtual ~Vars() {}

        // Copy ctor.
        // Copies list of var pointers, but not vars (shallow copy).
        Vars(const Vars& src) :
            _soln(src._soln),
            _vars(src._vars) {}

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
        virtual void set_dim_counts();
    };

    class LogicalVar;
    using LogicalVarPtr = shared_ptr<LogicalVar>;
    
    // A reference to a specific time offset and misc index values of a var,
    // i.e., a slice of a var in step and misc dims.
    class LogicalVar {
    protected:
        Solution* _soln;
        Var* _var;
        std::string _descr;
        bool _is_scratch;

    public:
        LogicalVar(Solution* soln,
                   VarPoint* vp) :
            _soln(soln), _var(vp->_get_var()) {
            _descr = vp->make_logical_var_str();
            _is_scratch = vp->_get_var()->is_scratch();
        }
        virtual ~LogicalVar() {}

        const std::string& get_descr() const {
            return _descr;
        }
        bool is_scratch() const {
            return _is_scratch;
        }
        Var& get_var() {
            return *_var;
        }
        const Var& get_var() const {
            return *_var;
        }

        bool operator==(const LogicalVar& rhs) const {
            return _descr == rhs._descr;
        }
        bool operator!=(const LogicalVar& rhs) const {
            return _descr != rhs._descr;
        }
        bool operator<(const LogicalVar& rhs) const {
            return _descr < rhs._descr;
        }
        bool operator>(const LogicalVar& rhs) const {
            return _descr > rhs._descr;
        }
        bool operator<=(const LogicalVar& rhs) const {
            return _descr <= rhs._descr;
        }
        bool operator>=(const LogicalVar& rhs) const {
            return _descr >= rhs._descr;
        }

        LogicalVarPtr clone() {

            // Don't copy the soln or the var.
            return make_shared<LogicalVar>(*this);
        }
    };

} // namespace yask.
