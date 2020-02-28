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

///////// Classes for Vars ////////////

#pragma once

#include "Expr.hpp"

using namespace std;

namespace yask {

    // Fwd decl.
    struct Dimensions;

    // A class for a Var.
    // This is a generic container for all variables to be accessed
    // from the kernel. A 0-D var is a scalar, a 1-D var is an array, etc.
    // Dims can be the step dim, a domain dim, or anything else.
    class Var : public virtual yc_var {

    protected:
        string _name;           // name of this var.
        indexExprPtrVec _dims;  // dimensions of this var.
        bool _isScratch = false; // true if a temp var.

        // Step-dim info.
        bool _isStepAllocFixed = true; // step alloc cannot be changed at run-time.
        idx_t _stepAlloc = 0;         // step-alloc override (0 => calculate).

        // Ptr to solution that this var belongs to (its parent).
        StencilSolution* _soln = 0;

        // How many dims are foldable.
        int _numFoldableDims = -1; // -1 => unknown.

        // Whether this var can be vector-folded.
        bool _isFoldable = false;

        ///// Values below are computed based on VarPoint accesses in equations.

        // Min and max const indices that are used to access each dim.
        IntTuple _minIndices, _maxIndices;

        // Max abs-value of domain-index halos required by all eqs at
        // various step-index values.
        // string key: name of pack.
        // bool key: true=left, false=right.
        // int key: step-dim offset or 0 if no step-dim.
        map<string, map<bool, map<int, IntTuple>>> _halos;

        // Greatest L1 dist of any halo point that accesses this var.
        int _l1Dist = 0;

    public:
        // Ctors.
        Var(string name,
             bool isScratch,
             StencilSolution* soln,
             const indexExprPtrVec& dims);

        // Dtor.
        virtual ~Var() { }

        // Name accessors.
        const string& getName() const { return _name; }
        void setName(const string& name) { _name = name; }
        string getDescr() const;

        // Access dims.
        virtual const indexExprPtrVec& getDims() const { return _dims; }

        // Step dim or null if none.
        virtual const indexExprPtr getStepDim() const {
            for (auto d : _dims)
                if (d->getType() == STEP_INDEX)
                    return d;
            return nullptr;
        }

        // Temp var?
        virtual bool isScratch() const { return _isScratch; }

        // Access to solution.
        virtual StencilSolution* getSoln() { return _soln; }
        virtual void setSoln(StencilSolution* soln) { _soln = soln; }

        // Get foldablity.
        virtual int getNumFoldableDims() const {
            assert(_numFoldableDims >= 0);
            return _numFoldableDims;
        }
        virtual bool isFoldable() const {
            assert(_numFoldableDims >= 0);
            return _isFoldable;
        }

        // Get min and max observed indices.
        virtual const IntTuple& getMinIndices() const { return _minIndices; }
        virtual const IntTuple& getMaxIndices() const { return _maxIndices; }

        // Get the max sizes of halo across all steps for given pack.
        virtual IntTuple getHaloSizes(const string& packName, bool left) const {
            IntTuple halo;
            if (_halos.count(packName) && _halos.at(packName).count(left)) {
                for (auto i : _halos.at(packName).at(left)) {
                    auto& hs = i.second; // halo at step-val 'i'.
                    halo = halo.makeUnionWith(hs);
                    halo = halo.maxElements(hs, false);
                }
            }
            return halo;
        }

        // Get the max size in 'dim' of halo across all packs and steps.
        virtual int getHaloSize(const string& dim, bool left) const {
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

        // Get max L1 dist of halos.
        virtual int getL1Dist() const {
            return _l1Dist;
        }

        // Determine whether dims are same.
        virtual bool areDimsSame(const Var& other) const {
            if (_dims.size() != other._dims.size())
                return false;
            size_t i = 0;
            for (auto& dim : _dims) {
                auto d2 = other._dims[i].get();
                if (!dim->isSame(d2))
                    return false;
                i++;
            }
            return true;
        }

        // Determine how many values in step-dim are needed.
        virtual int getStepDimSize() const;

        // Determine whether var can be folded.
        virtual void setFolding(const Dimensions& dims);

        // Determine whether halo sizes are equal.
        virtual bool isHaloSame(const Var& other) const;

        // Update halos and L1 dist based on halo in 'other' var.
        virtual void updateHalo(const Var& other);

        // Update halos and L1 dist based on each value in 'offsets'.
        virtual void updateHalo(const string& packName, const IntTuple& offsets);

        // Update L1 dist.
        virtual void updateL1Dist(int l1Dist) {
            _l1Dist = max(_l1Dist, l1Dist);
        }

        // Update const indices based on 'indices'.
        virtual void updateConstIndices(const IntTuple& indices);

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
            return dp->getName();
        }
        virtual std::vector<std::string> get_dim_names() const;
        virtual bool
        is_dynamic_step_alloc() const {
            return !_isStepAllocFixed;
        }
        virtual void
        set_dynamic_step_alloc(bool enable) {
            _isStepAllocFixed = !enable;
        }
        virtual idx_t
        get_step_alloc_size() const {
            return getStepDimSize();
        }
        virtual void
        set_step_alloc_size(idx_t size) {
            _stepAlloc = size;
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

        // Determine whether each var can be folded.
        virtual void setFolding(const Dimensions& dims) {
            for (auto gp : _vars)
                gp->setFolding(dims);
        }
    };

} // namespace yask.
