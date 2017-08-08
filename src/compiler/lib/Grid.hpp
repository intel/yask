/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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

///////// Classes for Grids and Eqs ////////////

#ifndef GRID_HPP
#define GRID_HPP

#include "Expr.hpp"

using namespace std;

namespace yask {

    // Fwd decl.
    struct Dimensions;
    
    // A class for a Grid.
    // This is a generic container for all variables to be accessed
    // from the kernel. A 0-D grid is a scalar, a 1-D grid is an array, etc.
    // Dims can be the step dim, a domain dim, or anything else.
    class Grid : public virtual yc_grid {

    protected:
        string _name;           // name of this grid.
        IndexExprPtrVec _dims;  // dimensions of this grid.

        // Ptr to solution that this grid belongs to (its parent).
        StencilSolution* _soln = 0;

        // How many dims are foldable.
        int _numFoldableDims = -1; // -1 => unknown.
        
        // Whether this grid can be vector-folded.
        bool _isFoldable = false;

        // Values below are computed based on equations.
        
        // Min and max const indices that are used to access each dim.
        IntTuple _minIndices, _maxIndices;
        
        // Max abs-value of domain-index halos required by all eqs at
        // various step-index values.
        // TODO: keep separate pos and neg halos.
        // TODO: keep separate halos for each equation.
        map<int, IntTuple> _halos;  // key: step-dim offset.
    
    public:
        // Ctors.
        Grid(string name, StencilSolution* soln,
             const IndexExprPtrVec& dims);
        Grid(string name, StencilSolution* soln,
             IndexExprPtr dim1 = nullptr,
             IndexExprPtr dim2 = nullptr,
             IndexExprPtr dim3 = nullptr,
             IndexExprPtr dim4 = nullptr,
             IndexExprPtr dim5 = nullptr,
             IndexExprPtr dim6 = nullptr);

        // Dtor.
        virtual ~Grid() { }

        // Name accessors.
        const string& getName() const { return _name; }
        void setName(const string& name) { _name = name; }

        // Access dims.
        virtual const IndexExprPtrVec& getDims() const { return _dims; }

        // Step dim or null if none.
        virtual const IndexExprPtr getStepDim() const {
            for (auto d : _dims)
                if (d->getType() == STEP_INDEX)
                    return d;
            return nullptr;
        }

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

        // Get the max size in 'dim' of halo across all step dims.
        virtual int getHaloSize(const string& dim) const {
            int h = 0;
            for (auto i : _halos) {
                auto& hi = i.second; // halo at step-val 'i'.
                auto* p = hi.lookup(dim);
                if (p)
                    h = std::max(h, *p);
            }
            return h;
        }

        // Determine how many values in step-dim are needed.
        virtual int getStepDimSize() const;

        // Determine whether grid can be folded.
        virtual void setFolding(const Dimensions& dims);
        
        // Update halos based on each value in 'offsets'.
        virtual void updateHalo(const IntTuple& offsets);

        // Update const indices based on 'indices'.
        virtual void updateConstIndices(const IntTuple& indices);
    
        // Create an expression to a specific point in this grid.
        // Note that this doesn't actually 'read' or 'write' a value;
        // it's just a node in an expression.
        virtual GridPointPtr makePoint(const NumExprPtrVec& args);
        virtual GridPointPtr makePoint() {
            NumExprPtrVec args;
            return makePoint(args);
        }

        // Convenience functions for zero dimensions (scalar).
        virtual operator NumExprPtr() { // implicit conversion.
            return makePoint();
        }
        virtual operator GridPointPtr() { // implicit conversion.
            return makePoint();
        }
        virtual GridPointPtr operator()() {
            return makePoint();
        }

        // Convenience functions for one dimension (array).
        virtual GridPointPtr operator[](const NumExprArg i1) {
            NumExprPtrVec args;
            args.push_back(i1);
            return makePoint(args);
        }
        virtual GridPointPtr operator()(const NumExprArg i1) {
            return operator[](i1);
        }

        // Convenience functions for more dimensions.
        virtual GridPointPtr operator()(const NumExprArg i1, const NumExprArg i2) {
            NumExprPtrVec args;
            args.push_back(i1);
            args.push_back(i2);
            return makePoint(args);
        }
        virtual GridPointPtr operator()(const NumExprArg i1, const NumExprArg i2,
                                        const NumExprArg i3) {
            NumExprPtrVec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            return makePoint(args);
        }
        virtual GridPointPtr operator()(const NumExprArg i1, const NumExprArg i2,
                                        const NumExprArg i3, const NumExprArg i4) {
            NumExprPtrVec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            args.push_back(i4);
            return makePoint(args);
        }
        virtual GridPointPtr operator()(const NumExprArg i1, const NumExprArg i2,
                                        const NumExprArg i3, const NumExprArg i4,
                                        const NumExprArg i5) {
            NumExprPtrVec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            args.push_back(i4);
            args.push_back(i5);
            return makePoint(args);
        }
        virtual GridPointPtr operator()(const NumExprArg i1, const NumExprArg i2,
                                        const NumExprArg i3, const NumExprArg i4,
                                        const NumExprArg i5, const NumExprArg i6) {
            NumExprPtrVec args;
            args.push_back(i1);
            args.push_back(i2);
            args.push_back(i3);
            args.push_back(i4);
            args.push_back(i5);
            args.push_back(i6);
            return makePoint(args);
        }

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
        virtual yc_grid_point_node_ptr
        new_relative_grid_point(int dim1_offset,
                                int dim2_offset,
                                int dim3_offset,
                                int dim4_offset,
                                int dim5_offset,
                                int dim6_offset);
        virtual yc_grid_point_node_ptr
        new_relative_grid_point(std::vector<int> dim_offsets);
    };

    // Set that retains order of things added.
    // Or, vector that allows insertion if element doesn't exist.
    // Keeps two copies of everything, so don't put large things in it.
    // TODO: hide vector inside class and provide proper accessor methods.
    template <typename T>
    class vector_set : public vector<T> {
        map<T, size_t> _posn;

    public:
        vector_set() {}
        virtual ~vector_set() {}

        // Copy ctor.
        vector_set(const vector_set& src) :
            vector<T>(src), _posn(src._posn) {}
    
        virtual size_t count(const T& val) const {
            return _posn.count(val);
        }
        virtual void insert(const T& val) {
            if (_posn.count(val) == 0) {
                vector<T>::push_back(val);
                _posn[val] = vector<T>::size() - 1;
            }
        }
        virtual void erase(const T& val) {
            if (_posn.count(val) > 0) {
                size_t op = _posn.at(val);
                vector<T>::erase(vector<T>::begin() + op);
                for (auto pi : _posn) {
                    auto& p = pi.second;
                    if (p > op)
                        p--;
                }
            }
        }
        virtual void clear() {
            vector<T>::clear();
            _posn.clear();
        }
    };

    // A list of grids.  This holds pointers to grids defined by the stencil
    // class in the order in which they are added via the INIT_GRID_* macros.
    class Grids : public vector_set<Grid*> {
    public:
    
        Grids() {}
        virtual ~Grids() {}

        // Copy ctor.
        // Copies list of grid pointers, but not grids (shallow copy).
        Grids(const Grids& src) : vector_set<Grid*>(src) {}

        // Determine whether each grid can be folded.
        virtual void setFolding(const Dimensions& dims) {
            for (auto gp : *this)
                gp->setFolding(dims);
        }
        
    };

    // Settings for the compiler.
    // May be provided via cmd-line or API.
    class CompilerSettings {
    public:
        int _elem_bytes = 4;    // bytes in an FP element.
        IntTuple _foldOptions;    // vector fold.
        IntTuple _clusterOptions; // cluster multipliers.
        bool _firstInner = true; // first dimension of fold is unit step.
        string _eq_group_basename_default = "stencil";
        bool _allowUnalignedLoads = false;
        int _haloSize = 0;      // 0 => calculate each halo separately and automatically.
        int _stepAlloc = 0;     // 0 => calculate step allocation automatically.
        int _maxExprSize = 50;
        int _minExprSize = 2;
        bool _doCse = true;      // do common-subexpr elim.
        bool _doComb = true;    // combine commutative operations.
        bool _doOptCluster = true; // apply optimizations also to cluster.
        bool _find_deps = true;  // find dependencies between equations.
        string _eqGroupTargets;  // how to group equations.
    };
    
    // Stencil dimensions.
    struct Dimensions {
        string _stepDim;         // step dimension, usually time.
        IntTuple _domainDims;    // domain dims, usually spatial (with zero value).
        IntTuple _stencilDims;   // both step and domain dims.
        IntTuple _miscDims;      // misc dims that are not the step or domain.

        // Following contain only domain dims.
        IntTuple _scalar;       // points in scalar (value 1 in each).
        IntTuple _fold;         // points in fold.
        IntTuple _foldGT1;      // subset of _fold w/values >1.
        IntTuple _clusterPts;    // cluster size in points.
        IntTuple _clusterMults;  // cluster size in vectors.

        Dimensions() {}
        virtual ~Dimensions() {}
    
        // Find the dimensions to be used.
        void setDims(Grids& grids,
                     CompilerSettings& settings,
                     int vlen,
                     bool is_folding_efficient,
                     ostream& os);
    };

} // namespace yask.
    
#endif
