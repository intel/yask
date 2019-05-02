/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2019, Intel Corporation

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

///////// Methods for Grids, etc. ////////////

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Parse.hpp"
#include "Grid.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {


    // grid APIs.
    yc_grid_point_node_ptr
    GridVar::new_grid_point(const std::vector<yc_number_node_ptr>& index_exprs) {

        // Check for correct number of indices.
        if (_dims.size() != index_exprs.size()) {
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: attempt to create a grid point in " <<
                                            _dims.size() << "D grid '" << _name << "' with " <<
                                            index_exprs.size() << " index expressions");
        }

        // Make args.
        numExprPtrVec args;
        for (size_t i = 0; i < _dims.size(); i++) {
            auto p = dynamic_pointer_cast<NumExpr>(index_exprs.at(i));
            assert(p);
            args.push_back(p->clone());
        }

        // Create a point from the args.
        gridPointPtr gpp = make_shared<GridPoint>(this, args);
        return gpp;
    }

    yc_grid_point_node_ptr
    GridVar::new_relative_grid_point(const std::vector<int>& dim_offsets) {

        // Check for correct number of indices.
        if (_dims.size() != dim_offsets.size()) {
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: attempt to create a relative grid point in " <<
                                            _dims.size() << "D grid '" << _name << "' with " <<
                                            dim_offsets.size() << " indices");
        }

        // Check dim types.
        // Make default args w/just index.
        numExprPtrVec args;
        for (size_t i = 0; i < _dims.size(); i++) {
            auto dim = _dims.at(i);
            if (dim->getType() == MISC_INDEX) {
                FORMAT_AND_THROW_YASK_EXCEPTION("Error: attempt to create a relative grid point in " <<
                                                _dims.size() << "D grid '" << _name <<
                                                "' containing non-step or non-domain dim '" <<
                                                dim->getName() << "'");
            }
            auto ie = dim->clone();
            args.push_back(ie);
        }

        // Create a point from the args.
        gridPointPtr gpp = make_shared<GridPoint>(this, args);

        // Set the offsets, which creates a new
        // expression for each index.
        for (size_t i = 0; i < _dims.size(); i++) {
            auto dim = _dims.at(i);
            IntScalar ofs(dim->getName(), dim_offsets.at(i));
            gpp->setArgOffset(ofs);
        }
        return gpp;
    }

    vector<string> GridVar::get_dim_names() const {
        vector<string> ret;
        for (auto dn : getDims())
            ret.push_back(dn->getName());
        return ret;
    }

    // Ctor for GridVar.
    GridVar::GridVar(string name,
                     bool isScratch,
                     StencilSolution* soln,
                     const indexExprPtrVec& dims) :
        _name(name),       // TODO: validate that name is legal C++ var.
        _isScratch(isScratch),
        _soln(soln)
    {
        assert(soln);

        // Name already used?
        auto& grids = soln->getGrids();
        for (auto gp : grids) {
            if (gp->getName() == name)
                THROW_YASK_EXCEPTION("Error: grid name '" + name + "' already used");
        }

        // Register in soln.
        grids.insert(this);

        // Define dims.
        _dims = dims;
    }

    // Determine whether grid can be folded.
    void GridVar::setFolding(const Dimensions& dims) {

        _numFoldableDims = 0;

        // Never fold scalars, even if there is no vectorization.
        if (get_num_dims() == 0) {
            _isFoldable = false;
            return;
        }

        // Find the number of folded dims used in this grid.
        for (auto fdim : dims._foldGT1.getDims()) {
            auto& fdname = fdim.getName();

            // Search for dim in grid.
            bool found = false;
            for (auto gdim : _dims) {
                auto& gdname = gdim->getName();
                if (fdname == gdname) {
                    found = true;
                    break;
                }
            }
            if (found)
                _numFoldableDims++;
        }

        // Can fold if ALL fold dims >1 are used in this grid.

        // NB: this will always be true if there is no vectorization, i.e.,
        // both are zero.  We do this because the compiler expects stencils
        // to be vectorizable.
        _isFoldable = _numFoldableDims == int(dims._foldGT1.size());
    }
    
    // Determine whether halo sizes are equal.
    bool GridVar::isHaloSame(const GridVar& other) const {

        // Same dims?
        if (!areDimsSame(other))
            return false;

        // Same halos?
        for (auto& dim : _dims) {
            auto& dname = dim->getName();
            auto dtype = dim->getType();
            if (dtype == DOMAIN_INDEX) {
                for (bool left : { false, true }) {
                    int sz = getHaloSize(dname, left);
                    int osz = other.getHaloSize(dname, left);
                    if (sz != osz)
                        return false;
                }
            }
        }
        return true;
    }

    // Update halos based on halo in 'other' grid.
    // This grid's halos can only be increased.
    void GridVar::updateHalo(const GridVar& other) {
        assert(areDimsSame(other));

        // Loop thru other grid's halo values.
        for (auto& hi : other._halos) {
            auto& pname = hi.first;
            auto& h2 = hi.second;
            for (auto& i0 : h2) {
                auto& left = i0.first;
                auto& m1 = i0.second;
                for (auto& i1 : m1) {
                    auto& step = i1.first;
                    const IntTuple& ohalos = i1.second;
                    for (auto& dim : ohalos.getDims()) {
                        auto& dname = dim.getName();
                        auto& val = dim.getVal();
                        
                        // Any existing value?
                        auto& halos = _halos[pname][left][step];
                        auto* p = halos.lookup(dname);

                        // If not, add this one.
                        if (!p)
                            halos.addDimBack(dname, val);

                        // Keep larger value.
                        else if (val > *p)
                            *p = val;

                        // Else, current value is larger than val, so don't update.
                    }
                }
            }
        }
    }

    // Update halos based on each value in 'offsets' in some
    // read or write to this grid.
    // This grid's halos can only be increased.
    void GridVar::updateHalo(const string& packName, const IntTuple& offsets) {

        // Find step value or use 0 if none.
        int stepVal = 0;
        auto stepDim = getStepDim();
        if (stepDim) {
            auto* p = offsets.lookup(stepDim->getName());
            if (p)
                stepVal = *p;
        }

        // Update halo vals.
        for (auto& dim : offsets.getDims()) {
            auto& dname = dim.getName();
            int val = dim.getVal();
            bool left = val <= 0;
            auto& halos = _halos[packName][left][stepVal];

            // Don't keep halo in step dim.
            if (stepDim && dname == stepDim->getName())
                continue;

            // Store abs value.
            val = abs(val);

            // Any existing value?
            auto* p = halos.lookup(dname);

            // If not, add this one.
            if (!p)
                halos.addDimBack(dname, val);

            // Keep larger value.
            else if (val > *p)
                *p = val;

            // Else, current value is larger than val, so don't update.
        }
    }

    // Update const indices based on 'indices'.
    void GridVar::updateConstIndices(const IntTuple& indices) {

        for (auto& dim : indices.getDims()) {
            auto& dname = dim.getName();
            int val = dim.getVal();

            // Update min.
            auto* minp = _minIndices.lookup(dname);
            if (!minp)
                _minIndices.addDimBack(dname, val);
            else if (val < *minp)
                *minp = val;

            // Update max.
            auto* maxp = _maxIndices.lookup(dname);
            if (!maxp)
                _maxIndices.addDimBack(dname, val);
            else if (val > *maxp)
                *maxp = val;
        }
    }

    // Determine how many values in step-dim are needed.
    int GridVar::getStepDimSize() const
    {
        // Specified by API.
        if (_stepAlloc > 0)
            return _stepAlloc;

        // No step-dim index used.
        auto stepDim = getStepDim();
        if (!stepDim)
            return 1;

        // Specified on cmd line.
        if (_soln->getSettings()._stepAlloc > 0)
            return _soln->getSettings()._stepAlloc;
        
        // No info stored?
        if (_halos.size() == 0)
            return 1;

        // Need the max across all packs.
        int max_sz = 1;

        // Loop thru each pack w/halos.
        for (auto& hi : _halos) {
#ifdef DEBUG_HALOS
            auto& pname = hi.first;
#endif
            auto& h2 = hi.second;

            // First and last step-dim found.
            const int unset = -9999;
            int first_ofs = unset, last_ofs = unset;

            // left and right.
            for (auto& i : h2) {
                //auto left = i.first;
                auto& h3 = i.second; // map of step-dims to halos.

                // Step-dim offsets.
                for (auto& j : h3) {
                    auto ofs = j.first;
                    auto& halo = j.second; // halo tuple at step-val 'ofs'.

                    // Any existing value?
                    if (halo.size()) {
#ifdef DEBUG_HALOS
                        cout << "** grid " << _name << " has halo " << halo.makeDimValStr() <<
                            " at ofs " << ofs << " in pack " << pname << endl;
#endif

                        // Update vars.
                        if (first_ofs == unset)
                            first_ofs = last_ofs = ofs;
                        else {
                            first_ofs = min(first_ofs, ofs);
                            last_ofs = max(last_ofs, ofs);
                        }
                    }
                }
            }
#ifdef DEBUG_HALOS
            cout << "** grid " << _name << " has halos from " << first_ofs <<
                " to " << last_ofs << " in pack " << pname << endl;
#endif

            // Only need to process if >1 offset.
            if (last_ofs != unset && first_ofs != unset && last_ofs != first_ofs) {

                // Default step-dim size is range of step offsets.
                // For example, if equation touches 't' through 't+2',
                // 'sz' is 3.
                int sz = last_ofs - first_ofs + 1;

                // First and last largest halos.
                int first_max_halo = 0, last_max_halo = 0;
                for (auto& i : h2) {
                    //auto left = i.first;
                    auto& h3 = i.second; // map of step-dims to halos.

                    if (h3.count(first_ofs) && h3.at(first_ofs).size())
                        first_max_halo = max(first_max_halo, h3.at(first_ofs).max());
                    if (h3.count(last_ofs) && h3.at(last_ofs).size())
                        last_max_halo = max(last_max_halo, h3.at(last_ofs).max());
                }

                // If first and last halos are zero, we can further optimize storage by
                // immediately reusing memory location.
                if (sz > 1 && first_max_halo == 0 && last_max_halo == 0)
                    sz--;

                // Keep max so far.
                max_sz = max(max_sz, sz);
            }

        } // packs.

        return max_sz;
    }

    // Description of this grid.
    string GridVar::getDescr() const {
        string d = _name + "(";
        int i = 0;
        for (auto dn : getDims()) {
            if (i++) d += ", ";
            d += dn->getName();
        }
        d += ")";
        return d;
    }

} // namespace yask.
