/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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
    Grid::new_relative_grid_point(std::vector<int> dim_offsets) {

        // Check for correct number of indices.
        if (_dims.size() != dim_offsets.size()) {
            THROW_YASK_EXCEPTION("Error: attempt to create a relative grid point in " <<
                                 _dims.size() << "D grid '" << _name << "' with " <<
                                 dim_offsets.size() << " indices");
        }

        // Check dim types.
        // Make default args w/just index.
        NumExprPtrVec args;
        for (size_t i = 0; i < _dims.size(); i++) {
            auto dim = _dims.at(i);
            if (dim->getType() == MISC_INDEX) {
                THROW_YASK_EXCEPTION("Error: attempt to create a relative grid point in " <<
                                     _dims.size() << "D grid '" << _name <<
                                     "' containing non-step or non-domain dim '" <<
                                     dim->getName() << "'");
            }
            auto ie = dim->clone();
            args.push_back(ie);
        }
        
        // Create a point from the args.
        GridPointPtr gpp = make_shared<GridPoint>(this, args);

        // Modify the offsets.
        for (size_t i = 0; i < _dims.size(); i++) {
            auto dim = _dims.at(i);
            IntScalar ofs(dim->getName(), dim_offsets.at(i));
            gpp->setArgOffset(ofs);
        }
        return gpp;
    }

    vector<string> Grid::get_dim_names() const {
        vector<string> ret;
        for (auto dn : getDims())
            ret.push_back(dn->getName());
        return ret;
    }

    // yask_compiler_factory API methods.
    // See yask_compiler_api.hpp.
    std::string yc_factory::get_version_string() {
    	return yask_get_version_string();
    }
    yc_solution_ptr yc_factory::new_solution(const std::string& name) const {
        return make_shared<EmptyStencil>(name);
    }
    
    // Create an expression to a specific point in this grid.
    // Note that this doesn't actually 'read' or 'write' a value;
    // it's just a node in an expression.
    GridPointPtr Grid::makePoint(const NumExprPtrVec& args) {
        auto gpp = make_shared<GridPoint>(this, args);
        return gpp;
    }

    // Ctor for Grid.
    Grid::Grid(string name,
               bool isScratch,
               StencilSolution* soln,
               IndexExprPtr dim1,
               IndexExprPtr dim2,
               IndexExprPtr dim3,
               IndexExprPtr dim4,
               IndexExprPtr dim5,
               IndexExprPtr dim6) :
            _name(name),       // TODO: validate that name is legal C++ var.
            _isScratch(isScratch),
            _soln(soln)
    {
        assert(soln);
        
        // Name already used?
        auto& grids = soln->getGrids();
        for (auto gp : grids) {
            if (gp->getName() == _name) {
                cerr << "Error: grid name '" << _name << "' already used.\n";
                exit(1);
            }
        }
        
        // Register in soln.
        if (soln)
            grids.insert(this);

        // Add dims that are not null.
        if (dim1)
            _dims.push_back(dim1);
        if (dim2)
            _dims.push_back(dim2);
        if (dim3)
            _dims.push_back(dim3);
        if (dim4)
            _dims.push_back(dim4);
        if (dim5)
            _dims.push_back(dim5);
        if (dim6)
            _dims.push_back(dim6);
    }
    Grid::Grid(string name,
               bool isScratch,
               StencilSolution* soln,
               const IndexExprPtrVec& dims) :
        Grid(name, isScratch, soln) {
        _dims = dims;
    }

    // Determine whether grid can be folded.
    void Grid::setFolding(const Dimensions& dims) {

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

#if 1
        // NB: this will always be true if there is no vectorization.
        // We do this because the compiler expects stencils to be vectorizable.
        _isFoldable = _numFoldableDims == dims._foldGT1.size();
#else
        _isFoldable = (_numFoldableDims > 0 ) && (_numFoldableDims == dims._foldGT1.size());
#endif
    }
    
    // Update halos based on each value in 'offsets' in some
    // read or write to this grid.
    void Grid::updateHalo(const IntTuple& offsets) {

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
            auto& halos = _halos[left][stepVal];

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
    void Grid::updateConstIndices(const IntTuple& indices) {

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
    int Grid::getStepDimSize() const
    {
        // Only need one value if no step-dim index used.
        auto stepDim = getStepDim();
        if (!stepDim)
            return 1;

        // No info stored?
        if (_halos.size() == 0)
            return 1;

        // First and last step-dim.
        int first_ofs = 0, last_ofs = 0;

        // left and right.
        for (auto& i : _halos) {
            //auto left = i.first;
            auto& h2 = i.second; // map of step-dims to halos.

            // Step-dim ofs.
            for (auto& j : h2) {
                auto ofs = j.first;
                //auto& halo = j.second; // halo tuple at step-val 'ofs'.

                // Update vars.
                first_ofs = min(first_ofs, ofs);
                last_ofs = max(last_ofs, ofs);
            }
        }

        // First and last largest halos.
        int first_max_halo = 0, last_max_halo = 0;
        for (auto& i : _halos) {
            //auto left = i.first;
            auto& h2 = i.second; // map of step-dims to halos.

            if (h2.count(first_ofs))
                first_max_halo = max(first_max_halo, h2.at(first_ofs).max());
            if (h2.count(last_ofs))
                last_max_halo = max(last_max_halo, h2.at(last_ofs).max());
        }
        
        // Default step-dim size is range of offsets.
        assert(last_ofs >= first_ofs);
        int sz = last_ofs - first_ofs + 1;
    
        // If first and last halos are zero, we can further optimize storage by
        // immediately reusing memory location.
        if (sz > 1 && first_max_halo == 0 && last_max_halo == 0)
            sz--;

        // TODO: recognize that reading in one eq-bundle and then writing in
        // another can also reuse storage.

        return sz;
    }

    // Find the dimensions to be used based on the grids in
    // the solution and the settings from the cmd-line or API.
    void Dimensions::setDims(Grids& grids,
                             CompilerSettings& settings,
                             int vlen,                  // SIMD len based on CPU arch.
                             bool is_folding_efficient, // heuristic based on CPU arch.
                             ostream& os)
    {
        _domainDims.clear();
        _stencilDims.clear();
        _scalar.clear();
        _fold.clear();
        _foldGT1.clear();
        _clusterPts.clear();
        _clusterMults.clear();
        _miscDims.clear();

        // Get dims from grids.
        for (auto gp : grids) {
            auto& gname = gp->getName();
                
            // Dimensions in this grid.
            for (auto dim : gp->getDims()) {
                auto& dname = dim->getName();
                auto type = dim->getType();

                switch (type) {

                case STEP_INDEX:
                    if (_stepDim.length() && _stepDim != dname) {
                        THROW_YASK_EXCEPTION("Error: step dimensions '" << _stepDim <<
                                             "' and '" << dname << "' found; only one allowed");
                    }
                    _stepDim = dname;
                    _stencilDims.addDimFront(dname, 0); // must be first!

                    // Scratch grids cannot use step dim.
                    if (gp->isScratch()) {
                        cerr << "Error: scratch grid '" << gname <<
                            "' cannot use step dimension '" << dname << "'.\n";
                        exit(1);
                    }
                    break;

                case DOMAIN_INDEX:
                    _domainDims.addDimBack(dname, 0);
                    _stencilDims.addDimBack(dname, 0);
                    _scalar.addDimBack(dname, 1);
                    _fold.addDimBack(dname, 1);
                    _clusterMults.addDimBack(dname, 1);
                    break;

                case MISC_INDEX:
                    _miscDims.addDimBack(dname, 0);
                    break;

                default:
                    THROW_YASK_EXCEPTION("Error: unexpected dim type " << type);
                }
            }
        }
        if (_stepDim.length() == 0) {
            THROW_YASK_EXCEPTION("Error: no step dimension defined");
        }
        if (!_domainDims.getNumDims()) {
            THROW_YASK_EXCEPTION("Error: no domain dimensions defined");
        }

        // Use last domain dim as inner one.
        // TODO: make this selectable.
        _innerDim = _domainDims.getDimName(_domainDims.getNumDims() - 1);
        
        // Layout of fold.
        _fold.setFirstInner(settings._firstInner);
        _foldGT1.setFirstInner(settings._firstInner);
        
        os << "Step dimension: " << _stepDim << endl;
        os << "Domain dimension(s): " << _domainDims.makeDimStr() << endl;
    
        // Set fold lengths based on cmd-line options.
        for (auto& dim : settings._foldOptions.getDims()) {
            auto& dname = dim.getName();
            int sz = dim.getVal();

            // Nothing to do for fold < 2.
            if (sz <= 1)
                continue;

            // Domain dim?
            if (!_domainDims.lookup(dname)) {
                os << "Warning: fold in '" << dname <<
                    "' dim ignored because it is not a domain dim.\n";
                continue;
            }

            // Set size.
            _fold.addDimBack(dname, sz);
            _foldGT1.addDimBack(dname, sz);
        }

        // Make sure folds cover vlen (unless vlen is 1).
        if (vlen > 1 && _fold.product() != vlen) {
            if (_fold.product() > 1)
                os << "Notice: adjusting requested fold to achieve SIMD length of " <<
                    vlen << ".\n";

            // Heuristics to determine which dims to modify.
            IntTuple targets = _foldGT1; // start with specified ones >1.
            const int nTargets = is_folding_efficient ? 2 : 1; // desired num targets.
            int fdims = _fold.getNumDims();
            if (targets.getNumDims() < nTargets && fdims > 1)
                targets.addDimBack(_fold.getDim(fdims - 2)); // 2nd from last.
            if (targets.getNumDims() < nTargets && fdims > 2)
                targets.addDimBack(_fold.getDim(fdims - 3)); // 3rd from last.
            if (targets.getNumDims() < nTargets)
                targets = _fold; // all.

            // Heuristic: incrementally increase targets by powers of 2.
            _fold.setValsSame(1);
            for (int n = 1; _fold.product() < vlen; n++) {
                for (auto i : targets.getDims()) {
                    auto& dname = i.getName();
                    if (_fold.product() < vlen)
                        _fold.setVal(dname, 1 << n);
                }
            }

            // Still wrong?
            if (_fold.product() != vlen) {
                _fold.setValsSame(1);

                // Heuristic: set first target to vlen.
                if (targets.getNumDims()) {
                    auto& dname = targets.getDim(0).getName();
                    _fold.setVal(dname, vlen);
                }
            }

            // Still wrong?
            if (_fold.product() != vlen) {
                _fold.setValsSame(1);
                os << "Warning: not able to adjust fold.\n";
            }

            // Fix foldGT1.
            _foldGT1.clear();
            for (auto i : _fold.getDims()) {
                auto& dname = i.getName();
                auto& val = i.getVal();
                if (val > 1)
                    _foldGT1.addDimBack(dname, val);
            }
        }
        os << " Number of SIMD elements: " << vlen << endl;
        os << " Vector-fold dimension(s) and point-size(s): " <<
            _fold.makeDimValStr(" * ") << endl;

        // Checks for unaligned loads.
        if (settings._allowUnalignedLoads) {
            if (_foldGT1.size() > 1) {
                THROW_YASK_EXCEPTION("Error: attempt to allow unaligned loads when there are " <<
                    _foldGT1.size() << " dimensions in the vector-fold that are > 1");
            }
            else if (_foldGT1.size() > 0)
                cout << "Notice: memory layout MUST have unit-stride in " <<
                    _foldGT1.makeDimStr() << " dimension!" << endl;
        }

        // Create final cluster lengths based on cmd-line options.
        for (auto& dim : settings._clusterOptions.getDims()) {
            auto& dname = dim.getName();
            int mult = dim.getVal();

            // Nothing to do for mult < 2.
            if (mult <= 1)
                continue;

            // Does it exist anywhere?
            if (!_domainDims.lookup(dname)) {
                os << "Warning: cluster-multiplier in '" << dname <<
                    "' dim ignored because it's not a domain dim.\n";
                continue;
            }

            // Set the size.
            _clusterMults.addDimBack(dname, mult);
        }
        _clusterPts = _fold.multElements(_clusterMults);
    
        os << " Cluster dimension(s) and multiplier(s): " <<
            _clusterMults.makeDimValStr(" * ") << endl;
        os << " Cluster dimension(s) and point-size(s): " <<
            _clusterPts.makeDimValStr(" * ") << endl;
        if (_miscDims.getNumDims())
            os << "Misc dimension(s): " << _miscDims.makeDimStr() << endl;
    }

    // Make string like "+(4/VLEN_X)" or "-(2/VLEN_Y)" or "" if ofs==zero.
    // given signed offset and direction.
    string Dimensions::makeNormStr(int ofs, string dname) const {
        ostringstream oss;

        if (ofs == 0)
            return "";
        
        if (_fold.lookup(dname)) {

            // Positive offset, e.g., '+(4 / VLEN_X)'.
            if (ofs > 0)
                oss << "+(" << ofs;
            
            // Neg offset, e.g., '-(4 / VLEN_X)'.
            // Put '-' sign outside division to fix truncated division problem.
            else
                oss << "-(" << (-ofs);
                    
            // add divisor.
            string cap_dname = PrinterBase::allCaps(dname);
            oss << " / VLEN_" << cap_dname << ")";
        }

        // No fold const avail.
        else
            oss << ofs;

        return oss.str();
    }
    
    // Make string like "t+1" or "t-1".
    string Dimensions::makeStepStr(int offset) const {
        IntTuple step;
        step.addDimBack(_stepDim, offset);
        return step.makeDimValOffsetStr();
    }

} // namespace yask.
