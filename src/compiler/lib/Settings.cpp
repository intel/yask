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

///////// Methods for Settings & Dimensions ////////////

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Parse.hpp"
#include "Grid.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {

    // yask_compiler_factory API methods.
    // See yask_compiler_api.hpp.
    std::string yc_factory::get_version_string() {
    	return yask_get_version_string();
    }
    yc_solution_ptr yc_factory::new_solution(const std::string& name) const {
        return make_shared<EmptyStencil>(name);
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

        // Get dims from settings.
        if (settings._stepDim.length()) {
            addStepDim(settings._stepDim);
            os << "Explicit step dimension: " << _stepDim << endl;
        }
        for (auto& dname : settings._domainDims)
            addDomainDim(dname);
        if (_domainDims.size())
            os << "Explicit domain dimension(s): " << _domainDims.makeDimStr() << endl;

        // Get dims from grids.
        for (auto& gp : grids) {
            auto& gname = gp->getName();
            os << "Grid: " << gp->getDescr() << endl;

            // Dimensions in this grid.
            for (auto dim : gp->getDims()) {
                auto& dname = dim->getName();
                auto type = dim->getType();

                switch (type) {

                case STEP_INDEX:
                    if (_stepDim.length() && _stepDim != dname) {
                        THROW_YASK_EXCEPTION("Error: step dimensions '" + _stepDim +
                                             "' and '" + dname + "' found; only one allowed");
                    }
                    addStepDim(dname);

                    // Scratch grids cannot use step dim.
                    if (gp->isScratch())
                        THROW_YASK_EXCEPTION("Error: scratch grid '" + gname +
                                             "' cannot use step dimension '" +
                                             dname + "'.\n");
                    break;

                case DOMAIN_INDEX:
                    addDomainDim(dname);
                    break;

                case MISC_INDEX:
                    _miscDims.addDimBack(dname, 0);
                    break;

                default:
                    THROW_YASK_EXCEPTION("Error: unexpected dim type " + type);
                }
            }
        }
        if (_stepDim.length() == 0) {
            THROW_YASK_EXCEPTION("Error: no step dimension defined");
        }
        if (!_domainDims.getNumDims()) {
            THROW_YASK_EXCEPTION("Error: no domain dimension(s) defined");
        }

        // Use last domain dim as inner one.
        _innerDim = _domainDims.getDimName(_domainDims.getNumDims() - 1);

        // Layout of fold.
        _fold.setFirstInner(settings._firstInner);
        _foldGT1.setFirstInner(settings._firstInner);

        os << "Step dimension: " << _stepDim << endl;
        os << "Domain dimension(s): " << _domainDims.makeDimStr() << endl;

        // Extract certain fold lengths based on cmd-line options.
        IntTuple foldOpts;
        for (auto& dim : _fold.getDims()) {
            auto& dname = dim.getName();

            // Was folding specified?
            auto* p = settings._foldOptions.lookup(dname);
            if (!p)
                continue;

            // Nothing to do for fold < 2.
            int sz = *p;
            if (sz <= 1)
                continue;

            // Set size.
            _fold.setVal(dname, sz);
            foldOpts.addDimBack(dname, sz);
        }
        os << " Number of SIMD elements: " << vlen << endl;
        if (foldOpts.getNumDims())
            os << " Requested vector-fold dimension(s) and point-size(s): " <<
                _fold.makeDimValStr(" * ") << endl;

        // Make sure folding exactly covers vlen (unless vlen is 1).
        // If vlen is 1, we will allow any folding.
        if (vlen > 1 && _fold.product() != vlen) {
            if (_fold.product() > 1)
                os << "Notice: adjusting requested fold to achieve SIMD length of " <<
                    vlen << ".\n";

            // Heuristics to determine which dims to modify.
            IntTuple targets = foldOpts; // start with specified ones >1.
            const int nTargets = is_folding_efficient ? 2 : 1; // desired num targets.
            int fdims = _fold.getNumDims();
            if (targets.getNumDims() < nTargets && fdims > 1)
                targets.addDimBack(_fold.getDim(fdims - 2)); // 2nd from last.
            if (targets.getNumDims() < nTargets && fdims > 2)
                targets.addDimBack(_fold.getDim(fdims - 3)); // 3rd from last.
            if (targets.getNumDims() < nTargets)
                targets = _fold; // all.
            assert(targets.getNumDims() > 0);

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
            if (_fold.product() != vlen)
                THROW_YASK_EXCEPTION("Internal error: cannot set folding for VLEN " +
                                     to_string(vlen));
        }

        // Set foldGT1.
        for (auto i : _fold.getDims()) {
            auto& dname = i.getName();
            auto& val = i.getVal();
            if (val > 1)
                _foldGT1.addDimBack(dname, val);
        }
        os << " Vector-fold dimension(s) and point-size(s): " <<
            _fold.makeDimValStr(" * ") << endl;

        // Checks for unaligned loads.
        if (settings._allowUnalignedLoads) {
            if (_foldGT1.size() > 1) {
                FORMAT_AND_THROW_YASK_EXCEPTION("Error: attempt to allow "
                                                "unaligned loads when there are " <<
                                                _foldGT1.size() <<
                                                " dimensions in the vector-fold that are > 1");
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
        else
            os << "No misc dimensions used\n";
    }

    // Make string like "+(4/VLEN_X)" or "-(2/VLEN_Y)" or "" if ofs==zero.
    // given signed offset and direction.
    string Dimensions::makeNormStr(int ofs, string dname) const {

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
            string cap_dname = PrinterBase::allCaps(dname);
            res += " / VLEN_" + cap_dname + ")";
        }

        // No fold const avail.
        else
            res += to_string(ofs);

        return res;
    }

    // Make string like "t+1" or "t-1".
    string Dimensions::makeStepStr(int offset) const {
        IntTuple step;
        step.addDimBack(_stepDim, offset);
        return step.makeDimValOffsetStr();
    }

} // namespace yask.
