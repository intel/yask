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
    void Dimensions::setDims(Vars& vars,
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

        // Get dims from vars.
        for (auto& gp : vars) {
            auto& gname = gp->getName();
            os << "Var: " << gp->getDescr() << endl;

            // Dimensions in this var.
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

                    // Scratch vars cannot use step dim.
                    if (gp->isScratch())
                        THROW_YASK_EXCEPTION("Error: scratch var '" + gname +
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

        // Set specific positional dims.
        _outerDim = _domainDims.getDimName(0);
        _innerDim = _domainDims.getDimName(_domainDims.getNumDims() - 1);
        string _nearInnerDim = _domainDims.getNumDims() >= 2 ?
            _domainDims.getDimName(_domainDims.getNumDims() - 2) : _outerDim;

        os << "Step dimension: " << _stepDim << endl;
        os << "Domain dimension(s): " << _domainDims.makeDimStr() << endl;

        // Extract domain fold lengths based on cmd-line options.
        IntTuple foldOpts;
        for (auto& dim : _domainDims) {
            auto& dname = dim.getName();

            // Was folding specified for this dim?
            auto* p = settings._foldOptions.lookup(dname);
            if (!p)
                continue;
            int sz = *p;
            if (sz < 1)
                continue;
            
            // Set size.
            _fold.setVal(dname, sz);
            foldOpts.addDimBack(dname, sz);
        }
        os << " Number of SIMD elements: " << vlen << endl;
        if (foldOpts.getNumDims())
            os << " Requested vector-fold dimension(s) and point-size(s): " <<
                _fold.makeDimValStr(" * ") << endl;
        else
            os << " No explicitly-requested vector-folding.\n";

        // If needed, adjust folding to exactly cover vlen unless vlen is 1.
        // If vlen is 1, we will allow any folding.
        if (vlen > 1 && _fold.product() != vlen) {
            if (foldOpts.getNumDims())
                os << "Notice: adjusting requested fold to achieve SIMD length of " <<
                    vlen << ".\n";

            // If 1D, there is only one option.
            if (_domainDims.getNumDims() == 1)
                _fold[_innerDim] = vlen;

            // If 2D+, adjust folding.
            else {

                // Determine inner-dim size separately because
                // vector-folding works best when folding is
                // applied in non-inner dims.
                int inner_sz = 1;

                // If specified dims are within vlen, try to use
                // specified inner-dim.
                if (foldOpts.product() < vlen) {

                    // Inner-dim fold-size requested and a factor of vlen?
                    auto* p = foldOpts.lookup(_innerDim);
                    if (p && (vlen % *p == 0))
                        inner_sz = *p;
                }

                // Remaining vlen to be split over non-inner dims.
                int upper_sz = vlen / inner_sz;

                // Tuple for non-inner dims.
                IntTuple innerFolds;
                
                // If we only want 1D folding, just set one to
                // needed value.
                if (!is_folding_efficient)
                    innerFolds.addDimBack(_nearInnerDim, upper_sz);

                // Else, make a tuple of hints to use for setting non-inner
                // sizes.
                else {
                    IntTuple innerOpts;
                    for (auto& dim : _domainDims) {
                        auto& dname = dim.getName();
                        if (dname == _innerDim)
                            continue;
                        auto* p = foldOpts.lookup(dname);
                        int sz = p ? *p : 0; // 0 => not specified.
                        innerOpts.addDimFront(dname, sz); // favor more inner ones.
                    }
                    assert(innerOpts.getNumDims() == _domainDims.getNumDims() - 1);

                    // Get final size of non-inner dims.
                    innerFolds = innerOpts.get_compact_factors(upper_sz);
                }

                // Put them into the fold.
                for (auto& dim : _domainDims) {
                    auto& dname = dim.getName();
                    if (dname == _innerDim)
                        _fold[dname] = inner_sz;
                    else if (innerFolds.lookup(dname))
                        _fold[dname] = innerFolds[dname];
                    else
                        _fold[dname] = 1;
                }
                assert(_fold.getNumDims() == _domainDims.getNumDims());
            }            

            // Check it.
            if (_fold.product() != vlen)
                THROW_YASK_EXCEPTION("Internal error: failed to set folding for VLEN " +
                                     to_string(vlen));
        }

        // Set foldGT1.
        for (auto i : _fold) {
            auto& dname = i.getName();
            auto& val = i.getVal();
            if (val > 1)
                _foldGT1.addDimBack(dname, val);
        }
        os << " Vector-fold dimension(s) and point-size(s): " <<
            _fold.makeDimValStr(" * ") << endl;

        // Layout used inside each folded vector.
        _fold.setFirstInner(settings._firstInner);
        _foldGT1.setFirstInner(settings._firstInner);

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
        for (auto& dim : settings._clusterOptions) {
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
