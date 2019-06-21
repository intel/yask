/*****************************************************************************

YASK: Yet Another Stencil Kit
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

///////// Classes for Settings & Dimensions ////////////

#pragma once

#include "Var.hpp"

using namespace std;

namespace yask {

    // Settings for the compiler.
    // May be provided via cmd-line or API.
    class CompilerSettings {
    public:
        string _target;             // format type.
        int _elem_bytes = 4;    // bytes in an FP element.
        string _stepDim;        // explicit step dim.
        vector<string> _domainDims; // explicit domain dims.
        IntTuple _foldOptions;    // vector fold.
        IntTuple _clusterOptions; // cluster multipliers.
        map<int, int> _prefetchDists;
        bool _firstInner = true; // first dimension of fold is unit step.
        string _eq_bundle_basename_default = "stencil_bundle";
        bool _allowUnalignedLoads = false;
        bool _bundleScratch = true;
        int _haloSize = 0;      // 0 => calculate each halo automatically.
        int _stepAlloc = 0;     // 0 => calculate each step allocation automatically.
        bool _innerMisc = false;
        int _maxExprSize = 50;
        int _minExprSize = 2;
        bool _doCse = true;      // do common-subexpr elim.
        bool _doComb = true;    // combine commutative operations.
        bool _doPairs = true;   // find equation pairs.
        bool _doOptCluster = true; // apply optimizations also to cluster.
        bool _doReorder = false;   // reorder commutative operations.
        string _eqBundleTargets;  // how to bundle equations.
        string _varRegex;       // vars to update.
        bool _findDeps = true;
        bool _printEqs = false;
    };

    // Stencil dimensions.
    struct Dimensions {
        string _stepDim;         // step dimension, usually time.
        IntTuple _domainDims;    // domain dims, usually spatial (with zero value).
        IntTuple _stencilDims;   // both step and domain dims.
        string _innerDim;        // domain dim that will be used in the inner loop.
        string _outerDim;        // domain dim that will be used in the outer loop.
        IntTuple _miscDims;      // misc dims that are not the step or domain.

        // Following contain only domain dims.
        IntTuple _scalar;       // points in scalar (value 1 in each).
        IntTuple _fold;         // points in fold.
        IntTuple _foldGT1;      // subset of _fold w/values >1.
        IntTuple _clusterPts;    // cluster size in points.
        IntTuple _clusterMults;  // cluster size in vectors.

        // Direction of stepping.
        int _stepDir = 0;       // 0: undetermined, +1: forward, -1: backward.

        Dimensions() {}
        virtual ~Dimensions() {}

        // Add step dim.
        void addStepDim(const string& dname) {
            _stepDim = dname;
            _stencilDims.addDimFront(dname, 0); // must be first!
        }
        
        // Add domain dims.
        // Last one added will be unit-stride.
        void addDomainDim(const string& dname) {
            _domainDims.addDimBack(dname, 0);
            _stencilDims.addDimBack(dname, 0);
            _scalar.addDimBack(dname, 1);
            _fold.addDimBack(dname, 1);
            _clusterMults.addDimBack(dname, 1);
        }
        
        // Find the dimensions to be used.
        void setDims(Vars& vars,
                     CompilerSettings& settings,
                     int vlen,
                     bool is_folding_efficient,
                     ostream& os);

        // Make string like "+(4/VLEN_X)" or "-(2/VLEN_Y)"
        // given signed offset and direction.
        string makeNormStr(int offset, string dim) const;

        // Make string like "t+1" or "t-1".
        string makeStepStr(int offset) const;
    };

} // namespace yask.
