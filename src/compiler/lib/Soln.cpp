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

// Base class for defining stencil equations.

#include "Print.hpp"
#include "CppIntrin.hpp"

using namespace std;

namespace yask {

    // Stencil-solution APIs.
    yc_grid_ptr StencilSolution::newGrid(const std::string& name,
                                         bool isScratch,
                                         const std::vector<yc_index_node_ptr>& dims) {
        
        // Make new grid and add to solution.
        // TODO: fix this mem leak--make smart ptr.

        // Copy pointers to concrete type.
        IndexExprPtrVec dims2;
        for (auto d : dims) {
            auto d2 = dynamic_pointer_cast<IndexExpr>(d);
            assert(d2);
            dims2.push_back(d2);
        }
        
        auto* gp = new Grid(name, isScratch, this, dims2);
        assert(gp);
        return gp;
    }

    // Stencil-solution APIs.
    void StencilSolution::set_fold_len(const yc_index_node_ptr dim,
                                       int len) {
        auto& fold = _settings._foldOptions;
        fold.addDimBack(dim->get_name(), len);
    }
    void StencilSolution::set_cluster_mult(const yc_index_node_ptr dim,
                                           int mult) {
        auto& cluster = _settings._clusterOptions;
        cluster.addDimBack(dim->get_name(), mult);
    }

    // Create the intermediate data for printing.
    void StencilSolution::analyze_solution(int vlen,
                                           bool is_folding_efficient) {

        // Call the stencil 'define' method to create ASTs.
        // ASTs can also be created via the APIs.
        define();

        // Find all the stencil dimensions from the grids.
        // Create the final folds and clusters from the cmd-line options.
        _dims.setDims(_grids, _settings, vlen, is_folding_efficient, *_dos);

        // Determine which grids can be folded.
        _grids.setFolding(_dims);

        // Determine which grid points can be vectorized and analyze inner-loop accesses.
        _eqs.analyzeVec(_dims);
        _eqs.analyzeLoop(_dims);

        // Find dependencies between equations.
        EqDepMap eq_deps;
        _eqs.findDeps(_dims, &eq_deps, *_dos);

        // Update access stats for the grids.
        _eqs.updateGridStats(eq_deps);
        
        // Create equation groups based on dependencies and/or target strings.
        _eqGroups.set_basename_default(_settings._eq_group_basename_default);
        _eqGroups.set_dims(_dims);
        _eqGroups.makeEqGroups(_eqs, _settings._gridRegex,
                               _settings._eqGroupTargets, eq_deps, *_dos);
        _eqGroups.optimizeEqGroups(_settings, "scalar & vector", false, *_dos);

        // Make a copy of each equation at each cluster offset.
        // We will use these for inter-cluster optimizations and code generation.
        *_dos << "Constructing cluster of equations containing " <<
            _dims._clusterMults.product() << " vector(s)...\n";
        _clusterEqGroups = _eqGroups;
        _clusterEqGroups.replicateEqsInCluster(_dims);
        if (_settings._doOptCluster)
            _clusterEqGroups.optimizeEqGroups(_settings, "cluster", true, *_dos);
    }

    // Format in given format-type.
    void StencilSolution::format(const string& format_type,
                                 yask_output_ptr output) {

        // Look for format match.
        // Most args to the printers just set references to data.
        // Data itself will be created in analyze_solution().
        PrinterBase* printer = 0;
        if (format_type == "cpp")
            printer = new YASKCppPrinter(*this, _eqGroups, _clusterEqGroups, &_dims);
        else if (format_type == "knc")
            printer = new YASKKncPrinter(*this, _eqGroups, _clusterEqGroups, &_dims);
        else if (format_type == "avx" || format_type == "avx2")
            printer = new YASKAvx256Printer(*this, _eqGroups, _clusterEqGroups, &_dims);
        else if (format_type == "avx512" || format_type == "avx512f")
            printer = new YASKAvx512Printer(*this, _eqGroups, _clusterEqGroups, &_dims);
        else if (format_type == "dot")
            printer = new DOTPrinter(*this, _clusterEqGroups, false);
        else if (format_type == "dot-lite")
            printer = new DOTPrinter(*this, _clusterEqGroups, true);
        else if (format_type == "pseudo")
            printer = new PseudoPrinter(*this, _clusterEqGroups);
        else if (format_type == "pov-ray") // undocumented.
            printer = new POVRayPrinter(*this, _clusterEqGroups);
        else {
            THROW_YASK_EXCEPTION("Error: format-type '" << format_type <<
                                 "' is not recognized");
        }
        assert(printer);
        int vlen = printer->num_vec_elems();
        bool is_folding_efficient = printer->is_folding_efficient();

        // Set data for equation groups, dims, etc.
        analyze_solution(vlen, is_folding_efficient);

        // Create the output.
        *_dos << "Generating '" << format_type << "' output...\n";
        printer->print(output->get_ostream());
        delete printer;
    }

} // namespace yask.

