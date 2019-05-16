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

// Base class for defining stencil equations.

#include "Print.hpp"
#include "CppIntrin.hpp"

using namespace std;

namespace yask {

    // Static registry.
    yc_solution_base::soln_map yc_solution_base::_soln_registry;
    
    // Stencil-solution APIs.
    yc_var_ptr StencilSolution::newVar(const std::string& name,
                                         bool isScratch,
                                         const std::vector<yc_index_node_ptr>& dims) {

        // Make new var and add to solution.
        // TODO: fix this mem leak--make smart ptr.

        // Copy pointers to concrete type.
        indexExprPtrVec dims2;
        for (auto d : dims) {
            auto d2 = dynamic_pointer_cast<IndexExpr>(d);
            assert(d2);
            dims2.push_back(d2);
        }

        auto* gp = new Var(name, isScratch, this, dims2);
        assert(gp);
        return gp;
    }

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
    yc_solution_base::yc_solution_base(const std::string& name) {
        if (_soln_registry.count(name)) {
            yask_exception e("Error: solution '" + name +
                             "' is already defined");
            throw e;
        }
        _soln = _yc_factory.new_solution(name);
        assert(_soln.get());
            
        // Add this new 'yc_solution_base' to the map.
        _soln_registry[name] = this;
    }
    yc_solution_base::yc_solution_base(yc_solution_base& base) {
        _soln = base.get_soln();
        assert(_soln.get());
    }
    void yc_solution_base::define() {
        yask_exception e("Error: no stencil equations are defined in solution '" +
                         get_soln()->get_name() +
                         "'. Implement the 'define()' method in the class "
                         "derived from 'yc_solution_base'");
        throw e;
    }
    void yc_solution_with_radius_base::define() {
        yask_exception e("Error: no stencil equations are defined in solution '" +
                         get_soln()->get_name() +
                         "'. Implement the 'define()' method in the class "
                         "derived from 'yc_solution_with_radius_base'");
        throw e;
    }
    
    // Create the intermediate data for printing.
    void StencilSolution::analyze_solution(int vlen,
                                           bool is_folding_efficient) {

        // Find all the stencil dimensions from the vars.
        // Create the final folds and clusters from the cmd-line options.
        _dims.setDims(_vars, _settings, vlen, is_folding_efficient, *_dos);

        // Determine which vars can be folded.
        _vars.setFolding(_dims);

        // Determine which var points can be vectorized and analyze inner-loop accesses.
        _eqs.analyzeVec(_dims);
        _eqs.analyzeLoop(_dims);

        // Find dependencies between equations.
        _eqs.analyzeEqs(_settings, _dims, *_dos);

        // Update access stats for the vars.
        _eqs.updateVarStats();

        // Create equation bundles based on dependencies and/or target strings.
        // This process may alter the halos in scratch vars.
        _eqBundles.set_basename_default(_settings._eq_bundle_basename_default);
        _eqBundles.set_dims(_dims);
        _eqBundles.makeEqBundles(_eqs, _settings, *_dos);

        // Optimize bundles.
        _eqBundles.optimizeEqBundles(_settings, "scalar & vector", false, *_dos);
        
        // Separate bundles into packs.
        _eqBundlePacks.makePacks(_eqBundles, *_dos);

        // Compute halos.
        _eqBundlePacks.calcHalos(_eqBundles);

        // Make a copy of each equation at each cluster offset.
        // We will use these for inter-cluster optimizations and code generation.
        // NB: these cluster bundles do not maintain dependencies, so cannot be used
        // for sorting, making packs, etc.
        *_dos << "\nConstructing cluster of equations containing " <<
            _dims._clusterMults.product() << " vector(s)...\n";
        _clusterEqBundles = _eqBundles;
        _clusterEqBundles.replicateEqsInCluster(_dims);
        if (_settings._doOptCluster)
            _clusterEqBundles.optimizeEqBundles(_settings, "cluster", true, *_dos);
    }

    // Format in given format-type.
    void StencilSolution::format(const string& format_type,
                                 yask_output_ptr output) {

        // Look for format match.
        // Most args to the printers just set references to data.
        // Data itself will be created in analyze_solution().
        PrinterBase* printer = 0;
        if (format_type == "cpp")
            printer = new YASKCppPrinter(*this, _eqBundles, _eqBundlePacks, _clusterEqBundles);
        else if (format_type == "knc")
            printer = new YASKKncPrinter(*this, _eqBundles, _eqBundlePacks, _clusterEqBundles);
        else if (format_type == "avx" || format_type == "avx2")
            printer = new YASKAvx256Printer(*this, _eqBundles, _eqBundlePacks, _clusterEqBundles);
        else if (format_type == "avx512" || format_type == "avx512f")
            printer = new YASKAvx512Printer(*this, _eqBundles, _eqBundlePacks, _clusterEqBundles);
        else if (format_type == "dot")
            printer = new DOTPrinter(*this, _clusterEqBundles, false);
        else if (format_type == "dot-lite")
            printer = new DOTPrinter(*this, _clusterEqBundles, true);
        else if (format_type == "pseudo")
            printer = new PseudoPrinter(*this, _clusterEqBundles, false);
        else if (format_type == "pseudo-long")
            printer = new PseudoPrinter(*this, _clusterEqBundles, true);
        else if (format_type == "pov-ray") // undocumented.
            printer = new POVRayPrinter(*this, _clusterEqBundles);
        else {
            THROW_YASK_EXCEPTION("Error: format-type '" + format_type +
                                 "' is not recognized");
        }
        assert(printer);
        int vlen = printer->num_vec_elems();
        bool is_folding_efficient = printer->is_folding_efficient();

        // Set data for equation bundles, dims, etc.
        analyze_solution(vlen, is_folding_efficient);

        // Create the output.
        *_dos << "\nGenerating '" << format_type << "' output...\n";
        printer->print(output->get_ostream());
        delete printer;
    }

} // namespace yask.

