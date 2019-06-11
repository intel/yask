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

    void StencilSolution::_free(bool free_printer) {
        if (free_printer && _printer)
            delete _printer;
        if (_eqBundles)
            delete _eqBundles;
        if (_eqBundlePacks)
            delete _eqBundlePacks;
        if (_clusterEqBundles)
            delete _clusterEqBundles;
    }
    
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
    int StencilSolution::get_prefetch_dist(int level) {
        if (level < 1 || level > 2)
            THROW_YASK_EXCEPTION("Error: cache-level " +
                                 to_string(level) +
                                 " is not 1 or 2.");
        if (_settings._prefetchDists.count(level))
            return _settings._prefetchDists.at(level);
        else if (is_target_set()) {
            auto target = get_target();

            // Defaults for various targets.
            if (target == "knc") {
                if (level == 1)
                    return 1;
                else
                    return 2;
            }
            if (target == "knl") {
                if (level == 1)
                    return 1;
                else
                    return 0;
            }
            else {
                if (level == 1)
                    return 0;
                else
                    return 2;
            }
        }
        return 0;
    }
    void StencilSolution::set_prefetch_dist(int level,
                                            int distance) {
        get_prefetch_dist(level); // check legality.
        if (distance < 0)
            THROW_YASK_EXCEPTION("Error: prefetch-distance " +
                                 to_string(distance) +
                                 " is not positive.");
        _settings._prefetchDists[level] = distance;
    }
    yc_solution_base::soln_map& yc_solution_base::get_registry() {
        static yc_solution_base::soln_map* rp = 0;
        if (!rp)

            // Creates a global singleton.
            // Small memory leak because it's never deleted, but not important.
            rp = new yc_solution_base::soln_map;
        assert(rp);
        return *rp;
    }
    yc_solution_base::yc_solution_base(const std::string& name) {
        auto& reg = yc_solution_base::get_registry();
        if (reg.count(name)) {
            yask_exception e("Error: solution '" + name +
                             "' is already defined");
            throw e;
        }
        _soln = _yc_factory.new_solution(name);
        assert(_soln.get());

        // Add this new 'yc_solution_base' to the map.
        reg[name] = this;
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
        _eqBundles->set_basename_default(_settings._eq_bundle_basename_default);
        _eqBundles->set_dims(_dims);
        _eqBundles->makeEqBundles(_eqs, _settings, *_dos);

        // Optimize bundles.
        _eqBundles->optimizeEqBundles(_settings, "scalar & vector", false, *_dos);
        
        // Separate bundles into packs.
        _eqBundlePacks->makePacks(*_eqBundles, *_dos);

        // Compute halos.
        _eqBundlePacks->calcHalos(*_eqBundles);

        // Make a copy of each equation at each cluster offset.
        // We will use these for inter-cluster optimizations and code generation.
        // NB: these cluster bundles do not maintain dependencies, so cannot be used
        // for sorting, making packs, etc.
        *_dos << "\nConstructing cluster of equations containing " <<
            _dims._clusterMults.product() << " vector(s)...\n";
        *_clusterEqBundles = *_eqBundles;
        _clusterEqBundles->replicateEqsInCluster(_dims);
        if (_settings._doOptCluster)
            _clusterEqBundles->optimizeEqBundles(_settings, "cluster", true, *_dos);
    }

    // Set format.
    // Create new printer and intermediate data.
    void StencilSolution::set_target(const std::string& format) {
        auto& target = _settings._target;
        target = format;
        
        // Aliases for backward-compatibility.
        if (target == "cpp")
            target = "intel64";
        else if (target == "snb" || target == "ivb")
            target = "avx";
        else if (target == "hsw" || target == "bdw")
            target = "avx2";
        else if (target == "avx512f" || target == "skx" ||
                 target == "skl" || target == "clx")
            target = "avx512";

        // Ensure all intermediate data is clean.
        _free(true);
        _eqBundles = new EqBundles;
        _eqBundlePacks = new EqBundlePacks;
        _clusterEqBundles = new EqBundles;
        
        // Create the appropriate printer object based on the format.
        // Most args to the printers just set references to data.
        // Data itself will be created in analyze_solution().
        if (target == "intel64")
            _printer = new YASKCppPrinter(*this, *_eqBundles, *_eqBundlePacks, *_clusterEqBundles);
        else if (target == "knc")
            _printer = new YASKKncPrinter(*this, *_eqBundles, *_eqBundlePacks, *_clusterEqBundles);
        else if (target == "avx" || target == "avx2")
            _printer = new YASKAvx256Printer(*this, *_eqBundles, *_eqBundlePacks, *_clusterEqBundles);
        else if (target == "avx512" || target == "knl")
            _printer = new YASKAvx512Printer(*this, *_eqBundles, *_eqBundlePacks, *_clusterEqBundles);
        else if (target == "dot")
            _printer = new DOTPrinter(*this, *_clusterEqBundles, false);
        else if (target == "dot-lite")
            _printer = new DOTPrinter(*this, *_clusterEqBundles, true);
        else if (target == "pseudo")
            _printer = new PseudoPrinter(*this, *_clusterEqBundles, false);
        else if (target == "pseudo-long")
            _printer = new PseudoPrinter(*this, *_clusterEqBundles, true);
        else if (target == "pov-ray") // undocumented.
            _printer = new POVRayPrinter(*this, *_clusterEqBundles);
        else {
            _printer = 0;
            target = "";
            THROW_YASK_EXCEPTION("Error: format-target '" + format +
                                 "' is not recognized");
        }
        assert(_printer);
    }
    
    // Format in given format-type.
    void StencilSolution::output_solution(yask_output_ptr output) {
        auto& target = _settings._target;

        if (!is_target_set())
            THROW_YASK_EXCEPTION("Error: output_solution() called before set_target()");

        // Call set_target() to ensure intermediate data is clean
        // before calling analyze_solution().
        set_target(target);
        
        // Set data for equation bundles, dims, etc.
        int vlen = _printer->num_vec_elems();
        bool is_folding_efficient = _printer->is_folding_efficient();
        analyze_solution(vlen, is_folding_efficient);

        // Call hooks.
        for (auto& hook : _output_hooks)
            hook(*this, output);
        
        // Create the output.
        *_dos << "\nGenerating '" << target << "' output...\n";
        _printer->print(output->get_ostream());
    }

} // namespace yask.

