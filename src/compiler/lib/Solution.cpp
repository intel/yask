/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2022, Intel Corporation

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
        if (_eq_bundles)
            delete _eq_bundles;
        if (_eq_stages)
            delete _eq_stages;
        if (_cluster_eq_bundles)
            delete _cluster_eq_bundles;
    }
    
    // Stencil-solution APIs.
    yc_var_ptr StencilSolution::new_var(const std::string& name,
                                         bool is_scratch,
                                         const std::vector<yc_index_node_ptr>& dims) {

        // Make new var and add to solution.
        // TODO: fix this mem leak--make smart ptr.

        // Copy pointers to concrete type.
        index_expr_ptr_vec dims2;
        for (auto d : dims) {
            auto d2 = dynamic_pointer_cast<IndexExpr>(d);
            assert(d2);
            dims2.push_back(d2);
        }

        auto* gp = new Var(name, is_scratch, this, dims2);
        assert(gp);
        return gp;
    }

    void StencilSolution::set_fold_len(const yc_index_node_ptr dim,
                                       int len) {
        auto& fold = _settings._fold_options;
        fold.add_dim_back(dim->get_name(), len);
    }
    void StencilSolution::set_cluster_mult(const yc_index_node_ptr dim,
                                           int mult) {
        auto& cluster = _settings._cluster_options;
        cluster.add_dim_back(dim->get_name(), mult);
    }
    int StencilSolution::get_prefetch_dist(int level) {
        if (level < 1 || level > 2)
            THROW_YASK_EXCEPTION("cache-level " +
                                 to_string(level) +
                                 " is not 1 or 2.");
        if (_settings._prefetch_dists.count(level))
            return _settings._prefetch_dists.at(level);
        return 0;
    }
    void StencilSolution::set_prefetch_dist(int level,
                                            int distance) {
        get_prefetch_dist(level); // check legality.
        if (distance < 0)
            THROW_YASK_EXCEPTION("prefetch-distance " +
                                 to_string(distance) +
                                 " is not zero or positive.");
        _settings._prefetch_dists[level] = distance;
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

        // Find all the stencil dimensions in the settings and/or vars.
        // Create the final folds and clusters.
        _dims.set_dims(_vars, _settings, vlen, is_folding_efficient, *_dos);

        // Count dim types in each var and determine foldability.
        _vars.set_dim_counts(_dims);

        // Determine which var points can be vectorized and analyze inner-loop accesses.
        _eqs.analyze_vec(_settings, _dims);
        _eqs.analyze_loop(_settings, _dims);

        // Find dependencies between equations.
        _eqs.analyze_eqs(_settings, _dims, *_dos);

        // Update access stats for the vars.
        _eqs.update_var_stats();

        // Create equation bundles based on dependencies and/or target strings.
        // This process may alter the halos in scratch vars.
        _eq_bundles->set_basename_default(_settings._eq_bundle_basename_default);
        _eq_bundles->set_dims(_dims);
        _eq_bundles->make_eq_bundles(_eqs, _settings, *_dos);

        // Optimize bundles.
        _eq_bundles->optimize_eq_bundles(_settings, "scalar & vector", false, *_dos);
        
        // Separate bundles into stages.
        _eq_stages->make_stages(*_eq_bundles, *_dos);

        // Compute halos.
        _eq_stages->calc_halos(*_eq_bundles);

        // Make a copy of each equation at each cluster offset.
        // We will use these for inter-cluster optimizations and code generation.
        // NB: these cluster bundles do not maintain dependencies, so cannot be used
        // for sorting, making stages, etc.
        *_dos << "\nConstructing cluster of equations containing " <<
            _dims._cluster_mults.product() << " vector(s)...\n";
        *_cluster_eq_bundles = *_eq_bundles;
        _cluster_eq_bundles->replicate_eqs_in_cluster(_dims);
        if (_settings._do_opt_cluster)
            _cluster_eq_bundles->optimize_eq_bundles(_settings, "cluster", true, *_dos);
    }

    // Set options as if command-line.
    string StencilSolution::apply_command_line_options(const string& argstr) {
        auto args = CommandLineParser::set_args(argstr);
        return apply_command_line_options(args);
    }

    string StencilSolution::apply_command_line_options(int argc, char* argv[]) {
        string_vec args;
        for (int i = 1; i < argc; i++)
            args.push_back(argv[i]);
        return apply_command_line_options(args);
    }

    string StencilSolution::apply_command_line_options(const vector<string>& args) {
        string rem;

        // Create a parser and add options to it.
        CommandLineParser parser;
        _settings.add_options(parser);
        
        // Parse cmd-line options, which sets values in _settings.
        rem = parser.parse_args("YASK", args);
        return rem;
    }

    // Get help.
    std::string StencilSolution::get_command_line_help() {

        // Create a parser and add options to it.
        CommandLineParser parser;
        _settings.add_options(parser);

        std::stringstream sstr;
        parser.print_help(sstr);
        return sstr.str();
    }
    std::string StencilSolution::get_command_line_values() {

        // Create a parser and add options to it.
        CommandLineParser parser;
        _settings.add_options(parser);

        std::stringstream sstr;
        parser.print_values(sstr);
        return sstr.str();
    }
    
    
    // Format in given format-type.
    void StencilSolution::output_solution(yask_output_ptr output) {

        // Ensure all intermediate data is clean.
        _free(true);
        _eq_bundles = new EqBundles;
        _eq_stages = new EqStages;
        _cluster_eq_bundles = new EqBundles;

        if (!is_target_set())
            THROW_YASK_EXCEPTION("output_solution() without format target being set");
        string target = _settings._target;

        // Aliases for backward-compatibility.
        if (target == "cpp")
            target = "intel64";
        else if (target == "snb" || target == "ivb")
            target = "avx";
        else if (target == "hsw" || target == "bdw")
            target = "avx2";
        else if (target == "avx512f" || target == "skx" ||
                 target == "skl" || target == "clx" ||
                 target == "icx" || target == "spr" ||
                 target == "avx512-zmm" || target == "avx512hi")
            target = "avx512";
        else if (target == "avx512lo")
            target = "avx512-ymm";

        // Create the appropriate printer object based on the format.
        // Most args to the printers just set references to data.
        // Data itself will be created in analyze_solution().
        if (target == "intel64")
            _printer = new YASKCppPrinter(*this, *_eq_bundles, *_eq_stages, *_cluster_eq_bundles);
        else if (target == "avx" || target == "avx2")
            _printer = new YASKAvx2Printer(*this, *_eq_bundles, *_eq_stages, *_cluster_eq_bundles);
        else if (target == "avx512" || target == "knl")
            _printer = new YASKAvx512Printer(*this, *_eq_bundles, *_eq_stages, *_cluster_eq_bundles);
        else if (target == "avx512-ymm")
            _printer = new YASKAvx512Printer(*this, *_eq_bundles, *_eq_stages, *_cluster_eq_bundles, true);
        else if (target == "dot")
            _printer = new DOTPrinter(*this, *_cluster_eq_bundles, false);
        else if (target == "dot-lite")
            _printer = new DOTPrinter(*this, *_cluster_eq_bundles, true);
        else if (target == "pseudo")
            _printer = new PseudoPrinter(*this, *_cluster_eq_bundles, false);
        else if (target == "pseudo-long")
            _printer = new PseudoPrinter(*this, *_cluster_eq_bundles, true);
        else if (target == "pov-ray") // undocumented.
            _printer = new POVRayPrinter(*this, *_cluster_eq_bundles);
        else {
            _printer = 0;
            THROW_YASK_EXCEPTION("format-target '" + target +
                                 "' is not recognized");
        }
        assert(_printer);
                
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

