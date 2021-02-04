/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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

#pragma once

// Generation code.
#include "ExprUtils.hpp"
#include "Settings.hpp"
#include "Eqs.hpp"

using namespace std;

namespace yask {
    class PrinterBase;

    // A base class for whole stencil solutions.  This is used by solutions
    // defined in C++ that are inherited from StencilBase as well as those
    // defined via the stencil-compiler API.
    class StencilSolution :
        public virtual yc_solution {
    protected:

        // Simple name for the stencil soln. Must be a legal C++ var name.
        string _name;

        // Longer descriptive string.
        string _long_name;

        // Debug output.
        yask_output_ptr _debug_output;
        ostream* _dos = &std::cout; // just a handy pointer to an ostream.

        // All vars accessible by the kernel.
        Vars _vars;

        // All equations defined in this solution.
        Eqs _eqs;

        // Settings for the solution.
        CompilerSettings _settings;

        // Code extensions.
        vector<string> _kernel_code;
        vector<output_hook_t> _output_hooks;

    private:

        // Intermediate data needed to format output.
        Dimensions _dims;          // various dimensions.
        PrinterBase* _printer = 0;
        EqBundles* _eq_bundles = 0;         // eq-bundles for scalar and vector.
        EqStages* _eq_stages = 0; // packs of bundles w/o inter-dependencies.
        EqBundles* _cluster_eq_bundles = 0;  // eq-bundles for scalar and vector.

        // Create the intermediate data.
        void analyze_solution(int vlen,
                              bool is_folding_efficient);

        // Free allocated objs.
        void _free(bool free_printer);

    public:
        StencilSolution(const string& name) : _name(name) { }
        virtual ~StencilSolution() { _free(true); }

        // Identification.
        virtual const string& _get_name() const { return _name; }
        virtual const string& get_long_name() const {
            return _long_name.length() ? _long_name : _name;
        }

        // Simple accessors.
        virtual Vars& _get_vars() { return _vars; }
        virtual Eqs& get_eqs() { return _eqs; }
        virtual CompilerSettings& get_settings() { return _settings; }
        virtual void set_settings(const CompilerSettings& settings) {
            _settings = settings;
        }
        virtual const Dimensions& get_dims() { return _dims; }
        virtual const vector<string>& get_kernel_code() { return _kernel_code; }

        // Get the messsage output stream.
        virtual std::ostream& get_ostr() const {
            assert(_dos);
            return *_dos;
        }

        // Make a new var.
        virtual yc_var_ptr new_var(const std::string& name,
                                    bool is_scratch,
                                    const std::vector<yc_index_node_ptr>& dims);

        // stencil_solution APIs.
        // See yask_stencil_api.hpp for documentation.
        virtual void set_debug_output(yask_output_ptr debug) override {
            _debug_output = debug;     // to share ownership of referent.
            _dos = &_debug_output->get_ostream();
        }
        virtual yask_output_ptr get_debug_output() {
            if (!_debug_output.get()) {
                yask_output_factory ofac;
                auto so = ofac.new_stdout_output();
                set_debug_output(so);
            }
            assert(_debug_output.get());
            return _debug_output;
        }
        virtual void set_name(std::string name) override {
            _name = name;
        }
        virtual void set_description(std::string str) override {
            _long_name = str;
        }
        virtual std::string get_name() const override {
            return _name;
        }
        virtual std::string get_description() const override {
            return get_long_name();
        }

        virtual yc_var_ptr new_var(const std::string& name,
                                     const std::vector<yc_index_node_ptr>& dims) override {
            return new_var(name, false, dims);
        }
        virtual yc_var_ptr new_var(const std::string& name,
                                     const std::initializer_list<yc_index_node_ptr>& dims) override {
            std::vector<yc_index_node_ptr> dim_vec(dims);
            return new_var(name, false, dim_vec);
        }
        virtual yc_var_ptr new_scratch_var(const std::string& name,
                                             const std::vector<yc_index_node_ptr>& dims) override {
            return new_var(name, true, dims);
        }
        virtual yc_var_ptr new_scratch_var(const std::string& name,
                                             const std::initializer_list<yc_index_node_ptr>& dims) override {
            std::vector<yc_index_node_ptr> dim_vec(dims);
            return new_var(name, true, dim_vec);
        }
        virtual int get_num_vars() const override {
            return int(_vars.size());
        }
        virtual yc_var_ptr get_var(const std::string& name) override {
            for (int i = 0; i < get_num_vars(); i++)
                if (_vars.at(i)->_get_name() == name)
                    return _vars.at(i);
            return nullptr;
        }
        virtual std::vector<yc_var_ptr> get_vars() override {
            std::vector<yc_var_ptr> gv;
            for (int i = 0; i < get_num_vars(); i++)
                gv.push_back(_vars.at(i));
            return gv;
        }

        virtual int get_num_equations() const override {
            return _eqs.get_num();
        }
        virtual std::vector<yc_equation_node_ptr> get_equations() override {
            std::vector<yc_equation_node_ptr> ev;
            for (int i = 0; i < get_num_equations(); i++)
                ev.push_back(_eqs.get_all().at(i));
            return ev;
        }
        virtual void
        call_after_new_solution(const string& code) override {
            _kernel_code.push_back(code);
        }
        virtual int
        get_prefetch_dist(int level) override ;
        virtual void
        set_prefetch_dist(int level,
                          int distance) override;
        virtual void add_flow_dependency(yc_equation_node_ptr from,
                                         yc_equation_node_ptr to) override {
            auto fp = dynamic_pointer_cast<EqualsExpr>(from);
            assert(fp);
            auto tp = dynamic_pointer_cast<EqualsExpr>(to);
            assert(tp);
            _eqs.get_deps().set_imm_dep_on(fp, tp);
        }
        virtual void clear_dependencies() override {
            _eqs.get_deps().clear_deps();
        }

        virtual void set_fold_len(const yc_index_node_ptr, int len) override;
        virtual bool is_folding_set() override {
            return _settings._fold_options.size() > 0;
        }
        virtual void clear_folding() override {
            _settings._fold_options.clear();
        }
        virtual void set_cluster_mult(const yc_index_node_ptr, int mult) override;
        virtual bool is_clustering_set() override {
            return _settings._cluster_options.size() > 0;
        }
        virtual void clear_clustering() override {
            _settings._cluster_options.clear();
        }

        virtual bool is_target_set() override {
            return _settings._target.length() > 0;
        }
        virtual std::string get_target() override {
            if (!is_target_set())
                THROW_YASK_EXCEPTION("Error: call to get_target() before set_target()");
            return _settings._target;
        }
        virtual void set_target(const std::string& format) override;
        virtual void set_element_bytes(int nbytes) override {
            _settings._elem_bytes = nbytes;
        }
        virtual int get_element_bytes() const override {
            return _settings._elem_bytes;
        }

        virtual bool is_dependency_checker_enabled() const override {
            return _settings._find_deps;
        }
        virtual void set_dependency_checker_enabled(bool enable) override {
            _settings._find_deps = enable;
        }

        virtual void output_solution(yask_output_ptr output) override;
        virtual void
        call_before_output(output_hook_t hook_fn) override {
                _output_hooks.push_back(hook_fn);
        }
        virtual void
        set_domain_dims(const std::vector<yc_index_node_ptr>& dims) override {
            _settings._domain_dims.clear();
            for (auto& d : dims) {
                auto dp = dynamic_pointer_cast<IndexExpr>(d);
                assert(dp);
                auto& dname = d->get_name();
                if (dp->get_type() != DOMAIN_INDEX)
                    THROW_YASK_EXCEPTION("Error: set_domain_dims() called with non-domain index '" +
                                         dname + "'");
                _settings._domain_dims.push_back(dname);
            }
        }
        virtual void
        set_domain_dims(const std::initializer_list<yc_index_node_ptr>& dims) override {
            vector<yc_index_node_ptr> vdims(dims);
            set_domain_dims(vdims);
        }
        virtual void
        set_step_dim(const yc_index_node_ptr dim) override {
            auto dp = dynamic_pointer_cast<IndexExpr>(dim);
            assert(dp);
            auto& dname = dim->get_name();
            if (dp->get_type() != STEP_INDEX)
                    THROW_YASK_EXCEPTION("Error: set_step_dim() called with non-step index '" +
                                         dname + "'");
            _settings._step_dim = dname;
        }
        
    };

} // namespace yask.
