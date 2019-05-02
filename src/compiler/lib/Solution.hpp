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

#pragma once

// Generation code.
#include "ExprUtils.hpp"
#include "Settings.hpp"
#include "Eqs.hpp"

using namespace std;

namespace yask {

    // TODO: add API to add to this.
    typedef enum { STENCIL_CONTEXT } YASKSection;
    typedef vector<string> CodeList;
    typedef map<YASKSection, CodeList > ExtensionsList;

// Convenience macros for adding 'extension' code to a stencil.
#define _REGISTER_CODE_EXTENSION(section, code) _extensions[section].push_back(code);
#define _REGISTER_STENCIL_CONTEXT_EXTENSION(...) REGISTER_CODE_EXTENSION(STENCIL_CONTEXT, #__VA_ARGS__)

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
        Grids _grids;

        // All equations defined in this solution.
        Eqs _eqs;

        // Settings for the solution.
        CompilerSettings _settings;

        // Code extensions that overload default functions from YASK in the
        // generated code for this solution.
        ExtensionsList _extensions;

    private:

        // Intermediate data needed to format output.
        Dimensions _dims;             // various dimensions.
        EqBundles _eqBundles;         // eq-bundles for scalar and vector.
        EqBundlePacks _eqBundlePacks; // packs of bundles w/o inter-dependencies.
        EqBundles _clusterEqBundles;  // eq-bundles for scalar and vector.

        // Create the intermediate data.
        void analyze_solution(int vlen,
                              bool is_folding_efficient);

    public:
        StencilSolution(const string& name) :
            _name(name) {
            yask_output_factory ofac;
            auto so = ofac.new_stdout_output();
            set_debug_output(so);
        }
        virtual ~StencilSolution() {}

        // Identification.
        virtual const string& getName() const { return _name; }
        virtual const string& getLongName() const {
            return _long_name.length() ? _long_name : _name;
        }

        // Simple accessors.
        virtual Grids& getGrids() { return _grids; }
        virtual Eqs& getEqs() { return _eqs; }
        virtual const CompilerSettings& getSettings() { return _settings; }
        virtual void setSettings(const CompilerSettings& settings) {
            _settings = settings;
        }
        virtual const Dimensions& getDims() { return _dims; }

        // Get user-provided code for the given section.
        CodeList * getExtensionCode(YASKSection section)
        {
            auto elem = _extensions.find(section);
            if ( elem != _extensions.end() ) {
                return &elem->second;
            }
            return NULL;
        }

        // Get the messsage output stream.
        virtual std::ostream& get_ostr() const {
            assert(_dos);
            return *_dos;
        }

        // Make a new grid.
        virtual yc_grid_ptr newGrid(const std::string& name,
                                    bool isScratch,
                                    const std::vector<yc_index_node_ptr>& dims);

        // stencil_solution APIs.
        // See yask_stencil_api.hpp for documentation.
        virtual void set_debug_output(yask_output_ptr debug) {
            _debug_output = debug;     // to share ownership of referent.
            _dos = &_debug_output->get_ostream();
        }
        virtual yask_output_ptr get_debug_output() const {
            return _debug_output;
        }
        virtual void set_name(std::string name) {
            _name = name;
        }
        virtual void set_description(std::string str) {
            _long_name = str;
        }
        virtual std::string get_name() const {
            return _name;
        }
        virtual std::string get_description() const {
            return getLongName();
        }

        virtual yc_grid_ptr new_grid(const std::string& name,
                                     const std::vector<yc_index_node_ptr>& dims) {
            return newGrid(name, false, dims);
        }
        virtual yc_grid_ptr new_grid(const std::string& name,
                                     const std::initializer_list<yc_index_node_ptr>& dims) {
            std::vector<yc_index_node_ptr> dim_vec(dims);
            return newGrid(name, false, dim_vec);
        }
        virtual yc_grid_ptr new_scratch_grid(const std::string& name,
                                             const std::vector<yc_index_node_ptr>& dims) {
            return newGrid(name, true, dims);
        }
        virtual yc_grid_ptr new_scratch_grid(const std::string& name,
                                             const std::initializer_list<yc_index_node_ptr>& dims) {
            std::vector<yc_index_node_ptr> dim_vec(dims);
            return newGrid(name, true, dim_vec);
        }
        virtual int get_num_grids() const {
            return int(_grids.size());
        }
        virtual yc_grid_ptr get_grid(const std::string& name) {
            for (int i = 0; i < get_num_grids(); i++)
                if (_grids.at(i)->getName() == name)
                    return _grids.at(i);
            return nullptr;
        }
        virtual std::vector<yc_grid_ptr> get_grids() {
            std::vector<yc_grid_ptr> gv;
            for (int i = 0; i < get_num_grids(); i++)
                gv.push_back(_grids.at(i));
            return gv;
        }

        virtual int get_num_equations() const {
            return _eqs.getNum();
        }
        virtual std::vector<yc_equation_node_ptr> get_equations() {
            std::vector<yc_equation_node_ptr> ev;
            for (int i = 0; i < get_num_equations(); i++)
                ev.push_back(_eqs.getAll().at(i));
            return ev;
        }
        virtual void add_flow_dependency(yc_equation_node_ptr from,
                                         yc_equation_node_ptr to) {
            auto fp = dynamic_pointer_cast<EqualsExpr>(from);
            assert(fp);
            auto tp = dynamic_pointer_cast<EqualsExpr>(to);
            assert(tp);
            _eqs.getDeps().set_imm_dep_on(fp, tp);
        }
        virtual void clear_dependencies() {
            _eqs.getDeps().clear_deps();
        }

        virtual void set_fold_len(const yc_index_node_ptr, int len);
        virtual void clear_folding() { _settings._foldOptions.clear(); }
        virtual void set_cluster_mult(const yc_index_node_ptr, int mult);
        virtual void clear_clustering() { _settings._clusterOptions.clear(); }

        virtual void set_element_bytes(int nbytes) { _settings._elem_bytes = nbytes; }
        virtual int get_element_bytes() const { return _settings._elem_bytes; }

        virtual bool is_dependency_checker_enabled() const { return _settings._findDeps; }
        virtual void set_dependency_checker_enabled(bool enable) { _settings._findDeps = enable; }

        virtual void format(const std::string& format_type,
                            yask_output_ptr output);
        virtual void
        set_domain_dims(const std::vector<yc_index_node_ptr>& dims) {
            _settings._domainDims.clear();
            for (auto& d : dims) {
                auto dp = dynamic_pointer_cast<IndexExpr>(d);
                assert(dp);
                auto& dname = d->get_name();
                if (dp->getType() != DOMAIN_INDEX)
                    THROW_YASK_EXCEPTION("Error: set_domain_dims() called with non-domain index '" +
                                         dname + "'");
                _settings._domainDims.push_back(dname);
            }
        }
        virtual void
        set_domain_dims(const std::initializer_list<yc_index_node_ptr>& dims) {
            vector<yc_index_node_ptr> vdims(dims);
            set_domain_dims(vdims);
        }
        virtual void
        set_step_dim(const yc_index_node_ptr dim) {
            auto dp = dynamic_pointer_cast<IndexExpr>(dim);
            assert(dp);
            auto& dname = dim->get_name();
            if (dp->getType() != STEP_INDEX)
                    THROW_YASK_EXCEPTION("Error: set_step_dim() called with non-step index '" +
                                         dname + "'");
            _settings._stepDim = dname;
        }
        
    };

} // namespace yask.
