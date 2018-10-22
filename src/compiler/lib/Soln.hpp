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

#ifndef SOLN_HPP
#define SOLN_HPP

// Generation code.
#include "ExprUtils.hpp"
#include "Grid.hpp"
#include "Eqs.hpp"

using namespace std;

namespace yask {

    typedef enum { STENCIL_CONTEXT } YASKSection;
    typedef vector<string> CodeList;
    typedef map<YASKSection, CodeList > ExtensionsList;

    class StencilSolution;
    class StencilBase;
    typedef map<string, StencilBase*> StencilList;

    // Global list for registering stencils.
    extern StencilList stencils;

    // A base class for whole stencil solutions.  This is used by solutions
    // defined in C++ that are inherited from StencilBase as well as those
    // defined via the stencil-compiler API.
    class StencilSolution :
        public virtual yc_solution {
    protected:

        // Simple name for the stencil soln.
        string _name;

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
            set_debug_output(ofac.new_stdout_output());
        }
        virtual ~StencilSolution() {}

        // Identification.
        virtual const string& getName() const { return _name; }

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

        // Define grid values relative to current domain indices in each dimension.
        // This must be implemented by each concrete stencil solution.
        virtual void define() = 0;

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
        virtual const std::string& get_name() const {
            return _name;
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
    };

    // A stencil solution that does not define any grids.
    // This is used by a program via the compiler API to add grids
    // programmatically.
    class EmptyStencil : public StencilSolution {

        // Do not define any dims.
        // Do not define any grids.

    public:
        EmptyStencil(std::string name) :
            StencilSolution(name) { }

        // Do not define any equations.
        virtual void define() { }
    };

    // An interface for all objects that participate in stencil definitions.
    // This allows a programmer to use object composition in addition to
    // inheritance from StencilBase to define stencils.
    class StencilPart {

    public:
        StencilPart() {}
        virtual ~StencilPart() {}

        // Return a reference to the main stencil object.
        virtual StencilSolution& get_stencil_solution() =0;
    };

    // The class all C++ stencil solutions must implement.
    class StencilBase : public StencilSolution,
                        public StencilPart {

    public:
        // Initialize name and register this new object in a list.
        StencilBase(const string name, StencilList& stencils) :
            StencilSolution(name)
        {
            if (stencils.count(name))
                THROW_YASK_EXCEPTION("Error: stencil '" + name +
                                     "' already defined");
            stencils[name] = this;
        }
        virtual ~StencilBase() { }

        // Return a reference to the main stencil-solution object.
        // For StencilBase, simply this object.
        virtual StencilSolution& get_stencil_solution() {
            return *this;
        }

        // Radius stub methods.
        virtual bool usesRadius() const { return false; }
        virtual bool setRadius(int radius) { return false; }
        virtual int getRadius() const { return 0; }

    };

    // A base class for stencils that have a 'radius'.
    class StencilRadiusBase : public StencilBase {
    protected:
        int _radius;         // stencil radius.

    public:
        StencilRadiusBase(const string name, StencilList& stencils, int radius) :
            StencilBase(name, stencils), _radius(radius) {}

        // Does use radius.
        virtual bool usesRadius() const { return true; }

        // Set radius.
        // Return true if successful.
        virtual bool setRadius(int radius) {
            _radius = radius;
            return radius >= 0;  // support only non-neg. radius.
        }

        // Get radius.
        virtual int getRadius() { return _radius; }
    };

} // namespace yask.

// Convenience macro for registering a stencil in a list.
#define REGISTER_STENCIL(Class) static Class registered_ ## Class(stencils)

// Convenience macros for adding 'extension' code to a stencil.
#define REGISTER_CODE_EXTENSION(section, code) _extensions[section].push_back(code);
#define REGISTER_STENCIL_CONTEXT_EXTENSION(...) REGISTER_CODE_EXTENSION(STENCIL_CONTEXT, #__VA_ARGS__)

// Convenience macro for declaring dims.
#define MAKE_STEP_INDEX(d) IndexExprPtr d = make_shared<IndexExpr>(#d, STEP_INDEX);
#define MAKE_DOMAIN_INDEX(d) IndexExprPtr d = make_shared<IndexExpr>(#d, DOMAIN_INDEX);
#define MAKE_MISC_INDEX(d) IndexExprPtr d = make_shared<IndexExpr>(#d, MISC_INDEX);

// Convenience macros for creating grids in a class implementing StencilPart.
// The 'gvar' arg is the var name and the grid name.
// The remaining args are the dimension names.
#define MAKE_GRID(gvar, ...)                                            \
    Grid gvar = Grid(#gvar, false, &get_stencil_solution(), ##__VA_ARGS__)
#define MAKE_SCRATCH_GRID(gvar, ...)                                    \
    Grid gvar = Grid(#gvar, true, &get_stencil_solution(), ##__VA_ARGS__)
#define MAKE_SCALAR(gvar) MAKE_GRID(gvar)
#define MAKE_ARRAY(gvar, d1) MAKE_GRID(gvar, d1)

#endif
