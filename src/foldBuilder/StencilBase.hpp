/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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

#ifndef STENCIL_BASE
#define STENCIL_BASE

#include <map>
using namespace std;

namespace yask {

    typedef enum { STENCIL_CONTEXT } YASKSection;
    typedef vector<string> CodeList;
    typedef map<YASKSection, CodeList > ExtensionsList;

    class StencilSolution;
    class StencilBase;
    typedef map<string, StencilBase*> StencilList;

    // A base class for whole stencil solutions.  This is used by solutions
    // defined in C++ that are inherited from StencilBase as well as those
    // defined via the stencil-compiler API.
    class StencilSolution : public virtual stencil_solution {
    protected:
        
        // Simple name for the stencil.
        string _name;
    
        // A grid is an n-dimensional tensor that is indexed by grid indices.
        // Vectorization will be applied to grid accesses.
        Grids _grids;       // keep track of all registered grids.

        // A parameter is an n-dimensional tensor that is NOT indexed by grid indices.
        // It is used to pass some sort of index-invarant setting to a stencil function.
        // Its indices must be resolved when define() is called.
        // At this time, this is not checked, so be careful!!
        Params _params;     // keep track of all registered non-grid vars.

        // Code extensions that overload default functions from YASK in the
        // generated code for this solution.
        ExtensionsList _extensions;

        // All equations defined in this solution.
        Eqs _eqs;

        // Settings for the solution.
        StencilSettings _settings;

    private:

        // Intermediate data needed to format output.
        Dimensions _dims;       // various dimensions.
        EqGroups _eqGroups;     // eq-groups for scalar and vector.
        EqGroups _clusterEqGroups; // eq-groups for scalar and vector.
        ofstream _nullos;        // Dummy output stream.

        // Create the intermediate data.
        void analyze_solution(int vlen,
                              bool is_folding_efficient,
                              ostream& os);

    public:
        StencilSolution(const string& name) :
            _name(name) { }
        virtual ~StencilSolution() {}

        // Identification.
        virtual const string& getName() const { return _name; }
    
        // Simple accessors.
        virtual Grids& getGrids() { return _grids; }
        virtual Grids& getParams() { return _params; }
        virtual Eqs& getEqs() { return _eqs; }
        virtual StencilSettings& getSettings() { return _settings; }

        // Get user-provided code for the given section.
        CodeList * getExtensionCode(YASKSection section)
        { 
            auto elem = _extensions.find(section);
            if ( elem != _extensions.end() ) {
                return &elem->second;
            }
            return NULL;
        }

        // Define grid values relative to given offsets in each dimension.
        // This must be implemented by each concrete stencil solution.
        virtual void define(const IntTuple& offsets) = 0;

        // stencil_solution APIs.
        // See yask_stencil_api.hpp for documentation.
        virtual void set_name(std::string name) {
            _name = name;
        }
        virtual const std::string& get_name() const {
            return _name;
        }

        virtual grid_ptr new_grid(std::string name,
                                  std::string dim1 = "",
                                  std::string dim2 = "",
                                  std::string dim3 = "",
                                  std::string dim4 = "",
                                  std::string dim5 = "",
                                  std::string dim6 = "");

        virtual int get_num_grids() const {
            return int(_grids.size());
        }
        virtual grid_ptr get_grid(int n) {
            assert(n >= 0 && n < get_num_grids());
            return _grids.at(n);
        }
        
        virtual int get_num_equations() const {
            return _eqs.getNumEqs();
        }
        virtual equation_node_ptr get_equation(int n) {
            assert(n >= 0 && n < get_num_equations());
            return _eqs.getEqs().at(n);
        }
        virtual void set_fold(const std::string& dim, int len) {
            auto& fold = _settings._foldOptions;
            auto* p = fold.lookup(dim);
            if (p)
                *p = len;
            else
                fold.addDimBack(dim, len);
        }
        virtual void set_fold_len(const std::string& dim, int len);
        virtual void clear_folding() { _settings._foldOptions.clear(); }
        virtual void set_cluster_mult(const std::string& dim, int mult);
        virtual void clear_clustering() { _settings._clusterOptions.clear(); }
        virtual void set_step_dim(const std::string& dim) {
            _settings._stepDim = dim;
        }
        virtual const std::string& get_step_dim() const {
            return _settings._stepDim;
        }
        virtual void set_elem_bytes(int nbytes) { _settings._elem_bytes = nbytes; }
        virtual int get_elem_bytes() const { return _settings._elem_bytes; }
        virtual std::string format(const std::string& format_type, ostream& msg_stream);
        virtual std::string format(const std::string& format_type, bool debug) {
            return format(format_type, debug? cout : _nullos);
        }
        virtual void write(const std::string& filename,
                           const std::string& format_type,
                           bool debug);
    };

    // A stencil solution that does not define any grids.
    // This is used by a program via the compiler API to add grids
    // programmatically.
    class EmptyStencil : public StencilSolution {
    public:
        EmptyStencil(std::string name) :
            StencilSolution(name) { }
        virtual void define(const IntTuple& offsets) { }
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
            stencils[name] = this;
        }
        virtual ~StencilBase() { }

        // Return a reference to the main stencil-solution object.
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
#define REGISTER_CODE_EXTENSION(section,code) _extensions[section].push_back(code);
#define REGISTER_STENCIL_CONTEXT_EXTENSION(...) REGISTER_CODE_EXTENSION(STENCIL_CONTEXT,#__VA_ARGS__)

// Convenience macros for initializing existing grids from a class
// implementing StencilPart.  Each names the grid according to the 'gvar'
// parameter and adds it to the stencil solution object.  The dimensions are
// named according to the remaining parameters.
#define INIT_GRID_0D(gvar)                              \
    get_stencil_solution().getGrids().insert(&gvar);    \
    gvar.setName(#gvar);                                \
    gvar.setEqs(&get_stencil_solution().getEqs())
#define INIT_GRID_1D(gvar, d1)                  \
    INIT_GRID_0D(gvar); gvar.addDimBack(#d1, 1)
#define INIT_GRID_2D(gvar, d1, d2)                      \
    INIT_GRID_1D(gvar, d1); gvar.addDimBack(#d2, 1)
#define INIT_GRID_3D(gvar, d1, d2, d3)                  \
    INIT_GRID_2D(gvar, d1, d2); gvar.addDimBack(#d3, 1)
#define INIT_GRID_4D(gvar, d1, d2, d3, d4)                      \
    INIT_GRID_3D(gvar, d1, d2, d3); gvar.addDimBack(#d4, 1)
#define INIT_GRID_5D(gvar, d1, d2, d3, d4, d5)                  \
    INIT_GRID_4D(gvar, d1, d2, d3, d4); gvar.addDimBack(#d5, 1)
#define INIT_GRID_6D(gvar, d1, d2, d3, d4, d5, d6)                      \
    INIT_GRID_5D(gvar, d1, d2, d3, d4, d5); gvar.addDimBack(#d6, 1)

// Convenience macros for initializing parameters from a class implementing StencilPart.
// Each names the param according to the 'pvar' parameter and adds it
// to the stencil solution object.
// The dimensions are named and sized according to the remaining parameters.
#define INIT_PARAM(pvar)                                \
    get_stencil_solution().getParams().insert(&pvar);   \
    pvar.setName(#pvar);                                \
    pvar.setParam(true)
#define INIT_PARAM_1D(pvar, d1, s1)             \
    INIT_PARAM(pvar); pvar.addDimBack(#d1, s1)
#define INIT_PARAM_2D(pvar, d1, s1, d2, s2)                     \
    INIT_PARAM_1D(pvar, d1, s1); pvar.addDimBack(#d2, s2)
#define INIT_PARAM_3D(pvar, d1, s1, d2, s2, d3, s3)                     \
    INIT_PARAM_2D(pvar, d1, s1, d2, s2); pvar.addDimBack(#d3, s3)
#define INIT_PARAM_4D(pvar, d1, s1, d2, s2, d3, s3, d4, s4)             \
    INIT_PARAM_3D(pvar, d1, s1, d2, s2, d3, s3); pvar.addDimBack(#d4, d4)
#define INIT_PARAM_5D(pvar, d1, s1, d2, s2, d3, s3, d4, s4, d5, s5)     \
    INIT_PARAM_4D(pvar, d1, s1, d2, s2, d3, s3, d4, s4); pvar.addDimBack(#d5, d5)
#define INIT_PARAM_6D(pvar, d1, s1, d2, s2, d3, s3, d4, s4, d5, s5, d6, s6) \
    INIT_PARAM_4D(pvar, d1, s1, d2, s2, d3, s3, d4, s4, d5, s5); pvar.addDimBack(#d6, d6)

// Convenience macro for getting one offset from the 'offsets' tuple.
#define GET_OFFSET(ovar)                                                \
    NumExprPtr ovar = make_shared<IntScalarExpr>(offsets.getDim(#ovar))
 
#endif
