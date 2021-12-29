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

////////// Support for YASK C++ scalar and vector-code generation //////////////

// See CppIntrin.hpp for explicit intrinsic generation.

#pragma once

#include "Vec.hpp"
#include "Var.hpp"

namespace yask {
    
    /////////// Scalar code /////////////

    // Outputs C++ scalar code for YASK.
    class CppPrintHelper : public PrintHelper {

    public:
        static constexpr const char* _var_ptr_type = "auto*";
        static constexpr const char* _var_ptr_restrict_type = "auto* __restrict";
        static constexpr const char* _step_val_type = "const auto";

        CppPrintHelper(const CompilerSettings& settings,
                       const Dimensions& dims,
                       const CounterVisitor* cv,
                       const string& var_type,
                       const string& line_prefix,
                       const string& line_suffix) :
            PrintHelper(settings, dims, cv, var_type, line_prefix, line_suffix) { }
        virtual ~CppPrintHelper() { }

        // Format a real, preserving precision.
        static string format_real(double v);

        // Return a constant expression.
        // This is overloaded to preserve precision.
        virtual string add_const_expr(ostream& os, double v) override {
            return format_real(v);
        }

        // Format a pointer to a var.
        virtual string get_var_ptr(const Var& var) {
            string gname = var._get_name();
            string expr;
            if (var.is_scratch())
                expr = "thread_core_data.";
            else
                expr = "core_data->";
            expr += "var_" + gname + "_core_p.get()";
            return expr;
        }
        virtual string get_var_ptr(const VarPoint& gp) {
            const auto* var = gp._get_var();
            assert(var);
            return get_var_ptr(*var);
        }        
        
        // Make call for a point.
        // This is a utility function used for both reads and writes.
        virtual string make_point_call(ostream& os,
                                       const VarPoint& gp,
                                       const string& fname,
                                       string opt_arg = "");

        // Return a var-point reference.
        virtual string read_from_point(ostream& os, const VarPoint& gp) override;

        // Return code to update a var point.
        virtual string write_to_point(ostream& os, const VarPoint& gp,
                                      const string& val) override;
    };

    /////////// Vector code /////////////

    // Output generic C++ vector code for YASK.
    class CppVecPrintHelper : public CppPrintHelper,
                              public VecPrintHelper {

    public:
        CppVecPrintHelper(VecInfoVisitor& vv,
                          const CompilerSettings& settings,
                          const Dimensions& dims,
                          const CounterVisitor* cv,
                          const string& var_type,
                          const string& line_prefix,
                          const string& line_suffix) :
            CppPrintHelper(settings, dims, cv,
                           var_type, line_prefix, line_suffix),
            VecPrintHelper(vv) { }
        
    protected:

        // Name of ptr to lowest-allocated vec for a given point in that var.
        // There is a unique ptr for each step-arg per var.
        // Thus, there is a many->one mapping for points that vary only by domain and/or misc indices.
        // Key: point expr; value: ptr-var name.
        map<VarPoint, string> _var_base_ptrs;

        // Name of ptr to current point in inner-loop dim.
        // Inner-layout dim index has no offset, and misc-dim indices are at min-value.
        // Key: point expr; value: ptr-var name.
        map<VarPoint, string> _inner_loop_base_ptrs;
 
        // Vars for tracking other info about vars.
        typedef pair<string, string> VarDimKey; // var and dim names.
        map<VarDimKey, string> _strides; // var containing stride expr for given dim in var.
        map<VarDimKey, string> _offsets; // var containing offset expr for given dim in var.
        map<string, string> _ptr_ofs; // var containing const offset expr for key var.
        map<string, int> _ptr_ofs_lo; // lowest read offset from var in inner loop dim. FIXME?
        map<string, int> _ptr_ofs_hi; // highest read offset from var in inner loop dim. FIXME?

        // Element indices.
        string _elem_suffix_global = "_global_elem";
        string _elem_suffix_local = "_local_elem";
        VarMap _vec2elem_local_map, _vec2elem_global_map;

        // Rank vars.
        string _rank_domain_offset_prefix = "rank_domain_offset_";

        // Set to var name of write mask if/when used.
        string _write_mask = "";

        // A simple constant.
        virtual string add_const_expr(ostream& os, double v) override {
            return CppPrintHelper::format_real(v);
        }

        // Any code.
        virtual string add_code_expr(ostream& os, const string& code) override {
            return code;
        }

        // Print a comment about a point.
        // This is a utility function used for both reads and writes.
        virtual void print_point_comment(ostream& os, const VarPoint& gp,
                                         const string& verb) const {

            os << endl << " // " << verb << " at " <<
                gp.make_str() << "." << endl;
        }

        // Return code for a var function call at a point.
        // This is a utility function used for both reads and writes.
        virtual string make_point_call_vec(ostream& os,
                                           const VarPoint& gp,
                                           const string& func_name,
                                           const string& first_arg,
                                           const string& last_arg,
                                           bool is_vector_normalized,
                                           const VarMap* var_map = 0);

        // Print aligned memory read.
        virtual string print_aligned_vec_read(ostream& os, const VarPoint& gp) override;

        // Print unaliged memory read.
        // Assumes this results in same values as print_unaligned_vec().
        virtual string print_unaligned_vec_read(ostream& os, const VarPoint& gp) override;

        // Print aligned memory write.
        virtual void print_aligned_vec_write(ostream& os, const VarPoint& gp,
                                             const string& val) override;

        // Print conversion from memory vars to point var gp if needed.
        // This calls print_unaligned_vec_ctor(), which can be overloaded
        // by derived classes.
        virtual string print_unaligned_vec(ostream& os, const VarPoint& gp) override;

        // Print per-element construction for one point var pv_name from elems.
        virtual void print_unaligned_vec_simple(ostream& os, const VarPoint& gp,
                                             const string& pv_name, string line_prefix,
                                             const set<size_t>* done_elems = 0);

        // Read from a single point to be broadcast to a vector.
        // Return code for read.
        virtual string read_from_scalar_point(ostream& os, const VarPoint& gp,
                                              const VarMap* var_map) override;

        // Read from multiple points that are not vectorizable.
        // Return var name.
        virtual string print_partial_vec_read(ostream& os, const VarPoint& gp) override;

        // Print construction for one point var pv_name from elems.
        // This version prints inefficient element-by-element assignment.
        // Override this in derived classes for more efficient implementations.
        virtual void print_unaligned_vec_ctor(ostream& os, const VarPoint& gp, const string& pv_name) override {
            print_unaligned_vec_simple(os, gp, pv_name, _line_prefix);
        }

        // Get offset from base pointer.
        virtual string get_var_base_ptr_offset(ostream& os, const VarPoint& gp,
                                               const VarMap* var_map = 0);
        virtual string get_inner_loop_ptr_offset(ostream& os, const VarPoint& gp,
                                                 const VarMap* var_map = 0,
                                                 const string& inner_ofs = "");

    public:

        // Whether to use masks during write.
        virtual void set_write_mask(string mask_var) {
            _write_mask = mask_var;
        }
        virtual string get_write_mask() const {
            return _write_mask;
        }

        // Print any needed memory reads and/or constructions to 'os'.
        // Return code containing a vector of var points.
        virtual string read_from_point(ostream& os, const VarPoint& gp) override;

        // Print any immediate memory writes to 'os'.
        // Return code to update a vector of var points or null string
        // if all writes were printed.
        virtual string write_to_point(ostream& os, const VarPoint& gp, const string& val) override;

        // Make var base point (first allocated point).
        virtual var_point_ptr make_var_base_point(const VarPoint& gp);

        // Make inner-loop base point (no inner-layout offset; misc-dim indices = min-val).
        virtual var_point_ptr make_inner_loop_base_point(const VarPoint& gp);

        // Print code to create base pointers for aligned reads.
        virtual void print_var_base_ptr(ostream& os, const VarPoint& gp);
        virtual void print_inner_loop_base_ptrs(ostream& os);

        // Print prefetches for each base pointer.
        // Print only 'ptr_var' if provided.
        virtual void print_prefetches(ostream& os, bool ahead, string ptr_var = "");

        // print init of rank constants.
        virtual void print_rank_data(ostream& os);
        
        // print init of un-normalized indices.
        virtual void print_elem_indices(ostream& os);

        // print increments of pointers.
        virtual void print_inc_inner_loop_ptrs(ostream& os, const string& inc_amt);

        // get un-normalized index.
        virtual const string& get_local_elem_index(const string& dname) const {
            return _vec2elem_local_map.at(dname);
        }
        virtual const string& get_global_elem_index(const string& dname) const {
            return _vec2elem_global_map.at(dname);
        }

        // Print strides for 'gp'.
        virtual void print_strides(ostream& os, const VarPoint& gp);

        // Access cached values.
        virtual string* lookup_var_base_ptr(const VarPoint& gp) {
            auto bgp = make_var_base_point(gp);
            if (_var_base_ptrs.count(*bgp))
                return &_var_base_ptrs.at(*bgp);
            return 0;
        }
        virtual string* lookup_inner_loop_base_ptr(const VarPoint& gp) {
            auto bgp = make_inner_loop_base_point(gp);
            if (_inner_loop_base_ptrs.count(*bgp))
                return &_inner_loop_base_ptrs.at(*bgp);
            return 0;
        }
        virtual string* lookup_stride(const Var& var, const string& dim) {
            auto key = VarDimKey(var.get_name(), dim);
            if (_strides.count(key))
                return &_strides.at(key);
            return 0;
        }
        virtual string* lookup_offset(const Var& var, const string& dim) {
            auto key = VarDimKey(var.get_name(), dim);
            if (_offsets.count(key))
                return &_offsets.at(key);
            return 0;
        }
    };

    // Outputs loop-invariant values.
    class CppPreLoopPrintVisitor : public PrintVisitorBase {
    protected:
        CppVecPrintHelper& _cvph;

    public:
        CppPreLoopPrintVisitor(ostream& os,
                               CppVecPrintHelper& ph,
                               const VarMap* var_map = 0) :
            PrintVisitorBase(os, ph, var_map),
            _cvph(ph) {
            _visit_equals_lhs = true;
            _visit_var_point_args = true;
            _visit_conds = true;
        }

        virtual string get_var_ptr(VarPoint& gp) {
            return _cvph.get_var_ptr(gp);
        }
    };

    // Meta values such as strides and pointers.
    class CppPreLoopPrintMetaVisitor : public CppPreLoopPrintVisitor {
    public:
        CppPreLoopPrintMetaVisitor(ostream& os,
                                   CppVecPrintHelper& ph,
                                   const VarMap* var_map = 0) :
            CppPreLoopPrintVisitor(os, ph, var_map) { }

        // A var access.
        virtual string visit(VarPoint* gp);
    };

    // Data values.
    class CppPreLoopPrintDataVisitor : public CppPreLoopPrintVisitor {
    public:
        CppPreLoopPrintDataVisitor(ostream& os,
                                   CppVecPrintHelper& ph,
                                   const VarMap* var_map = 0) :
            CppPreLoopPrintVisitor(os, ph, var_map) { }

        // A var access.
        virtual string visit(VarPoint* gp);
    };

    // Print out a stencil in C++ form for YASK.
    class YASKCppPrinter : public PrinterBase {
    protected:
        EqStages& _eq_stages; // stages of bundles w/o inter-dependencies.
        EqBundles& _cluster_eq_bundles;  // eq-bundles for scalar and vector.
        string _stencil_prefix;
        string _context, _context_hook; // class names;
        string _core_t, _thread_core_t; // core struct names;

        // Print an expression as a one-line C++ comment.
        void add_comment(ostream& os, EqBundle& eq);

        // A factory method to create a new PrintHelper.
        // This can be overridden in derived classes to provide
        // alternative PrintHelpers.
        virtual CppVecPrintHelper* new_cpp_vec_print_helper(VecInfoVisitor& vv,
                                                        CounterVisitor& cv) {
            return new CppVecPrintHelper(vv, _settings, _dims, &cv,
                                         "real_vec_t", " ", ";\n");
        }

        // Print extraction of indices.
        virtual void print_indices(ostream& os,
                                   bool print_step = true,
                                   bool print_domain = true,
                                   const string prefix = "",
                                   const string inner_var_prefix = "") const;

        // Print pieces of YASK output.
        virtual void print_macros(ostream& os);
        virtual void print_data(ostream& os);
        virtual void print_eq_bundles(ostream& os);
        virtual void print_context(ostream& os);

    public:
        YASKCppPrinter(StencilSolution& stencil,
                       EqBundles& eq_bundles,
                       EqStages& eq_stages,
                       EqBundles& cluster_eq_bundles) :
            PrinterBase(stencil, eq_bundles),
            _eq_stages(eq_stages),
            _cluster_eq_bundles(cluster_eq_bundles)
        {
            // name of C++ struct.
            _stencil_prefix = "stencil_" + _stencil._get_name() + "_";
            _context = _stencil_prefix + "context_t";
            _context_hook = _stencil_prefix + "hook_t";
            _core_t = _stencil_prefix + "core_t";
            _thread_core_t = _stencil_prefix + "thread_core_t";
        }
        virtual ~YASKCppPrinter() { }

        // Output all code for YASK.
        virtual void print(ostream& os);
    };

} // namespace yask.

