/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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
                       const string& var_prefix,
                       const string& var_type,
                       const string& line_prefix,
                       const string& line_suffix) :
            PrintHelper(settings, dims, cv, var_prefix, var_type,
                        line_prefix, line_suffix) { }
        virtual ~CppPrintHelper() { }

        // Format a real, preserving precision.
        static string format_real(double v);

        // Return a constant expression.
        // This is overloaded to preserve precision.
        virtual string add_const_expr(ostream& os, double v) override {
            return format_real(v);
        }

        // Format a pointer to a var.
        virtual string get_var_ptr(const VarPoint& gp) {
            const auto* var = gp._get_var();
            string gname = var->_get_name();
            string expr;
            if (var->is_scratch())
                expr = "thread_core_data.";
            else
                expr = "_core_p->";
            expr += "var_" + gname + "_core_p";
            return expr;
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
                          const string& var_prefix,
                          const string& var_type,
                          const string& line_prefix,
                          const string& line_suffix) :
            CppPrintHelper(settings, dims, cv,
                           var_prefix, var_type, line_prefix, line_suffix),
            VecPrintHelper(vv) { }
        
    protected:

        // Vars for tracking pointers to var values.
        map<VarPoint, string> _vec_ptrs; // pointers to var vecs. value: ptr-var name.
        map<string, int> _ptr_ofs_lo; // lowest read offset from _vec_ptrs in inner dim.
        map<string, int> _ptr_ofs_hi; // highest read offset from _vec_ptrs in inner dim.

        // Element indices.
        string _elem_suffix = "_elem";
        VarMap _vec2elem_map; // maps vector indices to elem indices; filled by print_elem_indices.

        bool _use_masked_writes = true;

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

            os << endl << " // " << verb << " vector starting at " <<
                gp.make_str() << "." << endl;
        }

        // Return code for a vectorized point.
        // This is a utility function used for both reads and writes.
        virtual string print_vec_point_call(ostream& os,
                                         const VarPoint& gp,
                                         const string& func_name,
                                         const string& first_arg,
                                         const string& last_arg,
                                         bool is_norm);

        // Print aligned memory read.
        virtual string print_aligned_vec_read(ostream& os, const VarPoint& gp) override;

        // Print unaliged memory read.
        // Assumes this results in same values as print_unaligned_vec().
        virtual string print_unaligned_vec_read(ostream& os, const VarPoint& gp) override;

        // Print aligned memory write.
        virtual string print_aligned_vec_write(ostream& os, const VarPoint& gp,
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
                                           const VarMap* v_map=0) override;

        // Read from multiple points that are not vectorizable.
        // Return var name.
        virtual string print_non_vec_read(ostream& os, const VarPoint& gp) override;

        // Print construction for one point var pv_name from elems.
        // This version prints inefficient element-by-element assignment.
        // Override this in derived classes for more efficient implementations.
        virtual void print_unaligned_vec_ctor(ostream& os, const VarPoint& gp, const string& pv_name) override {
            print_unaligned_vec_simple(os, gp, pv_name, _line_prefix);
        }

        // Get offset from base pointer.
        virtual string get_ptr_offset(const VarPoint& gp,
                                    const string& inner_expr = "");

    public:

        // Whether to use masks during write.
        virtual void set_use_masked_writes(bool do_use) {
            _use_masked_writes = do_use;
        }
        virtual bool get_use_masked_writes() const {
            return _use_masked_writes;
        }

        // Print any needed memory reads and/or constructions to 'os'.
        // Return code containing a vector of var points.
        virtual string read_from_point(ostream& os, const VarPoint& gp) override;

        // Print any immediate memory writes to 'os'.
        // Return code to update a vector of var points or null string
        // if all writes were printed.
        virtual string write_to_point(ostream& os, const VarPoint& gp, const string& val) override;

        // Print code to set pointers of aligned reads.
        virtual void print_base_ptrs(ostream& os);

        // Make base point (misc & inner-dim indices = 0).
        virtual var_point_ptr make_base_point(const VarPoint& gp);

        // Print prefetches for each base pointer.
        // Print only 'ptr_var' if provided.
        virtual void print_prefetches(ostream& os, bool ahead, string ptr_var = "");

        // print init of un-normalized indices.
        virtual void print_elem_indices(ostream& os);

        // get un-normalized index.
        virtual const string& get_elem_index(const string& dname) const {
            return _vec2elem_map.at(dname);
        }

        // Print code to set ptr_name to gp.
        virtual void print_point_ptr(ostream& os, const string& ptr_name, const VarPoint& gp);

        // Access cached values.
        virtual void save_point_ptr(const VarPoint& gp, string var) {
            _vec_ptrs[gp] = var;
        }
        virtual string* lookup_point_ptr(const VarPoint& gp) {
            if (_vec_ptrs.count(gp))
                return &_vec_ptrs.at(gp);
            return 0;
        }
    };

    // Outputs the time-invariant variables.
    class CppStepVarPrintVisitor : public PrintVisitorBase {
    protected:
        CppVecPrintHelper& _cvph;

    public:
        CppStepVarPrintVisitor(ostream& os,
                               CppVecPrintHelper& ph,
                               const VarMap* var_map = 0) :
            PrintVisitorBase(os, ph, var_map),
            _cvph(ph) { }

        // A var access.
        virtual string visit(VarPoint* gp);

        virtual string get_var_ptr(VarPoint& gp) {
            return _cvph.get_var_ptr(gp);
        }
    };

    // Outputs the loop-invariant variables for an inner loop.
    class CppLoopVarPrintVisitor : public PrintVisitorBase {
    protected:
        CppVecPrintHelper& _cvph;

    public:
        CppLoopVarPrintVisitor(ostream& os,
                               CppVecPrintHelper& ph,
                               const VarMap* var_map = 0) :
            PrintVisitorBase(os, ph, var_map),
            _cvph(ph) { }

        // A var access.
        virtual string visit(VarPoint* gp);
    };

    // Print out a stencil in C++ form for YASK.
    class YASKCppPrinter : public PrinterBase {
    protected:
        EqStages& _eq_stages; // stages of bundles w/o inter-dependencies.
        EqBundles& _cluster_eq_bundles;  // eq-bundles for scalar and vector.
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
                                         "temp", "real_vec_t", " ", ";\n");
        }

        // Print extraction of indices.
        virtual void print_indices(ostream& os) const;

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
            string sname = _stencil._get_name();
            _context = sname + "_context_t";
            _context_hook = sname + "_hook_t";
            _core_t = sname + "_core_t";
            _thread_core_t = sname + "_thread_core_t";
        }
        virtual ~YASKCppPrinter() { }

        // Output all code for YASK.
        virtual void print(ostream& os);
    };

} // namespace yask.

