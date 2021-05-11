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

// Support for vector code generation using SIMD instrinsics.

#pragma once

#include "Cpp.hpp"

//#define DEBUG_SHIFT

namespace yask {

    // Add specialization for 256 and 512-bit intrinsics.
    class CppIntrinPrintHelper : public CppVecPrintHelper {

    protected:
        set<string> defined_ctrls; // control vars already defined.
        bool _defined_na = false;        // NA var defined.

        // Ctor.
        CppIntrinPrintHelper(VecInfoVisitor& vv,
                             const CompilerSettings& settings,
                             const Dimensions& dims,
                             const CounterVisitor* cv,
                             const string& var_type,
                             const string& line_prefix,
                             const string& line_suffix) :
            CppVecPrintHelper(vv, settings, dims, cv,
                              var_type, line_prefix, line_suffix) { }

        // Dtor.
        virtual ~CppIntrinPrintHelper() { }

        // Add N/A var, just for readability.
        virtual void make_na(ostream& os, string line_prefix, string line_suffix) {
            if (!_defined_na) {
                os << line_prefix << "const int NA = 0; // indicates element not used." << line_suffix;
                _defined_na = true;
            }
        }

        // Print mask when needed for multi-step construction.
        virtual void print_mask(ostream& os, unsigned mask) {
            os << ", 0x" << hex << mask << dec;
        }

        // Try all applicable strategies to create nelems_target elements.
        // Elements needed are those from elems but not in done_elems.
        // Elements can be taken from aligned_vecs.
        // Update done_elems.
        virtual void try_strategies(ostream& os,
                                   const string& pv_name,
                                   size_t nelems_target,
                                   const VecElemList& elems,
                                   set<size_t>& done_elems,
                                   const VarPointSet& aligned_vecs) =0;

        // Try to use align instruction(s) to construct nelems_target elements
        // per instruction.
        virtual void try_align(ostream& os,
                              const string& pv_name,
                              size_t nelems_target,
                              const VecElemList& elems,
                              set<size_t>& done_elems,
                              const VarPointSet& aligned_vecs,
                              bool mask_allowed);

        // Try to use 1-var permute instruction(s) to construct nelems_target elements
        // per instruction.
        virtual void try_perm1(ostream& os,
                              const string& pv_name,
                              size_t nelems_target,
                              const VecElemList& elems,
                              set<size_t>& done_elems,
                              const VarPointSet& aligned_vecs);

        // Try to use 2-var permute instruction(s) to construct nelems_target elements
        // per instruction.
        virtual void try_perm2(ostream& os,
                              const string& pv_name,
                              size_t nelems_target,
                              const VecElemList& elems,
                              set<size_t>& done_elems,
                              const VarPointSet& aligned_vecs);

    public:
        // Print construction for one unaligned vector pv_name at gp.
        virtual void print_unaligned_vec_ctor(ostream& os,
                                           const VarPoint& gp,
                                           const string& pv_name);

    };

    // Specialization for KNL, SKX, etc.
    class CppAvx512PrintHelper : public CppIntrinPrintHelper {
    protected:

        // Try all applicable strategies.
        virtual void try_strategies(ostream& os,
                                   const string& pv_name,
                                   size_t nelems_target,
                                   const VecElemList& elems,
                                   set<size_t>& done_elems,
                                   const VarPointSet& aligned_vecs) {
            try_align(os, pv_name, nelems_target, elems, done_elems, aligned_vecs, true);
            try_perm2(os, pv_name, nelems_target, elems, done_elems, aligned_vecs);
            try_perm1(os, pv_name, nelems_target, elems, done_elems, aligned_vecs);
        }

    public:
        CppAvx512PrintHelper(VecInfoVisitor& vv,
                             const CompilerSettings& settings,
                             const Dimensions& dims,
                             const CounterVisitor* cv,
                             const string& var_type,
                             const string& line_prefix,
                             const string& line_suffix) :
            CppIntrinPrintHelper(vv, settings, dims, cv,
                                 var_type, line_prefix, line_suffix) { }
    };

    // Specialization for AVX, AVX2.
    class CppAvx256PrintHelper : public CppIntrinPrintHelper {
    protected:

        // Try all applicable strategies.
        virtual void try_strategies(ostream& os,
                                   const string& pv_name,
                                   size_t nelems_target,
                                   const VecElemList& elems,
                                   set<size_t>& done_elems,
                                   const VarPointSet& aligned_vecs) {
            try_align(os, pv_name, nelems_target, elems, done_elems, aligned_vecs, false);
        }

    public:
        CppAvx256PrintHelper(VecInfoVisitor& vv,
                             const CompilerSettings& settings,
                             const Dimensions& dims,
                             const CounterVisitor* cv,
                             const string& var_type,
                             const string& line_prefix,
                             const string& line_suffix) :
            CppIntrinPrintHelper(vv, settings, dims, cv,
                                 var_type, line_prefix, line_suffix) { }
    };

    // Print 256-bit AVX intrinsic code.
    class YASKAvx256Printer : public YASKCppPrinter {
    protected:
        virtual CppVecPrintHelper* new_cpp_vec_print_helper(VecInfoVisitor& vv,
                                                        CounterVisitor& cv) {
            return new CppAvx256PrintHelper(vv, _settings, _dims, &cv,
                                            "real_vec_t", " ", ";\n");
        }

    public:
        YASKAvx256Printer(StencilSolution& stencil,
                          EqBundles& eq_bundles,
                          EqStages& eq_stages,
                          EqBundles& cluster_eq_bundles) :
            YASKCppPrinter(stencil, eq_bundles, eq_stages, cluster_eq_bundles) { }

        virtual int num_vec_elems() const { return 32 / _settings._elem_bytes; }
    };

    // Print 512-bit AVX intrinsic code.
    class YASKAvx512Printer : public YASKCppPrinter {
    protected:
        bool _is_lo;
        virtual CppVecPrintHelper* new_cpp_vec_print_helper(VecInfoVisitor& vv,
                                                        CounterVisitor& cv) {
            return new CppAvx512PrintHelper(vv, _settings, _dims, &cv,
                                            "real_vec_t", " ", ";\n");
        }

    public:
        YASKAvx512Printer(StencilSolution& stencil,
                          EqBundles& eq_bundles,
                          EqStages& eq_stages,
                          EqBundles& cluster_eq_bundles,
                          bool is_lo = false) :
            YASKCppPrinter(stencil, eq_bundles, eq_stages, cluster_eq_bundles),
            _is_lo(is_lo) { }

        virtual int num_vec_elems() const {
            return (_is_lo ? 32 : 64) / _settings._elem_bytes;
        }

        // Whether multi-dim folding is efficient.
        virtual bool is_folding_efficient() const { return true; }
    };

} // namespace yask.

