/*****************************************************************************

YASK: Yet Another Stencil Kit
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

// Support for vector code generation using SIMD instrinsics.

#ifndef CPPINTRIN_HPP
#define CPPINTRIN_HPP

#include "Cpp.hpp"

//#define DEBUG_SHIFT

namespace yask {

    // Add specialization for 256 and 512-bit intrinsics.
    class CppIntrinPrintHelper : public CppVecPrintHelper {

    protected:
        set<string> definedCtrls; // control vars already defined.

        // Ctor.
        CppIntrinPrintHelper(VecInfoVisitor& vv,
                             const CompilerSettings& settings,
                             const Dimensions& dims,
                             const CounterVisitor* cv,
                             const string& varPrefix,
                             const string& varType,
                             const string& linePrefix,
                             const string& lineSuffix) :
            CppVecPrintHelper(vv, settings, dims, cv,
                              varPrefix, varType, linePrefix, lineSuffix) { }

        // Dtor.
        virtual ~CppIntrinPrintHelper() { }

        // Print mask when needed for multi-step construction.
        virtual void printMask(ostream& os, unsigned mask) {
            os << ", 0x" << hex << mask << dec;
        }

        // Try all applicable strategies to create nelemsTarget elements.
        // Elements needed are those from elems but not in doneElems.
        // Elements can be taken from alignedVecs.
        // Update doneElems.
        virtual void tryStrategies(ostream& os,
                                   const string& pvName,
                                   size_t nelemsTarget,
                                   const VecElemList& elems,
                                   set<size_t>& doneElems,
                                   const VarPointSet& alignedVecs) =0;

        // Try to use align instruction(s) to construct nelemsTarget elements
        // per instruction.
        virtual void tryAlign(ostream& os,
                              const string& pvName,
                              size_t nelemsTarget,
                              const VecElemList& elems,
                              set<size_t>& doneElems,
                              const VarPointSet& alignedVecs,
                              bool maskAllowed);

        // Try to use 1-var permute instruction(s) to construct nelemsTarget elements
        // per instruction.
        virtual void tryPerm1(ostream& os,
                              const string& pvName,
                              size_t nelemsTarget,
                              const VecElemList& elems,
                              set<size_t>& doneElems,
                              const VarPointSet& alignedVecs);

        // Try to use 2-var permute instruction(s) to construct nelemsTarget elements
        // per instruction.
        virtual void tryPerm2(ostream& os,
                              const string& pvName,
                              size_t nelemsTarget,
                              const VecElemList& elems,
                              set<size_t>& doneElems,
                              const VarPointSet& alignedVecs);

    public:
        // Print construction for one unaligned vector pvName at gp.
        virtual void printUnalignedVecCtor(ostream& os,
                                           const VarPoint& gp,
                                           const string& pvName);

    };

    // Specialization for KNC.
    class CppKncPrintHelper : public CppIntrinPrintHelper {
    protected:

        // Try all applicable strategies.
        virtual void tryStrategies(ostream& os,
                                   const string& pvName,
                                   size_t nelemsTarget,
                                   const VecElemList& elems,
                                   set<size_t>& doneElems,
                                   const VarPointSet& alignedVecs) {
            tryAlign(os, pvName, nelemsTarget, elems, doneElems, alignedVecs, true);
            tryPerm1(os, pvName, nelemsTarget, elems, doneElems, alignedVecs);
        }

    public:
        CppKncPrintHelper(VecInfoVisitor& vv,
                          const CompilerSettings& settings,
                          const Dimensions& dims,
                          const CounterVisitor* cv,
                          const string& varPrefix,
                          const string& varType,
                          const string& linePrefix,
                          const string& lineSuffix) :
            CppIntrinPrintHelper(vv, settings, dims, cv,
                                 varPrefix, varType, linePrefix, lineSuffix) { }
    };

    // Specialization for KNL, SKX, etc.
    class CppAvx512PrintHelper : public CppIntrinPrintHelper {
    protected:

        // Try all applicable strategies.
        virtual void tryStrategies(ostream& os,
                                   const string& pvName,
                                   size_t nelemsTarget,
                                   const VecElemList& elems,
                                   set<size_t>& doneElems,
                                   const VarPointSet& alignedVecs) {
            tryAlign(os, pvName, nelemsTarget, elems, doneElems, alignedVecs, true);
            tryPerm2(os, pvName, nelemsTarget, elems, doneElems, alignedVecs);
            tryPerm1(os, pvName, nelemsTarget, elems, doneElems, alignedVecs);
        }

    public:
        CppAvx512PrintHelper(VecInfoVisitor& vv,
                             const CompilerSettings& settings,
                             const Dimensions& dims,
                             const CounterVisitor* cv,
                             const string& varPrefix,
                             const string& varType,
                             const string& linePrefix,
                             const string& lineSuffix) :
            CppIntrinPrintHelper(vv, settings, dims, cv,
                                 varPrefix, varType, linePrefix, lineSuffix) { }
    };

    // Specialization for AVX, AVX2.
    class CppAvx256PrintHelper : public CppIntrinPrintHelper {
    protected:

        // Try all applicable strategies.
        virtual void tryStrategies(ostream& os,
                                   const string& pvName,
                                   size_t nelemsTarget,
                                   const VecElemList& elems,
                                   set<size_t>& doneElems,
                                   const VarPointSet& alignedVecs) {
            tryAlign(os, pvName, nelemsTarget, elems, doneElems, alignedVecs, false);
        }

    public:
        CppAvx256PrintHelper(VecInfoVisitor& vv,
                             const CompilerSettings& settings,
                             const Dimensions& dims,
                             const CounterVisitor* cv,
                             const string& varPrefix,
                             const string& varType,
                             const string& linePrefix,
                             const string& lineSuffix) :
            CppIntrinPrintHelper(vv, settings, dims, cv,
                                 varPrefix, varType, linePrefix, lineSuffix) { }
    };

    // Print KNC intrinsic code.
    class YASKKncPrinter : public YASKCppPrinter {
    protected:
        virtual CppVecPrintHelper* newCppVecPrintHelper(VecInfoVisitor& vv,
                                                        CounterVisitor& cv) {
            return new CppKncPrintHelper(vv, _settings, _dims, &cv,
                                         "temp", "real_vec_t", " ", ";\n");
        }

    public:
        YASKKncPrinter(StencilSolution& stencil,
                       EqBundles& eqBundles,
                       EqBundlePacks& eqBundlePacks,
                       EqBundles& clusterEqBundles) :
            YASKCppPrinter(stencil, eqBundles, eqBundlePacks, clusterEqBundles) { }

        virtual int num_vec_elems() const { return 64 / _settings._elem_bytes; }

        // Whether multi-dim folding is efficient.
        virtual bool is_folding_efficient() const { return true; }
    };

    // Print 256-bit AVX intrinsic code.
    class YASKAvx256Printer : public YASKCppPrinter {
    protected:
        virtual CppVecPrintHelper* newCppVecPrintHelper(VecInfoVisitor& vv,
                                                        CounterVisitor& cv) {
            return new CppAvx256PrintHelper(vv, _settings, _dims, &cv,
                                            "temp", "real_vec_t", " ", ";\n");
        }

    public:
        YASKAvx256Printer(StencilSolution& stencil,
                          EqBundles& eqBundles,
                          EqBundlePacks& eqBundlePacks,
                          EqBundles& clusterEqBundles) :
            YASKCppPrinter(stencil, eqBundles, eqBundlePacks, clusterEqBundles) { }

        virtual int num_vec_elems() const { return 32 / _settings._elem_bytes; }
    };

    // Print 512-bit AVX intrinsic code.
    class YASKAvx512Printer : public YASKCppPrinter {
    protected:
        virtual CppVecPrintHelper* newCppVecPrintHelper(VecInfoVisitor& vv,
                                                        CounterVisitor& cv) {
            return new CppAvx512PrintHelper(vv, _settings, _dims, &cv,
                                            "temp", "real_vec_t", " ", ";\n");
        }

    public:
        YASKAvx512Printer(StencilSolution& stencil,
                          EqBundles& eqBundles,
                          EqBundlePacks& eqBundlePacks,
                          EqBundles& clusterEqBundles) :
            YASKCppPrinter(stencil, eqBundles, eqBundlePacks, clusterEqBundles) { }

        virtual int num_vec_elems() const { return 64 / _settings._elem_bytes; }

        // Whether multi-dim folding is efficient.
        virtual bool is_folding_efficient() const { return true; }
    };

} // namespace yask.

#endif
