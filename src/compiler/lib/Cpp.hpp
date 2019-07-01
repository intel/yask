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

////////// Support for YASK C++ scalar and vector-code generation //////////////

// See CppIntrin.hpp for explicit intrinsic generation.

#ifndef CPP_HPP
#define CPP_HPP

#include "Vec.hpp"
#include "Var.hpp"

namespace yask {

    /////////// Scalar code /////////////

    // Outputs C++ scalar code for YASK.
    class CppPrintHelper : public PrintHelper {

    public:
        static constexpr const char* _var_ptr_type = "auto*";
        static constexpr const char* _var_ptr_restrict_type = "auto* restrict";
        static constexpr const char* _step_val_type = "const auto";

        CppPrintHelper(const CompilerSettings& settings,
                       const Dimensions& dims,
                       const CounterVisitor* cv,
                       const string& varPrefix,
                       const string& varType,
                       const string& linePrefix,
                       const string& lineSuffix) :
            PrintHelper(settings, dims, cv, varPrefix, varType,
                        linePrefix, lineSuffix) { }
        virtual ~CppPrintHelper() { }

        // Format a real, preserving precision.
        static string formatReal(double v);

        // Return a constant expression.
        // This is overloaded to preserve precision.
        virtual string addConstExpr(ostream& os, double v) {
            return formatReal(v);
        }

        // Format a pointer to a var.
        virtual string getVarPtr(const VarPoint& gp) {
            const auto* var = gp.getVar();
            string gname = var->getName();
            string expr = "(static_cast<_context_type::" + gname + "_type*>(_context_data->";
            if (var->isScratch())
                expr += gname + "_list[region_thread_idx]";
            else
                expr += gname + "_ptr";
            expr += ".get()->gbp()))";
            return expr;
        }
        
        // Make call for a point.
        // This is a utility function used for both reads and writes.
        virtual string makePointCall(ostream& os,
                                     const VarPoint& gp,
                                     const string& fname,
                                     string optArg = "");

        // Return a var-point reference.
        virtual string readFromPoint(ostream& os, const VarPoint& gp);

        // Return code to update a var point.
        virtual string writeToPoint(ostream& os, const VarPoint& gp,
                                    const string& val);
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
                          const string& varPrefix,
                          const string& varType,
                          const string& linePrefix,
                          const string& lineSuffix) :
            CppPrintHelper(settings, dims, cv,
                           varPrefix, varType, linePrefix, lineSuffix),
            VecPrintHelper(vv) { }
        
    protected:

        // Vars for tracking pointers to var values.
        map<VarPoint, string> _vecPtrs; // pointers to var vecs. value: ptr-var name.
        map<string, int> _ptrOfsLo; // lowest read offset from _vecPtrs in inner dim.
        map<string, int> _ptrOfsHi; // highest read offset from _vecPtrs in inner dim.

        // Element indices.
        string _elemSuffix = "_elem";
        VarMap _vec2elemMap; // maps vector indices to elem indices; filled by printElemIndices.

        // A simple constant.
        virtual string addConstExpr(ostream& os, double v) {
            return CppPrintHelper::formatReal(v);
        }

        // Any code.
        virtual string addCodeExpr(ostream& os, const string& code) {
            return code;
        }

        // Print a comment about a point.
        // This is a utility function used for both reads and writes.
        virtual void printPointComment(ostream& os, const VarPoint& gp,
                                       const string& verb) const {

            os << endl << " // " << verb << " vector starting at " <<
                gp.makeStr() << "." << endl;
        }

        // Return code for a vectorized point.
        // This is a utility function used for both reads and writes.
        virtual string printVecPointCall(ostream& os,
                                         const VarPoint& gp,
                                         const string& funcName,
                                         const string& firstArg,
                                         const string& lastArg,
                                         bool isNorm);

        // Print aligned memory read.
        virtual string printAlignedVecRead(ostream& os, const VarPoint& gp);

        // Print unaliged memory read.
        // Assumes this results in same values as printUnalignedVec().
        virtual string printUnalignedVecRead(ostream& os, const VarPoint& gp);

        // Print aligned memory write.
        virtual string printAlignedVecWrite(ostream& os, const VarPoint& gp,
                                            const string& val);

        // Print conversion from memory vars to point var gp if needed.
        // This calls printUnalignedVecCtor(), which can be overloaded
        // by derived classes.
        virtual string printUnalignedVec(ostream& os, const VarPoint& gp);

        // Print per-element construction for one point var pvName from elems.
        virtual void printUnalignedVecSimple(ostream& os, const VarPoint& gp,
                                             const string& pvName, string linePrefix,
                                             const set<size_t>* doneElems = 0);

        // Read from a single point to be broadcast to a vector.
        // Return code for read.
        virtual string readFromScalarPoint(ostream& os, const VarPoint& gp,
                                           const VarMap* vMap=0);

        // Read from multiple points that are not vectorizable.
        // Return var name.
        virtual string printNonVecRead(ostream& os, const VarPoint& gp);

        // Print construction for one point var pvName from elems.
        // This version prints inefficient element-by-element assignment.
        // Override this in derived classes for more efficient implementations.
        virtual void printUnalignedVecCtor(ostream& os, const VarPoint& gp, const string& pvName) {
            printUnalignedVecSimple(os, gp, pvName, _linePrefix);
        }

        // Get offset from base pointer.
        virtual string getPtrOffset(const VarPoint& gp,
                                    const string& innerExpr = "");

    public:


        // Print any needed memory reads and/or constructions to 'os'.
        // Return code containing a vector of var points.
        virtual string readFromPoint(ostream& os, const VarPoint& gp) override;

        // Print any immediate memory writes to 'os'.
        // Return code to update a vector of var points or null string
        // if all writes were printed.
        virtual string writeToPoint(ostream& os, const VarPoint& gp, const string& val) override;

        // Print code to set pointers of aligned reads.
        virtual void printBasePtrs(ostream& os);

        // Make base point (misc & inner-dim indices = 0).
        virtual varPointPtr makeBasePoint(const VarPoint& gp);

        // Print prefetches for each base pointer.
        // Print only 'ptrVar' if provided.
        virtual void printPrefetches(ostream& os, bool ahead, string ptrVar = "");

        // print init of un-normalized indices.
        virtual void printElemIndices(ostream& os);

        // get un-normalized index.
        virtual const string& getElemIndex(const string& dname) const {
            return _vec2elemMap.at(dname);
        }

        // Print code to set ptrName to gp.
        virtual void printPointPtr(ostream& os, const string& ptrName, const VarPoint& gp);

        // Access cached values.
        virtual void savePointPtr(const VarPoint& gp, string var) {
            _vecPtrs[gp] = var;
        }
        virtual string* lookupPointPtr(const VarPoint& gp) {
            if (_vecPtrs.count(gp))
                return &_vecPtrs.at(gp);
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
                               const VarMap* varMap = 0) :
            PrintVisitorBase(os, ph, varMap),
            _cvph(ph) { }

        // A var access.
        virtual string visit(VarPoint* gp);

        virtual string getVarPtr(VarPoint& gp) {
            return _cvph.getVarPtr(gp);
        }
};

    // Outputs the loop-invariant variables for an inner loop.
    class CppLoopVarPrintVisitor : public PrintVisitorBase {
    protected:
        CppVecPrintHelper& _cvph;

    public:
        CppLoopVarPrintVisitor(ostream& os,
                               CppVecPrintHelper& ph,
                               const VarMap* varMap = 0) :
            PrintVisitorBase(os, ph, varMap),
            _cvph(ph) { }

        // A var access.
        virtual string visit(VarPoint* gp);
    };

    // Print out a stencil in C++ form for YASK.
    class YASKCppPrinter : public PrinterBase {
    protected:
        EqBundlePacks& _eqBundlePacks; // packs of bundles w/o inter-dependencies.
        EqBundles& _clusterEqBundles;  // eq-bundles for scalar and vector.
        string _context, _context_base, _context_hook; // class names;

        // Print an expression as a one-line C++ comment.
        void addComment(ostream& os, EqBundle& eq);

        // A factory method to create a new PrintHelper.
        // This can be overridden in derived classes to provide
        // alternative PrintHelpers.
        virtual CppVecPrintHelper* newCppVecPrintHelper(VecInfoVisitor& vv,
                                                        CounterVisitor& cv) {
            return new CppVecPrintHelper(vv, _settings, _dims, &cv,
                                         "temp", "real_vec_t", " ", ";\n");
        }

        // Print extraction of indices.
        virtual void printIndices(ostream& os) const;

        // Print pieces of YASK output.
        virtual void printMacros(ostream& os);
        virtual void printData(ostream& os);
        virtual void printEqBundles(ostream& os);
        virtual void printContext(ostream& os);


    public:
        YASKCppPrinter(StencilSolution& stencil,
                       EqBundles& eqBundles,
                       EqBundlePacks& eqBundlePacks,
                       EqBundles& clusterEqBundles) :
            PrinterBase(stencil, eqBundles),
            _eqBundlePacks(eqBundlePacks),
            _clusterEqBundles(clusterEqBundles)
        {
            // name of C++ struct.
            _context = "StencilContext_" + _stencil.getName();
            _context_base = _context + "_data";
            _context_hook = _context + "_hook";
        }
        virtual ~YASKCppPrinter() { }

        // Output all code for YASK.
        virtual void print(ostream& os);
    };

} // namespace yask.

#endif
