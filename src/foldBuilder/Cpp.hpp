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

////////// Support for YASK C++ scalar and vector-code generation //////////////

#ifndef CPP_HPP
#define CPP_HPP

#include "Vec.hpp"

namespace yask {

    /////////// Scalar code /////////////

    // Outputs C++ scalar code for YASK.
    class CppPrintHelper : public PrintHelper {

    public:
        CppPrintHelper(const CounterVisitor* cv,
                       const string& varPrefix,
                       const string& varType,
                       const string& linePrefix,
                       const string& lineSuffix) :
            PrintHelper(cv, varPrefix, varType,
                        linePrefix, lineSuffix) { }
        virtual ~CppPrintHelper() { }

        // Format a real, preserving precision.
        static string formatReal(double v) {

            // IEEE double gives 15-17 significant decimal digits of precision per
            // https://en.wikipedia.org/wiki/Double-precision_floating-point_format.
            // Some precision might be lost if/when cast to a float, but that's ok.
            ostringstream oss;
            oss << setprecision(17) << scientific << v;
            return oss.str();
        }
    
        // Return a constant expression.
        // This is overloaded to preserve precision.
        virtual string addConstExpr(ostream& os, double v) {
            return formatReal(v);
        }

        // Return a parameter reference.
        virtual string readFromParam(ostream& os, const GridPoint& pp) {
            string str = "(*_context->" + pp.getName() + ")(" + pp.makeValStr() + ")";
            return str;
        }
    
        // Make call for a point.
        // This is a utility function used for both reads and writes.
        virtual string makePointCall(const GridPoint& gp,
                                     const string& fname, string optArg = "") const {
            ostringstream oss;
            oss << "_context->" << gp.getName() << "->" << fname << "(";
            if (optArg.length()) oss << optArg << ", ";
            oss << gp.makeDimValOffsetStr() << ", __LINE__)";
            return oss.str();
        }
    
        // Return a grid reference.
        virtual string readFromPoint(ostream& os, const GridPoint& gp) {
            return makePointCall(gp, "readElem");
        }

        // Update a grid point.
        virtual string writeToPoint(ostream& os, const GridPoint& gp, const string& val) {
            return makePointCall(gp, "writeElem", val);
        }
    };

    /////////// Vector code /////////////

    // Output generic C++ vector code for YASK.
    class CppVecPrintHelper : public VecPrintHelper {

    public:
        CppVecPrintHelper(VecInfoVisitor& vv,
                          bool allowUnalignedLoads,
                          const CounterVisitor* cv,
                          const string& varPrefix,
                          const string& varType,
                          const string& linePrefix,
                          const string& lineSuffix) :
            VecPrintHelper(vv, allowUnalignedLoads, cv,
                           varPrefix, varType, linePrefix, lineSuffix) { }

    protected:

        // A simple constant.
        virtual string addConstExpr(ostream& os, double v) {
            return CppPrintHelper::formatReal(v);
        }

        // Any code.
        virtual string addCodeExpr(ostream& os, const string& code) {
            return code;
        }

        // Return a parameter reference.
        virtual string readFromParam(ostream& os, const GridPoint& pp) {
            string str = "(*_context->" + pp.getName() + ")(" + pp.makeValStr() + ")";
            return str;
        }
    
        // Print a comment about a point.
        // This is a utility function used for both reads and writes.
        virtual void printPointComment(ostream& os, const GridPoint& gp, const string& verb) const {

            os << endl << " // " << verb << " " << gp.getName() << " at " <<
                gp.makeDimValOffsetStr() << "." << endl;
        }

        // Print call for a point.
        // This is a utility function used for both reads and writes.
        virtual void printPointCall(ostream& os,
                                    const GridPoint& gp,
                                    const string& funcName,
                                    const string& firstArg,
                                    const string& lastArg,
                                    bool isNorm) const {
            os << " _context->" << gp.getName() << "->" << funcName << "(";
            if (firstArg.length())
                os << firstArg << ", ";
            if (isNorm)
                os << gp.makeDimValNormOffsetStr(getFold());
            else
                os << gp.makeDimValOffsetStr();
            if (lastArg.length()) 
                os << ", " << lastArg;
            os << ")";
        }
    
        // Print aligned memory read.
        virtual string printAlignedVecRead(ostream& os, const GridPoint& gp) {
            printPointComment(os, gp, "Read aligned vector block from");

            // Read memory.
            string mvName = makeVarName();
            os << _linePrefix << getVarType() << " " << mvName << " = ";
            printPointCall(os, gp, "readVecNorm", "", "__LINE__", true);
            os << _lineSuffix;
            return mvName;
        }

        // Print unaliged memory read.
        // Assumes this results in same values as printUnalignedVec().
        virtual string printUnalignedVecRead(ostream& os, const GridPoint& gp) {
            printPointComment(os, gp, "Read unaligned vector block from");
            os << " // NOTICE: Assumes constituent vectors are consecutive in memory!" << endl;
            
            // Make a var.
            string mvName = makeVarName();
            os << _linePrefix << getVarType() << " " << mvName << _lineSuffix;
        
            // Read memory.
            os << _linePrefix << mvName << ".loadUnalignedFrom((const " << getVarType() << "*)";
            printPointCall(os, gp, "getElemPtr", "", "true", false);
            os << ")" << _lineSuffix;
            return mvName;
        }

        // Print aligned memory write.
        virtual string printAlignedVecWrite(ostream& os, const GridPoint& gp,
                                            const string& val) {
            printPointComment(os, gp, "Write aligned vector block to");

            // Write temp var to memory.
            printPointCall(os, gp, "writeVecNorm", val, "__LINE__", true);
            return val;
        }
    
        // Print conversion from memory vars to point var gp if needed.
        // This calls printUnalignedVecCtor(), which can be overloaded
        // by derived classes.
        virtual string printUnalignedVec(ostream& os, const GridPoint& gp) {
            printPointComment(os, gp, "Construct unaligned vector block from");

            // Declare var.
            string pvName = makeVarName();
            os << _linePrefix << getVarType() << " " << pvName << _lineSuffix;

            // Contruct it.
            printUnalignedVecCtor(os, gp, pvName);
            return pvName;
        }

        // Print per-element construction for one point var pvName from elems.
        virtual void printUnalignedVecSimple(ostream& os, const GridPoint& gp,
                                             const string& pvName, string linePrefix,
                                             const set<size_t>* doneElems = 0) {

            // just assign each element in vector separately.
            auto& elems = _vv._vblk2elemLists[gp];
            assert(elems.size() > 0);
            for (size_t pelem = 0; pelem < elems.size(); pelem++) {

                // skip if done.
                if (doneElems && doneElems->count(pelem))
                    continue;

                // one vector element from gp.
                auto& ve = elems[pelem];

                // Look up existing input var.
                assert(_readyPoints.count(ve._vec));
                string mvName = _readyPoints[ve._vec];

                // which element?
                int alignedElem = ve._offset;
                string elemStr = ve._offsets.makeDimValOffsetStr();

                os << linePrefix << pvName << "[" << pelem << "] = " <<
                    mvName << "[" << alignedElem << "];  // for " <<
                    elemStr << _lineSuffix;
            }
        }

        // Print construction for one point var pvName from elems.
        // This version prints inefficient element-by-element assignment.
        // Override this in derived classes for more efficient implementations.
        virtual void printUnalignedVecCtor(ostream& os, const GridPoint& gp, const string& pvName) {
            printUnalignedVecSimple(os, gp, pvName, _linePrefix);
        }

    public:

        // print init of normalized indices.
        virtual void printNorm(ostream& os, const IntTuple& dims) {
            const IntTuple& vlen = getFold();
            os << endl << " // Normalize indices by vector fold lengths." << endl;
            for (auto dim : dims.getDims()) {
                const int* p = vlen.lookup(dim);
                os << " const idx_t " << dim << "v = " << dim;
                if (p) os << " / " << *p;
                os << ";" << endl;
            }
        }

        // Print body of prefetch function.
        virtual void printPrefetches(ostream& os, const IntTuple& dir) const {

            // Points to prefetch.
            GridPointSet* pfPts = NULL;

            // Prefetch leading points only if dir is set.
            GridPointSet edge;
            if (dir.size() > 0) {
                _vv.getLeadingEdge(edge, dir);
                pfPts = &edge;
            }

            // if dir is not set, prefetch all points.
            else
                pfPts = &_vv._alignedVecs;

            for (auto gp : *pfPts) {
                printPointComment(os, gp, "Aligned");
            
                // Prefetch memory.
                printPointCall(os, gp, "prefetchVecNorm<level>", "", "__LINE__", true);
                os << ";" << endl;
            }
        }
    };

    // Settings for C++ printing.
    struct YASKCppSettings {
        bool _allowUnalignedLoads = false;
        int _haloSize = 0;
        int _stepAlloc = 2; 
        int _maxExprSize = 50, _minExprSize = 0;
    };

    // Print out a stencil in C++ form for YASK.
    class YASKCppPrinter : public PrinterBase {
    protected:
        EqGroups& _clusterEqGroups;
        Dimensions& _dims;
        YASKCppSettings& _settings;
        string _context, _context_base;
        IntTuple _yask_dims;        // spatial dims in yask.
        string _yask_step;          // step dim in yask.

        // Print an expression as a one-line C++ comment.
        void addComment(ostream& os, EqGroup& eq) {
        
            // Use a simple human-readable visitor to create a comment.
            PrintHelper ph(0, "temp", "", " // ", ".\n");
            PrintVisitorTopDown commenter(os, ph);
            eq.visitEqs(&commenter);
        }

        // A factory method to create a new PrintHelper.
        // This can be overridden in derived classes to provide
        // alternative PrintHelpers.
        virtual CppVecPrintHelper* newPrintHelper(VecInfoVisitor& vv,
                                                  CounterVisitor& cv) {
            return new CppVecPrintHelper(vv, _settings._allowUnalignedLoads, &cv,
                                         "temp_vec", "real_vec_t", " ", ";\n");
        }

    public:
        YASKCppPrinter(StencilBase& stencil,
                       EqGroups& eqGroups,
                       EqGroups& clusterEqGroups,
                       Dimensions& dims,
                       YASKCppSettings& settings) :
            PrinterBase(stencil, eqGroups,
                        settings._maxExprSize, settings._minExprSize),
            _clusterEqGroups(clusterEqGroups),
            _dims(dims), _settings(settings)
        {
            // name of C++ struct.
            _context = "StencilContext_" + _stencil.getName();
            _context_base = _context + "_data";

            // YASK dims are hard-coded.
            // TODO: fix YASK.
            _yask_step = "t";
            _yask_dims.addDimBack("w", 1);
            _yask_dims.addDimBack("x", 1);
            _yask_dims.addDimBack("y", 1);
            _yask_dims.addDimBack("z", 1);
        }
        virtual ~YASKCppPrinter() { }

        virtual void printMacros(ostream& os);
        virtual void printGrids(ostream& os);
        virtual void printCode(ostream& os);
        virtual void printShim(ostream& os, const string& fname,
                               bool use_template = false,
                               const string& dim = "");
    };

} // namespace yask.

#endif
