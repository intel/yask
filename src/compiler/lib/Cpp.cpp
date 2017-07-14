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

#include "Cpp.hpp"

namespace yask {

    /////////// Scalar code /////////////

    // Format a real, preserving precision.
    string CppPrintHelper::formatReal(double v) {

        // IEEE double gives 15-17 significant decimal digits of precision per
        // https://en.wikipedia.org/wiki/Double-precision_floating-point_format.
        // Some precision might be lost if/when cast to a float, but that's ok.
        ostringstream oss;
        oss << setprecision(15) << scientific << v;
        return oss.str();
    }
    
    // Make call for a point.
    // This is a utility function used for both reads and writes.
    string CppPrintHelper::makePointCall(const GridPoint& gp,
                                         const string& fname,
                                         string optArg) const {
        ostringstream oss;
        oss << "_context->" << gp.getGridName() << "->" << fname << "(";
        if (optArg.length()) oss << optArg << ", ";
        string args = gp.makeArgStr();
        if (args.length()) oss << args << ", ";
        oss << "__LINE__)";
        return oss.str();
    }
    
    // Return a grid-point reference.
    string CppPrintHelper::readFromPoint(ostream& os, const GridPoint& gp) {
        return makePointCall(gp, "readElem");
    }

    // Return code to update a grid point.
    string CppPrintHelper::writeToPoint(ostream& os, const GridPoint& gp,
                                        const string& val) {
        return makePointCall(gp, "writeElem", val);
    }

    /////////// Vector code /////////////

    // Read from a single point to be broadcast to a vector.
    // Return code for read.
    string CppVecPrintHelper::readFromScalarPoint(ostream& os, const GridPoint& gp) {

        // Assume that broadcast will be handled automatically by
        // operator overloading in kernel code.
        // Specify that any indices should use element vars.
        string str = "_context->" + gp.getGridName() + "->readElem(";
        string args = gp.makeArgStr(&_varMap);
        if (args.length()) str += args + ", ";
        str += "__LINE__)";
        return str;
    }

    // Read from multiple points that are not vectorizable.
    // Return var name.
    string CppVecPrintHelper::printNonVecRead(ostream& os, const GridPoint& gp) {
        printPointComment(os, gp, "Construct folded vector from non-folded");

        // Make a vec var.
        string mvName = makeVarName();
        os << _linePrefix << getVarType() << " " << mvName << _lineSuffix;

        // Loop through all points in the vector fold.
        size_t pelem = 0;
        getFold().visitAllPoints([&](const IntTuple& vecPoint){

                // Example: vecPoint contains x=0, y=2, z=1, where
                // each val is the offset in the given fold dim.
                // We want to map y=>(y+2), z=>(z+1) in
                // grid-point index args.
                VarMap vMap;
                for (auto& dim : vecPoint.getDims()) {
                    auto& dname = dim.getName();
                    int dofs = dim.getVal();
                    if (dofs != 0) {
                        ostringstream oss;
                        oss << "(" << dname << "+" << dofs << ")";
                        vMap[dname] = oss.str();
                    }
                }

                // Read or reuse.
                string varname;
                string args = gp.makeArgStr(&vMap);
                string stmt = "_context->" + gp.getGridName() + "->readElem(";
                if (args.length())
                    stmt += args + ", ";
                stmt += "__LINE__)";
                if (_elemVars.count(stmt))
                    varname = _elemVars.at(stmt);
                else {

                    // Read val into a new scalar var.
                    varname = makeVarName();
                    os << _linePrefix << "real_t " << varname <<
                        " = " << stmt << _lineSuffix;
                    _elemVars[stmt] = varname;
                }
                
                // Output translated expression for this element.
                os << _linePrefix << mvName << "[" << pelem << "] = " <<
                    varname << "; // for offset " << vecPoint.makeDimValStr() <<
                    _lineSuffix;

                pelem++;
            }); // end of lambda.
        return mvName;
    }
    
    // Print call for a point.
    // This is a utility function used for reads, writes, and prefetches.
    void CppVecPrintHelper::printVecPointCall(ostream& os,
                                              const GridPoint& gp,
                                              const string& funcName,
                                              const string& firstArg,
                                              const string& lastArg,
                                              bool isNorm) const {

        os << " _context->" << gp.getGridName() << "->" << funcName << "(";
        if (firstArg.length())
            os << firstArg << ", ";
        if (isNorm)
            os << gp.makeNormArgStr(getFold());
        else
            os << gp.makeArgStr();
        if (lastArg.length()) 
            os << ", " << lastArg;
        os << ")";
    }
    
    // Print aligned memory read.
    string CppVecPrintHelper::printAlignedVecRead(ostream& os, const GridPoint& gp) {
        printPointComment(os, gp, "Read aligned");

        // Read memory.
        string mvName = makeVarName();
        os << _linePrefix << getVarType() << " " << mvName << " = ";
        printVecPointCall(os, gp, "readVecNorm", "", "__LINE__", true);
        os << _lineSuffix;
        return mvName;
    }

    // Print unaliged memory read.
    // Assumes this results in same values as printUnalignedVec().
    string CppVecPrintHelper::printUnalignedVecRead(ostream& os, const GridPoint& gp) {
        printPointComment(os, gp, "Read unaligned");
        os << " // NOTICE: Assumes constituent vectors are consecutive in memory!" << endl;
            
        // Make a var.
        string mvName = makeVarName();
        os << _linePrefix << getVarType() << " " << mvName << _lineSuffix;
        
        // Read memory.
        os << _linePrefix << mvName << ".loadUnalignedFrom((const " << getVarType() << "*)";
        printVecPointCall(os, gp, "getElemPtr", "", "true", false);
        os << ")" << _lineSuffix;
        return mvName;
    }

    // Print aligned memory write.
    string CppVecPrintHelper::printAlignedVecWrite(ostream& os, const GridPoint& gp,
                                                   const string& val) {
        printPointComment(os, gp, "Write aligned");

        // Write temp var to memory.
        printVecPointCall(os, gp, "writeVecNorm", val, "__LINE__", true);
        return val;
    }
    
    // Print conversion from memory vars to point var gp if needed.
    // This calls printUnalignedVecCtor(), which can be overloaded
    // by derived classes.
    string CppVecPrintHelper::printUnalignedVec(ostream& os, const GridPoint& gp) {
        printPointComment(os, gp, "Construct unaligned");

        // Declare var.
        string pvName = makeVarName();
        os << _linePrefix << getVarType() << " " << pvName << _lineSuffix;

        // Contruct it.
        printUnalignedVecCtor(os, gp, pvName);
        return pvName;
    }

    // Print per-element construction for one point var pvName from elems.
    void CppVecPrintHelper::printUnalignedVecSimple(ostream& os, const GridPoint& gp,
                                                    const string& pvName, string linePrefix,
                                                    const set<size_t>* doneElems) {

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
            assert(_vecVars.count(ve._vec));
            string mvName = _vecVars[ve._vec];

            // which element?
            int alignedElem = ve._offset; // 1-D layout of this element.
            string elemStr = ve._offsets.makeDimValOffsetStr();

            os << linePrefix << pvName << "[" << pelem << "] = " <<
                mvName << "[" << alignedElem << "];  // for " <<
                gp.getGridName() << "(" << elemStr << ")" << _lineSuffix;
        }
    }

    // Print init of element indices.
    // Fill _varMap as side-effect.
    void CppVecPrintHelper::printElemIndices(ostream& os) {
        auto& fold = getFold();
        os << endl << " // Element indices derived from vector indices." << endl;
        for (auto& dim : fold.getDims()) {
            auto& dname = dim.getName();
            string ename = dname + _elemSuffix;
            string cap_dname = PrinterBase::allCaps(dname);
            os << " const idx_t " << ename << " = " <<
                dname << " * VLEN_" << cap_dname << ";" << endl;
            _varMap[dname] = ename;
        }
    }

    // Print body of prefetch function.
    void CppVecPrintHelper::printPrefetches(ostream& os, const IntScalar& dir) const {

        // Points to prefetch.
        GridPointSet* pfPts = NULL;

        // Prefetch leading points only if dir name is set.
        GridPointSet edge;
        if (dir.getName().length()) {
            _vv.getLeadingEdge(edge, dir);
            pfPts = &edge;
        }

        // if dir is not set, prefetch all aligned vectors.
        else
            pfPts = &_vv._alignedVecs;

        for (auto gp : *pfPts) {
            printPointComment(os, gp, "Prefetch aligned");
            
            // Prefetch memory.
            printVecPointCall(os, gp, "prefetchVecNorm<level>", "", "__LINE__", true);
            os << ";" << endl;
        }
    }

    // Print an expression as a one-line C++ comment.
    void YASKCppPrinter::addComment(ostream& os, EqGroup& eq) {
        
        // Use a simple human-readable visitor to create a comment.
        PrintHelper ph(0, "temp", "", " // ", ".\n");
        PrintVisitorTopDown commenter(os, ph);
        eq.visitEqs(&commenter);
    }

} // namespace yask.

