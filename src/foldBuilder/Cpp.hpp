/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

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

////////// Support for C++ scalar and vector-code generation //////////////

#include "Vec.hpp"

/////////// Scalar code /////////////

// Outputs a C++ compilable scalar code.
class CppPrintHelper : public PrintHelper {

public:
    CppPrintHelper(const string& varPrefix,
                   const string& varType,
                   const string& linePrefix,
                   const string& lineSuffix) :
        PrintHelper(varPrefix, varType, linePrefix, lineSuffix) { }
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

    // Make call for a point.
    virtual string makePointCall(const GridPoint& gp,
                                 const string& fname, string optArg = "") const {
        ostringstream os;
        os << "context." << gp.getName() << "->" << fname << "(";
        if (optArg.length()) os << optArg << ", ";
        os << gp.makeDimValOffsetStr() << ", __LINE__)";
        return os.str();
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

// Output simple C++ vector code.
class CppVecPrintHelper : public VecPrintHelper {

public:
    CppVecPrintHelper(VecInfoVisitor& vv,
                      const string& varPrefix,
                      const string& varType,
                      const string& linePrefix,
                      const string& lineSuffix) :
        VecPrintHelper(vv, varPrefix, varType, linePrefix, lineSuffix) { }

protected:
    map<string,string> _codes; // code expressions defined.
    map<double,string> _consts; // const expressions defined.

    // Define a constant if not already defined, writing definition
    // statement to os.  Creates a vector with all elements set to v.  If
    // not needed, do not write to os.  Either way, return var name.
    virtual string addConstExpr(ostream& os, double v) {
        if (_consts.count(v) == 0) {

            // Create new var.
            _consts[v] = makeVarName();
            int vlen = _vv.getFold().product();

            // A comment.
            os << endl << " // Create a const vector with all values set to " << v << "." << endl;

            string vstr = CppPrintHelper::formatReal(v);
            os << setprecision(17) << scientific <<
                _linePrefix << "static const " <<
                getVarType() << " " << _consts[v] << " = { .r = { " << vstr;
            for (int i = 1; i < vlen; i++)
                os << ", " << vstr;
            os << " } }" << _lineSuffix;
        }
        return _consts[v];
    }

    // Define a var if not already defined, writing definition statement to
    // os.  Creates a vector with all elements set to code.  If not needed,
    // do not write to os.  Either way, return var name.
    virtual string addCodeExpr(ostream& os, const string& code) {
        if (_codes.count(code) == 0) {

            // Create new var.
            _codes[code] = makeVarName();
            int vlen = _vv.getFold().product();

            // A comment.
            os << endl << " // Create a vector with all values set to same value." << endl <<
                _linePrefix << getVarType() << " " << _codes[code] << _lineSuffix <<
                _linePrefix << _codes[code] << " = " << code << _lineSuffix;
        }
        return _codes[code];
    }

    // Print a comment about a point.
    virtual void printPointComment(ostream& os, const GridPoint& gp, const string& verb) const {

        os << endl << " // " << verb << " " << gp.getName() << " at " <<
            gp.makeDimValOffsetStr() << "." << endl;
    }

    // Print call for a point.
    virtual void printPointCall(ostream& os, const GridPoint& gp,
                                const string& fname, string optArg = "") const {
        os << "context." << gp.getName() << "->" << fname << "(";
        if (optArg.length()) os << optArg << ", ";
        os << gp.makeDimValNormOffsetStr(getFold()) << ", __LINE__)";
    }
    
    // Print required memory read.
    virtual string printAlignedVecRead(ostream& os, const GridPoint& gp) {
        printPointComment(os, gp, "Read aligned vector block from");

        // Read memory.
        string mvName = makeVarName();
        os << _linePrefix << getVarType() << " " << mvName << " = ";
        printPointCall(os, gp, "readVecNorm");
        os << _lineSuffix;
        return mvName;
    }

    // Print memory write.
    virtual string printAlignedVecWrite(ostream& os, const GridPoint& gp,
                                        const string& val) {
        printPointComment(os, gp, "Write aligned vector block to");

        // Write temp var to memory.
        printPointCall(os, gp, "writeVecNorm", val);
        os << ";" << endl;
        return val;
    }
    
    // Print conversion from memory vars to point var gp if needed.
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
                                         const string& pvName, string linePrefix) {

        // just assign each element in vector separately.
        auto& elems = _vv._vblk2elemLists[gp];
        assert(elems.size() > 0);
        for (size_t pelem = 0; pelem < elems.size(); pelem++) {
            auto& ve = elems[pelem]; // one vector element from gp.

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
    virtual void printPrefetches(ostream& os, const IntTuple& dir, string hint) const {

        // Points to prefetch.
        GridPointSet* pfPts = NULL;

        // Leading points only if dir is set.
        GridPointSet edge;
        if (dir.size() > 0) {
            _vv.getLeadingEdge(edge, dir);
            pfPts = &edge;
        }

        // if dir is not set, prefetch all points.
        else
            pfPts = &_vv._alignedVecs;

        os << " const char* p = 0;" << endl;
        for (auto gp : *pfPts) {
            printPointComment(os, gp, "Aligned");
            
            // Prefetch memory.
            os << " p = (const char*)";
            printPointCall(os, gp, "getVecPtrNorm");
            os << ";" << endl;
            os << " MCP(p, " << hint << ", __LINE__);" << endl;
            os << " _mm_prefetch(p, " << hint << ");" << endl;
        }
    }
};
