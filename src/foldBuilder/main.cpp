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

/////////////// Main vector-folding code-generation code. /////////////

// Generation code.
#include "ExprUtils.hpp"
#include "CppIntrin.hpp"
#include "Parse.hpp"

// A register of stencils.
StencilList stencils;
#define REGISTER_STENCIL(Class) static Class registered_ ## Class(stencils)

// Stencils.
#include "ExampleStencil.hpp"
#include "Iso3dfdStencil.hpp"
#include "AveStencil.hpp"
#include "AwpStencil.hpp"
#include "AwpElasticStencil.hpp"
#include "StreamStencil.hpp"

// vars set via cmd-line options.
bool printPseudo = false;
bool printPOVRay = false;
bool printMacros = false;
bool printGrids = false;
bool printCpp = false;
bool printKncCpp = false;
bool print512Cpp = false;
bool print256Cpp = false;
int vlenForStats = 0;
StencilBase* stencilFunc = NULL;
string shapeName;
IntTuple foldOptions;                     // vector fold.
IntTuple clusterOptions;                  // cluster sizes.
int exprSize = 50;
int radius = 1;
bool firstInner = true;
bool allowUnalignedLoads = false;
string equationTargets;
bool doFuse = false;
bool hbwRW = true;
bool hbwRO = true;
bool doComb = false;
bool doCse = true;
string stepDim = "t";

void usage(const string& cmd) {

    cerr << "Options:\n"
        " -h                 print this help message.\n"
        "\n"
        " -st <name>         set stencil type (required); supported stencils:\n";
    for (auto si : stencils) {
        auto name = si.first;
        cerr << "                     " << name << endl;
    }
    cerr <<
        " -r <radius>        set stencil radius (ignored for some stencils; default=" << radius << ").\n"
        "\n"
        " -fold <dim>=<size>,...    set number of elements in each dimension in a vector block.\n"
        " -cluster <dim>=<size>,... set number of values to evaluate in each dimension.\n"
        " -eq <name>=<substr>,...   put updates to grids containing substring in equation name.\n"
        //" [-no]-fuse        do [not] pack grids together in meta container(s) (default=" << doFuse << ").\n"
        " -step <dim>        solution progresses in given dimension (default='" << stepDim << "').\n"
        " -lus               make last dimension of fold unit stride (instead of first).\n"
        " -aul               allow simple unaligned loads (memory map MUST be compatible).\n"
        " -es <expr-size>    set heuristic for expression-size threshold (default=" << exprSize << ").\n"
        " [-no]-comb         do [not] combine commutative operations (default=" << doComb << ").\n"
        " [-no]-cse          do [not] eliminate common subexpressions (default=" << doCse << ").\n"
        " [-no]-hbw-rw       do [not] allocate read/write grids in high-BW mem (default=" << hbwRW << ").\n"
        " [-no]-hbw-ro       do [not] allocate read-only grids in high-BW mem (default=" << hbwRO << ").\n"
        "\n"
        //" -ps <vec-len>      print stats for all folding options for given vector length.\n"
        " -ph                print human-readable scalar pseudo-code for one point.\n"
        " -pp                print POV-Ray code for one fold.\n"
        " -pm                print YASK pre-processor macros.\n"
        //" -pg                print YASK grid classes.\n"
        " -pcpp              print YASK stencil classes for generic C++.\n"
        " -p256              print YASK stencil classes for CORE AVX & AVX2 ISAs.\n"
        " -p512              print YASK stencil classes for CORE AVX-512 & MIC AVX-512 ISAs.\n"
        " -pknc              print YASK stencil classes for KNC ISA.\n"
        "\n"
        "Examples:\n"
        " " << cmd << " -st iso3dfd -or 8 -fold x=4,y=4 -p256\n"
        " " << cmd << " -st awp -fold y=4,z=2 -p512\n";
    exit(1);
}

// Parse command-line and set global cmd-line option vars.
// Exits on error.
void parseOpts(int argc, const char* argv[])
{
    if (argc <= 1)
        usage(argv[0]);

    int argi;               // current arg index.
    for (argi = 1; argi < argc; argi++) {
        if ( argv[argi][0] == '-' && argv[argi][1] ) {
            string opt = argv[argi];

            // options w/o values.
            if (opt == "-h" || opt == "-help" || opt == "--help")
                usage(argv[0]);

            else if (opt == "-lus")
                firstInner = false;
            else if (opt == "-aul")
                allowUnalignedLoads = true;
            else if (opt == "-comb")
                doComb = true;
            else if (opt == "-no-comb")
                doComb = false;
            else if (opt == "-cse")
                doCse = true;
            else if (opt == "-no-cse")
                doCse = false;
            else if (opt == "-fuse")
                doFuse = true;
            else if (opt == "-no-fuse")
                doFuse = false;

            else if (opt == "-hbw-rw")
                hbwRW = true;
            else if (opt == "-no-hbw-rw")
                hbwRW = false;
            else if (opt == "-hbw-ro")
                hbwRO = true;
            else if (opt == "-no-hbw-ro")
                hbwRO = false;
            
            else if (opt == "-ph")
                printPseudo = true;
            else if (opt == "-pp")
                printPOVRay = true;
            else if (opt == "-pm")
                printMacros = true;
            else if (opt == "-pg")
                printGrids = true;
            else if (opt == "-pcpp")
                printCpp = true;
            else if (opt == "-pknc")
                printKncCpp = true;
            else if (opt == "-p512")
                print512Cpp = true;
            else if (opt == "-p256")
                print256Cpp = true;
            
            // add any more options w/o values above.

            // options w/a value.
            else {

                // at least one value needed.
                if (argi + 1 >= argc) {
                    cerr << "error: value missing or bad option '" << opt << "'." << endl;
                    usage(argv[0]);
                }
                string argop = argv[++argi];

                // options w/a string value.
                if (opt == "-st")
                    shapeName = argop;
                else if (opt == "-step")
                    stepDim = argop;
                else if (opt == "-eq")
                    equationTargets = argop;
                else if (opt == "-fold" || opt == "-cluster") {

                    // example: x=4,y=2
                    ArgParser ap;
                    ap.parseKeyValuePairs
                        (argop, [&](const string& key, const string& value) {
                            int size = atoi(value.c_str());
                            
                            // set dim in tuple.
                            if (opt == "-fold")
                                foldOptions.addDimBack(key, size);
                            else
                                clusterOptions.addDimBack(key, size);
                        });
                }
                
                // add any more options w/a string value above.
                
                else {

                    // options w/an int value.
                    int val = atoi(argop.c_str());

                    if (opt == "-es")
                        exprSize = val;

                    else if (opt == "-r")
                        radius = val;

                    else if (opt == "-ps")
                        vlenForStats = val;

                    // add any more options w/int values here.

                    else {
                        cerr << "error: option '" << opt << "' not recognized." << endl;
                        usage(argv[0]);
                    }
                }
            }
        }
        else break;
    }
    if (argi < argc) {
        cerr << "error: unrecognized parameter '" << argv[argi] << "'." << endl;
        usage(argv[0]);
    }
    if (shapeName.length() == 0) {
        cerr << "error: shape not specified." << endl;
        usage(argv[0]);
    }

    // Find the stencil in the registry.
    auto stencilIter = stencils.find(shapeName);
    if (stencilIter == stencils.end()) {
        cerr << "error: unknown stencil shape '" << shapeName << "'." << endl;
        usage(argv[0]);
    }
    stencilFunc = stencilIter->second;
    assert(stencilFunc);
    
    cerr << "Stencil name: " << shapeName << endl;
    if (stencilFunc->usesRadius()) {
        bool rOk = stencilFunc->setRadius(radius);
        if (!rOk) {
            cerr << "error: invalid radius=" << radius << " for stencil type '" <<
                shapeName << "'." << endl;
            usage(argv[0]);
        }
        cerr << "Stencil radius: " << radius << endl;
    }
    cerr << "Expression-size threshold: " << exprSize << endl;
}

// Main program.
int main(int argc, const char* argv[]) {

    // parse options.
    parseOpts(argc, argv);

    // Set default fold ordering.
    IntTuple::setDefaultFirstInner(firstInner);
    
    // Reference to the grids and params in the stencil.
    Grids& grids = stencilFunc->getGrids();
    Params& params = stencilFunc->getParams();

    // Find all the stencil dimensions from the grids.
    // Create the final folds and clusters from the cmd-line options.
    Dimensions dims;
    dims.setDims(grids, stepDim,
                 foldOptions, clusterOptions,
                 allowUnalignedLoads, cerr);
    
    // Loop through all points in a cluster.
    // For each point, determine the offset from 0,..,0 based
    // on the cluster point and fold lengths.
    // Then, construct an AST for all equations at this offset.
    // When done, for each equation, we will have an AST for each
    // cluster point stored in its respective grid.
    // TODO: check for illegal dependences between cluster points.
    dims._clusterLengths.visitAllPoints([&](const IntTuple& clusterPoint){
            
            // Get starting offset of cluster, which is each cluster index
            // multipled by corresponding vector size.
            auto offsets = clusterPoint.multElements(dims._foldLengths);

            // Add any dims not in the cluster with offset 0.
            offsets.addDimBack(stepDim, 0);
            for (auto dim : dims._miscDims.getDims())
                offsets.addDimBack(dim, 0);
            
            // Construct AST in grids for this cluster point by calling
            // the 'define' method in the stencil.
            stencilFunc->define(offsets);
        });

    // Extract equations from grids.
    Equations equations;
    equations.findEquations(grids, equationTargets);
    equations.printInfo(cerr);

    // Get stats.
    equations.printStats(cerr, "for one vector", false);
    if (dims._clusterLengths.product() > 1)
        equations.printStats(cerr, "for one cluster", true);
    
    // Make a list of optimizations to apply to equations.
    vector<OptVisitor*> opts;
    if (doCse)
        opts.push_back(new CseVisitor);
    if (doComb) {
        opts.push_back(new CombineVisitor);
        if (doCse)
            opts.push_back(new CseVisitor);
    }
    
    // Apply opts.
    for (auto optimizer : opts) {

        grids.acceptToAll(optimizer);
        int numChanges = optimizer->getNumChanges();
        string descr = "after applying " + optimizer->getName();

        // Get new stats.
        if (numChanges) {
            equations.printStats(cerr, descr, true);
            //addComment(cerr, grids);
        }
        else
            cerr << "No changes " << descr << '.' << endl;
    }

    ///// Print out above data based on -p* option(s).
    
    // Human-readable output.
    if (printPseudo) {
        PseudoPrinter printer(*stencilFunc, equations, exprSize);
        printer.print(cout);
    }

    // POV-Ray output.
    if (printPOVRay) {
        POVRayPrinter printer(*stencilFunc, equations, exprSize);
        printer.print(cout);
    }

    // Print YASK classes to update grids and/or prefetch.
    YASKCppSettings yaskSettings;
    yaskSettings._allowUnalignedLoads = allowUnalignedLoads;
    yaskSettings._hbwRW = hbwRW;
    yaskSettings._hbwRO = hbwRO;
    if (printCpp) {
        YASKCppPrinter printer(*stencilFunc, equations,
                               exprSize, dims, yaskSettings);
        printer.printCode(cout);
    }
    if (printKncCpp) {
        YASKKncPrinter printer(*stencilFunc, equations,
                               exprSize, dims, yaskSettings);
        printer.printCode(cout);
    }
    if (print512Cpp) {
        YASKAvx512Printer printer(*stencilFunc, equations,
                                  exprSize, dims, yaskSettings);
        printer.printCode(cout);
    }
    if (print256Cpp) {
        YASKAvx256Printer printer(*stencilFunc, equations,
                                  exprSize, dims, yaskSettings);
        printer.printCode(cout);
    }

    // Print YASK classes for grids.
    if (printGrids) {
        YASKCppPrinter printer(*stencilFunc, equations,
                               exprSize, dims, yaskSettings);
        printer.printGrids(cout);
    }

    // Print CPP macros.
    if (printMacros) {
        YASKCppPrinter printer(*stencilFunc, equations,
                               exprSize, dims, yaskSettings);
        printer.printMacros(cout);
    }
    
#if 0
    // Print stats for various folding options.
    if (vlenForStats) {
        string separator(",");
        VecInfoVisitor::printStatsHeader(cout, separator);

        // Loop through all grids.
        for (auto gp : grids) {

            // Loop through possible folds of given length.
            for (int xlen = vlenForStats; xlen > 0; xlen--) {
                for (int ylen = vlenForStats / xlen; ylen > 0; ylen--) {
                    int zlen = vlenForStats / xlen / ylen;
                    if (vlenForStats == xlen * ylen * zlen) {
                        
                        // Create vectors needed to implement RHS.
                        VecInfoVisitor vv(xlen, ylen, zlen);
                        gp->acceptToAll(&vv);
                        
                        // Print stats.
                        vv.printStats(cout, gp->getName(), separator);
                    }
                }
            }
        }
    }
#endif
    
    return 0;
}
