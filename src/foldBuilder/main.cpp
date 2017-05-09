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

/////////////// Main vector-folding code-generation code. /////////////

// Generation code.
#include "ExprUtils.hpp"
#include "CppIntrin.hpp"
#include "Parse.hpp"

using namespace yask;

// A register of stencils.
StencilList stencils;

// Stencils.
#include "stencils.hpp"

// Misc headers.
#include <fstream>

// output streams.
ostream* printPseudo = NULL;
ostream* printPOVRay = NULL;
ostream* printDOT = NULL;
ostream* printSimpleDOT = NULL;
ostream* printCpp = NULL;
ostream* printKncCpp = NULL;
ostream* print512Cpp = NULL;
ostream* print256Cpp = NULL;

// other vars set via cmd-line options.
int vlenForStats = 0;
StencilBase* stencilSoln = NULL;
string shapeName;
IntTuple foldOptions;                     // vector fold.
IntTuple clusterOptions;                  // cluster sizes.
int maxExprSize = 50;
int minExprSize = 2;
int radius = 1;
bool firstInner = true;
bool allowUnalignedLoads = false;
string eqGroupTargets;
bool doFuse = false;
bool doComb = false;
bool doCse = true;
string stepDim = "t";
int haloSize = 0;                     // 0 means auto.
int stepAlloc = 0;                    // 0 means auto.
bool find_deps = true;                // find dependencies between equations.
string eq_group_basename_default = "stencil";

ostream* open_file(const string& name) {
    
    // Use '-' for stdout.
    if (name == "-")
        return &cout;

    ofstream* ofs = new ofstream(name, ofstream::out | ofstream::trunc);
    if (!ofs || !ofs->is_open()) {
        cerr << "Error: cannot open '" << name <<
            "' for output.\n";
        exit(1);
    }
    return ofs;
}

void close_file(ostream* os) {
    if (!os) return;
    if (os == &cout) return;
    if (os == &cerr) return;

    delete os;
}

void usage(const string& cmd) {

    cout << "Options:\n"
        " -h                 print this help message.\n"
        "\n"
        " -st <name>         set stencil type (required); supported stencils:\n";
    for (auto si : stencils) {
        auto name = si.first;
        auto sp = si.second;
        cout << "                     " << name;
        if (sp->usesRadius())
            cout << " *";
        cout << endl;
    }
    cout <<
        " -r <radius>\n"
        "     Set radius for stencils marked with '*' above (default=" << radius << ").\n"
        "\n"
        " -fold <dim>=<size>,...\n"
        "    Set number of elements in each dimension in a vector block.\n"
        " -cluster <dim>=<size>,...\n"
        "    Set number of vectors to evaluate in each dimension.\n"
        " -eq <name>=<substr>,...\n"
        "    Put updates to grids containing <substr> in equation-groups with base-name <name>.\n"
        "      By default, eq-groups are defined as needed based on dependencies with\n"
        "        base-name '" << eq_group_basename_default << "'.\n"
        "      Eq-groups are created in the order in which they are specified.\n"
        "        By default, they are created based on the order in which the grids are initialized.\n"
        "      Each eq-group base-name is appended with a unique index number.\n"
        "      Example: '-eq a=foo,b=bar' creates one or more eq-groups with base-name 'a'\n"
        "        containing updates to grids whose name contains 'foo' and one or more eq-groups\n"
        "        with base-name 'b' containing updates to grids whose name contains 'bar'.\n"
        //" [-no]-fuse         Do [not] pack grids together in meta container(s) (default=" << doFuse << ").\n"
        " -step <dim>\n"
        "    Specify stepping dimension <dim> (default='" << stepDim << "').\n"
        "      This is used for dependence calculation and memory allocation.\n"
        " -step-alloc <size>\n"
        "    Specify the size of the step-dimension memory allocation.\n"
        "      By default, allocations are calculated automatically for each grid.\n"
        " -halo <size>\n"
        "    Specify the sizes of the halos.\n"
        "      By default, halos are calculated automatically for each grid.\n"
        " [-no]-lus\n"
        "    Make last [first] dimension of fold unit stride (default=" << (!firstInner) << ").\n"
        "      This controls the intra-vector memory layout.\n"
        " [-no]-aul\n"
        "    Do [not] allow simple unaligned loads (default=" << allowUnalignedLoads << ").\n"
        "      To use this correctly, only 1D folds are allowed, and\n"
        "        the memory layout used by YASK must have that same dimension in unit stride.\n"
        " [-no]-comb\n"
        "    Do [not] combine commutative operations (default=" << doComb << ").\n"
        " [-no]-cse\n"
        "    Do [not] eliminate common subexpressions (default=" << doCse << ").\n"
        " -max-es <num-nodes>\n"
        "    Set heuristic for max single expression-size (default=" << maxExprSize << ").\n"
        " -min-es <num-nodes>\n"
        "    Set heuristic for min expression-size for reuse (default=" << minExprSize << ").\n"
        " [-no]-find-deps\n"
        "    Automatically find dependencies between equations (default=" << find_deps << ").\n"
        "\n"
        //" -ps <vec-len>         Print stats for all folding options for given vector length.\n"
        " -pm <filename>\n"
        "    Print YASK pre-processor macros.\n"
        //" -pg <filename>        Print YASK grid classes.\n"
        " -pcpp <filename>\n"
        "    Print YASK stencil classes for generic C++.\n"
        " -p256 <filename>\n"
        "    Print YASK stencil classes for CORE AVX & AVX2 ISAs.\n"
        " -p512 <filename>\n"
        "    Print YASK stencil classes for CORE AVX-512 & MIC AVX-512 ISAs.\n"
        " -pknc <filename>\n"
        "    Print YASK stencil classes for KNC ISA.\n"
        " -ph <filename>\n"
        "    Print human-readable scalar pseudo-code for one point.\n"
        " -pdot-full <filename>\n"
        "    Print DOT-language description of stencil equation(s).\n"
        " -pdot-lite <filename>\n"
        "    Print DOT-language description of grid dependencies.\n"
        //" -pp <filename>        Print POV-Ray code.\n"
        "\n"
        "Examples:\n"
        " " << cmd << " -st 3axis -r 2 -fold x=4,y=4 -ph -  # '-' for stdout\n"
        " " << cmd << " -st awp -fold x=4,y=2 -p256 stencil_code.hpp\n"
        " " << cmd << " -st iso3dfd -r 8 -fold x=4,y=4 -cluster y=2 -p512 stencil_code.hpp\n";
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
            else if (opt == "-no-lus")
                firstInner = true;
            else if (opt == "-aul")
                allowUnalignedLoads = true;
            else if (opt == "-no-aul")
                allowUnalignedLoads = false;
            else if (opt == "-find-deps")
                find_deps = true;
            else if (opt == "-no-find-deps")
                find_deps = false;
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

            // add any more options w/o values above.

            // options w/a value.
            else {

                // at least one value needed.
                if (argi + 1 >= argc) {
                    cerr << "Error: value missing or bad option '" << opt << "'." << endl;
                    usage(argv[0]);
                }
                string argop = argv[++argi];

                // options w/a string value.
                if (opt == "-st")
                    shapeName = argop;
                else if (opt == "-step")
                    stepDim = argop;
                else if (opt == "-eq")
                    eqGroupTargets = argop;
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

                // Print options w/filename arg.
                else if (opt == "-ph")
                    printPseudo = open_file(argop);
                else if (opt == "-pdot-full")
                    printDOT = open_file(argop);
                else if (opt == "-pdot-lite")
                    printSimpleDOT = open_file(argop);
                else if (opt == "-pp")
                    printPOVRay = open_file(argop);
                else if (opt == "-pcpp")
                    printCpp = open_file(argop);
                else if (opt == "-pknc")
                    printKncCpp = open_file(argop);
                else if (opt == "-p512")
                    print512Cpp = open_file(argop);
                else if (opt == "-p256")
                    print256Cpp = open_file(argop);
            
                // add any more options w/a string value above.
                
                else {

                    // options w/an int value.
                    int val = atoi(argop.c_str());

                    if (opt == "-max-es")
                        maxExprSize = val;
                    if (opt == "-min-es")
                        minExprSize = val;

                    else if (opt == "-r")
                        radius = val;

                    else if (opt == "-ps")
                        vlenForStats = val;

                    else if (opt == "-halo")
                        haloSize = val;
                    else if (opt == "-step-alloc")
                        stepAlloc = val;

                    // add any more options w/int values here.

                    else {
                        cerr << "Error: option '" << opt << "' not recognized." << endl;
                        usage(argv[0]);
                    }
                }
            }
        }
        else break;
    }
    if (argi < argc) {
        cerr << "Error: unrecognized parameter '" << argv[argi] << "'." << endl;
        usage(argv[0]);
    }
    if (shapeName.length() == 0) {
        cerr << "Error: shape not specified." << endl;
        usage(argv[0]);
    }

    // Find the stencil in the registry.
    auto stencilIter = stencils.find(shapeName);
    if (stencilIter == stencils.end()) {
        cerr << "Error: unknown stencil shape '" << shapeName << "'." << endl;
        usage(argv[0]);
    }
    stencilSoln = stencilIter->second;
    assert(stencilSoln);
    
    cout << "Stencil name: " << shapeName << endl;
    if (stencilSoln->usesRadius()) {
        bool rOk = stencilSoln->setRadius(radius);
        if (!rOk) {
            cerr << "Error: invalid radius=" << radius << " for stencil type '" <<
                shapeName << "'." << endl;
            usage(argv[0]);
        }
        cout << "Stencil radius: " << radius << endl;
    }
    cout << "Max expression-size threshold: " << maxExprSize << endl;
}

// Apply optimizations to eqGroups.
void optimizeEqGroups(EqGroups& eqGroups,
                      const string& descr,
                      bool printSets,
                      ostream& os) {

    // print stats.
    string edescr = "for " + descr + " equation-group(s)";
    eqGroups.printStats(os, edescr);
    
    // Make a list of optimizations to apply to eqGroups.
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

        eqGroups.visitEqs(optimizer);
        int numChanges = optimizer->getNumChanges();
        string odescr = "after applying " + optimizer->getName() + " to " +
            descr + " equation-group(s)";

        // Get new stats.
        if (numChanges)
            eqGroups.printStats(os, odescr);
        else
            os << "No changes " << odescr << '.' << endl;
    }

    // Final stats per equation group.
    if (printSets && eqGroups.size() > 1) {
        os << "Stats per equation-group:\n";
        for (auto eg : eqGroups)
            eg.printStats(os, "for " + eg.getDescription());
    }
}

// Main program.
int main(int argc, const char* argv[]) {

    cout << "YASK -- Yet Another Stencil Kernel\n"
        "YASK Stencil Compiler\n"
        "Copyright 2017, Intel Corporation.\n";
    
    // parse options.
    parseOpts(argc, argv);

    // Reference to objects in the stencil.
    Grids& grids = stencilSoln->getGrids();
    //Params& params = stencilSoln->getParams();
    Eqs& eqs = stencilSoln->getEqs();

    // Find all the stencil dimensions from the grids.
    // Create the final folds and clusters from the cmd-line options.
    Dimensions& dims = stencilSoln->getDims();
    dims.setDims(grids, stepDim,
                 foldOptions, firstInner, clusterOptions,
                 allowUnalignedLoads, cout);

    // Call the stencil 'define' method to create ASTs in grids.
    // All grid points will be relative to origin (0,0,...,0).
    stencilSoln->define(dims._allDims);

    // Check for illegal dependencies within equations for scalar size.
    if (find_deps) {
        cout << "Checking equation(s) with scalar operations...\n"
            " If this fails, review stencil equation(s) for illegal dependencies.\n";
        eqs.checkDeps(dims._scalar, dims._stepDim);
    }

    // Check for illegal dependencies within equations for vector size.
    if (find_deps) {
        cout << "Checking equation(s) with folded-vector operations...\n"
            " If this fails, the fold dimensions are not compatible with all equations.\n";
        eqs.checkDeps(dims._fold, dims._stepDim);
    }
    
    // Check for illegal dependencies within equations for cluster size and
    // also create equation groups based on legal dependencies.
    cout << "Checking equation(s) with clusters of vectors...\n"
        " If this fails, the cluster dimensions are not compatible with all equations.\n";
    EqGroups eqGroups(eq_group_basename_default, dims);
    eqGroups.makeEqGroups(eqs, eqGroupTargets, dims._clusterPts, find_deps);
    optimizeEqGroups(eqGroups, "scalar & vector", false, cout);

    // Make a copy of each equation at each cluster offset.
    // We will use these for inter-cluster optimizations and code generation.
    cout << "Constructing cluster of equations containing " <<
        dims._clusterMults.product() << " vector(s)...\n";
    EqGroups clusterEqGroups(eqGroups);
    clusterEqGroups.replicateEqsInCluster(dims);
    optimizeEqGroups(clusterEqGroups, "cluster", true, cout);

    ///// Print out above data based on -p* option(s).
    cout << "Generating requested output...\n";
    
    // Human-readable output.
    if (printPseudo) {
        PseudoPrinter printer(*stencilSoln, clusterEqGroups,
                              maxExprSize, minExprSize);
        printer.print(*printPseudo);
        close_file(printPseudo);
    }

    // DOT output.
    if (printDOT) {
        DOTPrinter printer(*stencilSoln, clusterEqGroups,
                           maxExprSize, minExprSize, false);
        printer.print(*printDOT);
        close_file(printDOT);
    }
    if (printSimpleDOT) {
        DOTPrinter printer(*stencilSoln, clusterEqGroups,
                           maxExprSize, minExprSize, true);
        printer.print(*printSimpleDOT);
        close_file(printSimpleDOT);
    }

    // POV-Ray output.
    if (printPOVRay) {
        POVRayPrinter printer(*stencilSoln, clusterEqGroups,
                              maxExprSize, minExprSize);
        printer.print(*printPOVRay);
        close_file(printPOVRay);
    }

    // Settings for YASK.
    YASKCppSettings yaskSettings;
    yaskSettings._allowUnalignedLoads = allowUnalignedLoads;
    yaskSettings._haloSize = haloSize;
    yaskSettings._stepAlloc = stepAlloc;
    yaskSettings._maxExprSize = maxExprSize;
    yaskSettings._minExprSize = minExprSize;
    
    // Print YASK code for various ISAs.
    if (printCpp) {
        YASKCppPrinter printer(*stencilSoln, eqGroups, clusterEqGroups,
                               dims, yaskSettings);
        printer.printCode(*printCpp);
        close_file(printCpp);
    }
    if (printKncCpp) {
        YASKKncPrinter printer(*stencilSoln, eqGroups, clusterEqGroups,
                               dims, yaskSettings);
        printer.printCode(*printKncCpp);
        close_file(printKncCpp);
    }
    if (print512Cpp) {
        YASKAvx512Printer printer(*stencilSoln, eqGroups, clusterEqGroups,
                                  dims, yaskSettings);
        printer.printCode(*print512Cpp);
        close_file(print512Cpp);
    }
    if (print256Cpp) {
        YASKAvx256Printer printer(*stencilSoln, eqGroups, clusterEqGroups,
                                  dims, yaskSettings);
        printer.printCode(*print256Cpp);
        close_file(print256Cpp);
    }

    // TODO: re-enable this.
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
                        gp->visitExprs(&vv);
                        
                        // Print stats.
                        vv.printStats(cout, gp->getName(), separator);
                    }
                }
            }
        }
    }
#endif

    cout << "YASK Stencil Compiler done.\n";
    return 0;
}
