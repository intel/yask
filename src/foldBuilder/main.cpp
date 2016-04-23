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

// Stencils.
#include "ExampleStencil.hpp"
#include "Iso3dfdStencil.hpp"
#include "AveStencil.hpp"
#include "AwpStencil.hpp"

// vars set via cmd-line options.
bool printPseudo = false;
bool printPOVRay = false;
bool printMacros = false;
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
bool deferCoeff = false;
int order = 2;
bool firstInner = true;
bool allowUnalignedLoads = false;

void usage(const string& cmd) {
    cout << "Options:\n"
        " -h                print this help message.\n"
        "\n"
        " -st <iso3dfd|3axis|9axis|3plane|cube|ave|awp>   set stencil type (required).\n"
        "\n"
        " -or <order>           set stencil order (ignored for awp; default=" << order << ").\n"
        " -dc                   defer coefficient lookup to runtime (for iso3dfd stencil only).\n"
        " -es <expr-size>       set heuristic for expression-size threshold (default=" << exprSize << ").\n"
        " -fold <dim>=<size>,...  set number of elements in each dimension in a vector block.\n"
        " -cluster <dim>=<size>,... set number of values to evaluate in each dimension.\n"
        " -lus                  make last dimension of fold unit stride (instead of first).\n"
        " -aul                  allow simple unaligned loads (memory layout MUST be compatible).\n"
        "\n"
        //" -ps <vec-len>      print stats for all folding options for given vector length.\n"
        " -ph                print human-readable scalar pseudo-code for one point.\n"
        " -pp                print POV-Ray code for one fold.\n"
        " -pm                print C++ pre-processor macros for current cluster and fold settings.\n"
        " -pcpp              print C++ stencil functions.\n"
        " -pknc              print C++ stencil functions for KNC ISA.\n"
        " -p512              print C++ stencil functions for CORE AVX-512 & MIC AVX-512 ISAs.\n"
        " -p256              print C++ stencil functions for CORE AVX & AVX2 ISAs.\n"
        "\n"
        "Examples:\n"
        " " << cmd << " -st iso3dfd -or 8 -fold x=4,y=4 -pscpp\n"
        " " << cmd << " -st awp -fold y=4,z=2 -pv512\n";
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

            else if (opt == "-ph")
                printPseudo = true;
            else if (opt == "-pp")
                printPOVRay = true;
            else if (opt == "-dc")
                deferCoeff = true;
            else if (opt == "-lus")
                firstInner = false;
            else if (opt == "-aul")
                allowUnalignedLoads = true;
            
            else if (opt == "-pm")
                printMacros = true;
            else if (opt == "-pcpp")
                printCpp = true;
            else if (opt == "-pknc") {
                printCpp = true;
                printKncCpp = true;
            }
            else if (opt == "-p512") {
                printCpp = true;
                print512Cpp = true;
            }
            else if (opt == "-p256") {
                printCpp = true;
                print256Cpp = true;
            }
            
            // add any more options w/o values above.

            // options w/a value.
            else {

                // at least one value needed.
                if (argi + 1 >= argc) {
                    cerr << "error: value missing or bad option '" << opt << "'." << endl;
                    usage(argv[0]);
                }

                // options w/a string value.
                // stencil type.
                if (opt == "-st") {
                    shapeName = argv[++argi];
                }

                // fold or cluster
                else if (opt == "-fold" || opt == "-cluster") {

                    // example: x=4,y=2
                    string argStr = argv[++argi];

                    // split by commas.
                    vector<string> args;
                    string arg;
                    for (char c1 : argStr) {
                        if (c1 == ',') {
                            args.push_back(arg);
                            arg = "";
                        } else
                            arg += c1;
                    }
                    args.push_back(arg);

                    // process each dim.
                    for (auto dimStr : args) {
                        
                        // split by equal sign.
                        size_t ep = dimStr.find("=");
                        if (ep == string::npos) {
                            cerr << "Error: no equal sign in '" << dimStr << "'." << endl;
                            usage(argv[0]);
                        }
                        string dim = dimStr.substr(0, ep);
                        string sizeStr = dimStr.substr(ep+1);
                        int size = atoi(sizeStr.c_str());
                        //cerr << dimStr << ": " << dim << "=" << size << endl;

                        // set dim in tuple.
                        if (opt == "-fold")
                            foldOptions.addDim(dim, size);
                        else
                            clusterOptions.addDim(dim, size);
                    }

                }
                
                // add any more options w/a string value above.
                
                else {

                    // options w/an int value.
                    int val = atoi(argv[++argi]);

                    if (opt == "-es")
                        exprSize = val;

                    else if (opt == "-or")
                        order = val;

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

    // construct the stencil object based on the name and other options.
    if (shapeName == "iso3dfd") {
        Iso3dfdStencil* iso3dfd = new Iso3dfdStencil(order);
        iso3dfd->setDeferCoeff(deferCoeff);
        stencilFunc = iso3dfd;
    }
    else if (shapeName == "ave")
        stencilFunc = new AveStencil(order);
    else if (shapeName == "awp")
        stencilFunc = new AwpStencil();
    else if (shapeName == "3axis")
        stencilFunc = new AxisStencil(order);
    else if (shapeName == "9axis")
            stencilFunc = new DiagStencil(order);
    else if (shapeName == "3plane")
        stencilFunc = new PlaneStencil(order);
    else if (shapeName == "cube")
        stencilFunc = new CubeStencil(order);
    else {
        cerr << "error: unknown stencil shape '" << shapeName << "'." << endl;
        usage(argv[0]);
    }
    assert(stencilFunc);
    
    cerr << "Stencil type: " << shapeName << endl;
    cerr << "Stencil order: " << order << endl;
    cerr << "Expression-size threshold: " << exprSize << endl;

    // Make sure order is ok by resetting it.
    bool orderOk = stencilFunc->setOrder(stencilFunc->getOrder());
      
    if (!orderOk) {
        cerr << "error: invalid order=" << order << " for stencil type '" <<
            shapeName << "'." << endl;
        usage(argv[0]);
    }
}

// Print an expression as a one-line C++ comment.
void addComment(ostream& os, Grid* gp) {

    // Use a simple human-readable visitor to create a comment.
    PrintHelper ph(0, "temp", "", " // ", ".\n");
    PrintVisitorTopDown commenter(os, ph);
    gp->acceptToFirst(&commenter);
}

// Return an upper-case string.
string allCaps(string str) {
    transform(str.begin(), str.end(), str.begin(), ::toupper);
    return str;
}

// Main program.
int main(int argc, const char* argv[]) {

    // parse options.
    parseOpts(argc, argv);

    // Set default fold ordering.
    IntTuple::setDefaultFirstInner(firstInner);
    
    // Reference to the grids in the stencil.
    Grids& grids = stencilFunc->getGrids();

    // Create a union of all dimensions in all grids.
    // Also keep count of how many grids have each dim.
    // Note that dimensions won't be in any particular order!
    IntTuple dimCounts;
    for (auto gp : grids) {

        // Count dimensions from this grid.
        for (auto dim : gp->getDims()) {
            if (dimCounts.lookup(dim))
                dimCounts.setVal(dim, dimCounts.getVal(dim) + 1);
            else
                dimCounts.addDim(dim, 1);
        }
    }

    // For now, there are only global specifications for vector and cluster
    // sizes. Also, vector folding and clustering is done identially for
    // every grid access. Thus, sizes > 1 must exist in all grids.  So, init
    // vector and cluster sizes based on dimensions that appear in ALL
    // grids.
    // TODO: relax this restriction.
    IntTuple foldLengths, clusterLengths;
    for (auto dim : dimCounts.getDims()) {
        if (dimCounts.getVal(dim) == grids.size()) {
            foldLengths.addDim(dim, 1);
            clusterLengths.addDim(dim, 1);
        }
    }

    // Create final fold lengths based on cmd-line options.
    IntTuple foldLengthsGT1;    // fold dimensions > 1.
    for (auto dim : foldOptions.getDims()) {
        int sz = foldOptions.getVal(dim);
        int* p = foldLengths.lookup(dim);
        if (!p) {
            cerr << "Error: fold-length of " << sz << " in '" << dim <<
                "' dimension not allowed because '" <<
                dim << "' doesn't exist in all grids." << endl;
            exit(1);
        }
        *p = sz;
        if (sz > 1)
            foldLengthsGT1.addDim(dim, sz);
            
    }
    cerr << "Vector-fold dimensions: " << foldLengths.makeDimValStr(" * ") << endl;

    // Checks for unaligned loads.
    if (allowUnalignedLoads) {
        if (foldLengthsGT1.size() > 1) {
            cerr << "Error: attempt to allow unaligned loads when there are " <<
                foldLengthsGT1.size() << " dimensions in the vector-fold that are > 1." << endl;
            exit(1);
        }
        else if (foldLengthsGT1.size() > 0)
            cerr << "Notice: memory layout MUST be with unit-stride in " <<
                foldLengthsGT1.makeDimStr() << " dimension!" << endl;
    }

    // Create final cluster lengths based on cmd-line options.
    for (auto dim : clusterOptions.getDims()) {
        int sz = clusterOptions.getVal(dim);
        int* p = clusterLengths.lookup(dim);
        if (!p) {
            cerr << "Error: cluster-length of " << sz << " in '" << dim <<
                "' dimension not allowed because '" <<
                dim << "' doesn't exist in all grids." << endl;
            exit(1);
        }
        *p = sz;
    }
    cerr << "Cluster dimensions: " << clusterLengths.makeDimValStr(" * ") << endl;
    
    // Loop through all points in a cluster.
    // For each point, determine the offset from 0,..,0 based
    // on the cluster point and fold lengths.
    // Then, construct an AST for all equations at this offset.
    // When done, for each equation, we will have an AST for each
    // cluster point stored in its respective grid.
    // TODO: check for illegal dependences between cluster points.
    clusterLengths.visitAllPoints([&](const IntTuple& clusterPoint){
            
            // Get starting offset of cluster, which is each cluster index
            // multipled by corresponding vector size.
            auto offsets = clusterPoint.multElements(foldLengths);

            // Add in any dims not in the cluster.
            for (auto dim : dimCounts.getDims()) {
                if (!offsets.lookup(dim))
                    offsets.addDim(dim, 0);
            }
            
            // Construct AST in grids for this cluster point.
            stencilFunc->define(offsets);
        });

    // Get some stats and eliminate common sub-expressions.
    int numFpOps = 0;
    for (int j = 0; j < 2; j++) {
        bool doCseElim = j > 0;
        string step = doCseElim ? "after CSE elimination" : "as written";
        
        for (int i = 0; i < 2; i++) {
            bool doCluster = i > 0;
            if (doCluster && clusterLengths.product() == 1)
                continue;
            string descr = doCluster ? "cluster" : "fold";
        
            // Eliminate the CSEs.
            if (doCseElim) {
                CseElimVisitor csee;
                for (auto gp : grids) {
                    if (doCluster)
                        gp->acceptToAll(&csee);
                    else
                        gp->acceptToFirst(&csee);
                }
                cerr << csee.getNumReplaced() << " CSE(s) eliminated in " << descr << "." << endl;
            }

            // Determine number of FP ops.
            CounterVisitor cv;
            for (auto gp : grids) {
                if (doCluster)
                    gp->acceptToAll(&cv);
                else
                    gp->acceptToFirst(&cv);
            }
            cerr << "Expression stats per " << descr << " " << step << ":" << endl <<
                "  " << cv.getNumNodes() << " nodes." << endl <<
                "  " << cv.getNumReads() << " grid reads." << endl <<
                "  " << cv.getNumWrites() << " grid writes." << endl <<
                "  " << cv.getNumOps() << " math operations." << endl;

            // Save final number of FP ops per point.
            if (!doCluster)
                numFpOps = cv.getNumOps();
        }
    }
    
    // Human-readable output.
    if (printPseudo) {

        // Loop through all grids.
        // Visit only first expression in each, since we don't want clustering.
        for (auto gp : grids) {

            // Skip if no expressions.
            if (gp->getExprs().size() == 0) {
                cout << endl << "////// No equation defined for grid '" << gp->getName() <<
                    "' //////" << endl;
                continue;
            }

            cout << endl << "////// Grid '" << gp->getName() <<
                "' //////" << endl;

            CounterVisitor cv;
            gp->acceptToFirst(&cv);
            PrintHelper ph(&cv, "temp", "real", " ", ".\n");

            cout << "// Top-down stencil calculation:" << endl;
            PrintVisitorTopDown pv1(cout, ph);
            gp->acceptToFirst(&pv1);
            
            cout << endl << "// Bottom-up stencil calculation:" << endl;
            PrintVisitorBottomUp pv2(cout, ph, exprSize);
            gp->acceptToFirst(&pv2);
        }
    }

    // POV-Ray output.
    if (printPOVRay) {

        cout << "#include \"stencil.inc\"" << endl;
        int cpos = stencilFunc->getOrder() + 1;
        cout << "camera { location <" <<
            cpos << ", " << (cpos-1) << ", " << cpos << ">" << endl <<
            "  look_at  <0, 0, 0>" << endl <<
            "}" << endl;

        // Loop through all grids.
        // Visit only first expression in each, since we don't want clustering.
        // TODO: separate different equations in space.
        for (auto gp : grids) {

            POVRayPrintVisitor pv(cout);
            gp->acceptToFirst(&pv);
            cout << "// " << pv.getNumPoints() << " stencil points" << endl;
        }
    }

    // Print classes to update grids and/or prefetch.
    if (printCpp) {

        cout << "// Automatically generated code; do not edit." << endl;
        cout << endl <<
            "// The following classes define equations for a total of " <<
            numFpOps << " FP operation(s) per point (as written)." << endl;
        string fnType = "void";

        // Loop through all grids.
        for (auto gp : grids) {
            string gname = gp->getName();

            // Skip if no expressions.
            if (gp->getExprs().size() == 0) {
                cout << endl << "////// No equation defined for grid '" << gp->getName() <<
                    "' //////" << endl;
                continue;
            }
            cout << endl << "////// Grid '" << gname <<
                "' //////" << endl;

            cout << endl << "class Stencil_" << gname << " {" << endl;

            // Ops for just this grid.
            CounterVisitor fpops;
            gp->acceptToFirst(&fpops);
            
            // Example computation.
            cout << " // " << fpops.getNumOps() << " FP operation(s) per point:" << endl;
            addComment(cout, gp);

            cout << endl << "public:" << endl;

            // Scalar code.
            {
                // C++ scalar print assistant.
                CounterVisitor cv;
                gp->acceptToFirst(&cv);
                CppPrintHelper* sp = new CppPrintHelper(&cv, "temp", "REAL", " ", ";\n");
            
                // Stencil-calculation code.
                // Function header.
                cout << endl << " // Calculate one scalar result relative to indices " <<
                    gp->makeDimStr(", ") << "." << endl;
                cout << fnType << " calc_scalar(StencilContext& context, " <<
                    gp->makeDimStr(", ", "idx_t ") << ") {" << endl;

                // C++ code generator.
                // The visitor is accepted at all nodes in the AST;
                // for each node in the AST, code is generated and
                // stored in the expression-string in the visitor.
                // Visit only first expression in each, since we don't want clustering.
                PrintVisitorBottomUp pcv(cout, *sp, exprSize);
                gp->acceptToFirst(&pcv);

                // End of function.
                cout << "} // scalar calculation." << endl;

                delete sp;
            }

            // Vector code.
            
            // Create vectors needed to implement.
            // The visitor is accepted at all nodes in the AST;
            // for each grid access node in the AST, the vectors
            // needed are determined and saved in the visitor.
            {
                // Create vector info.
                VecInfoVisitor vv(foldLengths);
                gp->acceptToAll(&vv);

                // Reorder based on vector info.
                ExprReorderVisitor erv(vv);
                gp->acceptToAll(&erv);
            
                // C++ vector print assistant.
                CounterVisitor cv;
                gp->acceptToFirst(&cv);
                CppVecPrintHelper* vp = NULL;
                if (print512Cpp)
                    vp = new CppAvx512PrintHelper(vv, allowUnalignedLoads, &cv,
                                                  "temp_vec", "realv", " ", ";\n");
                else if (print256Cpp)
                    vp = new CppAvx256PrintHelper(vv, allowUnalignedLoads, &cv,
                                                  "temp_vec", "realv", " ", ";\n");
                else if (printKncCpp)
                    vp = new CppKncPrintHelper(vv, allowUnalignedLoads, &cv,
                                               "temp_vec", "realv", " ", ";\n");
                else
                    vp = new CppVecPrintHelper(vv, allowUnalignedLoads, &cv,
                                               "temp_vec", "realv", " ", ";\n");
            
                // Stencil-calculation code.
                // Function header.
                int numResults = foldLengths.product() * clusterLengths.product();
                cout << endl << " // Calculate " << numResults <<
                    " result(s) relative to indices " << gp->makeDimStr(", ") <<
                    " in a '" << clusterLengths.makeDimValStr(" * ") << "' cluster of '" <<
                    foldLengths.makeDimValStr(" * ") << "' vector(s)." << endl;
                cout << " // Indices must be normalized, i.e., already divided by VLEN_*." << endl;
                cout << " // SIMD calculations use " << vv.getNumPoints() <<
                    " vector block(s) created from " << vv.getNumAlignedVecs() <<
                    " aligned vector-block(s)." << endl;
                cout << " // There are " << (fpops.getNumOps() * numResults) <<
                    " FP operation(s) per cluster (as written)." << endl;

                cout << fnType << " calc_vector(StencilContext& context, " <<
                    gp->makeDimStr(", ", "idx_t ", "v") << ") {" << endl;

                // Element indices.
                cout << endl << " // Un-normalized indices." << endl;
                for (auto dim : gp->getDims()) {
                    auto p = foldLengths.lookup(dim);
                    cout << " idx_t " << dim << " = " << dim << "v";
                    if (p) cout << " * " << *p;
                    cout << ";" << endl;
                }
                
                // Code generator visitor.
                // The visitor is accepted at all nodes in the AST;
                // for each node in the AST, code is generated and
                // stored in the expression-string in the visitor.
                PrintVisitorBottomUp pcv(cout, *vp, exprSize);
                gp->acceptToAll(&pcv);

                // End of function.
                cout << "} // vector calculation." << endl;

                // Generate prefetch code for no specific direction and then each
                // orthogonal direction.
                for (int diri = -1; diri < dimCounts.size(); diri++) {

                    // Create a direction object.
                    // If diri < 0, there is no direction.
                    // If diri >= 0, add a direction.
                    IntTuple dir;
                    if (diri >= 0) {

                        // skip for now--not yet enabled.
                        continue;
                        
                        string dim = dimCounts.getDims()[diri];
                        dir.addDim(dim, 1);
                    }

                    // Prefetch code for each cache level.
                    for (int l = 1; l <= 2; l++) {
                        string hint = (l == 1) ? "L1" : "L2";
            
                        // Function header.
                        cout << endl << "// Prefetches cache line(s) ";
                        if (dir.size())
                            cout << "for leading edge of stencil in '+" <<
                                dir.getDirName() << "' direction ";
                        else
                            cout << "for entire stencil ";
                        cout << "to " << hint << " cache" <<
                            " relative to indices " << gp->makeDimStr(", ") <<
                            " in a '" << clusterLengths.makeDimValStr(" * ") << "' cluster of '" <<
                            foldLengths.makeDimValStr(" * ") << "' vector(s)." << endl;
                        cout << "// Indices must be normalized, i.e., already divided by VLEN_*." << endl;

                        cout << fnType << " prefetch_" << hint << "_vector";
                        if (dir.size())
                            cout << "_" << dir.getDirName();
                        cout << "(StencilContext& context, " <<
                            gp->makeDimStr(", ", "idx_t ", "v") << ") {" << endl;

                        // C++ prefetch code.
                        vp->printPrefetches(cout, dir, hint);

                        // End of function.
                        cout << "}" << endl;

                    } // cache level.
                } // direction.

                delete vp;
            }

            cout << "};" << endl; // end of class.
            
        } // grids.
    } // Any C++ code.

    // Print CPP macros.
    // FIXME: many hacks below assume certain dimensions and usage model
    // by the kernel. Need to improve kernel to make it more flexible
    // and then communicate info more generically.
    if (printMacros) {

        cout << "// Automatically generated code; do not edit." << endl;

        cout << endl;
        cout << "// Stencil:" << endl;
        cout << "#define STENCIL_NAME \"" << shapeName << "\"" << endl;
        cout << "#define STENCIL_ORDER " << stencilFunc->getOrder() << endl; // FIXME: calculate actual halo sizes.
        cout << "#define STENCIL_NUM_FP_OPS_SCALAR (" << numFpOps << ")" << endl;

        cout << endl;
        cout << "// Dimensions:" << endl;
        for (auto dim : dimCounts.getDims()) {
            cout << "#define USING_DIM_" << allCaps(dim) << " (1)" << endl;
        }
        
        cout << endl;
        cout << "// Grids:" << endl;
        string ptrDefns;
        string gridAllocs;
        string gridCompares;
        string newStencils;
        int n = 0, m = 0;
        for (auto gp : grids) {
            string grid = gp->getName();
            string ucGrid = allCaps(grid);

            // Calculations that need to be made.
            if (gp->getExprs().size()) {

                // add code to create a new object.
                string stencilClass = "Stencil_" + grid;
                if (m) newStencils += ", ";
                newStencils += "new StencilTemplate<" + stencilClass + ">(\"" + grid + "\")";

                m++;
            }

            // Type name.
            // FIXME: make this MUCH more generic.
            string typeName = "Grid_";
            string dimArgs, padArgs;
            for (auto dim : gp->getDims()) {
                string ucDim = allCaps(dim);
                typeName += ucDim;

                // don't want the time dimension during construction.
                if (dim != "t") {
                    dimArgs += "d" + dim + ", ";
                    padArgs += "pad" + dim + ", ";
                }
            }

            // Define pointers.
            if (n) ptrDefns += "; ";
            ptrDefns += typeName + "* " + grid;

            // Allocate grids and set vector.
            if (!n) gridAllocs = "gridPtrs.clear()";
            gridAllocs += "; " + grid + " = new " + typeName +
                "(" + dimArgs + padArgs + "\"" + grid + "\");"
                " gridPtrs.push_back(" + grid + ")";

            // Compare grids.
            if (n) gridCompares += " + ";
            gridCompares += grid + "->compare(*(ref." + grid + "))";
            
            n++;
        }
        cout << "#define GRID_PTR_DEFNS " << ptrDefns << endl;
        cout << "#define GRID_ALLOCS " << gridAllocs << endl;
        cout << "#define GRID_COMPARES " << gridCompares << endl;
        cout << "#define NEW_STENCIL_OBJECTS " << newStencils << endl;
        
        // Vec/cluster lengths.
        cout << endl;
        cout << "// One vector fold: " << foldLengths.makeDimValStr(" * ") << endl;
        for (auto dim : foldLengths.getDims()) {
            string ucDim = allCaps(dim);
            cout << "#define VLEN_" << ucDim << " (" << foldLengths.getVal(dim) << ")" << endl;
        }
        cout << "#define VLEN (" << foldLengths.product() << ")" << endl;
        cout << "#define VLEN_FIRST_DIM_IS_UNIT_STRIDE (" <<
            (IntTuple::getDefaultFirstInner() ? 1 : 0) << ")" << endl;
        cout << "#define USING_UNALIGNED_LOADS (" <<
            (allowUnalignedLoads ? 1 : 0) << ")" << endl;

        cout << endl;
        cout << "// Cluster of vector folds: " << clusterLengths.makeDimValStr(" * ") << endl;
        for (auto dim : clusterLengths.getDims()) {
            string ucDim = allCaps(dim);
            cout << "#define CLEN_" << ucDim << " (" << clusterLengths.getVal(dim) << ")" << endl;
        }
        cout << "#define CLEN (" << clusterLengths.product() << ")" << endl;
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
                        gp->acceptToFirst(&vv);
                        
                        // Print stats.
                        vv.printStats(cout, gp->getName(), separator);
                    }
                }
            }
        }
    }
#endif
    
    delete stencilFunc;
    return 0;
}
