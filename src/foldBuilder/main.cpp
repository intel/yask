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
#define STENCIL_REGISTRY stencils
#include "StencilBase.hpp"
StencilList STENCIL_REGISTRY;

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
string equationTargets;
bool doComb = false;
bool doCse = true;

void usage(const string& cmd) {

    cerr << "Options:\n"
        " -h                print this help message.\n"
        "\n"
        " -st <name>        set stencil type (required); supported stencils:\n";
    for (auto si :  STENCIL_REGISTRY) {
        auto name = si.first;
        cerr << "                     " << name << endl;
    }
    cerr <<
        "\n"
        " -fold <dim>=<size>,...    set number of elements in each dimension in a vector block.\n"
        " -cluster <dim>=<size>,... set number of values to evaluate in each dimension.\n"
        " -eq <name>=<substr>,...   put updates to grids containing substring in equation name.\n"
        " -or <order>        set stencil order (ignored for some stencils; default=" << order << ").\n"
        //" -dc                defer coefficient lookup to runtime (for iso3dfd stencil only).\n"
        " -lus               make last dimension of fold unit stride (instead of first).\n"
        " -aul               allow simple unaligned loads (memory map MUST be compatible).\n"
        " -es <expr-size>    set heuristic for expression-size threshold (default=" << exprSize << ").\n"
        " -[no]comb          [do not] combine commutative operations (default=" << doComb << ").\n"
        " -[no]cse           [do not] eliminate common subexpressions (default=" << doCse << ").\n"
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
            else if (opt == "-comb")
                doComb = true;
            else if (opt == "-nocomb")
                doComb = false;
            else if (opt == "-cse")
                doCse = true;
            else if (opt == "-nocse")
                doCse = false;
            
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
                string argop = argv[++argi];

                // options w/a string value.
                
                // stencil type.
                if (opt == "-st")
                    shapeName = argop;

                // equations.
                else if (opt == "-eq")
                    equationTargets = argop;

                // fold or cluster
                else if (opt == "-fold" || opt == "-cluster") {

                    // example: x=4,y=2
                    ArgParser ap;
                    ap.parseKeyValuePairs
                        (argop, [&](const string& key, const string& value) {
                            int size = atoi(value.c_str());
                            
                            // set dim in tuple.
                            if (opt == "-fold")
                                foldOptions.addDim(key, size);
                            else
                                clusterOptions.addDim(key, size);
                        });
                }
                
                // add any more options w/a string value above.
                
                else {

                    // options w/an int value.
                    int val = atoi(argop.c_str());

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

    // Find the stencil in the registry.
    auto stencilIter = STENCIL_REGISTRY.find(shapeName);
    if (stencilIter == STENCIL_REGISTRY.end()) {
        cerr << "error: unknown stencil shape '" << shapeName << "'." << endl;
        usage(argv[0]);
    }
    stencilFunc = stencilIter->second;
    assert(stencilFunc);
    
    cerr << "Stencil name: " << shapeName << endl;
    if (stencilFunc->usesOrder()) {
        bool orderOk = stencilFunc->setOrder(order);
        if (!orderOk) {
            cerr << "error: invalid order=" << order << " for stencil type '" <<
                shapeName << "'." << endl;
            usage(argv[0]);
        }
        cerr << "Stencil order: " << order << endl;
    }
    cerr << "Expression-size threshold: " << exprSize << endl;
}

// Print an expression as a one-line C++ comment.
void addComment(ostream& os, Grids& grids) {

    // Use a simple human-readable visitor to create a comment.
    PrintHelper ph(0, "temp", "", " // ", ".\n");
    PrintVisitorTopDown commenter(os, ph);
    grids.acceptToFirst(&commenter);
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

    // name of C++ struct.
    string context = "StencilContext_" + shapeName;
    
    // Set default fold ordering.
    IntTuple::setDefaultFirstInner(firstInner);
    
    // Reference to the grids and params in the stencil.
    Grids& grids = stencilFunc->getGrids();
    Params& params = stencilFunc->getParams();

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
        if (dimCounts.getVal(dim) == (int)grids.size()) {
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
            cerr << "Notice: memory map MUST be with unit-stride in " <<
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

    // Extract equations from grids.
    Equations equations;
    equations.findEquations(grids, equationTargets);
    equations.printInfo(cerr);

    // Get stats.
    {
        CounterVisitor cv;
        grids.acceptToFirst(&cv);
        cv.printStats(cerr, "for one vector");
    }
    if (clusterLengths.product() > 1) {
        CounterVisitor cv;
        grids.acceptToAll(&cv);
        cv.printStats(cerr, "for one cluster");
    }
    
    // Make a list of optimizations to apply.
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
            CounterVisitor cv;
            grids.acceptToAll(&cv);
            cv.printStats(cerr, descr);
            //addComment(cerr, grids);
        }
        else
            cerr << "No changes " << descr << '.' << endl;
    }
    
    // Human-readable output.
    if (printPseudo) {

        // Loop through all equations.
        for (auto& eq : equations) {
            
            cout << endl << "////// Equation '" << eq.name <<
                "' //////" << endl;

            CounterVisitor cv;
            eq.grids.acceptToAll(&cv);
            PrintHelper ph(&cv, "temp", "real", " ", ".\n");

            cout << "// Top-down stencil calculation:" << endl;
            PrintVisitorTopDown pv1(cout, ph);
            eq.grids.acceptToAll(&pv1);
            
            cout << endl << "// Bottom-up stencil calculation:" << endl;
            PrintVisitorBottomUp pv2(cout, ph, exprSize);
            eq.grids.acceptToAll(&pv2);
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

        // Loop through all equations.
        for (auto& eq : equations) {

            // TODO: separate mutiple grids.
            POVRayPrintVisitor pv(cout);
            eq.grids.acceptToFirst(&pv);
            cout << "// " << pv.getNumPoints() << " stencil points" << endl;
        }
    }

    // Print classes to update grids and/or prefetch.
    if (printCpp) {

        cout << "// Automatically generated code; do not edit." << endl;

        cout << endl << "////// Implementation of the '" << shapeName <<
            "' stencil //////" << endl;
        cout << endl << "namespace yask {" << endl;

        // Create the overall context class.
        {
            // get stats for just one element (not cluster).
            CounterVisitor cve;
            grids.acceptToFirst(&cve);
            IntTuple maxHalos;

            cout << endl << "////// Overall stencil-context class //////" << endl <<
                "struct " << context << " : public StencilContext {" << endl;

            // Grids.
            cout << endl << " // Grids." << endl;
            map<Grid*, string> typeNames, dimArgs, padArgs;
            for (auto gp : grids) {
                assert (!gp->isParam());
                string grid = gp->getName();

                // Type name & ctor params.
                // Name in kernel is 'Grid_' followed by dimensions.
                string typeName = "Grid_";
                string dimArg, padArg;
                for (auto dim : gp->getDims()) {
                    string ucDim = allCaps(dim);
                    typeName += ucDim;

                    // don't want the time dimension during construction.
                    // TODO: make this more generic.
                    if (dim != "t") {
                        dimArg += "d" + dim + ", ";

                        // Halo for this dimension.
                        int halo = cve.getHalo(gp, dim);
                        string hvar = grid + "_halo_" + dim;
                        cout << " const idx_t " << hvar << " = " << halo << ";" << endl;

                        // Update max halo.
                        int* mh = maxHalos.lookup(dim);
                        if (mh)
                            *mh = max(*mh, halo);
                        else
                            maxHalos.addDim(dim, halo);

                        // Total padding = halo + extra.
                        padArg += hvar + " + p" + dim + ", ";
                    }
                }
                typeNames[gp] = typeName;
                dimArgs[gp] = dimArg;
                padArgs[gp] = padArg;
                cout << " " << typeName << "* " << grid << "; // ";
                if (equations.getEqGrids().count(gp) == 0)
                    cout << "not ";
                cout << "updated by stencil." << endl;
            }

            // Max halos.
            cout << endl << " // Max halos across all grids." << endl;
            for (auto dim : maxHalos.getDims())
                cout << " const idx_t max_halo_" << dim << " = " <<
                    maxHalos.getVal(dim) << ";" << endl;

            // Parameters.
            map<Param*, string> paramTypeNames, paramDimArgs;
            cout << endl << " // Parameters." << endl;
            for (auto pp : params) {
                assert(pp->isParam());
                string param = pp->getName();

                // Type name.
                // Name in kernel is 'GenericGridNd<real_t>'.
                ostringstream oss;
                oss << "GenericGrid" << pp->size() << "d<real_t";
                if (pp->size()) {
                    oss << ",Layout_";
                    for (int dn = pp->size(); dn > 0; dn--)
                        oss << dn;
                }
                oss << ">";
                string typeName = oss.str();

                // Ctor params.
                string dimArg = pp->makeValStr();
            
                paramTypeNames[pp] = typeName;
                paramDimArgs[pp] = dimArg;
                cout << " " << typeName << "* " << param << ";" << endl;
            }

            // Ctor.
            cout << endl << " " << context << "() {" << endl <<
                "  name = \"" << shapeName << "\";" << endl;

            // Init grid ptrs.
            for (auto gp : grids) {
                string grid = gp->getName();
                cout << "  " << grid << " = 0;" << endl;
            }

            // Init param ptrs.
            for (auto pp : params) {
                string param = pp->getName();
                cout << "  " << param << " = 0;" << endl;
            }

            // end of ctor.
            cout << " }" << endl;

            // Allocate grids.
            cout << endl << " virtual void allocGrids() {" << endl;
            cout << "  gridPtrs.clear();" << endl;
            cout << "  eqGridPtrs.clear();" << endl;
            for (auto gp : grids) {
                string grid = gp->getName();
                cout << "  " << grid << " = new " << typeNames[gp] <<
                    "(" << dimArgs[gp] << padArgs[gp] << "\"" << grid << "\");" << endl <<
                    "  gridPtrs.push_back(" << grid << ");" << endl;

                // Grids w/equations.
                if (equations.getEqGrids().count(gp))
                    cout << "  eqGridPtrs.push_back(" << grid  << ");" << endl;
            }
            cout << " }" << endl;

            // Allocate params.
            cout << endl << " virtual void allocParams() {" << endl;
            cout << "  paramPtrs.clear();" << endl;
            for (auto pp : params) {
                string param = pp->getName();
                cout << "  " << param << " = new " << paramTypeNames[pp] <<
                    "(" << paramDimArgs[pp] << ");" << endl <<
                    "  paramPtrs.push_back(" << param << ");" << endl;
            }
            cout << " }" << endl;

            // end of context.
            cout << "};" << endl;
        }
        
        // Loop through all equations.
        for (auto& eq : equations) {

            cout << endl << "////// Stencil equation '" << eq.name <<
                "' //////" << endl;

            cout << endl << "struct Stencil_" << eq.name << " {" << endl <<
                " std::string name = \"" << eq.name << "\";" << endl;

            // Ops for this equation.
            CounterVisitor fpops;
            eq.grids.acceptToFirst(&fpops);
            
            // Example computation.
            cout << endl << " // " << fpops.getNumOps() << " FP operation(s) per point:" << endl;
            addComment(cout, eq.grids);
            cout << " const int scalar_fp_ops = " << fpops.getNumOps() << ";" << endl;

            // Init code.
            {
                cout << endl << " // All grids updated by this equation." << endl <<
                     " std::vector<RealVecGridBase*> eqGridPtrs;" << endl;

                cout << " void init(" << context << "& context) {" << endl;

                // Grids w/equations.
                cout << "  eqGridPtrs.clear();" << endl;
                for (auto gp : eq.grids) {
                    cout << "  eqGridPtrs.push_back(context." << gp->getName() << ");" << endl;
                }
                cout << " }" << endl;
            }
            
            // Scalar code.
            {
                // C++ scalar print assistant.
                CounterVisitor cv;
                eq.grids.acceptToFirst(&cv);
                CppPrintHelper* sp = new CppPrintHelper(&cv, "temp", "real_t", " ", ";\n");
            
                // Stencil-calculation code.
                // Function header.
                cout << endl << " // Calculate one scalar result relative to indices " <<
                    dimCounts.makeDimStr(", ") << "." << endl;
                cout << " void calc_scalar(" << context << "& context, " <<
                    dimCounts.makeDimStr(", ", "idx_t ") << ") {" << endl;

                // C++ code generator.
                // The visitor is accepted at all nodes in the AST;
                // for each node in the AST, code is generated and
                // stored in the expression-string in the visitor.
                // Visit only first expression in each, since we don't want clustering.
                PrintVisitorBottomUp pcv(cout, *sp, exprSize);
                eq.grids.acceptToFirst(&pcv);

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
                eq.grids.acceptToAll(&vv);

#if 1
                // Reorder based on vector info.
                ExprReorderVisitor erv(vv);
                eq.grids.acceptToAll(&erv);
#endif
            
                // C++ vector print assistant.
                CounterVisitor cv;
                eq.grids.acceptToFirst(&cv);
                CppVecPrintHelper* vp = NULL;
                if (print512Cpp)
                    vp = new CppAvx512PrintHelper(vv, allowUnalignedLoads, &cv,
                                                  "temp_vec", "real_vec_t", " ", ";\n");
                else if (print256Cpp)
                    vp = new CppAvx256PrintHelper(vv, allowUnalignedLoads, &cv,
                                                  "temp_vec", "real_vec_t", " ", ";\n");
                else if (printKncCpp)
                    vp = new CppKncPrintHelper(vv, allowUnalignedLoads, &cv,
                                               "temp_vec", "real_vec_t", " ", ";\n");
                else
                    vp = new CppVecPrintHelper(vv, allowUnalignedLoads, &cv,
                                               "temp_vec", "real_vec_t", " ", ";\n");
            
                // Stencil-calculation code.
                // Function header.
                int numResults = foldLengths.product() * clusterLengths.product();
                cout << endl << " // Calculate " << numResults <<
                    " result(s) relative to indices " << dimCounts.makeDimStr(", ") <<
                    " in a '" << clusterLengths.makeDimValStr(" * ") << "' cluster of '" <<
                    foldLengths.makeDimValStr(" * ") << "' vector(s)." << endl;
                cout << " // Indices must be normalized, i.e., already divided by VLEN_*." << endl;
                cout << " // SIMD calculations use " << vv.getNumPoints() <<
                    " vector block(s) created from " << vv.getNumAlignedVecs() <<
                    " aligned vector-block(s)." << endl;
                 cout << " // There are " << (fpops.getNumOps() * numResults) <<
                    " FP operation(s) per cluster." << endl;

                cout << " void calc_cluster(" << context << "& context, " <<
                    dimCounts.makeDimStr(", ", "idx_t ", "v") << ") {" << endl;

                // Element indices.
                cout << endl << " // Un-normalized indices." << endl;
                for (auto dim : dimCounts.getDims()) {
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
                eq.grids.acceptToAll(&pcv);

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
                            " relative to indices " << dimCounts.makeDimStr(", ") <<
                            " in a '" << clusterLengths.makeDimValStr(" * ") << "' cluster of '" <<
                            foldLengths.makeDimValStr(" * ") << "' vector(s)." << endl;
                        cout << "// Indices must be normalized, i.e., already divided by VLEN_*." << endl;

                        cout << " void prefetch_" << hint << "_vector";
                        if (dir.size())
                            cout << "_" << dir.getDirName();
                        cout << "(" << context << "& context, " <<
                            dimCounts.makeDimStr(", ", "idx_t ", "v") << ") {" << endl;

                        // C++ prefetch code.
                        vp->printPrefetches(cout, dir, hint);

                        // End of function.
                        cout << "}" << endl;

                    } // cache level.
                } // direction.

                delete vp;
            }

            cout << "};" << endl; // end of class.
            
        } // stencil equations.

        // Create a class for all equations.
        cout << endl << "////// Overall stencil-equations class //////" << endl <<
            "template <typename ContextClass>" << endl <<
            "struct StencilEquations_" << shapeName <<
            " : public StencilEquations {" << endl;

        // Stencil equation objects.
        cout << endl << " // Stencils." << endl;
        for (auto& eq : equations)
            cout << " StencilTemplate<Stencil_" << eq.name << "," <<
                context << "> stencil_" << eq.name << ";" << endl;

        // Ctor.
        cout << endl << " StencilEquations_" << shapeName << "() {" << endl <<
            "name = \"" << shapeName << "\";" << endl;

        // Push stencils to list.
        for (auto& eq : equations)
            cout << "  stencils.push_back(&stencil_" << eq.name << ");" << endl;
        cout << " }" << endl;

        cout << "};" << endl;
        cout << "} // namespace yask." << endl;
        
    } // Any C++ code.

    // Print CPP macros.
    // TODO: many hacks below assume certain dimensions and usage model
    // by the kernel. Need to improve kernel to make it more flexible
    // and then communicate info more generically.
    if (printMacros) {

        cout << "// Automatically generated code; do not edit." << endl;

        cout << endl;
        cout << "// Stencil:" << endl;
        cout << "#define STENCIL_NAME \"" << shapeName << "\"" << endl;
        cout << "#define STENCIL_IS_" << allCaps(shapeName) << " (1)" << endl;
        cout << "#define STENCIL_CONTEXT " << context << endl;
        cout << "#define STENCIL_EQUATIONS StencilEquations_" << shapeName <<
            "<" << context << ">" << endl;

        cout << endl;
        cout << "// Dimensions:" << endl;
        for (auto dim : dimCounts.getDims()) {
            cout << "#define USING_DIM_" << allCaps(dim) << " (1)" << endl;
        }
        
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
