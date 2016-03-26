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
#include "Intrin512.hpp"

// Stencils.
#include "ExampleStencil.hpp"
#include "Iso3dfdStencil.hpp"
#include "AveStencil.hpp"

// vars set via cmd-line options.
bool printPseudo = false;
bool printPOVRay = false;
bool printMacros = false;
bool printScalarCpp = false;
bool printVecCpp = false;
bool printKncCpp = false;
bool print512Cpp = false;
int order = -1;
int vlenForStats = 0;
int timeSteps = 1;
StencilBase* stencilFunc = NULL;
string shapeName;
Triple vlen;
Triple clen(1,1,1);
int exprSize = 50;
bool deferCoeff = false;
bool usesVel = false;

void usage() {
    printf("options:\n"
           " -h                print this help message.\n"
           "\n"
           " -st <iso3dfd|3axis|9axis|3plane|cube|ave>   set stencil type (required).\n"
           " -or <order>        set stencil order (required).\n"
           " -ts <time-steps>   set number of time-steps to calc per call (default=1).\n"
           " -cluster <x> <y> <z>  set number of vectors used in each dimension (default=1 1 1).\n"
           " -dc                defer coefficient lookup to runtime (for iso3dfd stencil only).\n"
           " -es <expr-size>    set heuristic for expression-size threshold (default=%d).\n", exprSize);
    printf("\n"
           " -ps <vec-len>      print stats for all folding options for vec-len.\n"
           " -ph                print human-readable scalar pseudo-code for one point.\n"
           " -pp                print POV-Ray code.\n"
           " -pscpp             print C++ stencil function for scalar value.\n"
           " -pm <x> <y> <z>    print pre-processor macros for x*y*z vector block.\n"
           " -pvcpp <x> <y> <z> print C++ stencil and prefetch functions for x*y*z vector block.\n"
           " -pvknc <x> <y> <z> print KNC C++ stencil and prefetch functions for x*y*z vector block.\n"
           " -pv512 <x> <y> <z> print CORE/MIC AVX-512 C++ stencil and prefetch functions for x*y*z vector block.\n");
    exit(1);
}

// Parse command-line and set global cmd-line option vars.
// Exits on error.
void parseOpts(int argc, const char* argv[]) 
{
    if (argc <= 1)
        usage();

    int argi;               // current arg index.
    for (argi = 1; argi < argc; argi++) {
        if ( argv[argi][0] == '-' && argv[argi][1] ) {
            string opt = argv[argi];

            // options w/o values.
            if (opt == "-h" || opt == "-help" || opt == "--help")
                usage();

            else if (opt == "-ph")
                printPseudo = true;
            else if (opt == "-pp")
                printPOVRay = true;
            else if (opt == "-dc")
                deferCoeff = true;
            else if (opt == "-pscpp")
                printScalarCpp = true;
            
            // add any more options w/o values above.

            // options w/a value.
            else {

                // at least one value needed.
                if (argi + 1 >= argc) {
                    cerr << "error: value missing or bad option '" << opt << "'." << endl;
                    usage();
                }

                // options w/a string value.
                // stencil type.
                if (opt == "-st") {
                    shapeName = argv[++argi];
                }
                
                // add any more options w/a string value above.
                
                else {

                    // options w/an int value.
                    int val = atoi(argv[++argi]);

                    if (opt == "-ts")
                        timeSteps = val;

                    else if (opt == "-es")
                        exprSize = val;

                    else if (opt == "-or")
                        order = val;

                    else if (opt == "-ps")
                        vlenForStats = val;

                    // add any more options w/int values here.

                    else {
                        // options w/3 int values.

                        // need 2 more args.
                        if (argi + 2 >= argc) {
                            cerr << "error: value missing or bad option '" << opt << "'." << endl;
                            usage();
                        }
                        Triple args(val, atoi(argv[++argi]), atoi(argv[++argi]));

                        if (opt == "-cluster")
                            clen = args;

                        else {
                            vlen = args;
                            if (opt == "-pm")
                                printMacros = true;
                            else if (opt == "-pvcpp")
                                printVecCpp = true;
                            else if (opt == "-pvknc") {
                                printVecCpp = true;
                                printKncCpp = true;
                            }
                            else if (opt == "-pv512") {
                                printVecCpp = true;
                                print512Cpp = true;
                            }
                            else {
                                cerr << "error: option '" << opt << "' not recognized." << endl;
                                usage();
                            }
                        }
                    }
                }
            }
        }
        else break;
    }
    if (argi < argc) {
        cerr << "error: unrecognized parameter '" << argv[argi] << "'." << endl;
        usage();
    }
    if (order < 0) {
        cerr << "error: order not specified." << endl;
        usage();
    }
    if (shapeName.length() == 0) {
        cerr << "error: shape not specified." << endl;
        usage();
    }

    // construct the stencil object based on the name and other options.
    // iso-3dfd stencil.
    if (shapeName == "iso3dfd") {
        Iso3dfdStencil* iso3dfd = new Iso3dfdStencil(order);
        iso3dfd->setDeferCoeff(deferCoeff);
        stencilFunc = iso3dfd;
        usesVel = true;
    }

    // other symmetric stencils.
    else {
        if (shapeName == "3axis")
            stencilFunc = new AxisStencil(order);
        else if (shapeName == "9axis")
            stencilFunc = new DiagStencil(order);
        else if (shapeName == "3plane")
            stencilFunc = new PlaneStencil(order);
        else if (shapeName == "cube")
            stencilFunc = new CubeStencil(order);
        else if (shapeName == "ave")
            stencilFunc = new AveStencil(order);
        else {
            cerr << "error: unknown stencil shape '" << shapeName << "'." << endl;
            usage();
        }
    }

    cerr << "Number of time steps: " << timeSteps << endl;
    cerr << "Stencil type: " << shapeName << endl;
    cerr << "Stencil order: " << order << endl;
    cerr << "Expression-size threshold: " << exprSize << endl;

    // Make sure order is ok.
    bool orderOk = stencilFunc->setOrder(order);
      
    if (!orderOk) {
        cerr << "error: invalid order=" << order << " for stencil type '" <<
            shapeName << "'." << endl;
        usage();
    }
}

// Print an expression as a one-line C++ comment.
void addComment(ostream& os, ExprPtr ep, string descrip) {

    // Use a simple human-readable visitor to create a comment.
    PrintHelper ph;
    PrintVisitorTopDown commenter(os, ph);
    ep->accept(&commenter);
    os << endl <<
        " // Example: the following code calculates '" <<
        commenter.getExprStr() <<
        "' " << descrip << "." << endl;
}

// Main program.
int main(int argc, const char* argv[]) {

    // parse options.
    parseOpts(argc, argv);

    // Construct AST at the 0,0,0 corner of each vector
    // in the cluster by evaluating the stencil value function.
    // TODO: handle stencils that behave differently
    // at various locations, e.g., boundaries.
    TemporalGrid u("grid");
    typedef vector<GridValue> GridValueList;
    GridValueList asts;
    for (int k = 0; k < clen.getZLen(); k++) {
        int kp = k * vlen.getZLen();
        for (int j = 0; j < clen.getYLen(); j++) {
            int jp = j * vlen.getYLen();
            for (int i = 0; i < clen.getXLen(); i++) {
                int ip = i * vlen.getXLen();

                // ip,jp,kp is the 0,0,0 corner of the vector cluster i,j,k.
                // We want the value at timeSteps based on the value at 0.
                GridValue ast = stencilFunc->value(u, timeSteps, 0, ip, jp, kp);
                assert(ast);
                asts.push_back(ast);

                // map321() converts 3D indices to 1D index.
                assert(asts.size() == clen.map321(i, j, k) + 1); 
            }
        }
    }

    // Print stats for various folding options.
    if (vlenForStats) {
        string separator(",");

        VecInfoVisitor::printStatsHeader(cout, separator);

        for (int xlen = vlenForStats; xlen > 0; xlen--) {
            for (int ylen = vlenForStats / xlen; ylen > 0; ylen--) {
                int zlen = vlenForStats / xlen / ylen;
                if (vlenForStats == xlen * ylen * zlen) {

                    // Create vectors needed to implement.
                    VecInfoVisitor vv(xlen, ylen, zlen);
                    for (auto ast : asts)
                        ast->accept(&vv);
                
                    // Print stats.
                    vv.printStats(cout, separator);
                }
            }
        }
    }

    // Human-readable output.
    if (printPseudo) {
        cout << "// Top-down stencil calculation:" << endl;
        PrintHelper ph;
        PrintVisitorTopDown pv1(cout, ph);
        asts[0]->accept(&pv1);
        cout << u.name() << "(1, 0, 0, 0, 0) = " << pv1.getExprStr() << ";" << endl;

        cout << endl << "// Bottom-up stencil calculation:" << endl;
        PrintVisitorBottomUp pv2(cout, ph, exprSize);
        asts[0]->accept(&pv2);
        cout << u.name() << "(1, 0, 0, 0, 0) = " << pv2.getExprStr() << ";" << endl;
    }

    // POV-Ray output.
    if (printPOVRay) {

        cout << "#include \"stencil.inc\"" << endl;

        int cpos = order + 1;
        cout << "camera { location <" <<
            cpos << ", " << (cpos-1) << ", " << cpos << ">" << endl <<
            "  look_at  <0, 0, 0>" << endl <<
            "}" << endl;

        POVRayPrintVisitor pv(cout);
        asts[0]->accept(&pv);
        cout << "// " << pv.getNumPoints() << " stencil points" << endl;
    }

    // Print CPP macros.
    // TODO: make these real vars instead of macros.
    if (printMacros) {

        cout << "// Automatically generated code; do not edit." << endl;

        // Vec lengths.
        cout << endl;
        cout << "// Vector fold:" << endl;
        cout << "#define VLEN_X " << vlen.getXLen() << endl;
        cout << "#define VLEN_Y " << vlen.getYLen() << endl;
        cout << "#define VLEN_Z " << vlen.getZLen() << endl;
        cout << "#define VLEN " << vlen.product() << endl;
        cout << "#define CLEN_X " << clen.getXLen() << endl;
        cout << "#define CLEN_Y " << clen.getYLen() << endl;
        cout << "#define CLEN_Z " << clen.getZLen() << endl;
        cout << "#define CLEN " << clen.product() << endl;
        cout << "#define TIME_STEPS " << timeSteps << endl;
        cout << endl;
        cout << "// Stencil:" << endl;
        cout << "#define STENCIL_SHAPE_NAME \"" << shapeName << "\"" << endl;
        cout << "#define STENCIL_ORDER " << order << endl;
        if (usesVel)
            cout << "#define STENCIL_USES_VEL" << endl;
    }

    
    // Print a function to evaluate the stencil for a scalar value.
    if (printScalarCpp) {

        cout << "// Automatically generated code; do not edit." << endl;

        // C++ scalar print assistant.
        CppPrintHelper* vp = new CppPrintHelper();
        vp->setVarType("REAL");
            
        // Stencil-calculation code.
        // Function header.
        cout << endl << "// Calculate one result for order " << order <<
            " " << shapeName << " stencil at time t0 + TIME_STEPS,"
            " work variable v0, and scalar location i, j, k." << endl;
        cout << "void calc_stencil_scalar(StencilContext& context, "
            "int t0, int v0, int i, int j, int k) {" << endl;
        addComment(cout, asts[0], "when t0=0, v0=0, i=0, j=0, and k=0");

        // C++ code generator.
        // The visitor is accepted at all nodes in the AST;
        // for each node in the AST, code is generated and
        // stored in the expression-string in the visitor.
        PrintVisitorBottomUp pcv(cout, *vp, exprSize);
        asts[0]->accept(&pcv);

        // Result.
        string result = pcv.getExprStr();
        cout << endl << " // Set final result." << endl <<
            "context." << u.name() << "->writeElem(" << result <<
            ", t0 + TIME_STEPS, v0, i, j, k);" << endl;

        // End of function.
        cout << "}" << endl;

        delete vp;
    } // C++ scalar code.

    // Print a function to evaluate the stencil with the given fold.
    // For this, each AST will need to accept 2 different visitors.
    if (printVecCpp) {

        cout << "// Automatically generated code; do not edit." << endl;

        // Create vectors needed to implement.
        // The visitor is accepted at all nodes in the AST;
        // for each grid access node in the AST, the vectors
        // needed are determined and saved in the visitor.
        VecInfoVisitor vv(vlen);
        for (auto ast : asts)
            ast->accept(&vv);

        // C++ vector print assistant.
        CppVecPrintHelper* vp = NULL;
        if (print512Cpp)
            vp = new Avx512CppVecPrintHelper(vv);
        else if (printKncCpp)
            vp = new KncCppVecPrintHelper(vv);
        else
            vp = new CppVecPrintHelper(vv); // w/o intrinsics.
        vp->setVarType("realv");
            
        // Stencil-calculation code.
        // Function header.
        cout << endl << "// Calculate " << (vlen.getLen() * clen.getLen()) <<
            " results for order " << order << ", " <<
            vv.getNumPoints() << "-point " << shapeName <<
            " stencil at time t0 + TIME_STEPS,"
            " work variable v0, and vector location veci, vecj, veck." << endl;
        cout << "// Use " << vlen.getName3D() << " vector-folds in a " <<
            clen.getName3D() << " vector-fold cluster." << endl;
        cout << "// SIMD calculations use " << vv.getNumPoints() <<
            " vector blocks created from " << vv.getNumAlignedVecs() <<
            " aligned vector-blocks." << endl;
        cout << "ALWAYS_INLINE void calc_stencil_vector(StencilContext& context, "
            "int t0, int v0, int veci, int vecj, int veck) {" << endl;
        addComment(cout, asts[0],
                   "for the first scalar element when t0=0, v0=0, veci=0, vecj=0, and veck=0");

        // Loop through all ASTs.
        for (int k = 0; k < clen.getZLen(); k++) {
            int kp = k * vlen.getZLen();
            for (int j = 0; j < clen.getYLen(); j++) {
                int jp = j * vlen.getYLen();
                for (int i = 0; i < clen.getXLen(); i++) {
                    int ip = i * vlen.getXLen();

                    // Index into AST list.
                    int astIdx = clen.map321(i, j, k);
                    cout << endl << " // Calculate vector number " <<
                        (astIdx + 1) << " at offset " << 
                        ip << ", " << jp << ", " << kp << "." << endl;

                    // Code generator visitor.
                    // The visitor is accepted at all nodes in the AST;
                    // for each node in the AST, code is generated and
                    // stored in the expression-string in the visitor.
                    PrintVisitorBottomUp pcv(cout, *vp, exprSize);
                    asts[astIdx]->accept(&pcv);

                    // Result.
                    string result = pcv.getExprStr();
                    cout << endl << " // Set final result at offset " << 
                        ip << ", " << jp << ", " << kp << "." << endl <<
                        "context." << u.name() << "->writeVec(" << result <<
                        ", t0 + TIME_STEPS, v0, veci + " << i <<
                        ", vecj + " << j << ", veck + " << k <<
                        ", __LINE__);" << endl;
                }
            }
        }

        // End of function.
        cout << "}" << endl;

        // Prefetch code.
        cout << "// ==== Prefetch functions ====" << endl;
        
        // Generate code for no specific direction and then each orthogonal direction.
        for (int diri = 0; diri <= 3; diri++) {

            // Create a direction object.
            shared_ptr<Dir> dir;
            if (!diri)
                dir = make_shared<NoDir>();
            else {
                cout << endl;
                switch (diri) {
                case 1:
                    dir = make_shared<XDir>();
                    break;
                case 2:
                    dir = make_shared<YDir>();
                    break;
                case 3:
                    dir = make_shared<ZDir>();
                    break;
                }
                cout << endl;
            }

            // Upper-case direction.
            string ucDirStr = dir->getBaseName();
            transform(ucDirStr.begin(), ucDirStr.end(), ucDirStr.begin(), ::toupper);

            // Prefetch code for each cache level.
            for (int l = 1; l <= 2; l++) {
                string hint = (l == 1) ? "L1" : "L2";
            
                // Function header.
                cout << endl << "// Prefetch vector blocks ";
                if (!dir->isNone())
                    cout << "for leading edge in " << dir->getName() << " direction";
                cout << " to " << hint << " cache." << endl;
                cout << "// For stencil at vector location veci, vecj, veck." << endl;
                cout << "// For " << vv.getName3D() << " vector-fold." << endl;
                cout << "ALWAYS_INLINE void prefetch_" << hint << "_stencil_vector";
                if (!dir->isNone())
                    cout << "_" << dir->getBaseName();
                cout << "(StencilContext& context, "
                    "int t0, int v0, int veci, int vecj, int veck) {" << endl;

                // C++ prefetch code.
                vp->printPrefetches(cout, *dir, hint);

                // End of function.
                cout << "}" << endl;

            } // cache level.
        } // direction.

        delete vp;
    } // C++ vector code.
        

    delete stencilFunc;
    return 0;
}
