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
bool printCpp = false;
bool printKncCpp = false;
bool print512Cpp = false;
int order = -1;
int vlenForStats = 0;
int timeSteps = 1;
StencilBase* stencilFunc = NULL;
string shapeName;
string stencilCtor, stencilHeader;
Triple vlen;
Triple clen(1,1,1);
int exprSize = 50;
Grid3d* velGrid = NULL;

void usage() {
    printf("options:\n"
           " -h                print this help message.\n"
           "\n"
           " -st <iso3dfd|3axis|9axis|3plane|cube|ave>   set stencil type (required).\n"
           " -or <order>       set stencil order (required).\n"
           " -ts <time-steps>  set number of time-steps to calc per call (default=1).\n"
           " -cluster <x> <y> <z>  set number of vectors used in each dimension (default=1 1 1).\n"
           " -es <expr-size>   set heuristic for expression-size threshold (default=%d).\n", exprSize);
    printf("\n"
           " -ps <vec-len>     print stats for all folding options for vec-len.\n"
           " -ph               print human-readable scalar pseudo-code.\n"
           " -pp               print POV-Ray code.\n"
           " -pm <x> <y> <z>   print pre-processor macros for x*y*z vector block.\n"
           " -pcpp <x> <y> <z> print C++ stencil function for x*y*z vector block.\n"
           " -pknc <x> <y> <z> print KNC C++ stencil function for x*y*z vector block.\n"
           " -p512 <x> <y> <z> print CORE/MIC AVX-512 C++ stencil function for x*y*z vector block.\n");
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
            
            // add any more options w/o values here.

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

                    // iso-3dfd stencil.
                    if (shapeName == "iso3dfd") {
                        velGrid = new Grid3d("vel");
                        stencilFunc = new Iso3dfdStencil(*velGrid);
                        stencilCtor = "Iso3dfdStencil(*velGrid, STENCIL_ORDER)";
                        stencilHeader = "Iso3dfdStencil.hpp";
                    }

                    // average-value cube stencil.
                    else if (shapeName == "ave") {
                        stencilFunc = new AveStencil();
                        stencilCtor = "AveStencil(STENCIL_ORDER)";
                        stencilHeader = "AveStencil.hpp";
                    }

                    // other symmetric stencils.
                    else {
                        if (shapeName == "3axis")
                            stencilFunc = new ExampleStencil(ExampleStencil::SS_3AXIS);
                        else if (shapeName == "9axis")
                            stencilFunc = new ExampleStencil(ExampleStencil::SS_9AXIS);
                        else if (shapeName == "3plane")
                            stencilFunc = new ExampleStencil(ExampleStencil::SS_3PLANE);
                        else if (shapeName == "cube")
                            stencilFunc = new ExampleStencil(ExampleStencil::SS_CUBE);
                        else {
                            cerr << "error: unknown stencil shape '" << shapeName << "'." << endl;
                            usage();
                        }
                        string shapeStr = "SS_" + shapeName;
                        transform(shapeStr.begin(), shapeStr.end(), shapeStr.begin(), ::toupper);
                        stencilCtor = "ExampleStencil(ExampleStencil::" +
                            shapeStr + ", STENCIL_ORDER)";
                        stencilHeader = "ExampleStencil.hpp";
                    }
                    continue;
                }

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
                        else {
                            cerr << "error: option '" << opt << "' not recognized." << endl;
                            usage();
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
    if (!stencilFunc) {
        cerr << "error: shape not specified." << endl;
        usage();
    }

    cerr << "Number of time steps: " << timeSteps << endl;
    cerr << "Stencil type: " << shapeName << endl;
    cerr << "Stencil order: " << order << endl;
    cerr << "Expression-size threshold: " << exprSize << endl;

    // Finish configuring the stencil.
    bool orderOk = stencilFunc->setOrder(order);
      
    if (!orderOk) {
        cerr << "error: invalid order=" << order << " for stencil type '" <<
            shapeName << "'." << endl;
        usage();
    }
}

int main(int argc, const char* argv[]) {

    // parse options.
    parseOpts(argc, argv);

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
        cout << "// ==== Functions for direction '" << dir->getName() << "' ====" << endl;

        // Set direction.
        // Stencil calculation in this direction enabled?
        bool dirOk = stencilFunc->setDir(dir);

        // Only allow pipe if unit-stride in this direction.
        if (!dir->isNone() && vlen.getVal(*dir) > 1)
            dirOk = false;

        // Upper-case direction.
        string ucDirStr = dir->getBaseName();
        transform(ucDirStr.begin(), ucDirStr.end(), ucDirStr.begin(), ::toupper);

        // Construct AST at the 0,0,0 corner of each cluster.
        // TODO: handle stencils that behave differently
        // at various locations, e.g., boundaries.
        Grid5d u("grid");
        typedef vector<GridValue> GridValueList;
        GridValueList asts;
        for (int k = 0; k < clen.getZLen(); k++) {
            int kp = k * vlen.getZLen();
            for (int j = 0; j < clen.getYLen(); j++) {
                int jp = j * vlen.getYLen();
                for (int i = 0; i < clen.getXLen(); i++) {
                    int ip = i * vlen.getXLen();

                    // ip,jp,kp is the 0,0,0 corner of the vector cluster i,j,k.
                    GridValue ast = stencilFunc->value(u, timeSteps, 0, 0, ip, jp, kp);
                    assert(ast);
                    asts.push_back(ast);

                    // map321() converts 3D indices to 1D index.
                    assert(asts.size() == clen.map321(i, j, k) + 1); 
                }
            }
        }

        // Print stats for various folding options.
        if (vlenForStats && dirOk) {
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
        if (printPseudo && dirOk) {
            cout << "// Top-down stencil calculation:" << endl;
            PrintHelper ph;
            PrintVisitorTopDown pv1(cout, ph);
            asts[0]->accept(&pv1);
            cout << "u(1, 0, 0, 0, 0) = " << pv1.getExprStr() << ";" << endl;

            cout << endl << "// Bottom-up stencil calculation:" << endl;
            PrintVisitorBottomUp pv2(cout, ph, exprSize);
            asts[0]->accept(&pv2);
            cout << "u(1, 0, 0, 0, 0) = " << pv2.getExprStr() << ";" << endl;
        }

        // POV-Ray output.
        if (printPOVRay && dirOk) {

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

            // direction non-specific.
            if (dir->isNone()) {
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
                cout << "#define STENCIL_REF(velGrid) " << stencilCtor << endl;
                cout << "#define STENCIL_HEADER \"" << stencilHeader << "\"" << endl;
                if (velGrid)
                    cout << "#define STENCIL_USES_VEL" << endl;
            }

            // direction-specific.
            else {
                if (dirOk)
                    cout << "#define " << ucDirStr << "_PIPE_ENABLED" << endl;
                else
                    cout << "// Pipeline code in " << dir->getName() <<
                        " direction not enabled for this stencil." << endl;
            }
        }

        // Print a function to evaluate the stencil with the given fold.
        // For this, each AST will need to accept 2 different visitors.
        if (printCpp) {

            if (dir->isNone())
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
            if (dirOk) {

                // Function header.
                cout << endl << "// Calculate " << (vlen.getLen() * clen.getLen()) <<
                    " results for " << order << "-order, " << vv.getNumPoints() << "-point ";
                if (!dir->isNone())
                    cout << dir->getName() << "-direction pipeline leading-edge of ";
                cout << "stencil." << endl;
                cout << "// Find values at vector location veci, vecj, veck." << endl;
                cout << "// Use " << vlen.getName3D() << " vector-folds in a " <<
                    clen.getName3D() << " vector-fold cluster." << endl;
                cout << "// SIMD calculations use " << vv.getNumPoints() <<
                    " vector blocks created from " << vv.getNumAlignedVecs() <<
                    " aligned vector-blocks." << endl;
                cout << "ALWAYS_INLINE void calc_stencil_vector";
                if (!dir->isNone())
                    cout << "_" << dir->getBaseName() << "_pipe";
                cout << "(StencilContext& context, "
                    "int t0, int v0, long veci, long vecj, long veck) {" << endl;

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

                            // C++ vector code generator.
                            // The visitor is accepted at all nodes in the AST;
                            // for each node in the AST, code is generated and
                            // stored in the expression-string in the visitor.
                            CppVecPrintVisitor pcv(cout, *vp, exprSize);
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
            }

            else if (!dir->isNone())
                cout << "// Pipeline code in " << dir->getName() <<
                    " direction not enabled." << endl;

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
                    "int t0, int v0, long veci, long vecj, long veck) {" << endl;

                // C++ prefetch code.
                vp->printPrefetches(cout, *dir, hint);

                // End of function.
                cout << "}" << endl;

            } // Prefetch
            delete vp;
        }
        
    } // pipe directions.

    delete stencilFunc;
    if (velGrid)
        delete velGrid;
    return 0;
}
