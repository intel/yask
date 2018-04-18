/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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
#include "Grid.hpp"
#include "CppIntrin.hpp"
#include "Parse.hpp"

using namespace yask;

// A register of stencils.
StencilList stencils;

// Stencils.
#include "stencils.hpp"

// output streams.
map<string, string> outfiles;

// other vars set via cmd-line options.
CompilerSettings settings;
int vlenForStats = 0;
StencilBase* stencilSoln = NULL;
string solutionName;
int radius = 1;

void usage(const string& cmd) {

    cout << "Options:\n"
        " -h\n"
        "     Print this help message.\n"
        "\n"
        " -stencil <name>\n"
        "     Set stencil solution (required); supported solutions:\n";
    for (auto si : stencils) {
        auto name = si.first;
        auto sp = si.second;
        cout << "                     " << name;
        if (sp->usesRadius())
            cout << " *";
        cout << endl;
    }
    cout <<
        " -radius <radius>\n"
        "     Set radius for stencils marked with '*' above (default=" << radius << ").\n"
        "\n"
        " -elem-bytes <n>"
        "    Set number of bytes in each FP element (default=" << settings._elem_bytes << ").\n"
        " -fold <dim>=<size>,...\n"
        "    Set number of elements in each given dimension in a vector block.\n"
        "    Default depends on -elem-bytes setting and print format (below).\n"
        "    If product of fold lengths does not equal SIMD vector length for print\n"
        "      formats with explicit lengths, lengths will adjusted as needed.\n"
        " -cluster <dim>=<size>,...\n"
        "    Set number of vectors to evaluate in each dimension.\n"
        " -grids <regex>\n"
        "    Only process updates to grids whose names match <regex>.\n"
        "      This can be used to generate code for a subset of the stencil equations.\n"
        " -eq-bundles <name>=<regex>,...\n"
        "    Put updates to grids matching <regex> in equation-bundle with base-name <name>.\n"
        "      By default, eq-bundles are created as needed based on dependencies between equations:\n"
        "        equations that do not depend on each other are bundled together into bundles with the\n"
        "        base-name '" << settings._eq_bundle_basename_default << "'.\n"
        "      Each eq-bundle base-name is appended with a unique index number, so the default bundle\n"
        "        names are '" << settings._eq_bundle_basename_default << "_0', " <<
        settings._eq_bundle_basename_default << "_1', etc.\n"
        "      This option allows more control over this bundling.\n"
        "      Example: \"-eq-bundles a=foo,b=b[aeiou]r\" creates one or more eq-bundles named 'a_0', 'a_1', etc.\n"
        "        containing updates to each grid whose name contains 'foo' and one or more eq-bundles\n"
        "        named 'b_0', 'b_1', etc. containing updates to each grid whose name matches 'b[aeiou]r'.\n"
        "      Standard regex-format tokens in <name> will be replaced based on matches to <regex>.\n"
        "      Example: \"-eq-bundles 'g_$&=b[aeiou]r'\" with grids 'bar_x', 'bar_y', 'ber_x', and 'ber_y'\n"
        "        would create eq-bundle 'g_bar_0' for grids 'bar_x' and 'bar_y' and eq-bundle 'g_ber_0' for\n"
        "        grids 'ber_x' and 'ber_y' because '$&' is substituted by the string that matches the regex.\n"
        " -step-alloc <size>\n"
        "    Specify the size of the step-dimension memory allocation.\n"
        "      By default, allocations are calculated automatically for each grid.\n"
        " -halo <size>\n"
        "    Specify the sizes of the halos.\n"
        "      By default, halos are calculated automatically for each grid.\n"
        " -fus\n"
        "    Make first dimension of fold unit stride (default=" << settings._firstInner << ").\n"
        "      This controls the intra-vector memory layout.\n"
        " -lus\n"
        "    Make last dimension of fold unit stride (default=" << (!settings._firstInner) << ").\n"
        "      This controls the intra-vector memory layout.\n"
        " [-no]-ul\n"
        "    Do [not] generate simple unaligned loads (default=" <<
        settings._allowUnalignedLoads << ").\n"
        "      [Advanced] To use this correctly, only 1D folds are allowed, and\n"
        "        the memory layout used by YASK must have that same dimension in unit stride.\n"
        " [-no]-opt-comb\n"
        "    Do [not] combine commutative operations (default=" << settings._doComb << ").\n"
        " [-no]-opt-cse\n"
        "    Do [not] eliminate common subexpressions (default=" << settings._doCse << ").\n"
        " [-no]-opt-cluster\n"
        "    Do [not] apply optimizations across the cluster (default=" << settings._doOptCluster << ").\n"
        " -max-es <num-nodes>\n"
        "    Set heuristic for max single expression-size (default=" <<
        settings._maxExprSize << ").\n"
        " -min-es <num-nodes>\n"
        "    Set heuristic for min expression-size for reuse (default=" <<
        settings._minExprSize << ").\n"
        "\n"
        " -p <format-type> <filename>\n"
        "    Format output per <format-type> and write to <filename>.\n"
        "    Supported format-types:\n"
        "      cpp       YASK stencil classes for generic C++ (no explicit HW SIMD vectors).\n"
        "      avx       YASK stencil classes for CORE AVX ISA (256-bit HW SIMD vectors).\n"
        "      avx2      YASK stencil classes for CORE AVX2 ISA (256-bit HW SIMD vectors).\n"
        "      avx512    YASK stencil classes for CORE AVX-512 & MIC AVX-512 ISAs (512-bit HW SIMD vectors).\n"
        "      knc       YASK stencil classes for KNC ISA (512-bit HW SIMD vectors).\n"
        "      pseudo    Human-readable scalar pseudo-code for one point.\n"
        "      dot       DOT-language description.\n"
        "      dot-lite  DOT-language description of grid accesses only.\n"
        //"      pov-ray    POV-Ray code.\n"
        //" -ps <vec-len>         Print stats for all folding options for given vector length.\n"
        "\n"
        "Examples:\n"
        " " << cmd << " -stencil 3axis -radius 2 -fold x=4,y=4 -p pseudo -  # '-' for stdout\n"
        " " << cmd << " -stencil awp -elem-bytes 8 -fold x=4,y=2 -p avx2 stencil_code.hpp\n"
        " " << cmd << " -stencil iso3dfd -radius 8 -cluster y=2 -p avx512 stencil_code.hpp\n";
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

            else if (opt == "-fus")
                settings._firstInner = true;
            else if (opt == "-lus")
                settings._firstInner = false;
            else if (opt == "-ul")
                settings._allowUnalignedLoads = true;
            else if (opt == "-no-ul")
                settings._allowUnalignedLoads = false;
            else if (opt == "-opt-comb")
                settings._doComb = true;
            else if (opt == "-no-opt-comb")
                settings._doComb = false;
            else if (opt == "-opt-cse")
                settings._doCse = true;
            else if (opt == "-no-opt-cse")
                settings._doCse = false;
            else if (opt == "-opt-cluster")
                settings._doOptCluster = true;
            else if (opt == "-no-opt-cluster")
                settings._doOptCluster = false;

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
                if (opt == "-stencil")
                    solutionName = argop;
                else if (opt == "-grids")
                    settings._gridRegex = argop;
                else if (opt == "-eq-bundles")
                    settings._eqBundleTargets = argop;
                else if (opt == "-fold" || opt == "-cluster") {

                    // example: x=4,y=2
                    ArgParser ap;
                    ap.parseKeyValuePairs
                        (argop, [&](const string& key, const string& value) {
                            int size = atoi(value.c_str());
                            
                            // set dim in tuple.
                            if (opt == "-fold")
                                settings._foldOptions.addDimBack(key, size);
                            else
                                settings._clusterOptions.addDimBack(key, size);
                        });
                }

                // Print options w/format & filename args.
                else if (opt == "-p") {

                    // another arg needed.
                    if (argi + 1 >= argc) {
                        cerr << "Error: filename missing after '" << opt <<
                            " " << argop << "'." << endl;
                        usage(argv[0]);
                    }
                    string argop2 = argv[++argi];
                    outfiles[argop] = argop2;
                }
            
                // add any more options w/a string value above.
                
                else {

                    // options w/an int value.
                    int val = atoi(argop.c_str());

                    if (opt == "-max-es")
                        settings._maxExprSize = val;
                    else if (opt == "-min-es")
                        settings._minExprSize = val;

                    else if (opt == "-radius")
                        radius = val;
                    else if (opt == "-elem-bytes")
                        settings._elem_bytes = val;

                    else if (opt == "-ps")
                        vlenForStats = val;

                    else if (opt == "-halo")
                        settings._haloSize = val;
                    else if (opt == "-step-alloc")
                        settings._stepAlloc = val;

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
    if (solutionName.length() == 0) {
        cerr << "Error: solution not specified." << endl;
        usage(argv[0]);
    }

    // Find the stencil in the registry.
    auto stencilIter = stencils.find(solutionName);
    if (stencilIter == stencils.end()) {
        cerr << "Error: unknown stencil solution '" << solutionName << "'." << endl;
        usage(argv[0]);
    }
    stencilSoln = stencilIter->second;
    assert(stencilSoln);
    
    cout << "Stencil-solution name: " << solutionName << endl;
    if (stencilSoln->usesRadius()) {
        bool rOk = stencilSoln->setRadius(radius);
        if (!rOk) {
            cerr << "Error: invalid radius=" << radius << " for stencil type '" <<
                solutionName << "'." << endl;
            usage(argv[0]);
        }
        cout << "Stencil radius: " << radius << endl;
    }

    // Copy cmd-line settings into solution.
    stencilSoln->getSettings() = settings;
}

// Main program.
int main(int argc, const char* argv[]) {

    cout << "YASK -- Yet Another Stencil Kernel\n"
        "YASK Stencil Compiler\n"
        "Copyright (c) 2014-2018, Intel Corporation.\n"
        "Version: " << yask_get_version_string() << endl;
    
    try {
		// Parse options and create the stencil-solution object.
		parseOpts(argc, argv);

		// Create the requested output...
		for (auto i : outfiles) {
			auto& type = i.first;
			auto& fname = i.second;

			yask_output_factory ofac;
			yask_output_ptr os;
			if (fname == "-")
				os = ofac.new_stdout_output();
			else
				os = ofac.new_file_output(fname);
			stencilSoln->format(type, os);
		}
    } catch (yask_exception e) {
        cerr << "YASK Stencil Compiler: " << e.get_message() << ".\n";
        exit(1);
    }

    cout << "YASK Stencil Compiler: done.\n";
    return 0;
}
