/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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

// YASK compiler APIs.
#include "yask_compiler_api.hpp"

// Backward-compatible API for undocumented YASK v2 DSL.
#include "aux/Soln.hpp"

// YASK compiler-solution code.
// TODO: remove these non-API includes.
#include "Solution.hpp"
#include "Parse.hpp"

using namespace yask;

// Declarations to live in the 'yask' namespace.
namespace yask {

    // Compiler API factory.
    yc_factory factory;

    // output streams.
    string outfile;
    
    // other vars set via cmd-line options.
    string solution_name;
    int radius = -1;
    int vlen_for_stats = 0;
    
    // Dummy object for backward-compatibility with old stencil DSL.
    StencilList stub_stencils;

} // yask namespace.

void usage(const string& cmd,
           CompilerSettings& settings) {

    cout << "Options:\n"
        " -h\n"
        "     Print this help message.\n"
        "\n"
        " -stencil <name>\n"
        "     Set stencil solution (required)\n";
    for (bool show_test : { false, true }) {
        if (show_test)
            cout << "       Built-in test solutions:\n";
        else
            cout << "       Built-in example solutions:\n";
        for (auto si : yc_solution_base::get_registry()) {
            auto& name = si.first;
            if ((name.rfind("test_", 0) == 0) == show_test) {
                auto* sp = si.second;
                cout << "           " << name;

                // Add asterisk for solns with a radius.
                auto* srp = dynamic_cast<yc_solution_with_radius_base*>(sp);
                if (srp)
                    cout << " *";
                cout << endl;
            }
        }
    }
    cout <<
        " -radius <radius>\n"
        "     Set radius for stencils marked with '*' above (default is stencil-specific).\n"
        "\n"
        " -target <format>\n"
        "    Set the output format (required).\n"
        "    Supported formats:\n"
        "      avx         YASK code for CORE AVX ISA (256-bit HW SIMD vectors).\n"
        "      avx2        YASK code for CORE AVX2 ISA (256-bit HW SIMD vectors).\n"
        "      avx512      YASK code classes for CORE AVX-512 ISA (512-bit HW SIMD vectors).\n"
        "      avx512-ymm  YASK code for CORE AVX-512 ISA (256-bit HW SIMD vectors).\n"
        "      knl         YASK code for Knights-Landing (MIC) AVX-512 ISA (512-bit HW SIMD vectors).\n"
        "      intel64     YASK code for generic C++ with 64-bit indices (no explicit HW SIMD vectors).\n"
        "      pseudo      Human-readable scalar pseudo-code.\n"
        "      pseudo-long Human-readable scalar pseudo-code with intermediate variables.\n"
        "      dot         DOT-language description.\n"
        "      dot-lite    DOT-language description of var accesses only.\n"
        //"      pov-ray    POV-Ray code.\n"
        " -elem-bytes <n>\n"
        "    Set number of bytes in each FP element (default=" << settings._elem_bytes << ").\n"
        "      Currently, only 4 (single-precision) and 8 (double) are allowed.\n"
        " -fold <dim>=<size>,...\n"
        "    Set number of elements in each given dimension in a vector block.\n"
        "    Default depends on -elem-bytes setting, domain-dimension order, and print format (below).\n"
        "    If product of fold lengths does not equal SIMD vector length for print\n"
        "      formats with explicit lengths, lengths will adjusted as needed.\n"
        " -cluster <dim>=<size>,...\n"
        "    Set number of vectors to evaluate per inner-loop iteration in each dimension.\n"
        " -p <filename>\n"
        "    Write formatted output to <filename>.\n"
        //" -ps <vec-len>         Print stats for all folding options for given vector length.\n"
        "\n"
        "Advanced options for experimentation or debug:\n"
        " -step-dim <dim>\n"
        "    Explicitly set the name of the step dimension, e.g., 't'.\n"
        "    By default, the step dimension is defined implicitly when YASK variables are encountered\n"
        "      in the stencil DSL code.\n"
        " -domain-dims <dim>,<dim>,...\n"
        "    Explicitly name the domain dimensions and set their order, e.g., 'x,y,z'.\n"
        "    In addition, domain dimensions are added implicitly when YASK variables are encountered\n"
        "      in the stencil DSL code.\n"
        "    The domain-dimension order determines array memory layout, default loop nesting, and\n"
        "      MPI rank layout. Thus, this option can be used to override those traits compared to\n"
        "      what would be obtained from the DSL code only.\n"
        " -inner-loop-dim <dim>\n"
        "    Specify the domain dimension used for the inner-most stencil-computation loop.\n"
        "      The default is the last domain dimension specified via -domain_dims or in the\n"
        "        stencil DSL code.\n"
        "      For this option, a numerical index is allowed: '1' is the first domain-dim, etc.\n"
        " -min-buffer-len <n>\n"
        "    Create buffers in the inner loop if at least <n> points could be stored in it\n"
        "      (default=" << settings._min_buffer_len << ").\n"
        " [-no]-reloc-step\n"
        "    Define YASK vars with the 'step' dim just after the first domain dim\n"
        "      instead of just before the first domain dim (default=" << settings._inner_step << ").\n"
        " [-no]-interleave-misc\n"
        "    Define YASK vars with the 'misc' dim(s) as the inner-most dim(s) (default=" << settings._inner_misc << ").\n"
        "      This effectively creates an AoSoA-style layout instead of an SoAoA one.\n"
        "      This disallows dynamcally changing the 'misc' dim sizes during run-time.\n"
        " -[no]-fus\n"
        "    Make first dimension of fold unit stride (default=" << settings._first_inner << ").\n"
        "      This controls the intra-vector memory layout.\n"
        "      The order of dimensions in a folded vector is not necessarily the same as the order\n"
        "        of the dimensions in the array memory layouts as determined by -step_dim, -domain_dims,\n"
        "        and the stencil DSL code.\n"
        " -l1-prefetch-dist <n>\n"
        "    Set L1 prefetch distance to <n> iterations ahead. Use zero (0) to disable.\n"
        " -l2-prefetch-dist <n>\n"
        "    Set L2 prefetch distance to <n> iterations ahead. Use zero (0) to disable.\n"
        " -vars <regex>\n"
        "    Only process updates to vars whose names match <regex>.\n"
        "      This can be used to generate code for a subset of the stencil equations.\n"
        " -eq-bundles <name>=<regex>,...\n"
        "    Put updates to vars matching <regex> in equation-bundle with base-name <name>.\n"
        "      By default, eq-bundles are created as needed based on dependencies between equations:\n"
        "        equations that do not depend on each other are bundled together into bundles with the\n"
        "        base-name '" << settings._eq_bundle_basename_default << "'.\n"
        "      Each eq-bundle base-name is appended with a unique index number, so the default bundle\n"
        "        names are '" << settings._eq_bundle_basename_default << "_0', " <<
        settings._eq_bundle_basename_default << "_1', etc.\n"
        "      This option allows more control over this bundling.\n"
        "      Example: \"-eq-bundles a=foo,b=b[ae]r\" creates one or more eq-bundles named 'a_0', 'a_1', etc.\n"
        "        containing updates to each var whose name contains 'foo' and one or more eq-bundles\n"
        "        named 'b_0', 'b_1', etc. containing updates to each var whose name contains 'bar' or 'ber'.\n"
        "      Standard regex-format tokens in <name> will be replaced based on matches to <regex>.\n"
        "      Example: \"-eq-bundles 'g_$&=b[ae]r'\" with vars 'bar_x', 'bar_y', 'ber_x', and 'ber_y'\n"
        "        would create eq-bundle 'g_bar_0' for vars 'bar_x' and 'bar_y' and eq-bundle 'g_ber_0' for\n"
        "        vars 'ber_x' and 'ber_y' because '$&' is substituted by the string that matches the regex.\n"
        " [-no]-bundle-scratch\n"
        "    Bundle scratch equations even if the sizes of their scratch vars must be increased\n"
        "      to do so (default=" << settings._bundle_scratch << ").\n"
        " -halo <size>\n"
        "    Specify the size of the halos on all vars.\n"
        "      By default, halos are calculated automatically for each var.\n"
        " -step-alloc <size>\n"
        "    Specify the size of the step-dimension memory allocation on all vars.\n"
        "      By default, allocations are calculated automatically for each var.\n"
        " [-no]-ul\n"
        "    [Advanced] Do [not] generate simple unaligned loads (default=" << settings._allow_unaligned_loads << ").\n"
        "      To use this correctly, only 1D folds are allowed, and\n"
        "        the array memory layout must have that same dimension in unit stride.\n"
        " [-no]-opt-comb\n"
        "    Do [not] combine commutative operations (default=" << settings._do_comb << ").\n"
        " [-no]-opt-reorder\n"
        "    Do [not] reorder commutative operations (default=" << settings._do_reorder << ").\n"
        " [-no]-opt-cse\n"
        "    Do [not] eliminate common subexpressions (default=" << settings._do_cse << ").\n"
        " [-no]-opt-pair\n"
        "    Do [not] pair eligible function calls (default=" << settings._do_pairs << ").\n"
        "      Currently enables 'sin(x)' and 'cos(x)' to be replaced with 'sincos(x)'.\n"
        " [-no]-opt-cluster\n"
        "    Do [not] apply optimizations across the cluster (default=" << settings._do_opt_cluster << ").\n"
        " -max-es <num-nodes>\n"
        "    Set heuristic for max single expression-size (default=" << settings._max_expr_size << ").\n"
        " -min-es <num-nodes>\n"
        "    Set heuristic for min expression-size for reuse (default=" << settings._min_expr_size << ").\n"
        " [-no]-use-ptrs\n"
        "    Enable var access via pointers & strides (default=" << settings._use_ptrs << ").\n"
        " [-no]-find-deps\n"
        "    Find dependencies between stencil equations (default=" << settings._find_deps << ").\n"
        " [-no]-print-eqs\n"
        "    Print each equation when defined (default=" << settings._print_eqs << ").\n"
        "\n"
        "Examples:\n"
        " " << cmd << " -stencil 3axis -radius 2 -fold x=4,y=4 -target pseudo -p -  # '-' for stdout\n"
        " " << cmd << " -stencil awp -elem-bytes 8 -fold x=4,y=2 -target avx2 -p stencil_code.hpp\n"
        " " << cmd << " -stencil iso3dfd -radius 4 -cluster y=2 -target avx512 -p stencil_code.hpp\n";
    exit(1);
}

// Parse command-line and set global cmd-line option vars.
// Exits on error.
void parse_opts(int argc, const char* argv[],
               CompilerSettings& settings)
{
    if (argc <= 1)
        usage(argv[0], settings);

    int argi;               // current arg index.
    for (argi = 1; argi < argc; argi++) {
        if ( argv[argi][0] == '-' && argv[argi][1] ) {
            string opt = argv[argi];

            // options w/o values.
            if (opt == "-h" || opt == "-help" || opt == "--help")
                usage(argv[0], settings);

            else if (opt == "-fus")
                settings._first_inner = true;
            else if (opt == "-no-fus")
                settings._first_inner = false;
            else if (opt == "-ul")
                settings._allow_unaligned_loads = true;
            else if (opt == "-no-ul")
                settings._allow_unaligned_loads = false;
            else if (opt == "-opt-comb")
                settings._do_comb = true;
            else if (opt == "-no-opt-comb")
                settings._do_comb = false;
            else if (opt == "-opt-reorder")
                settings._do_reorder = true;
            else if (opt == "-no-opt-reorder")
                settings._do_reorder = false;
            else if (opt == "-opt-cse")
                settings._do_cse = true;
            else if (opt == "-no-opt-cse")
                settings._do_cse = false;
            else if (opt == "-opt-pair")
                settings._do_pairs = true;
            else if (opt == "-no-opt-pair")
                settings._do_pairs = false;
            else if (opt == "-opt-cluster")
                settings._do_opt_cluster = true;
            else if (opt == "-no-opt-cluster")
                settings._do_opt_cluster = false;
            else if (opt == "-find-deps")
                settings._find_deps = true;
            else if (opt == "-no-find-deps")
                settings._find_deps = false;
            else if (opt == "-bundle-scratch")
                settings._bundle_scratch = true;
            else if (opt == "-no-bundle-scratch")
                settings._bundle_scratch = false;
            else if (opt == "-print-eqs")
                settings._print_eqs = true;
            else if (opt == "-no-print-eqs")
                settings._print_eqs = false;
            else if (opt == "-interleave-misc")
                settings._inner_misc = true;
            else if (opt == "-no-interleave-misc")
                settings._inner_misc = false;
            else if (opt == "-reloc-step")
                settings._inner_step = true;
            else if (opt == "-no-reloc-step")
                settings._inner_step = false;
            else if (opt == "-use-ptrs")
                settings._use_ptrs = true;
            else if (opt == "-no-use-ptrs")
                settings._use_ptrs = false;
            
            // add any more options w/o values above.

            // options w/a value.
            else {

                // at least one value needed.
                if (argi + 1 >= argc) {
                    cerr << "Error: value missing or bad option '" << opt << "'." << endl;
                    usage(argv[0], settings);
                }
                string argop = argv[++argi];

                // options w/a string value.
                if (opt == "-stencil")
                    solution_name = argop;
                else if (opt == "-target")
                    settings._target = argop;
                else if (opt == "-p")
                    outfile = argop;
                else if (opt == "-vars")
                    settings._var_regex = argop;
                else if (opt == "-eq-bundles")
                    settings._eq_bundle_targets = argop;
                else if (opt == "-step-dim")
                    settings._step_dim = argop;
                else if (opt == "-inner-loop-dim")
                    settings._inner_loop_dim = argop;
                else if (opt == "-domain-dims") {
                    settings._domain_dims.clear();
                    
                    // example: y,z
                    ArgParser ap;
                    ap.parse_list
                        (argop,
                         [&](const string& dname) {
                             settings._domain_dims.push_back(dname);
                         });
                }
                else if (opt == "-fold" || opt == "-cluster") {

                    // example: x=4,y=2
                    ArgParser ap;
                    ap.parse_key_value_pairs
                        (argop,
                         [&](const string& key, const string& value) {
                             int size = atoi(value.c_str());

                             // set dim in tuple.
                             if (opt == "-fold")
                                 settings._fold_options.add_dim_back(key, size);
                             else
                                 settings._cluster_options.add_dim_back(key, size);
                         });
                }

                // add any more options w/a string value above.

                else {

                    // options w/an int value.
                    int val = atoi(argop.c_str());

                    if (opt == "-l1-prefetch-dist")
                        settings._prefetch_dists[1] = val;
                    else if (opt == "-l2-prefetch-dist")
                        settings._prefetch_dists[2] = val;
                    else if (opt == "-max-es")
                        settings._max_expr_size = val;
                    else if (opt == "-min-es")
                        settings._min_expr_size = val;
                    else if (opt == "-radius")
                        radius = val;
                    else if (opt == "-elem-bytes")
                        settings._elem_bytes = val;
                    else if (opt == "-ps")
                        vlen_for_stats = val;
                    else if (opt == "-halo")
                        settings._halo_size = val;
                    else if (opt == "-step-alloc")
                        settings._step_alloc = val;
                    else if (opt == "-min-buffer-len")
                        settings._min_buffer_len = val;

                    // add any more options w/int values here.

                    else {
                        cerr << "Error: option '" << opt << "' not recognized." << endl;
                        usage(argv[0], settings);
                    }
                }
            }
        }
        else break;
    }
    if (argi < argc) {
        cerr << "Error: unrecognized parameter '" << argv[argi] << "'." << endl;
        usage(argv[0], settings);
    }
    if (solution_name.length() == 0) {
        cerr << "Error: stencil solution not specified; use -stencil." << endl;
        usage(argv[0], settings);
    }
    if (settings._target.length() == 0) {
        cerr << "Error: target not specified; use -target." << endl;
        usage(argv[0], settings);
    }
}

// Main program.
int main(int argc, const char* argv[]) {

    cout << "YASK -- Yet Another Stencil Kit\n"
        "YASK Stencil Compiler Utility\n"
        "Copyright (c) 2014-2021, Intel Corporation.\n"
        "Version: " << yask_get_version_string() << endl;

    try {
        // Parse options.
        CompilerSettings settings;
        parse_opts(argc, argv, settings);

        // Find the requested stencil in the registry.
        auto& stencils = yc_solution_base::get_registry();
        auto stencil_iter = stencils.find(solution_name);
        if (stencil_iter == stencils.end()) {
            cerr << "Error: unknown stencil solution '" << solution_name << "'." << endl;
            usage(argv[0], settings);
        }
        auto* stencil_soln = stencil_iter->second;
        assert(stencil_soln);
        auto soln = stencil_soln->get_soln();
        cout << "Stencil-solution name: " << soln->get_name() << endl;

        // Set radius if applicable.
        auto* srp = dynamic_cast<yc_solution_with_radius_base*>(stencil_soln);
        if (srp) {
            if (radius >= 0) {
                bool r_ok = srp->set_radius(radius);
                if (!r_ok) {
                    cerr << "Error: invalid radius=" << radius << " for stencil type '" <<
                        solution_name << "'." << endl;
                    usage(argv[0], settings);
                }
            }
            cout << "Stencil radius: " << srp->get_radius() << endl;
        }
        cout << "Stencil-solution description: " << soln->get_description() << endl;

        // Make sure that target is legal.
        soln->set_target(settings._target);
        cout << "Output target: " << soln->get_target() << endl;

        // Copy cmd-line settings into solution.
        // TODO: remove this reliance on internal (non-API) functionality.
        auto sp = dynamic_pointer_cast<StencilSolution>(soln);
        assert(sp);
        sp->set_settings(settings);

        // Create equations and change settings from the overloaded 'define()' methods.
        stencil_soln->define();

        // Apply the cmd-line settings again to override the defaults
        // set in 'define()'
        parse_opts(argc, argv, sp->get_settings());

        // A bit more info.
        cout << "Num vars defined: " << soln->get_num_vars() << endl;
        cout << "Num equations defined: " << soln->get_num_equations() << endl;
        
        // Create the requested output.
        if (outfile.length() == 0)
            cout << "Use the '-p' option to generate output from this stencil.\n";
        else {
            yask_output_factory ofac;
            yask_output_ptr os;
            if (outfile == "-")
                os = ofac.new_stdout_output();
            else
                os = ofac.new_file_output(outfile);
            stencil_soln->get_soln()->output_solution(os);
        }
    } catch (yask_exception& e) {
        cerr << "YASK Stencil Compiler: " << e.get_message() << ".\n";
        exit(1);
    }

    cout << "YASK Stencil Compiler: done.\n";
    return 0;
}
