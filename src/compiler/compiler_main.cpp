/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2024, Intel Corporation

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

// An application built around the YASK compiler API to allow users
// to easily generate YASK stencil code. This application is linked with
// stencil descriptions found in src/stencils/*.cpp.

// Standard includes.
#include <iostream>
#include <sstream>

// YASK compiler APIs.
#include "yask_compiler_api.hpp"

using namespace yask;
using namespace std;

// Add some command-line options for this application in addition to the
// default ones provided by YASK library.
struct MySettings {
    static constexpr int def_radius = -1;

    // Local options.
    bool help = false;          // help requested.
    int radius = def_radius;    // stencil radius.
    string output_filename;     // output code file.
    string solution_name;       // stencil name.

    command_line_parser parser;

    MySettings() {

        // Add options to parser.
        parser.add_option(make_shared<command_line_parser::bool_option>
                          ("help",
                           "Print help message.",
                           help));
        parser.add_option(make_shared<command_line_parser::bool_option>
                          ("h",
                           "Print help message.",
                           help));
        parser.add_option(make_shared<command_line_parser::string_option>
                          ("p",
                           "Write formatted output to file <string>. "
                           "If <string> is '-', write to standard output.",
                           output_filename));
        parser.add_option(make_shared<command_line_parser::int_option>
                          ("radius",
                           "Radius for stencils marked with '*' in the list below. "
                           "If value is negative, stencil-specific default is used.",
                           radius));
        parser.add_option(make_shared<command_line_parser::string_option>
                          ("stencil",
                           "YASK stencil solution from the list below (required).",
                           solution_name));
    }

    // Parse options from the command-line and set corresponding vars.
    // Exit with message on error or request for help.
    void parse(int argc, char** argv,
               yc_solution_ptr csoln,
               bool print_info = true) {
        string pgm_name(argv[0]);
        string values;

        // Parse 'args' and 'argv' cmd-line options, which sets values.
        // Any remaining strings will be returned.
        auto rem_args = parser.parse_args(argc, argv);

        // Parse standard args.
        auto rem_args2 = csoln->apply_command_line_options(rem_args);
        
        if (help) {
            cout << "Usage: " << pgm_name << " [options]\n"
                "Options from the '" << pgm_name << "' binary:\n";
            parser.print_help(cout);

            auto chelp = csoln->get_command_line_help();
            cout << "Options from the YASK compiler library:\n" <<
                chelp;

            cout << "\nAvailable solutions that can be specified with the '-stencil' option:\n"
                "  (Solutions that allow the '-radius' option are marked with '*'.)\n";
            for (bool show_test : { false, true }) {
                if (show_test)
                    cout << "Built-in test solutions:\n";
                else
                    cout << "Built-in example solutions:\n";
                for (auto si : yc_solution_base::get_registry()) {
                    auto& name = si.first;
                    if ((name.rfind("test_", 0) == 0) == show_test) {
                        auto* sp = si.second;
                        cout << "  " << name;
                            
                        // Add asterisk for solns with a radius.
                        auto* srp = dynamic_cast<yc_solution_with_radius_base*>(sp);
                        if (srp)
                            cout << " *";
                        cout << endl;
                    }
                }
            }

            cout <<
                "\nExamples:\n"
                " " << pgm_name << " -stencil 3axis -radius 1 -target pseudo -p -  # '-' for stdout\n"
                " " << pgm_name << " -stencil awp -elem-bytes 8 -fold x=2,y=2 -target avx2 -p stencil_code.hpp\n"
                " " << pgm_name << " -stencil iso3dfd -radius 4 -target avx512 -p stencil_code.hpp\n" <<
                flush;
            exit(1);
        }

        if (solution_name.length() == 0)
            THROW_YASK_EXCEPTION("stencil solution not specified; use '-stencil <string>' (or '-help' for list)");
        if (output_filename.length() && !csoln->is_target_set())
            THROW_YASK_EXCEPTION("target format not specified for output to '" +
                                 output_filename + "'; use '-target <string>' (or'-help' for list)");
        
        // Show settings.
        if (print_info) {
            cout << "Settings of options from the '" << pgm_name << "' binary:\n";
            parser.print_values(cout);
            auto cvals = csoln->get_command_line_values();
            cout << "Settings of options from the YASK compiler library:\n" <<
                cvals;
        }
       
        if (rem_args2.length())
            THROW_YASK_EXCEPTION("extraneous parameter(s): '" +
                                 rem_args2 +
                                 "'; run with '-help' option for usage");
    }
};                              // MySettings.

// Main program.
int main(int argc, char* argv[]) {

    // Compiler API factory.
    yc_factory factory;

    try {
        yask_print_splash(cout, argc, argv, "YASK Stencil Compiler invocation: ");

        // Option parser.
        MySettings my_settings;
        
        // First, use an empty solution to allow option parsing
        // before the requested solution is chosen.
        {
            auto null_soln = factory.new_solution("temp");
            my_settings.parse(argc, argv, null_soln, false);
        }
       
        // Find the requested stencil in the registry.
        auto& stencils = yc_solution_base::get_registry();
        auto stencil_iter = stencils.find(my_settings.solution_name);
        if (stencil_iter == stencils.end())
            THROW_YASK_EXCEPTION("unknown stencil solution '" + my_settings.solution_name + "'");
        auto* stencil_soln = stencil_iter->second;
        assert(stencil_soln);
        auto soln = stencil_soln->get_soln();
        cout << "Stencil-solution name: " << soln->get_name() << endl;

        // Set radius if applicable.
        auto* srp = dynamic_cast<yc_solution_with_radius_base*>(stencil_soln);
        if (srp) {
            if (my_settings.radius >= 0) {
                bool r_ok = srp->set_radius(my_settings.radius);
                if (!r_ok)
                    THROW_YASK_EXCEPTION(string("invalid radius = ") +
                                         to_string(my_settings.radius) + " for stencil type '" +
                                         soln->get_name() + "'.");
            }
            cout << "Stencil radius: " << srp->get_radius() << endl;
        }
        cout << "Stencil-solution description: " << soln->get_description() << endl;

        // Parse options again to set options in real soln.
        my_settings.parse(argc, argv, soln, false);

        // Create equations and change settings from the overloaded 'define()' methods.
        stencil_soln->define();

        // Apply the cmd-line settings once again to override the defaults
        // set via the call-back functions from 'define()'
        my_settings.parse(argc, argv, soln);

        // High-level info.
        cout << "\nInfo from the definition of '" << soln->get_name() << "':\n" <<
            "Num vars defined: " << soln->get_num_vars() << endl << 
            "Num equations defined: " << soln->get_num_equations() << endl;
        
        // Create the requested output, which will print more info.
        if (my_settings.output_filename.length() == 0)
            cout << "Use the '-p <string>' option to print generated code to file '<string>'.\n";
        else {
            yask_output_factory ofac;
            yask_output_ptr os;
            if (my_settings.output_filename == "-")
                os = ofac.new_stdout_output();
            else
                os = ofac.new_file_output(my_settings.output_filename);
            stencil_soln->get_soln()->output_solution(os);
        }
    } catch (yask_exception& e) {
        cerr << "YASK Stencil Compiler: " << e.get_message() << ".\n";
        exit(1);
    }

    cout << "YASK Stencil Compiler: done.\n";
    return 0;
}
