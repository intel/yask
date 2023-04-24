/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

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

///////// APIs common to the YASK compiler and kernel. ////////////

// This file uses Doxygen markup for API documentation-generation.
// See https://www.doxygen.nl/manual/index.html.
/** @file yask_common_api.hpp */

#pragma once

#include <cstdint>
#include <cinttypes>
#include <climits>
#include <type_traits>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <ostream>
#include <sstream>
#include <memory>
#include <functional>

// Things SWIG can't handle.
#ifdef SWIG
#ifndef YASK_DEPRECATED
#define YASK_DEPRECATED
#endif
#define YASK_INT64_T long int
#else
/// Deprecated attribute.
#ifndef YASK_DEPRECATED
#define YASK_DEPRECATED [[deprecated]]
#endif
/// Signed 64-bit int.
#define YASK_INT64_T std::int64_t
#endif

namespace yask {

    /**
     * \defgroup yask YASK Common
     * Types, clases, and functions used in both the \ref sec_yc and \ref sec_yk.
     * @{
     */

    /// Version information.
    /**
       @returns String describing the current version.
    */
    std::string yask_get_version_string();

    /// Type to use for indexing grids.
    /** Index types are signed to allow negative indices in padding/halos. */
    typedef YASK_INT64_T idx_t;

    /// Vector of indices.
    typedef std::vector<idx_t> idx_t_vec;

    /// Initializer list of indices.
    /**
       @note This type is not available in the Python API.
       Use `idx_t_vec` instead.
    */
    typedef std::initializer_list<idx_t> idx_t_init_list;

    /// Vector of strings.
    typedef std::vector<std::string> string_vec;

    // Forward declarations of class-pointers.

    class yask_output;
    /// Shared pointer to \ref yask_output
    typedef std::shared_ptr<yask_output> yask_output_ptr;

    class yask_file_output;
    /// Shared pointer to \ref yask_file_output
    typedef std::shared_ptr<yask_file_output> yask_file_output_ptr;

    class yask_string_output;
    /// Shared pointer to \ref yask_string_output
    typedef std::shared_ptr<yask_string_output> yask_string_output_ptr;

    class yask_stdout_output;
    /// Shared pointer to \ref yask_stdout_output
    typedef std::shared_ptr<yask_stdout_output> yask_stdout_output_ptr;

    class yask_null_output;
    /// Shared pointer to \ref yask_null_output
    typedef std::shared_ptr<yask_null_output> yask_null_output_ptr;

    /// Exception from YASK framework
    /** Objects of this exception contain additional message from yask framework */
    class yask_exception: public std::exception {
    private:
    	/// Description of exception.
    	std::string _msg;

    public:

        /// Construct a YASK exception with default message.
    	yask_exception() :
            _msg("YASK error") { };

        /// Construct a YASK exception with `message`.
    	yask_exception(const std::string& message) :
            _msg(message) { };

    	virtual ~yask_exception() { };

        /// Get description.
        /** Returns a C-style character string describing the cause of the current error.
            @returns description of the exception. */
    	virtual const char* what() const noexcept;

    	/// Append `message` to description of this exception.
    	virtual void add_message(const std::string& message
                                 /**< [in] Additional message as string. */ );

        /// Get description.
        /** Same as what().
            @returns description of the exception. */
    	virtual const char* get_message() const;
    };

    #ifndef SWIG
    
    /// Macro for creating and throwing a yask_exception with a string.
    /**
       Example: THROW_YASK_EXCEPTION("all your base are belong to us");
       @note Not available in the Python API.
    */
    #define THROW_YASK_EXCEPTION(message) do {                      \
            auto msg = std::string("YASK error: ") + message;       \
            yask_exception e(msg);                                  \
            throw e;                                                \
        } while(0)
    
    /// Macro for creating and throwing a yask_exception using stream operators.
    /**
       Example: FORMAT_AND_THROW_YASK_EXCEPTION("bad value: x = " << x);
       @note Not available in the Python API.
    */
    #define FORMAT_AND_THROW_YASK_EXCEPTION(message) do {           \
            std::stringstream err;                                  \
            err << message;                                         \
            THROW_YASK_EXCEPTION(err.str());                        \
        } while(0)

    #endif
    
    /// Factory to create output objects.
    class yask_output_factory {
    public:
        virtual ~yask_output_factory() {}

        /// Create a file output object.
        /**
           This object is used to write output to a file.
           @returns Pointer to new output object or null pointer if
           file cannot be opened.
        */
        virtual yask_file_output_ptr
        new_file_output(const std::string& file_name
                        /**< [in] Name of file to open.
                           Any existing file will be truncated. */ ) const;

        /// Create a string output object.
        /**
           This object is used to write output to a string.
           @returns Pointer to new output object.
        */
        virtual yask_string_output_ptr
        new_string_output() const;

        /// Create a stdout output object.
        /**
           This object is used to write output to the standard output stream.
           @returns Pointer to new output object.
        */
        virtual yask_stdout_output_ptr
        new_stdout_output() const;

        /// Create a null output object.
        /**
           This object is used to discard output.
           @returns Pointer to new output object.
        */
        virtual yask_null_output_ptr
        new_null_output() const;
    };

    /// Base interface for output.
    class yask_output {
    public:
        virtual ~yask_output() {}

        /// Access underlying C++ ostream object.
        /** @returns Reference to ostream. */
        virtual std::ostream& get_ostream() =0;
    };

    /// File output.
    class yask_file_output : public virtual yask_output {
    public:
        virtual ~yask_file_output() {}

        /// Get the filename.
        /** @returns String containing filename given during creation. */
        virtual std::string get_filename() const =0;

        /// Close file.
        virtual void close() =0;
    };

    /// String output.
    class yask_string_output : public virtual yask_output {
    public:
        virtual ~yask_string_output() {}

        /// Get the output.
        /** Does not modify current buffer.
            @returns copy of current buffer's contents. */
        virtual std::string get_string() const =0;

        /// Discard contents of current buffer.
        virtual void discard() =0;
    };

    /// Stdout output.
    class yask_stdout_output : public virtual yask_output {
    public:
        virtual ~yask_stdout_output() {}
    };

    /// Null output.
    /** This object will discard all output. */
    class yask_null_output : public virtual yask_output {
    public:
        virtual ~yask_null_output() {}
    };

    /// Create finite-difference (FD) coefficients for the standard center form.
    /**
       Find FD coefficients with `radius` sample points to both the left and right
       of the center sample and evaluation point on a uniformly-spaced grid. 
       The FD has `radius * 2`-order accuracy.
       @returns `radius * 2 + 1` FD coefficients.
    */
    std::vector<double>
    get_center_fd_coefficients(int derivative_order
                               /**< [in] `1` for 1st derivative, `2` for 2nd, etc. */,
                               int radius
                               /**< [in] Number of points to either side of the center point. */ );

    /// Create finite-difference (FD) coefficients for the standard forward form.
    /**
       Find FD coefficients with `accuracy_order` sample points to the right
       of the center sample and evaluation point on a uniformly-spaced grid. 
       @returns `accuracy_order + 1` FD coefficients.
    */
    std::vector<double>
    get_forward_fd_coefficients(int derivative_order
                                /**< [in] `1` for 1st derivative, `2` for 2nd, etc. */,
                                int accuracy_order
                                /**< [in] Number of points to the right of the center point. */ );
    
    /// Create finite-difference (FD) coefficients for the standard backward form.
    /**
       Find FD coefficients with `accuracy_order` sample points to the left
       of the center sample and evaluation point on a uniformly-spaced grid. 
       @returns `accuracy_order + 1` FD coefficients.
    */
    std::vector<double>
    get_backward_fd_coefficients(int derivative_order
                                 /**< [in] `1` for 1st derivative, `2` for 2nd, etc. */,
                                 int accuracy_order
                                 /**< [in] Number of points to the left of the center point. */ );
    
    /// Create finite-difference (FD) coefficients at arbitrary evaluation and sample points.
    /**
       @returns `sample_points` FD coefficients.
    */
    std::vector<double>
    get_arbitrary_fd_coefficients(int derivative_order
                                  /**< [in] `1` for 1st derivative, `2` for 2nd, etc. */,
                                  double eval_point
                                  /**< [in] Location of evaluation point. */,
                                  const std::vector<double> sample_points
                                  /**< [in] Locations of sampled points. Must have at least 2. */ );

    #ifndef SWIG
    
    /// A class to parse command-line arguments.
    /**
       This is the class used to parse command-line arguments for the YASK kernel
       and compiler libraries.
       It is provided as a convenience for API programmers who want
       to parse application options in a consistent manner.

       @note Not available in the Python API.
    */
    class command_line_parser {

    public:

        /// Base class for a command-line option.
        /**
           The API programmer can extend this class to add new option types.
        */
        class option_base {

        private:
            std::string _name;
            std::string _help;
            std::string _help_leader;
            std::string _current_value_str;

        protected:

            /// Format and print help for option named `display_name` to `os`.
            virtual void _print_help(std::ostream& os,
                                     const std::string& display_name,
                                     int width) const;

            /// Check for matching option to `str` at `args[argi]`.
            /**
               @returns `true` and increments argi if match,
               `false` if not a match.
            */
            virtual bool _is_opt(const string_vec& args, int& argi,
                                 const std::string& str) const;

            /// Get one double value from `args[argi++]`.
            /**
               @returns the value at `args[argi]` and increments `argi`.

               @throws yask_exception if `args[argi]` is not a double.
            */
            virtual double _double_val(const string_vec& args, int& argi);

            /// Get one idx_t value from args[argi++].
            /**
               @returns the value at `args[argi]` and increments `argi`.

               @throws yask_exception if `args[argi]` is not an integer.
            */
            virtual idx_t _idx_val(const string_vec& args, int& argi);

            /// Get one string value from args[argi++].
            /**
               @returns the value at `args[argi]` and increments `argi`.

               @throws yask_exception if `args[argi]` does not exist.
            */
            virtual std::string _string_val(const string_vec& args, int& argi);

        public:
            /// Constructor.
            option_base(const std::string& name,
                        const std::string& help_msg,
                        const std::string& current_value_prefix = std::string("Current value = "),
                        const std::string& help_line_prefix = std::string("    ")) :
                _name(name), _help(help_msg),
                _help_leader(help_line_prefix),
                _current_value_str(current_value_prefix) 
            { }
            virtual ~option_base() { }

            /// Get the current option name.
            virtual const std::string& get_name() const {
                return _name;
            }

            /// Get the unformatted help string.
            virtual const std::string& get_help() const {
                return _help;
            }
            
            /// Print help on this option.
            virtual void print_help(std::ostream& os,
                                    int width) const {
                _print_help(os, _name, width);
            }

            /// Print current value of this option.
            virtual std::ostream& print_value(std::ostream& os) const =0;

            /// Check for matching option and any needed args at args[argi].
            /**
               @returns `true`, sets value of option, and increments `argi` if match;
               `false` if no match, and doesn't modify `argi`.
            */
            virtual bool check_arg(const string_vec& args, int& argi) =0;
        };

        /// Pointer to an option handler.
        typedef std::shared_ptr<option_base> option_ptr;

        /// A boolean option.
        class bool_option : public option_base {
            bool& _val;

        public:
            /// Constructor.
            bool_option(const std::string& name,
                       const std::string& help_msg,
                       bool& val) :
                option_base(name, help_msg), _val(val) { }

            /// Print help message for a boolean option.
            virtual void print_help(std::ostream& os,
                                    int width) const override;

            /// Print current value of the boolean.
            virtual std::ostream& print_value(std::ostream& os) const override {
                os << (_val ? "true" : "false");
                return os;
            }

            /// Check for a boolean option (set or unset variants).
            virtual bool check_arg(const string_vec& args, int& argi) override;
        };

        /// An integer option.
        class int_option : public option_base {
            int& _val;

        public:
            /// Constructor.
            int_option(const std::string& name,
                      const std::string& help_msg,
                      int& val) :
                option_base(name, help_msg), _val(val) { }

            /// Print help message for an int option.
            virtual void print_help(std::ostream& os,
                                    int width) const override;

            /// Print the current value of the int.
            virtual std::ostream& print_value(std::ostream& os) const override {
                os << _val;
                return os;
            }

            /// Check for the option and its integer argument.
            virtual bool check_arg(const string_vec& args, int& argi) override;
        };

        /// A double option.
        class double_option : public option_base {
            double& _val;

        public:
            /// Constructor.
            double_option(const std::string& name,
                      const std::string& help_msg,
                      double& val) :
                option_base(name, help_msg), _val(val) { }

            /// Print help message for a double option.
            virtual void print_help(std::ostream& os,
                                    int width) const override;

            /// Print the current value of the double.
            virtual std::ostream& print_value(std::ostream& os) const override {
                os << _val;
                return os;
            }

            /// Check for the option and its double argument.
            virtual bool check_arg(const string_vec& args, int& argi) override;
        };

        /// An idx_t option.
        class idx_option : public option_base {
            idx_t& _val;

        public:
            /// Constructor.
            idx_option(const std::string& name,
                       const std::string& help_msg,
                       idx_t& val) :
                option_base(name, help_msg), _val(val) { }

            /// Print help message for an int_t option.
            virtual void print_help(std::ostream& os,
                                    int width) const override;

            /// Print the current value of the int_t.
            virtual std::ostream& print_value(std::ostream& os) const override {
                os << _val;
                return os;
            }

            /// Check for the option and its int_t argument.
            virtual bool check_arg(const string_vec& args, int& argi) override;
        };

        /// A string option.
        class string_option : public option_base {
            std::string& _val;

        public:
            /// Constructor.
            string_option(const std::string& name,
                         const std::string& help_msg,
                         std::string& val) :
                option_base(name, help_msg), _val(val) { }

            /// Print help message for a string option.
            virtual void print_help(std::ostream& os,
                                    int width) const override;

            /// Print the current value of the string.
            virtual std::ostream& print_value(std::ostream& os) const override {
                os << "'" << _val << "'";
                return os;
            }

            /// Check for the option and its string argument.
            virtual bool check_arg(const string_vec& args,
                                   int& argi) override;
        };

        /// A list-of-strings option.
        /**
           Strings are separated by commas (without spaces).
        */
        class string_list_option : public option_base {
            std::set<std::string> _allowed_strs; // empty to allow any strings.
            string_vec& _val;

        public:
            /// Constructor allowing any strings.
            string_list_option(const std::string& name,
                               const std::string& help_msg,
                               string_vec& val) :
                option_base(name, help_msg),
                _val(val) { }

            /// Constructor with set of allowed strings.
            string_list_option(const std::string& name,
                               const std::string& help_msg,
                               const std::set<std::string>& allowed_strs,
                               string_vec& val) :
                option_base(name, help_msg),
                _allowed_strs(allowed_strs),
                _val(val) { }

            /// Print help message for a list-of-strings option.
            virtual void print_help(std::ostream& os,
                                    int width) const override;

            /// Print the current value of the strings.
            virtual std::ostream& print_value(std::ostream& os) const override {
                int n = 0;
                for (auto& v : _val) {
                    if (n)
                        os << ",";
                    os << v;
                    n++;
                }
                return os;
            }

            /// Check for the option and its string-list argument.
            virtual bool check_arg(const string_vec& args, int& argi) override;
        };

    private:
        std::map<std::string, option_ptr> _opts;
        int _width = 78;

    public:

        /// Constructor.
        command_line_parser() { }

        /// Destructor.
        virtual ~command_line_parser() { }

        /// Convenience funcion to tokenize args from a string.
        static string_vec set_args(const std::string& arg_string);

        /// Set help width.
        virtual void set_width(int width) {
            _width = width;
        }

        /// Get help width.
        virtual int get_width() const {
            return _width;
        }

        /// Add an allowed option to the parser.
        virtual void add_option(option_ptr opt) {
            _opts[opt->get_name()] = opt;
        }

        /// Print help info on all options to `os`.
        virtual void print_help(std::ostream& os) const;

        /// Print current settings of all options to `os`.
        virtual void print_values(std::ostream& os) const;

        /// Parse options from 'args' and set corresponding vars.
        /**
           Recognized strings from args are consumed, and unused ones
           remain for further processing by the application.

           @returns string of unused args.
        */
        virtual std::string parse_args(const std::string& pgm_name,
                                       const string_vec& args);

        /// Same as parse_args(), but splits 'arg_string' into tokens.
        virtual std::string parse_args(const std::string& pgm_name,
                                       const std::string& arg_string) {
            auto args = set_args(arg_string);
            return parse_args(pgm_name, args);
        }

        /// Same as parse_args(), but pgm_name is populated from `argv[0]`
        /// and rest of `argv` is parsed.
        virtual std::string parse_args(int argc, char** argv) {
            std::string pgm_name = argv[0];
            string_vec args;
            for (int i = 1; i < argc; i++)
                args.push_back(argv[i]);
            return parse_args(pgm_name, args);
        }
    };

    /// Print a YASK spash message to `os`.
    /**
       Splash message contains the YASK copyright, URL, and version.
       If `argc > 1`, also prints `invocation_leader` followed by
       the program invocation string.

       @note Not available in the Python API.
    */
    extern void
    yask_print_splash(std::ostream& os, int argc, char** argv,
                      std::string invocation_leader = "invocation: ");
    #endif
    
    /** @}*/

} // namespace yask.

