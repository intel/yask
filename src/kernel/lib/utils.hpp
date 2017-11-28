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

#ifndef YASK_UTILS
#define YASK_UTILS

namespace yask {

    // Fatal error.
    // TODO: enable exception throwing that works w/SWIG.
    inline void exit_yask(int code) {
#ifdef USE_MPI
        int flag;
        MPI_Initialized(&flag);
        if (flag)
            MPI_Abort(MPI_COMM_WORLD, code);
        else
            exit(code);
#else
        exit(code);
#endif
    }

    // Return num with SI multiplier and "iB" suffix,
    // e.g., 41.2KiB.
    extern std::string makeByteStr(size_t nbytes);

    // Return num with SI multiplier, e.g., 4.23M.
    extern std::string makeNumStr(double num);

    // Find sum of rank_vals over all ranks.
    extern idx_t sumOverRanks(idx_t rank_val, MPI_Comm comm);

    // Make sure rank_val is same over all ranks.
    extern void assertEqualityOverRanks(idx_t rank_val, MPI_Comm comm,
                                        const std::string& descr);
    
    // Round up val to a multiple of mult.
    // Print a message if rounding is done and do_print is set.
    extern idx_t roundUp(std::ostream& os, idx_t val, idx_t mult,
                         const std::string& name, bool do_print);

    // Helpers for shared and aligned malloc and free.
    // Use like this:
    // shared_ptr<char> p(alignedAlloc(nbytes), AlignedDeleter());
    extern char* alignedAlloc(std::size_t nbytes);
    struct AlignedDeleter {
        void operator()(char* p) {
            std::free(p);
        }
    };

    // A class for maintaining elapsed time.
    class YaskTimer {

        /* struct timespec {
           time_t   tv_sec;        // seconds
           long     tv_nsec;       // nanoseconds
           };
        */
        struct timespec _begin, _end, _elapsed;

    public:
        YaskTimer() { clear(); }
        virtual ~YaskTimer() { }

        // Reset elapsed time to zero.
        virtual void clear() {
            _begin.tv_sec = _end.tv_sec = _elapsed.tv_sec = 0;
            _begin.tv_nsec = _end.tv_nsec = _elapsed.tv_nsec = 0;
        }

        // Start a timed region.
        // start() and stop() can be called multiple times in
        // pairs before calling get_elapsed_secs(), which
        // will return the cumulative time over all timed regions.
        virtual void start() {
            clock_gettime(CLOCK_REALTIME, &_begin);
        }

        // End a timed region.
        virtual void stop() {
            clock_gettime(CLOCK_REALTIME, &_end);

            // Elapsed time is just end - begin times.
            _elapsed.tv_sec += _end.tv_sec - _begin.tv_sec;

            // No need to check for sign or to normalize, because tv_nsec is
            // signed and 64-bit.
            _elapsed.tv_nsec += _end.tv_nsec - _begin.tv_nsec;
        }

        // Get elapsed time in sec.
        // Does not reset value, so it may be used for cumulative time.
        virtual double get_elapsed_secs() const {
            return double(_elapsed.tv_sec) + double(_elapsed.tv_nsec) * 1e-9;
        }
    };
    
    // A class to parse command-line args.
    class CommandLineParser {

    public:

        // Base class for an allowed option.
        class OptionBase {
        protected:
            std::string _name;
            std::string _help;
            std::string _help_leader;
            std::string _current_value_str;

            // Internal function to print help.
            virtual void _print_help(std::ostream& os,
                                     const std::string& str,
                                     int width) const;

            // Check for matching option to str at args[argi].
            // Return true and increment argi if match.
            virtual bool _check_arg(std::vector<std::string>& args, int& argi,
                                    const std::string& str) const;

            // Get one idx_t value from args[argi++].
            // Exit on failure.
            virtual idx_t _idx_val(std::vector<std::string>& args, int& argi);
            
        public:
            OptionBase(const std::string& name,
                       const std::string& help_msg) :
                _name(name), _help(help_msg),
                _help_leader("    "),
                _current_value_str("Current value = ")
            { }
            virtual ~OptionBase() { }

            // Accessors.
            virtual const std::string& get_name() const {
                return _name;
            }
            virtual const std::string& get_help() const {
                return _help;
            }

            // Print help on this option.
            virtual void print_help(std::ostream& os,
                                    int width) const {
                _print_help(os, _name, width);
            }

            // Check for matching option and any needed args at args[argi].
            // Return true, set val, and increment argi if match.
            virtual bool check_arg(std::vector<std::string>& args, int& argi) =0;
        };

        // An allowed boolean option.
        class BoolOption : public OptionBase {
            bool& _val;
            
        public:
            BoolOption(const std::string& name,
                       const std::string& help_msg,
                       bool& val) :
                OptionBase(name, help_msg), _val(val) { }

            virtual void print_help(std::ostream& os,
                                    int width) const;
            virtual bool check_arg(std::vector<std::string>& args, int& argi);
        };

        // An allowed int option.
        class IntOption : public OptionBase {
            int& _val;
            
        public:
            IntOption(const std::string& name,
                      const std::string& help_msg,
                      int& val) :
                OptionBase(name, help_msg), _val(val) { }

            virtual void print_help(std::ostream& os,
                                    int width) const;
            virtual bool check_arg(std::vector<std::string>& args, int& argi);
        };

        // An allowed idx_t option.
        class IdxOption : public OptionBase {
            idx_t& _val;
            
        public:
            IdxOption(const std::string& name,
                       const std::string& help_msg,
                       idx_t& val) :
                OptionBase(name, help_msg), _val(val) { }

            virtual void print_help(std::ostream& os,
                                    int width) const;
            virtual bool check_arg(std::vector<std::string>& args, int& argi);
        };

        // An allowed idx_t option that sets multiple vars.
        class MultiIdxOption : public OptionBase {
            std::vector<idx_t*> _vals;
            
        public:
            MultiIdxOption(const std::string& name,
                           const std::string& help_msg,
                           std::vector<idx_t*> vals) :
                OptionBase(name, help_msg), _vals(vals) {
                _current_value_str = "Current values = ";
            }

            virtual void print_help(std::ostream& os,
                                    int width) const;
            virtual bool check_arg(std::vector<std::string>& args, int& argi);
        };

    protected:
        std::map<std::string, OptionBase*> _opts;
        int _width;

    public:

        // Ctor.
        CommandLineParser() : _width(78) { }

        // Dtor.
        ~CommandLineParser() {

            // Delete options.
            for (auto i : _opts) {
                delete i.second;
            }
        }

        // Set help width.
        virtual void set_width(int width) {
            _width = width;
        }
        
        // Add an allowed option.
        // Options will be deleted upon destruction.
        virtual void add_option(OptionBase* opt) {
            _opts[opt->get_name()] = opt;
        }

        // Print help info on all options.
        virtual void print_help(std::ostream& os) const;
        
        // Parse options from the command-line and set corresponding vars.
        // Recognized strings from args are consumed, and unused ones
        // remain for further processing by the application.
        virtual void parse_args(const std::string& pgmName,
                                std::vector<std::string>& args);

        // Same as above, but pgmName is populated from argv[0]
        // and args is appended from remainder of argv array.
        // Unused strings are returned in args vector.
        virtual void parse_args(int argc, char** argv,
                                std::vector<std::string>& args) {
            std::string pgmName = argv[0];
            for (int i = 1; i < argc; i++)
                args.push_back(argv[i]);
            parse_args(pgmName, args);
        }

        // Tokenize args from a string.
        virtual void set_args(std::string arg_string,
                              std::vector<std::string>& args);
    };
}

#endif
