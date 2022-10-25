/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2022, Intel Corporation

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

#pragma once

// Misc utilities.

namespace yask {

    // Fatal error.
    [[noreturn]] inline
    void exit_yask(int code) {

#ifdef USE_MPI
        int flag;
        MPI_Initialized(&flag);
        if (flag) {
            int num_ranks = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
            if (num_ranks > 1)
                MPI_Abort(MPI_COMM_WORLD, code);
        }
#endif
        exit(code);
    }

    // Get an int from an env var.
    inline int get_env_int(const std::string& name, int def) {
        int res = def;
        char* s = getenv(name.c_str());
        if (s)
            res = atoi(s);
        return res;
    }

    // Find sum of rank_vals over all ranks.
    extern idx_t sum_over_ranks(idx_t rank_val, MPI_Comm comm);

    // Make sure rank_val is same over all ranks.
    extern void assert_equality_over_ranks(idx_t rank_val, MPI_Comm comm,
                                           const std::string& descr);

    // A class for a simple producer-consumer memory lock on one item.
    class SimpleLock {

        // Put each value in a separate cache-line to
        // avoid false sharing.
        union LockVal {
            struct {
                volatile idx_t chk; // check for mem corruption.
                volatile idx_t val; // actual counter.
            };
            char pad[CACHELINE_BYTES];
        };

        LockVal _write_count, _read_count;
        LockVal _data; // Optional simple data field.

        static constexpr idx_t _ival = 1000;

#ifdef CHECK
        inline void _check(const std::string& fn) const {
            idx_t wcnt = _write_count.val;
            idx_t rcnt = _read_count.val;
            idx_t wchk = _write_count.chk;
            idx_t rchk = _read_count.chk;
            if (wcnt < _ival || rcnt < _ival ||
                wcnt < rcnt || wcnt - rcnt > 1 ||
                wchk != _ival || rchk != _ival)
                FORMAT_AND_THROW_YASK_EXCEPTION
                     ("Internal error: " << fn << "() w/lock @ " << (void*)this <<
                      " writes=" << wcnt << ", reads=" << rcnt <<
                      ", w-chk=" << wchk << ", r-chk=" << rchk);
        }
#else
        inline void _check(const char* fn) const { }
#endif

    public:
        SimpleLock() {
            init();
        }

        // Allow write and block read.
        void init() {
            _write_count.val = _read_count.val = _ival;
            _write_count.chk = _read_count.chk = _ival;
            _check("init");
        }

        // Check whether ok to read,
        // i.e., whether write is done.
        bool is_ok_to_read() const {
            _check("is_ok_to_read");
            return _write_count.val != _read_count.val;
        }

        // Wait until ok to read.
        void wait_for_ok_to_read() const {
            while (!is_ok_to_read())
                _mm_pause();
        }

        // Mark that read is done.
        void mark_read_done() {
            assert(is_ok_to_read());
            _read_count.val++;
            _check("mark_read_done");
        }

        // Check whether ok to write,
        // i.e., whether read is done for previous write.
        bool is_ok_to_write() const {
            _check("is_ok_to_write");
            return _write_count.val == _read_count.val;
        }

        // Wait until ok to write.
        void wait_for_ok_to_write() const {
            while (!is_ok_to_write())
                _mm_pause();
        }

        // Mark that write is done.
        void mark_write_done() {
            assert(is_ok_to_write());
            _write_count.val++;
            _check("mark_write_done");
        }

        // Access data value.
        // Of course, other data can be gated w/this lock.
        idx_t get_data() const {
            return _data.val;
        }
        void set_data(idx_t v) {
            _data.val = v;
        }
    };

    // A class for maintaining elapsed time.
    // NOT a virtual class.
    // Example:
    //   time --->
    //     start() ... stop() ... start() ... stop() ... get_elapsed_time()
    //     |   A secs  |          |   B secs  |
    // 1st call to stop() returns A.
    // 2nd call to stop() returns B.
    // Call to get_elapsed_time() returns A + B.
    class YaskTimer {

        /* struct timespec {
           time_t   tv_sec;        // seconds
           long     tv_nsec;       // nanoseconds
           };
        */
        struct timespec _begin, _elapsed;

    public:

        typedef struct timespec TimeSpec;

        YaskTimer() { clear(); }
        ~YaskTimer() { }

        // Reset elapsed time to zero.
        void clear() {
            _begin.tv_sec = _elapsed.tv_sec = 0;
            _begin.tv_nsec = _elapsed.tv_nsec = 0;
        }

        // Make a current timespec to be provided to start() or stop().
        // This allows multiple timers to use the same timespec.
        static TimeSpec get_timespec() {
            TimeSpec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            return ts;
        }

        // Start a timed region.
        // start() and stop() can be called multiple times in
        // pairs before calling get_elapsed_secs(), which
        // will return the cumulative time over all timed regions.
        void start(const TimeSpec& ts);
        void start() {
            auto ts = get_timespec();
            start(ts);
        }

        // End a timed region.
        // Return time since previous call to start(); this is *not*
        // generally the same as the value returned by get_elapsed_secs().
        double stop(const TimeSpec& ts);
        double stop() {
            auto ts = get_timespec();
            return stop(ts);
        }

        // Get elapsed time between all preceding start/stop pairs since
        // object creation or previous call to clear().  Does not reset
        // value, so it may be used for querying cumulative time.
        double get_elapsed_secs() const {

            // Make sure timer was stopped.
            assert(_begin.tv_sec == 0);

            return double(_elapsed.tv_sec) + double(_elapsed.tv_nsec) * 1e-9;
        }

        // Get elapsed time since previous start.
        // Used to check time w/o stopping timer.
        double get_secs_since_start() const;
    };

    // A class to parse command-line args.
    class CommandLineParser {

    public:

        // Base class for a command-line option.
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
            virtual bool _is_opt(const string_vec& args, int& argi,
                                 const std::string& str) const;

            // Get one double value from args[argi++].
            // Exit on failure.
            virtual double _double_val(const string_vec& args, int& argi);

            // Get one idx_t value from args[argi++].
            // Exit on failure.
            virtual idx_t _idx_val(const string_vec& args, int& argi);

            // Get one string value from args[argi++].
            // Exit on failure.
            virtual std::string _string_val(const string_vec& args, int& argi);

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

            // Print current value of this option.
            virtual std::ostream& print_value(std::ostream& os) const =0;

            // Check for matching option and any needed args at args[argi].
            // Return true, set val, and increment argi if match.
            virtual bool check_arg(const string_vec& args, int& argi) =0;
        };
        typedef std::shared_ptr<OptionBase> OptionPtr;

        // A boolean option.
        class BoolOption : public OptionBase {
            bool& _val;

        public:
            BoolOption(const std::string& name,
                       const std::string& help_msg,
                       bool& val) :
                OptionBase(name, help_msg), _val(val) { }

            virtual void print_help(std::ostream& os,
                                    int width) const override;
            virtual std::ostream& print_value(std::ostream& os) const override {
                os << (_val ? "true" : "false");
                return os;
            }
            virtual bool check_arg(const string_vec& args, int& argi) override;
        };

        // An int option.
        class IntOption : public OptionBase {
            int& _val;

        public:
            IntOption(const std::string& name,
                      const std::string& help_msg,
                      int& val) :
                OptionBase(name, help_msg), _val(val) { }

            virtual void print_help(std::ostream& os,
                                    int width) const override;
            virtual std::ostream& print_value(std::ostream& os) const override {
                os << _val;
                return os;
            }
            virtual bool check_arg(const string_vec& args, int& argi) override;
        };

        // A double option.
        class DoubleOption : public OptionBase {
            double& _val;

        public:
            DoubleOption(const std::string& name,
                      const std::string& help_msg,
                      double& val) :
                OptionBase(name, help_msg), _val(val) { }

            virtual void print_help(std::ostream& os,
                                    int width) const override;
            virtual std::ostream& print_value(std::ostream& os) const override {
                os << _val;
                return os;
            }
            virtual bool check_arg(const string_vec& args, int& argi) override;
        };

        // An idx_t option.
        class IdxOption : public OptionBase {
            idx_t& _val;

        public:
            IdxOption(const std::string& name,
                       const std::string& help_msg,
                       idx_t& val) :
                OptionBase(name, help_msg), _val(val) { }

            virtual void print_help(std::ostream& os,
                                    int width) const override;
            virtual std::ostream& print_value(std::ostream& os) const override {
                os << _val;
                return os;
            }
            virtual bool check_arg(const string_vec& args, int& argi) override;
        };

        // An idx_t option that sets multiple vars.
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
                                    int width) const override;
            virtual std::ostream& print_value(std::ostream& os) const override {
                for (size_t i = 0; i < _vals.size(); i++) {
                    if (i > 0)
                        os << ", ";
                    os << *_vals[i];
                }
                return os;
            }
            virtual bool check_arg(const string_vec& args,
                                   int& argi) override;
        };

        // A string option.
        class StringOption : public OptionBase {
            std::string& _val;

        public:
            StringOption(const std::string& name,
                         const std::string& help_msg,
                         std::string& val) :
                OptionBase(name, help_msg), _val(val) { }

            virtual void print_help(std::ostream& os,
                                    int width) const override;
            virtual std::ostream& print_value(std::ostream& os) const override {
                os << _val;
                return os;
            }
            virtual bool check_arg(const string_vec& args, int& argi) override;
        };

        // A list-of-strings option.
        class StringListOption : public OptionBase {
            std::set<std::string> _allowed_strs; // empty to allow any strings.
            string_vec& _val;

        public:
            StringListOption(const std::string& name,
                             const std::string& help_msg,
                             std::set<std::string> allowed_strs,
                             string_vec& val) :
                OptionBase(name, help_msg),
                _allowed_strs(allowed_strs), _val(val) { }

            virtual void print_help(std::ostream& os,
                                    int width) const override;
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
            virtual bool check_arg(const string_vec& args, int& argi) override;
        };

    protected:
        std::map<std::string, OptionPtr> _opts;
        int _width = 78;

    public:

        // Ctor.
        CommandLineParser() { }

        // Dtor.
        virtual ~CommandLineParser() { }

        // Tokenize args from a string.
        static string_vec set_args(const std::string& arg_string);

        // Set help width.
        virtual void set_width(int width) {
            _width = width;
        }

        // Add an allowed option.
        virtual void add_option(OptionPtr opt) {
            _opts[opt->get_name()] = opt;
        }

        // Print help info on all options.
        virtual void print_help(std::ostream& os) const;

        // Print current settings of all options.
        virtual void print_values(std::ostream& os) const;

        // Parse options from 'args' and set corresponding vars.
        // Recognized strings from args are consumed, and unused ones
        // remain for further processing by the application.
        virtual std::string parse_args(const std::string& pgm_name,
                                       const string_vec& args);

        // Same as above, but splits 'arg_string' into tokens.
        virtual std::string parse_args(const std::string& pgm_name,
                                       const std::string& arg_string) {
            auto args = set_args(arg_string);
            return parse_args(pgm_name, args);
        }

        // Same as above, but pgm_name is populated from argv[0]
        // and rest of argv is parsed.
        virtual std::string parse_args(int argc, char** argv) {
            std::string pgm_name = argv[0];
            string_vec args;
            for (int i = 1; i < argc; i++)
                args.push_back(argv[i]);
            return parse_args(pgm_name, args);
        }
    };
}

