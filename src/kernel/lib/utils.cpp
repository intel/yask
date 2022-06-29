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

#include "yask.hpp"
using namespace std;

// Set MODEL_CACHE to 1 or 2 to model that cache level
// and create a global cache object here.
#ifdef MODEL_CACHE
Cache cache_model(MODEL_CACHE);
#endif

namespace yask {

    // Timer.
    void YaskTimer::start(const TimeSpec& ts) {

        // Make sure timer was stopped.
        assert(_begin.tv_sec == 0);
        assert(_begin.tv_nsec == 0);

        _begin = ts;
    }
    double YaskTimer::stop(const TimeSpec& ts) {

        // Make sure timer was started.
        assert(_begin.tv_sec != 0);

        TimeSpec end = ts;
        
        // Make sure time is going forward.
        assert(end.tv_sec >= _begin.tv_sec);

        // Elapsed time is just end - begin times.
        TimeSpec delta;
        delta.tv_sec = end.tv_sec - _begin.tv_sec;
        _elapsed.tv_sec += delta.tv_sec;

        // No need to check for sign or to normalize, because tv_nsec is
        // signed and 64-bit.
        delta.tv_nsec = end.tv_nsec - _begin.tv_nsec;
        _elapsed.tv_nsec += delta.tv_nsec;

        // Clear begin to catch misuse.
        _begin.tv_sec = 0;
        _begin.tv_nsec = 0;

        return double(delta.tv_sec) + double(delta.tv_nsec) * 1e-9;
    }
    double YaskTimer::get_secs_since_start() const {

        // Make sure timer was started.
        assert(_begin.tv_sec != 0);

        TimeSpec now, delta;
        now = get_timespec();

        // Elapsed time is just now - begin times.
        delta.tv_sec = now.tv_sec - _begin.tv_sec;
        delta.tv_nsec = now.tv_nsec - _begin.tv_nsec;

        return double(delta.tv_sec) + double(delta.tv_nsec) * 1e-9;
    }

    ////// MPI utils //////
    
    // Find sum of rank_vals over all ranks.
    idx_t sum_over_ranks(idx_t rank_val, MPI_Comm comm) {
        idx_t sum_val = rank_val;
#ifdef USE_MPI
        MPI_Allreduce(&rank_val, &sum_val, 1, MPI_INTEGER8, MPI_SUM, comm);
#endif
        return sum_val;
    }

    // Make sure rank_val is same over all ranks.
    void assert_equality_over_ranks(idx_t rank_val,
                                 MPI_Comm comm,
                                 const string& descr) {
        idx_t min_val = rank_val;
        idx_t max_val = rank_val;
#ifdef USE_MPI
        MPI_Allreduce(&rank_val, &min_val, 1, MPI_INTEGER8, MPI_MIN, comm);
        MPI_Allreduce(&rank_val, &max_val, 1, MPI_INTEGER8, MPI_MAX, comm);
#endif

        if (min_val != rank_val || max_val != rank_val) {
            FORMAT_AND_THROW_YASK_EXCEPTION("error: " << descr << " ranges from " << min_val << " to " <<
                                            max_val << " across the ranks; they should all be identical");
        }
    }

    ///////////// Command-line parsing methods. /////////////

    // Internal function to print help for one option.
    void CommandLineParser::OptionBase::_print_help(ostream& os,
                                                    const string& str,
                                                    int width) const
    {
        os << "  -" << str;

        // Split help into words.
        vector<string> words;
        size_t pos = 0, prev = 0;
        while (pos != string::npos) {
            pos = _help.find(' ', prev);
            if (pos != string::npos) {
                string word = _help.substr(prev, pos - prev);
                if (word.length())
                    words.push_back(word);
                prev = pos + 1;
            }
        }
        if (prev < _help.length())
            words.push_back(_help.substr(prev)); // last word.

        // Format help message to fit in width.
        pos = 0;
        for (size_t i = 0; i < words.size(); i++) {
            if (i == 0 || pos + words[i].length() > size_t(width)) {
                os << endl << _help_leader;
                pos = _help_leader.length();
            }
            else {
                os << ' ';
                pos += 1;
            }
            os << words[i];
            pos += words[i].length();
        }
        os << endl;
    }

    // Check for matching option to "-"str at args[argi].
    // Return true and increment argi if match.
    bool CommandLineParser::OptionBase::_is_opt(const std::vector<std::string>& args,
                                                   int& argi,
                                                   const std::string& str) const
    {
        string opt_str = string("-") + str;
        if (args.at(argi) == opt_str) {
            argi++;
            return true;
        }
        return false;
    }

    // Get one double value from args[argi].
    // On failure, print msg using string from args[argi-1] and exit.
    // On success, increment argi and return value.
    double CommandLineParser::OptionBase::_double_val(const vector<string>& args,
                                                     int& argi)
    {
        if (size_t(argi) >= args.size() || args[argi].length() == 0) {
            THROW_YASK_EXCEPTION("Error: no argument for option '" + args[argi - 1] + "'");
        }

        const char* nptr = args[argi].c_str();
        char* endptr = 0;
        double val = strtod(nptr, &endptr);
        if (!isfinite(val) || *endptr != '\0') {
            THROW_YASK_EXCEPTION("Error: argument for option '" + args[argi - 1] +
                                 "' is not a valid floating-point number");
        }

        argi++;
        return val;
    }

    // Get one idx_t value from args[argi].
    // On failure, print msg using string from args[argi-1] and exit.
    // On success, increment argi and return value.
    idx_t CommandLineParser::OptionBase::_idx_val(const vector<string>& args,
                                                  int& argi)
    {
        if (size_t(argi) >= args.size() || args[argi].length() == 0) {
            THROW_YASK_EXCEPTION("Error: no argument for option '" + args[argi - 1] + "'");
        }

        const char* nptr = args[argi].c_str();
        char* endptr = 0;
        long long int val = strtoll(nptr, &endptr, 0);
        if (val == LLONG_MIN || val == LLONG_MAX || *endptr != '\0') {
            THROW_YASK_EXCEPTION("Error: argument for option '" + args[argi - 1] + "' is not an integer");
        }

        argi++;
        return idx_t(val);
    }

    // Get one string value from args[argi].
    // On failure, print msg using string from args[argi-1] and exit.
    // On success, increment argi and return value.
    string CommandLineParser::OptionBase::_string_val(const vector<string>& args,
                                                      int& argi)
    {
        if (size_t(argi) >= args.size())
            THROW_YASK_EXCEPTION("Error: no argument for option '" + args[argi - 1] + "'");

        auto v = args[argi];
        argi++;
        return v;
    }

    // Check for a boolean option.
    bool CommandLineParser::BoolOption::check_arg(const std::vector<std::string>& args,
                                                  int& argi) {
        if (_is_opt(args, argi, _name)) {
            _val = true;
            return true;
        }
        string false_name = string("no-") + _name;
        if (_is_opt(args, argi, false_name)) {
            _val = false;
            return true;
        }
        return false;
    }

    // Print help on a boolean option.
    void CommandLineParser::BoolOption::print_help(ostream& os,
                                                   int width) const {
        _print_help(os, string("[no-]" + _name), width);
        os << _help_leader << _current_value_str <<
            (_val ? "true" : "false") << "." << endl;
    }

    // Check for a double option.
    bool CommandLineParser::DoubleOption::check_arg(const std::vector<std::string>& args,
                                                    int& argi) {
        if (_is_opt(args, argi, _name)) {
            _val = _double_val(args, argi);
            return true;
        }
        return false;
    }

    // Print help on a double option.
    void CommandLineParser::DoubleOption::print_help(ostream& os,
                                                     int width) const {
        _print_help(os, _name + " <floating-point number>", width);
        os << _help_leader << _current_value_str <<
            _val << "." << endl;
    }

    // Check for an int option.
    bool CommandLineParser::IntOption::check_arg(const std::vector<std::string>& args,
                                                 int& argi) {
        if (_is_opt(args, argi, _name)) {
            _val = (int)_idx_val(args, argi); // TODO: check for over/underflow.
            return true;
        }
        return false;
    }

    // Print help on an int option.
    void CommandLineParser::IntOption::print_help(ostream& os,
                                                  int width) const {
        _print_help(os, _name + " <integer>", width);
        os << _help_leader << _current_value_str <<
            _val << "." << endl;
    }

    // Check for an idx_t option.
    bool CommandLineParser::IdxOption::check_arg(const std::vector<std::string>& args,
                                                 int& argi) {
        if (_is_opt(args, argi, _name)) {
            _val = _idx_val(args, argi);
            return true;
        }
        return false;
    }

    // Print help on an idx_t option.
    void CommandLineParser::IdxOption::print_help(ostream& os,
                                                  int width) const {
        _print_help(os, _name + " <integer>", width);
        os << _help_leader << _current_value_str <<
            _val << "." << endl;
    }

    // Print help on an multi-idx_t option.
    void CommandLineParser::MultiIdxOption::print_help(ostream& os,
                                                  int width) const {
        _print_help(os, _name + " <integer>", width);
        os << _help_leader << _current_value_str;
        for (size_t i = 0; i < _vals.size(); i++) {
            if (i > 0)
                os << ", ";
            os << *_vals[i];
        }
        os << "." << endl;
    }

    // Check for an multi-idx_t option.
    bool CommandLineParser::MultiIdxOption::check_arg(const std::vector<std::string>& args,
                                                      int& argi) {
        if (_is_opt(args, argi, _name)) {
            idx_t val = _idx_val(args, argi);
            for (size_t i = 0; i < _vals.size(); i++)
                *_vals[i] = val;
            return true;
        }
        return false;
    }

    // Check for a string option.
    bool CommandLineParser::StringOption::check_arg(const std::vector<std::string>& args,
                                                    int& argi) {
        if (_is_opt(args, argi, _name)) {
            _val = _string_val(args, argi);
            return true;
        }
        return false;
    }

    // Print help on a string option.
    void CommandLineParser::StringOption::print_help(ostream& os,
                                                     int width) const {
        _print_help(os, _name + " <string>", width);
        os << _help_leader << _current_value_str <<
            _val << "." << endl;
    }

    // Check for a string-list option.
    bool CommandLineParser::StringListOption::check_arg(const std::vector<std::string>& args,
                                                        int& argi) {
        if (_is_opt(args, argi, _name)) {
            _val.clear();
            string strs = _string_val(args, argi);
            stringstream ss(strs);
            string str;
            while (getline(ss, str, ',')) {
                if (_allowed_strs.size() && _allowed_strs.count(str) == 0) {
                    THROW_YASK_EXCEPTION("Error: illegal argument '" + str + "' to option '" +
                                         args[argi - 2] + "'");
                }
                _val.push_back(str);
            }
            return true;
        }
        return false;
    }

    // Print help on a string-list option.
    void CommandLineParser::StringListOption::print_help(ostream& os,
                                                         int width) const {
        _print_help(os, _name + " <string[,string[,...]]>", width);
        os << _help_leader << _current_value_str;
        int n = 0;
        for (auto& v : _val) {
            if (n)
                os << ",";
            os << v;
            n++;
        }
        os << endl;
    }

    // Print help on all options.
    void CommandLineParser::print_help(ostream& os) const {
        for (auto oi : _opts) {
            const auto opt = oi.second;
            opt->print_help(os, _width);
        }
    }

    // Parse options from the command-line and set corresponding vars.
    // Recognized strings from args are consumed, and unused ones
    // are returned.
    string CommandLineParser::parse_args(const std::string& pgm_name,
                                         const std::vector<std::string>& args) {
        vector<string> non_args;

        // Loop through strings in args.
        for (int argi = 0; argi < int(args.size()); ) {

            // Compare against all registered options.
            bool matched = false;
            for (auto oi : _opts) {
                auto opt = oi.second;

                // If a match is found, argi will be incremeted
                // as needed beyond option and/or its arg.
                if (opt->check_arg(args, argi)) {
                    matched = true;
                    break;
                }
            }

            // Save unused args.
            if (!matched) {
                string opt = args[argi];
                non_args.push_back(opt);
                argi++;
            }
        }

        // Return any left-over strings.
        string rem;
        for (auto r : non_args) {
            if (rem.length())
                rem += " ";
            // TODO: add quotes around 'r' if it has a space.
            rem += r;
        }
        return rem;
    }

    // Tokenize args from a string.
    vector<string> CommandLineParser::set_args(const string& arg_string) {
        string tmp;            // current arg.
        char in_quote = '\0';  // current string delimiter or null if none.
        vector<string> args;
        for (char c : arg_string) {

            // If in quotes, add to string or handle end.
            if (in_quote != '\0') {

                // End of quoted string, i.e., this char
                // matches opening quote.
                if (in_quote == c) {
                    args.push_back(tmp); // may be empty string.
                    tmp.clear();
                    in_quote = '\0';
                }

                else
                    tmp += c;
            }

            // If WS, save old string and start a new string.
            else if (isspace(c)) {
                if (tmp.length())
                    args.push_back(tmp);
                tmp.clear();
            }

            // If quote, remember delimiter.
            else if (c == '"' || c == '\'') {
                in_quote = c;
            }

            // Otherwise, just add to tmp.
            else
                tmp += c;
        }

        if (in_quote != '\0')
            THROW_YASK_EXCEPTION("Error: unterminated quote in '" +
                                 arg_string + "'");

        // Last string.
        if (tmp.length())
            args.push_back(tmp);
        return args;
    }
}
