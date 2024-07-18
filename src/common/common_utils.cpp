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

//////// Some common code shared between YASK compiler and kernel. //////////

// Include this first to assure NDEBUG is set properly.
#include "yask_assert.hpp"

#include <sstream>
#include <regex>
#include "common_utils.hpp"

using namespace std;

namespace yask {

    // Update this version string anytime changes are
    // committed to a repository, especially when
    // affecting master or develop branches.
    // Be sure to keep 2 digits in minor and patch
    // fields to allow proper alphanumeric sorting
    // for numbers above 9 (at least up to 99).

    // TODO: conform to Semantic Versioning:
    // https://semver.org/.

    // Format: "major.minor.patch[-alpha|-beta]".
    const string version = "4.05.04";

    string yask_get_version_string() {
        return version;
    }

    // Controls whether make*Str() functions add
    // suffixes or just print full number for
    // machine parsing.
    bool is_suffix_print_enabled = true;
    
    // Return num with SI multiplier and "iB" suffix,
    // e.g., 412KiB.
    // Use only for storage bytes, e.g., not for
    // rates like bytes/sec.
    string make_byte_str(size_t nbytes)
    {
        if (!is_suffix_print_enabled)
            return to_string(nbytes) + " Bytes";
        
        ostringstream os;
        double num = double(nbytes);
        const double one_ki = 1024;
        const double one_mi = one_ki * one_ki;
        const double one_gi = one_ki * one_mi;
        const double one_ti = one_ki * one_gi;
        const double one_pi = one_ki * one_ti;
        const double one_ei = one_ki * one_pi;
        if (num > one_ei)
            os << (num / one_ei) << "Ei";
        else if (num > one_pi)
            os << (num / one_pi) << "Pi";
        else if (num > one_ti)
            os << (num / one_ti) << "Ti";
        else if (num > one_gi)
            os << (num / one_gi) << "Gi";
        else if (num > one_mi)
            os << (num / one_mi) << "Mi";
        else if (num > one_ki)
            os << (num / one_ki) << "Ki";
        else
            os << nbytes;
        os << "B";
        return os.str();
    }

    // Return num with SI multiplier, e.g. "3.14M".
    // Use this one for printing any number that is
    // not number of storage bytes.
    string make_num_str(idx_t num) {
        if (!is_suffix_print_enabled || (num > -1000 && num < 1000))
            return to_string(num);
        return make_num_str(double(num));
    }
    string make_num_str(double num)
    {
        if (!is_suffix_print_enabled)
            return to_string(num);

        ostringstream os;
        const double one_k = 1e3;
        const double one_m = 1e6;
        const double one_g = 1e9;
        const double one_t = 1e12;
        const double one_p = 1e15;
        const double one_e = 1e18;
        const double onem = 1e-3;
        const double oneu = 1e-6;
        const double onen = 1e-9;
#ifdef USE_PICO
        const double onep = 1e-12;
        const double onef = 1e-15;
#endif
        double absnum = abs(num);
        if (num < 0.)
            os << "-";
        if (num == 0.)
            os << num;
        else if (absnum > one_e)
            os << (absnum / one_e) << "E";
        else if (absnum > one_p)
            os << (absnum / one_p) << "P";
        else if (absnum > one_t)
            os << (absnum / one_t) << "T";
        else if (absnum > one_g)
            os << (absnum / one_g) << "G";
        else if (absnum > one_m)
            os << (absnum / one_m) << "M";
        else if (absnum > one_k)
            os << (absnum / one_k) << "K"; // NB: official SI symbol is "k".
#ifdef USE_PICO
        else if (absnum < onep)
            os << (absnum / onef) << "f";
        else if (absnum < onen)
            os << (absnum / onep) << "p";
#endif
        else if (absnum < oneu)
            os << (absnum / onen) << "n";
        else if (absnum < onem)
            os << (absnum / oneu) << "u"; // NB: official SI symbol is Greek mu.
        else if (absnum < 1.)
            os << (absnum / onem) << "m";
        else
            os << absnum;
        return os.str();
    }

    // Add quotes around whitespace in string.
    std::string quote_whitespace(const std::string& str) {
        auto pos = str.find_first_of(" \t\r\n");
        if (pos != string::npos) {
            string r = str;

            // Must not use chars already in the string.
            auto apos = r.find('\'');
            auto qpos = r.find('"');
            if (apos == string::npos) // No apostrophes.
                r = string("\"") + r + '"'; // Add quotes.
            else if (qpos == string::npos) // No quotes.
                r = string("'") + r + "'"; // Add apostrophes.
            else { // Has both :(.
                r = regex_replace(r, regex("'"), "\\'"); // Escape apostrophes.
                r = string("'") + r + "'"; // Add apostrophes.
            }
            return r;
        }
        return str;
    }

    // A var that behaves like OMP_NUM_THREADS.
    int yask_num_threads[yask_max_levels] = { 0 };

    // See yask_common_api.hpp for documentation.
    const char* yask_exception::what() const noexcept {
        return _msg.c_str();
    }

    void yask_exception::add_message(const string& arg_msg) {
        _msg.append(arg_msg);
    }

    const char* yask_exception::get_message() const {
        return _msg.c_str();
    }

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

    ///////////// Command-line parsing methods. /////////////

    // Internal function to print help for one option.
    void command_line_parser::option_base::_print_help(ostream& os,
                                                    const string& str,
                                                    int width) const
    {
        os << "  -" << str << endl;

        // Split help into lines.
        vector<string> lines;
        size_t lpos = 0, lprev = 0;
        while (lpos != string::npos) {
            lpos = _help.find('\n', lprev);
            if (lpos != string::npos) {
                string line = _help.substr(lprev, lpos - lprev);
                lines.push_back(line);
                lprev = lpos + 1;
            }
        }
        if (lprev < _help.length())
            lines.push_back(_help.substr(lprev)); // last line.

        // Split lines into words.
        // Uses only spaces, not tabs.
        for (auto& line : lines) {
            vector<string> words;
            size_t pos = 0, prev = 0;
            while (pos != string::npos) {
                pos = line.find(' ', prev);
                if (pos != string::npos) {
                    string word = line.substr(prev, pos - prev);
                    if (word.length())
                        words.push_back(word);
                    prev = pos + 1;
                }
            }
            if (prev < line.length())
                words.push_back(line.substr(prev)); // last word.
            bool is_bullet = (words.size() > 1) &&
                words[0] == "-";

            // Format help message to fit in width.
            pos = 0;
            for (size_t i = 0; i < words.size(); i++) {
                if (i == 0 || pos + words[i].length() > size_t(width)) {
                    pos = 0;
                    if (i > 0) {
                        os << endl;
                        if (is_bullet) {
                            os << "  ";
                            pos += 2;
                        }
                    }
                    os << _help_leader;
                    pos += _help_leader.length();
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

        // Print current value.
        os << _help_leader << _current_value_str;
        print_value(os) << ".\n";
    }

    // Check for matching option to "-"str at args[argi].
    // Return true and increment argi if match.
    bool command_line_parser::option_base::_is_opt(const string_vec& args,
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
    double command_line_parser::option_base::_double_val(const vector<string>& args,
                                                     int& argi)
    {
        if (size_t(argi) >= args.size() || args[argi].length() == 0) {
            THROW_YASK_EXCEPTION("no argument for option '" + args[argi - 1] + "'");
        }

        const char* nptr = args[argi].c_str();
        char* endptr = 0;
        double val = strtod(nptr, &endptr);
        if (!isfinite(val) || *endptr != '\0') {
            THROW_YASK_EXCEPTION("argument for option '" + args[argi - 1] +
                                 "' is not a valid floating-point number");
        }

        argi++;
        return val;
    }

    // Get one idx_t value from args[argi].
    // On failure, print msg using string from args[argi-1] and exit.
    // On success, increment argi and return value.
    idx_t command_line_parser::option_base::_idx_val(const vector<string>& args,
                                                  int& argi)
    {
        if (size_t(argi) >= args.size() || args[argi].length() == 0) {
            THROW_YASK_EXCEPTION("no argument for option '" + args[argi - 1] + "'");
        }

        const char* nptr = args[argi].c_str();
        char* endptr = 0;
        long long int val = strtoll(nptr, &endptr, 0);
        if (val == LLONG_MIN || val == LLONG_MAX || *endptr != '\0') {
            THROW_YASK_EXCEPTION("argument for option '" + args[argi - 1] + "' is not an integer");
        }

        argi++;
        return idx_t(val);
    }

    // Get one string value from args[argi].
    // On failure, print msg using string from args[argi-1] and exit.
    // On success, increment argi and return value.
    string command_line_parser::option_base::_string_val(const vector<string>& args,
                                                      int& argi)
    {
        if (size_t(argi) >= args.size())
            THROW_YASK_EXCEPTION("no argument for option '" + args[argi - 1] + "'");

        auto v = args[argi];
        argi++;
        return v;
    }

    // Check for a boolean option.
    bool command_line_parser::bool_option::check_arg(const string_vec& args,
                                                  int& argi) {
        if (_is_opt(args, argi, get_name())) {
            _val = true;
            return true;
        }
        string false_name = string("no-") + get_name();
        if (_is_opt(args, argi, false_name)) {
            _val = false;
            return true;
        }
        return false;
    }

    // Print help on a boolean option.
    void command_line_parser::bool_option::print_help(ostream& os,
                                                   int width) const {
        _print_help(os, string("[no-]" + get_name()), width);
    }

    // Check for a double option.
    bool command_line_parser::double_option::check_arg(const string_vec& args,
                                                    int& argi) {
        if (_is_opt(args, argi, get_name())) {
            _val = _double_val(args, argi);
            return true;
        }
        return false;
    }

    // Print help on a double option.
    void command_line_parser::double_option::print_help(ostream& os,
                                                     int width) const {
        _print_help(os, get_name() + " <floating-point number>", width);
    }

    // Check for an int option.
    bool command_line_parser::int_option::check_arg(const string_vec& args,
                                                 int& argi) {
        if (_is_opt(args, argi, get_name())) {
            _val = (int)_idx_val(args, argi); // TODO: check for over/underflow.
            return true;
        }
        return false;
    }

    // Print help on an int option.
    void command_line_parser::int_option::print_help(ostream& os,
                                                  int width) const {
        _print_help(os, get_name() + " <integer>", width);
    }

    // Check for an idx_t option.
    bool command_line_parser::idx_option::check_arg(const string_vec& args,
                                                 int& argi) {
        if (_is_opt(args, argi, get_name())) {
            _val = _idx_val(args, argi);
            return true;
        }
        return false;
    }

    // Print help on an idx_t option.
    void command_line_parser::idx_option::print_help(ostream& os,
                                                  int width) const {
        _print_help(os, get_name() + " <integer>", width);
    }

    // Check for a string option.
    bool command_line_parser::string_option::check_arg(const string_vec& args,
                                                    int& argi) {
        if (_is_opt(args, argi, get_name())) {
            _val = _string_val(args, argi);
            return true;
        }
        return false;
    }

    // Print help on a string option.
    void command_line_parser::string_option::print_help(ostream& os,
                                                     int width) const {
        _print_help(os, get_name() + " <string>", width);
    }

    // Check for a string-list option.
    bool command_line_parser::string_list_option::check_arg(const string_vec& args,
                                                        int& argi) {
        if (_is_opt(args, argi, get_name())) {
            _val.clear();
            string strs = _string_val(args, argi);
            stringstream ss(strs);
            string str;
            while (getline(ss, str, ',')) {
                if (_allowed_strs.size() && _allowed_strs.count(str) == 0) {
                    THROW_YASK_EXCEPTION("illegal argument '" + quote_whitespace(str) +
                                         "' to option '" + args[argi - 2] + "'");
                }
                _val.push_back(str);
            }
            return true;
        }
        return false;
    }

    // Print help on a string-list option.
    void command_line_parser::string_list_option::print_help(ostream& os,
                                                         int width) const {
        _print_help(os, get_name() + " <string[,string[,...]]>", width);
    }

    // Print help on all options.
    void command_line_parser::print_help(ostream& os) const {
        for (auto oi : _opts) {
            const auto opt = oi.second;
            opt->print_help(os, _width);
        }
    }

    // Print settings of all options.
    void command_line_parser::print_values(ostream& os) const {
        const size_t name_wid = 22;
        for (auto oi : _opts) {
            const auto& name = oi.first;
            const auto& opt = oi.second;
            os << "  " << name << ":  ";
            if (name.length() < name_wid)
                for (size_t i = 0; i < name_wid - name.length(); i++)
                    os << " ";
            opt->print_value(os) << endl;
        }
    }

    // Parse options from the command-line and set corresponding vars.
    // Recognized strings from args are consumed, and unused ones
    // are returned.
    string command_line_parser::parse_args(const std::string& pgm_name,
                                         const string_vec& args) {
        vector<string> non_args;

        // Loop through strings in args.
        for (int argi = 0; argi < int(args.size()); ) {

            // Compare against all registered options.
            bool matched = false;
            for (auto oi : _opts) {
                auto opt = oi.second;

                // If a match is found, argi will be incremented
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
                rem += " "; // Space between words.

            // Add quotes around 'r' if it has whitespace.
            r = quote_whitespace(r);
            
            rem += r;
        }
        return rem;
    }

    // Tokenize args from a string.
    vector<string> command_line_parser::set_args(const string& arg_string) {
        string tmp;            // current arg.
        bool is_ok = false;    // current arg is valid.
        char in_quote = '\0';  // current string delimiter or null if none.
        vector<string> args;
        for (char c : arg_string) {

            // If in quotes, add to string or handle end.
            if (in_quote != '\0') {

                // End of quoted string, i.e., this char
                // matches opening quote.
                if (in_quote == c) {
                    is_ok = true; // even if empty.
                    in_quote = '\0';
                }

                else
                    tmp += c;
            }

            // If WS, save old string and start a new string.
            else if (isspace(c)) {
                if (is_ok)
                    args.push_back(tmp);
                tmp.clear();
                is_ok = false;
            }

            // If quote, remember delimiter.
            else if (c == '"' || c == '\'') {
                in_quote = c;
            }

            // Otherwise, just add to tmp.
            else {
                tmp += c;
                is_ok = true;
            }
        }

        if (in_quote != '\0')
            THROW_YASK_EXCEPTION("unterminated quote in '" +
                                 arg_string + "'");

        // Last string.
        if (is_ok)
            args.push_back(tmp);
        return args;
    }

    // Print a spash message.
    void yask_print_splash(ostream& os, int argc, char** argv,
                           string invocation_leader) {
        // See https://en.wikipedia.org/wiki/Box-drawing_character.
        os <<
            "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n"
            "\u2502     Y*A*S*K \u2500\u2500 Yet Another Stencil Kit     \u2502\n"
            "\u2502       https://github.com/intel/yask        \u2502\n"
            "\u2502 Copyright (c) 2014-2024, Intel Corporation \u2502\n"
            "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\n"
            "Version: " << yask_get_version_string() << endl;

        // Echo invocation parameters for record-keeping.
        if (argc) {
            os << invocation_leader;
            for (int argi = 0; argi < argc; argi++) {
                if (argi)
                    os << " ";
                os << quote_whitespace(argv[argi]);
            }
            os << endl;
        }
    }
    
} // namespace.
