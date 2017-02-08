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

#include "stencil.hpp"

// Set MODEL_CACHE to 1 or 2 to model that cache level
// and create a global cache object here.
#ifdef MODEL_CACHE
Cache cache_model(MODEL_CACHE);
#endif

using namespace std;
namespace yask {

    double getTimeInSecs() {

#if defined(_OPENMP)
        return omp_get_wtime();

#elif defined(WIN32)
        return 1e-3 * (double)GetTickCount64();

#else
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);

        return ((double)(ts.tv_sec) + 
                1e-9 * (double)(ts.tv_nsec));
#endif
    }

    // Return num with SI multiplier.
    // Use this one for bytes, etc.
    string printWithPow2Multiplier(double num)
    {
        ostringstream os;
        const double oneK = 1024.;
        const double oneM = oneK * oneK;
        const double oneG = oneK * oneM;
        const double oneT = oneK * oneG;
        if (num > oneT)
            os << (num / oneT) << "Ti";
        else if (num > oneG)
            os << (num / oneG) << "Gi";
        else if (num > oneM)
            os << (num / oneM) << "Mi";
        else if (num > oneK)
            os << (num / oneK) << "Ki";
        else
            os << num;
        return os.str();
    }

    // Return num with SI multiplier.
    // Use this one for rates, etc.
    string printWithPow10Multiplier(double num)
    {
        ostringstream os;
        const double oneK = 1e3;
        const double oneM = 1e6;
        const double oneG = 1e9;
        const double oneT = 1e12;
        if (num > oneT)
            os << (num / oneT) << "T";
        else if (num > oneG)
            os << (num / oneG) << "G";
        else if (num > oneM)
            os << (num / oneM) << "M";
        else if (num > oneK)
            os << (num / oneK) << "K";
        else
            os << num;
        return os.str();
    }

    // Round up val to a multiple of mult.
    // Print a message if rounding is done.
    idx_t roundUp(ostream& os, idx_t val, idx_t mult, const string& name)
    {
        assert(mult > 0);
        idx_t res = val;
        if (val % mult != 0) {
            res = ROUND_UP(res, mult);
            os << "Adjusting " << name << " from " << val << " to " <<
                res << " to be a multiple of " << mult << endl;
        }
        return res;
    }
    
    // Fix bsize, if needed, to fit into rsize and be a multiple of mult.
    // Return number of blocks.
    idx_t findNumSubsets(ostream& os,
                         idx_t& bsize, const string& bname,
                         idx_t rsize, const string& rname,
                         idx_t mult, const string& dim) {
        if (bsize < 1) bsize = rsize; // 0 => use full size.
        bsize = ROUND_UP(bsize, mult);
        if (bsize > rsize) bsize = rsize;
        idx_t nblks = (rsize + bsize - 1) / bsize;
        idx_t rem = rsize % bsize;
        idx_t nfull_blks = rem ? (nblks - 1) : nblks;

        os << " In '" << dim << "' dimension, " << rname << " of size " <<
            rsize << " is divided into " << nfull_blks << " " << bname << "(s) of size " << bsize;
        if (rem)
            os << " plus 1 remainder " << bname << " of size " << rem;
        os << "." << endl;
        return nblks;
    }

    // Find sum of rank_vals over all ranks.
    idx_t sumOverRanks(idx_t rank_val, MPI_Comm comm) {
        idx_t sum_val = rank_val;
#ifdef USE_MPI
        MPI_Allreduce(&rank_val, &sum_val, 1, MPI_INTEGER8, MPI_SUM, comm);
#endif
        return sum_val;
    }

    // Make sure rank_val is same over all ranks.
    void assertEqualityOverRanks(idx_t rank_val,
                                 MPI_Comm comm,
                                 const string& descr) {
        idx_t min_val = rank_val;
        idx_t max_val = rank_val;
#ifdef USE_MPI
        MPI_Allreduce(&rank_val, &min_val, 1, MPI_INTEGER8, MPI_MIN, comm);
        MPI_Allreduce(&rank_val, &max_val, 1, MPI_INTEGER8, MPI_MAX, comm);
#endif

        if (min_val != rank_val || max_val != rank_val) {
            cerr << "error: " << descr << " values range from " << min_val << " to " <<
                max_val << " across the ranks. They should all be identical." << endl;
            exit_yask(1);
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
    bool CommandLineParser::OptionBase::_check_arg(std::vector<std::string>& args,
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

    // Get one idx_t value from args[argi].
    // On failure, print msg using string from args[argi-1] and exit.
    // On success, increment argi and return value.
    idx_t CommandLineParser::OptionBase::_idx_val(vector<string>& args,
                                                  int& argi)
    {
        if (size_t(argi) >= args.size() || args[argi].length() == 0) {
            cerr << "Error: no argument for option '" << args[argi - 1] << "'." << endl;
            exit(1);
        }

        const char* nptr = args[argi].c_str();
        char* endptr = 0;
        long long int val = strtoll(nptr, &endptr, 0);
        if (val == LLONG_MIN || val == LLONG_MAX || *endptr != '\0') {
            cerr << "Error: argument for option '" << args[argi - 1] << "' is not an integer." << endl;
            exit(1);
        }

        argi++;
        return idx_t(val);
    }

    // Check for a boolean option.
    bool CommandLineParser::BoolOption::check_arg(std::vector<std::string>& args,
                                                  int& argi) {
        if (_check_arg(args, argi, _name)) {
            _val = true;
            return true;
        }
        string false_name = string("no-") + _name;
        if (_check_arg(args, argi, false_name)) {
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
    
    // Check for an int option.
    bool CommandLineParser::IntOption::check_arg(std::vector<std::string>& args,
                                                 int& argi) {
        if (_check_arg(args, argi, _name)) {
            _val = (int)_idx_val(args, argi); // TODO: check for under/overflow.
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
    bool CommandLineParser::IdxOption::check_arg(std::vector<std::string>& args,
                                                 int& argi) {
        if (_check_arg(args, argi, _name)) {
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
    bool CommandLineParser::MultiIdxOption::check_arg(std::vector<std::string>& args,
                                                      int& argi) {
        if (_check_arg(args, argi, _name)) {
            idx_t val = _idx_val(args, argi);
            for (size_t i = 0; i < _vals.size(); i++)
                *_vals[i] = val;
            return true;
        }
        return false;
    }

    // Print help on all options.
    void CommandLineParser::print_help(ostream& os) const {
        for (auto oi : _opts) {
            const auto* opt = oi.second;
            opt->print_help(os, _width);
        }
    }

    // Parse options from the command-line and set corresponding vars.
    // Recognized strings from args are consumed, and unused ones
    // remain for further processing by the application.
    void CommandLineParser::parse_args(const std::string& pgmName,
                                       std::vector<std::string>& args) {
        vector<string> non_args;

        // Loop through strings in args.
        for (int argi = 0; argi < int(args.size()); ) {

            // Compare against all registered options.
            bool matched = false;
            for (auto oi : _opts) {
                auto* opt = oi.second;

                // If a match is found, argi will be incremeted
                // as needed beyond option and/or its arg.
                if (opt->check_arg(args, argi)) {
                    matched = true;
                    break;
                }
            }                

            // Save unused args.
            if (!matched) {
                string& opt = args[argi];
                non_args.push_back(opt);
                argi++;
            }
        }        

        // Return unused args in args var.
        args.swap(non_args);
    }
    
}
