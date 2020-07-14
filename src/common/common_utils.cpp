/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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
#include "common_utils.hpp"

using namespace std;

namespace yask {

    // Update this version string anytime changes are
    // committed to a repository, especially when
    // affecting master or develop branches.
    // Be sure to keep 2 digits in minor and patch
    // fields to allow proper alphanumeric sorting
    // for numbers above 9 (at least up to 99).

    // Format: "major.minor.patch".
    const string version = "3.06.00";

    string yask_get_version_string() {
        return version;
    }

    // Controls whether make*Str() functions add
    // suffixes or just print full number for
    // machine parsing.
    bool is_suffix_print_enabled = true;
    
    // Return num with SI multiplier and "iB" suffix,
    // e.g., 412KiB.
    string make_byte_str(size_t nbytes)
    {
        if (!is_suffix_print_enabled)
            return to_string(nbytes);
        
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
            os << num;
        os << "B";
        return os.str();
    }

    // Return num with SI multiplier, e.g. "3.14M".
    // Use this one for rates, etc.
    string make_num_str(idx_t num) {
        if (!is_suffix_print_enabled || (num >= 0 && num < 1000))
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
        if (num == 0.)
            os << num;
        else if (num > one_e)
            os << (num / one_e) << "E";
        else if (num > one_p)
            os << (num / one_p) << "P";
        else if (num > one_t)
            os << (num / one_t) << "T";
        else if (num > one_g)
            os << (num / one_g) << "G";
        else if (num > one_m)
            os << (num / one_m) << "M";
        else if (num > one_k)
            os << (num / one_k) << "K"; // NB: official SI symbol is "k".
#ifdef USE_PICO
        else if (num < onep)
            os << (num / onef) << "f";
        else if (num < onen)
            os << (num / onep) << "p";
#endif
        else if (num < oneu)
            os << (num / onen) << "n";
        else if (num < onem)
            os << (num / oneu) << "u"; // NB: official SI symbol is Greek mu.
        else if (num < 1.)
            os << (num / onem) << "m";
        else
            os << num;
        return os.str();
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

}
