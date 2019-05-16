/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2019, Intel Corporation

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
    void YaskTimer::start(TimeSpec* ts) {

        // Make sure timer was stopped.
        assert(_begin.tv_sec == 0);
        assert(_begin.tv_nsec == 0);
        
        if (ts)
            _begin = *ts;
        else {
            auto cts = get_timespec();
            _begin = cts;
        }
    }
    double YaskTimer::stop(TimeSpec* ts) {
        TimeSpec end, delta;
        if (ts)
            end = *ts;
        else {
            auto cts = get_timespec();
            end = cts;
        }

        // Make sure timer was started.
        assert(_begin.tv_sec != 0);

        // Make sure time is going forward.
        assert(end.tv_sec >= _begin.tv_sec);
        
        // Elapsed time is just end - begin times.
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
    
    // Aligned allocation.
    char* alignedAlloc(std::size_t nbytes) {

        // Alignment to use based on size.
        const size_t _def_alignment = CACHELINE_BYTES;
        const size_t _def_big_alignment = YASK_HUGE_ALIGNMENT;
        size_t align = (nbytes >= _def_big_alignment) ?
            _def_big_alignment : _def_alignment;
        void *p = 0;

        // Some envs have posix_memalign(), some have aligned_alloc().
#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
        int ret = posix_memalign(&p, align, nbytes);
        if (ret) p = 0;
#else
        p = aligned_alloc(align, nbytes);
#endif

        if (!p)
            THROW_YASK_EXCEPTION("error: cannot allocate " + makeByteStr(nbytes));
        return static_cast<char*>(p);
    }

#ifdef USE_PMEM
    static int pmem_tmpfile(const char *dir, size_t size, int *fd, void **addr)
    {
        static char tmpl[] = "/appdirect_memXXXXXX";
        int err = 0;

        char fullname[strlen(dir) + sizeof (tmpl)];
        (void) strcpy(fullname, dir);
        (void) strcat(fullname, tmpl);

        if ((*fd = mkstemp(fullname)) < 0) {
            perror("mkstemp()");
            err = MEMKIND_ERROR_RUNTIME;
            THROW_YASK_EXCEPTION("Error: MEMKIND_ERROR_RUNTIME - mkstemp()\n");
        }

        (void) unlink(dir);

        if (ftruncate(*fd, size) != 0) {
            err = MEMKIND_ERROR_RUNTIME;
            THROW_YASK_EXCEPTION("Error: MEMKIND_ERROR_RUNTIME - ftruncate()\n");
        }

        *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
        if (*addr == MAP_FAILED) {
            err = MEMKIND_ERROR_RUNTIME;
            THROW_YASK_EXCEPTION("Error: MEMKIND_ERROR_RUNTIME - mmap()\n");
        }

        return err;
    }
#endif

    // NUMA allocation.
    // 'numa_pref' == yask_numa_none: use default aligned alloc.
    // 'numa_pref' >= 0: preferred NUMA node.
    // 'numa_pref' < 0: use NUMA policy corresponding to value.
    // TODO: get rid of magic-number scheme.
    char* numaAlloc(std::size_t nbytes, int numa_pref) {

        void *p = 0;

        if (numa_pref == yask_numa_none)
            return alignedAlloc(nbytes);

#ifdef USE_NUMA

        // Should we use the numa policy library?
#ifdef USE_NUMA_POLICY_LIB
#pragma omp single
        else if (numa_available() != -1) {
            numa_set_bind_policy(0);
            if (numa_pref >= 0 && numa_pref <= numa_max_node())
                numa_alloc_onnode(nbytes, numa_pref);
            else
                numa_alloc_local(nbytes);
            // Interleaved not available.
        }
        else
            THROW_YASK_EXCEPTION("Error: explicit NUMA policy allocation is not available");

        // Use mmap/mbind explicitly.
#else
        else if (get_mempolicy(NULL, NULL, 0, 0, 0) == 0) {

            // Set mmap flags.
            int mmprot = PROT_READ | PROT_WRITE;
            int mmflags = MAP_PRIVATE | MAP_ANONYMOUS;

            // Get an anonymous R/W memory map.
            p = mmap(0, nbytes, mmprot, mmflags, -1, 0);

            // If successful, apply the desired binding.
            if (p && p != MAP_FAILED) {
                if (numa_pref >= 0) {

                    // Prefer given node.
                    unsigned long nodemask = 0x1UL << numa_pref;
                    mbind(p, nbytes, MPOL_PREFERRED, &nodemask, sizeof(nodemask) * 8, 0);
                }
                else if (numa_pref == yask_numa_interleave) {

                    // Use all nodes.
                    unsigned long nodemask = (unsigned long)-1;
                    mbind(p, nbytes, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8, 0);
                }

                else{

                    // Use local node.
                    // MPOL_LOCAL was defined in Linux 3.8, so use
                    // MPOL_DEFAULT as backup on old systems.
#ifdef MPOL_LOCAL
                    mbind(p, nbytes, MPOL_LOCAL, 0, 0, 0);
#else
                    mbind(p, nbytes, MPOL_DEFAULT, 0, 0, 0);
#endif
                }
            }
            else
                THROW_YASK_EXCEPTION("Error: anonymous mmap of " + makeByteStr(nbytes) +
                                     " failed");
        }
        else
            THROW_YASK_EXCEPTION("Error: explicit NUMA policy allocation is not available");

#endif // not USE_NUMA_POLICY_LIB.

#else
        THROW_YASK_EXCEPTION("Error: NUMA allocation is not enabled; build with numa=1");
#endif // USE_NUMA.
        
        // Should not get here w/null p; throw exception.
        if (!p)
            THROW_YASK_EXCEPTION("Error: cannot allocate " + makeByteStr(nbytes));

        // Check alignment.
        if ((size_t(p) & (CACHELINE_BYTES - 1)) != 0)
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: NUMA-allocated " << p << " is not " <<
                                            CACHELINE_BYTES << "-byte aligned");

        // Return as a char* as required for shared_ptr ctor.
        return static_cast<char*>(p);
    }

    // Reverse numaAlloc().
    void NumaDeleter::operator()(char* p) {

        if (p && _numa_pref == yask_numa_none) {
            free(p);
            p = NULL;
        }

#ifdef USE_NUMA
#ifdef USE_NUMA_POLICY_LIB
        if (p && numa_available() != -1) {
            numa_free(p, _nbytes);
            p = NULL;
        }
#else
        if (p && get_mempolicy(NULL, NULL, 0, 0, 0) == 0) {
            munmap(p, _nbytes);
            p = NULL;
        }
#endif
#endif
        if (p) {
            free(p);
            p = NULL;
        }
    }
        
    // PMEM allocation.
    char* pmemAlloc(std::size_t nbytes, int dev_num) {

        void *p = 0;

        // Allocate into pmem.
#ifdef USE_PMEM
        int err = 0;
        int fd;
        // 'X' of pmemX should be matched with the NUMA node.
        string pmem_name("/mnt/pmem");
        pmem_name += to_string(dev_num);
        err = pmem_tmpfile(pmem_name.c_str(), nbytes, &fd, &p);
        if (err)
            THROW_YASK_EXCEPTION("Error: Unable to create temporary file for PMEM");
#else
        THROW_YASK_EXCEPTION("Error: PMEM allocation is not enabled; build with pmem=1");
#endif

        // Check alignment.
        if ((size_t(p) & (CACHELINE_BYTES - 1)) != 0)
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: PMEM-allocated " << p << " is not " <<
                                            CACHELINE_BYTES << "-byte aligned");

        // Return as a char* as required for shared_ptr ctor.
        return static_cast<char*>(p);
    }

    // Reverse pmemAlloc().
    void PmemDeleter::operator()(char* p) {
        if (p) {
            munmap(p, _nbytes);
            p = NULL;
        }
    }
        
    // MPI shm allocation.
    char* shmAlloc(std::size_t nbytes,
                   const MPI_Comm* shm_comm, MPI_Win* shm_win) {

        void *p = 0;

        // Allocate using MPI shm.
#ifdef USE_MPI
        assert(shm_comm);
        assert(shm_win);
        MPI_Info win_info;
        MPI_Info_create(&win_info);
        MPI_Info_set(win_info, "alloc_shared_noncontig", "true");
        MPI_Win_allocate_shared(nbytes, 1, win_info, *shm_comm, &p, shm_win);
        MPI_Info_free(&win_info);
        MPI_Win_lock_all(0, *shm_win);
#else
        THROW_YASK_EXCEPTION("Error: MPI shm allocation is not enabled; build with mpi=1");
#endif

        // Check alignment.
        if ((size_t(p) & (CACHELINE_BYTES - 1)) != 0)
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: MPI shm-allocated " << p << " is not " <<
                                            CACHELINE_BYTES << "-byte aligned");

        // Return as a char* as required for shared_ptr ctor.
        return static_cast<char*>(p);
    }

    // Reverse shmAlloc().
    void ShmDeleter::operator()(char* p) {

#ifdef USE_MPI
        assert(_shm_comm);
        assert(_shm_win);
        MPI_Win_unlock_all(*_shm_win);
        MPI_Win_free(_shm_win);
        p = NULL;
#else
        THROW_YASK_EXCEPTION("Error: MPI shm deallocation is not enabled; build with mpi=1");
#endif
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
    bool CommandLineParser::OptionBase::_check_arg(const std::vector<std::string>& args,
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

    // Check for a boolean option.
    bool CommandLineParser::BoolOption::check_arg(const std::vector<std::string>& args,
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
    bool CommandLineParser::IntOption::check_arg(const std::vector<std::string>& args,
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
    bool CommandLineParser::IdxOption::check_arg(const std::vector<std::string>& args,
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
    bool CommandLineParser::MultiIdxOption::check_arg(const std::vector<std::string>& args,
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
    // are returned.
    string CommandLineParser::parse_args(const std::string& pgmName,
                                         const std::vector<std::string>& args) {
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
        string tmp;
        bool in_quotes = false;
        vector<string> args;
        for (char c : arg_string) {

            // If WS, start a new string unless in quotes.
            if (isspace(c)) {
                if (in_quotes)
                    tmp += c;
                else {
                    if (tmp.length())
                        args.push_back(tmp);
                    tmp.clear();
                }
            }

            // If quote, start or end double-quotes.
            // TODO: handle single-quotes.
            else if (c == '"') {
                if (in_quotes) {
                    if (tmp.length())
                        args.push_back(tmp);
                    tmp.clear();
                    in_quotes = false;
                }
                else
                    in_quotes = true;
            }

            // Otherwise, just add to tmp.
            else
                tmp += c;
        }

        // Last string.
        if (tmp.length())
            args.push_back(tmp);
        return args;
    }
}
