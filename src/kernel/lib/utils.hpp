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
                     ("(internal fault) " << fn << "() w/lock @ " << (void*)this <<
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

}

