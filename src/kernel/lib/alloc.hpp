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

#pragma once

// Support for allocating data in various ways.

// Provide the needed definitions for NUMA support.
// This is fairly convoluted because of the inconsistency of
// support on various OS releases.
// The USE_NUMA* vars are set in the Makefile.
#ifdef USE_NUMA

// Use numa policy library?
#ifdef USE_NUMA_POLICY_LIB
#include <numa.h>

// Use <numaif.h> if available.
#elif defined(USE_NUMAIF_H)
#include <numaif.h>

// This is a hack, but some systems are missing <numaif.h>.
#elif !defined(NUMAIF_H)
extern "C" {
    extern long get_mempolicy(int *policy, const unsigned long *nmask,
                              unsigned long maxnode, void *addr, int flags);
    extern long mbind(void *start, unsigned long len, int mode,
                      const unsigned long *nmask, unsigned long maxnode, unsigned flags);
}

// Conservatively don't define MPOL_LOCAL.
#define MPOL_DEFAULT     0
#define MPOL_PREFERRED   1
#define MPOL_BIND        2
#define MPOL_INTERLEAVE  3

#endif
#endif

namespace yask {

    // Helpers for aligned malloc and free.
    extern char* yask_aligned_alloc(std::size_t nbytes);
    struct AlignedDeleter {
        std::size_t _nbytes;

        // Ctor saves size.
        AlignedDeleter(std::size_t nbytes) :
            _nbytes(nbytes) { }

        // Free p.
        void operator()(char* p);
    };

    // Alloc aligned data as a shared ptr.
    template<typename T>
    std::shared_ptr<T> shared_aligned_alloc(size_t nbytes) {
        auto _base = std::shared_ptr<T>(yask_aligned_alloc(nbytes),
                                        AlignedDeleter(nbytes));
        return _base;
    }

    // Helpers for NUMA malloc and free.
    extern char* numa_alloc(std::size_t nbytes, int numa_pref);
    struct NumaDeleter {
        std::size_t _nbytes;
        int _numa_pref;

        // Ctor saves data needed for freeing.
        NumaDeleter(std::size_t nbytes, int numa_pref) :
            _nbytes(nbytes),
            _numa_pref(numa_pref)
        { }

        // Free p.
        void operator()(char* p);
    };

    // Allocate NUMA memory from preferred node.
    template<typename T>
    std::shared_ptr<T> shared_numa_alloc(size_t nbytes, int numa_pref) {
        auto _base = std::shared_ptr<T>(numa_alloc(nbytes, numa_pref),
                                        NumaDeleter(nbytes, numa_pref));
        return _base;
    }

    // Helpers for PMEM malloc and free.
    extern char* pmem_alloc(std::size_t nbytes, int dev);
    struct PmemDeleter {
        std::size_t _nbytes;

        // Ctor saves data needed for freeing.
        PmemDeleter(std::size_t nbytes) :
            _nbytes(nbytes)
        { }

        // Free p.
        void operator()(char* p);
    };

    // Allocate PMEM memory from given device.
    template<typename T>
    std::shared_ptr<T> shared_pmem_alloc(size_t nbytes, int dev) {
        auto _base = std::shared_ptr<T>(pmem_alloc(nbytes, dev),
                                        PmemDeleter(nbytes));
        return _base;
    }

    // Helpers for MPI shm malloc and free.
    extern char* shm_alloc(std::size_t nbytes,
                          const MPI_Comm* shm_comm, MPI_Win* shm_win);
    struct ShmDeleter {
        std::size_t _nbytes;
        const MPI_Comm* _shm_comm;
        MPI_Win* _shm_win;

        // Ctor saves data needed for freeing.
        ShmDeleter(std::size_t nbytes,
                   const MPI_Comm* shm_comm, MPI_Win* shm_win):
            _nbytes(nbytes),
            _shm_comm(shm_comm),
            _shm_win(shm_win)
        { }

        // Free p.
        void operator()(char* p);
    };

    // Allocate MPI shm memory.
    template<typename T>
    std::shared_ptr<T> shared_shm_alloc(size_t nbytes,
                                        const MPI_Comm* shm_comm, MPI_Win* shm_win) {
        auto _base = std::shared_ptr<T>(shm_alloc(nbytes, shm_comm, shm_win),
                                        ShmDeleter(nbytes, shm_comm, shm_win));
        return _base;
    }

}

