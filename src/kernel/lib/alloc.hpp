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

    // Customer allocator for objects that may need to be offloaded.
    template <class T>
    struct yask_allocator {
        typedef T value_type;
        yask_allocator() noexcept {}

        template<typename U>
        yask_allocator(const yask_allocator<U>& other) throw() {};
    
        T* allocate (std::size_t n) {
            return static_cast<T*>(offload_alloc_host(n * sizeof(T)));
        }
        void deallocate (T* p, std::size_t n) {
            offload_free_host(p);
        }
    };

    // Generic deleter.
    struct DeleterBase {
        std::size_t _nbytes;

        // Ctor saves size & device ptr.
        DeleterBase(std::size_t nbytes) :
            _nbytes(nbytes) { }

        // Free device mem.
        void free_dev_mem(char* hostp);
    };
    
    // Helpers for aligned malloc and free.
    extern char* yask_aligned_alloc(std::size_t nbytes);
    struct AlignedDeleter : public DeleterBase {

        AlignedDeleter(std::size_t nbytes) :
            DeleterBase(nbytes) { }

        // Free p.
        void operator()(char* p);
    };

    // Alloc aligned data as a shared ptr.
    template<typename T>
    std::shared_ptr<T> shared_aligned_alloc(size_t nbytes) {

        // Alloc mem.
        char* cp = yask_aligned_alloc(nbytes);

        // Map alloc to device.
        offload_map_alloc(cp, nbytes);

        // Make shared ptr.
        auto _base = std::shared_ptr<T>(cp, AlignedDeleter(nbytes));
        return _base;
    }

    // Helpers for NUMA malloc and free.
    extern char* numa_alloc(std::size_t nbytes, int numa_pref);
    struct NumaDeleter : public DeleterBase {
        int _numa_pref;

        // Ctor saves data needed for freeing.
        NumaDeleter(std::size_t nbytes, int numa_pref) :
            DeleterBase(nbytes),
            _numa_pref(numa_pref)
        { }

        // Free p.
        void operator()(char* p);
    };

    // Allocate NUMA memory from preferred node.
    template<typename T>
    std::shared_ptr<T> shared_numa_alloc(size_t nbytes, int numa_pref) {

        // Alloc mem.
        char* cp = numa_alloc(nbytes, numa_pref);

        // Map alloc to device.
        offload_map_alloc(cp, nbytes);

        // Make shared ptr.
        auto _base = std::shared_ptr<T>(cp, NumaDeleter(nbytes, numa_pref));
        return _base;
    }

    // Helpers for PMEM malloc and free.
    extern char* pmem_alloc(std::size_t nbytes, int dev);
    struct PmemDeleter : public DeleterBase {

        // Ctor saves data needed for freeing.
        PmemDeleter(std::size_t nbytes) :
            DeleterBase(nbytes)
        { }

        // Free p.
        void operator()(char* p);
    };

    // Allocate PMEM memory from given device.
    template<typename T>
    std::shared_ptr<T> shared_pmem_alloc(size_t nbytes, int pmem_dev) {

        // Alloc mem.
        char* cp = pmem_alloc(nbytes, pmem_dev);

        // Map alloc to device.
        offload_map_alloc(cp, nbytes);

        // Make shared ptr.
        auto _base = std::shared_ptr<T>(cp, PmemDeleter(nbytes));
        return _base;
    }

    // Helpers for MPI shm malloc and free.
    extern char* shm_alloc(std::size_t nbytes,
                          const MPI_Comm* shm_comm, MPI_Win* shm_win);
    struct ShmDeleter : DeleterBase {
        const MPI_Comm* _shm_comm;
        MPI_Win* _shm_win;

        // Ctor saves data needed for freeing.
        ShmDeleter(std::size_t nbytes,
                   const MPI_Comm* shm_comm, MPI_Win* shm_win):
            DeleterBase(nbytes),
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

        // Alloc mem.
        char* cp = shm_alloc(nbytes, shm_comm, shm_win);

        // Map alloc to device.
        // Shm on device not currently supported.
        offload_map_alloc(cp, nbytes);

        // Make shared ptr.
        auto _base = std::shared_ptr<T>(cp, ShmDeleter(nbytes, shm_comm, shm_win));
        return _base;
    }

    // Key for allocating memory.
    // Pair is mem type and sequence number.
    typedef std::pair<int, int> AllocKey;

    // Data for allocating mem.
    struct AllocData {
        std::shared_ptr<char> ptr;
        size_t nbytes = 0;
        int nvars = 0;
    };

    // Map from alloc key to data.
    typedef std::map<AllocKey, AllocData> AllocMap;
    
}

