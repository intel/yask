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

// This file contains declarations and code needed for OpenMP offload.
// Various versions of code is provided depending on whether offloading and,
// if offloading, whether USM is used.
// Also see OMP_* macros in yask.hpp.

#pragma once

namespace yask {

    // Definitions to use when offloading WITH unified shared memory.
    // Not offloading:       0
    // Offloading w/USM:     1
    // Offloading w/o USM:   0
    #ifdef USE_OFFLOAD_USM

    // Allocate space for 'num' 'T' objects on host.
    inline void* offload_alloc_host(size_t nbytes) {
        auto devn = KernelEnv::_omp_devn;

        #ifdef INTEL_OMP
        TRACE_MSG("allocating " << make_byte_str(nbytes) << " shared, specifying OMP dev " << devn);
        void* p = omp_target_alloc_shared(nbytes, devn);
        #else
        TRACE_MSG("allocating " << make_byte_str(nbytes) << " on host");
        void* p = yask_aligned_alloc(nbytes, devn);
        #endif
        if (!p)
            THROW_YASK_EXCEPTION("error: cannot allocate " + make_byte_str(nbytes) + " on host");
        return p;
    }

    // Free memory allocated with offload_alloc_host().
    inline void offload_free_host(void* p) {
        auto devn = KernelEnv::_omp_devn;
        #ifdef INTEL_OMP
        omp_target_free(p, devn); // frees after omp_target_alloc_*().
        #else
        free(p);
        #endif
    }
    #endif
    
    // Definitions to use when offloading but NOT using unified shared memory.
    // Not offloading:       0
    // Offloading w/USM:     0
    // Offloading w/o USM:   1
    #ifdef USE_OFFLOAD_NO_USM

    // Allocate space for 'num' 'T' objects on host.
    inline void* offload_alloc_host(size_t nbytes) {
        auto devn = KernelEnv::_omp_devn;

        #ifdef INTEL_OMP
        TRACE_MSG("allocating " << make_byte_str(nbytes) << " on host, specifying OMP dev " << devn);
        void* p = omp_target_alloc_host(nbytes, devn);
        #else
        TRACE_MSG("allocating " << make_byte_str(nbytes) << " on host");
        void* p = yask_aligned_alloc(nbytes, devn);
        #endif
        if (!p)
            THROW_YASK_EXCEPTION("error: cannot allocate " + make_byte_str(nbytes) + " on host");
        return p;
    }

    // Free memory allocated with offload_alloc_host().
    inline void offload_free_host(void* p) {
        auto devn = KernelEnv::_omp_devn;
        #ifdef INTEL_OMP
        omp_target_free(p, devn); // frees after omp_target_alloc_*().
        #else
        free(p);
        #endif
    }
    
    // Get device addr from any mapped host addr 'hostp'.
    // If 'must_be_mapped' is true, assertion will fail if not mapped.
    // If 'enable_trace' is true, host and device addrs will be printed when
    // tracing. Make sure 'enable_trace' is false if called from 'TRACE_MSG'
    // to avoid deadlock.
    template <typename T>
    T* get_dev_ptr(const T* hostp,
                   bool must_be_mapped = true,
                   bool enable_trace = true) {
        if (!hostp)
            return NULL;
        auto devn = KernelEnv::_omp_devn;
        bool is_present = omp_target_is_present((void*)hostp, devn);
        if (!is_present && !must_be_mapped)
            return NULL;
        assert(is_present);

        void* mp = omp_get_mapped_ptr(hostp, devn);
        T* dp = (T*)mp;
        
        if (enable_trace) {
            TRACE_MSG("host addr == " << (void*)hostp <<
                      "; dev addr == " << (void*)dp);
        }
        return dp;
    }
        
    // Allocate space for 'num' 'T' objects on offload device.
    // Map 'hostp' to allocated mem.
    // Return device ptr.
    template <typename T>
    void* offload_map_alloc(T* hostp, size_t num) {
        if (KernelEnv::_use_offload) {
            assert(hostp);
            auto nb = sizeof(T) * num;
            auto devn = KernelEnv::_omp_devn;

            TRACE_MSG("allocating " << make_byte_str(nb) << " on OMP dev " << devn);
            #ifdef INTEL_OMP
            void* devp = omp_target_alloc_device(nb, devn);
            #else
            void* devp = omp_target_alloc(nb, devn);
            #endif
            if (!devp)
                THROW_YASK_EXCEPTION("error: cannot allocate " + make_byte_str(nb) + " on OMP device");

            TRACE_MSG("mapping " << (void*)hostp << " to " << devp << " on OMP dev " << devn);
            auto res = omp_target_associate_ptr(hostp, devp, nb, 0, devn);
            if (res)
                THROW_YASK_EXCEPTION("error: cannot map OMP device ptr");
            assert(omp_target_is_present(hostp, devn));
            assert(get_dev_ptr(hostp) == devp);

            TRACE_MSG("done allocating and mapping");
            return devp;
        }
        return hostp;
    }

    // Unmap 'hostp' from 'devp' on offload device.
    // Free space for 'num' 'T' objects on offload device.
    // Not checking KernelEnv::_use_offload because this is often
    // called from destructor which may be after _use_offload has
    // been changed since offload_map_alloc() was called.
    template <typename T>
    void _offload_map_free(void* devp, T* hostp, size_t num) {
        if (!devp || !hostp)
            return;
        auto nb = sizeof(T) * num;
        auto devn = KernelEnv::_omp_devn;
        TRACE_MSG("unmapping " << (void*)hostp << " from " << devp << " on OMP dev " << devn);
        assert(omp_target_is_present(hostp, devn));
        assert(get_dev_ptr(hostp) == devp);
        auto res = omp_target_disassociate_ptr(hostp, devn);
        if (res)
            THROW_YASK_EXCEPTION("error: cannot unmap OMP device ptr");
        TRACE_MSG("freeing " << make_byte_str(nb) << " on OMP dev " << devn);
        omp_target_free(devp, devn);
        TRACE_MSG("done unmapping and freeing");
    }

    // Unmap 'hostp' on offload device.
    // Automatically looks up dev ptr.
    // Free space for 'num' 'T' objects on offload device.
    template <typename T>
    void offload_map_free(T* hostp, size_t num) {
        void* devp = get_dev_ptr(hostp, false);
        _offload_map_free(devp, hostp, num);
    }

    // Copy data to device.
    template <typename T>
    void _offload_copy_to_device(void* devp, T* hostp, size_t num) {
        if (KernelEnv::_use_offload) {
            assert(hostp);
            assert(devp);
            auto nb = sizeof(T) * num;
            auto devn = KernelEnv::_omp_devn;
            TRACE_MSG("copying " << make_byte_str(nb) << " to OMP dev " << devn);
            assert(omp_target_is_present(hostp, devn));
            auto res = omp_target_memcpy(devp, hostp, // dst, src.
                                         nb, 0, 0,
                                         devn, KernelEnv::_omp_hostn);
            TRACE_MSG("done copying to OMP dev");
        }
    }
    template <typename T>
    void offload_copy_to_device(T* hostp, size_t num) {
        if (KernelEnv::_use_offload) {
            void* devp = get_dev_ptr(hostp);
            _offload_copy_to_device(devp, hostp, num);
        }
    }

    // Copy data from device.
    template <typename T>
    void _offload_copy_from_device(void* devp, T* hostp, size_t num) {
        if (KernelEnv::_use_offload) {
            assert(hostp);
            assert(devp);
            auto nb = sizeof(T) * num;
            auto devn = KernelEnv::_omp_devn;
            TRACE_MSG("copying " << make_byte_str(nb) << " from OMP dev " << devn);
            assert(omp_target_is_present(hostp, devn));
            auto res = omp_target_memcpy(hostp, devp, // dst, src.
                                         nb, 0, 0,
                                         KernelEnv::_omp_hostn, devn);
            TRACE_MSG("done copying from OMP dev");
        }
    }
    template <typename T>
    void offload_copy_from_device(T* hostp, size_t num) {
        if (KernelEnv::_use_offload) {
            void* devp = get_dev_ptr(hostp);
            _offload_copy_from_device(devp, hostp, num);
        }
    }
    #endif
    
    // Definitions to use when offloading with unified shared memory OR not offloading.
    // Not offloading:       1
    // Offloading w/USM:     1
    // Offloading w/o USM:   0
    #ifndef USE_OFFLOAD_NO_USM
    template <typename T>
    T* get_dev_ptr(T* hostp,
                   bool must_be_mapped = true,
                   bool enable_trace = true) { return hostp; }
    template <typename T>
    void* offload_map_alloc(T* hostp, size_t num) { return hostp; }
    template <typename T>
    void _offload_map_free(void* devp, T* hostp, size_t num) { }
    template <typename T>
    void offload_map_free(T* hostp, size_t num) { }
    template <typename T>
    void _offload_copy_to_device(void* devp, T* hostp, size_t num) { }
    template <typename T>
    void offload_copy_to_device(T* hostp, size_t num) { }
    template <typename T>
    void _offload_copy_from_device(void* devp, T* hostp, size_t num) { }
    template <typename T>
    void offload_copy_from_device(T* hostp, size_t num) { }
    #endif

    // Definitions to use when not offloading.
    // Not offloading:       1
    // Offloading w/USM:     0
    // Offloading w/o USM:   0
    #ifndef USE_OFFLOAD

    inline void* offload_alloc_host(size_t nbytes) {
        return malloc(nbytes);
    }
    inline void offload_free_host(void* p) {
        if (p)
            free(p);
    }

    #endif

    // Non-typed versions.
    inline void* get_dev_ptr(const void* hostp,
                             bool must_be_mapped = true,
                             bool enable_trace = true) {
        return (void*)get_dev_ptr((char*) hostp, must_be_mapped, enable_trace);
    }
    inline void _offload_copy_to_device(void* devp, void* hostp, size_t nbytes) {
        _offload_copy_to_device(devp, (char*)hostp, nbytes);
    }
    inline void offload_copy_to_device(void* hostp, size_t nbytes) {
        offload_copy_to_device((char*)hostp, nbytes);
    }
    inline void _offload_copy_from_device(void* devp, void* hostp, size_t nbytes) {
        _offload_copy_from_device(devp, (char*)hostp, nbytes);
    }
    inline void offload_copy_from_device(void* hostp, size_t nbytes) {
        offload_copy_from_device((char*)hostp, nbytes);
    }
    inline void* offload_map_alloc(void* hostp, size_t nbytes) {
        return offload_map_alloc((char*)hostp, nbytes);
    }

    // Type to track and sync pointers on target device.
    // A synced pointer has these characteristics:
    // - Pointer exists on host & dev.
    // - Object containing pointer is mapped (associated) on dev.
    // - On host copy of ptr:
    //   - Addr pointed to (value of '_p') is mapped on dev (value of '_dp').
    // - On dev copy of ptr:
    //   - Addr pointed to is mapped dev addr.
    template <typename T>
    class synced_ptr {
    private:
        T* _p = 0;                  // ptr to data; used on host and device.

        // Additional data when offloading without unified addresses.
        #ifdef USE_OFFLOAD_NO_USM
        T* _dp = 0;                 // val of ptr on device.
        #endif

    protected:
        // Sync this pointer.
        // To properly sync pointer on device, '*this' and '*_p' must
        // already be mapped to device mem.
        void _sync() {
            #ifdef USE_OFFLOAD_NO_USM
            if (KernelEnv::_use_offload) {
                auto devn = KernelEnv::_omp_devn;
                TRACE_MSG("omp: sync'ing ptr to " << _p << " on host...");
                
                // Value of ptr on dev.
                _dp = yask::get_dev_ptr(_p);

                // Addr of ptr on host & dev.
                T** pp = &_p;
                T** dpp = yask::get_dev_ptr(pp);

                // Set pointer on device to val of ptr on dev.
                _offload_copy_to_device(dpp, &_dp, 1);

                TRACE_MSG("omp: sync'd ptr to " << _p << " on host at " << (void*)pp <<
                          " -> " << _dp << " on device " << devn << " at " << (void*)dpp <<
                          ((dpp == 0) ? " *******" : ""));
            }
            else
                _dp = _p;
            #endif
        }
        
    public:
        synced_ptr(T* p) : _p(p) { }
        synced_ptr() : synced_ptr<T>(0) { }

        // Accessors.
        T* get() { return _p; }
        const T* get() const { return _p; }
        operator T*() { return _p; }
        operator const T*() { return _p; }
        T* operator->() { return _p; }
        const T* operator->() const { return _p; }
        T& operator*() { return *_p; }
        const T& operator*() const { return *_p; }
        T& operator[](size_t i) { return _p[i]; }
        const T& operator[](size_t i) const { return _p[i]; }
        T& operator[](long i) { return _p[i]; }
        const T& operator[](long i) const { return _p[i]; }
        T& operator[](int i) { return _p[i]; }
        const T& operator[](int i) const { return _p[i]; }

        // Get pointer on device or NULL if not yet resolved.
        #ifdef USE_OFFLOAD_NO_USM
        const T* get_dev_ptr() const { return _dp; }

        // Get pointer on host if not offloading or with USM.
        #else
        const T* get_dev_ptr() const { return _p; }
        #endif
        
        // Set pointer value.
        void operator=(T* p) {
            _p = p;
            #ifdef USE_OFFLOAD_NO_USM
            _dp = 0;
            #endif
        }

        // Sync pointer on device.
        inline void sync() {
            _sync();
        }
    
        // Set to given value and sync.
        inline void set_and_sync(T* p) {
            operator=(p);
            _sync();
        }

    };

    // Host/device coherency state machine.
    class Coherency {
    public:

        // Coherency states.
        enum coh_state { host_mod, // Host copy modified; device copy out-of-sync.
                         dev_mod, // Device copy modified; host copy out-of-sync.
                         in_sync, // Host and device have same data.
                         num_states };

        // Current state.
        inline coh_state get_state() const {
            return _state;
        }

        #ifdef USE_OFFLOAD_NO_USM
        
        // Set state directly (not recommended).
        void _force_state(coh_state state) {
            TRACE_MSG("coherency state forced to " << state);
            _state = state;
        }

    protected:
        coh_state _state = host_mod;

    public:

        // Boolean queries.
        inline bool need_to_update_host() const {
            return _state == dev_mod;
        }
        inline bool need_to_update_dev() const {
            return _state == host_mod;
        }

        // State-transition events.
        // Functions return new state.
        
        // Call when host copy is modified, but dev copy is not.
        coh_state mod_host() {
            if (_state == dev_mod)
                THROW_YASK_EXCEPTION("internal error: "
                                     "host copy modified, but device copy was newer");
            _state = host_mod;
            return _state;
        }

        // Call when dev copy is modified, but host copy is not.
        coh_state mod_dev() {
            if (_state == host_mod)
                THROW_YASK_EXCEPTION("internal error: "
                                     "device copy modified, but host copy was newer");
            _state = dev_mod;
            return _state;
        }

        // Call when both dev and host copies are modified w/the same changes.
        coh_state mod_both() {
            if (_state == dev_mod)
                THROW_YASK_EXCEPTION("internal error: "
                                     "host copy modified, but device copy was newer");
            if (_state == host_mod)
                THROW_YASK_EXCEPTION("internal error: "
                                     "device copy modified, but host copy was newer");
            assert(_state == in_sync);
            return _state;
        }

        // Call when host data is copied to dev.
        coh_state host_copied_to_dev() {
            if (_state == dev_mod)
                THROW_YASK_EXCEPTION("internal error: "
                                     "host data copied to dev, but device copy was newer");
            _state = in_sync;
            return _state;
        }

        // Call when dev data is copied to host.
        coh_state dev_copied_to_host() {
            if (_state == host_mod)
                THROW_YASK_EXCEPTION("internal error: "
                                     "device data copied to host, but host copy was newer");
            _state = in_sync;
            return _state;
        }

        #else
        // Stubs for no offload or USM.

        // Set state directly (not recommended).
        void _force_state(coh_state state) {
            TRACE_MSG("attempt to force coherency state to " << state <<
                      " ignored because of offload build state");
        }
        
    protected:
        coh_state _state = in_sync;

    public:

        // Boolean queries.
        inline bool need_to_update_host() const {
            return false;
        }
        inline bool need_to_update_dev() const {
            return false;
        }

        // State-transition event stubs.
        coh_state mod_host() {
            return _state;
        }
        coh_state mod_dev() {
            return _state;
        }
        coh_state mod_both() {
            return _state;
        }
        coh_state host_copied_to_dev() {
            return _state;
        }
        coh_state dev_copied_to_host() {
            return _state;
        }
        
        #endif
    };
    
} // namespace yask.
