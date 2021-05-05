/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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
// Active when USE_OFFLOAD macro is set, otherwise stubs are provided.
// Also see OMP_* macros in yask.hpp.

#pragma once

namespace yask {

    // Get device addr from any mapped host addr 'hostp'.
    // If 'must_be_mapped' is true, assertion will fail if not mapped.
    // If 'enable_trace' is true, host and device addrs will be printed when
    // tracing. Make sure 'enable_trace' is false if called from 'TRACE_MSG'
    // to avoid deadlock.
    template <typename T>
    T* get_dev_ptr(T* hostp, bool must_be_mapped = true, bool enable_trace = true) {
        #ifdef USE_OFFLOAD_NO_USM
        if (!hostp)
            return NULL;
        auto devn = KernelEnv::_omp_devn;
        bool is_present = omp_target_is_present((void*)hostp, devn);
        if (!is_present && !must_be_mapped)
            return NULL;
        assert(is_present);

        // Temp var to capture device ptr.
        T* dp = 0;

        // Get pointer on device via OMP offload map.  The "target data"
        // pragma maps variables to the device data environment, but doesn't
        // execute on the device.
        #pragma omp target data device(devn) use_device_ptr(hostp)
        {
            dp = hostp;
        }
        if (enable_trace) {
            TRACE_MSG("host addr == " << (void*)hostp <<
                      "; dev addr == " << (void*)dp);
        }
        return dp;
        #endif
        return hostp;
    }
        
    #ifdef USE_OFFLOAD_NO_USM

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
            void* devp = omp_target_alloc(nb, devn);
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

    #else

    // Stub functions when not offloading or when using USM.
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

    // Non-typed versions.
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

    // Type to track and sync pointers on target device.
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
    
} // namespace yask.
