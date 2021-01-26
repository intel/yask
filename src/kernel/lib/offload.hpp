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

// This file contains declarations and code needed for OpenMP offload.
// Active when USE_OFFLOAD macro is set, otherwise stubs are provided.

#pragma once

namespace yask {

    // Get device ptr from any host ptr.
    template <typename T>
    T* get_dev_ptr(T* hostp) {
        #ifdef USE_OFFLOAD_NO_USM
        if (KernelEnv::_use_offload) {
            auto devn = KernelEnv::_omp_devn;
            if (hostp)
                assert(omp_target_is_present(hostp, devn));
            else
                return NULL;

            // Temp var to capture device ptr.
            T* dp = 0;

            // Get pointer on device.
            #pragma omp target data device(devn) use_device_ptr(hostp)
            {
                dp = hostp;
            }
            TRACE_MSG("host ptr == " << (void*)hostp <<
                      "; dev ptr == " << (void*)dp);
            return dp;
        }
        #endif
        return hostp;
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
        // If 'force' is 'true', always sync; otherwise sync only
        // if it appears not to have been done since changed.
        // Returns 'true' if updated.
        void _sync() {
            #ifdef USE_OFFLOAD_NO_USM
            if (KernelEnv::_use_offload) {

                auto devn = KernelEnv::_omp_devn;
                TRACE_MSG("omp: sync'ing ptr to " << _p << " on host at " << (void*)&_p << "...");

                // Make sure the pointer itself is in mapped mem.
                assert(omp_target_is_present(&_p, devn));

                // Value on host; converted to target ptr in 'omp target'.
                T* p = _p;
                if (p)
                    assert(omp_target_is_present(p, devn));

                // Temp var to capture device ptr.
                T* dp;
                
                // With tracing.
                #ifdef TRACE

                // Addr of host ptr.
                T** pp = &_p;

                // Temp vars for dev ptr.
                #ifdef CHECK
                T** dpp1 = yask::get_dev_ptr(pp);
                #endif
                T** dpp2 = 0;

                // Set pointer on device and copy back to host.
                #pragma omp target device(devn) map(from: dp,dpp2)
                {
                    _p = p;
                    dp = p;
                    dpp2 = pp;
                }

                // Update values.
                TRACE_MSG("omp: sync'd ptr to " << _p << " on host at " << (void*)&_p <<
                          " -> " << _dp << " on device " << devn << " at " << (void*)dpp2 <<
                          ((dpp2 == 0) ? " *******" : ""));
                assert(dpp1 == dpp2);

                // Without tracing.
                #else

                // Set pointer on device and copy back to host.
                #pragma omp target device(devn) map(from: dp)
                {
                    _p = p;
                    dp = p;
                }
                #endif

                // Update values.
                _dp = dp;
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

        #ifdef USE_OFFLOAD_NO_USM
        const T* get_dev_ptr() const { return _dp; }
        #else
        const T* get_dev_ptr() const { return _p; }
        #endif
        
        // Set pointer value.
        void operator=(T* p) { _p = p; }

        // Sync pointer on device.
        inline void sync() {
            _sync();
        }
    
        // Set to given value and sync.
        inline void set_and_sync(T* p) {
            _p = p;
            _sync();
        }

    };
    
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
    template <typename T>
    void offload_map_free(void* devp, T* hostp, size_t num) {
        if (KernelEnv::_use_offload) {
            assert(hostp);
            assert(devp);
            auto nb = sizeof(T) * num;
            auto devn = KernelEnv::_omp_devn;
            TRACE_MSG("unmapping " << (void*)hostp << " from " << devp << " on OMP dev " << devn);
            assert(omp_target_is_present(hostp, devn));
            auto res = omp_target_disassociate_ptr(hostp, devn);
            if (res)
                THROW_YASK_EXCEPTION("error: cannot unmap OMP device ptr");
            TRACE_MSG("freeing " << make_byte_str(nb) << " on OMP dev " << devn);
            omp_target_free(devp, devn);
            TRACE_MSG("done unmapping and freeing");
        }
    }

    // Unmap 'hostp' on offload device.
    // Automatically looks up dev ptr.
    // Free space for 'num' 'T' objects on offload device.
    template <typename T>
    void offload_map_free(T* hostp, size_t num) {
        if (KernelEnv::_use_offload) {
            void* devp = get_dev_ptr(hostp);
            offload_map_free(devp, hostp, num);
        }
    }

    // Copy data to device.
    template <typename T>
    void offload_copy_to_device(void* devp, T* hostp, size_t num) {
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
            offload_copy_to_device(devp, hostp, num);
        }
    }

    // Copy data from device.
    template <typename T>
    void offload_copy_from_device(void* devp, T* hostp, size_t num) {
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
            offload_copy_from_device(devp, hostp, num);
        }
    }

    #else

    // Stub functions when not offloading or when using USM.
    template <typename T>
    void* offload_map_alloc(T* hostp, size_t num) { return hostp; }
    template <typename T>
    void offload_map_free(void* devp, T* hostp, size_t num) { }
    template <typename T>
    void offload_map_free(T* hostp, size_t num) { }
    template <typename T>
    void offload_copy_to_device(void* devp, T* hostp, size_t num) { }
    template <typename T>
    void offload_copy_to_device(T* hostp, size_t num) { }
    template <typename T>
    void offload_copy_from_device(void* devp, T* hostp, size_t num) { }
    template <typename T>
    void offload_copy_from_device(T* hostp, size_t num) { }
    #endif

    // Non-typed versions.
    inline void offload_copy_to_device(void* devp, void* hostp, size_t nbytes) {
        offload_copy_to_device(devp, (char*)hostp, nbytes);
    }
    inline void offload_copy_to_device(void* hostp, size_t nbytes) {
        offload_copy_to_device((char*)hostp, nbytes);
    }
    inline void offload_copy_from_device(void* devp, void* hostp, size_t nbytes) {
        offload_copy_from_device(devp, (char*)hostp, nbytes);
    }
    inline void offload_copy_from_device(void* hostp, size_t nbytes) {
        offload_copy_from_device((char*)hostp, nbytes);
    }
    
} // namespace yask.
