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
// Active when USE_OFFLOAD macro is set.

#pragma once

namespace yask {

    // Type to track and sync pointers on target device.
    template <typename T>
    class synced_ptr {
    private:
        T* _p = 0;                  // ptr to data.

        // Additional data when offloading.
        #ifdef USE_OFFLOAD
        T* _hp = 0;                 // latest sync'd ptr on host.
        T* _dp = 0;                 // val of ptr on device.

        // Additional data when printing debug info.
        #ifdef TRACE
        T** _dpp = 0;               // loc of ptr on device.
        #endif
        #endif

    protected:
        // Sync this pointer.
        // To properly sync pointer on device, '*this' and '*_p' must
        // already be mapped to device mem.
        // If 'force' is 'true', always sync; otherwise sync only
        // if it appears not to have been done since changed.
        // Returns 'true' if updated.
        bool _sync(bool force) {
            #ifdef USE_OFFLOAD
            if (force || _hp != _p) {

                // Value on host; converted to target ptr in 'omp target'.
                T* p = _p;
                auto devn = KernelEnv::_omp_devn;
                if (p)
                    assert(omp_target_is_present(p, devn));

                // Temp var to capture device data.
                T* dp;
                
                // With tracing.
                #ifdef TRACE
                T** pp = &_p;

                // Temp vars to capture device data.
                T** dpp;

                // Set pointer on device and copy back to host.
                #pragma omp target device(devn) map(from: dp,dpp)
                {
                    _p = p;
                    dp = p;
                    dpp = pp;
                }

                // Update values.
                _dpp = dpp;

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
                _hp = p;
                return true;
            }
            #else
            _hp = _p;
            _dp = _p;
            #endif
            return false;
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
        const T* get_dev_ptr() const { return _dp; }

        // Set pointer value.
        void operator=(T* p) { _p = p; }

        // Sync pointer on device.
        bool sync() {
            bool done = _sync(true);
            #if defined(USE_OFFLOAD) && defined(TRACE)
            auto devn = KernelEnv::_omp_devn;
            TRACE_MSG("omp: ptr to " << _hp << " on host at " << (void*)&_hp <<
                       (done ? "" : "already") << " set to " << _dp <<
                       " on device " << devn << " at " << (void*)_dpp <<
                       ((_dpp == 0) ? " *******" : ""));
            #endif
            return done;
        }
    
        // Set to given value and sync.
        bool set_and_sync(T* p) {
            _p = p;
            return sync();
        }

        // Get device ptr from any host ptr.
        static T* get_dev_ptr(T* hostp) {
            #ifdef USE_OFFLOAD
            auto devn = KernelEnv::_omp_devn;
            if (hostp)
                assert(omp_target_is_present(hostp, devn));
            else
                return NULL;

            // Temp var to capture device data.
            T* dp = 0;

            // Get pointer on device.
            #pragma omp target device(devn) map(from: dp)
            {
                dp = hostp;
            }
            return dp;

            #else
            return hostp;
            #endif
        }
        
    };
    
    #ifdef USE_OFFLOAD

    // Allocate space for 'num' 'T' objects on offload device.
    // Map 'hostp' to allocated mem.
    // Return device ptr.
    template <typename T>
    void* offload_map_alloc(T* hostp, size_t num) {
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
        TRACE_MSG("done allocating and mapping");
        return devp;
    }

    // Unmap 'hostp' from 'devp' on offload device.
    // Free space for 'num' 'T' objects on offload device.
    template <typename T>
    void offload_map_free(void* devp, T* hostp, size_t num) {
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

    // Unmap 'hostp' on offload device.
    // Free space for 'num' 'T' objects on offload device.
    template <typename T>
    void offload_map_free(T* hostp, size_t num) {
        void* devp = synced_ptr<T>::get_dev_ptr(hostp);
        offload_map_free(devp, hostp, num);
    }
    
    #define OFFLOAD_UPDATE_TO(p, num) do {                      \
            auto _nb = sizeof(*p) * num;                                \
            TRACE_MSG("#pragma omp target update to(" <<        \
                       (void*)p << "[0:" << num << "]); " << _nb << " bytes"); \
            YPRAGMA(omp target update to(p[0:num]))                     \
                TRACE_MSG(_nb << " bytes updated to device");   \
        } while(0)
    #define OFFLOAD_UPDATE_FROM(p, num) do {                    \
            auto _nb = sizeof(*p) * num;                                \
            TRACE_MSG("#pragma omp target update from(" <<      \
                       (void*)p << "[0:" << num << "]); " << _nb << " bytes"); \
            YPRAGMA(omp target update from(p[0:num]))                   \
                TRACE_MSG(_nb << " bytes updated from device"); \
        } while(0)

    #else
    template <typename T>
    void* offload_map_alloc(T* hostp, size_t num) { return NULL; }
    template <typename T>
    void offload_map_free(void* devp, T* hostp, size_t num) { }
    template <typename T>
    void offload_map_free(T* hostp, size_t num) { }

    #define OFFLOAD_UPDATE_TO(p, num) ((void)0)
    #define OFFLOAD_UPDATE_FROM(p, num) ((void)0)

    #endif

} // namespace yask.
