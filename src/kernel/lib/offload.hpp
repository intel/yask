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

#ifdef USE_OFFLOAD
#define OFFLOAD_MSG(msg) std::cout << "YASK: omp: " << msg << "\n" << std::flush
#define OFFLOAD_PRAGMA(...) do {                \
        _Pragma(#__VA_ARGS__)                   \
            } while(0)
#define OFFLOAD_TRACED(state, ...) do {         \
        TRACE_MSG1(state, #__VA_ARGS__);        \
        __VA_ARGS__;                            \
    } while(0)
#define OFFLOAD_PRAGMA_TRACED(state, ...) do {               \
        TRACE_MSG1(state, "omp: #pragma " #__VA_ARGS__);     \
        _Pragma(#__VA_ARGS__)                                \
            } while(0)
#define OFFLOAD_MAP_ALLOC(p, nbytes) do {                       \
        char* cp = static_cast<char*>(p);                       \
        size_t nb = size_t(nbytes);                             \
        OFFLOAD_MSG("#pragma omp target enter data map(alloc: " <<        \
                     (void*)cp << "[0:" << nb << "])");                 \
        _Pragma("omp target enter data map(alloc: cp[0:nb])")           \
            OFFLOAD_MSG("YASK: " << nb << " bytes allocated");  \
            } while(0)
#define OFFLOAD_MAP_FREE(p) do {                                \
        char* cp = static_cast<char*>(p);                       \
        OFFLOAD_MSG("#pragma omp target exit data map(release: " <<   \
                     (void*)cp << ")");                             \
        _Pragma("omp target exit data map(release: cp)")        \
            } while(0)
#else
#define OFFLOAD_PRAGMA(...) ((void)0)
#define OFFLOAD_TRACED(state, ...) ((void)0)
#define OFFLOAD_PRAGMA_TRACED(state, ...) ((void)0)
#define OFFLOAD_MAP_ALLOC(p, nbytes)
#define OFFLOAD_MAP_FREE(p)
#endif

namespace yask {

    // Type to track and sync pointers on target device.
    template <typename T>
    class synced_ptr {
    private:
        T* _p = 0;                  // ptr to data.

        // Additional data when offloading.
        #ifdef USE_OFFLOAD
        T* _hp = 0;                 // latest sync'd ptr on host.

        // Additional data when printing debug info.
        #ifdef TRACE
        T* _dp = 0;                 // ptr on device.
        int _devn = 0;              // device num.
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

                // With tracing.
                #ifdef TRACE

                // Temp vars to capture device data.
                T* dp;
                int devn;

                // Set pointer on device and copy back to host.
                #pragma omp target map(from: dp,devn)
                {
                    _p = p;
                    dp = p;
                    devn = omp_get_device_num();
                }

                // Update values;
                _devn = devn;
                _dp = dp;

                // Without tracing.
                #else

                // Set pointer on device.
                #pragma omp target
                {
                    _p = p;
                }
            
                #endif

                // Update copy of last-updated pointer.
                _hp = p;
                return true;
            }
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

        #if defined(USE_OFFLOAD) && defined(TRACE)
        const T& get_dev_ptr() const { return _dp; }
        int get_dev_num() const { return _devn; }
        #endif
    
        // Set pointer value.
        void operator=(T* p) { _p = p; }

        // Sync and print trace message using settings in KernelState.
        bool sync(const KernelState* state) {
            bool done = _sync(true);
            #if defined(USE_OFFLOAD) && defined(TRACE)
            if (done)
                TRACE_MSG1(state, "omp: host ptr to " << _hp << " set to " << _dp << " on device " << _devn);
            else
                TRACE_MSG1(state, "omp: host ptr to " << _hp << " already set to " << _dp << " on device " << _devn);
            #endif
            return done;
        }
    
        // Set to given value and sync.
        bool set_and_sync(const KernelState* state, T* p) {
            _p = p;
            return sync(state);
        }
        
    };

} // namespace yask.
