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

// Tracing directly to 'cout' when 'state' is not avail.
#ifdef TRACE
#define OFFLOAD_MSG(msg) DEBUG_MSG0(std::cout, "YASK: " << msg)
#else
#define OFFLOAD_MSG(msg) ((void)0)
#endif
#define OFFLOAD_MAP_ALLOC(p, num) do {                          \
        auto _nb = sizeof(*p) * num;                                     \
        OFFLOAD_MSG("#pragma omp target enter data map(alloc: " <<        \
                    (void*)p << "[0:" << num << "]); " << _nb << " bytes"); \
        YPRAGMA(omp target enter data map(alloc: p[0:num]))           \
            OFFLOAD_MSG(_nb << " device bytes allocated");      \
            } while(0)
#define OFFLOAD_MAP_FREE(p, num) do {                                   \
        auto _nb = sizeof(*p) * num;                                     \
        OFFLOAD_MSG("#pragma omp target exit data map(delete: " <<     \
                    (void*)p << "[0:" << num << "]); " << _nb << " bytes"); \
        YPRAGMA(omp target exit data map(delete: p[0:num]))            \
            OFFLOAD_MSG(_nb << " device bytes freed");      \
            } while(0)

// Conditional tracing using 'state'.
#define OFFLOAD_MAP_ALLOC2(state, p, num) do {               \
        auto _nb = sizeof(*p) * num;                                     \
        TRACE_MSG1(state, "#pragma omp target enter data map(alloc: " << \
                   (void*)p << "[0:" << num << "]); " << _nb << " bytes"); \
        YPRAGMA(omp target enter data map(alloc: p[0:num]))           \
            TRACE_MSG1(state, _nb << " device bytes allocated"); \
            } while(0)
#define OFFLOAD_UPDATE_TO2(state, p, num) do {                \
        auto _nb = sizeof(*p) * num;                           \
         TRACE_MSG1(state, "#pragma omp target update to(" << \
                   (void*)p << "[0:" << num << "]); " << _nb << " bytes"); \
         YPRAGMA(omp target update to(p[0:num]))              \
            TRACE_MSG1(state, _nb << " bytes updated to device"); \
            } while(0)
#define OFFLOAD_UPDATE_FROM2(state, p, num) do {                \
        auto _nb = sizeof(*p) * num;                             \
         TRACE_MSG1(state, "#pragma omp target update from(" << \
                  (void*)p << "[0:" << num << "]); " << _nb << " bytes"); \
         YPRAGMA(omp target update from(p[0:num]))              \
            TRACE_MSG1(state, _nb << " bytes updated from device"); \
            } while(0)
#define OFFLOAD_MAP_FREE2(state, p, num) do {                           \
        auto _nb = sizeof(*p) * num;                                     \
        TRACE_MSG1(state, "#pragma omp target exit data map(delete: " << \
                  (void*)p << "[0:" << num << "]); " << _nb << " bytes"); \
        YPRAGMA(omp target exit data map(delete: p))                  \
            TRACE_MSG1(state, _nb << " device bytes freed"); \
            } while(0)

#else
#define OFFLOAD_MAP_ALLOC(p, nbytes) ((void)0)
#define OFFLOAD_MAP_FREE(p, nbytes) ((void)0)
#define OFFLOAD_MAP_ALLOC2(state, p, num) ((void)0)
#define OFFLOAD_UPDATE_TO2(state, p, num) ((void)0)
#define OFFLOAD_UPDATE_FROM2(state, p, num) ((void)0)
#define OFFLOAD_MAP_FREE2(state, p, num) ((void)0)
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
        T* _dp = 0;                 // val of ptr on device.
        T** _dpp = 0;                // loc of ptr on device.
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
                T** pp = &_p;

                // Temp vars to capture device data.
                T* dp;
                T** dpp;
                int devn;

                // Set pointer on device and copy back to host.
                #pragma omp target map(from: dp,dpp,devn)
                {
                    _p = p;
                    dp = p;
                    dpp = pp;
                    devn = omp_get_device_num();
                }

                // Update values;
                _devn = devn;
                _dp = dp;
                _dpp = dpp;

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
            TRACE_MSG1(state, "omp: ptr to " << _hp << " on host at " << (void*)&_hp <<
                       (done ? "" : "already") << " set to " << _dp <<
                       " on device " << _devn << " at " << (void*)_dpp <<
                       ((_dpp == 0) ? " *******" : ""));
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
