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

// Test some OpenMP constructs with a hand-coded 2-stage stencil.
// Test hierarchical offloading if enabled.

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>
#include <omp.h>

using namespace std;

// Rounding macros for integer types.
#define CEIL_DIV(numer, denom) (((numer) + (denom) - 1) / (denom))
#define ROUND_UP(n, mult) (CEIL_DIV(n, mult) * (mult))
#define ROUND_DOWN(n, mult) (((n) / (mult)) * (mult))

// Determine whether val is "close enough" to ref.
template<typename T>
bool within_tolerance(T val, T ref, T epsilon) {
    if (val == ref)
        return true;
    bool ok;
    double adiff = fabs(val - ref);
    if (fabs(ref) > 1.0)
        epsilon = fabs(ref * epsilon);
    ok = adiff < epsilon;
    return ok;
}

// Various functions to implement.
#if 0
#define FN_A(a) (3.14 * a)
#define FN_B(b) (b * 1./2.5)
#define FN_C(am, ap, c) ((2. * (am) + (ap)) + 0.5 * (c))
#endif
#if 0
#define FN_A(a) (cos(a) - 2. * sin(a))
#define FN_B(b) (pow(b, 1./2.5))
#define FN_C(am, ap, c) (atan((am) + (ap)) + cbrt(c))
#endif
#if 1
#define FN_A(a) (sincos(a, &sa, &ca), ca - 2. * sa)
#define FN_B(b) (pow(b, 1./2.5))
#define FN_C(am, ap, c) (atan((am) + (ap)) + cbrt(c))
#endif

int main(int argc, char* argv[]) {

    int nthr = 4;               // Number of host threads.
    int nruns = 2;              // Number of runs.
    
    if (argc > 1)
        nthr = atoi(argv[1]);
    if (argc > 2)
        nruns = atoi(argv[2]);
    if (nthr < 1) {
        cerr << "Usage: " << argv[0] << " [num host threads] [num runs]\n";
        return 1;
    }

    #ifdef USE_OFFLOAD
    int ndev = omp_get_num_devices();
    int hostn = omp_get_initial_device();
    int devn = omp_get_default_device();
    cout << ndev << " OMP device(s)\n"
        " host: " << hostn << "\n"
        " dev: " << devn << endl << flush;

    // Dummy OMP offload section to trigger JIT.
    int MOLUE = 42;
    #pragma omp target data device(devn) map(MOLUE)
    { }
    #endif

    long n = 1171; // Array size.
    size_t bsz = n * sizeof(double);
    const int np = 3;
    double* p[np];
    void* devp[np];
    for (int k = 0; k < np; k++) {
        p[k] = new double[n];
        assert(p[k]);
    }
    
    for (int r = 0; r < nruns; r++) {
        cout << "Run " << (r+1) << endl << flush;
    
        for (int k = 0; k < np; k++) {
            for (long i = 0; i < n; i++)
                p[k][i] = double(i);

            // alloc buffers on target and copy to it.
            #ifdef USE_OFFLOAD
            devp[k] = omp_target_alloc(bsz, devn);
            omp_target_associate_ptr(p[k], devp[k], bsz, 0, devn);
            assert(omp_target_is_present(p[k], devn));
            omp_target_memcpy(devp[k], p[k], bsz, 0, 0, devn, hostn);

            // invalidate local copy.
            memset(p[k], 0x55, bsz);
            #endif
        }
        omp_set_max_active_levels(2);

        // Divide n into regions.
        long halo_sz = 2;
        int nreg = 2;
        long n_per_reg = CEIL_DIV(n, nreg);

        // Divide region into blocks.
        int nblks = nthr * 2 - nthr / 2;
        long n_per_blk = CEIL_DIV(n_per_reg, nblks);

        // Calc regions sequentially.
        for ( int reg = 0; reg < nreg; reg++) {
            long rbegin = max(reg * n_per_reg, halo_sz);
            long rend = min((reg+1) * n_per_reg, n - halo_sz);
            if (reg == nreg - 1)
                rend += halo_sz; // parallelogram adj.
            cout << "Region " << reg << " on [" <<
                rbegin << "..." << rend << ")\n" <<
                "Scheduling " << nthr << " host thread(s) on " <<
                nblks << " blk(s) of data...\n" << flush;

            // Calc both stages in this region.
            for ( int stage = 0; stage < 2; stage++) {
                long sbegin = rbegin;
                long send = rend;
                long shift = halo_sz * stage;
                sbegin -= shift;
                sbegin = max(sbegin, halo_sz);
                send -= shift;
                send = min(send, n - halo_sz);
                cout << "Stage " << stage << " on [" <<
                    sbegin << "..." << send << ")\n" << flush;
    
                // OMP on host.
                omp_set_num_threads(nthr);
                #pragma omp parallel for schedule(dynamic,1)
                for ( long i = 0; i < nblks; i++ )
                {
                    long begin = sbegin + i * n_per_blk;
                    long end = min(begin + n_per_blk, send);
                    #pragma omp critical
                    {
                        cout << " Running thread " << omp_get_thread_num() << " on blk [" <<
                            begin << "..." << end << ")\n" << flush;
                    }

                    #if 1
                    // Nested OMP on host.
                    // (Doesn't really do anything useful w/1 thread.)
                    // TODO: divide into >1 thread.
                    omp_set_num_threads(1);
                    #pragma omp parallel proc_bind(spread)
                    #endif
                    {
                        double* A = p[0];
                        double* B = p[1];
                        double* C = p[2];

                        // Calc current stage in current block.
                        // Use OMP on target if enabled.
                        if (stage == 0) {
                            #ifdef USE_OFFLOAD
                            #pragma omp target parallel for device(devn) schedule(static,1)
                            #endif
                            for (long j = begin; j < end; j++)
                            {
                                double sa, ca;
                                A[j] = FN_A(A[j]);
                                B[j] = FN_B(B[j]);
                            }
                        }
                        else {
                            #ifdef USE_OFFLOAD
                            #pragma omp target parallel for device(devn) schedule(static,1)
                            #endif
                            for (long j = begin; j < end; j++)
                            {
                                C[j] = FN_C(A[j - halo_sz], A[j + halo_sz], C[j]);
                            }
                        }
                    }
                }
            } // stages.
        } // regions.
    
        // Copy results back and free mem on dev.
        #ifdef USE_OFFLOAD
        for (int k = 0; k < np; k++) {
            omp_target_memcpy(p[k], devp[k], bsz, 0, 0, hostn, devn);
            omp_target_disassociate_ptr(p[k], devn);
            omp_target_free(devp[k], devn);
        }
        #endif
        
        // Check.
        long nbad = 0;
        for (long i = halo_sz; i < n - halo_sz; i++) {
            double orig = double(i);
            double sa, ca;
            double expected = FN_A(orig);
            if (!within_tolerance(p[0][i], expected, 1e-6)) {
                cout << "A[" << i << "] = " << p[0][i] << "; expecting " << expected << endl;
                nbad++;
            }
            expected = FN_B(orig);
            if (!within_tolerance(p[1][i], expected, 1e-6)) {
                cout << "B[" << i << "] = " << p[1][i] << "; expecting " << expected << endl;
                nbad++;
            }
            expected = FN_C(p[0][i-halo_sz], p[0][i+halo_sz], orig);
            if (!within_tolerance(p[2][i], expected, 1e-6)) {
                cout << "C[" << i << "] = " << p[2][i] << "; expecting " << expected << endl;
                nbad++;
            }
        }

        cout << "Num errors: " << nbad << endl;
        if (nbad)
            return nbad;
    }
    for (int k = 0; k < np; k++)
        delete[] p[k];

    return 0;
}
