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

// Test some OpenMP constructs with a hand-coded 2-stage, 1-D stencil.
// Test hierarchical offloading if enabled.

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>
#include <omp.h>

using namespace std;

// Macro defaults
#ifndef FN
#define FN 3
#endif
#ifndef USE_HOST_OMP
#define USE_HOST_OMP 1
#endif
#ifndef USE_NESTED_HOST_OMP
#define USE_NESTED_HOST_OMP 1
#endif

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

// Divide 'num' equally into 'nparts'.
// Returns the *cumulative* sizes of the 0-'n'th parts,
// if 0 <= 'n' < 'nparts' and 0 if n < 0.
// The <0 case is handy for calculating the initial
// starting point when passing 'n'-1 and 'n'==0.
template <typename T>
inline T div_equally_cumu_size_n(T num, T nparts, T n) {
    if (n < 0)
        return 0;
    T p = (num / nparts) * (n + 1);
    T rem = num % nparts;
    p += (n < rem) ? (n + 1) : rem;
    return p;
}

// Various functions to implement.
// FN_A and FN_B are 1-element computations from A and B, resp.
// FN_C is a stencil from 2 points in A and 1 point in C.
// Enable one.

// Using simple non-library math.
#if FN==1
#define FN_A(a) (3. * a)
#define FN_B(b) (b * 2. + 1.)
#define FN_C(am, ap, c) ((2. * (am) + (ap)) + 4. * (c))

// Using math lib fns.
#elif FN==2
#define FN_A(a) (cos(a) - 2. * sin(a))
#define FN_B(b) (pow(b, 1./2.5))
#define FN_C(am, ap, c) (atan((am) + (ap)) + cbrt(c))

// Same as FN=2, but using sincos() instead of separate sin() and cos().
#elif FN==3
#define FN_A(a) (sincos(a, &sa, &ca), ca - 2. * sa)
#define CHK_A(a) (cos(a) - 2. * sin(a))
#define FN_B(b) (pow(b, 1./2.5))
#define FN_C(am, ap, c) (atan((am) + (ap)) + cbrt(c))
#else
#error "FN macro value not valid"
#endif

// Checking fns are same as perf fns if not specifically defined.
#ifndef CHK_A
#define CHK_A(a) FN_A(a)
#endif
#ifndef CHK_B
#define CHK_B(b) FN_B(b)
#endif
#ifndef CHK_C
#define CHK_C(am, ap, c) FN_C(am, ap, c)
#endif

// Initial value.
#define INIT_FN(ai, i) (5. + (2. * ai) + i)

int main(int argc, char* argv[]) {

    cout << "Outer host OpenMP parallel region is "
        #if USE_HOST_OMP==0
        "NOT "
        #endif
        "enabled.\n";
    cout << "Inner host OpenMP parallel region is "
        #if USE_NESTED_HOST_OMP==0
        "NOT "
        #endif
        "enabled.\n";
    
    // Cmd-line settings.
    long pn = 1024;             // Problem size.
    int nreg = 2;               // Number of regions.
    int nthr = 4;               // Number of host threads.
    int nruns = 2;              // Number of runs.
    bool help = false;
    #if USE_HOST_OMP==0
    nthr = 1;
    #endif

    int ai = 0;
    ai++;
    if (argc > ai) {
        if (string(argv[ai]) == "-h")
            help = true;
        else
            nthr = atoi(argv[ai]);
    }
    ai++;
    if (argc > ai)
        pn = atol(argv[ai]);
    ai++;
    if (argc > ai)
        nreg = atoi(argv[ai]);
    int nblks = nthr * 2 - nthr / 2; // Blocks per region.
    ai++;
    if (argc > ai)
        nblks = atoi(argv[ai]);
    ai++;
    if (argc > ai)
        nruns = atoi(argv[ai]);

    cout << "Current settings: "
        " num-host-threads=" << nthr <<
        ", prob-size=" << pn <<
        ", num-regions=" << nreg <<
        ", num-blocks=" << nblks <<
        ", num-runs=" << nruns <<
        endl;
    if (help || nthr < 1 || pn < 1 || nreg < 1 || nblks < 1 || nruns < 1) {
        cerr << "Usage: " << argv[0] << " [num host threads] [problem size]"
            " [num regions] [num blocks per region] [num runs]\n";
        return 1;
    }

    #ifdef USE_OFFLOAD
    int ndev = omp_get_num_devices();
    int hostn = omp_get_initial_device();
    int devn = omp_get_default_device();
    cout << ndev << " OMP device(s)\n"
        " host: " << hostn << "\n"
        " dev: " << devn << endl << flush;
    if (ndev == 0) {
        cerr << "No OMP devices available.\n";
        exit(1);
    }

    // Dummy OMP offload section to trigger JIT.
    int MOLUE = 42;
    #pragma omp target data device(devn) map(MOLUE)
    { }

    #else
    cout << "NOT testing on offload device.\n";
    #endif

    constexpr long halo_sz = 2;
    long an = pn + 2 * halo_sz; // array size.
    size_t bsz = an * sizeof(double); // array size in bytes.
    constexpr int na = 3; // num of arrays.
    double* p[na]; // host ptrs.
    #ifdef USE_OFFLOAD
    void* devp[na]; // dev ptrs.
    #endif
    for (int k = 0; k < na; k++) {
        p[k] = new double[an];
        assert(p[k]);
    }
    
    for (int r = 0; r < nruns; r++) {
        long pbegin = halo_sz;
        long pend = pbegin + pn;
        cout << "Run " << (r+1) << endl <<
            " Working range = [" << pbegin << "..." <<
            pend << ")\n" << flush;

        // Init data.
        for (int k = 0; k < na; k++) {
            for (long i = 0; i < an; i++)
                p[k][i] = INIT_FN(k, i);

            // alloc buffers on target and copy to it.
            #ifdef USE_OFFLOAD
            devp[k] = omp_target_alloc(bsz, devn);
            assert(devp[k]);
            int res = omp_target_associate_ptr(p[k], devp[k], bsz, 0, devn);
            assert(res == 0);
            assert(omp_target_is_present(p[k], devn));
            res = omp_target_memcpy(devp[k], p[k], bsz, 0, 0, devn, hostn);
            assert(res == 0);

            // invalidate local copy to catch data transfer problems.
            memset(p[k], 0x55, bsz);
            #endif
        }
        omp_set_max_active_levels(2);

        // Divide pn into regions.
        long n_per_reg = CEIL_DIV(pn, nreg);

        // Divide region into blocks.
        long n_per_blk = CEIL_DIV(n_per_reg, nblks);
        cout << " Using " << nreg << " region(s) of " << n_per_reg <<
            " point(s) each comprising " << nblks << " block(s) of " <<
            n_per_blk << " point(s) each.\n";

        // Calc regions sequentially.
        for (int reg = 0; reg < nreg; reg++) {
            long rbegin = reg * n_per_reg + halo_sz;
            long rend = (reg+1) * n_per_reg + halo_sz; // Don't need to trim yet.
            cout << " Region " << reg << " on [" <<
                rbegin << "..." << rend << ")\n";

            // Calc both stages in this region.
            for (int stage = 0; stage < 2; stage++) {
                long sbegin = rbegin;
                long send = rend;

                // Shift after 1st stage to handle dependencies.
                long shift = halo_sz * stage;
                sbegin -= shift;
                if (reg < nreg-1)
                    send -= shift;
                else
                    send = an - halo_sz;

                // Trim to array size w/o halos.
                sbegin = max(sbegin, pbegin);
                send = min(send, pend);
                cout << " Stage " << stage << " on [" <<
                    sbegin << "..." << send << ")\n" <<
                    "   Scheduling " << nthr << " host thread(s) on " <<
                    nblks << " blk(s) of data...\n" << flush;
    
                // OMP on host.
                #if USE_HOST_OMP==1
                omp_set_num_threads(nthr);
                #pragma omp parallel for schedule(dynamic,1)
                #endif
                for ( long i = 0; i < nblks; i++ )
                {
                    long begin = sbegin + i * n_per_blk;
                    long end = min(begin + n_per_blk, send);
                    if (i == nblks-1)
                        end = send;
                    int tn0 = omp_get_thread_num();
                    #if USE_HOST_OMP==1
                    #pragma omp critical
                    #endif
                    {
                        cout << "   Running thread " << tn0 << " on blk [" <<
                            begin << "..." << end << ")\n" << flush;
                    }

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

                    // 2nd stage, 1st run.
                    else if (r == 0) {
                        #ifdef USE_OFFLOAD
                        #pragma omp target teams distribute parallel for device(devn)
                        #elif USE_NESTED_HOST_OMP==1
                        #pragma omp parallel for num_threads(nthr)
                        #endif
                        for (long j = begin; j < end; j++)
                        {
                            if (j == begin) {
                                int ntm1 = omp_get_num_teams();
                                int nt1 = omp_get_num_threads();
                                int nthr = ntm1 * nt1;
                                printf("     Running thread %i w/%i teams and %i threads/team: %i threads\n",
                                       tn0, ntm1, nt1, nthr);
                                #ifndef USE_OFFLOAD
                                fflush(stdout);
                                #endif
                            }
                            C[j] = FN_C(A[j - halo_sz], A[j + halo_sz], C[j]);
                        }
                    }

                    // 2nd stage, >1st run.
                    else {

                        // Use a manually-generated parallel loop after the 1st run.
                        #ifdef USE_OFFLOAD
                        #pragma omp critical
                        {
                            cout << "    Launching kernel from host thread " << tn0 <<
                                "\n" << flush;
                        }
                        #pragma omp target teams device(devn)
                        #pragma omp parallel
                        #elif USE_NESTED_HOST_OMP==1
                        #pragma omp parallel num_threads(nthr)
                        #endif
                        {
                            int ntm1 = omp_get_num_teams();
                            int nt1 = omp_get_num_threads();
                            int tmn1 = omp_get_team_num();
                            int tn1 = omp_get_thread_num();
                            long nthr = ntm1 * nt1;
                            long tn = (tmn1 * nt1 + tn1);
                            long niters = end - begin;

                            // Calculate begin and end points for this thread.
                            long tbegin = div_equally_cumu_size_n(niters, nthr, tn - 1) + begin;
                            long tend = div_equally_cumu_size_n(niters, nthr, tn) + begin;
                            #ifdef SHOW_THREADS
                            #pragma omp critical
                            {
                                printf("     Running thread %i w/team %i/%i & thread %i/%i (%li/%li) on [%li...%li)\n",
                                       tn0, tmn1, ntm1, tn1, nt1, tn, nthr, tbegin, tend);
                                #ifndef USE_OFFLOAD
                                fflush(stdout);
                                #endif
                            }
                            #endif
                            
                            for (long j = tbegin; j < tend; j++)
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
        for (int k = 0; k < na; k++) {
            omp_target_memcpy(p[k], devp[k], bsz, 0, 0, hostn, devn);
            omp_target_disassociate_ptr(p[k], devn);
            omp_target_free(devp[k], devn);
        }
        #endif
        
        // Check.
        
        long cbegin = halo_sz;
        long cend = cbegin + pn;
        cout << "Checking [" << cbegin << "..." << cend << ")\n" << flush;
        long nbad = 0;
        for (long i = cbegin; i < cend; i++) {
            double sa, ca;
            double expected = CHK_A(INIT_FN(0, i));
            if (!within_tolerance(p[0][i], expected, 1e-6)) {
                cout << "A[" << i << "] = " << p[0][i] << "; expecting " << expected << endl;
                nbad++;
            }
            expected = CHK_B(INIT_FN(1, i));
            if (!within_tolerance(p[1][i], expected, 1e-6)) {
                cout << "B[" << i << "] = " << p[1][i] << "; expecting " << expected << endl;
                nbad++;
            }
            expected = CHK_C(p[0][i-halo_sz], p[0][i+halo_sz], INIT_FN(2, i));
            if (!within_tolerance(p[2][i], expected, 1e-6)) {
                cout << "C[" << i << "] = " << p[2][i] << "; expecting " << expected << endl;
                nbad++;
            }
        }

        cout << "Run " << (r+1) << ": num errors: " << nbad << endl;
        if (nbad)
            return nbad;

    } // Runs.
    for (int k = 0; k < na; k++)
        delete[] p[k];

    return 0;
}
