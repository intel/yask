/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

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

// Test the YASK vars.
// This must be compiled with a kernel containing 'x', 'y', and 'z' dims.

// Enable extra checking.
#define DEBUG_LAYOUT

#include "yask_stencil.hpp"
using namespace std;
using namespace yask;

// Auto-generated stencil code that extends base types.
#include YSTR2(YK_SOLUTION_FILE)

void run_tests(int argc, char* argv[]) {

    // Bootstrap objects from kernel API.
    yk_factory kfac;
    yask_output_factory yof;

    // Set up the environment (e.g., OpenMP & MPI).
    yk_env::set_debug_output(yof.new_stdout_output());
    yk_env::set_trace_enabled(true);
    auto kenv = kfac.new_env();

    // Object containing data and parameters for stencil eval.
    // TODO: do everything through API without cast to StencilContext.
    auto ksoln = kfac.new_solution(kenv);
    ksoln->apply_command_line_options(argc, argv);
    auto ebytes = ksoln->get_element_bytes();
    auto context = dynamic_pointer_cast<StencilContext>(ksoln);
    assert(context.get());
    ostream& os = kenv->get_debug_output()->get_ostream();
    auto settings = context->get_actl_opts();
    os << "Solution: " << ksoln->get_name();
    os << "Precision: " << ksoln->get_element_bytes();
    assert(ksoln->get_real_bytes() == 8);

    // Problem dimensions.
    auto dims = YASK_STENCIL_SOLUTION::new_dims();
    os << "Folding: " << dims->_fold_pts.make_dim_val_str(" * ") << endl;
    os << "Domain dims: " << dims->_domain_dims.make_dim_str() << endl;

    // Set domain size.
    int i = 0;
    for (auto dname : ksoln->get_domain_dim_names()) {
        ksoln->set_rank_domain_size(dname, 3 + i * 2);
        i++;
    }

    // 0D test.
    {
        os << "0-D test...\n";
        VarDimNames gdims;

        // Make two scalar vars.
        auto gb0 = make_shared<YkElemVar<Layout_0d, false>>(*context, "var1", gdims);
        YkVarPtr g0 = make_shared<YkVarImpl>(gb0);
        g0->alloc_storage();
        os << gb0->make_info_string() << endl;
        auto gb1 = make_shared<YkElemVar<Layout_0d, false>>(*context, "var2", gdims);
        YkVarPtr g1 = make_shared<YkVarImpl>(gb1);
        g1->alloc_storage();
        os << gb1->make_info_string() << endl;

        double val = 3.14;
        os << "Testing with " << val << endl;
        g0->set_element(val, {}, true);
        g1->set_element(val, {}, true);
        auto v0 = g0->get_element({});
        auto v1 = g1->get_element({});
        assert(v0 == v1);
        os << "Exiting 0-D test\n";
    }

    // 3D test.
    {
        os << "3-D test...\n";
        VarDimNames gdims = {"x", "y", "z"};

        // Make two 3D vars.
        // An element-storage var.
        auto gb3 = make_shared<YkElemVar<Layout_321, false>>(*context, "var3", gdims);
        YkVarPtr g3 = make_shared<YkVarImpl>(gb3);

        // A vec-storage var (folded).
        auto gb3f = make_shared<YkVecVar<Layout_123, false, VLEN_X, VLEN_Y, VLEN_Z>>(*context, "var4", gdims);
        YkVarPtr g3f = make_shared<YkVarImpl>(gb3f);

        int i = 0;
        int min_pad = 1;
        for (auto dname : gdims) {
            g3->_set_domain_size(dname, ksoln->get_rank_domain_size(dname));
            g3->set_min_pad_size(dname, min_pad + i);
            g3f->_set_domain_size(dname, ksoln->get_rank_domain_size(dname));
            g3f->set_min_pad_size(dname, min_pad + i);
            i++;
        }
        g3->alloc_storage();
        g3f->alloc_storage();
        auto sizes = gb3->get_allocs();
        auto sizesf = gb3f->get_allocs();

        // gf3 may be larger because of folding.
        assert(sizes.product() <= sizesf.product());
        if (VLEN_X * VLEN_Y * VLEN_Z == 1)
            assert(sizes.product() == sizesf.product());

        os << "Setting vals in " << gb3->get_name() << endl;
        gb3->set_all_elements_in_seq(1.0);

        IdxTuple first, last;
        for (auto dname : gdims) {
            first.add_dim_back(dname, g3->get_first_local_index(dname));
            last.add_dim_back(dname, g3->get_last_local_index(dname));
        }
        Indices firsti(first), lasti(last);

        // Buffer for copying.
        size_t nelem = g3f->get_num_storage_elements();
        size_t bsz = nelem * sizeof(double);
        double* buf = new double[nelem];
        offload_map_alloc(buf, bsz);

        // Buffer with zeros.
        double* buf0 = new double[nelem];
        memset(buf0, 0, bsz);
                          
        bool done = false;
        for (int testn = 0; !done; testn++) {
            os << "\n*** test " << testn << endl;

            // Fill w/bad values.
            os << "initializing data...\n";
            gb3f->set_all_elements_in_seq(-1.0);

            os << "copying seq of vals to " << gb3f->get_name() << endl;
            switch (testn) {

            case 0: {
                os << " element-by-element in parallel on host...\n";
                sizes.visit_all_points_in_parallel
                    ( [&](const IdxTuple& pt, size_t idx, int thread) {
                          IdxTuple pt2 = pt;
                          for (auto dname : gdims)
                              pt2[dname] += first[dname];
                          Indices ipt(pt2);
                          auto val = gb3->read_elem(ipt, 0, __LINE__);
                          gb3f->write_elem(val, ipt, 0, __LINE__);
                          return true;
                      });
                break;
            }

            case 1: {
                os << " by slice on host...\n";
                auto n = gb3->get_elements_in_slice(buf, nelem, firsti, lasti, false);
                assert(n);
                gb3f->set_elements_in_slice(buf, nelem, firsti, lasti, false);
                break;
            }
                
                #ifdef USE_OFFLOAD
            case 2: {
                os << " by slice then copy to/from device...\n";

                // Same as test 1.
                gb3->get_elements_in_slice(buf, nelem, firsti, lasti, false);
                gb3f->set_elements_in_slice(buf, nelem, firsti, lasti, false);

                // Copy data to device; invalidate host data; copy data back.
                gb3f->copy_data_to_device();

                #ifndef USE_OFFLOAD_USM
                gb3f->set_elements_in_slice(buf0, nelem, firsti, lasti, false);
                gb3f->get_coh()._force_state(Coherency::dev_mod);
                #endif

                gb3f->copy_data_from_device();
                
                break;
            }

            case 3: {
                os << " by slice on device...\n";
                assert(VLEN_X * VLEN_Y * VLEN_Z == 1);

                // Copy from var to buffer on host.
                gb3->get_elements_in_slice(buf, nelem, firsti, lasti, false);
                gb3f->set_elements_in_slice(buf, nelem, firsti, lasti, false);
                gb3f->get_vecs_in_slice(buf, firsti, lasti, false);

                #ifndef USE_OFFLOAD_USM
                gb3f->set_elements_in_slice(buf0, nelem, firsti, lasti, false);
                gb3f->get_coh()._force_state(Coherency::dev_mod);
                #endif

                // Copy buffer to dev.
                offload_copy_to_device(buf, nelem);

                // Copy from buffer to var on dev.
                gb3f->set_vecs_in_slice(buf, firsti, lasti, true);

                // Copy var back to host.
                gb3f->copy_data_from_device();
                break;
            }
                #endif

            default:
                done = true;
            }

            if (!done) {
                os << "Checking vals...\n";
                idx_t nbad = 0, npts = 0;
                idx_t max_bad = 50;
                sizes.visit_all_points
                    ([&](const IdxTuple& pt, size_t idx, int thread) {
                         npts++;
                         IdxTuple pt2 = pt;
                         for (auto dname : gdims)
                             pt2[dname] += first[dname];
                         Indices ipt(pt2);
                         ipt.add_const(-min_pad);
                         auto val = gb3->read_elem(ipt, 0, __LINE__);
                         auto valf = gb3f->read_elem(ipt, 0, __LINE__);
                         if (val != valf) {
                             if (nbad < max_bad)
                                 os << "*** error: value at " << ipt.make_val_str() <<
                                     " is " << valf << "; expected " << val << endl;
                             else if (nbad == max_bad)
                                 os << "Additional errors not printed.\n";
                             nbad++;
                         }
                         return true;
                     });
                os << " done checking: " << nbad << "/" << npts << " are incorrect.\n";
                if (nbad)
                    exit(1);
            }
        }
        delete[] buf;
        offload_map_free(buf, bsz);
        os << "Exiting 3-D test\n";
    }
    kenv->finalize();
}

int main(int argc, char* argv[]) {
    run_tests(argc, argv);
    cout << "End of YASK var test.\n";
    return 0;
}
