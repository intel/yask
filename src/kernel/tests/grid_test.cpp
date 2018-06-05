/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

// Test the YASK grids.
// This must be compiled with a kernel lib containing 'x', 'y', and 'z' dims.

// enable assert().
#define DEBUG

#include "yask_stencil.hpp"
using namespace std;
using namespace yask;

// Auto-generated stencil code that extends base types.
#define DEFINE_CONTEXT
#include "yask_stencil_code.hpp"

int main(int argc, char** argv) {

    // Bootstrap factory from kernel API.
    yk_factory kfac;

    // Set up the environment (e.g., OpenMP & MPI).
    auto kenv = kfac.new_env();

    // Object containing data and parameters for stencil eval.
    // TODO: do everything through API without cast to StencilContext.
    auto ksoln = kfac.new_solution(kenv);
    auto context = dynamic_pointer_cast<StencilContext>(ksoln);
    assert(context.get());
    ostream& os = context->set_ostr();
    ostream* osp = &os;
    auto settings = context->get_settings();

    // Problem dimensions.
    auto dims = YASK_STENCIL_CONTEXT::new_dims();
    os << "Folding: " << dims->_fold_pts.makeDimValStr(" * ") << endl;
    os << "Domain dims: " << dims->_domain_dims.makeDimStr() << endl;

    // Set domain size.
    int i = 0;
    for (auto dname : ksoln->get_domain_dim_names()) {
        ksoln->set_rank_domain_size(dname, 9 + i++);
    }

    // 0D test.
    {
        os << "0-D test...\n";
        GridDimNames gdims;
        string name = "test grid";
        YkGridPtr g0 = make_shared<YkElemGrid<Layout_0d, false>>(dims, name, gdims, &settings, &osp);
        g0->alloc_storage();
        os << g0->make_info_string() << endl;
        YkGridPtr g1 = make_shared<YkElemGrid<Layout_0d, false>>(dims, name, gdims, &settings, &osp);
        g1->alloc_storage();
        os << g1->make_info_string() << endl;

        double val = 3.14;
        os << "Testing with " << val << endl;
        g0->set_element(val, {});
        g1->set_element(val, {});
        auto v0 = g0->get_element({});
        auto v1 = g1->get_element({});
        assert(v0 == v1);
    }

    // 3D test.
    {
        os << "3-D test...\n";
        GridDimNames gdims = {"x", "y", "z"};
        string name = "test grid";
        YkGridPtr g3 = make_shared<YkElemGrid<Layout_321, false>>(dims, name, gdims, &settings, &osp);
        YkGridPtr g3f = make_shared<YkVecGrid<Layout_123, false, VLEN_X, VLEN_Y, VLEN_Z>>(dims, name, gdims, &settings, &osp);
        int i = 0;
        int min_pad = 3;
        for (auto dname : gdims) {
            g3->_set_domain_size(dname, ksoln->get_rank_domain_size(dname));
            g3->set_min_pad_size(dname, min_pad + i);
            g3f->_set_domain_size(dname, ksoln->get_rank_domain_size(dname));
            g3f->set_min_pad_size(dname, min_pad + i);
            i++;
        }
        g3->alloc_storage();
        os << g3->make_info_string() << endl;
        g3f->alloc_storage();
        os << g3f->make_info_string() << endl;

        os << "Copying seq of vals\n";
        g3->set_all_elements_in_seq(1.0);
        auto sizes = g3->get_allocs();
        sizes.visitAllPointsInParallel([&](const IdxTuple& pt,
                                           size_t idx) {
                IdxTuple pt2 = pt;
                for (auto dname : gdims)
                    pt2[dname] += g3->get_first_rank_alloc_index(dname);
                Indices ipt(pt2);
                auto val = g3->readElem(ipt, 0, __LINE__);
                g3f->writeElem(val, ipt, 0, __LINE__);
                return true;
            });
        os << "Checking seq of vals\n";
        sizes.visitAllPoints([&](const IdxTuple& pt,
                                 size_t idx) {
                IdxTuple pt2 = pt;
                for (auto dname : gdims)
                    pt2[dname] += g3->get_first_rank_alloc_index(dname);
                Indices ipt(pt2);
                ipt.addConst(-min_pad);
                auto val = g3->readElem(ipt, 0, __LINE__);
                auto valf = g3f->readElem(ipt, 0, __LINE__);
                assert(val == valf);
                return true;
            });
    }

    os << "End of YASK grid test.\n";
    return 0;
}
