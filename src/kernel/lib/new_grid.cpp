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

#include "yask.hpp"
using namespace std;

namespace yask {

    // Make a new grid.
    YkGridPtr StencilContext::newGrid(const std::string& name,
                                      const GridDimNames& dims,
                                      const GridDimSizes* sizes) {

        // Check parameters.
        bool got_sizes = sizes != NULL;
        if (got_sizes) {
            if (dims.size() != sizes->size()) {
                THROW_YASK_EXCEPTION("Error: attempt to create grid '" << name << "' with " <<
                                     dims.size() << " dimension names but " << sizes->size() <<
                                     " dimension sizes");
            }
        }

        // First, try to make a grid that matches the layout in
        // the stencil.
        YkGridPtr gp = newStencilGrid(name, dims, _opts);

        // If there was no match, use default layout.
        if (!gp) {

            // Check dims.
            int ndims = dims.size();
            int step_posn = -1;      // -1 => not used.
            set<string> seenDims;
            for (int i = 0; i < ndims; i++) {

                // Already used?
                if (seenDims.count(dims[i])) {
                    THROW_YASK_EXCEPTION("Error: cannot create grid '" << name <<
                                         "': dimension '" << dims[i] << "' used more than once");
                }
            
                // Step dim?
                if (dims[i] == _dims->_step_dim) {
                    step_posn = i;
                    if (i > 0) {
                        THROW_YASK_EXCEPTION("Error: cannot create grid '" << name <<
                                             "' because step dimension '" << dims[i] <<
                                             "' must be first dimension");
                    }
                }
            }
            bool do_wrap = step_posn >= 0;

            // Scalar?
            if (ndims == 0)
                gp = make_shared<YkElemGrid<Layout_0d, false>>(_dims, name, dims, _opts, &_ostr);
            
            // Include auto-gen code for all other cases.
#include "yask_grid_code.hpp"
            
            if (!gp) {
                THROW_YASK_EXCEPTION("Error in new_grid: cannot create grid '" << name <<
                                     "' with " << ndims << " dimensions; only up to " << MAX_DIMS <<
                                     " dimensions supported");
            }
        }

        // Mark as non-resizable if sizes provided.
        gp->set_fixed_size(got_sizes);

        // Add to context.
        addGrid(gp, false);     // mark as non-output grid.

        // Set sizes as provided or via solution settings.
        if (got_sizes) {
            int ndims = dims.size();
            for (int i = 0; i < ndims; i++) {
                gp->_set_domain_size(i, sizes->at(i));
            }
        }

        else
            update_grids();

        return gp;
    }
} // namespace yask.
