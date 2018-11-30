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

#include "yask_stencil.hpp"
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
                FORMAT_AND_THROW_YASK_EXCEPTION("Error: attempt to create grid '" << name << "' with " <<
                                                dims.size() << " dimension names but " << sizes->size() <<
                                                " dimension sizes");
            }
        }

        // First, try to make a grid that matches the layout in
        // the stencil.
        YkGridPtr gp = newStencilGrid(name, dims);

        // No match.
        if (!gp) {

            // Tuple of dims.
            IdxTuple dtup;
            for (auto& d : dims)
                dtup.addDimBack(d, 0);

#if ALLOW_NEW_GRIDS
            // Allow new grid types.
            
            // Check dims.
            int ndims = dims.size();
            int step_posn = -1;      // -1 => not used.
            set<string> seenDims;
            for (int i = 0; i < ndims; i++) {

                // Already used?
                if (seenDims.count(dims[i])) {
                    THROW_YASK_EXCEPTION("Error: cannot create grid '" + name +
                                         "' because dimension '" + dims[i] +
                                         "' is used more than once");
                }

                // Step dim?
                if (dims[i] == _dims->_step_dim) {
                    step_posn = i;
                    if (i > 0) {
                        THROW_YASK_EXCEPTION("Error: cannot create grid '" + name +
                                             "' because step dimension '" + dims[i] +
                                             "' must be first dimension");
                    }
                }

                // Domain dim?
                else if (_dims->_domain_dims.lookup(dims[i])) {
                }

                // Known misc dim?
                else if (_dims->_misc_dims.lookup(dims[i])) {
                }

                // New misc dim?
                else {
                    _dims->_misc_dims.addDimBack(dims[i], 0);
                }
            }
            bool do_wrap = step_posn >= 0;

            // Scalar?
            if (ndims == 0)
                gp = make_shared<YkElemGrid<Layout_0d, false>>(_dims, name, dims, &_opts, &_ostr);

            // Include auto-gen code for all other cases.
#include "yask_grid_code.hpp"

            // Failed.
            if (!gp)
                FORMAT_AND_THROW_YASK_EXCEPTION("Error: cannot create new grid '" << name <<
                                                "' with dimensions '" << dtup.makeDimStr() <<
                                                "'; only up to " << MAX_DIMS <<
                                                " dimensions supported");
#else
            // Don't allow new grid types.
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: cannot create new grid '" << name <<
                                            "' with dimensions '" << dtup.makeDimStr() <<
                                            "'; this list of dimensions is not in any existing grid");
#endif
        }

        // Mark as non-resizable if sizes provided.
        gp->set_fixed_size(got_sizes);

        // Add to context.
        addGrid(gp, false);     // mark as non-output grid.

        // Set sizes as provided or via solution settings.
        if (got_sizes) {
            int ndims = dims.size();
            for (int i = 0; i < ndims; i++) {
                auto& dname = dims[i];

                // Domain size.
                gp->_set_domain_size(i, sizes->at(i));

                // Pads.
                // Set via both 'extra' and 'min'; larger result will be used.
                if (_dims->_domain_dims.lookup(dname)) {
                    gp->set_extra_pad_size(i, _opts->_extra_pad_sizes[dname]);
                    gp->set_min_pad_size(i, _opts->_min_pad_sizes[dname]);
                }

                // Offsets.
                gp->_set_offset(i, 0);
                gp->_set_local_offset(i, 0);
            }
        }

        else
            update_grid_info();

        return gp;
    }
} // namespace yask.
