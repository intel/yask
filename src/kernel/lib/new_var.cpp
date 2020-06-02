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

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Make a new var.
    YkVarPtr StencilContext::new_var(const std::string& name,
                                      const VarDimNames& gdims,
                                      const VarDimSizes* sizes) {
        STATE_VARS(this);

        // Check parameters.
        bool got_sizes = sizes != NULL;
        if (got_sizes) {
            if (gdims.size() != sizes->size()) {
                FORMAT_AND_THROW_YASK_EXCEPTION("Error: attempt to create var '" << name << "' with " <<
                                                gdims.size() << " dimension names but " << sizes->size() <<
                                                " dimension sizes");
            }
        }

        // First, try to make a var that matches the layout in
        // the stencil.
        VarBasePtr gp = new_stencil_var(name, gdims);

        // No match.
        if (!gp) {

            // Tuple of dims.
            IdxTuple dtup;
            for (auto& d : gdims)
                dtup.add_dim_back(d, 0);

#if ALLOW_NEW_VARS
            // Allow new var types.

            // Check dims.
            int ndims = gdims.size();
            bool step_used = false;
            set<string> seen_dims;
            for (int i = 0; i < ndims; i++) {
                auto& gdim = gdims[i];

                // Already used?
                if (seen_dims.count(gdim)) {
                    THROW_YASK_EXCEPTION("Error: cannot create var '" + name +
                                         "' because dimension '" + gdim +
                                         "' is used more than once");
                }

                // Step dim?
                if (gdim == step_dim) {
                    step_used = true;
                    if (i != step_posn) {
                        THROW_YASK_EXCEPTION("Error: cannot create var '" + name +
                                             "' because step dimension '" + gdim +
                                             "' is not first dimension");
                    }
                }

                // Domain dim?
                else if (domain_dims.lookup(gdim)) {
                }

                // Known misc dim?
                else if (misc_dims.lookup(gdim)) {
                }

                // New misc dim?
                else {
                    misc_dims.add_dim_back(gdim, 0);
                }
            }

            // Scalar?
            if (ndims == 0)
                gp = make_shared<YkElemVar<Layout_0d, false>>(*this, name, gdims);

            // Include auto-gen code for all other cases.
#include "yask_var_code.hpp"

            // Failed.
            if (!gp)
                FORMAT_AND_THROW_YASK_EXCEPTION("Error: cannot create new var '" << name <<
                                                "' with dimensions '" << dtup.make_dim_str() <<
                                                "'; only up to " << MAX_DIMS <<
                                                " dimensions supported");
#else
            // Don't allow new var types.
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: cannot create new var '" << name <<
                                            "' with dimensions '" << dtup.make_dim_str() <<
                                            "'; this list of dimensions is not in any existing var");
#endif
        }

        // Mark as non-resizable if sizes provided.
        gp->set_fixed_size(got_sizes);

        // Mark as created via API.
        gp->set_user_var(true);

        // Wrap with a Yk var.
        YkVarPtr ygp = make_shared<YkVarImpl>(gp);

        // Add to context.
        add_var(ygp, false, false);     // mark as non-orig, non-output var.

        // Set sizes as provided.
        if (got_sizes) {
            int ndims = gdims.size();
            for (int i = 0; i < ndims; i++) {
                auto& gdim = gdims[i];

                // Domain size.
                ygp->_set_domain_size(i, sizes->at(i));

                // Pads.
                // Set via both 'extra' and 'min'; larger result will be used.
                if (domain_dims.lookup(gdim)) {
                    ygp->set_extra_pad_size(i, opts->_extra_pad_sizes[gdim]);
                    ygp->set_min_pad_size(i, opts->_min_pad_sizes[gdim]);
                }

                // Offsets.
                ygp->_set_rank_offset(i, 0);
                ygp->_set_local_offset(i, 0);
            }
        }

        // Set sizes based on solution settings.
        else
            update_var_info(false);

        return ygp;
    }
} // namespace yask.
