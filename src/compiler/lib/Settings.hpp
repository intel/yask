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

///////// Classes for Settings & Dimensions ////////////

#pragma once

#include "Var.hpp"

using namespace std;

namespace yask {

    // Settings for the compiler.
    // May be provided via cmd-line or API.
    class CompilerSettings {
    public:
        string _target;             // format type.
        int _elem_bytes = 4;    // bytes in an FP element.
        string _step_dim;        // explicit step dim.
        vector<string> _domain_dims; // explicit domain dims.
        string _inner_loop_dim;      // explicit inner-loop dim.
        int _min_buffer_len = 1;     // min length of an inner-loop buffer.
        int _read_ahead_dist = 0;    // iterations to read ahead.
        IntTuple _fold_options;    // vector fold.
        IntTuple _cluster_options; // cluster multipliers.
        map<int, int> _prefetch_dists;
        bool _first_inner = true; // first dimension of fold is unit step.
        bool _allow_unaligned_loads = false;
        bool _bundle = true;
        bool _bundle_scratch = true;
        int _halo_size = 0;      // 0 => calculate each halo automatically.
        int _step_alloc = 0;     // 0 => calculate each step allocation automatically.
        bool _inner_misc = true;
        bool _outer_domain = false;
        int _max_expr_size = 50;
        int _min_expr_size = 2;
        bool _do_cse = true;      // do common-subexpr elim.
        bool _do_comb = true;    // combine commutative operations.
        bool _do_pairs = true;   // find equation pairs.
        bool _do_opt_cluster = true; // apply optimizations also to cluster.
        bool _do_reorder = false;   // reorder commutative operations.
        string _var_regex;       // vars to update.
        bool _find_deps = true;
        bool _print_eqs = false;
        bool _use_ptrs = true;  // enable access via pointers & strides.
        bool _use_many_ptrs = false;  // make pointer for almost every point.
        bool _use_offsets = false; // compute offsets from var alloc start.
        bool _early_loads = true; // issue loads early in the inner loop.

        // Add options to a cmd-line parser to set the settings.
        virtual void add_options(command_line_parser& parser);

        // Print informational messages.
        void print_usage(std::ostream& os);
        void print_values(std::ostream& os);
    };

    // Stencil dimensions.
    struct Dimensions {
        string _step_dim;         // step dimension, usually time.
        IntTuple _domain_dims;    // domain dims, usually spatial (with zero value).
        IntTuple _stencil_dims;   // both step and domain dims.
        IntTuple _misc_dims;      // misc dims that are not the step or domain.
        int _inner_loop_dim_num = 0; // stencil-dim index of inner-loop-dim.
        string _inner_layout_dim;        // inner-most domain dim in mem array layout.
        string _outer_layout_dim;        // outer-most domain dim in mem array layout.
        IntTuple _layout_dims;           // all dims in array-layout order.

        // Following contain only domain dims.
        IntTuple _scalar;       // points in scalar (value 1 in each).
        IntTuple _fold;         // points in fold.
        IntTuple _fold_gt1;      // subset of _fold w/values >1.
        IntTuple _cluster_pts;    // cluster size in points.
        IntTuple _cluster_mults;  // cluster size in vectors.

        // Direction of stepping.
        int _step_dir = 0;       // 0: undetermined, +1: forward, -1: backward.

        Dimensions() {}
        virtual ~Dimensions() {}

        // Add step dim.
        void add_step_dim(const string& dname) {
            _step_dim = dname;
            _stencil_dims.add_dim_front(dname, 0); // must be first!
        }
        
        // Add domain dims.
        // Last one added will be unit-stride.
        void add_domain_dim(const string& dname) {
            _domain_dims.add_dim_back(dname, 0);
            _stencil_dims.add_dim_back(dname, 0);
            _scalar.add_dim_back(dname, 1);
            _fold.add_dim_back(dname, 1);
            _cluster_mults.add_dim_back(dname, 1);
        }
        
        // Find the dimensions to be used.
        void set_dims(Vars& vars,
                      CompilerSettings& settings,
                      int vlen,
                      bool is_folding_efficient,
                      ostream& os);

        // Make string like "+(4/VLEN_X)" or "-(2/VLEN_Y)"
        // given signed offset and direction.
        string make_norm_str(int offset, string dim) const;

        // Make string like "t+1" or "t-1".
        string make_step_str(int offset) const;
    };

} // namespace yask.
