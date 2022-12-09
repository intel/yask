/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2022, Intel Corporation

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

///////// Methods for Settings & Dimensions ////////////

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Var.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {

    // yask_compiler_factory API methods.
    // See yask_compiler_api.hpp.
    std::string yc_factory::get_version_string() {
    	return yask_get_version_string();
    }
    yc_solution_ptr yc_factory::new_solution(const std::string& name) const {
        return make_shared<StencilSolution>(name);
    }

    // Find the dimensions to be used based on the vars in
    // the solution and the settings from the cmd-line or API.
    void Dimensions::set_dims(Vars& vars,
                             CompilerSettings& settings,
                             int vlen,                  // SIMD len based on CPU arch.
                             bool is_folding_efficient, // heuristic based on CPU arch.
                             ostream& os)
    {
        _domain_dims.clear();
        _stencil_dims.clear();
        _scalar.clear();
        _fold.clear();
        _fold_gt1.clear();
        _cluster_pts.clear();
        _cluster_mults.clear();
        _misc_dims.clear();

        // Get dims from settings.
        if (settings._step_dim.length()) {
            add_step_dim(settings._step_dim);
            os << "Explicit step dimension: " << _step_dim << endl;
        }
        for (auto& dname : settings._domain_dims)
            add_domain_dim(dname);
        if (_domain_dims.size())
            os << "Explicit domain dimension(s): " << _domain_dims.make_dim_str() << endl;

        // Get dims from vars.
        for (auto& gp : vars) {
            auto& gname = gp->_get_name();
            os << "Var: " << gp->get_descr() << endl;

            // Dimensions in this var.
            for (auto dim : gp->get_dims()) {
                auto& dname = dim->_get_name();
                auto type = dim->get_type();

                switch (type) {

                case STEP_INDEX:
                    if (_step_dim.length() && _step_dim != dname) {
                        THROW_YASK_EXCEPTION("step dimensions '" + _step_dim +
                                             "' and '" + dname + "' found; only one allowed");
                    }
                    add_step_dim(dname);

                    // Scratch vars cannot use step dim.
                    if (gp->is_scratch())
                        THROW_YASK_EXCEPTION("scratch var '" + gname +
                                             "' cannot use step dimension '" +
                                             dname + "'.\n");
                    break;

                case DOMAIN_INDEX:
                    add_domain_dim(dname);
                    break;

                case MISC_INDEX:
                    _misc_dims.add_dim_back(dname, 0);
                    break;

                default:
                    THROW_YASK_EXCEPTION("unexpected dim type " + to_string(type));
                }
            }
        }
        if (_step_dim.length() == 0) {
            THROW_YASK_EXCEPTION("no step dimension defined");
        }
        if (!_domain_dims.get_num_dims()) {
            THROW_YASK_EXCEPTION("no domain dimension(s) defined");
        }

        // Set specific positional dims.
        auto ndd = _domain_dims.get_num_dims();
        _outer_layout_dim = _domain_dims.get_dim_name(0);
        _inner_layout_dim = _domain_dims.get_dim_name(ndd - 1);
        string _near_inner_dim = _domain_dims.get_num_dims() >= 2 ?
            _domain_dims.get_dim_name(_domain_dims.get_num_dims() - 2) : _outer_layout_dim;
        if (settings._inner_loop_dim.length()) {
            if (isdigit(settings._inner_loop_dim[0])) {
                int dn = atoi(settings._inner_loop_dim.c_str());
                if (dn < 1) {
                    os << "Note: adjusting inner-loop-dim " << dn << " to 1.\n";
                    dn = 1;
                }
                if (dn > ndd) {
                    os << "Note: adjusting inner-loop-dim " << dn << " to " << ndd << ".\n";
                    dn = ndd;
                }
                settings._inner_loop_dim = _domain_dims.get_dim_name(dn - 1);
                _inner_loop_dim_num = dn;
            }
            int dp = _domain_dims.lookup_posn(settings._inner_loop_dim);
            if (dp < 0) {
                os << "Warning: inner-loop-dim '" << settings._inner_loop_dim <<
                    "' ignored because it's not a domain dim.\n";
                settings._inner_loop_dim.clear();
            } else
                _inner_loop_dim_num = dp + 1;
        }
        if (!settings._inner_loop_dim.length()) {
            settings._inner_loop_dim = _inner_layout_dim;
            _inner_loop_dim_num = ndd;
        }
        assert(_inner_loop_dim_num > 0);
        assert(_inner_loop_dim_num <= ndd);

        // Extract domain fold lengths based on cmd-line options.
        IntTuple fold_opts;
        for (auto& dim : _domain_dims) {
            auto& dname = dim._get_name();

            // Was folding specified for this dim?
            auto* p = settings._fold_options.lookup(dname);
            if (!p)
                continue;
            int sz = *p;
            if (sz < 1)
                continue;
            
            // Set size.
            _fold.set_val(dname, sz);
            fold_opts.add_dim_back(dname, sz);
        }
        os << "Folding and clustering:\n"
            " Number of SIMD elements: " << vlen << endl;
        if (fold_opts.get_num_dims())
            os << " Requested vector-fold dimension(s) and point-size(s): " <<
                _fold.make_dim_val_str(" * ") << endl;
        else
            os << " No explicitly-requested vector-folding.\n";

        // If needed, adjust folding to exactly cover vlen unless vlen is 1.
        // If vlen is 1, we will allow any folding.
        if (vlen > 1 && _fold.product() != vlen) {
            if (fold_opts.get_num_dims())
                os << "Note: adjusting requested fold to achieve SIMD length of " <<
                    vlen << ".\n";

            // If 1D, there is only one option.
            if (_domain_dims.get_num_dims() == 1)
                _fold[_inner_layout_dim] = vlen;

            // If 2D+, adjust folding.
            else {

                // Determine inner-dim size separately because
                // vector-folding works best when folding is
                // applied in non-inner-loop dims.
                int inner_sz = 1;

                // If specified dims are within vlen, try to use
                // specified inner-dim.
                if (fold_opts.product() < vlen) {

                    // Inner-dim fold-size requested and a factor of vlen?
                    auto* p = fold_opts.lookup(settings._inner_loop_dim);
                    if (p && (vlen % *p == 0))
                        inner_sz = *p;
                }

                // Remaining vlen to be split over non-inner dims.
                int upper_sz = vlen / inner_sz;

                // Tuple for non-inner dims.
                IntTuple inner_folds;
                
                // If we only want 1D folding, just set one to
                // needed value.
                if (!is_folding_efficient)
                    inner_folds.add_dim_back(_near_inner_dim, upper_sz);

                // Else, make a tuple of hints to use for setting non-inner
                // sizes.
                else {
                    IntTuple inner_opts;
                    for (auto& dim : _domain_dims) {
                        auto& dname = dim._get_name();
                        if (dname == settings._inner_loop_dim)
                            continue;
                        auto* p = fold_opts.lookup(dname);
                        int sz = p ? *p : 0; // 0 => not specified.
                        inner_opts.add_dim_front(dname, sz); // favor more inner ones.
                    }
                    assert(inner_opts.get_num_dims() == _domain_dims.get_num_dims() - 1);

                    // Get final size of non-inner dims.
                    inner_folds = inner_opts.get_compact_factors(upper_sz);
                }

                // Put them into the fold.
                for (auto& dim : _domain_dims) {
                    auto& dname = dim._get_name();
                    if (dname == settings._inner_loop_dim)
                        _fold[dname] = inner_sz;
                    else if (inner_folds.lookup(dname))
                        _fold[dname] = inner_folds[dname];
                    else
                        _fold[dname] = 1;
                }
                assert(_fold.get_num_dims() == _domain_dims.get_num_dims());
            }            

            // Check it.
            if (_fold.product() != vlen)
                THROW_YASK_EXCEPTION("(internal fault) failed to set folding for VLEN " +
                                     to_string(vlen));
        }

        // Set fold_gt1.
        for (auto i : _fold) {
            auto& dname = i._get_name();
            auto& val = i.get_val();
            if (val > 1)
                _fold_gt1.add_dim_back(dname, val);
        }
        os << " Vector-fold dimension(s) and point-size(s): " <<
            _fold.make_dim_val_str(" * ") << endl;

        // Layout used inside each folded vector.
        _fold.set_first_inner(settings._first_inner);
        _fold_gt1.set_first_inner(settings._first_inner);


        // Order all dims for layout.
        // Start w/all domain dims.
        _layout_dims = _domain_dims;

        // Insert step dim.
        _layout_dims.add_dim_front(_step_dim, 0);

        // Insert misc dims depending on setting.
        for (int i = 0; i < _misc_dims.get_num_dims(); i++) {
            auto& mdim = _misc_dims.get_dim(i);
            if (settings._inner_misc)
                _layout_dims.add_dim_back(mdim);
            else
                _layout_dims.add_dim_at(i, mdim);
        }

        // Move outer layout domain dim if requested.
        if (settings._outer_domain) {
            _layout_dims = _layout_dims.remove_dim(_outer_layout_dim);
            _layout_dims.add_dim_front(_outer_layout_dim, 0);
        }

        // Move inner layout domain dim if no explicit SIMD.
        // This will help enable implicit SIMD when possible.
        if (_fold.product() <= 1) {
            _layout_dims = _layout_dims.remove_dim(_inner_layout_dim);
            _layout_dims.add_dim_back(_inner_layout_dim, 0);
        }

        os << "Step dimension: " << _step_dim << endl;
        os << "Domain dimension(s): " << _domain_dims.make_dim_str() << endl;
        if (_misc_dims.get_num_dims())
            os << "Misc dimension(s): " << _misc_dims.make_dim_str() << endl;
        else
            os << "No misc dimensions used\n";
        os << "Dimension(s) in layout order: " << _layout_dims.make_dim_str() << endl;
        os << "Inner-loop dimension: " << settings._inner_loop_dim << endl;

        
        // Checks for unaligned loads.
        if (settings._allow_unaligned_loads) {
            if (_fold_gt1.size() > 1) {
                FORMAT_AND_THROW_YASK_EXCEPTION("attempt to allow "
                                                "unaligned loads when there are " <<
                                                _fold_gt1.size() <<
                                                " dimensions in the vector-fold that are > 1");
            }
            else if (_fold_gt1.size() > 0)
                cout << "Notice: memory layout MUST have unit-stride in " <<
                    _fold_gt1.make_dim_str() << " dimension!" << endl;
        }

        // Create final cluster lengths based on cmd-line options.
        for (auto& dim : settings._cluster_options) {
            auto& dname = dim._get_name();
            int mult = dim.get_val();

            // Nothing to do for mult < 2.
            if (mult <= 1)
                continue;

            // Does it exist anywhere?
            if (!_domain_dims.lookup(dname)) {
                os << "Warning: cluster-multiplier in '" << dname <<
                    "' dim ignored because it's not a domain dim.\n";
                continue;
            }

            // Set the size.
            _cluster_mults.add_dim_back(dname, mult);
        }
        _cluster_pts = _fold.mult_elements(_cluster_mults);

        os << " Cluster dimension(s) and multiplier(s): " <<
            _cluster_mults.make_dim_val_str(" * ") << endl;
        os << " Cluster dimension(s) and point-size(s): " <<
            _cluster_pts.make_dim_val_str(" * ") << endl;
    }

    // Make string like "+(4/VLEN_X)" or "-(2/VLEN_Y)" or "" if ofs==zero.
    // given signed offset and direction.
    string Dimensions::make_norm_str(int ofs, string dname) const {

        if (ofs == 0)
            return "";

        string res;
        if (_fold.lookup(dname)) {

            // Positive offset, e.g., '+(4 / VLEN_X)'.
            if (ofs > 0)
                res += "+(" + to_string(ofs);

            // Neg offset, e.g., '-(4 / VLEN_X)'.
            // Put '-' sign outside division to fix truncated division problem.
            else
                res += "-(" + to_string(-ofs);

            // add divisor.
            string cap_dname = PrinterBase::all_caps(dname);
            res += " / VLEN_" + cap_dname + ")";
        }

        // No fold const avail.
        else
            res += to_string(ofs);

        return res;
    }

    // Make string like "t+1" or "t-1".
    string Dimensions::make_step_str(int offset) const {
        IntTuple step;
        step.add_dim_back(_step_dim, offset);
        return step.make_dim_val_offset_str();
    }

    // A class to add fold and cluster options.
    class IntTupleOption : public CommandLineParser::OptionBase {
        IntTuple& _val;
        string_vec _strvec;
        CommandLineParser::StringListOption _slo;

    public:
        IntTupleOption(const std::string& name,
                       const std::string& help_msg,
                       IntTuple& val) :
            CommandLineParser::OptionBase(name, help_msg),
            _val(val),
            _slo(name, help_msg, _strvec) { }

        virtual std::ostream& print_value(std::ostream& os) const override {
            os << _val.make_dim_val_str(",");
            return os;
        }

        virtual bool check_arg(const string_vec& args, int& argi) override {

            // Get strings in _strvec if this is our option.
            if (_slo.check_arg(args, argi)) {

                for (auto& str : _strvec) {

                    // split by equal sign.
                    size_t ep = str.find("=");
                    if (ep == string::npos)
                        THROW_YASK_EXCEPTION("no equal sign in '" + str + "'");
                    string key = str.substr(0, ep);
                    string sval = str.substr(ep+1);

                    if (key.length() == 0)
                        THROW_YASK_EXCEPTION("empty dim name in '" + str + "'");
                    if (sval.length() == 0)
                        THROW_YASK_EXCEPTION("empty size in '" + str + "'");

                    const char* nptr = sval.c_str();
                    char* endptr = 0;
                    long ival = strtol(nptr, &endptr, 0);
                    if (ival == LONG_MIN || ival == LONG_MAX || *endptr != '\0')
                        THROW_YASK_EXCEPTION("argument for option '" + str + "' is not an integer");

                    _val.add_dim_back(key, ival);
                }
                return true;
            }
            return false;
        }

        virtual void print_help(ostream& os,
                                int width) const override {
            _print_help(os, _name + " <dim_name=value[,dim_name=value[,...]]>", width);
        }
    };

    // Add access to the compiler options from a cmd-line parser.
    void CompilerSettings::add_options(CommandLineParser& parser)
    {
        parser.add_option(make_shared<CommandLineParser::StringOption>
                          ("target",
                           "Output format (required).\n"
                           "Supported formats:\n"
                           "- avx:     YASK code for CORE AVX ISA (256-bit HW SIMD vectors).\n"
                           "- avx2:        YASK code for CORE AVX2 ISA (256-bit HW SIMD vectors).\n"
                           "- avx512: YASK code classes for CORE AVX-512 ISA (512-bit HW SIMD vectors).\n"
                           "- avx512-ymm:  YASK code for CORE AVX-512 ISA (256-bit HW SIMD vectors).\n"
                           "- knl:   YASK code for Knights-Landing (MIC) AVX-512 ISA (512-bit HW SIMD vectors).\n"
                           "- intel64:  YASK code for generic C++ with 64-bit indices (no explicit HW SIMD vectors).\n"
                           "- pseudo:  Human-readable scalar pseudo-code.\n"
                           "- pseudo-long: Human-readable scalar pseudo-code with intermediate variables.\n"
                           "- dot:    DOT-language description.\n"
                           "- dot-lite:  DOT-language description of var accesses only.",
                           _target));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("elem-bytes",
                           "Number of bytes in each FP element. "
                           "Currently, only 4 (single-precision) and 8 (double) are allowed.",
                           _elem_bytes));
        parser.add_option(make_shared<CommandLineParser::StringOption>
                          ("step-dim",
                           "[Advanced] "
                           "Name of the step dimension, e.g., 't'. "
                           "By default, the step dimension is defined implicitly when YASK variables are encountered "
                           "in the stencil DSL code.",
                           _step_dim));
        parser.add_option(make_shared<CommandLineParser::StringListOption>
                          ("domain-dims",
                           "[Advanced] "
                           "Name and order of the domain dimensions, e.g., 'x,y,z'. "
                           "In addition, domain dimensions are added implicitly when YASK variables are encountered "
                           "in the stencil DSL code. "
                           "The domain-dimension order determines array memory layout, default loop nesting, and "
                           "MPI rank layout. Thus, this option can be used to override those traits compared to "
                           "what would be obtained from the DSL code only.",
                           _domain_dims));
        parser.add_option(make_shared<CommandLineParser::StringOption>
                          ("inner-loop-dim",
                           "[Advanced] "
                           "Name of the dimension used for the inner-most stencil-computation loop. "
                           "The default is the last domain dimension specified via -domain_dims or in the "
                           "stencil DSL code. "
                           "For this option, a numerical index is allowed: '1' is the first domain-dim, etc.",
                           _inner_loop_dim));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("min-buffer-len",
                           "[Advanced] "
                           "Create inter-loop buffers used in the inner kernel loop if at least <n> points could be stored in it. "
                           "This may result in more values stored in registers rather than being re-read in each loop iteration "
                           "when multiple stencil inputs must be read along the inner-loop dimension",
                           _min_buffer_len));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("read-ahead-dist",
                           "[Advanced] "
                           "Number of iterations to read ahead into the inter-loop buffers. "
                           "This may be used as an alternative to prefetch hints.",
                           _read_ahead_dist));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("inner-misc-layout",
                           "[Advanced] "
                           "Set YASK-var memory layout so that the misc dim(s) are the inner-most dim(s) "
                           "instead of the outer-most. "
                           "This effectively creates an AoSoA-style layout instead of an SoAoA one, "
                           "where the last 'A' is the SIMD vector. "
                           "If the SIMD-vector length is 1, the last domain dim will always be in "
                           "the inner-most layout dim, even if this contradicts this setting. "
                           "This setting may help decrease the number of memory streams for complex "
                           "kernels when misc dims are used to consolidate vars. "
                           "This disallows dynamically changing the 'misc' dim sizes from the kernel APIs.",
                           _inner_misc));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("outer-domain-layout",
                           "[Advanced] "
                           "Set YASK-var memory layout so that the first domain dim is the outer-most "
                           "dim, even if the var contains step or misc dims. "
                           "This setting may be useful for run-time allocators that automatically partition "
                           "array layouts across NUMA nodes. "
                           "If the SIMD-vector length is 1, the last domain dim will always be in "
                           "the inner-most layout dim, possibly overriding this setting.",
                           _outer_domain));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("first-inner",
                           "[Advanced] "
                           "If true, each vector is saved in memory with the first given fold dimension as unit-stride "
                           "and so on until the last given fold dimension is the outer-most in the layout. "
                           "If false, each vector is saved in memory with the last fold dimension as unit-stride "
                           "and so on until the first given fold dimension is the outer-most in the layout. ",
                           _first_inner));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("fus",
                           "[Deprecated] Use -[no]-first-inner.",
                           _first_inner));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("l1-prefetch-dist",
                           "[Advanced] "
                           "Prefetch reads into the level-1 cache <integer> iterations "
                           "ahead of their usage in the inner kernel loop. "
                           "Use zero (0) to disable.",
                           _prefetch_dists[1]));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("l2-prefetch-dist",
                           "[Advanced] "
                           "Prefetch reads into the level-2 cache <integer> iterations "
                           "ahead of their usage in the inner kernel loop. "
                           "Use zero (0) to disable.",
                           _prefetch_dists[2]));
        parser.add_option(make_shared<CommandLineParser::StringOption>
                          ("vars",
                           "[Advanced] "
                           "Only process updates to vars whose names match regular expression defined in <string>. "
                           "This can be used to generate code for a subset of the stencil equations.",
                           _var_regex));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("bundle-scratch",
                           "[Advanced] "
                           "Bundle scratch equations together even if the sizes of their scratch vars must be increased "
                           "in order to do so.",
                           _bundle_scratch));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("halo",
                           "[Advanced] "
                           "If non-zero, override the calculation of the required halo sizes and force them to <integer>. "
                           "May cause memory-access faults and/or incorrect calculations "
                           "if specified to be less than the actual minimum. ",
                           _halo_size));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("step-alloc",
                           "[Advanced] "
                           "If non-zero, override the calculation of the required allocation of each variable in the "
                           "step (e.g., 't') dimension and force them to <integer>. "
                           "May cause memory-access faults and/or incorrect calculations "
                           "if specified to be less than the actual minimum. ",
                           _step_alloc));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("ul",
                           "[Advanced] "
                           "Generate simple unaligned loads instead of aligned loads followed by "
                           "shift operations when possible. "
                           "To use this correctly, only 1D folds are allowed, and "
                           "the array memory layout must have that same dimension in unit stride.",
                           _allow_unaligned_loads));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("opt-comb",
                           "[Advanced] "
                           "Combine a sequence of commutative operations, e.g., 'a + b + c' into a single parse-tree node.",
                           _do_comb));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("opt-reorder",
                           "[Advanced] "
                           "Allow reordering of commutative operations in a single parse-tree node.",
                           _do_reorder));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("opt-cse",
                           "[Advanced] "
                           "Eliminate common subexpressions in the parse-tree.",
                           _do_cse));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("opt-pair",
                           "[Advanced] "
                           "Combine matching pairs of eligible function calls into a single parse-tree node. "
                           "Currently enables 'sin(x)' and 'cos(x)' to be replaced with 'sincos(x)'.",
                           _do_pairs));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("opt-cluster",
                           "[Advanced] "
                           "Apply optimizations across a cluster of stencil equations. "
                           "Only has an effect if there are more than one vector in a cluster.",
                           _do_opt_cluster));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("max-es",
                           "[Advanced] "
                           "Heuristic for maximum expression-size threshold when outputting code from a parse-tree.",
                           _max_expr_size));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("min-es",
                           "[Advanced] "
                           "Heuristic for minimum expression-size threshold for creating a temporary variable for reuse "
                           "when outputting code from a parse-tree.",
                           _min_expr_size));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("use-ptrs",
                           "[Advanced] "
                           "Generate inner-kernel loop code using data pointers & strides, avoiding function calls.",
                           _use_ptrs));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("use-safe-ptrs",
                           "[Advanced] "
                           "Generate kernel code with pointer parameters to base addresses for each YASK var. "
                           "This is a workaround for offload-device drivers that don't allow negative indices from "
                           "a pointer that is a kernel argument",
                           _use_offsets));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("early-loads",
                           "[Advanced] "
                           "Generate code to load variables early in the inner-kernel loop instead of "
                           "immediately before they are needed.",
                           _early_loads));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("print-eqs",
                           "[Debug] "
                           "Print each equation when defined",
                           _print_eqs));
        parser.add_option(make_shared<IntTupleOption>
                          ("fold",
                           "The recommended number of elements in each given dimension in a vector block. "
                           "Default depends on -elem-bytes setting, domain-dimension order, and output format. "
                           "If product of fold lengths does not equal SIMD vector length for output "
                           "formats with defined lengths (e.g., 16 for 'avx512' when using 4-byte reals), "
                           "lengths will adjusted as needed.",
                           _fold_options));
        parser.add_option(make_shared<IntTupleOption>
                          ("cluster",
                           "The number of vectors to evaluate per inner-kernel loop iteration "
                           "in each domain dimension. "
                           "Default is one (1) in each unspecified dimension.",
                           _cluster_options));
    }    

} // namespace yask.
