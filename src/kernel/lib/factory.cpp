/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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

    // APIs.
    // See yask_kernel_api.hpp.
	std::string yk_factory::get_version_string() {
		return yask_get_version_string();
	}
    yk_solution_ptr yk_factory::new_solution(yk_env_ptr env,
                                             const yk_solution_ptr source) const {
        auto ep = dynamic_pointer_cast<KernelEnv>(env);
        assert(ep);
        auto dp = YASK_STENCIL_CONTEXT::new_dims();
        assert(dp);
        auto op = make_shared<KernelSettings>(dp, ep);
        assert(op);

        // Copy settings from source.
        if (source.get()) {
            auto ssp = dynamic_pointer_cast<StencilContext>(source);
            assert(ssp);
            auto sop = ssp->get_settings();
            assert(sop);
            *op = *sop;
        }

        // Create problem-specific object defined by stencil compiler.
        // TODO: allow more than one type of solution to be created.
        auto sp = make_shared<YASK_STENCIL_CONTEXT>(ep, op);
        assert(sp);

        // If no source, init settings from default args.
        if (!source.get())
            sp->apply_command_line_options(DEF_ARGS);
        
        return sp;
    }
    yk_solution_ptr yk_factory::new_solution(yk_env_ptr env) const {
        return new_solution(env, nullptr);
    }

} // namespace yask.
