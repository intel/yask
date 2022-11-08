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

#include "yask_stencil.hpp"
using namespace std;

// Auto-generated stencil code that extends base types.
#define DEFINE_CONTEXT
#include YSTR2(YK_CODE_FILE)
#undef DEFINE_CONTEXT

namespace yask {

    // APIs.
    // See yask_kernel_api.hpp.
    yk_env_ptr yk_factory::new_env(MPI_Comm comm) const {
        auto ep = make_shared<KernelEnv>();
        assert(ep);
        ep->init_env(0, 0, comm);
        TRACE_MSG("YASK env object created with MPI communicator " << comm);
        return ep;
    }
    yk_env_ptr yk_factory::new_env() const {
        return new_env(MPI_COMM_NULL);
    }
    std::string yk_factory::get_version_string() {
        return yask_get_version_string();
    }

    // Compiling new_solution() triggers compilation of the stencil kernels.
    yk_solution_ptr yk_factory::new_solution(yk_env_ptr env,
                                             const yk_solution_ptr source) const {
        TRACE_MSG("creating new YASK solution...");

        // Make sure JIT compiliation has happened.
        #ifdef USE_OFFLOAD
        {
            DEBUG_MSG("Initializing OpenMP offload; there may be a delay for JIT compilation...");
            YaskTimer init_timer;
            init_timer.start();

            // Dummy OMP section to trigger JIT.
            // This should be the first "omp target" pragma encountered.
            int dummy = 42;
            #pragma omp target data device(KernelEnv::_omp_devn) map(dummy)
            { }

            init_timer.stop();
            DEBUG_MSG("OpenMP offload initialization done in " <<
                      make_num_str(init_timer.get_elapsed_secs()) << " secs.");
        }
        #endif

        auto ep = dynamic_pointer_cast<KernelEnv>(env);
        assert(ep);
        auto dp = YASK_STENCIL_CONTEXT::new_dims(); // create Dims.
        assert(dp);
        auto req_opts = make_shared<KernelSettings>(dp, ep);
        assert(req_opts);
        auto actl_opts = make_shared<KernelSettings>(dp, ep);
        assert(actl_opts);

        // Copy settings from source, if any.
        if (source.get()) {
            auto ssp = dynamic_pointer_cast<StencilContext>(source);
            assert(ssp);
            auto sop = ssp->get_req_opts();
            assert(sop);
            *req_opts = *sop;
            sop = ssp->get_actl_opts();
            assert(sop);
            *actl_opts = *sop;
        }

        // Create problem-specific object defined by stencil compiler.
        // TODO: allow more than one type of solution to be created.
        auto sp = make_shared<YASK_STENCIL_CONTEXT>(ep, actl_opts, req_opts);
        assert(sp);

#ifdef DEF_ARGS
        // If no source, init settings from default args.
        if (!source.get())
            sp->apply_command_line_options(DEF_ARGS);
#endif

        return sp;
    }
    yk_solution_ptr yk_factory::new_solution(yk_env_ptr env) const {
        return new_solution(env, nullptr);
    }

} // namespace yask.
