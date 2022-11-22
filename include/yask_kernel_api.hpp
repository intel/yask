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

///////// API for the YASK stencil kernel. ////////////

// This file uses Doxygen 1.8 markup for API documentation-generation.
// See http://www.stack.nl/~dimitri/doxygen.
/** @file yask_kernel_api.hpp */

#pragma once

#include "yask_common_api.hpp"
#include <vector>
#include <cinttypes>

#ifndef MPI_VERSION
typedef int MPI_Comm;
#endif

namespace yask {

    /**
     * \defgroup yk YASK Kernel
     * Types, clases, and functions used in both the \ref sec_yk.
     * @{
     */

    // Forward declarations of classes and pointers.

    class yk_env;
    /// Shared pointer to \ref yk_env.
    typedef std::shared_ptr<yk_env> yk_env_ptr;

    class yk_solution;
    /// Shared pointer to \ref yk_solution.
    typedef std::shared_ptr<yk_solution> yk_solution_ptr;

    class yk_var;
    /// Shared pointer to \ref yk_var.
    typedef std::shared_ptr<yk_var> yk_var_ptr;

    class yk_stats;
    /// Shared pointer to \ref yk_stats.
    typedef std::shared_ptr<yk_stats> yk_stats_ptr;

    /** @}*/
} // namespace yask.

#include "aux/yk_solution_api.hpp"
#include "aux/yk_var_api.hpp"

namespace yask {

    /**
     * \addtogroup yk
     * @{
     */

    /// Bootstrap factory to create a stencil solution.
    class yk_factory {
    public:
        yk_factory();
        virtual ~yk_factory() {}

        /// Version information.
        /**
           @returns String describing the current version.
        */
        virtual std::string
        get_version_string();

        /// Create an object to hold environment information.
        /**
           Performs the following initialization steps:
           - Initializes MPI if MPI is enabled but not yet initialized.
           Does not initialize MPI if MPI is not enabled or already initialized.
           If MPI is enabled, uses `MPI_COMM_WORLD` as the communicator.
           - Sets flush-to-zero (FTZ) and denormals-are-zero (DAZ)
           floating-point controls.
           - Enables "hot teams" mode for Intel OpenMP. 
           Initializes OpenMP library if it is not already started.
           If it is already started, note that the "hot teams" mode may
           not get set correctly. This mode is important for optimal
           performance of nested OpenMP regions (used when the number
           of threads per block is greater than one).

           @note If you initialize MPI before calling this function,
           you should call `MPI_Init_thread(..., MPI_THREAD_SERIALIZED, ...)`
           or  `MPI_Init_thread(..., MPI_THREAD_MULTIPLE, ...)`.

           @note If you initialize OpenMP (by calling any OpenMP function)
           before calling this function, set the following environment
           variables prior to initializing OpenMP: 
           `KMP_HOT_TEAMS_MODE=` and `KMP_HOT_TEAMS_MAX_LEVEL=2`.

           Environment info is kept in a separate object to factilitate
           initializing the environment before creating a solution
           and sharing an environment among multiple solutions.
           @returns Pointer to new env object.
        */
        virtual yk_env_ptr
        new_env() const;

        /// Create a \ref yk_env object using the provided MPI communicator.
        /**
           Behaves like new_env(), but uses the provided MPI communicator 
           instead of using `MPI_COMM_WORLD`.
           MPI must be enabled and initialized before calling this function
           following the usage notes for new_env().

           @note #`include "mpi.h"` should precede #`include "yask_kernel_api.hpp"`
           to ensure proper MPI type definitions.
        */
        virtual yk_env_ptr
        new_env(MPI_Comm comm) const;

        /// Create a stencil solution.
        /**
           A stencil solution contains all the vars and equations
           that were created during stencil compilation.
           @returns Pointer to new solution object.
        */
        virtual yk_solution_ptr
        new_solution(yk_env_ptr env /**< [in] Pointer to env info. */) const;

        /// **[Advanced]** Create a stencil solution by copying the settings from another.
        /**
           All the settings that were specified via the `yk_solution::set_*()`
           functions in the source solution will be copied to the new solution.
           This does *not* copy any vars, var settings, or var data;
           see yk_solution::fuse_vars().
           @returns Pointer to new solution object.
        */
        virtual yk_solution_ptr
        new_solution(yk_env_ptr env /**< [in] Pointer to env info. */,
                     const yk_solution_ptr source
                     /**< [in] Pointer to existing \ref yk_solution from which
                        the settings will be copied. */ ) const;
    }; // yk_factory.

    /// Kernel environment.
    /**
       Created via yk_factory::new_env().
    */
    class yk_env {
    public:
        virtual ~yk_env() {}

        /// Set object to receive debug output.
        /**
           This is a static method, implying the following:
           - This setting may be changed before creating a `yk_env` object.
           - Calling this method applies settings globally.
         */
        static void
        set_debug_output(yask_output_ptr debug
                         /**< [out] Pointer to object to receive debug output.
                            See \ref yask_output_factory. */ );

        /// Disable the debug output.
        /**
           Shortcut for calling `set_debug_output()` with a `yask_null_output_ptr`;
         */
        static void
        disable_debug_output();

        /// Get object to receive debug output.
        /**
           This is a static method, implying the following:
           - This method may be called before creating a `yk_env` object.

           @returns Pointer to \ref yask_output set via set_debug_output
           or pointer to a \ref yask_stdout_output if not set.
        */
        static yask_output_ptr
        get_debug_output();

        /// Enable or disable additional debug tracing.
        /**
           This is a static method, implying the following:
           - This setting may be changed before creating a `yk_env` object.
           - Calling this method applies settings globally.

           Must also compile with general tracing and/or memory-access tracing enabled.
        */
        static void
        set_trace_enabled(bool enable);

        /// Get whether tracing is enabled.
        /**
           This is a static method, implying the following:
           - This method may be called before creating a `yk_env` object.

           @returns Whether tracing is enabled.
        */
        static bool
        is_trace_enabled();

        /// Get number of MPI ranks.
        /**
           @returns Number of ranks in MPI communicator or one (1) if MPI is not enabled.
        */
        virtual int get_num_ranks() const =0;

        /// Get MPI rank index.
        /**
           @returns Index of this MPI rank or zero (0) if MPI is not enabled.
        */
        virtual int get_rank_index() const =0;

        /// Wait until all ranks have reached this element.
        /**
           If MPI is enabled, calls `MPI_Barrier()`.
           Otherwise, has no effect.
         */
        virtual void
        global_barrier() const =0;

        /// Finalize the environment.
        /**
           If MPI is enabled and YASK initialized the MPI communicator,
           calls `MPI_Finalize()`.
           This function is automatically called when a yk_env object
           is destroyed.
           Cannot call global_barrier() or any MPI-dependent API after
           calling this.
        */
        virtual void
        finalize() =0;

    }; // yk_env.

    /// **[Deprecated]** Use yk_var.
    YASK_DEPRECATED
    typedef yk_var yk_grid;
    /// **[Deprecated]** Use yk_var_ptr.
    YASK_DEPRECATED
    typedef yk_var_ptr yk_grid_ptr;

    /** @}*/

} // namespace yask.
