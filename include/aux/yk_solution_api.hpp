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

///////// API for the YASK stencil kernel solution. ////////////

// This file uses Doxygen markup for API documentation-generation.
// See https://www.doxygen.nl/manual/index.html.
/** @file yk_solution_api.hpp */

#pragma once

#include "yask_kernel_api.hpp"

namespace yask {

    /**
     * \addtogroup yk
     * @{
     */

    /// Allocate vars on local NUMA node.
    /**
       This is used in yk_solution::set_default_numa_preferred
       and yk_var::set_numa_preferred.
       In Python, specify as `yask_kernel.cvar.yask_numa_local`.
    */
    const int yask_numa_local = -1;

    /// Allocate vars across all available NUMA nodes.
    /**
       This is used in yk_solution::set_default_numa_preferred
       and yk_var::set_numa_preferred.
       In Python, specify as `yask_kernel.cvar.yask_numa_interleave`.
    */
    const int yask_numa_interleave = -2;

    /// Do not specify any NUMA binding.
    /**
       This is used in yk_solution::set_default_numa_preferred
       and yk_var::set_numa_preferred.
       In Python, specify as `yask_kernel.cvar.yask_numa_none`.
    */
    const int yask_numa_none = -9;

    /// Do not specify any NUMA binding and use allocations optimized for offloading.
    /**
       This is used in yk_solution::set_default_numa_preferred
       and yk_var::set_numa_preferred.
       In Python, specify as `yask_kernel.cvar.yask_numa_host`.
    */
    const int yask_numa_offload = -11;

    /// Stencil solution as defined by the generated code from the YASK stencil compiler.
    /**
       Objects of this type contain all the vars and equations
       that comprise a solution.

       Created via yk_factory::new_solution().
    */
    class yk_solution {
    public:
        virtual ~yk_solution() {}

        /// Get the name of the solution.
        /**
           @returns String containing the solution name provided during stencil compilation.
        */
        virtual const std::string&
        get_name() const =0;

        /// Get the description (long name) of the solution.
        /**
           @returns String containing the solution description provided during stencil compilation
           or the name if no description was provided.
        */
        virtual const std::string&
        get_description() const =0;

        /// Get the target ISA.
        /**
           @returns String describing the instruction-set architecture of the CPU targeted
           during kernel compilation.
           See the allowed YASK kernel targets in yc_solution::set_target().
        */
        virtual std::string
        get_target() const =0;

        /// Get whether the stencil kernel will be offloaded to a device.
        /**
           @returns _true_ if kernel will be offloaded or _false_ if not.
        */
        virtual bool
        is_offloaded() const =0;

        /// Get the floating-point precision size.
        /**
           @returns Number of bytes in each FP element: 4 or 8.
        */
        virtual int
        get_element_bytes() const =0;

        /// Get the solution step dimension.
        /**
           @returns String containing the step-dimension name
           that was defined by yc_node_factory::new_step_index()
           and used in one or more vars.
        */
        virtual std::string
        get_step_dim_name() const =0;

        /// Get the number of domain dimensions used in this solution.
        /**
           The domain dimensions are those over which the stencil is
           applied in each step.
           Does *not* include the step dimension or any miscellaneous dimensions.
           @returns Number of dimensions that define the problem domain.
        */
        virtual int
        get_num_domain_dims() const =0;

        /// Get all the domain dimension names.
        /**
           @returns List of all domain-dimension names
           that were defined by yc_node_factory::new_domain_index()
           and used in one or more vars.
        */
        virtual string_vec
        get_domain_dim_names() const =0;

        /// Get all the miscellaneous dimension names.
        /**
           @returns List of all dimension names
           that were either
           * Defined by yc_node_factory::new_misc_index()
           and used in one or more vars, or
           * Created at run-time by adding a new dimension
           via yk_solution::new_var() or yk_solution::new_fixed_size_var().
        */
        virtual string_vec
        get_misc_dim_names() const =0;

        /// Set the local-domain size in the specified dimension, i.e., the size of the part of the domain that is in this rank.
        /**
           The domain defines the number of elements that will be evaluated with the stencil(s).
           If MPI is not enabled, this is equivalent to the global-domain size.
           If MPI is enabled, this is the domain size for the current rank only,
           and the global-domain size is the sum of all local-domain sizes
           in each dimension.
           The local-domain size in each rank does not have to be the same, but
           all local-domains in the same column of ranks must have the same width,
           all local-domains in the same row must have the same height,
           and so forth, for each domain dimension.
           The local-domain size does *not* include the halo area or any padding.
           For best performance, set the local-domain
           size to a multiple of the number of elements in a vector-cluster in
           each dimension.

           You should set either the local-domain size or the global-domain size
           in each dimension; the other should be set to zero (unspecified).
           The unspecified (zero) sizes will be calculated based on the 
           specified ones when prepare_solution() is called.

           See the "Detailed Description" for \ref yk_var for more information on var sizes.
        */
        virtual void
        set_rank_domain_size(const std::string& dim
                             /**< [in] Name of dimension to set.  Must be one of
                                the names from get_domain_dim_names(). */,
                             idx_t size /**< [in] Elements in the domain in this `dim`. */ ) =0;

        /// Set the local-domain size in all domain dimensions.
        /**
           See set_rank_domain_size().
        */
        virtual void
        set_rank_domain_size_vec(const idx_t_vec& vals
                             /**< [in] Elements in all domain dims. */) = 0;

        #ifndef SWIG
        /// Set the local-domain size in all domain dimensions.
        /**
           See set_rank_domain_size().
        */
        virtual void
        set_rank_domain_size_vec(const idx_t_init_list& vals
                             /**< [in] Elements in all domain dims. */) = 0;
        #endif

        /// Get the local-domain size in the specified dimension, i.e., the size in this rank.
        /**
           See documentation for set_rank_domain_size().

           @note get_rank_domain_size() may return zero in a dimension until
           prepare_solution() is called. After prepare_solution() is called,
           the computed size will be returned.

           @returns Current setting of rank domain size in specified dimension.
        */
        virtual idx_t
        get_rank_domain_size(const std::string& dim
                             /**< [in] Name of dimension to get.  Must be one of
                                the names from get_domain_dim_names(). */) const =0;

        /// Get the local-domain size in all domain dimensions.
        /**
           See get_rank_domain_size().
           @returns Vector of current setting of rank domain sizes.
        */
        virtual idx_t_vec
        get_rank_domain_size_vec() const =0;

        /// Get the global-domain size in the specified dimension, i.e., the total size across all MPI ranks.
        /**
           You should set either the local-domain size or the global-domain size
           in each dimension; the other should be set to zero (unspecified).
           The unspecified (zero) sizes will be calculated based on the 
           specified ones when prepare_solution() is called.

           See documentation for set_rank_domain_size().
           See the "Detailed Description" for \ref yk_var for more information on var sizes.
        */
        virtual void
        set_overall_domain_size(const std::string& dim
                                /**< [in] Name of dimension to set.  Must be one of
                                   the names from get_domain_dim_names(). */,
                                idx_t size /**< [in] Elements in the domain in this `dim`. */ ) =0;

        /// Set the global-domain size in all domain dimensions.
        /**
           See set_overall_domain_size().
        */
        virtual void
        set_overall_domain_size_vec(const idx_t_vec& vals
                                    /**< [in] Elements in all domain dims. */) = 0;

        #ifndef SWIG
        /// Set the global-domain size in all domain dimensions.
        /**
           See set_overall_domain_size().
        */
        virtual void
        set_overall_domain_size_vec(const idx_t_init_list& vals
                                    /**< [in] Elements in all domain dims. */) = 0;
        #endif

        /// Get the global-domain size in the specified dimension, i.e., the total size across all MPI ranks.
        /**
           The global-domain indices in the specified dimension will range from
           zero (0) to get_overall_domain_size() - 1, inclusive.
           Call get_first_rank_domain_index() and get_last_rank_domain_index()
           to find the subset of this domain in each rank.

           @note get_overall_domain_size() may return zero in a dimension until
           prepare_solution() is called. After prepare_solution() is called,
           the computed size will be returned.

           @returns Sum of all ranks' domain sizes in the given dimension.
        */
        virtual idx_t
        get_overall_domain_size(const std::string& dim
                                /**< [in] Name of dimension to get.  Must be one of
                                   the names from get_domain_dim_names(). */ ) const =0;

        /// Get the global-domain size in all domain dimensions.
        /**
           See get_overall_domain_size().

           @returns Vector of current setting of global domain sizes.
        */
        virtual idx_t_vec
        get_overall_domain_size_vec() const =0;

        /// Set the block size in the given dimension.
        /**
           This sets the approximate number of elements that are evaluated in
           each "block".
           This is a performance setting and should not affect the functional
           correctness or total number of elements evaluated.
           A block is typically the unit of work done by a
           top-level OpenMP thread.  The actual number of elements evaluated
           in a block may be greater than the specified size due to rounding
           up to fold-cluster sizes.  The number of elements in a block may
           also be smaller than the specified size when the block is at the
           edge of the domain. The block size cannot be set in the
           solution-step dimension (because temporal blocking is not yet enabled).

           Unless auto-tuning is disabled, the block size will be used as
           a starting point for an automated search for a higher-performing
           block size.

           This and all other tile sizes (Mega-blocks, blocks, micro-blocks, etc.)
           can be set via apply_command_line_options().
        */
        virtual void
        set_block_size(const std::string& dim
                       /**< [in] Name of dimension to set.  Must be one of
                          the names from get_step_dim_name() or
                          get_domain_dim_names(). */,
                       idx_t size
                       /**< [in] Elements in a block in this `dim`. */ ) =0;

        /// Set the block size in all domain dimensions.
        /**
           See set_block_size().

           @note Does _not_ set the block size in the step dim.
           Call set_block_size() with the name of the step dim to set the
           temporal block size.
        */
        virtual void
        set_block_size_vec(const idx_t_vec& vals
                           /**< [in] Elements in all domain dims. */) = 0;

        #ifndef SWIG
        /// Set the block size in all domain dimensions.
        /**
           See set_block_size().

           @note Does _not_ set the block size in the step dim.
           Call set_block_size() with the name of the step dim to set the
           temporal block size.
        */
        virtual void
        set_block_size_vec(const idx_t_init_list& vals
                           /**< [in] Elements in all domain dims. */) = 0;
        #endif

        /// Get the block size.
        /**
           Returned value may be slightly larger than the value provided
           via set_block_size() due to rounding.
           @returns Current settings of block size.
        */
        virtual idx_t
        get_block_size(const std::string& dim
                        /**< [in] Name of dimension to get.  Must be one of
                           the names from get_step_dim_name() or
                           get_domain_dim_names(). */) const =0;

        /// Get the block size in all domain dimensions.
        /**
           See get_block_size().

           @note Does _not_ return the block size in the step domain.
           Call get_block_size() with the name of the step-domain dimension
           to get the temporal block size.

           @returns Vector of current setting of block domain sizes.
        */
        virtual idx_t_vec
        get_block_size_vec() const =0;

        /// Set the number of MPI ranks in the given dimension.
        /**
           If set_num_ranks() is set to a non-zero value in all
           dimensions, then
           the *product* of the number of ranks across all dimensions must
           equal the value returned by yk_env::get_num_ranks().
           If the number of ranks is zero in one or more 
           dimensions, those values will be set by a heuristic when
           prepare_solution() is called.
           An exception will be thrown if no legal values are possible
           given the specified (non-zero) values.

           The curent MPI rank will be assigned a unique location
           within the overall problem domain based on its MPI rank index.
           Or, you can set it explicitly via set_rank_index().

           The same number of MPI ranks must be set via this API on each
           constituent MPI rank to ensure a consistent overall configuration.
           The number of ranks in each dimension must be properly set
           before calling yk_solution::prepare_solution().
           There is no rank setting allowed in the
           solution-step dimension (usually "t") or in a misc dimension.

           In fact, a practical definition of a domain dimension is one
           that is decomposable across MPI ranks. Specifically, a
           domain dimension does not have to correspond to a
           spatial dimension in the physical problem description.
        */
        virtual void
        set_num_ranks(const std::string& dim
                      /**< [in] Name of dimension to set.  Must be one of
                         the names from get_domain_dim_names(). */,
                      idx_t num /**< [in] Number of ranks in `dim`. */ ) =0;

        /// Set the number of MPI ranks in all domain dimensions.
        /**
           See set_num_ranks().
        */
        virtual void
        set_num_ranks_vec(const idx_t_vec& vals
                          /**< [in] Number of ranks in all domain dims. */) = 0;

        #ifndef SWIG
        /// Set the number of all MPI ranks in all domain dimensions.
        /**
           See set_num_ranks().
        */
        virtual void
        set_num_ranks_vec(const idx_t_init_list& vals
                          /**< [in] Number of ranks in all domain dims. */) = 0;
        #endif

        /// Get the number of MPI ranks in the given dimension.
        /**
           @note get_num_ranks() may return zero in a dimension until
           prepare_solution() is called. After prepare_solution() is called,
           the computed number of ranks will be returned.

           @returns Current number of ranks.
        */
        virtual idx_t
        get_num_ranks(const std::string& dim
                      /**< [in] Name of dimension to get.  Must be one of
                         the names from get_domain_dim_names(). */) const =0;

        /// Get the number of MPI ranks in all domain dimensions.
        /**
           See get_num_ranks();
           @returns Vector of current number of ranks in all domain dimensions.
        */
        virtual idx_t_vec
        get_num_ranks_vec() const =0;

        /// Set the rank index in the specified dimension.
        /**
           The overall rank index in the specified dimension must range from
           zero (0) to get_num_ranks() - 1, inclusive.
           If you do not call set_rank_index(), a rank index will be assigned
           when prepare_solution() is called.
           You should either call set_rank_index() on all ranks or allow
           YASK to assign on on all ranks, i.e., do not mix-and-match.

           Example using 6 MPI ranks in a 2-by-3 x, y domain:

           <table>
           <tr><td>MPI rank index = 0, x rank index = 0, y rank index = 0
               <td>MPI rank index = 1, x rank index = 1, y rank index = 0
           <tr><td>MPI rank index = 2, x rank index = 0, y rank index = 1
               <td>MPI rank index = 3, x rank index = 1, y rank index = 1
           <tr><td>MPI rank index = 4, x rank index = 0, y rank index = 2
               <td>MPI rank index = 5, x rank index = 1, y rank index = 2
           </table>

           See yk_env::get_num_ranks() and yk_env::get_rank_index() for MPI rank index.

           @note get_rank_index() may return zero in a dimension until
           prepare_solution() is called. After prepare_solution() is called,
           the computed index will be returned.
        */
        virtual void
        set_rank_index(const std::string& dim
                       /**< [in] Name of dimension to set.  Must be one of
                          the names from get_domain_dim_names(). */,
                       idx_t num /**< [in] Rank index in `dim`. */ ) =0;

        /// Set the rank index in all domain dimensions.
        /**
           See set_rank_index().
        */
        virtual void
        set_rank_index_vec(const idx_t_vec& vals
                           /**< [in] Index of this rank in all domain dims. */) = 0;

        #ifndef SWIG
        /// Set the rank index in all domain dimensions.
        /**
           See set_rank_index().
        */
        virtual void
        set_rank_index_vec(const idx_t_init_list& vals
                           /**< [in] Index of this rank in all domain dims. */) = 0;
        #endif

        /// Get the rank index in the specified dimension.
        /**
           The overall rank index in the specified dimension will range from
           zero (0) to get_num_ranks() - 1, inclusive.
           @returns Zero-based index of this rank.
        */
        virtual idx_t
        get_rank_index(const std::string& dim
                       /**< [in] Name of dimension to get.  Must be one of
                         the names from get_domain_dim_names(). */ ) const =0;

        /// Get the rank index in all domain dimensions.
        /**
           See get_rank_index();
           @returns Vector of zero-based indices of this rank in all domain dimensions.
        */
        virtual idx_t_vec
        get_rank_index_vec() const =0;

        /// Set kernel options from a string.
        /**
           Parses the string for options as if from a command-line.
           Example: "-bx 64 -inner_threads 4" sets the block-size in the *x*
           dimension to 64 and the number of nested OpenMp threads to 4.
           See the help message from the YASK kernel binary for documentation
           on the command-line options.
           Used to set less-common options not directly supported by the
           APIs above (set_block_size(), etc.).

           @returns Any parts of `args` that were not recognized by the parser as options.
           Thus, a non-empty returned string may be used to signal an error or
           interpreted by a custom application in another way.
        */
        virtual std::string
        apply_command_line_options(const std::string& args
                                   /**< [in] String of arguments to parse. */ ) =0;

        /// Set kernel options from standard C or C++ `argc` and `argv` parameters to `main()`.
        /**
           Discards `argv[0]`, which is the executable name.
           Then, parses the remaining `argv` values for options as
           described in apply_command_line_options() with a string argument.

           @returns Any parts of `argv` that were not recognized by the parser as options.
        */
        virtual std::string
        apply_command_line_options(int argc, char* argv[]) =0;

        /// Set kernel options from a vector of strings.
        /**
           Parses `args` values for options as
           described in apply_command_line_options() with a string argument.

           @returns Any parts of `args` that were not recognized by the parser as options.
        */
        virtual std::string
        apply_command_line_options(const string_vec& args) =0;

        /// Return a help-string for the command-line options.
        /**
           @returns A multi-line string.
        */
        virtual std::string
        get_command_line_help() =0;

        /// Return a description of the current settings of the command-line options.
        /**
           If options have been modified from the originally-requrested ones
           to legal ones, the updated ones will be shown. This occurs most
           frequently with tile-size options.
           @returns A multi-line string.
        */
        virtual std::string
        get_command_line_values() =0;

        /// Get the number of vars in the solution.
        /**
           Vars may be pre-defined by the stencil compiler
           (e.g., via yc_solution::new_var())
           or created explicitly via yk_solution::new_var()
           or yk_solution::new_fixed_size_var().
           @returns Number of YASK vars that have been created.
        */
        virtual int
        get_num_vars() const =0;

        /// Get the specified var.
        /**
           This cannot be used to access scratch vars.
           @returns Pointer to the specified var or null pointer if it does not exist.
        */
        virtual yk_var_ptr
        get_var(const std::string& name
                /**< [in] Name of the var. */ ) =0;

        /// Get all the vars.
        /**
           @returns List of all non-scratch vars in the solution.
        */
        virtual std::vector<yk_var_ptr>
        get_vars() =0;

        /// Prepare the solution for stencil application.
        /**
           Calculates the position of each rank in the overall problem domain
           if not previsouly specified.
           Calculates the sizes of each rank if not previsously specified.
           Allocates data in vars that do not already have storage allocated.
           Sets many other data structures needed for proper stencil application.
           Since this function initiates MPI communication, it must be called
           on all MPI ranks, and it will block until all ranks have completed.
           Must be called before applying any stencils.
        */
        virtual void
        prepare_solution() =0;

        /// Get the first index of the sub-domain in this rank in the specified dimension.
        /**
           This returns the first *overall* index at the beginning of the domain in this rank.
           Elements within the domain in this rank lie between the values returned by
           get_first_rank_domain_index() and get_last_rank_domain_index(), inclusive.
           If there is only one MPI rank, this is typically zero (0).
           If there is more than one MPI rank, the value depends
           on the the rank's position within the overall problem domain.

           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns First domain index in this rank.
        */
        virtual idx_t
        get_first_rank_domain_index(const std::string& dim
                                    /**< [in] Name of dimension to get.  Must be one of
                                       the names from get_domain_dim_names(). */ ) const =0;
        
        /// Get the first index of the sub-domain in this rank in all domain dimensions.
        /**
           See get_first_rank_domain_index().
           @returns Vector of first domain indices of this rank in all domain dimensions.
        */
        virtual idx_t_vec
        get_first_rank_domain_index_vec() const =0;

        /// Get the last index of the sub-domain in this rank the specified dimension.
        /**
           This returns the last *overall* index within the domain in this rank
           (*not* one past the end).
           If there is only one MPI rank, this is typically one less than the value
           provided by set_rank_domain_size().
           If there is more than one MPI rank, the value depends
           on the the rank's position within the overall problem domain.
           See get_first_rank_domain_index() for more information.

           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns Last index in this rank.
        */
        virtual idx_t
        get_last_rank_domain_index(const std::string& dim
                                   /**< [in] Name of dimension to get.  Must be one of
                                      the names from get_domain_dim_names(). */ ) const =0;

        /// Get the last index of the sub-domain in this rank in all domain dimensions.
        /**
           See get_last_rank_domain_index().
           @returns Vector of last domain indices of this rank in all domain dimensions.
        */
        virtual idx_t_vec
        get_last_rank_domain_index_vec() const =0;

        /// Run the stencil solution for the specified steps.
        /**
           The stencil(s) in the solution are applied to the var data, setting the
           index variables as follows:
           1. If temporal tiling is *not* used (the default):
            - The step index (e.g., `t` for "time") will be sequentially set to values
            from `first_step_index` to `last_step_index`, inclusive.
             + If the stencil equations were defined with dependencies on lower-valued steps,
             e.g., `t+1` depends on `t`, then `last_step_index` should be greater than or equal to
             `first_step_index` (forward solution).
             + If the stencil equations were defined with dependencies on higher-valued steps,
             e.g., `t-1` depends on `t`, then `last_step_index` should be less than or equal to
             `first_step_index` (reverse solution).
            - For each step index, the domain indices will be set
            to values across the entire domain as returned by yk_solution::get_overall_domain_size()
            (not necessarily sequentially).
            - MPI halo exchanges will occur as necessary before or during each step.
           2. **[Advanced]** If temporal wave-front tiling *is* enabled:
            - The step index (e.g., `t` for "time") will be sequentially set to values
            from `first_step_index` to `last_step_index`, inclusive, within each area configured
            for temporal tiling.
             + The number of steps in an area may also be restricted by the size
             of the area in the step dimension. 
             In that case, tiles will be done in temporal slices of that size.
            - For each step index within each area, the domain indices will be set
            to values across the entire area (not necessarily sequentially).
            - Ultimately, the stencil(s) will be applied to same the elements in both the step
            and domain dimensions as when wave-front tiling is not used.
            - MPI halo exchanges may occur at less frequent intervals.

           This function should be called only *after* calling prepare_solution().

           Since this function initiates MPI communication, it must be called
           on all MPI ranks, and it will block until all ranks have completed.
        */
        virtual void
        run_solution(idx_t first_step_index /**< [in] First index in the step dimension */,
                     idx_t last_step_index /**< [in] Last index in the step dimension */ ) =0;

        /// Run the stencil solution for the specified step.
        /**
           This function is simply an alias for `run_solution(step_index, step_index)`, i.e.,
           the solution will be applied for exactly one step across the domain.

           Typical C++ usage:

           \code{.cpp}
           soln->prepare_solution();
           for (idx_t t = 1; t <= num_steps; t++)
               soln->run_solution(t);
           soln->end_solution();
           \endcode

           As written, the above loop is identical to

           \code{.cpp}
           soln->prepare_solution();
           soln->run_solution(1, num_steps);
           soln->end_solution();
           \endcode

           @note The parameter is *not* the number of steps to run.
           @warning Since only one step is taken per call, using this function effectively disables
           wave-front tiling (except in the special case of tiling only across stages within a step).
        */
        virtual void
        run_solution(idx_t step_index /**< [in] Index in the step dimension */ ) =0;

        /// Update data on the device.
        /**
           Copies any YASK var data that has been modified on the host but
           not on the device from the host to the device.

           This is done automatically as needed, so calling this function is only
           needed when you want to control when the copy is done.

           If the kernel has been compiled for offloading using unified shared memory,
           calling this function will have no effect. 
           Similarly, if the kernel has not been compiled for offloading,
           calling this function will have no effect. 
        */
        virtual void
        copy_vars_to_device() const =0;

        /// Update data on the host.
        /**
           Copies any YASK var data that has been modified on the device but
           not on the host from the device to the host.

           This is done automatically as needed, so calling this function is only
           needed when you want to control when the copy is done.

           If the kernel has been compiled for offloading using unified shared memory,
           calling this function will have no effect. 
           Similarly, if the kernel has not been compiled for offloading,
           calling this function will have no effect. 
        */
        virtual void
        copy_vars_from_device() const =0;

        /// Finish using a solution.
        /**
           Performs a final MPI halo exchange.
           Releases shared ownership of memory used by the vars.  This will
           result in deallocating each memory block that is not
           referenced by another shared pointer.
        */
        virtual void
        end_solution() =0;

        /// Get performance statistics associated with preceding calls to run_solution().
        /**
           @note Side effect: resets all statistics, so each call
           returns only the elapsed time and counts since the previous call.
           @note Side effect: outputs stats in human-readable format
           to current debug output object.
           @returns Pointer to statistics object.
        */
        virtual yk_stats_ptr
        get_stats() =0;

        /// Start or stop the online auto-tuner on this rank.
        /**
           This function is used to apply the current best-known settings if the tuner is
           currently running, reset the state of the auto-tuner, and either
           restart its search (if `enable==true`) or stop it (if `enable==false`).
           This call must be made on each rank where the change is desired.

           This mode of running the auto-tuner is called "online" or "in-situ" because
           changes are made to the tile sizes between calls to run_solution().
           It will stop automatically when it converges.
           Call is_auto_tuner_enabled() to determine if it has converged.
        */
        virtual void
        reset_auto_tuner(bool enable
                         /**< [in] If _true_, start or restart the auto-tuner search on this rank.
                            If _false_, stop the auto-tuner. */,
                         bool verbose = false
                         /**< [in] If _true_, print progress information to the debug object
                            set via set_debug_output(). */ ) =0;

        /// Determine whether the online auto-tuner is enabled on this rank.
        /**
           The "online" or "in-situ" auto-tuner is disabled by default.
           It can be enabled by calling reset_auto_tuner(true).
           It will also become disabled after it has converged or after reset_auto_tuner(false) has been called.
           Auto-tuners run independently on each rank, so they will not generally finish at the same step
           across all ranks.
           @returns Whether the auto-tuner is still searching.
        */
        virtual bool
        is_auto_tuner_enabled() const =0;

        /// Run the offline auto-tuner immediately, not preserving variable data.
        /**
           This runs the auto-tuner in "offline" mode.
           (Under "online" operation, an auto-tuner is invoked during calls to
           run_solution(); see reset_auto_tuner() and is_auto_tuner_enabled()
           for more information on running in online mode.)

           This function causes the stencil solution to be run immediately
           until the auto-tuner converges on all ranks.
           It is useful for benchmarking, where performance is to be timed
           for a given number of steps after the best settings are found.
           This function should be called only *after* calling prepare_solution().
           This call must be made on each rank.

           @warning Modifies the contents of the YASK vars by automatically calling run_solution()
           an arbitrary number of times, but without halo exchanges.
           (See run_solution() for other restrictions and warnings.)
           Thus, var data should be set or reset *after* calling this function when
           used in a production or test setting where correct results are expected.
        */
        virtual void
        run_auto_tuner_now(bool verbose = true
                           /**< [in] If _true_, print progress information to the debug object
                              set via set_debug_output(). */ ) =0;

        /* Advanced APIs for yk_solution found below are not needed for most applications. */

        /// **[Advanced]** Set the minimum amount of padding for all vars.
        /**
           This sets the minimum number of elements in each var that is
           reserved outside of the rank domain in the given dimension.
           This padding area can be used for required halo areas.  At
           least the specified number of elements will be added to both
           sides, i.e., both "before" and "after" the domain.

           The *actual* padding size will be the largest of the following values,
           additionally rounded up based on the vector-folding dimensions,
           cache-line alignment, and/or extensions needed for wave-front tiles:
           - Halo size.
           - Value provided by any of the pad-size setting functions.

           Setting the minimum pad size is useful when an application needs
           to copy data back and forth between YASK vars and legacy C-style
           arrays that include a certain halo size that may be larger than
           the halo calculated by the YASK compiler. For example, for a
           given stencil problem, one or more YASK variables might need a
           halo of width 2 in the x dimension, but only 1 in the y dimension
           due to the stencil radii in the respective dimensions. However,
           an application might have an existing C-style array with halo
           data of width 2 in both x and y dimensions. By calling
           `set_min_pad_size("y", 2)`, all YASK vars will be created with
           padding widths of at least 2 in the y dimension, making it easier
           to copy data to and from the C-style arrays using 
           yk_var::get_elements_in_slice() and yk_var::set_elements_in_slice().

           The padding size cannot be changed after data storage
           has been allocated for a given var; attempted changes to the pad size for such
           vars will be ignored.

           Use yk_var::set_left_min_pad_size and yk_var::set_right_min_pad_size()
           for individual setting of each var.
           Call yk_var::get_left_pad_size() and yk_var::get_right_pad_size()
           to determine the actual padding sizes for a given var.
           See the "Detailed Description" for \ref yk_var for more information on var sizes.
           Padding is only allowed in the domain dimensions.
        */
        virtual void
        set_min_pad_size(const std::string& dim
                         /**< [in] Name of dimension to set.  Must
                            be one of the names from get_domain_dim_names(). */,
                         idx_t size
                         /**< [in] Elements in this `dim` applied
                            to both sides of the domain. */ ) =0;

        /// **[Advanced]** Get the minimum requested amount of padding for all vars.
        /**
           @note The actual padding for any given var may be greater than
           this minimum requested amount as described in set_min_pad_size().
           @returns Current setting of minimum amount of padding for all vars.
        */
        virtual idx_t
        get_min_pad_size(const std::string& dim
                         /**< [in] Name of dimension to get.  Must be one of
                            the names from get_domain_dim_names(). */) const =0;

        /// **[Advanced]** Add a new var to the solution.
        /**
           This is typically not needed because vars used by the stencils are pre-defined
           by the solution itself via the stencil compiler.
           However, a var may be created explicitly via this function
           in order to use it for purposes other than by the
           pre-defined stencils within the current solution.

           Vars created by this function will behave [mostly] like a pre-defined var.
           For example,
           - Step and domain dimensions must the same as those defined by
           yc_node_factory::new_step_index() and yc_node_factory::new_domain_index(),
           respectively.
           - For each domain dimension of the var,
           the new var's domain size will be the same as that returned by
           get_rank_domain_size().
           - Calls to set_rank_domain_size() will automatically resize the corresponding domain
           size in this var.
           - This var's first domain index in this rank will be determined
           by the position of this rank.
           - This var's initial padding size will be the same as that returned by
           get_min_pad_size().
           - After creating a new var, you can increase its padding
           sizes in the domain dimensions via yk_var::set_min_pad_size(),
           yk_solution::set_min_pad_size(), etc.
           - For step and misc dimensions, you can change the desired size
           yk_var::set_alloc_size().
           - Storage may be allocated via yk_var::alloc_storage() or
           yk_solution::prepare_solution().

           Some behaviors are different from pre-defined vars. For example,
           - You can create new "misc" dimensions during var creation simply
           by naming them in the `dims` argument. Any dimension name that is
           not a step or domain dimension will become a misc dimension,
           whether or not it was defined via yc_node_factory::new_misc_index().
           - Vars created via new_var() cannot be direct inputs or outputs of
           stencil equations. However, data in a var created via new_var()
           can be merged with a pre-defined var via yk_var::fuse_vars()
           if the vars are compatible.

           If you want a var that is not automatically resized based on the
           solution settings, use new_fixed_size_var() instead.

           @note A new var contains only the meta-data for the var; data storage
           is not yet allocated.
           Storage may be allocated in any of the methods listed
           in the "Detailed Description" for \ref yk_var.
           @returns Pointer to the new var.
        */
        virtual yk_var_ptr
        new_var(const std::string& name
                 /**< [in] Name of the var; must be unique
                    within the solution. */,
                 const string_vec& dims
                 /**< [in] List of names of all dimensions.
                    Names must be valid C++ identifiers and
                    not repeated within this var. */ ) =0;

#ifndef SWIG
        /// **[Advanced]** Add a new var to the solution.
        /**
           See documentation for the version of new_var() with a vector of dimension names
           as a parameter.
           @note This version is not available (or needed) in the Python API.
           @returns Pointer to the new var.
        */
        virtual yk_var_ptr
        new_var(const std::string& name
                 /**< [in] Name of the var; must be unique
                    within the solution. */,
                 const std::initializer_list<std::string>& dims
                 /**< [in] List of names of all dimensions.
                    Names must be valid C++ identifiers and
                    not repeated within this var. */ ) =0;
#endif

        /// **[Advanced]** Add a new var to the solution with a specified size.
        /**
           This is typically not needed because vars used by the stencils are pre-defined
           by the solution itself via the stencil compiler.
           However, a var may be created explicitly via this function
           in order to use it for purposes other than by the
           pre-defined stencils within the current solution.

           The following behaviors are different from both pre-defined vars
           and those created via new_var():
           - Calls to set_rank_domain_size() will *not* automatically resize
           the corresponding local-domain size in this var--this is where the term "fixed" applies.
           - In contrast, for each domain dimension of the var,
           the new var's local-domain size can be changed independently of the domain
           size of the solution.
           - This var's first domain index in this rank will be fixed at zero (0)
           in each domain dimension regardless of this rank's position.
           In other words, this var does not participate in "domain decomposition".
           - This var's padding size will be affected only by calls to
           yk_var::set_min_pad_size(), etc., i.e., *not* via
           yk_solution::set_min_pad_size().

           The following behaviors are the same as those of a pre-defined var
           and those created via new_var():
           - For step and misc dimensions, you can change the desired size
           yk_var::set_alloc_size().
           - Storage may be allocated via yk_var::alloc_storage() or
           yk_solution::prepare_solution().

           See yk_var::set_alloc_size().

           The following behaviors are different than a pre-defined var
           but the same as those created via new_var():
           - You can create new "misc" dimensions during var creation simply
           by naming them in the `dims` argument. Any dimension name that is
           not a step or domain dimension will become a misc dimension,
           whether or not it was defined via yc_node_factory::new_misc_index().
           - Vars created via new_fixed_size_var() cannot be direct inputs or outputs of
           stencil equations. However, data in a var created via new_fixed_size_var()
           can be shared with a pre-defined var via yk_var::fuse_vars()
           if the vars are compatible.

           @note A new var contains only the meta-data for the var; data storage
           is not yet allocated.
           Storage may be allocated in any of the methods listed
           in the "Detailed Description" for \ref yk_var.
           @returns Pointer to the new var.
        */
        virtual yk_var_ptr
        new_fixed_size_var(const std::string& name
                       /**< [in] Name of the var; must be unique
                          within the solution. */,
                       const string_vec& dims
                       /**< [in] List of names of all dimensions.
                          Names must be valid C++ identifiers and
                          not repeated within this var. */,
                       const idx_t_vec& dim_sizes
                       /**< [in] Initial allocation in each dimension.
                          Must be exatly one size for each dimension. */ ) =0;

#ifndef SWIG
        /// **[Advanced]** Add a new var to the solution with a specified size.
        /**
           See documentation for the version of new_fixed_size_var() with a vector of dimension names
           as a parameter.
           @note This version is not available (or needed) in the Python API.
           @returns Pointer to the new var.
        */
        virtual yk_var_ptr
        new_fixed_size_var(const std::string& name
                       /**< [in] Name of the var; must be unique
                          within the solution. */,
                       const std::initializer_list<std::string>& dims
                       /**< [in] List of names of all dimensions.
                          Names must be valid C++ identifiers and
                          not repeated within this var. */,
                       const idx_t_init_list& dim_sizes
                       /**< [in] Initial allocation in each dimension.
                          Must be exatly one size for each dimension. */ ) =0;
#endif

        /// **[Advanced]** Set the default preferred NUMA node on which to allocate data.
        /**
           This value is used when allocating vars and MPI buffers.
           The NUMA "preferred node allocation" policy is used, meaning that
           memory will be allocated in an alternative node if the preferred one
           doesn't have enough space available or is otherwise restricted.
           Instead of specifying a NUMA node, a special value may be used
           to specify another policy as listed.
           This setting may be overridden for any specific var.
           @returns `true` if NUMA preference was set;
           `false` if NUMA preferences are not enabled.
        */
        virtual bool
        set_default_numa_preferred(int numa_node
                                   /**< [in] Preferred NUMA node for data
                                      allocation.  Alternatively, use
                                      `yask_numa_local` for explicit
                                      local-node allocation,
                                      `yask_numa_interleave` for
                                      interleaving pages across all nodes,
                                      or `yask_numa_none` for no explicit NUMA
                                      policy. These constants are defined in
                                      the _Variable Documentation_ section of
                                      \ref yk_solution_api.hpp. */) =0;

        /// **[Advanced]** Get the default preferred NUMA node on which to allocate data.
        /**
           @returns Current setting of preferred NUMA node.
        */
        virtual int
        get_default_numa_preferred() const =0;

#ifndef SWIG
        /// **[Advanced]** Callback type with \ref yk_solution parameter.
        typedef std::function<void(yk_solution&)> hook_fn_t;
        
        /// **[Advanced]** Callback type with \ref yk_solution and step-index parameters.
        typedef std::function<void(yk_solution& soln,
                                   idx_t first_step_index,
                                   idx_t last_step_index)> hook_fn_2idx_t;

        /// **[Advanced]** Register a function to be called at the beginning of yk_solution::prepare_solution().
        /**
           A reference to the \ref yk_solution is passed to the `hook_fn`.

           If this method is called more than once, the hook functions will be
           called in the order registered.

           @note Not available in the Python API.
         */
        virtual void
        call_before_prepare_solution(hook_fn_t hook_fn
                                     /**< [in] callback function */) =0;

        /// **[Advanced]** Register a hook function to be called at the end of yk_solution::prepare_solution().
        /**
           A reference to the \ref yk_solution is passed to the `hook_fn`.

           If this method is called more than once, the hook functions will be
           called in the order registered.

           @note Not available in the Python API.
         */
        virtual void
        call_after_prepare_solution(hook_fn_t hook_fn
                                    /**< [in] callback function */) =0;

        /// **[Advanced]** Register a hook function to be called at the beginning of yk_solution::run_solution().
        /**
           A reference to the \ref yk_solution
           and the `first_step_index` and `last_step_index` passed to run_solution()
           are passed to the `hook_fn`.

           If this method is called more than once, the hook functions will be
           called in the order registered.

           @note Not available in the Python API.
         */
        virtual void
        call_before_run_solution(hook_fn_2idx_t hook_fn
                                 /**< [in] callback function */) =0;

        /// **[Advanced]** Register a hook function to be called at the end of yk_solution::run_solution().
        /**
           A reference to the \ref yk_solution
           and the `first_step_index` and `last_step_index` passed to run_solution()
           are passed to the `hook_fn`.

           If this method is called more than once, the hook functions will be
           called in the order registered.

           @note Not available in the Python API.
         */
        virtual void
        call_after_run_solution(hook_fn_2idx_t hook_fn
                                /**< [in] callback function */) =0;
#endif
        
        /// **[Advanced]** Merge YASK variables with another solution.
        /**
           Calls yk_var::fuse_vars() for each pair of vars that have the same name
           in this solution and the source solution.
           All conditions listed in yk_var::fuse_vars() must hold for each pair.
        */
        virtual void
        fuse_vars(yk_solution_ptr source
                   /**< [in] Solution from which vars will be merged. */) =0;

        /// **[Advanced]** Set whether invalid step indices alias to valid ones.
        virtual void
        set_step_wrap(bool do_wrap
                      /**< [in] Whether to allow any step index. */) =0;

        /// **[Advanced]** Get whether invalid step indices alias to valid ones.
        /**
           @returns Whether any step index is allowed.
        */
        virtual bool
        get_step_wrap() const =0;

        /// ***[Deprecated]*** Use yk_env::set_debug_output().
        YASK_DEPRECATED
        virtual void
        set_debug_output(yask_output_ptr debug) =0;

        /// **[Deprecated]** Use get_num_vars().
        YASK_DEPRECATED
        inline int
        get_num_grids() const {
            return get_num_vars();
        }

        /// **[Deprecated]** Use get_var().
        YASK_DEPRECATED
        inline yk_var_ptr
        get_grid(const std::string& name) {
            return get_var(name);
        }
                
        /// **[Deprecated]** Use get_vars().
        YASK_DEPRECATED
        inline std::vector<yk_var_ptr>
        get_grids() {
            return get_vars();
        }

        /// **[Deprecated]** Use new_var().
        YASK_DEPRECATED
        inline yk_var_ptr
        new_grid(const std::string& name,
                 const string_vec& dims) {
            return new_var(name, dims);
        }

#ifndef SWIG
        /// **[Deprecated]** Use new_var().
        YASK_DEPRECATED
        inline yk_var_ptr
        new_grid(const std::string& name,
                 const std::initializer_list<std::string>& dims) {
            return new_var(name, dims);
        }
#endif

        /// **[Deprecated]** Use new_fixed_size_var().
        YASK_DEPRECATED
        inline yk_var_ptr
        new_fixed_size_grid(const std::string& name,
                            const string_vec& dims,
                            const idx_t_vec& dim_sizes) {
            return new_fixed_size_var(name, dims, dim_sizes);
        }

#ifndef SWIG
        /// **[Deprecated]** Use new_fixed_size_var().
        YASK_DEPRECATED
        inline yk_var_ptr
        new_fixed_size_grid(const std::string& name,
                            const std::initializer_list<std::string>& dims,
                            const idx_t_vec& dim_sizes) {
            return new_fixed_size_var(name, dims, dim_sizes);
        }
#endif
        
        /// **[Deprecated]** Use fuse_vars().
        YASK_DEPRECATED
        inline void
        fuse_grids(yk_solution_ptr source) {
            fuse_vars(source);
        }
    };                          // yk_solution.

    /// Statistics from calls to run_solution().
    /**
       A throughput rate may be calculated by multiplying an
       amount-of-work-per-step quantity by the number of steps done and
       dividing by the number of seconds elapsed.
    */
    class yk_stats {
    public:
    	virtual ~yk_stats() {}

        /// Get the number of elements in the overall domain.
        /**
           @returns Product across all domain dimensions of the domain sizes across all ranks.
           Multiply this value by get_num_steps_done() to determine the number
           of points processed.
           Then, divide by get_elapsed_run_secs() to determine the throughput.
        */
        virtual idx_t
        get_num_elements() =0;

        /// Get the number of steps executed via run_solution().
        /**
           @returns A positive number, regardless of whether run_solution() steps were executed
           forward or backward.
        */
        virtual idx_t
        get_num_steps_done() =0;

        /// Get the number of elements written across all steps.
        /**
           @returns Number of elements written, summed over all output vars,
           steps executed, and ranks.
        */
        virtual idx_t
        get_num_writes_done() =0;

        /// Get the estimated number of floating-point operations executed across all steps.
        /**
           @returns Number of FP ops created by the stencil compiler, summed over
           all stencil-bundles, steps executed, and ranks.
           It may be slightly more or less than the actual number of FP ops executed
           by the CPU due to C++ compiler transformations.
        */
        virtual idx_t
        get_est_fp_ops_done() =0;

        /// Get the number of seconds elapsed during calls to run_solution().
        /**
           @returns Only the time spent in run_solution(), not in any other code in your
           application between calls.
        */
        virtual double
        get_elapsed_secs() =0;
    };                          // yk_stats.

    /** @}*/
} // namespace yask.
