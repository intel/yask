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

///////// API for the YASK stencil kernel. ////////////

// This file uses Doxygen 1.8 markup for API documentation-generation.
// See http://www.stack.nl/~dimitri/doxygen.
/** @file yask_kernel_api.hpp */ 

#ifndef YASK_KERNEL_API
#define YASK_KERNEL_API

#include "yask_common_api.hpp"
#include <vector>
#include <cinttypes>

namespace yask {

    /// Type to use for indexing grids.
    /** Index types are signed to allow negative indices in padding/halos. */
#ifdef SWIG
    typedef long int idx_t;     // SWIG doesn't seem to understand int64_t.
#else
    typedef std::int64_t idx_t;
#endif

    /// Allocate grids on local NUMA node.
    /**
       This is used in yk_solution::set_default_numa_preferred
       and yk_grid::set_numa_preferred.
       In Python, specify as `yask_kernel.cvar.yask_numa_local`. 
    */
    const int yask_numa_local = -1;

    /// Allocate grids across all available NUMA nodes.
    /**
       This is used in yk_solution::set_default_numa_preferred
       and yk_grid::set_numa_preferred.
       In Python, specify as `yask_kernel.cvar.yask_numa_interleave`. 
    */
    const int yask_numa_interleave = -2;

    /// Do not specify any NUMA binding.
    /**
       This is used in yk_solution::set_default_numa_preferred
       and yk_grid::set_numa_preferred.
       In Python, specify as `yask_kernel.cvar.yask_numa_none`. 
    */
    const int yask_numa_none = -9;

    // Forward declarations of classes and pointers.

    class yk_env;
    /// Shared pointer to \ref yk_env
    typedef std::shared_ptr<yk_env> yk_env_ptr;

    class yk_solution;
    /// Shared pointer to \ref yk_solution
    typedef std::shared_ptr<yk_solution> yk_solution_ptr;

    class yk_grid;
    /// Shared pointer to \ref yk_grid
    typedef std::shared_ptr<yk_grid> yk_grid_ptr;

    class yk_stats;
    /// Shared pointer to \ref yk_stats
    typedef std::shared_ptr<yk_stats> yk_stats_ptr;

    /// Factory to create a stencil solution.
    class yk_factory {
    public:
        virtual ~yk_factory() {}

        /// Version information.
        /**
           @returns String describing the current version.
        */
        virtual std::string
		get_version_string();

        /// Create an object to hold environment information.
        /**
           Initializes MPI if MPI is enabled.
           Environment info is kept in a separate object to factilitate
           initializing the environment before creating a solution
           and sharing an environment among multiple solutions.
           @returns Pointer to new env object.
        */
        virtual yk_env_ptr
        new_env() const;

        /// Create a stencil solution.
        /**
           A stencil solution contains all the grids and equations
           that were created during stencil compilation.
           @returns Pointer to new solution object. 
        */
        virtual yk_solution_ptr
        new_solution(yk_env_ptr env /**< [in] Pointer to env info. */) const;

        /// **[Advanced]** Create a stencil solution by copying the settings from another.
        /**
           All the settings that were specified via the `yk_solution::set_*()`
           functions in the source solution will be copied to the new solution.
           This does *not* copy any grids, grid settings, or grid data;
           see yk_solution::share_grid_storage().
           @returns Pointer to new solution object. 
        */
        virtual yk_solution_ptr
        new_solution(yk_env_ptr env /**< [in] Pointer to env info. */,
                     const yk_solution_ptr source
                     /**< [in] Pointer to existing \ref yk_solution from which
                        the settings will be copied. */ ) const;
    };

    /// Kernel environment.
    class yk_env {
    public:
        virtual ~yk_env() {}

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
    };

    /// Stencil solution as defined by the generated code from the YASK stencil compiler.
    /**
       Objects of this type contain all the grids and equations
       that comprise a solution.
    */
    class yk_solution {
    public:
        virtual ~yk_solution() {}

        /// Set object to receive debug output.
        virtual void
        set_debug_output(yask_output_ptr debug
                         /**< [out] Pointer to object to receive debug output. 
                            See \ref yask_output_factory. */ ) =0;

        /// Get the name of the solution.
        /**
           @returns String containing the solution name provided during stencil compilation.
        */
        virtual const std::string&
        get_name() const =0;

        /// Get the floating-point precision size.
        /**
           @returns Number of bytes in each FP element: 4 or 8.
        */
        virtual int 
        get_element_bytes() const =0;
        
        /// Get the solution step dimension.
        /**
           @returns String containing the step-dimension name. 
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
           @returns List of all domain-dimension names.
        */
        virtual std::vector<std::string>
        get_domain_dim_names() const =0;

        /// Get all the miscellaneous dimension names.
        /**
           @returns List of all dimension names used in the solution
           that are not step or domain dimensions.
        */
        virtual std::vector<std::string>
        get_misc_dim_names() const =0;

        /// Set the size of the solution domain for this rank.
        /**
           The domain defines the number of elements that will be evaluated with the stencil(s). 
           If MPI is not enabled, this is the entire problem domain.
           If MPI is enabled, this is the domain for the current rank only,
           and the problem domain consists of the sum of all rank domains
           in each dimension (weak-scaling).
           The domain size in each rank does not have to be the same, but
           all domains in the same column must have the same width,
           all domains in the same row must have the same height,
           and so forth, for each domain dimension.
           The domain size does *not* include the halo region or any padding.
           For best performance, set the rank domain
           size to a multiple of the number of elements in a vector-cluster in
           each dimension whenever possible.
           See the "Detailed Description" for \ref yk_grid for more information on grid sizes.
           There is no domain-size setting allowed in the
           solution-step dimension (usually "t"). 
        */
        virtual void
        set_rank_domain_size(const std::string& dim
                             /**< [in] Name of dimension to set.  Must be one of
                                the names from get_domain_dim_names(). */,
                             idx_t size /**< [in] Elements in the domain in this `dim`. */ ) =0;

        /// Get the domain size for this rank.
        /**
           @returns Current setting of rank domain size in specified dimension.
        */
        virtual idx_t
        get_rank_domain_size(const std::string& dim
                             /**< [in] Name of dimension to get.  Must be one of
                                the names from get_domain_dim_names(). */) const =0;

        /// Set the minimum amount of grid padding for all grids.
        /**
           This sets the minimum number of elements in each grid that is
           reserved outside of the rank domain in the given dimension.
           This padding area can be used for required halo regions.  At
           least the specified number of elements will be added to both
           sides, i.e., both "before" and "after" the domain.
           
           The *actual* padding size will be the largest of the following values,
           additionally rounded up based on the vector-folding dimensions
           and/or cache-line alignment:
           - Halo size.
           - Value provided by any of the pad-size setting functions.
           
           The padding size cannot be changed after data storage
           has been allocated for a given grid; attempted changes to the pad size for such
           grids will be ignored.
           In addition, once a grid's padding is set, it cannot be reduced, only increased.
           Call yk_grid::get_pad_size() to determine the actual padding size for a given grid.
           See the "Detailed Description" for \ref yk_grid for more information on grid sizes.
           There is no padding allowed in the solution-step dimension (usually "t").
        */
        virtual void
        set_min_pad_size(const std::string& dim
                         /**< [in] Name of dimension to set.  Must
                            be one of the names from get_domain_dim_names(). */,
                         idx_t size
                         /**< [in] Elements in this `dim` applied
                            to both sides of the domain. */ ) =0;

        /// Get the minimum amount of grid padding for all grids.
        /**
           @returns Current setting of minimum amount of grid padding for all grids.
        */
        virtual idx_t
        get_min_pad_size(const std::string& dim
                         /**< [in] Name of dimension to get.  Must be one of
                            the names from get_domain_dim_names(). */) const =0;

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
        */
        virtual void
        set_block_size(const std::string& dim
                       /**< [in] Name of dimension to set.  Must be one of
                          the names from get_domain_dim_names(). */,
                       idx_t size
                       /**< [in] Elements in a block in this `dim`. */ ) =0;

        /// Get the block size.
        /**
           Returned value may be slightly larger than the value provided
           via set_block_size() due to rounding.
           @returns Current settings of block size.
        */
        virtual idx_t
        get_block_size(const std::string& dim
                        /**< [in] Name of dimension to get.  Must be one of
                           the names from get_domain_dim_names(). */) const =0;

        /// Set the number of MPI ranks in the given dimension.
        /**
           The *product* of the number of ranks across all dimensions must
           equal yk_env::get_num_ranks().
           The curent MPI rank will be assigned a unique location 
           within the overall problem domain based on its MPI rank index.
           The same number of MPI ranks must be set via this API on each
           constituent MPI rank to ensure a consistent overall configuration.
           The number of ranks in each dimension must be properly set
           before calling yk_solution::prepare_solution().
           There is no rank setting allowed in the
           solution-step dimension (usually "t").
        */
        virtual void
        set_num_ranks(const std::string& dim
                      /**< [in] Name of dimension to set.  Must be one of
                         the names from get_domain_dim_names(). */,
                      idx_t num /**< [in] Number of ranks in `dim`. */ ) =0;

        /// Get the number of MPI ranks in the given dimension.
        /**
           @returns Current setting of rank size.
        */
        virtual idx_t
        get_num_ranks(const std::string& dim
                      /**< [in] Name of dimension to get.  Must be one of
                         the names from get_domain_dim_names(). */) const =0;

        /// Get the rank index in the specified dimension.
        /**
           The overall rank indices in the specified dimension will range from
           zero (0) to get_num_ranks() - 1, inclusive.
           @returns Zero-based index of this rank.
        */
        virtual idx_t
        get_rank_index(const std::string& dim
                       /**< [in] Name of dimension to get.  Must be one of
                         the names from get_domain_dim_names(). */ ) const =0;

        /// Get the number of grids in the solution.
        /**
           Grids may be pre-defined by the stencil compiler
           (e.g., via yc_solution::new_grid())
           or created explicitly via yk_solution::new_grid().
           @returns Number of grids that have been created.
        */
        virtual int
        get_num_grids() const =0;
        
        /// Get the specified grid.
        /**
           This cannot be used to access scratch grids.
           @returns Pointer to the specified grid or null pointer if it does not exist.
        */
        virtual yk_grid_ptr
        get_grid(const std::string& name /**< [in] Name of the grid. */ ) =0;

        /// Get all the grids.
        /**
           @returns List of all non-scratch grids in the solution.
        */
        virtual std::vector<yk_grid_ptr>
        get_grids() =0;

        /// Prepare the solution for stencil application.
        /**
           Allocates data in grids that do not already have storage allocated.
           Calculates the position of each rank in the overall problem domain.
           Sets many other data structures needed for proper stencil application.
           Since this function initiates MPI communication, it must be called
           on all MPI ranks, and it will block until all ranks have completed.
           Must be called before applying any stencils.
        */
        virtual void
        prepare_solution() =0;

        /// Get the first index of the sub-domain in this rank in the specified dimension.
        /**
           This returns the first *overall* index at the beginning of the domain.
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

        /// Get the overall problem size in the specified dimension.
        /**
           The overall domain indices in the specified dimension will range from
           zero (0) to get_overall_domain_size() - 1, inclusive.
           Call get_first_rank_domain_index() and get_last_rank_domain_index()
           to find the subset of this domain in each rank.

           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() obtains the sub-domain sizes from other ranks.
           @returns Sum of all ranks' domain sizes in the given dimension.
        */
        virtual idx_t
        get_overall_domain_size(const std::string& dim
                                /**< [in] Name of dimension to get.  Must be one of
                                   the names from get_domain_dim_names(). */ ) const =0;

        /// Run the stencil solution for the specified steps.
        /**
           The stencil(s) in the solution are applied to the grid data, setting the
           index variables as follows:
           1. If temporal wave-fronts are *not* used (the default):
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
            - MPI halo exchanges will occur as necessary before, after, or during a step.
            - Since this function initiates MPI communication, it must be called
              on all MPI ranks, and it will block until all ranks have completed.
           2. **[Advanced]** If temporal wave-fronts *are* enabled (currently only possible via apply_command_line_options()):
            - The step index (e.g., `t` for "time") will be sequentially set to values
            from `first_step_index` to `last_step_index`, inclusive, within each wave-front tile.
             + The number of steps in a wave-front tile may also be restricted by the size
             of the tile in the step dimension. In that case, tiles will be done in slices of that size.
             + Reverse solutions are not allowed with wave-front tiling.
            - For each step index within each wave-front tile, the domain indices will be set
            to values across the entire tile (not necessarily sequentially).
            - Ultimately, the stencil(s) will be applied to same the elements in both the step 
            and domain dimensions as when wave-front tiling is not used.
            - MPI is not supported with wave-front tiling.

           This function should be called only *after* calling prepare_solution().
        */
        virtual void
        run_solution(idx_t first_step_index /**< [in] First index in the step dimension */,
                     idx_t last_step_index /**< [in] Last index in the step dimension */ ) =0;

        /// Run the stencil solution for the specified step.
        /**
           This function is simply an alias for `run_solution(step_index, step_index)`, i.e.,
           the solution will be applied for exactly one step across the domain.

           Typical C++ usage:

           ~~~{.cpp}
           for (idx_t t = 1; t <= num_steps; t++)
               run_solution(t);
           ~~~

           As written, the above loop is identical to

           ~~~{.cpp}
           run_solution(1, num_steps);
           ~~~

           @note The parameter is *not* the number of steps to run.
           @note Since only one step is taken per call, using this function effectively disables
           wave-front tiling.
        */
        virtual void
        run_solution(idx_t step_index /**< [in] Index in the step dimension */ ) =0;

        /// Finish using a solution.
        /**
           Releases shared ownership of memory used by the grids.  This will
           result in deallocating each memory block whose ownership is not
           shared by another shared pointer.
        */
        virtual void
        end_solution() =0;


        /// Get performance statistics associated with preceding calls to run_solution().
        /**
           Side effect: resets all statistics, so a subsequent call will
           measure performance after the current call.
           @returns Pointer to statistics object.
        */
        virtual yk_stats_ptr
        get_stats() =0;

        /// Determine whether the auto-tuner is enabled on this rank.
        /**
           The auto-tuner is enabled by default.
           It will become disabled after it has converged or after reset_auto_tuner(false) has been called.
           @returns Whether the auto-tuner is still searching.
        */
        virtual bool
        is_auto_tuner_enabled() =0;

        /* Advanced APIs for yk_solution found below are not needed for most applications. */
        
        /// **[Advanced]** Restart or disable the auto-tuner on this rank.
        /**
           Under normal operation, an auto-tuner is invoked automatically during calls to
           run_solution().
           Currently, only the block size is set by the auto-tuner, and the search begins from the 
           sizes set via set_block_size() or the default size if set_block_size() has
           not been called.
           This function is used to apply the current best-known settings if the tuner has
           been running, reset the state of the auto-tuner, and either
           restart its search or disable it from running.
           This call must be made on each rank where the change is desired.
        */
        virtual void
        reset_auto_tuner(bool enable
                         /**< [in] If _true_, start or restart the auto-tuner search.
                            If _false_, disable the auto-tuner from running. */,
                         bool verbose = false
                         /**< [in] If _true_, print progress information to the debug object
                            set via set_debug_output(). */ ) =0;

        /// **[Advanced]** Automatically tune selected settings immediately.
        /**
           Executes a search algorithm to find [locally] optimum values for some of the
           settings.
           Under normal operation, an auto-tuner is invoked during calls to
           run_solution().
           See reset_auto_tuner() for more information.
           This function causes the stencil solution to be run immediately
           until the auto-tuner converges on all ranks.
           It is useful for benchmarking, where performance is to be timed
           for a given number of steps after the best settings are found.
           This function should be called only *after* calling prepare_solution().
           This call must be made on each rank.
           @warning Modifies the contents of the grids by calling run_solution()
           an arbitrary number of times, but without halo exchange.
           (See run_solution() for other restrictions and warnings.)
           Thus, grid data should be set *after* calling this function when
           used in a production or test setting where correct results are expected.
        */
        virtual void
        run_auto_tuner_now(bool verbose = true
                           /**< [in] If _true_, print progress information to the debug object
                              set via set_debug_output(). */ ) =0;
        
        /// **[Advanced]** Add a new grid to the solution.
        /**
           This is typically not needed because grids used by the stencils are pre-defined
           by the solution itself via the stencil compiler.
           However, a grid may be created explicitly via this function
           in order to use it for purposes other than by the
           pre-defined stencils within the current solution.

           Grids created by this function will be treated like a pre-defined grid.
           For example,
           - For each domain dimension of the grid,
           the new grid's domain size will be the same as that returned by
           get_rank_domain_size().
           - Calls to set_rank_domain_size() will resize the corresponding domain 
           size in this grid.
           - This grid's first domain index in this rank will be determined
           by the position of this rank.
           - This grid's initial padding size will be the same as that returned by
           get_min_pad_size().
           - After creating a new grid, you can increase its padding
           sizes in the domain dimensions via yk_grid::set_min_pad_size(), etc.
           - For step and misc dimensions, you can change the allocation via
           yk_grid::set_alloc_size().

           If you want a grid that is not automatically resized based on the
           solution settings, use new_fixed_size_grid() instead.

           @note A new grid contains only the meta-data for the grid; data storage
           is not yet allocated.
           Storage may be allocated in any of the methods listed
           in the "Detailed Description" for \ref yk_grid.
           @returns Pointer to the new grid.
        */
        virtual yk_grid_ptr
        new_grid(const std::string& name
                 /**< [in] Name of the grid; must be unique
                    within the solution. */,
                 const std::vector<std::string>& dims
                 /**< [in] List of names of all dimensions. 
                    Names must be valid C++ identifiers and 
                    not repeated within this grid. */ ) =0;

#ifndef SWIG
        /// **[Advanced]** Add a new grid to the solution.
        /**
           See documentation for the version of new_grid() with a vector of dimension names
           as a parameter.
           @note This version is not available (or needed) in SWIG-based APIs, e.g., Python.
           @returns Pointer to the new grid.
        */
        virtual yk_grid_ptr
        new_grid(const std::string& name
                 /**< [in] Name of the grid; must be unique
                    within the solution. */,
                 const std::initializer_list<std::string>& dims
                 /**< [in] List of names of all dimensions. 
                    Names must be valid C++ identifiers and 
                    not repeated within this grid. */ ) =0;
#endif

        /// **[Advanced]** Add a new grid to the solution with a specified size.
        /**
           This is typically not needed because grids used by the stencils are pre-defined
           by the solution itself via the stencil compiler.
           However, a grid may be created explicitly via this function
           in order to use it for purposes other than by the
           pre-defined stencils within the current solution.

           Unlike new_grid(),
           grids created by this function will *not* be treated like a pre-defined grid.
           For example,
           - For each domain dimension of the grid,
           the new grid's domain size is provided during creation and cannot be changed.
           - Calls to set_rank_domain_size() will *not* resize the corresponding domain 
           size in this grid.
           - This grid's first domain index in this rank will be fixed at zero (0)
           regardless of this rank's position.
           - This grid's padding size will be affected only by calls to 
           yk_grid::set_min_pad_size(), etc.
           - For step and misc dimensions, you can still change the allocation via
           yk_grid::set_alloc_size().

           @note A new grid contains only the meta-data for the grid; data storage
           is not yet allocated.
           Storage may be allocated in any of the methods listed
           in the "Detailed Description" for \ref yk_grid.
           @returns Pointer to the new grid.
        */
        virtual yk_grid_ptr
        new_fixed_size_grid(const std::string& name
                       /**< [in] Name of the grid; must be unique
                          within the solution. */,
                       const std::vector<std::string>& dims
                       /**< [in] List of names of all dimensions. 
                          Names must be valid C++ identifiers and 
                          not repeated within this grid. */,
                       const std::vector<idx_t>& dim_sizes
                       /**< [in] Initial allocation in each dimension.
                          Must be exatly one size for each dimension. */ ) =0;

#ifndef SWIG
        /// **[Advanced]** Add a new grid to the solution with a specified size.
        /**
           See documentation for the version of new_fixed_size_grid() with a vector of dimension names
           as a parameter.
           @note This version is not available (or needed) in SWIG-based APIs, e.g., Python.
           @returns Pointer to the new grid.
        */
        virtual yk_grid_ptr
        new_fixed_size_grid(const std::string& name
                       /**< [in] Name of the grid; must be unique
                          within the solution. */,
                       const std::initializer_list<std::string>& dims
                       /**< [in] List of names of all dimensions. 
                          Names must be valid C++ identifiers and 
                          not repeated within this grid. */,
                       const std::initializer_list<idx_t>& dim_sizes
                       /**< [in] Initial allocation in each dimension.
                          Must be exatly one size for each dimension. */ ) =0;
#endif

        /// **[Advanced]** Set the default preferred NUMA node on which to allocate data.
        /**
           This value is used when allocating grids and MPI buffers.
           The NUMA "preferred node allocation" policy is used, meaning that
           memory will be allocated in an alternative node if the preferred one
           doesn't have enough space available or is otherwise restricted.
           Instead of specifying a NUMA node, a special value may be used
           to specify another policy as listed.
           This setting may be overridden for any specific grid.
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
                                      \ref yask_kernel_api.hpp. */) =0;

        /// **[Advanced]** Get the default preferred NUMA node on which to allocate data.
        /**
           @returns Current setting of preferred NUMA node.
        */
        virtual int
        get_default_numa_preferred() const =0;

        /// **[Advanced]** Set performance parameters from an option string.
        /**
           Parses the string for options as if from a command-line.
           Example: "-bx 64 -block_threads 4" sets the block-size in the *x*
           dimension to 64 and the number of threads used to process each
           block to 4.
           See the help message from the YASK kernel binary for documentation
           on the command-line options.

           @returns Any strings that were not recognized by the parser as options.
        */
        virtual std::string
        apply_command_line_options(const std::string& args
                                   /**< [in] String of arguments to parse. */ ) =0;

        /// **[Advanced]** Use data-storage from existing grids in specified solution.
        /**
           Calls yk_grid::share_storage() for each pair of grids that have the same name
           in this solution and the source solution.
           All conditions listed in yk_grid::share_storage() must hold for each pair.
        */
        virtual void
        share_grid_storage(yk_solution_ptr source
                           /**< [in] Solution from which grid storage will be shared. */) =0;
    };

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
           @returns Product of all the overal domain sizes across all domain dimensions.
        */
        virtual idx_t
        get_num_elements() =0;

        /// Get the number of elements written in each step.
        /**
           @returns Number of elements written to each output grid.
           This is the same value as get_num_elements() if there is only one output grid.
        */
        virtual idx_t
        get_num_writes() =0;

        /// Get the estimated number of floating-point operations required for each step.
        /**
           @returns Number of FP ops created by the stencil compiler.
           It may be slightly more or less than the actual number of FP ops executed 
           by the CPU due to C++ compiler transformations.
        */
        virtual idx_t
        get_est_fp_ops() =0;

        /// Get the number of steps calculated via run_solution().
        /**
           @returns A positive number, regardless of whether run_solution() steps were executed
           forward or backward.
        */
        virtual idx_t
        get_num_steps_done() =0;

        /// Get the number of seconds elapsed during calls to run_solution().
        /**
           @returns Only the time spent in run_solution(), not in any other code in your
           application between calls.
        */
        virtual double
        get_elapsed_run_secs() =0;
    };
    
    /// A run-time grid.
    /**
       "Grid" is a generic term for any n-dimensional array.  A 0-dim grid
       is a scalar, a 1-dim grid is an array, etc.  A run-time grid contains
       data, unlike yc_grid, a compile-time grid variable.  
       
       Typically, access to each grid is obtained via yk_solution::get_grid().
       You may also use yk_solution::new_grid() or yk_solution::new_fixed_size_grid() 
       if you need a grid that is not part of the pre-defined solution.

       Each dimension of a grid is one of the following:
       - The *step* dimension, typically time ("t"), as identified via yk_solution::get_step_dim_name().
       - A *domain* dimension, typically a spatial dimension such as "x" or "y",
       as identified via yk_solution:get_domain_dim_names().
       - A *miscellaneous* dimension, which is any dimension that is not a domain or step dimension,
       as identified via yk_solution:get_misc_dim_names().
       
       In the step dimension, there is no fixed domain size, and no
       specified first or last index.
       However, there is an allocation size, which is the number of values in the step
       dimension that are stored in memory.
       Step-dimension indices "wrap-around" within this allocation to reuse memory.
       For example, if the step dimension is "t", and the t-dimension allocation size is 3,
       then t=-2, t=0, t=3, t=6, ..., t=303, etc. would all alias to the same spatial values in memory.

       In each domain dimension,
       grid sizes include the following components:
       - The *domain* is the elements to which the stencils are applied.
       - The *left padding* is all the elements before the domain and includes the left halo.
       - The *right padding* is all the elements before the domain and includes the right halo.
       - The *left halo* is the elements just before the domain which must be
       copied between preceding ranks during halo exchanges. The left halo is contained within the left padding.
       - The *right halo* is the elements just after the domain which must be
       copied between following ranks during halo exchanges. The right halo is contained within the right padding.
       - The *extra left padding* is the elements before the domain and left halo
       and thus does not include the left halo.
       - The *extra right padding* is the elements after the domain and right halo
       and thus does not include the right halo.
       - The *allocation* includes the left padding, domain, and right padding.
       
       Domain sizes specified via yk_solution::set_rank_domain_size() apply to each MPI rank.
       Visually, in each of the domain dimensions, these sizes are related as follows
       in each rank:
       <table>
       <tr><td>extra left padding <td>left halo <td rowspan="2">domain <td>right halo <td>extra right padding
       <tr><td colspan="2"><center>left padding</center> <td colspan="2"><center>right padding</center>
       <tr><td colspan="5"><center>allocation</center>
       </table>

       If MPI is not enabled, a rank's domain is equivalent to the entire problem size.
       If MPI is enabled, the domains of the ranks are logically abutted to create the 
       overall problem domain in each dimension:
       <table>
       <tr><td>extra left padding of rank A <td>halo of rank A <td>domain of rank A <td>domain of rank B
         <td>... <td>domain of rank Z <td>halo of rank Z <td>extra right padding of rank Z
       <tr><td colspan="2"><center>left padding of rank A</center>
         <td colspan="4"><center>overall problem domain</center>
         <td colspan="2"><center>right padding of rank Z</center>
       </table>
       The intermediate halos and paddings also exist, but are not shown in the above diagram.
       The halos overlap the domains of adjacent ranks.
       For example, the left halo of rank B in the diagram would overlap the domain of rank A.
       Data in these overlapped regions is exchanged as needed during stencil application
       to maintain a consistent values as if there was only one rank.

       In each miscellaneous dimension, there is only an allocation size,
       and there is no wrap-around as in the step dimension.
       Each index must be between its first and last allowed value.

       All sizes are expressed in numbers of elements.
       Each element may be a 4-byte (single precision)
       or 8-byte (double precision) floating-point value as returned by
       yk_solution::get_element_bytes().
       
       Initially, a grid is not assigned any allocated storage.
       This is done to allow modification of domain, padding, and other allocation sizes
       before allocation.
       Once the allocation sizes have been set in all dimensions, the data storage itself may
       be allocated.
       This can be done in any of the following ways:
       - Storage for all grids without data storage will be automatically allocated when
       prepare_solution() is called.
       - Storage for a specific grid may be allocated before calling prepare_solution()
       via yk_grid::alloc_storage().
       - **[Advanced]** Storage for a specific grid may be shared with another grid with
       existing storage via yk_grid::share_storage().
       
       @note The domain index arguments to the \ref yk_grid functions that require indices
       are *always* relative to the overall problem; they are *not* relative to the current rank.
       The first and last overall-problem index that lies within a rank can be
       retrieved via yk_solution::get_first_rank_domain_index() and 
       yk_solution::get_last_rank_domain_index(), respectively.
       The first and last accessible index that lies within a rank for a given grid can be
       retrieved via yk_grid::get_first_rank_alloc_index() and 
       yk_grid::get_last_rank_alloc_index(), respectively.
       Also, index arguments are always inclusive. 
       Specifically, for functions that return or require a "last" index, that
       index indicates the last one in the relevant range, i.e., *not* one past the last value
       (this is more like Fortran and Perl than Python and Lisp).
    */
    class yk_grid {
    public:
        virtual ~yk_grid() {}

        /// Get the name of the grid.
        /**
           @returns String containing name provided via yc_solution::new_grid().
        */
        virtual const std::string& get_name() const =0;

        /// Determine whether this grid is automatically resized based on the solution.
        /**
           @returns `true` if this grid was created via yk_solution::new_fixed_size_grid()
           or `false` otherwise.
        */
        virtual bool is_fixed_size() const =0;

        /// Get the number of dimensions used in this grid.
        /**
           This may include domain, step, and/or miscellaneous dimensions.
           @returns Number of dimensions created via yc_solution::new_grid(),
           yk_solution::new_grid(), or yk_solution::new_fixed_size_grid().
        */
        virtual int get_num_dims() const =0;

        /// Get all the dimensions in this grid.
        /**
           This may include domain, step, and/or miscellaneous dimensions.
           @returns List of names of all the dimensions.
        */
        virtual std::vector<std::string>
        get_dim_names() const =0;
        
        /// Determine whether specified dimension exists in this grid.
        /**
           @returns `true` if dimension exists (including step-dimension),
           `false` otherwise.
        */
        virtual bool
        is_dim_used(const std::string& dim) const =0;

        /// Get the domain size for this rank.
        /**
           @returns The same value as yk_solution::get_rank_domain_size() if
           is_fixed_size() returns `false` or the fixed sized provided via
           yk_solution::new_fixed_size_grid() otherwise.
        */
        virtual idx_t
        get_rank_domain_size(const std::string& dim
                             /**< [in] Name of dimension to get.  Must be one of
                                the names from yk_solution::get_domain_dim_names(). */) const =0;

        /// Get the first index of the sub-domain in this rank in the specified dimension.
        /**
           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns The same value as yk_solution::get_first_rank_domain_index() if
           is_fixed_size() returns `false` or zero (0) otherwise.
        */
        virtual idx_t
        get_first_rank_domain_index(const std::string& dim
                                    /**< [in] Name of dimension to get.  Must be one of
                                       the names from yk_solution::get_domain_dim_names(). */ ) const =0;
        
        /// Get the last index of the sub-domain in this rank in the specified dimension.
        /**
           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns The same value as yk_solution::get_last_rank_domain_index() if
           is_fixed_size() returns `false` or one less than the fixed sized provided via
           yk_solution::new_fixed_size_grid() otherwise.
        */
        virtual idx_t
        get_last_rank_domain_index(const std::string& dim
                                    /**< [in] Name of dimension to get.  Must be one of
                                       the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the left halo size in the specified dimension.
        /**
           This value is typically set by the stencil compiler.
           @returns Elements in halo in given dimension before the domain.
        */
        virtual idx_t
        get_left_halo_size(const std::string& dim
                      /**< [in] Name of dimension to get.
                         Must be one of
                         the names from yk_solution::get_domain_dim_names(). */ ) const =0;
        
        /// Get the right halo size in the specified dimension.
        /**
           This value is typically set by the stencil compiler.
           @returns Elements in halo in given dimension after the domain.
        */
        virtual idx_t
        get_right_halo_size(const std::string& dim
                      /**< [in] Name of dimension to get.
                         Must be one of
                         the names from yk_solution::get_domain_dim_names(). */ ) const =0;
        
        /// Get the first index of the left halo in this rank in the specified dimension.
        /**
           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns The first index of left halo in this rank or
           the same value as yk_grid::get_first_rank_domain_index()
           if the left halo has zero size.
        */
        virtual idx_t
        get_first_rank_halo_index(const std::string& dim
                                    /**< [in] Name of dimension to get.  Must be one of
                                       the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the last index of the right halo in this rank in the specified dimension.
        /**
           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns The last index of right halo in this rank or
           the same value as yk_grid::get_last_rank_domain_index()
           if the right halo has zero size.
        */
        virtual idx_t
        get_last_rank_halo_index(const std::string& dim
                                    /**< [in] Name of dimension to get.  Must be one of
                                       the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the left padding in the specified dimension.
        /**
           The left padding is the memory allocated before
           the domain in a given dimension.
           The left padding size includes the left halo size.
           The value may be slightly
           larger than that provided via set_min_pad_size(), etc. due to rounding.
           @returns Elements in left padding in given dimension.
        */
        virtual idx_t
        get_left_pad_size(const std::string& dim
                     /**< [in] Name of dimension to get.
                         Must be one of
                         the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the right padding in the specified dimension.
        /**
           The right padding is the memory allocated after
           the domain in a given dimension.
           The right padding size includes the right halo size.
           The value may be slightly
           larger than that provided via set_min_pad_size(), etc. due to rounding.
           @returns Elements in right padding in given dimension.
        */
        virtual idx_t
        get_right_pad_size(const std::string& dim
                     /**< [in] Name of dimension to get.
                         Must be one of
                         the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the extra left padding in the specified dimension.
        /**
           The *extra* padding size is the left padding size minus the left halo size.
           @returns Elements in padding in given dimension before the
           left halo region.
        */
        virtual idx_t
        get_left_extra_pad_size(const std::string& dim
                           /**< [in] Name of dimension to get.
                              Must be one of
                              the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the extra right padding in the specified dimension.
        /**
           The *extra* padding size is the right padding size minus the right halo size.
           @returns Elements in padding in given dimension after the
           right halo region.
        */
        virtual idx_t
        get_right_extra_pad_size(const std::string& dim
                           /**< [in] Name of dimension to get.
                              Must be one of
                              the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Set the padding in the specified dimension.
        /**
           This sets the minimum number of elements in this grid 
           in both left and right pads.
           This padding area can be used for required halo regions.
           
           The *actual* padding size will be the largest of the following values,
           additionally rounded up based on the vector-folding dimensions
           and/or cache-line alignment:
           - Halo size.
           - Value provided by any of the pad-size setting functions.
           
           The padding size cannot be changed after data storage
           has been allocated for this grid; attempted changes to the pad size
           will be ignored.
           In addition, once a grid's padding is set, it cannot be reduced, only increased.
           Call get_pad_size() to determine the actual padding size for the grid.
           See the "Detailed Description" for \ref yk_grid for information on grid sizes.
        */
        virtual void
        set_min_pad_size(const std::string& dim
                         /**< [in] Name of dimension to set.
                            Must be one of
                            the names from yk_solution::get_domain_dim_names(). */,
                         idx_t size
                         /**< [in] Minimum number of elements to allocate beyond the domain size. */ ) =0;
        
        /// Get the storage allocation in the specified dimension.
        /**
           For the step dimension, this is the specified allocation and
           does not typically depend on the number of steps evaluated.
           For the non-step dimensions, this includes the domain and padding sizes.
           See the "Detailed Description" for \ref yk_grid for information on grid sizes.
           @returns allocation in number of elements (not bytes).
        */
        virtual idx_t
        get_alloc_size(const std::string& dim
                       /**< [in] Name of dimension to get. */ ) const =0;

        /// Get the first index of a specified miscellaneous dimension.
        /**
           @returns the first allowed index in a non-step and non-domain dimension.
        */
        virtual idx_t
        get_first_misc_index(const std::string& dim
                             /**< [in] Name of dimension to get.  Must be one of
                                the names from yk_solution::get_misc_dim_names(). */ ) const =0;
        
        /// Get the last index of a specified miscellaneous dimension.
        /**
           @returns the last allowed index in a non-step and non-domain dimension.
        */
        virtual idx_t
        get_last_misc_index(const std::string& dim
                            /**< [in] Name of dimension to get.  Must be one of
                               the names from yk_solution::get_misc_dim_names(). */ ) const =0;

        /// Determine whether the given indices are allocated in this rank.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           @returns `true` if index values fall within the allocated space as returned by
           get_first_rank_alloc_index() and get_last_rank_alloc_index() for
           each dimension; `false` otherwise.
        */
        virtual bool
        is_element_allocated(const std::vector<idx_t>& indices
                             /**< [in] List of indices, one for each grid dimension. */ ) const =0;
        
#ifndef SWIG
        /// Determine whether the given indices are allocated in this rank.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           @note This version is not available (or needed) in SWIG-based APIs, e.g., Python.
           @returns `true` if index values fall within the allocated space as returned by
           get_first_rank_alloc_index() and get_last_rank_alloc_index() for
           each dimension; `false` otherwise.
        */
        virtual bool
        is_element_allocated(const std::initializer_list<idx_t>& indices
                             /**< [in] List of indices, one for each grid dimension. */ ) const =0;
#endif
        
        /// Get the value of one grid element.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall within the allocated space as returned by
           get_first_rank_alloc_index() and get_last_rank_alloc_index() for
           each dimension.
           @returns value in grid at given multi-dimensional location.
        */
        virtual double
        get_element(const std::vector<idx_t>& indices
                    /**< [in] List of indices, one for each grid dimension. */ ) const =0;

#ifndef SWIG
        /// Get the value of one grid element.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall within the allocated space as returned by
           get_first_rank_alloc_index() and get_last_rank_alloc_index() for
           each dimension.
           @note The return value is a double-precision floating-point value, but
           it will be converted from a single-precision if 
           yk_solution::get_element_bytes() returns 4.
           @note This version is not available (or needed) in SWIG-based APIs, e.g., Python.
           @returns value in grid at given multi-dimensional location.
        */
        virtual double
        get_element(const std::initializer_list<idx_t>& indices
                    /**< [in] List of indices, one for each grid dimension. */ ) const =0;
#endif

        /// Get grid elements within specified subset of the grid.
        /**
           Reads all elements from `first_indices` to `last_indices` in each dimension
           and writes them to consecutive memory locations in the buffer.
           Indices in the buffer progress in row-major order.
           The buffer pointed to must contain the number of bytes equal to
           yk_solution::get_element_bytes() multiplied by the number of
           elements in the specified slice.
           Since the reads proceed in row-major order, the last index is "unit-stride"
           in the buffer.
           Provide indices in two lists in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall within the allocated space as returned by
           get_first_rank_alloc_index() and get_last_rank_alloc_index() for
           each dimension.
           @returns Number of elements read.
        */
        virtual idx_t
        get_elements_in_slice(void* buffer_ptr
                              /**< [out] Pointer to buffer where values will be written. */,
                              const std::vector<idx_t>& first_indices
                              /**< [in] List of initial indices, one for each grid dimension. */,
                              const std::vector<idx_t>& last_indices
                              /**< [in] List of final indices, one for each grid dimension. */ ) const =0;
        
        /// Set the value of one grid element.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall within the allocated space as returned by
           get_first_rank_alloc_index() and get_last_rank_alloc_index() for
           each dimension.
           @note The parameter value is a double-precision floating-point value, but
           it will be converted to single-precision if
           yk_solution::get_element_bytes() returns 4.
           If storage has not been allocated for this grid, this will have no effect.
           @returns Number of elements set.
        */
        virtual idx_t
        set_element(double val /**< [in] Element in grid will be set to this. */,
                    const std::vector<idx_t>& indices
                    /**< [in] List of indices, one for each grid dimension. */,
                    bool strict_indices = false
                    /**< [in] If true, indices must be within domain or padding.
                       If false, indices outside of domain and padding result
                       in no change to grid. */ ) =0;

#ifndef SWIG        
        /// Set the value of one grid element.
        /**
           Provide the number of indices equal to the number of dimensions in the grid.
           Indices beyond that will be ignored.
           Indices are relative to the *overall* problem domain.
           If any index values fall outside of the allocated space as returned by
           get_first_rank_alloc_index() and get_last_rank_alloc_index() for
           each dimension, this will have no effect.
           @note The parameter value is a double-precision floating-point value, but
           it will be converted to single-precision if
           yk_solution::get_element_bytes() returns 4.
           If storage has not been allocated for this grid, this will have no effect.
           @note This version is not available (or needed) in SWIG-based APIs, e.g., Python.
           @returns Number of elements set.
        */
        virtual idx_t
        set_element(double val /**< [in] Element in grid will be set to this. */,
                    const std::initializer_list<idx_t>& indices
                    /**< [in] List of indices, one for each grid dimension. */,
                    bool strict_indices = false
                    /**< [in] If true, indices must be within domain or padding.
                       If false, indices outside of domain and padding result
                       in no change to grid. */ ) =0;
#endif
        /// Atomically add to the value of one grid element.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall within the allocated space as returned by
           get_first_rank_alloc_index() and get_last_rank_alloc_index() for
           each dimension if `strict_indices` is set to true.
           Updates are OpenMP atomic, meaning that this function can be called by
           several OpenMP threads without causing a race condition.
           @note The parameter value is a double-precision floating-point value, but
           it will be converted to single-precision if
           yk_solution::get_element_bytes() returns 4.
           If storage has not been allocated for this grid, this will have no effect.
           @returns Number of elements updated.
        */
        virtual idx_t
        add_to_element(double val /**< [in] This value will be added to element in grid. */,
                       const std::vector<idx_t>& indices
                       /**< [in] List of indices, one for each grid dimension. */,
                       bool strict_indices = false
                       /**< [in] If true, indices must be within domain or padding.
                          If false, indices outside of domain and padding result
                          in no change to grid. */ ) =0;

#ifndef SWIG        
        /// Atomically add to the value of one grid element.
        /**
           Provide the number of indices equal to the number of dimensions in the grid.
           Indices beyond that will be ignored.
           Indices are relative to the *overall* problem domain.
           Index values must fall within the allocated space as returned by
           get_first_rank_alloc_index() and get_last_rank_alloc_index() for
           each dimension if `strict_indices` is set to true.
           Updates are OpenMP atomic, meaning that this function can be called by
           several OpenMP threads without causing a race condition.
           @note The parameter value is a double-precision floating-point value, but
           it will be converted to single-precision if
           yk_solution::get_element_bytes() returns 4.
           If storage has not been allocated for this grid, this will have no effect.
           @note This version is not available (or needed) in SWIG-based APIs, e.g., Python.
           @returns Number of elements set.
        */
        virtual idx_t
        add_to_element(double val /**< [in] This value will be added to element in grid. */,
                       const std::initializer_list<idx_t>& indices
                       /**< [in] List of indices, one for each grid dimension. */,
                       bool strict_indices = false
                       /**< [in] If true, indices must be within domain or padding.
                          If false, indices outside of domain and padding result
                          in no change to grid. */ ) =0;
#endif
        
        /// Initialize all grid elements to the same value.
        /**
           Sets all allocated elements, including those in the domain and padding
           area to the same specified value.
           @note The parameter is a double-precision floating-point value, but
           it will be converted to single-precision if
           yk_solution::get_element_bytes() returns 4.
           @note If storage has not been allocated via yk_solution::prepare_solution(),
           this will have no effect.
        */
        virtual void
        set_all_elements_same(double val /**< [in] All elements will be set to this. */ ) =0;

        /// Initialize grid elements within specified subset of the grid to the same value.
        /**
           Sets all elements from `first_indices` to `last_indices` in each dimension to the
           specified value.
           Provide indices in two lists in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall within the allocated space as returned by
           get_first_rank_alloc_index() and get_last_rank_alloc_index() for
           each dimension.
           Indices are relative to the *overall* problem domain.
           If storage has not been allocated for this grid, this will have no effect.
           @returns Number of elements set.
        */
        virtual idx_t
        set_elements_in_slice_same(double val /**< [in] All elements in the slice will be set to this. */,
                                   const std::vector<idx_t>& first_indices
                                   /**< [in] List of initial indices, one for each grid dimension. */,
                                   const std::vector<idx_t>& last_indices
                                   /**< [in] List of final indices, one for each grid dimension. */,
                                   bool strict_indices = false
                                   /**< [in] If true, indices must be within domain or padding.
                                      If false, only elements within the allocation of this grid
                                      will be set, and elements outside will be ignored. */ ) =0;

        /// Set grid elements within specified subset of the grid.
        /**
           Reads elements from consecutive memory locations,
           starting at `buffer_ptr`
           and writes them from `first_indices` to `last_indices` in each dimension.
           Indices in the buffer progress in row-major order.
           The buffer pointed to must contain either 4 or 8 byte FP values per element in the 
           subset, depending on the FP precision of the solution.
           The buffer pointed to must contain the number of FP values in the specified slice,
           where each FP value is the size of yk_solution::get_element_bytes().
           Since the writes proceed in row-major order, the last index is "unit-stride"
           in the buffer.
           Provide indices in two lists in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall within the allocated space as returned by
           get_first_rank_alloc_index() and get_last_rank_alloc_index() for
           each dimension.
           Indices are relative to the *overall* problem domain.
           If storage has not been allocated for this grid, this will have no effect.
           @returns Number of elements written.
        */
        virtual idx_t
        set_elements_in_slice(const void* buffer_ptr
                              /**< [out] Pointer to buffer where values will be read. */,
                              const std::vector<idx_t>& first_indices
                              /**< [in] List of initial indices, one for each grid dimension. */,
                              const std::vector<idx_t>& last_indices
                              /**< [in] List of final indices, one for each grid dimension. */ ) =0;
        
        /// Format the indices for pretty-printing.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           @returns A string containing the grid name and the index values.
        */
        virtual std::string
        format_indices(const std::vector<idx_t>& indices
                       /**< [in] List of indices, one for each grid dimension. */ ) const =0;
        
#ifndef SWIG
        /// Format the indices for pretty-printing.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           @note This version is not available (or needed) in SWIG-based APIs, e.g., Python.
           @returns A string containing the grid name and the index values.
        */
        virtual std::string
        format_indices(const std::initializer_list<idx_t>& indices
                       /**< [in] List of indices, one for each grid dimension. */ ) const =0;
#endif
        
        /// Determine whether storage has been allocated.
        /**
           @returns `true` if storage has been allocated,
           `false` otherwise.
        */
        virtual bool
        is_storage_allocated() const =0;

        /// Determine size of raw storage in bytes.
        /**
           @returns Minimum number of bytes required for
           storage given the current domain size and padding settings.
        */
        virtual idx_t
        get_num_storage_bytes() const =0;

        /// Determine size of raw storage in elements.
        /**
           @returns get_num_storage_bytes() / yk_solution.get_element_bytes().
        */
        virtual idx_t
        get_num_storage_elements() const =0;

        /* Advanced APIs for yk_grid found below are not needed for most applications. */

        /// **[Advanced]** Set the default preferred NUMA node on which to allocate data.
        /**
           This value is used when allocating data for this grid.
           Thus, the desired NUMA policy must be set before calling alloc_data()
           or yk_solution::prepare_solution().
           @returns `true` if NUMA preference was set;
           `false` if NUMA preferences are not enabled.
        */
        virtual bool
        set_numa_preferred(int numa_node
                           /**< [in] Preferred NUMA node.
                              See yk_solution::set_default_numa_preferred() for other options. */) =0;

        /// **[Advanced]** Get the default preferred NUMA node on which to allocate data.
        /**
           @returns Current setting of preferred NUMA node for this grid.
        */
        virtual int
        get_numa_preferred() const =0;

        /// **[Advanced]** Set the left halo size in the specified dimension.
        /**
           This value is typically set by the stencil compiler, but
           this function allows you to override that value.
           If the left halo is set to a value larger than the left padding size, the
           left padding size will be automatically increase to accomodate it.
           @note After data storage has been allocated, the left halo size
           can only be set to a value less than or equal to the left padding size
           in the given dimension.
        */
        virtual void
        set_left_halo_size(const std::string& dim
                      /**< [in] Name of dimension to get.
                         Must be one of
                         the names from yk_solution::get_domain_dim_names(). */,
                      idx_t size
                      /**< [in] Number of elements in the left halo. */ ) =0;

        /// **[Advanced]** Set the right halo size in the specified dimension.
        /**
           This value is typically set by the stencil compiler, but
           this function allows you to override that value.
           If the right halo is set to a value larger than the right padding size, the
           right padding size will be automatically increase to accomodate it.
           @note After data storage has been allocated, the right halo size
           can only be set to a value less than or equal to the right padding size
           in the given dimension.
        */
        virtual void
        set_right_halo_size(const std::string& dim
                      /**< [in] Name of dimension to get.
                         Must be one of
                         the names from yk_solution::get_domain_dim_names(). */,
                      idx_t size
                      /**< [in] Number of elements in the right halo. */ ) =0;

        /// **[Advanced]** Set the left and right halo sizes in the specified dimension.
        /**
           Alias for set_left_halo_size(dim, size); set_right_halo_size(dim, size).
        */
        virtual void
        set_halo_size(const std::string& dim
                      /**< [in] Name of dimension to get.
                         Must be one of
                         the names from yk_solution::get_domain_dim_names(). */,
                      idx_t size
                      /**< [in] Number of elements in the halo. */ ) =0;


        /// **[Advanced]** Set the number of elements to allocate in the specified dimension.
        /** 
           This setting is only allowed in the step dimension.
           Typically, the allocation in the step dimension is determined by the
           stencil compiler, but
           this function allows you to override that value.
           Allocations in other dimensions should be set indirectly
           via the domain and padding sizes.
           The allocation size cannot be changed after data storage
           has been allocated for this grid.
        */
        virtual void
        set_alloc_size(const std::string& dim
                       /**< [in] Name of dimension to set.
                          Must *not* be one of
                          the names from yk_solution::get_domain_dim_names(). */,
                       idx_t size /**< [in] Number of elements to allocate. */ ) =0;

        /// **[Advanced]** Set the first index of a specified miscellaneous dimension.
        /**
           Sets the first allowed index in a non-step and non-domain dimension.
           After calling this function, the last allowed index will be the first index
           as set by this function plus the allocation size set by set_alloc_size()
           minus one.
        */
        virtual void
        set_first_misc_index(const std::string& dim
                             /**< [in] Name of dimension to get.  Must be one of
                                the names from yk_solution::get_misc_dim_names(). */,
                             idx_t idx /**< [in] New value for first index.
                                        May be negative. */ ) =0;
        
        /// **[Advanced]** Get the first accessible index in this grid in this rank in the specified dimension.
        /**
           This returns the first *overall* index allowed in this grid.
           This element may be in the domain, left halo, or extra left padding area.
           This function is only for checking the legality of an index.
           @returns First allowed index in this grid.
        */
        virtual idx_t
        get_first_rank_alloc_index(const std::string& dim
                                   /**< [in] Name of dimension to get.
                                      Must be one of
                                      the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// **[Advanced]** Get the last accessible index in this grid in this rank in the specified dimension.
        /**
           This returns the last *overall* index allowed in this grid.
           This element may be in the domain, right halo, or extra right padding area.
           This function is only for checking the legality of an index.
           @returns Last allowed index in this grid.
        */
        virtual idx_t
        get_last_rank_alloc_index(const std::string& dim
                                  /**< [in] Name of dimension to get.
                                     Must be one of
                                     the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// **[Advanced]** Explicitly allocate data-storage memory for this grid.
        /**
           Amount of allocation is calculated based on domain, padding, and 
           step-dimension allocation sizes.
           Any pre-existing storage will be released before allocation as via release_storage().
           See allocation options in the "Detailed Description" for \ref yk_grid.
         */
        virtual void
        alloc_storage() =0;

        /// **[Advanced]** Explicitly release any allocated data-storage for this grid.
        /**
           This will release storage allocated via any of the options
           described in the "Detailed Description" for \ref yk_grid.
           If the data was shared between two or more grids, the data will
           be retained by the remaining grids.
        */
        virtual void
        release_storage() =0;

        /// **[Advanced]** Determines whether storage layout is the same as another grid.
        /**
           In order for the storage layout to be identical, the following
           must be the same:
           - Number of dimensions.
           - Name of each dimension, in the same order.
           - Allocation size in each dimension.
           - Rank domain size in each domain dimension.
           - Padding size in each domain dimension.

           The following do not have to be identical:
           - Halo size.

           @returns `true` if storage for this grid has the same layout as
           `other` or `false` otherwise.
        */
        virtual bool
        is_storage_layout_identical(const yk_grid_ptr other) const =0;
        
        /// **[Advanced]** Use existing data-storage from specified grid.
        /**
           This is an alternative to allocating data storage via 
           yk_solution::prepare_solution() or alloc_storage().
           In this case, data from a grid in this or another solution will be shared with
           this grid.
           In order to successfully share storage, the following conditions must hold:
           - The source grid must already have storage allocated.
           - The two grids must have the same dimensions in the same order.
           - The two grids must have the same domain sizes in all domain dimensions.
           - The two grids must have the same allocation sizes in non-domain dimensions.
           - The required padding size of this grid must be less than or
           equal to the actual padding size of the source grid in all domain
           dimensions. The required padding size of this grid will be equal to
           or greater than its halo size. It is not strictly necessary that the
           two grids have the same halo sizes, but that is a sufficient condition.

           Any pre-existing storage will be released before allocation as via release_storage().
           The padding size(s) of this grid will be set to that of the source grid.
           After calling share_storage(), changes in one grid via set_all_elements()
           or set_element() will be visible in the other grid.

           See allocation options and more information about grid sizes
           in the "Detailed Description" for \ref yk_grid.
        */
        virtual void
        share_storage(yk_grid_ptr source
                      /**< [in] Grid from which storage will be shared. */) =0;

        /// **[Advanced]** Get pointer to raw data storage buffer.
        /**
           The following assumptions about the contents of data are safe:
           - Each FP element starts at a number of bytes from the beginning
           of the buffer which is a multiple of yk_solution::get_element_bytes().
           - All the FP elements will be located within get_num_storage_bytes()
           bytes from the beginning of the buffer.
           - A call to set_all_elements_same() will initialize all elements
           within get_num_storage_bytes() bytes from the beginning of the buffer.
           - If is_storage_layout_identical() returns `true` between this
           and some other grid, any given element index applied to both grids
           will refer to an element at the same offset into their respective
           data buffers. 

           Thus,
           - You can perform element-wise unary mathematical operations on
           all elements of a grid via its raw buffer, e.g., add some constant
           value to all elements.
           - If the layouts of two grids are identical, you can use their
           raw buffers to copy or compare the grid contents for equality or
           perform element-wise binary mathematical operations on them,
           e.g., add all elements from one grid to another.

           The following assumptions are not safe:
           - Any expectations regarding the relationship between an element
           index and that element's offset from the beginning of the buffer
           such as row-major or column-major layout.
           - All elements in the buffer are part of the rank domain or halo.

           Thus,
           - You should not perform any operations dependent on
           the logical indices of any element via raw buffer, e.g., matrix
           multiply.

           @returns Pointer to raw data storage if is_storage_allocated()
           returns `true` or NULL otherwise.
        */
        virtual void* get_raw_storage_buffer() =0;

        /* Deprecated APIs for yk_grid found below should be avoided.
           Use the more explicit form found in the documentation. */
        
        /// **[Deprecated]** Get the left halo size in the specified dimension.
        /**
           Alias for get_left_halo_size(dim, size).
           @returns Elements in halo in given dimension before the domain.
        */
        virtual idx_t
        get_halo_size(const std::string& dim
                      /**< [in] Name of dimension to get.
                         Must be one of
                         the names from yk_solution::get_domain_dim_names(). */ ) const =0;
        
        /// **[Deprecated]** Get the left padding in the specified dimension.
        /**
           Alias for get_left_pad_size(dim).
           @returns Elements in left padding in given dimension.
        */
        virtual idx_t
        get_pad_size(const std::string& dim
                     /**< [in] Name of dimension to get.
                         Must be one of
                         the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// **[Deprecated]** Get the extra left padding in the specified dimension.
        /**
           Alias for get_extra_left_pad_size(dim).
           @returns Elements in padding in given dimension before the
           left halo region.
        */
        virtual idx_t
        get_extra_pad_size(const std::string& dim
                           /**< [in] Name of dimension to get.
                              Must be one of
                              the names from yk_solution::get_domain_dim_names(). */ ) const =0;

    };


} // namespace yask.

#endif
