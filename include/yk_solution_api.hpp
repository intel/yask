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

///////// API for the YASK stencil kernel solution. ////////////

// This file uses Doxygen 1.8 markup for API documentation-generation.
// See http://www.stack.nl/~dimitri/doxygen.
/** @file yk_solution_api.hpp */ 

#ifndef YK_SOLN_API
#define YK_SOLN_API

#include "yask_kernel_api.hpp"

namespace yask {

    /**
     * \addtogroup yk
     * @{
     */

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
           @returns String containing the step-dimension name
           that was defined by yc_node_factory::new_step_index()
           and used in one or more grids.
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
           and used in one or more grids.
        */
        virtual std::vector<std::string>
        get_domain_dim_names() const =0;

        /// Get all the miscellaneous dimension names.
        /**
           @returns List of all dimension names
           that were either
           * Defined by yc_node_factory::new_misc_index()
           and used in one or more grids, or
           * Created at run-time by adding a new dimension
           via yk_solution::new_grid() or yk_solution::new_fixed_size_grid().
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
           The domain size does *not* include the halo area or any padding.
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
           or created explicitly via yk_solution::new_grid()
           or yk_solution::new_fixed_size_grid().
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
           1. If temporal wave-front tiling is *not* used (the default):
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
            - Since this function initiates MPI communication, it must be called
              on all MPI ranks, and it will block until all ranks have completed.
           2. **[Advanced]** If temporal wave-front tiling *is* enabled via set_region_size():
            - The step index (e.g., `t` for "time") will be sequentially set to values
            from `first_step_index` to `last_step_index`, inclusive, within each region.
             + The number of steps in a region may also be restricted by the size
             of the region in the step dimension. In that case, tiles will be done in slices of that size.
            - For each step index within each region, the domain indices will be set
            to values across the entire region (not necessarily sequentially).
            - Ultimately, the stencil(s) will be applied to same the elements in both the step 
            and domain dimensions as when wave-front tiling is not used.
            - MPI halo exchanges will occur before each number of steps in a region.

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
           wave-front tiling.
        */
        virtual void
        run_solution(idx_t step_index /**< [in] Index in the step dimension */ ) =0;

        /// Finish using a solution.
        /**
           Performs a final MPI halo exchange.
           Releases shared ownership of memory used by the grids.  This will
           result in deallocating each memory block that is not
           referenced by another shared pointer.
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
        
        /// **[Advanced]** Set the region size in the given dimension.
        /**
           This sets the approximate number of elements that are evaluated in
           each "region".
           This is a performance setting and should not affect the functional
           correctness or total number of elements evaluated.
           A region is typically the unit of work done by each
           top-level OpenMP parallel region.  The actual number of elements evaluated
           in a region may be greater than the specified size due to rounding.
           The number of elements in a region may
           also be smaller than the specified size when the region is at the
           edge of the domain.

           A region is most often used to specify the size of a temporal
           wave-front tile. Thus, you will normally specify the size of the
           region in the step dimension as well as all the domain dimensions.
           For example, `set_region_size("t", 4)` specifies that four
           time-steps will be executed in each region.
           The sizes of regions in the domain dimensions are typically
           set to fit within a large cache structure such as MCDRAM cache
           in an Intel(R) Xeon Phi(TM) processor.

           In order to get the benefit of regions with multiple steps,
           you must also call run_solution() where the number of steps
           between its `first_step_index` and `last_step_index`
           arguments is greater than or equal to the step-size of the 
           regions.
        */
        virtual void
        set_region_size(const std::string& dim
                        /**< [in] Name of dimension to set.  Must be one of
                           the names from get_step_dim_name() or
                           get_domain_dim_names(). */,
                        idx_t size
                        /**< [in] Elements in a region in this `dim`. */ ) =0;

        /// **[Advanced]** Get the region size.
        /**
           Returned value may be slightly larger than the value provided
           via set_region_size() due to rounding.
           @returns Current settings of region size.
        */
        virtual idx_t
        get_region_size(const std::string& dim
                        /**< [in] Name of dimension to get.  Must be one of
                           the names from get_step_dim_name() or
                           get_domain_dim_names(). */) const =0;

        /// **[Advanced]** Set the minimum amount of grid padding for all grids.
        /**
           This sets the minimum number of elements in each grid that is
           reserved outside of the rank domain in the given dimension.
           This padding area can be used for required halo areas.  At
           least the specified number of elements will be added to both
           sides, i.e., both "before" and "after" the domain.
           
           The *actual* padding size will be the largest of the following values,
           additionally rounded up based on the vector-folding dimensions,
           cache-line alignment, and/or extensions needed for wave-front tiles:
           - Halo size.
           - Value provided by any of the pad-size setting functions.
           
           The padding size cannot be changed after data storage
           has been allocated for a given grid; attempted changes to the pad size for such
           grids will be ignored.
           In addition, once a grid's padding is set, it cannot be reduced, only increased.

           Use yk_grid::set_left_min_pad_size and yk_grid::set_right_min_pad_size()
           for specific setting of each grid.
           Call yk_grid::get_left_pad_size() and yk_grid::get_right_pad_size()
           to determine the actual padding sizes for a given grid.
           See the "Detailed Description" for \ref yk_grid for more information on grid sizes.
           Padding is only allowed in the domain dimensions.
        */
        virtual void
        set_min_pad_size(const std::string& dim
                         /**< [in] Name of dimension to set.  Must
                            be one of the names from get_domain_dim_names(). */,
                         idx_t size
                         /**< [in] Elements in this `dim` applied
                            to both sides of the domain. */ ) =0;

        /// **[Advanced]** Get the minimum amount of grid padding for all grids.
        /**
           @returns Current setting of minimum amount of grid padding for all grids.
        */
        virtual idx_t
        get_min_pad_size(const std::string& dim
                         /**< [in] Name of dimension to get.  Must be one of
                            the names from get_domain_dim_names(). */) const =0;

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

           Grids created by this function will behave [mostly] like a pre-defined grid.
           For example,
           - Step and domain dimensions must the same as those defined by
           yc_node_factory::new_step_index() and yc_node_factory::new_domain_index(),
           respectively.
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
           sizes in the domain dimensions via yk_grid::set_min_pad_size(),
           yk_solution::set_min_pad_size(), etc.
           - For step and misc dimensions, you can change the desired size
           yk_grid::set_alloc_size().
           - Storage may be allocated via yk_grid::alloc_storage() or
           yk_solution::prepare_solution().

           Some behaviors are different from pre-defined grids. For example,
           - You can create new "misc" dimensions during grid creation simply
           by naming them in the `dims` argument. Any dimension name that is 
           not a step or domain dimension will become a misc dimension,
           whether or not it was defined via yc_node_factory::new_misc_index().
           - Grids created via new_grid() cannot be direct inputs or outputs of
           stencil equations. However, data in a grid created via new_grid()
           can be shared with a pre-defined grid via yk_grid::share_storage()
           if and only if the sizes of all dimensions are compatible.

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

           The following behaviors are different from both pre-defined grids
           and those created via new_grid():
           - For each domain dimension of the grid,
           the new grid's domain size is provided during creation and cannot be changed.
           - Calls to set_rank_domain_size() will *not* resize the corresponding domain 
           size in this grid.
           - This grid's first domain index in this rank will be fixed at zero (0)
           regardless of this rank's position.
           - This grid's padding size will be affected only by calls to 
           yk_grid::set_min_pad_size(), etc., i.e., *not* via
           yk_solution::set_min_pad_size().

           The following behaviors are the same as those of a pre-defined grid
           and those created via new_grid():
           - For step and misc dimensions, you can change the desired size
           yk_grid::set_alloc_size().
           - Storage may be allocated via yk_grid::alloc_storage() or
           yk_solution::prepare_solution().

           The following behaviors are different than a pre-defined grid
           but the same as those created via new_grid():
           - You can create new "misc" dimensions during grid creation simply
           by naming them in the `dims` argument. Any dimension name that is 
           not a step or domain dimension will become a misc dimension,
           whether or not it was defined via yc_node_factory::new_misc_index().
           - Grids created via new_fixed_size_grid() cannot be direct inputs or outputs of
           stencil equations. However, data in a grid created via new_grid()
           can be shared with a pre-defined grid via yk_grid::share_storage()
           if and only if the sizes of all dimensions are compatible.

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
                                      \ref yk_solution_api.hpp. */) =0;

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
    
    /** @}*/
} // namespace yask.

#endif
