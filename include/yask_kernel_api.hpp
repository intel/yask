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

///////// API for the YASK stencil kernel. ////////////

// This file uses Doxygen 1.8 markup for API documentation-generation.
// See http://www.stack.nl/~dimitri/doxygen.

#ifndef YASK_KERNEL_API
#define YASK_KERNEL_API

#include <string>
#include <memory>
#include <cinttypes>

namespace yask {

    /// Type to use for indexing grids.
    /** Index types are signed to allow negative indices in halos. */
#ifdef SWIG
    typedef long long int idx_t;     // SWIG doesn't seem to understand int64_t.
#else
    typedef std::int64_t idx_t;
#endif
    
    // Forward declarations of classes and pointers.
    class yk_env;
    typedef std::shared_ptr<yk_env> yk_env_ptr;
    class yk_settings;
    typedef std::shared_ptr<yk_settings> yk_settings_ptr;
    class yk_solution;
    typedef std::shared_ptr<yk_solution> yk_solution_ptr;
    class yk_grid;
    typedef yk_grid* yk_grid_ptr;

    /// Factory to create a stencil solution.
    class yk_factory {
    public:
        virtual ~yk_factory() {}

        /// Create an object to hold environment information.
        /**
         * Initializes MPI if MPI is enabled.
         * Environment info is kept in a separate object to factilitate
         * initializing the environment before creating a solution
         * and sharing an environment among multiple solutions.
         * @returns Pointer to new env object.
         */
        virtual yk_env_ptr
        new_env() const;

        /// Create an object to hold kernel solution settings.
        /**
         * Settings are kept in a separate object to facilitate the
         * sharing of settings among multiple solutions.
         * The settings may be modified via the returned pointer
         * after a solution has been created via new_solution().
         * However, the settings should not be changed after
         * calling yk_solution::prepare_solution().
         * @returns Pointer to new settings object.
         */
        virtual yk_settings_ptr
        new_settings() const;
        
        /// Create a stencil solution.
        /** A stencil solution contains all the grids and equations
         * that were created during stencil compilation.
         * @returns Pointer to new solution object. */
        virtual yk_solution_ptr
        new_solution(yk_env_ptr env /**< [in] Pointer to env info. */,
                     yk_settings_ptr settings /**< [in] Pointer to kernel settings. */ ) const;
    };

    /// Kernel environment.
    class yk_env {
    public:
        virtual ~yk_env() {}

        /// Get number of MPI ranks.
        /** @returns Number of ranks in MPI communicator or one (1) if MPI is not enabled. */
        virtual int get_num_ranks() const =0;

        /// Get MPI rank index.
        /** @returns Index of this MPI rank or zero (0) if MPI is not enabled. */
        virtual int get_rank_index() const =0;

        /// Wait until all ranks have reached this point.
        /** If MPI is enabled, calls MPI_Barrier().
         * Otherwise, has no effect.
         */
        virtual void
        global_barrier() const =0;
    };
    
    /// Kernel settings.
    /** Contains preferences for constructing a solution. */
    class yk_settings {
    public:
        virtual ~yk_settings() {}

        /// Set the size of the solution domain.
        /** The domain defines the points that will be evaluated with the
         * stencil(s). 
         * If MPI is not enabled, this is the entire problem domain.
         * If MPI is enabled, this is the domain for the current rank only,
         * and the problem domain consist of the sum of all rank domains
         * in each dimension.
         * The domain size affects the size of the grids, but it is not
         * equivalent.  Specifically, it does *not* include the halo region
         * or any additional padding.  The actual number of points evaluated
         * may be less than the domain if a given stencil equation has a
         * non-default condition.  Also, the actual number of points in the
         * domain of a given grid may be greater than this specified size
         * due to rounding up to fold-cluster sizes. 
         * Call yk_grid::get_domain_size() to determine the actual size.
         * See the "Detailed Description" for \ref yk_grid for more information on grid sizes.
         * There is no domain-size setting allowed in the
         * solution-step dimension (usually "t"). */
        virtual void
        set_domain_size(const std::string& dim
                        /**< [in] Name of dimension to set.  Must be one of
                           the names from
                           yk_solution::get_domain_dim_name(). */,
                        idx_t size /**< [in] Points in the domain in this `dim`. */ ) =0;

        /// Set the amount of additional grid padding.
        /** This sets the default minimum number of points in each grid that is
         * allocated outside of the grid domain + halos in the given dimension.  The
         * specified number of points is added to both sides, i.e., both "before" and
         * "after" the domain and halo.
         * The actual pad may be greater than
         * the specified size due to rounding up to fold sizes.  
         * Call yk_grid::get_extra_pad_size() to determine the actual size.
         * See the "Detailed Description" for \ref yk_grid for more information on grid sizes.
         * This function sets the extra-padding size that is added by default to *all* grids;
         * you can also set the padding of each grid individually via yk_grid::set_extra_pad_size()
         * and/or yk_grid::set_total_pad_size().
         * There is no padding allowed in the solution-step dimension (usually "t").
         */
        virtual void
        set_default_extra_pad_size(const std::string& dim
                                   /**< [in] Name of dimension to set.  Must
                                      be one of the names from
                                      yk_solution::get_domain_dim_name(). */,
                                   idx_t size
                                   /**< [in] Points in this `dim` applied
                                      to both sides of the domain. */ ) =0;

        /// Set the block size in the given dimension.
        /** This sets the approximate number of points that are evaluated in
         * each "block".  
         * This is a performance setting and should not affect the functional
         * correctness or total number of points evaluated.
         * A block is typically the unit of work done by a
         * top-level OpenMP thread.  The actual number of points evaluated
         * in a block may be greater than the specified size due to rounding
         * up to fold-cluster sizes.  The number of points in a block may
         * also be smaller than the specified size when the block is at the
         * edge of the domain. The block size cannot be set in the
         * solution-step dimension (because temporal blocking is not yet enabled). */
        virtual void
        set_block_size(const std::string& dim
                       /**< [in] Name of dimension to set.  Must be one of
                          the names from
                          yk_solution::get_domain_dim_name(). */,
                       idx_t size /**< [in] Points in a block in this `dim`. */ ) =0;

        /// Set the number of MPI ranks in the given dimension.
        /**
         * The product of the number of ranks across all dimensions must
         * equal the total number of ranks available when yk_factory::new_env() 
         * was called.
         * The curent MPI rank will be assigned a unique location 
         * within the overall problem domain based on its MPI rank index.
         * The same number of MPI ranks must be set via this API on each
         * MPI rank to ensure a consistent overall configuration.
         */
        virtual void
        set_num_ranks(const std::string& dim
                      /**< [in] Name of dimension to set.  Must be one of
                         the names from
                         yk_solution::get_domain_dim_name(). */,
                      idx_t num /**< [in] Number of ranks in `dim`. */ ) =0;
    };
    
    /// Stencil solution as defined by the generated code from the YASK stencil compiler.
    /** Objects of this type contain all the grids and equations
     * that comprise a solution. */
    class yk_solution {
    public:
        virtual ~yk_solution() {}

        /// Get the name of the solution.
        /** @returns String containing the solution name provided during stencil compilation. */
        virtual const std::string&
        get_name() const =0;

        /// Get the solution step dimension.
        /** @returns String containing the step-dimension name. */
        virtual std::string
        get_step_dim() const =0;

        /// Get the number of domain dimensions used in this solution.
        /**
         * Does *not* include the step dimension.
         * @returns Number of dimensions that define the problem domain. */
        virtual int
        get_num_domain_dims() const =0;

        /// Get the name of the specified domain dimension.
        /**
         * @returns String containing name of domain dimension. */
        virtual std::string
        get_domain_dim_name(int n /**< [in] Index of dimension between zero (0)
                                     and get_num_domain_dims()-1. */ ) const =0;

        /// Get the number of grids in the solution.
        /** @returns Number of grids that have been created via new_grid(). */
        virtual int
        get_num_grids() const =0;
        
        /// Get the specified grid.
        /** @returns Pointer to the nth grid. */
        virtual yk_grid_ptr
        get_grid(int n /**< [in] Index of grid between zero (0)
                              and get_num_grids()-1. */ ) =0;

        /// Prepare the solution for stencil application.
        /** Allocates data in grids that do not already have storage allocated.
         * Sets many other data structures needed for proper stencil application.
         * Must be called before querying grids for their final sizes or domain positions.
         * Must be called before applying any stencils.
         */
        virtual void
        prepare_solution() =0;

        /// Apply the stencil solution for one step.
        /** The stencil(s) in the solution are applied
         * across the entire domain as specified in yk_settings::set_domain_size().
         * MPI halo exchanges will occur as necessary.
         */
        virtual void
        apply_solution(idx_t step_index /**< [in] Index in the step dimension */ ) =0;

        /// Apply the stencil solution for the specified number of steps.
        /** The stencil(s) in the solution are applied from
         * the first to last step index, inclusive.
         * MPI halo exchanges will occur as necessary.
         */
        virtual void
        apply_solution(idx_t first_step_index /**< [in] First index in the step dimension */,
                       idx_t last_step_index /**< [in] Last index in the step dimension */ ) =0;
    };

    /// A run-time grid.
    /** "Grid" is a generic term for any n-dimensional array.  A 0-dim grid
     * is a scalar, a 1-dim grid is an array, etc.  A run-time grid contains
     * data, unlike yc_grid, a compile-time grid variable.  
     * Gain access to each grid via yk_solution::get_grid().
     *
     * In each dimension, grid sizes include the following components:
     * - The *domain* is the points to which the stencils are applied.
     * - The *total padding* is the points outside the domain on either side
     * and includes the halo.
     * - The *extra padding* is the points outside the domain and halo on
     * either side and thus does not include the halo.
     * - The *allocation* includes the domain and the padding.
     *
     * All sizes are expressed in numbers of elements, which may be 4-byte (single precision)
     * or 8-byte (double precision) floating-point values.
     *
     * Visually, in each of the non-step dimensions, these sizes are related as follows:
     * <table>
     * <tr><td>extra pad <td>halo <td rowspan="2">domain <td>halo <td>extra pad
     * <tr><td colspan="2"><center>total pad</center> <td colspan="2"><center>total pad</center>
     * <tr><td colspan="5"><center>allocation</center>
     * </table>
     * In the step dimension (usually "t"), there is no fixed domain size.
     * But there is an allocation size, which is the number of values in the step
     * dimension that are stored in memory. 
     * Step-dimension indices "wrap-around" within this allocation to reuse memory.
     * For example, if the step dimension is "t", and the t-dimension allocation size is 3,
     * then t=-2, t=0, t=3, t=6, ..., t=303, etc. would all alias to the same spatial values in memory.
     * 
     * Domain sizes specified via yk_settings::set_domain_size() apply to each process or *rank*.
     * If MPI is not enabled, a rank's domain is the entire problem size.
     * If MPI is enabled, the domains of the ranks are logically adjoined to create the 
     * overall problem domain in each dimension:
     * <table>
     * <tr><td>extra pad of rank A <td>halo of rank A <td>domain of rank A <td>domain of rank B
     *   <td>... <td>domain of rank Z <td>halo of rank Z <td>extra pad of rank Z
     * <tr><td colspan="2"><center>total pad of rank A</center>
     *   <td colspan="4"><center>overall problem domain</center>
     *   <td colspan="2"><center>total pad of rank Z</center>
     * </table>
     * The beginning and ending overall-problem index that lies within a rank can be
     * retrieved via get_first_domain_index() and get_last_domain_index(), respectively.
     *
     * The intermediate halos and pads also exist, but are not shown in the above diagram.
     * They overlap the domains of adjacent ranks.
     * For example, the left halo of rank B in the diagram would overlap the domain of rank A.
     * Data in these overlapped regions is exchanged as needed during stencil application
     * to maintain a consistent values as if there was only one rank.
     */
    class yk_grid {
    public:
        virtual ~yk_grid() {}

        /// Get the name of the grid.
        /** @returns String containing name provided via yc_solution::new_grid(). */
        virtual const std::string& get_name() const =0;

        /// Get the number of dimensions used in this grid.
        /**
         * May include step dimension.
         * May be different than value returned from yk_solution::get_num_domain_dims().
         * @returns Number of dimensions created via yc_solution::new_grid(). */
        virtual int get_num_dims() const =0;

        /// Get the name of the specified dimension in this grid.
        /**
         * Dimensions are not necessarily in the same order as those returned
         * via yk_solution::get_domain_dim_name().
         * @returns String containing name of dimension created via new_grid(). */
        virtual const std::string&
        get_dim_name(int n /**< [in] Index of dimension between zero (0)
                              and get_num_dims()-1. */ ) const =0;

        /// Get the first logical index of the domain in the specified dimension.
        /** This returns the first index at the beginning of the domain.
         * Points within the domain lie between the values returned by
         * get_first_domain_index() and get_last_domain_index(), inclusive.
         * This value does *not* include
         * indices within the pad area, including the halo region.
         * Call get_halo_size() to find the number of points that may be
         * indexed before the first domain index.
         * If there is only one MPI, this is typically zero (0).
         * However, if there is more than one MPI rank, the value depends
         * on the the rank's position within the overall problem domain.
         * @returns First logical domain index. */
        virtual idx_t
        get_first_domain_index(const std::string& dim
                               /**< [in] Name of dimension from get_dim_name().
                                  Cannot be the step dimension. */ ) const =0;

        /// Get the last logical index of the domain in the specified dimension.
        /** This returns the last index at the end of the domain
         * (*not* one past the end).
         * Points within the domain lie between the values returned by
         * get_first_domain_index() and get_last_domain_index(), inclusive.
         * This value does *not* include
         * indices within the pad area, including the halo region.
         * Call get_halo_size() to find the number of points that may be
         * indexed after the last domain index.
         * If there is only one MPI, this is typically one less than the value
         * provided by yk_settings::set_domain_size().
         * However, if there is more than one MPI rank, the value depends
         * on the the rank's position within the overall problem domain.
         * @returns Last logical index. */
        virtual idx_t
        get_last_domain_index(const std::string& dim
                               /**< [in] Name of dimension from get_dim_name().
                                  Cannot be the step dimension. */ ) const =0;

        /// Get the domain size in the specified dimension.
        /** Value does *not* include padding or halo.  Value may be slightly
         * different than that provided via yk_settings::set_domain_size()
         * as described in that function's documentation.  
         * @returns Points in domain in given dimension. */
        virtual idx_t
        get_domain_size(const std::string& dim
                        /**< [in] Name of dimension from get_dim_name().
                           Cannot be the step dimension. */ ) const =0;

        /// Get the halo size in the specified dimension.
        /** This value is typically set by the stencil compiler.
         * @returns Points in halo in given dimension. */
        virtual idx_t
        get_halo_size(const std::string& dim
                      /**< [in] Name of dimension from get_dim_name().
                         Cannot be the step dimension. */ ) const =0;

        /// Get the extra padding in the specified dimension.
        /** The *extra* pad size is the total pad size minus the halo size.
         * The value may be slightly
         * different than that provided via yk_settings::set_default_extra_pad_size()
         * or set_extra_pad_size() as described in those functions' documentation.
         * @returns Points in extra padding in given dimension. */
        virtual idx_t
        get_extra_pad_size(const std::string& dim
                           /**< [in] Name of dimension from get_dim_name().
                              Cannot be the step dimension. */ ) const =0;

        /// Get the total padding in the specified dimension.
        /** The *total* pad size includes the halo size.
         * The value may be slightly
         * different than that provided via set_total_pad_size() due to rounding.
         * @returns Points in extra padding in given dimension. */
        virtual idx_t
        get_total_pad_size(const std::string& dim
                           /**< [in] Name of dimension from get_dim_name().
                              Cannot be the step dimension. */ ) const =0;

        /// Get the storage allocation in the specified dimension.
        /** For the step dimension, this is the specified allocation and
         * does not typically depend on the number of steps evaluated.
         * For the non-step dimensions, this is get_domain_size() + 
         * (2 * (get_halo_size() + get_extra_pad_size())).
         * See the "Detailed Description" for \ref yk_grid for information on grid sizes.
         * @returns allocation in number of points (not bytes). */
        virtual idx_t
        get_alloc_size(const std::string& dim
                       /**< [in] Name of dimension from get_dim_name(). */ ) const =0;

        /// Set the number of points to allocate in the specified dimension.
        /** This is only allowed in the step dimension.
         * Allocations in other dimensions should be set indirectly
         * via the domain and padding sizes. */
        virtual void
        set_alloc_size(const std::string& dim
                       /**< [in] Name of dimension from get_dim_name().
                          Must be the step dimension. */,
                       idx_t size /**< [in] Number of points to allocate. */ ) =0;

        /// Set the total padding in the specified dimension.
        /** This overrides yk_settings::set_default_extra_pad_size()
            and any earlier call to set_extra_pad_size().
            See the "Detailed Description" for \ref yk_grid for information on grid sizes. */
        virtual void
        set_total_pad_size(const std::string& dim
                           /**< [in] Name of dimension from get_dim_name().
                              Cannot be the step dimension. */,
                           idx_t size
                           /**< [in] Number of points to allocate beyond the domain size.
                              If less than the halo size, it will be increased as needed. */ ) =0;
        
        /// Set the extra padding in the specified dimension.
        /** This overrides yk_settings::set_default_extra_pad_size()
            and any earlier call to set_total_pad_size().
            See the "Detailed Description" for \ref yk_grid for information on grid sizes. */
        virtual void
        set_extra_pad_size(const std::string& dim
                           /**< [in] Name of dimension from get_dim_name().
                              Cannot be the step dimension. */,
                           idx_t size
                           /**< [in] Number of points to allocate beyond the domain and halo size.
                              Must be greater than or equal to zero. */ ) =0;

        /// Initialize all grid points to the same value.
        /** Sets all allocated elements, including those in the domain, halo, and extra padding
         * area to the same provided value.
         * @note The parameter is a double-precision floating-point value, but
         * it may be converted to single-precision if the solution was configured with 4-byte
         * elements via yc_solution::set_element_bytes().
         * @note If storage has not been allocated via yk_solution::prepare_solution(),
         * this will have no effect.
         */
        virtual void
        set_all_elements(double val /**< [in] All points will be set to this. */ ) =0;

        /// Get the value of one grid point.
        /** Provide the number of indices equal to the number of dimensions in the grid.
         * Indices beyond that will be ignored.
         * Indices are relative to the overall problem domain.
         * You can only get elements that are located in the grid in the current rank.
         * See get_first_domain_index(), get_last_domain_index(), and the 
         * "Detailed Description" for \ref yk_grid for more information on grid sizes.
         * @note The return value is a double-precision floating-point value, but
         * it may be converted from a single-precision if the solution was configured with 4-byte
         * elements via yc_solution::set_element_bytes().
         */
        virtual double
        get_element(idx_t dim1_index=0 /**< [in] Index in dimension 1. */,
                    idx_t dim2_index=0 /**< [in] Index in dimension 2. */,
                    idx_t dim3_index=0 /**< [in] Index in dimension 3. */,
                    idx_t dim4_index=0 /**< [in] Index in dimension 4. */,
                    idx_t dim5_index=0 /**< [in] Index in dimension 5. */,
                    idx_t dim6_index=0 /**< [in] Index in dimension 6. */ ) const =0;

        /// Set the value of one grid point.
        /** Provide the number of indices equal to the number of dimensions in the grid.
         * Indices beyond that will be ignored.
         * Indices are relative to the overall problem domain.
         * You can only set elements that are located in the grid in the current rank.
         * See get_first_domain_index(), get_last_domain_index(), and the 
         * "Detailed Description" for \ref yk_grid for more information on grid sizes.
         * @note The parameter value is a double-precision floating-point value, but
         * it may be converted to single-precision if the solution was configured with 4-byte
         * elements via yc_solution::set_element_bytes().
         */
        virtual void
        set_element(double val /**< [in] Point in grid will be set to this. */,
                    idx_t dim1_index=0 /**< [in] Index in dimension 1. */,
                    idx_t dim2_index=0 /**< [in] Index in dimension 2. */,
                    idx_t dim3_index=0 /**< [in] Index in dimension 3. */,
                    idx_t dim4_index=0 /**< [in] Index in dimension 4. */,
                    idx_t dim5_index=0 /**< [in] Index in dimension 5. */,
                    idx_t dim6_index=0 /**< [in] Index in dimension 6. */ ) =0;
    };


} // namespace yask.

#endif
