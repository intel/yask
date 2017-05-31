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

        /// Create an object to hold kernel settings.
        virtual yk_settings_ptr
        new_settings() const;
        
        /// Create a stencil solution.
        /** A stencil solution contains all the grids and equations
         * that were created during stencil compilation.
         * @returns Pointer to new solution object. */
        virtual yk_solution_ptr
        new_solution(yk_settings_ptr settings /**< [in] Pointer to kernel settings. */ ) const;
    };

    /// Kernel settings.
    /** Contains preferences for constructing a solution. */
    class yk_settings {
    public:
        virtual ~yk_settings() {}

        /// Set the size of the solution domain.
        /** The domain defines the points that will be evaluated with the
         * stencil(s). It affects the size of the grids, but it is not
         * equivalent.  Specifically, it does *not* include the halo region
         * or any additional padding.  The actual number of points evaluated
         * may be less than the domain if a given stencil equation has a
         * non-default condition.  Also, the actual number of points in the
         * domain of a given grid may be greater than this specified size
         * due to rounding up to fold-cluster sizes. Get the
         * domain size of a specific grid to determine the actual size.
         * There is no domain-size setting allowed in the
         * solution-step dimension (usually "t"). */
        virtual void
        set_domain_size(const std::string& dim /**< [in] Name of dimension to set.
                                                Must correspond to a dimension set via
                                                INIT_GRID() or yc_solution::new_grid(). */,
                        idx_t size /**< [in] Points in the domain in this `dim`. */ ) =0;

        /// Set the minimum size of additional grid padding.
        /** This sets the default minimum number of points in each grid that is
         * allocated outside of the grid domain + halo in the given dimension.  The
         * specified number of points are added to both sides, i.e., both "before" and
         * "after" the domain and halo.
         * The *total* padding is the number of points outside the domain on either side
         * and includes the halo.
         * The *extra* padding is the number of points outside the domain and halo on
         * either side and does not include the halo.
         * The actual pad may be greater than
         * the specified size due to rounding up to fold sizes.  Get the pad
         * size of a specific grid to determine the actual size.
         * There is no padding allowed in the
         * solution-step dimension (usually "t").
         */
        virtual void
        set_extra_pad_size(const std::string& dim /**< [in] Name of dimension to set.
                                                     Must correspond to a dimension created via
                                                     INIT_GRID() or yc_solution::new_grid(). */,
                     idx_t size /**< [in] Points in this `dim` applied
                                   to both sides of the domain. */ ) =0;

        /// Set the block size.
        /** This sets the approximate number of points that are evaluated in
         * each "block".  A block is typically the unit of work done by a
         * top-level OpenMP thread.  The actual number of points evaluated
         * in a block may be greater than the specified size due to rounding
         * up to fold-cluster sizes.  The number of points in a block may
         * also be smaller than the specified size when the block is at the
         * edge of the domain. The block size cannot be set in the
         * solution-step dimension (because temporal blocking is not yet enabled). */
        virtual void
        set_block_size(const std::string& dim
                       /**< [in] Name of dimension to set.
                          Must correspond to a dimension created via
                          INIT_GRID() or yc_solution::new_grid(). */,
                       idx_t size /**< [in] Points in a block in this `dim`. */ ) =0;
    };
    
    /// Stencil solution.
    /** Objects of this type contain all the grids and equations
     * that comprise a solution. */
    class yk_solution {
    public:
        virtual ~yk_solution() {}

        /// Initialize the solution's environment.
        /** Either init_env() or copy_env() must be called before
         * a solution is used.
         * This call initializes the MPI (if enabled) and OpenMP
         * settings.
         */
        virtual void
        init_env() =0;
        
        /// Copy the solution's environment from an existing one.
        /** Either init_env() or copy_env() must be called before
         * a solution is used.
         * This call copies the MPI (if enabled) and OpenMP
         * settings.
         */
        virtual void
        copy_env(const yk_solution_ptr src /**< [in] Pointer to solution that environment
                                              settings will be copied from. */ ) =0;
        
        /// Get the name of the solution.
        /** @returns String containing the solution name provided during stencil compilation. */
        virtual const std::string&
        get_name() const =0;

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
    };

    /// A run-time grid.
    /** "Grid" is a generic term for any n-dimensional array.  A 0-dim grid
     * is a scalar, a 1-dim grid is an array, etc.  A run-time grid contains
     * data, unlike yc_grid, a compile-time grid.  Access grids via
     * yk_solution::get_grid(). */
    class yk_grid {
    public:
        virtual ~yk_grid() {}

        /// Get the name of the grid.
        /** @returns String containing name provided via new_grid(). */
        virtual const std::string& get_name() const =0;

        /// Get the number of dimensions.
        /** @returns Number of dimensions created via new_grid(). */
        virtual int get_num_dims() const =0;

        /// Get the name of the specified dimension.
        /** @returns String containing name of dimension created via new_grid(). */
        virtual const std::string&
        get_dim_name(int n /**< [in] Index of dimension between zero (0)
                              and get_num_dims()-1. */ ) const =0;

        /// Get the domain size in the specified dimension.
        /** Value does *not* include padding or halo.  Value may be slightly
         * different than that provided via yk_settings::set_domain_size()
         * as described in that function's documentation.  
         * @returns Points in domain in given dimension. */
        virtual idx_t
        get_domain_size(const std::string& dim
                        /**< [in] Name of dimension from get_dim_name().
                           Cannot be the step dimension. */ ) =0;

        /// Get the halo size in the specified dimension.
        /** This value is set by the stencil compiler.
         * @returns Points in halo in given dimension. */
        virtual idx_t
        get_halo_size(const std::string& dim
                      /**< [in] Name of dimension from get_dim_name().
                         Cannot be the step dimension. */ ) =0;

        /// Get the extra padding in the specified dimension.
        /** The *extra* pad size is the total pad size minus the halo size.
         * Value may be slightly
         * different than that provided via yk_settings::set_extra_pad_size()
         * as described in that function's documentation.
         * @returns Points in extra padding in given dimension. */
        virtual idx_t
        get_extra_pad_size(const std::string& dim
                           /**< [in] Name of dimension from get_dim_name().
                              Cannot be the step dimension. */ ) =0;

        /// Get the storage allocation in the specified dimension.
        /** For the step dimension, this is the specified allocation and
         * does not typically depend on the number of steps evaluated.
         * For the non-step dimensions, this is get_domain_size() + 
         * (2 * (get_halo_size() + get_extra_pad_size())).
         * @returns allocation in number of points (not bytes). */
        virtual idx_t
        get_alloc_size(const std::string& dim
                       /**< [in] Name of dimension from get_dim_name(). */ ) =0;

        /// Set the storage allocation in the specified dimension.
        /** This is only allowed in the step dimension.
         * Allocations in other dimensions should be set indirectly
         * via the domain and padding sizes. */
        virtual void
        set_alloc_size(const std::string& dim
                       /**< [in] Name of dimension from get_dim_name().
                          Must be the step dimension. */,
                       idx_t size /**< [in] Number of points to allocate. */ ) =0;

        /// Set the total padding in the specified dimension.
        /** See yk_settings::set_extra_pad_size() for a discussion
         * of extra vs total padding. */
        virtual void
        set_total_pad_size(const std::string& dim
                           /**< [in] Name of dimension from get_dim_name().
                              Cannot be the step dimension. */,
                           idx_t size
                           /**< [in] Number of points to allocate beyond the domain size.
                              Must be greater than or equal to the halo size. */ ) =0;
        
    };


} // namespace yask.

#endif
