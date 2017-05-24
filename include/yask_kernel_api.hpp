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

namespace yask {

    /// Type to use for indexing grids.
    /** Index types are signed to allow negative indices in halos. */
    typedef int64_t idx_t;
   
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

        /// Set the domain size.
        /** The domain defines the points that will be
         * evaluated with the stencil(s). It affects the size of the grids, but it is not equivalent.
         * Specifically, it does *not* include the halo region or any additional padding.
         * The actual number of points evaluated may be less than the domain if a given stencil
         * equation has a non-default condition.
         * Also, actual number of points in the domain of a given grid may be greater than
         * this specified size due to rounding up to fold-cluster sizes. 
         * Get the domain size of a specific grid to determine the actual size. */
        virtual void
        set_domain_size(const std::string& dim /**< [in] Dimension to set. */,
                        idx_t size /**< [in] Points in this `dim`. */ ) =0;

        /// Set the grid pad size.
        /** This sets a minimum number of points in each grid that is allocated outside of the domain
         * in the given dimension.
         * There is no padding allowed in the solution-step dimension (usually "t").
         * If a grid has a halo size set in a given dimension before allocation, 
         * this pad will be added to that halo size. 
         * (Thus, a pad is not required a priori if the halo size is already set.)
         * If a grid does *not* have a halo size set in a given dimension before allocation,
         * the halo can be set after allocation to a value less than or equal to the pad size without
         * requiring re-allocation, and the pad size will be reduced as needed.
         * Also, the actual pad may be greater than
         * the specified size due to rounding up to fold sizes. 
         * Get the pad size of a specific grid to determine the actual size. */
        virtual void
        set_pad_size(const std::string& dim /**< [in] Dimension to set. */,
                     idx_t size /**< [in] Points in this `dim`. */ ) =0;
    };
    
    /// Stencil solution.
    /** Objects of this type contain all the grids and equations
     * that comprise a solution. */
    class yk_solution {
    public:
        virtual ~yk_solution() {}

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
    };

    /// A grid.
    /** "Grid" is a generic term for any n-dimensional array.  A 0-dim grid
     * is a scalar, a 1-dim grid is an array, etc. 
     * Access grids via yk_solution::get_grid(). */
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

    };


} // namespace yask.

#endif
