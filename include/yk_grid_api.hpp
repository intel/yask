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

///////// API for the YASK stencil kernel grid. ////////////

// This file uses Doxygen 1.8 markup for API documentation-generation.
// See http://www.stack.nl/~dimitri/doxygen.
/** @file yk_grid_api.hpp */ 

#ifndef YK_GRID_API
#define YK_GRID_API

#include "yask_kernel_api.hpp"

namespace yask {

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
