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

///////// API for the YASK stencil kernel var. ////////////

// This file uses Doxygen 1.8 markup for API documentation-generation.
// See http://www.stack.nl/~dimitri/doxygen.
/** @file yk_var_api.hpp */

#pragma once

#include "yask_kernel_api.hpp"

namespace yask {

    /**
     * \addtogroup yk
     * @{
     */

    /// A run-time YASK data container.
    /**
       A run-time YASK variable is a generic term for any n-dimensional
       array.  A 0-dim var is a scalar, a 1-dim var is an array, etc.  A
       run-time variable actually contains data, unlike yc_var, a
       compile-time variable.

       Typically, access to each var is obtained via yk_solution::get_var().
       You may also use yk_solution::new_var() or yk_solution::new_fixed_size_var()
       if you need a var that is not part of the pre-defined solution.

       # Var Dimensions #

       Each dimension of a YASK var is one of the following:
       - The *step* dimension, typically time (`t`),
       as returned from yk_solution::get_step_dim_name().
       - A *domain* dimension, typically a spatial dimension such as `x` or `y`,
       as returned from yk_solution::get_domain_dim_names().
       - A *miscellaneous* dimension, which is any dimension that is not a step or domain dimension.
       These may be returned via yk_solution::get_misc_dim_names() if they were defined
       in the YASK compiler, or they may be any other name that is not a step or domain dimension.

       ## Step Dimensions #

       The step dimension, as defined during YASK compilation,
       is the dimension in which the simulation proceeds, often "t" for time.
       In the step dimension, there is no fixed first or last index.
       However, there is a finite allocation size, which is the number of
       values in the step dimension that are stored in memory.  The valid
       indices in the step dimension are always consecutive and change based
       on what was last written to the var.  

       For example: If a var `A` has
       an allocation size of two (2) in the `t` step dimension, its initial
       valid `t` indices are 0 and 1.  Calling `A->get_element({0, x})` or
       `A->get_element({1, x})` would return a value from `A` assuming `x`
       is a valid index, but `A->get_element({2, x})` would cause a run-time
       exception.  Let's say the YASK solution defines `A(t+1, x) EQUALS
       (A(t, x) + A(t, x+1))/2`.  Calling yk_solution::run_solution(1) means
       that `A(2, x)` would be defined for all `x` in the domain because
       `t+1 == 2` on the left-hand-side of the equation.  Thus, the new
       valid `t` indices in `A` would be 1 and 2, and `A(0, x)` is no longer
       stored in memory because the allocation size is only 2.  Then,
       calling `A->get_element({1, x})` or `A->get_element({2, x})` would
       succeed and `A->get_element({0, x})` would fail.  

       Calling APIs that set values in a var such as set_element() will
       also update the valid step index range.  The current valid indices in
       the step dimension can be retrieved via
       yk_var::get_first_valid_step_index() and
       yk_var::get_last_valid_step_index().

       If yk_solution::set_step_wrap(true) is called, any invalid value of a
       step index provided to an API will silently "wrap-around" to a valid
       value by effectively adding or subtracing multiples of the allocation
       size as needed. For example, if the valid step indices are 7 and 8
       for a given var, the indices 0 and 1 will wrap-around to 8 and 7,
       respectively.  This is not recommended for general use because it can
       hide off-by-one-type errors. However, it may be useful for
       applications that need to access a var using absolute rather than
       logical step indices.

       ## Domain Dimensions #

       A domain dimension typically corresponds to a physical spatial dimension,
       but it can be any dimension that is used for "domain decomposition" in
       the solution.
       The indices in the domain dimensions that correspond to spatial dimensions (e.g.,
       "x" and "y") often denote locations in the implicit "grid" of the problem domain.

       In each domain dimension,
       the number of elements allocated in a given var include the following components:
       - The *domain* is the elements to which the stencils are applied.
       - The *left padding* is all the accessible elements before the domain and includes the left halo.
       - The *right padding* is all the accessible elements after the domain and includes the right halo.
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
       Halos and paddings between ranks also exist, but are not shown in the above diagram.
       The halos overlap the domains of adjacent ranks.
       For example, the left halo of rank B in the diagram would overlap the domain of rank A.
       Data in these overlapped areas are exchanged as needed during stencil application
       to maintain a consistent values as if there was only one rank.

       ## Miscellaneous Dimensions #

       In each miscellaneous dimension, there is no padding or halos.
       There is a fixed allocation size, and 
       each index must be between its first and last valid value.
       The valid miscellaneous indices may be retrieved via
       yk_var::get_first_misc_index() and yk_var::get_last_misc_index().

       # Other Details #

       ## Elements #

       All sizes are expressed in numbers of elements.
       Each element may be a 4-byte (single precision)
       or 8-byte (double precision) floating-point value as returned by
       yk_solution::get_element_bytes().

       ## Data Storage #

       Initially, a var is not assigned any allocated storage.
       This is done to allow modification of domain, padding, and other allocation sizes
       before allocation.
       Once the allocation sizes have been set in all dimensions, the data storage itself may
       be allocated.
       This can be done in any of the following ways:
       - Storage for all vars without data storage will be automatically allocated when
       yk_solution::prepare_solution() is called.
       - Storage for a specific var may be allocated before calling yk_solution::prepare_solution()
       via yk_var::alloc_storage().
       - **[Advanced]** A var may be merged with another var with existing storage
       via yk_var::fuse_vars().
    */
    class yk_var {
    public:
        virtual ~yk_var() {}

        /// Get the name of the var.
        /**
           @returns String containing name provided via yc_solution::new_var(),
           yk_solution::new_var(), or yk_solution::new_fixed_size_var()..
           @note Recall that scratch vars created via yc_solution::new_scratch_var()
           are temporary variables and thus are not accessible via the kernel APIs.
        */
        virtual const std::string& get_name() const =0;

        /// Get the number of dimensions used in this var.
        /**
           This may include domain, step, and/or miscellaneous dimensions.
           @returns Number of dimensions declared in the stencil code
           or created via yc_solution::new_var(),
           yk_solution::new_var(), or yk_solution::new_fixed_size_var().
        */
        virtual int get_num_dims() const =0;

        /// Get all the dimensions in this var.
        /**
           This may include domain, step, and/or miscellaneous dimensions.
           @returns List of names of all the dimensions.
        */
        virtual string_vec
        get_dim_names() const =0;

        /// Get the number of _domain_ dimensions used in this var.
        /**
           @returns Number of domain dimensions declared in the stencil code
           or created via yc_solution::new_var(),
           yk_solution::new_var(), or yk_solution::new_fixed_size_var().
        */
        virtual int get_num_domain_dims() const =0;

        /// Determine whether specified dimension exists in this var.
        /**
           @returns `true` if dimension exists (including step-dimension),
           `false` otherwise.
        */
        virtual bool
        is_dim_used(const std::string& dim) const =0;

        /// Determine whether this var is *not* automatically resized based on the solution.
        /**
           @returns `true` if this var was created via yk_solution::new_fixed_size_var()
           or `false` otherwise.
        */
        virtual bool is_fixed_size() const =0;

        /// Get the first valid index in this rank in the specified dimension.
        /**
           This is a convenience function that provides the first possible
           index in any var dimension regardless of the dimension type.
           If `dim` is a domain dimension, returns the first accessible index
           in the left padding area.
           It is equivalent to get_first_misc_index(dim)
           for a misc dimension, and get_first_valid_step_index()
           for the step dimension.
           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns the first valid index.
        */
        virtual idx_t
        get_first_local_index(const std::string& dim
                                /**< [in] Name of dimension to get.  Must be one of
                                   the names from get_dim_names(). */ ) const =0;

        /// Get the first valid index in this rank in all dimensions in this var.
        /**
           See get_first_local_index().
           @returns vector of first valid indices.
        */
        virtual idx_t_vec
        get_first_local_index_vec() const =0;

        /// Get the last index in this rank in the specified dimension.
        /**
           This is a convenience function that provides the last possible
           index in any var dimension regardless of the dimension type.
           If `dim` is a domain dimension, returns the last accessible index
           in the right padding area.
           It is equivalent to get_last_misc_index(dim)
           for a misc dimension, and get_last_valid_step_index()
           for the step dimension.
           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns the last valid index.
        */
        virtual idx_t
        get_last_local_index(const std::string& dim
                               /**< [in] Name of dimension to get.  Must be one of
                                  the names from get_dim_names(). */ ) const =0;

        /// Get the last valid index in this rank in all dimensions in this var.
        /**
           See get_last_local_index().
           @returns vector of last valid indices.
        */
        virtual idx_t_vec
        get_last_local_index_vec() const =0;

        /// Get the number of elements allocated in the specified dimension.
        /**
           For the domain dimensions, this includes the rank-domain and padding sizes.
           See the "Detailed Description" for \ref yk_var for information on var sizes.
           For any dimension `dim`, `get_alloc_size(dim) ==
           get_last_local_index(dim) - get_first_local_index(dim) + 1`;
           @returns allocation size in number of elements (not bytes).
        */
        virtual idx_t
        get_alloc_size(const std::string& dim
                       /**< [in] Name of dimension to get. Must be one of
                          the names from get_dim_names(). */ ) const =0;

        /// Get the number of elements allocated in all dimensions in this var.
        /**
           See get_alloc_size().
           @returns vector of allocation sizes in number of elements (not bytes).
        */
        virtual idx_t_vec
        get_alloc_size_vec() const =0;

        /// Get the first valid index in the step dimension.
        /**
           The valid step indices in a var are updated by calling yk_solution::run_solution()
           or one of the element-setting API functions.
           Equivalient to get_first_local_index(dim), where `dim` is the step dimension.
           @returns the first index in the step dimension that can be used in one of the
           element-getting API functions.
           This var must use the step index.
        */
        virtual idx_t
        get_first_valid_step_index() const =0;

        /// Get the last valid index in the step dimension.
        /**
           The valid step indices in a var are updated by calling yk_solution::run_solution()
           or one of the element-setting API functions.
           Equivalient to get_last_local_index(dim), where `dim` is the step dimension.
           @returns the last index in the step dimension that can be used in one of the
           element-getting API functions.
           This var must use the step index.
        */
        virtual idx_t
        get_last_valid_step_index() const =0;
        
        /// Get the domain size for this rank in the specified dimension.
        /**
           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's size.
           @returns The same value as yk_solution::get_rank_domain_size() if
           is_fixed_size() returns `false` or the fixed sized provided via
           yk_solution::new_fixed_size_var() otherwise.
        */
        virtual idx_t
        get_rank_domain_size(const std::string& dim
                             /**< [in] Name of dimension to get.  Must be one of
                                the names from yk_solution::get_domain_dim_names(). */) const =0;

        /// Get the domain size for this rank in all domain dimensions in this var.
        /**
           See get_rank_domain_size().
           @returns vector of values, one for each domain dimension in this var.
        */
        virtual idx_t_vec
        get_rank_domain_size_vec() const =0;

        /// Get the first index of the sub-domain in this rank in the specified dimension.
        /**
           Does _not_ include indices of padding area.
           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns The same value as yk_solution::get_first_rank_domain_index() if
           is_fixed_size() returns `false` or zero (0) otherwise.
        */
        virtual idx_t
        get_first_rank_domain_index(const std::string& dim
                                    /**< [in] Name of dimension to get.  Must be one of
                                       the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the first index of the sub-domain in this rank in all domain dimensions in this var.
        /**
           See get_first_rank_domain_index().
           @returns vector of values, one for each domain dimension in this var.
        */
        virtual idx_t_vec
        get_first_rank_domain_index_vec() const =0;

        /// Get the last index of the sub-domain in this rank in the specified dimension.
        /**
           Does _not_ include indices of padding area.
           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns The same value as yk_solution::get_last_rank_domain_index() if
           is_fixed_size() returns `false` or one less than the fixed sized provided via
           yk_solution::new_fixed_size_var() otherwise.
        */
        virtual idx_t
        get_last_rank_domain_index(const std::string& dim
                                    /**< [in] Name of dimension to get.  Must be one of
                                       the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the last index of the sub-domain in this rank in all domain dimensions in this var.
        /**
           See get_last_rank_domain_index().
           @returns vector of values, one for each domain dimension in this var.
        */
        virtual idx_t_vec
        get_last_rank_domain_index_vec() const =0;

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
           the same value as yk_var::get_first_rank_domain_index()
           if the left halo has zero size.
        */
        virtual idx_t
        get_first_rank_halo_index(const std::string& dim
                                    /**< [in] Name of dimension to get.  Must be one of
                                       the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the first index of the left halo in this rank in all domain dimensions in this var.
        /**
           See get_first_rank_halo_index().
           @returns vector of values, one for each domain dimension in this var.
        */
        virtual idx_t_vec
        get_first_rank_halo_index_vec() const =0;
        
        /// Get the last index of the right halo in this rank in the specified dimension.
        /**
           @note This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns The last index of right halo in this rank or
           the same value as yk_var::get_last_rank_domain_index()
           if the right halo has zero size.
        */
        virtual idx_t
        get_last_rank_halo_index(const std::string& dim
                                    /**< [in] Name of dimension to get.  Must be one of
                                       the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the last index of the right halo in this rank in all domain dimensions in this var.
        /**
           See get_last_rank_halo_index().
           @returns vector of values, one for each domain dimension in this var.
        */
        virtual idx_t_vec
        get_last_rank_halo_index_vec() const =0;

        /// Get the actual left padding in the specified dimension.
        /**
           The left padding is the memory allocated before
           the domain in a given dimension.
           The left padding size includes the left halo size.
           The value may be slightly
           larger than that provided via set_left_min_pad_size(), etc. due to rounding.
           @returns Elements in left padding in given dimension.
        */
        virtual idx_t
        get_left_pad_size(const std::string& dim
                     /**< [in] Name of dimension to get.
                         Must be one of
                         the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the actual right padding in the specified dimension.
        /**
           The right padding is the memory allocated after
           the domain in a given dimension.
           The right padding size includes the right halo size.
           The value may be slightly
           larger than that provided via set_right_min_pad_size(), etc. due to rounding.
           @returns Elements in right padding in given dimension.
        */
        virtual idx_t
        get_right_pad_size(const std::string& dim
                     /**< [in] Name of dimension to get.
                         Must be one of
                         the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the actual extra left padding in the specified dimension.
        /**
           The *extra* padding size is the left padding size minus the left halo size.
           @returns Elements in padding in given dimension before the
           left halo area.
        */
        virtual idx_t
        get_left_extra_pad_size(const std::string& dim
                           /**< [in] Name of dimension to get.
                              Must be one of
                              the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the actual extra right padding in the specified dimension.
        /**
           The *extra* padding size is the right padding size minus the right halo size.
           @returns Elements in padding in given dimension after the
           right halo area.
        */
        virtual idx_t
        get_right_extra_pad_size(const std::string& dim
                           /**< [in] Name of dimension to get.
                              Must be one of
                              the names from yk_solution::get_domain_dim_names(). */ ) const =0;

        /// Get the first index of a specified miscellaneous dimension.
        /**
           Equivalent to get_first_local_index(dim), where `dim` is a misc dimension.
           @returns the first valid index in a non-step and non-domain dimension.
        */
        virtual idx_t
        get_first_misc_index(const std::string& dim
                             /**< [in] Name of dimension to get.  Must be one of
                                the names from yk_solution::get_misc_dim_names(). */ ) const =0;

        /// Get the last index of a specified miscellaneous dimension.
        /**
           Equivalent to get_last_local_index(dim), where `dim` is a misc dimension.
           @returns the last valid index in a non-step and non-domain dimension.
        */
        virtual idx_t
        get_last_misc_index(const std::string& dim
                            /**< [in] Name of dimension to get.  Must be one of
                               the names from yk_solution::get_misc_dim_names(). */ ) const =0;

        /// Determine whether the given indices refer to an accessible element in this rank.
        /**
           Provide indices in a list in the same order returned by get_dim_names() for this var.
           Domain index values are relative to the *overall* problem domain.
           @returns `true` if index values fall within the range returned by
           get_first_local_index(dim) and get_last_local_index(dim) for each dimension
           `dim` in the var; `false` otherwise.
        */
        virtual bool
        are_indices_local(const idx_t_vec& indices
                          /**< [in] List of indices, one for each var dimension. */ ) const =0;

#ifndef SWIG
        /// Determine whether the given indices refer to an accessible element in this rank.
        /**
           See are_indices_local().
        */
        virtual bool
        are_indices_local(const idx_t_init_list& indices
                          /**< [in] List of indices, one for each var dimension. */ ) const =0;
#endif

        /// Read the value of one element in this var.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall between the values returned by
           get_first_local_index() and get_last_local_index(), inclusive,
           for each dimension in the var.
           @returns value in var at given indices.
        */
        virtual double
        get_element(const idx_t_vec& indices
                    /**< [in] List of indices, one for each var dimension. */ ) const =0;

#ifndef SWIG
        /// Read the value of one element in this var.
        /**
           See get_element().
           @returns value in var at given indices.
        */
        virtual double
        get_element(const idx_t_init_list& indices
                    /**< [in] List of indices, one for each var dimension. */ ) const =0;
#endif

        /// Set the value of one element in this var.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           If the var uses the step dimension, the value of the step index
           will be used to update the current valid step indices in the var.
           If `strict_indices` is `false` and any non-step index values
           are invalid as defined by are_indices_local(),
           the API will have no effect and return zero (0).
           If `strict_indices` is `true` and any non-step index values
           are invalid, the API will throw an exception.
           If storage has not been allocated for this var, this will have no effect
           and return zero (0) if `strict_indices` is `false`,
           or it will throw an exception if `strict_indices` is `true`.
           @note The parameter value is a double-precision floating-point value, but
           it will be converted to single-precision if
           yk_solution::get_element_bytes() returns 4.
           @returns Number of elements set, which will be one (1) if the indices
           are valid and zero (0) if they are not.
        */
        virtual idx_t
        set_element(double val /**< [in] Element in var will be set to this. */,
                    const idx_t_vec& indices
                    /**< [in] List of indices, one for each var dimension. */,
                    bool strict_indices = true
                    /**< [in] If true, indices must be within domain or padding.
                       If false, indices outside of domain and padding result
                       in no change to var. */ ) =0;

#ifndef SWIG
        /// Set the value of one element in this var.
        /**
           See set_element().
           @returns Number of elements set.
        */
        virtual idx_t
        set_element(double val /**< [in] Element in var will be set to this. */,
                    const idx_t_init_list& indices
                    /**< [in] List of indices, one for each var dimension. */,
                    bool strict_indices = true
                    /**< [in] If true, indices must be within domain or padding.
                       If false, indices outside of domain and padding result
                       in no change to var. */ ) =0;
#endif

        /// Copy elements within specified subset of this var into a buffer.
        /**
           Reads all elements from `first_indices` to `last_indices` in each dimension
           and writes them to consecutive memory locations in the buffer.
           Indices in the buffer progress in row-major order, i.e.,
           traditional C-language layout.
           The buffer pointed to must contain the number of bytes equal to
           yk_solution::get_element_bytes() multiplied by the number of
           elements in the specified slice.
           Since the reads proceed in row-major order, the last index is "unit-stride"
           in the buffer.

           Provide indices in two lists in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall between the values returned by
           get_first_local_index() and get_last_local_index(), inclusive.
           @returns Number of elements read.
        */
        virtual idx_t
        get_elements_in_slice(void* buffer_ptr
                              /**< [out] Pointer to buffer where values will be written. */,
                              const idx_t_vec& first_indices
                              /**< [in] List of initial indices, one for each var dimension. */,
                              const idx_t_vec& last_indices
                              /**< [in] List of final indices, one for each var dimension. */ ) const =0;

        /// Atomically add to the value of one var element.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall between the values returned by
           get_first_local_index() and get_last_local_index(), inclusive.
           Updates are OpenMP atomic, meaning that this function can be called by
           several OpenMP threads without causing a race condition.
           If storage has not been allocated for this var, this will have no effect
           and return zero (0) if `strict_indices` is `false`,
           or it will throw an exception if `strict_indices` is `true`.
           @note The parameter value is a double-precision floating-point value, but
           it will be converted to single-precision if
           yk_solution::get_element_bytes() returns 4.
           @returns Number of elements updated.
        */
        virtual idx_t
        add_to_element(double val /**< [in] This value will be added to element in var. */,
                       const idx_t_vec& indices
                       /**< [in] List of indices, one for each var dimension. */,
                       bool strict_indices = true
                       /**< [in] If true, indices must be within domain or padding.
                          If false, indices outside of domain and padding result
                          in no change to var. */ ) =0;

#ifndef SWIG
        /// Atomically add to the value of one var element.
        /**
           See add_to_element().
           @returns Number of elements set.
        */
        virtual idx_t
        add_to_element(double val /**< [in] This value will be added to element in var. */,
                       const idx_t_init_list& indices
                       /**< [in] List of indices, one for each var dimension. */,
                       bool strict_indices = true
                       /**< [in] If true, indices must be within domain or padding.
                          If false, indices outside of domain and padding result
                          in no change to var. */ ) =0;
#endif

        /// Initialize all var elements to the same value.
        /**
           Sets all allocated elements, including those in the domain and padding
           area to the same specified value.
           If storage has not been allocated, this will have no effect.
           @note The parameter is a double-precision floating-point value, but
           it will be converted to single-precision if
           yk_solution::get_element_bytes() returns 4.
        */
        virtual void
        set_all_elements_same(double val /**< [in] All elements will be set to this. */ ) =0;

        /// Initialize var elements within specified subset of the var to the same value.
        /**
           Sets all elements from `first_indices` to `last_indices` in each dimension to the
           specified value.
           Provide indices in two lists in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall between the values returned by
           get_first_local_index() and get_last_local_index(), inclusive,
           if `strict_indices` is `true`.
           If storage has not been allocated for this var, this will have no effect
           and return zero (0) if `strict_indices` is `false`,
           or it will throw an exception if `strict_indices` is `true`.
           @returns Number of elements set.
        */
        virtual idx_t
        set_elements_in_slice_same(double val /**< [in] All elements in the slice will be set to this. */,
                                   const idx_t_vec& first_indices
                                   /**< [in] List of initial indices, one for each var dimension. */,
                                   const idx_t_vec& last_indices
                                   /**< [in] List of final indices, one for each var dimension. */,
                                   bool strict_indices = true
                                   /**< [in] If true, indices must be within domain or padding.
                                      If false, only elements within the allocation of this var
                                      will be set, and elements outside will be ignored. */ ) =0;

        /// Set var elements within specified subset of the var from values in a buffer.
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
           Index values must fall between the values returned by
           get_first_local_index() and get_last_local_index(), inclusive.
           If storage has not been allocated for this var, this will
           throw an exception.
           @returns Number of elements written.
        */
        virtual idx_t
        set_elements_in_slice(const void* buffer_ptr
                              /**< [out] Pointer to buffer where values will be read. */,
                              const idx_t_vec& first_indices
                              /**< [in] List of initial indices, one for each var dimension. */,
                              const idx_t_vec& last_indices
                              /**< [in] List of final indices, one for each var dimension. */ ) =0;

#ifdef COPY_SLICE_IMPLEMENTED
        /// Copy specified var elements from another (source) var into this (target) var.
        /**
           Reads elements starting at `first_source_indices` in the `source` var and
           writes them starting at `first_target_indices` and ending
           at `last_target_indices` in this var.
           The size of the copy is determined by the differences bewteen
           `first_target_indices` and `last_target_indices` in each dimension.
           Provide indices in the same order returned by get_dim_names().
           Indices are relative to the *overall* problem domain.
           Index values must fall between the values returned by
           get_first_local_index() and get_last_local_index(), inclusive, for
           each dimension in both vars.
           @returns Number of elements copied.
        */
        virtual idx_t
        set_elements_in_slice(const yk_var_ptr source
                              /**< [in] Var from which elements will be read. */,
                              const idx_t_vec& first_source_indices
                              /**< [in] List of starting indices in the source var,
                                 one for each var dimension. */,
                              const idx_t_vec& first_target_indices
                              /**< [in] List of starting indices in this (target) var,
                                 one for each var dimension. */,
                              const idx_t_vec& last_target_indices
                              /**< [in] List of final indices in this (target) var,
                                 one for each var dimension. */ ) =0;
#endif
        
        /// Format the indices for human-readable display.
        /**
           Provide indices in a list in the same order returned by get_dim_names().
           @returns A string containing the var name and the index values.
        */
        virtual std::string
        format_indices(const idx_t_vec& indices
                       /**< [in] List of indices, one for each var dimension. */ ) const =0;

#ifndef SWIG
        /// Format the indices for human-readable display.
        /**
           See format_indices().
           @returns A string containing the var name and the index values.
        */
        virtual std::string
        format_indices(const idx_t_init_list& indices
                       /**< [in] List of indices, one for each var dimension. */ ) const =0;
#endif

        /* Advanced APIs for yk_var found below are not needed for most applications. */

        /// **[Advanced]** Get the maximum L1-norm of a neighbor rank for halo exchange.
        /**
           This setting determines which MPI neighbors participate in a halo exchange.
           The L1-norm is also known as the "Manhattan distance" or "taxicab norm".
           In this case, the distance in *each* domain dimension can be only zero or one,
           so the sum can range from zero to the number of domain dimensions.
           
           Examples for a domain size with 2 spatial dimensions (e.g., "x" and "y"):
           * L1-norm = 0: no halos are exchanged for this var.
           * L1-norm = 1: halos are exchanged between "up", "down", "left" and "right"
             neighbors.
           * L1-norm = 2: halos are exchanged as above plus diagonal neighbors.

           The actual exchanges are further controlled by the size of the halo in each
           direction per get_halo_size().
           
           @returns L1-norm, ranging from zero to number of domain dimensions.
        */
        virtual int
        get_halo_exchange_l1_norm() const =0;
        
        /// **[Advanced]** Set the maximum L1-norm of a neighbor rank for halo exchange.
        /**
           This should only be used to override the value calculated automatically by
           the YASK compiler.
           @see get_halo_exchange_l1_norm().
        */
        virtual void
        set_halo_exchange_l1_norm(int norm
                                  /**< [in] Maximum L1-norm of neighbor rank
                                     with which to exchange halos. */) =0;
        
        /// **[Advanced]** Get whether the allocation of the step dimension of this var can be modified at run-time.
        /**
           See set_alloc_size().
         */
        virtual bool
        is_dynamic_step_alloc() const =0;

        /// **[Advanced]** Set the default preferred NUMA node on which to allocate data.
        /**
           This value is used when allocating data for this var.
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
           @returns Current setting of preferred NUMA node for this var.
        */
        virtual int
        get_numa_preferred() const =0;

        /// **[Advanced]** Set the minimum left padding in the specified dimension.
        /**
           This sets the minimum number of elements in this var in the left padding area.
           This padding area can be used for required halo areas.
           This function may be useful in the unusual case where the final halo size
           is unknown when the storage is allocated.

           Call get_left_pad_size() to determine the actual padding size for the var.
           See additional behavior related to setting pad size under yk_solution::set_min_pad_size().
           See the "Detailed Description" for \ref yk_var for information on var sizes.
        */
        virtual void
        set_left_min_pad_size(const std::string& dim
                              /**< [in] Name of dimension to set.
                                 Must be one of
                                 the names from yk_solution::get_domain_dim_names(). */,
                              idx_t size
                              /**< [in] Minimum number of elements to allocate
                                 before the domain size. */ ) =0;

        /// **[Advanced]** Set the minimum right padding in the specified dimension.
        /**
           This sets the minimum number of elements in this var in the right padding area.
           This padding area can be used for required halo areas.
           This function may be useful in the unusual case where the final halo size
           is unknown when the storage is allocated.

           Call get_right_pad_size() to determine the actual padding size for the var.
           See additional behavior related to setting pad size under yk_solution::set_min_pad_size().
           See the "Detailed Description" for \ref yk_var for information on var sizes.
        */
        virtual void
        set_right_min_pad_size(const std::string& dim
                              /**< [in] Name of dimension to set.
                                 Must be one of
                                 the names from yk_solution::get_domain_dim_names(). */,
                              idx_t size
                              /**< [in] Minimum number of elements to allocate
                                 after the domain size. */ ) =0;

        /// **[Advanced]** Set the minimum padding in the specified dimension.
        /**
           Shorthand for calling set_left_min_pad_size() and set_right_min_pad_size().
        */
        virtual void
        set_min_pad_size(const std::string& dim
                         /**< [in] Name of dimension to set.
                            Must be one of
                            the names from yk_solution::get_domain_dim_names(). */,
                         idx_t size
                         /**< [in] Minimum number of elements to allocate
                            before and after the domain size. */ ) =0;

        /// **[Advanced]** Set the left halo size in the specified dimension.
        /**
           This value is typically set by the stencil compiler, but
           this function allows you to override that value.
           If the left halo is set to a value larger than the left padding size, the
           left padding size will be automatically increase to accommodate it.
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
           right padding size will be automatically increase to accommodate it.
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
           Setting an allocation is only allowed in the following cases.
           Vars created via yc_solution::new_var() are defined at YASK
           compile time, and vars created via \ref yk_solution methods
           are defined at YASK kernel run time.
           
           Var creation time | Var creation method | Step dim | Domain dim | Misc dim |
           -------------------|----------------------|----------|------------|----------|
           Compile-time | yc_solution::new_var() + yc_var::set_dynamic_step_alloc (false) [1] | No | No | Yes [2] |
           Compile-time | yc_solution::new_var() + yc_var::set_dynamic_step_alloc (true) [1] | Yes | No | Yes [2] |
           Run-time | yk_solution::new_var() | Yes | No | Yes |
           Run-time | yk_solution::new_fixed_size_var() [3] | Yes | Yes | Yes |

           @note [1] By default, variables created via yc_solution::new_var()
           do _not_ allow dynamic step allocation.
           @note [2] Misc dim allocations cannot be changed for compile-time vars if the YASK
           compiler was run with the "-interleave-misc" option.
           @note [3] The term "fixed" in yk_solution::new_fixed_size_var() means that the
           domain size will not change automatically when its solution domain
           size changes. It does not mean that the sizes cannot be changed
           via the APIs--quite the opposite.

           The allocation size cannot be changed after data storage
           has been allocated for this var.
        */
        virtual void
        set_alloc_size(const std::string& dim
                       /**< [in] Name of dimension to set.
                          Must be a domain dimension or
                          a misc dimension for user-created vars. */,
                       idx_t size /**< [in] Number of elements to allocate. */ ) =0;

        /// **[Advanced]** Set the first index of a specified miscellaneous dimension.
        /**
           Sets the first valid index in a non-step and non-domain dimension.
           After calling this function, the last valid index will be the first index
           as set by this function plus the allocation size set by set_alloc_size()
           minus one.
        */
        virtual void
        set_first_misc_index(const std::string& dim
                             /**< [in] Name of dimension to get.  Must be one of
                                the names from yk_solution::get_misc_dim_names(). */,
                             idx_t idx /**< [in] New value for first index.
                                        May be negative. */ ) =0;

        /// **[Advanced]** Determine whether storage has been allocated.
        /**
           @returns `true` if storage has been allocated,
           `false` otherwise.
        */
        virtual bool
        is_storage_allocated() const =0;

        /// **[Advanced]** Determine size of raw storage in bytes.
        /**
           @returns Minimum number of bytes required for
           storage given the current domain size and padding settings.
        */
        virtual idx_t
        get_num_storage_bytes() const =0;

        /// **[Advanced]** Determine size of raw storage in elements.
        /**
           @returns get_num_storage_bytes() / yk_solution.get_element_bytes().
        */
        virtual idx_t
        get_num_storage_elements() const =0;

        /// **[Advanced]** Explicitly allocate data-storage memory for this var.
        /**
           Amount of allocation is calculated based on domain, padding, and
           step-dimension allocation sizes.
           Any pre-existing storage will be released before allocation as via release_storage().
           See allocation options in the "Detailed Description" for \ref yk_var.
         */
        virtual void
        alloc_storage() =0;

        /// **[Advanced]** Explicitly release any allocated data-storage for this var.
        /**
           This will release storage allocated via any of the options
           described in the "Detailed Description" for \ref yk_var.
        */
        virtual void
        release_storage() =0;

        /// **[Advanced]** Determines whether storage layout is the same as another var.
        /**
           In order for the storage layout to be identical, the following
           must be the same:
           - Number of dimensions.
           - Name of each dimension, in the same order.
           - Vector folding in each dimension.
           - Allocation size in each dimension.
           - Rank (local) domain size in each domain dimension.
           - Padding size in each domain dimension.

           The following do not have to be identical:
           - Halo size.

           @returns `true` if storage for this var has the same layout as
           `other` or `false` otherwise.
        */
        virtual bool
        is_storage_layout_identical(const yk_var_ptr other) const =0;

        /// **[Advanced]** Merge this var with another var.
        /**
           After calling this API, this var 
           var will effectively become another reference to the `source` var.
           Any subsequent API applied to this var or the
           `source` var will access the same data and/or
           effect the same changes.

           Storage implications:
           - Any pre-existing storage in this var will be released.
           - The storage of the this var will become
           allocated or unallocated depending on that of the source var.
           - After fusing, calling release_storage() on this var
           or the `source` var will apply to both.

           To ensure that the kernels created by the YASK compiler work
           properly, if this var is used in a kernel, the dimensions and
           fold-lengths of the `source` var must be identical or an
           exception will the thrown.  If the `source` var is a fixed-size
           var, the storage, local domain sizes, halos, etc. of the var
           must be set to be compatible with the solution before 
           calling yk_solution::prepare_solution(). Otherwise,
           yk_solution::prepare_solution() will throw an exception.

           See allocation options and more information about var sizes
           in the "Detailed Description" for \ref yk_var.
        */
        virtual void
        fuse_vars(yk_var_ptr source
                   /**< [in] Var to be merged with this var. */) =0;

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
           and some other var, any given element index applied to both vars
           will refer to an element at the same offset into their respective
           data buffers.

           Thus,
           - You can perform element-wise unary mathematical operations on
           all elements of a var via its raw buffer, e.g., add some constant
           value to all elements.
           - If the layouts of two vars are identical, you can use their
           raw buffers to copy all data from one to the other or
           perform element-wise binary mathematical operations on them,
           e.g., add all elements from one var to another.

           The following assumptions are not safe:
           - Any expectations regarding the relationship between an element
           index and that element's offset from the beginning of the buffer
           such as row-major or column-major layout.
           - All elements in the buffer are part of the rank domain or halo.
           - All elements in the buffer contain valid floating-point values.

           Thus,
           - You should not perform any operations dependent on
           the logical indices of any element via raw buffer, e.g., matrix
           multiply.

           @returns Pointer to raw data storage if is_storage_allocated()
           returns `true` or NULL otherwise.
        */
        virtual void* get_raw_storage_buffer() =0;


        /// **[Deprecated]** Use get_first_local_index().
        YASK_DEPRECATED
        virtual idx_t
        get_first_rank_alloc_index(const std::string& dim) const {
            return get_first_local_index(dim);
        }

        /// **[Deprecated]** Use get_last_local_index().
        YASK_DEPRECATED
        virtual idx_t
        get_last_rank_alloc_index(const std::string& dim) const {
            return get_last_local_index(dim);
        }

    }; // yk_var.

    /// **[Deprecated]** Use yk_var.
    YASK_DEPRECATED
    typedef yk_var yk_grid;
    
    /** @}*/
} // namespace yask.
