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
#include <vector>
#include <memory>
#include <cinttypes>

namespace yask {

    /// Type to use for indexing grids.
    /** Index types are signed to allow negative indices in halos. */
#ifdef SWIG
    typedef long int idx_t;     // SWIG doesn't seem to understand int64_t.
#else
    typedef std::int64_t idx_t;
#endif

    // Forward declarations of classes and pointers.
    class yk_env;
    typedef std::shared_ptr<yk_env> yk_env_ptr;
    class yk_solution;
    typedef std::shared_ptr<yk_solution> yk_solution_ptr;
    class yk_grid;
    typedef std::shared_ptr<yk_grid> yk_grid_ptr;

    /// Factory to create a stencil solution.
    class yk_factory {
    public:
        virtual ~yk_factory() {}

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

        /// Create a stencil solution by copying the settings from another.
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

        /// Wait until all ranks have reached this point.
        /**
           If MPI is enabled, calls MPI_Barrier().
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

        /// Get the name of the solution.
        /**
           @returns String containing the solution name provided during stencil compilation.
        */
        virtual const std::string&
        get_name() const =0;

        /// Get the floating-point precision size.
        /**
           @returns Number of bytes in each FP element.
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
           Does *not* include the step dimension.
           @returns Number of dimensions that define the problem domain.
        */
        virtual int
        get_num_domain_dims() const =0;

        /// Get the name of the specified domain dimension.
        /**
            @returns String containing name of domain dimension. 
        */
        virtual std::string
        get_domain_dim_name(int n /**< [in] Index of dimension between zero (0)
                                     and get_num_domain_dims()-1. */ ) const =0;

        /// Get all the domain dimension names.
        /**
           @returns List of all domain-dimension names.
        */
        virtual std::vector<std::string>
        get_domain_dim_names() const =0;

        /// Set the size of the solution domain for this rank.
        /**
           The domain defines the number of points that will be evaluated with the stencil(s). 
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
           size to a multiple of the number of points in a vector-cluster in
           each dimension whenever possible.
           See the "Detailed Description" for \ref yk_grid for more information on grid sizes.
           There is no domain-size setting allowed in the
           solution-step dimension (usually "t"). 
        */
        virtual void
        set_rank_domain_size(const std::string& dim
                             /**< [in] Name of dimension to set.  Must be one of
                                the names from
                                yk_solution::get_domain_dim_name(). */,
                             idx_t size /**< [in] Points in the domain in this `dim`. */ ) =0;

        /// Get the domain size for this rank.
        /**
           @returns Current setting of rank domain size in specified dimension.
        */
        virtual idx_t
        get_rank_domain_size(const std::string& dim
                             /**< [in] Name of dimension to get.  Must be one of
                                the names from
                                yk_solution::get_domain_dim_name(). */) const =0;

        /// Set the minimum amount of grid padding for all grids.
        /**
           This sets the minimum number of points in each grid that is
           reserved outside of the rank domain in the given dimension.
           This padding area can be used for required halo regions.  At
           least the specified number of elements will be added to both
           sides, i.e., both "before" and "after" the domain.
           
           The *actual* padding size will be the largest of the following values,
           additionally rounded up based on the vector-folding dimensions
           and/or cache-line alignment:
           - Halo size.
           - Value provided by this function, set_min_pad_size().
           - Value provided by yk_grid::set_min_pad_size().
           
           The padding size cannot be changed after data storage
           has been allocated for a given grid.
           In addition, once a grid's padding is set, it cannot be reduced, only increased.
           Call yk_grid::get_pad_size() to determine the actual padding size for a given grid.
           See the "Detailed Description" for \ref yk_grid for more information on grid sizes.
           There is no padding allowed in the solution-step dimension (usually "t").
        */
        virtual void
        set_min_pad_size(const std::string& dim
                         /**< [in] Name of dimension to set.  Must
                            be one of the names from
                            yk_solution::get_domain_dim_name(). */,
                         idx_t size
                         /**< [in] Points in this `dim` applied
                            to both sides of the domain. */ ) =0;

        /// Get the minimum amount of grid padding for all grids.
        /**
           @returns Current setting of minimum amount of grid padding for all grids.
        */
        virtual idx_t
        get_min_pad_size(const std::string& dim
                         /**< [in] Name of dimension to get.  Must be one of
                            the names from
                            yk_solution::get_domain_dim_name(). */) const =0;

        /// Set the block size in the given dimension.
        /**
           This sets the approximate number of points that are evaluated in
           each "block".
           This is a performance setting and should not affect the functional
           correctness or total number of points evaluated.
           A block is typically the unit of work done by a
           top-level OpenMP thread.  The actual number of points evaluated
           in a block may be greater than the specified size due to rounding
           up to fold-cluster sizes.  The number of points in a block may
           also be smaller than the specified size when the block is at the
           edge of the domain. The block size cannot be set in the
           solution-step dimension (because temporal blocking is not yet enabled).
        */
        virtual void
        set_block_size(const std::string& dim
                       /**< [in] Name of dimension to set.  Must be one of
                          the names from
                          yk_solution::get_domain_dim_name(). */,
                       idx_t size /**< [in] Points in a block in this `dim`. */ ) =0;

        /// Get the block size.
        /**
           Returned value may be slightly larger than the value provided
           via set_block_size() due to rounding.
           @returns Current settings of block size.
        */
        virtual idx_t
        get_block_size(const std::string& dim
                        /**< [in] Name of dimension to get.  Must be one of
                           the names from
                           yk_solution::get_domain_dim_name(). */) const =0;

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
                         the names from
                         yk_solution::get_domain_dim_name(). */,
                      idx_t num /**< [in] Number of ranks in `dim`. */ ) =0;

        /// Get the number of MPI ranks in the given dimension.
        /**
           @returns Current setting of rank size.
        */
        virtual idx_t
        get_num_ranks(const std::string& dim
                      /**< [in] Name of dimension to get.  Must be one of
                         the names from
                         yk_solution::get_domain_dim_name(). */) const =0;

        /// Get the rank index in the specified dimension.
        /**
           The overall rank indices in the specified dimension will range from
           zero (0) to get_num_ranks() - 1, inclusive.
           @returns Zero-based index of this rank.
        */
        virtual idx_t
        get_rank_index(const std::string& dim
                       /**< [in] Name of dimension from get_domain_dim_name().
                          Cannot be the step dimension. */ ) const =0;

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
           @returns Pointer to the nth grid.
        */
        virtual yk_grid_ptr
        get_grid(int n /**< [in] Index of grid between zero (0)
                              and get_num_grids()-1. */ ) =0;

        /// Get all the grids.
        /**
           @returns List of all grids in the solution.
        */
        virtual std::vector<yk_grid_ptr>
        get_grids() =0;

        /// **[Advanced]** Add a new grid to the solution.
        /**
           This is typically not needed because grids are usually pre-defined
           by the solution itself via the stencil compiler.
           However, a grid may be created explicitly via this function
           in order to use it for purposes other than by the
           pre-defined stencils within the current solution.
           A new grid's domain size will be the same as that returned by
           get_rank_domain_size().
           A new grid's initial padding size will be the same as that returned by
           get_min_pad_size().
           After creating a new grid, you can increase its padding
           sizes via yk_grid::set_min_pad_size().
           A new grid contains only the meta-data for the grid; data storage
           is not yet allocated.
           Storage may be allocated in any of the methods listed
           in the "Detailed Description" for \ref yk_grid.
           @returns Pointer to the new grid.
        */
        virtual yk_grid_ptr
        new_grid(const std::string& name /**< [in] Unique name of the grid; must be
                                            a valid C++ identifier and unique
                                            across grids. */,
                 const std::string& dim1 = "" /**< [in] Name of 1st dimension. All
                                                 dimension names must be valid C++
                                                 identifiers and unique within this
                                                 grid. */,
                 const std::string& dim2 = "" /**< [in] Name of 2nd dimension. */,
                 const std::string& dim3 = "" /**< [in] Name of 3rd dimension. */,
                 const std::string& dim4 = "" /**< [in] Name of 4th dimension. */,
                 const std::string& dim5 = "" /**< [in] Name of 5th dimension. */,
                 const std::string& dim6 = "" /**< [in] Name of 6th dimension. */ ) =0;

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

        /// Get the first logical index of the sub-domain in this rank in the specified dimension.
        /**
           This returns the first index at the beginning of the domain.
           Points within the domain in this rank lie between the values returned by
           get_first_rank_domain_index() and get_last_rank_domain_index(), inclusive.
           If there is only one MPI rank, this is typically zero (0).
           If there is more than one MPI rank, the value depends
           on the the rank's position within the overall problem domain.
           This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns First logical domain index in this rank. 
        */
        virtual idx_t
        get_first_rank_domain_index(const std::string& dim
                                    /**< [in] Name of dimension from get_domain_dim_name().
                                       Cannot be the step dimension. */ ) const =0;

        /// Get the last logical index of the sub-domain in this rank the specified dimension.
        /**
           This returns the last index within the domain in this rank
           (*not* one past the end).
           If there is only one MPI rank, this is typically one less than the value
           provided by set_rank_domain_size().
           If there is more than one MPI rank, the value depends
           on the the rank's position within the overall problem domain.
           See get_first_rank_domain_index() for more information.
           This function should be called only *after* calling prepare_solution()
           because prepare_solution() assigns this rank's position in the problem domain.
           @returns Last logical index in this rank.
        */
        virtual idx_t
        get_last_rank_domain_index(const std::string& dim
                                   /**< [in] Name of dimension from get_domain_dim_name().
                                      Cannot be the step dimension. */ ) const =0;

        /// Get the overall problem size in the specified dimension.
        /**
           The overall domain indices in the specified dimension will range from
           zero (0) to get_overall_domain_size() - 1, inclusive.
           Call get_first_rank_domain_index() and get_last_rank_domain_index()
           to find the subset of this domain in each rank.
           This function should be called only *after* calling prepare_solution()
           because prepare_solution() obtains the sub-domain sizes from other ranks.
           @returns Sum of the ranks' domain sizes in the given dimension.
        */
        virtual idx_t
        get_overall_domain_size(const std::string& dim
                                /**< [in] Name of dimension from get_domain_dim_name().
                                   Cannot be the step dimension. */ ) const =0;

        /// Run the stencil solution for one step.
        /**
           The stencil(s) in the solution are applied
           at the given step index
           across the entire domain as returned by yk_solution::get_overall_domain_size().
           MPI halo exchanges will occur as necessary.
           Since this function initiates MPI communication, it must be called
           on all MPI ranks, and it will block until all ranks have completed.
        */
        virtual void
        run_solution(idx_t step_index /**< [in] Index in the step dimension */ ) =0;

        /// Run the stencil solution for the specified number of steps.
        /**
           The stencil(s) in the solution are applied from
           the first to last step index, inclusive,
           across the entire domain as returned by yk_solution::get_overall_domain_size().
           MPI halo exchanges will occur as necessary.
           Since this function initiates MPI communication, it must be called
           on all MPI ranks, and it will block until all ranks have completed.
        */
        virtual void
        run_solution(idx_t first_step_index /**< [in] First index in the step dimension */,
                     idx_t last_step_index /**< [in] Last index in the step dimension */ ) =0;

        /// **[Advanced]** Use data-storage from existing grids in specified solution.
        /**
           Calls yk_grid::share_storage() for each pair of grids that have the same name
           in this solution and the source solution.
           All conditions listed in yk_grid::share_storage() must hold for each of the pairs.
        */
        virtual void
        share_grid_storage(yk_solution_ptr source
                           /**< [in] Solution from which grid storage will be shared. */) =0;
    };

    /// A run-time grid.
    /**
       "Grid" is a generic term for any n-dimensional array.  A 0-dim grid
       is a scalar, a 1-dim grid is an array, etc.  A run-time grid contains
       data, unlike yc_grid, a compile-time grid variable.  
       
       Typically, access to each grid is obtained via yk_solution::get_grid().
       You may also use yk_solution::new_grid() if you need a grid that is not part
       of the pre-defined solution.
       
       In each dimension, grid sizes include the following components:
       - The *domain* is the points to which the stencils are applied.
       - The *padding* is the points outside the domain on either side
       and includes the halo.
       - The *halo* is the points just outside the domain which must be
       copied between ranks during halo exchanges. The halo is part of the padding.
       - The *extra padding* is the points outside the domain and halo on
       either side and thus does not include the halo.
       - The *allocation* includes the domain and the padding.
       
       All sizes are expressed in numbers of elements, which may be 4-byte (single precision)
       or 8-byte (double precision) floating-point values.
       
       Visually, in each of the non-step dimensions, these sizes are related as follows:
       <table>
       <tr><td>extra padding <td>halo <td rowspan="2">domain <td>halo <td>extra padding
       <tr><td colspan="2"><center>padding</center> <td colspan="2"><center>padding</center>
       <tr><td colspan="5"><center>allocation</center>
       </table>
       In the step dimension (usually "t"), there is no fixed domain size.
       But there is an allocation size, which is the number of values in the step
       dimension that are stored in memory. 
       Step-dimension indices "wrap-around" within this allocation to reuse memory.
       For example, if the step dimension is "t", and the t-dimension allocation size is 3,
       then t=-2, t=0, t=3, t=6, ..., t=303, etc. would all alias to the same spatial values in memory.

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
       
       Domain sizes specified via yk_solution::set_rank_domain_size() apply to each process or *rank*.
       If MPI is not enabled, a rank's domain is the entire problem size.
       If MPI is enabled, the domains of the ranks are logically abutted to create the 
       overall problem domain in each dimension:
       <table>
       <tr><td>extra padding of rank A <td>halo of rank A <td>domain of rank A <td>domain of rank B
         <td>... <td>domain of rank Z <td>halo of rank Z <td>extra padding of rank Z
       <tr><td colspan="2"><center>padding of rank A</center>
         <td colspan="4"><center>overall problem domain</center>
         <td colspan="2"><center>padding of rank Z</center>
       </table>
       The beginning and ending overall-problem index that lies within a rank can be
       retrieved via yk_solution::get_first_rank_domain_index() and 
       yk_solution::get_last_rank_domain_index(), respectively.
       
       The intermediate halos and paddings also exist, but are not shown in the above diagram.
       They overlap the domains of adjacent ranks.
       For example, the left halo of rank B in the diagram would overlap the domain of rank A.
       Data in these overlapped regions is exchanged as needed during stencil application
       to maintain a consistent values as if there was only one rank.
    */
    class yk_grid {
    public:
        virtual ~yk_grid() {}

        /// Get the name of the grid.
        /**
           @returns String containing name provided via yc_solution::new_grid().
        */
        virtual const std::string& get_name() const =0;

        /// Get the number of dimensions used in this grid.
        /**
           Includes step dimension if it is a dimension of this grid.
           May be different than value returned from yk_solution::get_num_domain_dims().
           @returns Number of dimensions created via yc_solution::new_grid().
        */
        virtual int get_num_dims() const =0;

        /// Get the name of the specified dimension in this grid.
        /**
           Includes step dimension if it is a dimension of this grid.
           @note Domain dimensions are not necessarily in the same order as
           those returned via yk_solution::get_domain_dim_name().
           @returns String containing name of dimension created via new_grid().
        */
        virtual const std::string&
        get_dim_name(int n /**< [in] Index of dimension between zero (0)
                              and get_num_dims()-1. */ ) const =0;

        /// Get all the dimensions in this grid.
        /**
           Includes step dimension if it is a dimension of this grid.
           May be different than values returned from yk_solution::get_domain_dim_names().
           @returns List of names of all the dimensions.
        */
        virtual std::vector<std::string>
        get_dim_names() const =0;
        
        /// Determine whether specified dimension exists in this grid.
        /**
           @returns 'true' if dimension exists (including step-dimension),
           'false' otherwise.
        */
        virtual bool
        is_dim_used(const std::string& dim) const =0;

        /// Get the halo size in the specified dimension.
        /**
           This value is typically set by the stencil compiler.
           @returns Points in halo in given dimension.
        */
        virtual idx_t
        get_halo_size(const std::string& dim
                      /**< [in] Name of dimension from get_dim_name().
                         Cannot be the step dimension. */ ) const =0;
        
        /// **[Advanced]** Set the halo size in the specified dimension.
        /**
           This value is typically set by the stencil compiler, but
           this function allows you to override that value.
           If the halo is set to a value larger than the padding size, the
           padding size will be automatically increase to accomodate it.
           @note After data storage has been allocated, the halo size
           can only be set to a value less than or equal to the padding size
           in the given dimension.
           @returns Points in halo in given dimension.
        */
        virtual void
        set_halo_size(const std::string& dim
                      /**< [in] Name of dimension from get_dim_name().
                         Cannot be the step dimension. */,
                      idx_t size
                      /**< [in] Number of points in the halo. */ ) =0;

        /// Get the padding in the specified dimension.
        /**
           The padding size includes the halo size.
           The value may be slightly
           larger than that provided via set_min_pad_size()
           or yk_solution::set_min_pad_size() due to rounding.
           @returns Points in padding in given dimension.
        */
        virtual idx_t
        get_pad_size(const std::string& dim
                     /**< [in] Name of dimension from get_dim_name().
                        Cannot be the step dimension. */ ) const =0;

        /// Get the extra padding in the specified dimension.
        /**
           The *extra* padding size is the padding size minus the halo size.
           @returns Points in extra padding in given dimension.
        */
        virtual idx_t
        get_extra_pad_size(const std::string& dim
                           /**< [in] Name of dimension from get_dim_name().
                              Cannot be the step dimension. */ ) const =0;

        /// Set the padding in the specified dimension.
        /**
           This sets the minimum number of points in this grid that is
           reserved outside of the rank domain in the given dimension.
           This padding area can be used for required halo regions.
           The specified number of points is added to both sides, i.e., both "before" and
           "after" the domain.
           
           The *actual* padding size will be the largest of the following values,
           additionally rounded up based on the vector-folding dimensions
           and/or cache-line alignment:
           - Halo size.
           - Value provided by yk_solution::set_min_pad_size().
           - Value provided by this function, set_min_pad_size().
           
           The padding size cannot be changed after data storage
           has been allocated for this grid.
           In addition, once a grid's padding is set, it cannot be reduced, only increased.
           Call get_pad_size() to determine the actual padding size for the grid.
           See the "Detailed Description" for \ref yk_grid for information on grid sizes.
        */
        virtual void
        set_min_pad_size(const std::string& dim
                         /**< [in] Name of dimension from get_dim_name().
                            Cannot be the step dimension. */,
                         idx_t size
                         /**< [in] Minimum number of points to allocate beyond the domain size. */ ) =0;
        
        /// Get the storage allocation in the specified dimension.
        /**
           For the step dimension, this is the specified allocation and
           does not typically depend on the number of steps evaluated.
           For the non-step dimensions, this is yk_solution::get_rank_domain_size() + 
           (2 * (get_halo_size() + get_extra_pad_size())).
           See the "Detailed Description" for \ref yk_grid for information on grid sizes.
           @returns allocation in number of points (not bytes).
        */
        virtual idx_t
        get_alloc_size(const std::string& dim
                       /**< [in] Name of dimension from get_dim_name(). */ ) const =0;

        /// **[Advanced]** Set the number of points to allocate in the specified dimension.
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
                       /**< [in] Name of dimension from get_dim_name().
                          Must be the step dimension. */,
                       idx_t size /**< [in] Number of points to allocate. */ ) =0;

        /// Get the value of one grid point.
        /**
           Provide the number of indices equal to the number of dimensions in the grid.
           Indices beyond that will be ignored.
           Indices are relative to the *overall* problem domain.
           Index values must fall within the rank domain or padding area.
           See yk_solution::get_first_rank_domain_index(),
           yk_solution::get_last_domain_index(), and the 
           "Detailed Description" for \ref yk_grid for more information on grid sizes.
           @note The return value is a double-precision floating-point value, but
           it will be converted from a single-precision if 
           yk_solution::get_element_bytes() returns 4.
           @returns value in grid at given multi-dimensional location.
        */
        virtual double
        get_element(idx_t dim1_index=0 /**< [in] Index in dimension 1. */,
                    idx_t dim2_index=0 /**< [in] Index in dimension 2. */,
                    idx_t dim3_index=0 /**< [in] Index in dimension 3. */,
                    idx_t dim4_index=0 /**< [in] Index in dimension 4. */,
                    idx_t dim5_index=0 /**< [in] Index in dimension 5. */,
                    idx_t dim6_index=0 /**< [in] Index in dimension 6. */ ) const =0;

        /// Get the value of one grid point.
        /**
           Same as version of get_element() with individual indices, but
           indices are provided in a list per the order returned by
           get_dim_names().
           @returns value in grid at given multi-dimensional location. 
        */
        virtual double
        get_element(const std::vector<idx_t>& indices
                    /**< [in] List of indices, one for each grid dimension. */ ) const =0;

        /// Get grid points within specified subset of the grid.
        /**
           Reads all elements from 'first' to 'last' indices in each dimension in 
           row-major order, and writes them to consecutive memory locations,
           starting at 'buffer_ptr'.
           The buffer pointed to must contain the number of bytes equal to
           yk_solution::get_element_bytes() multiplied by the number of 
           points in the specified slice.
           Since the reads proceed in row-major order, the last index is "unit-stride"
           in the buffer.
           Indices are relative to the *overall* problem domain.
           Index values must fall within the rank domain or padding area.
           @returns Number of elements read.
        */
        virtual idx_t
        get_elements_in_slice(void* buffer_ptr
                              /**< [out] Pointer to buffer where values will be written. */,
                              const std::vector<idx_t>& first_indices
                              /**< [in] List of beginning indices, one for each grid dimension. */,
                              const std::vector<idx_t>& last_indices
                              /**< [in] List of ending indices, one for each grid dimension. */ ) const =0;
        
        /// Set the value of one grid point.
        /**
           Caller must provide the number of indices equal to the number of dimensions in the grid.
           Indices beyond that will be ignored.
           Indices are relative to the *overall* problem domain.
           Index values must fall within the rank domain or padding area.
           See yk_solution::get_first_rank_domain_index(), 
           yk_solution::get_last_domain_index(), and the 
           "Detailed Description" for \ref yk_grid for more information on grid sizes.
           @note The parameter value is a double-precision floating-point value, but
           it will be converted to single-precision if
           yk_solution::get_element_bytes() returns 4.
           @note If storage has not been allocated via yk_solution::prepare_solution(),
           this will have no effect.
        */
        virtual void
        set_element(double val /**< [in] Point in grid will be set to this. */,
                    idx_t dim1_index=0 /**< [in] Index in dimension 1. */,
                    idx_t dim2_index=0 /**< [in] Index in dimension 2. */,
                    idx_t dim3_index=0 /**< [in] Index in dimension 3. */,
                    idx_t dim4_index=0 /**< [in] Index in dimension 4. */,
                    idx_t dim5_index=0 /**< [in] Index in dimension 5. */,
                    idx_t dim6_index=0 /**< [in] Index in dimension 6. */ ) =0;

        /// Set the value of one grid point.
        /**
           Same as version of set_element() with individual indices, but
           indices are provided in a list per the order returned by
           get_dim_names().
        */
        virtual void
        set_element(double val /**< [in] Point in grid will be set to this. */,
                    const std::vector<idx_t>& indices
                    /**< [in] List of indices, one for each grid dimension. */ ) =0;

        /// Initialize all grid points to the same value.
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
        set_all_elements_same(double val /**< [in] All points will be set to this. */ ) =0;

        /// Initialize grid points within specified subset of the grid to the same value.
        /**
           Sets all elements from 'first' to 'last' indices in each dimension to the
           specified value.
           Indices are relative to the *overall* problem domain.
           Both indices are inclusive (more like Fortran and Perl than Python and Lisp).
           Index values must fall within the rank domain or padding area.
           Notes in set_all_elements() documentation apply.
           @returns Number of elements set.
        */
        virtual idx_t
        set_elements_in_slice_same(double val /**< [in] All points in the slice will be set to this. */,
                                   const std::vector<idx_t>& first_indices
                                   /**< [in] List of beginning indices, one for each grid dimension. */,
                                   const std::vector<idx_t>& last_indices
                                   /**< [in] List of ending indices, one for each grid dimension. */ ) =0;

        /// Set grid points within specified subset of the grid.
        /**
           Reads elements from consecutive memory locations,
           starting at 'buffer_ptr',
           and writes them from 'first' to 'last' indices in each dimension in 
           row-major order.
           The buffer pointed to must contain either 4 or 8 byte FP values per point in the 
           subset, depending on the FP precision of the solution.
           The buffer pointed to must contain the number of FP values in the specified slice,
           where each FP value is the size of yk_solution::get_element_bytes().
           Since the writes proceed in row-major order, the last index is "unit-stride"
           in the buffer.
           Indices are relative to the *overall* problem domain.
           Both indices are inclusive (more like Fortran and Perl than Python and Lisp).
           Index values must fall within the rank domain or padding area.
           @returns Number of elements written.
        */
        virtual idx_t
        set_elements_in_slice(const void* buffer_ptr
                              /**< [out] Pointer to buffer where values will be read. */,
                              const std::vector<idx_t>& first_indices
                              /**< [in] List of beginning indices, one for each grid dimension. */,
                              const std::vector<idx_t>& last_indices
                              /**< [in] List of ending indices, one for each grid dimension. */ ) =0;
        
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

        /// Determine whether storage has been allocated.
        /**
           @returns 'true' if storage has been allocated,
           'false' otherwise.
        */
        virtual bool
        is_storage_allocated() const =0;
        
        /// **[Advanced]** Use existing data-storage from specified grid.
        /**
           This is an alternative to allocating data storage via 
           yk_solution::prepare_solution() or alloc_storage().
           In this case, data from a grid in this or another solution will be shared with
           this grid.
           In order to successfully share storage, the following conditions must hold:
           - The source grid must already have storage allocated.
           - The two grids must have the same dimensions in the same order.
           - The two grids must have the same domain sizes in all dimensions.
           - The two grids must have the same step-dimension allocation.
           - The halo size of this grid must be less than or equal to the padding
           size of the source grid in all dimensions. In other words, the halo
           of this grid must be able to "fit inside" the source padding.

           Any pre-existing storage will be released before allocation as via release_storage().
           The padding size of this grid will be set to that of the source grid.
           After calling share_storage(), changes in one grid via set_all_elements()
           or set_element() will be visible in the other grid.

           The halo sizes of the two grids do not need to be equal, but (as usual)
           they must remain less than or equal to the padding size after data storage is
           allocated.
           See allocation options and more information about grid sizes
           in the "Detailed Description" for \ref yk_grid.
        */
        virtual void
        share_storage(yk_grid_ptr source
                      /**< [in] Grid from which storage will be shared. */) =0;
    };


} // namespace yask.

#endif
