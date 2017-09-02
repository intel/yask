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

#include "yask.hpp"
using namespace std;

namespace yask {

    // Check whether dim is of allowed type.
    void Dims::checkDimType(const std::string& dim,
                            const std::string& fn_name,
                            bool step_ok,
                            bool domain_ok,
                            bool misc_ok) const {
        if (dim != _step_dim &&
            !_domain_dims.lookup(dim) &&
            !_misc_dims.lookup(dim)) {
            cerr << "Error in " << fn_name << "(): dimension '" <<
                dim << "' is not recognized in this solution.\n";
            exit_yask(1);
        }
        if (!step_ok && dim == _step_dim) {
            cerr << "Error in " << fn_name << "(): dimension '" <<
                dim << "' is the step dimension, which is not allowed.\n";
            exit_yask(1);
        }
        if (!domain_ok && _domain_dims.lookup(dim)) {
            cerr << "Error in " << fn_name << "(): dimension '" <<
                dim << "' is a domain dimension, which is not allowed.\n";
            exit_yask(1);
        }
        if (!misc_ok && _misc_dims.lookup(dim)) {
            cerr << "Error in " << fn_name << "(): dimension '" <<
                dim << "' is a misc dimension, which is not allowed.\n";
            exit_yask(1);
        }
    }
    
    // APIs.
    // See yask_kernel_api.hpp.
    yk_env_ptr yk_factory::new_env() const {
        auto ep = make_shared<KernelEnv>();
        assert(ep);
        ep->initEnv(0, 0);
        return ep;
    }
    yk_solution_ptr yk_factory::new_solution(yk_env_ptr env,
                                             const yk_solution_ptr source) const {
        auto ep = dynamic_pointer_cast<KernelEnv>(env);
        assert(ep);
        auto dp = YASK_STENCIL_CONTEXT::new_dims();
        assert(dp);
        auto op = make_shared<KernelSettings>(dp);
        assert(op);

        // Copy settings from source.
        if (source.get()) {
            auto ssp = dynamic_pointer_cast<StencilContext>(source);
            assert(ssp);
            auto sop = ssp->get_settings();
            assert(sop);
            *op = *sop;
        }
        
        // Create problem-specific object defined by stencil compiler.
        // TODO: allow more than one type of solution to be created.
        auto sp = make_shared<YASK_STENCIL_CONTEXT>(ep, op);
        assert(sp);

        // Set time-domain size because it's not really used in the API.
        auto& step_dim = op->_dims->_step_dim;
        op->_rank_sizes[step_dim] = 1;

        return sp;
    }
    yk_solution_ptr yk_factory::new_solution(yk_env_ptr env) const {
        return new_solution(env, nullptr);
    }

#define GET_SOLN_API(api_name, expr, step_ok, domain_ok, misc_ok)   \
    idx_t StencilContext::api_name(const string& dim) const {           \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        return expr;                                                    \
    }
    GET_SOLN_API(get_first_rank_domain_index, bb_begin[dim], false, true, false)
    GET_SOLN_API(get_last_rank_domain_index, bb_end[dim] - 1, false, true, false)
    GET_SOLN_API(get_overall_domain_size, overall_domain_sizes[dim], false, true, false)
    GET_SOLN_API(get_rank_domain_size, _opts->_rank_sizes[dim], false, true, false)
    GET_SOLN_API(get_min_pad_size, _opts->_min_pad_sizes[dim], false, true, false)
    GET_SOLN_API(get_block_size, _opts->_block_sizes[dim], false, true, false)
    GET_SOLN_API(get_num_ranks, _opts->_num_ranks[dim], false, true, false)
    GET_SOLN_API(get_rank_index, _opts->_rank_indices[dim], false, true, false)
#undef GET_SOLN_API

#define SET_SOLN_API(api_name, expr, step_ok, domain_ok, misc_ok)       \
    void StencilContext::api_name(const string& dim, idx_t n) {         \
        checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok);      \
        expr;                                                           \
    }
    SET_SOLN_API(set_rank_domain_size, _opts->_rank_sizes[dim] = n, false, true, false)
    SET_SOLN_API(set_min_pad_size, _opts->_min_pad_sizes[dim] = n, false, true, false)
    SET_SOLN_API(set_block_size, _opts->_block_sizes[dim] = n, false, true, false)
    SET_SOLN_API(set_num_ranks, _opts->_num_ranks[dim] = n, false, true, false)
#undef SET_SOLN_API
    
    string StencilContext::get_domain_dim_name(int n) const {
        auto* p = _dims->_domain_dims.lookup(n);
        if (!p) {
            cerr << "Error: get_domain_dim_name(): bad index '" << n << "'\n";
            exit_yask(1);
        }
        return _dims->_domain_dims.getDimName(n);
    }

    yk_grid_ptr StencilContext::new_grid(const std::string& name,
                                         const std::string& dim1,
                                         const std::string& dim2,
                                         const std::string& dim3,
                                         const std::string& dim4,
                                         const std::string& dim5,
                                         const std::string& dim6) {
        GridDimNames dims;
        if (dim1.length())
            dims.push_back(dim1);
        if (dim2.length())
            dims.push_back(dim2);
        if (dim3.length())
            dims.push_back(dim3);
        if (dim4.length())
            dims.push_back(dim4);
        if (dim5.length())
            dims.push_back(dim5);
        if (dim6.length())
            dims.push_back(dim6);
        return newGrid(name, dims);
    }
    
    void StencilContext::share_grid_storage(yk_solution_ptr source) {
        auto sp = dynamic_pointer_cast<StencilContext>(source);
        assert(sp);
        
        for (auto gp : gridPtrs) {
            auto gname = gp->get_name();
            auto si = sp->gridMap.find(gname);
            if (si != sp->gridMap.end()) {
                auto sgp = si->second;
                gp->share_storage(sgp);
            }
        }
    }

    ///// KernelEnv functions:

    // Init MPI, OMP.
    void KernelEnv::initEnv(int* argc, char*** argv)
    {
        // MPI init.
        my_rank = 0;
        num_ranks = 1;
        
#ifdef USE_MPI
        int is_init = false;
        MPI_Initialized(&is_init);

        if (!is_init) {
            int provided = 0;
            MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
            if (provided < MPI_THREAD_SERIALIZED) {
                cerr << "error: MPI_THREAD_SERIALIZED not provided.\n";
                exit_yask(1);
            }
            is_init = true;
        }
        comm = MPI_COMM_WORLD;
        MPI_Comm_rank(comm, &my_rank);
        MPI_Comm_size(comm, &num_ranks);
#else
        comm = 0;
#endif

        // There is no specific call to init OMP, but we make a gratuitous
        // OMP call to trigger any debug output.
        omp_get_num_procs();
    }

    ///// StencilContext functions:

    // Set ostr to given stream if provided.
    // If not provided, set to cout if my_rank == msg_rank
    // or a null stream otherwise.
    ostream& StencilContext::set_ostr(std::ostream* os) {
        if (os)
            _ostr = os;
        else if (_env->my_rank == _opts->msg_rank)
            _ostr = &cout;
        else
            _ostr = new ofstream;    // null stream (unopened ofstream). TODO: fix leak.
        assert(_ostr);
        return *_ostr;
    }
    
    
    ///// Top-level methods for evaluating reference and optimized stencils.

    // Eval stencil equation group(s) over grid(s) using scalar code.
    void StencilContext::calc_rank_ref()
    {
        auto& step_dim = _dims->_step_dim;
        idx_t begin_t = 0;
        idx_t end_t = _opts->_rank_sizes[step_dim];
        idx_t step_t = 1;

        // Begin, end, step, group-size tuples.
        IdxTuple begin(_dims->_stencil_dims);
        begin.setVals(bb_begin, false);
        begin[step_dim] = begin_t;
        IdxTuple end(_dims->_stencil_dims);
        end.setVals(bb_end, false);
        end[step_dim] = end_t;

        TRACE_MSG("calc_rank_ref: " << begin.makeDimValStr() << " ... " <<
                  end.subElements(1).makeDimValStr());

        // Indices needed for the 'general' loops.
        ScanIndices gen_idxs;
        gen_idxs.begin = begin;
        gen_idxs.end = end;

        // Number of iterations to get from begin_t to end_t-1,
        // stepping by step_t.
        const idx_t num_t = ((end_t - begin_t) + (step_t - 1)) / step_t;
        for (idx_t index_t = 0; index_t < num_t; index_t++)
        {
            // This value of index_t steps from start_t to stop_t-1.
            const idx_t start_t = begin_t + (index_t * step_t);
            const idx_t stop_t = min(start_t + step_t, end_t);

            // Set indices that will pass through generated code
            // because the step loop is coded here.
            gen_idxs.index[_step_posn] = index_t;
            gen_idxs.start[_step_posn] = start_t;
            gen_idxs.stop[_step_posn] = stop_t;
        
            // Loop thru eq-groups.
            for (auto* eg : eqGroups) {

                // Halo exchange(s) needed for this eq-group.
                exchange_halos(start_t, stop_t, *eg);

                // Define general calc function.  Since step is always 1, we
                // ignore gen_stop.  If point is in sub-domain for this eq
                // group, then evaluate the reference scalar code.
#define calc_gen(gen_idxs) \
                if (eg->is_in_valid_domain(gen_idxs.start)) \
                    eg->calc_scalar(gen_idxs.start)
                
                // Scan through n-D space.
#include "yask_gen_loops.hpp"
#undef calc_gen
                
                // Remember grids that have been written to by this eq-group.
                mark_grids_dirty(*eg);
                
            } // eq-groups.
        } // iterations.

        // Make sure all ranks are done.
        _env->global_barrier();
    }

    // Eval equation group(s) over grid(s) using optimized code.
    void StencilContext::run_solution(idx_t first_step_index,
                                      idx_t last_step_index)
    {
        auto& step_dim = _dims->_step_dim;
        idx_t begin_t = first_step_index;
        idx_t end_t = last_step_index + 1; // end is 1 past last.
        idx_t step_t = _opts->_region_sizes[step_dim];

        // Begin, end, and step tuples.
        IdxTuple begin(_dims->_stencil_dims);
        begin.setVals(bb_begin, false);
        begin[step_dim] = begin_t;
        IdxTuple end(_dims->_stencil_dims);
        end.setVals(bb_end, false);
        end[step_dim] = end_t;
        IdxTuple step(_dims->_stencil_dims);
        step.setVals(_opts->_region_sizes, false); // step by region sizes.

        TRACE_MSG("run_solution: " << begin.makeDimValStr() << " ... " <<
                  end.subElements(1).makeDimValStr());
        if (!bb_valid) {
            cerr << "Error: attempt to run solution without preparing it first.\n";
            exit_yask(1);
        }
        if (bb_size < 1) {
            TRACE_MSG("nothing to do in solution");
            return;
        }
        
#ifdef MODEL_CACHE
        ostream& os = get_ostr();
        if (context.my_rank != context.msg_rank)
            cache_model.disable();
        if (cache_model.isEnabled())
            os << "Modeling cache...\n";
#endif
        
        // Extend end points for overlapping regions due to wavefront angle.
        // For each subsequent time step in a region, the spatial location
        // of each block evaluation is shifted by the angle for each
        // eq-group. So, the total shift in a region is the angle * num
        // stencils * num timesteps. Thus, the number of overlapping regions
        // is ceil(total shift / region size). This assumes all eq-groups
        // are inter-dependent to find minimum extension. Actual required
        // extension may be less, but this will just result in some calls to
        // calc_region() that do nothing.
        //
        // Conceptually (showing 4 regions in t and x dims):
        // -----------------------------  t = rt
        //  \    |\     \     \ |   \     .
        //   \   | \     \     \|    \    .
        //    \  |  \     \     |     \   .
        //     \ |r0 \  r1 \ r2 |\ r3  \  .
        //      \|    \     \   | \     \ .
        // ------------------------------ t = 0
        //       |              |     |
        // x = begin_dx      end_dx end_dx
        //                   (orig) (after extension)
        //
        idx_t nshifts = (idx_t(eqGroups.size()) * step_t) - 1;
        for (auto& dim : _dims->_domain_dims.getDims()) {
            auto& dname = dim.getName();
            end[dname] += angles[dname] * nshifts;
        }
        TRACE_MSG("after wave-front adjustment: " <<
                  begin.makeDimValStr() << " ... " <<
                  end.subElements(1).makeDimValStr());

        // Indices needed for the 'rank' loops.
        ScanIndices rank_idxs;
        rank_idxs.begin = begin;
        rank_idxs.end = end;
        rank_idxs.step = step;

        // Set number of threads for a region.
        set_region_threads();

        // Number of iterations to get from begin_t to end_t-1,
        // stepping by step_t.
        const idx_t num_t = ((end_t - begin_t) + (step_t - 1)) / step_t;
        for (idx_t index_t = 0; index_t < num_t; index_t++)
        {
            // This value of index_t steps from start_t to stop_t-1.
            const idx_t start_t = begin_t + (index_t * step_t);
            const idx_t stop_t = min(start_t + step_t, end_t);

            // Set indices that will pass through generated code.
            rank_idxs.index[_step_posn] = index_t;
            rank_idxs.start[_step_posn] = start_t;
            rank_idxs.stop[_step_posn] = stop_t;
            
            // If doing only one time step in a region (default), loop
            // through equations here, and do only one equation group at a
            // time in calc_region(). This is similar to loop in
            // calc_rank_ref().
            if (step_t == 1) {

                for (auto* eg : eqGroups) {

                    // Halo exchange(s) needed for this eq-group.
                    exchange_halos(start_t, stop_t, *eg);

                    // Eval this eq-group in calc_region().
                    EqGroupSet eqGroup_set;
                    eqGroup_set.insert(eg);
                    EqGroupSet* eqGroup_ptr = &eqGroup_set;

                    // Include automatically-generated loop code that calls
                    // calc_region() for each region.
#include "yask_rank_loops.hpp"

                    // Remember grids that have been written to by this eq-group.
                    mark_grids_dirty(*eg);
                }
            }

            // If doing more than one time step in a region (temporal wave-front),
            // must loop through all eq-groups in calc_region().
            else {

                // TODO: enable halo exchange for wave-fronts.
                if (_env->num_ranks > 1) {
                    cerr << "Error: halo exchange with wave-fronts not yet supported.\n";
                    exit_yask(1);
                }
                
                // Eval all equation groups.
                EqGroupSet* eqGroup_ptr = NULL;
                
                // Include automatically-generated loop code that calls calc_region() for each region.
#include "yask_rank_loops.hpp"
            }

        }

        // Reset threads back to max.
        set_all_threads();

        // Make sure all ranks are done.
        _env->global_barrier();

#ifdef MODEL_CACHE
        // Print cache stats, then disable.
        // Thus, cache is only modeled for first call.
        if (cache_model.isEnabled()) {
            os << "Done modeling cache...\n";
            cache_model.dumpStats();
            cache_model.disable();
        }
#endif
    }

    // Apply solution for time-steps specified in _rank_sizes.
    void StencilContext::calc_rank_opt()
    {
        auto& step_dim = _dims->_step_dim;
        idx_t first_t = 0;
        idx_t last_t = _opts->_rank_sizes[step_dim] - 1;

        run_solution(first_t, last_t);
    }

    // Calculate results within a region.
    // Each region is typically computed in a separate OpenMP 'for' region.
    // In it, we loop over the time steps and the stencil
    // equations and evaluate the blocks in the region.
    void StencilContext::calc_region(EqGroupSet* eqGroup_set,
                                     const ScanIndices& rank_idxs) {

        int ndims = _dims->_stencil_dims.size();
        auto& step_dim = _dims->_step_dim;
        TRACE_MSG("calc_region: " << rank_idxs.start.makeValStr(ndims) <<
                  " ... " << rank_idxs.stop.addElements(-1).makeValStr(ndims));

        // Init region begin & end from rank start & stop indices.
        ScanIndices region_idxs;
        region_idxs.initFromOuter(rank_idxs);

        // Make a copy of the original start and stop indices because
        // we will be shifting these for temporal wavefronts.
        Indices rank_start(rank_idxs.start);
        Indices rank_stop(rank_idxs.stop);

        // Not yet supporting temporal blocking.
        if (_opts->_block_sizes[step_dim] != 1) {
            cerr << "Error: temporal blocking not yet supported." << endl;
            exit_yask(1);
        }
        
        // Steps within a region are based on block sizes.
        region_idxs.step = _opts->_block_sizes;

        // Groups in region loops are based on block-group sizes.
        region_idxs.group_size = _opts->_block_group_sizes;

        // Time loop.
        idx_t begin_t = region_idxs.begin[_step_posn];
        idx_t end_t = region_idxs.end[_step_posn];
        idx_t step_t = region_idxs.step[_step_posn];
        const idx_t num_t = ((end_t - begin_t) + (step_t - 1)) / step_t;
        for (idx_t index_t = 0; index_t < num_t; index_t++)
        {
            // This value of index_t steps from start_t to stop_t-1.
            const idx_t start_t = begin_t + (index_t * step_t);
            const idx_t stop_t = min(start_t + step_t, end_t);

            // TODO: remove this when temporal blocking is implemented.
            assert(stop_t == start_t + 1);

            // Set indices that will pass through generated code.
            region_idxs.index[_step_posn] = index_t;
            region_idxs.start[_step_posn] = start_t;
            region_idxs.stop[_step_posn] = stop_t;
            
            // equations to evaluate at this time step.
            for (auto* eg : eqGroups) {
                if (!eqGroup_set || eqGroup_set->count(eg)) {
                    TRACE_MSG("calc_region: eq-group '" << eg->get_name() << "' w/BB " <<
                              eg->bb_begin.makeDimValStr() << " ... " <<
                              eg->bb_end.subElements(1).makeDimValStr());

                    // For wavefront adjustments, see conceptual diagram in
                    // calc_rank_opt().  In this function, 1 of the 4
                    // parallelogram-shaped regions is being evaluated.  At
                    // each time-step, the parallelogram may be trimmed
                    // based on the BB.
                    
                    // Actual region boundaries must stay within BB for this eq group.
                    // Note that i-loop is over domain vars only (skipping over step var).
                    bool ok = true;
                    for (int i = _step_posn + 1; i < ndims; i++) {
                        auto& dname = _dims->_stencil_dims.getDimName(i);
                        assert(eg->bb_begin.lookup(dname));
                        region_idxs.begin[i] = max<idx_t>(rank_start[i], eg->bb_begin[dname]);
                        assert(eg->bb_end.lookup(dname));
                        region_idxs.end[i] = min<idx_t>(rank_stop[i], eg->bb_end[dname]);
                        if (region_idxs.end[i] <= region_idxs.begin[i])
                            ok = false;
                    }
                    TRACE_MSG("calc_region, after trimming: " <<
                              region_idxs.begin.makeValStr(ndims) <<
                              " ... " << region_idxs.end.addElements(-1).makeValStr(ndims));
                    
                    // Only need to loop through the spatial extent of the
                    // region if any of its blocks are at least partly
                    // inside the domain. For overlapping regions, they may
                    // start outside the domain but enter the domain as time
                    // progresses and their boundaries shift. So, we don't
                    // want to return if this condition isn't met.
                    if (ok) {

                        // Include automatically-generated loop code that
                        // calls calc_block() for each block in this region.
                        // Loops through x from begin_rx to end_rx-1;
                        // similar for y and z.  This code typically
                        // contains the outer OpenMP loop(s).
#include "yask_region_loops.hpp"

                    }
            
                    // Shift spatial region boundaries for next iteration to
                    // implement temporal wavefront.  We only shift
                    // backward, so region loops must increment. They may do
                    // so in any order.  TODO: shift only what is needed by
                    // this eq-group, not the global max.
                    // Note that i-loop is over domain vars only (skipping over step var).
                    for (int i = _step_posn + 1; i < ndims; i++) {
                        auto& dname = _dims->_stencil_dims.getDimName(i);
                        auto angle = angles[dname];
                        rank_start[i] -= angle;
                        rank_stop[i] -= angle;
                    }
                }            
            } // stencil equations.
        } // time.
    }

    // Calculate results within a block.
    void EqGroupBase::calc_block(const ScanIndices& region_idxs) {

        auto opts = _generic_context->get_settings();
        auto dims = _generic_context->get_dims();
        int ndims = dims->_stencil_dims.size();
        auto& step_dim = dims->_step_dim;
        TRACE_MSG3("calc_block: " << region_idxs.start.makeValStr(ndims) <<
                  " ... " << region_idxs.stop.addElements(-1).makeValStr(ndims));

        // Init block begin & end from region start & stop indices.
        ScanIndices block_idxs;
        block_idxs.initFromOuter(region_idxs);

        // Steps within a block are based on sub-block sizes.
        block_idxs.step = opts->_sub_block_sizes;

        // Groups in block loops are based on sub-block-group sizes.
        block_idxs.group_size = opts->_sub_block_group_sizes;

        // Set number of threads for a block.
        // This should be nested within a top-level OpenMP task.
        _generic_context->set_block_threads();

        // Include automatically-generated loop code that calls
        // calc_sub_block() for each sub-block in this block.  Loops through
        // x from begin_bx to end_bx-1; similar for y and z.  This
        // code typically contains the nested OpenMP loop(s).
#include "yask_block_loops.hpp"
    }
    
    // Calculate results for one sub-block.
    // Each block is typically computed in a separate OpenMP thread.
    void EqGroupBase::calc_sub_block(const ScanIndices& block_idxs) {

        auto* cp = _generic_context;
        auto opts = cp->get_settings();
        auto dims = cp->get_dims();
        int ndims = dims->_stencil_dims.size();
        auto& step_dim = dims->_step_dim;
        TRACE_MSG3("calc_sub_block: " << block_idxs.start.makeValStr(ndims) <<
                  " ... " << block_idxs.stop.addElements(-1).makeValStr(ndims));

        // Init sub-block begin & end from block start & stop indices.
        ScanIndices sub_block_idxs;
        sub_block_idxs.initFromOuter(block_idxs);
        
        // If not a 'simple' domain, use scalar code.  TODO: this
        // is very inefficient--need to vectorize as much as possible.
        if (!bb_simple) {
            bool full_bb = true;
            
            // If no holes, don't need to check each point in domain.
            if (bb_num_points == bb_size)
                TRACE_MSG3("...using scalar code without sub-domain checking.");
            
            else {
                TRACE_MSG3("...using scalar code with sub-domain checking.");
                full_bb = false;
            }

            // Use the 'general' loop. The OMP will be ignored because we're already in
            // a nested OMP region.
            ScanIndices gen_idxs(sub_block_idxs);

            // Define general calc function.  Since step is always 1, we
            // ignore gen_stop.  If point is in sub-domain for this eq
            // group, then evaluate the reference scalar code.
#define calc_gen(gen_idxs)                                      \
            if (full_bb || is_in_valid_domain(gen_idxs.start))  \
                calc_scalar(gen_idxs.start)
                
            // Scan through n-D space.
#include "yask_gen_loops.hpp"
#undef calc_gen
        }

        // Full rectangular polytope of aligned vectors: use optimized code.
        else {
            TRACE_MSG3("...using vector code without sub-domain checking.");

            // Make sure we're doing a multiple of clusters.
            for (int i = cp->_step_posn + 1; i < ndims; i++) {
                auto& dname = dims->_stencil_dims.getDimName(i);
                assert((sub_block_idxs.end[i] - sub_block_idxs.begin[i]) % 
                       dims->_cluster_pts[dname] == 0);
            }

            // Subtract rank offset and divide indices by vector lengths as
            // needed by read/writeVecNorm().  Use idiv_flr() instead of '/'
            // because begin/end vars may be negative (if in halo).
            for (int i = cp->_step_posn + 1; i < ndims; i++) {
                auto& dname = dims->_stencil_dims.getDimName(i);
                sub_block_idxs.begin[i] =
                    idiv_flr<idx_t>(sub_block_idxs.begin[i] -
                                    cp->rank_domain_offsets[dname],
                                    dims->_fold_pts[dname]);
                sub_block_idxs.end[i] =
                    idiv_flr<idx_t>(sub_block_idxs.end[i] -
                                    cp->rank_domain_offsets[dname],
                                    dims->_fold_pts[dname]);
            }

            // Evaluate sub-block of clusters.
            calc_sub_block_of_clusters(sub_block_idxs);
        }
        
        // Make sure streaming stores are visible for later loads.
        make_stores_visible();
    }

    // Add a new grid to the containers.
    void StencilContext::addGrid(YkGridPtr gp, bool is_output) {
        auto& gname = gp->get_name();
        if (gridMap.count(gname)) {
            cerr << "Error: grid '" << gname << "' already exists.\n";
            exit_yask(1);
        }

        // Add to list and map.
        gridPtrs.push_back(gp);
        gridMap[gname] = gp;

        // Add to output list and map if 'is_output'.
        if (is_output) {
            outputGridPtrs.push_back(gp);
            outputGridMap[gname] = gp;
        }
    }
        
    
    // Make a new grid.
    YkGridPtr StencilContext::newGrid(const std::string& name,
                                      const GridDimNames& dims,
                                      bool is_visible) {

        // Check for step dim.
        bool got_step = false;
        for (size_t i = 0; i < dims.size(); i++) {
            if (dims[i] == _dims->_step_dim) {
                if (i == 0)
                    got_step = true;
                else {
                    cerr << "Error: cannot create grid '" << name <<
                        "' with dimension '" << dims[i] << "' in position " <<
                        i << "; step dimension must be first dimension.\n";
                    exit_yask(1);
                }
            }
        }
        
        // NB: the behavior of this algorithm must follow that in the
        // YASK compiler to allow grids created via new_grid() to share
        // storage with those created via the compiler.
        // TODO: auto-gen this code.
#warning FIXME: make folded grids.
        YkGridPtr gp;
        switch (dims.size()) {
        case 0:
            gp = make_shared<YkElemGrid<Layout_0d, false>>(_dims, name, dims, &_ostr);
            break;
        case 1:
            if (got_step)
                gp = make_shared<YkElemGrid<Layout_1, true>>(_dims, name, dims, &_ostr);
            else
                gp = make_shared<YkElemGrid<Layout_1, false>>(_dims, name, dims, &_ostr);
            break;
        case 2:
            if (got_step)
                gp = make_shared<YkElemGrid<Layout_12, true>>(_dims, name, dims, &_ostr);
            else
                gp = make_shared<YkElemGrid<Layout_12, false>>(_dims, name, dims, &_ostr);
            break;
        case 3:
            if (got_step)
                gp = make_shared<YkElemGrid<Layout_123, true>>(_dims, name, dims, &_ostr);
            else
                gp = make_shared<YkElemGrid<Layout_123, false>>(_dims, name, dims, &_ostr);
            break;
        case 4:
            if (got_step)
                gp = make_shared<YkElemGrid<Layout_1234, true>>(_dims, name, dims, &_ostr);
            else
                gp = make_shared<YkElemGrid<Layout_1234, false>>(_dims, name, dims, &_ostr);
            break;
#if MAX_DIMS >= 5
        case 5:
            if (got_step)
                gp = make_shared<YkElemGrid<Layout_12345, true>>(_dims, name, dims, &_ostr);
            else
                gp = make_shared<YkElemGrid<Layout_12345, false>>(_dims, name, dims, &_ostr);
            break;
#endif
#if MAX_DIMS >= 6
        case 6:
            if (got_step)
                gp = make_shared<YkElemGrid<Layout_123456, true>>(_dims, name, dims, &_ostr);
            else
                gp = make_shared<YkElemGrid<Layout_123456, false>>(_dims, name, dims, &_ostr);
            break;
#endif
#if MAX_DIMS >= 7
        case 7:
            if (got_step)
                gp = make_shared<YkElemGrid<Layout_1234567, true>>(_dims, name, dims, &_ostr);
            else
                gp = make_shared<YkElemGrid<Layout_1234567, false>>(_dims, name, dims, &_ostr);
            break;
#endif
        default:
            cerr << "Error in new_grid: cannot create grid '" << name <<
                "' with " << dims.size() << " dimensions.\n";
            exit_yask(1);
        }

        // Add to context.
        if (is_visible)
            addGrid(gp, false);     // mark as non-output grid; TODO: determine if this is ok.

        // Set default sizes from settings and get offset, if set.
        if (is_visible)
            update_grids();

        return gp;
    }

    // Init MPI-related vars and other vars related to my rank's place in
    // the global problem: rank index, offset, etc.  Need to call this even
    // if not using MPI to properly init these vars.  Called from
    // prepare_solution(), so it doesn't normally need to be called from user code.
    void StencilContext::setupRank()
    {
        ostream& os = get_ostr();
        auto& step_dim = _dims->_step_dim;

        // Check ranks.
        idx_t req_ranks = _opts->_num_ranks.product();
        if (req_ranks != _env->num_ranks) {
            cerr << "error: " << req_ranks << " rank(s) requested (" <<
                _opts->_num_ranks.makeDimValStr(" * ") << "), but " <<
                _env->num_ranks << " rank(s) are active." << endl;
            exit_yask(1);
        }
        assertEqualityOverRanks(_opts->_rank_sizes[step_dim], _env->comm, "num steps");

        // Determine my coordinates if not provided already.
        // TODO: do this more intelligently based on proximity.
        if (_opts->find_loc)
            _opts->_rank_indices = _opts->_num_ranks.unlayout(_env->my_rank);

        // A table of rank-coordinates for everyone.
        auto num_ddims = _opts->_rank_indices.size(); // domain-dims only!
        idx_t coords[_env->num_ranks][num_ddims];

        // Init coords for this rank.
        for (int i = 0; i < num_ddims; i++)
            coords[_env->my_rank][i] = _opts->_rank_indices[i];

        // A table of rank-sizes for everyone.
        idx_t rsizes[_env->num_ranks][num_ddims];

        // Init sizes for this rank.
        for (int di = 0; di < num_ddims; di++) {
            auto& dname = _opts->_rank_indices.getDimName(di);
            rsizes[_env->my_rank][di] = _opts->_rank_sizes[dname];
        }

#ifdef USE_MPI
        // Exchange coord and size info between all ranks.
        for (int rn = 0; rn < _env->num_ranks; rn++) {
            MPI_Bcast(&coords[rn][0], num_ddims, MPI_INTEGER8,
                      rn, _env->comm);
            MPI_Bcast(&rsizes[rn][0], num_ddims, MPI_INTEGER8,
                      rn, _env->comm);
        }
        // Now, the tables are filled in for all ranks.
#endif

        // Init offsets and total sizes.
        rank_domain_offsets.setValsSame(0);
        overall_domain_sizes.setValsSame(0);

        // Loop over all ranks, including myself.
        int num_neighbors = 0, num_exchanges = 0;
        for (int rn = 0; rn < _env->num_ranks; rn++) {

            // Coord offset of rn from me: prev => negative, self => 0, next => positive.
            IdxTuple rcoords(_dims->_domain_dims);
            IdxTuple rdeltas(_dims->_domain_dims);
            for (int di = 0; di < num_ddims; di++) {
                rcoords[di] = coords[rn][di];
                rdeltas[di] = coords[rn][di] - _opts->_rank_indices[di];
            }
        
            for (int di = 0; di < num_ddims; di++) {
                auto& dname = _opts->_rank_indices.getDimName(di);

                // Does this rank "intersect" mine?
                // Rank rn intersects when deltas in other dims are zero.
                bool intersect = true;
                for (int dj = 0; dj < num_ddims; dj++) {
                    if (di != dj && rdeltas[dj] != 0) {
                        intersect = false;
                        break;
                    }
                }
                if (intersect) {
                    
                    // Accumulate total problem size in each dim for ranks that
                    // intersect with this rank, including myself.
                    overall_domain_sizes[dname] += rsizes[rn][di];

                    // Adjust my offset in the global problem by adding all domain
                    // sizes from prev ranks only.
                    if (rdeltas[di] < 0)
                        rank_domain_offsets[dname] += rsizes[rn][di];
                }
            }

            // Manhattan distance from rn (sum of abs deltas in all dims).
            // Max distance in any dim.
            int mandist = 0;
            int maxdist = 0;
            for (int di = 0; di < num_ddims; di++) {
                mandist += abs(rdeltas[di]);
                maxdist = max(maxdist, abs(int(rdeltas[di])));
            }
            
            // Myself.
            if (rn == _env->my_rank) {
                if (mandist != 0) {
                    cerr << "Internal error: distance to own rank == " << mandist << endl;
                    exit_yask(1);
                }
                continue; // nothing else to do for self.
            }

            // Someone else.
            else {
                if (mandist == 0) {
                    cerr << "Error: ranks " << _env->my_rank <<
                        " and " << rn << " at same coordinates." << endl;
                    exit_yask(1);
                }
            }
            
            // Rank rn is my immediate neighbor if its distance <= 1 in
            // every dim.  Assume we do not need to exchange halos except
            // with immediate neighbor. TODO: validate domain size is larger
            // than halo.
            if (maxdist > 1)
                continue;

            // At this point, rdeltas contains only -1..+1 for each domain dim.
            // Add one to -1..+1 to get 0..2 range for my_neighbors indices.
            IdxTuple roffsets = rdeltas.addElements(1);

            // Convert these nD offsets into a 1D index.
            auto rn_ofs = _mpiInfo->neighbor_offsets.layout(roffsets);
            TRACE_MSG("neighbor_offsets = " << _mpiInfo->neighbor_offsets.makeDimValStr() <<
                      " & roffsets of rank " << rn << " = " << roffsets.makeDimValStr() <<
                      " => " << rn_ofs);
            assert(rn_ofs < _mpiInfo->neighborhood_size);

            // Save rank of this neighbor into the MPI info object.
            _mpiInfo->my_neighbors.at(rn_ofs) = rn;
            num_neighbors++;
            os << "Neighbor #" << num_neighbors << " is rank " << rn <<
                " at absolute rank indices " << rcoords.makeDimValStr() <<
                " (" << rdeltas.makeDimValOffsetStr() << " relative to rank " <<
                _env->my_rank << ")\n";
                    
            // Check against max dist needed.  TODO: determine max dist
            // automatically from stencil equations; may not be same for all
            // grids.
#ifndef MAX_EXCH_DIST
#define MAX_EXCH_DIST 3
#endif

            // Is buffer needed?
            // TODO: calculate and use exch dist for each grid.
            if (mandist > MAX_EXCH_DIST) {
                os << "- no halo exchanges needed with rank " << rn << '.' << endl;
                continue;
            }

            // Determine size of MPI buffers between rn and my rank.
            // Create send and receive for each updated grid.
            for (auto gp : gridPtrs) {
                auto& gname = gp->get_name();

                // Size of MPI buffer in each dim of this grid:
                // For domain dims, if dist to neighbor is
                // zero in given dim (i.e., is perpendicular to this
                // rank), use full rank size; otherwise, use halo size for this grid.
                // For step dims, use size 1. Otherwise, use allocated grid size.
                // These sizes must match those calculated in exchange_halos().
                IdxTuple bufsizes;
                bool found_delta = false;
                for (auto& dname : gp->get_dim_names()) {
                    idx_t dsize = 1;
                    
                    // domain dim?
                    if (rdeltas.lookup(dname)) {
                        dsize = _opts->_rank_sizes[dname];
                        if (rdeltas[dname] != 0) {
                            found_delta = true;
                            dsize = gp->get_halo_size(dname);
                        }
                    }

                    // step dim?
                    else if (dname == _dims->_step_dim)
                        dsize = 1; // TODO: change when supporting wavefronts.

                    // misc?
                    else
                        dsize = gp->get_alloc_size(dname);
                    
                    bufsizes.addDimBack(dname, dsize);
                }

                if (!found_delta || bufsizes.size() == 0 || bufsizes.product() == 0) {
                    os << "- no halo exchanges needed for grid '" << gname <<
                        "' with rank " << rn << '.' << endl;
                }
                else {

                    // Make a buffer in both directions (send & receive).
                    size_t num_bytes = 0;
                    for (int bd = 0; bd < MPIBufs::nBufDirs; bd++) {
                        ostringstream oss;
                        oss << gname;
                        if (bd == MPIBufs::bufSend)
                            oss << "_send_halo_from_" << _env->my_rank << "_to_" << rn;
                        else
                            oss << "_get_halo_to_" << _env->my_rank << "_from_" << rn;
                        string bufname = oss.str();

                        // Buffers for this grid.
                        auto gbp = mpiBufs.emplace(gname, _mpiInfo);
                        auto& gbi = gbp.first; // iterator from pair returned by emplace().
                        auto& gbv = gbi->second; // value from iterator.
                        auto bp = gbv.makeBuf(MPIBufs::BufDir(bd),
                                              roffsets,
                                              bufsizes,
                                              bufname,
                                              *this);
                        num_bytes = bp->get_num_storage_bytes();
                        num_exchanges++;
                    }

                    os << "- 2 halo-exchange buffers of shape " << bufsizes.makeDimValStr(" * ") <<
                        " and size " << makeByteStr(num_bytes) <<
                        " enabled for grid '" << gname << "' with rank " << rn << '.' << endl;
                }
            }
        }
        os << "Number of halo-exchange buffers enabled on this rank: " << num_exchanges << endl;

        // Set offsets in grids.
        update_grids();
    }

    // Allocate memory for grids that do not
    // already have storage.
    // TODO: allow different types of memory for different grids, MPI bufs, etc.
    void StencilContext::allocData()
    {
        ostream& os = get_ostr();

        // Base ptrs for all default-alloc'd data.
        // These pointers will be shared by the ones in the grid
        // objects, which will take over ownership when these go
        // out of scope.
        shared_ptr<char> _grid_data_buf;
        shared_ptr<char> _mpi_data_buf;

        // TODO: release old MPI buffers.
        
        // Pass 0: count required size, allocate memory.
        // Pass 1: distribute already-allocated memory.
        for (int pass = 0; pass < 2; pass++) {
            TRACE_MSG("allocData pass " << pass << "; " << gridPtrs.size() <<
                      " grid(s) and " << mpiBufs.size() << " MPI buffer pair(s)");
        
            // Determine how many bytes are needed and actually alloc'd.
            size_t gbytes = 0, agbytes = 0;
            size_t bbytes = 0, abbytes = 0;
            int ngrids = 0, nbufs = 0;
        
            // Grids.
            for (auto gp : gridPtrs) {
                if (!gp)
                    continue;
                auto& gname = gp->get_name();

                // Grid data.
                // Don't alloc if already done.
                if (!gp->is_storage_allocated()) {

                    // Set storage if buffer has been allocated.
                    if (pass == 1) {
                        gp->set_storage(_grid_data_buf, agbytes);
                        gp->print_info(os);
                        os << endl;
                    }

                    // Determine size used (also offset to next location).
                    gbytes += gp->get_num_storage_bytes();
                    agbytes += ROUND_UP(gp->get_num_storage_bytes() + _data_buf_pad,
                                        CACHELINE_BYTES);
                    ngrids++;
                    TRACE_MSG(" grid '" << gname << "' needs " <<
                              gp->get_num_storage_bytes() << " bytes");
                }
                
                // MPI bufs for this grid.
                if (mpiBufs.count(gname)) {

                    // Visit buffers for each neighbor for this grid.  Don't
                    // check whether grid has allocated storage, because we
                    // want to replace old MPI buffers in case the size has
                    // changed.
                    mpiBufs.at(gname).visitNeighbors
                        ([&](const IdxTuple& offsets,
                             int rank, int idx,
                             YkGridPtr sendBuf,
                             YkGridPtr recvBuf)
                         {
                             // Send.
                             if (sendBuf) {
                                 if (pass == 1)
                                     sendBuf->set_storage(_mpi_data_buf, abbytes);
                                 auto sbytes = sendBuf->get_num_storage_bytes();
                                 bbytes += sbytes;
                                 abbytes += ROUND_UP(sbytes + _data_buf_pad,
                                                     CACHELINE_BYTES);
                                 nbufs++;
                                 TRACE_MSG("  send buf '" << sendBuf->get_name() << "' needs " <<
                                           sbytes << " bytes");
                             }

                             // Recv.
                             if (recvBuf) {
                                 if (pass == 1)
                                     recvBuf->set_storage(_mpi_data_buf, abbytes);
                                 auto rbytes = recvBuf->get_num_storage_bytes();
                                 bbytes += rbytes;
                                 abbytes += ROUND_UP(rbytes + _data_buf_pad,
                                                     CACHELINE_BYTES);
                                 nbufs++;
                                 TRACE_MSG("  recv buf '" << recvBuf->get_name() << "' needs " <<
                                           rbytes<< " bytes");
                             }
                         } );
                }
            }

            // Don't need pad after last one.
            if (agbytes >= _data_buf_pad)
                agbytes -= _data_buf_pad;
            if (abbytes >= _data_buf_pad)
                abbytes -= _data_buf_pad;

            // Allocate data.
            if (pass == 0) {
                os << "Allocating " << makeByteStr(agbytes) <<
                    " for " << ngrids << " grid(s)...\n" << flush;
                _grid_data_buf = shared_ptr<char>(alignedAlloc(agbytes), AlignedDeleter());

#ifdef USE_MPI
                os << "Allocating " << makeByteStr(abbytes) <<
                    " for " << nbufs << " MPI buffer(s)...\n" << flush;
                _mpi_data_buf = shared_ptr<char>(alignedAlloc(abbytes), AlignedDeleter());
#endif
            }
        }
    }

    // Set grid sizes and offsets based on settings.
    // Set max halos across grids.
    // This should be called anytime a setting or rank offset is changed.
    void StencilContext::update_grids()
    {
        assert(_opts);

        // Reset halos.
        max_halos = _dims->_domain_dims;
        
        // Loop through each grid.
        for (auto gp : gridPtrs) {

            // Loop through each domain dim.
            for (auto& dim : _dims->_domain_dims.getDims()) {
                auto& dname = dim.getName();

                if (gp->is_dim_used(dname)) {

                    // Rank domains.
                    gp->_set_domain_size(dname, _opts->_rank_sizes[dname]);
                    
                    // Pads.
                    // Set via both 'extra' and 'min'; larger result will be used.
                    gp->set_extra_pad_size(dname, _opts->_extra_pad_sizes[dname]);
                    gp->set_min_pad_size(dname, _opts->_min_pad_sizes[dname]);
                    
                    // Offsets.
                    gp->_set_offset(dname, rank_domain_offsets[dname]);

                    // Update max halo across grids, used for wavefront angles.
                    auto hsz = gp->get_halo_size(dname);
                    max_halos[dname] = max(max_halos[dname], hsz);
                }
            }
        }
    }
    
    // Allocate grids and MPI bufs.
    // Initialize some data structures.
    void StencilContext::prepare_solution()
    {
        // Don't continue until all ranks are this far.
        _env->global_barrier();

        ostream& os = get_ostr();
#ifdef DEBUG
        os << "*** WARNING: YASK compiled with DEBUG; ignore performance results.\n";
#endif
#if defined(NO_INTRINSICS) && (VLEN > 1)
        os << "*** WARNING: YASK compiled with NO_INTRINSICS; ignore performance results.\n";
#endif
#ifdef MODEL_CACHE
        os << "*** WARNING: YASK compiled with MODEL_CACHE; ignore performance results.\n";
#endif
#ifdef TRACE_MEM
        os << "*** WARNING: YASK compiled with TRACE_MEM; ignore performance results.\n";
#endif
#ifdef TRACE_INTRINSICS
        os << "*** WARNING: YASK compiled with TRACE_INTRINSICS; ignore performance results.\n";
#endif
        
        // Adjust all settings before setting MPI buffers or sizing grids.
        // Prints out final settings.
        _opts->adjustSettings(os);

        // Size grids based on finalized settings.
        update_grids();
        
        // Report ranks.
        os << endl;
        os << "Num ranks: " << _env->get_num_ranks() << endl;
        os << "This rank index: " << _env->get_rank_index() << endl;

        // report threads.
        os << "Num OpenMP procs: " << omp_get_num_procs() << endl;
        set_all_threads();
        os << "Num OpenMP threads: " << omp_get_max_threads() << endl;
        set_region_threads(); // Temporary; just for reporting.
        os << "  Num threads per region: " << omp_get_max_threads() << endl;
        set_block_threads(); // Temporary; just for reporting.
        os << "  Num threads per block: " << omp_get_max_threads() << endl;
        set_all_threads(); // Back to normal.

        // TODO: enable multi-rank wave-front tiling.
        auto& step_dim = _dims->_step_dim;
        if (_opts->_region_sizes[step_dim] > 1 && _env->num_ranks > 1) {
            cerr << "MPI communication is not currently enabled with wave-front tiling." << endl;
            exit_yask(1);
        }

        os << endl;
        os << "Num grids: " << gridPtrs.size() << endl;
        os << "Num grids to be updated: " << outputGridPtrs.size() << endl;
        os << "Num stencil equation-groups: " << eqGroups.size() << endl;
        
        // Set up data based on MPI rank, including grid positions.
        setupRank();

        // Determine bounding-boxes for all eq-groups.
        find_bounding_boxes();

        // Alloc grids and MPI bufs.
        allocData();

        // Report some stats.
        idx_t dt = _opts->_rank_sizes[step_dim];
        os << "\nProblem sizes in points (from smallest to largest):\n"
            " vector-size:          " << _dims->_fold_pts.makeDimValStr(" * ") << endl <<
            " cluster-size:         " << _dims->_cluster_pts.makeDimValStr(" * ") << endl <<
            " sub-block-size:       " << _opts->_sub_block_sizes.makeDimValStr(" * ") << endl <<
            " sub-block-group-size: " << _opts->_sub_block_group_sizes.makeDimValStr(" * ") << endl <<
            " block-size:           " << _opts->_block_sizes.makeDimValStr(" * ") << endl <<
            " block-group-size:     " << _opts->_block_group_sizes.makeDimValStr(" * ") << endl <<
            " region-size:          " << _opts->_region_sizes.makeDimValStr(" * ") << endl <<
            " rank-domain-size:     " << _opts->_rank_sizes.makeDimValStr(" * ") << endl <<
            " overall-problem-size: " << overall_domain_sizes.makeDimValStr(" * ") << endl <<
            endl <<
            "Other settings:\n"
#ifdef USE_MPI
            " num-ranks:            " << _opts->_num_ranks.makeDimValStr(" * ") << endl <<
            " rank-indices:         " << _opts->_rank_indices.makeDimValStr() << endl <<
            " rank-domain-offsets:  " << rank_domain_offsets.makeDimValOffsetStr() << endl <<
#endif
            " stencil-name:         " << get_name() << endl << 
            " vector-len:           " << VLEN << endl <<
            " extra-padding:        " << _opts->_extra_pad_sizes.makeDimValStr() << endl <<
            " minimum-padding:      " << _opts->_min_pad_sizes.makeDimValStr() << endl <<
            " wave-front-angles:    " << angles.makeDimValStr() << endl <<
            " max-halos:            " << max_halos.makeDimValStr() << endl <<
            " manual-L1-prefetch-distance: " << PFDL1 << endl <<
            " manual-L2-prefetch-distance: " << PFDL2 << endl <<
            endl;
        
        // sums across eqs for this rank.
        rank_numpts_1t = 0;
        rank_reads_1t = 0;
        rank_numFpOps_1t = 0;
        for (auto eg : eqGroups) {
            idx_t updates1 = eg->get_scalar_points_written();
            idx_t updates_domain = updates1 * eg->bb_num_points;
            rank_numpts_1t += updates_domain;
            idx_t reads1 = eg->get_scalar_points_read();
            idx_t reads_domain = reads1 * eg->bb_num_points;
            rank_reads_1t += reads_domain;
            idx_t fpops1 = eg->get_scalar_fp_ops();
            idx_t fpops_domain = fpops1 * eg->bb_num_points;
            rank_numFpOps_1t += fpops_domain;
            os << "Stats for equation-group '" << eg->get_name() << "':\n" <<
                " sub-domain:                 " << eg->bb_begin.makeDimValStr() <<
                " ... " << eg->bb_end.subElements(1).makeDimValStr() << endl <<
                " sub-domain size:            " << eg->bb_len.makeDimValStr(" * ") << endl <<
                " valid points in sub domain: " << makeNumStr(eg->bb_num_points) << endl <<
                " grid-updates per point:     " << updates1 << endl <<
                " grid-updates in sub-domain: " << makeNumStr(updates_domain) << endl <<
                " grid-reads per point:       " << reads1 << endl <<
                " grid-reads in sub-domain:   " << makeNumStr(reads_domain) << endl <<
                " est FP-ops per point:       " << fpops1 << endl <<
                " est FP-ops in sub-domain:   " << makeNumStr(fpops_domain) << endl;
        }

        // Report total allocation.
        rank_nbytes = get_num_bytes();
        os << "Total allocation in this rank: " <<
            makeByteStr(rank_nbytes) << "\n";
        tot_nbytes = sumOverRanks(rank_nbytes, _env->comm);
        os << "Total overall allocation in " << _env->num_ranks << " rank(s): " <<
            makeByteStr(tot_nbytes) << "\n";
    
        // Various metrics for amount of work.
        rank_numpts_dt = rank_numpts_1t * dt;
        tot_numpts_1t = sumOverRanks(rank_numpts_1t, _env->comm);
        tot_numpts_dt = tot_numpts_1t * dt;

        rank_reads_dt = rank_reads_1t * dt;
        tot_reads_1t = sumOverRanks(rank_reads_1t, _env->comm);
        tot_reads_dt = tot_reads_1t * dt;

        rank_numFpOps_dt = rank_numFpOps_1t * dt;
        tot_numFpOps_1t = sumOverRanks(rank_numFpOps_1t, _env->comm);
        tot_numFpOps_dt = tot_numFpOps_1t * dt;

        rank_domain_dt = _opts->_rank_sizes.product();
        rank_domain_1t = rank_domain_dt / dt;
        tot_domain_1t = sumOverRanks(rank_domain_1t, _env->comm);
        tot_domain_dt = tot_domain_1t * dt;
    
        // Print some more stats.
        os << endl <<
            "Amount-of-work stats:\n" <<
            " domain-size in this rank, for one time-step: " <<
            makeNumStr(rank_domain_1t) << endl <<
            " overall-problem-size in all ranks, for one time-step: " <<
            makeNumStr(tot_domain_1t) << endl <<
            " domain-size in this rank, for all time-steps: " <<
            makeNumStr(rank_domain_dt) << endl <<
            " overall-problem-size in all ranks, for all time-steps: " <<
            makeNumStr(tot_domain_dt) << endl <<
            endl <<
            " grid-point-updates in this rank, for one time-step: " <<
            makeNumStr(rank_numpts_1t) << endl <<
            " grid-point-updates in all ranks, for one time-step: " <<
            makeNumStr(tot_numpts_1t) << endl <<
            " grid-point-updates in this rank, for all time-steps: " <<
            makeNumStr(rank_numpts_dt) << endl <<
            " grid-point-updates in all ranks, for all time-steps: " <<
            makeNumStr(tot_numpts_dt) << endl <<
            endl <<
            " grid-point-reads in this rank, for one time-step: " <<
            makeNumStr(rank_reads_1t) << endl <<
            " grid-point-reads in all ranks, for one time-step: " <<
            makeNumStr(tot_reads_1t) << endl <<
            " grid-point-reads in this rank, for all time-steps: " <<
            makeNumStr(rank_reads_dt) << endl <<
            " grid-point-reads in all ranks, for all time-steps: " <<
            makeNumStr(tot_reads_dt) << endl <<
            endl <<
            " est-FP-ops in this rank, for one time-step: " <<
            makeNumStr(rank_numFpOps_1t) << endl <<
            " est-FP-ops in all ranks, for one time-step: " <<
            makeNumStr(tot_numFpOps_1t) << endl <<
            " est-FP-ops in this rank, for all time-steps: " <<
            makeNumStr(rank_numFpOps_dt) << endl <<
            " est-FP-ops in all ranks, for all time-steps: " <<
            makeNumStr(tot_numFpOps_dt) << endl <<
            endl << 
            "Notes:\n"
            " Domain-sizes and overall-problem-sizes are based on rank-domain sizes (dw * dx * dy * dz)\n"
            "  and number of ranks (nrw * nrx * nry * nrz) regardless of number of grids or sub-domains.\n"
            " Grid-point-updates are based on sum of grid-updates in sub-domain across equation-group(s).\n"
            " Grid-point-reads are based on sum of grid-reads in sub-domain across equation-group(s).\n"
            " Est-FP-ops are based on sum of est-FP-ops in sub-domain across equation-group(s).\n"
            "\n";
    }

    // Init all grids & params by calling initFn.
    void StencilContext::initValues(function<void (YkGridPtr gp, 
                                                   real_t seed)> realInitFn) {
        ostream& os = get_ostr();
        real_t v = 0.1;
        os << "Initializing grids..." << endl;
        for (auto gp : gridPtrs) {
            realInitFn(gp, v);
            v += 0.01;
        }
    }

    // Compare grids in contexts.
    // Return number of mis-compares.
    idx_t StencilContext::compareData(const StencilContext& ref) const {
        ostream& os = get_ostr();

        os << "Comparing grid(s) in '" << name << "' to '" << ref.name << "'..." << endl;
        if (gridPtrs.size() != ref.gridPtrs.size()) {
            cerr << "** number of grids not equal." << endl;
            return 1;
        }
        idx_t errs = 0;
        for (size_t gi = 0; gi < gridPtrs.size(); gi++) {
            os << "Grid '" << ref.gridPtrs[gi]->get_name() << "'..." << endl;
            errs += gridPtrs[gi]->compare(ref.gridPtrs[gi].get());
        }

        return errs;
    }

    // Compute convenience values for a bounding-box.
    void BoundingBox::update_bb(ostream& os,
                                const string& name,
                                StencilContext& context,
                                bool force_full) {

        auto dims = context.get_dims();
        auto& domain_dims = dims->_domain_dims;
        bb_len = bb_end.subElements(bb_begin);
        bb_size = bb_len.product();
        if (force_full)
            bb_num_points = bb_size;
        bb_simple = true;       // assume ok.

        // Solid rectangle?
        if (bb_num_points != bb_size) {
            os << "Warning: '" << name << "' domain has only " <<
                makeNumStr(bb_num_points) <<
                " valid point(s) inside its bounding-box of " <<
                makeNumStr(bb_size) <<
                " point(s); slower scalar calculations will be used.\n";
            bb_simple = false;
        }

        else {

            // Lengths are cluster-length multiples?
            bool is_cluster_mult = true;
            for (auto& dim : domain_dims.getDims()) {
                auto& dname = dim.getName();
                if (bb_len[dname] % dims->_cluster_pts[dname]) {
                    is_cluster_mult = false;
                    break;
                }
            }
            if (!is_cluster_mult) {
                os << "Warning: '" << name << "' domain"
                    " has one or more sizes that are not vector-cluster multiples;"
                    " slower scalar calculations will be used.\n";
                bb_simple = false;
            }

            else {

                // Does everything start on a vector-length boundary?
                bool is_aligned = true;
                for (auto& dim : domain_dims.getDims()) {
                    auto& dname = dim.getName();
                    if ((bb_begin[dname] - context.rank_domain_offsets[dname]) %
                        dims->_fold_pts[dname]) {
                        is_aligned = false;
                        break;
                    }
                }
                if (!is_aligned) {
                    os << "Warning: '" << name << "' domain"
                        " has one or more starting edges not on vector boundaries;"
                        " slower scalar calculations will be used.\n";
                    bb_simple = false;
                }
            }
        }

#warning FIXME: re-enable vectorization.
        if (bb_simple) {
            os << "Warning: vectorization disabled for this alpha version;"
                " slower scalar calculations will be used.\n";
            bb_simple = false;
        }
        
        // All done.
        bb_valid = true;
    }
    
    // Set the bounding-box for each eq-group and whole domain.
    // Also sets wave-front angles.
    void StencilContext::find_bounding_boxes()
    {
        ostream& os = get_ostr();

        // Find BB for each eq group.
        for (auto eg : eqGroups)
            eg->find_bounding_box();

        // Overall BB based only on rank offsets and rank domain sizes.
        bb_begin = rank_domain_offsets;
        bb_end = rank_domain_offsets.addElements(_opts->_rank_sizes, false);
        update_bb(os, "rank", *this, true);

        // Determine the max spatial skewing angles for temporal wavefronts
        // based on the max halos.  This assumes the smallest granularity of
        // calculation is CPTS_* in each dim.  We only need non-zero angles
        // if the region size is less than the rank size, i.e., if the
        // region covers the whole rank in a given dimension, no wave-front
        // is needed in thar dim.
        // TODO: make rounding-up an option.
        for (auto& dim : _dims->_domain_dims.getDims()) {
            auto& dname = dim.getName();
            angles[dname] = (_opts->_region_sizes[dname] < bb_len[dname]) ?
                ROUND_UP(max_halos[dname], _dims->_cluster_pts[dname]) : 0;
        }
    }

    // Set the bounding-box vars for this eq group in this rank.
    void EqGroupBase::find_bounding_box() {
        StencilContext& context = *_generic_context;
        auto optsp = context.get_settings();
        assert(optsp);
        auto opts = *optsp.get();
        ostream& os = context.get_ostr();
        auto& domain_dims = context.get_dims()->_domain_dims;
        auto& step_dim = context.get_dims()->_step_dim;

        // Init bb vars to ensure correct dims.
        bb_begin = domain_dims;
        bb_end = domain_dims;
        
        // Init min vars w/max val and vice-versa.
        Indices min_pts(idx_max);
        Indices max_pts(idx_min);
        idx_t npts = 0;

        // Tuple of rank sizes w/o step dim.
        auto rank_sizes = context.get_settings()->_rank_sizes.removeDim(step_dim);
        idx_t rsz = rank_sizes.product();

        // Loop through rank domain.
        // Assume result is valid for all step-dim values.
        // Find the min and max valid points in this space.
        // Count all the valid points.
#pragma omp parallel for                        \
    reduction(min_idxs:min_pts)                 \
    reduction(max_idxs:max_pts)                 \
    reduction(+:npts)
        for (idx_t i = 0; i < rsz; i++) {

            // Get n-D indices.
            // TODO: make this more efficient.
            auto pt = rank_sizes.unlayout(i);

            // Translate to overall-problem indices.
            pt = pt.addElements(context.rank_domain_offsets);
                
            // Add step dim for domain test.
            auto fpt = pt;
            fpt.addDimFront(step_dim, 0);

            // Update only if point is in domain for this eq group.
            Indices fidxs(fpt);
            if (is_in_valid_domain(fidxs)) {

                Indices idxs(pt);   // w/o step dim.
                min_pts = min_pts.minElements(idxs);
                max_pts = max_pts.maxElements(idxs);
                npts++;
            }
        }

        // Set begin vars to min indices and end vars to one beyond max indices.
        if (npts) {
            min_pts.setTupleVals(bb_begin);
            max_pts.setTupleVals(bb_end);
            bb_end = bb_end.addElements(1); // end = last + 1.
        } else {
            bb_begin.setValsSame(0);
            bb_end.setValsSame(0);
        }
        bb_num_points = npts;

        // Finalize BB.
        update_bb(os, get_name(), context);
    }
    
    // Exchange halo data needed by eq-group 'eg' at the given time.
    // Data is needed for input grids that have not already been updated.
    // [BIG] TODO: overlap halo exchange with computation.
    void StencilContext::exchange_halos(idx_t start_dt, idx_t stop_dt, EqGroupBase& eg)
    {
        auto opts = get_settings();
        TRACE_MSG("exchange_halos: " << start_dt << " ... " << (stop_dt-1) <<
                  " for eq-group '" << eg.get_name() << "'");

#ifdef USE_MPI
        double start_time = getTimeInSecs();

        // 1D array to store send request handles.
        // We use a 1D array so we can call MPI_Waitall().
        MPI_Request send_reqs[eg.inputGridPtrs.size() * _mpiInfo->neighborhood_size];
        int num_send_reqs = 0;

        // 2D array for receive request handles.
        // We use a 2D array to simplify individual indexing.
        MPI_Request recv_reqs[eg.inputGridPtrs.size()][_mpiInfo->neighborhood_size];

        // Sequence of things to do for each grid's neighbors
        // (isend includes packing).
        enum halo_steps { halo_irecv, halo_isend, halo_unpack, halo_nsteps };
        for (int hi = 0; hi < halo_nsteps; hi++) {

            if (hi == halo_irecv)
                TRACE_MSG("exchange_halos: requesting data...");
            else if (hi == halo_isend)
                TRACE_MSG("exchange_halos: packing and sending data...");
            else if (hi == halo_unpack)
                TRACE_MSG("exchange_halos: unpacking data...");
            
            // Loop thru all input grids in this group.
            for (size_t gi = 0; gi < eg.inputGridPtrs.size(); gi++) {
                auto gp = eg.inputGridPtrs[gi];
                MPI_Request* grid_recv_reqs = recv_reqs[gi];

                // Only need to swap grids whose halos are not up-to-date.
                if (gp->is_updated())
                    continue;

                // Only need to swap grids that have MPI buffers.
                auto& gname = gp->get_name();
                if (mpiBufs.count(gname) == 0)
                    continue;
                TRACE_MSG(" for grid '" << gname << "'...");

                // Lookup first & last domain indices and halo sizes.
                IdxTuple halo_sizes, first_idx, last_idx;
                for (auto& dim : _dims->_domain_dims.getDims()) {
                    auto& dname = dim.getName();
                    if (gp->is_dim_used(dname)) {
                        halo_sizes.addDimBack(dname, gp->get_halo_size(dname));
                        first_idx.addDimBack(dname, gp->get_first_rank_domain_index(dname));
                        last_idx.addDimBack(dname, gp->get_last_rank_domain_index(dname));
                    }
                }

                // Visit all this rank's neighbors.
                mpiBufs.at(gname).visitNeighbors
                    ([&](const IdxTuple& offsets, // NeighborOffset.
                         int neighbor_rank,
                         int ni, // simple counter from 0.
                         YkGridPtr sendBuf,
                         YkGridPtr recvBuf)
                     {
                         if (sendBuf.get() == 0 ||
                             recvBuf.get() == 0)
                             return;
                         TRACE_MSG("  with rank " << neighbor_rank << " at relative position " <<
                                   offsets.subElements(1).makeDimValOffsetStr() << "...");
                         
                         // Submit request to receive data from neighbor.
                         if (hi == halo_irecv) {
                             auto nbytes = recvBuf->get_num_storage_bytes();
                             void* buf = recvBuf->get_raw_storage_buffer();
                             TRACE_MSG("   requesting " << makeByteStr(nbytes) << "...");
                             MPI_Irecv(buf, nbytes, MPI_BYTE,
                                       neighbor_rank, int(gi), _env->comm, &grid_recv_reqs[ni]);
                         }

                         // Wait for data from neighbor.
                         else if (hi == halo_unpack) {
                             TRACE_MSG("   waiting for data...");
                             MPI_Wait(&grid_recv_reqs[ni], MPI_STATUS_IGNORE);
                             TRACE_MSG("   done waiting for data");
                         }
                         
                         // Common code for packing/unpacking data to/from
                         // the MPI buffers from/to the current input grid.
                         // TODO: move this code to setupRank().
                         if (hi == halo_isend || hi == halo_unpack) {

                             // Begin/end vars to indicate what part
                             // of main grid to read from or write to based on
                             // the current neighbor being processed.
                             IdxTuple copy_begin, copy_end;
                             for (auto& dim : halo_sizes.getDims()) {
                                 auto& dname = dim.getName();

                                 // Init range to whole rank domain (inside halos).
                                 // These may be changed below depending on the
                                 // neighbor's direction.
                                 copy_begin.addDimBack(dname, first_idx[dname]);
                                 copy_end.addDimBack(dname, last_idx[dname] + 1); // end = last + 1.

                                 // Neighbor direction in this dim.
                                 auto neigh_ofs = offsets[dname];
                                 
                                 // Region to read from, i.e., data from inside
                                 // this rank's halo to be put into receiver's
                                 // halo.
                                 if (hi == halo_isend) {

                                     // Is this neighbor 'before' me in this dim?
                                     if (neigh_ofs == idx_t(MPIInfo::rank_prev)) {

                                         // Only read slice as wide as halo from beginning.
                                         copy_end[dname] = first_idx[dname] + halo_sizes[dname];
                                     }

                                     // Is this neighbor 'after' me in this dim?
                                     else if (neigh_ofs == idx_t(MPIInfo::rank_next)) {

                                         // Only read slice as wide as halo before end.
                                         copy_begin[dname] = last_idx[dname] + 1 - halo_sizes[dname];
                                     }

                                     // Else, this neighbor is in same posn as I am in this dim,
                                     // so we leave the default begin/end settings.
                                 }

                                 // Region to write to, i.e., into this rank's halo.
                                 else if (hi == halo_unpack) {

                                     // Is this neighbor 'before' me in this dim?
                                     if (neigh_ofs == idx_t(MPIInfo::rank_prev)) {

                                         // Only read slice as wide as halo before beginning.
                                         copy_begin[dname] = first_idx[dname] - halo_sizes[dname];
                                         copy_end[dname] = first_idx[dname];
                                     }

                                     // Is this neighbor 'after' me in this dim?
                                     else if (neigh_ofs == idx_t(MPIInfo::rank_next)) {

                                         // Only read slice as wide as halo after end.
                                         copy_begin[dname] = last_idx[dname] + 1;
                                         copy_end[dname] = last_idx[dname] + 1 + halo_sizes[dname];
                                     }

                                     // Else, this neighbor is in same posn as I am in this dim,
                                     // so we leave the default begin/end settings.
                                 }
                             } // dims.

                             // Assume only one time-step to exchange.
                             // TODO: fix this when MPI + wave-front is enabled.
                             assert(stop_dt = start_dt + 1);
                             idx_t ht = start_dt;
                             
                             // Sizes of buffer in all dims of this grid.
                             // This must match the algorithm in setupRanks().
                             IdxTuple buf_sizes;
                             for (auto& dname : gp->get_dim_names()) {
                                 idx_t dsize = 1;

                                 // domain dim?
                                 if (halo_sizes.lookup(dname))
                                     dsize = copy_end[dname] - copy_begin[dname];

                                 // step dim?
                                 else if (dname == _dims->_step_dim) {
                                     dsize = 1;
                                     copy_begin.addDimBack(dname, ht);
                                     copy_end.addDimBack(dname, ht + 1);
                                 }

                                 // misc?
                                 else {
                                     dsize = gp->get_alloc_size(dname);
                                     copy_begin.addDimBack(dname, gp->get_first_misc_index(dname));
                                     copy_end.addDimBack(dname, gp->get_last_misc_index(dname));
                                 }

                                 // This must match buffer allocation.
                                 idx_t bsize = recvBuf->get_alloc_size(dname);
                                 assert(bsize == dsize);
                                 buf_sizes.addDimBack(dname, dsize);
                             }

                             // Overall size should also match.
                             idx_t buf_elems = buf_sizes.product();
                             assert(buf_elems == recvBuf->get_num_storage_elements());
                             assert(buf_elems == sendBuf->get_num_storage_elements());
                             // Visit every point to copy.
                             // TODO: parallelize.
                             buf_sizes.visitAllPoints([&](const IdxTuple& bpt) {
                                     IdxTuple gpt = bpt.addElements(copy_begin);
                                     Indices bidxs(bpt), gidxs(gpt);
                             
                                     // Pack data for sending.
                                     if (hi == halo_isend) {

                                         // Copy this point from grid to buffer.
                                         real_t hval = gp->readElem(gidxs, __LINE__);
                                         sendBuf->writeElem(hval, bidxs, __LINE__);
                                     }

                                     // Unpack data after receiving.
                                     else if (hi == halo_unpack) {
                                         real_t hval = recvBuf->readElem(bidxs, __LINE__);
                                         gp->writeElem(hval, gidxs, __LINE__);
                                     }
                                     return true; // keep visiting.
                                 }); // visit points.
                         } // pack/unpack.

                         // Send filled buffer to neighbor.
                         if (hi == halo_isend) {
                             auto nbytes = sendBuf->get_num_storage_bytes();
                             const void* buf = (const void*)(sendBuf->get_raw_storage_buffer());
                             TRACE_MSG("   sending " << makeByteStr(nbytes) << "...");
                             MPI_Isend(buf, nbytes, MPI_BYTE,
                                       neighbor_rank, int(gi), _env->comm,
                                       &send_reqs[num_send_reqs++]);
                         }
                     }); // visit neighbors.
                
                // Mark this grid as up-to-date.
                if (hi == halo_unpack) {
                    gp->set_updated(true);
                    TRACE_MSG("  grid '" << gp->get_name() << "' marked as updated");
                }
                
            } // grids.

        } // halo sequence.

        // Wait for all send requests to complete.
        // TODO: delay this until next attempted halo exchange.
        if (num_send_reqs) {
            TRACE_MSG("exchange_halos: waiting for " << num_send_reqs <<
                      " MPI send request(s) to complete...");
            MPI_Waitall(num_send_reqs, send_reqs, MPI_STATUS_IGNORE);
            TRACE_MSG(" done waiting for MPI send request(s)");
        }
        
        double end_time = getTimeInSecs();
        mpi_time += end_time - start_time;
#endif
    }

    // Mark grids that have been written to by eq-group 'eg'.
    // TODO: only mark grids that are written to in their halo-read area.
    void StencilContext::mark_grids_dirty(EqGroupBase& eg)
    {
        for (auto gp : eg.outputGridPtrs) {
            gp->set_updated(false);
            TRACE_MSG("grid '" << gp->get_name() << "' marked as modified");
        }
    }

    // Apply a function to each neighbor rank.
    void MPIInfo::visitNeighbors(std::function<void
                                 (const IdxTuple& offsets, // NeighborOffset vals.
                                  int rank, // MPI rank.
                                  int index)> visitor) {

        for (int i = 0; i < neighborhood_size; i++) {
            auto offsets = neighbor_offsets.unlayout(i);
            int rank = my_neighbors.at(i);
            visitor(offsets, rank, i);
        }
    }
    
    // Apply a function to each neighbor rank.
    // Called visitor function will contain the rank index of the neighbor.
    void MPIBufs::visitNeighbors(std::function<void
                                 (const IdxTuple& offsets, // NeighborOffset.
                                  int rank,
                                  int index,
                                  YkGridPtr sendBuf,
                                  YkGridPtr recvBuf)> visitor) {

        _mpiInfo->visitNeighbors
            ([&](const IdxTuple& offsets,
                 int rank, int i) {

                if (rank != MPI_PROC_NULL) {
                    auto sendBuf = send_bufs.at(i);
                    auto recvBuf = recv_bufs.at(i);
                    visitor(offsets, rank, i, sendBuf, recvBuf);
                }
            });
    }

    // Access a buffer by direction and neighbor offsets.
    YkGridPtr& MPIBufs::getBuf(BufDir bd, const IdxTuple& offsets) {
        auto i = _mpiInfo->neighbor_offsets.layout(offsets); // 1D index.
        assert(i >= 0);
        assert(i < _mpiInfo->neighborhood_size);
        assert(int(bd) < int(nBufDirs));
        return (bd == bufSend) ? send_bufs.at(i) : recv_bufs.at(i);
    }

    // Create new buffer in given direction and size.
    // Does not yet allocate space in it.
    YkGridPtr MPIBufs::makeBuf(BufDir bd,               // send or recv.
                               const IdxTuple& offsets, // offset of this neighbor.
                               const IdxTuple& sizes,   // size in each grid dim.
                               const std::string& name, // name for this buffer.
                               StencilContext& context) {

        ostream& os = context.get_ostr();
        TRACE_MSG0(os, "making MPI buffer '" << name << "' for rank at " <<
                   offsets.subElements(1).makeDimValStr() << " with size " <<
                   sizes.makeDimValStr(" * "));
        auto& gp = getBuf(bd, offsets);
        gp = context.newGrid(name, sizes.getDimNames(), false);
        assert(gp);
        for (auto& dim : sizes.getDims()) {
            auto& dname = dim.getName();
            auto sz = dim.getVal();
            gp->_set_domain_size(dname, sz);
        }
        TRACE_MSG0(os, " buffer '" << name << "' has " <<
                   gp->get_num_storage_elements() << " element(s)");
        assert(getBuf(bd, offsets) == gp);
        return gp;
    }

    // Add options to set one domain var to a cmd-line parser.
    void KernelSettings::_add_domain_option(CommandLineParser& parser,
                                            const std::string& prefix,
                                            const std::string& descrip,
                                            IdxTuple& var) {

        // Add step + domain vars.
        vector<idx_t*> multi_vars;
        string multi_help;
        for (auto& dim : var.getDims()) {
            auto& dname = dim.getName();
            idx_t* dp = var.lookup(dname); // use lookup() to get non-const ptr.

            // Option for individual dim.
            parser.add_option(new CommandLineParser::IdxOption
                              (prefix + dname,
                               descrip + " in '" + dname + "' dimension.",
                               *dp));

            // Add to domain list if a domain var.
            if (_dims->_domain_dims.lookup(dname)) {
                multi_vars.push_back(dp);
                multi_help += " -" + prefix + dname + " <integer>";
            }
        }

        // Option for all domain dims.
        if (multi_vars.size() > 1) {
            parser.add_option(new CommandLineParser::MultiIdxOption
                              (prefix,
                               "Shorthand for" + multi_help,
                               multi_vars));
        }
    }
    
    // Add these settigns to a cmd-line parser.
    void KernelSettings::add_options(CommandLineParser& parser)
    {
        _add_domain_option(parser, "d", "Rank-domain size", _rank_sizes);
        _add_domain_option(parser, "r", "Region size", _region_sizes);
        _add_domain_option(parser, "bg", "Block-group size", _block_group_sizes);
        _add_domain_option(parser, "b", "Block size", _block_sizes);
        _add_domain_option(parser, "sbg", "Sub-block-group size", _sub_block_group_sizes);
        _add_domain_option(parser, "sb", "Sub-block size", _sub_block_sizes);
        _add_domain_option(parser, "mp", "Minimum grid-padding size (including halo)", _min_pad_sizes);
        _add_domain_option(parser, "ep", "Extra grid-padding size (beyond halo)", _extra_pad_sizes);
#ifdef USE_MPI
        _add_domain_option(parser, "nr", "Num ranks", _num_ranks);
        _add_domain_option(parser, "ri", "This rank's logical index", _rank_indices);
        parser.add_option(new CommandLineParser::IntOption
                          ("msg_rank",
                           "Index of MPI rank that will print informational messages.",
                           msg_rank));
#endif
        parser.add_option(new CommandLineParser::IntOption
                          ("max_threads",
                           "Max OpenMP threads to use.",
                           max_threads));
        parser.add_option(new CommandLineParser::IntOption
                          ("thread_divisor",
                           "Divide max OpenMP threads by <integer>.",
                           thread_divisor));
        parser.add_option(new CommandLineParser::IntOption
                          ("block_threads",
                           "Number of threads to use within each block.",
                           num_block_threads));
    }
    
    // Print usage message.
    void KernelSettings::print_usage(ostream& os,
                                      CommandLineParser& parser,
                                      const string& pgmName,
                                      const string& appNotes,
                                      const vector<string>& appExamples) const
    {
        os << "Usage: " << pgmName << " [options]\n"
            "Options:\n";
        parser.print_help(os);
        os << "Terms for the various levels of tiling from smallest to largest:\n"
            " A 'point' is a single floating-point (FP) element.\n"
            "  This binary uses " << REAL_BYTES << "-byte FP elements.\n"
            " A 'vector' is composed of points.\n"
            "  A 'folded vector' contains points in more than one dimension.\n"
            "  The size of a vector is typically that of a SIMD register.\n"
            " A 'vector-cluster' is composed of vectors.\n"
            "  This is the unit of work done in each inner-most loop iteration.\n"
            " A 'sub-block' is composed of vector-clusters.\n"
            "  If the number of threads-per-block is greater than one,\n"
            "   then this is the unit of work for one nested OpenMP thread;\n"
            "   else, sub-blocks are evaluated sequentially within each block.\n"
            " A 'block' is composed of sub-blocks.\n"
            "  This is the unit of work for one top-level OpenMP thread.\n"
            " A 'region' is composed of blocks.\n"
            "  If using temporal wave-front tiling (see region-size guidelines),\n"
            "   then, this is the unit of work for each wave-front tile;\n"
            "   else, there is typically only one region the sie of the rank-domain.\n"
            " A 'rank-domain' is composed of regions.\n"
            "  This is the unit of work for one MPI rank.\n"
            " The 'overall-problem' is composed of rank-domains.\n"
            "  This is the unit of work across all MPI ranks.\n" <<
#ifndef USE_MPI
            "   This binary has NOT been compiled with MPI support,\n"
            "   so the overall-problem is equivalent to the single rank-domain.\n" <<
#endif
            "Guidelines for setting tiling sizes:\n"
            " The vector and vector-cluster sizes are set at compile-time, so\n"
            "  there are no run-time options to set them.\n"
            " Set sub-block sizes to specify a unit of work done by each thread.\n"
            "  A sub-block size of 0 in dimensions 'w' or 'x' =>\n"
            "   sub-block size is set to vector-cluster size in that dimension.\n"
            "  A sub-block size of 0 in dimensions 'y' or 'z' =>\n"
            "   sub-block size is set to block size in that dimension.\n"
            "  Thus, the default sub-block is a 'y-z' slab.\n"
            "  Temporal tiling in sub-blocks is not yet supported, so effectively, sbt = 1.\n"
            " Set sub-block-group sizes to control the ordering of sub-blocks within a block.\n"
            "  All sub-blocks that intersect a given sub-block-group are evaluated\n"
            "   before sub-blocks in the next sub-block-group.\n"
            "  A sub-block-group size of 0 in a given dimension =>\n"
            "   sub-block-group size is set to sub-block size in that dimension.\n"
            " Set block sizes to specify a unit of work done by each thread team.\n"
            "  A block size of 0 in a given dimension =>\n"
            "   block size is set to region size in that dimension.\n"
            "  Temporal tiling in blocks is not yet supported, so effectively, bt = 1.\n"
            " Set block-group sizes to control the ordering of blocks within a region.\n"
            "  All blocks that intersect a given block-group are evaluated before blocks\n"
            "   in the next block-group.\n"
            "  A block-group size of 0 in a given dimension =>\n"
            "   block-group size is set to block size in that dimension.\n"
            " Set region sizes to control temporal wave-front tile sizes.\n"
            "  The temopral region size should be larger than one, and\n"
            "   the spatial region sizes should be less than the rank-domain sizes\n"
            "   in at least one dimension to enable temporal wave-front tiling.\n"
            "  The spatial region sizes should be greater than corresponding block sizes\n"
            "   to enable threading withing each wave-front tile.\n"
            "  Control the time-steps in each temporal wave-front with -rt.\n"
            "   Special cases:\n"
            "    Using '-rt 1' disables wave-front tiling.\n"
            "    Using '-rt 0' => all time-steps done in one wave-front.\n"
            "  A region size of 0 in a given dimension =>\n"
            "   region size is set to rank-domain size in that dimension.\n"
            " Set rank-domain sizes to specify the work done on this rank.\n"
            "  This and the number of grids affect the amount of memory used.\n"
            "Controlling OpenMP threading:\n"
            " Using '-max_threads 0' =>\n"
            "  max_threads is set to OpenMP's default number of threads.\n"
            " The -thread_divisor option is a convenience to control the number of\n"
            "  hyper-threads used without having to know the number of cores,\n"
            "  e.g., using '-thread_divisor 2' will halve the number of OpenMP threads.\n"
            " For stencil evaluation, threads are allocated using nested OpenMP:\n"
            "  Num threads per region = max_threads / thread_divisor / block_threads.\n"
            "  Num threads per block = block_threads.\n"
            "  Num threads per sub-block = 1.\n"
            "  Num threads used for halo exchange is same as num per region.\n" <<
#ifdef USE_MPI
            "Controlling MPI scaling:\n"
            "  To 'weak-scale' to a larger overall-problem size, use multiple MPI ranks\n"
            "   and keep the rank-domain sizes constant.\n"
            "  To 'strong-scale' a given overall-problem size, use multiple MPI ranks\n"
            "   and reduce the size of each rank-domain appropriately.\n" <<
#endif
            appNotes <<
            "Examples:\n" <<
            " " << pgmName << " -d 768 -dt 25\n" <<
            " " << pgmName << " -dx 512 -dy 256 -dz 128\n" <<
            " " << pgmName << " -d 2048 -dt 20 -r 512 -rt 10  # temporal tiling.\n" <<
            " " << pgmName << " -d 512 -nrx 2 -nry 1 -nrz 2   # multi-rank.\n";
        for (auto ae : appExamples)
            os << " " << pgmName << " " << ae << endl;
        os << flush;
    }

    // For each one of 'inner_sizes' that is zero,
    // make it equal to corresponding one in 'outer_sizes'.
    // Round up each of 'inner_sizes' to be a multiple of corresponding one in 'mults'.
    // Output info to 'os' using '*_name' and dim names.
    // Return product of number of inner subsets.
    idx_t  KernelSettings::findNumSubsets(ostream& os,
                                          IdxTuple& inner_sizes, const string& inner_name,
                                          const IdxTuple& outer_sizes, const string& outer_name,
                                          const IdxTuple& mults) {

        idx_t prod = 1;
        for (auto& dim : inner_sizes.getDims()) {
            auto& dname = dim.getName();
            idx_t* dptr = inner_sizes.lookup(dname); // use lookup() to get non-const ptr.

            idx_t outer_size = outer_sizes[dname];
            if (*dptr <= 0)
                *dptr = outer_size; // 0 => use full size as default.
            if (mults.lookup(dname))
                *dptr = ROUND_UP(*dptr, mults[dname]);
            idx_t inner_size = *dptr;
            idx_t ninner = (inner_size <= 0) ? 0 :
                (outer_size + inner_size - 1) / inner_size; // full or partial.
            idx_t rem = outer_size % inner_size;                       // size of remainder.
            idx_t nfull = rem ? (ninner - 1) : ninner; // full only.
            
            os << " In '" << dname << "' dimension, " << outer_name << " of size " <<
                outer_size << " contains " << nfull << " " <<
                    inner_name << "(s) of size " << inner_size;
            if (rem)
                os << " plus 1 remainder " << inner_name << " of size " << rem;
            os << "." << endl;
            prod *= ninner;
        }
        return prod;
    }

    // Make sure all user-provided settings are valid and finish setting up some
    // other vars before allocating memory.
    // Called from prepare_solution(), so it doesn't normally need to be called from user code.
    void KernelSettings::adjustSettings(std::ostream& os) {
        
        // Set max number of threads.
        if (max_threads <= 0)
            max_threads = omp_get_max_threads();
        
        // Determine num regions.
        // Also fix up region sizes as needed.
        // Default region size (if 0) will be size of rank-domain.
        os << "\nRegions:" << endl;
        auto nr = findNumSubsets(os, _region_sizes, "region",
                                 _rank_sizes, "rank-domain",
                                 _dims->_cluster_pts);
        auto rt = _rank_sizes[_dims->_step_dim];
        os << " num-regions-per-rank-domain: " << nr << endl;
        os << " Since the temporal region size is " << rt <<
            ", temporal wave-front tiling is ";
        if (rt <= 1) os << "NOT ";
        os << "enabled.\n";

        // Determine num blocks.
        // Also fix up block sizes as needed.
        // Default block size (if 0) will be size of region.
        os << "\nBlocks:" << endl;
        auto nb = findNumSubsets(os, _block_sizes, "block",
                                 _region_sizes, "region",
                                 _dims->_cluster_pts);
        os << " num-blocks-per-region: " << nb << endl;
        os << " num-blocks-per-rank-domain: " << (nb * nr) << endl;

        // Adjust defaults for sub-blocks to be slab.
        // Otherwise, findNumSubsets() would set default
        // to entire block.
        for (auto& dim : _dims->_domain_dims.getDims()) {
            auto& dname = dim.getName();
            if (_sub_block_sizes[dname] == 0)
                _sub_block_sizes[dname] = 1; // will be rounded up to min size.
            
            // only want to set 1st dim; others will be set to max.
            break;
        }

        // Determine num sub-blocks.
        // Also fix up sub-block sizes as needed.
        os << "\nSub-blocks:" << endl;
        auto nsb = findNumSubsets(os, _sub_block_sizes, "sub-block",
                                 _block_sizes, "block",
                                 _dims->_cluster_pts);
        os << " num-sub-blocks-per-block: " << nsb << endl;

        // Now, we adjust groups. These are done after all the above sizes
        // because group sizes are more like 'guidelines' and don't have
        // their own loops.
        os << "\nGroups:" << endl;
        
        // Adjust defaults for groups to be min size.
        // Otherwise, findNumBlockGroupsInRegion() would set default
        // to entire region.
        for (auto& dim : _dims->_domain_dims.getDims()) {
            auto& dname = dim.getName();
            if (_block_group_sizes[dname] == 0)
                _block_group_sizes[dname] = 1; // will be rounded up to min size.
            if (_sub_block_group_sizes[dname] == 0)
                _sub_block_group_sizes[dname] = 1; // will be rounded up to min size.
        }

        // Determine num block-groups.
        // Also fix up block-group sizes as needed.
        // TODO: only print this if block-grouping is enabled.
        auto nbg = findNumSubsets(os, _block_group_sizes, "block-group",
                                  _region_sizes, "region",
                                  _block_sizes);
        os << " num-block-groups-per-region: " << nbg << endl;
        auto nb_g = findNumSubsets(os, _block_sizes, "block",
                                   _block_group_sizes, "block-group",
                                   _dims->_cluster_pts);
        os << " num-blocks-per-block-group: " << nb_g << endl;

        // Determine num sub-block-groups.
        // Also fix up sub-block-group sizes as needed.
        // TODO: only print this if sub-block-grouping is enabled.
        auto nsbg = findNumSubsets(os, _sub_block_group_sizes, "sub-block-group",
                                   _block_sizes, "block",
                                   _sub_block_sizes);
        os << " num-sub-block-groups-per-block: " << nsbg << endl;
        auto nsb_g = findNumSubsets(os, _sub_block_sizes, "block",
                                   _sub_block_group_sizes, "sub-block-group",
                                   _dims->_cluster_pts);
        os << " num-sub-blocks-per-sub-block-group: " << nsb_g << endl;
    }

} // namespace yask.
