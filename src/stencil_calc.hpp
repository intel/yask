/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

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

#ifndef STENCIL_CALC
#define STENCIL_CALC

#include <string.h>
#include <algorithm>
#include <functional>

// Macro for automatically adding N dimension's arg.
// TODO: make all args programmatic.
#if USING_DIM_N
#define ARG_N(n) n,
#else
#define ARG_N(n)
#endif

namespace yask {

    // Forward defn.
    struct StencilContext;

    // MPI buffers for *one* grid.
    struct MPIBufs {

        // A type to store ranks of all possible neighbors in all
        // directions, including diagonals.
        enum NeighborOffset { rank_prev, rank_self, rank_next, num_neighbors };
        typedef int Neighbors[num_neighbors][num_neighbors][num_neighbors][num_neighbors];

        // Neighborhood size includes self.
        static const int neighborhood_size = num_neighbors * num_neighbors * num_neighbors * num_neighbors;
        
        // Need one buf for send and one for receive for each neighbor.
        enum BufDir { bufSend, bufRec, nBufDirs };

        // A type to store buffers for all possible neighbors.
        typedef Grid_NXYZ* NeighborBufs[nBufDirs][num_neighbors][num_neighbors][num_neighbors][num_neighbors];
        NeighborBufs bufs;

        MPIBufs() {
            memset(bufs, 0, sizeof(bufs));
        }

        // Access a buffer by direction and 4D neighbor indices.
        Grid_NXYZ* operator()(int bd,
                              idx_t nn, idx_t nx, idx_t ny, idx_t nz) {
            assert(bd >= 0);
            assert(bd < nBufDirs);
            assert(nn >= 0);
            assert(nn < num_neighbors);
            assert(nx >= 0);
            assert(nx < num_neighbors);
            assert(ny >= 0);
            assert(ny < num_neighbors);
            assert(nz >= 0);
            assert(nn < num_neighbors);
            return bufs[bd][nn][nx][ny][nz];
        }

        // Apply a function to each neighbor rank and/or buffer.
        virtual void visitNeighbors(StencilContext& context,
                                    std::function<void (idx_t nn, idx_t nx, idx_t ny, idx_t nz,
                                                        int rank,
                                                        Grid_NXYZ* sendBuf,
                                                        Grid_NXYZ* rcvBuf)> visitor);
            
        // Allocate new buffer in given direction and size.
        virtual Grid_NXYZ* allocBuf(int bd,
                                    idx_t nn, idx_t nx, idx_t ny, idx_t nz,
                                    idx_t dn, idx_t dx, idx_t dy, idx_t dz,
                                    const std::string& name,
                                    std::ostream& os);
    };

    // A 4D bounding-box.
    struct BoundingBox {
        idx_t begin_bbn, begin_bbx, begin_bby, begin_bbz;
        idx_t end_bbn, end_bbx, end_bby, end_bbz;
        idx_t len_bbn, len_bbx, len_bby, len_bbz;
        idx_t bb_size;
        bool bb_valid;

        BoundingBox() :
            begin_bbn(0), begin_bbx(0), begin_bby(0), begin_bbz(0),
            end_bbn(0), end_bbx(0), end_bby(0), end_bbz(0),
            len_bbn(0), len_bbx(0), len_bby(0), len_bbz(0),
            bb_size(0), bb_valid(false) { }
    };
    
    // Data and hierarchical sizes.
    // This is a pure-virtual class that must be implemented
    // for a specific problem.
    // Each eq group is valid within its bounding box (BB).
    // The context's BB encompasses all eq-group BBs.
    struct StencilContext : public BoundingBox {

        // Name.
        std::string name;

        // Default output stream for messages.
        std::ostream* ostr;

        // A list of all grids.
        std::vector<RealVecGridBase*> gridPtrs;

        // Only grids that are updated by the stencil equations.
        std::vector<RealVecGridBase*> eqGridPtrs;

        // A list of all non-grid parameters.
        std::vector<RealGrid*> paramPtrs;

        // Sizes in elements (points).
        // - time sizes (t) are in steps to be done (not grid allocation).
        // - spatial sizes (x, y, z) are in elements (not vectors).
        // Sizes are the same for all grids and all stencil equations.
        // TODO: relax this restriction.
        idx_t begin_dt;           // begin time (end is begin_dt + dt);
        idx_t dt, dn, dx, dy, dz; // rank size (without halos).
        idx_t rt, rn, rx, ry, rz; // region size.
        idx_t bt, bn, bx, by, bz; // block size.
        idx_t gn, gx, gy, gz;     // group-of-blocks size.
        idx_t hn, hx, hy, hz;     // spatial halos (max over grids as required by stencil).
        idx_t pn, px, py, pz;     // spatial padding (extra to avoid aliasing).
        idx_t angle_n, angle_x, angle_y, angle_z; // temporal skewing angles.

        // MPI meta-data.
        MPI_Comm comm;
        int num_ranks, my_rank;   // MPI-assigned index.
        idx_t nrn, nrx, nry, nrz; // number of ranks in each dim.
        idx_t rin, rix, riy, riz;    // my rank index in each dim.
        double mpi_time;          // time spent doing MPI.
        MPIBufs::Neighbors my_neighbors;   // neighbor ranks.

        // Actual MPI buffers.
        // MPI buffers are tagged by their grid pointers.
        // Only grids in eqGridPtrs will have buffers.
        std::map<RealVecGridBase*, MPIBufs> mpiBufs;

        // Shadow grids.
        // These are used to time copies back and forth between buffers used
        // by YASK for computation and those that might be needed by
        // traditional C or FORTRAN functions.  Only grids in eqGridPtrs
        // will have shadows; it is assumed that other grids will only need
        // to be copied once.
        std::map<RealVecGridBase*, RealGrid_NXYZ*> shadowGrids;
        idx_t shadow_in_freq, shadow_out_freq; // copy frequencies;
        double shadow_time;          // time spent doing shadow copies.
        
        // Threading.
        // Remember original number of threads avail.
        // We use this instead of omp_get_num_procs() so the user
        // can limit threads via OMP_NUM_THREADS env var.
        int orig_max_threads;

        // Number of threads to use in a nested OMP region.
        int num_block_threads;

        // Ctor, dtor.
        StencilContext() :
            ostr(&std::cout),
            begin_dt(TIME_DIM_SIZE),
            dt(0), dn(0), dx(0), dy(0), dz(0),
            rt(0), rn(0), rx(0), ry(0), rz(0),
            bt(0), bn(0), bx(0), by(0), bz(0),
            gn(0), gx(0), gy(0), gz(0),
            hn(0), hx(0), hy(0), hz(0),
            pn(0), px(0), py(0), pz(0),
            angle_n(0), angle_x(0), angle_y(0), angle_z(0),
            comm(0), num_ranks(1), my_rank(0), mpi_time(0.0),
            shadow_in_freq(0), shadow_out_freq(0), shadow_time(0.0),
            orig_max_threads(1), num_block_threads(1)
        {
            // Init my_neighbors to indicate no neighbor.
            int *p = (int *)my_neighbors;
            for (int i = 0; i < MPIBufs::neighborhood_size; i++)
                p[i] = MPI_PROC_NULL;
        }
        virtual ~StencilContext() { }

        // Allocate grid memory and set gridPtrs.
        virtual void allocGrids() =0;

        // Allocate param memory and set paramPtrs.
        virtual void allocParams() =0;

        // Allocate MPI buffers, etc. if enabled.
        virtual void setupMPI(bool findLocation);

        // Alloc shadow grids.
        virtual void allocShadowGrids();
        
        // Allocate grids, params, MPI bufs, and shadow grids.
        // Prints and returns num bytes.
        virtual idx_t allocAll(bool findRankLocation = true);
        
        // Get total size.
        virtual idx_t get_num_bytes();

        // Init all grids & params by calling initFn.
        virtual void initValues(std::function<void (RealVecGridBase* gp, 
                                                    real_t seed)> realVecInitFn,
                                std::function<void (RealGrid* gp,
                                                    real_t seed)> realInitFn);

        // Init all grids & params to same value within grids,
        // but different for each grid.
        virtual void initSame() {
            initValues([&](RealVecGridBase* gp, real_t seed){ gp->set_same(seed); },
                       [&](RealGrid* gp, real_t seed){ gp->set_same(seed); });
        }

        // Init all grids & params to different values within grids,
        // and different for each grid.
        virtual void initDiff() {
            initValues([&](RealVecGridBase* gp, real_t seed){ gp->set_diff(seed); },
                       [&](RealGrid* gp, real_t seed){ gp->set_diff(seed); });
        }

        // Init all grids & params 
        // By default it uses the initSame initialization routine
        virtual void init() {
            initSame();
        }

        // Compare grids in contexts for validation.
        // Params should not be written to, so they are not compared.
        // Return number of mis-compares.
        virtual idx_t compare(const StencilContext& ref) const;

        // Set number of threads to use for something other than a region.
        inline int set_max_threads() {

            // Reset number of OMP threads back to max allowed.
            int nt = orig_max_threads;
            omp_set_num_threads(nt);
            return nt;
        }

        // Set number of threads to use for a region.
        // Return that number.
        inline int set_region_threads() {

            int nt = orig_max_threads;

#if !USE_CREW
            // If not using crew, limit outer nesting to allow
            // num_block_threads per nested block loop.
            nt /= num_block_threads;
#endif

            omp_set_num_threads(nt);
            return nt;
        }

        // Set number of threads for a block.
        inline int set_block_threads() {

            // This should be a nested OMP region.
            omp_set_num_threads(num_block_threads);
            return num_block_threads;
        }

    };

    /// Classes that support evaluation of one stencil equation-set.

    // A pure-virtual class base for a stencil equation-set.
    class EqGroupBase : public BoundingBox {
    public:

        // ctor, dtor.
        EqGroupBase() { }
        virtual ~EqGroupBase() { }

        // Get name of this equation set.
        virtual const std::string& get_name() const =0;

        // Get estimated number of FP ops done for one scalar eval.
        virtual int get_scalar_fp_ops() const =0;

        // Get number of points updated for one scalar eval.
        virtual int get_scalar_points_updated() const =0;

        // Get list of grids updated by this equation.
        virtual std::vector<RealVecGridBase*>& get_eq_grid_ptrs() =0;

        // Set eqGridPtrs.
        virtual void init(StencilContext& generic_context) =0;
    
        // Set the bounding-box vars for this eq group.
        virtual void find_bounding_box(StencilContext& context);

        // Determine whether indices are in [sub-]domain.
        virtual bool is_in_valid_domain(StencilContext& generic_context,
                                        idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) =0;

        // Calculate one scalar result at time t.
        virtual void calc_scalar(StencilContext& generic_context,
                                 idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) =0;

        // Calculate one block of results from begin to end-1 on each dimension.
        // Note: this interface cannot support temporal blocking with >1 stencil because
        // it only operates on one stencil.
        virtual void calc_block(StencilContext& generic_context, idx_t bt,
                                idx_t begin_bn, idx_t begin_bx, idx_t begin_by, idx_t begin_bz,
                                idx_t end_bn, idx_t end_bx, idx_t end_by, idx_t end_bz) =0;

        // Exchange halo and shadow data for the updated grids at the given time.
        virtual void exchange_halos(StencilContext& generic_context, idx_t start_dt, idx_t stop_dt);
    };

    // Collections of equation sets.
    typedef std::vector<EqGroupBase*> EqGroupList;
    typedef std::set<EqGroupBase*> EqGroupSet;

    // Define a method named cfn to prefetch a cluster by calling vfn.
#define PREFETCH_CLUSTER_METHOD(cfn, vfn)                               \
    template<int level>                                                 \
    ALWAYS_INLINE void                                                  \
    cfn (ContextClass& context, idx_t ct,                               \
         idx_t begin_cnv, idx_t begin_cxv, idx_t begin_cyv, idx_t begin_czv, \
         idx_t end_cnv, idx_t end_cxv, idx_t end_cyv, idx_t end_czv) {  \
        TRACE_MSG("%s.%s<%d>(%ld, %ld, %ld, %ld, %ld)",                 \
                  get_name().c_str(), #cfn, level, ct,                  \
                  begin_cnv, begin_cxv, begin_cyv, begin_czv);          \
        _eqGroup.vfn<level>(context, ct,                                \
                     ARG_N(begin_cnv) begin_cxv, begin_cyv, begin_czv); \
    }

    // A template that provides wrappers around a stencil-equation class
    // created by the foldBuilder. A template is used instead of inheritance
    // for performance.  By using templates, the compiler can inline stencil
    // code into loops and avoid indirect calls. TODO: maybe templating
    // isn't really necessary: optimization through a known concrete derived
    // class may be just as good and much simpler. Need to try this.
    template <typename EqGroupClass, typename ContextClass>
    class EqGroupTemplate : public EqGroupBase {

    protected:

        // EqGroupClass must implement calc_scalar(), calc_cluster(),
        // etc., that are used below.
        // This class is generated by the foldBuilder.
        EqGroupClass _eqGroup;

    public:

        EqGroupTemplate() {}
        EqGroupTemplate(const std::string& name) :
            EqGroupBase(name) { }
        virtual ~EqGroupTemplate() {}

        // Get values from _eqGroup.
        virtual const std::string& get_name() const {
            return _eqGroup.name;
        }
        virtual int get_scalar_fp_ops() const {
            return _eqGroup.scalar_fp_ops;
        }
        virtual int get_scalar_points_updated() const {
            return _eqGroup.scalar_points_updated;
        }
        virtual std::vector<RealVecGridBase*>& get_eq_grid_ptrs() {
            return _eqGroup.eqGridPtrs;
        }

        // Init data.
        // This function implements the interface in the base class.
        virtual void init(StencilContext& generic_context) {

            // Convert to a problem-specific context.
            auto context = dynamic_cast<ContextClass&>(generic_context);

            // Call the generated code.
            _eqGroup.init(context);
        }
    
        // Determine whether indices are in [sub-]domain for this eq group.
        virtual bool is_in_valid_domain(StencilContext& generic_context,
                                        idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) {
            return _eqGroup.is_in_valid_domain(generic_context, t,
                                               ARG_N(n) x, y, z);
        }

        // Calculate one scalar result.
        // This function implements the interface in the base class.
        virtual void calc_scalar(StencilContext& generic_context,
                                 idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) {

            // Convert to a problem-specific context.
            auto context = dynamic_cast<ContextClass&>(generic_context);

            // Call the generated code.
            _eqGroup.calc_scalar(context, t, ARG_N(n) x, y, z);
        }

        // Calculate results within a cluster of vectors.
        // Called from calc_block().
        // The begin/end_c* vars are the start/stop_b* vars from the block loops.
        ALWAYS_INLINE void
        calc_cluster(ContextClass& context, idx_t ct,
                     idx_t begin_cnv, idx_t begin_cxv, idx_t begin_cyv, idx_t begin_czv,
                     idx_t end_cnv, idx_t end_cxv, idx_t end_cyv, idx_t end_czv)
        {
            TRACE_MSG("%s.calc_cluster(%ld, %ld, %ld, %ld, %ld)",
                      get_name().c_str(), ct, begin_cnv, begin_cxv, begin_cyv, begin_czv);

            // The step vars are hard-coded in calc_block below, and there should
            // never be a partial step at this level. So, we can assume one var and
            // exactly CLEN_d steps in each given direction d are calculated in this
            // function.  Thus, we can ignore the end_* vars in the calc function.
            assert(end_cnv == begin_cnv + CLEN_N);
            assert(end_cxv == begin_cxv + CLEN_X);
            assert(end_cyv == begin_cyv + CLEN_Y);
            assert(end_czv == begin_czv + CLEN_Z);
        
            // Calculate results.
            _eqGroup.calc_cluster(context, ct, ARG_N(begin_cnv) begin_cxv, begin_cyv, begin_czv);
        }

        // Prefetch a cluster.
        // Separate methods for full cluster and each direction.
        PREFETCH_CLUSTER_METHOD(prefetch_cluster, prefetch_cluster)
#if USING_DIM_N
        PREFETCH_CLUSTER_METHOD(prefetch_cluster_bnv, prefetch_cluster_n)
#endif
        PREFETCH_CLUSTER_METHOD(prefetch_cluster_bxv, prefetch_cluster_x)
        PREFETCH_CLUSTER_METHOD(prefetch_cluster_byv, prefetch_cluster_y)
        PREFETCH_CLUSTER_METHOD(prefetch_cluster_bzv, prefetch_cluster_z)
    
        // Calculate results within a cache block.
        // This function implements the interface in the base class.
        // Each block is typically computed in a separate OpenMP task.
        // The begin/end_b* vars are the start/stop_r* vars from the region loops.
        virtual void
        calc_block(StencilContext& generic_context, idx_t bt,
                   idx_t begin_bn, idx_t begin_bx, idx_t begin_by, idx_t begin_bz,
                   idx_t end_bn, idx_t end_bx, idx_t end_by, idx_t end_bz)
        {
            TRACE_MSG("%s.calc_block(%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld)", 
                      get_name().c_str(), bt,
                      begin_bn, end_bn-1,
                      begin_bx, end_bx-1,
                      begin_by, end_by-1,
                      begin_bz, end_bz-1);

            // Convert to a problem-specific context.
            auto context = dynamic_cast<ContextClass&>(generic_context);

            // Divide indices by vector lengths.  Use idiv() instead of '/'
            // because begin/end vars may be negative (if in halo).
            const idx_t begin_bnv = idiv<idx_t>(begin_bn, VLEN_N);
            const idx_t begin_bxv = idiv<idx_t>(begin_bx, VLEN_X);
            const idx_t begin_byv = idiv<idx_t>(begin_by, VLEN_Y);
            const idx_t begin_bzv = idiv<idx_t>(begin_bz, VLEN_Z);
            const idx_t end_bnv = idiv<idx_t>(end_bn, VLEN_N);
            const idx_t end_bxv = idiv<idx_t>(end_bx, VLEN_X);
            const idx_t end_byv = idiv<idx_t>(end_by, VLEN_Y);
            const idx_t end_bzv = idiv<idx_t>(end_bz, VLEN_Z);

            // Vector-size steps are based on cluster lengths.
            // Using CLEN_* instead of CPTS_* because we want vector lengths.
            const idx_t step_bnv = CLEN_N;
            const idx_t step_bxv = CLEN_X;
            const idx_t step_byv = CLEN_Y;
            const idx_t step_bzv = CLEN_Z;

            // Groups in block loops are set to smallest size.
            const idx_t group_size_bnv = 1;
            const idx_t group_size_bxv = 1;
            const idx_t group_size_byv = 1;
            const idx_t group_size_bzv = 1;
            
#if !defined(DEBUG) && defined(__INTEL_COMPILER)
#pragma forceinline recursive
#endif
            {
                // Set threads for a block.
                context.set_block_threads();

                // Include automatically-generated loop code that calls calc_cluster()
                // and optionally, the prefetch functions().
#include "stencil_block_loops.hpp"
            }
        }
    };

    // Collection of all stencil equations to be evaluated.
    struct StencilEqs {

        // Name of the problem.
        std::string name;

        // List of all stencil equations in the order in which
        // they should be evaluated. Current assumption is that
        // later ones are dependent on their predecessors.
        // TODO: relax this assumption, determining which eqGroups
        // are actually dependent on which others, allowing
        // more parallelism.
        EqGroupList eqGroups;

        // ctor, dtor.
        StencilEqs() {}
        virtual ~StencilEqs() {}

        // Reference stencil calculations.
        virtual void calc_rank_ref(StencilContext& context);

        // Vectorized and blocked stencil calculations.
        virtual void calc_rank_opt(StencilContext& context);

        // Set the bounding-box vars for all eq groups.
        virtual void find_bounding_boxes(StencilContext& context);

        // Initialize some data structures.
        // Must be called after the context grids are allocated.
        virtual void init(StencilContext& context,
                          idx_t* sum_points = NULL,
                          idx_t* sum_fpops = NULL);
    
    protected:
    
        // Calculate results within a region.
        void calc_region(StencilContext& context, idx_t start_dt, idx_t stop_dt,
                         EqGroupSet& eqGroup_set,
                         idx_t start_dn, idx_t start_dx, idx_t start_dy, idx_t start_dz,
                         idx_t stop_dn, idx_t stop_dx, idx_t stop_dy, idx_t stop_dz);
    
    };
}

#endif
