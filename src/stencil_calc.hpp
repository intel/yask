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

#ifndef STENCIL_CALC
#define STENCIL_CALC

#include <string.h>
#include <algorithm>
#include <functional>

// Stencil types.
#include "stencil.hpp"

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
        Grid_NXYZ** getBuf(int bd,
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
            return &bufs[bd][nn][nx][ny][nz];
        }

        // Apply a function to each neighbor rank.
        // Called visitor function will contain the rank index of the neighbor.
        // The send and receive buffer pointers may be null.
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

    // Application settings to control size and perf of stencil code.
    struct StencilSettings {

        // Sizes in elements (points).
        // - time sizes (t) are in steps to be done.
        // - spatial sizes (n, x, y, z) are in elements (not vectors).
        // Sizes are the same for all grids. TODO: relax this restriction.
        idx_t dt, dn, dx, dy, dz; // rank size (without halos).
        idx_t rt, rn, rx, ry, rz; // region size (used for wave-front tiling).
        idx_t bt, bn, bx, by, bz; // block size (used for cache locality).
        idx_t gn, gx, gy, gz;     // group-of-blocks size (only used for 'grouped' loop paths).
        idx_t pn, px, py, pz;     // spatial padding (in addition to halos, to avoid aliasing).

        // MPI settings.
        idx_t nrn, nrx, nry, nrz; // number of ranks in each dim.
        idx_t rin, rix, riy, riz; // my rank index in each dim.
        bool find_loc;            // whether my rank index needs to be calculated.
        int msg_rank;             // rank that prints informational messages.

        // OpenMP settings.
        int max_threads;       // Initial number of threads to use overall.
        int thread_divisor;    // Reduce number of threads by this amount.
        int num_block_threads; // Number of threads to use for a block.

        // Ctor.
        StencilSettings() :
            dt(50), dn(1), dx(DEF_RANK_SIZE), dy(DEF_RANK_SIZE), dz(DEF_RANK_SIZE),
            rt(1), rn(0), rx(0), ry(0), rz(0),
            bt(1), bn(1), bx(DEF_BLOCK_SIZE), by(DEF_BLOCK_SIZE), bz(DEF_BLOCK_SIZE),
            gn(0), gx(0), gy(0), gz(0),
            pn(0), px(DEF_PAD), py(DEF_PAD), pz(DEF_PAD),
            nrn(1), nrx(1), nry(1), nrz(1),
            rin(0), rix(0), riy(0), riz(0),
            find_loc(true),
            msg_rank(0),
            max_threads(1),
            thread_divisor(DEF_THREAD_DIVISOR),
            num_block_threads(DEF_BLOCK_THREADS)
        {
            max_threads = omp_get_max_threads();
        }

        // Add these settigns to a cmd-line parser.
        virtual void add_options(CommandLineParser& parser);

        // Print usage message.
        void print_usage(std::ostream& os,
                         CommandLineParser& parser,
                         const std::string& pgmName,
                         const std::string& appNotes,
                         const std::vector<std::string>& appExamples) const;
        
        // Make sure all user-provided settings are valid.
        // Called from allocAll(), so it doesn't normally need to be called from user code.
        virtual void finalizeSettings(std::ostream& os);
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

    // Collections of things in a context.
    class EqGroupBase;
    typedef std::vector<EqGroupBase*> EqGroupList;
    typedef std::set<EqGroupBase*> EqGroupSet;
    typedef std::vector<RealVecGridBase*> GridPtrs;
    typedef std::vector<RealGrid*> ParamPtrs;
    typedef std::set<std::string> NameSet;
    
    // Data and hierarchical sizes.
    // This is a pure-virtual class that must be implemented
    // for a specific problem.
    // Each eq group is valid within its bounding box (BB).
    // The context's BB encompasses all eq-group BBs.
    class StencilContext : public BoundingBox {

    protected:
        
        // Output stream for messages.
        std::ostream* _ostr;

        // Command-line and env parameters.
        StencilSettings* _opts;
        
        // TODO: move vars into private or protected sections and
        // add accessor methods.
    public:

        // Name.
        std::string name;

        // List of all stencil equations in the order in which
        // they should be evaluated. Current assumption is that
        // later ones are dependent on their predecessors.
        // TODO: relax this assumption, determining which eqGroups
        // are actually dependent on which others, allowing
        // more parallelism.
        EqGroupList eqGroups;

        // A list of all grids.
        GridPtrs gridPtrs;
        NameSet gridNames;

        // Only grids that are updated by the stencil equations.
        GridPtrs outputGridPtrs;
        NameSet outputGridNames;

        // A list of all non-grid parameters.
        ParamPtrs paramPtrs;
        NameSet paramNames;

        // Some calculated sizes.
        idx_t ofs_t, ofs_n, ofs_x, ofs_y, ofs_z; // Index offsets for this rank.
        idx_t tot_n, tot_x, tot_y, tot_z; // Total of rank domains over all ranks.
        idx_t hn, hx, hy, hz;     // spatial halos (max over grids as required by stencil).
        idx_t angle_n, angle_x, angle_y, angle_z; // temporal skewing angles.

        // Various metrics calculated in allocAll().
        // 'rank_' prefix indicates for this rank.
        // 'tot_' prefix indicates over all ranks.
        // 'numpts' indicates points actually calculated in sub-domains.
        // 'domain' indicates points in domain-size specified on cmd-line.
        // 'numFpOps' indicates est. number of FP ops.
        // 'nbytes' indicates number of bytes allocated.
        // '_1t' suffix indicates work for one time-step.
        // '_dt' suffix indicates work for all time-steps.
        idx_t rank_numpts_1t, rank_numpts_dt, tot_numpts_1t, tot_numpts_dt;
        idx_t rank_domain_1t, rank_domain_dt, tot_domain_1t, tot_domain_dt;
        idx_t rank_numFpOps_1t, rank_numFpOps_dt, tot_numFpOps_1t, tot_numFpOps_dt;
        idx_t rank_nbytes, tot_nbytes;
        
        // MPI environment.
        MPI_Comm comm;
        int num_ranks, my_rank;   // MPI-assigned index.
        double mpi_time;          // time spent doing MPI.
        MPIBufs::Neighbors my_neighbors;   // neighbor ranks.

        // Actual MPI buffers.
        // MPI buffers are tagged by their grid names.
        // Only grids in outputGridPtrs will have buffers.
        std::map<std::string, MPIBufs> mpiBufs;
        
        // Constructor.
        StencilContext(StencilSettings& settings) :
            _ostr(&std::cout),
            _opts(&settings),
            ofs_t(0), ofs_n(0), ofs_x(0), ofs_y(0), ofs_z(0),
            hn(0), hx(0), hy(0), hz(0),
            angle_n(0), angle_x(0), angle_y(0), angle_z(0),
            comm(0), num_ranks(1), my_rank(0),
            mpi_time(0.0)
        {
            // Init my_neighbors to indicate no neighbor.
            int *p = (int *)my_neighbors;
            for (int i = 0; i < MPIBufs::neighborhood_size; i++)
                p[i] = MPI_PROC_NULL;
        }

        // Destructor.
        virtual ~StencilContext() { }

        // Init MPI, OMP, etc.
        // This is normally called very early in the program.
        virtual void initEnv(int* argc, char*** argv);

        // Set ostr to given stream if provided.
        // If not provided, set to cout if my_rank == msg_rank
        // or a null stream otherwise.
        virtual std::ostream& set_ostr(std::ostream* os = NULL);

        // Get the default output stream.
        virtual std::ostream& get_ostr() const {
            assert(_ostr);
            return *_ostr;
        }

        // Get access to settings.
        virtual StencilSettings& get_settings() {
            assert(_opts);
            return *_opts;
        }
        
        // Set vars related to this rank's role in global problem.
        // Allocate MPI buffers as needed.
        // Called from allocAll(), so it doesn't normally need to be called from user code.
        virtual void setupRank();

        // Allocate grid memory.
        // Called from allocAll(), so it doesn't normally need to be called from user code.
        virtual void allocGrids() =0;

        // Allocate param memory.
        // Called from allocAll(), so it doesn't normally need to be called from user code.
        virtual void allocParams() =0;

        // Set pointers to allocated grids.
        // Called from allocAll(), so it doesn't normally need to be called from user code.
        virtual void setPtrs() =0;
        
        // Allocate grids, params, MPI bufs, etc.
        // Initialize some other data structures.
        virtual void allocAll();
        
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

        // Init all grids & params.
        // By default it uses the initSame initialization routine.
        virtual void initData() {
            initSame();
        }

        // Compare grids in contexts for validation.
        // Params should not be written to, so they are not compared.
        // Return number of mis-compares.
        virtual idx_t compareData(const StencilContext& ref) const;
        
        // Set number of threads to use for something other than a region.
        virtual int set_all_threads() {

            // Reset number of OMP threads to max allowed.
            int nt = _opts->max_threads / _opts->thread_divisor;
            nt = std::max(nt, 1);
            TRACE_MSG("set_all_threads: omp_set_num_threads(%d)", nt);
            omp_set_num_threads(nt);
            return nt;
        }

        // Set number of threads to use for a region.
        // Return that number.
        virtual int set_region_threads() {

            // Start with "all" threads.
            int nt = _opts->max_threads / _opts->thread_divisor;

            // Limit outer nesting to allow num_block_threads per nested
            // block loop.
            nt /= _opts->num_block_threads;
            nt = std::max(nt, 1);
            if (_opts->num_block_threads > 1)
                omp_set_nested(1);

            TRACE_MSG("set_region_threads: omp_set_num_threads(%d)", nt);
            omp_set_num_threads(nt);
            return nt;
        }

        // Set number of threads for a block.
        virtual int set_block_threads() {

            // This should be a nested OMP region.
            int nt = _opts->num_block_threads;
            nt = std::max(nt, 1);
            TRACE_MSG("set_block_threads: omp_set_num_threads(%d)", nt);
            omp_set_num_threads(nt);
            return nt;
        }

        // Wait for a global barrier.
        virtual void global_barrier() {
            MPI_Barrier(comm);
        }
        
        // Reference stencil calculations.
        virtual void calc_rank_ref();

        // Vectorized and blocked stencil calculations.
        virtual void calc_rank_opt();

        // Calculate results within a region.
        // TODO: create a public interface w/a more logical index ordering.
        virtual void calc_region(idx_t start_dt, idx_t stop_dt,
                                 EqGroupSet* eqGroup_set,
                                 idx_t start_dn, idx_t start_dx, idx_t start_dy, idx_t start_dz,
                                 idx_t stop_dn, idx_t stop_dx, idx_t stop_dy, idx_t stop_dz);

        // Set the bounding-box around all eq groups.
        virtual void find_bounding_boxes();

    };

    /// Classes that support evaluation of one stencil equation-group.
    /// A context contains of one or more equation-groups.

    // A pure-virtual class base for a stencil equation-set.
    class EqGroupBase : public BoundingBox {
    protected:
        StencilContext* _generic_context;
        
    public:

        // Grids that are updated by these stencil equations.
        GridPtrs outputGridPtrs;

        // ctor, dtor.
        EqGroupBase() { }
        virtual ~EqGroupBase() { }

        // Get name of this equation set.
        virtual const std::string& get_name() const =0;

        // Get estimated number of FP ops done for one scalar eval.
        virtual int get_scalar_fp_ops() const =0;

        // Get number of points updated for one scalar eval.
        virtual int get_scalar_points_updated() const =0;

        // Set various pointers.
        virtual void setPtrs(StencilContext& context) =0;
        
        // Set the bounding-box vars for this eq group in this rank.
        virtual void find_bounding_box();

        // Determine whether indices are in [sub-]domain.
        virtual bool is_in_valid_domain(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) =0;

        // Calculate one scalar result at time t.
        virtual void calc_scalar(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) =0;

        // Calculate one block of results from begin to end-1 on each dimension.
        virtual void calc_block(idx_t bt,
                                idx_t begin_bn, idx_t begin_bx, idx_t begin_by, idx_t begin_bz,
                                idx_t end_bn, idx_t end_bx, idx_t end_by, idx_t end_bz) =0;

        // Exchange halo data for the updated grids at the given time.
        virtual void exchange_halos(idx_t start_dt, idx_t stop_dt);
    };

    // Define a method named cfn to prefetch a cluster by calling vfn.
#define PREFETCH_CLUSTER_METHOD(cfn, vfn)                               \
    template<int level>                                                 \
    ALWAYS_INLINE void                                                  \
    cfn (idx_t ct,                                                      \
         idx_t begin_cnv, idx_t begin_cxv, idx_t begin_cyv, idx_t begin_czv, \
         idx_t end_cnv, idx_t end_cxv, idx_t end_cyv, idx_t end_czv) {  \
        TRACE_MSG("%s.%s<%d>(%ld, %ld, %ld, %ld, %ld)",                 \
                  get_name().c_str(), #cfn, level, ct,                  \
                  begin_cnv, begin_cxv, begin_cyv, begin_czv);          \
        _eqGroup.vfn<level>(*_context, ct,                               \
                     ARG_N(begin_cnv) begin_cxv, begin_cyv, begin_czv); \
    }

    // A template that provides wrappers around a stencil-equation class
    // created by the foldBuilder. A template is used instead of inheritance
    // for performance.  By using templates, the compiler can inline stencil
    // code into loops and avoid indirect calls.
    template <typename EqGroupClass, typename ContextClass>
    class EqGroupTemplate : public EqGroupBase {

    protected:
        ContextClass* _context;
        
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

        // Set some pointers.
        virtual void setPtrs(StencilContext& generic_context) {
            _generic_context = &generic_context;

            // Convert to a problem-specific context.
            _context = dynamic_cast<ContextClass*>(_generic_context);
            assert(_context);

            outputGridPtrs.clear();
            _eqGroup.setPtrs(*_context, outputGridPtrs);
        }
    
        // Determine whether indices are in [sub-]domain for this eq group.
        virtual bool is_in_valid_domain(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) {
            return _eqGroup.is_in_valid_domain(*_context, t,
                                               ARG_N(n) x, y, z);
        }

        // Calculate one scalar result.
        // This function implements the interface in the base class.
        virtual void calc_scalar(idx_t t, idx_t n, idx_t x, idx_t y, idx_t z) {
            _eqGroup.calc_scalar(*_context, t, ARG_N(n) x, y, z);
        }

        // Calculate results within a cluster of vectors.
        // Called from calc_block().
        // The begin/end_c* vars are the start/stop_b* vars from the block loops.
        ALWAYS_INLINE void
        calc_cluster(idx_t ct,
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
            _eqGroup.calc_cluster(*_context, ct, ARG_N(begin_cnv) begin_cxv, begin_cyv, begin_czv);
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
        calc_block(idx_t bt,
                   idx_t begin_bn, idx_t begin_bx, idx_t begin_by, idx_t begin_bz,
                   idx_t end_bn, idx_t end_bx, idx_t end_by, idx_t end_bz)
        {
            TRACE_MSG("%s.calc_block(%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld)", 
                      get_name().c_str(), bt,
                      begin_bn, end_bn-1,
                      begin_bx, end_bx-1,
                      begin_by, end_by-1,
                      begin_bz, end_bz-1);

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
            // Using CLEN_* instead of CPTS_* because we want multiples of vector lengths.
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
                _context->set_block_threads();

                // Include automatically-generated loop code that calls calc_cluster()
                // and optionally, the prefetch functions().
#include "stencil_block_loops.hpp"
            }
        }
    };

}

// Include auto-generated stencil code.
#include "stencil_code.hpp"

#endif
