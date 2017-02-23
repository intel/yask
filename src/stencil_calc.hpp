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

// Macro for automatically adding W dimension's arg.
// TODO: make all args programmatic.
#if USING_DIM_W
#define ARG_W(w) w,
#else
#define ARG_W(w)
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
        typedef Grid_WXYZ* NeighborBufs[nBufDirs][num_neighbors][num_neighbors][num_neighbors][num_neighbors];
        NeighborBufs bufs;

        MPIBufs() {
            memset(bufs, 0, sizeof(bufs));
        }

        // Access a buffer by direction and 4D neighbor indices.
        Grid_WXYZ** getBuf(int bd,
                           idx_t nw, idx_t nx, idx_t ny, idx_t nz) {
            assert(bd >= 0);
            assert(bd < nBufDirs);
            assert(nw >= 0);
            assert(nw < num_neighbors);
            assert(nx >= 0);
            assert(nx < num_neighbors);
            assert(ny >= 0);
            assert(ny < num_neighbors);
            assert(nz >= 0);
            assert(nz < num_neighbors);
            return &bufs[bd][nw][nx][ny][nz];
        }

        // Apply a function to each neighbor rank.
        // Called visitor function will contain the rank index of the neighbor.
        // The send and receive buffer pointers may be null if 'null_ok' is true.
        // TODO: remove 'null_ok' when non-symmetrical halos are supported.
        virtual void visitNeighbors(StencilContext& context,
                                    bool null_ok,
                                    std::function<void (idx_t nw, idx_t nx, idx_t ny, idx_t nz,
                                                        int rank,
                                                        Grid_WXYZ* sendBuf,
                                                        Grid_WXYZ* rcvBuf)> visitor);
            
        // Create new buffer in given direction and size.
        virtual Grid_WXYZ* makeBuf(int bd,
                                   idx_t nw, idx_t nx, idx_t ny, idx_t nz,
                                   idx_t dw, idx_t dx, idx_t dy, idx_t dz,
                                   const std::string& name);
    };

    // Application settings to control size and perf of stencil code.
    struct StencilSettings {

        // Sizes in elements (points).
        // - time sizes (t) are in steps to be done.
        // - spatial sizes (w, x, y, z) are in elements (not vectors).
        // Sizes are the same for all grids. TODO: relax this restriction.
        idx_t dt=1, dw=0, dx=0, dy=0, dz=0; // rank size (without halos).
        idx_t rt=1, rw=0, rx=0, ry=0, rz=0; // region size (used for wave-front tiling).
        idx_t bt=1, bw=0, bx=0, by=0, bz=0; // block size (used for cache locality).
        idx_t gw=0, gx=0, gy=0, gz=0;     // group-of-blocks size (only used for 'grouped' loop paths).
        idx_t pw=0, px=0, py=0, pz=0;     // spatial padding (in addition to halos, to avoid aliasing).

        // MPI settings.
        idx_t nrw=1, nrx=1, nry=1, nrz=1; // number of ranks in each dim.
        idx_t riw=0, rix=0, riy=0, riz=0; // my rank index in each dim.
        bool find_loc=true;            // whether my rank index needs to be calculated.
        int msg_rank=0;             // rank that prints informational messages.

        // OpenMP settings.
        int max_threads=1;       // Initial number of threads to use overall.
        int thread_divisor;    // Reduce number of threads by this amount.
        int num_block_threads; // Number of threads to use for a block.

        // Ctor.
        StencilSettings() :
            dt(50), dw(1), dx(DEF_RANK_SIZE), dy(DEF_RANK_SIZE), dz(DEF_RANK_SIZE),
            bt(1), bw(1), bx(DEF_BLOCK_SIZE), by(DEF_BLOCK_SIZE), bz(DEF_BLOCK_SIZE),
            pw(0), px(DEF_PAD), py(DEF_PAD), pz(DEF_PAD),
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
        idx_t begin_bbw=0, begin_bbx=0, begin_bby=0, begin_bbz=0;
        idx_t end_bbw=1, end_bbx=1, end_bby=1, end_bbz=1; // one past last value.
        idx_t len_bbw=1, len_bbx=1, len_bby=1, len_bbz=1;
        idx_t bb_size=1;        // points in the entire box.
        idx_t bb_num_points=1;  // valid points within the box.
        bool bb_simple=true;    // full box with vector-length sizes.
        bool bb_valid=false;    // lengths and sizes have been calculated.
        
        BoundingBox() {}

        // Find lengths and set valid to true.
        virtual void update_lengths() {
            len_bbw = end_bbw - begin_bbw;
            len_bbx = end_bbx - begin_bbx;
            len_bby = end_bby - begin_bby;
            len_bbz = end_bbz - begin_bbz;
            bb_size = len_bbw * len_bbx * len_bby * len_bbz;
            bb_valid = true;
        }
    };

    // Collections of things in a context.
    class EqGroupBase;
    typedef std::vector<EqGroupBase*> EqGroupList;
    typedef std::set<EqGroupBase*> EqGroupSet;
    typedef std::vector<RealVecGridBase*> GridPtrs;
    typedef std::set<RealVecGridBase*> GridPtrSet;
    typedef std::vector<RealGrid*> ParamPtrs;
    typedef std::map<std::string, RealVecGridBase*> GridPtrMap;
    typedef std::map<std::string, RealGrid*> ParamPtrMap;
    
    // Data and hierarchical sizes.
    // This is a pure-virtual class that must be implemented
    // for a specific problem.
    // Each eq group is valid within its bounding box (BB).
    // The context's BB encompasses all eq-group BBs.
    class StencilContext : public BoundingBox {

    private:
        // Disallow copying.
        StencilContext(const StencilContext& src) {
            exit_yask(1);
        }
        void operator=(const StencilContext& src) {
            exit_yask(1);
        }
        
    protected:
        
        // Output stream for messages.
        std::ostream* _ostr;

        // Command-line and env parameters.
        StencilSettings* _opts;

        // Underlying data allocation.
        // TODO: create different types of memory, e.g., HBM.
        void* _data_buf = 0;
        size_t _data_buf_size = 0;

        // Byes between each buffer to help avoid aliasing
        // in the HW.
        size_t _data_buf_pad = (YASK_PAD * CACHELINE_BYTES);

        // Alignment for _data_buf;
        size_t _data_buf_alignment = YASK_ALIGNMENT;
        
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

        // All grids.
        GridPtrs gridPtrs;
        GridPtrMap gridMap;

        // Only grids that are updated by the stencil equations.
        GridPtrs outputGridPtrs;
        GridPtrMap outputGridMap;

        // Non-grid parameters.
        ParamPtrs paramPtrs;
        ParamPtrMap paramMap;

        // Some calculated sizes.
        idx_t ofs_t=0, ofs_w=0, ofs_x=0, ofs_y=0, ofs_z=0; // Index offsets for this rank.
        idx_t tot_w=0, tot_x=0, tot_y=0, tot_z=0; // Total of rank domains over all ranks.

        // Maximum halos and skewing angles over all grids and
        // equations. Used for calculating worst-case minimum regions.
        idx_t hw=0, hx=0, hy=0, hz=0;                     // spatial halos.
        idx_t angle_w=0, angle_x=0, angle_y=0, angle_z=0; // temporal skewing angles.

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
        MPI_Comm comm=0;
        int num_ranks=1, my_rank=0;   // MPI-assigned index.
        double mpi_time=0.0;          // time spent doing MPI.
        MPIBufs::Neighbors my_neighbors;   // neighbor ranks.

        // Actual MPI buffers.
        // MPI buffers are tagged by their grid names.
        std::map<std::string, MPIBufs> mpiBufs;
        
        // Constructor.
        StencilContext(StencilSettings& settings) :
            _ostr(&std::cout),
            _opts(&settings)
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

        // Copy env settings from another context.
        virtual void copyEnv(const StencilContext& src);

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

        // Allocate grid, param, and MPI memory.
        // Called from allocAll(), so it doesn't normally need to be called from user code.
        virtual void allocData();

        // Allocate grids, params, MPI bufs, etc.
        // Initialize some other data structures.
        // Print lots of stats.
        virtual void allocAll();
        
        // Get total memory allocation.
        virtual size_t get_num_bytes() {
            return _data_buf_size;
        }

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
            TRACE_MSG("set_all_threads: omp_set_num_threads=" << nt);
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

            TRACE_MSG("set_region_threads: omp_set_num_threads=" << nt);
            omp_set_num_threads(nt);
            return nt;
        }

        // Set number of threads for a block.
        virtual int set_block_threads() {

            // This should be a nested OMP region.
            int nt = _opts->num_block_threads;
            nt = std::max(nt, 1);
            TRACE_MSG("set_block_threads: omp_set_num_threads=" << nt);
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
                                 idx_t start_dw, idx_t start_dx, idx_t start_dy, idx_t start_dz,
                                 idx_t stop_dw, idx_t stop_dx, idx_t stop_dy, idx_t stop_dz);

        // Exchange halo data needed by eq-group 'eg' at the given time.
        virtual void exchange_halos(idx_t start_dt, idx_t stop_dt, EqGroupBase& eg);

        // Mark grids that have been written to by eq-group 'eg'.
        virtual void mark_grids_dirty(EqGroupBase& eg);
        
        // Set the bounding-box around all eq groups.
        virtual void find_bounding_boxes();

    };
    
    /// Classes that support evaluation of one stencil equation-group.
    /// A context contains of one or more equation-groups.

    // Types of dependencies.
    enum DepType {
        certain_dep,
        possible_dep,
        num_deps
    };

    // A pure-virtual class base for a stencil equation-set.
    class EqGroupBase : public BoundingBox {
    protected:
        StencilContext* _generic_context;
        
    public:

        // Grids that are written to by these stencil equations.
        GridPtrs outputGridPtrs;

        // Grids that are read by these stencil equations.
        GridPtrs inputGridPtrs;

        // ctor, dtor.
        EqGroupBase(StencilContext* context) :
            _generic_context(context) { }
        virtual ~EqGroupBase() { }

        // Get name of this equation set.
        virtual const std::string& get_name() const =0;

        // Get estimated number of FP ops done for one scalar eval.
        virtual int get_scalar_fp_ops() const =0;

        // Get number of points updated for one scalar eval.
        virtual int get_scalar_points_updated() const =0;

        // Set the bounding-box vars for this eq group in this rank.
        virtual void find_bounding_box();

        // Determine whether indices are in [sub-]domain.
        virtual bool is_in_valid_domain(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z) =0;

        // Calculate one scalar result at time t.
        virtual void calc_scalar(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z) =0;

        // Calculate one block of results from begin to end-1 on each dimension.
        virtual void calc_block(idx_t bt,
                                idx_t begin_bw, idx_t begin_bx, idx_t begin_by, idx_t begin_bz,
                                idx_t end_bw, idx_t end_bx, idx_t end_by, idx_t end_bz) =0;
    };

    // Define a method named cfn to prefetch a cluster by calling vfn.
#define PREFETCH_CLUSTER_METHOD(cfn, vfn)                               \
    template<int level>                                                 \
    ALWAYS_INLINE void                                                  \
    cfn (idx_t ct,                                                      \
         idx_t begin_cwv, idx_t begin_cxv, idx_t begin_cyv, idx_t begin_czv, \
         idx_t end_cwv, idx_t end_cxv, idx_t end_cyv, idx_t end_czv) {  \
        TRACE_MSG2(get_name() << "." #cfn "<" << level << ">("          \
                  "t=" << ct <<                                         \
                  ", wv=" << begin_cwv <<                               \
                  ", xv=" << begin_cxv <<                               \
                  ", yv=" << begin_cyv <<                               \
                  ", zv=" << begin_czv << ")");                         \
        _eqGroup.vfn<level>(*_context, ct,                              \
                     ARG_W(begin_cwv) begin_cxv, begin_cyv, begin_czv); \
    }

    // A template that provides wrappers around a stencil-equation class
    // created by the foldBuilder. A template is used instead of inheritance
    // for performance.  By using templates, the compiler can inline stencil
    // code into loops and avoid indirect calls.
    template <typename EqGroupClass, typename ContextClass>
    class EqGroupTemplate : public EqGroupBase {

    protected:

        // Pointer to a more specific context.
        ContextClass* _context;
        
        // EqGroupClass must implement calc_scalar(), calc_cluster(),
        // etc., that are used below.
        // This class is generated by the foldBuilder.
        EqGroupClass _eqGroup;

        // Eq-groups that this one depends on.
        std::map<DepType, EqGroupSet> _depends_on;
        
    public:

        // Ctor.
        EqGroupTemplate(ContextClass* context) :
            EqGroupBase(context),
            _context(context),
            _eqGroup(*_context, outputGridPtrs, inputGridPtrs)
        {
            assert(_generic_context);
            assert(_context);

            // Make sure map entries exist.
            for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1)) {
                _depends_on[dt];
            }
        }
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

        // Add dependency.
        virtual void add_dep(DepType dt, EqGroupBase* eg) {
            _depends_on.at(dt).insert(eg);
        }

        // Get dependencies.
        virtual const EqGroupSet& get_deps(DepType dt) const {
            return _depends_on.at(dt);
        }
    
        // Determine whether indices are in [sub-]domain for this eq group.
        virtual bool is_in_valid_domain(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z) {
            return _eqGroup.is_in_valid_domain(*_context, t,
                                               ARG_W(w) x, y, z);
        }

        // Calculate one scalar result.
        // This function implements the interface in the base class.
        virtual void calc_scalar(idx_t t, idx_t w, idx_t x, idx_t y, idx_t z) {
            TRACE_MSG2(get_name() << ".calc_scalar(t=" << t <<
                       ", w=" << w << ", x=" << x <<
                       ", y=" << y << ", z=" << z << ")");
            _eqGroup.calc_scalar(*_context, t, ARG_W(w) x, y, z);
        }

        // Calculate results within a cluster of vectors.
        // Called from calc_block().
        // The begin/end_c* vars are the start/stop_b* vars from the block loops.
        ALWAYS_INLINE void
        calc_cluster(idx_t ct,
                     idx_t begin_cwv, idx_t begin_cxv, idx_t begin_cyv, idx_t begin_czv,
                     idx_t end_cwv, idx_t end_cxv, idx_t end_cyv, idx_t end_czv)
        {
            TRACE_MSG2("calc_cluster(t=" << ct <<
                      ", wv=" << begin_cwv << ".." << (end_cwv-1) <<
                      ", xv=" << begin_cxv << ".." << (end_cxv-1) <<
                      ", yv=" << begin_cyv << ".." << (end_cyv-1) <<
                      ", zv=" << begin_czv << ".." << (end_czv-1) <<
                      ")");

            // The step vars are hard-coded in calc_block below, and there should
            // never be a partial step at this level. So, we can assume one var and
            // exactly CLEN_d steps in each given direction d are calculated in this
            // function.  Thus, we can ignore the end_* vars in the calc function.
            assert(end_cwv == begin_cwv + CLEN_W);
            assert(end_cxv == begin_cxv + CLEN_X);
            assert(end_cyv == begin_cyv + CLEN_Y);
            assert(end_czv == begin_czv + CLEN_Z);
        
            // Calculate results.
            _eqGroup.calc_cluster(*_context, ct, ARG_W(begin_cwv) begin_cxv, begin_cyv, begin_czv);
        }

        // Prefetch a cluster.
        // Separate methods for full cluster and each direction.
        // TODO: handle pre-fetching correctly for non-simple BBs.
        PREFETCH_CLUSTER_METHOD(prefetch_cluster, prefetch_cluster)
#if USING_DIM_W
        PREFETCH_CLUSTER_METHOD(prefetch_cluster_bwv, prefetch_cluster_w)
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
                   idx_t begin_bw, idx_t begin_bx, idx_t begin_by, idx_t begin_bz,
                   idx_t end_bw, idx_t end_bx, idx_t end_by, idx_t end_bz)
        {
            TRACE_MSG2(get_name() << ".calc_block(t=" << bt <<
                       ", w=" << begin_bw << ".." << (end_bw-1) <<
                       ", x=" << begin_bx << ".." << (end_bx-1) <<
                       ", y=" << begin_by << ".." << (end_by-1) <<
                       ", z=" << begin_bz << ".." << (end_bz-1) <<
                       ").");

            // If not a 'simple' domain, must use scalar code.  TODO: this
            // is very inefficient--need to vectorize as much as possible.
            if (!bb_simple) {

                TRACE_MSG2("...using scalar code.");
                for (idx_t w = begin_bw; w < end_bw; w++)
                    for (idx_t x = begin_bx; x < end_bx; x++)
                        for (idx_t y = begin_by; y < end_by; y++) {

                            // Are there holes in the BB?
                            if (bb_num_points != bb_size) {
                                for (idx_t z = begin_bz; z < end_bz; z++) {

                                    // Update only if point is in sub-domain for this eq group.
                                    if (is_in_valid_domain(bt, w, x, y, z))
                                        calc_scalar(bt, w, x, y, z);
                                }
                            }

                            // If no holes, don't need to check domain.
                            else {
                                for (idx_t z = begin_bz; z < end_bz; z++) {
                                    calc_scalar(bt, w, x, y, z);
                                }
                            }
                }
                
                return;
            }

            // Divide indices by vector lengths.  Use idiv_flr() instead of '/'
            // because begin/end vars may be negative (if in halo).
            const idx_t begin_bwv = idiv_flr<idx_t>(begin_bw, VLEN_W);
            const idx_t begin_bxv = idiv_flr<idx_t>(begin_bx, VLEN_X);
            const idx_t begin_byv = idiv_flr<idx_t>(begin_by, VLEN_Y);
            const idx_t begin_bzv = idiv_flr<idx_t>(begin_bz, VLEN_Z);
            const idx_t end_bwv = idiv_flr<idx_t>(end_bw, VLEN_W);
            const idx_t end_bxv = idiv_flr<idx_t>(end_bx, VLEN_X);
            const idx_t end_byv = idiv_flr<idx_t>(end_by, VLEN_Y);
            const idx_t end_bzv = idiv_flr<idx_t>(end_bz, VLEN_Z);

            // Vector-size steps are based on cluster lengths.
            // Using CLEN_* instead of CPTS_* because we want multiples of vector lengths.
            const idx_t step_bwv = CLEN_W;
            const idx_t step_bxv = CLEN_X;
            const idx_t step_byv = CLEN_Y;
            const idx_t step_bzv = CLEN_Z;

            // Groups in block loops are set to smallest size.
            const idx_t group_size_bwv = 1;
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
