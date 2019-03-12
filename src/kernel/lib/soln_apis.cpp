/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2019, Intel Corporation

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

// This file contains implementations of StencilContext methods for API calls.
// Also see setup.cpp and context.cpp.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // APIs.
    // See yask_kernel_api.hpp.

#define GET_SOLN_API(api_name, expr, step_ok, domain_ok, misc_ok, prep_req) \
    idx_t StencilContext::api_name(const string& dim) const {           \
        STATE_VARS(this);                                               \
        if (prep_req && !rank_bb.bb_valid)                              \
            THROW_YASK_EXCEPTION("Error: '" #api_name \
                                 "()' called before calling 'prepare_solution()'"); \
        dims->checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok); \
        return expr;                                                    \
    }
    GET_SOLN_API(get_num_ranks, opts->_num_ranks[dim], false, true, false, false)
    GET_SOLN_API(get_overall_domain_size, opts->_global_sizes[dim], false, true, false, true)
    GET_SOLN_API(get_rank_domain_size, opts->_rank_sizes[dim], false, true, false, false)
    GET_SOLN_API(get_region_size, opts->_region_sizes[dim], true, true, false, false)
    GET_SOLN_API(get_block_size, opts->_block_sizes[dim], true, true, false, false)
    GET_SOLN_API(get_first_rank_domain_index, rank_bb.bb_begin[dim], false, true, false, true)
    GET_SOLN_API(get_last_rank_domain_index, rank_bb.bb_end[dim] - 1, false, true, false, true)
    GET_SOLN_API(get_min_pad_size, opts->_min_pad_sizes[dim], false, true, false, false)
    GET_SOLN_API(get_rank_index, opts->_rank_indices[dim], false, true, false, true)
#undef GET_SOLN_API

    // The grid sizes updated any time these settings are changed.
#define SET_SOLN_API(api_name, expr, step_ok, domain_ok, misc_ok, reset_prep) \
    void StencilContext::api_name(const string& dim, idx_t n) {         \
        STATE_VARS(this);                                               \
        dims->checkDimType(dim, #api_name, step_ok, domain_ok, misc_ok); \
        expr;                                                           \
        update_grid_info();                                             \
        if (reset_prep) rank_bb.bb_valid = ext_bb.bb_valid = false;     \
    }
    SET_SOLN_API(set_rank_index, opts->_rank_indices[dim] = n;
                 opts->find_loc = false, false, true, false, true)
    SET_SOLN_API(set_num_ranks, opts->_num_ranks[dim] = n, false, true, false, true)
    SET_SOLN_API(set_overall_domain_size, opts->_global_sizes[dim] = n;
                 if (n) opts->_rank_sizes[dim] = 0, false, true, false, true)
    SET_SOLN_API(set_rank_domain_size, opts->_rank_sizes[dim] = n;
                 if (n) opts->_global_sizes[dim] = 0, false, true, false, true)
    SET_SOLN_API(set_region_size, opts->_region_sizes[dim] = n, true, true, false, true)
    SET_SOLN_API(set_block_size, opts->_block_sizes[dim] = n, true, true, false, true)
    SET_SOLN_API(set_min_pad_size, opts->_min_pad_sizes[dim] = n, false, true, false, false)
#undef SET_SOLN_API

    // Allocate grids and MPI bufs.
    // Initialize some data structures.
    void StencilContext::prepare_solution() {
        STATE_VARS(this);

        // Don't continue until all ranks are this far.
        env->global_barrier();

#ifdef CHECK
        os << "*** WARNING: YASK compiled with CHECK; ignore performance results.\n";
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
        TRACE_MSG(" WARNING: YASK run with -trace; ignore performance results");

        // reset time keepers.
        clear_timers();

        // Init auto-tuner to run silently during normal operation.
        reset_auto_tuner(true, false);

        // Report ranks.
        os << endl;
        os << "Num MPI ranks:            " << env->get_num_ranks() << endl;
        os << "This MPI rank index:      " << env->get_rank_index() << endl;

        // report threads.
        {
            os << "Num OpenMP procs:         " << omp_get_num_procs() << endl;
            int rt, bt;
            int at = get_num_comp_threads(rt, bt);
            os << "Num OpenMP threads avail: " << opts->max_threads <<
                "\nNum OpenMP threads used:  " << at <<
                "\n  Num threads per region: " << rt <<
                "\n  Num threads per block:  " << bt << endl;
        }

        // Set the number of threads for a region. The number of threads
        // used in top-level OpenMP parallel sections should not change
        // during execution.
        int rthreads = set_region_threads();

        // Run a dummy nested OMP loop to make sure nested threading is
        // initialized.
        yask_for(0, rthreads * 100, 1,
                 [&](idx_t start, idx_t stop, idx_t thread_num) { });

        // Some grid stats.
        os << endl;
        os << "Num grids: " << gridPtrs.size() << endl;
        os << "Num grids to be updated: " << outputGridPtrs.size() << endl;

        // Set up data based on MPI rank, including local or global sizes,
        // grid positions.
        setupRank();

        // Adjust all settings before setting MPI buffers or sizing grids.
        // Prints adjusted settings.
        // TODO: print settings again after auto-tuning.
        opts->adjustSettings(os);

        // Set offsets in grids and find WF extensions
        // based on the grids' halos.
        update_grid_info();

        // Determine bounding-boxes for all bundles.
        // This must be done after finding WF extensions.
        find_bounding_boxes();

        // Copy current settings to packs.  Needed here because settings may
        // have been changed via APIs or from call to setupRank() since last
        // call to prepare_solution().  This will wipe out any previous
        // auto-tuning.
        for (auto& sp : stPacks)
            sp->getLocalSettings() = *opts;

        // Alloc grids, scratch grids, MPI bufs.
        // This is the order in which preferred NUMA nodes (e.g., HBW mem)
        // will be used.
        // We free the scratch and MPI data first to give grids preference.
        YaskTimer allocTimer;
        allocTimer.start();
        freeScratchData();
        freeMpiData();
        allocGridData();
        allocScratchData();
        allocMpiData();
        allocTimer.stop();
        os << "Allocation done in " <<
            makeNumStr(allocTimer.get_elapsed_secs()) << " secs.\n" << flush;

        print_info();

    } // prepare_solution().

    void StencilContext::print_temporal_tiling_info() {
        ostream& os = get_ostr();

        os <<
            " num-wave-front-steps:      " << wf_steps << endl;
        if (wf_steps > 0) {
            os <<
                " wave-front-angles:         " << wf_angles.makeDimValStr() << endl <<
                " num-wave-front-shifts:     " << num_wf_shifts << endl <<
                " wave-front-shift-amounts:  " << wf_shift_pts.makeDimValStr() << endl <<
                " left-wave-front-exts:      " << left_wf_exts.makeDimValStr() << endl <<
                " right-wave-front-exts:     " << right_wf_exts.makeDimValStr() << endl <<
                " ext-local-domain:          " << ext_bb.bb_begin.makeDimValStr() <<
                " ... " << ext_bb.bb_end.subElements(1).makeDimValStr() << endl <<
                " num-temporal-block-steps:  " << tb_steps << endl <<
                " temporal-block-angles:     " << tb_angles.makeDimValStr() << endl <<
                " num-temporal-block-shifts: " << num_tb_shifts << endl <<
                " temporal-block-long-base:  " << tb_widths.makeDimValStr(" * ") << endl <<
                " temporal-block-short-base: " << tb_tops.makeDimValStr(" * ") << endl <<
                " mini-block-angles:         " << mb_angles.makeDimValStr() << endl;
        }
    }
    
    void StencilContext::print_info() {
        STATE_VARS(this);

        // Calc and report total allocation and domain sizes.
        rank_nbytes = get_num_bytes();
        tot_nbytes = sumOverRanks(rank_nbytes, env->comm);
        rank_domain_pts = rank_bb.bb_num_points;
        tot_domain_pts = sumOverRanks(rank_domain_pts, env->comm);
        os <<
            "\nDomain size in this rank (points):          " << makeNumStr(rank_domain_pts) <<
            "\nTotal allocation in this rank:              " << makeByteStr(rank_nbytes) <<
            "\nOverall problem size in " << env->num_ranks << " rank(s) (points): " <<
            makeNumStr(tot_domain_pts) <<
            "\nTotal overall allocation in " << env->num_ranks << " rank(s):      " <<
            makeByteStr(tot_nbytes) <<
            endl;

        // Report some sizes and settings.
        os << "\nWork-unit sizes in points (from smallest to largest):\n"
            " vector-size:           " << dims->_fold_pts.makeDimValStr(" * ") << endl <<
            " cluster-size:          " << dims->_cluster_pts.makeDimValStr(" * ") << endl <<
            " sub-block-size:        " << opts->_sub_block_sizes.removeDim(step_posn).makeDimValStr(" * ") << endl <<
            " mini-block-size:       " << opts->_mini_block_sizes.makeDimValStr(" * ") << endl <<
            " block-size:            " << opts->_block_sizes.makeDimValStr(" * ") << endl <<
            " region-size:           " << opts->_region_sizes.makeDimValStr(" * ") << endl <<
            " local-domain-size:     " << opts->_rank_sizes.removeDim(step_posn).makeDimValStr(" * ") << endl <<
            " global-domain-size:    " << opts->_global_sizes.removeDim(step_posn).makeDimValStr(" * ") << endl;
#ifdef SHOW_GROUPS
        os << 
            " sub-block-group-size:  " << opts->_sub_block_group_sizes.makeDimValStr(" * ") << endl <<
            " block-group-size:      " << opts->_block_group_sizes.makeDimValStr(" * ") << endl <<
#endif
        os << "\nOther settings:\n"
            " yask-version:          " << yask_get_version_string() << endl <<
            " stencil-name:          " << get_name() << endl <<
            " stencil-description:   " << get_description() << endl <<
            " element-size:          " << makeByteStr(get_element_bytes()) << endl <<
            " local-domain:          " << rank_bb.bb_begin.makeDimValStr() <<
            " ... " << rank_bb.bb_end.subElements(1).makeDimValStr() << endl;
#ifdef USE_MPI
        os <<
            " num-ranks:             " << opts->_num_ranks.makeDimValStr(" * ") << endl <<
            " rank-indices:          " << opts->_rank_indices.makeDimValStr() << endl <<
            " local-domain-offsets:  " << rank_domain_offsets.makeDimValStr() << endl;
        if (opts->overlap_comms)
            os <<
                " mpi-interior:          " << mpi_interior.bb_begin.makeDimValStr() <<
                " ... " << mpi_interior.bb_end.subElements(1).makeDimValStr() << endl;
#endif
        os <<
            " vector-len:            " << VLEN << endl <<
            " extra-padding:         " << opts->_extra_pad_sizes.makeDimValStr() << endl <<
            " minimum-padding:       " << opts->_min_pad_sizes.makeDimValStr() << endl <<
            " L1-prefetch-distance:  " << PFD_L1 << endl <<
            " L2-prefetch-distance:  " << PFD_L2 << endl <<
            " max-halos:             " << max_halos.makeDimValStr() << endl;
        print_temporal_tiling_info();
        os << endl;

        // Info about eqs, packs and bundles.
        os << "Num stencil packs:      " << stPacks.size() << endl;
        os << "Num stencil bundles:    " << stBundles.size() << endl;
        os << "Num stencil equations:  " << NUM_STENCIL_EQS << endl;

        // Info on work in packs.
        os << "\nBreakdown of work stats in this rank:\n";
        for (auto& sp : stPacks)
            sp->init_work_stats();
    }

    // Dealloc grids, etc.
    void StencilContext::end_solution() {
        STATE_VARS(this);

        // Final halo exchange (usually not needed).
        exchange_halos();

        // Release any MPI data.
        env->global_barrier();
        mpiData.clear();

        // Release grid data.
        for (auto gp : gridPtrs) {
            if (!gp)
                continue;
            gp->release_storage();
        }

	// Reset threads to original value.
	set_max_threads();
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

    string StencilContext::apply_command_line_options(const string& args) {
        STATE_VARS(this);

        // Create a parser and add base options to it.
        CommandLineParser parser;
        opts->add_options(parser);

        // Tokenize default args.
        vector<string> argsv;
        parser.set_args(args, argsv);

        // Parse cmd-line options, which sets values in settings.
        parser.parse_args("YASK", argsv);

        // Return any left-over strings.
        string rem;
        for (auto r : argsv) {
            if (rem.length())
                rem += " ";
            rem += r;
        }
        return rem;
    }

    // Add a new grid to the containers.
    void StencilContext::addGrid(YkGridPtr gp, bool is_output) {
        STATE_VARS(this);
        assert(gp);
        auto& gname = gp->get_name();
        if (gridMap.count(gname))
            THROW_YASK_EXCEPTION("Error: grid '" + gname + "' already exists");

        // Add to list and map.
        gridPtrs.push_back(gp);
        gridMap[gname] = gp;

        // Add to output list and map if 'is_output'.
        if (is_output) {
            outputGridPtrs.push_back(gp);
            outputGridMap[gname] = gp;
        }
    }

    static void print_pct(ostream& os, double ntime, double dtime) {
        if (dtime > 0.) {
            float pct = 100. * ntime / dtime;
            os << " (" << pct << "%)";
        }
        os << endl;
    }

    /// Get statistics associated with preceding calls to run_solution().
    yk_stats_ptr StencilContext::get_stats() {
        STATE_VARS(this);

        // Numbers of threads.
        int rthr, bthr;
        int athr = get_num_comp_threads(rthr, bthr);

        // 'run_time' covers all of 'run_solution()' and subsumes
        // all other timers. Measured outside parallel region.
        double rtime = run_time.get_elapsed_secs();

        // 'halo_time' covers calls to 'exchange_halos()'.
        // Measured outside parallel region.
        double hetime = min(halo_time.get_elapsed_secs(), rtime);

        // 'wait_time' is part of 'halo_time'.
        double wtime = min(wait_time.get_elapsed_secs(), hetime);

        // Exterior and interior parts. Measured outside parallel region.
        // Does not include 'halo_time'.
        double etime = min(ext_time.get_elapsed_secs(), rtime - hetime);
        double itime = int_time.get_elapsed_secs();

        // 'test_time' is part of 'int_time', but only on region thread 0.
        // It's not part of 'halo_time'.
        double ttime = test_time.get_elapsed_secs() / rthr; // ave.

        // Remove average test time from interior time.
        itime -= ttime;
        itime = min(itime, rtime - hetime - etime);

        // Compute time.
        double ctime = etime + itime;

        // All halo time.
        double htime = hetime + ttime;

        // Other.
        double otime = max(rtime - ctime - htime, 0.0);

        // Init return object.
        auto p = make_shared<Stats>();
        p->npts = tot_domain_pts; // NOT sum over steps.
        p->nsteps = steps_done;
        p->run_time = rtime;
        p->halo_time = htime;
        p->nreads = 0;
        p->nwrites = 0;
        p->nfpops = 0;
        p->pts_ps = 0.;
        p->reads_ps = 0.;
        p->writes_ps = 0.;
        p->flops = 0.;

        // Sum work done across packs using per-pack step counters.
        double tptime = 0.;
        double optime = 0.;
        idx_t psteps = 0;
        for (auto& sp : stPacks) {

            // steps in this pack.
            idx_t ns = sp->steps_done;

            auto& ps = sp->stats;
            ps.nsteps = ns;
            ps.npts = tot_domain_pts; // NOT sum over steps.
            ps.nreads = sp->tot_reads_per_step * ns;
            ps.nwrites = sp->tot_writes_per_step * ns;
            ps.nfpops = sp->tot_fpops_per_step * ns;

            // Add to total work.
            psteps += ns;
            p->nreads += ps.nreads;
            p->nwrites += ps.nwrites;
            p->nfpops += ps.nfpops;

            // Adjust pack time to make sure total time is <= compute time.
            double ptime = sp->timer.get_elapsed_secs();
            ptime = min(ptime, ctime - tptime);
            tptime += ptime;
            ps.run_time = ptime;
            ps.halo_time = 0.;

            // Pack rates.
            idx_t np = tot_domain_pts * ns; // Sum over steps.
            ps.reads_ps = 0.;
            ps.writes_ps = 0.;
            ps.flops = 0.;
            ps.pts_ps = 0.;
            if (ptime > 0.) {
                ps.reads_ps = ps.nreads / ptime;
                ps.writes_ps = ps.nwrites / ptime;
                ps.flops = ps.nfpops / ptime;
                ps.pts_ps = np / ptime;
            }
        }
        optime = max(ctime - tptime, 0.); // remaining time.

        // Overall rates.
        idx_t npts_done = tot_domain_pts * steps_done;
        if (rtime > 0.) {
            p->reads_ps= double(p->nreads) / rtime;
            p->writes_ps= double(p->nwrites) / rtime;
            p->flops = double(p->nfpops) / rtime;
            p->pts_ps = double(npts_done) / rtime;
        }
      
        if (steps_done > 0) {
            os <<
                "\nWork stats:\n"
                " num-steps-done:                   " << makeNumStr(steps_done) << endl <<
                " num-reads-per-step:               " << makeNumStr(double(p->nreads) / steps_done) << endl <<
                " num-writes-per-step:              " << makeNumStr(double(p->nwrites) / steps_done) << endl <<
                " num-est-FP-ops-per-step:          " << makeNumStr(double(p->nfpops) / steps_done) << endl <<
                " num-points-per-step:              " << makeNumStr(tot_domain_pts) << endl;
            if (psteps != steps_done) {
                os <<
                    " Work breakdown by stencil pack(s):\n";
                for (auto& sp : stPacks) {
                    idx_t ns = sp->steps_done;
                    idx_t nreads = sp->tot_reads_per_step;
                    idx_t nwrites = sp->tot_writes_per_step;
                    idx_t nfpops = sp->tot_fpops_per_step;
                    string pfx = "  '" + sp->get_name() + "' ";
                    os << pfx << "num-steps-done:           " << ns << endl <<
                        pfx << "num-reads-per-step:       " << makeNumStr(nreads) << endl <<
                        pfx << "num-writes-per-step:      " << makeNumStr(nwrites) << endl <<
                        pfx << "num-est-FP-ops-per-step:  " << makeNumStr(nfpops) << endl;
                }
            }
            os << 
                "\nTime stats:\n"
                " elapsed-time (sec):               " << makeNumStr(rtime) << endl <<
                " Time breakdown by activity type:\n"
                "  compute time (sec):                " << makeNumStr(ctime);
            print_pct(os, ctime, rtime);
#ifdef USE_MPI
            os <<
                "  halo exchange time (sec):          " << makeNumStr(htime);
            print_pct(os, htime, rtime);
#endif
            os <<
                "  other time (sec):                  " << makeNumStr(otime);
            print_pct(os, otime, rtime);
            if (psteps != steps_done) {
                os <<
                    " Compute-time breakdown by stencil pack(s):\n";
                for (auto& sp : stPacks) {
                    auto& ps = sp->stats;
                    double ptime = ps.run_time;
                    string pfx = "  '" + sp->get_name() + "' ";
                    os << pfx << "time (sec):       " << makeNumStr(ptime);
                    print_pct(os, ptime, ctime);
                }
                os << "  other (sec):                       " << makeNumStr(optime);
                print_pct(os, optime, ctime);
            }
#ifdef USE_MPI
            os <<
                " Compute-time breakdown by halo area:\n"
                "  rank-exterior compute (sec):       " << makeNumStr(etime);
            print_pct(os, etime, ctime);
            os <<
                "  rank-interior compute (sec):       " << makeNumStr(itime);
            print_pct(os, itime, ctime);
            os <<
                " Halo-time breakdown:\n"
                "  MPI waits (sec):                   " << makeNumStr(wtime);
            print_pct(os, wtime, htime);
            os <<
                "  MPI tests (sec):                   " << makeNumStr(ttime);
            print_pct(os, ttime, htime);
            double ohtime = max(htime - wtime - ttime, 0.);
            os <<
                "  packing, unpacking, etc. (sec):    " << makeNumStr(ohtime);
            print_pct(os, ohtime, htime);
#endif

            // Note that rates are reported with base-10 suffixes per common convention, not base-2.
            // See https://www.speedguide.net/articles/bits-bytes-and-bandwidth-reference-guide-115.
            os <<
                "\nRate stats:\n"
                " throughput (num-reads/sec):       " << makeNumStr(p->reads_ps) << endl <<
                " throughput (num-writes/sec):      " << makeNumStr(p->writes_ps) << endl <<
                " throughput (est-FLOPS):           " << makeNumStr(p->flops) << endl <<
                " throughput (num-points/sec):      " << makeNumStr(p->pts_ps) << endl;
            if (psteps != steps_done) {
                os <<
                    " Rate breakdown by stencil pack(s):\n";
                for (auto& sp : stPacks) {
                    auto& ps = sp->stats;
                    string pfx = "  '" + sp->get_name() + "' ";
                    os <<
                        pfx << "throughput (num-reads/sec):   " << makeNumStr(ps.reads_ps) << endl <<
                        pfx << "throughput (num-writes/sec):  " << makeNumStr(ps.writes_ps) << endl <<
                        pfx << "throughput (est-FLOPS):       " << makeNumStr(ps.flops) << endl <<
                        pfx << "throughput (num-points/sec):  " << makeNumStr(ps.pts_ps) << endl;
                    
                }
            }
        }

        // Clear counters.
        clear_timers();

        return p;
    }

    // Reset elapsed times to zero.
    void StencilContext::clear_timers() {
        run_time.clear();
        ext_time.clear();
        int_time.clear();
        halo_time.clear();
        wait_time.clear();
        test_time.clear();
        steps_done = 0;
        for (auto& sp : stPacks) {
            sp->timer.clear();
            sp->steps_done = 0;
        }
    }
    
} // namespace yask.
