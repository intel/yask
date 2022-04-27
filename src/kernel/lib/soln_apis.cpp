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

// This file contains implementations of StencilContext methods for API calls.
// Also see setup.cpp and context.cpp.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // APIs.
    // See yask_kernel_api.hpp.

    #define GET_SOLN_API(api_name, expr, step_ok, domain_ok, misc_ok, prep_req) \
        idx_t StencilContext::api_name(const string& dim) const {       \
            STATE_VARS(this);                                           \
            if (prep_req && !is_prepared())                             \
                THROW_YASK_EXCEPTION("Error: '" #api_name               \
                                     "()' called before calling 'prepare_solution()'"); \
            dims->check_dim_type(dim, #api_name, step_ok, domain_ok, misc_ok); \
            return expr;                                                \
        }
    GET_SOLN_API(get_num_ranks,
                 actl_opts->_num_ranks[dim],
                 false, true, false, false)
    GET_SOLN_API(get_overall_domain_size,
                 actl_opts->_global_sizes[dim],
                 false, true, false, true)
    GET_SOLN_API(get_rank_domain_size,
                 actl_opts->_rank_sizes[dim],
                 false, true, false, false)
    GET_SOLN_API(get_mega_block_size,
                 actl_opts->_mega_block_sizes[dim],
                 true, true, false, false)
    GET_SOLN_API(get_block_size,
                 actl_opts->_block_sizes[dim],
                 true, true, false, false)
    GET_SOLN_API(get_min_pad_size,
                 actl_opts->_min_pad_sizes[dim],
                 false, true, false, false)
    GET_SOLN_API(get_rank_index,
                 actl_opts->_rank_indices[dim],
                 false, true, false, true)

    GET_SOLN_API(get_first_rank_domain_index,
                 rank_bb.bb_begin_tuple(domain_dims)[dim],
                 false, true, false, true)
    GET_SOLN_API(get_last_rank_domain_index,
                 rank_bb.bb_end_tuple(domain_dims)[dim] - 1,
                 false, true, false, true)
    #undef GET_SOLN_API

    // The var sizes are updated any time these settings are changed.
    #define SET_SOLN_API(api_name, expr, step_ok, domain_ok, misc_ok, reset_prep) \
        void StencilContext::api_name(const string& dim, idx_t n) {     \
            STATE_VARS(this);                                           \
            TRACE_MSG("solution '" << get_name() << "'."                \
                      #api_name "('" << dim << "', " << n << ")");      \
            dims->check_dim_type(dim, #api_name, step_ok, domain_ok, misc_ok); \
            expr;                                                       \
            update_var_info(false);                                     \
            if (reset_prep) set_prepared(false);                        \
        }
    SET_SOLN_API(set_rank_index,
                 actl_opts->_rank_indices[dim] = n;
                 req_opts->_rank_indices[dim] = n;
                 actl_opts->find_loc = false;
                 req_opts->find_loc = false,
                 false, true, false, true)
    SET_SOLN_API(set_num_ranks,
                 actl_opts->_num_ranks[dim] = n;
                 req_opts->_num_ranks[dim] = n,
                 false, true, false, true)
    SET_SOLN_API(set_overall_domain_size,
                 actl_opts->_global_sizes[dim] = n;
                 req_opts->_global_sizes[dim] = n;
                 if (n) { actl_opts->_rank_sizes[dim] = 0;
                     req_opts->_rank_sizes[dim] = 0; },
                 false, true, false, true)
    SET_SOLN_API(set_rank_domain_size,
                 actl_opts->_rank_sizes[dim] = n;
                 req_opts->_rank_sizes[dim] = n;
                 if (n) { actl_opts->_global_sizes[dim] = 0;
                     req_opts->_global_sizes[dim] = 0; },
                 false, true, false, true)
    SET_SOLN_API(set_mega_block_size,
                 actl_opts->_mega_block_sizes[dim] = n;
                 req_opts->_mega_block_sizes[dim] = n,
                 true, true, false, true)
    SET_SOLN_API(set_block_size,
                 actl_opts->_block_sizes[dim] = n;
                 req_opts->_block_sizes[dim] = n,
                 true, true, false, true)
    SET_SOLN_API(set_min_pad_size,
                 actl_opts->_min_pad_sizes[dim] = n;
                 req_opts->_min_pad_sizes[dim] = n,
                 false, true, false, false)
    #undef SET_SOLN_API

    // Callbacks.
    void StencilContext::call_hooks(hook_fn_vec& hook_fns) {
        STATE_VARS(this);
        int n=0;
        for (auto& cb : hook_fns) {
            TRACE_MSG("Calling hook " << (++n) << "...");
            cb(*this);
        }
    }
    void StencilContext::call_2idx_hooks(hook_fn_2idx_vec& hook_fns,
                                         idx_t first_step_index,
                                         idx_t last_step_index) {
        STATE_VARS(this);
        int n=0;
        for (auto& cb : hook_fns) {
            TRACE_MSG("Calling hook " << (++n) << "...");
            cb(*this, first_step_index, last_step_index);
        }
    }

    // Allocate vars and MPI bufs.
    // Initialize some data structures.
    void StencilContext::prepare_solution() {
        STATE_VARS(this);
        TRACE_MSG("Calling prepare_solution()...");

        // User-provided code.
        call_hooks(_before_prepare_solution_hooks);

        // Don't continue until all ranks are this far.
        env->global_barrier();

        // reset time keepers.
        clear_timers();

        // Init auto-tuner to run silently during normal operation.
        reset_auto_tuner(true, false);

        // Report ranks.
        DEBUG_MSG("\nNum MPI ranks:             " << env->get_num_ranks() <<
                  "\nThis MPI rank index:       " << env->get_rank_index());

        // report threads.
        {
            DEBUG_MSG("Num OpenMP procs:          " << omp_get_num_procs());
            int rt, bt;
            int at = get_num_comp_threads(rt, bt);
            DEBUG_MSG("Num OpenMP threads avail:  " << actl_opts->max_threads <<
                      "\nNum OpenMP threads used:   " << at <<
                      "\n  Num outer threads:       " << rt <<
                      "\n  Num inner threads:       " << bt);
            #ifdef USE_OFFLOAD
            DEBUG_MSG("Num OpenMP devices:        " << omp_get_num_devices() <<
                      "\nOpenMP host device:        " << KernelEnv::_omp_hostn <<
                      "\nOpenMP offload device:     " << KernelEnv::_omp_devn <<
                      "\ndevice thread limit:       " << actl_opts->thread_limit);
            #endif
        }

        // Set the number of threads for a region. The number of threads
        // used in top-level OpenMP parallel sections should not change
        // during execution.
        int rthreads = set_num_outer_threads();

        // Run a dummy nested OMP loop to make sure nested threading is
        // initialized.
        yask_parallel_for(0, rthreads * 10, 1,
                          [&](idx_t start, idx_t stop, idx_t thread_num) { });

        // Some var stats.
        DEBUG_MSG("\nNum vars: " << all_var_ptrs.size() <<
                  "\nNum vars to be updated: " << output_var_ptrs.size() <<
                  "\nNum vars created via APIs: " << (all_var_ptrs.size() - orig_var_ptrs.size()));

        // Set up data based on MPI rank, including local or global sizes,
        // var positions.
        setup_rank();

        // Adjust all settings before setting MPI buffers or sizing vars.
        // Prints adjusted settings.
        actl_opts->adjust_settings(this);

        // Set offsets in vars and find WF extensions
        // based on the vars' halos. Force setting
        // the size of all solution vars.
        update_var_info(true);

        // Set core data needed in kernels.
        set_core();
        
        // Determine bounding-boxes for all bundles.  This must be done
        // after finding WF extensions.  And, this must be done after
        // set_core() because is_in_valid_domain() needs the core data.
        find_bounding_boxes();

        // Copy current settings to stages.  Needed here because settings may
        // have been changed via APIs or from call to setup_rank() since last
        // call to prepare_solution().  FIXME: This will wipe out any previous
        // stage-specific auto-tuning.
        for (auto& sp : st_stages)
            sp->get_local_settings() = *actl_opts;

        // Free the scratch and MPI data first to give vars preference.
        // Alloc vars (if needed), scratch vars, MPI bufs.
        // This is the order in which preferred NUMA nodes (e.g., HBW mem)
        // will be used.
        YaskTimer alloc_timer;
        alloc_timer.start();
        free_scratch_data();
        free_mpi_data();
        alloc_var_data(); // Does nothing if already done.
        alloc_scratch_data();
        alloc_mpi_data();
        alloc_timer.stop();
        DEBUG_MSG("Allocation done in " <<
                  make_num_str(alloc_timer.get_elapsed_secs()) << " secs.");

        init_stats();

        // User-provided code.
        call_hooks(_after_prepare_solution_hooks);

    } // prepare_solution().

    // Dealloc vars, etc.
    void StencilContext::end_solution() {
        STATE_VARS(this);
        TRACE_MSG("end_solution()...");

        // Release any MPI data.
        env->global_barrier();
        mpi_data.clear();

        // Release var data.
        for (auto gp : all_var_ptrs) {
            if (!gp)
                continue;
            gp->release_storage();
        }

	// Reset threads to original value.
	set_max_threads();
    }

    void StencilContext::fuse_vars(yk_solution_ptr source) {
        auto sp = dynamic_pointer_cast<StencilContext>(source);
        assert(sp);

        for (auto gp : all_var_ptrs) {
            auto gname = gp->get_name();
            auto si = sp->all_var_map.find(gname);
            if (si != sp->all_var_map.end()) {
                auto sgp = si->second;
                gp->fuse_vars(sgp);
            }
        }
    }

    string StencilContext::apply_command_line_options(const string& argstr) {
        auto args = CommandLineParser::set_args(argstr);
        return apply_command_line_options(args);
    }

    string StencilContext::apply_command_line_options(int argc, char* argv[]) {
        std::vector<std::string> args;
        for (int i = 1; i < argc; i++)
            args.push_back(argv[i]);
        return apply_command_line_options(args);
    }

    string StencilContext::apply_command_line_options(const vector<string>& args) {
        STATE_VARS(this);
        string rem;

        // Apply settings to actual and requested options.
        for (auto& cur_opts : { actl_opts, req_opts }) {
        
            // Create a parser and add base options to it.
            CommandLineParser parser;
            cur_opts->add_options(parser);

            // Parse cmd-line options, which sets values in opts.
            rem = parser.parse_args("YASK", args);
        }

        return rem;
    }

    static string print_pct(double ntime, double dtime) {
        string msg;
        if (dtime > 0.) {
            float pct = 100. * ntime / dtime;
            msg = " (" + to_string(pct) + "%)";
        }
        return msg;
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

        // These are part of 'halo_time'.
        double hwtime = min(halo_wait_time.get_elapsed_secs(), hetime);
        double hptime = min(halo_pack_time.get_elapsed_secs(), hetime - hwtime);
        double hutime = min(halo_unpack_time.get_elapsed_secs(), hetime - hwtime - hptime);
        double hctime = min(halo_copy_time.get_elapsed_secs(), hetime - hwtime - hptime - hutime);

        // Exterior and interior parts. Measured outside parallel region.
        // Does not include 'halo_time'.
        double etime = min(ext_time.get_elapsed_secs(), rtime - hetime);
        double itime = int_time.get_elapsed_secs();

        // 'test_time' is part of 'int_time', but only on outer thread 0.
        // It's not part of 'halo_time', since it's done outside of 'halo_exchange'.
        // We calculate an average to distribute this time across threads,
        // although it's just an estimate.
        double ttime = halo_test_time.get_elapsed_secs() / rthr; // ave.

        // Remove average test time from interior time.
        itime -= ttime;
        itime = min(itime, rtime - hetime - etime);

        // Compute time.
        double ctime = etime + itime;

        // All halo-related time.
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

        // Sum work done across stages using per-stage step counters.
        double tptime = 0.;
        double optime = 0.;
        idx_t psteps = 0;
        for (auto& sp : st_stages) {

            // steps in this stage.
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

            // Adjust stage time to make sure total time is <= compute time.
            double ptime = sp->timer.get_elapsed_secs();
            ptime = min(ptime, ctime - tptime);
            tptime += ptime;
            ps.run_time = ptime;
            ps.halo_time = 0.;

            // Stage rates.
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
            DEBUG_MSG("\nWork stats:\n"
                      " num-steps-done:                   " << make_num_str(steps_done) << endl <<
                      " num-reads-per-step:               " << make_num_str(double(p->nreads) / steps_done) << endl <<
                      " num-writes-per-step:              " << make_num_str(double(p->nwrites) / steps_done) << endl <<
                      " num-est-FP-ops-per-step:          " << make_num_str(double(p->nfpops) / steps_done) << endl <<
                      " num-points-per-step:              " << make_num_str(tot_domain_pts));
            if (psteps != steps_done) {
                DEBUG_MSG(" Work breakdown by stage(s):");
                for (auto& sp : st_stages) {
                    idx_t ns = sp->steps_done;
                    idx_t nreads = sp->tot_reads_per_step;
                    idx_t nwrites = sp->tot_writes_per_step;
                    idx_t nfpops = sp->tot_fpops_per_step;
                    string pfx = "  '" + sp->get_name() + "' ";
                    DEBUG_MSG(pfx << "num-steps-done:           " << ns << endl <<
                              pfx << "num-reads-per-step:       " << make_num_str(nreads) << endl <<
                              pfx << "num-writes-per-step:      " << make_num_str(nwrites) << endl <<
                              pfx << "num-est-FP-ops-per-step:  " << make_num_str(nfpops));
                }
            }
            DEBUG_MSG("\nTime stats:\n"
                      " elapsed-time (sec):               " << make_num_str(rtime) << endl <<
                      " Time breakdown by activity type:\n"
                      "  compute time (sec):                " << make_num_str(ctime) <<
                      print_pct(ctime, rtime) << endl <<
                      "  halo exchange time (sec):          " << make_num_str(htime) <<
                      print_pct(htime, rtime) << endl <<
                      "  other time (sec):                  " << make_num_str(otime) <<
                      print_pct(otime, rtime));
            if (psteps != steps_done) {
                DEBUG_MSG(" Compute-time breakdown by stage(s):");
                for (auto& sp : st_stages) {
                    auto& ps = sp->stats;
                    double ptime = ps.run_time;
                    string pfx = "  '" + sp->get_name() + "' ";
                    DEBUG_MSG(pfx << "time (sec):       " << make_num_str(ptime) <<
                              print_pct(ptime, ctime));
                }
                DEBUG_MSG("  other (sec):                       " << make_num_str(optime) <<
                          print_pct(optime, ctime));
            }

            #ifdef USE_MPI
            if (etime > 0.0)
                DEBUG_MSG(" Compute-time breakdown by halo area:\n"
                          "  rank-exterior compute (sec):         " << make_num_str(etime) <<
                          print_pct(etime, ctime) << endl <<
                          "  rank-interior compute (sec):         " << make_num_str(itime) <<
                          print_pct(itime, ctime));
            double hotime = max(hetime - hwtime - hptime - hutime - hctime, 0.);
            DEBUG_MSG(" Halo-time breakdown:\n"
                      "  MPI waits (sec):                     " << make_num_str(hwtime) <<
                      print_pct(hwtime, htime) << endl <<
                      "  MPI tests (sec):                     " << make_num_str(ttime) <<
                      print_pct(ttime, htime) << endl <<
                      "  buffer packing (sec):                " << make_num_str(hptime) <<
                      print_pct(hptime, htime) << endl <<
                      "  buffer unpacking (sec):              " << make_num_str(hutime) <<
                      print_pct(hutime, htime) << endl <<
                      #ifdef USE_OFFLOAD
                      "  buffer copying to/from device (sec): " << make_num_str(hctime) <<
                      print_pct(hctime, htime) << endl <<
                      #endif
                      "  other halo time (sec):               " << make_num_str(hotime) <<
                      print_pct(hotime, htime));
            #endif

            // Note that rates are reported with base-10 suffixes per common convention, not base-2.
            // See https://www.speedguide.net/articles/bits-bytes-and-bandwidth-reference-guide-115.
            DEBUG_MSG("\nRate stats:\n"
                      " throughput (num-reads/sec):       " << make_num_str(p->reads_ps) << endl <<
                      " throughput (num-writes/sec):      " << make_num_str(p->writes_ps) << endl <<
                      " throughput (est-FLOPS):           " << make_num_str(p->flops) << endl <<
                      " throughput (num-points/sec):      " << make_num_str(p->pts_ps));
            if (psteps != steps_done) {
                DEBUG_MSG(" Rate breakdown by stage(s):");
                for (auto& sp : st_stages) {
                    auto& ps = sp->stats;
                    string pfx = "  '" + sp->get_name() + "' ";
                    DEBUG_MSG(pfx << "throughput (num-reads/sec):   " << make_num_str(ps.reads_ps) << endl <<
                              pfx << "throughput (num-writes/sec):  " << make_num_str(ps.writes_ps) << endl <<
                              pfx << "throughput (est-FLOPS):       " << make_num_str(ps.flops) << endl <<
                              pfx << "throughput (num-points/sec):  " << make_num_str(ps.pts_ps));

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
        halo_pack_time.clear();
        halo_unpack_time.clear();
        halo_copy_time.clear();
        halo_wait_time.clear();
        halo_test_time.clear();
        steps_done = 0;
        for (auto& sp : st_stages) {
            sp->timer.clear();
            sp->steps_done = 0;
        }
    }

} // namespace yask.
