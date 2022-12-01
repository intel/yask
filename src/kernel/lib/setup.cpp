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

// This file contains implementations of configuration-related methods
// from several classes.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Init MPI, OMP.
    // This function can have some calls to TRACE_MSG(), but it shouldn't
    // call DEBUG_MSG(), because the user might not want debug messages
    // before setting up the env.
    void KernelEnv::init_env(int* argc, char*** argv, MPI_Comm existing_comm)
    {
        TRACE_MSG("Initializing YASK environment...");
        YaskTimer init_timer;
        init_timer.start();
         
        // MPI init.
        my_rank = 0;
        num_ranks = 1;

        #ifdef USE_MPI
        TRACE_MSG("Initializing MPI...");
        int is_init = false;
        MPI_Initialized(&is_init);

        // No MPI communicator provided.
        if (existing_comm == MPI_COMM_NULL ||
            existing_comm == MPI_COMM_WORLD) {
            if (!is_init) {

                int provided = 0;
                MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
                if (provided < MPI_THREAD_SERIALIZED) {
                    THROW_YASK_EXCEPTION("error: MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE not provided");
                }
                is_init = true;
                finalize_needed = true;
            }
            comm = MPI_COMM_WORLD;
        }

        // MPI communicator provided.
        else {
            if (!is_init)
                THROW_YASK_EXCEPTION("error: YASK environment created with"
                                     " an existing MPI communicator, but MPI is not initialized");
            comm = existing_comm;
        }

        // Get some info on this communicator.
        MPI_Comm_rank(comm, &my_rank);
        MPI_Comm_group(comm, &group);
        MPI_Comm_size(comm, &num_ranks);
        if (num_ranks < 1)
            THROW_YASK_EXCEPTION("error: MPI_Comm_size() returns less than one rank");

        // Create a shm communicator.
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
        MPI_Comm_rank(shm_comm, &my_shm_rank);
        MPI_Comm_group(shm_comm, &shm_group);
        MPI_Comm_size(shm_comm, &num_shm_ranks);

        #else
        comm = MPI_COMM_NULL;
        #endif

        // Turn off denormals unless the USE_DENORMALS macro is set.
        #ifndef USE_DENORMALS
        // Enable FTZ
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

        //Enable DAZ
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        #endif

        #ifdef _OPENMP
        TRACE_MSG("Initializing OpenMP...");

        // Set env vars needed by OMP.
        // TODO: make this visible to the user.
        int ret = setenv("OMP_PLACES", "cores", 0); // default placement for outer loop.
        assert(ret == 0);
        ret = setenv("KMP_HOT_TEAMS_MODE", "1", 0); // more efficient nesting.
        assert(ret == 0);
        ret = setenv("KMP_HOT_TEAMS_MAX_LEVEL", "3", 0); // nesting.

        // Check initial value of OMP max threads.
        // Side effect: causes OMP to dump debug info if env var set.
        int mt = omp_get_max_threads();
        if (!max_threads)
            max_threads = mt;

        #ifdef USE_OFFLOAD
        _omp_hostn = omp_get_initial_device();
        _omp_devn = omp_get_default_device();
        #endif

        #else
        max_threads = 1;
        #endif

        init_timer.stop();
        TRACE_MSG("Environment initialization done in " <<
                  make_num_str(init_timer.get_elapsed_secs()) << " secs.");
    }

    // Bootstrap factory ctor.
    yk_factory::yk_factory() {
    }

    // Context ctor.
    StencilContext::StencilContext(KernelEnvPtr& kenv,
                                   KernelSettingsPtr& actl_settings,
                                   KernelSettingsPtr& req_settings) :
        KernelStateBase(kenv, actl_settings, req_settings),
        _at(this, actl_settings.get())
    {
        STATE_VARS(this);

        // Init various tuples to make sure they have the correct dims.
        rank_domain_offsets.set_from_tuple(domain_dims);
        max_halos = domain_dims;
        wf_angles = domain_dims;
        wf_shift_pts = domain_dims;
        tb_angles = domain_dims;
        tb_widths = domain_dims;
        tb_tops = domain_dims;
        mb_angles = domain_dims;
        left_wf_exts = domain_dims;
        right_wf_exts = domain_dims;
    }

    // Init MPI-related vars and other vars related to my rank's place in
    // the global problem: rank index, offset, etc.  Need to call this even
    // if not using MPI to properly init these vars.  Called from
    // prepare_solution().
    void StencilContext::setup_rank() {
        STATE_VARS(this);
        TRACE_MSG("setup_rank()...");
        auto me = env->my_rank;
        auto nr = env->num_ranks;

        // All ranks should have the same settings for certain options.
        assert_equality_over_ranks(nr, env->comm, "total number of MPI ranks");
        assert_equality_over_ranks(idx_t(actl_opts->use_shm), env->comm, "use_shm setting");
        assert_equality_over_ranks(idx_t(actl_opts->find_loc), env->comm, "defined rank indices");
        DOMAIN_VAR_LOOP(i, j) {
            auto& dname = domain_dims.get_dim_name(j);
            assert_equality_over_ranks(actl_opts->_global_sizes[i], env->comm,
                                       "global-domain size in '" + dname + "' dimension");
            assert_equality_over_ranks(actl_opts->_num_ranks[j], env->comm,
                                       "number of ranks in '" + dname + "' dimension");

            // Check that either local or global size is set.
            if (!actl_opts->_global_sizes[i] && !actl_opts->_rank_sizes[i])
                THROW_YASK_EXCEPTION("Error: both local-domain size and "
                                     "global-domain size are zero in '" +
                                     dname + "' dimension on rank " +
                                     to_string(me) + "; specify one, "
                                     "and the other will be calculated");
        }

        #ifndef USE_MPI

        // Simple settings.
        actl_opts->_num_ranks.set_vals_same(1);
        actl_opts->_rank_indices.set_vals_same(0);
        rank_domain_offsets.set_vals_same(0);
        assert(nr == 1);

        // Init vars w/o MPI.
        DOMAIN_VAR_LOOP(i, j) {

            // Need to set local size.
            if (!actl_opts->_rank_sizes[i])
                actl_opts->_rank_sizes[i] = actl_opts->_global_sizes[i];

            // Need to set global size.
            else if (!actl_opts->_global_sizes[i])
                actl_opts->_global_sizes[i] = actl_opts->_rank_sizes[i];

            // Check that settings are equal.
            else if (actl_opts->_global_sizes[i] != actl_opts->_rank_sizes[i]) {
                auto& dname = domain_dims.get_dim_name(j);
                FORMAT_AND_THROW_YASK_EXCEPTION("Error: specified local-domain size of " <<
                                                actl_opts->_rank_sizes[i] <<
                                                " does not equal specified global-domain size of " <<
                                                actl_opts->_global_sizes[i] << " in '" << dname <<
                                                "' dimension");
            }
        }

        #else
        // Set number of ranks in each dim if any is unset (zero).
        TRACE_MSG("rank layout " << actl_opts->_num_ranks.make_dim_val_str(" * ") << " requested");
        actl_opts->_num_ranks = actl_opts->_num_ranks.get_compact_factors(nr);
        TRACE_MSG("rank layout " << actl_opts->_num_ranks.make_dim_val_str(" * ") << " selected");

        // Check ranks.
        idx_t req_ranks = actl_opts->_num_ranks.product();
        if (req_ranks != nr)
            FORMAT_AND_THROW_YASK_EXCEPTION("error: " << req_ranks << " rank(s) requested (" +
                                            actl_opts->_num_ranks.make_dim_val_str(" * ") + "), but " <<
                                            nr << " rank(s) are active");

        // Determine my coordinates if not provided already.
        // TODO: do this more intelligently based on proximity.
        if (actl_opts->find_loc)
            actl_opts->_rank_indices = actl_opts->_num_ranks.unlayout(me);

        // Check rank indices.
        DOMAIN_VAR_LOOP(i, j) {
            auto& dname = domain_dims.get_dim_name(j);
            if (actl_opts->_rank_indices[j] < 0 ||
                actl_opts->_rank_indices[j] >= actl_opts->_num_ranks[j])
                THROW_YASK_EXCEPTION("Error: rank index of " +
                                     to_string(actl_opts->_rank_indices[j]) +
                                     " is not within allowed range [0 ... " +
                                     to_string(actl_opts->_num_ranks[j] - 1) +
                                     "] in '" + dname + "' dimension on rank " +
                                     to_string(me));
        }

        // Init starting indices for this rank.
        rank_domain_offsets.set_vals_same(0);

        // Tables to share data across ranks.
        idx_t coords[nr][nddims]; // rank indices.
        idx_t rsizes[nr][nddims]; // rank sizes.

        // Two passes over ranks:
        // 0: sum all specified local sizes.
        // 1: set final sums and offsets.
        for (int pass : { 0, 1 }) {

            // Init rank-size sums.
            IdxTuple rank_domain_sums(domain_dims);
            rank_domain_sums.set_vals_same(0);

            // Init tables for this rank.
            DOMAIN_VAR_LOOP(i, j) {
                coords[me][j] = actl_opts->_rank_indices[j];
                rsizes[me][j] = actl_opts->_rank_sizes[i];
            }

            // Exchange coord and size info between all ranks.
            for (int rn = 0; rn < nr; rn++) {
                MPI_Bcast(&coords[rn][0], nddims, MPI_INTEGER8,
                          rn, env->comm);
                MPI_Bcast(&rsizes[rn][0], nddims, MPI_INTEGER8,
                          rn, env->comm);
            }
            // Now, the tables are filled in for all ranks.
            // Some rank sizes may be zero on the 1st pass,
            // but they should all be non-zero on 2nd pass.

            // Loop over all ranks, including myself.
            int num_neighbors = 0;
            for (int rn = 0; rn < nr; rn++) {

                // Coord offset of rn from me: prev => negative, self => 0, next => positive.
                IdxTuple rcoords(domain_dims);
                IdxTuple rdeltas(domain_dims);
                DOMAIN_VAR_LOOP(i, di) {
                    rcoords[di] = coords[rn][di];
                    rdeltas[di] = coords[rn][di] - coords[me][di];
                }

                // Manhattan distance from rn (sum of abs deltas in all dims).
                // Max distance in any dim.
                int mandist = 0;
                int maxdist = 0;
                DOMAIN_VAR_LOOP(i, di) {
                    mandist += abs(rdeltas[di]);
                    maxdist = max(maxdist, abs(int(rdeltas[di])));
                }

                // Myself.
                if (rn == me) {
                    if (mandist != 0)
                        FORMAT_AND_THROW_YASK_EXCEPTION
                            ("Internal error: distance to own rank == " << mandist);
                }

                // Someone else.
                else {
                    if (mandist == 0)
                        FORMAT_AND_THROW_YASK_EXCEPTION
                            ("Error: ranks " << me <<
                             " and " << rn << " at same coordinates");
                }

                // Loop through domain dims.
                DOMAIN_VAR_LOOP(i, di) {
                    auto& dname = domain_dims.get_dim_name(di);

                    // Is rank 'rn' in-line with my rank in 'dname' dim?
                    // True when deltas in all other dims are zero.
                    bool is_inline = true;
                    DOMAIN_VAR_LOOP(j, dj) {
                        if (di != dj && rdeltas[dj] != 0) {
                            is_inline = false;
                            break;
                        }
                    }

                    // Process this rank if it is in-line with me in 'dname', including myself.
                    if (is_inline) {

                        // Sum rank sizes in this dim.
                        rank_domain_sums[di] += rsizes[rn][di];

                        if (pass == 1) {

                            // Make sure all the other dims are the same size.
                            // This ensures that all the ranks' domains line up
                            // properly along their edges and at their corners.
                            DOMAIN_VAR_LOOP(j, dj) {
                                if (di != dj) {
                                    auto& dnamej = domain_dims.get_dim_name(dj);
                                    auto mysz = rsizes[me][dj];
                                    auto rnsz = rsizes[rn][dj];
                                    if (mysz != rnsz) {
                                        FORMAT_AND_THROW_YASK_EXCEPTION
                                            ("Error: rank " << rn << " and " << me <<
                                             " are both at rank-index " << coords[me][di] <<
                                             " in the '" << dname <<
                                             "' dimension, but their local-domain sizes are " <<
                                             rnsz << " and " << mysz <<
                                             " (resp.) in the '" << dnamej <<
                                             "' dimension, making them unaligned");
                                    }
                                }
                            }

                            // Adjust my offset in the global problem by adding all domain
                            // sizes from prev ranks only.
                            if (rdeltas[di] < 0)
                                rank_domain_offsets[di] += rsizes[rn][di];

                        } // 2nd pass.
                    } // is inline w/me.
                } // dims.

                // Rank rn is myself or my immediate neighbor if its distance <= 1 in
                // every dim.  Assume we do not need to exchange halos except
                // with immediate neighbor. We enforce this assumption below by
                // making sure that the rank domain size is at least as big as the
                // largest halo.
                if (pass == 1 && maxdist <= 1) {

                    // At this point, rdeltas contains only -1..+1 for each domain dim.
                    // Add one to -1..+1 to get 0..2 range for my_neighbors offsets.
                    IdxTuple roffsets = rdeltas.add_elements(1);
                    assert(rdeltas.min() >= -1);
                    assert(rdeltas.max() <= 1);
                    assert(roffsets.min() >= 0);
                    assert(roffsets.max() <= 2);

                    // Convert the offsets into a 1D index.
                    auto rn_ofs = mpi_info->get_neighbor_index(roffsets);
                    TRACE_MSG("neighborhood size = " << mpi_info->neighborhood_sizes.make_dim_val_str() <<
                              " & roffsets of rank " << rn << " = " << roffsets.make_dim_val_str() <<
                              " => " << rn_ofs);
                    assert(idx_t(rn_ofs) < mpi_info->neighborhood_size);

                    // Save rank of this neighbor into the MPI info object.
                    mpi_info->my_neighbors.at(rn_ofs) = rn;
                    if (rn == me) {
                        assert(mpi_info->my_neighbor_index == rn_ofs);
                        mpi_info->shm_ranks.at(rn_ofs) = env->my_shm_rank;
                    }
                    else {
                        num_neighbors++;
                        DEBUG_MSG("Neighbor #" << num_neighbors << " is MPI rank " << rn <<
                                  " at absolute rank indices " << rcoords.make_dim_val_str() <<
                                  " (" << rdeltas.make_dim_val_offset_str() << " relative to rank " <<
                                  me << ")");

                        // Determine whether neighbor is in my shm group.
                        // If so, record rank number in shmcomm.
                        if (actl_opts->use_shm && env->shm_comm != MPI_COMM_NULL) {
                            int g_rank = rn;
                            int s_rank = MPI_PROC_NULL;
                            MPI_Group_translate_ranks(env->group, 1, &g_rank,
                                                      env->shm_group, &s_rank);
                            if (s_rank != MPI_UNDEFINED) {
                                mpi_info->shm_ranks.at(rn_ofs) = s_rank;
                                DEBUG_MSG("  is MPI shared-memory rank " << s_rank);
                            }
                        }
                    }

                    // Save manhattan dist.
                    mpi_info->man_dists.at(rn_ofs) = mandist;

                    // Loop through domain dims.
                    bool vlen_mults = true;
                    DOMAIN_VAR_LOOP(i, j) {
                        auto& dname = domain_dims.get_dim_name(j);
                        auto nranks = actl_opts->_num_ranks[j];
                        bool is_last = (actl_opts->_rank_indices[j] == nranks - 1);

                        // Does rn have all VLEN-multiple sizes?
                        // TODO: allow last rank in each dim to be non-conformant.
                        auto rnsz = rsizes[rn][j];
                        auto vlen = fold_pts[j];
                        if (rnsz % vlen != 0) {
                            TRACE_MSG("cannot use vector halo exchange with rank " << rn <<
                                      " because its size in '" << dname << "' is " << rnsz);
                            vlen_mults = false;
                        }
                    }

                    // Save vec-mult flag.
                    mpi_info->has_all_vlen_mults.at(rn_ofs) = vlen_mults;

                } // self or immediate neighbor in any direction.
            } // ranks.

            // At end of 1st pass, known ranks sizes have
            // been summed in each dim. Determine global size
            // or other rank sizes for each dim.
            if (pass == 0) {
                DOMAIN_VAR_LOOP(i, j) {
                    auto& dname = domain_dims.get_dim_name(j);
                    auto nranks = actl_opts->_num_ranks[j];
                    auto gsz = actl_opts->_global_sizes[i];
                    bool is_last = (actl_opts->_rank_indices[j] == nranks - 1);

                    // Need to determine my rank size.
                    if (!actl_opts->_rank_sizes[i]) {
                        if (rank_domain_sums[j] != 0)
                            FORMAT_AND_THROW_YASK_EXCEPTION
                                ("Error: local-domain size is not specified in the '" <<
                                 dname << "' dimension on rank " << me <<
                                 ", but it is specified on another rank; "
                                 "it must be specified or unspecified consistently across all ranks");

                        // Divide sum by num of ranks in this dim.
                        auto rsz = CEIL_DIV(gsz, nranks);

                        // Round up to whole vector-clusters.
                        rsz = ROUND_UP(rsz, dims->_cluster_pts[j]);

                        // Remainder for last rank.
                        auto rem = gsz - (rsz * (nranks - 1));
                        if (rem <= 0)
                            FORMAT_AND_THROW_YASK_EXCEPTION
                                ("Error: global-domain size of " << gsz <<
                                 " is not large enough to split across " << nranks <<
                                 " ranks in the '" << dname << "' dimension");
                        if (is_last)
                            rsz = rem;

                        // Set rank size depending on whether it is last one.
                        actl_opts->_rank_sizes[i] = rsz;
                        TRACE_MSG("local-domain-size[" << dname << "] = " << rem);
                    }

                    // Need to determine global size.
                    // Set it to sum of rank sizes.
                    else if (!actl_opts->_global_sizes[i])
                        actl_opts->_global_sizes[i] = rank_domain_sums[j];
                }
            }

            // After 2nd pass, check for consistency.
            else {
                DOMAIN_VAR_LOOP(i, j) {
                    auto& dname = domain_dims.get_dim_name(j);
                    if (actl_opts->_global_sizes[i] != rank_domain_sums[j]) {
                        FORMAT_AND_THROW_YASK_EXCEPTION("Error: sum of local-domain sizes across " <<
                                                        nr << " ranks is " <<
                                                        rank_domain_sums[j] <<
                                                        ", which does not equal global-domain size of " <<
                                                        actl_opts->_global_sizes[i] << " in '" << dname <<
                                                        "' dimension");
                    }
                }
            }

        } // passes.
        #endif

    } // setup_rank().

    void StencilContext::print_warnings() const {
        STATE_VARS(this);
#ifdef CHECK
        DEBUG_MSG("*** WARNING: YASK compiled with CHECK; ignore performance.");
#endif
#if defined(NO_INTRINSICS) && (VLEN > 1)
        DEBUG_MSG("*** WARNING: YASK compiled with NO_INTRINSICS; ignore performance.");
#endif
#ifdef MODEL_CACHE
        DEBUG_MSG("*** WARNING: YASK compiled with MODEL_CACHE; ignore performance.");
#endif
#ifdef TRACE_MEM
        DEBUG_MSG("*** WARNING: YASK compiled with TRACE_MEM; ignore performance.");
#endif
#ifdef TRACE_INTRINSICS
        DEBUG_MSG("*** WARNING: YASK compiled with TRACE_INTRINSICS; ignore performance.");
#endif
        TRACE_MSG("***  WARNING: YASK run with -trace; ignore performance");
        if (!actl_opts->do_halo_exchange)
            DEBUG_MSG("*** WARNING: YASK run without halo exchanges; ignore performance; invalid results.");
    }

    void StencilContext::print_temporal_tiling_info(string prefix) const {
        STATE_VARS(this);

        DEBUG_MSG(prefix << "num-wave-front-steps:      " << wf_steps << endl <<
                  prefix << "num-temporal-block-steps:  " << tb_steps);

        // Print detailed info only if temporal tiling enabled.
        if (wf_steps > 0 || tb_steps > 0) {
            DEBUG_MSG(prefix << "wave-front-angles:         " << wf_angles.make_dim_val_str() << endl <<
                      prefix << "num-wave-front-shifts:     " << num_wf_shifts << endl <<
                      prefix << "wave-front-shift-amounts:  " << wf_shift_pts.make_dim_val_str() << endl <<
                      prefix << "left-wave-front-exts:      " << left_wf_exts.make_dim_val_str() << endl <<
                      prefix << "right-wave-front-exts:     " << right_wf_exts.make_dim_val_str() << endl <<
                      prefix << "ext-local-domain:          " << ext_bb.make_range_string(domain_dims) << endl <<
                      prefix << "temporal-block-angles:     " << tb_angles.make_dim_val_str() << endl <<
                      prefix << "num-temporal-block-shifts: " << num_tb_shifts << endl <<
                      prefix << "temporal-block-long-base:  " << tb_widths.make_dim_val_str(" * ") << endl <<
                      prefix << "temporal-block-short-base: " << tb_tops.make_dim_val_str(" * ") << endl <<
                      prefix << "micro-block-angles:        " << mb_angles.make_dim_val_str());
        }
    }

    void StencilContext::print_sizes(string prefix) const {
        STATE_VARS(this);
#ifdef USE_TILING
        DEBUG_MSG(prefix << "local-domain-tile-size: " <<
                  actl_opts->_rank_tile_sizes.remove_dim(step_posn).make_dim_val_str(" * "));
#endif
        DEBUG_MSG(prefix << "mega-block-size:        " <<
                  actl_opts->_mega_block_sizes.make_dim_val_str(" * "));
#ifdef USE_TILING
        DEBUG_MSG(prefix << "mega-block-tile-size:   " <<
                  actl_opts->_mega_block_tile_sizes.remove_dim(step_posn).make_dim_val_str(" * "));
#endif
        DEBUG_MSG(prefix << "block-size:             " <<
                  actl_opts->_block_sizes.make_dim_val_str(" * "));
#ifdef USE_TILING
        DEBUG_MSG(prefix << "block-tile-size:        " <<
                  actl_opts->_block_tile_sizes.remove_dim(step_posn).make_dim_val_str(" * "));
#endif
        DEBUG_MSG(prefix << "micro-block-size:       " <<
                  actl_opts->_micro_block_sizes.make_dim_val_str(" * "));
#ifdef USE_TILING
        DEBUG_MSG(prefix << "micro-block-tile-size:  " <<
                  actl_opts->_micro_block_tile_sizes.remove_dim(step_posn).make_dim_val_str(" * "));
#endif
        DEBUG_MSG(prefix << "nano-block-size:        " <<
                  actl_opts->_nano_block_sizes.remove_dim(step_posn).make_dim_val_str(" * "));
#ifdef USE_TILING
        DEBUG_MSG(prefix << "nano-block-tile-size:   " <<
                  actl_opts->_nano_block_tile_sizes.remove_dim(step_posn).make_dim_val_str(" * "));
#endif
        DEBUG_MSG(prefix << "pico-block-size:        " <<
                  actl_opts->_pico_block_sizes.remove_dim(step_posn).make_dim_val_str(" * "));
    }

    void StencilContext::init_stats() {
        STATE_VARS(this);

        // Calc and report total allocation and domain sizes.
        rank_nbytes = get_num_bytes();
        tot_nbytes = sum_over_ranks(rank_nbytes, env->comm);
        rank_domain_pts = rank_bb.bb_num_points;
        tot_domain_pts = sum_over_ranks(rank_domain_pts, env->comm);
        DEBUG_MSG("\nDomain size in this rank (points):          " << make_num_str(rank_domain_pts) <<
                  "\nTotal allocation in this rank:              " << make_byte_str(rank_nbytes) <<
                  "\nOverall problem size in " << env->num_ranks << " rank(s) (points): " <<
                  make_num_str(tot_domain_pts) <<
                  "\nTotal overall allocation in " << env->num_ranks << " rank(s):      " <<
                  make_byte_str(tot_nbytes));

        // Report some sizes and settings.
        DEBUG_MSG("\nWork-unit sizes in grid points (from largest to smallest):");
        DEBUG_MSG(" global-domain-size:     " <<
                  actl_opts->_global_sizes.remove_dim(step_posn).make_dim_val_str(" * "));
        DEBUG_MSG(" local-domain-size:      " <<
                  actl_opts->_rank_sizes.remove_dim(step_posn).make_dim_val_str(" * "));
        print_sizes(" ");
        DEBUG_MSG(" cluster-size:           " << dims->_cluster_pts.make_dim_val_str(" * "));
        DEBUG_MSG(" vector-size:            " << dims->_fold_pts.make_dim_val_str(" * "));
        DEBUG_MSG("\nOther settings:\n"
                  " yask-version:           " << yask_get_version_string() << endl <<
                  " target:                 " << get_target() << endl <<
                  " stencil-name:           " << get_name() << endl <<
                  " stencil-description:    " << get_description() << endl <<
                  " element-size:           " << make_byte_str(get_element_bytes()) << endl <<
                  " inner-layout dim:       " << dims->_inner_layout_dim << endl <<
                  " inner-loop dim:         " << dims->_inner_loop_dim);
#ifdef USE_MPI
        DEBUG_MSG(" num-ranks:              " << actl_opts->_num_ranks.make_dim_val_str(" * ") << endl <<
                  " rank-indices:           " << actl_opts->_rank_indices.make_dim_val_str() << endl <<
                  " local-domain-offsets:   " << rank_domain_offsets.make_dim_val_str(dims->_domain_dims));
        if (actl_opts->overlap_comms)
            DEBUG_MSG(" mpi-interior:           " << mpi_interior.make_range_string(domain_dims));
#endif
        DEBUG_MSG(" vector-len:             " << VLEN << endl <<
                  " extra-padding:          " << actl_opts->_extra_pad_sizes.remove_dim(step_posn).make_dim_val_str() << endl <<
                  " min-padding:            " << actl_opts->_min_pad_sizes.remove_dim(step_posn).make_dim_val_str() << endl <<
                  " allow-addl-padding:     " << actl_opts->_allow_addl_pad << endl <<
                  " L1-prefetch-distance:   " << PFD_L1 << endl <<
                  " L2-prefetch-distance:   " << PFD_L2 << endl <<
                  " max-halos:              " << max_halos.make_dim_val_str());
        print_temporal_tiling_info(" ");

        // Info about eqs, stages and bundles.
        DEBUG_MSG("\n"
                  "Num stages:              " << st_stages.size() << endl <<
                  "Num stencil bundles:     " << st_bundles.size() << endl <<
                  "Num stencil equations:   " << NUM_STENCIL_EQS);

        // Info on work in stages.
        DEBUG_MSG("\nBreakdown of work stats in this rank:");
        for (auto& sp : st_stages)
            sp->init_work_stats();
    }

    // Set non-scratch var sizes and offsets based on settings.
    // Set wave-front settings.
    // This should be called anytime a setting or rank offset is changed.
    void StencilContext::update_var_info(bool force) {
        STATE_VARS(this);
        TRACE_MSG("update_var_info(" << force << ")...");

        // If we haven't finished constructing the context, it's too early
        // to do this.
        if (!st_stages.size())
            return;

        // Reset max halos to zero.
        max_halos = dims->_domain_dims;

        // Loop through each domain dim.
        for (auto& dim : domain_dims) {
            auto& dname = dim._get_name();

            // Each non-scratch var.
            for (auto gp : all_var_ptrs) {
                assert(gp);
                if (!gp->is_dim_used(dname))
                    continue;
                auto& gb = gp->gb();

                // Don't resize manually-sized var
                // unless it is a solution var and 'force' is 'true'.
                if (!gp->is_fixed_size() ||
                    (!gb.is_user_var() && force)) {

                    // Rank domains.
                    gp->_set_domain_size(dname, actl_opts->_rank_sizes[dname]);

                    // Pads.
                    // Set via both 'extra' and 'min'; larger result will be used.
                    gp->update_extra_pad_size(dname, actl_opts->_extra_pad_sizes[dname]);
                    gp->update_min_pad_size(dname, actl_opts->_min_pad_sizes[dname]);

                    // Offsets.
                    auto dp = dims->_domain_dims.lookup_posn(dname);
                    gp->_set_rank_offset(dname, rank_domain_offsets[dp]);
                    gp->_set_local_offset(dname, 0);
                }

                // Update max halo across vars, used for temporal angles.
                if (!gb.is_user_var()) {
                    max_halos[dname] = max(max_halos[dname], gp->get_left_halo_size(dname));
                    max_halos[dname] = max(max_halos[dname], gp->get_right_halo_size(dname));
                }
            }
        }

        // Calculate wave-front shifts.
        // See the wavefront diagram in run_solution() for description
        // of angles and extensions.
        idx_t tb_steps = actl_opts->_block_sizes[step_dim]; // use requested size; actual may be less.
        assert(tb_steps >= 0);
        wf_steps = actl_opts->_mega_block_sizes[step_dim];
        wf_steps = max(wf_steps, tb_steps); // round up WF steps if less than TB steps.
        assert(wf_steps >= 0);
        num_wf_shifts = 0;
        if (wf_steps > 0) {

            // Need to shift for each stage.
            assert(st_stages.size() > 0);
            num_wf_shifts = idx_t(st_stages.size()) * wf_steps;

            // Don't need to shift first one.
            if (num_wf_shifts > 0)
                num_wf_shifts--;
        }
        assert(num_wf_shifts >= 0);

        // Calculate angles and related settings.
        for (auto& dim : domain_dims) {
            auto& dname = dim._get_name();
            auto rnsize = actl_opts->_mega_block_sizes[dname];
            auto rksize = actl_opts->_rank_sizes[dname];
            auto nranks = actl_opts->_num_ranks[dname];

            // Req'd shift in this dim based on max halos.
            idx_t angle = ROUND_UP(max_halos[dname], dims->_fold_pts[dname]);

            // Determine the spatial skewing angles for WF tiling.  We
            // only need non-zero angles if the mega-block size is less than the
            // rank size or there are other ranks in this dim, i.e., if
            // the mega-block covers the *global* domain in a given dim, no
            // wave-front shifting is needed in that dim.
            idx_t wf_angle = 0;
            if (rnsize < rksize || nranks > 1)
                wf_angle = angle;
            wf_angles.add_dim_back(dname, wf_angle);
            assert(angle >= 0);

            // Determine the total WF shift to be added in each dim.
            idx_t shifts = wf_angle * num_wf_shifts;
            wf_shift_pts[dname] = shifts;
            assert(shifts >= 0);

            // Is domain size at least as large as halo + wf_ext in direction
            // when there are multiple ranks?
            auto min_size = max_halos[dname] + shifts;
            if (actl_opts->_num_ranks[dname] > 1 && rksize < min_size) {
                FORMAT_AND_THROW_YASK_EXCEPTION
                    ("Error: local-domain size of " << rksize << " in '" <<
                     dname << "' dim is less than minimum size of " << min_size <<
                     ", which is based on stencil halos and temporal wave-front sizes");
            }

            // If there is another rank to the left, set wave-front
            // extension on the left.
            left_wf_exts[dname] = actl_opts->is_first_rank(dname) ? 0 : shifts;

            // If there is another rank to the right, set wave-front
            // extension on the right.
            right_wf_exts[dname] = actl_opts->is_last_rank(dname) ? 0 : shifts;
        }

        // Now that wave-front settings are known, we can push this info
        // back to the vars. It's useful to store this redundant info
        // in the vars, because there it's indexed by var dims instead
        // of domain dims. This makes it faster to do var indexing.
        for (auto gp : orig_var_ptrs) {
            assert(gp);

            // Loop through each domain dim.
            for (auto& dim : domain_dims) {
                auto& dname = dim._get_name();
                if (gp->is_dim_used(dname)) {
                    // Set extensions to be the same as the global ones.
                    gp->_set_left_wf_ext(dname, left_wf_exts[dname]);
                    gp->_set_right_wf_ext(dname, right_wf_exts[dname]);
                }
            }
        } // vars.

        // Calculate temporal-block shifts.
        // NB: this will change if/when block sizes change.
        update_tb_info();

    } // update_var_info().

    // Set temporal blocking data.  This should be called anytime a block
    // size is changed.  Must be called after update_var_info() to ensure
    // angles are properly set.  TODO: calculate 'tb_steps' dynamically
    // considering temporal conditions; this assumes worst-case, which is
    // all stages always done.
    void StencilContext::update_tb_info() {
        STATE_VARS(this);
        TRACE_MSG("update_tb_info()...");

        // Get requested size.
        tb_steps = actl_opts->_block_sizes[step_dim];

        // Reset all TB and MB vars.
        num_tb_shifts = 0;
        tb_angles.set_vals_same(0);
        tb_widths.set_vals_same(0);
        tb_tops.set_vals_same(0);
        mb_angles.set_vals_same(0);

        // Set angles.
        // Determine max temporal depth based on block sizes
        // and requested temporal depth.
        // When using temporal blocking, all block sizes
        // across all stages must be the same.
        TRACE_MSG("update_tb_info: original TB steps = " << tb_steps);
        if (tb_steps > 0) {

            // TB is inside WF, so can't be larger.
            idx_t max_steps = min(tb_steps, wf_steps);
            TRACE_MSG("update_tb_info: min(TB, WF) steps = " << max_steps);

            // Loop through each domain dim.
            DOMAIN_VAR_LOOP(i, j) {
                auto& dim = domain_dims.get_dim(j);
                auto& dname = dim._get_name();
                auto rnsize = actl_opts->_mega_block_sizes[i];
                auto blksize = actl_opts->_block_sizes[i];
                auto mblksize = actl_opts->_micro_block_sizes[i];

                // Req'd shift in this dim based on max halos.
                // Can't use separate L & R shift because of possible data reuse in vars.
                // Can't use separate shifts for each stage for same reason.
                // TODO: make round-up optional.
                auto fpts = dims->_fold_pts[j];
                idx_t angle = ROUND_UP(max_halos[j], fpts);

                // Determine the spatial skewing angles for MB.
                // If MB covers whole blk, no shifting is needed in that dim.
                idx_t mb_angle = 0;
                if (mblksize < blksize)
                    mb_angle = angle;
                mb_angles[j] = mb_angle;

                // Determine the max spatial skewing angles for TB.
                // If blk covers whole mega-block, no shifting is needed in that dim.
                idx_t tb_angle = 0;
                if (blksize < rnsize)
                    tb_angle = angle;
                tb_angles[j] = tb_angle;

                // Calculate max number of temporal steps in
                // allowed this dim.
                if (tb_angle > 0) {

                    // min_blk_sz = min_top_sz + 2 * angle * (nstages * nsteps - 1).
                    // bs = ts + 2*a*np*ns - 2*a.
                    // 2*a*np*ns = bs - ts + 2*a.
                    // s = flr[ (bs - ts + 2*a) / 2*a*np ].
                    idx_t top_sz = fpts; // min pts on top row. TODO: is zero ok?
                    idx_t sh_pts = tb_angle * 2 * st_stages.size(); // pts shifted per step.
                    idx_t nsteps = (blksize - top_sz + tb_angle * 2) / sh_pts; // might be zero.
                    TRACE_MSG("update_tb_info: max TB steps in dim '" <<
                              dname << "' = " << nsteps <<
                              " due to base block size of " << blksize <<
                              ", TB angle of " << tb_angle <<
                              ", and " << st_stages.size() << " stage(s)");
                    max_steps = min(max_steps, nsteps);
                }
            }
            tb_steps = min(tb_steps, max_steps);
            TRACE_MSG("update_tb_info: final TB steps = " << tb_steps);
        }
        assert(tb_steps >= 0);

        // Calc number of shifts based on steps.
        if (tb_steps > 0) {

            // Need to shift for each stage.
            assert(st_stages.size() > 0);
            num_tb_shifts = idx_t(st_stages.size()) * tb_steps;

            // Don't need to shift first one.
            if (num_tb_shifts > 0)
                num_tb_shifts--;
        }
        assert(num_tb_shifts >= 0);
        TRACE_MSG("update_tb_info: num TB shifts = " << num_tb_shifts);

        // Calc size of base of phase 0 trapezoid.
        // Initial width is half of base plus one shift distance.  This will
        // make 'up' and 'down' trapezoids approx same size.

        //   x->
        // ^   ----------------------
        // |  /        \            /^
        // t /  phase 0 \ phase 1  / |
        //  /            \        /  |
        //  ----------------------   |
        //  ^             ^       ^  |
        //  |<-blk_width->|    -->|  |<--sa=nshifts*angle
        //  |             |       |
        // blk_start  blk_stop  next_blk_start
        //  |                     |
        //  |<-----blk_sz-------->|
        // blk_width = blk_sz/2 + sa.

        // Ex: blk_sz=12, angle=4, nshifts=1, fpts=4,
        // sa=1*4=4, blk_width=rnd_up(12/2+4,4)=12.
        //     111122222222
        // 111111111111

        // Ex: blk_sz=16, angle=4, nshifts=1, fpts=4,
        // sa=1*4=4, blk_width=rnd_up(16/2+4,4)=12.
        //     1111222222222222
        // 1111111111112222

        // Ex: blk_sz=16, angle=2, nshifts=2, fpts=2,
        // sa=2*2=4, blk_width=rnd_up(16/2+4,2)=12.
        //     1111222222222222
        //   1111111122222222
        // 1111111111112222

        // TODO: use actual number of shifts dynamically instead of this
        // max.
        DOMAIN_VAR_LOOP(i, j) {
            auto blk_sz = actl_opts->_block_sizes[i];
            auto tb_angle = tb_angles[j];
            tb_widths[j] = blk_sz;
            tb_tops[j] = blk_sz;

            // If no shift or angle in this dim, we don't need
            // bridges at all, so base is entire block.
            if (num_tb_shifts > 0 && tb_angle > 0) {

                // See equations above for block size.
                auto fpts = dims->_fold_pts[j];
                idx_t min_top_sz = fpts;
                idx_t sa = num_tb_shifts * tb_angle;
                idx_t min_blk_width = min_top_sz + 2 * sa;
                idx_t blk_width = ROUND_UP(CEIL_DIV(blk_sz, idx_t(2)) + sa, fpts);
                blk_width = max(blk_width, min_blk_width);
                idx_t top_sz = max(blk_width - 2 * sa, idx_t(0));
                tb_widths[j] = blk_width;
                tb_tops[j] = top_sz;
            }
        }
        TRACE_MSG("update_tb_info: trapezoid bases = " << tb_widths.make_dim_val_str() <<
                  ", tops = " << tb_tops.make_dim_val_str());
    } // update_tb_info().

    // Init all vars & params by calling init_fn.
    void StencilContext::init_values(real_t seed0,
                                     function<void (YkVarPtr gp,
                                                    real_t seed)> real_init_fn) {
        STATE_VARS(this);

        real_t seed = seed0;
        DEBUG_MSG("Initializing stencil vars...");
        YaskTimer itimer;
        itimer.start();
        for (auto gp : orig_var_ptrs) {
            real_init_fn(gp, seed);
            gp->gb().get_coh().mod_host();
            seed += seed0;
        }
        itimer.stop();
        DEBUG_MSG("Var initialization done in " <<
                  make_num_str(itimer.get_elapsed_secs()) << " secs.");
    }

    // Set the bounding-box for each stencil-bundle and whole domain.
    void StencilContext::find_bounding_boxes()
    {
        STATE_VARS(this);
        DEBUG_MSG("Constructing bounding boxes for " <<
                  st_bundles.size() << " stencil-bundles(s)...");
        YaskTimer bbtimer;
        bbtimer.start();

        // Rank BB is based only on rank offsets and rank domain sizes.
        rank_bb.bb_begin = rank_domain_offsets;
        rank_bb.bb_end = rank_bb.bb_begin_tuple(domain_dims).add_elements(actl_opts->_rank_sizes, false);
        rank_bb.update_bb("rank", this, true, true);

        // BB may be extended for wave-fronts.
        ext_bb.bb_begin = rank_bb.bb_begin.sub_elements(left_wf_exts);
        ext_bb.bb_end = rank_bb.bb_end.add_elements(right_wf_exts);
        ext_bb.update_bb("extended-rank", this, true);

        // Remember sub-domain for each bundle.
        map<string, StencilBundleBase*> bb_descrs;

        // Find BB for each stage.
        for (auto sp : st_stages) {
            auto& spbb = sp->get_bb();
            spbb.bb_begin = domain_dims;
            spbb.bb_end = domain_dims;

            // Find BB for each bundle in this stage.
            for (auto sb : *sp) {

                // Already done?
                auto bb_descr = sb->get_domain_description();
                if (bb_descrs.count(bb_descr)) {

                    // Copy existing.
                    auto* src = bb_descrs.at(bb_descr);
                    sb->copy_bounding_box(src);
                }

                // Find bundle BB.
                else {
                    sb->find_bounding_box();
                    bb_descrs[bb_descr] = sb;
                }

                auto& sbbb = sb->get_bb();

                // Expand stage BB to encompass bundle BB.
                spbb.bb_begin = spbb.bb_begin.min_elements(sbbb.bb_begin);
                spbb.bb_end = spbb.bb_end.max_elements(sbbb.bb_end);
            }
            spbb.update_bb(sp->get_name(), this, false);
        }

        // Init MPI interior to extended BB.
        mpi_interior = ext_bb;

        bbtimer.stop();
        DEBUG_MSG("Bounding-box construction done in " <<
                  make_num_str(bbtimer.get_elapsed_secs()) << " secs.");
    }

    // Copy BB vars from another.
    void StencilBundleBase::copy_bounding_box(const StencilBundleBase* src) {
        STATE_VARS(this);
        TRACE_MSG("copy_bounding_box for '" << get_name() << "' from '" <<
                  src->get_name() << "'...");

        _bundle_bb = src->_bundle_bb;
        assert(_bundle_bb.bb_valid);
        _bb_list = src->_bb_list;
    }

    // Find the bounding-boxes for this bundle in this rank.
    // Only tests domain-var values, not step-vars.
    // Step-vars are tested dynamically for each step
    // as it is executed.
    void StencilBundleBase::find_bounding_box() {
        STATE_VARS(this);
        TRACE_MSG("find_bounding_box for '" << get_name() << "'...");

        // Init overall bundle BB to that of parent and clear list.
        assert(_context);
        _bundle_bb = _context->ext_bb;
        assert(_bundle_bb.bb_valid);
        _bb_list.clear();

        // If BB is empty, we are done.
        if (!_bundle_bb.bb_size)
            return;

        // If there is no condition, just add full BB to list.
        if (!is_sub_domain_expr()) {
            TRACE_MSG("adding 1 sub-BB: [" << _bundle_bb.make_range_string(domain_dims) << "]");
            _bb_list.push_back(_bundle_bb);
            return;
        }

        // Goal: Create list of full BBs (non-overlapping & with no invalid
        // points) inside overall BB.
        YaskTimer bbtimer;
        bbtimer.start();

        // Divide the overall BB into a slice for each thread
        // across the outer dim.
        const int odim = 0;     // Split across first domain dim; TODO: pick smarter.
        idx_t outer_len = _bundle_bb.bb_len[odim];
        idx_t nthreads = yask_get_num_threads();
        idx_t len_per_thr = CEIL_DIV(outer_len, nthreads);
        TRACE_MSG("find_bounding_box: running " << nthreads << " thread(s) over " <<
                  outer_len << " point(s) in outer dim");

        // Struct w/padding to avoid false sharing.
        struct BBL_t {
            BBList bbl;
            char pad[CACHELINE_BYTES - sizeof(BBList)];
        };

        // List of full BBs for each thread.
        vector<BBL_t> bb_lists(nthreads);

        // Run rect-finding code on each thread.
        // When these are done, we will merge the
        // rects from all threads.
        yask_parallel_for
            (0, nthreads, 1,
             [&](idx_t start, idx_t stop, idx_t thread_num) {
                 auto& cur_bb_list = bb_lists[start].bbl;

                 // Begin and end of this slice.
                 // These Indices contain domain dims.
                 Indices islice_begin(_bundle_bb.bb_begin);
                 islice_begin[odim] += start * len_per_thr;
                 Indices islice_end(_bundle_bb.bb_end);
                 islice_end[odim] = min(islice_end[odim], islice_begin[odim] + len_per_thr);
                 if (islice_end[odim] <= islice_begin[odim])
                     return; // from lambda.

                 // Construct len of slice in all dims.
                 Indices islice_len = islice_end.sub_elements(islice_begin);

                 // Visit all points in slice, looking for a new
                 // valid beginning point, 'ib*pt'.
                 Indices ibspt(stencil_dims); // in stencil dims.
                 ibspt[step_posn] = 0;
                 Indices ibdpt(domain_dims);  // in domain dims.
                 islice_len.visit_all_points
                     (true,
                      [&](const Indices& iofs, size_t idx) {

                          // Find global point from 'iofs' in domain
                          // and stencil dims.
                          ibdpt = islice_begin.add_elements(iofs); // domain indices.
                          DOMAIN_VAR_LOOP(i, j)
                              ibspt[i] = ibdpt[j];            // stencil indices.

                          // Valid point must be in sub-domain and
                          // not seen before in this slice.
                          bool is_valid = is_in_valid_domain(ibspt);
                          if (is_valid) {
                              for (auto& bb : cur_bb_list) {
                                  if (bb.is_in_bb(ibdpt)) {
                                      is_valid = false;
                                      break;
                                  }
                              }
                          }

                          // Process this new rect starting at 'ib*pt'.
                          if (is_valid) {

                              // Scan from 'ib*pt' to end of this slice
                              // looking for end of rect.
                              auto iscan_len = islice_end.sub_elements(ibdpt);

                              // End point to be found, 'ie*pt'.
                              Indices iespt(stencil_dims); // stencil dims.
                              iespt[step_posn] = 0;
                              Indices iedpt(domain_dims);  // domain dims.

                              // Repeat scan until no adjustment is made.
                              bool do_scan = true;
                              while (do_scan) {
                                  do_scan = false;

                                  TRACE_MSG("scanning " <<
                                            iscan_len.make_dim_val_str(domain_dims, " * ") <<
                                            " starting at " <<
                                            ibdpt.make_dim_val_str(domain_dims) <<
                                            " in thread " << thread_num);
                                  iscan_len.visit_all_points
                                      (true,
                                       [&](const Indices& ieofs, size_t eidx) {

                                           // Make sure iscan_len range is observed.
                                           DOMAIN_VAR_LOOP(i, j)
                                               assert(ieofs[j] < iscan_len[j]);

                                           // Find global point from 'ieofs'.
                                           iedpt = ibdpt.add_elements(ieofs); // domain tuple.
                                           DOMAIN_VAR_LOOP(i, j)
                                               iespt[i] = iedpt[j];            // stencil tuple.

                                           // Valid point must be in sub-domain and
                                           // not seen before in this slice.
                                           bool is_evalid = is_in_valid_domain(iespt);
                                           if (is_evalid) {
                                               for (auto& bb : cur_bb_list) {
                                                   if (bb.is_in_bb(iedpt)) {
                                                       is_evalid = false;
                                                       break;
                                                   }
                                               }
                                           }

                                           // If this is an invalid point, adjust
                                           // scan range appropriately.
                                           if (!is_evalid) {

                                               // Adjust 1st dim that is beyond its starting pt.
                                               // This will reduce the range of the scan.
                                               DOMAIN_VAR_LOOP(i, j) {

                                                   // Beyond starting point in this dim?
                                                   if (iedpt[j] > ibdpt[j]) {
                                                       iscan_len[j] = iedpt[j] - ibdpt[j];

                                                       // restart scan for
                                                       // remaining dims.
                                                       // TODO: be smarter
                                                       // about where to
                                                       // restart scan.
                                                       if (j < nddims - 1)
                                                           do_scan = true;

                                                       return false; // stop this scan.
                                                   }
                                               }
                                           }

                                           return true; // keep looking for invalid point.
                                       }); // Looking for invalid point.
                              } // while scan is adjusted.
                              TRACE_MSG("found BB " << iscan_len.make_dim_val_str(domain_dims, " * ") <<
                                        " starting at " << ibdpt.make_dim_val_str(domain_dims) <<
                                        " in thread " << thread_num);

                              // 'iscan_len' now contains sizes of the new BB.
                              BoundingBox new_bb;
                              new_bb.bb_begin = ibdpt;
                              new_bb.bb_end = ibdpt.add_elements(iscan_len);
                              new_bb.update_bb("sub-bb", _context, true);
                              cur_bb_list.push_back(new_bb);

                          } // new rect found.

                          return true;  // from labmda; keep looking.
                      }); // Looking for new rects.
             }); // threads/slices.
        TRACE_MSG("sub-bbs found in " <<
                  bbtimer.get_secs_since_start() << " secs.");
        // At this point, we have a set of full BBs.

        // Reset overall BB.
        _bundle_bb.bb_num_points = 0;

        // Collect BBs in all slices.
        // TODO: merge in a parallel binary tree instead of sequentially.
        for (int n = 0; n < nthreads; n++) {
            auto& cur_bb_list = bb_lists[n].bbl;
            TRACE_MSG("processing " << cur_bb_list.size() <<
                      " sub-BB(s) in bundle '" << get_name() <<
                      "' from thread " << n);

            // BBs in slice 'n'.
            for (auto& bbn : cur_bb_list) {
                TRACE_MSG(" sub-BB: [" << bbn.make_range_string(domain_dims) << "]");

                // Don't bother with empty BB.
                if (bbn.bb_size == 0)
                    continue;

                // Init or update overall BB.
                if (!_bundle_bb.bb_num_points) {
                    _bundle_bb.bb_begin = bbn.bb_begin;
                    _bundle_bb.bb_end = bbn.bb_end;
                } else {
                    _bundle_bb.bb_begin = _bundle_bb.bb_begin.min_elements(bbn.bb_begin);
                    _bundle_bb.bb_end = _bundle_bb.bb_end.max_elements(bbn.bb_end);
                }
                _bundle_bb.bb_num_points += bbn.bb_size;

                // Scan existing final BBs looking for one to merge with.
                bool do_merge = false;
                for (auto& bb : _bb_list) {

                    // Can 'bbn' be merged with 'bb'?
                    do_merge = true;
                    for (int i = 0; i < nddims && do_merge; i++) {

                        // Must be adjacent in outer dim.
                        if (i == odim) {
                            if (bb.bb_end[i] != bbn.bb_begin[i])
                                do_merge = false;
                        }

                        // Must be aligned in other dims.
                        else {
                            if (bb.bb_begin[i] != bbn.bb_begin[i] ||
                                bb.bb_end[i] != bbn.bb_end[i])
                                do_merge = false;
                        }
                    }
                    if (do_merge) {

                        // Merge by just increasing the size of 'bb'.
                        bb.bb_end[odim] = bbn.bb_end[odim];
                        TRACE_MSG("  merging to form [" << bb.make_range_string(domain_dims) << "]");
                        bb.update_bb("sub-bb", _context, true);
                        break;
                    }
                }

                // If not merged, add 'bbn' as new.
                if (!do_merge) {
                    _bb_list.push_back(bbn);
                    TRACE_MSG("  adding as final sub-BB #" << _bb_list.size());
                }
            }
        }

        // Finalize overall BB.
        _bundle_bb.update_bb(get_name(), _context, false);
        bbtimer.stop();
        TRACE_MSG("find-bounding-box: done in " <<
                  bbtimer.get_elapsed_secs() << " secs.");
    }

    // Compute convenience values for a bounding-box.
    void BoundingBox::update_bb(const string& name,
                                StencilContext* context,
                                bool force_full,
                                bool print_info) {

        STATE_VARS(context);

        bb_len = bb_end.sub_elements(bb_begin);
        bb_size = bb_len.product();
        if (force_full)
            bb_num_points = bb_size;

        // Solid rectangle?
        bb_is_full = true;
        if (bb_num_points != bb_size) {
            if (print_info)
                DEBUG_MSG("Note: '" << name << "' domain has only " <<
                          make_num_str(bb_num_points) <<
                          " valid point(s) inside its bounding-box of " <<
                          make_num_str(bb_size) <<
                          " point(s); multiple sub-boxes will be used.");
            bb_is_full = false;
        }

        // Does everything start on a vector-length boundary?
        bb_is_aligned = true;
        DOMAIN_VAR_LOOP(i, j) {
            if ((bb_begin[j] - context->rank_domain_offsets[j]) %
                dims->_fold_pts[j] != 0) {
                if (print_info)
                    DEBUG_MSG("Note: '" << name << "' domain"
                              " has one or more starting edges not on vector boundaries;"
                              " masked calculations will be used in peel and remainder nano-blocks.");
                bb_is_aligned = false;
                break;
            }
        }

        // Lengths are cluster-length multiples?
        bb_is_cluster_mult = true;
        DOMAIN_VAR_LOOP(i, j) {
            auto& dim = domain_dims.get_dim(j);
            auto& dname = dim._get_name();
            if (bb_len[j] % dims->_cluster_pts[dname] != 0) {
                if (bb_is_full && bb_is_aligned)
                    if (print_info && bb_is_aligned)
                        DEBUG_MSG("Note: '" << name << "' domain"
                                  " has one or more sizes that are not vector-cluster multiples;"
                                  " masked calculations will be used in peel and remainder nano-blocks.");
                bb_is_cluster_mult = false;
                break;
            }
        }

        // All done.
        bb_valid = true;
    }

    // Add a new non-scratch var to the containers.
    void StencilContext::add_var(YkVarPtr gp, bool is_orig, bool is_output) {
        STATE_VARS(this);
        assert(gp);
        auto& gname = gp->get_name();
        if (all_var_map.count(gname))
            THROW_YASK_EXCEPTION("Error: var '" + gname + "' already exists");

        // Add to list and map.
        all_var_ptrs.push_back(gp);
        all_var_map[gname] = gp;

        // Add to orig list and map if 'is_orig'.
        if (is_orig) {
            orig_var_ptrs.push_back(gp);
            orig_var_map[gname] = gp;
        }

        // Add to output list and map if 'is_output'.
        if (is_output) {
            output_var_ptrs.push_back(gp);
            output_var_map[gname] = gp;
        }
    }
} // namespace yask.
