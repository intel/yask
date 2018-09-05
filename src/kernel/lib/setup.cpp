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

// This file contains implementations of StencilContext and
// StencilBundleBase methods specific to the preparation steps.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Constructor.
    StencilContext::StencilContext(KernelEnvPtr env,
                                   KernelSettingsPtr settings) :
        _ostr(&std::cout),
        _env(env),
        _opts(settings),
        _dims(settings->_dims),
        _at(this, _opts.get(), "solution")
    {
        // Set debug output object.
        yask_output_factory yof;
        set_debug_output(yof.new_stdout_output());

        // Create MPI Info object.
        _mpiInfo = std::make_shared<MPIInfo>(settings->_dims);

        // Init various tuples to make sure they have the correct dims.
        rank_domain_offsets = _dims->_domain_dims;
        rank_domain_offsets.setValsSame(-1); // indicates prepare_solution() not called.
        overall_domain_sizes = _dims->_domain_dims;
        max_halos = _dims->_domain_dims;
        wf_angles = _dims->_domain_dims;
        wf_shifts = _dims->_domain_dims;
        tb_angles = _dims->_domain_dims;
        tb_shifts = _dims->_domain_dims;
        left_wf_exts = _dims->_domain_dims;
        right_wf_exts = _dims->_domain_dims;

        // Set output to msg-rank per settings.
        set_ostr();
    }

    // Init MPI-related vars and other vars related to my rank's place in
    // the global problem: rank index, offset, etc.  Need to call this even
    // if not using MPI to properly init these vars.  Called from
    // prepare_solution(), so it doesn't normally need to be called from user code.
    void StencilContext::setupRank() {
        ostream& os = get_ostr();
        auto& step_dim = _dims->_step_dim;
        auto me = _env->my_rank;
        int num_neighbors = 0;

        // Check ranks.
        idx_t req_ranks = _opts->_num_ranks.product();
        if (req_ranks != _env->num_ranks) {
            FORMAT_AND_THROW_YASK_EXCEPTION("error: " << req_ranks << " rank(s) requested (" +
                                            _opts->_num_ranks.makeDimValStr(" * ") + "), but " <<
                                            _env->num_ranks << " rank(s) are active");
        }
        assertEqualityOverRanks(_opts->_rank_sizes[step_dim], _env->comm, "num steps");

        // Determine my coordinates if not provided already.
        // TODO: do this more intelligently based on proximity.
        if (_opts->find_loc)
            _opts->_rank_indices = _opts->_num_ranks.unlayout(me);

        // A table of rank-coordinates for everyone.
        auto num_ddims = _opts->_rank_indices.size(); // domain-dims only!
        idx_t coords[_env->num_ranks][num_ddims];

        // Init offsets and total sizes.
        rank_domain_offsets.setValsSame(0);
        overall_domain_sizes.setValsSame(0);

        // Init coords for this rank.
        for (int i = 0; i < num_ddims; i++)
            coords[me][i] = _opts->_rank_indices[i];

        // A table of rank-domain sizes for everyone.
        idx_t rsizes[_env->num_ranks][num_ddims];

        // Init sizes for this rank.
        for (int di = 0; di < num_ddims; di++) {
            auto& dname = _opts->_rank_indices.getDimName(di);
            auto rsz = _opts->_rank_sizes[dname];
            rsizes[me][di] = rsz;
            overall_domain_sizes[dname] = rsz;
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

        // Loop over all ranks, including myself.
        for (int rn = 0; rn < _env->num_ranks; rn++) {

            // Coord offset of rn from me: prev => negative, self => 0, next => positive.
            IdxTuple rcoords(_dims->_domain_dims);
            IdxTuple rdeltas(_dims->_domain_dims);
            for (int di = 0; di < num_ddims; di++) {
                rcoords[di] = coords[rn][di];
                rdeltas[di] = coords[rn][di] - _opts->_rank_indices[di];
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
            for (int di = 0; di < num_ddims; di++) {
                auto& dname = _opts->_rank_indices.getDimName(di);

                // Is rank 'rn' in-line with my rank in 'dname' dim?
                // True when deltas in other dims are zero.
                bool is_inline = true;
                for (int dj = 0; dj < num_ddims; dj++) {
                    if (di != dj && rdeltas[dj] != 0) {
                        is_inline = false;
                        break;
                    }
                }

                // Process ranks that are in-line in 'dname', including self.
                if (is_inline) {

                    // Accumulate total problem size in each dim for ranks that
                    // intersect with this rank, not including myself.
                    if (rn != me)
                        overall_domain_sizes[dname] += rsizes[rn][di];

                    // Adjust my offset in the global problem by adding all domain
                    // sizes from prev ranks only.
                    if (rdeltas[di] < 0)
                        rank_domain_offsets[dname] += rsizes[rn][di];

                    // Make sure all the other dims are the same size.
                    // This ensures that all the ranks' domains line up
                    // properly along their edges and at their corners.
                    for (int dj = 0; dj < num_ddims; dj++) {
                        if (di != dj) {
                            auto mysz = rsizes[me][dj];
                            auto rnsz = rsizes[rn][dj];
                            if (mysz != rnsz) {
                                auto& dnamej = _opts->_rank_indices.getDimName(dj);
                                FORMAT_AND_THROW_YASK_EXCEPTION
                                    ("Error: rank " << rn << " and " << me <<
                                     " are both at rank-index " << coords[me][di] <<
                                     " in the '" << dname <<
                                     "' dimension , but their rank-domain sizes are " <<
                                     rnsz << " and " << mysz <<
                                     " (resp.) in the '" << dj <<
                                     "' dimension, making them unaligned");
                            }
                        }
                    }
                }
            }

            // Rank rn is myself or my immediate neighbor if its distance <= 1 in
            // every dim.  Assume we do not need to exchange halos except
            // with immediate neighbor. We validate this assumption below by
            // making sure that the rank domain size is at least as big as the
            // largest halo.
            if (maxdist <= 1) {

                // At this point, rdeltas contains only -1..+1 for each domain dim.
                // Add one to -1..+1 to get 0..2 range for my_neighbors offsets.
                IdxTuple roffsets = rdeltas.addElements(1);
                assert(rdeltas.min() >= -1);
                assert(rdeltas.max() <= 1);
                assert(roffsets.min() >= 0);
                assert(roffsets.max() <= 2);

                // Convert the offsets into a 1D index.
                auto rn_ofs = _mpiInfo->getNeighborIndex(roffsets);
                TRACE_MSG("neighborhood size = " << _mpiInfo->neighborhood_sizes.makeDimValStr() <<
                          " & roffsets of rank " << rn << " = " << roffsets.makeDimValStr() <<
                          " => " << rn_ofs);
                assert(idx_t(rn_ofs) < _mpiInfo->neighborhood_size);

                // Save rank of this neighbor into the MPI info object.
                _mpiInfo->my_neighbors.at(rn_ofs) = rn;
                if (rn != me) {
                    num_neighbors++;
                    os << "Neighbor #" << num_neighbors << " is rank " << rn <<
                        " at absolute rank indices " << rcoords.makeDimValStr() <<
                        " (" << rdeltas.makeDimValOffsetStr() << " relative to rank " <<
                        me << ")\n";
                }

                // Save manhattan dist.
                _mpiInfo->man_dists.at(rn_ofs) = mandist;

                // Loop through domain dims.
                bool vlen_mults = true;
                for (int di = 0; di < num_ddims; di++) {
                    auto& dname = _opts->_rank_indices.getDimName(di);

                    // Does rn have all VLEN-multiple sizes?
                    auto rnsz = rsizes[rn][di];
                    auto vlen = _dims->_fold_pts[di];
                    if (rnsz % vlen != 0) {
                        TRACE_MSG("cannot use vector halo exchange with rank " << rn <<
                                  " because its size in '" << dname << "' is " << rnsz);
                        vlen_mults = false;
                    }
                }

                // Save vec-mult flag.
                _mpiInfo->has_all_vlen_mults.at(rn_ofs) = vlen_mults;

            } // self or immediate neighbor in any direction.

        } // ranks.
#endif

        // Set offsets in grids and find WF extensions
        // based on the grids' halos.
        update_grid_info();

        // Determine bounding-boxes for all bundles.
        // This must be done after finding WF extensions.
        find_bounding_boxes();

    } // setupRank().

    // Alloc 'nbytes' on each requested NUMA node.
    // Map keys are preferred NUMA nodes or -1 for local.
    // Pointers are returned in '_data_buf'.
    // 'ngrids' and 'type' are only used for debug msg.
    void StencilContext::_alloc_data(const map <int, size_t>& nbytes,
                                     const map <int, size_t>& ngrids,
                                     map <int, shared_ptr<char>>& data_buf,
                                     const std::string& type) {
        ostream& os = get_ostr();

        for (const auto& i : nbytes) {
            int numa_pref = i.first;
            size_t nb = i.second;
            size_t ng = ngrids.at(numa_pref);

            // Don't need pad after last one.
            if (nb >= _data_buf_pad)
                nb -= _data_buf_pad;

            // Allocate data.
            os << "Allocating " << makeByteStr(nb) <<
                " for " << ng << " " << type << "(s)";
#ifdef USE_NUMA
            if (numa_pref >= 0)
                os << " preferring NUMA node " << numa_pref;
            else
                os << " using NUMA policy " << numa_pref;
#endif
            os << "...\n" << flush;
            auto p = shared_numa_alloc<char>(nb, numa_pref);
            TRACE_MSG("Got memory at " << static_cast<void*>(p.get()));

            // Save using original key.
            data_buf[numa_pref] = p;
        }
    }

    // Allocate memory for grids that do not already have storage.
    void StencilContext::allocGridData(ostream& os) {

        // Base ptrs for all default-alloc'd data.
        // These pointers will be shared by the ones in the grid
        // objects, which will take over ownership when these go
        // out of scope.
        // Key is preferred numa node or -1 for local.
        map <int, shared_ptr<char>> _grid_data_buf;

        // Pass 0: count required size for each NUMA node, allocate chunk of memory at end.
        // Pass 1: distribute parts of already-allocated memory chunk.
        for (int pass = 0; pass < 2; pass++) {
            TRACE_MSG("allocGridData pass " << pass << " for " <<
                      gridPtrs.size() << " grid(s)");

            // Count bytes needed and number of grids for each NUMA node.
            map <int, size_t> npbytes, ngrids;

            // Grids.
            for (auto gp : gridPtrs) {
                if (!gp)
                    continue;
                auto& gname = gp->get_name();

                // Grid data.
                // Don't alloc if already done.
                if (!gp->is_storage_allocated()) {
                    int numa_pref = gp->get_numa_preferred();

                    // Set storage if buffer has been allocated in pass 0.
                    if (pass == 1) {
                        auto p = _grid_data_buf[numa_pref];
                        assert(p);
                        gp->set_storage(p, npbytes[numa_pref]);
                        os << gp->make_info_string() << endl;
                    }

                    // Determine padded size (also offset to next location).
                    size_t nbytes = gp->get_num_storage_bytes();
                    npbytes[numa_pref] += ROUND_UP(nbytes + _data_buf_pad,
                                                   CACHELINE_BYTES);
                    ngrids[numa_pref]++;
                    if (pass == 0)
                        TRACE_MSG(" grid '" << gname << "' needs " << makeByteStr(nbytes) <<
                                  " on NUMA node " << numa_pref);
                }

                // Otherwise, just print existing grid info.
                else if (pass == 0)
                    os << gp->make_info_string() << endl;
            }

            // Alloc for each node.
            if (pass == 0)
                _alloc_data(npbytes, ngrids, _grid_data_buf, "grid");

        } // grid passes.
    };

    // Determine the size and shape of all MPI buffers.
    // Create buffers and allocate them.
    void StencilContext::allocMpiData(ostream& os) {

        // Remove any old MPI data.
        freeMpiData(os);

        // Init interior.
        mpi_interior = ext_bb;
        mpi_interior.bb_valid = false;

#ifdef USE_MPI

        map<int, int> num_exchanges; // send/recv => count.
        map<int, idx_t> num_elems; // send/recv => count.
        auto me = _env->my_rank;
        auto& step_dim = _dims->_step_dim;
        auto* settings = get_settings().get();

        // Need to determine the size and shape of all MPI buffers.
        // Loop thru all neighbors of this rank.
        _mpiInfo->visitNeighbors
            ([&](const IdxTuple& neigh_offsets, int neigh_rank, int neigh_idx) {
                if (neigh_rank == MPI_PROC_NULL)
                    return; // from lambda fn.

                // Determine max dist needed.  TODO: determine max dist
                // automatically from stencils; may not be same for all
                // grids.
#ifndef MAX_EXCH_DIST
#define MAX_EXCH_DIST (NUM_STENCIL_DIMS - 1)
#endif
                // Always use max dist with WF.
                // TODO: determine if this is overkill.
                int maxdist = MAX_EXCH_DIST;
                if (num_wf_shifts > 0)
                    maxdist = NUM_STENCIL_DIMS - 1;

                // Manhattan dist.
                int mandist = _mpiInfo->man_dists.at(neigh_idx);

                // Check distance.
                // TODO: calculate and use exch dist for each grid.
                if (mandist > maxdist) {
                    TRACE_MSG("no halo exchange needed with rank " << neigh_rank <<
                              " because L1-norm = " << mandist);
                    return;     // from lambda fn.
                }

                // Is vectorized exchange allowed based on domain sizes?
                // Both my rank and neighbor rank must have *all* domain sizes
                // of vector multiples.
                bool vec_ok = allow_vec_exchange &&
                    _mpiInfo->has_all_vlen_mults[_mpiInfo->my_neighbor_index] &&
                    _mpiInfo->has_all_vlen_mults[neigh_idx];

                // Determine size of MPI buffers between neigh_rank and my
                // rank for each grid and create those that are needed.  It
                // is critical that the number, size, and shape of my
                // send/receive buffers match those of the receive/send
                // buffers of my neighbors.  Important: Current algorithm
                // assumes my left neighbor's buffer sizes can be calculated
                // by considering my rank's right side data and vice-versa.
                // Thus, all ranks must have consistent data that contribute
                // to these calculations.
                for (auto gp : gridPtrs) {
                    if (!gp || gp->is_scratch() || gp->is_fixed_size())
                        continue;
                    auto& gname = gp->get_name();
                    bool grid_vec_ok = vec_ok;

                    // Lookup first & last domain indices and calc exchange sizes
                    // for this grid.
                    bool found_delta = false;
                    IdxTuple my_halo_sizes, neigh_halo_sizes;
                    IdxTuple first_inner_idx, last_inner_idx;
                    IdxTuple first_outer_idx, last_outer_idx;
                    for (auto& dim : _dims->_domain_dims.getDims()) {
                        auto& dname = dim.getName();

                        // Only consider domain dims that are used in this grid.
                        if (gp->is_dim_used(dname)) {
                            auto vlen = gp->_get_vec_len(dname);
                            auto lhalo = gp->get_left_halo_size(dname);
                            auto rhalo = gp->get_right_halo_size(dname);

                            // Get domain indices for this grid.  If there
                            // are no more ranks in the given direction,
                            // extend the "outer" index to include the halo
                            // in that direction to make sure all data are
                            // sync'd when using WF tiling.
                            idx_t fidx = gp->get_first_rank_domain_index(dname);
                            idx_t lidx = gp->get_last_rank_domain_index(dname);
                            first_inner_idx.addDimBack(dname, fidx);
                            last_inner_idx.addDimBack(dname, lidx);
                            if (_opts->is_time_tiling()) {
                                if (_opts->is_first_rank(dname))
                                    fidx -= lhalo;
                                if (_opts->is_last_rank(dname))
                                    lidx += rhalo;
                            }
                            first_outer_idx.addDimBack(dname, fidx);
                            last_outer_idx.addDimBack(dname, lidx);

                            // Determine if it is possible to round the
                            // outer indices to vec-multiples. This will
                            // be required to allow full vec exchanges for
                            // this grid. We won't do the actual rounding
                            // yet, because we need to see if it's safe
                            // in all dims.
                            // Need +1 and then -1 trick for last.
                            fidx = round_down_flr(fidx, vlen);
                            lidx = round_up_flr(lidx + 1, vlen) - 1;
                            if (fidx < gp->get_first_rank_alloc_index(dname))
                                grid_vec_ok = false;
                            if (lidx > gp->get_last_rank_alloc_index(dname))
                                grid_vec_ok = false;

                            // Determine size of exchange in this dim. This
                            // will be the actual halo size plus any
                            // wave-front shifts. In the current
                            // implementation, we need the wave-front shifts
                            // regardless of whether there is a halo on a
                            // given grid. This is because each
                            // stencil-bundle gets shifted by the WF angles
                            // at each step in the WF.

                            // Neighbor is to the left in this dim.
                            if (neigh_offsets[dname] == MPIInfo::rank_prev) {

                                // Number of points to be added for WFs.
                                auto ext = wf_shifts[dname];

                                // My halo on my left.
                                my_halo_sizes.addDimBack(dname, lhalo + ext);

                                // Neighbor halo on their right.
                                // Assume my right is same as their right.
                                neigh_halo_sizes.addDimBack(dname, rhalo + ext);

                                // Flag that this grid has a neighbor to left or right.
                                found_delta = true;
                            }

                            // Neighbor is to the right in this dim.
                            else if (neigh_offsets[dname] == MPIInfo::rank_next) {

                                // Number of points to be added for WFs.
                                auto ext = wf_shifts[dname];

                                // My halo on my right.
                                my_halo_sizes.addDimBack(dname, rhalo + ext);

                                // Neighbor halo on their left.
                                // Assume my left is same as their left.
                                neigh_halo_sizes.addDimBack(dname, lhalo + ext);

                                // Flag that this grid has a neighbor to left or right.
                                found_delta = true;
                            }

                            // Neighbor in-line in this dim.
                            else {
                                my_halo_sizes.addDimBack(dname, 0);
                                neigh_halo_sizes.addDimBack(dname, 0);
                            }

                        } // domain dims in this grid.
                    } // domain dims.

                    // Is buffer needed?
                    // Example: if this grid is 2D in y-z, but only neighbors are in
                    // x-dim, we don't need any exchange.
                    if (!found_delta) {
                        TRACE_MSG("no halo exchange needed for grid '" << gname <<
                                  "' with rank " << neigh_rank <<
                                  " because the neighbor is not in a direction"
                                  " corresponding to a grid dim");
                        continue; // to next grid.
                    }

                    // Round halo sizes if vectorized exchanges allowed.
                    // Both self and neighbor must be vec-multiples
                    // and outer indices must be vec-mults or extendable
                    // to be so.
                    // TODO: add a heuristic to avoid increasing by a large factor.
                    if (grid_vec_ok) {
                        for (auto& dim : _dims->_domain_dims.getDims()) {
                            auto& dname = dim.getName();
                            if (gp->is_dim_used(dname)) {
                                auto vlen = gp->_get_vec_len(dname);

                                // First index rounded down.
                                auto fidx = first_outer_idx[dname];
                                fidx = round_down_flr(fidx, vlen);
                                first_outer_idx.setVal(dname, fidx);

                                // Last index rounded up.
                                // Need +1 and then -1 trick because it's last, not end.
                                auto lidx = last_outer_idx[dname];
                                lidx = round_up_flr(lidx + 1, vlen) - 1;
                                last_outer_idx.setVal(dname, lidx);

                                // sizes rounded up.
                                my_halo_sizes.setVal(dname, ROUND_UP(my_halo_sizes[dname], vlen));
                                neigh_halo_sizes.setVal(dname, ROUND_UP(neigh_halo_sizes[dname], vlen));

                            } // domain dims in this grid.
                        } // domain dims.
                    }

                    // Make a buffer in both directions (send & receive).
                    for (int bd = 0; bd < MPIBufs::nBufDirs; bd++) {

                        // Begin/end vars to indicate what part
                        // of main grid to read from or write to based on
                        // the current neighbor being processed.
                        IdxTuple copy_begin = gp->get_allocs();
                        IdxTuple copy_end = gp->get_allocs(); // one past last!

                        // Adjust along domain dims in this grid.
                        for (auto& dim : _dims->_domain_dims.getDims()) {
                            auto& dname = dim.getName();
                            if (gp->is_dim_used(dname)) {

                                // Init range to whole rank domain (including
                                // outer halos).  These may be changed below
                                // depending on the neighbor's direction.
                                copy_begin[dname] = first_outer_idx[dname];
                                copy_end[dname] = last_outer_idx[dname] + 1; // end = last + 1.

                                // Neighbor direction in this dim.
                                auto neigh_ofs = neigh_offsets[dname];

                                // Region to read from, i.e., data from inside
                                // this rank's domain to be put into neighbor's
                                // halo. So, use neighbor's halo sizes when
                                // calculating buffer size.
                                if (bd == MPIBufs::bufSend) {

                                    // Neighbor is to the left.
                                    if (neigh_ofs == idx_t(MPIInfo::rank_prev)) {

                                        // Only read slice as wide as halo from beginning.
                                        copy_begin[dname] = first_inner_idx[dname];
                                        copy_end[dname] = first_inner_idx[dname] + neigh_halo_sizes[dname];

                                        // Adjust LHS of interior.
                                        mpi_interior.bb_begin[dname] =
                                            max(mpi_interior.bb_begin[dname], copy_end[dname]);
                                    }

                                    // Neighbor is to the right.
                                    else if (neigh_ofs == idx_t(MPIInfo::rank_next)) {

                                        // Only read slice as wide as halo before end.
                                        copy_begin[dname] = last_inner_idx[dname] + 1 - neigh_halo_sizes[dname];
                                        copy_end[dname] = last_inner_idx[dname] + 1;

                                        // Adjust RHS of interior.
                                        mpi_interior.bb_end[dname] =
                                            min(mpi_interior.bb_end[dname], copy_begin[dname]);
                                    }

                                    // Else, this neighbor is in same posn as I am in this dim,
                                    // so we leave the default begin/end settings.
                                }

                                // Region to write to, i.e., into this rank's halo.
                                // So, use my halo sizes when calculating buffer sizes.
                                else if (bd == MPIBufs::bufRecv) {

                                    // Neighbor is to the left.
                                    if (neigh_ofs == idx_t(MPIInfo::rank_prev)) {

                                        // Only read slice as wide as halo before beginning.
                                        copy_begin[dname] = first_inner_idx[dname] - my_halo_sizes[dname];
                                        copy_end[dname] = first_inner_idx[dname];
                                    }

                                    // Neighbor is to the right.
                                    else if (neigh_ofs == idx_t(MPIInfo::rank_next)) {

                                        // Only read slice as wide as halo after end.
                                        copy_begin[dname] = last_inner_idx[dname] + 1;
                                        copy_end[dname] = last_inner_idx[dname] + 1 + my_halo_sizes[dname];
                                    }

                                    // Else, this neighbor is in same posn as I am in this dim,
                                    // so we leave the default begin/end settings.
                                }
                            } // domain dims in this grid.
                        } // domain dims.

                        // Sizes of buffer in all dims of this grid.
                        // Also, set begin/end value for non-domain dims.
                        IdxTuple buf_sizes = gp->get_allocs();
                        bool buf_vec_ok = grid_vec_ok;
                        for (auto& dname : gp->get_dim_names()) {
                            idx_t dsize = 1;

                            // domain dim?
                            if (_dims->_domain_dims.lookup(dname)) {
                                dsize = copy_end[dname] - copy_begin[dname];

                                // Check whether alignment and size are multiple of vlen.
                                auto vlen = gp->_get_vec_len(dname);
                                if (dsize % vlen != 0)
                                    buf_vec_ok = false;
                                if (imod_flr(copy_begin[dname], vlen) != 0)
                                    buf_vec_ok = false;
                            }

                            // step dim?
                            // Allowing only one step to be exchanged.
                            // TODO: consider exchanging mutiple steps at once for WFs.
                            else if (dname == step_dim) {

                                // Use 0..1 as a place-holder range.
                                // The actual values will be supplied during
                                // halo exchange.
                                copy_begin[dname] = 0;
                                copy_end[dname] = 1;
                            }

                            // misc?
                            // Copy over entire range.
                            // TODO: make dirty flags for misc dims in grids.
                            else {
                                dsize = gp->get_alloc_size(dname);
                                copy_begin[dname] = gp->get_first_misc_index(dname);
                                copy_end[dname] = gp->get_last_misc_index(dname) + 1;
                            }

                            // Save computed size.
                            buf_sizes[dname] = dsize;

                        } // all dims in this grid.

                        // Unique name for buffer based on grid name, direction, and ranks.
                        ostringstream oss;
                        oss << gname;
                        if (bd == MPIBufs::bufSend)
                            oss << "_send_halo_from_" << me << "_to_" << neigh_rank;
                        else if (bd == MPIBufs::bufRecv)
                            oss << "_recv_halo_from_" << neigh_rank << "_to_" << me;
                        string bufname = oss.str();

                        // Does buffer have non-zero size?
                        if (buf_sizes.size() == 0 || buf_sizes.product() == 0) {
                            TRACE_MSG("MPI buffer '" << bufname <<
                                      "' not needed because there is no data to exchange");
                            continue;
                        }

                        // At this point, buf_sizes, copy_begin, and copy_end
                        // should be set for each dim in this grid.

                        // Compute last from end.
                        IdxTuple copy_last = copy_end.subElements(1);

                        // Make MPI data entry for this grid.
                        auto gbp = mpiData.emplace(gname, _mpiInfo);
                        auto& gbi = gbp.first; // iterator from pair returned by emplace().
                        auto& gbv = gbi->second; // value from iterator.
                        auto& buf = gbv.getBuf(MPIBufs::BufDir(bd), neigh_offsets);

                        // Config buffer for this grid.
                        // (But don't allocate storage yet.)
                        buf.begin_pt = copy_begin;
                        buf.last_pt = copy_last;
                        buf.num_pts = buf_sizes;
                        buf.name = bufname;
                        buf.vec_copy_ok = buf_vec_ok;

                        TRACE_MSG("MPI buffer '" << buf.name <<
                                  "' configured for rank at relative offsets " <<
                                  neigh_offsets.subElements(1).makeDimValStr() << " with " <<
                                  buf.num_pts.makeDimValStr(" * ") << " = " << buf.get_size() <<
                                  " element(s) at [" << buf.begin_pt.makeDimValStr() <<
                                  " ... " << buf.last_pt.makeDimValStr() <<
                                  "] with vector-copy " <<
                                  (buf.vec_copy_ok ? "enabled" : "disabled"));
                        num_exchanges[bd]++;
                        num_elems[bd] += buf.get_size();

                    } // send, recv.
                } // grids.
            });   // neighbors.
        TRACE_MSG("number of MPI send buffers on this rank: " << num_exchanges[int(MPIBufs::bufSend)]);
        TRACE_MSG("number of elements in send buffers: " << makeNumStr(num_elems[int(MPIBufs::bufSend)]));
        TRACE_MSG("number of MPI recv buffers on this rank: " << num_exchanges[int(MPIBufs::bufRecv)]);
        TRACE_MSG("number of elements in recv buffers: " << makeNumStr(num_elems[int(MPIBufs::bufRecv)]));

        // Finalize interior BB if there are multiple ranks and overlap enabled.
        if (_env->num_ranks > 1 && settings->overlap_comms) {
            mpi_interior.update_bb("interior", *this, true);
            TRACE_MSG("MPI interior BB: [" << mpi_interior.bb_begin.makeDimValStr() <<
                      " ... " << mpi_interior.bb_end.makeDimValStr() << ")");
        }
        
        // Base ptrs for all alloc'd data.
        // These pointers will be shared by the ones in the grid
        // objects, which will take over ownership when these go
        // out of scope.
        map <int, shared_ptr<char>> _mpi_data_buf;

        // Allocate MPI buffers.
        // Pass 0: count required size, allocate chunk of memory at end.
        // Pass 1: distribute parts of already-allocated memory chunk.
        for (int pass = 0; pass < 2; pass++) {
            TRACE_MSG("allocMpiData pass " << pass << " for " <<
                      mpiData.size() << " MPI buffer set(s)");

            // Count bytes needed and number of buffers for each NUMA node.
            map <int, size_t> npbytes, nbufs;

            // Grids.
            for (auto gp : gridPtrs) {
                if (!gp)
                    continue;
                auto& gname = gp->get_name();
                int numa_pref = gp->get_numa_preferred();

                // MPI bufs for this grid.
                if (mpiData.count(gname)) {
                    auto& grid_mpi_data = mpiData.at(gname);

                    // Visit buffers for each neighbor for this grid.
                    grid_mpi_data.visitNeighbors
                        ([&](const IdxTuple& roffsets,
                             int rank,
                             int idx,
                             MPIBufs& bufs) {

                            // Send and recv.
                            for (int bd = 0; bd < MPIBufs::nBufDirs; bd++) {
                                auto& buf = grid_mpi_data.getBuf(MPIBufs::BufDir(bd), roffsets);
                                if (buf.get_size() == 0)
                                    continue;

                                // Set storage if buffer has been allocated in pass 0.
                                if (pass == 1) {
                                    auto p = _mpi_data_buf[numa_pref];
                                    assert(p);
                                    buf.set_storage(p, npbytes[numa_pref]);
                                }

                                // Determine padded size (also offset to next location).
                                auto sbytes = buf.get_bytes();
                                npbytes[numa_pref] += ROUND_UP(sbytes + _data_buf_pad,
                                                               CACHELINE_BYTES);
                                nbufs[numa_pref]++;
                                if (pass == 0)
                                    TRACE_MSG("  MPI buf '" << buf.name << "' needs " <<
                                              makeByteStr(sbytes) <<
                                              " on NUMA node " << numa_pref);
                            }
                        } );
                }
            }

            // Alloc for each node.
            if (pass == 0)
                _alloc_data(npbytes, nbufs, _mpi_data_buf, "MPI buffer");

        } // MPI passes.
#endif
    }

    // Allocate memory for scratch grids based on number of threads and
    // block sizes.
    void StencilContext::allocScratchData(ostream& os) {
        auto nddims = _dims->_domain_dims.size();
        auto nsdims = _dims->_stencil_dims.size();
        auto step_posn = +Indices::step_posn;

        // Remove any old scratch data.
        freeScratchData(os);

        // Base ptrs for all alloc'd data.
        // This pointer will be shared by the ones in the grid
        // objects, which will take over ownership when it goes
        // out of scope.
        map <int, shared_ptr<char>> _scratch_data_buf;

        // Make sure the right number of threads are set so we
        // have the right number of scratch grids.
        int rthreads = set_region_threads();

        // Delete any existing scratch grids.
        // Create new scratch grids.
        makeScratchGrids(rthreads);

        // Find the max block size across all packs.  TODO: use the specific
        // block size for the pack containing a given scratch grid.
        IdxTuple blksize(_dims->_domain_dims);
        for (auto& sp : stPacks) {
            auto& psettings = sp->getSettings();
            for (int i = 0, j = 0; i < nsdims; i++) {
                if (i == step_posn) continue;
                auto sz = round_up_flr(psettings._block_sizes[i],
                                       _dims->_fold_pts[j]);
                blksize[j] = max(blksize[j], sz);
                j++;
            }
        }
        TRACE_MSG("allocScratchData: max block size across pack(s) is " <<
                  blksize.makeDimValStr(" * "));
        
        // Pass 0: count required size, allocate chunk of memory at end.
        // Pass 1: distribute parts of already-allocated memory chunk.
        for (int pass = 0; pass < 2; pass++) {
            TRACE_MSG("allocScratchData pass " << pass << " for " <<
                      scratchVecs.size() << " set(s) of scratch grids");

            // Count bytes needed and number of grids for each NUMA node.
            map <int, size_t> npbytes, ngrids;

            // Loop through each scratch grid vector.
            for (auto* sgv : scratchVecs) {
                assert(sgv);

                // Loop through each scratch grid in this vector.
                // There will be one for each region thread.
                assert(int(sgv->size()) == rthreads);
                int thr_num = 0;
                for (auto gp : *sgv) {
                    assert(gp);
                    auto& gname = gp->get_name();
                    int numa_pref = gp->get_numa_preferred();

                    // Loop through each domain dim.
                    for (auto& dim : _dims->_domain_dims.getDims()) {
                        auto& dname = dim.getName();

                        if (gp->is_dim_used(dname)) {

                            // Set domain size of grid to block size.
                            gp->_set_domain_size(dname, blksize[dname]);

                            // Pads.
                            // Set via both 'extra' and 'min'; larger result will be used.
                            gp->set_extra_pad_size(dname, _opts->_extra_pad_sizes[dname]);
                            gp->set_min_pad_size(dname, _opts->_min_pad_sizes[dname]);
                        }
                    } // dims.

                    // Set storage if buffer has been allocated.
                    if (pass == 1) {
                        auto p = _scratch_data_buf[numa_pref];
                        assert(p);
                        gp->set_storage(p, npbytes[numa_pref]);
                        TRACE_MSG(gp->make_info_string());
                    }

                    // Determine size used (also offset to next location).
                    size_t nbytes = gp->get_num_storage_bytes();
                    npbytes[numa_pref] += ROUND_UP(nbytes + _data_buf_pad,
                                                   CACHELINE_BYTES);
                    ngrids[numa_pref]++;
                    if (pass == 0)
                        TRACE_MSG(" scratch grid '" << gname << "' for thread " <<
                                  thr_num << " needs " << makeByteStr(nbytes) <<
                                  " on NUMA node " << numa_pref);
                    thr_num++;
                } // scratch grids.
            } // scratch-grid vecs.

            // Alloc for each node.
            if (pass == 0)
                _alloc_data(npbytes, ngrids, _scratch_data_buf, "scratch grid");

        } // scratch-grid passes.
    }
    
    // Set temporal blocking data.
    // This should be called anytime a block size is changed.
    // Must be called after update_grid_info() to ensure
    // angles are properly set.
    void StencilContext::update_block_info() {
        auto& step_dim = _dims->_step_dim;

        // Start w/original temporal setting.
        tb_steps = _opts->_block_sizes[step_dim];
        assert(tb_steps >= 1);

        // Determine max setting based on block sizes.
        // When using temporal blocking, all block sizes
        // across all packs must be the same.
        if (tb_steps > 1) {
            TRACE_MSG("update_block_info: original TB steps = " << tb_steps);
            idx_t max_steps = min(tb_steps, wf_steps);
            TRACE_MSG("update_block_info: max(TB, WF) steps = " << max_steps);

            // Loop through each domain dim.
            for (auto& dim : _dims->_domain_dims.getDims()) {
                auto& dname = dim.getName();

                // Calculate max number of temporal steps in
                // this dim.
                auto bsz = _opts->_block_sizes[dname];
                auto angle = tb_angles[dname];
                if (angle > 0) {
                    auto cur_max = bsz / angle / 2 + 1;
                    TRACE_MSG("update_block_info: max TB steps in dim '" <<
                              dname << "' = " << cur_max <<
                              " due to base block size of " << bsz <<
                              " and TB angle of " << angle);
                    max_steps = min(max_steps, cur_max);
                }
            }
            tb_steps = min(tb_steps, max_steps);
            TRACE_MSG("update_block_info: final TB steps = " << tb_steps);
        }

        // Calc number of shifts based on steps.
        num_tb_shifts = 0;
        if (tb_steps > 1) {

            // Need to shift for each bundle pack.
            assert(stPacks.size() > 0);
            num_tb_shifts = idx_t(stPacks.size()) * tb_steps;
            assert(num_tb_shifts > 1);

            // Don't need to shift first one.
            num_tb_shifts--;
        }
        assert(num_tb_shifts >= 0);

    } // update_block_info().

    // Set non-scratch grid sizes and offsets based on settings.
    // Set wave-front settings.
    // This should be called anytime a setting or rank offset is changed.
    void StencilContext::update_grid_info() {
        assert(_opts);
        auto& step_dim = _dims->_step_dim;

        // If we haven't finished constructing the context, it's too early
        // to do this.
        if (!stPacks.size())
            return;

        // Reset halos to zero.
        max_halos = _dims->_domain_dims;

        // Loop through each non-scratch grid.
        for (auto gp : gridPtrs) {
            assert(gp);

            // Ignore manually-sized grid.
            if (gp->is_fixed_size())
                continue;

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
                    gp->_set_local_offset(dname, 0);

                    // Update max halo across grids, used for temporal angles.
                    max_halos[dname] = max(max_halos[dname], gp->get_left_halo_size(dname));
                    max_halos[dname] = max(max_halos[dname], gp->get_right_halo_size(dname));
                }
            }
        } // grids.

        // Calculate wave-front shifts.
        // See the wavefront diagram in run_solution() for description
        // of angles and extensions.
        tb_steps = _opts->_block_sizes[step_dim]; // use original size; actual may be less.
        assert(tb_steps >= 1);
        wf_steps = _opts->_region_sizes[step_dim];
        wf_steps = max(wf_steps, tb_steps); // round up WF steps if less than TB steps.
        assert(wf_steps >= 1);
        num_wf_shifts = 0;
        if (wf_steps > 1) {

            // Need to shift for each bundle pack.
            assert(stPacks.size() > 0);
            num_wf_shifts = idx_t(stPacks.size()) * wf_steps;
            assert(num_wf_shifts > 1);

            // Don't need to shift first one.
            num_wf_shifts--;
        }
        assert(num_wf_shifts >= 0);

        // Determine whether separate tuners can be used.
        _use_pack_tuners = (tb_steps == 1) && (stPacks.size() > 1);

        // Calculate angles and related settings.
        for (auto& dim : _dims->_domain_dims.getDims()) {
            auto& dname = dim.getName();
            auto rnsize = _opts->_region_sizes[dname];
            auto rksize = _opts->_rank_sizes[dname];
            auto nranks = _opts->_num_ranks[dname];

            // Req'd shift in this dim based on max halos.
            idx_t angle = ROUND_UP(max_halos[dname], _dims->_fold_pts[dname]);
            
            // Determine the max spatial skewing angles for TB.
            // We assume that the block size will always require
            // non-zero angles.
            // TODO: adjust TB angle to zero iff block covers whole
            // rank in given dim. Do this in update_block_info().
            tb_angles.addDimBack(dname, angle);
            
            // Determine the max spatial skewing angles for WF tiling.  We
            // only need non-zero angles if the region size is less than the
            // rank size or there are other ranks in this dim, i.e., if
            // the region covers the *global* domain in a given dim, no
            // wave-front shifting is needed in that dim.
            idx_t wf_angle = 0;
            if (rnsize < rksize || nranks > 0)
                wf_angle = angle;
            wf_angles.addDimBack(dname, wf_angle);
            assert(angle >= 0);

            // Determine the total WF shift to be added in each dim.
            idx_t shifts = wf_angle * num_wf_shifts;
            wf_shifts[dname] = shifts;
            assert(shifts >= 0);

            // Is domain size at least as large as halo + wf_ext in direction
            // when there are multiple ranks?
            auto min_size = max_halos[dname] + shifts;
            if (_opts->_num_ranks[dname] > 1 && rksize < min_size) {
                FORMAT_AND_THROW_YASK_EXCEPTION("Error: rank-domain size of " << rksize << " in '" <<
                                                dname << "' dim is less than minimum size of " << min_size <<
                                                ", which is based on stencil halos and temporal wave-front sizes");
            }

            // If there is another rank to the left, set wave-front
            // extension on the left.
            left_wf_exts[dname] = _opts->is_first_rank(dname) ? 0 : shifts;

            // If there is another rank to the right, set wave-front
            // extension on the right.
            right_wf_exts[dname] = _opts->is_last_rank(dname) ? 0 : shifts;
        }

        // Calculate temporal-block shifts.
        // NB: this will change if/when block sizes change.
        update_block_info();
        
        // Now that wave-front settings are known, we can push this info
        // back to the grids. It's useful to store this redundant info
        // in the grids, because there it's indexed by grid dims instead
        // of domain dims. This makes it faster to do grid indexing.
        for (auto gp : gridPtrs) {
            assert(gp);

            // Ignore manually-sized grid.
            if (gp->is_fixed_size())
                continue;

            // Loop through each domain dim.
            for (auto& dim : _dims->_domain_dims.getDims()) {
                auto& dname = dim.getName();
                if (gp->is_dim_used(dname)) {

                    // Set extensions to be the same as the global ones.
                    gp->_set_left_wf_ext(dname, left_wf_exts[dname]);
                    gp->_set_right_wf_ext(dname, right_wf_exts[dname]);
                }
            }
        }
    } // update_grid_info().

    // Allocate grids and MPI bufs.
    // Initialize some data structures.
    void StencilContext::prepare_solution() {
        auto& step_dim = _dims->_step_dim;

        // Don't continue until all ranks are this far.
        _env->global_barrier();

        ostream& os = get_ostr();
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

        // reset time keepers.
        clear_timers();

        // Adjust all settings before setting MPI buffers or sizing grids.
        // Prints adjusted settings.
        // TODO: print settings again after auto-tuning.
        _opts->adjustSettings(os, _env);

        // Copy current settings to packs.
        // Needed here because settings may have been changed via APIs
        // since last call to prepare_solution().
        // This will wipe out any previous auto-tuning.
        for (auto& sp : stPacks)
            sp->getSettings() = *_opts;

        // Init auto-tuner to run silently during normal operation.
        reset_auto_tuner(true, false);

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

        // Set the number of threads for a region. It should stay this
        // way for top-level OpenMP parallel sections.
        int rthreads = set_region_threads();

        // Run a dummy nested OMP loop to make sure nested threading is
        // initialized.
#ifdef _OPENMP
#pragma omp parallel for
        for (int i = 0; i < rthreads * 100; i++) {

            idx_t dummy = 0;
            set_block_threads();
#pragma omp parallel for reduction(+:dummy)
            for (int j = 0; j < i * 100; j++) {
                dummy += j;
            }
        }
#endif

        // Some grid stats.
        os << endl;
        os << "Num grids: " << gridPtrs.size() << endl;
        os << "Num grids to be updated: " << outputGridPtrs.size() << endl;

        // Set up data based on MPI rank, including grid positions.
        // Update all the grid sizes.
        setupRank();

        // Alloc grids, scratch grids, MPI bufs.
        // This is the order in which preferred NUMA nodes (e.g., HBW mem)
        // will be used.
        // We free the scratch and MPI data first to give grids preference.
        freeScratchData(os);
        freeMpiData(os);
        allocGridData(os);
        allocScratchData(os);
        allocMpiData(os);

        print_info();

    } // prepare_solution().

    void StencilContext::print_info() {
        auto& step_dim = _dims->_step_dim;
        ostream& os = get_ostr();

        // Report total allocation.
        rank_nbytes = get_num_bytes();
        os << "Total allocation in this rank: " <<
            makeByteStr(rank_nbytes) << "\n";
        tot_nbytes = sumOverRanks(rank_nbytes, _env->comm);
        os << "Total overall allocation in " << _env->num_ranks << " rank(s): " <<
            makeByteStr(tot_nbytes) << "\n";

        // Report some stats.
        idx_t dt = _opts->_rank_sizes[step_dim];
        os << "\nProblem sizes in points (from smallest to largest):\n"
            " vector-size:           " << _dims->_fold_pts.makeDimValStr(" * ") << endl <<
            " cluster-size:          " << _dims->_cluster_pts.makeDimValStr(" * ") << endl <<
            " sub-block-size:        " << _opts->_sub_block_sizes.makeDimValStr(" * ") << endl <<
            " sub-block-group-size:  " << _opts->_sub_block_group_sizes.makeDimValStr(" * ") << endl <<
            " block-size:            " << _opts->_block_sizes.makeDimValStr(" * ") << endl <<
            " block-group-size:      " << _opts->_block_group_sizes.makeDimValStr(" * ") << endl <<
            " region-size:           " << _opts->_region_sizes.makeDimValStr(" * ") << endl <<
            " rank-domain-size:      " << _opts->_rank_sizes.makeDimValStr(" * ") << endl <<
            " overall-problem-size:  " << overall_domain_sizes.makeDimValStr(" * ") << endl;
        os << "\nOther settings:\n"
            " yask-version:          " << yask_get_version_string() << endl <<
            " stencil-name:          " << get_name() << endl <<
            " element-size:          " << makeByteStr(get_element_bytes()) << endl <<
#ifdef USE_MPI
            " num-ranks:             " << _opts->_num_ranks.makeDimValStr(" * ") << endl <<
            " rank-indices:          " << _opts->_rank_indices.makeDimValStr() << endl <<
            " rank-domain-offsets:   " << rank_domain_offsets.makeDimValOffsetStr() << endl <<
#endif
            " rank-domain:           " << rank_bb.bb_begin.makeDimValStr() <<
            " ... " << rank_bb.bb_end.subElements(1).makeDimValStr() << endl <<
            " vector-len:            " << VLEN << endl <<
            " extra-padding:         " << _opts->_extra_pad_sizes.makeDimValStr() << endl <<
            " minimum-padding:       " << _opts->_min_pad_sizes.makeDimValStr() << endl <<
            " L1-prefetch-distance:  " << PFD_L1 << endl <<
            " L2-prefetch-distance:  " << PFD_L2 << endl <<
            " max-halos:             " << max_halos.makeDimValStr() << endl;
        if (num_wf_shifts > 0) {
            os <<
                " wave-front-angles:     " << wf_angles.makeDimValStr() << endl <<
                " num-wave-front-shifts: " << num_wf_shifts << endl <<
                " wave-front-shift-lens: " << wf_shifts.makeDimValStr() << endl <<
                " left-wave-front-exts:  " << left_wf_exts.makeDimValStr() << endl <<
                " right-wave-front-exts: " << right_wf_exts.makeDimValStr() << endl <<
                " ext-rank-domain:       " << ext_bb.bb_begin.makeDimValStr() <<
                " ... " << ext_bb.bb_end.subElements(1).makeDimValStr() << endl;
        }
        os << endl;

        // Info about eqs, packs and bundles.
        os << "Num stencil packs:      " << stPacks.size() << endl;
        os << "Num stencil bundles:    " << stBundles.size() << endl;
        os << "Num stencil equations:  " << NUM_STENCIL_EQS << endl;

#if NUM_STENCIL_EQS

        // sums across bundles for this rank.
        rank_numWrites_1t = 0;
        rank_reads_1t = 0;
        rank_numFpOps_1t = 0;

        for (auto& sp : stPacks) {
            auto& pbb = sp->getBB();
            os << "Pack '" << sp->get_name() << "':\n" <<
                " num bundles:                 " << sp->size() << endl <<
                " sub-domain scope:            " << pbb.bb_begin.makeDimValStr() <<
                " ... " << pbb.bb_end.subElements(1).makeDimValStr() << endl;

            for (auto* sg : *sp) {
                idx_t updates1 = 0, reads1 = 0, fpops1 = 0;

                // Loop through all the needed bundles to
                // count stats for scratch bundles.
                // Does not count extra ops needed in scratch halos
                // since this varies depending on block size.
                auto sg_list = sg->get_reqd_bundles();
                for (auto* rsg : sg_list) {
                    updates1 += rsg->get_scalar_points_written();
                    reads1 += rsg->get_scalar_points_read();
                    fpops1 += rsg->get_scalar_fp_ops();
                }

                auto& bb = sg->getBB();
                idx_t updates_domain = updates1 * bb.bb_num_points;
                rank_numWrites_1t += updates_domain;
                idx_t reads_domain = reads1 * bb.bb_num_points;
                rank_reads_1t += reads_domain;
                idx_t fpops_domain = fpops1 * bb.bb_num_points;
                rank_numFpOps_1t += fpops_domain;

                os << " Bundle '" << sg->get_name() << "':\n" <<
                    "  scratch bundles:            " << (sg_list.size() - 1) << endl <<
                    "  sub-domain scope:           " << bb.bb_begin.makeDimValStr() <<
                    " ... " << bb.bb_end.subElements(1).makeDimValStr() << endl <<
                    "  sub-domain size:            " << bb.bb_len.makeDimValStr(" * ") << endl <<
                    "  valid points in sub domain: " << makeNumStr(bb.bb_num_points) << endl <<
                    "  rectangles in sub domain:   " << sg->getBBs().size() << endl <<
                    "  grid-updates per point:     " << updates1 << endl <<
                    "  grid-updates in sub-domain: " << makeNumStr(updates_domain) << endl <<
                    "  grid-reads per point:       " << reads1 << endl <<
                    "  grid-reads in sub-domain:   " << makeNumStr(reads_domain) << endl <<
                    "  est FP-ops per point:       " << fpops1 << endl <<
                    "  est FP-ops in sub-domain:   " << makeNumStr(fpops_domain) << endl;
                os << "  input-grids:                ";
                int i = 0;
                for (auto gp : sg->inputGridPtrs) {
                    if (i++) os << ", ";
                    os << gp->get_name();
                }
                os << "\n  output-grids:               ";
                i = 0;
                for (auto gp : sg->outputGridPtrs) {
                    if (i++) os << ", ";
                    os << gp->get_name();
                }
                os << endl;
            } // bundles.
        } // packs.

        // Various metrics for amount of work.
        rank_numWrites_dt = rank_numWrites_1t * dt;
        tot_numWrites_1t = sumOverRanks(rank_numWrites_1t, _env->comm);
        tot_numWrites_dt = tot_numWrites_1t * dt;

        rank_reads_dt = rank_reads_1t * dt;
        tot_reads_1t = sumOverRanks(rank_reads_1t, _env->comm);
        tot_reads_dt = tot_reads_1t * dt;

        rank_numFpOps_dt = rank_numFpOps_1t * dt;
        tot_numFpOps_1t = sumOverRanks(rank_numFpOps_1t, _env->comm);
        tot_numFpOps_dt = tot_numFpOps_1t * dt;

        rank_domain_1t = rank_bb.bb_num_points;
        rank_domain_dt = rank_domain_1t * dt; // same as _opts->_rank_sizes.product();
        tot_domain_1t = sumOverRanks(rank_domain_1t, _env->comm);
        tot_domain_dt = tot_domain_1t * dt;

        // Print some more stats.
        os << endl <<
            "Amount-of-work stats:\n" <<
            " domain-size in this rank for one time-step: " <<
            makeNumStr(rank_domain_1t) << endl <<
            " overall-problem-size in all ranks for one time-step: " <<
            makeNumStr(tot_domain_1t) << endl <<
            endl <<
            " num-writes-required in this rank for one time-step: " <<
            makeNumStr(rank_numWrites_1t) << endl <<
            " num-writes-required in all ranks for one time-step: " <<
            makeNumStr(tot_numWrites_1t) << endl <<
            endl <<
            " num-reads-required in this rank for one time-step: " <<
            makeNumStr(rank_reads_1t) << endl <<
            " num-reads-required in all ranks for one time-step: " <<
            makeNumStr(tot_reads_1t) << endl <<
            endl <<
            " est-FP-ops in this rank for one time-step: " <<
            makeNumStr(rank_numFpOps_1t) << endl <<
            " est-FP-ops in all ranks for one time-step: " <<
            makeNumStr(tot_numFpOps_1t) << endl <<
            endl;

        if (dt > 1) {
            os <<
                " domain-size in this rank for all time-steps: " <<
                makeNumStr(rank_domain_dt) << endl <<
                " overall-problem-size in all ranks for all time-steps: " <<
                makeNumStr(tot_domain_dt) << endl <<
                endl <<
                " num-writes-required in this rank for all time-steps: " <<
                makeNumStr(rank_numWrites_dt) << endl <<
                " num-writes-required in all ranks for all time-steps: " <<
                makeNumStr(tot_numWrites_dt) << endl <<
                endl <<
                " num-reads-required in this rank for all time-steps: " <<
                makeNumStr(rank_reads_dt) << endl <<
                " num-reads-required in all ranks for all time-steps: " <<
                makeNumStr(tot_reads_dt) << endl <<
                endl <<
                " est-FP-ops in this rank for all time-steps: " <<
                makeNumStr(rank_numFpOps_dt) << endl <<
                " est-FP-ops in all ranks for all time-steps: " <<
                makeNumStr(tot_numFpOps_dt) << endl <<
                endl;
        }
        os <<
            "Notes:\n"
            " Domain-sizes and overall-problem-sizes are based on rank-domain sizes\n"
            "  and number of ranks regardless of number of grids or sub-domains.\n"
            " Num-writes-required is based on sum of grid-updates in sub-domain across stencil-bundle(s).\n"
            " Num-reads-required is based on sum of grid-reads in sub-domain across stencil-bundle(s).\n"
            " Est-FP-ops are based on sum of est-FP-ops in sub-domain across stencil-bundle(s).\n"
            "\n";
#endif
    }

    // Dealloc grids, etc.
    void StencilContext::end_solution() {

        // Final halo exchange (usually not needed).
        exchange_halos();

        // Release any MPI data.
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

    // Set the bounding-box for each stencil-bundle and whole domain.
    void StencilContext::find_bounding_boxes()
    {
        ostream& os = get_ostr();

        // Rank BB is based only on rank offsets and rank domain sizes.
        rank_bb.bb_begin = rank_domain_offsets;
        rank_bb.bb_end = rank_domain_offsets.addElements(_opts->_rank_sizes, false);
        rank_bb.update_bb("rank", *this, true, &os);

        // BB may be extended for wave-fronts.
        ext_bb.bb_begin = rank_bb.bb_begin.subElements(left_wf_exts);
        ext_bb.bb_end = rank_bb.bb_end.addElements(right_wf_exts);
        ext_bb.update_bb("extended-rank", *this, true);

        // Find BB for each pack.
        for (auto sp : stPacks) {
            auto& spbb = sp->getBB();
            spbb.bb_begin = _dims->_domain_dims;
            spbb.bb_end = _dims->_domain_dims;

            // Find BB for each bundle in this pack.
            for (auto sb : *sp) {

                // Find bundle BB.
                sb->find_bounding_box();
                auto& sbbb = sb->getBB();

                // Expand pack BB to encompass bundle BB.
                spbb.bb_begin = spbb.bb_begin.minElements(sbbb.bb_begin);
                spbb.bb_end = spbb.bb_end.maxElements(sbbb.bb_end);
            }
            spbb.update_bb(sp->get_name(), *this, false);
        }

        // Init MPI interior to extended BB.
        mpi_interior = ext_bb;
    }

    // Find the bounding-boxes for this bundle in this rank.
    void StencilBundleBase::find_bounding_box() {
        StencilContext& context = *_generic_context;
        ostream& os = context.get_ostr();
        auto settings = context.get_settings();
        auto dims = context.get_dims();
        auto& domain_dims = dims->_domain_dims;
        auto& step_dim = dims->_step_dim;
        auto& stencil_dims = dims->_stencil_dims;
        auto nddims = domain_dims.size();
        auto nsdims = stencil_dims.size();
        TRACE_MSG3(get_name() << ".find_bounding_box()...");

        // First, find an overall BB around all the
        // valid points in the bundle.

        // Init min vars w/max val and vice-versa.
        Indices min_pts(idx_max, nsdims);
        Indices max_pts(idx_min, nsdims);
        idx_t npts = 0;

        // Begin, end tuples. Use 'ext_bb' to scan across domain in this
        // rank including any extensions for wave-fronts.
        IdxTuple begin(stencil_dims);
        begin.setVals(context.ext_bb.bb_begin, false);
        begin[step_dim] = 0;
        IdxTuple end(stencil_dims);
        end.setVals(context.ext_bb.bb_end, false);
        end[step_dim] = 1;      // one time-step only.

        // Indices needed for the generated 'misc' loops.
        ScanIndices misc_idxs(*dims, false, 0);
        misc_idxs.begin = begin;
        misc_idxs.end = end;

        // Define misc-loop function.  Since step is always 1, we ignore
        // misc_stop.  Update only if point is in domain for this bundle.
#define misc_fn(misc_idxs) do {                                 \
            if (is_in_valid_domain(misc_idxs.start)) {          \
                min_pts = min_pts.minElements(misc_idxs.start); \
                max_pts = max_pts.maxElements(misc_idxs.start); \
                npts++;                                         \
            } } while(0)
        
        // Define OMP reductions to be used in generated code.
#define OMP_PRAGMA_SUFFIX reduction(+:npts)     \
            reduction(min_idxs:min_pts)         \
            reduction(max_idxs:max_pts)

        // Scan through n-D space.  This scan sets min_pts & max_pts for all
        // stencil dims (including step dim) and npts to the number of valid
        // points.
#include "yask_misc_loops.hpp"
#undef misc_fn

        // Init bb vars to ensure they contain correct dims.
        _bundle_bb.bb_begin = domain_dims;
        _bundle_bb.bb_end = domain_dims;

        // If any points, set begin vars to min indices and end vars to one
        // beyond max indices.
        if (npts) {
            IdxTuple tmp(stencil_dims); // create tuple w/stencil dims.
            min_pts.setTupleVals(tmp);  // convert min_pts to tuple.
            _bundle_bb.bb_begin.setVals(tmp, false); // set bb_begin to domain dims of min_pts.

            max_pts.setTupleVals(tmp); // convert min_pts to tuple.
            _bundle_bb.bb_end.setVals(tmp, false); // set bb_end to domain dims of max_pts.
            _bundle_bb.bb_end = _bundle_bb.bb_end.addElements(1); // end = last + 1.
        }

        // No points, just set to zero.
        else {
            _bundle_bb.bb_begin.setValsSame(0);
            _bundle_bb.bb_end.setValsSame(1);
        }
        _bundle_bb.bb_num_points = npts;

        // Finalize overall BB.
        _bundle_bb.update_bb(get_name(), context, false);

        // If the BB is full (solid) or completely empty, this BB is the only bb.
        if (_bundle_bb.bb_is_full || !npts) {
            TRACE_MSG3("adding 1 sub-BB: [" << _bundle_bb.bb_begin.makeDimValStr() <<
                       " ... " << _bundle_bb.bb_end.makeDimValStr() << ")");

            // Add it to the list, and we're done.
            _bb_list.push_back(_bundle_bb);
        }

        // Create list of full BBs (non-overlapping & with no invalid
        // points) inside overall BB.
        else {

            // Divide the overall BB into a slice for each thread
            // across the outer dim.
            const int odim = 0;
            idx_t outer_len = _bundle_bb.bb_len[odim];
            idx_t nthreads = omp_get_max_threads();
            idx_t len_per_thr = CEIL_DIV(outer_len, nthreads);

            // List of BBs for each thread.
            BBList bb_lists[nthreads];

            // Run rect-finding code on each thread.
            // When these are done, we will merge the
            // rects from all threads.
#pragma omp parallel for
            for (int n = 0; n < nthreads; n++) {
                auto& cur_bb_list = bb_lists[n];

                // Begin and end of this slice.
                IdxTuple slice_begin(_bundle_bb.bb_begin);
                slice_begin[odim] += n * len_per_thr;
                IdxTuple slice_end(_bundle_bb.bb_end);
                slice_end[odim] = min(slice_end[odim], slice_begin[odim] + len_per_thr);
                if (slice_end[odim] <= slice_begin[odim])
                    continue;

                // Construct len of slice in all dims.
                IdxTuple slice_len = slice_end.subElements(slice_begin);
                
                // Visit all points in slice, looking for a new
                // valid starting point, 'pt'.
                IdxTuple spt(stencil_dims);
                IdxTuple dpt(domain_dims);
                slice_len.visitAllPoints
                    ([&](const IdxTuple& ofs, size_t idx) {

                        // Find global point from 'ofs'.
                        dpt = slice_begin.addElements(ofs); // domain tuple.
                        spt.setVals(dpt, false);            // stencil tuple.
                        Indices pt(spt);                    // stencil indices.

                        // Valid point must be in sub-domain and
                        // not seen before in this slice.
                        bool is_valid = is_in_valid_domain(pt);
                        if (is_valid) {
                            for (auto& bb : cur_bb_list) {
                                if (bb.is_in_bb(dpt)) {
                                    is_valid = false;
                                    break;
                                }
                            }
                        }
                        
                        // Process this new rect starting at 'pt'.
                        if (is_valid) {
                            IdxTuple espt(stencil_dims);
                            IdxTuple edpt(domain_dims);

                            // Scan from 'pt' to end of this slice
                            // looking for end of rect.
                            IdxTuple scan_len = slice_end.subElements(dpt);

                            // Repeat scan until no adjustment is made.
                            bool do_scan = true;
                            while (do_scan) {
                                do_scan = false;

                                TRACE_MSG3("scanning " << scan_len.makeDimValStr(" * ") <<
                                           " starting at " << dpt.makeDimValStr());
                                scan_len.visitAllPoints
                                    ([&](const IdxTuple& eofs, size_t eidx) {

                                        // Make sure scan_len range is observed.
                                        for (int i = 0; i < nddims; i++)
                                            assert(eofs[i] < scan_len[i]);

                                        // Find global point from 'eofs'.
                                        edpt = dpt.addElements(eofs); // domain tuple.
                                        espt.setVals(edpt, false); // stencil tuple.
                                        Indices ept(espt); // stencil indices.

                                        // Valid point must be in sub-domain and
                                        // not seen before in this slice.
                                        bool is_evalid = is_in_valid_domain(ept);
                                        if (is_evalid) {
                                            for (auto& bb : cur_bb_list) {
                                                if (bb.is_in_bb(edpt)) {
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
                                            for (int i = 0; i < nddims; i++) {

                                                // Beyond starting point in this dim?
                                                if (edpt[i] > dpt[i]) {
                                                    scan_len[i] = edpt[i] - dpt[i];

                                                    // restart scan for
                                                    // remaining dims.
                                                    // TODO: be smarter
                                                    // about where to
                                                    // restart scan.
                                                    if (i < nddims - 1)
                                                        do_scan = true;

                                                    return false; // stop this scan.
                                                }
                                            }
                                        }

                                        return true; // keep looking for invalid point.
                                    }); // Looking for invalid point.
                            } // while scan is adjusted.
                            TRACE_MSG3("found BB " << scan_len.makeDimValStr(" * ") <<
                                       " starting at " << dpt.makeDimValStr());

                            // 'scan_len' now contains sizes of the new BB.
                            BoundingBox new_bb;
                            new_bb.bb_begin = dpt;
                            new_bb.bb_end = dpt.addElements(scan_len);
                            new_bb.update_bb("sub-bb", context, true);
                            cur_bb_list.push_back(new_bb);
                            
                        } // new rect found.

                        return true;  // from labmda; keep looking.
                    }); // Looking for new rects.
            } // threads/slices.

            // Collect BBs in all slices.
            // TODO: merge in a binary tree instead of sequentially.
            for (int n = 0; n < nthreads; n++) {
                auto& cur_bb_list = bb_lists[n];
                TRACE_MSG3("processing " << cur_bb_list.size() <<
                           " sub-BB(s) in bundle '" << get_name() <<
                           "' from thread " << n);

                // BBs in slice 'n'.
                for (auto& bbn : cur_bb_list) {
                    TRACE_MSG3(" sub-BB: [" << bbn.bb_begin.makeDimValStr() <<
                               " ... " << bbn.bb_end.makeDimValStr() << ")");

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
                            TRACE_MSG3("  merging to form [" << bb.bb_begin.makeDimValStr() <<
                                       " ... " << bb.bb_end.makeDimValStr() << ")");
                            break;
                        }
                    }

                    // If not merged, add 'bbn' as new.
                    if (!do_merge) {
                        _bb_list.push_back(bbn);
                        TRACE_MSG3("  adding as final sub-BB #" << _bb_list.size());
                    }
                }
            }
        } // Finding constituent rects.
    }

    // Compute convenience values for a bounding-box.
    void BoundingBox::update_bb(const string& name,
                                StencilContext& context,
                                bool force_full,
                                ostream* os) {

        auto dims = context.get_dims();
        auto& domain_dims = dims->_domain_dims;
        bb_len = bb_end.subElements(bb_begin);
        bb_size = bb_len.product();
        if (force_full)
            bb_num_points = bb_size;

        // Solid rectangle?
        bb_is_full = true;
        if (bb_num_points != bb_size) {
            if (os)
                *os << "Note: '" << name << "' domain has only " <<
                    makeNumStr(bb_num_points) <<
                    " valid point(s) inside its bounding-box of " <<
                    makeNumStr(bb_size) <<
                    " point(s); multiple sub-boxes will be used.\n";
            bb_is_full = false;
        }

        // Does everything start on a vector-length boundary?
        bb_is_aligned = true;
        for (auto& dim : domain_dims.getDims()) {
            auto& dname = dim.getName();
            if ((bb_begin[dname] - context.rank_domain_offsets[dname]) %
                dims->_fold_pts[dname] != 0) {
                if (os)
                    *os << "Note: '" << name << "' domain"
                        " has one or more starting edges not on vector boundaries;"
                        " masked calculations will be used in peel and remainder sub-blocks.\n";
                bb_is_aligned = false;
                break;
            }
        }

        // Lengths are cluster-length multiples?
        bb_is_cluster_mult = true;
        for (auto& dim : domain_dims.getDims()) {
            auto& dname = dim.getName();
            if (bb_len[dname] % dims->_cluster_pts[dname] != 0) {
                if (bb_is_full && bb_is_aligned)
                    if (os && bb_is_aligned)
                        *os << "Note: '" << name << "' domain"
                            " has one or more sizes that are not vector-cluster multiples;"
                            " masked calculations will be used in peel and remainder sub-blocks.\n";
                bb_is_cluster_mult = false;
                break;
            }
        }

        // All done.
        bb_valid = true;
    }

} // namespace yask.
