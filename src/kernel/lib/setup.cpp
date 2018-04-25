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

// This file contains implementations of StencilContext methods
// specific to the preparation steps.

#include "yask.hpp"
using namespace std;

namespace yask {

    // Init MPI-related vars and other vars related to my rank's place in
    // the global problem: rank index, offset, etc.  Need to call this even
    // if not using MPI to properly init these vars.  Called from
    // prepare_solution(), so it doesn't normally need to be called from user code.
    void StencilContext::setupRank() {
        ostream& os = get_ostr();
        auto& step_dim = _dims->_step_dim;
        auto me = _env->my_rank;

        // Check ranks.
        idx_t req_ranks = _opts->_num_ranks.product();
        if (req_ranks != _env->num_ranks) {
            THROW_YASK_EXCEPTION("error: " << req_ranks << " rank(s) requested (" <<
                _opts->_num_ranks.makeDimValStr(" * ") << "), but " <<
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

        // Init coords for this rank.
        for (int i = 0; i < num_ddims; i++)
            coords[me][i] = _opts->_rank_indices[i];

        // A table of rank-domain sizes for everyone.
        idx_t rsizes[_env->num_ranks][num_ddims];

        // Init sizes for this rank.
        for (int di = 0; di < num_ddims; di++) {
            auto& dname = _opts->_rank_indices.getDimName(di);
            rsizes[me][di] = _opts->_rank_sizes[dname];
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
        int num_neighbors = 0;
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
                    THROW_YASK_EXCEPTION("Internal error: distance to own rank == " << mandist);
            }

            // Someone else.
            else {
                if (mandist == 0)
                    THROW_YASK_EXCEPTION("Error: ranks " << me <<
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
                    // intersect with this rank, including myself.
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
                                THROW_YASK_EXCEPTION("Error: rank " << rn << " and " << me <<
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

        // Set offsets in grids and find WF extensions
        // based on the grids' halos.
        update_grids();

        // Determine bounding-boxes for all bundles.
        // This must be done after finding WF extensions.
        find_bounding_boxes();

    } // setupRank.

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
            }

            // Alloc for each node.
            if (pass == 0)
                _alloc_data(npbytes, ngrids, _grid_data_buf, "grid");

        } // grid passes.
    };
    
    // Create MPI and allocate buffers.
    void StencilContext::allocMpiData(ostream& os) {

        // Remove any old MPI data.
        freeMpiData(os);

#ifdef USE_MPI

        int num_exchanges = 0;
        auto me = _env->my_rank;
        
        // Need to determine the size and shape of all MPI buffers.
        // Visit all neighbors of this rank.
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
        
                // Determine size of MPI buffers between neigh_rank and my rank
                // for each grid and create those that are needed.
                for (auto gp : gridPtrs) {
                    if (!gp)
                        continue;
                    auto& gname = gp->get_name();

                    // Lookup first & last domain indices and calc exchange sizes
                    // for this grid.
                    bool found_delta = false;
                    IdxTuple my_halo_sizes, neigh_halo_sizes;
                    IdxTuple first_inner_idx, last_inner_idx;
                    IdxTuple first_outer_idx, last_outer_idx;
                    for (auto& dim : _dims->_domain_dims.getDims()) {
                        auto& dname = dim.getName();
                        if (gp->is_dim_used(dname)) {

                            // Get domain indices for this grid.
                            // If there are no more ranks in the given direction, extend
                            // the index into the outer halo to make sure all data are sync'd.
                            // This is critical for WFs.
                            idx_t fidx = gp->get_first_rank_domain_index(dname);
                            idx_t lidx = gp->get_last_rank_domain_index(dname);
                            first_inner_idx.addDimBack(dname, fidx);
                            last_inner_idx.addDimBack(dname, lidx);
                            if (_opts->is_first_rank(dname))
                                fidx -= gp->get_left_halo_size(dname);
                            if (_opts->is_last_rank(dname))
                                lidx += gp->get_right_halo_size(dname);
                            first_outer_idx.addDimBack(dname, fidx);
                            last_outer_idx.addDimBack(dname, lidx);

                            // Determine size of exchange. This will be the actual halo size
                            // plus any wave-front extensions. In the current implementation,
                            // we need the wave-front extensions regardless of whether there
                            // is a halo on a given grid. This is because each stencil-bundle
                            // gets shifted by the WF angles at each step in the WF.

                            // Neighbor is to the left.
                            if (neigh_offsets[dname] == MPIInfo::rank_prev) {
                                auto ext = left_wf_exts[dname];

                                // my halo.
                                auto halo_size = gp->get_left_halo_size(dname);
                                halo_size += ext;
                                my_halo_sizes.addDimBack(dname, halo_size);

                                // neighbor halo.
                                halo_size = gp->get_right_halo_size(dname); // their right is on my left.
                                halo_size += ext;
                                neigh_halo_sizes.addDimBack(dname, halo_size);
                            }

                            // Neighbor is to the right.
                            else if (neigh_offsets[dname] == MPIInfo::rank_next) {
                                auto ext = right_wf_exts[dname];

                                // my halo.
                                auto halo_size = gp->get_right_halo_size(dname);
                                halo_size += ext;
                                my_halo_sizes.addDimBack(dname, halo_size);

                                // neighbor halo.
                                halo_size = gp->get_left_halo_size(dname); // their left is on my right.
                                halo_size += ext;
                                neigh_halo_sizes.addDimBack(dname, halo_size);
                            }

                            // Neighbor in-line.
                            else {
                                my_halo_sizes.addDimBack(dname, 0);
                                neigh_halo_sizes.addDimBack(dname, 0);
                            }

                            // Vectorized exchange allowed based on domain sizes?
                            // Both my rank and neighbor rank must have all domain sizes
                            // of vector multiples.
                            bool vec_ok = allow_vec_exchange &&
                                _mpiInfo->has_all_vlen_mults[_mpiInfo->my_neighbor_index] &&
                                _mpiInfo->has_all_vlen_mults[neigh_idx];
                            
                            // Round up halo sizes if vectorized exchanges allowed.
                            // TODO: add a heuristic to avoid increasing by a large factor.
                            if (vec_ok) {
                                auto vec_size = _dims->_fold_pts[dname];
                                my_halo_sizes.setVal(dname, ROUND_UP(my_halo_sizes[dname], vec_size));
                                neigh_halo_sizes.setVal(dname, ROUND_UP(neigh_halo_sizes[dname], vec_size));
                            }
                            
                            // Is this neighbor before or after me in this domain direction?
                            if (neigh_offsets[dname] != MPIInfo::rank_self)
                                found_delta = true;
                        }
                    }

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

                    // Make a buffer in both directions (send & receive).
                    for (int bd = 0; bd < MPIBufs::nBufDirs; bd++) {

                        // Begin/end vars to indicate what part
                        // of main grid to read from or write to based on
                        // the current neighbor being processed.
                        IdxTuple copy_begin = gp->get_allocs();
                        IdxTuple copy_end = gp->get_allocs();

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
                                // halo.
                                if (bd == MPIBufs::bufSend) {

                                    // Neighbor is to the left.
                                    if (neigh_ofs == idx_t(MPIInfo::rank_prev)) {

                                        // Only read slice as wide as halo from beginning.
                                        copy_end[dname] = first_inner_idx[dname] + neigh_halo_sizes[dname];
                                    }
                            
                                    // Neighbor is to the right.
                                    else if (neigh_ofs == idx_t(MPIInfo::rank_next)) {
                                    
                                        // Only read slice as wide as halo before end.
                                        copy_begin[dname] = last_inner_idx[dname] + 1 - neigh_halo_sizes[dname];
                                    }
                            
                                    // Else, this neighbor is in same posn as I am in this dim,
                                    // so we leave the default begin/end settings.
                                }
                        
                                // Region to write to, i.e., into this rank's halo.
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
                        bool vlen_mults = true;
                        for (auto& dname : gp->get_dim_names()) {
                            idx_t dsize = 1;

                            // domain dim?
                            if (_dims->_domain_dims.lookup(dname)) {
                                dsize = copy_end[dname] - copy_begin[dname];

                                // Check whether size is multiple of vlen.
                                auto vlen = _dims->_fold_pts[dname];
                                if (dsize % vlen != 0)
                                    vlen_mults = false;
                            }

                            // step dim?
                            // Allowing only one step to be exchanged.
                            // TODO: consider exchanging mutiple steps at once for WFs.
                            else if (dname == _dims->_step_dim) {

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

                        // Does buffer have non-zero size?
                        if (buf_sizes.size() == 0 || buf_sizes.product() == 0) {
                            TRACE_MSG("no halo exchange needed for grid '" << gname <<
                                      "' with rank " << neigh_rank <<
                                      " because there is no data to exchange");
                            continue;
                        }

                        // At this point, buf_sizes, copy_begin, and copy_end
                        // should be set for each dim in this grid.
                        // Convert end to last.
                        IdxTuple copy_last = copy_end.subElements(1);

                        // Unique name for buffer based on grid name, direction, and ranks.
                        ostringstream oss;
                        oss << gname;
                        if (bd == MPIBufs::bufSend)
                            oss << "_send_halo_from_" << me << "_to_" << neigh_rank;
                        else if (bd == MPIBufs::bufRecv)
                            oss << "_recv_halo_from_" << neigh_rank << "_to_" << me;
                        string bufname = oss.str();

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
                        buf.has_all_vlen_mults = vlen_mults;
                        
                        TRACE_MSG("configured MPI buffer object '" << buf.name <<
                                  "' for rank at relative offsets " <<
                                  neigh_offsets.subElements(1).makeDimValStr() << " with " <<
                                  buf.num_pts.makeDimValStr(" * ") << " = " << buf.get_size() <<
                                  " element(s) at " << buf.begin_pt.makeDimValStr() <<
                                  " ... " << buf.last_pt.makeDimValStr());
                        num_exchanges++;

                    } // send, recv.
                } // grids.
            });   // neighbors.
        TRACE_MSG("number of halo-exchanges needed on this rank: " << num_exchanges);

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
                            gp->_set_domain_size(dname, _opts->_block_sizes[dname]);
                    
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

    // Set non-scratch grid sizes and offsets based on settings.
    // Set wave-front settings.
    // This should be called anytime a setting or rank offset is changed.
    void StencilContext::update_grids()
    {
        assert(_opts);

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

                    // Update max halo across grids, used for wavefront angles.
                    max_halos[dname] = max(max_halos[dname], gp->get_left_halo_size(dname));
                    max_halos[dname] = max(max_halos[dname], gp->get_right_halo_size(dname));
                }
            }
        } // grids.

        // Calculate wave-front settings based on max halos.
        // See the wavefront diagram in run_solution() for description
        // of angles and extensions.
        auto& step_dim = _dims->_step_dim;
        auto wf_steps = _opts->_region_sizes[step_dim];
        num_wf_shifts = 0;
        if (wf_steps > 1)

            // TODO: don't shift for scratch grids.
            num_wf_shifts = max((idx_t(stBundles.size()) * wf_steps) - 1, idx_t(0));
        for (auto& dim : _dims->_domain_dims.getDims()) {
            auto& dname = dim.getName();
            auto rksize = _opts->_rank_sizes[dname];
            auto nranks = _opts->_num_ranks[dname];

            // Determine the max spatial skewing angles for temporal
            // wave-fronts based on the max halos.  We only need non-zero
            // angles if the region size is less than the rank size and
            // there are no other ranks in this dim, i.e., if the region
            // covers the global domain in a given dim, no wave-front is
            // needed in that dim.  TODO: make rounding-up an option.
            idx_t angle = 0;
            if (_opts->_region_sizes[dname] < rksize || nranks > 0)
                angle = ROUND_UP(max_halos[dname], _dims->_cluster_pts[dname]);
            wf_angles[dname] = angle;

            // Determine the total WF shift to be added in each dim.
            idx_t shifts = angle * num_wf_shifts;
            wf_shifts[dname] = shifts;

            // Is domain size at least as large as halo + wf_ext in direction
            // when there are multiple ranks?
            auto min_size = max_halos[dname] + shifts;
            if (_opts->_num_ranks[dname] > 1 && rksize < min_size) {
                THROW_YASK_EXCEPTION("Error: rank-domain size of " << rksize << " in '" <<
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
    }
    
    // Allocate grids and MPI bufs.
    // Initialize some data structures.
    void StencilContext::prepare_solution() {
        auto& step_dim = _dims->_step_dim;

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
        
        // reset time keepers.
        clear_timers();

        // Init auto-tuner to run silently during normal operation.
        _at.clear(false, false);

        // Adjust all settings before setting MPI buffers or sizing grids.
        // Prints final settings.
        // TODO: print settings again after auto-tuning.
        _opts->adjustSettings(os, _env);

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
            " overall-problem-size:  " << overall_domain_sizes.makeDimValStr(" * ") << endl <<
            endl <<
            "Other settings:\n"
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
        
        // sums across bundles for this rank.
        rank_numWrites_1t = 0;
        rank_reads_1t = 0;
        rank_numFpOps_1t = 0;
        os << "Num stencil bundles: " << stBundles.size() << endl;
        for (auto* sg : stBundles) {
            idx_t updates1 = sg->get_scalar_points_written();
            idx_t updates_domain = updates1 * sg->bb_num_points;
            rank_numWrites_1t += updates_domain;
            idx_t reads1 = sg->get_scalar_points_read();
            idx_t reads_domain = reads1 * sg->bb_num_points;
            rank_reads_1t += reads_domain;
            idx_t fpops1 = sg->get_scalar_fp_ops();
            idx_t fpops_domain = fpops1 * sg->bb_num_points;
            rank_numFpOps_1t += fpops_domain;
            os << "Stats for bundle '" << sg->get_name() << "':\n" <<
                " sub-domain:                 " << sg->bb_begin.makeDimValStr() <<
                " ... " << sg->bb_end.subElements(1).makeDimValStr() << endl <<
                " sub-domain size:            " << sg->bb_len.makeDimValStr(" * ") << endl <<
                " valid points in sub domain: " << makeNumStr(sg->bb_num_points) << endl <<
                " grid-updates per point:     " << updates1 << endl <<
                " grid-updates in sub-domain: " << makeNumStr(updates_domain) << endl <<
                " grid-reads per point:       " << reads1 << endl <<
                " grid-reads in sub-domain:   " << makeNumStr(reads_domain) << endl <<
                " est FP-ops per point:       " << fpops1 << endl <<
                " est FP-ops in sub-domain:   " << makeNumStr(fpops_domain) << endl;
        }

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
    }

    // Dealloc grids, etc.
    void StencilContext::end_solution() {

        // Final halo exchange.
        exchange_halos_all();

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

        // Solid rectangle?
        bb_is_full = true;
        if (bb_num_points != bb_size) {
            os << "Warning: '" << name << "' domain has only " <<
                makeNumStr(bb_num_points) <<
                " valid point(s) inside its bounding-box of " <<
                makeNumStr(bb_size) <<
                " point(s); slower scalar calculations will be used.\n";
            bb_is_full = false;
        }

        // Does everything start on a vector-length boundary?
        bb_is_aligned = true;
        for (auto& dim : domain_dims.getDims()) {
            auto& dname = dim.getName();
            if ((bb_begin[dname] - context.rank_domain_offsets[dname]) %
                dims->_fold_pts[dname] != 0) {
                os << "Note: '" << name << "' domain"
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
                    os << "Note: '" << name << "' domain"
                        " has one or more sizes that are not vector-cluster multiples;"
                        " masked calculations will be used in peel and remainder sub-blocks.\n";
                bb_is_cluster_mult = false;
                break;
            }
        }

        // All done.
        bb_valid = true;
    }
    
    // Set the bounding-box for each stencil-bundle and whole domain.
    void StencilContext::find_bounding_boxes()
    {
        ostream& os = get_ostr();

        // Rank BB is based only on rank offsets and rank domain sizes.
        rank_bb.bb_begin = rank_domain_offsets;
        rank_bb.bb_end = rank_domain_offsets.addElements(_opts->_rank_sizes, false);
        rank_bb.update_bb(os, "rank", *this, true);

        // Overall BB may be extended for wave-fronts.
        ext_bb.bb_begin = rank_bb.bb_begin.subElements(left_wf_exts);
        ext_bb.bb_end = rank_bb.bb_end.addElements(right_wf_exts);
        ext_bb.update_bb(os, "extended-rank", *this, true);

        // Find BB for each bundle.
        for (auto sg : stBundles)
            sg->find_bounding_box();
    }

} // namespace yask.
