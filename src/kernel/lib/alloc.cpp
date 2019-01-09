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

// This file contains implementations of StencilContext
// specific to the allocating data.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

#ifdef USE_PMEM
    static inline int getnode() {
        #ifdef SYS_getcpu
        int node, status;
       status = syscall(SYS_getcpu, NULL, &node, NULL);
        return (status == -1) ? status : node;
        #else
        return -1; // unavailable
        #endif
    }
#endif

    // Magic numbers for memory types in addition to those for NUMA.
    // TODO: get rid of magic-number scheme.
    constexpr int _shmem_key = 1000;
    constexpr int _pmem_key = 2000; // leave space after this for pmem devices.

    // Alloc 'nbytes' for each requested mem type.
    // Pointers are returned in '_data_buf'.
    // 'ngrids' and 'type' are only used for debug msg.
    void StencilContext::_alloc_data(const map <int, size_t>& nbytes,
                                     const map <int, size_t>& ngrids,
                                     map <int, shared_ptr<char>>& data_buf,
                                     const std::string& type) {
        CONTEXT_VARS(this);

        // Loop through each mem type.
        for (const auto& i : nbytes) {
            int mem_key = i.first;
            size_t nb = i.second;
            size_t ng = ngrids.at(mem_key);

            // Alloc data depending on magic key.
            shared_ptr<char> p;
            os << "Allocating " << makeByteStr(nb) <<
                " for " << ng << " " << type << "(s) ";
            if (mem_key == _shmem_key) {
                os << "using MPI shm...\n" << flush;
                p = shared_shm_alloc<char>(nb, &env->shm_comm, &mpiInfo->halo_win);

                // Get pointer for each neighbor rank.
#ifdef USE_MPI
                int ns = int(mpiInfo->neighborhood_size);
                for (int ni = 0; ni < ns; ni++) {
                    int nr = mpiInfo->my_neighbors.at(ni);
                    if (nr == MPI_PROC_NULL)
                        continue;
                    int sr = mpiInfo->shm_ranks.at(ni);
                    MPI_Aint sz;
                    int dispunit;
                    void* baseptr;
                    MPI_Win_shared_query(mpiInfo->halo_win, sr, &sz,
                                         &dispunit, &baseptr);
                    mpiInfo->halo_buf_ptrs.at(ni) = baseptr;
                    mpiInfo->halo_buf_sizes.at(ni) = sz;
                    TRACE_MSG("MPI shm halo buffer for rank " << nr << " is at " <<
                              baseptr << " for " << makeByteStr(sz));
                }
#endif
            }
            else if (mem_key >= _pmem_key) {
                auto dev_num = mem_key - _pmem_key;
                os << "on PMEM device " << dev_num << "...\n" << flush;
                p = shared_pmem_alloc<char>(nb, dev_num);
            }
            else {
                if (mem_key == yask_numa_none)
                    os << "using default allocator";
                else if (mem_key == yask_numa_local)
                    os << "preferring local NUMA node";
                else if (mem_key == yask_numa_interleave)
                    os << "interleaved across all NUMA nodes";
                else if (mem_key >= 0)
                    os << "preferring NUMA node " << mem_key;
                else
                    os << "using mem policy " << mem_key;
                os << "...\n" << flush;
                p = shared_numa_alloc<char>(nb, mem_key);
            }

            // Save using original key.
            data_buf[mem_key] = p;
            TRACE_MSG("Got memory at " << static_cast<void*>(p.get()));
        }
    }

    // Allocate memory for grids that do not already have storage.
    void StencilContext::allocGridData() {
        CONTEXT_VARS(this);

        // Allocate I/O grids before read-only grids.
        GridPtrs sortedGridPtrs;
        GridPtrSet done;
        for (auto op : outputGridPtrs) {
            sortedGridPtrs.push_back(op);
            done.insert(op);
        }
        for (auto gp : gridPtrs) {
            if (!done.count(gp))
                sortedGridPtrs.push_back(gp);
        }
	done.clear();

#ifdef USE_PMEM
        os << "PMEM grid-allocation priority:" << endl;
        for (auto sp : sortedGridPtrs) {
            os << " '" << sp->get_name() << "'";
            if (done.find(sp)!=done.end())
                os << " (output)";
            os << endl;
        }
#endif
        
        // Base ptrs for all default-alloc'd data.
        // These pointers will be shared by the ones in the grid
        // objects, which will take over ownership when these go
        // out of scope.
        // Key is preferred numa node or -1 for local.
        map <int, shared_ptr<char>> _grid_data_buf;

#ifdef USE_PMEM
        auto preferredNUMASize = opts->_numa_pref_max * 1024*1024*(size_t)1024;
#endif
        
        // Pass 0: assign PMEM node when preferred NUMA node is not enough.
        // Pass 1: count required size for each NUMA node, allocate chunk of memory at end.
        // Pass 2: distribute parts of already-allocated memory chunk.
        for (int pass = 0; pass < 3; pass++) {
            TRACE_MSG("allocGridData pass " << pass << " for " <<
                      gridPtrs.size() << " grid(s)");

            // Count bytes needed and number of grids for each NUMA node.
            map <int, size_t> npbytes, ngrids;

            // Grids.
            for (auto gp : sortedGridPtrs) {
                if (!gp)
                    continue;
                auto& gname = gp->get_name();

                // Grid data.
                // Don't alloc if already done.
                if (!gp->is_storage_allocated()) {
                    int numa_pref = gp->get_numa_preferred();

                    // Set storage if buffer has been allocated in pass 1.
                    if (pass == 2) {
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

                    if (pass == 0) {
#ifdef USE_PMEM
                        if (preferredNUMASize < npbytes[numa_pref])
                            if (getnode() == -1) {
                                os << "Warning: cannot get numa_node information for PMEM allocation;"
                                    " using default numa_pref " << endl;
                            }
                            else

                                // TODO: change this behavior so that it doesn't actually
                                // modify the NUMA pref of the grid.
                                gp->set_numa_preferred(_pmem_key + getnode());
#endif
                    }

                    if (pass == 1)
                        TRACE_MSG(" grid '" << gname << "' needs " << makeByteStr(nbytes) <<
                                  " on NUMA node " << numa_pref);
                }

                // Otherwise, just print existing grid info.
                else if (pass == 1)
                    os << gp->make_info_string() << endl;
            }

            // Reset the counters
            if (pass == 0) {
                npbytes.clear();
                ngrids.clear();
            }

            // Alloc for each node.
            if (pass == 1)
                _alloc_data(npbytes, ngrids, _grid_data_buf, "grid");

        } // grid passes.
    };

    // Determine the size and shape of all MPI buffers.
    // Create buffers and allocate them.
    void StencilContext::allocMpiData() {
        CONTEXT_VARS(this);

        // Remove any old MPI data.
        env->global_barrier();
        freeMpiData();

        // Init interior.
        mpi_interior = ext_bb;
        mpi_interior.bb_valid = false;

#ifdef USE_MPI

        map<int, int> num_exchanges; // send/recv => count.
        map<int, idx_t> num_elems; // send/recv => count.
        auto me = _env->my_rank;

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
                // Always use max dist with WF. Do this because edge
                // and/or corner values may be needed in WF extensions
                // even it not needed w/o WFs.
                // TODO: determine if max is always needed.
                int maxdist = MAX_EXCH_DIST;
                if (wf_steps > 0)
                    maxdist = NUM_STENCIL_DIMS - 1;

                // Manhattan dist. of current neighbor.
                int mandist = _mpiInfo->man_dists.at(neigh_idx);

                // Check distance.
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
                            // sync'd. Critical for temporal tiling.
                            idx_t fidx = gp->get_first_rank_domain_index(dname);
                            idx_t lidx = gp->get_last_rank_domain_index(dname);
                            first_inner_idx.addDimBack(dname, fidx);
                            last_inner_idx.addDimBack(dname, lidx);
                            if (_opts->is_first_rank(dname))
                                fidx -= lhalo; // extend into left halo.
                            if (_opts->is_last_rank(dname))
                                lidx += rhalo; // extend into right halo.
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
                                auto ext = wf_shift_pts[dname];

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
                                auto ext = wf_shift_pts[dname];

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

                                // Min MPI exterior options.
                                idx_t min_ext = opts->_min_exterior;

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
                                        idx_t ext_end = ROUND_UP(first_inner_idx[dname] +
                                                                 max(min_ext, neigh_halo_sizes[dname]),
                                                                 _dims->_fold_pts[dname]);
                                        mpi_interior.bb_begin[dname] =
                                            max(mpi_interior.bb_begin[dname], ext_end);
                                    }

                                    // Neighbor is to the right.
                                    else if (neigh_ofs == idx_t(MPIInfo::rank_next)) {

                                        // Only read slice as wide as halo before end.
                                        copy_begin[dname] = last_inner_idx[dname] + 1 - neigh_halo_sizes[dname];
                                        copy_end[dname] = last_inner_idx[dname] + 1;

                                        // Adjust RHS of interior.
                                        idx_t ext_begin = ROUND_DOWN(last_inner_idx[dname] + 1 -
                                                                     max(min_ext, neigh_halo_sizes[dname]),
                                                                     _dims->_fold_pts[dname]);
                                        mpi_interior.bb_end[dname] =
                                            min(mpi_interior.bb_end[dname], ext_begin);
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
                            // Enable copy over entire allocated range.
                            // May only copy one step when not using WFs.
                            else if (dname == step_dim) {

                                // Use 0..N as a place-holder range.
                                // The actual values will be supplied during
                                // halo exchange.
                                dsize = gp->get_alloc_size(dname);
                                copy_begin[dname] = 0;
                                copy_end[dname] = dsize;
                            }

                            // misc?
                            // Copy over entire range.
                            // TODO: make dirty flags for misc dims in grids.
                            else {
                                dsize = gp->get_alloc_size(dname);
                                copy_begin[dname] = gp->get_first_misc_index(dname);
                                copy_end[dname] = gp->get_last_misc_index(dname) + 1;
                                assert(copy_end[dname] - copy_begin[dname] == dsize);
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
        if (env->num_ranks > 1 && opts->overlap_comms) {
            mpi_interior.update_bb("interior", *this, true);
            TRACE_MSG("MPI interior BB: [" << mpi_interior.bb_begin.makeDimValStr() <<
                      " ... " << mpi_interior.bb_end.makeDimValStr() << ")");
        }

        // At this point, we have all the buffers configured.
        // Now we need to allocate space for them.
        
        // Base ptrs for all alloc'd data.
        // These pointers will be shared by the ones in the grid
        // objects, which will take over ownership when these go
        // out of scope. Key is memory type.
        map <int, shared_ptr<char>> _mpi_data_buf;

        // A table for send-buffer offsets for all rank pairs for every grid:
        // [grid-name][sending-rank][receiving-rank]
        map<string, vector<vector<size_t>>> sb_ofs;
        bool do_shm = false;
        auto my_shm_rank = env->my_shm_rank;
        assert(my_shm_rank == mpiInfo->shm_ranks.at(mpiInfo->my_neighbor_index));

        // Make sure pad is big enough for shm locks.
        assert(_data_buf_pad >= sizeof(SimpleLock));
 
        // Allocate MPI buffers.
        // Pass 0: count required size, allocate chunk of memory at end.
        // Pass 1: distribute parts of already-allocated memory chunk.
        // Pass 2: set pointers to shm of other ranks.
        for (int pass = 0; pass < 3; pass++) {
            TRACE_MSG("allocMpiData pass " << pass << " for " <<
                      mpiData.size() << " MPI buffer set(s)");

            // Count bytes needed and number of buffers for each NUMA node.
            map <int, size_t> npbytes, nbufs;

            // Grids. Use the map to ensure same order in all ranks.
            for (auto gi : gridMap) {
                auto& gname = gi.first;
                auto& gp = gi.second;

                // Are there MPI bufs for this grid?
                if (mpiData.count(gname) == 0)
                    continue;
                auto& grid_mpi_data = mpiData.at(gname);

                // Resize table.
                if (pass == 0) {
                    assert(env->num_shm_ranks > 0);
                    sb_ofs[gname].resize(env->num_shm_ranks);
                    for (auto& gtab : sb_ofs[gname])
                        gtab.resize(env->num_shm_ranks, 0);
                }
                
                // Visit buffers for each neighbor for this grid.
                grid_mpi_data.visitNeighbors
                    ([&](const IdxTuple& roffsets,
                         int nrank,
                         int nidx,
                         MPIBufs& bufs) {

                        // Default is global numa pref setting for MPI
                        // buffer, not possible override for this grid.
                        int numa_pref = opts->_numa_pref;

                        // If neighbor can use MPI shm, set key, etc.
                        auto nshm_rank = mpiInfo->shm_ranks.at(nidx);
                        if (nshm_rank != MPI_PROC_NULL) {
                            do_shm = true;
                            numa_pref = _shmem_key;
                            assert(nshm_rank < env->num_shm_ranks);
                        }
                            
                        // Send and recv.
                        for (int bd = 0; bd < MPIBufs::nBufDirs; bd++) {
                            auto& buf = grid_mpi_data.getBuf(MPIBufs::BufDir(bd), roffsets);
                            if (buf.get_size() == 0)
                                continue;

                            // Don't use my mem for the recv buf if using shm.
                            bool use_mine = !(bd == MPIBufs::bufRecv && nshm_rank != MPI_PROC_NULL);

                            // Set storage if buffer has been allocated in pass 0.
                            if (pass == 1 && use_mine) {
                                auto base = _mpi_data_buf[numa_pref];
                                auto ofs = npbytes[numa_pref];
                                assert(base);
                                auto* rp = buf.set_storage(base, ofs);
                                TRACE_MSG("  MPI buf '" << buf.name << "' at " << rp <<
                                          " for " << makeByteStr(buf.get_bytes()));

                                // Write test values & init lock.
                                *((int*)rp) = me;
                                *((char*)rp + buf.get_bytes() - 1) = 'Z';
                                buf.shm_lock_init();

                                // Save offset.
                                if (nshm_rank != MPI_PROC_NULL && bd == MPIBufs::bufSend)
                                    sb_ofs[gname].at(my_shm_rank).at(nshm_rank) = ofs;
                            }

                            // Using shm from another rank.
                            else if (pass == 2 && !use_mine) {
                                char* base = (char*)mpiInfo->halo_buf_ptrs[nidx];
                                auto sz = mpiInfo->halo_buf_sizes[nidx];
                                auto ofs = sb_ofs[gname].at(nshm_rank).at(my_shm_rank);
                                assert(sz >= ofs + buf.get_bytes() + YASK_PAD_BYTES);
                                auto* rp = buf.set_storage(base, ofs);
                                TRACE_MSG("  MPI shm buf '" << buf.name << "' at " << rp <<
                                          " for " << makeByteStr(buf.get_bytes()));

                                // Check values written by owner rank.
                                assert(*((int*)rp) == nrank);
                                assert(*((char*)rp + buf.get_bytes() - 1) == 'Z');
                                assert(!buf.is_ok_to_read());
                            }

                            // Determine padded size (also offset to next location)
                            // in my mem.
                            if (use_mine) {
                                auto sbytes = buf.get_bytes();
                                npbytes[numa_pref] += ROUND_UP(sbytes + _data_buf_pad,
                                                               CACHELINE_BYTES);
                                nbufs[numa_pref]++;
                                if (pass == 0)
                                    TRACE_MSG("  MPI buf '" << buf.name << "' needs " <<
                                              makeByteStr(sbytes) <<
                                              " using mem-key " << numa_pref);
                            }
                        } // snd/rcv.
                    } );  // neighbors.

                // Share offsets between ranks.
                if (pass == 1 && do_shm) {

                    for (int rn = 0; rn < env->num_shm_ranks; rn++) {
                        TRACE_MSG("Sharing MPI shm offsets from shm-rank " << rn);
                        MPI_Bcast(sb_ofs[gname][rn].data(), env->num_shm_ranks, MPI_INTEGER8,
                                  rn, env->shm_comm);
                        for (int rn2 = 0; rn2 < env->num_shm_ranks; rn2++)
                            TRACE_MSG("  offset on rank " << rn << " for rank " << rn2 <<
                                      " is " << sb_ofs[gname][rn][rn2]);
                    }
                }

            } // grids.

            // Alloc for each mem type.
            if (pass == 0)
                _alloc_data(npbytes, nbufs, _mpi_data_buf, "MPI buffer");

            MPI_Barrier(env->shm_comm);
        } // MPI passes.

#endif
    }

    // Allocate memory for scratch grids based on number of threads and
    // block sizes.
    void StencilContext::allocScratchData() {
        CONTEXT_VARS(this);

        // Remove any old scratch data.
        freeScratchData();

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

        // Find the max mini-block size across all packs.
        // They can be different across packs when pack-specific
        // auto-tuning has been used.
        IdxTuple mblksize(_dims->_domain_dims);
        for (auto& sp : stPacks) {
            auto& psettings = sp->getActiveSettings();
            DOMAIN_VAR_LOOP(i, j) {

                auto sz = round_up_flr(psettings._mini_block_sizes[i],
                                       fold_pts[j]);
                mblksize[j] = max(mblksize[j], sz);
            }
        }
        TRACE_MSG("allocScratchData: max mini-block size across pack(s) is " <<
                  mblksize.makeDimValStr(" * "));
        
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

                            // Set domain size of scratch grid to mini-block size.
                            gp->_set_domain_size(dname, mblksize[dname]);

                            // Conservative allowance for WF exts and/or temporal shifts.
                            idx_t shift_pts = max(wf_shift_pts[dname], tb_angles[dname] * num_tb_shifts) * 2;
                            gp->_set_left_wf_ext(dname, shift_pts);
                            gp->_set_right_wf_ext(dname, shift_pts);

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
    
} // namespace yask.
