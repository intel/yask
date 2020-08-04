/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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

// This file contains implementations of functions for allocating data.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    ////// Allocators and deleters //////

    // Aligned allocation.
    char* yask_aligned_alloc(std::size_t nbytes) {

        // Alignment to use based on size.
        const size_t _def_alignment = CACHELINE_BYTES;
        const size_t _def_big_alignment = YASK_HUGE_ALIGNMENT;
        size_t align = (nbytes >= _def_big_alignment) ?
            _def_big_alignment : _def_alignment;
        void *p = 0;

        // Some envs have posix_memalign(), some have aligned_alloc().
#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
        int ret = posix_memalign(&p, align, nbytes);
        if (ret) p = 0;
#else
        p = aligned_alloc(align, nbytes);
#endif

        if (!p)
            THROW_YASK_EXCEPTION("error: cannot allocate " + make_byte_str(nbytes) +
                                 " aligned to " + make_byte_str(align));

        // Map alloc to device.
        char* cp = static_cast<char*>(p);
        OFFLOAD_MAP_ALLOC(cp, nbytes);

        // Return as a char* as required for shared_ptr ctor.
        return static_cast<char*>(p);
    }

    // Reverse yask_aligned_alloc().
    void AlignedDeleter::operator()(char* p) {
        if (p) {
            OFFLOAD_MAP_FREE(p);
            free(p);
            p = NULL;
        }
    }
    
#ifdef USE_PMEM
    static int pmem_tmpfile(const char *dir, size_t size, int *fd, void **addr)
    {
        static char tmpl[] = "/appdirect_mem_xxxxxx";
        int err = 0;

        char fullname[strlen(dir) + sizeof (tmpl)];
        (void) strcpy(fullname, dir);
        (void) strcat(fullname, tmpl);

        if ((*fd = mkstemp(fullname)) < 0) {
            perror("mkstemp()");
            err = MEMKIND_ERROR_RUNTIME;
            THROW_YASK_EXCEPTION("Error: MEMKIND_ERROR_RUNTIME - mkstemp()\n");
        }

        (void) unlink(dir);

        if (ftruncate(*fd, size) != 0) {
            err = MEMKIND_ERROR_RUNTIME;
            THROW_YASK_EXCEPTION("Error: MEMKIND_ERROR_RUNTIME - ftruncate()\n");
        }

        *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
        if (*addr == MAP_FAILED) {
            err = MEMKIND_ERROR_RUNTIME;
            THROW_YASK_EXCEPTION("Error: MEMKIND_ERROR_RUNTIME - mmap()\n");
        }

        return err;
    }
#endif

    // NUMA allocation.
    // 'numa_pref' == yask_numa_none: use default aligned alloc.
    // 'numa_pref' >= 0: preferred NUMA node.
    // 'numa_pref' < 0: use NUMA policy corresponding to value.
    // TODO: get rid of magic-number scheme.
    char* numa_alloc(std::size_t nbytes, int numa_pref) {

        void *p = 0;

        if (numa_pref == yask_numa_none)
            return yask_aligned_alloc(nbytes);

#ifdef USE_NUMA

        // Should we use the numa policy library?
#ifdef USE_NUMA_POLICY_LIB
#pragma omp single
        else if (numa_available() != -1) {
            numa_set_bind_policy(0);
            if (numa_pref >= 0 && numa_pref <= numa_max_node())
                numa_alloc_onnode(nbytes, numa_pref);
            else
                numa_alloc_local(nbytes);
            // Interleaved not available.
        }
        else
            THROW_YASK_EXCEPTION("Error: explicit NUMA policy allocation is not available");

        // Use mmap/mbind explicitly.
#else
        else if (get_mempolicy(NULL, NULL, 0, 0, 0) == 0) {

            // Set mmap flags.
            int mmprot = PROT_READ | PROT_WRITE;
            int mmflags = MAP_PRIVATE | MAP_ANONYMOUS;

            // Get an anonymous R/W memory map.
            p = mmap(0, nbytes, mmprot, mmflags, -1, 0);

            // If successful, apply the desired binding.
            if (p && p != MAP_FAILED) {
                if (numa_pref >= 0) {

                    // Prefer given node.
                    unsigned long nodemask = 0x1UL << numa_pref;
                    mbind(p, nbytes, MPOL_PREFERRED, &nodemask, sizeof(nodemask) * 8, 0);
                }
                else if (numa_pref == yask_numa_interleave) {

                    // Use all nodes.
                    unsigned long nodemask = (unsigned long)-1;
                    mbind(p, nbytes, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8, 0);
                }

                else{

                    // Use local node.
                    // MPOL_LOCAL was defined in Linux 3.8, so use
                    // MPOL_DEFAULT as backup on old systems.
#ifdef MPOL_LOCAL
                    mbind(p, nbytes, MPOL_LOCAL, 0, 0, 0);
#else
                    mbind(p, nbytes, MPOL_DEFAULT, 0, 0, 0);
#endif
                }
            }
            else
                THROW_YASK_EXCEPTION("Error: anonymous mmap of " + make_byte_str(nbytes) +
                                     " failed");
        }
        else
            THROW_YASK_EXCEPTION("Error: explicit NUMA policy allocation is not available");

#endif // not USE_NUMA_POLICY_LIB.

#else
        THROW_YASK_EXCEPTION("Error: NUMA allocation is not enabled; build with numa=1");
#endif // USE_NUMA.

        // Should not get here w/null p; throw exception.
        if (!p)
            THROW_YASK_EXCEPTION("Error: cannot allocate " + make_byte_str(nbytes) +
                                 " using numa-node (or policy) " + to_string(numa_pref));

        // Check alignment.
        if ((size_t(p) & (CACHELINE_BYTES - 1)) != 0)
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: NUMA-allocated " << p << " is not " <<
                                            CACHELINE_BYTES << "-byte aligned");

        // Map alloc to device.
        char* cp = static_cast<char*>(p);
        OFFLOAD_MAP_ALLOC(cp, nbytes);

        // Return as a char* as required for shared_ptr ctor.
        return static_cast<char*>(p);
    }

    // Reverse numa_alloc().
    void NumaDeleter::operator()(char* p) {

        if (p)
            OFFLOAD_MAP_FREE(p);

        if (p && _numa_pref == yask_numa_none) {
            free(p);
            p = NULL;
        }

#ifdef USE_NUMA
#ifdef USE_NUMA_POLICY_LIB
        if (p && numa_available() != -1) {
            numa_free(p, _nbytes);
            p = NULL;
        }
#else
        if (p && get_mempolicy(NULL, NULL, 0, 0, 0) == 0) {
            munmap(p, _nbytes);
            p = NULL;
        }
#endif
#endif
        if (p) {
            free(p);
            p = NULL;
        }
    }

    // PMEM allocation.
    char* pmem_alloc(std::size_t nbytes, int dev_num) {

        void *p = 0;

        // Allocate into pmem.
#ifdef USE_PMEM
        int err = 0;
        int fd;
        // 'X' of pmem_x should be matched with the NUMA node.
        string pmem_name("/mnt/pmem");
        pmem_name += to_string(dev_num);
        err = pmem_tmpfile(pmem_name.c_str(), nbytes, &fd, &p);
        if (err)
            THROW_YASK_EXCEPTION("Error: Unable to create temporary file for PMEM");
#else
        THROW_YASK_EXCEPTION("Error: PMEM allocation is not enabled; build with pmem=1");
#endif

        if (!p)
            THROW_YASK_EXCEPTION("Error: cannot allocate " + make_byte_str(nbytes) +
                                 " on pmem dev " + to_string(dev_num));

        // Check alignment.
        if ((size_t(p) & (CACHELINE_BYTES - 1)) != 0)
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: PMEM-allocated " << p << " is not " <<
                                            CACHELINE_BYTES << "-byte aligned");

        // Map alloc to device.
        char* cp = static_cast<char*>(p);
        OFFLOAD_MAP_ALLOC(cp, nbytes);

        // Return as a char* as required for shared_ptr ctor.
        return static_cast<char*>(p);
    }

    // Reverse pmem_alloc().
    void PmemDeleter::operator()(char* p) {
        if (p) {
            OFFLOAD_MAP_FREE(p);
            munmap(p, _nbytes);
            p = NULL;
        }
    }

    // MPI shm allocation.
    char* shm_alloc(std::size_t nbytes,
                   const MPI_Comm* shm_comm, MPI_Win* shm_win) {

        void *p = 0;

        // Allocate using MPI shm.
#ifdef USE_MPI
        assert(shm_comm);
        assert(shm_win);
        MPI_Info win_info;
        MPI_Info_create(&win_info);
        MPI_Info_set(win_info, "alloc_shared_noncontig", "true");
        MPI_Win_allocate_shared(nbytes, 1, win_info, *shm_comm, &p, shm_win);
        MPI_Info_free(&win_info);
        MPI_Win_lock_all(0, *shm_win);
#else
        THROW_YASK_EXCEPTION("Error: MPI shm allocation is not enabled; build with mpi=1");
#endif

        if (!p)
            THROW_YASK_EXCEPTION("Error: cannot allocate " + make_byte_str(nbytes) +
                                 " using MPI shm");

        // Check alignment.
        if ((size_t(p) & (CACHELINE_BYTES - 1)) != 0)
            FORMAT_AND_THROW_YASK_EXCEPTION("Error: MPI shm-allocated " << p << " is not " <<
                                            CACHELINE_BYTES << "-byte aligned");

        #ifdef USE_OFFLOAD
        THROW_YASK_EXCEPTION("Error: mapping offload device memory to shm not yet supported; "
                             "use '-no-use_shm'");
        #endif
        
        // Return as a char* as required for shared_ptr ctor.
        return static_cast<char*>(p);
    }

    // Reverse shm_alloc().
    void ShmDeleter::operator()(char* p) {

#ifdef USE_MPI
        assert(_shm_comm);
        assert(_shm_win);
        MPI_Win_unlock_all(*_shm_win);
        MPI_Win_free(_shm_win);
        p = NULL;
#else
        THROW_YASK_EXCEPTION("Error: MPI shm deallocation is not enabled; build with mpi=1");
#endif
    }

    ///// Memory-alloc functions in StencilContext /////
    
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
    // Pointers are returned in 'data_buf'.
    // 'nvars' and 'type' are only used for debug msg.
    void StencilContext::_alloc_data(const map <int, size_t>& nbytes,
                                     const map <int, size_t>& nvars,
                                     map <int, shared_ptr<char>>& data_buf,
                                     const std::string& type) {
        STATE_VARS(this);

        // Loop through each mem type.
        for (const auto& i : nbytes) {
            int mem_key = i.first;
            size_t nb = i.second;
            size_t ng = nvars.at(mem_key);

            // Alloc data depending on magic key.
            shared_ptr<char> p;
            string msg = "Allocating " + make_byte_str(nb) +
                " for " + to_string(ng) + " " + type + "(s) ";
            if (mem_key == _shmem_key) {
                msg += "using MPI shm";
                DEBUG_MSG(msg << "...");
                p = shared_shm_alloc<char>(nb, &env->shm_comm, &mpi_info->halo_win);

                // Get pointer for each neighbor rank.
#ifdef USE_MPI
                int ns = int(mpi_info->neighborhood_size);
                for (int ni = 0; ni < ns; ni++) {
                    int nr = mpi_info->my_neighbors.at(ni);
                    if (nr == MPI_PROC_NULL)
                        continue;
                    int sr = mpi_info->shm_ranks.at(ni);
                    MPI_Aint sz;
                    int dispunit;
                    void* baseptr;
                    MPI_Win_shared_query(mpi_info->halo_win, sr, &sz,
                                         &dispunit, &baseptr);
                    mpi_info->halo_buf_ptrs.at(ni) = baseptr;
                    mpi_info->halo_buf_sizes.at(ni) = sz;
                    TRACE_MSG("MPI shm halo buffer for rank " << nr << " is at " <<
                              baseptr << " for " << make_byte_str(sz));
                }
#endif
            }
            else if (mem_key >= _pmem_key) {
                auto dev_num = mem_key - _pmem_key;
                msg += "on PMEM device " + to_string(dev_num);
                DEBUG_MSG(msg << "...");
                p = shared_pmem_alloc<char>(nb, dev_num);
            }
            else {
                if (mem_key == yask_numa_none)
                    msg += "using default allocator";
                else if (mem_key == yask_numa_local)
                    msg += "preferring local NUMA node";
                else if (mem_key == yask_numa_interleave)
                    msg += "interleaved across all NUMA nodes";
                else if (mem_key >= 0)
                    msg += "preferring NUMA node " + to_string(mem_key);
                else
                    msg += "using mem policy " + to_string(mem_key);
                DEBUG_MSG(msg << "...");
                p = shared_numa_alloc<char>(nb, mem_key);
            }

            // Save using original key.
            data_buf[mem_key] = p;
            TRACE_MSG("Got memory at " << static_cast<void*>(p.get()));
        }
    }

    // Allocate memory for vars that do not already have storage.
    void StencilContext::alloc_var_data() {
        STATE_VARS(this);

        // Allocate I/O vars before read-only vars.
        VarPtrs sorted_var_ptrs;
        VarPtrSet done;
        for (auto op : output_var_ptrs) {
            sorted_var_ptrs.push_back(op);
            done.insert(op);
        }
        for (auto gp : var_ptrs) {
            if (!done.count(gp))
                sorted_var_ptrs.push_back(gp);
        }
	done.clear();

#ifdef USE_PMEM
        os << "PMEM var-allocation priority:" << endl;
        for (auto sp : sorted_var_ptrs) {
            os << " '" << sp->get_name() << "'";
            if (done.find(sp)!=done.end())
                os << " (output)";
            os << endl;
        }
#endif

        // Base ptrs for all default-alloc'd data.
        // These pointers will be shared by the ones in the var
        // objects, which will take over ownership when these go
        // out of scope.
        // Key is preferred numa node or -1 for local.
        map <int, shared_ptr<char>> _var_data_buf;

#ifdef USE_PMEM
        auto preferred_numasize = opts->_numa_pref_max * 1024*1024*(size_t)1024;
#endif

        // Pass 0: assign PMEM node when preferred NUMA node is not enough.
        // Pass 1: count required size for each NUMA node, allocate chunk of memory at end.
        // Pass 2: distribute parts of already-allocated memory chunk.
        for (int pass = 0; pass < 3; pass++) {
            TRACE_MSG("alloc_var_data pass " << pass << " for " <<
                      var_ptrs.size() << " var(s)");

            // Count bytes needed and number of vars for each NUMA node.
            map <int, size_t> npbytes, nvars;

            // Vars.
            for (auto gp : sorted_var_ptrs) {
                if (!gp)
                    continue;
                auto& gname = gp->get_name();
                auto& gb = gp->gb();

                // Var data.
                // Don't alloc if already done.
                if (!gp->is_storage_allocated()) {
                    int numa_pref = gp->get_numa_preferred();

                    // Set storage if buffer has been allocated in pass 1.
                    if (pass == 2) {
                        auto p = _var_data_buf[numa_pref];
                        assert(p);
                        gp->set_storage(p, npbytes[numa_pref]);
                        DEBUG_MSG(gb.make_info_string());
                    }

                    // Determine padded size (also offset to next location).
                    size_t nbytes = gp->get_num_storage_bytes();
                    npbytes[numa_pref] += ROUND_UP(nbytes + _data_buf_pad,
                                                   CACHELINE_BYTES);
                    nvars[numa_pref]++;

                    if (pass == 0) {
#ifdef USE_PMEM
                        if (preferred_numasize < npbytes[numa_pref])
                            if (getnode() == -1) {
                                os << "Warning: cannot get numa_node information for PMEM allocation;"
                                    " using default numa_pref " << endl;
                            }
                            else

                                // TODO: change this behavior so that it doesn't actually
                                // modify the NUMA pref of the var.
                                gp->set_numa_preferred(_pmem_key + getnode());
#endif
                    }

                    if (pass == 1)
                        TRACE_MSG(" var '" << gname << "' needs " << make_byte_str(nbytes) <<
                                  " on NUMA node " << numa_pref);
                }

                // Otherwise, just print existing var info.
                else if (pass == 1)
                    DEBUG_MSG(gb.make_info_string());
            }

            // Reset the counters
            if (pass == 0) {
                npbytes.clear();
                nvars.clear();
            }

            // Alloc for each node.
            if (pass == 1)
                _alloc_data(npbytes, nvars, _var_data_buf, "var");

        } // var passes.
    };

    // Determine the size and shape of all MPI buffers.
    // Create buffers and allocate them.
    void StencilContext::alloc_mpi_data() {
        STATE_VARS(this);

        // Remove any old MPI data.
        env->global_barrier();
        free_mpi_data();

        // Init interior.
        mpi_interior = ext_bb;
        mpi_interior.bb_valid = false;

#ifdef USE_MPI

        map<int, int> num_exchanges; // send/recv => count.
        map<int, idx_t> num_elems; // send/recv => count.
        auto me = env->my_rank;

        // Need to determine the size and shape of all MPI buffers.
        // Loop thru all neighbors of this rank.
        mpi_info->visit_neighbors
            ([&](const IdxTuple& neigh_offsets, int neigh_rank, int neigh_idx) {
                if (neigh_rank == MPI_PROC_NULL)
                    return; // from lambda fn.

                // Is vectorized exchange allowed based on domain sizes?
                // Both my rank and neighbor rank must have *all* domain sizes
                // of vector multiples.
                bool vec_ok = allow_vec_exchange &&
                    mpi_info->has_all_vlen_mults[mpi_info->my_neighbor_index] &&
                    mpi_info->has_all_vlen_mults[neigh_idx];

                // Determine size of MPI buffers between neigh_rank and my
                // rank for each var and create those that are needed.  It
                // is critical that the number, size, and shape of my
                // send/receive buffers match those of the receive/send
                // buffers of my neighbors.  Important: Current algorithm
                // assumes my left neighbor's buffer sizes can be calculated
                // by considering my rank's right side data and vice-versa.
                // Thus, all ranks must have consistent data that contribute
                // to these calculations.
                for (auto& gp : orig_var_ptrs) {
                    auto& gb = gp->gb();
                    auto& gname = gp->get_name();
                    bool var_vec_ok = vec_ok;


                    // Get calculated max dist needed for this var.
                    int maxdist = gp->get_halo_exchange_l1_norm();

                    // Always use max dist with WF. Do this because edge
                    // and/or corner values may be needed in WF extensions
                    // even it not needed w/o WFs.
                    // TODO: determine if max is always needed.
                    if (wf_steps > 0)
                        maxdist = NUM_STENCIL_DIMS - 1;

                    // Manhattan dist. of current neighbor.
                    int mandist = mpi_info->man_dists.at(neigh_idx);

                    // Check distance.
                    if (mandist > maxdist) {
                        TRACE_MSG("no halo exchange needed with rank " << neigh_rank <<
                                  " (L1-norm = " << mandist <<
                                  ") for var '" << gname <<
                                  "' (max L1-norm = " << maxdist << ")");
                        continue; // to next var.
                    }

                    // Lookup first & last domain indices and calc exchange sizes
                    // for this var.
                    bool found_delta = false;
                    IdxTuple my_halo_sizes, neigh_halo_sizes;
                    IdxTuple first_inner_idx, last_inner_idx;
                    IdxTuple first_outer_idx, last_outer_idx;
                    for (auto& dim : domain_dims) {
                        auto& dname = dim._get_name();

                        // Only consider domain dims that are used in this var.
                        if (gp->is_dim_used(dname)) {
                            auto vlen = gp->_get_var_vec_len(dname);
                            auto lhalo = gp->get_left_halo_size(dname);
                            auto rhalo = gp->get_right_halo_size(dname);

                            // Get domain indices for this var.  If there
                            // are no more ranks in the given direction,
                            // extend the "outer" index to include the halo
                            // in that direction to make sure all data are
                            // sync'd. Critical for temporal tiling.
                            idx_t fidx = gp->get_first_rank_domain_index(dname);
                            idx_t lidx = gp->get_last_rank_domain_index(dname);
                            first_inner_idx.add_dim_back(dname, fidx);
                            last_inner_idx.add_dim_back(dname, lidx);
                            if (opts->is_first_rank(dname))
                                fidx -= lhalo; // extend into left halo.
                            if (opts->is_last_rank(dname))
                                lidx += rhalo; // extend into right halo.
                            first_outer_idx.add_dim_back(dname, fidx);
                            last_outer_idx.add_dim_back(dname, lidx);

                            // Determine if it is possible to round the
                            // outer indices to vec-multiples. This will
                            // be required to allow full vec exchanges for
                            // this var. We won't do the actual rounding
                            // yet, because we need to see if it's safe
                            // in all dims.
                            // Need +1 and then -1 trick for last.
                            fidx = round_down_flr(fidx, vlen);
                            lidx = round_up_flr(lidx + 1, vlen) - 1;
                            if (fidx < gp->get_first_rank_alloc_index(dname))
                                var_vec_ok = false;
                            if (lidx > gp->get_last_rank_alloc_index(dname))
                                var_vec_ok = false;

                            // Determine size of exchange in this dim. This
                            // will be the actual halo size plus any
                            // wave-front shifts. In the current
                            // implementation, we need the wave-front shifts
                            // regardless of whether there is a halo on a
                            // given var. This is because each
                            // stencil-bundle gets shifted by the WF angles
                            // at each step in the WF.

                            // Neighbor is to the left in this dim.
                            if (neigh_offsets[dname] == MPIInfo::rank_prev) {

                                // Number of points to be added for WFs.
                                auto ext = wf_shift_pts[dname];

                                // My halo on my left.
                                my_halo_sizes.add_dim_back(dname, lhalo + ext);

                                // Neighbor halo on their right.
                                // Assume my right is same as their right.
                                neigh_halo_sizes.add_dim_back(dname, rhalo + ext);

                                // Flag that this var has a neighbor to left or right.
                                found_delta = true;
                            }

                            // Neighbor is to the right in this dim.
                            else if (neigh_offsets[dname] == MPIInfo::rank_next) {

                                // Number of points to be added for WFs.
                                auto ext = wf_shift_pts[dname];

                                // My halo on my right.
                                my_halo_sizes.add_dim_back(dname, rhalo + ext);

                                // Neighbor halo on their left.
                                // Assume my left is same as their left.
                                neigh_halo_sizes.add_dim_back(dname, lhalo + ext);

                                // Flag that this var has a neighbor to left or right.
                                found_delta = true;
                            }

                            // Neighbor in-line in this dim.
                            else {
                                my_halo_sizes.add_dim_back(dname, 0);
                                neigh_halo_sizes.add_dim_back(dname, 0);
                            }

                        } // domain dims in this var.
                    } // domain dims.

                    // Is buffer needed?
                    // Example: if this var is 2D in y-z, but only neighbors are in
                    // x-dim, we don't need any exchange.
                    if (!found_delta) {
                        TRACE_MSG("no halo exchange needed for var '" << gname <<
                                  "' with rank " << neigh_rank <<
                                  " because the neighbor is not in a direction"
                                  " corresponding to a var dim");
                        continue; // to next var.
                    }

                    // Round halo sizes if vectorized exchanges allowed.
                    // Both self and neighbor must be vec-multiples
                    // and outer indices must be vec-mults or extendable
                    // to be so.
                    // TODO: add a heuristic to avoid increasing by a large factor.
                    if (var_vec_ok) {
                        for (auto& dim : domain_dims) {
                            auto& dname = dim._get_name();
                            if (gp->is_dim_used(dname)) {
                                auto vlen = gp->_get_var_vec_len(dname);

                                // First index rounded down.
                                auto fidx = first_outer_idx[dname];
                                fidx = round_down_flr(fidx, vlen);
                                first_outer_idx.set_val(dname, fidx);

                                // Last index rounded up.
                                // Need +1 and then -1 trick because it's last, not end.
                                auto lidx = last_outer_idx[dname];
                                lidx = round_up_flr(lidx + 1, vlen) - 1;
                                last_outer_idx.set_val(dname, lidx);

                                // sizes rounded up.
                                my_halo_sizes.set_val(dname, ROUND_UP(my_halo_sizes[dname], vlen));
                                neigh_halo_sizes.set_val(dname, ROUND_UP(neigh_halo_sizes[dname], vlen));

                            } // domain dims in this var.
                        } // domain dims.
                    }

                    // Make a buffer in both directions (send & receive).
                    for (int bd = 0; bd < MPIBufs::n_buf_dirs; bd++) {

                        // Begin/end vars to indicate what part
                        // of main var to read from or write to based on
                        // the current neighbor being processed.
                        IdxTuple copy_begin = gb.get_allocs();
                        IdxTuple copy_end = gb.get_allocs(); // one past last!

                        // Adjust along domain dims in this var.
                        DOMAIN_VAR_LOOP(i, j) {
                            auto& dim = domain_dims.get_dim(j);
                            auto& dname = dim._get_name();
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
                                if (bd == MPIBufs::buf_send) {

                                    // Neighbor is to the left.
                                    if (neigh_ofs == idx_t(MPIInfo::rank_prev)) {

                                        // Only read slice as wide as halo from beginning.
                                        copy_begin[dname] = first_inner_idx[dname];
                                        copy_end[dname] = first_inner_idx[dname] + neigh_halo_sizes[dname];

                                        // Adjust LHS of interior.
                                        idx_t ext_end = ROUND_UP(first_inner_idx[dname] +
                                                                 max(min_ext, neigh_halo_sizes[dname]),
                                                                 dims->_fold_pts[dname]);
                                        mpi_interior.bb_begin[j] =
                                            max(mpi_interior.bb_begin[j], ext_end);
                                    }

                                    // Neighbor is to the right.
                                    else if (neigh_ofs == idx_t(MPIInfo::rank_next)) {

                                        // Only read slice as wide as halo before end.
                                        copy_begin[dname] = last_inner_idx[dname] + 1 - neigh_halo_sizes[dname];
                                        copy_end[dname] = last_inner_idx[dname] + 1;

                                        // Adjust RHS of interior.
                                        idx_t ext_begin = ROUND_DOWN(last_inner_idx[dname] + 1 -
                                                                     max(min_ext, neigh_halo_sizes[dname]),
                                                                     dims->_fold_pts[dname]);
                                        mpi_interior.bb_end[j] =
                                            min(mpi_interior.bb_end[j], ext_begin);
                                    }

                                    // Else, this neighbor is in same posn as I am in this dim,
                                    // so we leave the default begin/end settings.
                                }

                                // Region to write to, i.e., into this rank's halo.
                                // So, use my halo sizes when calculating buffer sizes.
                                else if (bd == MPIBufs::buf_recv) {

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
                            } // domain dims in this var.
                        } // domain dims.

                        // Sizes of buffer in all dims of this var.
                        // Also, set begin/end value for non-domain dims.
                        IdxTuple buf_sizes = gb.get_allocs();
                        bool buf_vec_ok = var_vec_ok;
                        for (auto& dname : gp->get_dim_names()) {
                            idx_t dsize = 1;

                            // domain dim?
                            if (domain_dims.lookup(dname)) {
                                dsize = copy_end[dname] - copy_begin[dname];

                                // Check whether alignment and size are multiple of vlen.
                                auto vlen = gp->_get_var_vec_len(dname);
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
                            // TODO: make dirty flags for misc dims in vars.
                            else {
                                dsize = gp->get_alloc_size(dname);
                                copy_begin[dname] = gp->get_first_misc_index(dname);
                                copy_end[dname] = gp->get_last_misc_index(dname) + 1;
                                assert(copy_end[dname] - copy_begin[dname] == dsize);
                            }

                            // Save computed size.
                            buf_sizes[dname] = dsize;

                        } // all dims in this var.

                        // Unique name for buffer based on var name, direction, and ranks.
                        string bname = gname;
                        if (bd == MPIBufs::buf_send)
                            bname += "_send_halo_from_" + to_string(me) + "_to_" + to_string(neigh_rank);
                        else if (bd == MPIBufs::buf_recv)
                            bname += "_recv_halo_from_" + to_string(neigh_rank) + "_to_" + to_string(me);

                        // Does buffer have non-zero size?
                        if (buf_sizes.size() == 0 || buf_sizes.product() == 0) {
                            TRACE_MSG("MPI buffer '" << bname <<
                                      "' not needed because there is no data to exchange");
                            continue;
                        }

                        // At this point, buf_sizes, copy_begin, and copy_end
                        // should be set for each dim in this var.

                        // Compute last from end.
                        IdxTuple copy_last = copy_end.sub_elements(1);

                        // Make MPI data entry for this var.
                        auto gbp = mpi_data.emplace(gname, state->_mpi_info);
                        auto& gbi = gbp.first; // iterator from pair returned by emplace().
                        auto& gbv = gbi->second; // value from iterator.
                        auto& buf = gbv.get_buf(MPIBufs::BufDir(bd), neigh_offsets);

                        // Config buffer for this var.
                        // (But don't allocate storage yet.)
                        buf.begin_pt = copy_begin;
                        buf.last_pt = copy_last;
                        buf.num_pts = buf_sizes;
                        buf.name = bname;
                        buf.vec_copy_ok = buf_vec_ok;

                        TRACE_MSG("MPI buffer '" << buf.name <<
                                  "' configured for rank at relative offsets " <<
                                  neigh_offsets.sub_elements(1).make_dim_val_str() << " with " <<
                                  buf.num_pts.make_dim_val_str(" * ") << " = " << buf.get_size() <<
                                  " element(s) at [" << buf.begin_pt.make_dim_val_str() <<
                                  " ... " << buf.last_pt.make_dim_val_str() <<
                                  "] with vector-copy " <<
                                  (buf.vec_copy_ok ? "enabled" : "disabled"));
                        num_exchanges[bd]++;
                        num_elems[bd] += buf.get_size();

                    } // send, recv.
                } // vars.
            });   // neighbors.
        TRACE_MSG("number of MPI send buffers on this rank: " << num_exchanges[int(MPIBufs::buf_send)]);
        TRACE_MSG("number of elements in send buffers: " << make_num_str(num_elems[int(MPIBufs::buf_send)]));
        TRACE_MSG("number of MPI recv buffers on this rank: " << num_exchanges[int(MPIBufs::buf_recv)]);
        TRACE_MSG("number of elements in recv buffers: " << make_num_str(num_elems[int(MPIBufs::buf_recv)]));

        // Finalize interior BB if there are multiple ranks and overlap enabled.
        if (env->num_ranks > 1 && opts->overlap_comms) {
            mpi_interior.update_bb("interior", this, true);
            TRACE_MSG("MPI interior BB: [" << mpi_interior.make_range_string(domain_dims) << "]");
        }

        // At this point, we have all the buffers configured.
        // Now we need to allocate space for them.

        // Base ptrs for all alloc'd data.
        // These pointers will be shared by the ones in the var
        // objects, which will take over ownership when these go
        // out of scope. Key is memory type.
        map <int, shared_ptr<char>> _mpi_data_buf;

        // A table for send-buffer offsets for all rank pairs for every var:
        // [var-name][sending-rank][receiving-rank]
        map<string, vector<vector<size_t>>> sb_ofs;
        bool do_shm = false;
        auto my_shm_rank = env->my_shm_rank;
        assert(my_shm_rank == mpi_info->shm_ranks.at(mpi_info->my_neighbor_index));

        // Make sure pad is big enough for shm locks.
        assert(_data_buf_pad >= sizeof(SimpleLock));

        // Allocate MPI buffers.
        // Pass 0: count required size, allocate chunk of memory at end.
        // Pass 1: distribute parts of already-allocated memory chunk.
        // Pass 2: set pointers to shm of other ranks.
        for (int pass = 0; pass < 3; pass++) {
            TRACE_MSG("alloc_mpi_data pass " << pass << " for " <<
                      mpi_data.size() << " MPI buffer set(s)");

            // Count bytes needed and number of buffers for each NUMA node.
            map <int, size_t> npbytes, nbufs;

            // Vars. Use the map to ensure same order in all ranks.
            for (auto gi : var_map) {
                auto& gname = gi.first;
                auto& gp = gi.second;

                // Are there MPI bufs for this var?
                if (mpi_data.count(gname) == 0)
                    continue;
                auto& var_mpi_data = mpi_data.at(gname);

                // Resize table.
                if (pass == 0) {
                    assert(env->num_shm_ranks > 0);
                    sb_ofs[gname].resize(env->num_shm_ranks);
                    for (auto& gtab : sb_ofs[gname])
                        gtab.resize(env->num_shm_ranks, 0);
                }

                // Visit buffers for each neighbor for this var.
                var_mpi_data.visit_neighbors
                    ([&](const IdxTuple& roffsets,
                         int nrank,
                         int nidx,
                         MPIBufs& bufs) {

                        // Default is global numa pref setting for MPI
                        // buffer, not possible override for this var.
                        int numa_pref = opts->_numa_pref;

                        // If neighbor can use MPI shm, set key, etc.
                        auto nshm_rank = mpi_info->shm_ranks.at(nidx);
                        if (nshm_rank != MPI_PROC_NULL) {
                            do_shm = true;
                            numa_pref = _shmem_key;
                            assert(nshm_rank < env->num_shm_ranks);
                        }

                        // Send and recv.
                        for (int bd = 0; bd < MPIBufs::n_buf_dirs; bd++) {
                            auto& buf = var_mpi_data.get_buf(MPIBufs::BufDir(bd), roffsets);
                            if (buf.get_size() == 0)
                                continue;

                            // Don't use my mem for the recv buf if using shm.
                            bool use_mine = !(bd == MPIBufs::buf_recv && nshm_rank != MPI_PROC_NULL);

                            // Set storage if buffer has been allocated in pass 0.
                            if (pass == 1 && use_mine) {
                                auto base = _mpi_data_buf[numa_pref];
                                auto ofs = npbytes[numa_pref];
                                assert(base);
                                auto* rp = buf.set_storage(base, ofs);
                                TRACE_MSG("  MPI buf '" << buf.name << "' at " << rp <<
                                          " for " << make_byte_str(buf.get_bytes()));

                                // Write test values & init lock.
                                *((int*)rp) = me;
                                if (buf.get_bytes() > sizeof(int)) // Room to mark end?
                                    *((char*)rp + buf.get_bytes() - 1) = 'Z';
                                buf.shm_lock_init();

                                // Save offset.
                                if (nshm_rank != MPI_PROC_NULL && bd == MPIBufs::buf_send)
                                    sb_ofs[gname].at(my_shm_rank).at(nshm_rank) = ofs;
                            }

                            // Using shm from another rank.
                            else if (pass == 2 && !use_mine) {
                                char* base = (char*)mpi_info->halo_buf_ptrs[nidx];
                                auto sz = mpi_info->halo_buf_sizes[nidx];
                                auto ofs = sb_ofs[gname].at(nshm_rank).at(my_shm_rank);
                                assert(sz >= ofs + buf.get_bytes() + YASK_PAD_BYTES);
                                auto* rp = buf.set_storage(base, ofs);
                                TRACE_MSG("  MPI shm buf '" << buf.name << "' at " << rp <<
                                          " for " << make_byte_str(buf.get_bytes()));

                                // Check values written by owner rank.
                                assert(*((int*)rp) == nrank);
                                if (buf.get_bytes() > sizeof(int)) // Room to mark end?
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
                                              make_byte_str(sbytes) <<
                                              " (mem-key = " << numa_pref << ")");
                            }
                        } // snd/rcv.
                    } );  // neighbors.

                // Share offsets between ranks.
                if (pass == 1 && do_shm) {

                    for (int rn = 0; rn < env->num_shm_ranks; rn++) {
                        TRACE_MSG("Sharing MPI shm offsets from shm-rank " << rn <<
                                  " for var '" << gname << "'");
                        MPI_Bcast(sb_ofs[gname][rn].data(), env->num_shm_ranks, MPI_INTEGER8,
                                  rn, env->shm_comm);
                        for (int rn2 = 0; rn2 < env->num_shm_ranks; rn2++)
                            TRACE_MSG("  offset on rank " << rn << " for rank " << rn2 <<
                                      " is " << sb_ofs[gname][rn][rn2]);
                    }
                }

            } // vars.

            // Alloc for each mem type.
            if (pass == 0)
                _alloc_data(npbytes, nbufs, _mpi_data_buf, "MPI buffer");

            MPI_Barrier(env->shm_comm);
        } // MPI passes.

#endif
    }

    // Delete and re-create all the scratch vars.  Delete and re-allocate
    // memory for scratch vars based on number of threads and block sizes.
    void StencilContext::alloc_scratch_data() {
        STATE_VARS(this);

        // Remove any old scratch data.
        free_scratch_data();

        // Base ptrs for all alloc'd data.
        // This pointer will be shared by the ones in the var
        // objects, which will take over ownership when it goes
        // out of scope.
        map <int, shared_ptr<char>> _scratch_data_buf;

        // Make sure the right number of threads are set so we
        // have the right number of scratch vars.
        int rthreads = set_region_threads();

        // Delete any existing scratch vars.
        // Create new scratch vars, but without any
        // data allocated.
        // Update core pointers in generated bundles.
        make_scratch_vars(rthreads);

        // Find the max mini-block size across all stages.
        // They can be different across stages when stage-specific
        // auto-tuning has been used.
        IdxTuple mblksize(domain_dims);
        for (auto& sp : st_stages) {
            auto& psettings = sp->get_active_settings();
            DOMAIN_VAR_LOOP(i, j) {

                auto sz = round_up_flr(psettings._mini_block_sizes[i],
                                       fold_pts[j]);
                mblksize[j] = max(mblksize[j], sz);
            }
        }
        TRACE_MSG("alloc_scratch_data: max mini-block size across stage(s) is " <<
                  mblksize.make_dim_val_str(" * "));

        // Pass 0: count required size, allocate chunk of memory at end.
        // Pass 1: distribute parts of already-allocated memory chunk.
        for (int pass = 0; pass < 2; pass++) {
            TRACE_MSG("alloc_scratch_data pass " << pass << " for " <<
                      scratch_vecs.size() << " set(s) of scratch vars");

            // Count bytes needed and number of vars for each NUMA node.
            map <int, size_t> npbytes, nvars;

            // Loop through each scratch var vector.
            for (auto* sgv : scratch_vecs) {
                assert(sgv);

                // Loop through each scratch var in this vector.
                // There will be one for each region thread.
                assert(int(sgv->size()) == rthreads);
                int thr_num = 0;
                for (auto gp : *sgv) {
                    assert(gp);
                    auto& gname = gp->get_name();
                    int numa_pref = gp->get_numa_preferred();
                    auto& gb = gp->gb();

                    // Loop through each domain dim.
                    for (auto& dim : domain_dims) {
                        auto& dname = dim._get_name();

                        if (gp->is_dim_used(dname)) {

                            // Set domain size of scratch var to mini-block size.
                            gp->_set_domain_size(dname, mblksize[dname]);

                            // Conservative allowance for WF exts and/or temporal shifts.
                            idx_t shift_pts = max(wf_shift_pts[dname], tb_angles[dname] * num_tb_shifts) * 2;
                            gp->_set_left_wf_ext(dname, shift_pts);
                            gp->_set_right_wf_ext(dname, shift_pts);

                            // Pads.
                            // Set via both 'extra' and 'min'; larger result will be used.
                            gp->set_extra_pad_size(dname, opts->_extra_pad_sizes[dname]);
                            gp->set_min_pad_size(dname, opts->_min_pad_sizes[dname]);
                        }
                    } // dims.

                    // Set storage if buffer has been allocated.
                    if (pass == 1) {
                        auto p = _scratch_data_buf[numa_pref];
                        assert(p);
                        gp->set_storage(p, npbytes[numa_pref]);
                        TRACE_MSG(gb.make_info_string());
                    }

                    // Determine size used (also offset to next location).
                    size_t nbytes = gp->get_num_storage_bytes();
                    npbytes[numa_pref] += ROUND_UP(nbytes + _data_buf_pad,
                                                   CACHELINE_BYTES);
                    nvars[numa_pref]++;
                    if (pass == 0)
                        TRACE_MSG(" scratch var '" << gname << "' for thread " <<
                                  thr_num << " needs " << make_byte_str(nbytes) <<
                                  " on NUMA node " << numa_pref);
                    thr_num++;
                } // scratch vars.
            } // scratch-var vecs.

            // Alloc for each node.
            if (pass == 0)
                _alloc_data(npbytes, nvars, _scratch_data_buf, "scratch var");

        } // scratch-var passes.
    }

} // namespace yask.
