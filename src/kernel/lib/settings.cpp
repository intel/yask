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

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Check whether dim is of allowed type.
    void Dims::checkDimType(const std::string& dim,
                            const std::string& fn_name,
                            bool step_ok,
                            bool domain_ok,
                            bool misc_ok) const {
        if (step_ok && domain_ok && misc_ok)
            return;
        if (dim == _step_dim) {
            if (!step_ok) {
                THROW_YASK_EXCEPTION("Error in " + fn_name + "(): dimension '" +
                                     dim + "' is the step dimension, which is not allowed");
            }
        }
        else if (_domain_dims.lookup(dim)) {
            if (!domain_ok) {
                THROW_YASK_EXCEPTION("Error in " + fn_name + "(): dimension '" +
                                     dim + "' is a domain dimension, which is not allowed");
            }
        }
        else if (!misc_ok) {
            THROW_YASK_EXCEPTION("Error in " + fn_name + "(): dimension '" +
                                 dim + "' is a misc dimension, which is not allowed");
        }
    }

    // APIs.
    // See yask_kernel_api.hpp.
    yk_env_ptr yk_factory::new_env(MPI_Comm comm) const {
        auto ep = make_shared<KernelEnv>();
        assert(ep);
        ep->initEnv(0, 0, comm);
        return ep;
    }
    yk_env_ptr yk_factory::new_env() const {
        return new_env(MPI_COMM_NULL);
    }

    ///// KernelEnv functions:

    omp_lock_t KernelEnv::_debug_lock;
    bool KernelEnv::_debug_lock_init_done = false;
    
    // Init MPI, OMP.
    void KernelEnv::initEnv(int* argc, char*** argv, MPI_Comm existing_comm)
    {
        // MPI init.
        my_rank = 0;
        num_ranks = 1;

#ifdef USE_MPI
        int is_init = false;
        MPI_Initialized(&is_init);

        // No MPI communicator provided.
        if (existing_comm == MPI_COMM_NULL) {
            if (!is_init) {
                int provided = 0;
                MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
                if (provided < MPI_THREAD_SERIALIZED) {
                    THROW_YASK_EXCEPTION("error: MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE not provided");
                }
                is_init = true;
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

        // Set env vars needed by OMP.
        // TODO: make this visible to the user.
        int ret = setenv("OMP_PLACES", "cores", 0); // default placement for outer loop.
        assert(ret == 0);
        ret = setenv("KMP_HOT_TEAMS_MODE", "1", 0); // more efficient nesting.
        assert(ret == 0);
        ret = setenv("KMP_HOT_TEAMS_MAX_LEVEL", "2", 0); // 2-level nesting.

        // Save initial value of OMP max threads.
        // Side effect: causes OMP to dump debug info if env var set.
        if (!max_threads)
            max_threads = omp_get_max_threads();
    }

    // Apply a function to each neighbor rank.
    // Does NOT visit self.
    void MPIInfo::visitNeighbors(std::function<void
                                 (const IdxTuple& neigh_offsets, // NeighborOffset vals.
                                  int neigh_rank, // MPI rank.
                                  int neigh_index)> visitor) {

        // TODO: convert to use visitAllPoints().
        for (int i = 0; i < neighborhood_size; i++) {
            auto neigh_offsets = neighborhood_sizes.unlayout(i);
            int neigh_rank = my_neighbors.at(i);
            assert(i == getNeighborIndex(neigh_offsets));

            if (i != my_neighbor_index)
                visitor(neigh_offsets, neigh_rank, i);
        }
    }

    // Set pointer to storage.
    // Free old storage.
    // 'base' should provide get_num_bytes() bytes at offset bytes.
    void* MPIBuf::set_storage(std::shared_ptr<char>& base, size_t offset) {

        void* p = set_storage(base.get(), offset);
        
        // Share ownership of base.
        // This ensures that last MPI buffer to use a shared allocation
        // will trigger dealloc.
        _base = base;

        return p;
    }

    // Set internal pointer, but does not share ownership.
    void* MPIBuf::set_storage(char* base, size_t offset) {

        // Release any old data.
        release_storage();

        // Set plain pointer to new data.
        if (base) {
            char* p = base + offset;
            _elems = (real_t*)p;
        } else {
            _elems = 0;
        }
        return (void*)_elems;
    }
    
    // Apply a function to each neighbor rank.
    // Does NOT visit self or non-existent neighbors.
    void MPIData::visitNeighbors(std::function<void
                                 (const IdxTuple& neigh_offsets, // NeighborOffset.
                                  int neigh_rank,
                                  int neigh_index,
                                  MPIBufs& bufs)> visitor) {

        _mpiInfo->visitNeighbors
            ([&](const IdxTuple& neigh_offsets, int neigh_rank, int i) {

                if (neigh_rank != MPI_PROC_NULL)
                    visitor(neigh_offsets, neigh_rank, i, bufs[i]);
            });
    }

    // Access a buffer by direction and neighbor offsets.
    MPIBuf& MPIData::getBuf(MPIBufs::BufDir bd, const IdxTuple& offsets) {
        assert(int(bd) < int(MPIBufs::nBufDirs));
        auto i = _mpiInfo->getNeighborIndex(offsets); // 1D index.
        assert(i < _mpiInfo->neighborhood_size);
        return bufs[i].bufs[bd];
    }

    // Add options to set one domain var to a cmd-line parser.
    void KernelSettings::_add_domain_option(CommandLineParser& parser,
                                            const std::string& prefix,
                                            const std::string& descrip,
                                            IdxTuple& var,
                                            bool allow_step) {

        // Add step + domain vars.
        vector<idx_t*> multi_vars;
        string multi_help;
        for (auto& dim : var.getDims()) {
            auto& dname = dim.getName();
            if (!allow_step && _dims->_step_dim == dname)
                continue;
            idx_t* dp = var.lookup(dname); // use lookup() to get non-const ptr.

            // Option for individual dim.
            parser.add_option(new CommandLineParser::IdxOption
                              (prefix + dname,
                               descrip + " in '" + dname + "' dimension.",
                               *dp));

            // Add to domain list if a domain var.
            if (_dims->_domain_dims.lookup(dname)) {
                multi_vars.push_back(dp);
                multi_help += " -" + prefix + dname + " <integer>";
            }
        }

        // Option for setting all domain dims.
        parser.add_option(new CommandLineParser::MultiIdxOption
                          (prefix,
                           "Shorthand for" + multi_help,
                           multi_vars));
    }

    // Add these settigns to a cmd-line parser.
    void KernelSettings::add_options(CommandLineParser& parser)
    {
        _add_domain_option(parser, "d", "Rank-domain size", _rank_sizes);
        _add_domain_option(parser, "r", "Region size", _region_sizes, true);
        _add_domain_option(parser, "b", "Block size", _block_sizes, true);
        _add_domain_option(parser, "mb", "Mini-block size", _mini_block_sizes);
        _add_domain_option(parser, "sb", "Sub-block size", _sub_block_sizes);
#ifdef SHOW_GROUPS
        _add_domain_option(parser, "bg", "Block-group size", _block_group_sizes);
        _add_domain_option(parser, "mbg", "Mini-block-group size", _mini_block_group_sizes);
        _add_domain_option(parser, "sbg", "Sub-block-group size", _sub_block_group_sizes);
#endif
        _add_domain_option(parser, "mp", "Minimum grid-padding size (including halo)", _min_pad_sizes);
        _add_domain_option(parser, "ep", "Extra grid-padding size (beyond halo)", _extra_pad_sizes);
#ifdef USE_MPI
        _add_domain_option(parser, "nr", "Num ranks", _num_ranks);
        _add_domain_option(parser, "ri", "This rank's logical index (0-based)", _rank_indices);
        parser.add_option(new CommandLineParser::IntOption
                          ("msg_rank",
                           "Index of MPI rank that will print informational messages.",
                           msg_rank));
        parser.add_option(new CommandLineParser::BoolOption
                          ("overlap_comms",
                           "Overlap MPI communication with calculation of interior elements whenever possible.",
                           overlap_comms));
        parser.add_option(new CommandLineParser::BoolOption
                          ("use_shm",
                           "Use shared memory for MPI buffers when possible.",
                           use_shm));
        parser.add_option(new CommandLineParser::IdxOption
                          ("min_exterior",
                           "Minimum width of MPI exterior section to compute before starting MPI communication.",
                           _min_exterior));
#endif
        parser.add_option(new CommandLineParser::BoolOption
                          ("force_scalar",
                           "Evaluate every grid point with scalar stencil operations (for debug).",
                           force_scalar));
        parser.add_option(new CommandLineParser::IntOption
                          ("max_threads",
                           "Max OpenMP threads to use. Overrides default number of OpenMP threads "
                           "or the value set by OMP_NUM_THREADS.",
                           max_threads));
        parser.add_option(new CommandLineParser::IntOption
                          ("thread_divisor",
                           "Divide the number of OpenMP threads by the argument value. "
                           "If -max_threads is also used, divide the argument to that option by the "
                           "argument to this one. If -max_threads is not used, "
                           "divide the default number of OpenMP threads. "
                           "In either case, use the resulting truncated value as the "
                           "maximum number of OpenMP threads to use.",
                           thread_divisor));
        parser.add_option(new CommandLineParser::IntOption
                          ("block_threads",
                           "Number of threads to use in a nested OpenMP region for each block. "
                           "Will be restricted to a value less than or equal to "
                           "the maximum number of OpenMP threads specified by -max_threads "
                           "and/or -thread_divisor. "
                           "Each thread is used to execute stencils within a sub-block, and "
                           "sub-blocks are executed in parallel within mini-blocks.",
                           num_block_threads));
        parser.add_option(new CommandLineParser::BoolOption
                          ("bind_block_threads",
                           "Execute stencils at each vector-cluster index on a fixed thread-id "
                           "based on sub-block and mini-block sizes. "
                           "Applies only to domain dimensions not including the inner-most "
                           "one, effectively disabling parallelism across sub-blocks in the "
                           "inner-most domain dimension. "
                           "May increase cache locality when using multiple "
                           "block-threads when temporal blocking is active.",
                           bind_block_threads));
#ifdef USE_NUMA
        stringstream msg;
        msg << "Preferred NUMA node on which to allocate data for "
            "grids and MPI buffers. Alternatively, use special values " <<
            yask_numa_local << " for explicit local-node allocation, " <<
            yask_numa_interleave << " for interleaving pages across all nodes, or " <<
            yask_numa_none << " for no NUMA policy.";
        parser.add_option(new CommandLineParser::IntOption
                          ("numa_pref", msg.str(),
                           _numa_pref));
#endif
#ifdef USE_PMEM
        parser.add_option(new CommandLineParser::IntOption
                          ("numa_pref_max",
                           "Maximum GiB to allocate on preferred NUMA node before allocating on pmem device.",
                           _numa_pref_max));
#endif
    }

    // Print usage message.
    void KernelSettings::print_usage(ostream& os,
                                      CommandLineParser& parser,
                                      const string& pgmName,
                                      const string& appNotes,
                                      const vector<string>& appExamples) const
    {
        os << "Usage: " << pgmName << " [options]\n"
            "Options:\n";
        parser.print_help(os);
        os << "\nTerms for the various levels of tiling from smallest to largest:\n"
            " A 'point' is a single floating-point (FP) element.\n"
            "  This binary uses " << REAL_BYTES << "-byte FP elements.\n"
            " A 'vector' is composed of points.\n"
            "  A 'folded vector' contains points in more than one dimension.\n"
            "  The size of a vector is typically that of a SIMD register.\n"
            " A 'vector-cluster' is composed of vectors.\n"
            "  This is the unit of work done in each inner-most loop iteration.\n"
            " A 'sub-block' is composed of vector-clusters.\n"
            "  If the number of block-threads is greater than one,\n"
            "   then this is the unit of work for one nested OpenMP thread;\n"
            "   else, sub-blocks are evaluated sequentially within each mini-block.\n"
            " A 'mini-block' is composed of sub-blocks.\n"
            "  If using temporal wave-front block tiling (see mini-block-size guidelines),\n"
            "   then this is the unit of work for each wave-front block tile,\n"
            "   and the number temporal steps in the mini-block is always equal\n"
            "   to the number temporal steps a temporal block;\n"
            "   else, there is typically only one mini-block the size of a block.\n"
            "  Mini-blocks are evaluated sequentially within blocks.\n"
            " A 'block' is composed of mini-blocks.\n"
            "  If the number of threads is greater than one (typical),\n"
            "   then this is the unit of work for one OpenMP thread;\n"
            "   else, blocks are evaluated sequentially within each region.\n"
            " A 'region' is composed of blocks.\n"
            "  If using temporal wave-front rank tiling (see region-size guidelines),\n"
            "   then this is the unit of work for each wave-front rank tile;\n"
            "   else, there is typically only one region the size of the rank-domain.\n"
            "  Regions are evaluated sequentially within ranks.\n"
            " A 'rank-domain' is composed of regions.\n"
            "  This is the unit of work for one MPI rank.\n"
            "  Ranks are evaluated in parallel in separate MPI processes.\n"
            " The 'overall-problem' is composed of rank-domains.\n"
            "  This is the unit of work across all MPI ranks.\n" <<
#ifndef USE_MPI
            "   This binary has NOT been compiled with MPI support,\n"
            "   so the overall-problem is equivalent to the single rank-domain.\n" <<
#endif
            "\nGuidelines for setting tiling sizes:\n"
            " The vector and vector-cluster sizes are set at compile-time, so\n"
            "  there are no run-time options to set them.\n"
            " Set sub-block sizes to specify a unit of work done by each nested OpenMP thread.\n"
            "  Multiple sub-blocks are intended to allow sharing of caches\n"
            "   among multiple hyper-threads in a core when there is more than\n"
            "   one block-thread. It can also be used to share data between caches\n"
            "   among multiple cores.\n"
            "  A sub-block size of 0 in a given domain dimension =>\n"
            "   sub-block size is set to mini-block size in that dimension;\n"
            "   when there is more than one block-thread, the first dimension\n"
            "   will instead be set to the vector length to create \"slab\" shapes.\n"
            " Set mini-block sizes to control temporal wave-front tile sizes within a block.\n"
            "  Multiple mini-blocks are intended to increase locality in level-2 caches\n"
            "   when blocks are larger than L2 capacity.\n"
            "  A mini-block size of 0 in a given domain dimension =>\n"
            "   mini-block size is set to block size in that dimension.\n"
            "  The size of a mini-block in the step dimension is always implicitly\n"
            "   the same as that of a block.\n"
            " Set block sizes to specify a unit of work done by each top-level OpenMP thread.\n"
            "  A block size of 0 in a given domain dimension =>\n"
            "   block size is set to region size in that dimension.\n"
            "  A block size of 0 in the step dimension (e.g., '-bt') disables any temporal blocking.\n"
            "  A block size of 1 in the step dimension enables temporal blocking, but only between\n"
            "   packs in the same step.\n"
            "  A block size >1 in the step dimension enables temporal blocking across multiple steps.\n"
            "  The temporal block size may be automatically reduced if needed based on the\n"
            "   domain block sizes and the stencil halos.\n"
            " Set region sizes to control temporal wave-front tile sizes within a rank.\n"
            "  Multiple regions are intended to increase locality in level-3 caches\n"
            "   when ranks are larger than L3 capacity.\n"
            "  A region size of 0 in the step dimension (e.g., '-rt') => region size is\n"
            "   set to block size in the step dimension.\n"
            "  A region size >1 in the step dimension enables wave-front rank tiling.\n"
            "  The region size in the step dimension affects how often MPI halo-exchanges occur:\n"
            "   A region size of 0 in the step dimension => exchange after every pack.\n"
            "   A region size >0 in the step dimension => exchange after that many steps.\n"
            " Set rank-domain sizes to specify the work done on this rank.\n"
            "  Set the domain sizes to specify the problem size for this rank.\n"
            "  This and the number of grids affect the amount of memory used.\n"
#ifdef SHOW_GROUPS
            " Setting 'group' sizes controls only the order of tiles.\n"
            "  These are advanced settings that are not commonly used.\n"
#endif
            "\nControlling OpenMP threading:\n"
            " Using '-max_threads 0' =>\n"
            "  max_threads is set to OpenMP's default number of threads.\n"
            " The -thread_divisor option is a convenience to control the number of\n"
            "  hyper-threads used without having to know the number of cores,\n"
            "  e.g., using '-thread_divisor 2' will halve the number of OpenMP threads.\n"
            " For stencil evaluation, threads are allocated using nested OpenMP:\n"
            "  Num threads per region = max_threads / thread_divisor / block_threads.\n"
            "  Num threads per block = block_threads.\n"
            "  Num threads per sub-block = 1.\n"
            "  Num threads used for halo exchange is same as num per region.\n" <<
#ifdef USE_MPI
            "\nControlling MPI scaling:\n"
            "  To 'weak-scale' to a larger overall-problem size, use multiple MPI ranks\n"
            "   and keep the rank-domain sizes constant.\n"
            "  To 'strong-scale' a given overall-problem size, use multiple MPI ranks\n"
            "   and reduce the size of each rank-domain appropriately.\n" <<
#endif
            appNotes <<
            "Examples for a 3D (x, y, z) over time (t) problem:\n"
            " " << pgmName << " -d 768\n"
            " " << pgmName << " -dx 512 -dy 256 -dz 128\n"
            " " << pgmName << " -d 2048 -r 512 -rt 10  # temporal rank tiling.\n"
            " " << pgmName << " -d 512 -nrx 2 -nry 1 -nrz 2   # multi-rank.\n";
        for (auto ae : appExamples)
            os << " " << pgmName << " " << ae << endl;
        os << flush;
    }

    // For each one of 'inner_sizes' that is zero,
    // make it equal to corresponding one in 'outer_sizes'.
    // Round up each of 'inner_sizes' to be a multiple of corresponding one in 'mults'.
    // Output info to 'os' using '*_name' and dim names.
    // Does not process 'step_dim'.
    // Return product of number of inner subsets.
    idx_t  KernelSettings::findNumSubsets(ostream& os,
                                          IdxTuple& inner_sizes, const string& inner_name,
                                          const IdxTuple& outer_sizes, const string& outer_name,
                                          const IdxTuple& mults, const std::string& step_dim) {

        idx_t prod = 1;
        for (auto& dim : inner_sizes.getDims()) {
            auto& dname = dim.getName();
            if (dname == step_dim)
                continue;
            idx_t* dptr = inner_sizes.lookup(dname); // use lookup() to get non-const ptr.

            idx_t outer_size = outer_sizes[dname];
            if (*dptr <= 0)
                *dptr = outer_size; // 0 => use full size as default.
            if (mults.lookup(dname) && mults[dname] > 1)
                *dptr = ROUND_UP(*dptr, mults[dname]);
            idx_t inner_size = *dptr;
            idx_t ninner = (inner_size <= 0) ? 0 :
                (outer_size + inner_size - 1) / inner_size; // full or partial.
            idx_t rem = (inner_size <= 0) ? 0 :
                outer_size % inner_size;                       // size of remainder.
            idx_t nfull = rem ? (ninner - 1) : ninner; // full only.

            if (outer_size > 0) {
                os << " In '" << dname << "' dimension, " <<
                    outer_name << " of size " <<
                    outer_size << " contains " << nfull << " " <<
                    inner_name << "(s) of size " << inner_size;
                if (rem)
                    os << " plus 1 remainder " << inner_name << " of size " << rem;
                os << "." << endl;
            }
            prod *= ninner;
        }
        return prod;
    }

    // Make sure all user-provided settings are valid and finish setting up some
    // other vars before allocating memory.
    // Called from prepare_solution(), so it doesn't normally need to be called from user code.
    void KernelSettings::adjustSettings(std::ostream& os, KernelEnvPtr env) {

        auto& step_dim = _dims->_step_dim;
        auto& rt = _region_sizes[step_dim];
        auto& bt = _block_sizes[step_dim];
        auto& mbt = _mini_block_sizes[step_dim];
        
        // Fix up step-dim sizes.
        rt = max(rt, idx_t(0));
        bt = max(bt, idx_t(0));
        mbt = max(mbt, idx_t(0));
        if (!rt)
            rt = bt;       // Default region steps to block steps.
        if (!mbt)
            mbt = bt;       // Default mini-blk steps to block steps.

        // Determine num regions.
        // Also fix up region sizes as needed.
        // Temporal region size will be increase to
        // current temporal block size if needed.
        // Default region size (if 0) will be size of rank-domain.
        os << "\nRegions:" << endl;
        auto nr = findNumSubsets(os, _region_sizes, "region",
                                 _rank_sizes, "rank-domain",
                                 _dims->_cluster_pts, step_dim);
        os << " num-regions-per-rank-domain-per-step: " << nr << endl;
        os << " Since the region size in the '" << step_dim <<
            "' dim is " << rt << ", temporal wave-front rank tiling is ";
        if (!rt) os << "NOT ";
        os << "enabled.\n";

        // Determine num blocks.
        // Also fix up block sizes as needed.
        // Default block size (if 0) will be size of region.
        os << "\nBlocks:" << endl;
        auto nb = findNumSubsets(os, _block_sizes, "block",
                                 _region_sizes, "region",
                                 _dims->_cluster_pts, step_dim);
        os << " num-blocks-per-region-per-step: " << nb << endl;
        os << " num-blocks-per-rank-domain-per-step: " << (nb * nr) << endl;
        os << " Since the block size in the '" << step_dim <<
            "' dim is " << bt << ", temporal blocking is ";
        if (!bt) os << "NOT ";
        os << "enabled.\n";

        // Determine num mini-blocks.
        // Also fix up mini-block sizes as needed.
        os << "\nMini-blocks:" << endl;
        auto nmb = findNumSubsets(os, _mini_block_sizes, "mini-block",
                                 _block_sizes, "block",
                                 _dims->_cluster_pts, step_dim);
        os << " num-mini-blocks-per-block-per-step: " << nmb << endl;
        os << " num-mini-blocks-per-region-per-step: " << (nmb * nb) << endl;
        os << " num-mini-blocks-per-rank-domain-per-step: " << (nmb * nb * nr) << endl;
        os << " Since the mini-block size in the '" << step_dim <<
            "' dim is " << mbt << ", temporal wave-front block tiling is ";
        if (!mbt) os << "NOT ";
        os << "enabled.\n";

        // Adjust defaults for sub-blocks to be slab if
        // we are using more than one block thread.
        // Otherwise, findNumSubsets() would set default
        // to entire block.
        if (num_block_threads > 1 && _sub_block_sizes.sum() == 0) {
            DOMAIN_VAR_LOOP(i, j) {
                auto bsz = _block_sizes[i];
                auto fpts = fold_pts[j];
                auto vecs_per_blk = bsz / fpts;

                // Subdivide this dim if there are enough vectors in
                // the block for each thread.
                if (vecs_per_blk >= num_block_threads) {

                    // Use narrow slabs if at least 3D.
                    // TODO: consider a better heuristic.
                    if (_dims->_domain_dims.getNumDims() >= 3)
                        _sub_block_sizes[i] = 1; // will be rounded up to min size.

                    // Divide equally.
                    else
                        _sub_block_sizes[i] = CEIL_DIV(vecs_per_blk, num_block_threads) * fpts;

                    // Only want to set 1 dim; others will be set to max.
                    break;
                }                        
            }
        }

        // Determine num sub-blocks.
        // Also fix up sub-block sizes as needed.
        os << "\nSub-blocks:" << endl;
        auto nsb = findNumSubsets(os, _sub_block_sizes, "sub-block",
                                  _mini_block_sizes, "mini-block",
                                 _dims->_cluster_pts, step_dim);
        os << " num-sub-blocks-per-mini-block-per-step: " << nsb << endl;
        os << " num-sub-blocks-per-block-per-step: " << (nsb * nmb) << endl;
        os << " num-sub-blocks-per-region-per-step: " << (nsb * nmb * nb) << endl;
        os << " num-sub-blocks-per-rank-per-step: " << (nsb * nmb * nb * nr) << endl;

        // Now, we adjust groups. These are done after all the above sizes
        // because group sizes are more like 'guidelines' and don't have
        // their own loops.

        // Adjust defaults for groups to be min size.
        // Otherwise, findNumBlockGroupsInRegion() would set default
        // to entire region.
        DOMAIN_VAR_LOOP(i, j) {
            if (_block_group_sizes[i] == 0)
                _block_group_sizes[i] = 1; // will be rounded up to min size.
            if (_mini_block_group_sizes[i] == 0)
                _mini_block_group_sizes[i] = 1; // will be rounded up to min size.
            if (_sub_block_group_sizes[i] == 0)
                _sub_block_group_sizes[i] = 1; // will be rounded up to min size.
        }

#ifdef SHOW_GROUPS
        os << "\nGroups (only affect ordering):" << endl;

        // Show num block-groups.
        // TODO: only print this if block-grouping is enabled.
        auto nbg = findNumSubsets(os, _block_group_sizes, "block-group",
                                  _region_sizes, "region",
                                  _block_sizes, step_dim);
        os << " num-block-groups-per-region-per-step: " << nbg << endl;
        auto nb_g = findNumSubsets(os, _block_sizes, "block",
                                   _block_group_sizes, "block-group",
                                   _dims->_cluster_pts, step_dim);
        os << " num-blocks-per-block-group-per-step: " << nb_g << endl;

        // Show num mini-block-groups.
        // TODO: only print this if mini-block-grouping is enabled.
        auto nmbg = findNumSubsets(os, _mini_block_group_sizes, "mini-block-group",
                                   _block_sizes, "block",
                                   _mini_block_sizes, step_dim);
        os << " num-mini-block-groups-per-block-per-step: " << nmbg << endl;
        auto nmb_g = findNumSubsets(os, _mini_block_sizes, "mini-block",
                                    _mini_block_group_sizes, "mini-block-group",
                                    _dims->_cluster_pts, step_dim);
        os << " num-mini-blocks-per-block-group-per-step: " << nmb_g << endl;

        // Show num sub-block-groups.
        // TODO: only print this if sub-block-grouping is enabled.
        auto nsbg = findNumSubsets(os, _sub_block_group_sizes, "sub-block-group",
                                   _mini_block_sizes, "mini-block",
                                   _sub_block_sizes, step_dim);
        os << " num-sub-block-groups-per-mini-block-per-step: " << nsbg << endl;
        auto nsb_g = findNumSubsets(os, _sub_block_sizes, "sub-block",
                                   _sub_block_group_sizes, "sub-block-group",
                                   _dims->_cluster_pts, step_dim);
        os << " num-sub-blocks-per-sub-block-group-per-step: " << nsb_g << endl;
#endif
    }

} // namespace yask.
