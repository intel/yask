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
    yk_env_ptr yk_factory::new_env() const {
        auto ep = make_shared<KernelEnv>();
        assert(ep);
        ep->initEnv(0, 0);
        return ep;
    }

    ///// KernelEnv functions:

    // Init MPI, OMP.
    void KernelEnv::initEnv(int* argc, char*** argv)
    {
        // MPI init.
        my_rank = 0;
        num_ranks = 1;

#ifdef USE_MPI
        int is_init = false;
        MPI_Initialized(&is_init);

        if (!is_init) {
            int provided = 0;
            MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
            if (provided < MPI_THREAD_SERIALIZED) {
                THROW_YASK_EXCEPTION("error: MPI_THREAD_SERIALIZED not provided");
            }
            is_init = true;
        }
        comm = MPI_COMM_WORLD;
        MPI_Comm_rank(comm, &my_rank);
        MPI_Comm_size(comm, &num_ranks);
#else
        comm = 0;
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
    void MPIBuf::set_storage(std::shared_ptr<char>& base, size_t offset) {

        // Release any old data if last owner.
        release_storage();

        // Share ownership of base.
        // This ensures that last grid to use a shared allocation
        // will trigger dealloc.
        _base = base;

        // Set plain pointer to new data.
        if (base.get()) {
            char* p = _base.get() + offset;
            _elems = (real_t*)p;
        } else {
            _elems = 0;
        }
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
                                            IdxTuple& var) {

        // Add step + domain vars.
        vector<idx_t*> multi_vars;
        string multi_help;
        for (auto& dim : var.getDims()) {
            auto& dname = dim.getName();
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
        _add_domain_option(parser, "r", "Region size", _region_sizes);
        _add_domain_option(parser, "bg", "Block-group size", _block_group_sizes);
        _add_domain_option(parser, "b", "Block size", _block_sizes);
        _add_domain_option(parser, "sbg", "Sub-block-group size", _sub_block_group_sizes);
        _add_domain_option(parser, "sb", "Sub-block size", _sub_block_sizes);
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
                           "Overlap MPI communication with calculation of grid cells whenever possible.",
                           overlap_comms));
#endif
        parser.add_option(new CommandLineParser::IntOption
                          ("max_threads",
                           "Max OpenMP threads to use.",
                           max_threads));
        parser.add_option(new CommandLineParser::IntOption
                          ("thread_divisor",
                           "Divide max OpenMP threads by <integer>.",
                           thread_divisor));
        parser.add_option(new CommandLineParser::IntOption
                          ("block_threads",
                           "Number of threads to use within each block.",
                           num_block_threads));
#ifdef USE_NUMA
        stringstream msg;
        msg << "Preferred NUMA node on which to allocate data for "
            "grids and MPI buffers. "
            "Alternatively, use " << yask_numa_local << " for explicit local-node allocation, " <<
            yask_numa_interleave << " for interleaving pages across all nodes, or " <<
            yask_numa_none << " for no NUMA policy.";
        parser.add_option(new CommandLineParser::IntOption
                          ("numa_pref", msg.str(),
                           _numa_pref));
#endif
#ifdef USE_PMEM
        parser.add_option(new CommandLineParser::IntOption
                          ("numa_pref_max",
                           "Maximum size of preferred NUMA node.",
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
        os << "Terms for the various levels of tiling from smallest to largest:\n"
            " A 'point' is a single floating-point (FP) element.\n"
            "  This binary uses " << REAL_BYTES << "-byte FP elements.\n"
            " A 'vector' is composed of points.\n"
            "  A 'folded vector' contains points in more than one dimension.\n"
            "  The size of a vector is typically that of a SIMD register.\n"
            " A 'vector-cluster' is composed of vectors.\n"
            "  This is the unit of work done in each inner-most loop iteration.\n"
            " A 'sub-block' is composed of vector-clusters.\n"
            "  If the number of threads-per-block is greater than one,\n"
            "   then this is the unit of work for one nested OpenMP thread;\n"
            "   else, sub-blocks are evaluated sequentially within each block.\n"
            " A 'block' is composed of sub-blocks.\n"
            "  This is the unit of work for one top-level OpenMP thread.\n"
            " A 'region' is composed of blocks.\n"
            "  If using temporal wave-front tiling (see region-size guidelines),\n"
            "   then, this is the unit of work for each wave-front tile;\n"
            "   else, there is typically only one region the sie of the rank-domain.\n"
            " A 'rank-domain' is composed of regions.\n"
            "  This is the unit of work for one MPI rank.\n"
            " The 'overall-problem' is composed of rank-domains.\n"
            "  This is the unit of work across all MPI ranks.\n" <<
#ifndef USE_MPI
            "   This binary has NOT been compiled with MPI support,\n"
            "   so the overall-problem is equivalent to the single rank-domain.\n" <<
#endif
            "Guidelines for setting tiling sizes:\n"
            " The vector and vector-cluster sizes are set at compile-time, so\n"
            "  there are no run-time options to set them.\n"
            " Set sub-block sizes to specify a unit of work done by each thread.\n"
            "  A sub-block size of 0 in a given dimension =>\n"
            "   sub-block size is set to block size in that dimension;\n"
            "   when there is more than one block-thread, the first dimension\n"
            "   will instead be set to a small value to create \"slab\" shapes.\n"
            " Set sub-block-group sizes to control the ordering of sub-blocks within a block.\n"
            "  All sub-blocks that intersect a given sub-block-group are evaluated\n"
            "   before sub-blocks in the next sub-block-group.\n"
            "  A sub-block-group size of 0 in a given dimension =>\n"
            "   sub-block-group size is set to sub-block size in that dimension.\n"
            " Set block sizes to specify a unit of work done by each thread team.\n"
            "  A block size of 0 in a given dimension =>\n"
            "   block size is set to region size in that dimension.\n"
            "  Temporal tiling in blocks is not yet supported, so effectively, bt = 1.\n"
            " Set block-group sizes to control the ordering of blocks within a region.\n"
            "  All blocks that intersect a given block-group are evaluated before blocks\n"
            "   in the next block-group.\n"
            "  A block-group size of 0 in a given dimension =>\n"
            "   block-group size is set to block size in that dimension.\n"
            " Set region sizes to control temporal wave-front tile sizes.\n"
            "  The temopral region size should be larger than one, and\n"
            "   the spatial region sizes should be less than the rank-domain sizes\n"
            "   in at least one dimension to enable temporal wave-front tiling.\n"
            "  The spatial region sizes should be greater than corresponding block sizes\n"
            "   to enable threading withing each wave-front tile.\n"
            "  Control the time-steps in each temporal wave-front with -rt.\n"
            "   Special cases:\n"
            "    Using '-rt 1' disables wave-front tiling.\n"
            "    Using '-rt 0' => all time-steps done in one wave-front.\n"
            "  A region size of 0 in a given dimension =>\n"
            "   region size is set to rank-domain size in that dimension.\n"
            " Set rank-domain sizes to specify the work done on this rank.\n"
            "  This and the number of grids affect the amount of memory used.\n"
            "Controlling OpenMP threading:\n"
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
            "Controlling MPI scaling:\n"
            "  To 'weak-scale' to a larger overall-problem size, use multiple MPI ranks\n"
            "   and keep the rank-domain sizes constant.\n"
            "  To 'strong-scale' a given overall-problem size, use multiple MPI ranks\n"
            "   and reduce the size of each rank-domain appropriately.\n" <<
#endif
            appNotes <<
            "Examples:\n" <<
            " " << pgmName << " -d 768 -dt 25\n" <<
            " " << pgmName << " -dx 512 -dy 256 -dz 128\n" <<
            " " << pgmName << " -d 2048 -dt 20 -r 512 -rt 10  # temporal tiling.\n" <<
            " " << pgmName << " -d 512 -nrx 2 -nry 1 -nrz 2   # multi-rank.\n";
        for (auto ae : appExamples)
            os << " " << pgmName << " " << ae << endl;
        os << flush;
    }

    // For each one of 'inner_sizes' that is zero,
    // make it equal to corresponding one in 'outer_sizes'.
    // Round up each of 'inner_sizes' to be a multiple of corresponding one in 'mults'.
    // Output info to 'os' using '*_name' and dim names.
    // Return product of number of inner subsets.
    idx_t  KernelSettings::findNumSubsets(ostream& os,
                                          IdxTuple& inner_sizes, const string& inner_name,
                                          const IdxTuple& outer_sizes, const string& outer_name,
                                          const IdxTuple& mults) {

        idx_t prod = 1;
        for (auto& dim : inner_sizes.getDims()) {
            auto& dname = dim.getName();
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

            os << " In '" << dname << "' dimension, " << outer_name << " of size " <<
                outer_size << " contains " << nfull << " " <<
                    inner_name << "(s) of size " << inner_size;
            if (rem)
                os << " plus 1 remainder " << inner_name << " of size " << rem;
            os << "." << endl;
            prod *= ninner;
        }
        return prod;
    }

    // Make sure all user-provided settings are valid and finish setting up some
    // other vars before allocating memory.
    // Called from prepare_solution(), so it doesn't normally need to be called from user code.
    void KernelSettings::adjustSettings(std::ostream& os, KernelEnvPtr env) {
        auto& step_dim = _dims->_step_dim;

        // Determine num regions.
        // Also fix up region sizes as needed.
        // Default region size (if 0) will be size of rank-domain.
        os << "\nRegions:" << endl;
        auto nr = findNumSubsets(os, _region_sizes, "region",
                                 _rank_sizes, "rank-domain",
                                 _dims->_cluster_pts);
        auto rt = _region_sizes[step_dim];
        os << " num-regions-per-rank-domain: " << nr << endl;
        os << " Since the region size in the '" << step_dim <<
            "' dim is " << rt << ", temporal wave-front tiling is ";
        if (rt <= 1) os << "NOT ";
        os << "enabled.\n";

        // Determine num blocks.
        // Also fix up block sizes as needed.
        // Default block size (if 0) will be size of region.
        os << "\nBlocks:" << endl;
        auto nb = findNumSubsets(os, _block_sizes, "block",
                                 _region_sizes, "region",
                                 _dims->_cluster_pts);
        os << " num-blocks-per-region: " << nb << endl;
        os << " num-blocks-per-rank-domain: " << (nb * nr) << endl;

        // Adjust defaults for sub-blocks to be slab if
        // we are using more than one block thread.
        // Otherwise, findNumSubsets() would set default
        // to entire block.
        if (num_block_threads > 1) {
            for (auto& dim : _dims->_domain_dims.getDims()) {
                auto& dname = dim.getName();
                if (_sub_block_sizes[dname] == 0)
                    _sub_block_sizes[dname] = 1; // will be rounded up to min size.

                // Only want to set 1st dim; others will be set to max.
                // TODO: make sure we're not setting inner dim.
                break;
            }
        }

        // Determine num sub-blocks.
        // Also fix up sub-block sizes as needed.
        os << "\nSub-blocks:" << endl;
        auto nsb = findNumSubsets(os, _sub_block_sizes, "sub-block",
                                 _block_sizes, "block",
                                 _dims->_cluster_pts);
        os << " num-sub-blocks-per-block: " << nsb << endl;

        // Now, we adjust groups. These are done after all the above sizes
        // because group sizes are more like 'guidelines' and don't have
        // their own loops.
        os << "\nGroups:" << endl;

        // Adjust defaults for groups to be min size.
        // Otherwise, findNumBlockGroupsInRegion() would set default
        // to entire region.
        for (auto& dim : _dims->_domain_dims.getDims()) {
            auto& dname = dim.getName();
            if (_block_group_sizes[dname] == 0)
                _block_group_sizes[dname] = 1; // will be rounded up to min size.
            if (_sub_block_group_sizes[dname] == 0)
                _sub_block_group_sizes[dname] = 1; // will be rounded up to min size.
        }

        // Determine num block-groups.
        // Also fix up block-group sizes as needed.
        // TODO: only print this if block-grouping is enabled.
        auto nbg = findNumSubsets(os, _block_group_sizes, "block-group",
                                  _region_sizes, "region",
                                  _block_sizes);
        os << " num-block-groups-per-region: " << nbg << endl;
        auto nb_g = findNumSubsets(os, _block_sizes, "block",
                                   _block_group_sizes, "block-group",
                                   _dims->_cluster_pts);
        os << " num-blocks-per-block-group: " << nb_g << endl;

        // Determine num sub-block-groups.
        // Also fix up sub-block-group sizes as needed.
        // TODO: only print this if sub-block-grouping is enabled.
        auto nsbg = findNumSubsets(os, _sub_block_group_sizes, "sub-block-group",
                                   _block_sizes, "block",
                                   _sub_block_sizes);
        os << " num-sub-block-groups-per-block: " << nsbg << endl;
        auto nsb_g = findNumSubsets(os, _sub_block_sizes, "block",
                                   _sub_block_group_sizes, "sub-block-group",
                                   _dims->_cluster_pts);
        os << " num-sub-blocks-per-sub-block-group: " << nsb_g << endl;
    }

} // namespace yask.
