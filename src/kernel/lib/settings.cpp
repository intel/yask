/*****************************************************************************

YASK: Yet Another Stencil Kit
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

    // KernelEnv global lock objects.
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
        if (existing_comm == MPI_COMM_NULL ||
            existing_comm == MPI_COMM_WORLD) {
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
    // 'base' should provide get_bytes() + YASK_PAD_BYTES bytes at offset bytes.
    void* MPIBuf::set_storage(std::shared_ptr<char>& base, size_t offset) {

        void* p = set_storage(base.get(), offset);
        
        // Share ownership of base.  This ensures that [only] last MPI
        // buffer to use a shared allocation will trigger dealloc.
        _base = base;

        return p;
    }

    // Set internal pointer, but does not share ownership.
    // Use when shm buffer is owned by another rank.
    void* MPIBuf::set_storage(char* base, size_t offset) {

        // Release any old data.
        release_storage();

        // Set plain pointer to new data.
        if (base) {
            char* p = base + offset;
            _elems = (real_t*)p;

            // Shm lock lives at end of data in buffer.
            _shm_lock = (SimpleLock*)(p + get_bytes());
        } else {
            _elems = 0;
            _shm_lock = 0;
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

    // Settings ctor.
    KernelSettings::KernelSettings(DimsPtr dims, KernelEnvPtr env) :
        _dims(dims), max_threads(env->max_threads) {
        auto& step_dim = dims->_step_dim;

        // Target-dependent defaults.
        int def_blk_size = 32;  // TODO: calculate based on actual cache size and stencil.
        num_block_threads = 2;
        if (string(YASK_TARGET) == "knl") {
            def_blk_size = 64;   // larger L2.
            num_block_threads = 8; // 4 threads per core * 2 cores per tile.
        }
        else if (string(YASK_TARGET) == "knc") {
            num_block_threads = 4; // 4 threads per core.
        }

        // Use both step and domain dims for all size tuples.
        _global_sizes = dims->_stencil_dims;
        _global_sizes.setValsSame(0); // 0 => calc from rank.

        _rank_sizes = dims->_stencil_dims;
        _rank_sizes.setValsSame(0); // 0 => calc from global.

        _region_sizes = dims->_stencil_dims;
        _region_sizes.setValsSame(0);   // 0 => rank size.

        _block_group_sizes = dims->_stencil_dims;
        _block_group_sizes.setValsSame(0); // 0 => min size.

        _block_sizes = dims->_stencil_dims;
        _block_sizes.setValsSame(def_blk_size);
        _block_sizes.setVal(step_dim, 0); // 0 => default.

        _mini_block_group_sizes = dims->_stencil_dims;
        _mini_block_group_sizes.setValsSame(0); // 0 => min size.

        _mini_block_sizes = dims->_stencil_dims;
        _mini_block_sizes.setValsSame(0);            // 0 => calc from block.

        _sub_block_group_sizes = dims->_stencil_dims;
        _sub_block_group_sizes.setValsSame(0); // 0 => min size.

        _sub_block_sizes = dims->_stencil_dims;
        _sub_block_sizes.setValsSame(0);            // 0 => calc from mini-block.

        _min_pad_sizes = dims->_stencil_dims;
        _min_pad_sizes.setValsSame(0);

        _extra_pad_sizes = dims->_stencil_dims;
        _extra_pad_sizes.setValsSame(0);

        // Use only domain dims for MPI tuples.
        _num_ranks = dims->_domain_dims;
        _num_ranks.setValsSame(0); // 0 => set using heuristic.

        _rank_indices = dims->_domain_dims;
        _rank_indices.setValsSame(0);
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
        for (auto& dim : var) {
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
        auto shortcut = prefix;
        if (shortcut.back() == '_')
            shortcut.pop_back();
        parser.add_option(new CommandLineParser::MultiIdxOption
                          (shortcut,
                           "Shortcut for" + multi_help,
                           multi_vars));
    }

    // Add access to these options from a cmd-line parser.
    void KernelSettings::add_options(CommandLineParser& parser)
    {
        // Following options are in the 'yask' namespace, i.e., no object.
        parser.add_option(new CommandLineParser::BoolOption
                          ("print_suffixes",
                           "Format output with suffixes for human readibility, e.g., 6.15K, 12.3GiB, 7.45m."
                           " If disabled, prints without suffixes for computer parsing, e.g., 6150, 1.23e+10, 7.45e-3.",
                           yask::is_suffix_print_enabled));

        // Following options are in 'this' object.
        _add_domain_option(parser, "g", "Global-domain (overall-problem) size", _global_sizes);
        _add_domain_option(parser, "l", "Local-domain (rank) size", _rank_sizes);
        _add_domain_option(parser, "d", "Deprecated alias for local-domain size", _rank_sizes);
        _add_domain_option(parser, "r", "Region size", _region_sizes, true);
        _add_domain_option(parser, "b", "Block size", _block_sizes, true);
        _add_domain_option(parser, "mb", "Mini-block size", _mini_block_sizes);
        _add_domain_option(parser, "sb", "Sub-block size", _sub_block_sizes);
#ifdef SHOW_GROUPS
        _add_domain_option(parser, "bg", "Block-group size", _block_group_sizes);
        _add_domain_option(parser, "mbg", "Mini-block-group size", _mini_block_group_sizes);
        _add_domain_option(parser, "sbg", "Sub-block-group size", _sub_block_group_sizes);
#endif
        _add_domain_option(parser, "mp", "Minimum var-padding size (including halo)", _min_pad_sizes);
        _add_domain_option(parser, "ep", "Extra var-padding size (beyond halo)", _extra_pad_sizes);
        parser.add_option(new CommandLineParser::BoolOption
                          ("allow_addl_padding",
                           "Allow automatic extension of padding beyond what is needed for"
                           " vector alignment for additional performance reasons",
                           _allow_addl_pad));
#ifdef USE_MPI
        _add_domain_option(parser, "nr", "Num ranks", _num_ranks);
        _add_domain_option(parser, "ri", "This rank's logical index (0-based)", _rank_indices);
        parser.add_option(new CommandLineParser::BoolOption
                          ("overlap_comms",
                           "Overlap MPI communication with calculation of interior elements whenever possible.",
                           overlap_comms));
        parser.add_option(new CommandLineParser::BoolOption
                          ("use_shm",
                           "Use shared memory for MPI halo-exchange buffers between ranks on the same node when possible.",
                           use_shm));
        parser.add_option(new CommandLineParser::IdxOption
                          ("min_exterior",
                           "Minimum width of MPI exterior section to compute before starting MPI communication.",
                           _min_exterior));
#endif
        parser.add_option(new CommandLineParser::BoolOption
                          ("force_scalar",
                           "Evaluate every var point with scalar stencil operations (for debug).",
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
                           "Divide mini-blocks into sub-blocks of slabs along the first valid dimension "
                           "(usually the outer-domain dimension), ignoring other sub-block sizes. "
                           "Assign each slab to a block thread based on its global index in that dimension. "
                           "This setting may increase cache locality when using multiple "
                           "block-threads when scratch vars are used and/or "
                           "when temporal blocking is active. "
                           "This option is ignored if there are fewer than two block threads.",
                           bind_block_threads));
#ifdef USE_NUMA
        stringstream msg;
        msg << "Preferred NUMA node on which to allocate data for "
            "vars and MPI buffers. Alternatively, use special values " <<
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
        parser.add_option(new CommandLineParser::BoolOption
                          ("auto_tune",
                           "Adjust block sizes *during* normal operation to tune for performance. "
                           "May cause varying performance between steps.",
                           _do_auto_tune));
        parser.add_option(new CommandLineParser::DoubleOption
                          ("auto_tune_min_secs",
                           "Minimum seconds to run trial during auto-tuning for trial settings to be "
                           "considered better than the existing best.",
                           _tuner_min_secs));
        parser.add_option(new CommandLineParser::BoolOption
                          ("auto_tune_mini_blocks",
                           "Apply the auto-tuner to mini-block sizes instead of block sizes. "
                           "Particularly useful when using temporal block tiling.",
                           _tune_mini_blks));
        parser.add_option(new CommandLineParser::BoolOption
                          ("auto_tune_each_pass",
                           "Apply the auto-tuner separately to each stencil pack when "
                           "those packs are applied in separate passes across the entire var, "
                           "i.e., when no temporal tiling is used.",
                           _allow_pack_tuners));
    }

    // Print usage message.
    void KernelSettings::print_usage(ostream& os,
                                     CommandLineParser& appParser,
                                     const string& pgmName,
                                     const string& appNotes,
                                     const vector<string>& appExamples)
    {
        os << "Usage: " << pgmName << " [options]\n"
            "Options:\n";
        appParser.print_help(os);
        CommandLineParser solnParser;
        add_options(solnParser);
        solnParser.print_help(os);
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
            " A 'local-domain' or 'rank-domain' is composed of regions.\n"
            "  This is the unit of work for one MPI rank.\n"
            "  Ranks are evaluated in parallel in separate MPI processes.\n"
            " The 'global-domain' or 'overall-problem' is composed of local-domains.\n"
            "  This is the unit of work across all MPI ranks.\n" <<
#ifndef USE_MPI
            "   This binary has NOT been compiled with MPI support,\n"
            "   so the global-domain is equivalent to the single local-domain.\n" <<
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
            " Set local-domain sizes to specify the work done on this MPI rank.\n"
            "  A local-domain size of 0 in a given domain dimension =>\n"
            "   local-domain size is determined by the global-domain size in that dimension.\n"
            "  This and the number of vars affect the amount of memory used.\n"
            " Set global-domain sizes to specify the work done across all MPI ranks.\n"
            "  A global-domain size of 0 in a given domain dimension =>\n"
            "   global-domain size is the sum of local-domain sizes in that dimension.\n"
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
            "  To 'strong-scale' a given overall-problem size, use multiple MPI ranks\n"
            "   and keep the global-domain sizes constant.\n"
            "  To 'weak-scale' to a larger overall-problem size, use multiple MPI ranks\n"
            "   and keep the local-domain sizes constant.\n" <<
#endif
            appNotes;

        // Make example knobs.
        string ex1, ex2;
        DOMAIN_VAR_LOOP(i, j) {
            auto& dname = _dims->_domain_dims.getDimName(j);
            ex1 += " -g" + dname + " " + to_string(i * 128);
            ex2 += " -nr" + dname + " " + to_string(i + 1);
        }
        os <<
            "\nExamples:\n"
            " " << pgmName << " -g 768  # global-domain size in all dims.\n"
            " " << pgmName << ex1 << "  # global-domain size in each dim.\n"
            " " << pgmName << " -l 2048 -r 512 -rt 10  # local-domain size and temporal rank tiling.\n"
            " " << pgmName << " -g 512" << ex2 << "  # number of ranks in each dim.\n";
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
        for (auto& dim : inner_sizes) {
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
    // Called from prepare_solution(), during auto-tuning, etc.
    void KernelSettings::adjustSettings(KernelStateBase* ksb) {
        yask_output_ptr op = ksb ? ksb->get_debug_output() : nullop;
        ostream& os = op->get_ostream();

        auto& step_dim = _dims->_step_dim;
        auto& inner_dim = _dims->_inner_dim;
        auto& rt = _region_sizes[step_dim];
        auto& bt = _block_sizes[step_dim];
        auto& mbt = _mini_block_sizes[step_dim];
        auto& cluster_pts = _dims->_cluster_pts;
        int nddims = _dims->_domain_dims.getNumDims();
        
        // Fix up step-dim sizes.
        rt = max(rt, idx_t(0));
        bt = max(bt, idx_t(0));
        mbt = max(mbt, idx_t(0));
        if (!rt)
            rt = bt;       // Default region steps == block steps.
        if (!mbt)
            mbt = bt;       // Default mini-blk steps == block steps.

        // Determine num regions.
        // Also fix up region sizes as needed.
        // Temporal region size will be increase to
        // current temporal block size if needed.
        // Default region size (if 0) will be size of rank-domain.
        os << "\nRegions:" << endl;
        auto nr = findNumSubsets(os, _region_sizes, "region",
                                 _rank_sizes, "local-domain",
                                 cluster_pts, step_dim);
        os << " num-regions-per-local-domain-per-step: " << nr << endl;
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
                                 cluster_pts, step_dim);
        os << " num-blocks-per-region-per-step: " << nb << endl;
        os << " num-blocks-per-local-domain-per-step: " << (nb * nr) << endl;
        os << " Since the block size in the '" << step_dim <<
            "' dim is " << bt << ", temporal blocking is ";
        if (!bt) os << "NOT ";
        os << "enabled.\n";

        // Determine num mini-blocks.
        // Also fix up mini-block sizes as needed.
        os << "\nMini-blocks:" << endl;
        auto nmb = findNumSubsets(os, _mini_block_sizes, "mini-block",
                                  _block_sizes, "block",
                                  cluster_pts, step_dim);
        os << " num-mini-blocks-per-block-per-step: " << nmb << endl;
        os << " num-mini-blocks-per-region-per-step: " << (nmb * nb) << endl;
        os << " num-mini-blocks-per-local-domain-per-step: " << (nmb * nb * nr) << endl;
        os << " Since the mini-block size in the '" << step_dim <<
            "' dim is " << mbt << ", temporal wave-front block tiling is ";
        if (!mbt) os << "NOT ";
        os << "enabled.\n";

        // Adjust defaults for sub-blocks to be slab if
        // we are using more than one block thread.
        // Otherwise, findNumSubsets() would set default
        // to entire block.
        if (num_block_threads > 1 && _sub_block_sizes.sum() == 0) {

            // Default dim is outer one.
            _bind_posn = 1;

            // Look for best dim to split and bind threads to
            // if binding is enabled.
            DOMAIN_VAR_LOOP(i, j) {

                // Don't pick inner dim.
                auto& dname = _dims->_domain_dims.getDimName(j);
                if (dname == inner_dim)
                    continue;

                auto bsz = _block_sizes[i];
                auto cpts = cluster_pts[j];
                auto clus_per_blk = bsz / cpts;

                // Subdivide this dim if there are enough clusters in
                // the block for each thread.
                if (clus_per_blk >= num_block_threads) {
                    _bind_posn = i;

                    // Stop when first dim picked.
                    break;
                }
            }

            // Divide on best dim.
            auto bsz = _block_sizes[_bind_posn - 1]; // "-1" to adjust stencil to domain dims.
            auto cpts = cluster_pts[_bind_posn - 1];

            // Use narrow slabs if at least 2D.
            // TODO: consider a better heuristic.
            if (nddims >= 2)
                _sub_block_sizes[_bind_posn] = cpts;

            // Divide block equally.
            else
                _sub_block_sizes[_bind_posn] = ROUND_UP(bsz / num_block_threads, cpts);
        }

        // Determine num sub-blocks.
        // Also fix up sub-block sizes as needed.
        os << "\nSub-blocks:" << endl;
        auto nsb = findNumSubsets(os, _sub_block_sizes, "sub-block",
                                  _mini_block_sizes, "mini-block",
                                  cluster_pts, step_dim);
        os << " num-sub-blocks-per-mini-block-per-step: " << nsb << endl;
        os << " num-sub-blocks-per-block-per-step: " << (nsb * nmb) << endl;
        os << " num-sub-blocks-per-region-per-step: " << (nsb * nmb * nb) << endl;
        os << " num-sub-blocks-per-rank-per-step: " << (nsb * nmb * nb * nr) << endl;

        // Determine binding dimension. Do this again if it was done above
        // by default because it may have changed during adjustment.
        if (bind_block_threads && num_block_threads > 1) {
            DOMAIN_VAR_LOOP(i, j) {

                // Don't pick inner dim.
                auto& dname = _dims->_domain_dims.getDimName(j);
                if (dname == inner_dim)
                    continue;

                auto bsz = _block_sizes[i];
                auto sbsz = _sub_block_sizes[i];
                auto sb_per_b = CEIL_DIV(bsz, sbsz);

                // Choose first dim with enough sub-blocks
                // per block.
                if (sb_per_b >= num_block_threads) {
                    _bind_posn = i;
                    break;
                }                        
            }
            os << " Note: only the sub-block size in the '" << 
                _dims->_stencil_dims.getDimName(_bind_posn) << "' dimension may be used at run-time\n"
                "  because block-thread binding is enabled on " << num_block_threads << " block threads.\n";
        }

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
                                   cluster_pts, step_dim);
        os << " num-blocks-per-block-group-per-step: " << nb_g << endl;

        // Show num mini-block-groups.
        // TODO: only print this if mini-block-grouping is enabled.
        auto nmbg = findNumSubsets(os, _mini_block_group_sizes, "mini-block-group",
                                   _block_sizes, "block",
                                   _mini_block_sizes, step_dim);
        os << " num-mini-block-groups-per-block-per-step: " << nmbg << endl;
        auto nmb_g = findNumSubsets(os, _mini_block_sizes, "mini-block",
                                    _mini_block_group_sizes, "mini-block-group",
                                    cluster_pts, step_dim);
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

    // Ctor.
    KernelStateBase::KernelStateBase(KernelEnvPtr& kenv,
                                     KernelSettingsPtr& ksettings)
    {
        // Create state. All other objects that need to share
        // this state should use a shared ptr to it.
        _state = make_shared<KernelState>();

        // Share passed ptrs.
        assert(kenv);
        _state->_env = kenv;
        assert(ksettings);
        _state->_opts = ksettings;
        assert(ksettings->_dims);
        _state->_dims = ksettings->_dims;

       // Create MPI Info object.
        _state->_mpiInfo = make_shared<MPIInfo>(ksettings->_dims);

        // Set vars after above inits.
        STATE_VARS(this);

        // Find index posns in stencil dims.
        DOMAIN_VAR_LOOP(i, j) {
            auto& dname = stencil_dims.getDimName(i);
            if (state->_outer_posn < 0)
                state->_outer_posn = i;
            if (dname == dims->_inner_dim)
                state->_inner_posn = i;
        }
        assert(outer_posn == state->_outer_posn);
    }

    // Set number of threads w/o using thread-divisor.
    // Return number of threads.
    // Do nothing and return 0 if not properly initialized.
    int KernelStateBase::set_max_threads() {
        STATE_VARS(this);

        // Get max number of threads.
        int mt = max(opts->max_threads, 1);

        // Reset number of OMP threads to max allowed and disable nesting.
        omp_set_num_threads(mt);
        omp_set_nested(0);
        omp_set_max_active_levels(1);
        return mt;
    }

    // Get total number of computation threads to use.
    int KernelStateBase::get_num_comp_threads(int& region_threads, int& blk_threads) const {
        STATE_VARS(this);

        // Max threads / divisor.
        int mt = max(opts->max_threads, 1);
        int td = max(opts->thread_divisor, 1);
        int at = mt / td;
        at = max(at, 1);

        // Blk threads per region thread.
        int bt = max(opts->num_block_threads, 1);
        bt = min(bt, at); // Cannot be > 'at'.
        blk_threads = bt;
        assert(bt >= 1);

        // Region threads.
        int rt = at / bt;
        rt = max(rt, 1);
        region_threads = rt;
        assert(rt >= 1);

        // Total number of block threads.
        // Might be less than max threads due to truncation.
        int ct = bt * rt;
        assert(ct <= mt);
        return ct;
    }
        
    // Set number of threads to use for a region.
    // Enable nested OMP if there are >1 block threads,
    // disable otherwise.
    // Return number of threads.
    // Do nothing and return 0 if not properly initialized.
    int KernelStateBase::set_region_threads() {
        int rt=0, bt=0;
        int at = get_num_comp_threads(rt, bt);

        // Limit outer nesting to allow num_block_threads per nested
        // block loop.
        yask_num_threads[0] = rt;

        if (bt > 1) {
            omp_set_nested(1);
            omp_set_max_active_levels(2);
            int mal = omp_get_max_active_levels();
            assert (mal == 2);
            yask_num_threads[1] = bt;
        }
        else {
            assert(bt == 1);
            omp_set_nested(0);
            omp_set_max_active_levels(1);
            int mal = omp_get_max_active_levels();
            assert (mal == 1);
            yask_num_threads[1] = 0;
        }

        omp_set_num_threads(rt);
        return rt;
    }

    // Set number of threads for a block.
    // Return number of threads.
    // Do nothing and return 0 if not properly initialized.
    int KernelStateBase::set_block_threads() {
        int rt=0, bt=0;
        int at = get_num_comp_threads(rt, bt);

        if (bt > 1) {
            int mal = omp_get_max_active_levels();
            assert (mal == 2);
            omp_set_num_threads(bt);
        }
        return bt;
    }

    
    // ContextLinker ctor.
    ContextLinker::ContextLinker(StencilContext* context) :
        KernelStateBase(context->get_state()),
        _context(context) {
        assert(context);
    }

} // namespace yask.
